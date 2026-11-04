"""SAM 3.1 grounded person segmenter — experimental (MASK_SEGMENTER=sam3).

Drop-in alternative to PersonSegmenter for the render/mask stage's
all-persons masking: same contract, ``all_persons(frame_rgb)`` -> union
mask of every person (HxW bool) or None. Only that contract is
implemented — detector-box masking (``infer``/``infer_boxes``) stays
RF-DETR-only; run_mask rejects mixed renders early.

Engine recipe = the visiontouch deployment (video_pipeline.py
Sam3TRTWrapper): FP8-quantized dynk ONNX built as a strongly-typed
TensorRT engine in FP32 build mode (fp32=True — no FP16 layering, the
graph's FP8 quantization stands), dynamic-K optimization profile over
the text-prompt inputs, 1008x1008 input, one execute+sync per pass.

Prompt tokens are CLIP BPE rows (openai/clip-vit-base-patch32, 77-token
budget truncated to the 32-token window the engine expects). The
bundled ``sam3_tokens_person.npz`` was minted with:

  HF_HUB_OFFLINE=0 python3 -c "from transformers import CLIPTokenizer; \\
      t = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32'); \\
      print(t.encode('person')[:32])"   # then zero-pad to 32

Timing (RTX 4090, TRT 11.2.1.2): ~36 ms/frame at K=1 vs RF-DETR
PersonSegmenter's ~6.7 ms — see scans/SAM3_MASK_TIMING.md. Off by
default; enabled per-job via POST /mask segmenter=sam3.
"""

from __future__ import annotations

import logging
import os

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def person_row(tokens: np.ndarray, prompts: list[str]) -> int:
    """Index of the person prompt row: exact 'person' first, then any
    prompt containing it, then 'player' (sports-table naming, e.g.
    visiontouch's 'cricket player')."""
    lowered = [str(p).lower() for p in prompts]
    if "person" in lowered:
        return lowered.index("person")
    for sub in ("person", "player"):
        for i, p in enumerate(lowered):
            if sub in p:
                return i
    raise ValueError(
        f"SAM3 token table has no person row (prompts: {prompts})")


def union_person_mask(outs: dict, out_names: list[str], thresh: float,
                      h0: int, w0: int, max_queries: int):
    """Decode one grounded pass into the all-persons union mask.

    outs: raw output tensors keyed by name (torch, device); row 0 = the
    person prompt. probs = sigmoid(logits) * sigmoid(presence); every
    query >= thresh contributes its 288^2 mask logits -> sigmoid ->
    bilinear -> frame size -> >0.5. Returns HxW bool | None (None when
    no query passes — the caller's fallbacks take over).
    """
    import torch

    def find(sub: str):
        return next(n for n in out_names if sub in n)

    probs = (torch.sigmoid(outs[find("pred_logits")][0, :, 0])
             * torch.sigmoid(outs[find("presence")][0, 0]))
    sel = np.where(probs.detach().cpu().numpy() >= thresh)[0]
    sel = sel[:max_queries]
    if len(sel) == 0:
        return None
    m = torch.sigmoid(outs[find("pred_masks")][0, torch.from_numpy(sel)])
    up = torch.nn.functional.interpolate(
        m[None], size=(h0, w0), mode="bilinear", align_corners=False)[0]
    union = (up > 0.5).any(0)
    return union if bool(union.any()) else None


class Sam3Segmenter:
    """SAM 3.1 TRT wrapper with the PersonSegmenter.all_persons contract."""

    backend_name = "trt"
    model_name = "sam3-grounding-fp8"

    def __init__(self, threshold: float | None = None):
        import torch

        self._torch = torch
        self.threshold = (config.SAM3_THRESH_PERSON if threshold is None
                          else threshold)
        self.in_size = config.SAM3_IN_SIZE

        npz = np.load(config.SAM3_TOKENS, allow_pickle=True)
        self.prompts = [str(p) for p in npz["prompts"]]
        self.row = person_row(npz["text_tokens"], self.prompts)
        # single-row passes: K=1 uses the profile's min shape
        self.tokens = torch.from_numpy(
            np.ascontiguousarray(
                npz["text_tokens"][self.row:self.row + 1])).long()
        self.ids = torch.zeros((1,), dtype=torch.long)

        from .trt import TrtEngine, engine_path
        onnx = config.SAM3_ONNX
        self.engine = TrtEngine(
            onnx, engine_path(onnx, recipe="fp8-fp32"),
            name="sam3_fp8_dynk", fp32=True,
            workspace=config.SAM3_WORKSPACE,
            profile=[
                ("text_tokens", ((config.SAM3_KMIN, 32),
                                 (config.SAM3_KOPT, 32),
                                 (config.SAM3_KMAX, 32))),
                ("img_ids", ((config.SAM3_KMIN,),
                             (config.SAM3_KOPT,),
                             (config.SAM3_KMAX,))),
            ])
        self.out_names = self.engine.out_names
        self.device = self.engine.device
        self.tokens = self.tokens.to(self.device)
        self.ids = self.ids.to(self.device)
        self._pix = torch.empty((1, 3, self.in_size, self.in_size),
                                dtype=torch.float32, device=self.device)
        logger.info("[SAM3] grounded segmenter ready: row=%d (%r) "
                    "thresh=%.2f", self.row, self.prompts[self.row],
                    self.threshold)

    def _preprocess(self, frame_rgb: np.ndarray):
        """RGB HxWx3 uint8 -> normalized float32 CHW on device."""
        import cv2

        x = cv2.resize(frame_rgb, (self.in_size, self.in_size),
                       interpolation=cv2.INTER_LINEAR)
        x = ((x.astype(np.float32) / 255.0) - 0.5) / 0.5
        return self._torch.from_numpy(
            x.transpose(2, 0, 1)[None]).to(self.device)

    def all_persons(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """-> union mask of every person instance, or None when the model
        finds nobody (same contract as PersonSegmenter.all_persons)."""
        h0, w0 = frame_rgb.shape[:2]
        self._pix[0].copy_(self._preprocess(frame_rgb)[0])
        outs = self.engine.run({"pixel_values": self._pix,
                                "text_tokens": self.tokens,
                                "img_ids": self.ids})
        return union_person_mask(outs, self.out_names, self.threshold,
                                 h0, w0, config.SAM3_MAX_QUERIES)
