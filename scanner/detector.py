"""NudeNet detector: exact-parity pre/post-processing, TRT or ORT execution.

Reference behaviour replicated from nudenet 3.4.2 (`nudenet/nudenet.py`):

  preprocessing
    - RGB frame (we already decode RGB; the reference reads BGR and swaps)
    - pad right/bottom to square with black (copyMakeBorder semantics)
    - resize the padded square to (res, res), INTER_LINEAR
    - /255.0, NCHW, float32

  postprocessing (per image, output [B, 22, N] -> [N, 22])
    - class score = max over channels 4..21, class = argmax
    - candidates: score >= 0.2
    - box (cx, cy, w, h in model px) -> top-left xy
    - scale x/w by (W + x_pad)/model_w, y/h by (H + y_pad)/model_h
    - clip to frame, then CLASS-AGNOSTIC greedy NMS (entry 0.25, IoU 0.45)

Backends:
  - "trt": ONNX -> TensorRT engine (FP16 default, cached per GPU x TRT x
    weights version) — dynamic batch profile. Container path.
  - "ort": onnxruntime, CPU or CUDA EP if available. Host/test path and
    equivalence reference.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Sequence

import cv2
import numpy as np

from . import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x, y, w, h (top-left, px)

    @property
    def area(self) -> float:
        return max(0.0, self.box[2]) * max(0.0, self.box[3])

    def area_fraction(self, frame_w: int, frame_h: int) -> float:
        return self.area / float(frame_w * frame_h)

    def as_dict(self) -> dict:
        x, y, w, h = self.box
        return {"class": self.label, "score": round(self.score, 4),
                "box": [int(x), int(y), int(w), int(h)]}


# --- backends ---------------------------------------------------------------

class OrtBackend:
    """onnxruntime execution: numpy in, numpy out. No torch dependency."""

    def __init__(self, onnx_path: str):
        import onnxruntime as ort

        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")
        self.sess = ort.InferenceSession(onnx_path, providers=providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.dtype = np.float32

    def infer(self, blob: np.ndarray) -> np.ndarray:
        return self.sess.run(None, {self.input_name: blob.astype(self.dtype)})[0]


class TrtBackend:
    """TensorRT execution via the cached TrtEngine. Requires torch+TRT
    (i.e. the container). Lazy imports keep the host venv torch-free."""

    MAX_BATCH = 32

    def __init__(self, onnx_path: str, resolution: int, *, batch_size: int,
                 fp32: bool = False):
        import torch  # noqa: F401
        from . import trt as trt_mod

        res = resolution
        profile = [("images",
                    ((1, 3, res, res),
                     (max(1, min(batch_size, self.MAX_BATCH)), 3, res, res),
                     (self.MAX_BATCH, 3, res, res)))]
        self.engine = trt_mod.TrtEngine(
            onnx_path,
            trt_mod.engine_path(onnx_path,
                                recipe="fp32" if fp32 else "fp16"),
            name=os.path.basename(onnx_path), fp32=fp32, profile=profile)
        self.input_name = self.engine.in_names[0]
        self.output_name = self.engine.out_names[0]
        self._torch = torch

    def infer(self, blob: np.ndarray) -> np.ndarray:
        torch = self._torch
        import tensorrt as trt

        io_dtype = torch.float16 if trt.float16 == self.engine.engine.get_tensor_dtype(
                self.input_name) else torch.float32
        t = torch.from_numpy(blob).to(device=self.engine.device, dtype=io_dtype)
        out = self.engine.run({self.input_name: t})
        buf = out[self.output_name]
        return buf.float().cpu().numpy()


def _trt_available() -> bool:
    try:
        import tensorrt  # noqa: F401
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


# --- preprocessing ----------------------------------------------------------

def preprocess(frame_rgb: np.ndarray, resolution: int) -> tuple[np.ndarray, int, int]:
    """Return (blob [1,3,res,res] float32, x_pad, y_pad) — reference parity.

    NOTE on channel order (do not "fix"): the reference reads BGR via
    cv2.imread, then cvtColor(RGBA2BGR) *swaps* 3-channel input to RGB,
    then blobFromImage(swapRB=True) swaps back — the net blob the NudeNet
    weights were trained on is **BGR-ordered**. Our input frames are RGB,
    so swapRB=True here reproduces the identical BGR blob.
    """
    h, w = frame_rgb.shape[:2]
    max_side = max(h, w)
    x_pad, y_pad = max_side - w, max_side - h
    if x_pad or y_pad:
        frame = cv2.copyMakeBorder(frame_rgb, 0, y_pad, 0, x_pad,
                                   cv2.BORDER_CONSTANT, value=(0, 0, 0))
    else:
        frame = frame_rgb
    frame = cv2.resize(frame, (resolution, resolution), interpolation=cv2.INTER_LINEAR)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (resolution, resolution),
                                 (0, 0, 0), swapRB=True, crop=False)
    return blob, x_pad, y_pad


# --- postprocessing -----------------------------------------------------------

def _nms_agnostic(boxes_xywh: np.ndarray, scores: np.ndarray,
                  score_thr: float, iou_thr: float) -> list[int]:
    """Greedy class-agnostic NMS — cv2.dnn.NMSBoxes semantics."""
    idx = np.flatnonzero(scores >= score_thr)
    if idx.size == 0:
        return []
    order = idx[np.argsort(-scores[idx], kind="stable")]
    # work in original-index space throughout
    x, y, w, h = boxes_xywh.T
    x2, y2 = x + w, y + h
    areas = w * h
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        iw = np.maximum(0.0, np.minimum(x2[i], x2[rest]) - np.maximum(x[i], x[rest]))
        ih = np.maximum(0.0, np.minimum(y2[i], y2[rest]) - np.maximum(y[i], y[rest]))
        inter = iw * ih
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-9)
        order = rest[iou <= iou_thr]
    return keep


def postprocess(output_b: np.ndarray, frame_w: int, frame_h: int,
                x_pad: int, y_pad: int, model_res: int) -> list[Detection]:
    """Decode one image's raw ONNX output [22, N] (see module docstring)."""
    rows = output_b.T  # [N, 22]
    class_scores = rows[:, 4:]
    cand = np.flatnonzero(class_scores.max(axis=1) >= config.DETECT_SCORE_THRESHOLD)
    if cand.size == 0:
        return []

    scores = class_scores[cand].max(axis=1)
    class_ids = class_scores[cand].argmax(axis=1)
    cx, cy, bw, bh = rows[cand, 0], rows[cand, 1], rows[cand, 2], rows[cand, 3]

    # center -> top-left, scale out of padded-model space into frame space
    sx = (frame_w + x_pad) / model_res
    sy = (frame_h + y_pad) / model_res
    x = (cx - bw / 2.0) * sx
    y = (cy - bh / 2.0) * sy
    w = bw * sx
    h = bh * sy

    # clip to frame boundaries (reference order: clamp x/y, then fit w/h)
    x = np.clip(x, 0.0, frame_w)
    y = np.clip(y, 0.0, frame_h)
    w = np.minimum(w, frame_w - x)
    h = np.minimum(h, frame_h - y)

    boxes = np.stack([x, y, w, h], axis=1)
    keep = _nms_agnostic(boxes, scores,
                         config.NMS_SCORE_THRESHOLD, config.NMS_IOU_THRESHOLD)
    return [Detection(label=config.NUDENET_LABELS[int(class_ids[i])],
                      score=float(scores[i]),
                      box=tuple(float(v) for v in boxes[i]))
            for i in keep]


# --- detector -----------------------------------------------------------------

class NudeNetDetector:
    def __init__(self, model_name: str = None, *, backend: str = "trt",
                 batch_size: int = config.DEFAULT_BATCH_SIZE, fp32: bool = False):
        model_name = model_name or config.DEFAULT_MODEL
        if model_name not in config.MODELS:
            raise ValueError(f"unknown model {model_name!r}; "
                             f"available: {sorted(config.MODELS)}")
        self.model_name = model_name
        fname, res = config.MODELS[model_name]
        self.resolution = res
        self.onnx_path = str(config.MODEL_DIR / fname)
        if not os.path.exists(self.onnx_path):
            raise FileNotFoundError(
                f"{self.onnx_path} missing — models are baked into the image "
                f"at /opt/models (see scanner.Dockerfile)")
        self.batch_size = batch_size

        # TRT is the production backend; "ort" exists for host unit tests
        # only. No silent fallback: a TRT failure must fail the job loudly
        # (fix the engine/container, don't crawl on CPU).
        if backend == "trt":
            self.backend = TrtBackend(self.onnx_path, res,
                                      batch_size=batch_size, fp32=fp32)
            self.backend_name = f"trt({'fp32' if fp32 else 'fp16'})"
        elif backend == "ort":
            self.backend = OrtBackend(self.onnx_path)
            self.backend_name = "ort"
        else:
            raise ValueError(f"unknown backend {backend!r} "
                             f"(use 'trt'; 'ort' is test-only)")
        logger.info("[detector] %s @ %dpx via %s",
                    model_name, res, self.backend_name)

    def detect_frames(self, frames: Sequence[np.ndarray]) -> list[list[Detection]]:
        """Batch-detect a list of same-size RGB uint8 frames.

        Chunks at self.batch_size: TRT engines are profiled up to that batch
        (an oversized vstack would exceed the profile max batch).
        """
        if not frames:
            return []
        results: list[list[Detection]] = []
        for off in range(0, len(frames), self.batch_size):
            chunk = frames[off:off + self.batch_size]
            blobs, pads = [], []
            for f in chunk:
                blob, xp, yp = preprocess(f, self.resolution)
                blobs.append(blob)
                pads.append((xp, yp))
            blob = np.vstack(blobs)
            out = self.backend.infer(blob)  # [B, 22, N]
            h, w = chunk[0].shape[:2]
            for b, (xp, yp) in enumerate(pads):
                results.append(postprocess(out[b], w, h, xp, yp, self.resolution))
        return results
