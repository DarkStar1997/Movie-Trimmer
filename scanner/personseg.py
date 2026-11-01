"""RF-DETR Seg Medium person segmenter (stage C: mask-precision blurring).

Ported from fitness-coach `visioncoach.models.PersonSegmenter`, keeping the
numerics identical:

  preprocessing — RGB (our decoder already yields RGB; the reference reads
    BGR and swaps — same thing), plain resize to 432x432 (NOT letterbox:
    RF-DETR convention), /255, ImageNet normalize (means 0.485/0.456/0.406,
    stds 0.229/0.224/0.225), NCHW float32.

  decode (rfdetr PostProcess replicated) — sigmoid over ALL query x class
    logits, global top-k (num_select), q_idx = flat // C, label = flat % C,
    boxes cxcywh normalized -> xyxy scaled to the frame; keep person
    (class 1 in COCO-91 indexing) above PERSON_DETECT_THRESHOLD.

  match_box association — person instance with best IoU against the
    explicit-part box (fall back to top score below MATCH_IOU); mask logits
    of only the selected query bilinear-interpolated to frame size, > 0.

Precision: TRT engine is FP32 (fitness-coach lesson — TRT 11 strongly-typed
FP16 drifts RF-DETR logits by ~9; trt.TrtEngine(fp32=True) builds from the
original ONNX untouched). ORT is the host/test path and equivalence
reference, same as the NudeNet detector.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from . import config
from .detector import OrtBackend

logger = logging.getLogger(__name__)

# ImageNet normalization (rfdetr.detr defaults)
_MEANS = np.asarray([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
_STDS = np.asarray([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)

PERSON_CLASS = 1          # COCO-91 indexing (0 is background)
PERSON_DETECT_THRESHOLD = 0.35
MATCH_IOU = 0.3
NUM_SELECT = 300


def _box_iou(a, b) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = ((a[2] - a[0]) * (a[3] - a[1])
          + (b[2] - b[0]) * (b[3] - b[1]) - inter)
    return inter / ua if ua > 0 else 0.0


def preprocess(frame_rgb: np.ndarray, size: int) -> np.ndarray:
    """RGB uint8 HxWx3 -> (1,3,size,size) normalized float32.

    ascontiguousarray is LOAD-BEARING: the arithmetic chain on a
    transposed view lets numpy allocate the output Fortran-ordered
    (HWC bytes). ORT honors numpy strides and reads it correctly, but a
    TRT engine reads data_ptr() as LINEAR NCHW — feeding it HWC bytes
    permutes channels with spatial and the engine outputs garbage (the
    'TRT is blind' mystery, and the fitness-coach 'FP16 drifts by ~9'
    lesson before it)."""
    r = cv2.resize(frame_rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    t = r.astype(np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(((t - _MEANS) / _STDS)[None])


def decode(outs: dict, in_names_map: dict, h0: int, w0: int,
           match_box: np.ndarray | None,
           num_select: int = NUM_SELECT,
           det_thr: float = PERSON_DETECT_THRESHOLD
           ) -> tuple[np.ndarray | None, float, float, np.ndarray | None]:
    """Raw outputs -> (mask HxW bool | None, mask_area_fraction, score,
    person_bbox xyxy | None).

    person_bbox is set only when the match to `match_box` succeeded (IoU
    above MATCH_IOU, or the explicit-part box is >=60% inside the person
    box — containment matters for part boxes, whose IoU against a person
    box is inherently low). Unmatched callers must NOT use the returned
    mask as a full-person blur: it is whatever person scored highest.
    """
    logits = np.asarray(outs[in_names_map["labels"]], dtype=np.float32)
    boxes_cxcy = np.asarray(outs[in_names_map["dets"]], dtype=np.float32)
    masks = np.asarray(outs[in_names_map["masks"]], dtype=np.float32)
    _, n_q, n_c = logits.shape
    k = min(num_select, n_q * n_c)
    prob = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))       # sigmoid
    top_flat = np.argpartition(prob, -k)[-k:]
    top_scores = prob[top_flat]
    order = np.argsort(-top_scores)
    top_flat, top_scores = top_flat[order], top_scores[order]
    q_idx = top_flat // n_c
    labels = top_flat % n_c

    b = boxes_cxcy[0][q_idx]                               # (K,4) normalized
    xyxy = np.stack([b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2,
                     b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2], axis=-1)
    xyxy *= np.asarray([w0, h0, w0, h0], dtype=np.float32)

    keep = (top_scores >= det_thr) & (labels == PERSON_CLASS)
    if not keep.any():
        return None, 0.0, 0.0, None
    boxes_k, scores_k = xyxy[keep], top_scores[keep]
    rows_k = np.nonzero(keep)[0]

    r = int(np.argmax(scores_k))
    matched_bbox: np.ndarray | None = None
    if match_box is not None:
        best_key, best_r = -1.0, r
        for i, bx in enumerate(boxes_k):
            iou = _box_iou(bx, match_box)
            x1, y1 = max(bx[0], match_box[0]), max(bx[1], match_box[1])
            x2, y2 = min(bx[2], match_box[2]), min(bx[3], match_box[3])
            mb_area = max(0.0, (match_box[2] - match_box[0])
                          * (match_box[3] - match_box[1]))
            contained = (max(0.0, x2 - x1) * max(0.0, y2 - y1)
                         / mb_area) if mb_area > 0 else 0.0
            key = max(iou, contained)
            if key > best_key:
                best_key, best_r = key, int(i)
        if best_key > MATCH_IOU or best_key > 0.6:
            r = best_r
            matched_bbox = boxes_k[r]

    m_logits = masks[0, q_idx[rows_k[r]]]                  # (108,108)
    m = cv2.resize(m_logits, (w0, h0), interpolation=cv2.INTER_LINEAR)
    mask = m > 0.0
    return mask, float(mask.mean()), float(scores_k[r]), matched_bbox


def persons_union(outs: dict, in_names_map: dict, h0: int, w0: int,
                  num_select: int = NUM_SELECT,
                  det_thr: float = PERSON_DETECT_THRESHOLD
                  ) -> np.ndarray | None:
    """Union mask of EVERY person instance above det_thr (or None). Used for
    VLM-sourced segments, where no explicit-part box exists to match on."""
    logits = np.asarray(outs[in_names_map["labels"]], dtype=np.float32)
    masks = np.asarray(outs[in_names_map["masks"]], dtype=np.float32)
    _, n_q, n_c = logits.shape
    k = min(num_select, n_q * n_c)
    prob = 1.0 / (1.0 + np.exp(-logits.reshape(-1)))
    top_flat = np.argpartition(prob, -k)[-k:]
    order = np.argsort(-prob[top_flat])
    top_flat = top_flat[order]
    q_idx = top_flat // n_c
    labels = top_flat % n_c
    keep = (prob[top_flat] >= det_thr) & (labels == PERSON_CLASS)
    if not keep.any():
        return None
    union = np.zeros((h0, w0), dtype=bool)
    for q in q_idx[keep]:
        m = cv2.resize(masks[0, q], (w0, h0),
                       interpolation=cv2.INTER_LINEAR)
        union |= m > 0.0
    return union


def person_presence(outs: dict, in_names_map: dict,
                    det_thr: float = PERSON_DETECT_THRESHOLD
                    ) -> tuple[bool, float]:
    """Raw outputs -> (person_present, max_person_score). Pure.

    Unlike decode()/persons_union() (global top-k over all classes, then
    keep person), this takes the max over the person column directly —
    a presence gate wants recall: a person scoring below the global
    top-k cut still counts. No boxes, no masks — the gate only asks
    'is anybody here'."""
    logits = np.asarray(outs[in_names_map["labels"]], dtype=np.float32)
    _, n_q, n_c = logits.shape
    prob = 1.0 / (1.0 + np.exp(
        -logits.reshape(n_q, n_c)[:, PERSON_CLASS]))
    mx = float(prob.max())
    return mx >= det_thr, mx


class PersonSegmenter:
    """person mask for an explicit-part box, via TRT FP32.

    "ort" is accepted explicitly for host unit tests; production is TRT-only
    with no silent fallback — a TRT failure fails the job loudly."""

    def __init__(self, backend: str = "trt"):
        cfg = config.PERSON_SEG
        self.onnx_path = str(config.MODEL_DIR / cfg["onnx"])
        self.resolution = int(cfg["resolution"])
        self.backend = backend
        self._names: dict[str, str] | None = None
        if backend == "trt":
            self._init_trt()
            self.backend_name = "trt"
        elif backend == "ort":
            self.ort = OrtBackend(self.onnx_path)
            self._names = self._ort_names()
            self.backend_name = "ort"
        else:
            raise ValueError(f"unknown backend {backend!r} "
                             f"(use 'trt'; 'ort' is test-only)")

    # -- backends ----------------------------------------------------------
    def _ort_names(self) -> dict[str, str]:
        names = [o.name for o in self.ort.sess.get_outputs()]
        return {
            "dets": next(n for n in names if "dets" in n),
            "labels": next(n for n in names if "labels" in n),
            "masks": next(n for n in names if "mask" in n),
        }

    def _init_trt(self) -> None:
        from .trt import TrtEngine, engine_path

        # profile input name must match the ONNX graph input
        import onnx
        m = onnx.load(self.onnx_path, load_external_data=False)
        in_name = m.graph.input[0].name
        profile = [(in_name, ((1, 3, self.resolution, self.resolution),) * 3)]
        self.engine = TrtEngine(
            self.onnx_path, engine_path(self.onnx_path, recipe="fp32"),
            name="person_seg_rfdetr_m", fp32=True, profile=profile)
        names = self.engine.out_names
        self._names = {
            "dets": next(n for n in names if "dets" in n),
            "labels": next(n for n in names if "labels" in n),
            "masks": next(n for n in names if "mask" in n),
        }
        import torch  # noqa: F401
        self._torch = torch

    # -- inference ----------------------------------------------------------
    def infer(self, frame_rgb: np.ndarray,
              match_box: np.ndarray | None = None) -> tuple[np.ndarray | None,
                                                            float, float,
                                                            np.ndarray | None]:
        """-> (mask HxW bool | None, mask_area_fraction, score,
        matched person_bbox xyxy | None). matched bbox is None when the
        association failed — callers must not treat that mask as the
        matched person's full-body mask (see decode)."""
        h0, w0 = frame_rgb.shape[:2]
        blob = preprocess(frame_rgb, self.resolution)
        if self.backend_name == "ort":
            outs = {
                n: self.ort.sess.run(None, {self.ort.input_name: blob})[i]
                for i, n in enumerate(("dets", "labels", "masks"))
            }
            names = {"dets": "dets", "labels": "labels", "masks": "masks"}
        else:
            torch = self._torch
            t = torch.from_numpy(blob).to(self.engine.device)
            raw = self.engine.run({self.engine.in_names[0]: t})
            outs = {k: raw[n].float().cpu().numpy()
                    for k, n in
                    (("dets", self._names["dets"]),
                     ("labels", self._names["labels"]),
                     ("masks", self._names["masks"]))}
            names = {"dets": "dets", "labels": "labels", "masks": "masks"}
        return decode(outs, names, h0, w0, match_box)

    def infer_boxes(self, frame_rgb: np.ndarray,
                    boxes_xyxy: list[np.ndarray]) -> np.ndarray:
        """Union mask of the person instances matching `boxes_xyxy` (HxW bool;
        all-zero when nothing matches). One personseg pass per distinct box."""
        h, w = frame_rgb.shape[:2]
        union = np.zeros((h, w), dtype=bool)
        for bx in boxes_xyxy:
            mask, _, _, pbbox = self.infer(frame_rgb,
                                           np.asarray(bx, np.float32))
            # only masks whose person actually matched the box
            if mask is not None and pbbox is not None:
                union |= mask
        return union

    def all_persons(self, frame_rgb: np.ndarray) -> np.ndarray | None:
        """-> union mask of every person instance, or None when the
        segmenter finds nobody. VLM-sourced segments have no explicit-part
        box to associate on, so every person present gets masked."""
        h0, w0 = frame_rgb.shape[:2]
        blob = preprocess(frame_rgb, self.resolution)
        if self.backend_name == "ort":
            outs = {
                n: self.ort.sess.run(None, {self.ort.input_name: blob})[i]
                for i, n in enumerate(("dets", "labels", "masks"))
            }
            return persons_union(outs, {"dets": "dets", "labels": "labels",
                                        "masks": "masks"}, h0, w0)
        torch = self._torch
        t = torch.from_numpy(blob).to(self.engine.device)
        raw = self.engine.run({self.engine.in_names[0]: t})
        outs = {k: raw[n].float().cpu().numpy()
                for k, n in
                (("dets", self._names["dets"]),
                 ("labels", self._names["labels"]),
                 ("masks", self._names["masks"]))}
        return persons_union(outs, {"dets": "dets", "labels": "labels",
                                    "masks": "masks"}, h0, w0)

    # -- scan-time presence gate --------------------------------------------
    def presence(self, frame_rgb: np.ndarray,
                 det_thr: float | None = None) -> tuple[bool, float]:
        """Synchronous person-presence check (gate path; no masks)."""
        from . import config as _config

        thr = _config.PERSON_GATE_SCORE if det_thr is None else det_thr
        blob = preprocess(frame_rgb, self.resolution)
        if self.backend_name == "ort":
            outs = {"labels": self.ort.sess.run(
                None, {self.ort.input_name: blob})[1]}
        else:
            torch = self._torch
            t = torch.from_numpy(blob).to(self.engine.device)
            raw = self.engine.run({self.engine.in_names[0]: t})
            outs = {"labels": raw[self._names["labels"]].float().cpu().numpy()}
        return person_presence(outs, {"labels": "labels"}, thr)

    def presence_enqueue(self, frame_rgb: np.ndarray,
                         stream) -> tuple["object", str]:
        """Async gate pass on `stream`: enqueues without syncing.

        -> (labels_tensor_on_device, labels_output_name). The caller owns
        the join (stream.synchronize()) and then converts:
            numpy = tensor.float().cpu().numpy()
            person_presence({"labels": numpy}, {"labels": name}, thr)
        Preprocess (CPU) happens here; only H2D + the engine run land on
        the side stream, so the NudeNet engine's stream keeps flowing."""
        blob = preprocess(frame_rgb, self.resolution)
        torch = self._torch
        with torch.cuda.stream(stream):
            t = torch.from_numpy(blob).to(self.engine.device)
            raw = self.engine.run({self.engine.in_names[0]: t}, sync=False)
        return raw[self._names["labels"]], self._names["labels"]
