"""All-frames detector benchmark harness (runs inside the movie-trimmer container).

One decode pass per movie; every detector sees the identical frame stream.
Per-model inference timed separately (NudeNet via its production path,
candidates via TrtEngine with custom pre/post). Raw per-frame scores saved
to <outdir>/allframes.npz; thresholds swept offline on the host.

Usage:
  python3 scripts/eval/run_allframes.py <movie> <tag> <W> <H> <outdir>
  e.g. python3 scripts/eval/run_allframes.py 'Gone.Girl...mkv' gg 960 540 evals/gg
"""

import json
import os
import pathlib
import sys
import time

# The image sets TRT_CACHE_DIR=/cache/trt (container-ephemeral): without this
# override every run rebuilds all engines (~330s each) and the plans vanish
# with the container. Eval engines belong in the repo cache the mount exposes.
os.environ["TRT_CACHE_DIR"] = "trt_cache"

import cv2
import numpy as np
import torch

from scanner.decode import FFmpegPipeDecoder
from scanner.detector import NudeNetDetector
from scanner.trt import TrtEngine, engine_path

TRIGGER_LABELS = {
    "FEMALE_GENITALIA_EXPOSED", "MALE_GENITALIA_EXPOSED", "ANUS_EXPOSED",
    "FEMALE_BREAST_EXPOSED", "BUTTOCKS_EXPOSED",
}
MIN_BOX_AREA_FRAC = 0.005
BATCH = 16
FPS = 24.0

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).cuda().view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.47853944, 0.4732864, 0.47434163]).cuda().view(1, 3, 1, 1)


class ClassifierEngine:
    """viddexa nsfw-detection-2-mini: 380x380 squash, ImageNet norm, 5 probs."""

    def __init__(self):
        onnx = "models/nsfw2mini/nsfw2mini_v1.onnx"
        shp = ((1, 3, 380, 380), (BATCH, 3, 380, 380), (BATCH, 3, 380, 380))
        # fp32, not modelopt-fp16: the full-cast head corrupts logits (A/B
        # vs onnxruntime-fp32 showed saturated frames flip porn<->hentai).
        # Same lesson as scanner/personseg.py (RF-DETR fp32).
        self.eng = TrtEngine(onnx, engine_path(onnx, recipe="fp32"),
                             name="nsfw2mini", fp32=True, profile=[("images", shp)])
        # engine IO dtype decides the feed (fp32 here, but stay generic)
        self.half_in = (self.eng.engine.get_tensor_dtype("images")
                        == self.eng._trt.float16)
        self.prep_ms = 0.0

    def prep(self, frames):
        t0 = time.perf_counter()
        batch = np.stack([cv2.resize(f, (380, 380),
                                     interpolation=cv2.INTER_LINEAR)
                          for f in frames])
        x = torch.from_numpy(batch).cuda().permute(0, 3, 1, 2)
        x = x.float().div_(255).sub_(IMAGENET_MEAN).div_(IMAGENET_STD)
        if self.half_in:
            x = x.half()
        self.prep_ms += (time.perf_counter() - t0) * 1e3
        return {"images": x}

    def post(self, out, n):
        logits = list(out.values())[0].float()
        return torch.softmax(logits, dim=-1)[:n].cpu().numpy()


class YoloEngine:
    """EraX YOLO11: letterbox 640, RGB /255, raw [B,4+nc,8400].

    Frame score = per-class max anchor confidence (pre-NMS; sufficient for
    frame-level flagging and threshold sweeps).
    """

    def __init__(self, tag):
        onnx = f"models/erax/erax-yolo11{tag}_v1.onnx"
        shp = ((1, 3, 640, 640), (BATCH, 3, 640, 640), (BATCH, 3, 640, 640))
        # fp32: modelopt fp16 crushes YOLO cls sigmoids ~10x (0.17 -> 0.02
        # in the A/B vs onnxruntime-fp32) — EraX would look dead when alive.
        self.eng = TrtEngine(onnx, engine_path(onnx, recipe="fp32"),
                             name=f"erax-{tag}", fp32=True,
                             profile=[("images", shp)])
        self.half_in = (self.eng.engine.get_tensor_dtype("images")
                        == self.eng._trt.float16)
        self.prep_ms = 0.0

    def prep(self, frames):
        t0 = time.perf_counter()
        h, w = frames[0].shape[:2]
        s = 640 / max(h, w)
        nw, nh = int(round(w * s)), int(round(h * s))
        batch = np.empty((len(frames), 640, 640, 3), dtype=np.uint8)
        batch[:] = 114
        for i, f in enumerate(frames):
            r = cv2.resize(f, (nw, nh), interpolation=cv2.INTER_LINEAR)
            batch[i, :nh, :nw] = r
        x = torch.from_numpy(batch).cuda().permute(0, 3, 1, 2).float().div_(255)
        if self.half_in:
            x = x.half()
        self.prep_ms += (time.perf_counter() - t0) * 1e3
        return {"images": x}

    def post(self, out, n):
        raw = list(out.values())[0]           # [B, 4+nc, A]
        conf = raw[:, 4:, :]                  # [B, nc, A]
        return conf.max(dim=-1).values.float()[:n].cpu().numpy()


def main():
    movie, tag, W, H, outdir = sys.argv[1], sys.argv[2], int(sys.argv[3]), \
        int(sys.argv[4]), pathlib.Path(sys.argv[5])
    outdir.mkdir(parents=True, exist_ok=True)
    log = outdir / "progress.log"

    def say(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(log, "a") as fh:
            fh.write(line + "\n")

    say(f"warming engines ({tag})")
    t0 = time.time()
    nudenet = NudeNetDetector("640m", backend="trt", batch_size=BATCH)
    engines = {"nsfw2mini": ClassifierEngine(),
               "eraxn": YoloEngine("n"), "eraxs": YoloEngine("s"),
               "eraxm": YoloEngine("m")}
    say(f"engines ready in {time.time()-t0:.1f}s "
        f"(cache hits: nudenet prod engine + 4 candidates)")

    per_model_ms = {k: 0.0 for k in ["nudenet"] + list(engines)}
    frames_seen = 0
    buf = []
    rec = {k: [] for k in ["nudenet_max", "nudenet_n", "nsfw2_probs",
                           "eraxn_max", "eraxs_max", "eraxm_max"]}
    t_decode = 0.0

    def flush(batch):
        nonlocal frames_seen
        n = len(batch)
        area_thr = MIN_BOX_AREA_FRAC * W * H
        t = time.perf_counter()
        dets = nudenet.detect_frames(batch)
        per_model_ms["nudenet"] += (time.perf_counter() - t) * 1e3
        for dl in dets:
            best, cnt = 0.0, 0
            for d in dl:
                if d.label in TRIGGER_LABELS and d.area >= area_thr:
                    cnt += 1
                    best = max(best, d.score)
            rec["nudenet_max"].append(best)
            rec["nudenet_n"].append(cnt)
        for name, eng in engines.items():
            t = time.perf_counter()
            out = eng.eng.run(eng.prep(batch))
            probs = eng.post(out, n)
            per_model_ms[name] += (time.perf_counter() - t) * 1e3
            key = "nsfw2_probs" if name == "nsfw2mini" else f"{name}_max"
            rec[key].append(probs)
        frames_seen += n

    t0 = time.time()
    with FFmpegPipeDecoder(movie, fps=FPS, width=W, height=H) as decoder:
        t_last = time.time()
        for f in decoder.frames():
            buf.append(f)
            if len(buf) == BATCH:
                flush(buf)
                buf = []
                now = time.time()
                t_decode += now - t_last
                t_last = now
                if frames_seen % 20000 < BATCH:
                    rate = frames_seen / (time.time() - t0)
                    say(f"{frames_seen} frames, {rate:.0f} fps overall")
        if buf:
            flush(buf)

    wall = time.time() - t0
    np.savez_compressed(
        outdir / "allframes.npz",
        fps=FPS, width=W, height=H,
        nudenet_max=np.asarray(rec["nudenet_max"], dtype=np.float32),
        nudenet_n=np.asarray(rec["nudenet_n"], dtype=np.int16),
        nsfw2_probs=np.vstack(rec["nsfw2_probs"]).astype(np.float32),
        eraxn_max=np.vstack(rec["eraxn_max"]).astype(np.float32),
        eraxs_max=np.vstack(rec["eraxs_max"]).astype(np.float32),
        eraxm_max=np.vstack(rec["eraxm_max"]).astype(np.float32),
    )
    meta = {
        "movie": movie, "tag": tag, "frames": frames_seen, "fps": FPS,
        "decode_wall_s": t_decode, "total_wall_s": wall,
        "inference_ms_per_frame": {k: v / max(1, frames_seen)
                                   for k, v in per_model_ms.items()},
        "effective_inference_fps": {k: 1000.0 / (v / max(1, frames_seen))
                                    for k, v in per_model_ms.items()},
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=1))
    say(f"DONE {frames_seen} frames in {wall:.0f}s "
        f"(decode {t_decode:.0f}s); " + " ".join(
            f"{k}={v/1000:.0f}s" for k, v in per_model_ms.items()))


if __name__ == "__main__":
    main()
