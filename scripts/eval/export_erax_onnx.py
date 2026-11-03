"""Export erax-ai/EraX-Anti-NSFW-V1.1 YOLO11 n/s/m to ONNX (host, CPU ok).

ULTRALYTICS POLICY: this is the ONLY step that ever imports ultralytics
(unpacking their .pt checkpoints needs the package). It runs offline:
YOLO_OFFLINE=1 disables is_online() entirely, and ~/.config/Ultralytics/
settings.json must carry sync=false, analytics=false, crash_reports=false,
uuid="" (the harness/container never import ultralytics — engines run as
plain TensorRT ONNX after this).

Run via: YOLO_OFFLINE=1 uv run --with torch --with torchvision \
             --with ultralytics --with onnx \
             python scripts/eval/export_erax_onnx.py
Outputs: models/erax/erax-yolo11{n,s,m}_v1.onnx (dynamic batch, 640x640,
raw [B, 4+nc, 8400] output — no NMS in graph) + classes.json.
"""

import json
import pathlib
import time

from ultralytics import YOLO

ER = pathlib.Path("models/erax")
classes = None

for size in ("n", "s", "m"):
    pt = ER / f"erax-yolo11{size}.pt"
    out = ER / f"erax-yolo11{size}_v1.onnx"
    t0 = time.time()
    model = YOLO(str(pt))
    if classes is None:
        classes = {int(k): str(v) for k, v in model.model.names.items()}
        print("classes:", classes)
    model.export(format="onnx", imgsz=640, dynamic=True, opset=17,
                 simplify=True, device="cpu")
    # ultralytics writes <stem>.onnx next to the .pt
    produced = pt.with_suffix(".onnx")
    produced.rename(out)
    print(f"OK {out.name} ({out.stat().st_size/1e6:.1f} MB) "
          f"in {time.time()-t0:.1f}s")

(ER / "classes.json").write_text(json.dumps(classes, indent=1))
print("done")
