"""Export viddexa/nsfw-detection-2-mini (EfficientNet-B4 classifier) to ONNX.

Runs inside the movie-trimmer container (torch + transformers present).
Output: models/nsfw2mini/nsfw2mini_v1.onnx  (dynamic batch, 380x380, logits).
"""

import json
import pathlib
import time

import torch
from transformers import AutoModelForImageClassification

MP = pathlib.Path("models/nsfw2mini")
OUT = MP / "nsfw2mini_v1.onnx"

t0 = time.time()
model = AutoModelForImageClassification.from_pretrained(
    str(MP), torch_dtype=torch.float32).eval()


class LogitsOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
        return self.m(x).logits


wrapped = LogitsOnly(model)
dummy = torch.randn(1, 3, 380, 380)

with torch.no_grad():
    ref = wrapped(dummy)
print("logits shape:", tuple(ref.shape))

torch.onnx.export(
    wrapped, dummy, str(OUT),
    input_names=["images"], output_names=["logits"],
    dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
    opset_version=17,
    dynamo=False,   # legacy exporter: dynamo emits AdaptiveAvgPool as an
                    # AveragePool with a TRT-invalid kernel + IR v10/opset 18
)

import onnx  # noqa: E402
sess_check = onnx.load(str(OUT))
onnx.checker.check_model(sess_check)
meta = {
    "id2label": model.config.id2label,
    "input": "images [B,3,380,380] RGB, ImageNet norm "
             "(mean .485/.456/.406 std .4785/.4733/.4743), resized 380x380 squash",
    "output": "logits [B,5]",
}
(MP / "export_meta.json").write_text(json.dumps(meta, indent=1))
print(f"OK {OUT} ({OUT.stat().st_size/1e6:.1f} MB) in {time.time()-t0:.1f}s")
