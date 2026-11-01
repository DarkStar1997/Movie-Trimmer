"""TensorRT engine build/load with a version-aware cache.

Ported from fitness-coach/visiontouch `trt.py`, simplified for a single-node
bind-mounted cache directory (no S3/runpod sync). The cache-key discipline is
kept intact — a stale plan can never be silently reused:

    {TRT_CACHE_DIR}/{gpu_slug}/{trt_slug}/{base}.{weights_vN}.{engine}

Precision policy (production lessons from visiontouch):
  - YOLOv8 detectors (NudeNet): FP16 engines — safe, 2x throughput.
    `fp32=True` stays available as the escape hatch if label flips appear.
  - TRT >= 11 strongly-typed networks take precision from the ONNX itself,
    hence the modelopt fp16 cast with the fp16 -> mixed -> noconv fallback.
"""

from __future__ import annotations

import logging
import os
import re
from contextlib import nullcontext as _nullcontext
from pathlib import Path

logger = logging.getLogger(__name__)


def gpu_slug() -> str:
    """Stable slug of the current GPU (engines are arch/GPU-model specific)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        name = torch.cuda.get_device_name(0).lower()
    except Exception:
        return "gpu-unknown"
    name = name.replace("nvidia ", "").replace("geforce ", "")
    return re.sub(r"[^a-z0-9]+", "_", name).strip("_")


def trt_slug() -> str:
    """TensorRT version segment: engines only deserialize within one TRT version."""
    try:
        import tensorrt as trt

        slug = f"trt-{trt.__version__}"
    except Exception:
        return "trt-na"
    # TRT records build-time NVIDIA_TF32_OVERRIDE and refuses to create an
    # execution context when the runtime value differs (Myelin inconsistency
    # error). Keep strict-fp32 builds apart in the cache key.
    if os.environ.get("NVIDIA_TF32_OVERRIDE") == "0":
        slug += "-strictfp32"
    return slug


def weights_version(source: str) -> str:
    """``nudenet_640m_v1.onnx`` -> ``v1`` (see config.MODELS naming)."""
    stem = os.path.splitext(os.path.basename(source))[0]
    m = re.search(r"_v(\d+(?:\.\d+)*)", stem)
    return f"v{m.group(1)}" if m else "v0"


def engine_cache_dir(base_dir: str | os.PathLike | None = None) -> Path:
    """Bind-mount-friendly cache root: {root}/{gpu_slug}/{trt_slug}/."""
    from . import config

    root = Path(base_dir) if base_dir else config.TRT_CACHE_DIR
    d = root / gpu_slug() / trt_slug()
    d.mkdir(parents=True, exist_ok=True)
    return d


def engine_path(onnx_path: str) -> str:
    """Versioned engine path inside the cache for a given ONNX source."""
    stem = os.path.splitext(os.path.basename(onnx_path))[0]
    d = engine_cache_dir()
    return str(d / f"{stem}.{weights_version(onnx_path)}.engine")


def prepare_onnx_fp16(onnx_path: str, mode: str) -> str:
    """ONNX-level FP16 conversion for strongly-typed TRT (>=11) networks.

    Modes: 'fp16' (everything), 'mixed', 'noconv' (all Conv in FP32). Falls
    back to the original file when modelopt is unavailable (older TRT with
    the FP16 builder flag instead).
    """
    import onnx

    path = {
        "fp16": onnx_path.replace(".onnx", "_fp16.onnx"),
        "mixed": onnx_path.replace(".onnx", "_mixed.onnx"),
        "noconv": onnx_path.replace(".onnx", "_fp16_noconv.onnx"),
    }[mode]
    if os.path.exists(path):
        return path

    import logging as _logging

    for ln in ("modelopt", "modelopt.onnx", "modelopt.onnx.autocast",
               "modelopt.onnx.autocast.nodeclassifier"):
        lg = _logging.getLogger(ln)
        lg.setLevel(_logging.ERROR)
        lg.propagate = False

    from modelopt.onnx import autocast

    try:
        import onnxslim as slim
    except ImportError:
        slim = None

    block = ["InstanceNormalization"]
    model = onnx.load(onnx_path)
    if slim:
        model = slim.slim(model)

    if mode == "fp16":
        out = autocast.convert_to_f16(model, keep_io_types=False, op_block_list=block)
    elif mode == "mixed":
        excl = []
        for node in model.graph.node:
            if node.op_type == "Conv":
                for a in node.attribute:
                    if a.name == "group":
                        from onnx import helper

                        if helper.get_attribute_value(a) > 1:
                            excl.append(node.name)
                            break
        out = (autocast.convert_to_mixed_precision(
                    onnx_path, keep_io_types=False,
                    nodes_to_exclude=excl, op_types_to_exclude=block)
               if excl else
               autocast.convert_to_f16(model, keep_io_types=False, op_block_list=block))
    else:  # noconv
        out = autocast.convert_to_f16(
            model, keep_io_types=False,
            op_block_list=["Conv", "ConvTranspose"] + block)

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    onnx.save(out, path)
    logger.info("[TRT] ONNX converted (%s): %s", mode, path)
    return path


class TrtEngine:
    """ONNX -> engine (cached) -> executable TensorRT context.

    ``profile``: list of (input_name, (min_shape, opt_shape, max_shape)) for
    dynamic-dim inputs.
    """

    def __init__(self, onnx_path: str, engine_path: str, *, name: str = "engine",
                 fp32: bool = False, workspace: int = 1 << 31,
                 profile: list[tuple[str, tuple]] | None = None):
        import torch  # noqa: F401  (buffers are torch tensors, visiontouch parity)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("TrtEngine requires CUDA")
        self.onnx_path, self.engine_path = onnx_path, engine_path
        self.name, self.fp32, self.workspace, self.profile = name, fp32, workspace, profile

        if not os.path.exists(engine_path):
            self._build()

        import tensorrt as trt

        self._trt = trt
        trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING), "")
        self.logger = trt.Logger(trt.Logger.WARNING)
        with open(engine_path, "rb") as fh:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(fh.read())
        if self.engine is None:
            raise RuntimeError(f"{name}: engine deserialization failed ({engine_path})")
        self.context = self.engine.create_execution_context()
        self.in_names: list[str] = []
        self.out_names: list[str] = []
        for i in range(self.engine.num_io_tensors):
            nm = self.engine.get_tensor_name(i)
            (self.in_names if self.engine.get_tensor_mode(nm) == trt.TensorIOMode.INPUT
             else self.out_names).append(nm)

    # -- build ---------------------------------------------------------------
    def _build(self) -> None:
        trt = __import__("tensorrt")
        strongly_typed = hasattr(trt.NetworkDefinitionCreationFlag, "STRONGLY_TYPED")
        has_fp16_flag = hasattr(trt.BuilderFlag, "FP16")
        modes = ("fp16", "mixed", "noconv")

        for mode in modes:
            label = "FP32" if self.fp32 else f"FP16/{mode}"
            logger.info("[TRT] Building %s engine (%s)...", self.name, label)
            try:
                builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
                flags = 0
                if hasattr(trt.NetworkDefinitionCreationFlag, "EXPLICIT_BATCH"):
                    flags |= 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
                if strongly_typed:
                    flags |= 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
                net = builder.create_network(flags)
                parser = trt.OnnxParser(net, trt.Logger(trt.Logger.WARNING))
                src = self.onnx_path if self.fp32 else prepare_onnx_fp16(self.onnx_path, mode)
                # parse_from_file (not bytes): resolves external .onnx.data files
                if not parser.parse_from_file(src):
                    for i in range(parser.num_errors):
                        logger.error("  ONNX err: %s", parser.get_error(i))
                    raise RuntimeError(f"{self.name}: ONNX parse failed")
                cfg = builder.create_builder_config()
                if not self.fp32:
                    if has_fp16_flag:
                        cfg.set_flag(trt.BuilderFlag.FP16)
                    elif hasattr(trt.BuilderFlag, "TF32"):
                        cfg.set_flag(trt.BuilderFlag.TF32)
                cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, self.workspace)

                if self.profile:
                    prof = builder.create_optimization_profile()
                    for pname, (mn, opt, mx) in self.profile:
                        prof.set_shape(pname, tuple(mn), tuple(opt), tuple(mx))
                        logger.info("[TRT] %s profile %s: min=%s opt=%s max=%s",
                                    self.name, pname, mn, opt, mx)
                    cfg.add_optimization_profile(prof)

                plan = builder.build_serialized_network(net, cfg)
                if plan is None:
                    raise RuntimeError(f"{self.name}: build returned None")
                os.makedirs(os.path.dirname(os.path.abspath(self.engine_path)), exist_ok=True)
                with open(self.engine_path, "wb") as fh:
                    fh.write(plan)
                logger.info("[TRT] %s engine saved: %s", self.name, self.engine_path)
                return
            except Exception as e:
                if mode != modes[-1]:
                    logger.warning("[TRT] %s build failed (%s/%s: %s); trying next mode...",
                                   self.name, self.fp32, mode, e)
                else:
                    raise

    # -- inference -----------------------------------------------------------
    def run(self, inputs: dict[str, "object"], stream=None,
            sync: bool = True) -> dict:
        """One async v3 pass. Values: torch tensors on device. Returns torch
        output tensors keyed by output name (fresh buffers per call — batch
        sizes vary across the scan, reuse would complicate the caller).

        sync=False leaves the pass in flight on its stream: the caller owns
        the join (stream/event synchronize) BEFORE consuming the returned
        buffers. Used by the scan-time person gate to overlap the RF-DETR
        engine with the NudeNet engine on separate streams."""
        import torch

        trt = self._trt
        ctx = torch.cuda.stream(stream) if stream is not None else _nullcontext()
        with ctx:
            current = torch.cuda.current_stream()
            for nm, tensor in inputs.items():
                self.context.set_input_shape(nm, tuple(tensor.shape))
                self.context.set_tensor_address(nm, tensor.data_ptr())
            outputs: dict = {}
            for nm in self.out_names:
                shape = tuple(self.context.get_tensor_shape(nm))
                dtype = torch.float16 if trt.float16 == self.engine.get_tensor_dtype(nm) \
                    else torch.float32
                buf = torch.zeros(shape, dtype=dtype, device=self.device)
                self.context.set_tensor_address(nm, buf.data_ptr())
                outputs[nm] = buf
            self.context.execute_async_v3(current.cuda_stream)
            if sync:
                current.synchronize()
        return outputs
