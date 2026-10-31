"""Full-GPU-native blur render (stage C, production path).

One continuous pass over the movie — there are no cut boundaries to get
wrong (the stream-copy splice approach failed twice on ffmpeg's dts/pts
cut semantics; this design removes the class of failure entirely):

    CreateDemuxer -> CreateDecoder (NVDEC, device memory)
                  -> torch.from_dlpack(frame)        # NV12 view, zero-copy
                  -> in-place block pixelate of Y+UV through the held
                     person-mask (torch ops on the same CUDA surface)
                  -> CreateEncoder (NVENC, P5/hq, VBR + explicit CQ)
                  -> FFmpegMuxer (audio/subs passthrough, bit-exact PTS)

Frames never touch the CPU; the blur is ~2 pooled resamples + a where() on
the GPU per frame, and the warning overlay is two masked_fill_ passes over
precomputed BT.709 NV12 color constants. The demux/decode/encode/mux loop is
the PyNvVideoCodec `advanced/transcode.py` reference with the blur inserted
between Decode() and Encode().
"""

from __future__ import annotations

import logging
import time

import numpy as np

from . import config

logger = logging.getLogger(__name__)


def _torch_mask(mask_np: np.ndarray, device) -> "torch.Tensor":
    import torch
    return torch.from_numpy(mask_np.astype(np.uint8)).to(device).bool()


def render_gpu(video_path: str, out_path: str, segments: list[dict],
               per_seg: dict[int, list], w: int, h: int, aw: int, ah: int,
               cq: int | None = None, pixels: int | None = None) -> dict:
    """-> stats dict. Requires the container's GPU stack (torch + CUDA)."""
    import torch
    import PyNvVideoCodec as nvc
    from .blur import probe_fps
    import bisect
    import cv2

    t_start = time.monotonic()
    cq = cq or config.BLUR_CQ
    pixels = pixels or config.BLUR_PIXELS
    block = max(2, int(round(pixels * w / 1920)))
    src_fps = probe_fps(video_path)
    device = torch.device("cuda")
    torch.cuda.init()

    nv_dmx = nvc.CreateDemuxer(filename=video_path)
    dw, dh = nv_dmx.Width(), nv_dmx.Height()
    if (dw, dh) != (w, h):
        raise RuntimeError(f"demuxer dims {dw}x{dh} != probe {w}x{h}")

    nv_dec = nvc.CreateDecoder(
        gpuid=0, codec=nv_dmx.GetNvCodecId(),
        cudacontext=0, cudastream=0, usedevicememory=1)

    # NVENC with explicit rate control (the standing requirement): P5 +
    # high-quality tuning + VBR at constant quality, no default bitrate.
    # source_av_format_context carries source fps/VUI/container-mode.
    nv_enc = nvc.CreateEncoder(
        w, h, nvc.Pixel_Format.NV12.name, False,
        codec="hevc", gpu_id=0, preset="P5", tuning_info="high_quality",
        rc="vbr", cq=int(cq),
        source_av_format_context=int(nv_dmx.GetAVFormatInputContext()))

    muxer = nvc.FFmpegMuxer(
        file_path=str(out_path), media_format=nvc.GetMediaFormat(str(out_path)),
        in_format_context=nv_dmx.GetAVFormatInputContext(),
        codec="hevc", width=w, height=h,
        extradata=nv_enc.GetSequenceParams())

    seg_bounds = [(float(s["start"]), float(s["end"])) for s in segments]
    starts = [a for a, _ in seg_bounds]
    kf_index = {si: [e[0] for e in entries]
                for si, entries in per_seg.items()}

    # full-res mask cache: (si, ki) -> (y bool (h,w), uv bool (h/2, w))
    mask_cache: dict[tuple, tuple] = {}
    hu = h // 2

    def full_masks(si: int, ki: int):
        key = (si, ki)
        if key not in mask_cache:
            packed = np.frombuffer(per_seg[si][ki][1], dtype=np.uint8)
            m = np.unpackbits(packed, count=ah * aw).reshape(ah, aw)
            m = cv2.resize(m.astype(np.uint8), (w, h),
                           interpolation=cv2.INTER_NEAREST).astype(bool)
            my = _torch_mask(m, device)
            m2 = (m[0::2, 0::2] | m[1::2, 0::2]
                  | m[0::2, 1::2] | m[1::2, 1::2])          # 2x2 union
            muv = _torch_mask(np.repeat(m2, 2, axis=1), device)
            if len(mask_cache) > 8:          # bounded: keyframes are local
                mask_cache.clear()
            mask_cache[key] = (my, muv)
        return mask_cache[key]

    def pixelate_yuv(t: torch.Tensor, my: torch.Tensor,
                     muv: torch.Tensor) -> None:
        y, uv = t[:h], t[h:]
        fy = y.unsqueeze(0).unsqueeze(0).float()
        pooled = torch.nn.functional.avg_pool2d(fy, block)
        up = torch.nn.functional.interpolate(
            pooled, size=(h, w), mode="nearest")
        y.copy_(torch.where(my, up.squeeze(0).squeeze(0).round().clamp(0, 255),
                             y).to(torch.uint8))
        fuv = uv.unsqueeze(0).unsqueeze(0).float()
        pooled_u = torch.nn.functional.avg_pool2d(fuv, max(2, block // 2))
        up_u = torch.nn.functional.interpolate(
            pooled_u, size=(hu, w), mode="nearest")
        uv.copy_(torch.where(
            muv, up_u.squeeze(0).squeeze(0).round().clamp(0, 255),
            uv).to(torch.uint8))

    # warning overlay, NV12 / BT.709 limited range: pure red -> Y 63 Cb 102
    # Cr 240; black outline -> Y 16 Cb 128 Cr 128. Cached per (si, ki).
    RED_Y, RED_CB, RED_CR = 63, 102, 240
    BLK_Y, BLK_CB, BLK_CR = 16, 128, 128

    def overlay_masks(si: int, ki: int):
        """-> (red_y, blk_y, red_uv, blk_uv) bool tensors, or None when the
        overlay is disabled. Masks are derived from a sentinel-colored
        canvas (fill R-channel, outline B-channel) rendered once per
        keyframe entry — the label/score ride in the analysis entries."""
        if not config.BLUR_OVERLAY:
            return None
        key = ("overlay", si, ki)
        if key not in mask_cache:
            from .blur import draw_warning
            entry = per_seg[si][ki]
            canvas = np.zeros((h, w, 3), dtype=np.uint8)
            draw_warning(canvas, entry[2], entry[3],
                         fill=(255, 0, 0), outline=(0, 0, 255))
            red = canvas[..., 0] > 127
            blk = canvas[..., 2] > 127

            def uv_of(m: np.ndarray) -> torch.Tensor:
                m2 = (m[0::2, 0::2] | m[1::2, 0::2]
                      | m[0::2, 1::2] | m[1::2, 1::2])   # 2x2 union
                return _torch_mask(np.repeat(m2, 2, axis=1), device)

            mask_cache[key] = (_torch_mask(red, device),
                               _torch_mask(blk, device),
                               uv_of(red), uv_of(blk))
        return mask_cache[key]

    def composite_overlay(t: torch.Tensor, ov) -> None:
        ry, by, ruv, buv = ov
        y, uv = t[:h], t[h:]
        y.masked_fill_(by, BLK_Y)
        y.masked_fill_(ry, RED_Y)
        cb, cr = uv[:, 0::2], uv[:, 1::2]
        cb.masked_fill_(buv[:, 0::2], BLK_CB)
        cr.masked_fill_(buv[:, 1::2], BLK_CR)
        cb.masked_fill_(ruv[:, 0::2], RED_CB)
        cr.masked_fill_(ruv[:, 1::2], RED_CR)

    frames = blurred = 0
    held = {"si": -1, "ki": -1}
    last_progress = time.monotonic()
    try:
        while True:
            if time.monotonic() - last_progress > config.BLUR_STALL_TIMEOUT_S:
                raise RuntimeError("GPU decode stalled %.0fs — aborting"
                                   % config.BLUR_STALL_TIMEOUT_S)
            pkt = nv_dmx.DemuxNoSkipAudio()
            if pkt.is_video or pkt.bsl == 0:
                muxer.RecordSourceVideoPacket(pkt)
                for frame in nv_dec.Decode(pkt):
                    last_progress = time.monotonic()
                    t_sec = frames / src_fps
                    si = bisect.bisect_right(starts, t_sec) - 1
                    if (0 <= si < len(seg_bounds)
                            and seg_bounds[si][0] <= t_sec <= seg_bounds[si][1]
                            and per_seg.get(si)):
                        ki = max(0, bisect.bisect_right(kf_index[si], t_sec) - 1)
                        if held["si"] != si or held["ki"] != ki:
                            held = {"si": si, "ki": ki}
                        surf = torch.from_dlpack(frame)
                        pixelate_yuv(surf, *full_masks(si, ki))
                        ov = overlay_masks(si, ki)
                        if ov is not None:
                            composite_overlay(surf, ov)
                        # our kernels are async on torch's stream; NVENC
                        # reads the surface without waiting for that queue
                        # — without this sync the race drops the tail of
                        # the queue (observed: overlay flickering off on
                        # ~half the flagged frames)
                        torch.cuda.synchronize()
                        blurred += 1
                    pp = nvc.NV_ENC_PIC_PARAMS()
                    pp.inputTimeStamp = frames
                    frames += 1
                    for enc_pkt in nv_enc.Encode(frame, pp):
                        muxer.MuxVideoPacket(bytes(enc_pkt["data"]),
                                             enc_pkt["picture_type"],
                                             enc_pkt["timestamp"])
                if pkt.bsl == 0:
                    break
            else:
                muxer.MuxPassthroughPacket(pkt)

        for enc_pkt in nv_enc.EndEncode():
            muxer.MuxVideoPacket(bytes(enc_pkt["data"]),
                                 enc_pkt["picture_type"],
                                 enc_pkt["timestamp"])
        muxer.Finalize()
    finally:
        del muxer, nv_enc, nv_dec, nv_dmx
        torch.cuda.empty_cache()

    wall = time.monotonic() - t_start
    logger.info("[blur:gpu] %d frames (%d blurred, %.1f%%) in %.1fs "
                "(%.1fx realtime)", frames, blurred,
                100 * blurred / max(frames, 1), wall,
                frames / src_fps / max(wall, 0.01))
    return {"frames": frames, "frames_blurred": blurred,
            "codec": "hevc_nvenc", "transport": "gpu-native",
            "wall_time_s": round(wall, 2)}
