"""Full-GPU-native blackout render (stage C, production path).

One continuous pass over the movie — there are no cut boundaries to get
wrong (the stream-copy splice approach failed twice on ffmpeg's dts/pts
cut semantics; this design removes the class of failure entirely):

    CreateDemuxer -> CreateDecoder (NVDEC, device memory)
                  -> torch.from_dlpack(frame)        # NV12/P016 view, zero-copy
                  -> in-place solid-black fill of Y+UV through the held
                     person-mask (torch ops on the same CUDA surface)
                  -> CreateEncoder (NVENC, P5/hq, VBR + explicit CQ)
                  -> FFmpegMuxer (audio passthrough, bit-exact PTS; subtitle-bearing
    sources are sub-stripped upstream — see mask.run_mask)

Frames never touch the CPU; the mask is ~2 pooled resamples + a where() on
the GPU per frame, and the warning overlay is a handful of masked where()
passes over precomputed BT.709 NV12 color constants. The demux/decode/encode/mux loop is
the PyNvVideoCodec `advanced/transcode.py` reference with the mask inserted
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
               cq: int | None = None, seg_ready=None) -> dict:
    """-> stats dict. Requires the container's GPU stack (torch + CUDA).

    seg_ready(si) blocks until segment si's analysis entries are final
    (analysis/render overlap); None means per_seg is complete up front.
    Either way a segment is only touched once the render head enters its
    bounds, so with overlap the render starts as soon as segment 0 is
    done and waits never outlast the analysis lead."""
    import torch
    import PyNvVideoCodec as nvc
    from .mask import probe_fps, probe_pix_fmt
    import bisect
    import cv2

    t_start = time.monotonic()
    cq = cq or config.MASK_CQ
    src_fps = probe_fps(video_path)
    # Bit-depth adapter: 8-bit sources decode to NV12 surfaces, 10-bit
    # (HEVC main10 etc.) to P016 — same Y + interleaved-UV layout, but
    # uint16 words with samples MSB-aligned (10-bit << 6). The zero-copy
    # path works for both: NVENC encodes main10 and every depth-dependent
    # constant flows from here. The encoder must be created BEFORE the
    # first decode, so the depth comes from the probe, not the surface.
    # (Found by Gone Girl's 10-bit WEBRip: NV12-forced NVENC aborted the
    # process on the first header write — 'Error writing header'.)
    ten_bit = "10" in probe_pix_fmt(video_path)
    # NVENC's 10-bit format is "P010"; the binding's parser rejects the
    # P016 enum name ("Unknown format") even though NVDEC labels its main10
    # surfaces P016 — same Y + interleaved-UV layout, 16-bit words,
    # samples MSB-aligned (10-bit << 6).
    enc_pf = "P010" if ten_bit else nvc.Pixel_Format.NV12.name
    word_dtype = torch.uint16 if ten_bit else torch.uint8
    lvl = (lambda v: v << 8) if ten_bit else (lambda v: v)  # 8-bit level -> word

    # PyNvVideoCodec 2.2.3 consumes Encode() inputs by calling
    # Tensor.__dlpack__(stream) POSITIONALLY; torch 2.14's is keyword-only
    # and raises TypeError. Bridge once, process-wide — no-arg and keyword
    # callers keep working unchanged. (Probed: without this, NV12 tensors
    # die on "takes 1 positional argument" and the encoder falls back to
    # "incorrect usage of CPU input buffer" for anything it can't sniff.)
    if not getattr(torch.Tensor.__dlpack__, "_posarg_ok", False):
        _orig_dlpack = torch.Tensor.__dlpack__

        def _dlpack_posarg(self, *args, **kwargs):
            return _orig_dlpack(
                self, stream=args[0] if args else kwargs.get("stream"))

        _dlpack_posarg._posarg_ok = True
        torch.Tensor.__dlpack__ = _dlpack_posarg
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
        w, h, enc_pf, False,
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
    # keyframe times per segment, built lazily: with the analysis/render
    # overlap per_seg fills segment-by-segment just ahead of the render
    # head, so a segment's index may only be built (and seg_ready waited
    # on) once the head actually reaches that segment
    kf_index: dict[int, list] = {}

    def seg_entries(si: int) -> list:
        """Segment si's analysis entries — blocking for them when the
        analysis runs concurrently, and caching the keyframe-time index.
        -> [] when the segment produced no masks (frames copy through)."""
        if si not in kf_index:
            if seg_ready is not None:
                seg_ready(si)
            kf_index[si] = [e[0] for e in (per_seg.get(si) or [])]
        return per_seg.get(si) or []

    # full-res mask cache: (si, ki) -> (y bool (h,w), uv bool (h/2, w))
    mask_cache: dict[tuple, tuple] = {}

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

    # blackout fill levels: BT.709 limited-range black Y=16 with neutral
    # chroma Cb=Cr=128 — the same convention the overlay outline uses, so
    # 8- and 10-bit words both come out of lvl().
    MASK_BLACK_Y = lvl(16)
    MASK_NEUTRAL_UV = lvl(128)

    def blackout_yuv(t: torch.Tensor, my: torch.Tensor,
                     muv: torch.Tensor) -> None:
        """Solid black through the person silhouette mask. The mosaic this
        replaced kept silhouettes (and bright-scene detail) legible; a
        constant fill leaves nothing. promote -> where -> cast, the same
        UInt16-safe path the overlay fill uses."""
        y, uv = t[:h], t[h:]
        y.copy_(torch.where(
            my, torch.full_like(my, MASK_BLACK_Y, dtype=torch.float32),
            y.float()).to(word_dtype))
        uv.copy_(torch.where(
            muv, torch.full_like(muv, MASK_NEUTRAL_UV, dtype=torch.float32),
            uv.float()).to(word_dtype))

    # warning overlay, NV12 / BT.709 limited range: pure red -> Y 63 Cb 102
    # Cr 240; black outline -> Y 16 Cb 128 Cr 128. Cached per (si, ki).
    # lvl() lifts the 8-bit levels into P016 words on 10-bit sources
    # (v << 8; the 0.2% rounding vs exact 1023/255 scaling is invisible).
    RED_Y, RED_CB, RED_CR = lvl(63), lvl(102), lvl(240)
    BLK_Y, BLK_CB, BLK_CR = lvl(16), lvl(128), lvl(128)

    def overlay_masks(si: int, ki: int):
        """-> (red_y, blk_y, red_uv, blk_uv) bool tensors, or None when the
        overlay is disabled. Masks are derived from a sentinel-colored
        canvas (fill R-channel, outline B-channel) rendered once per
        keyframe entry — the label/score ride in the analysis entries."""
        if not config.MASK_OVERLAY:
            return None
        key = ("overlay", si, ki)
        if key not in mask_cache:
            from .mask import draw_warning
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
        # masked_fill_ is not implemented for UInt16 CUDA (torch 2.14 op
        # gap — same family as remainder/min/max) — fill via the same
        # promote -> where -> cast path blackout_yuv uses
        def fill(x: torch.Tensor, mask: torch.Tensor, value: int) -> None:
            x.copy_(torch.where(mask, torch.full_like(mask, value,
                                                      dtype=torch.float32),
                                x.float()).to(word_dtype))

        fill(y, by, BLK_Y)
        fill(y, ry, RED_Y)
        cb, cr = uv[:, 0::2], uv[:, 1::2]
        fill(cb, buv[:, 0::2], BLK_CB)
        fill(cr, buv[:, 1::2], BLK_CR)
        fill(cb, ruv[:, 0::2], RED_CB)
        fill(cr, ruv[:, 1::2], RED_CR)

    frames = masked = 0
    held = {"si": -1, "ki": -1}
    last_progress = time.monotonic()
    try:
        while True:
            if time.monotonic() - last_progress > config.MASK_STALL_TIMEOUT_S:
                raise RuntimeError("GPU decode stalled %.0fs — aborting"
                                   % config.MASK_STALL_TIMEOUT_S)
            pkt = nv_dmx.DemuxNoSkipAudio()
            if pkt.is_video or pkt.bsl == 0:
                muxer.RecordSourceVideoPacket(pkt)
                for frame in nv_dec.Decode(pkt):
                    last_progress = time.monotonic()
                    t_sec = frames / src_fps
                    if ten_bit:
                        # every frame takes this path, not just masked
                        # ones — Encode() rejects the raw dlpack frame (its
                        # export carries u16 geometry on a u1 typestr, and
                        # the binding validates typestr). Re-stride to the
                        # true word layout for the kernels, then hand NVENC
                        # the u8 HWC view of the same memory that its P010
                        # stride check expects.
                        t0 = torch.from_dlpack(frame)
                        surf = torch.as_strided(t0, (h * 3 // 2, w), (w, 1))
                        enc_in = surf.unsqueeze(-1).view(torch.uint8)
                    else:
                        enc_in = frame
                    si = bisect.bisect_right(starts, t_sec) - 1
                    if (0 <= si < len(seg_bounds)
                            and seg_bounds[si][0] <= t_sec <= seg_bounds[si][1]
                            and seg_entries(si)):
                        ki = max(0, bisect.bisect_right(kf_index[si], t_sec) - 1)
                        if held["si"] != si or held["ki"] != ki:
                            held = {"si": si, "ki": ki}
                        if not ten_bit:
                            surf = torch.from_dlpack(frame)
                        blackout_yuv(surf, *full_masks(si, ki))
                        ov = overlay_masks(si, ki)
                        if ov is not None:
                            composite_overlay(surf, ov)
                        # our kernels are async on torch's stream; NVENC
                        # reads the surface without waiting for that queue
                        # — without this sync the race drops the tail of
                        # the queue (observed: overlay flickering off on
                        # ~half the flagged frames)
                        torch.cuda.synchronize()
                        masked += 1
                    pp = nvc.NV_ENC_PIC_PARAMS()
                    pp.inputTimeStamp = frames
                    frames += 1
                    for enc_pkt in nv_enc.Encode(enc_in, pp):
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
    logger.info("[mask:gpu] %d frames (%d masked, %.1f%%) in %.1fs "
                "(%.1fx realtime)", frames, masked,
                100 * masked / max(frames, 1), wall,
                frames / src_fps / max(wall, 0.01))
    return {"frames": frames, "frames_masked": masked,
            "codec": "hevc_nvenc", "transport": "gpu-native",
            "render_wall_s": round(wall, 2)}
