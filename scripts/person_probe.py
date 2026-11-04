"""Person-pass probe: run ONLY the RF-DETR gate over a movie at 2fps.

Mirrors the scan loop's gate wiring (PersonGateRunner + open_decoder +
per-batch sync) standalone, so gate recall can be
audited on the GPU before a full scan is committed to:

  host> docker compose exec -T movie-trimmer \
          timeout -s KILL 2400 python /movies/scripts/person_probe.py \
          /movies/<movie>.mkv > person_probe.json

Prints one JSON object: {duration, eff_fps, frames_fed, wall_time_s,
gate: <PersonGateRunner.finish() stats block>} (gate.ranges included).
"""

from __future__ import annotations

import json
import sys
import time

# HOST scanner (mounted read-only at /movies): run the probe against the
# working-tree code — the fix under test — instead of the baked /app copy.
# TRT_CACHE_DIR points at the container's writable engine cache.
sys.path.insert(0, "/movies")

from scanner import config          # noqa: E402
from scanner.decode import (analysis_size, open_decoder, probe_dimensions,  # noqa: E402
                            probe_duration)
from scanner.persongate import PersonGateRunner  # noqa: E402


def main() -> int:
    video = sys.argv[1]
    duration = probe_duration(video)
    w0, h0 = probe_dimensions(video)
    aw, ah = analysis_size(w0, h0, config.DECODE_WIDTH)

    gate = PersonGateRunner(backend="trt")
    t0 = time.monotonic()
    frames_fed = 0
    decoder, eff_fps = open_decoder(video, fps=config.PERSON_GATE_FPS,
                                    width=aw, height=ah)
    stride = max(1, round(eff_fps / max(config.PERSON_GATE_FPS, 0.1)))
    try:
        with decoder:
            batch: list = []
            first = 0
            for frame in decoder.frames():
                batch.append(frame)
                if len(batch) >= 32:
                    gate.feed_batch(batch, first, eff_fps, stride)
                    gate.sync()
                    frames_fed += len(batch)
                    first += len(batch)
                    batch = []
            if batch:
                gate.feed_batch(batch, first, eff_fps, stride)
                gate.sync()
                frames_fed += len(batch)
    finally:
        decoder.close()

    ranges, stats = gate.finish(duration)
    stats["probe_wall_time_s"] = round(time.monotonic() - t0, 1)
    stats["frames_fed"] = frames_fed
    stats["eff_fps"] = eff_fps
    stats["stride"] = stride
    print(json.dumps({
        "video": video.rsplit("/", 1)[-1],
        "duration": duration,
        "gate": stats,
        "ranges": [[round(a, 3), round(b, 3)] for a, b in (ranges or [])],
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
