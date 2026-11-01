#!/usr/bin/env python3
"""Run the all-in-one Movie-Trimmer container under a definite name.

  python run.py                 # run the published image as 'movie-trimmer'
  python run.py --build         # rebuild scanner + allinone images first
  python run.py --wait 900      # block until /health answers (seconds cap)

The container runs detached with --rm: `docker stop movie-trimmer`
removes it completely — no compose project names, no orphaned
containers. Volumes/env mirror docker-compose.yml exactly (engine
cache, repo read-only at /movies, blur outputs).
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build import (ALLINONE as DEFAULT_IMAGE,  # noqa: E402
                   SCANNER as SCANNER_IMAGE, build)

ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_NAME = "movie-trimmer"
PORT = 8081


def run(*cmd: str) -> None:
    subprocess.run(cmd, check=True)


def container_exists(name: str) -> bool:
    out = subprocess.run(
        ["docker", "ps", "-aq", "--filter", f"name=^/{name}$"],
        capture_output=True, text=True).stdout.strip()
    return bool(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--build", action="store_true",
                    help="rebuild the scanner + allinone images first")
    ap.add_argument("--image", default=DEFAULT_IMAGE)
    ap.add_argument("--name", default=DEFAULT_NAME)
    ap.add_argument("--wait", type=int, default=0, metavar="SECONDS",
                    help="poll /health until it answers, cap in seconds")
    args = ap.parse_args()

    if args.build:
        build("scanner.Dockerfile", SCANNER_IMAGE)
        print("[run] building allinone image...")
        build("allinone.Dockerfile", args.image)

    if container_exists(args.name):
        print(f"[run] stopping + removing previous '{args.name}'...")
        run("docker", "rm", "-f", args.name)

    run("docker", "run", "-d", "--rm",
        "--name", args.name,
        "--init",
        "--gpus", "all",
        "-p", f"{PORT}:{PORT}",
        "-v", f"{ROOT}/trt_cache:/cache/trt",
        "-v", f"{ROOT}:/movies:ro",
        "-v", f"{ROOT}/blur_outputs:/blur_outputs",
        "-e", "ARBITER_HTTP_CONCURRENCY=4",
        "-e", "BLUR_OUTPUT_DIR=/blur_outputs",
        args.image)
    print(f"[run] {args.name} up — http://localhost:{PORT}/health")

    if args.wait > 0:
        print(f"[run] waiting for health (cap {args.wait}s)...")
        t0 = time.monotonic()
        while time.monotonic() - t0 < args.wait:
            try:
                urllib.request.urlopen(
                    f"http://localhost:{PORT}/health", timeout=3)
                print("[run] healthy.")
                return 0
            except Exception:
                time.sleep(5)
        print("[run] health wait TIMED OUT", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
