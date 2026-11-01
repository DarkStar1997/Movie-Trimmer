#!/usr/bin/env python3
"""Build the Movie-Trimmer images.

  python3 build.py            # scanner + allinone (the common case)
  python3 build.py --vlm      # also rebuild the VLM sidecar image
  python3 build.py --push     # build, then push to GHCR
  python3 build.py --no-cache # docker build --no-cache

Build order matters: allinone composes FROM the scanner image (and the
vlm image), so those build first. Tags are the pinned :v1 set the
deployment expects (docker-compose.yml / run.py).
"""
from __future__ import annotations

import argparse
import subprocess
import sys

REGISTRY = "ghcr.io/darkstar1997"
SCANNER = f"{REGISTRY}/movie-trimmer-scanner:v1"
VLM = f"{REGISTRY}/movie-trimmer-vlm-qwen3vl-30b-a3b:v1"
ALLINONE = f"{REGISTRY}/movie-trimmer:v1"

VLM_ARGS = [
    "--build-arg", "MODEL_DIR=models/qwen3-vl-30b-a3b",
    "--build-arg", "MODEL_FILE=Qwen3-VL-30B-A3B-Instruct-UD-Q4_K_XL.gguf",
    "--build-arg", "MMPROJ_FILE=mmproj-BF16.gguf",
]


def build(dockerfile: str, tag: str, extra: list[str] | None = None,
          no_cache: bool = False) -> None:
    cmd = ["docker", "build", "-f", dockerfile, "-t", tag]
    if no_cache:
        cmd.append("--no-cache")
    cmd += extra or []
    cmd.append(".")
    print(f"[build] {dockerfile} -> {tag}")
    subprocess.run(cmd, check=True)


def push(tag: str) -> None:
    print(f"[push] {tag}")
    subprocess.run(["docker", "push", tag], check=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--vlm", action="store_true",
                    help="also rebuild the VLM sidecar image")
    ap.add_argument("--push", action="store_true",
                    help="push built images to GHCR")
    ap.add_argument("--no-cache", action="store_true",
                    help="pass --no-cache to docker build")
    args = ap.parse_args()

    try:
        build("scanner.Dockerfile", SCANNER, no_cache=args.no_cache)
        if args.vlm:
            build("vlm.Dockerfile", VLM, extra=VLM_ARGS,
                  no_cache=args.no_cache)
        build("allinone.Dockerfile", ALLINONE, no_cache=args.no_cache)
        if args.push:
            for tag in ([SCANNER, VLM, ALLINONE] if args.vlm
                        else [SCANNER, ALLINONE]):
                push(tag)
    except subprocess.CalledProcessError as e:
        print(f"[build] FAILED (rc={e.returncode})", file=sys.stderr)
        return e.returncode
    print("[build] done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
