"""Pre-download model weights (used during the Docker build).

Running this once bakes the CLIP / BLIP / ResNet-50 weights into the image so
the first request after deployment is already warm instead of downloading.
"""

import sys

from core import pipeline


def main() -> None:
    print("Prefetching model weights…", flush=True)
    pipeline.warmup()
    print("Done. Models ready:", pipeline.status(), flush=True)


if __name__ == "__main__":
    sys.exit(main())
