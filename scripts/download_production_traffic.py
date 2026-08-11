"""Download a SECOND, different PPE dataset to act as simulated production traffic.

Usage:
    python scripts/download_production_traffic.py

Stage 4 needs images the model has never seen — a different site, different
cameras, different labelling conventions. This is the "new customer's camera
feed" the deployed model would meet in reality, and the source of both the
hard cases and any measurable drift.

Its labels are used ONLY as a review aid during curation; the flywheel treats
these images as unlabelled production data.
"""

import os
import sys
from pathlib import Path

from roboflow import Roboflow

WORKSPACE = "objet-detect-yolov5"
PROJECT = "eep_detection-u9bbd"
VERSION = 1
DEST = Path(__file__).resolve().parents[1] / "data" / "production"


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        env = Path(__file__).resolve().parents[1] / ".env"
        if env.exists():
            for line in env.read_text().splitlines():
                if line.startswith("ROBOFLOW_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY not set")

    if DEST.exists():
        sys.exit(f"{DEST} already exists — delete it first to re-download")

    rf = Roboflow(api_key=api_key)
    ds = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION).download(
        "yolov8", location=str(DEST)
    )
    n = len(list(Path(ds.location).rglob("*.jpg")))
    print(f"downloaded {n} production-traffic images to {ds.location}")


if __name__ == "__main__":
    main()
