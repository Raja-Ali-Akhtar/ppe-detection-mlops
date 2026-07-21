"""Download the PPE dataset from Roboflow Universe in YOLO format.

Usage:
    set ROBOFLOW_API_KEY=xxx   (or put it in a .env file)
    python scripts/download_dataset.py

Dataset: Construction Site Safety (roboflow-universe-projects), CC BY 4.0.
Classes include: Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person, ...
The raw download lands in data/raw/ (gitignored; DVC will track it).
"""

import os
import sys
from pathlib import Path

from roboflow import Roboflow

WORKSPACE = "roboflow-universe-projects"
PROJECT = "construction-site-safety"
VERSION = 30  # pin the version so the download is reproducible
DEST = Path(__file__).resolve().parents[1] / "data" / "raw"


def main() -> None:
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        env_file = Path(__file__).resolve().parents[1] / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("ROBOFLOW_API_KEY="):
                    api_key = line.split("=", 1)[1].strip()
    if not api_key:
        sys.exit("ROBOFLOW_API_KEY not set (env var or .env file)")

    DEST.mkdir(parents=True, exist_ok=True)
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(VERSION).download("yolov8", location=str(DEST))
    print(f"Downloaded to {dataset.location}")


if __name__ == "__main__":
    main()
