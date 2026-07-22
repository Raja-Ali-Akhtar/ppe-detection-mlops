"""Config-driven YOLO training with MLflow tracking.

Usage:
    python scripts/train.py --config configs/baseline.yaml

Every run is defined by a config file — no hyperparameters on the command line
(CLI overrides don't leave a paper trail; config files do). Ultralytics' MLflow
integration logs params, per-epoch metrics, and weight artifacts automatically.
Start the UI with `mlflow ui` and open http://127.0.0.1:5000 to compare runs.
"""

import argparse
import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="path to a training config yaml")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    # local tracking: file store in mlruns/ (gitignored). Stage 3 swaps this
    # URI for a remote server — nothing else changes.
    os.environ.setdefault("MLFLOW_TRACKING_URI", (ROOT / "mlruns").resolve().as_uri())
    os.environ.setdefault("MLFLOW_EXPERIMENT_NAME", cfg.get("project", "ppe-detection"))

    from ultralytics import YOLO, settings

    settings.update({"mlflow": True})

    data_path = ROOT / cfg["data"]
    model = YOLO(cfg["model"])
    model.train(
        data=str(data_path),
        epochs=cfg["epochs"],
        imgsz=cfg["imgsz"],
        batch=cfg["batch"],
        patience=cfg.get("patience", 50),
        seed=cfg.get("seed", 0),
        project=cfg.get("project", "ppe-detection"),
        name=cfg.get("name"),
    )


if __name__ == "__main__":
    main()
