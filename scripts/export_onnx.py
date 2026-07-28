"""Export the registered champion model to ONNX, tracked in MLflow.

Usage:
    python scripts/export_onnx.py --config configs/export-onnx.yaml

The model comes from the MLflow Registry BY ALIAS — this script contains no
run ids and no weight paths. Optimization work is experiment work: the export
is logged as a run in the `ppe-optimization` experiment (params + onnx artifact),
so every downstream engine traces back to an exact, reproducible export.
"""

import argparse
import os
import shutil
from pathlib import Path

import mlflow
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    os.environ["MLFLOW_EXPERIMENT_NAME"] = "ppe-optimization"
    mlflow.set_experiment("ppe-optimization")

    weights = mlflow.artifacts.download_artifacts(cfg["model_uri"])
    print(f"champion resolved: {weights}")

    from ultralytics import YOLO, settings
    settings.update({"mlflow": False})  # we log manually here — one run, our fields

    model = YOLO(weights)
    onnx_path = model.export(
        format="onnx",
        imgsz=cfg["imgsz"],
        dynamic=cfg["dynamic"],
        opset=cfg["opset"],
        simplify=cfg["simplify"],
        half=cfg["half"],
    )

    out_dir = ROOT / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "dynamic" if cfg["dynamic"] else "static"
    dest = out_dir / f"ppe-detector-v1-{suffix}.onnx"
    shutil.move(onnx_path, dest)

    with mlflow.start_run(run_name=f"export-onnx-{suffix}"):
        mlflow.log_params({
            "source_model": cfg["model_uri"],
            "imgsz": cfg["imgsz"],
            "dynamic": cfg["dynamic"],
            "opset": cfg["opset"],
            "simplify": cfg["simplify"],
        })
        mlflow.log_artifact(str(dest))
        size_mb = dest.stat().st_size / 1e6
        mlflow.log_metric("onnx_size_mb", round(size_mb, 2))

    print(f"exported: {dest} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
