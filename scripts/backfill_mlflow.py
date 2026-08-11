"""Backfill an Ultralytics run into MLflow from its on-disk artifacts.

    python scripts/backfill_mlflow.py --run v1-yolov8s-b4

WHY THIS EXISTS: mid-Stage-4 an Evidently install downgraded pydantic, which
silently broke Ultralytics' MLflow callback — a training run executed perfectly
but was never tracked. Rather than throw away 100 epochs of GPU time, this
reconstructs the run from what Ultralytics always writes to disk:

    runs/<project>/<name>/args.yaml     every hyperparameter
    runs/<project>/<name>/results.csv   per-epoch metrics
    runs/<project>/<name>/weights/      best.pt / last.pt

The lesson worth keeping: a tracking system that fails SILENTLY is worse than
one that crashes. The artifacts on disk are what made recovery possible.
"""

import argparse
from pathlib import Path

import mlflow
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory name, e.g. v1-yolov8s-b4")
    ap.add_argument("--project", default="ppe-detection")
    args = ap.parse_args()

    d = ROOT / "runs" / args.project / args.run
    if not (d / "results.csv").exists():
        raise SystemExit(f"no results.csv in {d} — is the run finished?")

    cfg = yaml.safe_load((d / "args.yaml").read_text())
    df = pd.read_csv(d / "results.csv")
    df.columns = [c.strip() for c in df.columns]

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment(args.project)

    with mlflow.start_run(run_name=args.run):
        mlflow.log_params({k: cfg[k] for k in
                           ("model", "data", "epochs", "imgsz", "batch", "seed",
                            "patience", "optimizer", "lr0") if k in cfg})
        mlflow.set_tag("backfilled", "true")
        mlflow.set_tag("backfill_reason", "pydantic downgrade broke the MLflow callback")

        for _, row in df.iterrows():
            step = int(row["epoch"])
            for col in df.columns:
                if col == "epoch":
                    continue
                try:
                    v = float(row[col])
                except (TypeError, ValueError):
                    continue
                # match the metric names the live callback uses
                name = col if col.startswith("metrics/") else col.replace("/", "_")
                mlflow.log_metric(name.replace("(B)", "B"), v, step=step)

        for w in ("best.pt", "last.pt"):
            p = d / "weights" / w
            if p.exists():
                mlflow.log_artifact(str(p), artifact_path="weights")

        best = df["metrics/mAP50(B)"].max() if "metrics/mAP50(B)" in df else float("nan")
        print(f"backfilled {args.run}: {len(df)} epochs, best mAP50 {best:.4f}")


if __name__ == "__main__":
    main()
