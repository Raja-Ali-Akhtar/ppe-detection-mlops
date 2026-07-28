"""Register the winning run's weights in the MLflow Model Registry.

Usage:
    python scripts/register_model.py --run-name exp-yolov8s --alias baseline-champion

Promotion is a committed, reviewable act: which run, which artifact, which alias.
Downstream (export, Triton packaging) pulls `models:/ppe-detector@<alias>` and
never touches a file path.
"""

import argparse
from pathlib import Path

import mlflow
from mlflow import MlflowClient

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "ppe-detector"


def find_weights_artifact(client: MlflowClient, run_id: str) -> str:
    """Locate best.pt among the run's artifacts — verify, never assume."""
    def walk(path=""):
        for f in client.list_artifacts(run_id, path or None):
            if f.is_dir:
                yield from walk(f.path)
            else:
                yield f.path

    paths = list(walk())
    best = [p for p in paths if p.endswith("best.pt")]
    if not best:
        raise SystemExit(f"no best.pt among artifacts of run {run_id}: {paths}")
    return best[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--alias", required=True)
    args = parser.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    client = MlflowClient()

    runs = mlflow.search_runs(experiment_names=["ppe-detection"])
    match = runs.loc[runs["tags.mlflow.runName"] == args.run_name]
    if match.empty:
        raise SystemExit(f"no run named {args.run_name}")
    run_id = match.iloc[0].run_id

    artifact = find_weights_artifact(client, run_id)
    print(f"run {run_id}: registering {artifact}")

    # mlflow 3's register_model() only accepts log_model()-style "logged models";
    # ultralytics logs plain file artifacts -> use the lower-level version API,
    # which registers any artifact URI directly
    try:
        client.create_registered_model(MODEL_NAME)
    except mlflow.exceptions.MlflowException:
        pass  # name already exists

    source = f"{client.get_run(run_id).info.artifact_uri}/{artifact}"
    mv = client.create_model_version(MODEL_NAME, source, run_id=run_id)
    client.set_registered_model_alias(MODEL_NAME, args.alias, mv.version)
    print(f"{MODEL_NAME} v{mv.version} @ {args.alias}  <-  {source}")


if __name__ == "__main__":
    main()
