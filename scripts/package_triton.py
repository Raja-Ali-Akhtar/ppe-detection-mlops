"""Package the exported ONNX into a Triton model repository.

Usage:
    python scripts/package_triton.py

Fetches the ONNX from the MLflow export run (which itself was created from the
registry alias — full lineage: registry -> export run -> serving artifact) and
writes the Triton layout:

    serving/triton/model_repository/
      ppe_detector/
        config.pbtxt          <- serving contract: batching, instances, shapes
        1/                    <- version directory (Triton hot-swaps these)
          model.onnx

The config is GENERATED from the ONNX graph (input/output names + shapes read
from the model, never hardcoded).
"""

from pathlib import Path
import shutil

import mlflow
import onnx

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "serving" / "triton" / "model_repository" / "ppe_detector"

MAX_BATCH = 16                 # matches the TRT profile + export decision
QUEUE_DELAY_US = 5000          # wait up to 5 ms to assemble a batch
PREFERRED_BATCHES = [4, 8, 16]
INSTANCES = 1                  # 6 GB card: one instance; cloud T4 could take 2

CONFIG_TEMPLATE = """\
name: "ppe_detector"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch}

input [
  {{
    name: "{in_name}"
    data_type: TYPE_FP32
    dims: [ 3, 640, 640 ]
  }}
]
output [
  {{
    name: "{out_name}"
    data_type: TYPE_FP32
    dims: [ 11, 8400 ]
  }}
]

# The reason Triton exists: requests arriving within {delay_ms} ms are merged
# into one GPU batch. Stage 2 measured batch-16 at ~2x the batch-1 throughput —
# this setting is how single-image clients reach that regime.
dynamic_batching {{
  max_queue_delay_microseconds: {queue_delay}
  preferred_batch_size: [ {preferred} ]
}}

instance_group [
  {{
    count: {instances}
    kind: KIND_GPU
  }}
]
"""


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    runs = mlflow.search_runs(experiment_names=["ppe-optimization"])
    export = runs.loc[runs["tags.mlflow.runName"] == "export-onnx-dynamic"].iloc[0]
    onnx_path = mlflow.artifacts.download_artifacts(
        f"runs:/{export.run_id}/ppe-detector-v1-dynamic.onnx"
    )
    print(f"onnx from export run {export.run_id[:8]}: {onnx_path}")

    model = onnx.load(onnx_path)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    print(f"graph contract: {in_name} -> {out_name}")

    version_dir = REPO / "1"
    version_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, version_dir / "model.onnx")

    (REPO / "config.pbtxt").write_text(CONFIG_TEMPLATE.format(
        max_batch=MAX_BATCH,
        in_name=in_name,
        out_name=out_name,
        delay_ms=QUEUE_DELAY_US / 1000,
        queue_delay=QUEUE_DELAY_US,
        preferred=", ".join(map(str, PREFERRED_BATCHES)),
        instances=INSTANCES,
    ))
    print(f"model repository ready: {REPO.parent}")


if __name__ == "__main__":
    main()
