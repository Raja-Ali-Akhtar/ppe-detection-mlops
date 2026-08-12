"""Regression tests written from REAL incidents in this project.

Every test here exists because something broke and cost hours. CI runs them on
every push so the same failure cannot come back quietly.
"""

import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "serving"))


# ---------------------------------------------------------------- INCIDENT #1
# Stage 4: an `evidently` install downgraded pydantic -> broke fastapi -> broke
# mlflow. Ultralytics' tracking callback then failed SILENTLY: 40 epochs trained,
# nothing recorded. A tracking system that fails quietly is data loss.
def test_tracking_stack_imports():
    import fastapi  # noqa: F401
    import mlflow

    assert mlflow.__version__, "mlflow must import — a broken dependency chain kills tracking silently"


def test_mlflow_can_actually_log(tmp_path):
    """Importing is not enough: the Stage 4 break only showed up at log time."""
    import mlflow

    mlflow.set_tracking_uri(f"sqlite:///{(tmp_path / 'ci.db').as_posix()}")
    mlflow.set_experiment("ci-smoke")
    with mlflow.start_run(run_name="ci"):
        mlflow.log_param("p", 1)
        mlflow.log_metric("m", 0.5, step=0)
    runs = mlflow.search_runs(experiment_names=["ci-smoke"])
    assert len(runs) == 1 and runs.iloc[0]["metrics.m"] == 0.5


# ---------------------------------------------------------------- INCIDENT #2
# Stage 3: Triton refused to load because config.pbtxt declared output dims
# [11, 8400] while the ONNX graph declares [-1, 11, -1] (dynamic anchors follow
# from dynamic H/W). The config must describe the GRAPH's capability, not our
# deployment intention.
def test_triton_config_generator_uses_dynamic_output_dim():
    src = (ROOT / "scripts" / "package_triton.py").read_text()
    assert "dims: [ 11, -1 ]" in src, (
        "output dims must stay dynamic — hardcoding 8400 makes Triton refuse the model"
    )
    assert "{in_name}" in src and "{out_name}" in src, (
        "tensor names must be read FROM the onnx graph, never hardcoded"
    )


def test_triton_config_cudnn_algo_is_integer_enum():
    """The ORT parameter takes an integer enum as a string ('1' = HEURISTIC).
    Passing the NAME made Triton fail with 'failed to convert HEURISTIC to
    integral number'; leaving it at the EXHAUSTIVE default gave p95 8,000 ms."""
    src = (ROOT / "scripts" / "package_triton.py").read_text()
    assert 'key: "cudnn_conv_algo_search"' in src
    assert 'string_value: "1"' in src, "must be the integer enum, not 'HEURISTIC'"


# ---------------------------------------------------------------- INCIDENT #3
# Stage 3: the gateway used tritonclient.http, which is gevent-based and
# collides with uvicorn's asyncio loop ("Cannot switch to a different thread").
def test_gateway_uses_grpc_client():
    src = (ROOT / "serving" / "gateway" / "app.py").read_text()
    imports = [ln.strip() for ln in src.splitlines()
               if ln.strip().startswith(("import ", "from "))]
    assert any("tritonclient.grpc" in ln for ln in imports)
    assert not any("tritonclient.http" in ln for ln in imports), \
        "http client's gevent breaks under uvicorn (mentions in comments are fine)"


# ---------------------------------------------------------------- CONSISTENCY
# One class list, many consumers. A mismatch silently relabels every detection.
def test_class_list_is_consistent_everywhere():
    from common import CLASSES

    v1_yaml = ROOT / "data" / "processed" / "ppe-7cls-v1" / "data.yaml"
    if v1_yaml.exists():                       # dataset is DVC-tracked, may be absent in CI
        assert yaml.safe_load(v1_yaml.read_text())["names"] == CLASSES

    # scripts that hardcode the list must agree with it; scripts that read it
    # from config (the better pattern — int8_class_report.py) are checked via
    # the config instead
    for script in ("build_processed.py", "build_dataset_v2.py"):
        src = (ROOT / "scripts" / script).read_text()
        for name in CLASSES:
            assert f'"{name}"' in src, f"{script} is missing class {name!r}"

    bench = yaml.safe_load((ROOT / "configs" / "benchmark.yaml").read_text())
    assert bench["classes"] == CLASSES, "configs/benchmark.yaml class order must match serving"


def test_training_configs_are_valid():
    for cfg_path in (ROOT / "configs").glob("*.yaml"):
        cfg = yaml.safe_load(cfg_path.read_text())
        if "model" not in cfg:                 # export/engine configs have their own shape
            continue
        # 320 is legitimate — exp-imgsz320 is the resolution ablation. The real
        # contract is YOLO's stride: imgsz must be a multiple of 32.
        assert cfg["imgsz"] % 32 == 0, f"{cfg_path.name}: imgsz must be a multiple of 32"
        if cfg_path.stem not in ("exp-imgsz320", "smoke"):   # ablation + fast pipeline check
            assert cfg["imgsz"] == 640, f"{cfg_path.name}: non-ablation runs use the serving size"
        assert cfg["model"].endswith(".pt")
        assert cfg["epochs"] > 0 and cfg["batch"] > 0
        assert "seed" in cfg, f"{cfg_path.name}: runs must be seeded to be comparable"


def test_no_secrets_committed():
    """The AWS key leak in Stage 1 cost a rotation. Never again from a commit."""
    import re

    pattern = re.compile(r"AKIA[0-9A-Z]{16}")
    for path in list(ROOT.glob("scripts/*.py")) + list(ROOT.glob("serving/**/*.py")) \
            + list(ROOT.glob("terraform/*.tf")) + list(ROOT.glob("configs/*.yaml")):
        assert not pattern.search(path.read_text()), f"AWS access key id found in {path}"

