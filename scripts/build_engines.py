"""Build TensorRT engines (fp32 / fp16 / int8) from the exported ONNX.

Usage:
    python scripts/build_engines.py --config configs/engines.yaml

Every build is an MLflow run in `ppe-optimization`: precision, profile, TRT
version, build time, engine size, and the engine file's SHA256 (engines are
too large to store as artifacts — the hash gives traceability without bloat).

INT8 calibration reads training images preprocessed EXACTLY like inference
(letterbox 640, pad 114, BGR->RGB, /255) — mismatched calibration
preprocessing is the classic silent INT8 quality killer. The calibration
cache is logged as an artifact so identical rebuilds skip calibration.

NOTE: engines are LOCKED to this machine's TRT version + GPU. They are
benchmarking artifacts; deployment (Stage 3) ships the ONNX to Triton.
"""

import argparse
import hashlib
import time
from pathlib import Path

import mlflow
import numpy as np
import tensorrt as trt
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOGGER = trt.Logger(trt.Logger.WARNING)


def letterbox(img, size):
    """Resize with aspect ratio preserved, pad to size x size with 114-gray."""
    import cv2

    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    top, left = (size - nh) // 2, (size - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    return canvas


def preprocess(path, size):
    import cv2

    img = cv2.imread(str(path))
    img = letterbox(img, size)
    img = img[..., ::-1].transpose(2, 0, 1)          # BGR->RGB, HWC->CHW
    return np.ascontiguousarray(img, dtype=np.float32) / 255.0


def make_calibrator(kind, image_dir, count, batch, size, cache_path):
    """Torch-backed calibrator (no pycuda); kind = 'entropy' | 'minmax'.

    The cache file must be per-(kind, count): a cache written by one calibrator
    type would be silently reused by another, skipping calibration entirely."""
    base = trt.IInt8MinMaxCalibrator if kind == "minmax" else trt.IInt8EntropyCalibrator2

    class Calibrator(base):
        def __init__(self):
            super().__init__()
            import torch

            self.cache_path = Path(cache_path)
            self.batch = batch
            images = sorted(Path(image_dir).glob("*.jpg"))[:count]
            if len(images) < count:
                print(f"warning: only {len(images)} calibration images found")
            self.batches = [images[i:i + batch] for i in range(0, len(images) - batch + 1, batch)]
            self.idx = 0
            self.device_input = torch.empty((batch, 3, size, size), dtype=torch.float32, device="cuda")
            self._torch = torch

        def get_batch_size(self):
            return self.batch

        def get_batch(self, names):
            if self.idx >= len(self.batches):
                return None
            data = np.stack([preprocess(p, size) for p in self.batches[self.idx]])
            self.device_input.copy_(self._torch.from_numpy(data))
            self.idx += 1
            print(f"  calibration batch {self.idx}/{len(self.batches)}", flush=True)
            return [int(self.device_input.data_ptr())]

        def read_calibration_cache(self):
            if self.cache_path.exists():
                print("  using existing calibration cache")
                return self.cache_path.read_bytes()
            return None

        def write_calibration_cache(self, cache):
            self.cache_path.write_bytes(cache)

    return Calibrator()


def build_engine(cfg, precision):
    builder = trt.Builder(LOGGER)
    network = builder.create_network(0)  # explicit batch (the only mode in TRT 10)
    parser = trt.OnnxParser(network, LOGGER)
    onnx_path = str(ROOT / cfg["onnx"])
    if not parser.parse_from_file(onnx_path):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        raise SystemExit(f"ONNX parse failed: {onnx_path}")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, cfg["workspace_gb"] << 30)

    size = cfg["imgsz"]
    inp = network.get_input(0)
    profile = builder.create_optimization_profile()
    profile.set_shape(
        inp.name,
        (cfg["batch_min"], 3, size, size),
        (cfg["batch_opt"], 3, size, size),
        (cfg["batch_max"], 3, size, size),
    )
    config.add_optimization_profile(profile)

    calibrator = None
    if precision == "fp16":
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8":
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.FP16)  # let TRT pick fp16 where int8 hurts
        c = cfg["calib"]
        cache = ROOT / cfg["out_dir"] / f"int8-{c['type']}-{c['count']}.cache"
        calibrator = make_calibrator(
            c["type"], ROOT / c["dir"], c["count"], c["batch"], size, cache
        )
        config.int8_calibrator = calibrator
        config.set_calibration_profile(profile)

    variant = f"int8-{cfg['calib']['type']}" if precision == "int8" else precision
    print(f"building {variant} engine (this can take minutes)...", flush=True)
    t0 = time.time()
    serialized = builder.build_serialized_network(network, config)
    build_s = time.time() - t0
    if serialized is None:
        raise SystemExit(f"{variant} build failed")

    out = ROOT / cfg["out_dir"] / f"ppe-v1-{variant}.engine"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(serialized)
    return out, build_s


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--only", help="build just this precision (fp32/fp16/int8)")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.only:
        cfg["precisions"] = [args.only]

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("ppe-optimization")

    for precision in cfg["precisions"]:
        with mlflow.start_run(run_name=f"build-{precision}"):
            mlflow.log_params({
                "precision": precision,
                "tensorrt_version": trt.__version__,
                "onnx": cfg["onnx"],
                "batch_profile": f"{cfg['batch_min']}/{cfg['batch_opt']}/{cfg['batch_max']}",
                "imgsz": cfg["imgsz"],
                "calib_images": cfg["calib"]["count"] if precision == "int8" else 0,
                "calib_type": cfg["calib"]["type"] if precision == "int8" else "n/a",
            })
            out, build_s = build_engine(cfg, precision)
            sha = hashlib.sha256(out.read_bytes()).hexdigest()[:16]
            mlflow.log_params({"engine_path": str(out.relative_to(ROOT)), "engine_sha256": sha})
            mlflow.log_metric("build_seconds", round(build_s, 1))
            mlflow.log_metric("engine_size_mb", round(out.stat().st_size / 1e6, 1))
            if precision == "int8":
                c = cfg["calib"]
                cache = ROOT / cfg["out_dir"] / f"int8-{c['type']}-{c['count']}.cache"
                mlflow.log_artifact(str(cache))
            print(f"{precision}: {out.name}  {out.stat().st_size/1e6:.1f} MB  in {build_s:.0f}s  sha {sha}")


if __name__ == "__main__":
    main()
