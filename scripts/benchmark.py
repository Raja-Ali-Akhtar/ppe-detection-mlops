"""Benchmark every model variant: latency, throughput, VRAM, and mAP retention.

Usage:
    python scripts/benchmark.py --config configs/benchmark.yaml

Method notes (depth-series rigor):
  - warmup iterations excluded; latencies from perf_counter around a
    stream-synchronized execute; p50/p95/p99 reported, never means alone
  - VRAM = device free-memory delta across engine+context creation
  - mAP: every variant (INCLUDING the pytorch reference) goes through the
    SAME decode + NMS + FiftyOne evaluation path, so retention numbers
    compare precision, not pipeline differences
  - each variant = one MLflow run in `ppe-optimization`
"""

import argparse
import time
from pathlib import Path

import mlflow
import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from build_engines import preprocess  # same preprocessing, one source


# --------------------------------------------------------------- TRT runner
class EngineRunner:
    def __init__(self, engine_path, imgsz):
        import tensorrt as trt

        free0, _ = torch.cuda.mem_get_info()
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        self.engine = runtime.deserialize_cuda_engine(Path(engine_path).read_bytes())
        self.ctx = self.engine.create_execution_context()
        self.inp = self.engine.get_tensor_name(0)
        self.out = self.engine.get_tensor_name(1)
        self.imgsz = imgsz
        torch.cuda.synchronize()
        free1, _ = torch.cuda.mem_get_info()
        self.vram_mb = (free0 - free1) / 1e6

    def __call__(self, batch_tensor):
        self.ctx.set_input_shape(self.inp, tuple(batch_tensor.shape))
        out_shape = tuple(self.ctx.get_tensor_shape(self.out))
        out = torch.empty(out_shape, dtype=torch.float32, device="cuda")
        self.ctx.set_tensor_address(self.inp, batch_tensor.data_ptr())
        self.ctx.set_tensor_address(self.out, out.data_ptr())
        stream = torch.cuda.current_stream().cuda_stream
        self.ctx.execute_async_v3(stream)
        torch.cuda.synchronize()
        return out


class TorchRunner:
    def __init__(self, model_uri, imgsz):
        from ultralytics import YOLO

        free0, _ = torch.cuda.mem_get_info()
        weights = mlflow.artifacts.download_artifacts(model_uri)
        self.model = YOLO(weights).model.eval().cuda()
        torch.cuda.synchronize()
        free1, _ = torch.cuda.mem_get_info()
        self.vram_mb = (free0 - free1) / 1e6
        self.imgsz = imgsz

    @torch.no_grad()
    def __call__(self, batch_tensor):
        out = self.model(batch_tensor)
        out = out[0] if isinstance(out, (list, tuple)) else out
        torch.cuda.synchronize()
        return out


# ------------------------------------------------------------ decode + eval
def decode(pred, conf_thr, iou_thr):
    """(11, 8400) raw head -> [x1,y1,x2,y2,conf,cls] in 640-canvas coords."""
    import torchvision

    pred = pred.T                                   # (8400, 11)
    scores, cls = pred[:, 4:].max(dim=1)
    keep = scores > conf_thr
    if keep.sum() == 0:
        return torch.zeros((0, 6), device=pred.device)
    box, scores, cls = pred[keep, :4], scores[keep], cls[keep]
    xy1 = box[:, :2] - box[:, 2:] / 2
    xy2 = box[:, :2] + box[:, 2:] / 2
    boxes = torch.cat([xy1, xy2], dim=1)
    keep = torchvision.ops.batched_nms(boxes, scores, cls, iou_thr)[:300]
    return torch.cat([boxes[keep], scores[keep, None], cls[keep, None].float()], dim=1)


def evaluate_map(runner, cfg, variant):
    """Run the test set through a runner, score with FiftyOne -> mAP50, mAP50-95."""
    import cv2
    import fiftyone as fo

    classes = cfg["classes"]
    name = f"bench-{variant}"
    if fo.dataset_exists(name):
        fo.delete_dataset(name)
    ds = fo.Dataset(name)

    size = cfg["imgsz"]
    for img_path in sorted((ROOT / cfg["test_images"]).glob("*.jpg")):
        raw = cv2.imread(str(img_path))
        h, w = raw.shape[:2]
        r = min(size / h, size / w)
        top, left = (size - round(h * r)) // 2, (size - round(w * r)) // 2

        t = torch.from_numpy(preprocess(img_path, size)).unsqueeze(0).cuda()
        det = decode(runner(t)[0], cfg["conf_eval"], cfg["iou_nms"]).cpu().numpy()

        dets = []
        for x1, y1, x2, y2, score, cl in det:
            x1, x2 = (x1 - left) / r, (x2 - left) / r
            y1, y2 = (y1 - top) / r, (y2 - top) / r
            dets.append(fo.Detection(
                label=classes[int(cl)],
                bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                confidence=float(score),
            ))

        gts = []
        lbl = ROOT / cfg["test_labels"] / (img_path.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) == 5:
                    ci, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                    gts.append(fo.Detection(
                        label=classes[ci],
                        bounding_box=[cx - bw / 2, cy - bh / 2, bw, bh],
                    ))

        s = fo.Sample(filepath=str(img_path))
        s["ground_truth"] = fo.Detections(detections=gts)
        s["predictions"] = fo.Detections(detections=dets)
        ds.add_sample(s)

    res = ds.evaluate_detections("predictions", gt_field="ground_truth",
                                 eval_key="e", compute_mAP=True)
    map5095 = res.mAP()
    # mAP() always returns the mean over the eval's IoU sweep — to get mAP50,
    # restrict the sweep itself to a single threshold
    res50 = ds.evaluate_detections("predictions", gt_field="ground_truth",
                                   eval_key="e50", compute_mAP=True, iou_threshs=[0.5])
    map50 = res50.mAP()
    fo.delete_dataset(name)
    return map50, map5095


# ----------------------------------------------------------------- latency
def bench_latency(runner, cfg):
    metrics = {}
    for b in cfg["batches"]:
        x = torch.rand((b, 3, cfg["imgsz"], cfg["imgsz"]), device="cuda")
        for _ in range(cfg["warmup"]):
            runner(x)
        times = []
        for _ in range(cfg["iters"]):
            t0 = time.perf_counter()
            runner(x)
            times.append((time.perf_counter() - t0) * 1000)
        times = np.array(times)
        metrics[f"lat_b{b}_p50_ms"] = float(np.percentile(times, 50))
        metrics[f"lat_b{b}_p95_ms"] = float(np.percentile(times, 95))
        metrics[f"lat_b{b}_p99_ms"] = float(np.percentile(times, 99))
        metrics[f"throughput_b{b}_ips"] = float(b * 1000 / np.percentile(times, 50))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    mlflow.set_experiment("ppe-optimization")

    reference_map = {}
    for v in cfg["variants"]:
        name, kind = v["name"], v["kind"]
        print(f"\n=== {name} ===")
        if kind == "torch":
            runner = TorchRunner(v["path"], cfg["imgsz"])
        else:
            path = ROOT / v["path"]
            if not path.exists():
                print(f"skip {name}: {path} missing")
                continue
            runner = EngineRunner(path, cfg["imgsz"])

        with mlflow.start_run(run_name=f"bench-{name}"):
            mlflow.log_params({"variant": name, "kind": kind, "iters": cfg["iters"],
                               "warmup": cfg["warmup"], "batches": str(cfg["batches"])})
            lat = bench_latency(runner, cfg)
            print("  " + ", ".join(f"{k}={val:.2f}" for k, val in lat.items() if "p50" in k or "ips" in k))
            map50, map5095 = evaluate_map(runner, cfg, name)
            print(f"  mAP50={map50:.4f}  mAP50-95={map5095:.4f}")
            metrics = {**lat, "vram_mb": runner.vram_mb,
                       "map50": map50, "map50_95": map5095}
            if name == "pytorch":
                reference_map["m"] = map50
            elif "m" in reference_map:
                metrics["map50_retention_pct"] = 100 * map50 / reference_map["m"]
            mlflow.log_metrics({k: round(v, 4) for k, v in metrics.items()})

        del runner
        torch.cuda.empty_cache()

    print("\nall variants done — compare runs in the ppe-optimization experiment")


if __name__ == "__main__":
    main()

