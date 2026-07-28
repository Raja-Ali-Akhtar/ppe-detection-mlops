"""Per-class mAP50 across precision variants — where does quantization bite?

Usage:
    python scripts/int8_class_report.py --config configs/benchmark.yaml

Hypothesis under test: efficiency levers (resolution in Stage 1, precision here)
tax the NO-* violation classes first. Prints a per-class table of mAP50 per
variant with deltas vs the pytorch reference, and writes it to
reports/int8-class-report.md.
"""

import argparse
from pathlib import Path

import mlflow
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]

from benchmark import EngineRunner, TorchRunner, decode
from build_engines import preprocess


def eval_dataset(runner, cfg, variant):
    import cv2
    import fiftyone as fo

    classes = cfg["classes"]
    name = f"clsreport-{variant}"
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
            dets.append(fo.Detection(label=classes[int(cl)],
                                     bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                                     confidence=float(score)))
        gts = []
        lbl = ROOT / cfg["test_labels"] / (img_path.stem + ".txt")
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.split()
                if len(p) == 5:
                    ci, cx, cy, bw, bh = int(p[0]), *map(float, p[1:])
                    gts.append(fo.Detection(label=classes[ci],
                                            bounding_box=[cx - bw / 2, cy - bh / 2, bw, bh]))
        s = fo.Sample(filepath=str(img_path))
        s["ground_truth"] = fo.Detections(detections=gts)
        s["predictions"] = fo.Detections(detections=dets)
        ds.add_sample(s)

    res = ds.evaluate_detections("predictions", gt_field="ground_truth",
                                 eval_key="e", compute_mAP=True, iou_threshs=[0.5])
    per_class = {c: res.mAP(classes=[c]) for c in classes}
    fo.delete_dataset(name)
    return per_class


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())

    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")
    classes = cfg["classes"]
    results = {}
    for v in cfg["variants"]:
        path = v["path"]
        if v["kind"] == "engine" and not (ROOT / path).exists():
            print(f"skip {v['name']}: missing")
            continue
        print(f"evaluating {v['name']}...")
        runner = TorchRunner(path, cfg["imgsz"]) if v["kind"] == "torch" \
            else EngineRunner(ROOT / path, cfg["imgsz"])
        results[v["name"]] = eval_dataset(runner, cfg, v["name"])
        del runner
        torch.cuda.empty_cache()

    ref = results.get("pytorch", {})
    variants = list(results)
    header = "| Class | " + " | ".join(variants) + " | int8 worst Δ |"
    sep = "|---" * (len(variants) + 2) + "|"
    lines = [header, sep]
    for c in classes:
        row = [f"| {c} "]
        worst_delta = 0.0
        for v in variants:
            val = results[v].get(c)
            cell = f"{val:.3f}" if val is not None else "–"
            if v.startswith("trt-int8") and val is not None and ref.get(c):
                d = val - ref[c]
                worst_delta = min(worst_delta, d)
                cell += f" ({d:+.3f})"
            row.append(f"| {cell} ")
        row.append(f"| {worst_delta:+.3f} |")
        lines.append("".join(row))

    table = "\n".join(lines)
    print("\nPer-class mAP50 (deltas vs pytorch):\n")
    print(table)
    out = ROOT / "reports" / "int8-class-report.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("# Per-class mAP50 by precision variant\n\n" + table + "\n")
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
