"""Recover the class mapping of an unlabelled-in-name dataset by IoU agreement.

    python scripts/infer_label_mapping.py --limit 400

The production dataset ships with class NAMES stripped to numbers ('0','11','3'...),
which is exactly what messy real-world data looks like. Rather than guess, we let
the model vote: run our detector, match its boxes to the numeric ground-truth boxes
by IoU, and count which of OUR classes each numeric class co-occurs with.

A numeric class whose boxes overwhelmingly match one of our classes IS that class.
Weak or split agreement means "don't trust it" — reported, not silently accepted.

Output: data/production/inferred_mapping.json
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "data" / "production"
IOU_MATCH = 0.45
MIN_SUPPORT = 15          # boxes needed before we trust a mapping
MIN_PURITY = 0.60         # dominant class must own this share of matches


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:9000/detect")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    import cv2
    import yaml

    cfg = yaml.safe_load((PROD / "data.yaml").read_text())
    numeric_names = [str(n) for n in cfg["names"]]

    images = sorted((PROD / "train" / "images").glob("*.jpg"))[: args.limit]
    votes = defaultdict(Counter)
    support = Counter()
    matched = unmatched_gt = 0

    for i, img in enumerate(images, 1):
        lbl = PROD / "train" / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        h, w = cv2.imread(str(img)).shape[:2]
        gts = []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            ci, cx, cy, bw, bh = int(p[0]), *map(float, p[1:5])
            gts.append((numeric_names[ci],
                        ((cx - bw / 2) * w, (cy - bh / 2) * h,
                         (cx + bw / 2) * w, (cy + bh / 2) * h)))
        if not gts:
            continue

        with open(img, "rb") as f:
            try:
                r = requests.post(args.endpoint, files={"file": f},
                                  params={"conf": args.conf}, timeout=60)
                r.raise_for_status()
            except Exception:
                continue
        preds = r.json()["detections"]

        for gname, gbox in gts:
            support[gname] += 1
            best, best_iou = None, 0.0
            for p in preds:
                v = iou(gbox, p["box_xyxy"])
                if v > best_iou:
                    best, best_iou = p, v
            if best and best_iou >= IOU_MATCH:
                votes[gname][best["label"]] += 1
                matched += 1
            else:
                unmatched_gt += 1
        if i % 100 == 0:
            print(f"  {i}/{len(images)} images, {matched} boxes matched")

    mapping, report = {}, {}
    for gname in numeric_names:
        c = votes[gname]
        total = sum(c.values())
        if total == 0:
            report[gname] = {"verdict": "no_matches", "gt_boxes": support[gname]}
            continue
        top, n = c.most_common(1)[0]
        purity = n / total
        ok = total >= MIN_SUPPORT and purity >= MIN_PURITY
        if ok:
            mapping[gname] = top
        report[gname] = {
            "verdict": "mapped" if ok else "ambiguous",
            "maps_to": top, "purity": round(purity, 3),
            "matched_boxes": total, "gt_boxes": support[gname],
            "distribution": dict(c.most_common(4)),
        }

    out = {"mapping": mapping, "report": report,
           "gt_boxes_total": sum(support.values()),
           "gt_boxes_matched": matched,
           "recall_of_our_model_on_production": round(matched / max(1, sum(support.values())), 3)}
    (PROD / "inferred_mapping.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
