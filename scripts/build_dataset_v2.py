"""Build dataset v2 = curated v1 + production frames with FUSED labels.

    python scripts/build_dataset_v2.py --limit 500

THE PARTIAL-LABEL TRAP (the reason this script exists in this shape):
we recovered only 3 of the production set's 7 class mappings (Hardhat, Safety
Vest, Mask). Dropping those images in with only those three classes labelled
would teach the detector that every unlabelled worker, bare head and mask-less
face is BACKGROUND — actively destroying the NO-* classes we are trying to fix.
Partial labels are worse than no labels for detection.

So we FUSE two sources per image:
  * GROUND TRUTH for the mapped classes (real human labels from the source set)
  * PSEUDO-LABELS from our own model (conf >= --pseudo-conf) for everything else,
    skipping any pseudo-box that overlaps a ground-truth box (IoU >= 0.5)

That is standard practice when merging datasets with mismatched taxonomies, and
it carries a known risk we measure rather than assume: the pseudo-labels come
from a model that is weak on exactly the classes it is pseudo-labelling
(NO-Hardhat, 0.276 AP), so self-training can reinforce its own blind spots.
The retraining run is the experiment that decides.
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PROD = ROOT / "data" / "production"
V1 = ROOT / "data" / "processed" / "ppe-7cls-v1"
V2 = ROOT / "data" / "processed" / "ppe-7cls-v2"
CLASSES = ["Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest", "Person", "Mask", "NO-Mask"]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:9000/detect")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--pseudo-conf", type=float, default=0.50)
    args = ap.parse_args()

    import cv2
    import yaml

    mapping = json.loads((PROD / "inferred_mapping.json").read_text())["mapping"]
    numeric_names = [str(n) for n in yaml.safe_load((PROD / "data.yaml").read_text())["names"]]
    print(f"using recovered mapping: {mapping}")

    if V2.exists():
        shutil.rmtree(V2)
    for split in ("train", "valid", "test"):
        (V2 / split / "images").mkdir(parents=True)
        (V2 / split / "labels").mkdir(parents=True)

    # 1) copy v1 unchanged — v2 is a SUPERSET, so val/test stay comparable
    copied = 0
    for split in ("train", "valid", "test"):
        for img in (V1 / split / "images").glob("*.jpg"):
            shutil.copy2(img, V2 / split / "images" / img.name)
            lbl = V1 / split / "labels" / f"{img.stem}.txt"
            shutil.copy2(lbl, V2 / split / "labels" / lbl.name) if lbl.exists() else None
            copied += 1
    print(f"v1 carried over: {copied} images")

    # 2) production frames -> train split only (never contaminate val/test)
    stats = Counter()
    images = sorted((PROD / "train" / "images").glob("*.jpg"))[: args.limit]
    for i, img in enumerate(images, 1):
        lbl = PROD / "train" / "labels" / f"{img.stem}.txt"
        if not lbl.exists():
            continue
        h, w = cv2.imread(str(img)).shape[:2]

        gt_lines, gt_boxes = [], []
        for line in lbl.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            name = numeric_names[int(p[0])]
            if name not in mapping:              # unmapped class -> left to pseudo-labelling
                continue
            cx, cy, bw, bh = map(float, p[1:5])
            gt_lines.append(f"{CLASSES.index(mapping[name])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            gt_boxes.append(((cx - bw / 2) * w, (cy - bh / 2) * h,
                             (cx + bw / 2) * w, (cy + bh / 2) * h))
            stats[f"gt:{mapping[name]}"] += 1

        with open(img, "rb") as f:
            try:
                r = requests.post(args.endpoint, files={"file": f},
                                  params={"conf": args.pseudo_conf}, timeout=60)
                r.raise_for_status()
            except Exception:
                continue
        for d in r.json()["detections"]:
            box = d["box_xyxy"]
            if any(iou(box, g) >= 0.5 for g in gt_boxes):     # human label wins
                continue
            x1, y1, x2, y2 = box
            cx, cy = (x1 + x2) / 2 / w, (y1 + y2) / 2 / h
            bw, bh = (x2 - x1) / w, (y2 - y1) / h
            gt_lines.append(f"{CLASSES.index(d['label'])} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            stats[f"pseudo:{d['label']}"] += 1

        if not gt_lines:
            stats["background_frames"] += 1
        shutil.copy2(img, V2 / "train" / "images" / img.name)
        (V2 / "train" / "labels" / f"{img.stem}.txt").write_text(
            "\n".join(gt_lines) + ("\n" if gt_lines else ""))
        stats["production_frames_added"] += 1
        if i % 100 == 0:
            print(f"  {i}/{len(images)} production frames fused")

    (V2 / "data.yaml").write_text(yaml.safe_dump({
        "path": str(V2.resolve()), "train": "train/images", "val": "valid/images",
        "test": "test/images", "nc": len(CLASSES), "names": CLASSES}, sort_keys=False))

    n_train = len(list((V2 / "train" / "images").glob("*.jpg")))
    summary = {"v1_images": copied, "v2_train_images": n_train, **dict(stats)}
    (V2 / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
