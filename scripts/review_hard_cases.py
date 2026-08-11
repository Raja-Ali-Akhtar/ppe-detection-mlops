"""FiftyOne review of harvested hard cases — the human-in-the-loop step.

    python scripts/review_hard_cases.py --review     # load + open the app
    python scripts/review_hard_cases.py --export     # tags -> YOLO labels for v2

WORKFLOW (verification-style annotation, the realistic one):
the model's predictions are loaded as PRE-ANNOTATIONS. You do not draw boxes
from scratch — you judge what the model produced, which is 5-10x faster and is
exactly how annotation teams actually work (model proposes, human disposes).

Tag every frame with ONE of:
    keep        predictions are right (or right enough) -> becomes a labelled sample
    background  genuinely nothing to detect            -> becomes a negative sample
    drop        predictions are wrong / image unusable -> excluded from v2

Optional: tag individual BOXES with `bad-box` and they are removed on export.

Only `keep` + `background` frames enter dataset v2. Everything is exported to
YOLO format so build_processed.py can merge it.
"""

import argparse
import json
import shutil
from pathlib import Path

import fiftyone as fo

ROOT = Path(__file__).resolve().parents[1]
HARD = ROOT / "data" / "hard_cases"
OUT = ROOT / "data" / "hard_cases_reviewed"
NAME = "ppe-hard-cases"
CLASSES = ["Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest", "Person", "Mask", "NO-Mask"]


def load() -> fo.Dataset:
    if fo.dataset_exists(NAME):
        ds = fo.load_dataset(NAME)
        if len(ds) > 0:
            return ds
        fo.delete_dataset(NAME)

    preds = json.loads((HARD / "predictions.json").read_text())
    ds = fo.Dataset(NAME, persistent=True)
    samples = []
    import cv2

    for img in sorted((HARD / "images").glob("*.jpg")):
        rec = preds.get(img.name, {})
        h, w = cv2.imread(str(img)).shape[:2]
        dets = []
        for d in rec.get("detections", []):
            x1, y1, x2, y2 = d["box_xyxy"]
            dets.append(fo.Detection(
                label=d["label"],
                bounding_box=[x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h],
                confidence=d["confidence"],
            ))
        s = fo.Sample(filepath=str(img))
        s["predictions"] = fo.Detections(detections=dets)
        s["harvest_score"] = rec.get("score", 0.0)
        s["harvest_reasons"] = ", ".join(rec.get("reasons", []))
        samples.append(s)
    ds.add_samples(samples)
    return ds


def export(ds: fo.Dataset) -> None:
    keep = ds.match_tags(["keep", "background"])
    if len(keep) == 0:
        raise SystemExit("nothing tagged keep/background yet — review first")

    img_dir, lbl_dir = OUT / "images", OUT / "labels"
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)

    n_boxes = n_bg = 0
    for s in keep:
        name = Path(s.filepath).name
        shutil.copy2(s.filepath, img_dir / name)
        lines = []
        if "background" not in s.tags:
            for d in s["predictions"].detections:
                if "bad-box" in (d.tags or []):
                    continue
                x, y, w, h = d.bounding_box                 # relative xywh (top-left)
                lines.append(f"{CLASSES.index(d.label)} {x + w/2:.6f} {y + h/2:.6f} {w:.6f} {h:.6f}")
        if not lines:
            n_bg += 1
        n_boxes += len(lines)
        (lbl_dir / f"{Path(name).stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

    print(f"exported {len(keep)} reviewed frames -> {OUT}")
    print(f"  {n_boxes} boxes, {n_bg} background frames")
    print(f"  dropped: {len(ds) - len(keep)} frames tagged 'drop' or untagged")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", action="store_true")
    ap.add_argument("--export", action="store_true")
    args = ap.parse_args()

    ds = load()
    print(f"{len(ds)} hard cases loaded")
    print("tagged so far:", {t: len(ds.match_tags(t)) for t in ("keep", "background", "drop")})

    if args.export:
        export(ds)
        return

    if args.review:
        # most uncertain first — spend attention where it counts
        session = fo.launch_app(ds.sort_by("harvest_score", reverse=True))
        print("\nTag each frame: keep / background / drop   (press '5' … or use the tag icon)")
        print("Then rerun with --export")
        session.wait(-1)


if __name__ == "__main__":
    main()
