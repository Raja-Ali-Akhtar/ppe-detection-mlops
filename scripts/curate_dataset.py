"""First-look curation of the raw PPE dataset with FiftyOne.

Usage:
    python scripts/curate_dataset.py            # compute + print stats, launch app
    python scripts/curate_dataset.py --no-app   # stats only (CI-friendly)

Loads the YOLO-format download from data/raw/ into FiftyOne, then reports:
  - sample counts per split
  - class distribution (the class-imbalance number for the post)
  - image size distribution
  - exact-duplicate candidates via file hashes
Findings drive what data/processed/ will look like — raw/ is never modified.
"""

import argparse
from collections import Counter
from pathlib import Path

import fiftyone as fo
import fiftyone.utils.yolo as fouy

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
DATASET_NAME = "ppe-raw"


def _fixed_yaml(original: Path) -> Path:
    """Roboflow yamls point at ../train/images (one level above the yaml), which is
    wrong for local tooling. Write a corrected copy with absolute paths OUTSIDE
    raw/ (raw stays byte-identical to the download)."""
    import yaml

    cfg = yaml.safe_load(original.read_text())
    base = original.parent
    for key, subdir in (("train", "train"), ("val", "valid"), ("test", "test")):
        if key in cfg:
            cfg[key] = str((base / subdir / "images").resolve())
    fixed = original.parents[2] / "css-v30-local.yaml"  # lands in data/
    fixed.write_text(yaml.safe_dump(cfg))
    return fixed


def load_dataset() -> fo.Dataset:
    """Load all YOLO splits from data/raw into one FiftyOne dataset with split tags."""
    if fo.dataset_exists(DATASET_NAME):
        existing = fo.load_dataset(DATASET_NAME)
        if len(existing) > 0:
            return existing
        # empty cached dataset = a failed/unflushed prior import; never trust it
        fo.delete_dataset(DATASET_NAME)

    dataset = fo.Dataset(DATASET_NAME, persistent=True)
    yaml_path = _fixed_yaml(next(RAW.rglob("data.yaml")))
    base = next(RAW.rglob("data.yaml")).parent
    # disk dirs are train/valid/test but the yaml keys are train/val/test
    for split_dir_name, yaml_key in (("train", "train"), ("valid", "val"), ("test", "test")):
        if not (base / split_dir_name).exists():
            continue
        dataset.add_dir(
            dataset_type=fo.types.YOLOv5Dataset,
            dataset_dir=str(base),
            split=yaml_key,
            tags=split_dir_name,
            yaml_path=str(yaml_path),
        )
    return dataset


def print_stats(dataset: fo.Dataset) -> None:
    print(f"\n=== {dataset.name}: {len(dataset)} samples ===")

    for split in ("train", "valid", "test"):
        view = dataset.match_tags(split)
        print(f"{split}: {len(view)} images")

    counts = dataset.count_values("ground_truth.detections.label")
    total = sum(counts.values())
    print("\n--- Class distribution ---")
    for label, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"{label:20s} {n:6d}  ({100 * n / total:.1f}%)")
    if counts:
        most, least = max(counts.values()), min(counts.values())
        print(f"Imbalance ratio (most:least): {most / least:.1f}:1")

    widths = dataset.distinct("metadata.width")
    heights = dataset.distinct("metadata.height")
    print(f"\nDistinct image widths: {sorted(widths)[:10]}{'...' if len(widths) > 10 else ''}")
    print(f"Distinct image heights: {sorted(heights)[:10]}{'...' if len(heights) > 10 else ''}")


def find_exact_duplicates(dataset: fo.Dataset) -> None:
    """Flag byte-identical images (cross-split duplicates = train/val leakage)."""
    import hashlib

    hashes: dict = {}
    dupes = 0
    for sample in dataset.select_fields("filepath"):
        digest = hashlib.md5(Path(sample.filepath).read_bytes()).hexdigest()
        if digest in hashes:
            dupes += 1
            sample.tags.append("exact-duplicate")
            sample.save()
        else:
            hashes[digest] = sample.filepath
    print(f"\nExact duplicates: {dupes} (tagged 'exact-duplicate')")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-app", action="store_true", help="skip launching the app")
    args = parser.parse_args()

    dataset = load_dataset()
    dataset.compute_metadata()
    print_stats(dataset)
    find_exact_duplicates(dataset)

    if not args.no_app:
        session = fo.launch_app(dataset)
        session.wait()


if __name__ == "__main__":
    main()
