"""Build data/processed from data/raw: scope 25 raw classes down to the 7 PPE classes.

Usage:
    python scripts/build_processed.py

Data lineage rule: raw/ is never modified; processed/ is fully regenerable by
re-running this script. Labels for dropped classes are removed; images left with
zero labels are KEPT as background/negative samples (YOLO treats an empty label
file as "nothing to detect here" — removing them would teach the model that
every image contains an object).
"""

import shutil
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "css-v30"
PROCESSED = ROOT / "data" / "processed" / "ppe-7cls-v1"

# old class id (from raw data.yaml `names` order) -> new class id
KEEP = {
    2: 0,   # Hardhat
    5: 1,   # NO-Hardhat
    11: 2,  # Safety Vest
    7: 3,   # NO-Safety Vest
    8: 4,   # Person
    4: 5,   # Mask
    6: 6,   # NO-Mask
}
NEW_NAMES = ["Hardhat", "NO-Hardhat", "Safety Vest", "NO-Safety Vest", "Person", "Mask", "NO-Mask"]


def remap_label_file(src: Path, dst: Path) -> tuple[int, int]:
    """Rewrite one YOLO label file, keeping/remapping only PPE classes.

    Returns (kept, dropped) box counts."""
    kept, dropped = 0, 0
    lines_out = []
    for line in src.read_text().splitlines():
        parts = line.split()
        if not parts:
            continue
        old_id = int(parts[0])
        if old_id in KEEP:
            lines_out.append(" ".join([str(KEEP[old_id])] + parts[1:]))
            kept += 1
        else:
            dropped += 1
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""))
    return kept, dropped


def main() -> None:
    if PROCESSED.exists():
        shutil.rmtree(PROCESSED)  # regenerable by definition — always rebuild clean

    total_kept, total_dropped, backgrounds = 0, 0, 0
    for split in ("train", "valid", "test"):
        img_src = RAW / split / "images"
        lbl_src = RAW / split / "labels"
        img_dst = PROCESSED / split / "images"
        lbl_dst = PROCESSED / split / "labels"
        img_dst.mkdir(parents=True)
        lbl_dst.mkdir(parents=True)

        for img in img_src.iterdir():
            shutil.copy2(img, img_dst / img.name)
            label = lbl_src / (img.stem + ".txt")
            if label.exists():
                kept, dropped = remap_label_file(label, lbl_dst / label.name)
                total_kept += kept
                total_dropped += dropped
                if kept == 0:
                    backgrounds += 1
            else:
                (lbl_dst / (img.stem + ".txt")).write_text("")
                backgrounds += 1

    cfg = {
        "path": str(PROCESSED.resolve()),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(NEW_NAMES),
        "names": NEW_NAMES,
    }
    (PROCESSED / "data.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    print(f"boxes kept: {total_kept}, dropped: {total_dropped}")
    print(f"background (label-free) images: {backgrounds}")
    print(f"written to {PROCESSED}")


if __name__ == "__main__":
    main()
