"""Find Roboflow baked-in augmented copies and write exclusion lists.

Usage:
    python scripts/find_augmented_copies.py

Roboflow names every exported file `<source-stem>_jpg.rf.<hash>.jpg`; augmented
copies of one source photo share the stem and differ only in hash. We keep the
first file of each family and exclude the rest (fresh augmentation is applied
at train time by Ultralytics anyway — baked-in copies just double-count sources).

Also converts any `bad-label` SAMPLE tags (from GUI review) into an exclusion
list. Writes:
    curation/augmented_exclusions.txt
    curation/bad_label_exclusions.txt   (only if such tags exist)
"""

from collections import defaultdict
from pathlib import Path

import fiftyone as fo

ROOT = Path(__file__).resolve().parents[1]
CURATION = ROOT / "curation"
SPLITS = ("train", "valid", "test")


def split_of(sample) -> str:
    return next(t for t in sample.tags if t in SPLITS)


def main() -> None:
    dataset = fo.load_dataset("ppe-raw")
    n = len(dataset)
    if n == 0:
        raise SystemExit("ppe-raw is empty/corrupt — run scripts/curate_dataset.py --no-app first")
    print(f"{n} samples loaded")

    groups = defaultdict(list)
    for s in dataset:
        stem = Path(s.filepath).name.split("_jpg.rf.")[0].split(".rf.")[0]
        groups[stem].append(s)

    families = {k: v for k, v in groups.items() if len(v) > 1}
    sizes = sorted((len(v) for v in families.values()), reverse=True)
    print(f"{len(families)} source images with multiple copies; family sizes (top 10): {sizes[:10]}")

    aug_lines = set()
    for members in families.values():
        for s in sorted(members, key=lambda s: s.filepath)[1:]:  # keep first per family
            if "aug-copy" not in s.tags:
                s.tags.append("aug-copy")
                s.save()
            aug_lines.add(f"{split_of(s)}/{Path(s.filepath).name}")

    # Filename matching misses renamed augmented copies — the embedding-based
    # near-duplicate index catches them regardless of name. duplicate_ids holds
    # every sample that is a near-copy of some kept representative, so excluding
    # them keeps exactly one image per visual family.
    import fiftyone.brain as fob

    # temporary index: harvest duplicate_ids in-process (this API version has no
    # brain_key persistence; the index cleanup corrupts dataset counts afterward,
    # so a reimport may be needed for later interactive sessions — data files unaffected)
    index = fob.compute_near_duplicates(dataset, model="mobilenet-v2-imagenet-torch")
    for sample_id in index.duplicate_ids:
        s = dataset[sample_id]
        aug_lines.add(f"{split_of(s)}/{Path(s.filepath).name}")
    print(f"after embedding pass: {len(aug_lines)} total copies to exclude")

    CURATION.mkdir(exist_ok=True)
    (CURATION / "augmented_exclusions.txt").write_text("\n".join(sorted(aug_lines)) + "\n")
    print(f"{len(aug_lines)} copies -> curation/augmented_exclusions.txt")

    bad = dataset.match_tags("bad-label")
    if len(bad) > 0:
        bad_lines = sorted(f"{split_of(s)}/{Path(s.filepath).name}" for s in bad)
        (CURATION / "bad_label_exclusions.txt").write_text("\n".join(bad_lines) + "\n")
        print(f"{len(bad_lines)} bad-label images -> curation/bad_label_exclusions.txt")
    else:
        print("no bad-label SAMPLE tags found (box-level label tags are not exclusions)")


if __name__ == "__main__":
    main()
