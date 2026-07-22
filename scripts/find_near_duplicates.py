"""Near-duplicate detection via embedding similarity (GPU).

Usage:
    python scripts/find_near_duplicates.py

Embeds every image with a small ImageNet backbone, then flags pairs whose
embeddings are nearly identical — catches re-crops, re-compressions, and
frame-adjacent shots that byte-level hashing (curate_dataset.py) cannot.
The critical finding is CROSS-SPLIT near-dups: same scene in train and
valid/test = leakage = inflated validation metrics.
"""

import fiftyone as fo
import fiftyone.brain as fob

DATASET_NAME = "ppe-raw"


def main() -> None:
    dataset = fo.load_dataset(DATASET_NAME)
    n_samples = len(dataset)  # capture BEFORE compute: its temp-index cleanup
    # corrupts the dataset's count state (fiftyone 1.19 bug, seen on Windows)

    # NOTE: this API version has no brain_key persistence for near-dups; the
    # temp-index cleanup corrupts the dataset's COUNT state on exit (data is
    # fine — reimport with curate_dataset.py if the app shows "No samples")
    index = fob.compute_near_duplicates(dataset, model="mobilenet-v2-imagenet-torch")

    dup_ids = index.duplicate_ids
    print(f"\nnear-duplicate candidates: {len(dup_ids)} of {n_samples} samples")

    for sample_id in dup_ids:
        sample = dataset[sample_id]
        if "near-duplicate" not in sample.tags:
            sample.tags.append("near-duplicate")
            sample.save()

    # cross-split pairs = train/val leakage. Remedy: keep the train member,
    # exclude the valid/test twin. The exclusion list is committed to git —
    # curation decisions are data lineage, not throwaway state.
    from pathlib import Path

    drop_rank = {"train": 0, "valid": 1, "test": 2}  # higher rank gets dropped
    leaks = 0
    exclusions = set()
    for sample_id, neighbors in index.neighbors_map.items():
        anchor = dataset[sample_id]
        anchor_split = next(iter(set(anchor.tags) & drop_rank.keys()), None)
        for neighbor_id, _dist in neighbors:
            neighbor = dataset[neighbor_id]
            neighbor_split = next(iter(set(neighbor.tags) & drop_rank.keys()), None)
            if anchor_split and neighbor_split and anchor_split != neighbor_split:
                leaks += 1
                victim = anchor if drop_rank[anchor_split] > drop_rank[neighbor_split] else neighbor
                victim_split = anchor_split if victim is anchor else neighbor_split
                exclusions.add(f"{victim_split}/{Path(victim.filepath).name}")
                for s in (anchor, neighbor):
                    if "leakage" not in s.tags:
                        s.tags.append("leakage")
                        s.save()
    print(f"cross-split near-dup pairs (leakage!): {leaks}")

    out = Path(__file__).resolve().parents[1] / "curation" / "leakage_exclusions.txt"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(sorted(exclusions)) + "\n")
    print(f"{len(exclusions)} valid/test samples excluded -> {out}")

    tiny = dataset.match(
        (fo.ViewField("metadata.width") < 224) | (fo.ViewField("metadata.height") < 224)
    )
    print(f"tiny images (<224px on a side): {len(tiny)}")


if __name__ == "__main__":
    main()
