"""Stage 4 verdict: did adding production data help — and did NO-Hardhat move?

    python scripts/compare_v1_v2.py

Reads both runs from MLflow, then re-evaluates each best.pt on the SAME held-out
test set to get per-class AP50 (overall mAP can rise while the weak class stays
broken — the whole point of Stage 4 is that specific class).

Writes reports/stage4/comparison.md + comparison.png
"""

import json
from pathlib import Path

import mlflow
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "stage4"
RUNS = {"v1 (509 imgs)": "baseline-yolov8n", "v2 (901 imgs)": "v2-yolov8n"}
TEST_DATA = ROOT / "data/processed/ppe-7cls-v1/data.yaml"   # identical eval set for both


def main() -> None:
    mlflow.set_tracking_uri(f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}")

    OUT.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO

    # weights on disk are the source of truth here — the MLflow run may be absent
    # (a broken pydantic silently killed the tracking callback mid-stage; metrics
    # are backfilled separately by scripts/backfill_mlflow.py)
    overall, per_class = {}, {}
    for label, name in RUNS.items():
        weights = ROOT / "runs" / "ppe-detection" / name / "weights" / "best.pt"
        if not weights.exists():
            raise SystemExit(f"{weights} missing — training may still be running")

        print(f"evaluating {label} on the held-out test set …")
        m = YOLO(str(weights))
        res = m.val(data=str(TEST_DATA), split="test", verbose=False,
                    project=str(OUT), name=f"eval-{name}", exist_ok=True)
        overall[label] = {"mAP50": round(float(res.box.map50), 4),
                          "mAP50-95": round(float(res.box.map), 4),
                          "precision": round(float(res.box.mp), 4),
                          "recall": round(float(res.box.mr), 4)}
        per_class[label] = {m.names[c]: round(float(res.box.ap50[i]), 4)
                            for i, c in enumerate(res.box.ap_class_index)}

    a, b = list(RUNS.keys())
    tbl = pd.DataFrame(per_class)
    tbl["delta"] = (tbl[b] - tbl[a]).round(4)
    tbl["delta_%"] = (100 * tbl["delta"] / tbl[a].replace(0, float("nan"))).round(1)
    tbl = tbl.sort_values("delta")

    ov = pd.DataFrame(overall).T
    ov.loc["delta"] = (ov.loc[b] - ov.loc[a]).round(4)

    print("\n=== OVERALL (held-out test set) ===")
    print(ov.to_string())
    print("\n=== PER-CLASS AP50 ===")
    print(tbl.to_string())

    verdict = ("IMPROVED" if tbl.loc["NO-Hardhat", "delta"] > 0.01 else
               "UNCHANGED" if abs(tbl.loc["NO-Hardhat", "delta"]) <= 0.01 else "REGRESSED")
    print(f"\nNO-Hardhat (the target class): {verdict} "
          f"({tbl.loc['NO-Hardhat', a]:.3f} -> {tbl.loc['NO-Hardhat', b]:.3f})")

    (OUT / "comparison.md").write_text(
        "# Stage 4 — v1 vs v2 (identical hyperparameters, only the dataset differs)\n\n"
        "## Overall, held-out test set\n\n" + ov.to_markdown() +
        "\n\n## Per-class AP50\n\n" + tbl.to_markdown() +
        f"\n\n**NO-Hardhat verdict: {verdict}**\n")
    (OUT / "comparison.json").write_text(json.dumps(
        {"overall": overall, "per_class": per_class, "no_hardhat_verdict": verdict}, indent=2))

    # chart
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=150)
    y = range(len(tbl))
    ax.barh([i + 0.2 for i in y], tbl[a], 0.38, label=a, color="#898781")
    ax.barh([i - 0.2 for i in y], tbl[b], 0.38, label=b, color="#2a78d6")
    ax.set_yticks(list(y))
    ax.set_yticklabels(tbl.index)
    ax.set_xlabel("AP50 (held-out test set)")
    ax.set_title("Does +500 production images move the weak classes?", fontweight="bold")
    ax.legend(frameon=False)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "comparison.png")
    print(f"\nwritten: {OUT}")


if __name__ == "__main__":
    main()



