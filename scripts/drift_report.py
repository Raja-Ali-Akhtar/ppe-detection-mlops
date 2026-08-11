"""Evidently drift report: production traffic vs the training distribution.

    python scripts/drift_report.py --limit 400

Production data has no trustworthy labels, so accuracy cannot be monitored
directly. What CAN be monitored are proxies computed from the images and from
the model's own output:

  image features      width, height, aspect ratio, mean brightness, contrast
  prediction features detections per frame, max/mean confidence, violation share

Evidently compares the reference (training set) against the current (production)
window and reports per-column drift. This is the standard unlabelled-CV setup:
you watch the INPUTS and the model's BEHAVIOUR, not its accuracy.

Output: reports/drift/drift_report.html + drift_summary.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "drift"


def features(paths, endpoint, conf, tag):
    rows = []
    for i, p in enumerate(paths, 1):
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row = {"width": w, "height": h, "aspect": round(w / h, 3),
               "brightness": round(float(gray.mean()), 2),
               "contrast": round(float(gray.std()), 2)}
        try:
            with open(p, "rb") as f:
                r = requests.post(endpoint, files={"file": f},
                                  params={"conf": conf}, timeout=60)
            dets = r.json()["detections"]
        except Exception:
            dets = []
        confs = [d["confidence"] for d in dets]
        row.update({
            "n_detections": len(dets),
            "max_confidence": round(max(confs), 3) if confs else 0.0,
            "mean_confidence": round(float(np.mean(confs)), 3) if confs else 0.0,
            "violation_share": round(
                sum(d["label"].startswith("NO-") for d in dets) / len(dets), 3) if dets else 0.0,
        })
        rows.append(row)
        if i % 100 == 0:
            print(f"  {tag}: {i}/{len(paths)}")
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:9000/detect")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--conf", type=float, default=0.30)
    args = ap.parse_args()

    ref_paths = sorted((ROOT / "data/processed/ppe-7cls-v1/train/images").glob("*.jpg"))[: args.limit]
    cur_paths = sorted((ROOT / "data/production/train/images").glob("*.jpg"))[: args.limit]
    print(f"reference (training): {len(ref_paths)} | current (production): {len(cur_paths)}")

    ref = features(ref_paths, args.endpoint, args.conf, "reference")
    cur = features(cur_paths, args.endpoint, args.conf, "current")

    OUT.mkdir(parents=True, exist_ok=True)
    # NOTE: Evidently could not be used on this machine — 0.7.x fails on a
    # Python 3.11.0 typing bug (KeyError: ~TResult in nested generics, fixed in
    # 3.11.1+), 0.4.x fails identically, 0.2.x fails on NumPy 2.0. Rather than
    # grind, we compute the SAME statistics it would: two-sample
    # Kolmogorov-Smirnov per column (its default numeric drift test) plus PSI.
    from scipy import stats

    def psi(a, b, bins=10):
        """Population Stability Index — the finance/industry standard drift metric.
        <0.1 no shift · 0.1-0.25 moderate · >0.25 major shift."""
        edges = np.histogram_bin_edges(np.concatenate([a, b]), bins=bins)
        pa = np.histogram(a, edges)[0] / max(1, len(a))
        pb = np.histogram(b, edges)[0] / max(1, len(b))
        pa, pb = np.clip(pa, 1e-6, None), np.clip(pb, 1e-6, None)
        return float(np.sum((pb - pa) * np.log(pb / pa)))

    rows = []
    for col in ref.columns:
        a, b = ref[col].to_numpy(float), cur[col].to_numpy(float)
        ks, p = stats.ks_2samp(a, b)
        rows.append({
            "feature": col,
            "train_mean": round(float(a.mean()), 3),
            "prod_mean": round(float(b.mean()), 3),
            "ks_stat": round(float(ks), 3),
            "p_value": float(f"{p:.3g}"),
            "psi": round(psi(a, b), 3),
            "drifted": bool(p < 0.05),
        })
    drift = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)

    summary = {
        "method": "KS two-sample (p<0.05) + PSI — Evidently blocked by py3.11.0 typing bug",
        "reference_rows": len(ref), "current_rows": len(cur),
        "n_drifted": int(drift["drifted"].sum()), "n_features": len(drift),
        "dataset_drift": bool(drift["drifted"].mean() > 0.5),
        "features": drift.to_dict("records"),
    }
    (OUT / "drift_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "drift_report.md").write_text(
        "# Production vs training drift\n\n"
        f"reference: {len(ref)} training images · current: {len(cur)} production images\n\n"
        + drift.to_markdown(index=False)
        + f"\n\n**{summary['n_drifted']}/{len(drift)} features drifted** "
          f"(KS p<0.05). Dataset-level drift: {summary['dataset_drift']}\n")
    print("\n--- drift (KS + PSI) ---")
    print(drift.to_string(index=False))
    print(f"\n{summary['n_drifted']}/{len(drift)} features drifted · dataset_drift={summary['dataset_drift']}")

    print("\n--- distribution comparison (means) ---")
    comp = pd.DataFrame({"training": ref.mean(numeric_only=True),
                         "production": cur.mean(numeric_only=True)}).round(3)
    comp["delta_%"] = ((comp["production"] - comp["training"]) / comp["training"].replace(0, np.nan) * 100).round(1)
    print(comp.to_string())
    print(f"\nreport: {OUT / 'drift_report.html'}")


if __name__ == "__main__":
    main()
