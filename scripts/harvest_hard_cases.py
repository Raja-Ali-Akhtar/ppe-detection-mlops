"""Run production traffic through the served model and harvest HARD CASES.

Usage:
    python scripts/harvest_hard_cases.py --limit 300
    python scripts/harvest_hard_cases.py --limit 300 --endpoint http://localhost:9000

Why confidence-band harvesting: a detection at 0.35 is the model saying "I think
there's a bare head here, but I'm not sure." Those are the samples worth a human's
time — random new images are mostly redundant, uncertainty is a free prioritizer.

Harvest rules (any one triggers):
  * a detection in the UNCERTAIN band (0.25 <= conf < 0.50)
  * ZERO detections at all on a real scene (silent failure — the worst kind)
  * a NO-* violation class at any confidence below 0.60 (the weak classes we
    are explicitly hunting; NO-Hardhat sits at 0.276 AP)

Outputs:
  data/hard_cases/images/*.jpg          the harvested frames
  data/hard_cases/predictions.json      model output per frame (review aid)
  data/hard_cases/summary.json          counts + reasons
  and mirrors them to S3 (the "inference-time logging" the plan calls for)
"""

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "hard_cases"
S3_PREFIX = "s3://ppe-mlops-dvc-428232898120/hard-cases"

UNCERTAIN_LO, UNCERTAIN_HI = 0.25, 0.50
VIOLATION_REVIEW_MAX = 0.60


def score_frame(dets):
    """Uncertainty score in [0,1] — higher means 'a human should look at this'.

    LESSON (measured, not assumed): the first version of this used trigger RULES
    ("any box in 0.25-0.50") and selected 84% of production frames — but also
    93.8% of the in-distribution TEST set. A rule that fires on almost everything
    is not a prioritiser. Real active learning SCORES frames and spends a fixed
    annotation budget on the top of the ranking.

    Score components:
      1.0                      nothing detected at all (silent failure)
      1 - max_confidence       the model's best guess on this frame is weak
      + 0.15 per weak NO-*     bias toward the classes we know are weak
    """
    if not dets:
        return 1.0, ["no_detections"]

    reasons = []
    top = max(d["confidence"] for d in dets)
    score = 1.0 - top
    if top < UNCERTAIN_HI:
        reasons.append("whole_frame_uncertain")

    weak_viol = [d for d in dets
                 if d["label"].startswith("NO-") and d["confidence"] < VIOLATION_REVIEW_MAX]
    if weak_viol:
        score += 0.15 * min(2, len(weak_viol))
        reasons.append("weak_violation:" + weak_viol[0]["label"])

    band = [d for d in dets if UNCERTAIN_LO <= d["confidence"] < UNCERTAIN_HI]
    if len(band) >= 2:                       # several hesitant boxes, not just one
        score += 0.10
        reasons.append("multiple_uncertain")

    return min(1.0, score), reasons or ["confident"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://localhost:9000/detect")
    ap.add_argument("--source", default="data/production")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--conf", type=float, default=0.20,
                    help="collection threshold — LOWER than serving, so we see hesitation")
    ap.add_argument("--budget", type=int, default=150,
                    help="annotation budget: keep the top-N most uncertain frames")
    ap.add_argument("--score-only", action="store_true", help="report the score distribution, copy nothing")
    ap.add_argument("--no-s3", action="store_true")
    args = ap.parse_args()

    images = sorted((ROOT / args.source).rglob("*.jpg"))[: args.limit]
    if not images:
        raise SystemExit(f"no images under {args.source}")
    print(f"{len(images)} frames of production traffic -> {args.endpoint}")

    img_dir = OUT / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    scored, latencies = [], []

    for i, img in enumerate(images, 1):
        with open(img, "rb") as f:
            try:
                r = requests.post(args.endpoint, files={"file": f},
                                  params={"conf": args.conf}, timeout=60)
                r.raise_for_status()
            except Exception as e:
                print(f"  request failed on {img.name}: {e}")
                continue
        payload = r.json()
        dets = payload["detections"]
        latencies.append(payload.get("infer_ms", 0))
        score, why = score_frame(dets)
        scored.append({"file": str(img), "name": img.name, "score": round(score, 4),
                       "reasons": why, "n_det": len(dets), "detections": dets})
        if i % 100 == 0:
            print(f"  {i}/{len(images)} scored")

    scored.sort(key=lambda s: -s["score"])
    # Stratified selection: taking the raw top-N gives 100% "no detections" frames
    # (score 1.0 dominates) — that teaches the model ONE failure mode. Split the
    # budget across failure types so the curated batch is diverse.
    silent = [s for s in scored if s["n_det"] == 0]
    weak_v = [s for s in scored if s["n_det"] and any("weak_violation" in r for r in s["reasons"])]
    uncert = [s for s in scored if s["n_det"] and s not in weak_v]
    quota = [(silent, 0.45), (weak_v, 0.35), (uncert, 0.20)]
    selected, seen = [], set()
    for pool, share in quota:
        for s in pool[: int(args.budget * share)]:
            if s["name"] not in seen:
                selected.append(s)
                seen.add(s["name"])
    for s in scored:                                   # top up if a pool ran dry
        if len(selected) >= args.budget:
            break
        if s["name"] not in seen:
            selected.append(s)
            seen.add(s["name"])
    reason_counts = Counter(r.split(":")[0] for s in selected for r in s["reasons"])

    records = {}
    if not args.score_only:
        for s in selected:
            shutil.copy2(s["file"], img_dir / s["name"])
            records[s["name"]] = {"detections": s["detections"], "reasons": s["reasons"],
                                  "score": s["score"]}
        (OUT / "predictions.json").write_text(json.dumps(records, indent=2))

    all_scores = [s["score"] for s in scored]
    srt = sorted(all_scores)

    def q(p):
        return round(srt[int(p * (len(srt) - 1))], 3)

    summary = {
        "source": args.source,
        "frames_scored": len(scored),
        "annotation_budget": args.budget,
        "selected": len(selected),
        "score_p50": q(0.50), "score_p90": q(0.90), "score_max": round(max(all_scores), 3),
        "mean_score": round(sum(all_scores) / len(all_scores), 3),
        "frac_zero_detections": round(sum(1 for s in scored if s["n_det"] == 0) / len(scored), 3),
        "selection_reasons": dict(reason_counts),
        "mean_infer_ms": round(sum(latencies) / max(1, len(latencies)), 1),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))

    if not args.no_s3:
        import subprocess
        print("mirroring to S3 …")
        subprocess.run(["aws", "s3", "sync", str(OUT), S3_PREFIX,
                        "--region", "eu-central-1", "--only-show-errors"], check=False)
        print(f"logged to {S3_PREFIX}")


if __name__ == "__main__":
    main()


