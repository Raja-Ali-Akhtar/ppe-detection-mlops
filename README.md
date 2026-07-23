# PPE Detection — Production MLOps Pipeline

Detecting hardhats and safety vests on construction sites — built as a **production-grade, cloud-native MLOps system**, not just a model.

This is the sequel to my [monocular depth estimation series](https://github.com/Raja-Ali-Akhtar/monocular-depth-benchmark-Depth-estimation-) (edge deployment). That series was edge; this one is cloud. Together: **Production CV — Edge & Cloud.**

## Stack

| Layer | Tool |
|---|---|
| Experiment tracking + model registry | MLflow |
| Data versioning | DVC (S3 remote) |
| Dataset curation | FiftyOne |
| Training | Ultralytics YOLO (local GTX 1660 Ti) |
| Optimization | ONNX → TensorRT (FP16 / INT8) |
| Serving | NVIDIA Triton Inference Server + FastAPI gateway |
| Infrastructure as Code | Terraform |
| CI/CD | GitHub Actions |
| Monitoring | Prometheus + Grafana (system), Evidently (drift) |
| Cloud | AWS (S3, ECR, EC2/SageMaker) |

## Stages

- [x] **Stage 1 — Data + baseline training with tracking** — mAP50 0.528 baseline; 717→509 images after curation; DVC+S3; 4 MLflow runs
- [ ] Stage 2 — Optimization (ONNX / TensorRT FP16 / INT8)
- [ ] Stage 3 — Cloud deployment with IaC (Triton + Terraform + monitoring)
- [ ] Stage 4 — The retraining loop (data flywheel + drift detection)
- [ ] Stage 5 — CI/CD

## Stage 1 — Data + Baseline Training with Tracking ✅

**TL;DR: the dataset advertised 717 images. After curation it was 509 unique photos — and the baked-in augmentation had leaked near-identical images across train/val. Found before training a single epoch, with free tools.**

### Know your data (FiftyOne curation)

Dataset: [Construction Site Safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety/dataset/30) (Roboflow Universe, CC BY 4.0), pinned at v30.

| Finding | Number |
|---|---|
| Images as downloaded | 717 (25 classes) |
| Classes after scoping to PPE | 7 (Hardhat, NO-Hardhat, Safety Vest, NO-Safety Vest, Person, Mask, NO-Mask) |
| Raw class imbalance | 1148:1 (Person vs. bus — a 1-instance class) |
| Imbalance after scoping | ~5.7:1 |
| Exact duplicates (MD5) | 0 |
| Near-duplicates (embeddings) | 198 candidates (~28%) — mostly augmented copies baked into the export |
| **Cross-split near-dup pairs (train/val leakage)** | **83 — validation metrics would have been inflated** |
| Final training set | 509 images, leakage-free splits |

Every curation decision is a committed, regenerable artifact: detector scripts in [`scripts/`](scripts/) write exclusion lists to [`curation/`](curation/), and [`build_processed.py`](scripts/build_processed.py) deterministically rebuilds `data/processed/` from immutable `data/raw/`. Data lineage: raw → committed decisions → processed.

### Data versioning (DVC + S3)

The repo holds 5-line `.dvc` pointer files; content lives hash-addressed in S3.
Reproduce any commit's exact dataset:

```
git checkout <commit>
dvc pull
```

### Experiments (MLflow, SQLite backend)

Config-driven training ([`scripts/train.py`](scripts/train.py) + [`configs/`](configs/)) — one YAML per run, no CLI hyperparameters. All runs logged with params, per-epoch metrics, and weight artifacts; winner registered in the MLflow Model Registry as `ppe-detector` v1.

| Run | Model | imgsz | mAP50 | mAP50-95 |
|---|---|---|---|---|
| exp-yolov8s | YOLOv8s (11M) | 640 | **0.551** | **0.314** |
| baseline-yolov8n | YOLOv8n (3M) | 640 | 0.528 | 0.306 |
| exp-imgsz320 | YOLOv8n | 320 | 0.481 | 0.262 |

**Held-out test set** (48 images, untouched by training or model selection): the winning model scores **0.577 mAP50 / 0.332 mAP50-95** — agreeing with val within noise, i.e. no evidence the val-driven model selection inflated the numbers.

**Findings (honest, small-val-set caveats apply — 60 val / 48 test images):**
- **Halving resolution barely hurts `Hardhat` (0.737 → 0.727 mAP50) but collapses `NO-Hardhat` (0.456 → 0.296).** Detecting a bright hardhat survives 320px; detecting a *bare head* — the violation that safety systems exist to catch — does not. Resolution decisions should be driven by the violation classes, not the average.
- The NO-* (violation) classes lag their positive counterparts consistently: absence is harder to detect than presence. **NO-Hardhat is the confirmed weakest class on every evaluation (0.456 val, 0.276 test) — the retraining loop's first data-collection target.**
- 3.7× more parameters bought only +2.3 mAP50 points: with 509 images, **data is the bottleneck, not architecture** (the Stage 4 retraining loop exists for exactly this reason).
- `Mask` mAP is reported but meaningless (6 val instances).

Hardware note: trained on a GTX 1660 Ti (6 GB). AMP auto-disabled (known Turing FP16 issue). YOLOv8s at batch 8 silently spilled into system RAM via Windows CUDA fallback — 4-8× slower epochs; watch `GPU_mem` exceeding VRAM.
