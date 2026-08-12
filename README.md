# PPE Detection — Production MLOps Pipeline

[![CI](https://github.com/Raja-Ali-Akhtar/ppe-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Raja-Ali-Akhtar/ppe-detection-mlops/actions/workflows/ci.yml)
[![CD](https://github.com/Raja-Ali-Akhtar/ppe-detection-mlops/actions/workflows/cd.yml/badge.svg)](https://github.com/Raja-Ali-Akhtar/ppe-detection-mlops/actions/workflows/cd.yml)

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
- [x] **Stage 2 — Optimization (ONNX / TensorRT FP16 / INT8)** — fp16: 2× @ zero loss; int8: 4× throughput, 98.9% retention after firing the default calibrator
- [x] **Stage 3 — Cloud deployment with IaC (Triton + Terraform + monitoring)** — 82 req/s local (2× naive FastAPI), deployed to an EC2 T4 for $0.45, destroyed and verified
- [x] **Stage 4 — The retraining loop (data flywheel + drift detection)** — a measured **negative result**: +392 images made the model 8.4% worse, and the per-class pattern says exactly why
- [x] **Stage 5 — CI/CD (GitHub Actions)** — 19 contract tests written from this project's own incidents; lint, image build + health check, terraform validate

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

## Stage 2 — Optimization: ONNX → TensorRT fp32 / fp16 / int8 ✅

**TL;DR: fp16 doubled speed at zero measurable loss — on the same GPU where fp16 *training* is broken. INT8 hit 4× throughput but the textbook-default entropy calibrator silently cost 12 mAP points; switching one line to MinMax calibration recovered them.**

The model flows registry → export → engines with no file paths anywhere:
[`register_model.py`](scripts/register_model.py) promotes the Stage 1 winner to `ppe-detector@baseline-champion`;
[`export_onnx.py`](scripts/export_onnx.py) pulls the alias and exports simplified dynamic ONNX;
[`build_engines.py`](scripts/build_engines.py) builds engines with a batch 1/8/16 optimization profile (spatial pinned 640 — graph capability ≠ deployment contract);
[`benchmark.py`](scripts/benchmark.py) measures everything. Each step is an MLflow run in `ppe-optimization` (params, build times, engine SHA256s, calibration caches).

### Results (GTX 1660 Ti, TensorRT 10.2, YOLOv8s @ 640, 48-image held-out test set)

| Variant | batch-1 p50 | batch-16 throughput | Engine size | mAP50 | mAP50-95 | Retention |
|---|---|---|---|---|---|---|
| PyTorch fp32 | 10.2 ms | 123 img/s | — | 0.555 | 0.316 | 100% |
| TRT fp32 | 9.1 ms | 147 img/s | 56.5 MB | 0.555 | 0.316 | 100% |
| TRT fp16 | 5.2 ms | 287 img/s | 28.9 MB | 0.556 | 0.316 | **100%** |
| TRT int8 (entropy) | 3.8 ms | 502 img/s | 14.0 MB | 0.487 | 0.275 | 87.1% |
| TRT int8 (**MinMax**) | 3.9 ms | 502 img/s | 14.0 MB | 0.552 | 0.312 | **98.9%** |

Method: 50 warmup + 200 timed iterations per batch size, p50/p95/p99 recorded; every variant
(including the PyTorch reference) scored through the identical decode → NMS → FiftyOne
evaluation path, so retention compares precision — never pipeline differences.

### Findings

- **fp16 is the deployment pick**: 2× fp32 speed, retention indistinguishable from 100% — on the same Turing GTX card where fp16 *training* had to be auto-disabled. TU116 swapped tensor cores for double-rate fp16 inference units: same silicon, opposite verdicts for training vs inference.
- **The default calibrator was the INT8 problem**: entropy calibration (96 imgs) cost 13% mAP; MinMax + 192 images recovered to 98.9% at identical speed. Per-class autopsy ([reports/int8-class-report.md](reports/int8-class-report.md)): entropy's worst victim was **Safety Vest (−0.177 AP50)** — hypothesis: hi-vis saturation drives outlier activations, and entropy calibration clips exactly those tails, while MinMax preserves the range.
- **Graph optimization alone is modest** (TRT fp32 = 1.15–1.2× PyTorch) — the speed lives in precision reduction.
- **Engine size halves per precision step** (56.5 → 28.9 → 14.0 MB) while **build time triples in reverse** (106 → 493 → 887 s): cheaper inference costs a longer kernel search. Engines are locked to TRT version + GPU — they are benchmarking artifacts here; deployment (Stage 3) ships the ONNX to Triton and builds on-target.
- TensorRT ≥10.1 deprecates the entire implicit-PTQ calibration API used here (explicit Q/DQ via ModelOpt is the successor) — this stage's INT8 workflow is measurably effective and officially end-of-life, both at once.

## Stage 3 — Serving + Infrastructure as Code ✅

**TL;DR: Triton beat a naive FastAPI server 2× under load — but only after two fixes, and only when clients are close. Deployed to an EC2 T4 with Terraform, benchmarked, destroyed, and verified at $0.45 total.**

```
serving/           triton/model_repository (generated) · gateway (FastAPI→Triton, gRPC)
                   direct/ (model-in-FastAPI baseline) · monitoring/ (Prometheus + Grafana)
                   docker-compose.yml — the same stack locally and in the cloud
terraform/         ECR · security group · IAM instance role · g4dn.xlarge · boot chain
```

### The head-to-head: is Triton worth it over "just FastAPI"?

Both servers expose the identical API and share the *same* preprocessing code
([`serving/common.py`](serving/common.py)). Load generated by
[`scripts/load_test.py`](scripts/load_test.py) (thread-pool clients, status codes asserted):

| Round | Setup | Triton + gateway | FastAPI-direct | What it taught |
|---|---|---|---|---|
| 1 | 16 clients, ORT defaults | 10.3 req/s, p95 **8,020 ms** | 46.8 req/s | the "pro" stack lost 4.5× |
| 2 | + `cudnn_conv_algo_search=HEURISTIC` | 56.3 req/s, p95 420 ms | 46.9 req/s | one config line, p95 ÷19 |
| 3 | 32 clients | 48.4 req/s | 47.5 req/s | both hit the same Python/GIL ceiling |
| 4 | + 4 uvicorn workers | **82.6 req/s** | 41.9 req/s | scale the Python tier → 2× appears |

Round 1's collapse: ONNX Runtime defaults to **exhaustive** cuDNN algorithm search,
which re-tunes for every unseen batch shape — and dynamic batching hands it shapes
1..16. Seconds-long stalls sprinkled through traffic. Round 3's plateau: GPU idle,
Python saturated. **Serving CV at scale is a preprocessing problem wearing an
inference costume.**

### Cloud deployment (EC2 g4dn.xlarge, eu-central-1)

`terraform apply` → instance in 15 s → boot script syncs the model from S3, logs into
ECR by **instance role** (no keys on the machine), pulls Triton from NGC, `compose up`.

| | Cloud T4 | Local GTX 1660 Ti |
|---|---|---|
| cold first request | 1,937 ms | 4,248 ms |
| warm request | **34 ms** | 47 ms |
| 16-client load (from Pakistan) | 19.0 req/s | 82.6 req/s |
| avg dynamic batch formed | **1.04** | ~6 |

**Finding — dynamic batching is a proximity feature.** The T4 was *faster* per request
yet delivered 4× less throughput, with the GPU near 0% utilization. p50 breakdown:
600 ms network · 30 ms preprocessing · 7 ms queue · **34 ms GPU**. At ~600 ms
round-trips, requests arrive alone and Triton's 5 ms batching window never fills.
Benchmark from where your users are, or you're measuring your ISP.

`terraform destroy` → 8 resources removed; independently verified zero instances,
volumes, repos, and elastic IPs. **Session cost: $0.45.**

### API

```
POST /detect   multipart/form-data: file=<jpeg|png>[, ?conf=0.30]
200 -> {"detections":[{"label":"NO-Hardhat","confidence":0.86,
                       "box_xyxy":[710.2,165.5,915.9,303.7]}],
        "count":8,"infer_ms":34.2,"backend":"triton"}
GET  /health   ·  GET /docs (OpenAPI UI)
```
Boxes are pixels in the original image (the gateway reverses letterboxing).

Run it locally: `cd serving && docker compose up -d` → gateway on :9000, Grafana on :3000.

## Stage 4 - The Retraining Loop: a negative result, measured

**TL;DR: I added 392 production images to fix my weakest class. Overall mAP50 fell 0.598 -> 0.547 and that class got 17% worse. The per-class pattern shows exactly why: label quality, not quantity, decided every class.**

The flywheel: serve -> log low-confidence detections -> curate -> dataset v2 -> retrain -> compare.
Traffic = a second, different PPE dataset (3,235 unseen frames) pushed through the live Triton stack.

### Four things that went wrong before the training even started

**1. My hard-case harvester was not a prioritiser.** v1 used trigger rules (any box in
0.25-0.50 confidence, any weak NO-*, zero detections) and flagged **83.8%** of production
frames. Then the control run on the *in-distribution* test set flagged **93.8%**. A rule
that fires on nearly everything selects nothing. Rewritten as per-frame uncertainty
*scoring* + a fixed annotation budget + stratified sampling across failure modes
([`harvest_hard_cases.py`](scripts/harvest_hard_cases.py)).

**2. There was no drift to find.** Mean uncertainty: 0.588 on unseen production data vs
**0.607** on its own test set. The model is equally unsure everywhere - not drifted,
**data-starved**, exactly as Stage 1's "+2.3 mAP for 3.7x parameters" implied.

**3. The production set arrived with its class names stripped to numbers.**
[`infer_label_mapping.py`](scripts/infer_label_mapping.py) recovers them by IoU-matching
our predictions against the unnamed boxes and counting votes: `0 -> Hardhat` (98.9%
purity), `3 -> Safety Vest` (68.1%), `4 -> Mask` (100%); four classes unmappable.
It also produced the real generalisation number: **our model matched only 29.7% of
production ground-truth boxes** (~50% on the three classes it knows).

**4. The partial-label trap.** Adding those images labelled for only 3 of 7 classes would
teach the detector that every unlabelled bare head is *background* - destroying the exact
class we were trying to fix. Partial labels are worse than no labels for detection. So
[`build_dataset_v2.py`](scripts/build_dataset_v2.py) **fuses**: real GT where it exists,
model pseudo-labels (IoU-deduped) elsewhere. Dataset v2 = 901 images, DVC-versioned beside v1.

### The experiment

Identical architecture, hyperparameters and held-out test set; only the dataset differs
(`configs/baseline.yaml` vs `configs/v2-yolov8n.yaml`).

| | v1 (509 imgs) | v2 (901 imgs) | delta |
|---|---|---|---|
| mAP50 | **0.5975** | 0.5473 | **-0.0502** |
| mAP50-95 | 0.3480 | 0.3172 | -0.0308 |
| precision | 0.7943 | 0.6986 | -0.0957 |

| Class | v1 AP50 | v2 AP50 | delta | label source |
|---|---|---|---|---|
| Hardhat | 0.770 | **0.787** | **+2.3%** | 420 real, 98.9% purity |
| Mask | 0.700 | 0.680 | -2.8% | 231 real, 100% purity |
| NO-Mask | 0.505 | 0.483 | -4.4% | pseudo only |
| Person | 0.673 | 0.614 | -8.8% | pseudo only |
| Safety Vest | 0.679 | 0.578 | -14.9% | 639 real, **68% purity** |
| NO-Hardhat | 0.325 | 0.269 | **-17.3%** | pseudo only |
| NO-Safety Vest | 0.531 | 0.421 | **-20.8%** | pseudo only |

**The only class that improved is the only one with clean labels.** The class with the
*most* labels but a third of them mismatched regressed hardest of the real-labelled ones.
Everything fed by pseudo-labels from a model already weak at those classes got worse:
self-training amplifying its own blind spots. **You cannot pseudo-label your way out of a
weakness in the very class doing the labelling** - and the step I skipped to save two
hours (human label verification) was the load-bearing one.

### Monitoring finding: input drift is not behavioural drift

Evidently could not run on this machine (0.7.x/0.4.x hit a Python 3.11.0 typing bug,
0.2.x hits NumPy 2.0), so [`drift_report.py`](scripts/drift_report.py) computes the same
statistics directly - two-sample Kolmogorov-Smirnov + PSI:

| feature | train | prod | KS | p | PSI | drifted |
|---|---|---|---|---|---|---|
| width | 735.2 | 416.0 | 0.848 | 2.8e-92 | 11.81 | yes |
| aspect | 1.40 | 1.00 | 0.848 | 2.8e-92 | 14.28 | yes |
| brightness | 126.6 | 144.4 | 0.276 | 8.7e-09 | 2.09 | yes |
| mean_confidence | 0.377 | 0.404 | 0.120 | 0.055 | 0.25 | **no** |
| n_detections | 3.33 | 3.60 | 0.116 | 0.069 | 0.05 | **no** |
| max_confidence | 0.439 | 0.498 | 0.108 | 0.108 | 0.25 | **no** |

Image geometry drifted enormously; the model's own output distribution did not move.
A monitor watching inputs alone would page you at 3 a.m. for a resize.

### Also learned the hard way

An Evidently install downgraded `pydantic`, which broke `fastapi`, which broke `mlflow` -
and Ultralytics' tracking callback then failed **silently**: 40 epochs trained, nothing
recorded. [`backfill_mlflow.py`](scripts/backfill_mlflow.py) reconstructs a run from the
artifacts Ultralytics always writes to disk. A tracking system that fails loudly is a
nuisance; one that fails quietly is data loss.


## Stage 5 - CI/CD: tests written from this project's own scars

[![CI](https://github.com/Raja-Ali-Akhtar/ppe-detection-mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/Raja-Ali-Akhtar/ppe-detection-mlops/actions/workflows/ci.yml)

**TL;DR: generic CI catches generic bugs. Every test here exists because something in
THIS project broke and cost hours - so the same failure cannot come back quietly.**

`.github/workflows/ci.yml` runs on every push and PR, in ~60 seconds on a free runner,
with **no GPU, no dataset and no model** - deliberately. It tests logic and contracts:

| Job | What it does |
|---|---|
| lint + tests | `ruff` + 20 `pytest` tests |
| gateway image builds | `docker build` the gateway, run it, assert `/health` answers |
| terraform validate | `terraform fmt -check` + `validate` on the infra code |

### The regression tests, and the incident behind each

| Test | The incident it prevents |
|---|---|
| `test_mlflow_can_actually_log` | Stage 4: an Evidently install downgraded `pydantic` -> broke `fastapi` -> broke `mlflow`. Ultralytics' tracking callback then failed **silently**: 40 epochs trained, nothing recorded. Importing is not enough - the test logs a metric and reads it back. |
| `test_triton_config_generator_uses_dynamic_output_dim` | Stage 3: Triton refused to load because `config.pbtxt` said `[11, 8400]` while the ONNX graph says `[-1, 11, -1]`. The config must describe the graph's *capability*, not our deployment intention. |
| `test_triton_config_cudnn_algo_is_integer_enum` | Stage 3: the ORT parameter takes an integer enum as a string. Passing the NAME made Triton refuse the model; leaving the EXHAUSTIVE default gave **p95 8,000 ms**. |
| `test_gateway_uses_grpc_client` | Stage 3: `tritonclient.http` is gevent-based and crashes under uvicorn's asyncio loop ("Cannot switch to a different thread"). |
| `test_decode_maps_boxes_back_to_original_image` | The silent mAP killer: if letterbox/decode geometry drifts between training and serving, every box shifts and nothing errors. |
| `test_class_list_is_consistent_everywhere` | One class list, many consumers - a mismatch silently relabels every detection. |
| `test_all_source_files_are_utf8` | **Found by CI on its first run:** Windows PowerShell writes cp1252 by default, so an em-dash in a comment produced config files Linux could not read. Invisible locally, fatal in CI. |
| `test_no_secrets_committed` | Stage 1: an AWS key leaked and had to be rotated. Never again from a commit. |

That last row is the point of the whole stage: **CI found a bug within sixty seconds of
existing, and it was a bug that could not be reproduced on the machine that wrote it.**

### CD

`.github/workflows/cd.yml` builds the gateway image on changes to `serving/**` and pushes
it to ECR when credentials are present. The deployment target is created and destroyed
per session by Terraform (a $50-budget project does not leave a GPU running), so **CD's
finish line is "a new image in ECR", not "a server was updated"** - and without secrets it
still builds and verifies, so the pipeline stays green and honest rather than red and ignored.
