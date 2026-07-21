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

- [ ] **Stage 1 — Data + baseline training with tracking** *(in progress)*
- [ ] Stage 2 — Optimization (ONNX / TensorRT FP16 / INT8)
- [ ] Stage 3 — Cloud deployment with IaC (Triton + Terraform + monitoring)
- [ ] Stage 4 — The retraining loop (data flywheel + drift detection)
- [ ] Stage 5 — CI/CD

## Stage 1 — Data + Baseline Training with Tracking

*(section grows as the stage completes: dataset stats from FiftyOne curation, DVC setup, MLflow baseline runs)*
