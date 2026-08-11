# Stage 4 — v1 vs v2 (identical hyperparameters, only the dataset differs)

## Overall, held-out test set

|               |   mAP50 |   mAP50-95 |   precision |   recall |
|:--------------|--------:|-----------:|------------:|---------:|
| v1 (509 imgs) |  0.5975 |     0.348  |      0.7943 |   0.5553 |
| v2 (901 imgs) |  0.5473 |     0.3172 |      0.6986 |   0.5415 |
| delta         | -0.0502 |    -0.0308 |     -0.0957 |  -0.0138 |

## Per-class AP50

|                |   v1 (509 imgs) |   v2 (901 imgs) |   delta |   delta_% |
|:---------------|----------------:|----------------:|--------:|----------:|
| NO-Safety Vest |          0.5312 |          0.4208 | -0.1104 |     -20.8 |
| Safety Vest    |          0.6785 |          0.5776 | -0.1009 |     -14.9 |
| Person         |          0.6731 |          0.6142 | -0.0589 |      -8.8 |
| NO-Hardhat     |          0.325  |          0.2689 | -0.0561 |     -17.3 |
| NO-Mask        |          0.5053 |          0.4829 | -0.0224 |      -4.4 |
| Mask           |          0.6997 |          0.6801 | -0.0196 |      -2.8 |
| Hardhat        |          0.7695 |          0.787  |  0.0175 |       2.3 |

**NO-Hardhat verdict: REGRESSED**
