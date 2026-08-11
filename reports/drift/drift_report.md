# Production vs training drift

reference: 250 training images · current: 250 production images

| feature         |   train_mean |   prod_mean |   ks_stat |   p_value |    psi | drifted   |
|:----------------|-------------:|------------:|----------:|----------:|-------:|:----------|
| width           |      735.196 |     416     |     0.848 |  2.82e-92 | 11.814 | True      |
| aspect          |        1.401 |       1     |     0.848 |  2.82e-92 | 14.278 | True      |
| height          |      555.008 |     416     |     0.548 |  8.06e-35 |  4.064 | True      |
| brightness      |      126.589 |     144.352 |     0.276 |  8.69e-09 |  2.089 | True      |
| violation_share |        0.188 |       0.154 |     0.144 |  0.0111   |  0.531 | True      |
| mean_confidence |        0.377 |       0.404 |     0.12  |  0.0546   |  0.246 | False     |
| n_detections    |        3.328 |       3.596 |     0.116 |  0.0691   |  0.053 | False     |
| max_confidence  |        0.439 |       0.498 |     0.108 |  0.108    |  0.246 | False     |
| contrast        |       61.303 |      58.375 |     0.096 |  0.2      |  0.265 | False     |

**5/9 features drifted** (KS p<0.05). Dataset-level drift: True
