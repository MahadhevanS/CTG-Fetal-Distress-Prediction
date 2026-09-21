# Protocol sweep -- does the split rule or the architecture drive AUROC?

Data: `data/processed_clinical/` | target `y_primary` | 20 epochs | seed 42

`window_random` is a NEGATIVE CONTROL, not a result. It is reported only to quantify how much a window-level split inflates this benchmark.


## Window-level AUROC (5-fold CV mean)

| encoder | params | patient-grouped | window-random | inflation |
|---|---:|---:|---:|---:|
| cnn1d | 143,745 | 0.5798 | 0.6784 | +0.0986 |
| bilstm | 167,553 | 0.5799 | 0.5944 | +0.0146 |
| gru | 195,969 | 0.5625 | 0.6512 | +0.0887 |
| tcn | 383,521 | 0.5764 | 0.6151 | +0.0386 |
| mslstm | 600,545 | 0.6200 | 0.6638 | +0.0438 |
| patchctg | 687,361 | 0.5791 | 0.6556 | +0.0765 |
| patchtst | 701,697 | 0.6215 | 0.6627 | +0.0412 |
| crossformer | 2,504,385 | 0.5319 | 0.7642 | +0.2322 |

## Patient-level AUROC -- the clinically meaningful number

| encoder | grouped (max-agg) | grouped (mean-agg) | window-random (max-agg) |
|---|---:|---:|---:|
| cnn1d | 0.6920 | 0.6292 | 0.7299 |
| bilstm | 0.6593 | 0.6673 | 0.6224 |
| gru | 0.6624 | 0.5761 | 0.6930 |
| tcn | 0.6184 | 0.6045 | 0.6452 |
| mslstm | 0.7178 | 0.7034 | 0.7227 |
| patchctg | 0.6775 | 0.6224 | 0.6910 |
| patchtst | 0.7175 | 0.6857 | 0.7067 |
| crossformer | 0.6167 | 0.5637 | 0.8035 |

## Reference -- 19 features + logistic regression, identical data

| protocol | real labels | random labels |
|---|---:|---:|
| window-random | 0.8979 | 0.9094 |
| patient-grouped | 0.6254 | 0.5170 |

Patient-grouped patient-level feature baseline: **0.7290 +/- 0.045** (pooled 5x5 CV, 547 patients).

