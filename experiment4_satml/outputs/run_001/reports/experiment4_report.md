# Experiment 4 — SaTML statistical bigram suppression

Method: **SaTML statistical bigram suppression with reconstructed public calibration assets.** This is not an exact-data reproduction.

Calibration is fixed at 128 contexts. The 64-context view is diagnostic only; the final mapping uses all 128. No fingerprint outcome is used to tune mapping or parameters.

## Main results

| Metric | Clean control | No attack | δ=2 | δ=4 | δ=8 | δ=12 |
|---|---|---|---|---|---|---|
| TPR Q20 alpha=0.01 | n/a | 1.0 | 1.0 | 0.92 | 0.0 | 0.0 |
| TPR Q20 alpha=0.0001 | n/a | 1.0 | 0.96 | 0.08 | 0.0 | 0.0 |
| TPR Q50 alpha=0.01 | n/a | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| TPR Q50 alpha=0.0001 | n/a | 1.0 | 1.0 | 0.15 | 0.0 | 0.0 |
| TPR Q100 alpha=0.01 | n/a | 1.0 | 1.0 | 1.0 | 0.0 | 0.0 |
| TPR Q100 alpha=0.0001 | n/a | 1.0 | 1.0 | 0.1 | 0.0 | 0.0 |
| Wrong-key acceptance mean | n/a | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| ARC-C accuracy (%) | 43.68600682593857 | 43.68600682593857 | 44.36860068259386 | 43.77133105802048 | 42.491467576791806 | 41.467576791808874 |
| Mean response length (words) | n/a | 486.29 | 497.681 | 511.789 | 568.856 | 687.339 |
| Truncation rate (%) | n/a | 22.8 | 25.2 | 28.199999999999996 | 50.0 | 73.7 |
| Severe repetition rate (%) | n/a | 0.1 | 0.2 | 0.3 | 20.4 | 65.7 |
| Suppression coverage (%) | n/a | n/a | 51.21544837653548 | 52.23403478540239 | 59.01743116820811 | 70.81270820077104 |
| Clean acceptance Q100 alpha=0.01 | n/a | 0.0 | n/a | n/a | n/a | n/a |
| Clean acceptance Q100 alpha=0.0001 | n/a | 0.0 | n/a | n/a | n/a | n/a |
| ARC-C correct/total | 512/1172 | 512/1172 | 520/1172 | 513/1172 | 498/1172 | 486/1172 |
| Mean generated tokens/response | n/a | 656.451 | 668.601 | 684.77 | 750.38 | 854.944 |

## Fingerprint verification

| Metric | No attack | δ=2 | δ=4 | δ=8 | δ=12 |
|---|---|---|---|---|---|
| TPR Q20 α=0.01 | n/a | 50/50 (1.000) | 50/50 (1.000) | 46/50 (0.920) | 0/50 (0.000) | 0/50 (0.000) |
| TPR Q20 α=0.0001 | n/a | 50/50 (1.000) | 48/50 (0.960) | 4/50 (0.080) | 0/50 (0.000) | 0/50 (0.000) |
| TPR Q50 α=0.01 | n/a | 20/20 (1.000) | 20/20 (1.000) | 20/20 (1.000) | 0/20 (0.000) | 0/20 (0.000) |
| TPR Q50 α=0.0001 | n/a | 20/20 (1.000) | 20/20 (1.000) | 3/20 (0.150) | 0/20 (0.000) | 0/20 (0.000) |
| TPR Q100 α=0.01 | n/a | 10/10 (1.000) | 10/10 (1.000) | 10/10 (1.000) | 0/10 (0.000) | 0/10 (0.000) |
| TPR Q100 α=0.0001 | n/a | 10/10 (1.000) | 10/10 (1.000) | 1/10 (0.100) | 0/10 (0.000) | 0/10 (0.000) |
| TPR Q1000 α=0.01 | n/a | 1/1 (1.000) | 1/1 (1.000) | 1/1 (1.000) | 0/1 (0.000) | 0/1 (0.000) |
| TPR Q1000 α=0.0001 | n/a | 1/1 (1.000) | 1/1 (1.000) | 0/1 (0.000) | 0/1 (0.000) | 0/1 (0.000) |

See per-condition CSVs under `verification/` for group decisions, exact p-values, owner and 20 wrong keys.

Alpha is the nominal spherical random-key null significance level, not empirical clean FPR. Q=1000 has one group and is descriptive only.

## Q=1000 exact owner result

| Condition | Exact p-value | Decision at α=.01 | Decision at α=.0001 |
|---|---:|:---:|:---:|
| baseline | 1.4282119e-27 | True | True |
| delta_2 | 3.292579e-09 | True | True |
| delta_4 | 0.00019473981 | True | False |
| delta_8 | 0.038514257 | False | False |
| delta_12 | 0.053511782 | False | False |

## ARC-Challenge

ARC results are present in `arc/results.csv`.

## Attack diagnostics

| δ | Previous-token coverage | Expected adjusted steps | Expected nonzero mapped values | Mean expected absolute adjustment |
|---:|---:|---:|---:|---:|
| 2 | 0.5122 | 0.5122 | 1.000000 | 0.53381413 |
| 4 | 0.5223 | 0.5223 | 1.000000 | 1.0670165 |
| 8 | 0.5902 | 0.5902 | 1.000000 | 2.1417981 |
| 12 | 0.7081 | 0.7081 | 1.000000 | 3.2762367 |

Interpret retained TPR together with coverage before claiming robustness.
