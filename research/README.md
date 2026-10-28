# Model research — backtest & calibration

Offline, free, out-of-sample validation + calibration of the football model
(the same Dixon-Coles attack/defence used by the app engine). No API key / Pro
needed — uses the free football-data.co.uk mirror.

## Data (not committed)
```
mkdir -p data
curl -L -o data/Matches.csv \
  https://raw.githubusercontent.com/xgabora/Club-Football-Match-Data-2000-2025/main/data/Matches.csv
```
Results + half-time + corners + yellow/red cards + bookmaker odds, 2000–2025,
for E0/SP1/I1/D1/F1/T1 (and more).

## Run
```
python3 backtest_research.py   # baseline metrics + calibration curves + bookmaker benchmark
python3 tune_calibrate.py      # grid search + per-market calibration (train pre-2021 / test 2021+)
```

## Findings (2026-10-28, ~30k out-of-sample matches)
- Model is competitive with the bookmaker on the core markets:
  1X2 log-loss 0.993 vs market 0.970 (acc 51.8% vs 53.9%); O/U2.5 0.682 vs 0.673.
- Best hyper-params: half-life **540d**, rho **-0.11** (now applied in config.py /
  football_model.py). Shrinkage barely matters.
- **Calibration is the key win** — raw probabilities are over-confident. Per-market
  temperature (p_cal = sigmoid(logit(p)/T + b), fit pre-2021, validated 2021+):
  - 1X2: T≈1.0 (already calibrated)
  - Over/Under goals: T≈1.5, b≈-0.05
  - BTTS: T≈2.0
  - Cards O/U: T≈1.1 (has real signal)
  - Corners O/U: ~no signal at 9.5 (actual ≈50% regardless) → never headline a
    corners "edge"; damp confidence toward 50%.
- Candidate next improvement: xG-style ratings from shots-on-target (data has
  HomeTarget/AwayTarget) — often less noisy than goals-only.
