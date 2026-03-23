# Recency Weighting Backtest — Design Spec

## Goal

Determine whether blending recent actual performance with preseason projections improves prediction accuracy for fantasy-relevant stats, specifically for next-week predictions (start/sit decisions) vs rest-of-season predictions (trade/waiver decisions).

## Data

**Players:** All 230 drafted players from the 2025 season (DRAFT_2025 in backtest_2025.py).

**Game logs:** Per-game stats from the MLB Stats API for all 230 players, cached to `data/stats/game_logs_2025.json`. Hitter logs include AB, H, HR, R, RBI, SB. Pitcher logs include IP, K, ER, BB, H (allowed), W, SV.

**Preseason projections:** Steamer+ZiPS 2025 blend already available in `data/projections/*-2025.csv`.

## Checkpoints

Five dates: **May 1, June 1, July 1, Aug 1, Sep 1**.

At each checkpoint, every model produces per-PA rate predictions for hitters and per-IP rate predictions for pitchers. These predictions are compared against two targets.

## Models

| # | Model | Description |
|---|-------|-------------|
| 1 | Preseason only | Static Steamer+ZiPS blend. Control A. |
| 2 | Season-to-date | Actual cumulative stats through checkpoint, no projection. Control B. |
| 3 | Fixed blend | 30% last-30-day actuals, 70% preseason projection. |
| 4 | Reliability blend | Blend weight scales with sample size. Uses per-stat reliability constants: actual_weight = actual_PA / (actual_PA + reliability_constant). Reliability constants: K%=100 PA, BB%=200 PA, HR rate=200 PA, AVG=400 PA, R/RBI/SB=300 PA. For pitchers: K/9=50 IP, ERA=120 IP, WHIP=80 IP. |
| 5 | Exponential decay | Games weighted by recency with ~7-day half-life. More recent games count more. No hard window cutoff. |

## Accuracy Targets

**Next-week:** Actual per-PA or per-IP rates in the 7 days following the checkpoint. Players with fewer than 10 PA (hitters) or 3 IP (pitchers) in that week are excluded from that checkpoint's measurement.

**Rest-of-season:** Actual per-PA or per-IP rates from checkpoint through end of season. Players with fewer than 50 PA or 20 IP are excluded.

## Stats Measured

**Hitters (per-PA rates):** HR/PA, R/PA, RBI/PA, SB/PA, AVG (H/AB).

**Pitchers (per-IP rates):** K/IP (K/9 / 9), ERA (ER*9/IP), WHIP ((BB+H)/IP). Also W/GS and SV/G but flagged as high-noise.

## Error Metric

Mean absolute error (MAE) across all qualifying players at each checkpoint. Each model-checkpoint-target combination produces one MAE per stat.

## Output

A summary table printed to stdout and saved as CSV (`data/stats/recency_backtest_results.csv`):

```
checkpoint, model, target, stat, mae, n_players
2025-05-01, preseason, next_week, HR/PA, 0.0045, 187
2025-05-01, fixed_blend, next_week, HR/PA, 0.0038, 187
...
```

Plus a human-readable summary answering:
- Does any recency blend beat preseason-only for next-week prediction? By how much?
- Does recency blending hurt ROS prediction accuracy?
- Which blend approach works best?
- Does the answer change across checkpoints (early vs late season)?
- Which stats benefit most from recency weighting?

## Architecture

Single script: `scripts/backtest_recency.py`.

Helper module: `src/fantasy_baseball/analysis/recency.py` — contains the five model functions (preseason, season_to_date, fixed_blend, reliability_blend, exponential_decay) so they're testable independently.

Game log fetcher: utility function in recency.py to fetch and cache game logs via MLB Stats API (raw `requests.get` to the statsapi.mlb.com endpoint, since the `statsapi` Python package doesn't support game logs cleanly).

## Scope

This is a research/analysis script, not a production feature. It answers "should we build recency weighting?" before we invest in building it. No changes to the lineup optimizer or run_lineup.py.
