# Injury-Aware Projection & Dynamic Replacement Rates

## Problem

The draft valuation pipeline has two related issues that overvalue injury-prone pitchers:

1. **Hardcoded replacement ERA/WHIP (4.50/1.35)** are more generous than the actual replacement-level pitcher in the pool (~3.89 ERA). This inflates every pitcher's rate stat SGP, compounding the SP overvaluation flagged in the group review.

2. **No injury backfill cost in VAR.** A pitcher projected for 145 IP at 3.20 ERA gets valued on those stats alone. In reality, the team must fill ~33 IP with waiver-quality pitching (~4.20 ERA), dragging the effective contribution down. The same applies to hitters projected for fewer AB than a healthy season.

## Design

### Change 1: Dynamic replacement rates from the pool

`calculate_replacement_levels` currently returns `{position: sgp}`. It will additionally return the replacement-level pitcher's ERA and WHIP, and the replacement-level hitter's AVG, derived from the actual player pool at the replacement threshold (e.g., the 91st pitcher).

`build_draft_board` passes these pool-derived rates to `calculate_player_sgp` instead of the hardcoded defaults. This means every pitcher's rate stat SGP is computed against the real replacement baseline.

**Interface change:**

```python
# replacement.py
def calculate_replacement_levels(player_pool, starters_per_position=None):
    """Returns (replacement_sgp_dict, replacement_rates_dict).

    replacement_rates_dict contains:
      - "era": float  (replacement pitcher ERA)
      - "whip": float (replacement pitcher WHIP)
      - "avg": float  (replacement hitter AVG)
    """
```

### Change 2: Injury backfill blending

Before SGP is calculated, players projected below a healthy baseline have their stats blended with waiver-quality replacement stats for the gap innings/ABs. This produces an "effective" stat line that reflects the true team-level cost of drafting an injury-prone player.

**Healthy baselines:**
- SP: 178 IP (~31 starts x 5.7 IP/start)
- Closer: 60 IP
- Hitter: 600 AB

**Thresholds** (gap must exceed these before backfill applies):
- SP: 15 IP
- Closer: 10 IP
- Hitter: 50 AB

**Waiver-quality replacement stats:**
- SP: ~4.20 ERA, ~1.30 WHIP, 7 W/140 IP, 120 K/140 IP (scaled to gap)
- RP: ~4.50 ERA, ~1.35 WHIP, 5 SV/60 IP (to be calibrated post-season)
- Hitter: ~.250 AVG, modest counting stats (55 R, 12 HR, 50 RBI, 5 SB per 600 AB, scaled to gap)

**Blend formula** (pitcher example):

```
gap_ip = max(0, baseline_ip - projected_ip)
if gap_ip <= threshold:
    # Normal projection, no adjustment
    use projected stats as-is

effective_ip = projected_ip + gap_ip  # = baseline
effective_er = projected_er + (waiver_era / 9 * gap_ip)
effective_bb = projected_bb + (waiver_bb_rate * gap_ip)
effective_h_allowed = projected_h_allowed + (waiver_ha_rate * gap_ip)
effective_w = projected_w + (waiver_w_rate * gap_ip)
effective_k = projected_k + (waiver_k_rate * gap_ip)

# SGP calculated from effective stats
```

For hitters, the same pattern: gap AB filled with replacement-quality counting stats and .250 AVG.

**Closer detection:** A pitcher with projected SV >= `CLOSER_SV_THRESHOLD` (20) uses the closer baseline (60 IP) and threshold (10 IP). All other pitchers use the SP baseline (178 IP) and threshold (15 IP).

**Display:** The original projected stats are preserved on the board for the dashboard. The blending only affects the `total_sgp` column used for VAR/VONA ranking. The dashboard could optionally show "effective IP" or a health flag, but that's a future enhancement.

### Files changed

| File | Change |
|------|--------|
| `utils/constants.py` | New constants: healthy baselines, thresholds, waiver replacement stats |
| `sgp/replacement.py` | Return replacement pitcher ERA/WHIP and hitter AVG alongside SGP levels |
| `sgp/player_value.py` | Accept optional backfill config; blend stats before SGP when gap exceeds threshold |
| `draft/board.py` | Pass pool-derived replacement rates and backfill config to SGP calculation |
| `config/league.yaml.example` | Document optional baseline/threshold overrides |

### Files NOT changed

- `draft/recommender.py` — consumes VAR, unaffected
- `draft/strategy.py` — consumes recommendations, unaffected
- `lineup/` — uses standings leverage, not replacement levels
- `draft/projections.py` — Monte Carlo already models injuries independently

### Testing

- Unit test: `calculate_replacement_levels` returns correct rate stats from a known pool
- Unit test: backfill blending produces expected effective stats for a 145 IP pitcher vs a 185 IP pitcher
- Unit test: backfill does NOT apply when gap is below threshold
- Unit test: closer uses 60 IP baseline, SP uses 178 IP baseline
- Integration test: `build_draft_board` with backfill produces lower VAR for injury-prone pitchers than without
- Regression: existing tests continue to pass (backfill is opt-in via config, defaults match current behavior during transition)

### Rollout

Backfill blending defaults to ON for new boards. The hardcoded replacement rates are replaced by pool-derived rates unconditionally (this is strictly more correct). Both changes can be disabled by setting baselines to 0 in league.yaml overrides.
