# Injury-Aware Projection & Dynamic Replacement Rates

## Problem

The draft valuation pipeline has two related issues that overvalue injury-prone pitchers:

1. **Hardcoded replacement ERA/WHIP (4.50/1.35)** are more generous than the actual replacement-level pitcher in the pool (~3.89 ERA). This inflates every pitcher's rate stat SGP, compounding the SP overvaluation flagged in the group review.

2. **No injury backfill cost in VAR.** A pitcher projected for 145 IP at 3.20 ERA gets valued on those stats alone. In reality, the team must fill ~33 IP with waiver-quality pitching (~4.20 ERA), dragging the effective contribution down. The same applies to hitters projected for fewer AB than a healthy season.

## Design

### Change 1: Dynamic replacement rates from the pool

`calculate_replacement_levels` currently returns `{position: sgp}`. A new companion function `calculate_replacement_rates` will compute the replacement-level pitcher's ERA/WHIP and hitter's AVG from the pool. This avoids changing the existing return type and breaking callers.

**Algorithm:** To smooth noise from a single player, average the ERA/WHIP/AVG of pitchers (or hitters) ranked from `num_starters - 5` to `num_starters + 5` around the replacement threshold. Players with 0 IP (or 0 AB) are excluded from the average to avoid division-by-zero artifacts.

**New function in `replacement.py`:**

```python
def calculate_replacement_rates(
    player_pool: pd.DataFrame,
    starters_per_position: dict[str, int] | None = None,
) -> dict[str, float]:
    """Derive replacement-level rate stats from the pool.

    Returns {"era": float, "whip": float, "avg": float} using a band
    of players around each replacement threshold to smooth noise.
    """
```

`build_draft_board` calls this once at board-build time and passes the rates as keyword arguments to `calculate_player_sgp`. The existing hardcoded defaults in `player_value.py` (`REPLACEMENT_ERA = 4.50`, etc.) remain as fallback defaults — they are not removed or changed, so `waivers.py`, `weighted_sgp.py`, and `recommender.py` (which use them for in-season and in-draft leverage calculations) are unaffected.

**Conscious decision:** `total_sgp` is computed once at board-build time using the initial pool's replacement rates. The live per-pick recalculation in `get_recommendations` only recomputes replacement *levels* (SGP thresholds) and VAR, not replacement *rates* or total_sgp. Recomputing total_sgp per pick would require re-running SGP for all ~3,600 players, which is too expensive for the interactive draft path. The initial pool rates are a good approximation since the marginal players near replacement don't shift much over a draft.

### Change 2: Injury backfill blending

**Where it happens:** In `build_draft_board` (in `board.py`), after the pool DataFrame is assembled but before the `.apply(calculate_player_sgp)` call. The blending modifies counting stat columns (`er`, `bb`, `h_allowed`, `w`, `k`, `ip`, `h`, `ab`, `r`, `hr`, `rbi`, `sb`) on the pool DataFrame directly. These component columns already exist on the board from projection blending.

The player's original stats are preserved in separate columns (e.g., `orig_ip`, `orig_era`) for dashboard display. The modified columns feed into `calculate_player_sgp` which then produces a `total_sgp` that reflects the backfill cost.

**Player classification:**
- **Closer**: projected SV >= `CLOSER_SV_THRESHOLD` (20). Baseline 60 IP, threshold 10 IP.
- **Starter**: `"SP" in positions` or IP >= 100. Baseline 178 IP, threshold 15 IP.
- **Middle reliever**: all other pitchers (RP with < 20 SV and < 100 IP). No backfill — their low IP is their normal workload, not an injury signal.
- **Hitter**: baseline 600 AB, threshold 50 AB.

This three-tier pitcher classification avoids the cliff effect where a setup man (19 SV, 60 IP) would get penalized against a 178 IP baseline.

**Healthy baselines** (configurable in `league.yaml`):
- SP: 178 IP (~31 starts × 5.7 IP/start)
- Closer: 60 IP
- Hitter: 600 AB

**Thresholds** (gap must exceed these before backfill applies):
- SP: 15 IP
- Closer: 10 IP
- Hitter: 50 AB

**Waiver-quality replacement stats** (new constants, separate from existing `REPLACEMENT_SP`/`REPLACEMENT_RP`/`REPLACEMENT_HITTER` which are used by Monte Carlo):

```python
# Waiver-quality stats for backfill blending (10-team league)
WAIVER_SP = {"w": 7, "k": 120, "sv": 0, "ip": 140, "er": 65, "bb": 48, "h_allowed": 133}
# ERA = 65*9/140 = 4.18, WHIP = (48+133)/140 = 1.29

WAIVER_RP = {"w": 2, "k": 55, "sv": 5, "ip": 60, "er": 30, "bb": 21, "h_allowed": 60}
# ERA = 4.50, WHIP = 1.35 (to be calibrated post-season, see post_season_analysis.md)

WAIVER_HITTER = {"r": 55, "hr": 12, "rbi": 50, "sb": 5, "h": 150, "ab": 600}
# AVG = 150/600 = .250
```

These are intentionally separate from the Monte Carlo `REPLACEMENT_*` constants. The MC constants model true replacement level (worst rostered player). The `WAIVER_*` constants model what you'd actually stream from waivers in a 10-team league — slightly better quality.

**Pitcher blend formula:**

```
gap_ip = max(0, baseline_ip - projected_ip)
if gap_ip <= threshold_ip:
    no adjustment

# Scale waiver stats to gap innings
waiver = WAIVER_SP  # or WAIVER_RP for closers
scale = gap_ip / waiver["ip"]

effective_ip = projected_ip + gap_ip
effective_er = projected_er + waiver["er"] * scale
effective_bb = projected_bb + waiver["bb"] * scale
effective_h_allowed = projected_h_allowed + waiver["h_allowed"] * scale
effective_w = projected_w + waiver["w"] * scale
effective_k = projected_k + waiver["k"] * scale
effective_sv = projected_sv + waiver["sv"] * scale  # only for closers

# Rate stats recomputed from blended components by calculate_player_sgp
```

**Hitter blend formula:**

```
gap_ab = max(0, baseline_ab - projected_ab)
if gap_ab <= threshold_ab:
    no adjustment

waiver = WAIVER_HITTER
scale = gap_ab / waiver["ab"]

effective_ab = projected_ab + gap_ab
effective_h = projected_h + waiver["h"] * scale
effective_r = projected_r + waiver["r"] * scale
effective_hr = projected_hr + waiver["hr"] * scale
effective_rbi = projected_rbi + waiver["rbi"] * scale
effective_sb = projected_sb + waiver["sb"] * scale

# AVG recomputed as effective_h / effective_ab by calculate_player_sgp
```

### Double-counting with Monte Carlo

The backfill blending adjusts the draft board's expected value — it answers "what is the expected team-level contribution of drafting this player?" The Monte Carlo simulation models variance around that expected value — it answers "how wide is the range of outcomes?"

These serve different purposes and both are useful. However, there is a mild double-penalization: a fragile pitcher gets lower VAR (from backfill) AND more simulated bad seasons (from MC injury model). This is acceptable because:
- Backfill affects draft pick order (deterministic ranking)
- MC affects win probability estimates (probabilistic range)
- A fragile pitcher *should* rank lower AND have wider outcome variance

If post-season analysis shows the double penalty is too harsh, the MC injury probability could be reduced for players who already had significant backfill applied. But this is a future calibration, not a launch concern.

### Files changed

| File | Change |
|------|--------|
| `utils/constants.py` | New: `WAIVER_SP`, `WAIVER_RP`, `WAIVER_HITTER`, healthy baselines, thresholds |
| `sgp/replacement.py` | New function `calculate_replacement_rates` (existing function unchanged) |
| `draft/board.py` | Call `calculate_replacement_rates`, apply backfill blending to pool before SGP, pass pool-derived rates to `calculate_player_sgp` |
| `config/league.yaml.example` | Document optional baseline/threshold overrides |

### Files NOT changed

- `sgp/player_value.py` — hardcoded defaults remain as fallback; board.py overrides via kwargs
- `draft/recommender.py` — calls `calculate_replacement_levels` (unchanged return type), uses hardcoded rates for leverage (intentional, see Change 1)
- `draft/strategy.py` — consumes recommendations, unaffected
- `lineup/` — uses standings leverage and hardcoded rates, unaffected
- `lineup/waivers.py` — uses hardcoded rates for waiver SGP, intentionally unchanged
- `draft/projections.py` — Monte Carlo models injuries independently (see double-counting note)

### Testing

**Change 1 (dynamic rates):**
- `calculate_replacement_rates` returns correct ERA/WHIP/AVG from a known pool
- Empty pitcher pool returns sensible defaults (the hardcoded fallbacks)
- Pool where replacement-level pitcher has 0 IP: excluded from band average
- Pool with only closers (no SP): still computes valid rates from available pitchers
- Pool with only SP (no closers): same

**Change 2 (backfill blending):**
- SP projected at 145 IP gets blended to 178 IP with worse effective ERA/WHIP
- SP projected at 170 IP (gap = 8, below 15 IP threshold) gets NO adjustment
- Closer projected at 45 IP gets blended to 60 IP with closer replacement stats
- Middle reliever projected at 55 IP with 5 SV gets NO adjustment (not a starter or closer)
- Hitter projected at 520 AB gets blended to 600 AB with .250 replacement hitting
- Hitter projected at 570 AB (gap = 30, below 50 AB threshold) gets NO adjustment

**Integration:**
- `build_draft_board` with backfill produces lower VAR for deGrom (145 IP) than for a durable SP (185 IP) at similar rate stats
- Full test suite passes

### Rollout

Both changes default to ON. The hardcoded fallback defaults in `player_value.py` are unchanged, so any code path that doesn't go through `build_draft_board` (e.g., waivers, in-season optimizer) behaves identically. Baselines can be set to 0 in `league.yaml` to disable backfill blending.
