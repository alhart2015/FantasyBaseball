# Lineup overall-pace coloring: SGP-deviation, leaguewide percentile

**Date:** 2026-07-21
**Status:** Approved design (brainstormed), ready for planning.

## Problem

The lineup page colors each player's Slot cell green/red by an **equal-weight
mean of per-category z-scores** (`compute_overall_pace` in
`src/fantasy_baseball/analysis/pace.py`). Two flaws combine to make the color
misleading:

1. **Averaging washes out broad slumps.** A player down moderately in every
   category averages to a moderate z that never crosses the +/-1.0 threshold, so
   he reads neutral despite clearly underperforming. Real example (2026-07-21
   data): Julio Rodriguez is below projection in all five categories
   (equal-weight mean-z -0.75) yet shows **neutral**.

2. **A fluke in a low-stakes category swings the color.** The z-score measures
   statistical surprise, not fantasy value, so a big deviation in a category
   where the player has little at stake dominates the average. Real example:
   Austin Riley has 6 SB vs 1.6 expected (**z +2.52**) on a category worth ~0.4
   SGP to him; that single number drags his average from ~-1.1 up to **-0.50
   (neutral)**, masking a season in which his overall leaguewide rank fell from
   ~42 (preseason) to ~134.

The fix is to change **both** the metric (measure fantasy value delivered vs
expected, not average per-category surprise) and the scale (rank players against
the league rather than thresholding an absolute z).

Investigation confirming the diagnosis and the chosen metric's behavior is
summarized inline above; the numbers were pulled from the live Upstash cache on
2026-07-21.

## Goals

- The overall Slot-cell color reflects **how much roto value (SGP) a player has
  delivered versus what his preseason projection expected by now**, dominated by
  the categories that actually matter to that player.
- A meaningless over/under-performance (Riley's +3 SB) contributes negligibly; a
  broad-but-moderate slump (Julio) accumulates into a clear signal.
- The color scale is a **leaguewide percentile** with fixed tercile/sextile
  cutpoints, so a fixed fraction of players is colored regardless of calendar
  month (the metric grows over the season, but the percentile buckets do not).
- Hitters are ranked among hitters and pitchers among pitchers.
- The tooltip shows the literal, interpretable number ("delivered X of Y
  expected SGP").

## Non-goals

- **No change to per-category cell coloring.** The individual stat cells keep
  their per-category z-score coloring (`_z_to_color`); only the overall Slot cell
  changes.
- **No change to `rest_of_season_deviation_sgp`.** That per-category field
  measures preseason -> ROS *projection revision* (how the market re-rated the
  player) and is a distinct concept from YTD-performance deviation. It stays.
- **Display only.** This metric must NOT feed roster decisions, trade/waiver
  evaluation, Monte Carlo, or projected standings, consistent with the existing
  `pace.py` module contract.
- **No rank-computation change** and **no new page/route.** Same Slot cell +
  tooltip surface.

## Chosen approach

### The metric: SGP delivered vs expected

For each player, sum a per-category value delta across the five roto categories:

```
sgp_dev = sum over categories of (actual_sgp_c - expected_sgp_c)
```

where `expected` is the **preseason** projection prorated to the player's actual
playing time so far (the same `expected = proj * actual_opp / proj_opp`
proration already used for the per-category z-scores).

- **Counting** (R/HR/RBI/SB for hitters, W/K/SV for pitchers):
  `actual_sgp_c = actual / denom`, `expected_sgp_c = expected / denom`, so the
  delta is `(actual - expected) / denom`.
- **AVG:** value-above-replacement over the **actual** at-bats accrued:
  - `actual_sgp = (actual_avg - REPLACEMENT_AVG) * actual_AB / (denom_AVG * team_AB)`
  - `expected_sgp = (proj_avg - REPLACEMENT_AVG) * actual_AB / (denom_AVG * team_AB)`
  - delta = `(actual_avg - proj_avg) * actual_AB / (denom_AVG * team_AB)`
- **ERA/WHIP** (inverse; lower is better), over the **actual** innings:
  - divisor 9 for ERA, 1 for WHIP.
  - `actual_sgp = (REPLACEMENT_rate - actual_rate) * actual_IP / divisor / (denom * team_IP / divisor)`
  - `expected_sgp = (REPLACEMENT_rate - proj_rate) * actual_IP / divisor / (denom * team_IP / divisor)`
  - delta = `(proj_rate - actual_rate) * actual_IP / divisor / (denom * team_IP / divisor)`
    (positive when the pitcher's actual rate is below his projection = good).

Using **actual** volume (AB/IP) for both the actual and expected rate SGP makes
the metric rate-fair: it judges rate performance over the playing time actually
accrued, and does not penalize/credit a player for missing time (consistent with
the per-category z proration). The replacement baseline cancels in the delta but
is retained in the displayed `actual_sgp`/`expected_sgp` so the tooltip reads as
value-above-replacement.

**Denominators / team volumes:** league SGP denominators
(`get_sgp_denominators(config.sgp_overrides)` -- already threaded into
`compute_player_pace` as `sgp_denoms`) and the constant `DEFAULT_TEAM_AB` /
`DEFAULT_TEAM_IP` (consistent across all players; the absolute team volume is
irrelevant to relative ordering).

**Sample-size gates:** identical to the existing per-category gates -- counting
colored only when PA >= 10 / IP >= 5, rates only when PA >= 30 / IP >= 10. A
category below its gate contributes 0 to `sgp_dev` (and to `actual_sgp` /
`expected_sgp`). A player with no games above the counting gate, or with no
preseason projection, has an **undefined** `sgp_dev` (None).

### The coloring: leaguewide percentile, hitters vs pitchers separate

1. **Reference population:** all **rostered** players in the league (the user's
   roster plus every opponent roster) that have a defined `sgp_dev` (preseason
   projection present and at least the counting gate met). Restricting to
   rostered players keeps the pool fantasy-relevant and avoids part-time MLB
   players diluting the middle third.
2. **Two pools:** hitters ranked among hitters, pitchers among pitchers.
3. **Cutpoints:** for each pool, compute the four quantile values at 1/6, 1/3,
   2/3, 5/6 using a deterministic nearest-rank rule
   (`sorted[round(q * (n - 1))]`).
4. **Buckets** (dev `d` vs the pool's cutpoints `q16 <= q33 <= q66 <= q83`):

   | Condition | Bucket | CSS class |
   |---|---|---|
   | `d >= q83` | bright green | `stat-hot-2` |
   | `q66 <= d < q83` | light green | `stat-hot-1` |
   | `q33 <= d < q66` | neutral | `stat-neutral` |
   | `q16 <= d < q33` | light red | `stat-cold-1` |
   | `d < q16` | bright red | `stat-cold-2` |

5. A player with undefined `sgp_dev`, or when cutpoints are unavailable, or when
   the pool is too small to bucket meaningfully (fewer than
   `MIN_POOL_SIZE = 6` qualified players), renders **neutral / uncolored**.

The metric grows in magnitude as the season progresses, but the percentile
buckets are re-fit against the current distribution on every refresh, so the
fraction of players colored stays fixed (bottom 1/6 bright red, next 1/6 light
red, middle 1/3 neutral, and mirrored greens).

### Architecture

Three pure units plus wiring. The metric is computed once, leaguewide, in the
refresh pipeline; the color is applied at display time as a **pure cache lookup**
so the user roster and opponent lineups use an identical, consistent metric basis
(never a display-time recompute).

1. **`compute_sgp_deviation(actual_stats, projected_stats, player_type, denoms)`
   -> `{sgp_dev, actual_sgp, expected_sgp}`** (new, in `pace.py`). Pure function
   implementing the metric above (including gates). Returns `sgp_dev=None` when
   undefined. It takes the **raw** actual/projected stat dicts (the same
   lowercase-key shapes `compute_player_pace` receives: `actual_stats` with
   `pa`/`ab`/`ip`/counting/`h`/`er`/etc., `projected_stats` with the projected
   counting + rate keys and `pa`/`ip`) and computes the prorated `expected`
   internally -- it does NOT consume a pre-built `pace` dict. The proration
   (`expected = proj * actual_opp / proj_opp`) is the same rule
   `compute_player_pace` uses; factor it into a small shared helper so the two
   functions cannot drift, rather than duplicating it. **This function is called
   only from the pipeline pass in unit 2, never from the display path.**

2. **Pipeline step (new): compute leaguewide deviations + cutpoints.** Placed
   after opponent rosters are hydrated and the game logs + projections are loaded
   (alongside / just after `_compute_rankings`, which already has these inputs).
   Reusing the game logs (`hitter_logs` / `pitcher_logs`) and the **preseason**
   projections (the same `preseason_lookup` / preseason frames used by
   `_compute_rankings` and `attach_pace_to_roster`), call `compute_sgp_deviation`
   for **every rostered player** (user + every opponent). Cache two things under a
   new `CacheKey.PACE_DEVIATIONS` payload:
   - `deviations`: `{ "<normalized_name>::<player_type>": {sgp_dev, actual_sgp,
     expected_sgp} }` for all rostered players (keyed by the same
     normalized-name form the game-log and preseason lookups use, so the display
     path resolves by `(normalize_name(player.name), player_type)`). Both the
     user and opponent display paths look up a precomputed value rather than
     recomputing -- guaranteeing one consistent preseason basis.
   - `cutpoints`: `{ "hitter": [q16, q33, q66, q83], "pitcher": [...] }`.

   The reference population for cutpoints is the rostered players with a defined
   `sgp_dev` (undefined ones are excluded from the pools and rendered neutral).

3. **`compute_overall_pace(sgp_summary, cutpoints_for_type)` ->
   `{color_class, sgp_dev, actual_sgp, expected_sgp}`** (reshaped in `pace.py`).
   Pure bucketing of one player's cached `sgp_dev` against its pool cutpoints.
   Neutral when `sgp_dev` is None, cutpoints missing, or the pool was too small.

4. **Wiring (`season_data.py`):** the three existing `compute_overall_pace` call
   sites (user lineup; opponent-lineup matched and unmatched) read
   `CacheKey.PACE_DEVIATIONS`, look up the player's summary by
   `(normalize_name(player.name), player_type)`, and bucket against the cutpoints
   for the player's type. The opponent overall color no longer derives from a
   ROS-basis pace dict; it uses the same preseason-basis leaguewide map, so every
   player's **overall** color -- user and opponent alike -- is scored on one
   consistent basis and the percentile is comparable across the whole pool.

**Scope of the basis change:** only the **overall** Slot color is unified onto
the leaguewide preseason basis. The per-category `pace` cell colors are unchanged
(non-goal) and keep computing their own per-cell z-scores exactly as today -- the
user roster's cells against preseason, the opponent cells against ROS. This means
an opponent row's per-category cells (ROS basis) and its overall color (preseason
basis) use different bases; that pre-existing per-cell inconsistency is
deliberately left out of scope here and is not "fixed" by this change.

### Display

Both lineup tbody templates (`_lineup_hitters_tbody.html`,
`_lineup_pitchers_tbody.html`) replace the "Avg z-score" tooltip row with the
SGP-pace line, e.g.:

```
<Player> -- Overall Pace
SGP pace   -2.4   (delivered 5.1 of 7.5 expected)
```

The tooltip is shown only when `sgp_dev` is not None; the Slot cell uses
`overall_pace.color_class` as today.

## Edge cases / failure modes

- **No preseason projection** (rookies, late call-ups): `sgp_dev = None` ->
  excluded from cutpoints, rendered neutral.
- **Below sample gate** (early-season / bench): categories under the gate
  contribute 0; a player entirely under the counting gate has `sgp_dev = None`
  -> neutral.
- **Cold cache / missing `PACE_DEVIATIONS`** (first run before a refresh
  populates it): all players render neutral (no color), no error.
- **Small pool** (`n < MIN_POOL_SIZE = 6` qualified in a pool): that pool
  renders neutral (percentiles not meaningful with too few players).
- **Two-way players** (e.g. Ohtani): the hitter and pitcher rows are separate
  `(name, type)` entries, each bucketed in its own pool -- reuses the existing
  `name::player_type` disambiguation.
- **Inverse stats (ERA/WHIP):** delta sign must be positive when the actual rate
  is *below* projection; unit tests pin the sign.
- **Zero/missing denominator or zero projected opportunity:** that category is
  skipped (guarded), as in the current code.
- **Ties at cutpoints:** deterministic via `<` / `>=` boundaries as tabulated.
- **Name-key drift** between game logs, preseason projections, and rosters: a
  player who fails to match on `(normalized_name, player_type)` gets
  `sgp_dev = None` -> neutral (fail-soft, never raises). The leaguewide pass logs
  a debug count of unmatched rostered players.

## Testing expectations

- **Unit -- `compute_sgp_deviation`:** hitter counting-only, hitter with AVG,
  pitcher counting-only, pitcher with ERA/WHIP (assert inverse sign), gating
  below thresholds zeroes a category, no preseason projection returns
  `sgp_dev=None`, replacement cancels so `actual_sgp - expected_sgp == sgp_dev`.
- **Unit -- cutpoint computation:** known list -> expected quantile values;
  pool below `MIN_POOL_SIZE` handled.
- **Unit -- `compute_overall_pace` bucketing:** a dev in each of the five bands
  maps to the right class; boundary/tie behavior; `sgp_dev=None` -> neutral;
  missing cutpoints -> neutral; small pool -> neutral.
- **Integration:** the refresh pipeline computes and caches `PACE_DEVIATIONS`
  (deviations + cutpoints); `season_data` wires them into the user and opponent
  lineups; `tests/test_web/test_opponent_lineup.py` updated for the new
  `overall_pace` shape.
- **Rewrite `tests/test_analysis/test_overall_pace.py`:** the function contract
  changes by design (metric replacement, not a regression). Old avg-z assertions
  are replaced with dev-bucketing assertions.
- **ASCII-only**, Windows-safe (no non-ASCII in source, logs, or templates that
  hit `print`).
- Final gate: `pytest`, `ruff check .`, `ruff format --check .`, `vulture`, and
  `mypy` for any touched file listed under `[tool.mypy].files`.

## Phasing

The existing avg-z overall coloring stays live and unchanged through Phases 1-2;
the swap to the new metric happens atomically in Phase 3. The app is in a
working, shippable state after every phase.

1. **Metric (pure).** Add the shared proration helper + `compute_sgp_deviation`
   + unit tests. No wiring yet; old coloring untouched.
2. **Leaguewide pass + cache.** New `CacheKey.PACE_DEVIATIONS`, the pipeline step
   computing deviations + cutpoints, cutpoint helper + tests. The payload is
   written but not yet consumed; old coloring still live.
3. **Coloring + wiring.** Reshape `compute_overall_pace`; wire the three
   `season_data` call sites to the cached map; rewrite/adjust the affected tests.
   This is the atomic behavior swap.
4. **Display.** Update both tbody templates' tooltip; verify end-to-end (drive
   the lineup page via the `verify` / `run` skill and confirm the Slot color and
   tooltip render for a colored and a neutral player).
