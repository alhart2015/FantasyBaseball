# Lineup page: YTD / ROS / Total three-way toggle

**Date:** 2026-06-02
**Status:** Approved design, pending implementation plan

## Problem

The `/lineup` page shows each player's SGP and rank on a **rest-of-season (ROS)**
basis, while the per-category stat cells show **YTD actuals** (via the `pace`
dict). There is no way to view a player on a consistent basis, and no way to see
either (a) how valuable a player has actually been so far this season, or (b)
what their projected full-season value is.

Add a three-way toggle — **YTD / ROS / Total** — that re-bases the lineup table:

- **YTD** — season-to-date actual production (what they've done so far).
- **ROS** — rest-of-season projection (forward-looking; today's SGP/rank basis).
- **Total** — full-season value = YTD actuals + ROS projection.

## Confirmed decisions

1. **Display-only.** The toggle changes only what is shown (stat cells, SGP,
   rank badge, row sort). It does **not** re-run the lineup optimizer or change
   the delta-Roto column / "moves" recommendations — those stay ROS-based and
   forward-looking. You cannot start a player for stats already banked.
2. **YTD = total season-to-date** production (the player's full year actuals),
   not "while on my roster." This is the data we already have via
   `game_log_totals`; per-roster-tenure tracking is out of scope.
3. **Whole row re-bases.** The R/HR/RBI/SB/AVG (and IP/W/K/SV/ERA/WHIP) cell
   **values** follow the selected basis.
4. **Pace tooltip, z-score, and the +/- deviation corner are UNCHANGED across
   all three bases.** Seeing that a player is exceeding pace stays useful in
   every view. Only the cell's main number, SGP, rank badge, and sort change.
5. **Default basis is ROS.** This preserves today's default SGP/rank. Note this
   means the default view's stat **cells** now show ROS-remaining projections
   instead of today's YTD actuals (YTD actuals move to the YTD tab). This change
   to the default screen is intended.
6. **Approach B (server-side re-render).** The toggle re-renders the table
   server-side via a partial endpoint, reusing the existing opponent-lineup
   partial pattern. Chosen over a client-side swap for a single formatting
   source of truth and testability; the round-trip is negligible at 25 rows.

## Data model (already present)

Each `Player` carries three independent stat lines:

- `preseason` — original full-season projection (NOT used by this feature).
- `rest_of_season` — FanGraphs remaining-games-only projection.
- `full_season_projection` — `rest_of_season + YTD actuals`, built by
  `normalize_rest_of_season_to_full_season` (projections.py:87,
  `result[col] = ros[col] + game_log_totals[col]`); rates recomputed from the
  summed counting components.

Therefore:

| Basis | Stat line | SGP source | Rank source |
|-------|-----------|------------|-------------|
| ROS   | `rest_of_season` | `rest_of_season.sgp` | `rank.rest_of_season` (exists) |
| YTD   | `full_season_projection - rest_of_season` | `calculate_player_sgp(YTD line)` | `rank.current` (exists, from game logs) |
| Total | `full_season_projection` | `full_season_projection.compute_sgp()` | `rank.total` (**new**) |

`rank.current` is already computed today by
`compute_rankings_from_game_logs`. Only the **Total** leaguewide rank is new.

## Components

### 1. Data layer (`refresh_pipeline.py`, `models/player.py`, `sgp/rankings.py`)

- **`RankInfo`**: add a `total: int | None` field with `from_dict` / `to_dict`
  support (alongside `rest_of_season`, `preseason`, `current`).
- **`_compute_rankings`**: add
  `total_ranks = compute_sgp_rankings(self.full_hitters_proj, self.full_pitchers_proj)`
  and thread it through `build_rankings_lookup` and `lookup_rank`. Guard for a
  missing full-season pool (preseason / no ROS): `total` falls back to `None`.
  `self.full_hitters_proj` / `self.full_pitchers_proj` are already populated in
  the pipeline, so no new data load.
- **`build_rankings_lookup` / `lookup_rank`**: extend to carry the `total` rank
  per player. The RANKINGS cache shape gains a `total` key per entry.

### 2. YTD SGP derivation (single-sourced with `rank.current`)

YTD SGP is computed as `calculate_player_sgp` on a YTD stat line built from
`full_season_projection - rest_of_season`:

- Hitters: counting cols (R/HR/RBI/SB, plus AB/H for the rate); AVG = YTD_H /
  YTD_AB.
- Pitchers: counting cols (W/SV/K, plus IP/ER/BB/H_allowed); ERA = 9*YTD_ER /
  YTD_IP, WHIP = (YTD_BB + YTD_H_allowed) / YTD_IP.
- Clamp each counting component at `>= 0`; guard zero AB/IP (SGP contribution
  from that line is 0 / cell renders "--").

Because `full_season = ROS + game_log_totals` by construction, this YTD line
equals the game-log YTD that produced `rank.current`, so the displayed YTD SGP
and the YTD rank are consistent **without** refactoring the shared RANKINGS
cache (smaller blast radius than threading SGP values through the lookup).

### 3. Formatter (`web/season_data.py::format_lineup_for_display`)

Signature gains `basis: str = "ros"` (one of `ros` / `ytd` / `total`; unknown
values fall back to `ros`).

For each player the formatter selects, for the chosen basis: the per-category
**cell display value**, the **SGP**, and the **rank**. It then:

- Injects a `display` value into each per-category `pace` sub-dict (the basis
  value: ROS-remaining / YTD-actual / full-season). The `pace` dict is otherwise
  emitted unchanged, so `actual` (tooltip), `z_score`, `color_class`, and
  `rest_of_season_deviation_sgp` (the +/- corner) are identical across bases.
- Sets `entry["sgp"]` to the basis SGP.
- Sets `entry["rank_display"]` to the basis rank int (ROS -> `rank.rest_of_season`,
  YTD -> `rank.current`, Total -> `rank.total`). The badge renders
  `rank_display`; the full `rank` object still flows through for the tooltip.
- Sorts rows by the basis SGP, preserving today's slot-group ordering
  (hitters: slot order then -sgp; pitchers: is_bench then -sgp).

### 4. Templates

- `_lineup_hitters_tbody.html` / `_lineup_pitchers_tbody.html`: the cell's main
  number changes from `st.actual` to `st.display`. The tooltip rows (`Actual`,
  `Expected pace`, `Z-score`, `Pre-season proj`, `ROS proj`) and the
  `stat-ros-up/down` deviation classes are unchanged (still read `st.actual` /
  `st.rest_of_season_deviation_sgp`).
- `rank_badge` (macros.html): render `rank_display` as the badge number; the
  tooltip continues to show ROS / Preseason / Current, plus the new Total row.
- `lineup.html`: add the three-way segmented toggle in the table header,
  reflecting the active basis.

### 5. Route + partial endpoint (`web/season_routes.py`)

- `/lineup`: read `basis = request.args.get("basis", "ros")`, pass to
  `format_lineup_for_display`, render with the toggle in the active state.
- Partial endpoint (e.g. `/lineup/tbodies?basis=<b>`): returns the two
  re-rendered `<tbody>` fragments for the requested basis, mirroring the
  existing `/api/opponent/<key>/lineup` server-rendered-partial pattern.

### 6. Frontend JS (`web/static/season.js` or inline on lineup.html)

On toggle change: `fetch` the partial for the new basis, swap the two tbody
`innerHTML`s, update the active toggle state, and update `?basis=` in the URL
(via `history.replaceState`) so refresh/share is stable.

## Edge cases

- **Injured / no-games players** (e.g. Hader): `full == ros` -> YTD line is 0 ->
  YTD SGP 0, cells "--". `rank.total` may be `None`; the badge already renders
  "--" for missing ranks.
- **Unmatched prospects / recent call-ups** (no `mlbam_id` match in game logs):
  `full == ros`, so derived YTD collapses to 0 even if they've played. Their
  `rank.current` may still place them via the game-log path, so SGP and rank can
  momentarily disagree for these edge players. **Documented limitation**, not
  special-cased.
- **Zero AB/IP in a basis**: guard division; cell renders "--", SGP contribution
  from that line is 0.
- **Stale ROS snapshot**: YTD derivation inherits the existing
  `ROS_SNAPSHOT_STALE_DAYS` assumption (full = YTD + ROS can double-count if the
  snapshot is stale). The pipeline already warns; no new handling here.
- **Unknown `basis` query value**: defaults to `ros`.

## Testing

- **Formatter unit tests** on all three bases with a fixture roster containing a
  played player (YTD > 0) and an injured player (YTD = 0): assert per-basis SGP,
  rank, and cell display values; assert the `pace` payload (`actual`, `z_score`,
  `rest_of_season_deviation_sgp`) is **identical across bases**.
- **Pipeline test**: `rank.total` populates when a full-season pool is present;
  is `None` when it is absent.
- **Route tests**: `/lineup?basis=ytd` renders; the partial endpoint returns the
  two tbodies for a given basis; unknown basis falls back to ROS.
- **Existing lineup tests**: the default ROS view's stat cells change from YTD
  actuals to ROS-remaining. Any existing assertion that the default cell shows
  the actual must be **updated deliberately** (this is an intended behavior
  change, per decision 5) — not silently loosened. Call out each such test in
  the PR.

## Out of scope

- Changing the optimizer, delta-Roto, or "moves" basis (stays ROS).
- "While on my roster" YTD attribution.
- Applying the toggle to the roster-audit or other pages (lineup page only).
