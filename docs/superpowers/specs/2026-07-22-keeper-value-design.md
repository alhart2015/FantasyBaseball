# Keeper-Asset-Value Metric

**Date:** 2026-07-22
**Status:** Approved

## Problem

The toolkit scores players for a single season (VAR / standings-point value) but
has no way to quantify a player's value *as a multi-year keeper*. In this league
you protect 3 players every year, and the manager's live question -- "is Caminero
a better keeper than a trade for Kyle Tucker, and how much of that is his youth?"
-- has no number behind it. We want a reusable, general keeper-asset-value metric
that any player can be scored on, which then feeds the concrete decisions
(which 3 to protect, whether to chase a trade target) as filters on top.

## League mechanic (the assumption everything rests on)

**Flat, unlimited keep-3.** Every team keeps exactly 3 players each year; the 30
keepers consume draft rounds 1-3 and the live draft runs rounds 4-23. There is
**no per-player round penalty, no escalating cost, and no year cap** -- keeping
any player costs the same slot every year regardless of the player's age.

Consequence: the *cost* of a keeper slot is uniform, so it is a roughly constant
subtrahend across all players and **drops out of the ranking**. Therefore the
ranking of keeper value reduces to a player's discounted multi-year roto value.
The "youth premium" is **not a separate term** -- it emerges because a young
player holds his value across the horizon while an older player's value decays.
The decline curve does all the work.

## Core formula

```
KeeperValue(p) = sum over k in {0 .. horizon-1}  [ discount^k * V(p, 2026 + k) ]
```

- `V(p, y)` = player p's roto value in year y, in the toolkit's existing
  **VAR / standings-point currency** (`sgp/var.py::calculate_var`, netting a
  player's SGP against position-aware empirical replacement floors). This is
  already position- and category-aware, so saves scarcity and steals scarcity
  are priced correctly.
- `discount` (default 0.80) and `horizon` (default 3, = ZiPS availability) are
  **tunable knobs**. The report sweeps `discount` so the manager watches the
  ranking slide from win-now to dynasty.

## Computing V(p, y): anchor + trajectory (approach B)

We do **not** feed ZiPS out-year lines through the pipeline directly (that would
lean entirely on ZiPS absolute levels and discard the 5-system consensus for the
current year). Instead:

- **2026 anchor line:** the player's existing 5-system blended stat line
  (`data/projections.py::blend_projections` over `data/projections/2026/*.csv`).
  This is the most accurate current estimate and the value level we trust.
- **Out-year lines:** the anchor line scaled **per stat** by ZiPS's own
  year-over-year trajectory ratio:

  ```
  ratio(stat, y) = ZiPS_y(stat) / ZiPS_2026(stat)
  scaled_line_y(stat) = anchor_line(stat) * ratio(stat, y)
  ```

  ZiPS supplies *only the aging shape*; the level stays anchored to consensus.
  Playing-time decline is captured because PA/AB/IP are themselves scaled stats.

- **Which fields get scaled:** exactly the ones `calculate_player_sgp` consumes,
  each by its own ZiPS ratio -- hitters `r/hr/rbi/sb` (counting), `ab` (volume),
  `avg` (rate); pitchers `w/k/sv` (counting), `ip` (volume), `era/whip` (rate).
  **Rate stats ARE scaled directly** by their ZiPS rate-ratio
  (`ZiPS_y(rate) / ZiPS_2026(rate)`), because that ratio *is* ZiPS's projected
  aging of the rate and `calculate_player_sgp` reads `avg`/`era`/`whip` as rates
  (it does not reconstruct them from `h`/`er`/`bb`). Fields the SGP function does
  not read (e.g. `h`) are ignored -- no need to scale them.

- Each year's scaled line then runs through the **full-season SGP -> VAR path**,
  the same one the draft board uses: `sgp/player_value.py::calculate_player_sgp(
  series, denoms=denoms, team_ab=team_ab, team_ip=team_ip)` to get `total_sgp`,
  then `sgp/var.py::calculate_var(series_with_total_sgp, floors, role_ip=...)` to
  net against the positional floor -- exactly the `board.py:99` sequence.

  **Do not** reuse `analysis/draft_value.py::_sgp` / `_value_of_line`: those are
  the *in-season, to-date* deltaRoto seam that scales counting stats by a partial
  `fraction` via `ScaleInputs`. Keeper value is a **full-season** valuation
  (`fraction == 1.0`), so the `fraction`/`ScaleInputs` machinery is not used at
  all; call `calculate_player_sgp` directly like the board does.

- **Full-season inputs come from the shared league context** (see Component 4,
  "shared context"): `denoms` from the league's SGP denominators, and
  `team_ab` / `team_ip` at their full-season league values (the board's
  defaults / `config/league.yaml`), NOT any to-date-scaled values.

- **Held constant across the horizon** (documented simplifying assumptions):
  2026 position eligibility, 2026 position-aware replacement floors
  (`position_aware_replacement_levels`), 2026 SGP denominators, and full-season
  `team_ab` / `team_ip`. We are pricing the aging curve, not future league drift.

## Components

### 1. Out-year projection loader

ZiPS multi-year projections are **manual CSV exports** (FanGraphs auto-fetch is
Cloudflare-blocked; same constraint as the ROS restore procedure). They reuse the
existing `{system}-{type}.csv` convention under a per-year folder:

```
data/projections/2027/zips-hitters.csv
data/projections/2027/zips-pitchers.csv
data/projections/2028/zips-hitters.csv
data/projections/2028/zips-pitchers.csv
```

The loader reuses `data/fangraphs.py::load_projection_set(dir, "zips")` and, on a
missing file, raises with the **exact FanGraphs ZiPS URL** to download, matching
the actionable-error pattern in `data/projections.py::validate_projections_dir`.

### 2. Trajectory engine

Per player, per out-year, per stat: compute `ratio(stat, y)` with two numeric
guardrails against the small-denominator blowup:

- **Denominator floor:** if `ZiPS_2026(stat)` is below `eps` (default `1e-6`), or
  the player's ZiPS-2026 line is below `min_pt_for_trajectory` (concrete defaults:
  **100 AB** for hitters -- the projection CSVs carry `ab`, not `pa` -- and
  **20 IP** for pitchers; both tunable), the ratio is
  unstable -> **fall back to approach A** for that player (feed the ZiPS out-year
  line directly through the pipeline), flag `fallback_A`.
- **Ratio clamp:** clamp each ratio to a configurable band (default `[0.25, 2.5]`)
  so a single small denominator can't explode a counting stat.

Guard the numeric-default trap per repo convention: `V == 0.0` is a real value,
never treated as falsy/missing in any sort or share computation.

### 3. `keeper_value()` metric function

New module `src/fantasy_baseball/analysis/keeper_value.py`. A pure, testable
function that returns a structured result per player:

```python
@dataclass(frozen=True)
class KeeperValueResult:
    player_id: str
    name: str
    per_year_var: dict[int, float]     # {2026:.., 2027:.., 2028:..}
    total: float                       # discounted sum (the ranking key)
    used_fallback: bool                # anchor missing -> approach A
    flags: list[str]                   # e.g. ["no_zips_2028", "fallback_A"]
    pct_from_out_years: float | None   # display-only; None when total <= eps_share
    pct_from_saves: float | None       # display-only; None when total <= eps_share
```

`keeper_value(...)` takes the anchor line, the per-year ZiPS lines, the shared
league context (replacement floors, SGP denoms, `team_ab`/`team_ip`, positions,
player_type), and the `discount` / `horizon` / `ratio_band` /
`min_pt_for_trajectory` knobs.

**Transparency-column definitions (display-only; never part of the ranking):**

- `pct_from_out_years` = `discounted(V_2027 + V_2028) / total`. Because VAR can be
  negative (sub-replacement), `total` can be near-zero or negative, which makes
  this ratio explode or flip sign. **Guard:** when `total <= eps_share` (default
  `eps_share = 1.0` standings-point), report `None` (rendered `N/A`), never a
  number. This is the exact numeric-default trap the repo warns about.
- `pct_from_saves` = the **SV category's share of the player's 2026-anchor SGP**,
  not of VAR. SGP is category-additive (a sum of per-category standings gains), so
  a per-category share is well defined; VAR (a single scalar netted against a
  positional floor) is not. Computed on the 2026 anchor line only; `0.0` for any
  hitter (no SV component). Guard is on **its own denominator**, the 2026-anchor
  total SGP (which is also sign-unstable for weak players): when
  `abs(sgp_2026) <= eps_share`, report `None`.

### 4. Report script

New `scripts/keeper_value.py`:

1. Build the **shared context** once via the existing board seam:
   `draft/board.py::build_board_from_frames(hitters, pitchers, positions,
   roster_slots, num_teams, sgp_overrides)` returns `(board_df, ScaleInputs)`.
   `ScaleInputs` already carries the entire context the metric needs -- `denoms`,
   `repl_rates`, `replacement_levels`, `team_ab`, `team_ip` -- and `board_df`
   carries per-player `player_id`, `name_normalized`, `positions`, `player_type`,
   the 2026 blended stat line, and `var`. Because the metric scores its 2026 year
   on this same `ScaleInputs`, the 2026 column reconciles with the board exactly.
   Load `hitters`/`pitchers`/`positions` from SQLite (`get_blended_projections`,
   `get_positions`), matching `build_draft_board`. This script does **no** network
   I/O and does **not** apply keepers/ADP/draft state -- it scores the raw
   projection universe.
2. Load ZiPS 2026/2027/2028 lines via the out-year loader.
3. Score every board player with `keeper_value(...)`.
4. **Sweep `discount`** over a configurable list (default `[0.60, 0.70, 0.80,
   0.90]`); for each, rank by `total`.
5. Emit an ASCII table: each player's `total` and rank at each discount, the
   per-year VAR, and the two transparency columns (`% from out-years`,
   `% from saves`, rendered `N/A` when `None`). **Highlight the manager's
   candidate set** (Soto, Julio Rodriguez, Junior Caminero, CJ Abrams, Mason
   Miller, Kyle Tucker) -- match by **normalized name** (`utils/name_utils.
   normalize_name`) / `name::player_type` id, never bare name, tie-breaking on VAR
   per `draft/keepers.py::find_keeper_match`, so a namesake collision can't
   highlight the wrong player. ASCII-only output;
   `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the entry point
   in case a player name carries an accent.

## Data flow

```
manual ZiPS export
  -> data/projections/{2027,2028}/zips-{hitters,pitchers}.csv
  -> out-year loader (load_projection_set)
  -> trajectory ratios vs ZiPS 2026 (floored + clamped)
  -> scale the 2026 blended anchor line per stat
  -> calculate_player_sgp(...) + calculate_var(...) per year  -> V(p, y)
  -> discount + sum                            -> KeeperValueResult.total
  -> discount sweep + ranked ASCII report
```

## Edge cases / numeric guards

- **Player missing from 2026 blend** (prospect not in consensus): no anchor ->
  approach A for all years using ZiPS lines directly; flag `fallback_A`.
- **Player missing a ZiPS out-year line** (aging vet ZiPS stops projecting): that
  year's `V = 0.0` with a visible flag (`no_zips_{y}`). Conservative by design --
  correctly penalizes "we don't even know if he'll be rosterable." No silent zeros
  masquerading as real values.
- **Tiny ZiPS-2026 denominator:** handled by the denominator floor + ratio clamp
  (see Trajectory engine).
- **Rate-stat handling:** scale `avg`/`era`/`whip` by their ZiPS rate-ratios
  (same clamp), alongside the counting/volume fields -- `calculate_player_sgp`
  reads them as rates, so no component reconstruction is needed.
- **Saves and other role/opportunity stats in out-years:** taken straight from
  ZiPS (per decision -- the manager filters out players at risk of losing a closer
  role *before* they are keeper candidates, so role-loss is handled upstream by
  human judgment, not the model). The `% from saves` column keeps that dependence
  visible.

## Testing (the guardrails)

- **Youth-premium test:** two fixtures with **identical 2026 VAR** but different
  ZiPS decline curves -> the younger (flatter curve) ranks higher, and the gap
  **widens as the discount shallows**. Directly pins the behavior the metric
  exists to produce.
- **Currency-parity test:** `horizon=1` reproduces the existing single-year VAR
  exactly -- proves the pipeline is reused faithfully, not reimplemented.
- **Blowup-guard test:** a near-zero ZiPS-2026 denominator stays within the ratio
  band and does not explode the scaled counting stat.
- **Fallback tests:** missing anchor -> approach A + `fallback_A` flag; missing
  out-year line -> `V=0.0` + `no_zips_{y}` flag (assert the flag, not just the
  number).
- **Transparency-column tests:** a hitter has `pct_from_saves == 0.0`; a pure
  closer's is high. A **sub-replacement player** (negative discounted `total`, and
  a near-zero 2026 SGP) returns `pct_from_out_years is None` / `pct_from_saves is
  None`, rendered `N/A` -- never a blown-up or sign-flipped ratio.
- **Numeric-default test:** a player whose year `V` is exactly `0.0` is not sorted
  or shared as if missing.

## Non-goals (v1)

Deferred, explicitly out of scope for this build:

- **Risk modeling / downside bands** (injury, pitcher attrition, closer-role loss).
  Separate future conversation. v1 is a point estimate off the ZiPS mean.
- **Trade give-up valuation** (net value of acquiring Tucker minus what you ship).
  The metric scores Tucker's asset value; comparing both sides is a later consumer.
- **Dashboard tab.** Ships as a library primitive + report script first; the tab
  comes once the number is trusted.
- **Out-year league-context drift** (replacement floors, SGP denominators, roster
  composition changing across years). Held at 2026 values.

## What doesn't change

- `sgp/var.py`, `sgp/replacement.py`, `data/projections.py::blend_projections`,
  and the existing draft board build path are **reused, not modified**. The new
  module composes them; it does not fork the scoring math.
