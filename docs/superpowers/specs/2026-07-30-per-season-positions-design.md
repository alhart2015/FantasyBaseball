# Per-season position eligibility for the scarcity measurement

- **Addresses:** PR #279 review findings 1 (historical map contamination), 5 (UTIL
  floor), 6 (unregenerable prose). Branch `fix/273-per-season-positions`, PRs into
  `feat/273-league-keeper-board`.
- **Status:** approved design, ready for planning.

## Problem

`run_scarcity` measures the positional credits (`NATIVE_CREDITS`) over the four
complete seasons 2022-2025, but every season is scored against the **2026** Yahoo
eligibility map (`pricing_table()` -> `load_positions()`). Yahoo eligibility is a
current-season snapshot with no history, so:

1. A player active in an older season but gone by 2026 (retired, demoted) is absent
   from the map -> falls to `FALLBACK_POS["hitter"] == ["UTIL"]`, which fills or sets
   **no** dedicated slot. He vanishes from every per-position measurement. This
   shrinks the older seasons' position pools (measured: 23 catcher-eligible hitters
   in 2022 vs 42 in 2025), and a floor defined as "the 11th-best of N eligible" is a
   different, shallower measurement when N is artificially small.
2. Even players present in 2026 are scored with their **2026** eligibility, not the
   season's own -- eligibility drifts year to year.

Both push the same way: `marginal_starter_floors`' own precondition ("fill every
starting slot with the best eligible player") is violated for older seasons. The
shipped `NATIVE_CREDITS["C"] = 1.176` averages coverage-degraded seasons; the
best-covered season (2025) measures 0.32. The overcredit (~0.85 SGP) decides the top
of the live board (four catchers in the top seven) and invalidates the PR's "catcher
premium was roughly right" claim, which used the same contaminated measurement on
both sides.

## Goals

- Measure each historical season's positional floors against **that season's own**
  eligibility, derived from MLB Stats fielding data.
- Regenerate `NATIVE_CREDITS`; `--scarcity` reproduces the shipped literal to delta
  0.00 (the subsystem's self-consistency check).
- Fix the UTIL floor (finding 5) and remove the unregenerable prose (finding 6), so
  the regenerated credits are fully correct and no number is asserted that a flag
  doesn't reproduce.
- Make contamination un-hideable: `--scarcity` prints per-season map coverage and a
  one-time agreement check of the derived rule against the real 2026 Yahoo map.
- Update PR #279's scarcity table and "catcher premium" claim to the corrected
  numbers.

## Non-goals

- **Findings 2, 3, 4, 7, 8, 9, 10** -- out of scope for this change (the broader
  review-response was cancelled). In particular finding 4's config-drift test is NOT
  included.
- **Changing the live board's position source.** `build` (the shipped 2026 board and
  `--league`/`--roster`) keeps the Yahoo map: it is the real league eligibility a
  keeper decision is actually made under. Only the historical *measurement* changes.
- Splitting outfield into LF/CF/RF slots, or P into SP/RP. The league rosters a
  single `OF` and a single `P`.
- Carrying prior-season eligibility forward (Yahoo does; the measurement wants each
  season's own contemporaneous eligibility).

## Chosen approach

### Eligibility rule
A player is eligible at a position in a season iff he appears in **>= 10 games** at
that position that season (using the fielding `games` count, not `gamesStarted`).
Outfield is combined: `OF` eligibility is **>= 10 games across LF + CF + RF summed**,
because the league rosters one `OF` slot. Base slots only are stored; the existing
`can_fill_slot` derives the `IF`/`UTIL`/`OF` flex slots from them.

### Data source and fetch boundary
The MLB Stats fielding leaderboard (`stats=season&group=fielding&playerPool=all`,
paginated), which returns one row per (player, position) with `games`, keyed by
**MLBAM id** -- matching the keeper board's index, so no name join is needed for the
measurement. `keepers/mlb_stats.py`'s `_fetch_mlb_season(group, year)` already
paginates and json-normalizes; the fetch (with the `fetch_or_cache` layer, like the
hitting/pitching pulls) lives there and returns a raw frame. After json_normalize
the columns the derivation needs are `player.id` (the MLBAM id, an int),
`position.abbreviation` (e.g. `"C"`, `"LF"`, `"P"`), and `stat.games` (an int). The
fetch that Phase 0 caches and Phase 1 exposes is a thin wrapper returning that frame
for a season.

### Derivation module
`keepers/appearances.py` (pure, I/O-free like the rest of the normalization layer):

    season_eligibility(fielding: pd.DataFrame) -> dict[int, set[str]]

Reads `player.id`, `position.abbreviation`, `stat.games` from the raw frame. Group by
(`player.id`, mapped base slot), sum `stat.games`, keep base slots whose summed games
>= 10. Mapping: `C->C`, `1B/2B/3B/SS->same`, `LF/CF/RF->OF`, `P->P`; `DH` and any
other non-fielding token contribute to no base slot. Returns `int` MLBAM id -> set of
base slots. A player with no >= 10-game base slot is simply absent from the dict (the
caller supplies the pool fallback). The returned key dtype is `int` to match the
board's MLBAM index; a str/int mismatch would silently make every join miss, so the
derivation coerces `player.id` to int and the wiring asserts the index dtype agrees.

### Wiring into `run_scarcity`
Replace the per-season `eligible` construction. For season `year`, load
`season_eligibility(fetch_fielding(year))` and, for each board row keyed by its MLBAM
index:
- hitter pool: `slots = derived.get(idx, set())` restricted to `HITTER_ELIGIBLE`; if
  empty, `{DH}` (UTIL-only, matching the old fallback's intent).
- pitcher pool: `{P}` for every row (the pool has one slot).

This mirrors `_slots_for`'s pool-restriction and fallback, but keyed by MLBAM and
per season. `build`/`projected` for the LIVE board is untouched and still uses the
Yahoo map via `pricing_table()`.

### Finding 5 -- UTIL floor
In `marginal_starter_floors`, compute the `UTIL` floor the same way as every other
slot: the best leftover player who `can_fill_slot(..., "UTIL")`, and keep
`max(dedicated floors)` only as the fallback when no such leftover exists. Today it
is unconditionally `max(dedicated)`, so a genuinely UTIL-only best-leftover (a real
DH) can never set it.

### Finding 6 -- prose
`--scarcity` prints the **centred per-season credit** beside each season's floor (it
prints only floors today). Delete the "the catcher figure ranges 0.50 to 2.18"
sentence from `keepers/scarcity.py`; the range is now a flag output, not a literal.

### Validation and coverage (finding 1's DoD)
`--scarcity` additionally prints:
- **per-season coverage**: how many board rows got a real (non-DH) position vs fell
  to the DH fallback, per pool and season, so the contamination cannot recur
  silently.
- **derived-vs-Yahoo agreement**: match the 10-game-rule eligibility against
  `load_positions()` (the real Yahoo map) and print per-slot agreement. The season
  compared is **2025, not 2026**: Yahoo's 2026 map derives its eligibility largely
  from the 2025 season, and 2026 is only half-played, so derived-2026 would spuriously
  under-match (a player Yahoo lists off 2025 who has not yet reached 10 games in 2026
  is not a rule error). Derived-2025 is a complete season and is the apples-to-apples
  comparison. The bridge is the **2025 board** (the season being validated), which
  carries both the MLBAM index and clean player `name`s: map derived-2025 (MLBAM) ->
  name via that board, then compare to Yahoo (`normalize_name` -> slots); names that
  do not bridge are counted as unmatched, not dropped. (The raw fielding pull's own
  `player.fullName` is cp1252-mangled, so the clean board names are used for the
  bridge, not the fielding frame's.) This is a diagnostic, not a hard gate -- but if a
  dedicated slot's agreement is grossly low (well under ~80% of Yahoo's players at
  that slot recovered), that signals a mapping bug to investigate BEFORE reshipping
  credits, not a number to accept.

## Requirements

R1. `keepers/mlb_stats.py` exposes a cached fielding pull for a season
    (`group='fielding'`), returning the raw splits keyed usably by MLBAM.
R2. `keepers/appearances.py::season_eligibility(fielding)` implements the 10-game
    rule with OF combined and the base-slot mapping above; pure and unit-tested.
R3. `run_scarcity` scores each season 2022-2025 against its own derived eligibility,
    keyed by the board's MLBAM index, with hitter fallback `{DH}` and pitcher `{P}`.
R4. `marginal_starter_floors` sets the UTIL floor from the best UTIL-eligible
    leftover, falling back to `max(dedicated)` only when none exists.
R5. `--scarcity` prints, per season: floors, centred credits, and coverage
    (real-position vs DH-fallback counts); and once, the derived-**2025**-vs-Yahoo
    agreement (bridged through the 2025 board's MLBAM<->name).
R6. The "0.50 to 2.18" sentence is removed from `keepers/scarcity.py`.
R7. `NATIVE_CREDITS` is regenerated from the corrected measurement; `--scarcity`
    reproduces it to delta 0.00 on every slot.
R8. The live 2026 `build` output is unchanged by R1-R4 until R7 lands (the Yahoo map
    still prices it); only the credits it applies change, and only at R7.
R9. `--league` and `--roster` re-run and the new top board recorded; PR #279's
    scarcity table and "catcher premium roughly right" claim updated to the
    corrected numbers.
R10. `pytest -n auto`, `ruff check .`, `ruff format --check .`, `mypy`, `vulture` all
     clean; no test loosened to pass (fix code or justify).

## Edge cases and failure modes

- **Pure DH / no fielding rows.** Absent from the fielding pull -> absent from
  `season_eligibility` -> caller assigns `{DH}` -> UTIL-only via `can_fill_slot`.
  Must NOT become C/OF/IF eligible. Unit-tested.
- **Scattered outfielder.** LF 6 + CF 5 = 11 combined -> OF eligible even though
  neither corner reached 10. This is intended (OF is combined). Unit-tested.
- **Sub-threshold everywhere.** A player with 9 games at his only position -> no base
  slot -> `{DH}`. Unit-tested (the user's catcher-with-9-games example).
- **Two-way player (Ohtani).** Appears in fielding as both P and a hitter position;
  the measurement is per pool, so the hitter-pool pass reads his hitter slots and the
  pitcher-pool pass assigns `{P}`. No cross-pool leakage.
- **Accented names.** The measurement joins on MLBAM id, so cp1252-mangled names in
  the raw pull are irrelevant. Only the 2026 validation touches names -> use
  `normalize_name` for the bridge, and count unbridgeable names as unmatched rather
  than silently dropping them.
- **2026 is deliberately unused for fielding.** The measurement uses complete seasons
  2022-2025, and the validation uses derived-2025 vs the Yahoo map; nothing needs 2026
  fielding, so it is not fetched. (The Yahoo 2026 map still comes from
  `load_positions()`, as today, for the live board.) An implementer must not add a
  2026 fielding fetch "for completeness" -- a half-season derived map is exactly the
  misleading artifact the validation-season choice avoids.
- **Numeric-default trap.** `games` must be read with an explicit missing->skip, not
  `games or 0` semantics that could sink a real 0 (there is no 0-game row, but follow
  the CLAUDE.md rule).
- **ASCII-only** in all source and printed strings; the script already reconfigures
  stdout to UTF-8 for player names (the documented exception).

## Testing expectations

- Unit tests for `season_eligibility`: the 10-game threshold (9 -> excluded, 10 ->
  included), OF combined across corners, base-slot mapping, DH/absent handling, and
  that pitchers map to `{P}`.
- Unit test for `marginal_starter_floors`' new UTIL-floor branch: a UTIL-only
  leftover that is better than every dedicated floor sets the UTIL floor; with no
  UTIL-only leftover it falls back to `max(dedicated)`.
- `test_scarcity.py` reviewed and updated deliberately for any assertion the F5/F6
  changes invalidate, with a stated justification (requirement changed, not a
  silent loosening).
- Before Phase 3 reships `NATIVE_CREDITS`, grep the suite for any test pinning a
  `NATIVE_CREDITS` literal or a credit-dependent board value (`proj_var`, a named
  keeper's rank/credit); update each to the regenerated numbers with justification,
  or confirm none exist. A silently-passing golden test over the old credits would
  otherwise hide the reship.
- **Phase 2 no-live-board-change gate.** `marginal_starter_floors` is called only by
  `run_scarcity`, never by `build`, so wiring the derived eligibility and fixing the
  UTIL floor must not touch the live board while `NATIVE_CREDITS` is unchanged.
  Verify: capture the 2026 `build` output CSV before Phase 2 and after, and require
  `rank`/`composite`/`proj_sgp`/`sd`/`proj_var` identical (the credits literal has not
  changed yet, so the board must not move until Phase 3).
- Integration (needs the fielding cache): `run_scarcity` completes, reproduces the
  new `NATIVE_CREDITS` to delta 0.00, and prints coverage + agreement.
- Full suite + lint/type/dead-code checks (R10).

## Prerequisites and risks

- **Fielding cache.** The fielding pull must be fetched for **2022-2025** and cached
  (like the skills cache). 2026 is not fetched (see the edge case). If the MLB Stats
  API is unreachable, the measurement cannot be regenerated -- surface, do not
  fabricate.
- **Credits move the board.** R7 changes every `proj_var`; the catcher-led top order
  will change. This is the intended outcome, but the PR narrative (R9) must be
  updated in the same change so the description never asserts the old numbers.

## Phasing

- **Phase 0 -- prerequisite.** Fetch and cache MLB Stats fielding for 2022-2025;
  surface if the API is unreachable.
- **Phase 1 -- derivation (no behaviour change).** Add the fielding pull to
  `mlb_stats.py` and `keepers/appearances.py::season_eligibility`; unit tests.
  Nothing wired in yet.
- **Phase 2 -- wire the measurement + F5/F6 (credits not yet reshipped).**
  `run_scarcity` uses derived eligibility; add coverage + 2026-agreement prints; fix
  the UTIL floor; delete the prose. `--scarcity` now prints new deltas against the
  still-shipped `NATIVE_CREDITS`.
- **Phase 3 -- reship + propagate.** Regenerate `NATIVE_CREDITS` to delta 0.00;
  re-run `--league`/`--roster`; update PR #279's description; full verification.
