# fix(standings): trust Yahoo's points_for to avoid display-tie drift

## Summary

Our `/standings` page showed Spacemen 75.5 / Hart of the Order 74.5,
but Yahoo's official standings had them tied at 74.5. Root cause:
Yahoo's standings API exposes stat totals only at display precision
(e.g. WHIP `1.03`), while Yahoo's scoring engine runs on full-precision
internal stats. Our `score_roto` saw rounded ties that aren't real ties
inside Yahoo and awarded averaged ranks (9.5 / 9.5) instead of Yahoo's
tie-broken (10 / 9). Each display-level tie shifted our total by ±0.5
per team.

Fix: parse Yahoo's own `team_standings.points_for` (its authoritative
roto total) and use it for the displayed total/rank whenever it's
available. Score-based path still used for projected standings, where
we have full-precision inputs and no artificial ties.

## Why

The scoring-engine mismatch breaks the primary source of truth on the
dashboard. The user makes lineup, trade, and waiver decisions looking
at totals that should match Yahoo exactly. An off-by-0.5 total is a
correctness issue, not a cosmetic one — it misranks teams at the edges
of tie groups.

## Changes

- `src/fantasy_baseball/lineup/yahoo_roster.py` — `parse_standings_raw`
  now extracts `team_standings.points_for` (checking both
  `team_entry[1]` and `team_entry[2]` positions where Yahoo has been
  observed to place `team_standings`). Existing rank parsing consolidated
  into the same pass.
- `src/fantasy_baseball/models/standings.py` —
  `StandingsEntry` gains `yahoo_points_for: float | None = None`.
- `src/fantasy_baseball/web/season_data.py`:
  - `_standings_to_snapshot` propagates `points_for` onto the entry.
  - `format_standings_for_display` prefers `yahoo_points_for` for the
    displayed total and uses Yahoo's `rank` when available. Falls back
    to `score_roto` when no entry has `points_for` (projected
    standings). The original `score_roto` total is preserved under
    `roto_points["score_roto_total"]` for diagnostics.
- `scripts/debug_standings_mismatch.py` — investigation script that
  pulls raw Yahoo standings, prints per-team diffs (`score_roto` vs
  `points_for`), and dumps the raw snapshot for offline inspection.
- `scripts/verify_standings_fix.py` — smoke test that asserts displayed
  total == `points_for` for every team on the live league.
- Tests:
  - `tests/test_lineup/test_yahoo_roster.py` — coverage for
    `points_for` extraction (present and absent cases).
  - `tests/test_web/test_season_data.py` — coverage for Yahoo-override
    path and score_roto fallback.

## Behavior impact

- Live standings page (`/standings` → Current view) now matches Yahoo
  exactly on total and rank.
- Per-category points still come from `score_roto` over rounded stats.
  On teams involved in display-level rate-stat ties, per-cat cells may
  not sum to the headline total; the gap is ±0.5 per tie. Net across
  the league is zero.
- Projected standings (Preseason, Current Projected), Monte Carlo,
  leverage, roster audit, trade evaluator — all unchanged. They build
  snapshots from our own ROS projections, which have no rounding-tie
  issue.

## Cache compatibility

`yahoo_points_for` defaults to `None`. Caches written before this PR
have no `points_for` field; the code treats that as "not available" and
falls back to `score_roto`. The first refresh after deploy repopulates
the cache with `points_for` set.

## Verification

On 2026-04-18 against the live league:

```
$ python scripts/verify_standings_fix.py
Rank  Team                            DisplayedTotal  YahooPointsFor    CatSum
1     Spacemen                                 74.50            74.5     75.50
1     Hart of the Order                        74.50            74.5     74.50
3     Hello Peanuts!                           69.50            69.5     69.00
4     Jon's Underdogs                          56.50            56.5     56.00
4     Tortured Baseball Department             56.50            56.5     56.50
6     Springfield Isotopes                     55.00            55.0     55.00
7     Boston Estrellas                         45.00            45.0     45.00
8     SkeleThor                                44.50            44.5     44.50
9     Work in Progress                         39.00            39.0     39.50
10    Send in the Cavalli                      35.00            35.0     34.50

0 total mismatches
```

- `DisplayedTotal` — what `/standings` now renders.
- `YahooPointsFor` — Yahoo's authoritative total.
- `CatSum` — `score_roto`'s per-category sum; shown for transparency.

## Tests

- `pytest tests/` — 1123 passed, 0 failed.
- `ruff check` — no new violations introduced on touched files (17
  pre-existing; 17 after).
- `mypy src/fantasy_baseball/models/standings.py
  src/fantasy_baseball/scoring.py` — clean.

## Follow-ups (not in this PR)

- Tooltip on the Total cell in `standings.html` explaining that the
  headline comes from Yahoo and may differ from the category sum by
  ±0.5 per display tie.
- Longer-term: fetch raw component stats (H, AB, BB, H_allowed, ER, IP)
  so we can recompute rate stats at full precision and get exact
  agreement with Yahoo per category too.
