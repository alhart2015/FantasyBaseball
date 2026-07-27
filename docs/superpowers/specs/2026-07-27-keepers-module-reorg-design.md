# Keepers Module Reorg -- Raw Ingest Only -- Design

- Date: 2026-07-27
- Status: Approved (design); pending implementation plan
- Issue: #265 (fresh `/keepers` module; keep only the Statcast/Savant ingest).
  Pairs with #264 (memory cleanup, done) and blocks #266 (feature definition).
- Related code (delete set): `src/fantasy_baseball/data/skill_luck.py`,
  `src/fantasy_baseball/analysis/breakout.py`,
  `src/fantasy_baseball/analysis/hr_confirm.py`,
  `src/fantasy_baseball/analysis/breakout_backtest.py`, and the breakout/HR
  backtest scripts + tests + specs/plans enumerated in "Delete set" below.
- Related code (untouched): `src/fantasy_baseball/analysis/keeper_value.py`,
  `src/fantasy_baseball/analysis/keeper_trades.py`, and their scripts/tests.

## Problem

The last two keeper-value PRs (#261 breakout/mirage diagnostic, #263
HR-confirmation + barrel-anchored backtests) produced analysis the owner has **no
confidence in**. The only durable output worth keeping is the **data-acquisition
layer**: the raw pulls from the MLB Stats API and Baseball Savant (both public,
no-auth, MLBAM-native -- no FanGraphs/Cloudflare). Everything computed on top of
those pulls -- rate derivations, unit conversions, joins, the `SkillLuckRow`
shape, the breakout/mirage classifier, the HR-confirmation term, and all the
backtests -- is distrusted and in the way of the fresh feature definition (#266).

Today the trusted pulls are interleaved with the distrusted calculation inside
one module (`data/skill_luck.py`), and the shared data shape (`SkillLuckRow`)
lives inside the classifier module (`analysis/breakout.py`) -- so the ingest
cannot simply be deleted around. This reorg separates the two and keeps only the
pulls.

## Goals

1. Stand up a fresh `src/fantasy_baseball/keepers/` module whose only content is
   the raw network fetchers for the MLB Stats API and Baseball Savant.
2. The fetchers return the **fully raw** upstream response -- no column selection,
   no rename, no arithmetic, no unit conversion. #266 rebuilds every derivation,
   join, and shape from scratch.
3. Delete all distrusted analysis, backtests, reports, and their specs/plans/CSV
   artifacts from the last two PRs.
4. Leave `keeper_value.py` / `keeper_trades.py` (earlier PRs #254/#255/#259)
   untouched -- their migration/redefinition is #266's call, not this reorg's.
5. `pytest` / `ruff` / `ruff format` / `mypy` / `vulture` all green.

## Non-goals (explicit; deferred to #266)

- Any derivation or value model (rates, joins, `SkillLuckRow`-style shapes, the
  keeper ranking).
- Any CLI or report. The breakout report (`run_breakout_report.py`) is deleted;
  the `keeper_value` / `keeper_trades` CLIs keep working unchanged.
- Migrating `keeper_value.py` / `keeper_trades.py` into `/keepers`.
- Caching redesign. `fetch_or_cache` is preserved as-is (plumbing, not
  calculation).

## Design

### Module layout (split by source)

```
src/fantasy_baseball/keepers/
  __init__.py     # re-exports the fetchers (public surface #266 imports)
  mlb_stats.py    # MLB Stats API pull
  savant.py       # Baseball Savant pulls (pybaseball expected-stats + xHR CSV)
  cache.py        # fetch_or_cache plumbing
tests/test_keepers/
  test_cache.py
  test_mlb_stats.py
  test_savant.py
```

Rationale: the two upstreams (statsapi.mlb.com via `requests`; Baseball Savant
via `pybaseball` + a direct CSV) are unrelated dependencies; separate files keep
each independently readable and testable and leave obvious homes for #266 to add
derivation modules (e.g. `keepers/value.py`) beside the ingest without touching
it. Raw pulls cache to `data/keepers/` (renamed off the old `data/skill_luck/`).

### `cache.py` -- plumbing (moved verbatim)

- `fetch_or_cache(path, fetcher, *, tolerate_empty=False) -> pd.DataFrame`:
  fetch-on-miss; read the CSV cache if present and non-empty; on a fresh fetch,
  refuse to write/overwrite when the pull is empty/failed (raise), except when
  `tolerate_empty=True` returns an expected-empty frame without caching it (the
  pre-2016 xHR case). `_read_cached(path)` helper.

### `mlb_stats.py` -- MLB Stats API season line, fully raw

- `fetch_mlb_season(cache_dir, year, group, *, fetcher=None) -> pd.DataFrame`
  where `group` is `"hitting"` or `"pitching"`. Paginated season-stats
  leaderboard (`playerPool="all"`), all pages concatenated.
- **Fully raw representation:** each split is flattened with `pd.json_normalize`
  so the frame keeps **every** field the API returns (`player.id`,
  `player.fullName`, all `stat.*`, `team.*`, etc.) -- nothing dropped. This is a
  deliberate change from the old `{mlbam: player.id, **stat}`, which silently
  dropped player name / team; "don't trust decisions to drop columns" (owner)
  makes keep-everything the correct default. The wider/messier frame is accepted
  for v1; #266 selects what it needs.
- Pagination and the `fetcher` injection seam (for tests) are preserved; the
  network call is a local import (`requests`) so the module stays import-safe.

### `savant.py` -- Baseball Savant pulls, fully raw

Each returns the upstream frame **as-is** -- no rename, no `/100`, no merge:

- `fetch_batter_expected(cache_dir, year, *, fetcher=None)` ->
  `pybaseball.statcast_batter_expected_stats(year, minPA=1)`.
- `fetch_batter_barrels(cache_dir, year, *, fetcher=None)` ->
  `pybaseball.statcast_batter_exitvelo_barrels(year, minBBE=1)`.
- `fetch_pitcher_expected(cache_dir, year, *, fetcher=None)` ->
  `pybaseball.statcast_pitcher_expected_stats(year, minPA=1)`.
- `fetch_savant_hr(cache_dir, year, *, fetcher=None)` -> the park-adjusted xHR
  leaderboard CSV (direct `urllib` fetch, browser UA, utf-8-sig BOM), raw. Uses
  `tolerate_empty=True` (pre-2016 returns a header-only body).

Note: `minPA=1` / `minBBE=1` are fetch **parameters** ("return the full
population," not Savant's default qualified-only filter), not calculations on the
returned data -- kept so the pulls are maximally complete. `pybaseball` calls are
local imports.

### `__init__.py`

Re-exports `fetch_or_cache` and the five fetchers as the module's public surface.

### Delete set (complete)

- **src:** `data/skill_luck.py`, `analysis/breakout.py`, `analysis/hr_confirm.py`,
  `analysis/breakout_backtest.py`.
- **scripts:** `run_breakout_report.py`, `backtest_breakout.py`,
  `backtest_coefficient.py`, `backtest_hr_confirm.py`, `backtest_hr_level.py`.
- **tests:** `test_analysis/test_breakout.py`,
  `test_analysis/test_breakout_backtest.py`, `test_analysis/test_hr_confirm.py`,
  `test_scripts/test_backtest_breakout.py`,
  `test_scripts/test_backtest_hr_confirm.py`,
  `test_scripts/test_backtest_hr_level.py`, `test_data/test_skill_luck.py`.
- **specs:** `docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md`,
  `docs/superpowers/specs/2026-07-26-barrel-anchored-hr-design.md`,
  `docs/superpowers/specs/2026-07-26-hr-confirmation-backtest-design.md`.
- **plans:** `docs/superpowers/plans/2026-07-24-keeper-breakout-diagnostic.md`,
  `docs/superpowers/plans/2026-07-26-barrel-anchored-hr.md`,
  `docs/superpowers/plans/2026-07-26-hr-confirmation-backtest.md`.
- **artifacts (untracked):** `data/stats/breakout_backtest_results.csv`,
  `data/stats/hr_confirm_backtest_results.csv`,
  `data/stats/hr_level_backtest_results.csv`.

### Untouched (verified independent of the delete set)

`analysis/keeper_value.py`, `analysis/keeper_trades.py`, `scripts/keeper_value.py`,
`scripts/keeper_trades.py`, `tests/test_analysis/test_keeper_value.py`,
`tests/test_analysis/test_keeper_trades.py`,
`tests/test_scripts/test_keeper_value_script.py`,
`tests/test_scripts/test_keeper_trades_script.py`, and the three keeper-value /
keeper-trade specs+plans. Their only lost consumer was `breakout.breakout_rows`;
neither module imports anything in the delete set (grep-verified).

## Testing

New `tests/test_keepers/` uses the `fetcher` injection seam (no real network):

- `test_cache.py`: `fetch_or_cache` behaviors -- miss -> fetch -> write; hit ->
  no fetch; empty pull -> raise; `tolerate_empty` -> return-without-cache.
- `test_mlb_stats.py`: a fake paginated payload flows through unmodified;
  `json_normalize` keeps the full column set (assert a name/team column survives
  that the old code dropped); pagination stops correctly.
- `test_savant.py`: each fetcher returns the injected frame byte-for-byte (raw
  pass-through -- no rename, no `/100`); `fetch_savant_hr` tolerates the pre-2016
  empty body.

No derivation, join, or shape remains to test.

## Gates + config touch-ups

- **mypy:** add `src/fantasy_baseball/keepers/` to `[tool.mypy] files` (the
  `analysis/` tree is already covered; keep the new module covered).
- **vulture:** the fetchers have **no in-repo caller until #266**, so vulture will
  report them as unused. Whitelist `keepers/` (or add the fetcher names to the
  vulture allowlist) so the "no new dead-code findings" gate passes -- they are
  intentional public API for #266. Deleting the old modules should otherwise
  reduce vulture findings.
- `pytest` / `ruff check` / `ruff format --check` all green.

## Acceptance (mirrors #265)

- `/keepers` module exists and holds the raw MLB Stats API + Baseball Savant
  pulls (and nothing else).
- All analysis/backtest/report artifacts from the last two PRs are removed.
- `pytest` / `ruff` / `mypy` (and `vulture`) green.

## Decisions settled during brainstorming

- **Intent = clean quarantine (option A):** move in only the ingest; delete the
  distrusted analysis; leave `keeper_value` / `keeper_trades` in `analysis/`.
- **Keep only the raw pulls; all calculation dropped** (owner: "all calculation
  should not carry over"), including `SkillLuckRow` and the rate derivations.
- **Fully raw frames** -- no column selection/rename (owner: "I don't trust any
  decisions made including choosing which columns to drop").
- **Split by source** layout.
