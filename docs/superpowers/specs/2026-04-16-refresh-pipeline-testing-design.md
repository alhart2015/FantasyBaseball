# Refresh Pipeline Testing & Decomposition

## Goal

Make `run_full_refresh` understandable, testable, and resistant to wiring regressions, without locking down its non-deterministic outputs (Monte Carlo) or rewriting domain code that already has tests.

## Motivation

`src/fantasy_baseball/web/refresh_pipeline.py::run_full_refresh` is 847 lines containing 24 sequential `# --- Step N` blocks. There is no test file for it. The existing `tests/test_web/test_season_data.py` has source-string regression guards but no behavioral coverage.

The pipeline is the single most important orchestration in the project: it produces every cache artifact the season dashboard reads. The bugs it actually suffers from are not arithmetic errors inside individual steps (the underlying domain modules — `optimizer`, `simulation`, `leverage`, `pace`, `spoe` — already have tests) but data-shape mismatches between steps and silent skips when intermediate state is wrong.

## Approach

Two layers of coverage, matched to two kinds of bug:

1. **Pure helpers** with unit tests — nine functions extracted from the inline computational logic currently buried inside step blocks. These are the pieces where a per-function unit test catches a real arithmetic or shape bug.
2. **One integration test** with shape + invariant assertions — for the wiring contracts between steps. Shape-only smoke is too weak; full value snapshots are too brittle (MC randomness, frequent projection updates). Invariant assertions are the sweet spot.

Plus a structural change: `run_full_refresh` becomes `RefreshRun(cache_dir).run()`, with each current step block as a named method on the class. The class holds shared state as instance attributes so methods don't need 10-arg signatures. This is for skim-readability of the top-level flow — the methods themselves are not individually unit-tested; the integration test covers them collectively.

## Pure helpers and destinations

Each helper goes to the module that owns its domain. New unit tests live in the existing test directory for that module.

| Helper | Destination | Test location |
|---|---|---|
| `compute_effective_date(end_date)` | `utils/time_utils.py` | `tests/test_utils/test_time_utils.py` |
| `compute_fraction_remaining(season_start, season_end, today)` | `utils/time_utils.py` | `tests/test_utils/test_time_utils.py` |
| `build_projected_standings(rosters)` | `scoring/team_projection.py` (new) | `tests/test_scoring/test_team_projection.py` (new) |
| `build_team_sds(rosters, sd_scale)` | `scoring/team_projection.py` (new) | same as above |
| `attach_pace_to_roster(players, hitter_logs, pitcher_logs, preseason_lookup, sgp_denoms)` | `analysis/pace.py` | `tests/test_analysis/test_pace.py` |
| `build_rankings_lookup(ros, preseason, current)` | `sgp/rankings.py` | `tests/test_sgp/test_rankings.py` |
| `merge_matched_and_raw_roster(matched, roster_raw, preseason_lookup)` | `web/refresh_steps.py` (new) | `tests/test_web/test_refresh_steps.py` (new) |
| `compute_lineup_moves(optimal_hitters, roster_players)` | `web/refresh_steps.py` | same as above |
| `build_positions_map(roster_players, opp_rosters, fa_players)` | `web/refresh_steps.py` | same as above |

Note: `compute_pending_moves_diff` already exists in `season_data` — not in this list.

## RefreshRun class

Lives in `web/refresh_pipeline.py`. Replaces the body of `run_full_refresh`.

```python
class RefreshRun:
    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir
        self.logger = JobLogger("refresh")
        # Shared state — populated as steps run
        self.config = None
        self.league = None              # Yahoo session-bound league
        self.league_model = None        # League dataclass loaded from Redis
        self.user_team_key = None
        self.standings = None
        self.standings_snap = None
        self.projected_standings = None
        self.projected_standings_snap = None
        self.team_sds = None
        self.fraction_remaining = None
        self.effective_date = None
        self.start_date = None
        self.end_date = None
        self.roster_raw = None
        self.raw_rosters_by_team = None
        self.opp_rosters = None
        self.matched = None
        self.roster_players = None
        self.preseason_lookup = None
        self.preseason_hitters = None
        self.preseason_pitchers = None
        self.hitters_proj = None
        self.pitchers_proj = None
        self.has_rest_of_season = False
        self.hitter_logs = None
        self.pitcher_logs = None
        self.leverage = None
        self.rankings_lookup = None
        self.optimal_hitters = None
        self.optimal_pitchers_starters = None
        self.optimal_pitchers_bench = None
        self.fa_players = None

    def run(self) -> None:
        # Same try/except/finally + _refresh_status protocol as today
        ...

    # One method per current `# --- Step N` block, in order:
    def _authenticate(self): ...
    def _find_user_team(self): ...
    def _fetch_standings_and_roster(self): ...
    def _load_projections(self): ...
    def _fetch_opponent_rosters(self): ...
    def _write_snapshots_and_load_league(self): ...
    def _hydrate_rosters(self): ...
    def _build_projected_standings(self): ...
    def _compute_leverage(self): ...
    def _match_roster_to_projections(self): ...
    def _fetch_game_logs(self): ...
    def _compute_pace(self): ...
    def _compute_wsgp(self): ...
    def _compute_rankings(self): ...
    def _optimize_lineup(self): ...
    def _compute_moves(self): ...
    def _fetch_probable_starters(self): ...
    def _audit_roster(self): ...
    def _compute_per_team_leverage(self): ...
    def _run_monte_carlo(self): ...
    def _run_ros_monte_carlo(self): ...
    def _compute_spoe(self): ...
    def _analyze_transactions(self): ...
    def _write_meta(self): ...

    def _progress(self, msg: str) -> None:
        _set_refresh_progress(msg)
        self.logger.log(msg)
        log.info(msg)
```

`run_full_refresh` is preserved as a thin module-level wrapper:

```python
def run_full_refresh(cache_dir: Path = CACHE_DIR) -> None:
    RefreshRun(cache_dir).run()
```

This keeps `scripts/run_lineup.py` and `season_routes.py` working unchanged.

Module-level state — `_refresh_lock`, `_refresh_status`, `get_refresh_status`, `_set_refresh_progress` — stays at module level because it is shared across threads.

## Bonus dedup

`fraction_remaining` is currently computed twice: once in step 4e for SD scaling, once in step 13b for ROS Monte Carlo. With the class, this is computed once in `__init__` (or in an early step) and read by both consumers via `self.fraction_remaining`.

## Integration test

File: `tests/test_web/test_refresh_pipeline.py`

Setup:
- Hand-built minimal League: 12 teams (matches league size), ~10 players per roster (mix of hitters and pitchers, IL slots, bench).
- `fake_redis` from existing `conftest.py` for Redis.
- Mock Yahoo: patch `get_league`, `get_yahoo_session`, `fetch_roster`, `fetch_standings`, `fetch_scoring_period`, `fetch_all_transactions`, `fetch_and_match_free_agents`, `fetch_game_log_totals`, `get_week_schedule`, `get_team_batting_stats` to return canned data from fixtures.
- Use `tmp_path` for `cache_dir`.

Assertions (shape):
- Every expected cache file is written: `standings`, `pending_moves`, `projections`, `roster`, `rankings`, `lineup_optimal`, `probable_starters`, `positions`, `roster_audit`, `leverage`, `monte_carlo`, `spoe`, `transaction_analyzer`, `meta`, `opp_rosters`.
- Each file's top-level structure has the expected keys and types.

Assertions (invariants):
- Every team in `standings` appears in `projected_standings`.
- Every player in cached `roster` has `pace` populated.
- Every player referenced in `lineup_optimal.moves` exists in `roster`.
- `positions` covers every player from roster + opp_rosters + fa_players.
- `monte_carlo` has `base` and `with_management` always; has `rest_of_season` and `rest_of_season_with_management` only when ROS projections are present.
- `meta.last_refresh` is set; `meta.team_name` matches config.

Parametrize on `has_rest_of_season` (true/false) so the ROS-MC branch is exercised both ways.

## Update existing regression tests

`tests/test_web/test_season_data.py` currently uses `inspect.getsource(refresh_pipeline.run_full_refresh)` for two guards:
- Every `config.<attr>` reference resolves to a real `LeagueConfig` field.
- No local `from datetime import date` inside the function.

These guards now need to walk every method on `RefreshRun` instead. Helper:

```python
def _refresh_run_source() -> str:
    cls = refresh_pipeline.RefreshRun
    return "\n".join(
        inspect.getsource(getattr(cls, name))
        for name in dir(cls)
        if callable(getattr(cls, name)) and not name.startswith("__")
    )
```

Both regression guards swap their `inspect.getsource(run_full_refresh)` call for `_refresh_run_source()`.

## Out of scope

- Per-method unit tests for `RefreshRun` methods (covered by integration test).
- Pushing Yahoo I/O behind an interface or further abstracting the fetch layer.
- Changing Monte Carlo determinism, seeding, or iteration count.
- Re-architecting `_refresh_status` / progress reporting.
- Snapshot-style golden-file tests for cache outputs.
- Refactoring projection blending or any downstream domain module beyond the helper extractions listed above.
