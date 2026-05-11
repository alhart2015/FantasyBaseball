# Hot Streaks — Phase B (Dashboard Panel) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Two new dashboard surfaces — a `/streaks` page that ports the Phase 5 Sunday report into HTML (sortable tables, FA-count selector) and a per-row chip on the existing `/lineup` hitters table — both driven by a new `CacheKey.STREAK_SCORES` populated by the existing refresh pipeline.

**Architecture:** Extract the Phase 5 Sunday CLI's orchestration into a shared `streaks/pipeline.py::compute_streak_report` that does the full sequence (DB refresh → refit-or-load models → Yahoo fetch → `build_report`) and returns the existing render-agnostic `Report` dataclass. Refactor the Sunday CLI to delegate, then add a `RefreshRun._compute_streaks` method that calls the same function, serializes the `Report` to JSON, and writes the cache. Both new dashboard surfaces are pure read-from-cache consumers — no inference at request time. Model refit runs only when `model_fits` is missing or ≥14 days old (or when forced via flag), keeping refresh wall time predictable.

**Tech Stack:** Flask + Jinja2 (existing dashboard), DuckDB (existing streaks DB), htmx is available but not used here (small datasets, client-side sort sufficient), `unittest.mock` for Yahoo stubbing in tests. No new runtime dependencies.

**Spec:** `docs/superpowers/specs/2026-05-11-hot-streaks-phase-b-dashboard-panel-design.md`.

---

## Design Decisions (resolved during brainstorm — see spec for full rationale)

1. **Two surfaces**: `/streaks` page + per-row chip on `/lineup` hitters table.
2. **Lineup-indicator density**: composite chip (`HOT` / `COLD` / `—`) + label naming the strongest hot-or-cold category when active (e.g., `HOT · HR`). Five mini per-cat cells were ruled out as too wide.
3. **Streaks-page interaction**: sortable column headers (client-side JS) + FA-count `<select>` (10 / 25 / 50). No filters in v1.
4. **Hitters only**, inherited from Phase 5. Pitcher table on `/lineup` is unchanged.
5. **Inference reuse**: dashboard step calls `compute_streak_report`; no duplicated math. The shared function wraps the same DB-refresh + refit-or-load + score sequence the Sunday CLI runs.
6. **Refit policy**: refit when `model_fits` is missing or the most recent `refit_at` is ≥14 days old, else load from `model_fits`. A `force_refit` parameter overrides.
7. **Cache shape**: faithful JSON serialization of the `Report` dataclass (no derived fields). Top 50 FAs are always cached so the dropdown is a pure client-side slice.
8. **Name lookup**: `normalize_name` on both sides at runtime. Unresolved players get a neutral `—` chip.
9. **Top category** (chip label): the cat whose `PlayerCategoryScore.label` matches the composite direction with the highest `probability`. Alphabetical tie-break for determinism.
10. **Empty-cache UX**: Streaks page shows "No streak data yet — run a refresh"; Lineup chips render `—` everywhere. Dashboard still works fully without streak data.
11. **Failure isolation**: `_compute_streaks` logs and swallows on failure; other refresh steps continue. Cache is not overwritten on failure.

---

## File Structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/data/cache_keys.py` | Modify | Add `STREAK_SCORES = "streak_scores"` to `CacheKey` enum |
| `src/fantasy_baseball/streaks/inference.py` | Modify | Add `load_models_from_fits(conn)` next to existing `refit_models_for_report` |
| `src/fantasy_baseball/streaks/pipeline.py` | Create | `compute_streak_report(conn, *, league, config, projections_root, scoring_season, season_set_train, force_refit) -> Report`. Wraps DB refresh + refit-or-load + Yahoo fetch + `build_report`. |
| `src/fantasy_baseball/streaks/dashboard.py` | Create | `serialize_report(report) -> dict`, `deserialize_report(payload) -> Report`, `build_indicator(name, cached_payload) -> Indicator`, `Indicator` dataclass |
| `scripts/streaks/run_sunday_report.py` | Modify | Replace inline orchestration in `main()` with a single `compute_streak_report(...)` call. Add `--force-refit`; remove the `--skip-refit` SystemExit stub. |
| `src/fantasy_baseball/web/refresh_pipeline.py` | Modify | Add `_compute_streaks` method on `RefreshRun`; wire into `run()` after `_analyze_transactions`. |
| `src/fantasy_baseball/web/season_routes.py` | Modify | Add `/streaks` route; modify `lineup` route to inject `streak_indicator` per hitter |
| `src/fantasy_baseball/web/templates/season/streaks.html` | Create | Three sections (Roster / Top FAs / Drivers), sortable tables, FA-count `<select>` |
| `src/fantasy_baseball/web/templates/season/_lineup_hitters_tbody.html` | Modify | New rightmost `<td>` rendering the chip |
| `src/fantasy_baseball/web/templates/season/base.html` | Modify | Add "Streaks" sidebar entry between Lineup and Roster Audit |
| `src/fantasy_baseball/web/static/season.css` | Modify | Chip classes (`streak-chip.streak-hot/cold/neutral`), Streaks-page section/table styles |
| `tests/test_streaks/test_inference.py` | Modify | Add `load_models_from_fits` round-trip test |
| `tests/test_streaks/test_pipeline.py` | Create | Refit-vs-reuse decision tests; end-to-end with seeded DB + mocked Yahoo |
| `tests/test_streaks/test_dashboard.py` | Create | `Report` ↔ JSON round-trip; `build_indicator` cases (hot/cold/neutral/missing/tiebreak) |
| `tests/test_web/test_refresh_pipeline.py` | Modify | Extend with `_compute_streaks` cache-write + failure-isolation tests |
| `tests/test_web/test_streaks_route.py` | Create | `/streaks` integration (seeded cache + empty cache) |
| `tests/test_web/test_season_routes.py` | Modify | Extend with `/lineup` chip injection tests |
| `tests/test_web/test_streaks_snapshot.py` | Create | One canonical Streaks-page HTML snapshot |
| `pyproject.toml` | Modify | Add `streaks.pipeline`, `streaks.dashboard` to `[tool.mypy].files` |

---

## Tasks

### Task 1: Add STREAK_SCORES to CacheKey

**Files:**
- Modify: `src/fantasy_baseball/data/cache_keys.py`

- [ ] **Step 1: Add the enum member**

```python
# in CacheKey
STREAK_SCORES = "streak_scores"
```

Append after the existing `STANDINGS_BREAKDOWN` line so the diff is one-liner.

- [ ] **Step 2: Run the test suite**

Run: `pytest tests/test_web/test_no_json_file_cache.py tests/test_web/test_season_data.py -v`
Expected: PASS — new member doesn't break existing logic.

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/data/cache_keys.py
git commit -m "feat(streaks): add STREAK_SCORES cache key"
```

---

### Task 2: load_models_from_fits — round-trip from model_fits table

**Files:**
- Modify: `src/fantasy_baseball/streaks/inference.py`
- Test: `tests/test_streaks/test_inference.py`

The Phase 4 `refit_models_for_report` persists fit metadata (coefficients, intercept, scaler params) into `model_fits`. Phase 5 always refits. Phase B needs the inverse: reconstruct fitted sklearn `Pipeline` objects from the stored rows so we can score without refitting.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_streaks/test_inference.py`:

```python
def test_load_models_from_fits_round_trips_predictions(seeded_pipeline_conn) -> None:
    """A model loaded from model_fits scores identically to the freshly-refit one (within tolerance)."""
    fresh = refit_models_for_report(seeded_pipeline_conn, season_set_train="2023-2024", window_days=14)
    loaded = load_models_from_fits(seeded_pipeline_conn)

    assert set(loaded.keys()) == set(fresh.keys())
    # Predict on a deterministic feature row; the two pipelines must
    # produce identical (or near-identical) probabilities.
    X = _example_feature_row()  # 1-row DataFrame matching EXPECTED_FEATURE_COLUMNS
    for key, fresh_model in fresh.items():
        loaded_model = loaded[key]
        p_fresh = fresh_model.pipeline.predict_proba(X)[0, 1]
        p_loaded = loaded_model.pipeline.predict_proba(X)[0, 1]
        assert abs(p_fresh - p_loaded) < 1e-9, (
            f"Mismatch for {key}: fresh={p_fresh}, loaded={p_loaded}"
        )
```

`seeded_pipeline_conn` is a fixture that calls `_seed_pipeline` from `tests/test_streaks/test_predictors.py` for both 2023 and 2024 plus the full Phase 4 fit + persist sequence. If a similar fixture doesn't already exist in `test_inference.py`, add one in the same file using the same `_seed_pipeline` import — match the pattern already used by `test_refit_models_for_report_writes_model_fits_rows` (search that test name in the file).

`_example_feature_row()` returns a 1-row DataFrame with constant values for each column in `EXPECTED_FEATURE_COLUMNS`. Put it next to the test as a private helper.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streaks/test_inference.py::test_load_models_from_fits_round_trips_predictions -v`
Expected: FAIL with `AttributeError: module 'fantasy_baseball.streaks.inference' has no attribute 'load_models_from_fits'`.

- [ ] **Step 3: Implement `load_models_from_fits`**

Add to `src/fantasy_baseball/streaks/inference.py` (next to `refit_models_for_report`):

```python
def load_models_from_fits(conn: duckdb.DuckDBPyConnection) -> dict[tuple[StreakCategory, StreakDirection], FittedModel]:
    """Reconstruct fitted Pipelines from the most recent ``model_fits`` rows.

    Selects the most recent ``refit_at`` group; for each ``(category,
    direction)`` row in that group, rebuilds a ``Pipeline`` of
    ``StandardScaler`` + ``LogisticRegression`` whose parameters are
    set directly from the persisted coefficients / intercept / scaler
    mean / scaler scale. The result is byte-identical predictions to
    the original fit (no retraining).

    Raises ``RuntimeError`` if ``model_fits`` is empty.
    """
    rows = conn.execute(
        """
        SELECT category, direction, cold_method, dense_quintile_cutoffs,
               feature_columns, coef, intercept,
               scaler_mean, scaler_scale, refit_at
        FROM model_fits
        WHERE refit_at = (SELECT MAX(refit_at) FROM model_fits)
        """
    ).fetchall()
    if not rows:
        raise RuntimeError("model_fits is empty; refit before loading")

    out: dict[tuple[StreakCategory, StreakDirection], FittedModel] = {}
    for (
        category, direction, cold_method, quintile_cutoffs,
        feature_columns, coef, intercept, scaler_mean, scaler_scale, _refit_at,
    ) in rows:
        scaler = StandardScaler()
        scaler.mean_ = np.asarray(scaler_mean, dtype=np.float64)
        scaler.scale_ = np.asarray(scaler_scale, dtype=np.float64)
        scaler.var_ = scaler.scale_ ** 2
        scaler.n_features_in_ = len(feature_columns)
        scaler.feature_names_in_ = np.asarray(feature_columns, dtype=object)

        clf = LogisticRegression()
        clf.coef_ = np.asarray([coef], dtype=np.float64)
        clf.intercept_ = np.asarray([intercept], dtype=np.float64)
        clf.classes_ = np.asarray([0, 1])
        clf.n_features_in_ = len(feature_columns)
        clf.feature_names_in_ = np.asarray(feature_columns, dtype=object)

        pipeline = Pipeline([("scaler", scaler), ("clf", clf)])
        cutoffs = tuple(quintile_cutoffs) if quintile_cutoffs is not None else None
        out[(StreakCategory(category), StreakDirection(direction))] = FittedModel(
            pipeline=pipeline,
            category=StreakCategory(category),
            direction=StreakDirection(direction),
            cold_method=ColdMethod(cold_method),
            dense_quintile_cutoffs=cutoffs,
        )
    return out
```

Imports to add at the top of `inference.py` if not already present: `from sklearn.preprocessing import StandardScaler`, `from sklearn.linear_model import LogisticRegression`, `from sklearn.pipeline import Pipeline`, `import numpy as np`.

The exact column names (`coef`, `scaler_mean`, etc.) must match the `model_fits` schema in `streaks/data/schema.py` — verify before writing the SQL. If column names differ, use the actual names and update the unpacking.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_streaks/test_inference.py::test_load_models_from_fits_round_trips_predictions -v`
Expected: PASS.

- [ ] **Step 5: Run the wider inference test file**

Run: `pytest tests/test_streaks/test_inference.py -v`
Expected: PASS — existing tests untouched.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/streaks/inference.py tests/test_streaks/test_inference.py
git commit -m "feat(streaks): load_models_from_fits — reconstruct Pipeline from model_fits row"
```

---

### Task 3: streaks/pipeline.py — compute_streak_report (staleness logic + skeleton)

**Files:**
- Create: `src/fantasy_baseball/streaks/pipeline.py`
- Test: `tests/test_streaks/test_pipeline.py`

This task ships the staleness-decision logic only. Yahoo fetch + full orchestration come in Task 4. Splitting the task keeps each TDD cycle small.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_streaks/test_pipeline.py`:

```python
"""Tests for streaks/pipeline.py — refit-or-load decision."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from fantasy_baseball.streaks.pipeline import _should_refit


def test_should_refit_true_when_no_fits(seeded_pipeline_conn_no_fits) -> None:
    assert _should_refit(seeded_pipeline_conn_no_fits, max_age_days=14, force=False) is True


def test_should_refit_false_when_recent_fits(seeded_pipeline_conn_with_recent_fits) -> None:
    assert _should_refit(seeded_pipeline_conn_with_recent_fits, max_age_days=14, force=False) is False


def test_should_refit_true_when_stale_fits(seeded_pipeline_conn_with_old_fits) -> None:
    assert _should_refit(seeded_pipeline_conn_with_old_fits, max_age_days=14, force=False) is True


def test_should_refit_true_when_forced_even_if_recent(seeded_pipeline_conn_with_recent_fits) -> None:
    assert _should_refit(seeded_pipeline_conn_with_recent_fits, max_age_days=14, force=True) is True
```

Fixtures live in a `conftest.py` next to the test file. Add:

```python
# tests/test_streaks/conftest.py (extend if exists, create otherwise)
import pytest
from datetime import datetime, timedelta, timezone

from fantasy_baseball.streaks.data.schema import get_connection

from tests.test_streaks.test_predictors import _seed_pipeline


@pytest.fixture
def seeded_pipeline_conn_no_fits(tmp_path):
    db = tmp_path / "s.duckdb"
    conn = get_connection(db)
    _seed_pipeline(conn, season=2023)
    _seed_pipeline(conn, season=2024)
    yield conn
    conn.close()


def _stamp_fits(conn, *, age_days: int) -> None:
    """Backdate every row in model_fits to (now - age_days)."""
    ts = datetime.now(timezone.utc) - timedelta(days=age_days)
    conn.execute("UPDATE model_fits SET refit_at = ?", [ts])


@pytest.fixture
def seeded_pipeline_conn_with_recent_fits(seeded_pipeline_conn_no_fits):
    from fantasy_baseball.streaks.inference import refit_models_for_report
    refit_models_for_report(seeded_pipeline_conn_no_fits, season_set_train="2023-2024", window_days=14)
    _stamp_fits(seeded_pipeline_conn_no_fits, age_days=1)
    return seeded_pipeline_conn_no_fits


@pytest.fixture
def seeded_pipeline_conn_with_old_fits(seeded_pipeline_conn_no_fits):
    from fantasy_baseball.streaks.inference import refit_models_for_report
    refit_models_for_report(seeded_pipeline_conn_no_fits, season_set_train="2023-2024", window_days=14)
    _stamp_fits(seeded_pipeline_conn_no_fits, age_days=30)
    return seeded_pipeline_conn_no_fits
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_streaks/test_pipeline.py -v`
Expected: FAIL with `ImportError: cannot import name '_should_refit' from 'fantasy_baseball.streaks.pipeline'`.

- [ ] **Step 3: Create the module with `_should_refit`**

Create `src/fantasy_baseball/streaks/pipeline.py`:

```python
"""End-to-end orchestration for the hot-streaks pipeline.

Wraps the DB-refresh sequence (fetch logs/statcast, upsert projection
rates, recompute windows/thresholds/labels), the refit-or-load model
decision, the Yahoo fetch, and ``build_report`` into a single function
called by both the Sunday CLI and the dashboard refresh pipeline.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import duckdb

logger = logging.getLogger("streaks.pipeline")


_DEFAULT_MAX_FIT_AGE_DAYS = 14


def _should_refit(
    conn: duckdb.DuckDBPyConnection, *, max_age_days: int, force: bool
) -> bool:
    """Return True iff models should be refit rather than loaded.

    True when ``force`` is set, when ``model_fits`` is empty, or when
    the most recent ``refit_at`` is older than ``max_age_days``.
    """
    if force:
        return True
    row = conn.execute("SELECT MAX(refit_at) FROM model_fits").fetchone()
    if row is None or row[0] is None:
        return True
    most_recent = row[0]
    if most_recent.tzinfo is None:
        most_recent = most_recent.replace(tzinfo=timezone.utc)
    return most_recent < datetime.now(timezone.utc) - timedelta(days=max_age_days)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_streaks/test_pipeline.py -v`
Expected: 4 PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/streaks/pipeline.py tests/test_streaks/test_pipeline.py tests/test_streaks/conftest.py
git commit -m "feat(streaks): pipeline._should_refit (refit-or-load decision)"
```

---

### Task 4: compute_streak_report — full orchestration

**Files:**
- Modify: `src/fantasy_baseball/streaks/pipeline.py`
- Test: `tests/test_streaks/test_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_streaks/test_pipeline.py`:

```python
def test_compute_streak_report_end_to_end(seeded_pipeline_conn_no_fits, monkeypatch, tmp_path) -> None:
    """End-to-end with seeded DB + stubbed Yahoo fetch.

    Verifies the returned Report has the expected shape (one roster
    row + one FA row) and that models were refit (model_fits populated).
    """
    from fantasy_baseball.streaks.pipeline import compute_streak_report
    from fantasy_baseball.streaks.reports.sunday import YahooHitter

    # Project a single hitter with mlbam=1 via a synthetic CSV so the
    # name→mlbam map resolves both stubbed Yahoo names.
    projections_root = tmp_path / "projections"
    season_dir = projections_root / "2024"
    season_dir.mkdir(parents=True)
    (season_dir / "steamer-hitters.csv").write_text(
        "Name,MLBAMID,PA,HR,R,RBI,SB,AVG\n"
        "Roster Guy,1,600,20,80,80,10,0.270\n"
        "FA Guy,2,600,20,80,80,10,0.270\n"
    )

    def _fake_fetch_yahoo(league, *, team_name):
        return (
            [YahooHitter(name="Roster Guy", positions=("OF",), yahoo_id="1", status="")],
            [YahooHitter(name="FA Guy", positions=("OF",), yahoo_id="2", status="")],
        )

    monkeypatch.setattr(
        "fantasy_baseball.streaks.pipeline._fetch_yahoo_hitters", _fake_fetch_yahoo
    )

    # Stub fetch_season — seeded DB already has data.
    monkeypatch.setattr(
        "fantasy_baseball.streaks.pipeline.fetch_season",
        lambda *, season, conn, **kw: {"season": season, "stub": True},
    )

    fake_league = object()
    report = compute_streak_report(
        seeded_pipeline_conn_no_fits,
        league=fake_league,
        team_name="Hart of the Order",
        league_id=5652,
        projections_root=projections_root,
        scoring_season=2024,
        season_set_train="2023-2024",
        force_refit=False,
    )

    assert report.team_name == "Hart of the Order"
    assert report.league_id == 5652
    assert len(report.roster_rows) == 1
    # FA row may or may not appear depending on composite=0 filter; assert
    # the overall report shape rather than the FA count.

    # Models must have been refit (no prior fits).
    n_fits = seeded_pipeline_conn_no_fits.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    assert n_fits > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streaks/test_pipeline.py::test_compute_streak_report_end_to_end -v`
Expected: FAIL with `ImportError: cannot import name 'compute_streak_report'`.

- [ ] **Step 3: Implement `compute_streak_report` and `_fetch_yahoo_hitters`**

Extend `src/fantasy_baseball/streaks/pipeline.py`:

```python
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

from fantasy_baseball.lineup.yahoo_roster import (
    fetch_free_agents,
    fetch_roster,
    fetch_teams,
    find_user_team_key,
)
from fantasy_baseball.streaks.data.fetch_history import fetch_season
from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.projections import (
    load_projection_rates_for_seasons,
)
from fantasy_baseball.streaks.inference import (
    load_models_from_fits,
    refit_models_for_report,
)
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.reports.sunday import (
    Report,
    YahooHitter,
    build_name_to_mlbam_map,
    build_report,
)
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows
from fantasy_baseball.utils.time_utils import local_today


_HITTER_FA_POSITIONS: tuple[str, ...] = ("C", "1B", "2B", "3B", "SS", "OF", "Util")


def _normalize_position(p: str) -> str:
    return p.upper()


def _to_yahoo_hitter(entry: dict) -> YahooHitter:
    positions = tuple(_normalize_position(p) for p in entry.get("positions", []))
    return YahooHitter(
        name=entry["name"],
        positions=positions,
        yahoo_id=str(entry.get("player_id", "")),
        status=entry.get("status", "") or "",
    )


def _fetch_yahoo_hitters(league, *, team_name: str) -> tuple[list[YahooHitter], list[YahooHitter]]:
    """Identical to scripts/streaks/run_sunday_report.py::_fetch_yahoo_data.

    Lifted here so dashboard refresh and the Sunday CLI share one
    implementation. The CLI will be refactored in Task 5 to delegate.
    """
    teams = fetch_teams(league)
    user_team_key = find_user_team_key(teams, team_name)
    roster_raw = fetch_roster(league, user_team_key)
    roster_hitters = [_to_yahoo_hitter(p) for p in roster_raw]

    def _fetch_one(pos: str) -> list[dict]:
        try:
            return fetch_free_agents(league, pos, count=50)
        except Exception:
            logger.exception("Free agent fetch failed at position %s; continuing", pos)
            return []

    seen: set[str] = set()
    fa_hitters: list[YahooHitter] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for fa_raw in pool.map(_fetch_one, _HITTER_FA_POSITIONS):
            for fa in fa_raw:
                key = fa["name"].lower().strip()
                if key in seen:
                    continue
                seen.add(key)
                fa_hitters.append(_to_yahoo_hitter(fa))
    return roster_hitters, fa_hitters


def _refresh_streaks_db(
    conn: duckdb.DuckDBPyConnection,
    *,
    season: int,
    season_set_train: str,
    projections_root: Path,
    skip_fetch: bool,
) -> None:
    """Steps 1-4 of the pipeline. Lifted from the Sunday CLI."""
    if not skip_fetch:
        logger.info("Fetching %d game logs + Statcast (incremental)...", season)
        summary = fetch_season(season=season, conn=conn)
        logger.info("fetch_season summary: %s", summary)
    else:
        logger.info("--skip-fetch set; using cached game logs + Statcast")

    logger.info("Loading %d projection rates from %s...", season, projections_root)
    rates = load_projection_rates_for_seasons(projections_root, [season])
    upsert_projection_rates(conn, rates)

    logger.info("Recomputing hitter_windows...")
    n_windows = compute_windows(conn)
    logger.info("  wrote %d window rows", n_windows)

    logger.info("Recomputing thresholds and labels on %s...", season_set_train)
    compute_thresholds(conn, season_set=season_set_train)
    n_labels = apply_labels(conn, season_set=season_set_train)
    logger.info("  wrote %d label rows", n_labels)


def compute_streak_report(
    conn: duckdb.DuckDBPyConnection,
    *,
    league,
    team_name: str,
    league_id: int,
    projections_root: Path,
    scoring_season: int,
    season_set_train: str = "2023-2025",
    window_days: int = 14,
    top_n_fas: int = 50,
    force_refit: bool = False,
    skip_fetch: bool = False,
    max_fit_age_days: int = _DEFAULT_MAX_FIT_AGE_DAYS,
    today: date | None = None,
) -> Report:
    """End-to-end streak report orchestration.

    Runs DB refresh, refit-or-load models, Yahoo fetch, score, return.
    ``league`` is an opaque Yahoo league handle (e.g. the value returned
    by ``yahoo_auth.get_league``); not type-annotated to keep this
    module testable without importing yahoo_fantasy_api.
    """
    _refresh_streaks_db(
        conn,
        season=scoring_season,
        season_set_train=season_set_train,
        projections_root=projections_root,
        skip_fetch=skip_fetch,
    )

    if _should_refit(conn, max_age_days=max_fit_age_days, force=force_refit):
        logger.info("Refitting models on %s...", season_set_train)
        models = refit_models_for_report(
            conn, season_set_train=season_set_train, window_days=window_days
        )
    else:
        logger.info("Reusing models from model_fits")
        models = load_models_from_fits(conn)

    roster_hitters, fa_hitters = _fetch_yahoo_hitters(league, team_name=team_name)
    logger.info(
        "Yahoo fetch complete: %d roster, %d FAs (deduped)",
        len(roster_hitters),
        len(fa_hitters),
    )

    name_to_mlbam = build_name_to_mlbam_map(projections_root, season=scoring_season)
    if not name_to_mlbam:
        raise RuntimeError(
            f"No name→mlbam mappings built — check that {projections_root}/"
            f"{scoring_season}/ contains hitter CSVs with Name + MLBAMID columns."
        )

    return build_report(
        conn,
        league_config_team_name=team_name,
        league_config_league_id=league_id,
        models=models,
        roster_hitters=roster_hitters,
        fa_hitters=fa_hitters,
        name_to_mlbam=name_to_mlbam,
        today=today or local_today(),
        season_set_train=season_set_train,
        scoring_season=scoring_season,
        window_days=window_days,
        top_n_fas=top_n_fas,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_streaks/test_pipeline.py::test_compute_streak_report_end_to_end -v`
Expected: PASS.

- [ ] **Step 5: Verify mypy on the new module**

Run: `mypy src/fantasy_baseball/streaks/pipeline.py`
Expected: clean (or matches the pattern in `streaks/reports/sunday.py` if `league` typing requires `Any`).

If mypy complains, add `from typing import Any` and annotate `league: Any`.

- [ ] **Step 6: Add to pyproject mypy files**

Edit `pyproject.toml`, find the `[tool.mypy].files` list, add `"src/fantasy_baseball/streaks/pipeline.py"`.

Re-run: `mypy` (full project).
Expected: clean.

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/streaks/pipeline.py tests/test_streaks/test_pipeline.py pyproject.toml
git commit -m "feat(streaks): compute_streak_report — shared end-to-end orchestration"
```

---

### Task 5: Refactor Sunday CLI to delegate

**Files:**
- Modify: `scripts/streaks/run_sunday_report.py`
- (Existing test) `tests/test_scripts/test_run_sunday_report.py` — must still pass

- [ ] **Step 1: Read the existing test**

Run: `pytest tests/test_scripts/test_run_sunday_report.py -v`
Expected: PASS — baseline before refactor.

- [ ] **Step 2: Refactor `main()` to delegate**

Replace the body of `main()` from the line after `args = parser.parse_args(argv)` through the line before `args.output_dir.mkdir(...)` with:

```python
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    config = load_config(args.league_config)
    scoring_season = args.scoring_season or config.season_year
    logger.info(
        "Sunday report for league %d (%s) — scoring season %d",
        config.league_id,
        config.team_name,
        scoring_season,
    )

    if args.skip_refit:
        raise SystemExit(
            "--skip-refit was removed in favor of model_fits reuse; "
            "use --force-refit to bypass reuse when needed."
        )

    conn = get_connection(args.db_path)
    try:
        from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session

        session = get_yahoo_session()
        league = get_league(session, config.league_id, config.game_code)

        report = compute_streak_report(
            conn,
            league=league,
            team_name=config.team_name,
            league_id=config.league_id,
            projections_root=args.projections_root,
            scoring_season=scoring_season,
            season_set_train=args.season_set_train,
            window_days=14,
            top_n_fas=10,
            force_refit=args.force_refit,
            skip_fetch=args.skip_fetch,
        )
    finally:
        conn.close()
```

Update the argparse block: replace the `--skip-refit` action with `--force-refit`:

```python
    parser.add_argument(
        "--force-refit",
        action="store_true",
        help="Refit models even if model_fits is recent. Default is "
        "reuse-when-recent (≤14 days).",
    )
```

Remove now-unused imports at the top of the file:
- `from fantasy_baseball.streaks.data.fetch_history import fetch_season`
- `from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates`
- `from fantasy_baseball.streaks.data.projections import load_projection_rates_for_seasons`
- `from fantasy_baseball.streaks.inference import refit_models_for_report`
- `from fantasy_baseball.streaks.labels import apply_labels`
- `from fantasy_baseball.streaks.reports.sunday import (build_name_to_mlbam_map, build_report)` — keep `render_markdown`, `render_terminal`, `YahooHitter` removed too
- `from fantasy_baseball.streaks.thresholds import compute_thresholds`
- `from fantasy_baseball.streaks.windows import compute_windows`
- `from fantasy_baseball.utils.time_utils import local_today`
- `from concurrent.futures import ThreadPoolExecutor`

Remove the helper functions `_normalize_position`, `_to_yahoo_hitter`, `_fetch_yahoo_data`, `_refresh_streaks_db` from this file (they live in `streaks/pipeline.py` now).

Add the new import at the top:

```python
from fantasy_baseball.streaks.pipeline import compute_streak_report
from fantasy_baseball.streaks.reports.sunday import render_markdown, render_terminal
```

- [ ] **Step 3: Run the existing CLI test**

Run: `pytest tests/test_scripts/test_run_sunday_report.py -v`
Expected: PASS — the test stubs Yahoo + `fetch_season`. Update the stub target if the existing patch path no longer resolves: previously patched `run_sunday_report._fetch_yahoo_data` and `fetch_season` — after the refactor, patch `fantasy_baseball.streaks.pipeline._fetch_yahoo_hitters` and `fantasy_baseball.streaks.pipeline.fetch_season`.

If the Phase 5 test was patching by string path inside the script module (e.g., `patch("run_sunday_report.fetch_season", ...)`) you'll need to update the strings. Skim the patch calls and rewrite as needed.

- [ ] **Step 4: Re-run full streaks tests**

Run: `pytest tests/test_streaks/ tests/test_scripts/test_run_sunday_report.py -v`
Expected: all PASS.

- [ ] **Step 5: Verify lint**

Run: `ruff check scripts/streaks/run_sunday_report.py`
Expected: clean. Fix any unused-import (F401) findings — those should disappear with the import removals above.

Run: `ruff format scripts/streaks/run_sunday_report.py`
Expected: no diff (or run with `--check` first).

- [ ] **Step 6: Commit**

```bash
git add scripts/streaks/run_sunday_report.py tests/test_scripts/test_run_sunday_report.py
git commit -m "refactor(streaks): Sunday CLI delegates to streaks.pipeline.compute_streak_report"
```

---

### Task 6: Report ↔ JSON serialization

**Files:**
- Create: `src/fantasy_baseball/streaks/dashboard.py`
- Test: `tests/test_streaks/test_dashboard.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_streaks/test_dashboard.py`:

```python
"""Tests for streaks/dashboard.py — serialization and indicator."""

from __future__ import annotations

from datetime import date

from fantasy_baseball.streaks.dashboard import (
    deserialize_report,
    serialize_report,
)
from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
from fantasy_baseball.streaks.models import (
    StreakCategory,
    StreakDirection,  # adjust import if location differs
    StreakLabel,
)
from fantasy_baseball.streaks.reports.sunday import (
    DriverLine,
    Report,
    ReportRow,
)


def _example_report() -> Report:
    score_hr = PlayerCategoryScore(
        player_id=665742,
        category=StreakCategory.HR,
        label=StreakLabel.HOT,
        probability=0.62,
        drivers=(Driver(feature="barrel_pct", z_score=1.8),),
        window_end=date(2026, 5, 10),
    )
    score_avg = PlayerCategoryScore(
        player_id=665742,
        category=StreakCategory.AVG,
        label=StreakLabel.NEUTRAL,
        probability=None,
        drivers=(),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="Juan Soto",
        positions=("OF",),
        player_id=665742,
        composite=1,
        scores={StreakCategory.HR: score_hr, StreakCategory.AVG: score_avg},
        max_probability=0.62,
    )
    driver_line = DriverLine(
        player_name="Juan Soto",
        category=StreakCategory.HR,
        label=StreakLabel.HOT,
        probability=0.62,
        drivers=(Driver(feature="barrel_pct", z_score=1.8),),
    )
    return Report(
        report_date=date(2026, 5, 11),
        window_end=date(2026, 5, 10),
        team_name="Hart of the Order",
        league_id=5652,
        season_set_train="2023-2025",
        roster_rows=(row,),
        fa_rows=(),
        driver_lines=(driver_line,),
        skipped=("Foo — no_window",),
    )


def test_serialize_report_round_trips() -> None:
    original = _example_report()
    payload = serialize_report(original)
    rebuilt = deserialize_report(payload)
    assert rebuilt == original
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_streaks/test_dashboard.py::test_serialize_report_round_trips -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement serialization**

Create `src/fantasy_baseball/streaks/dashboard.py`:

```python
"""Dashboard glue for the hot-streaks pipeline.

Serialization helpers translate the in-memory ``Report`` dataclass to
JSON-safe dicts (and back) for transport through Redis/SQLite cache.
``build_indicator`` is the Lineup-page hook: given a hitter name and
a cached payload, returns the chip's tone + label + tooltip.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Literal

from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
from fantasy_baseball.streaks.models import (
    StreakCategory,
    StreakLabel,
)
from fantasy_baseball.streaks.reports.sunday import (
    DriverLine,
    Report,
    ReportRow,
)


def _serialize_driver(d: Driver) -> dict[str, Any]:
    return {"feature": d.feature, "z_score": d.z_score}


def _deserialize_driver(p: dict[str, Any]) -> Driver:
    return Driver(feature=p["feature"], z_score=float(p["z_score"]))


def _serialize_score(s: PlayerCategoryScore) -> dict[str, Any]:
    return {
        "player_id": s.player_id,
        "category": s.category.value,
        "label": s.label.value,
        "probability": s.probability,
        "drivers": [_serialize_driver(d) for d in s.drivers],
        "window_end": s.window_end.isoformat() if s.window_end else None,
    }


def _deserialize_score(p: dict[str, Any]) -> PlayerCategoryScore:
    return PlayerCategoryScore(
        player_id=int(p["player_id"]),
        category=StreakCategory(p["category"]),
        label=StreakLabel(p["label"]),
        probability=p["probability"],
        drivers=tuple(_deserialize_driver(d) for d in p["drivers"]),
        window_end=date.fromisoformat(p["window_end"]) if p["window_end"] else None,
    )


def _serialize_row(r: ReportRow) -> dict[str, Any]:
    return {
        "name": r.name,
        "positions": list(r.positions),
        "player_id": r.player_id,
        "composite": r.composite,
        "max_probability": r.max_probability,
        "scores": {cat.value: _serialize_score(score) for cat, score in r.scores.items()},
    }


def _deserialize_row(p: dict[str, Any]) -> ReportRow:
    return ReportRow(
        name=p["name"],
        positions=tuple(p["positions"]),
        player_id=int(p["player_id"]),
        composite=int(p["composite"]),
        max_probability=float(p["max_probability"]),
        scores={
            StreakCategory(cat): _deserialize_score(score)
            for cat, score in p["scores"].items()
        },
    )


def _serialize_driver_line(dl: DriverLine) -> dict[str, Any]:
    return {
        "player_name": dl.player_name,
        "category": dl.category.value,
        "label": dl.label.value,
        "probability": dl.probability,
        "drivers": [_serialize_driver(d) for d in dl.drivers],
    }


def _deserialize_driver_line(p: dict[str, Any]) -> DriverLine:
    return DriverLine(
        player_name=p["player_name"],
        category=StreakCategory(p["category"]),
        label=StreakLabel(p["label"]),
        probability=float(p["probability"]),
        drivers=tuple(_deserialize_driver(d) for d in p["drivers"]),
    )


def serialize_report(report: Report) -> dict[str, Any]:
    return {
        "report_date": report.report_date.isoformat(),
        "window_end": report.window_end.isoformat() if report.window_end else None,
        "team_name": report.team_name,
        "league_id": report.league_id,
        "season_set_train": report.season_set_train,
        "roster_rows": [_serialize_row(r) for r in report.roster_rows],
        "fa_rows": [_serialize_row(r) for r in report.fa_rows],
        "driver_lines": [_serialize_driver_line(dl) for dl in report.driver_lines],
        "skipped": list(report.skipped),
    }


def deserialize_report(payload: dict[str, Any]) -> Report:
    return Report(
        report_date=date.fromisoformat(payload["report_date"]),
        window_end=date.fromisoformat(payload["window_end"]) if payload["window_end"] else None,
        team_name=payload["team_name"],
        league_id=int(payload["league_id"]),
        season_set_train=payload["season_set_train"],
        roster_rows=tuple(_deserialize_row(r) for r in payload["roster_rows"]),
        fa_rows=tuple(_deserialize_row(r) for r in payload["fa_rows"]),
        driver_lines=tuple(_deserialize_driver_line(dl) for dl in payload["driver_lines"]),
        skipped=tuple(payload["skipped"]),
    )
```

Verify the enum import paths — `StreakCategory` and `StreakLabel` may live in `streaks/models.py` or `streaks/labels.py`. Run `grep -n "class StreakCategory" src/fantasy_baseball/streaks/` to confirm before writing the import.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_streaks/test_dashboard.py::test_serialize_report_round_trips -v`
Expected: PASS.

- [ ] **Step 5: Add to mypy files**

Edit `pyproject.toml`: add `"src/fantasy_baseball/streaks/dashboard.py"` to `[tool.mypy].files`.

Run: `mypy src/fantasy_baseball/streaks/dashboard.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/streaks/dashboard.py tests/test_streaks/test_dashboard.py pyproject.toml
git commit -m "feat(streaks): dashboard.serialize_report ↔ deserialize_report"
```

---

### Task 7: build_indicator

**Files:**
- Modify: `src/fantasy_baseball/streaks/dashboard.py`
- Modify: `tests/test_streaks/test_dashboard.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_streaks/test_dashboard.py`:

```python
from fantasy_baseball.streaks.dashboard import Indicator, build_indicator


def test_build_indicator_hot_picks_top_hot_cat() -> None:
    payload = serialize_report(_example_report())
    ind = build_indicator("Juan Soto", payload)
    assert ind.tone == "hot"
    assert ind.label == "HOT · HR"


def test_build_indicator_cold_picks_top_cold_cat() -> None:
    # Modify the example: flip HR to COLD, AVG stays NEUTRAL.
    # Resulting composite = -1, top cat = HR with prob 0.62.
    original = _example_report()
    row = original.roster_rows[0]
    cold_score = PlayerCategoryScore(
        player_id=row.player_id,
        category=StreakCategory.HR,
        label=StreakLabel.COLD,
        probability=0.62,
        drivers=(Driver(feature="barrel_pct", z_score=-1.8),),
        window_end=date(2026, 5, 10),
    )
    flipped_row = ReportRow(
        name=row.name,
        positions=row.positions,
        player_id=row.player_id,
        composite=-1,
        scores={StreakCategory.HR: cold_score, StreakCategory.AVG: row.scores[StreakCategory.AVG]},
        max_probability=0.62,
    )
    flipped = Report(
        report_date=original.report_date, window_end=original.window_end,
        team_name=original.team_name, league_id=original.league_id,
        season_set_train=original.season_set_train,
        roster_rows=(flipped_row,), fa_rows=(),
        driver_lines=(), skipped=(),
    )
    payload = serialize_report(flipped)
    ind = build_indicator("Juan Soto", payload)
    assert ind.tone == "cold"
    assert ind.label == "COLD · HR"


def test_build_indicator_neutral_when_composite_zero() -> None:
    # All cats NEUTRAL → composite=0 → neutral chip, no label.
    neutral_score = PlayerCategoryScore(
        player_id=1,
        category=StreakCategory.HR,
        label=StreakLabel.NEUTRAL,
        probability=None,
        drivers=(),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="Neutral Guy", positions=("OF",), player_id=1,
        composite=0,
        scores={StreakCategory.HR: neutral_score},
        max_probability=0.0,
    )
    rpt = Report(
        report_date=date(2026, 5, 11), window_end=date(2026, 5, 10),
        team_name="t", league_id=1, season_set_train="2023-2025",
        roster_rows=(row,), fa_rows=(), driver_lines=(), skipped=(),
    )
    payload = serialize_report(rpt)
    ind = build_indicator("Neutral Guy", payload)
    assert ind.tone == "neutral"
    assert ind.label == "—"


def test_build_indicator_unresolved_player() -> None:
    payload = serialize_report(_example_report())
    ind = build_indicator("Unknown Hitter", payload)
    assert ind.tone == "neutral"
    assert ind.label == "—"
    assert "No streak data" in ind.tooltip


def test_build_indicator_returns_none_when_cache_missing() -> None:
    assert build_indicator("Juan Soto", None) is None


def test_build_indicator_tiebreak_alphabetical() -> None:
    # Two HOT cats with identical probability — alphabetical tiebreak (HR < R).
    score_hr = PlayerCategoryScore(
        player_id=1, category=StreakCategory.HR, label=StreakLabel.HOT,
        probability=0.6, drivers=(), window_end=date(2026, 5, 10),
    )
    score_r = PlayerCategoryScore(
        player_id=1, category=StreakCategory.R, label=StreakLabel.HOT,
        probability=0.6, drivers=(), window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="Tied Guy", positions=("OF",), player_id=1, composite=2,
        scores={StreakCategory.HR: score_hr, StreakCategory.R: score_r},
        max_probability=0.6,
    )
    rpt = Report(
        report_date=date(2026, 5, 11), window_end=date(2026, 5, 10),
        team_name="t", league_id=1, season_set_train="2023-2025",
        roster_rows=(row,), fa_rows=(), driver_lines=(), skipped=(),
    )
    payload = serialize_report(rpt)
    ind = build_indicator("Tied Guy", payload)
    assert ind.label == "HOT · HR"  # HR alphabetically before R
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_streaks/test_dashboard.py -v`
Expected: 6 new tests FAIL with `ImportError`.

- [ ] **Step 3: Implement `Indicator` and `build_indicator`**

Append to `src/fantasy_baseball/streaks/dashboard.py`:

```python
@dataclass(frozen=True)
class Indicator:
    """One Lineup-page chip: tone + label + tooltip."""

    tone: Literal["hot", "cold", "neutral"]
    label: str
    tooltip: str


def _row_lookup_by_normalized_name(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Build {normalize_name(row.name): row_dict} from roster + FAs.

    Roster wins ties with FAs (a player can theoretically appear in both
    if the cache was written mid-roster-move). Already-normalized name
    comparison is the contract.
    """
    from fantasy_baseball.utils.name_utils import normalize_name

    out: dict[str, dict[str, Any]] = {}
    for row in payload.get("fa_rows", []):
        out[normalize_name(row["name"])] = row
    for row in payload.get("roster_rows", []):
        out[normalize_name(row["name"])] = row
    return out


def _top_cat_label(row: dict[str, Any], tone: Literal["hot", "cold"]) -> str:
    """Find the cat with the highest probability matching the tone.

    Alphabetical tiebreak on the category enum value for determinism.
    """
    target = "HOT" if tone == "hot" else "COLD"
    candidates: list[tuple[float, str]] = []
    for cat_value, score in row["scores"].items():
        if score["label"] != target:
            continue
        prob = score["probability"] or 0.0
        candidates.append((prob, cat_value))
    if not candidates:
        return "—"
    # max by (probability desc, name asc) → sort ascending by name then
    # take max by probability with stable order.
    candidates.sort(key=lambda x: (-x[0], x[1]))
    top_cat = candidates[0][1]
    return f"{target} · {top_cat}"


def build_indicator(name: str, payload: dict[str, Any] | None) -> Indicator | None:
    """Build the Lineup-page chip for one hitter name.

    Returns ``None`` when the cache is missing (so the route can decide
    to render a default placeholder). Returns ``Indicator(tone='neutral',
    label='—', tooltip='No streak data')`` when the name doesn't resolve.
    """
    if payload is None:
        return None

    from fantasy_baseball.utils.name_utils import normalize_name

    lookup = _row_lookup_by_normalized_name(payload)
    row = lookup.get(normalize_name(name))
    if row is None:
        return Indicator(tone="neutral", label="—", tooltip="No streak data")

    composite = row["composite"]
    if composite > 0:
        tone: Literal["hot", "cold", "neutral"] = "hot"
    elif composite < 0:
        tone = "cold"
    else:
        return Indicator(
            tone="neutral",
            label="—",
            tooltip=f"composite=0 (no active streaks)",
        )

    label = _top_cat_label(row, tone)
    # Tooltip lists active cats in composite direction with their probs.
    target = "HOT" if tone == "hot" else "COLD"
    bits: list[str] = []
    for cat_value, score in row["scores"].items():
        if score["label"] == target and score["probability"] is not None:
            bits.append(f"{cat_value} ({int(round(score['probability'] * 100))}%)")
    bits.sort()
    tooltip = f"composite={'+' if composite > 0 else ''}{composite} · top: " + ", ".join(bits)
    return Indicator(tone=tone, label=label, tooltip=tooltip)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_streaks/test_dashboard.py -v`
Expected: all PASS.

- [ ] **Step 5: Lint + mypy**

Run: `ruff check src/fantasy_baseball/streaks/dashboard.py && mypy src/fantasy_baseball/streaks/dashboard.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/streaks/dashboard.py tests/test_streaks/test_dashboard.py
git commit -m "feat(streaks): build_indicator — lineup chip from cached report"
```

---

### Task 8: _compute_streaks pipeline method + wire into RefreshRun.run

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`
- Modify: `tests/test_web/test_refresh_pipeline.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web/test_refresh_pipeline.py`:

```python
def test_compute_streaks_writes_cache(monkeypatch, kv_isolation) -> None:
    """_compute_streaks wraps compute_streak_report + serializes + writes cache."""
    from datetime import date
    from fantasy_baseball.data import kv_store
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
    from fantasy_baseball.streaks.models import (
        StreakCategory, StreakLabel,
    )
    from fantasy_baseball.streaks.reports.sunday import Report, ReportRow
    from fantasy_baseball.web.refresh_pipeline import RefreshRun

    score = PlayerCategoryScore(
        player_id=1, category=StreakCategory.HR, label=StreakLabel.HOT,
        probability=0.6, drivers=(Driver(feature="barrel_pct", z_score=1.0),),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="X", positions=("OF",), player_id=1, composite=1,
        scores={StreakCategory.HR: score}, max_probability=0.6,
    )
    fake_report = Report(
        report_date=date(2026, 5, 11), window_end=date(2026, 5, 10),
        team_name="t", league_id=1, season_set_train="2023-2025",
        roster_rows=(row,), fa_rows=(), driver_lines=(), skipped=(),
    )

    monkeypatch.setattr(
        "fantasy_baseball.web.refresh_pipeline.compute_streak_report",
        lambda *a, **kw: fake_report,
    )

    run = _refresh_run_stub()  # helper that builds a RefreshRun with league + config preloaded
    run._compute_streaks()

    cached = kv_store.get_kv().get(redis_key(CacheKey.STREAK_SCORES))
    assert cached is not None
    import json
    payload = json.loads(cached)
    assert payload["team_name"] == "t"
    assert len(payload["roster_rows"]) == 1


def test_compute_streaks_swallows_failures(monkeypatch, kv_isolation, caplog) -> None:
    """A failure in compute_streak_report logs but doesn't crash the pipeline."""
    from fantasy_baseball.data import kv_store
    from fantasy_baseball.data.cache_keys import CacheKey, redis_key

    def _boom(*a, **kw):
        raise RuntimeError("DuckDB unhappy")

    monkeypatch.setattr(
        "fantasy_baseball.web.refresh_pipeline.compute_streak_report", _boom
    )

    run = _refresh_run_stub()
    run._compute_streaks()  # must not raise

    cached = kv_store.get_kv().get(redis_key(CacheKey.STREAK_SCORES))
    assert cached is None  # not overwritten on failure
    assert any("DuckDB unhappy" in r.message for r in caplog.records)
```

The `_refresh_run_stub()` helper instantiates a `RefreshRun` and sets the minimum state `_compute_streaks` reads: `self.config` (with `team_name`, `league_id`, `season_year`), `self.league` (opaque sentinel), `self.logger` (existing `JobLogger` or a `MagicMock` if available). Add this helper to the existing test file alongside any analogous helpers — read the file first to follow its conventions.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web/test_refresh_pipeline.py::test_compute_streaks_writes_cache tests/test_web/test_refresh_pipeline.py::test_compute_streaks_swallows_failures -v`
Expected: FAIL with `AttributeError: 'RefreshRun' object has no attribute '_compute_streaks'`.

- [ ] **Step 3: Implement `_compute_streaks` + wire it into `run()`**

At the top of `src/fantasy_baseball/web/refresh_pipeline.py`, add imports:

```python
from fantasy_baseball.streaks.dashboard import serialize_report
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection
from fantasy_baseball.streaks.pipeline import compute_streak_report
```

Find the existing `_analyze_transactions` method and add `_compute_streaks` after it:

```python
    # --- Step N: Compute streak scores for /streaks + lineup chips ---
    def _compute_streaks(self) -> None:
        """Run the full streak pipeline + serialize + write cache.

        Failures are logged but not re-raised — streak data is
        non-load-bearing for the rest of the dashboard.
        """
        import json

        self._progress("Computing streak scores...")
        try:
            assert self.config is not None
            assert self.league is not None
            project_root = Path(__file__).resolve().parents[3]
            conn = get_connection(DEFAULT_DB_PATH)
            try:
                report = compute_streak_report(
                    conn,
                    league=self.league,
                    team_name=self.config.team_name,
                    league_id=self.config.league_id,
                    projections_root=project_root / "data" / "projections",
                    scoring_season=self.config.season_year,
                    top_n_fas=50,
                )
            finally:
                conn.close()
            payload = serialize_report(report)
            write_cache(CacheKey.STREAK_SCORES, payload)
            self._progress(
                f"Streak scores cached: {len(report.roster_rows)} roster, "
                f"{len(report.fa_rows)} FAs"
            )
        except Exception:
            log.exception("Streak computation failed; cache unchanged")
            self._progress("Streak computation failed (continuing)")
```

Add `self._compute_streaks()` to the call sequence in `run()`, after `self._analyze_transactions()` and before `self._write_meta()`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: all PASS — new tests green, existing tests unchanged.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py tests/test_web/test_refresh_pipeline.py
git commit -m "feat(streaks): RefreshRun._compute_streaks — populate STREAK_SCORES cache"
```

---

### Task 9: /streaks route + base template Roster section + empty state

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Create: `src/fantasy_baseball/web/templates/season/streaks.html`
- Create: `tests/test_web/test_streaks_route.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web/test_streaks_route.py`:

```python
"""Integration tests for the /streaks route."""

from __future__ import annotations

import json
from datetime import date

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.streaks.dashboard import serialize_report
from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
from fantasy_baseball.streaks.models import StreakCategory, StreakLabel
from fantasy_baseball.streaks.reports.sunday import (
    DriverLine, Report, ReportRow,
)
from fantasy_baseball.web.season_app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


@pytest.fixture
def kv_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "test.db"))
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


def _seed_streak_cache() -> None:
    score = PlayerCategoryScore(
        player_id=1, category=StreakCategory.HR, label=StreakLabel.HOT,
        probability=0.6, drivers=(Driver(feature="barrel_pct", z_score=1.0),),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="Test Player", positions=("OF",), player_id=1, composite=1,
        scores={StreakCategory.HR: score}, max_probability=0.6,
    )
    dl = DriverLine(
        player_name="Test Player", category=StreakCategory.HR,
        label=StreakLabel.HOT, probability=0.6,
        drivers=(Driver(feature="barrel_pct", z_score=1.0),),
    )
    rpt = Report(
        report_date=date(2026, 5, 11), window_end=date(2026, 5, 10),
        team_name="Hart of the Order", league_id=5652,
        season_set_train="2023-2025",
        roster_rows=(row,), fa_rows=(), driver_lines=(dl,), skipped=(),
    )
    kv_store.get_kv().set(redis_key(CacheKey.STREAK_SCORES), json.dumps(serialize_report(rpt)))


def test_streaks_route_with_seeded_cache(client, kv_isolation) -> None:
    _seed_streak_cache()
    resp = client.get("/streaks")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "Your Roster" in body
    assert "Top Free Agent Signals" in body
    assert "Drivers" in body
    assert "Test Player" in body


def test_streaks_route_empty_state(client, kv_isolation) -> None:
    resp = client.get("/streaks")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "No streak data yet" in body
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web/test_streaks_route.py -v`
Expected: FAIL with 404 (route not registered).

- [ ] **Step 3: Add the route**

Find `register_routes` in `src/fantasy_baseball/web/season_routes.py`. Add a new route after the existing `/transactions` route (any location is fine — pick one consistent with the surrounding routes):

```python
    @app.route("/streaks")
    def streaks():
        import json as _json
        payload_raw = read_cache(CacheKey.STREAK_SCORES)
        payload = payload_raw if isinstance(payload_raw, dict) else None
        return render_template(
            "season/streaks.html",
            payload=payload,
            meta=_load_meta(),
            active_page="streaks",
        )
```

Use `_load_meta()` if that's the existing helper for sidebar meta; otherwise pass whatever the other routes pass.

`read_cache` lives at module top alongside `CacheKey` — verify the import is already there.

- [ ] **Step 4: Create the template**

Create `src/fantasy_baseball/web/templates/season/streaks.html`:

```html
{% extends "season/base.html" %}
{% block title %}Streaks · {{ super() }}{% endblock %}

{% block content %}
<div class="page-header">
  <h1>Streaks</h1>
  {% if payload %}
    <p class="page-sub">
      Week of {{ payload.report_date }}
      {% if payload.window_end %} · Window ends {{ payload.window_end }}{% endif %}
    </p>
  {% endif %}
</div>

{% if not payload %}
  <div class="empty-state">
    <p>No streak data yet — run a refresh to compute streak signals.</p>
  </div>
{% else %}
  <section class="streaks-section">
    <h2>Your Roster</h2>
    <table class="streaks-table" data-sortable>
      <thead>
        <tr>
          <th onclick="sortStreaksTable(this, 'name', 'asc')">Name</th>
          <th onclick="sortStreaksTable(this, 'pos', 'asc')">Pos</th>
          <th onclick="sortStreaksTable(this, 'avg', 'desc')">AVG</th>
          <th onclick="sortStreaksTable(this, 'hr', 'desc')">HR</th>
          <th onclick="sortStreaksTable(this, 'r', 'desc')">R</th>
          <th onclick="sortStreaksTable(this, 'rbi', 'desc')">RBI</th>
          <th onclick="sortStreaksTable(this, 'sb', 'desc')">SB</th>
          <th onclick="sortStreaksTable(this, 'cmp', 'desc')">Cmp</th>
        </tr>
      </thead>
      <tbody>
        {% for row in payload.roster_rows %}
          {% include "season/_streaks_row.html" %}
        {% endfor %}
      </tbody>
    </table>
  </section>

  <section class="streaks-section">
    <div class="section-head">
      <h2>Top Free Agent Signals</h2>
      <label>Show top:
        <select id="fa-count" onchange="filterFaRows(this.value)">
          <option value="10">10</option>
          <option value="25">25</option>
          <option value="50">50</option>
        </select>
      </label>
    </div>
    <table class="streaks-table" data-sortable id="fa-table">
      <thead>
        <tr>
          <th>Name</th><th>Pos</th><th>Active Cats</th><th>|Cmp|</th>
        </tr>
      </thead>
      <tbody>
        {% for row in payload.fa_rows %}
        <tr data-rank="{{ loop.index }}">
          <td>{{ row.name }}</td>
          <td>{{ row.positions | join(", ") }}</td>
          <td>
            {% for cat, score in row.scores.items() if score.label != "NEUTRAL" %}
              <span class="streak-chip streak-{{ score.label | lower }}">{{ cat }}</span>
            {% endfor %}
          </td>
          <td>{{ row.composite | abs }}</td>
        </tr>
        {% endfor %}
      </tbody>
    </table>
  </section>

  <section class="streaks-section">
    <h2>Drivers</h2>
    <ul class="drivers">
      {% for dl in payload.driver_lines %}
        <li>
          <strong>{{ dl.player_name }}</strong> ·
          {{ dl.category }} {{ dl.label }} ({{ (dl.probability * 100) | round | int }}%):
          {% for d in dl.drivers %}
            {{ d.feature }} {{ "%+0.1f"|format(d.z_score) }}z{% if not loop.last %}, {% endif %}
          {% endfor %}
        </li>
      {% endfor %}
    </ul>
  </section>

  <script src="{{ url_for('static', filename='streaks.js') }}"></script>
{% endif %}
{% endblock %}
```

Also create `src/fantasy_baseball/web/templates/season/_streaks_row.html`:

```html
<tr>
  <td>{{ row.name }}</td>
  <td>{{ row.positions | join(", ") }}</td>
  {% for cat in ["AVG", "HR", "R", "RBI", "SB"] %}
    {% set score = row.scores.get(cat) %}
    <td>
      {% if score and score.label != "NEUTRAL" %}
        <span class="streak-chip streak-{{ score.label | lower }}">{{ score.label }}</span>
      {% else %}
        —
      {% endif %}
    </td>
  {% endfor %}
  <td>{{ "%+d"|format(row.composite) }}</td>
</tr>
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_web/test_streaks_route.py -v`
Expected: both PASS. If the template references `streaks.js` and Flask can't find it yet, that's fine — the page still renders. Add the file in Task 11.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py \
        src/fantasy_baseball/web/templates/season/streaks.html \
        src/fantasy_baseball/web/templates/season/_streaks_row.html \
        tests/test_web/test_streaks_route.py
git commit -m "feat(web): /streaks route + template (roster + FAs + drivers)"
```

---

### Task 10: Streaks-page JS (client-side sort + FA-count slice)

**Files:**
- Create: `src/fantasy_baseball/web/static/streaks.js`

No new test — covered by the snapshot test in Task 13.

- [ ] **Step 1: Write `streaks.js`**

Create `src/fantasy_baseball/web/static/streaks.js`:

```js
/* Streaks page client-side helpers. Small datasets — pure DOM ops. */
(function () {
  "use strict";

  function _cellValue(row, key) {
    // Map column key → comparable value. Keys correspond to the
    // onclick="sortStreaksTable(this, '<key>', ...)" arguments in the
    // streaks.html template.
    const cells = row.children;
    switch (key) {
      case "name": return cells[0].textContent.trim().toLowerCase();
      case "pos":  return cells[1].textContent.trim().toLowerCase();
      case "avg":  return _toneOrder(cells[2]);
      case "hr":   return _toneOrder(cells[3]);
      case "r":    return _toneOrder(cells[4]);
      case "rbi":  return _toneOrder(cells[5]);
      case "sb":   return _toneOrder(cells[6]);
      case "cmp":  return parseFloat(cells[7].textContent) || 0;
      default:     return 0;
    }
  }

  function _toneOrder(cell) {
    // HOT > NEUTRAL > COLD, so HOT sorts highest in desc order.
    if (cell.querySelector(".streak-hot")) return 1;
    if (cell.querySelector(".streak-cold")) return -1;
    return 0;
  }

  window.sortStreaksTable = function (th, key, defaultDir) {
    const table = th.closest("table");
    const tbody = table.querySelector("tbody");
    const rows = Array.from(tbody.querySelectorAll("tr"));
    const currentDir = th.getAttribute("data-sort-dir");
    const dir = currentDir === "asc" ? "desc" : "asc";
    rows.sort((a, b) => {
      const av = _cellValue(a, key);
      const bv = _cellValue(b, key);
      if (av < bv) return dir === "asc" ? -1 : 1;
      if (av > bv) return dir === "asc" ? 1 : -1;
      return 0;
    });
    // Reset all headers, mark this one.
    Array.from(table.querySelectorAll("th")).forEach(h => h.removeAttribute("data-sort-dir"));
    th.setAttribute("data-sort-dir", dir);
    rows.forEach(r => tbody.appendChild(r));
  };

  window.filterFaRows = function (count) {
    const limit = parseInt(count, 10);
    const rows = document.querySelectorAll("#fa-table tbody tr");
    rows.forEach(r => {
      const rank = parseInt(r.dataset.rank, 10);
      r.style.display = rank <= limit ? "" : "none";
    });
  };

  // Apply default FA-count on load.
  document.addEventListener("DOMContentLoaded", () => {
    const sel = document.getElementById("fa-count");
    if (sel) window.filterFaRows(sel.value);
  });
})();
```

- [ ] **Step 2: Verify the page loads with the JS attached**

Run: `pytest tests/test_web/test_streaks_route.py -v`
Expected: PASS — the JS file is now referenced and Flask will serve it.

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/static/streaks.js
git commit -m "feat(web): streaks.js — client-side sort + FA-count slice"
```

---

### Task 11: CSS — chip styles + Streaks-page tables

**Files:**
- Modify: `src/fantasy_baseball/web/static/season.css`

- [ ] **Step 1: Append chip + table styles**

Append to `src/fantasy_baseball/web/static/season.css`:

```css
/* Streaks chip — used on /streaks tables and /lineup hitters tbody */
.streak-chip {
  display: inline-block;
  padding: 2px 6px;
  border-radius: 3px;
  font-size: 11px;
  font-weight: 600;
  line-height: 1.2;
  white-space: nowrap;
  font-family: var(--font-mono, "JetBrains Mono", monospace);
}
.streak-chip.streak-hot { background: #5b2a2a; color: #ffb1a8; }
.streak-chip.streak-cold { background: #1f3556; color: #9fc1ff; }
.streak-chip.streak-neutral { background: #2a2e36; color: #6a727f; }

/* Streaks page layout */
.streaks-section { margin-bottom: 32px; }
.streaks-section h2 { font-size: 18px; margin: 0 0 8px; }
.streaks-section .section-head {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  margin-bottom: 8px;
}
.streaks-table {
  width: 100%;
  border-collapse: collapse;
  font-size: 13px;
}
.streaks-table th {
  text-align: left;
  padding: 8px 10px;
  cursor: pointer;
  user-select: none;
  border-bottom: 1px solid var(--border, #2a2e36);
}
.streaks-table th[data-sort-dir="asc"]::after { content: " ▲"; opacity: 0.6; }
.streaks-table th[data-sort-dir="desc"]::after { content: " ▼"; opacity: 0.6; }
.streaks-table td { padding: 6px 10px; border-bottom: 1px solid var(--border-soft, #1f2228); }
.streaks-table tr:hover td { background: rgba(255, 255, 255, 0.02); }

/* Drivers list */
.drivers { list-style: none; padding: 0; margin: 0; font-size: 13px; }
.drivers li { padding: 4px 0; border-bottom: 1px solid var(--border-soft, #1f2228); }

/* Empty state for /streaks before first refresh */
.empty-state {
  padding: 32px;
  background: #1a1d22;
  border-radius: 4px;
  color: var(--text-secondary, #8a92a0);
  text-align: center;
}
```

The colour variables (`--border`, `--border-soft`, `--text-secondary`, `--font-mono`) may or may not exist in the existing `season.css`. Read the top of the file before writing to use the actual variable names; fall back to the literal hex values shown.

- [ ] **Step 2: Verify the dashboard still renders**

Run: `pytest tests/test_web/test_streaks_route.py tests/test_web/test_season_routes.py -v`
Expected: PASS — CSS doesn't affect tests.

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/static/season.css
git commit -m "feat(web): chip + streaks-table CSS"
```

---

### Task 12: Lineup route — inject streak_indicator into hitters context

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Modify: `src/fantasy_baseball/web/templates/season/_lineup_hitters_tbody.html`
- Modify: `tests/test_web/test_season_routes.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web/test_season_routes.py`:

```python
def test_lineup_injects_streak_chip_when_cache_present(client, kv_isolation, monkeypatch) -> None:
    """When STREAK_SCORES is in cache, the lineup hitters table renders chips."""
    # Seed a minimal streak cache.
    _seed_streak_cache_for("Roster Guy", composite=2, hot_cat="HR", prob=0.62)
    # Seed the rest of the lineup-page caches so the route renders at all.
    _seed_minimum_lineup_caches(monkeypatch, hitter_names=["Roster Guy"])

    resp = client.get("/lineup")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "streak-chip" in body
    assert "HOT · HR" in body


def test_lineup_renders_dash_chip_when_no_streak_cache(client, kv_isolation, monkeypatch) -> None:
    _seed_minimum_lineup_caches(monkeypatch, hitter_names=["Roster Guy"])
    resp = client.get("/lineup")
    assert resp.status_code == 200
    body = resp.data.decode()
    assert "streak-chip streak-neutral" in body
```

The `_seed_streak_cache_for` and `_seed_minimum_lineup_caches` helpers should be added in the test file. The former builds a one-row `Report` and writes it via `serialize_report` (lift from `test_streaks_route.py`). The latter seeds whatever the existing `/lineup` route reads (likely `CacheKey.LINEUP_OPTIMAL`, `CacheKey.ROSTER`, etc.) — read the lineup-route code to see what's required. If existing test infrastructure already does this (search test_season_routes.py for `/lineup` tests), reuse it.

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web/test_season_routes.py::test_lineup_injects_streak_chip_when_cache_present -v`
Expected: FAIL — the chip isn't injected yet.

- [ ] **Step 3: Update the lineup route to inject indicators**

In `src/fantasy_baseball/web/season_routes.py`, find the `lineup()` view function. Near where the hitter list is prepared for the template (look for whatever variable holds the rows passed to `render_template("season/lineup.html", ...)`), enrich each hitter dict:

```python
from fantasy_baseball.streaks.dashboard import build_indicator

# Inside lineup():
streak_payload = read_cache(CacheKey.STREAK_SCORES)
streak_payload = streak_payload if isinstance(streak_payload, dict) else None
for hitter in hitters_for_template:  # whatever the existing variable is
    indicator = build_indicator(hitter["name"], streak_payload)
    hitter["streak_indicator"] = indicator  # may be None
```

Place this after the hitters list is built but before `render_template`. If the existing code uses Pydantic models or dataclasses for hitters, attach the indicator via the appropriate mechanism (extra dict in the context, or a new field on the model). Keep the change minimal — match the file's existing style.

- [ ] **Step 4: Update the tbody template**

Find the existing rightmost `<td>` in `src/fantasy_baseball/web/templates/season/_lineup_hitters_tbody.html`. Add a new `<td>` after it:

```html
<td class="streak-cell">
  {% if hitter.streak_indicator %}
    <span class="streak-chip streak-{{ hitter.streak_indicator.tone }}"
          title="{{ hitter.streak_indicator.tooltip }}">
      {{ hitter.streak_indicator.label }}
    </span>
  {% else %}
    <span class="streak-chip streak-neutral">—</span>
  {% endif %}
</td>
```

If the hitters-table `<thead>` lives in `lineup.html` (not in `_lineup_hitters_tbody.html`), add a new `<th>Streak</th>` to it as the rightmost header.

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_web/test_season_routes.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py \
        src/fantasy_baseball/web/templates/season/_lineup_hitters_tbody.html \
        src/fantasy_baseball/web/templates/season/lineup.html \
        tests/test_web/test_season_routes.py
git commit -m "feat(web): inject streak chip into /lineup hitters table"
```

(Drop `lineup.html` from the `git add` if you didn't need to add a `<th>` there.)

---

### Task 13: Sidebar nav entry + HTML snapshot test

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/base.html`
- Create: `tests/test_web/test_streaks_snapshot.py`

- [ ] **Step 1: Add the nav link**

In `src/fantasy_baseball/web/templates/season/base.html`, find the existing sidebar nav (the section with `<a href="{{ url_for('lineup') }}"...>`). Insert between Lineup and Roster Audit:

```html
            <a href="{{ url_for('streaks') }}"
               class="nav-link {% if active_page == 'streaks' %}active{% endif %}">
                <span class="diamond"></span>Streaks
            </a>
```

- [ ] **Step 2: Write the snapshot test**

Create `tests/test_web/test_streaks_snapshot.py`:

```python
"""Snapshot test for the rendered /streaks page.

Catches unintentional structural drift in the HTML. The snapshot is a
plain text file under ``tests/test_web/snapshots/`` — update with the
test's `-s --snapshot-update` style if your project uses one, or just
copy the actual response into the expected file once on first failure.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.streaks.dashboard import serialize_report
from fantasy_baseball.streaks.inference import Driver, PlayerCategoryScore
from fantasy_baseball.streaks.models import StreakCategory, StreakLabel
from fantasy_baseball.streaks.reports.sunday import (
    DriverLine, Report, ReportRow,
)
from fantasy_baseball.web.season_app import create_app

SNAPSHOT_DIR = Path(__file__).parent / "snapshots"
SNAPSHOT_PATH = SNAPSHOT_DIR / "streaks.html"


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


@pytest.fixture
def kv_isolation(tmp_path, monkeypatch):
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "test.db"))
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


def _seed_canonical_report() -> None:
    score = PlayerCategoryScore(
        player_id=1, category=StreakCategory.HR, label=StreakLabel.HOT,
        probability=0.62, drivers=(Driver(feature="barrel_pct", z_score=1.8),),
        window_end=date(2026, 5, 10),
    )
    row = ReportRow(
        name="Canon Player", positions=("OF",), player_id=1, composite=1,
        scores={StreakCategory.HR: score}, max_probability=0.62,
    )
    dl = DriverLine(
        player_name="Canon Player", category=StreakCategory.HR,
        label=StreakLabel.HOT, probability=0.62,
        drivers=(Driver(feature="barrel_pct", z_score=1.8),),
    )
    rpt = Report(
        report_date=date(2026, 5, 11), window_end=date(2026, 5, 10),
        team_name="Hart of the Order", league_id=5652,
        season_set_train="2023-2025",
        roster_rows=(row,), fa_rows=(), driver_lines=(dl,), skipped=(),
    )
    kv_store.get_kv().set(redis_key(CacheKey.STREAK_SCORES), json.dumps(serialize_report(rpt)))


def test_streaks_html_snapshot(client, kv_isolation) -> None:
    _seed_canonical_report()
    resp = client.get("/streaks")
    assert resp.status_code == 200
    actual = resp.data.decode()

    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    if not SNAPSHOT_PATH.exists():
        SNAPSHOT_PATH.write_text(actual, encoding="utf-8")
        pytest.skip("Snapshot created; rerun.")
    expected = SNAPSHOT_PATH.read_text(encoding="utf-8")
    assert actual == expected, (
        "Streaks HTML drift detected. "
        f"Diff the response against {SNAPSHOT_PATH} and either fix the route/template "
        "or delete the snapshot to regenerate."
    )
```

- [ ] **Step 3: Run the snapshot test twice**

Run: `pytest tests/test_web/test_streaks_snapshot.py -v`
Expected: SKIP (snapshot created on first run).

Run again: `pytest tests/test_web/test_streaks_snapshot.py -v`
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/base.html \
        tests/test_web/test_streaks_snapshot.py \
        tests/test_web/snapshots/streaks.html
git commit -m "feat(web): sidebar Streaks nav + HTML snapshot test"
```

---

### Task 14: Final verification pass

**Files:** none changed in this task.

- [ ] **Step 1: Run the full project test suite**

Run: `pytest -v`
Expected: all PASS. Fix any failures attributable to this branch (don't touch unrelated failures — call them out in the commit message if pre-existing).

- [ ] **Step 2: Run ruff**

Run: `ruff check .`
Expected: clean.

Run: `ruff format --check .`
Expected: clean. If not, run `ruff format .` and commit the diff.

- [ ] **Step 3: Run mypy**

Run: `mypy`
Expected: clean for every file in `[tool.mypy].files`, including the new modules.

- [ ] **Step 4: Run vulture**

Run: `vulture`
Expected: no NEW findings from this branch. Pre-existing findings are acceptable; if you see one introduced by your work, fix it inline.

- [ ] **Step 5: Manual smoke test (optional but recommended)**

Run a real refresh end-to-end against the live league:

```bash
python scripts/run_season_dashboard.py
# In the dashboard, click "Refresh Data" with admin auth, wait, then visit
# /streaks and /lineup. Verify chips render and tables look right.
```

This is the final reality check — the test suite mocks Yahoo, so this is the only thing that exercises the real DuckDB + real Yahoo + real Redis path together.

- [ ] **Step 6: Commit any cleanup**

If steps 2-4 surfaced lint/format fixes:

```bash
git add -p  # stage cleanup only
git commit -m "chore(streaks): lint + format fixes"
```

---

## Self-Review (executed before handoff)

**Spec coverage:**
- `/streaks` page (Roster / FAs / Drivers, sortable, FA selector) — Tasks 9-11
- `/lineup` chip (composite tone + top hot/cold cat) — Tasks 7, 12
- New `RefreshStreaks` step / `_compute_streaks` method — Task 8
- New `CacheKey.STREAK_SCORES` — Task 1
- `streaks/pipeline.py::compute_streak_report` — Tasks 3-4
- `streaks/dashboard.py` serialization + indicator — Tasks 6-7
- `load_models_from_fits` — Task 2
- Sunday CLI refactor — Task 5
- HTML snapshot — Task 13
- Final verification (pytest, ruff, mypy, vulture) — Task 14
- Sidebar nav — Task 13

**Placeholder scan:** None of "TBD", "TODO", "implement later", "Similar to Task N", "add appropriate error handling" appears in step bodies. Every step has either concrete code, an exact command, or a specific instruction.

**Type consistency:**
- `Indicator` defined in Task 7 used in Task 12.
- `compute_streak_report` signature defined in Task 4 called consistently in Task 5 (Sunday CLI) and Task 8 (`_compute_streaks`).
- `serialize_report` defined Task 6 used Tasks 8, 9, 12, 13.
- `build_indicator` defined Task 7 used Task 12.
- `load_models_from_fits` defined Task 2 used Task 4 (`compute_streak_report`).
- `_should_refit` defined Task 3 used Task 4.
- All function names, parameter names, and dataclass fields agree across tasks.
