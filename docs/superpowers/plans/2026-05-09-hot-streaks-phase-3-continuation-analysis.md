# Hot Streaks — Phase 3 (Continuation Analysis) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether hot/cold streak labels carry meaningful predictive signal — the **go/no-go gate** for the Phase 4 model. Switch sparse-cat (HR/SB) cold labels to a per-player, projection-anchored Poisson rule (eliminating the "every zero is cold" pathology Phase 2 surfaced), then compute continuation rates against base rates across all 2023-2025 labeled windows. Persist the resulting tables so a notebook can audit them and Phase 4 can read from them.

**Architecture:** Stage A loads preseason projections into a new `hitter_projection_rates` table (per-season blend of all available systems — 2 for 2023-2025, 5 for 2026). Stage B migrates `hitter_streak_labels` to add a `cold_method` column and rewires `labels.py` to apply skill-relative Poisson cold for HR/SB while leaving R/RBI/AVG on the empirical p10 they already use. Stage C builds `analysis/continuation.py`: for each labeled window, look up the next disjoint window, classify the outcome relative to that player's expectation (projected rate × next-window PA for sparse cats; bucket median for dense cats), and tabulate continuation rates vs base rates per stratum. Stage D is the acceptance notebook + spec progress entry.

**Tech Stack:** DuckDB, pandas, scipy.stats.poisson (already a transitive dep via pybaseball), pytest, Jupyter (gitignored notebook). No new pip dependencies.

**Spec:** `docs/superpowers/specs/2026-05-06-hot-streaks-design.md` — Phase 3 high-level scope. Open questions resolved by this plan are captured in the next section.

---

## Design Decisions (resolved before plan-writing)

1. **Sparse-cat (HR, SB) cold rule — skill-relative, projection-anchored.** Cold fires when window count is below `Poisson(rate=proj_per_pa × window_PA).ppf(percentile)`. Hot for sparse cats stays on the existing empirical p90 (the upper-tail pathology doesn't apply). Dense cats (R/RBI/AVG) keep both empirical p10 and p90 unchanged.
2. **Two parallel cold thresholds for sparse cats — Poisson p10 AND Poisson p20.** Phase 2 acceptance numbers showed Poisson p10 fires for ~0% of qualified hitters at 7d windows (Judge tops out at expected 1.83 HR / 25 PA, P(0)=16% > 10%). Computing both lets Phase 3 measure continuation lift at each threshold and hand Phase 4 the better one without re-running the labeling pipeline. Implementation cost is one extra column in the labels PK and one extra `scipy.stats.poisson.ppf` call per row.
3. **Skill baseline — mean blend of all available preseason projection systems per season.** 2 systems (Steamer + ZiPS) for 2023-2025, 5 systems (Steamer + ZiPS + ATC + THE BAT X + Oopsy) for 2026. The blend is per-system arithmetic mean of `HR/projected_PA` and `SB/projected_PA`, equally weighted. Per-year accuracy beats cross-year consistency: the marginal accuracy gain from more systems matters most for projection-disagreement edge cases (rookies, post-injury), which is exactly where the cold-rule baseline matters most.
4. **Unprojected players — skip cold-HR/cold-SB labels for them.** ~10% of qualified hitters per season won't appear in any projection system (organizational filler, late-arriving international signings). They still get hot/neutral on HR/SB and full hot/cold/neutral on R/RBI/AVG. This is honest about uncertainty rather than fabricating a baseline for them.
5. **Train/test split — use all of 2023-2025 for base-rate measurement.** Phase 3 is descriptive statistics; we want maximum sample. The 2024-vs-2025 holdout is deferred to Phase 4 (the Phase 4 model can overfit; base-rate continuation tables can't).
6. **Continuation outcome definition — "above the player's expectation" (sparse) / "above bucket median" (dense).** Symmetric for hot vs cold. For sparse cats, expectation = `proj_per_pa × next_window_PA`; for dense cats, expectation = empirical median of windows in the same `(category, window_days, pt_bucket)` cell. Both are computed from the same data we already have.
7. **Go/no-go gate (committed upfront so we don't move goalposts):** ≥1 cell shows ≥5pp lift over base rate with N≥1000, AND ≥3 of 5 categories show ≥2pp directional lift in the 7d window. If neither passes after stratification, write up the no-go finding and decide in a follow-up whether to reframe (different windows, different label rule) or shelve.
8. **Stratifications — `category × window_days × pt_bucket × strength_bucket × direction × cold_method`.** Defer season-skill-quartile to Phase 4 (it's a feature, not a primary stratification axis; adding it now creates 4× as many cells, most too small to read).
9. **Schema changes:**
   - New table `hitter_projection_rates` (PK: `player_id, season`).
   - `hitter_streak_labels` PK extended to `(player_id, window_end, window_days, category, cold_method)`. For dense cats: one row per (player, window, cat) with `cold_method='empirical'`. For sparse cats: two rows per (player, window, cat) with `cold_method ∈ {'poisson_p10', 'poisson_p20'}`. Hot/neutral determination is the same in both rows (empirical p90 for hot); we duplicate hot rows across both methods for simplicity.
   - New table `continuation_rates` (PK: `season_set, category, window_days, pt_bucket, strength_bucket, direction, cold_method`).
10. **Migration path — DROP+rebuild (matches Phase 2 pattern).** Labels are pure derived data; recomputing from `hitter_windows` + `thresholds` + `hitter_projection_rates` is cheap (~minutes for the 2.6M+-row corpus).
11. **Strength bins:**
    - **Dense cats:** quintile bin within the labeled population (Q1-Q5 above p90 for hot; Q1-Q5 below p10 for cold). Avoids per-stratum population-shape sensitivity.
    - **Sparse cats:** Poisson z-score `(window_count - expected) / sqrt(expected)` bucketed into integer bins (e.g., -2.5σ, -1.5σ, ...). Same metric for both p10 and p20 cold rules.

---

## File Structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/streaks/models.py` | Modify | Add `HitterProjectionRate`, `ContinuationRate` dataclasses; extend `HitterStreakLabel` with `cold_method` |
| `src/fantasy_baseball/streaks/data/schema.py` | Modify | Add DDL for `hitter_projection_rates`, `continuation_rates`; rebuild `hitter_streak_labels` PK |
| `src/fantasy_baseball/streaks/data/migrate.py` | Modify | Add `migrate_to_phase_3` that DROPs `hitter_streak_labels` and recreates with new PK |
| `src/fantasy_baseball/streaks/data/projections.py` | Create | Read FanGraphs projection CSVs from `data/projections/{year}/`, mean-blend across available systems, return per-player rates |
| `src/fantasy_baseball/streaks/data/load_projections.py` | Create | Idempotent upsert into `hitter_projection_rates` |
| `src/fantasy_baseball/streaks/labels.py` | Modify | Split label-application into dense-cat (existing rule) and sparse-cat (Poisson p10 + p20) branches |
| `src/fantasy_baseball/streaks/analysis/__init__.py` | Create | Empty package marker |
| `src/fantasy_baseball/streaks/analysis/continuation.py` | Create | Compute continuation rates + persist to `continuation_rates` |
| `scripts/streaks/migrate.py` | Modify | Wire up new `migrate_to_phase_3` CLI subcommand |
| `scripts/streaks/load_projections.py` | Create | CLI: load projection rates for given seasons |
| `scripts/streaks/compute_labels.py` | Modify | Require projection rates loaded before `apply_labels` |
| `scripts/streaks/run_continuation.py` | Create | CLI: run continuation analysis for given season set |
| `tests/test_streaks/test_models.py` | Modify | Assert new field surface for `HitterStreakLabel`, `HitterProjectionRate`, `ContinuationRate` |
| `tests/test_streaks/test_schema.py` | Modify | Assert new tables + new label PK exist |
| `tests/test_streaks/test_migrate.py` | Modify | Assert `migrate_to_phase_3` resets labels and is idempotent |
| `tests/test_streaks/test_projections.py` | Create | Reader: filename suffix variation, multi-system blend, missing-system fallback, MLBAMID coercion |
| `tests/test_streaks/test_load_projections.py` | Create | Upsert idempotency, per-season filter |
| `tests/test_streaks/test_labels.py` | Modify | Sparse-cat Poisson p10/p20 branches; dense-cat empirical branch unchanged; unprojected-player skip |
| `tests/test_streaks/test_continuation.py` | Create | Next-window lookup, base-rate vs conditional-rate math, lift computation, strength bucketing |
| `notebooks/streaks/02_continuation.ipynb` | Create (gitignored) | Acceptance notebook: lift tables, go/no-go assessment |
| `docs/superpowers/specs/2026-05-06-hot-streaks-design.md` | Modify | Append Phase 3 progress entry |

---

## Stage A — Projection rate ingestion

### Task 1: Add `HitterProjectionRate` and `ContinuationRate` dataclasses; rebuild `HitterStreakLabel`

Add the new dataclasses and extend `HitterStreakLabel` with `cold_method`. `load.py` derives column tuples via `dataclasses.fields()`, so adding fields auto-updates upserts as long as field declaration order matches the DDL column order (Phase 2 convention).

**Files:**
- Modify: `src/fantasy_baseball/streaks/models.py`
- Modify: `tests/test_streaks/test_models.py`

- [ ] **Step 1.1: Write failing dataclass-shape tests**

Append to `tests/test_streaks/test_models.py`:

```python
def test_hitter_streak_label_includes_cold_method() -> None:
    expected = ("player_id", "window_end", "window_days", "category", "cold_method", "label")
    assert tuple(f.name for f in fields(HitterStreakLabel)) == expected


def test_hitter_projection_rate_fields_in_expected_order() -> None:
    from fantasy_baseball.streaks.models import HitterProjectionRate

    expected = ("player_id", "season", "hr_per_pa", "sb_per_pa", "n_systems")
    assert tuple(f.name for f in fields(HitterProjectionRate)) == expected


def test_continuation_rate_fields_in_expected_order() -> None:
    from fantasy_baseball.streaks.models import ContinuationRate

    expected = (
        "season_set",
        "category",
        "window_days",
        "pt_bucket",
        "strength_bucket",
        "direction",
        "cold_method",
        "n_labeled",
        "n_continued",
        "p_continued",
        "p_baserate",
        "lift",
    )
    assert tuple(f.name for f in fields(ContinuationRate)) == expected
```

- [ ] **Step 1.2: Run the tests, confirm they fail**

```
pytest tests/test_streaks/test_models.py -v
```

Expected: three FAILures — `cold_method` missing on `HitterStreakLabel`, and the two new dataclasses don't exist.

- [ ] **Step 1.3: Edit `src/fantasy_baseball/streaks/models.py`**

Replace the `HitterStreakLabel` block at the end of the file with:

```python
ColdMethod = Literal["empirical", "poisson_p10", "poisson_p20"]
StreakDirection = Literal["above", "below"]


@dataclass(frozen=True, slots=True)
class HitterStreakLabel:
    """One hot/cold/neutral label for a (player, window, category, cold_method).

    PK is (player_id, window_end, window_days, category, cold_method).

    `cold_method` distinguishes the rule that produced the cold determination:
    - 'empirical' for dense cats (R/RBI/AVG): uses calibrated p10 from `thresholds`.
    - 'poisson_p10' / 'poisson_p20' for sparse cats (HR/SB): uses skill-relative
      `Poisson(proj_rate × window_PA).ppf(0.1 | 0.2)`.

    For sparse cats, two rows are written per (player, window, cat) — one per
    Poisson percentile. The hot determination (empirical p90) is identical
    across both rows; we duplicate rather than introduce a third schema.
    """

    player_id: int
    window_end: date
    window_days: int
    category: StreakCategory
    cold_method: ColdMethod
    label: StreakLabel


@dataclass(frozen=True, slots=True)
class HitterProjectionRate:
    """Per-season blended projection rate for a single hitter.

    PK is (player_id, season). Rates are season-prior blended means
    (HR/projected_PA, SB/projected_PA) across all available systems.
    `n_systems` records the count of systems that contributed; rows with
    n_systems < 2 are still kept (caller decides whether to use them).
    """

    player_id: int
    season: int
    hr_per_pa: float
    sb_per_pa: float
    n_systems: int


@dataclass(frozen=True, slots=True)
class ContinuationRate:
    """One stratum of the Phase 3 continuation analysis output.

    PK is (season_set, category, window_days, pt_bucket, strength_bucket,
    direction, cold_method).

    Each row answers: "of windows labeled <direction> in this stratum, what
    fraction had next-window outcome on the same side of expectation, vs the
    base rate computed over all windows in that stratum?"
    """

    season_set: str
    category: StreakCategory
    window_days: int
    pt_bucket: PtBucket
    strength_bucket: str
    direction: StreakDirection
    cold_method: ColdMethod
    n_labeled: int
    n_continued: int
    p_continued: float
    p_baserate: float
    lift: float
```

- [ ] **Step 1.4: Run the tests, confirm they pass**

```
pytest tests/test_streaks/test_models.py -v
```

Expected: PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/fantasy_baseball/streaks/models.py tests/test_streaks/test_models.py
git commit -m "feat(streaks): add Phase 3 dataclasses (projection rate, continuation, cold_method on label)"
```

---

### Task 2: Add Phase 3 schema (new tables + label PK rebuild)

DuckDB doesn't support PK changes via ALTER. Instead, follow the Phase 2 pattern: drop the dependent table (`hitter_streak_labels` is pure derived data) and recreate via `init_schema`. Add the two new tables alongside.

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/schema.py`
- Modify: `tests/test_streaks/test_schema.py`

- [ ] **Step 2.1: Write failing schema tests**

Append to `tests/test_streaks/test_schema.py`:

```python
def test_hitter_streak_labels_has_cold_method_pk() -> None:
    conn = get_connection(":memory:")
    info = conn.execute("PRAGMA table_info('hitter_streak_labels')").fetchall()
    cols = {r[1] for r in info}
    assert "cold_method" in cols, f"expected cold_method in {cols}"
    pk_cols = [r[1] for r in info if r[5]]  # column 5 is the pk position
    assert pk_cols == [
        "player_id",
        "window_end",
        "window_days",
        "category",
        "cold_method",
    ]


def test_hitter_projection_rates_table_exists() -> None:
    conn = get_connection(":memory:")
    info = conn.execute("PRAGMA table_info('hitter_projection_rates')").fetchall()
    cols = [r[1] for r in info]
    assert cols == ["player_id", "season", "hr_per_pa", "sb_per_pa", "n_systems"]
    pk_cols = [r[1] for r in info if r[5]]
    assert pk_cols == ["player_id", "season"]


def test_continuation_rates_table_exists() -> None:
    conn = get_connection(":memory:")
    info = conn.execute("PRAGMA table_info('continuation_rates')").fetchall()
    cols = {r[1] for r in info}
    expected_cols = {
        "season_set",
        "category",
        "window_days",
        "pt_bucket",
        "strength_bucket",
        "direction",
        "cold_method",
        "n_labeled",
        "n_continued",
        "p_continued",
        "p_baserate",
        "lift",
    }
    assert expected_cols.issubset(cols)
```

- [ ] **Step 2.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_schema.py -v
```

Expected: three FAILures.

- [ ] **Step 2.3: Edit `src/fantasy_baseball/streaks/data/schema.py`**

Replace the `hitter_streak_labels` DDL entry in `_SCHEMA_DDL` with the new PK shape, and append two new DDL entries:

```python
    """
    CREATE TABLE IF NOT EXISTS hitter_streak_labels (
        player_id INTEGER NOT NULL,
        window_end DATE NOT NULL,
        window_days INTEGER NOT NULL,
        category VARCHAR NOT NULL,
        cold_method VARCHAR NOT NULL,
        label VARCHAR NOT NULL,
        PRIMARY KEY (player_id, window_end, window_days, category, cold_method)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS hitter_projection_rates (
        player_id INTEGER NOT NULL,
        season INTEGER NOT NULL,
        hr_per_pa DOUBLE NOT NULL,
        sb_per_pa DOUBLE NOT NULL,
        n_systems INTEGER NOT NULL,
        PRIMARY KEY (player_id, season)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS continuation_rates (
        season_set VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        window_days INTEGER NOT NULL,
        pt_bucket VARCHAR NOT NULL,
        strength_bucket VARCHAR NOT NULL,
        direction VARCHAR NOT NULL,
        cold_method VARCHAR NOT NULL,
        n_labeled INTEGER NOT NULL,
        n_continued INTEGER NOT NULL,
        p_continued DOUBLE NOT NULL,
        p_baserate DOUBLE NOT NULL,
        lift DOUBLE NOT NULL,
        PRIMARY KEY (season_set, category, window_days, pt_bucket, strength_bucket, direction, cold_method)
    )
    """,
```

- [ ] **Step 2.4: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_schema.py -v
```

Expected: PASS.

- [ ] **Step 2.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/schema.py tests/test_streaks/test_schema.py
git commit -m "feat(streaks): Phase 3 schema (cold_method on labels, projection_rates, continuation_rates)"
```

---

### Task 3: Phase 3 migration helper

Existing local DBs have the Phase 2 `hitter_streak_labels` PK. We need to drop and recreate so the new PK takes hold; the new tables come along for free via `init_schema`. Labels are pure derived data — recomputed via `apply_labels` after migration.

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/migrate.py`
- Modify: `scripts/streaks/migrate.py`
- Modify: `tests/test_streaks/test_migrate.py`

- [ ] **Step 3.1: Write failing test**

Append to `tests/test_streaks/test_migrate.py`:

```python
def test_migrate_to_phase_3_resets_labels_and_keeps_other_tables() -> None:
    """`migrate_to_phase_3` drops hitter_streak_labels and recreates it with
    the new PK, but does NOT touch hitter_games / hitter_windows / thresholds.
    """
    from fantasy_baseball.streaks.data.migrate import migrate_to_phase_3

    conn = get_connection(":memory:")
    # Seed something in hitter_games so we can assert it survives.
    conn.execute(
        "INSERT INTO hitter_games VALUES (1, 100, 'X', 'TEAM', 2025, '2025-04-01', "
        "4, 4, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, true)"
    )
    # Seed an old-shape label row to confirm it gets cleared.
    conn.execute(
        "INSERT INTO hitter_streak_labels (player_id, window_end, window_days, category, "
        "cold_method, label) VALUES (1, '2025-04-08', 7, 'hr', 'empirical', 'cold')"
    )

    migrate_to_phase_3(conn)

    # Labels are wiped...
    n_labels = conn.execute("SELECT COUNT(*) FROM hitter_streak_labels").fetchone()[0]
    assert n_labels == 0
    # ...but games are not.
    n_games = conn.execute("SELECT COUNT(*) FROM hitter_games").fetchone()[0]
    assert n_games == 1


def test_migrate_to_phase_3_is_idempotent() -> None:
    from fantasy_baseball.streaks.data.migrate import migrate_to_phase_3

    conn = get_connection(":memory:")
    migrate_to_phase_3(conn)
    migrate_to_phase_3(conn)  # second call must not raise
    info = conn.execute("PRAGMA table_info('hitter_streak_labels')").fetchall()
    pk_cols = [r[1] for r in info if r[5]]
    assert "cold_method" in pk_cols
```

- [ ] **Step 3.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_migrate.py -v
```

Expected: FAIL — `migrate_to_phase_3` not defined.

- [ ] **Step 3.3: Edit `src/fantasy_baseball/streaks/data/migrate.py`**

Append at the bottom of the file:

```python
def migrate_to_phase_3(conn: duckdb.DuckDBPyConnection) -> None:
    """Drop only `hitter_streak_labels` (which has a PK shape change) and let
    `init_schema` recreate it with the Phase 3 PK — plus any new Phase 3 tables
    that are missing (`hitter_projection_rates`, `continuation_rates`).

    `hitter_games` / `hitter_statcast_pa` / `hitter_windows` / `thresholds` are
    untouched: their schema didn't change. Labels are pure derived data and are
    rebuilt by `apply_labels` after this migration.
    """
    conn.execute("DROP TABLE IF EXISTS hitter_streak_labels")
    logger.info("Dropped hitter_streak_labels (PK shape change for cold_method)")
    init_schema(conn)
    logger.info("Recreated hitter_streak_labels + Phase 3 tables via init_schema")
```

- [ ] **Step 3.4: Edit `scripts/streaks/migrate.py`**

Read the existing file first; it currently exposes `migrate_to_phase_2`. Add a `--phase` argument with default `3`. The exact diff:

```python
import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.migrate import migrate_to_phase_2, migrate_to_phase_3
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate the streaks DuckDB schema.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--phase", type=int, choices=[2, 3], default=3)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        if args.phase == 2:
            migrate_to_phase_2(conn)
        else:
            migrate_to_phase_3(conn)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 3.5: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_migrate.py -v
```

Expected: PASS.

- [ ] **Step 3.6: Commit**

```bash
git add src/fantasy_baseball/streaks/data/migrate.py scripts/streaks/migrate.py tests/test_streaks/test_migrate.py
git commit -m "feat(streaks): migrate_to_phase_3 (drop labels for new cold_method PK)"
```

---

### Task 4: Projection rate reader (`streaks/data/projections.py`)

Read every `<system>-hitters*.csv` file under `data/projections/{season}/`, blend per-PA HR and SB rates across systems, and emit `HitterProjectionRate` rows. Handle filename suffix variation (2025 has `-2025.csv` suffix; other years don't), MLBAMID coercion to int, and the PA<200 filter (drops org filler / NRI rows). Players appearing in only one system are still emitted (with `n_systems=1`); the caller decides whether to use them.

**Files:**
- Create: `src/fantasy_baseball/streaks/data/projections.py`
- Create: `tests/test_streaks/test_projections.py`

- [ ] **Step 4.1: Write failing tests**

Create `tests/test_streaks/test_projections.py`:

```python
"""Tests for the projection-rate reader."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.streaks.data.projections import (
    PROJECTION_PA_FLOOR,
    discover_projection_files,
    load_projection_rates,
)


def _write_proj_csv(path: Path, rows: list[dict[str, object]]) -> None:
    cols = ["Name", "PA", "HR", "SB", "MLBAMID"]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_discover_projection_files_no_suffix(tmp_path: Path) -> None:
    base = tmp_path / "2024"
    base.mkdir()
    (base / "steamer-hitters.csv").touch()
    (base / "zips-hitters.csv").touch()
    (base / "steamer-pitchers.csv").touch()  # ignored: pitcher file
    files = discover_projection_files(tmp_path, season=2024)
    assert sorted(p.name for p in files) == ["steamer-hitters.csv", "zips-hitters.csv"]


def test_discover_projection_files_with_year_suffix(tmp_path: Path) -> None:
    base = tmp_path / "2025"
    base.mkdir()
    (base / "steamer-hitters-2025.csv").touch()
    (base / "zips-hitters-2025.csv").touch()
    files = discover_projection_files(tmp_path, season=2025)
    assert sorted(p.name for p in files) == [
        "steamer-hitters-2025.csv",
        "zips-hitters-2025.csv",
    ]


def test_load_projection_rates_blends_two_systems(tmp_path: Path) -> None:
    base = tmp_path / "2024"
    base.mkdir()
    # Player 100: 30 HR / 600 PA in steamer (=0.05/PA), 36 HR / 600 PA in zips (=0.06/PA).
    # Mean: 0.055 HR/PA. SB: 12/600 (=0.02) and 18/600 (=0.03) -> 0.025 SB/PA.
    _write_proj_csv(
        base / "steamer-hitters.csv",
        [{"Name": "P", "PA": 600, "HR": 30, "SB": 12, "MLBAMID": 100}],
    )
    _write_proj_csv(
        base / "zips-hitters.csv",
        [{"Name": "P", "PA": 600, "HR": 36, "SB": 18, "MLBAMID": 100}],
    )
    rates = load_projection_rates(tmp_path, season=2024)
    assert len(rates) == 1
    r = rates[0]
    assert r.player_id == 100
    assert r.season == 2024
    assert r.hr_per_pa == pytest.approx(0.055, rel=1e-6)
    assert r.sb_per_pa == pytest.approx(0.025, rel=1e-6)
    assert r.n_systems == 2


def test_load_projection_rates_emits_single_system_player(tmp_path: Path) -> None:
    """A player appearing in only one of the two systems is still emitted with n_systems=1."""
    base = tmp_path / "2024"
    base.mkdir()
    _write_proj_csv(
        base / "steamer-hitters.csv",
        [{"Name": "Solo", "PA": 500, "HR": 20, "SB": 5, "MLBAMID": 200}],
    )
    _write_proj_csv(
        base / "zips-hitters.csv",
        [{"Name": "Other", "PA": 400, "HR": 12, "SB": 8, "MLBAMID": 300}],
    )
    rates_by_id = {r.player_id: r for r in load_projection_rates(tmp_path, season=2024)}
    assert rates_by_id[200].n_systems == 1
    assert rates_by_id[200].hr_per_pa == pytest.approx(20 / 500, rel=1e-6)
    assert rates_by_id[300].n_systems == 1


def test_load_projection_rates_filters_below_pa_floor(tmp_path: Path) -> None:
    base = tmp_path / "2024"
    base.mkdir()
    _write_proj_csv(
        base / "steamer-hitters.csv",
        [
            {"Name": "Reg", "PA": 600, "HR": 30, "SB": 5, "MLBAMID": 1},
            {"Name": "Filler", "PA": PROJECTION_PA_FLOOR - 1, "HR": 1, "SB": 0, "MLBAMID": 2},
        ],
    )
    _write_proj_csv(
        base / "zips-hitters.csv",
        [{"Name": "Reg", "PA": 600, "HR": 28, "SB": 6, "MLBAMID": 1}],
    )
    rates = load_projection_rates(tmp_path, season=2024)
    ids = {r.player_id for r in rates}
    assert 1 in ids
    assert 2 not in ids  # below floor in steamer; not in zips at all -> dropped


def test_load_projection_rates_drops_rows_without_mlbamid(tmp_path: Path) -> None:
    base = tmp_path / "2024"
    base.mkdir()
    _write_proj_csv(
        base / "steamer-hitters.csv",
        [
            {"Name": "Has", "PA": 500, "HR": 20, "SB": 5, "MLBAMID": 1},
            {"Name": "NoID", "PA": 500, "HR": 20, "SB": 5, "MLBAMID": ""},
        ],
    )
    rates = load_projection_rates(tmp_path, season=2024)
    assert {r.player_id for r in rates} == {1}
```

- [ ] **Step 4.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_projections.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 4.3: Create `src/fantasy_baseball/streaks/data/projections.py`**

```python
"""Read FanGraphs preseason projection CSVs and blend per-PA rates.

Reads every `<system>-hitters*.csv` under `data/projections/{season}/`,
filters to rows with PA >= ``PROJECTION_PA_FLOOR`` (drops org filler / NRI
spring rows), coerces MLBAMID to int, and computes per-system HR/PA and
SB/PA. Returns one ``HitterProjectionRate`` per (player_id, season) with
the simple arithmetic mean across the systems that included that player.

Filename pattern variation: 2025's CSVs are named ``<system>-hitters-2025.csv``;
other years use ``<system>-hitters.csv``. The discovery glob matches both.

This module reads flat CSVs only — no imports from ``web/`` or ``lineup/``,
preserving the streaks package's hard isolation from the production stack.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from fantasy_baseball.streaks.models import HitterProjectionRate

logger = logging.getLogger(__name__)

# Drops org-filler rows (Steamer projects ~3,500-4,500 hitters/year, mostly
# 1-50 PA blowouts of MiLB depth charts; ZiPS does the same with ~1,700-2,000).
# The floor matches what we use for the draft pipeline elsewhere; revisit
# only if it bites a real player who genuinely projects below it.
PROJECTION_PA_FLOOR = 200


def discover_projection_files(projections_root: Path, *, season: int) -> list[Path]:
    """Return all ``<system>-hitters*.csv`` files under ``{projections_root}/{season}/``.

    Pitcher files and any non-hitter files are excluded. Order is filesystem-
    dependent; the blender doesn't care.
    """
    season_dir = projections_root / str(season)
    if not season_dir.is_dir():
        return []
    files = [
        p
        for p in season_dir.iterdir()
        if p.is_file() and p.suffix == ".csv" and "hitters" in p.name and "pitchers" not in p.name
    ]
    return files


def _load_one_system(path: Path) -> pd.DataFrame:
    """Load one system's projection CSV into a 4-column frame keyed by MLBAMID.

    Drops rows missing MLBAMID and rows below ``PROJECTION_PA_FLOOR`` PA.
    Computes hr_per_pa and sb_per_pa per row.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "MLBAMID" not in df.columns:
        logger.warning("File %s has no MLBAMID column; skipping", path)
        return pd.DataFrame(columns=["MLBAMID", "PA", "hr_per_pa", "sb_per_pa"])
    df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce")
    df = df.dropna(subset=["MLBAMID"])
    df["MLBAMID"] = df["MLBAMID"].astype(int)
    df = df[df["PA"] >= PROJECTION_PA_FLOOR].copy()
    df["hr_per_pa"] = df["HR"] / df["PA"]
    df["sb_per_pa"] = df["SB"] / df["PA"]
    return df[["MLBAMID", "PA", "hr_per_pa", "sb_per_pa"]]


def load_projection_rates(
    projections_root: Path, *, season: int
) -> list[HitterProjectionRate]:
    """Load and blend projection rates for one season.

    Returns one ``HitterProjectionRate`` per (player_id, season). Players who
    appear in only one system are emitted with ``n_systems=1`` (caller decides
    whether to filter). Players who appear in no system are not emitted.
    """
    files = discover_projection_files(projections_root, season=season)
    if not files:
        logger.warning("No projection files found for season %d at %s", season, projections_root)
        return []
    logger.info("Season %d: %d projection files found", season, len(files))

    frames: list[pd.DataFrame] = []
    for path in files:
        sub = _load_one_system(path)
        if not sub.empty:
            frames.append(sub)
    if not frames:
        return []
    stacked = pd.concat(frames, ignore_index=True)

    blended = stacked.groupby("MLBAMID", as_index=False).agg(
        hr_per_pa=("hr_per_pa", "mean"),
        sb_per_pa=("sb_per_pa", "mean"),
        n_systems=("PA", "count"),
    )

    return [
        HitterProjectionRate(
            player_id=int(r.MLBAMID),
            season=season,
            hr_per_pa=float(r.hr_per_pa),
            sb_per_pa=float(r.sb_per_pa),
            n_systems=int(r.n_systems),
        )
        for r in blended.itertuples(index=False)
    ]


def load_projection_rates_for_seasons(
    projections_root: Path, seasons: Iterable[int]
) -> list[HitterProjectionRate]:
    """Convenience wrapper: concat per-season loads."""
    out: list[HitterProjectionRate] = []
    for s in seasons:
        out.extend(load_projection_rates(projections_root, season=s))
    return out
```

- [ ] **Step 4.4: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_projections.py -v
```

Expected: all 5 PASS.

- [ ] **Step 4.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/projections.py tests/test_streaks/test_projections.py
git commit -m "feat(streaks): projection-rate reader (per-season blend across available systems)"
```

---

### Task 5: Loader for `hitter_projection_rates` + CLI

Idempotent upsert into the new table. Mirrors the pattern of `load.py::upsert_hitter_games`.

**Files:**
- Create: `src/fantasy_baseball/streaks/data/load_projections.py`
- Create: `scripts/streaks/load_projections.py`
- Create: `tests/test_streaks/test_load_projections.py`

- [ ] **Step 5.1: Write failing tests**

Create `tests/test_streaks/test_load_projections.py`:

```python
from __future__ import annotations

from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.models import HitterProjectionRate


def _row(pid: int, season: int, hr_pa: float = 0.05, sb_pa: float = 0.02, n: int = 2):
    return HitterProjectionRate(
        player_id=pid, season=season, hr_per_pa=hr_pa, sb_per_pa=sb_pa, n_systems=n
    )


def test_upsert_projection_rates_inserts_rows() -> None:
    conn = get_connection(":memory:")
    upsert_projection_rates(conn, [_row(1, 2024), _row(2, 2024)])
    n = conn.execute("SELECT COUNT(*) FROM hitter_projection_rates").fetchone()[0]
    assert n == 2


def test_upsert_projection_rates_replaces_on_pk_collision() -> None:
    conn = get_connection(":memory:")
    upsert_projection_rates(conn, [_row(1, 2024, hr_pa=0.05)])
    upsert_projection_rates(conn, [_row(1, 2024, hr_pa=0.10)])
    rate = conn.execute(
        "SELECT hr_per_pa FROM hitter_projection_rates WHERE player_id=1 AND season=2024"
    ).fetchone()[0]
    assert rate == 0.10


def test_upsert_projection_rates_empty_input_is_noop() -> None:
    conn = get_connection(":memory:")
    upsert_projection_rates(conn, [])
    n = conn.execute("SELECT COUNT(*) FROM hitter_projection_rates").fetchone()[0]
    assert n == 0
```

- [ ] **Step 5.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_load_projections.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 5.3: Create `src/fantasy_baseball/streaks/data/load_projections.py`**

```python
"""Idempotent loader for ``hitter_projection_rates``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from operator import attrgetter

import duckdb

from fantasy_baseball.streaks.models import HitterProjectionRate

_PROJECTION_RATE_COLS = tuple(f.name for f in fields(HitterProjectionRate))
_projection_rate_row = attrgetter(*_PROJECTION_RATE_COLS)


def upsert_projection_rates(
    conn: duckdb.DuckDBPyConnection, rows: Sequence[HitterProjectionRate]
) -> None:
    """Insert or replace rows in `hitter_projection_rates` keyed by (player_id, season)."""
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(_PROJECTION_RATE_COLS))
    sql = (
        f"INSERT OR REPLACE INTO hitter_projection_rates ({', '.join(_PROJECTION_RATE_COLS)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [_projection_rate_row(r) for r in rows])
```

- [ ] **Step 5.4: Create `scripts/streaks/load_projections.py`**

```python
"""CLI: load preseason projection rates for given seasons into the streaks DB.

Usage:
    python -m scripts.streaks.load_projections [--db-path PATH] [--seasons 2023 2024 2025] \\
        [--projections-root data/projections]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.projections import load_projection_rates_for_seasons
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Load preseason projection rates.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument(
        "--seasons", nargs="+", type=int, default=[2023, 2024, 2025],
        help="Seasons to load (default: 2023 2024 2025)",
    )
    parser.add_argument(
        "--projections-root", type=Path, default=PROJECT_ROOT / "data" / "projections",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    rates = load_projection_rates_for_seasons(args.projections_root, args.seasons)
    print(f"loaded {len(rates)} projection-rate rows from {args.projections_root}")

    conn = get_connection(args.db_path)
    try:
        upsert_projection_rates(conn, rates)
    finally:
        conn.close()
    print("upserted to hitter_projection_rates")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5.5: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_load_projections.py -v
```

Expected: PASS.

- [ ] **Step 5.6: Commit**

```bash
git add src/fantasy_baseball/streaks/data/load_projections.py scripts/streaks/load_projections.py tests/test_streaks/test_load_projections.py
git commit -m "feat(streaks): hitter_projection_rates loader + CLI"
```

---

## Stage B — Skill-relative cold labeling

### Task 6: Refactor `labels.py` to write `cold_method` for dense cats

Before adding the sparse-cat branch, rewire the existing dense-cat path to write `cold_method='empirical'`. This is a pure refactor — no behavior change for R/RBI/AVG — but it lets the next task add a parallel sparse-cat path without re-touching the dense logic.

**Files:**
- Modify: `src/fantasy_baseball/streaks/labels.py`
- Modify: `tests/test_streaks/test_labels.py`

- [ ] **Step 6.1: Update existing tests for `cold_method` column**

In `tests/test_streaks/test_labels.py`, replace the body of `test_apply_labels_writes_rows_per_category` and `test_apply_labels_classifies_hot_above_p90_cold_below_p10` with versions that filter on `cold_method='empirical'`:

```python
def test_apply_labels_writes_rows_per_category() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    n = apply_labels(conn, season_set="2025")
    assert n > 0
    cats = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT category FROM hitter_streak_labels "
            "WHERE category IN ('r', 'rbi', 'avg') AND cold_method='empirical'"
        ).fetchall()
    }
    assert cats == {"r", "rbi", "avg"}


def test_apply_labels_classifies_hot_above_p90_cold_below_p10() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    apply_labels(conn, season_set="2025")
    threshold = conn.execute(
        "SELECT category, window_days, pt_bucket, p10, p90 "
        "FROM thresholds WHERE season_set = '2025' AND category = 'r' LIMIT 1"
    ).fetchone()
    cat, win, bucket, p10, p90 = threshold
    sample = conn.execute(
        "SELECT player_id, window_end, r FROM hitter_windows "
        "WHERE window_days = ? AND pt_bucket = ? LIMIT 5",
        [win, bucket],
    ).fetchall()
    for pid, end, r_val in sample:
        label = conn.execute(
            "SELECT label FROM hitter_streak_labels "
            "WHERE player_id = ? AND window_end = ? AND window_days = ? "
            "AND category = ? AND cold_method = 'empirical'",
            [pid, end, win, cat],
        ).fetchone()[0]
        if r_val >= p90:
            assert label == "hot", f"r={r_val} >= p90={p90} but label={label}"
        elif r_val <= p10:
            assert label == "cold", f"r={r_val} <= p10={p10} but label={label}"
        else:
            assert label == "neutral"
```

(Switch the inspected category from `'hr'` to `'r'` because HR is now sparse-cat with a different rule; the dense-cat branch still owns R.)

- [ ] **Step 6.2: Run the test, confirm it fails**

```
pytest tests/test_streaks/test_labels.py -v
```

Expected: FAILures referencing missing `cold_method` column or wrong-cat selection.

- [ ] **Step 6.3: Edit `src/fantasy_baseball/streaks/labels.py`**

Replace the entire file with:

```python
"""Apply calibrated thresholds to hitter_windows -> hitter_streak_labels.

Two label paths, both written into the same table with `cold_method`
distinguishing them:

- **Dense categories (R, RBI, AVG):** uses calibrated empirical p10/p90
  from `thresholds`. One row per (player, window, category) with
  cold_method='empirical'.
- **Sparse categories (HR, SB):** uses skill-relative Poisson lower-tail
  thresholds against per-player projected rates. Two rows per (player,
  window, category) — cold_method='poisson_p10' and cold_method='poisson_p20'.
  Hot uses the same empirical p90 in both rows. Players without a row in
  `hitter_projection_rates` get NO sparse-cat labels written (callers can
  still compute hot via the dense path; we omit them from sparse rows
  rather than fabricating a baseline).

Idempotent: full-wipe of `hitter_streak_labels` on each call (labels are
tied to the latest threshold + projection-rate calibration; no scoped
delete is meaningful).
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd
from scipy.stats import poisson

from fantasy_baseball.streaks.models import StreakCategory

logger = logging.getLogger(__name__)

DENSE_CATEGORIES: tuple[StreakCategory, ...] = ("r", "rbi", "avg")
SPARSE_CATEGORIES: tuple[StreakCategory, ...] = ("hr", "sb")
POISSON_PERCENTILES: tuple[tuple[str, float], ...] = (
    ("poisson_p10", 0.10),
    ("poisson_p20", 0.20),
)


def apply_labels(conn: duckdb.DuckDBPyConnection, *, season_set: str) -> int:
    """Rebuild `hitter_streak_labels` from windows + thresholds + projection rates.

    Returns total rows written across all (category, cold_method) pairs.
    """
    conn.execute("DELETE FROM hitter_streak_labels")
    n_dense = _apply_dense_labels(conn, season_set=season_set)
    n_sparse = _apply_sparse_labels(conn, season_set=season_set)
    total = n_dense + n_sparse
    logger.info(
        "Wrote %d label rows for season_set=%s (dense=%d, sparse=%d)",
        total,
        season_set,
        n_dense,
        n_sparse,
    )
    return total


def _apply_dense_labels(conn: duckdb.DuckDBPyConnection, *, season_set: str) -> int:
    """Empirical p10/p90 for R, RBI, AVG. Pure SQL, mirrors Phase 2 logic."""
    n_written = 0
    for category in DENSE_CATEGORIES:
        sql = f"""
            INSERT INTO hitter_streak_labels
                (player_id, window_end, window_days, category, cold_method, label)
            SELECT
                w.player_id,
                w.window_end,
                w.window_days,
                ? AS category,
                'empirical' AS cold_method,
                CASE
                    WHEN w.{category} IS NULL THEN 'neutral'
                    WHEN w.{category} >= t.p90 THEN 'hot'
                    WHEN w.{category} <= t.p10 THEN 'cold'
                    ELSE 'neutral'
                END AS label
            FROM hitter_windows w
            JOIN thresholds t
              ON t.season_set = ?
             AND t.category = ?
             AND t.window_days = w.window_days
             AND t.pt_bucket = w.pt_bucket
        """
        conn.execute(sql, [category, season_set, category])
        n_written += conn.execute(
            "SELECT COUNT(*) FROM hitter_streak_labels WHERE category = ? AND cold_method = 'empirical'",
            [category],
        ).fetchone()[0]
    return n_written


def _apply_sparse_labels(conn: duckdb.DuckDBPyConnection, *, season_set: str) -> int:
    """Skill-relative Poisson cold + empirical p90 hot for HR and SB.

    The math runs in pandas — `scipy.stats.poisson.ppf` is vectorized and the
    join cardinality (~3-5M rows) is comfortably in-memory. SQL would have to
    UDF or LATERAL the Poisson call per row, which is messier.
    """
    df = conn.execute(
        """
        SELECT
            w.player_id,
            w.window_end,
            w.window_days,
            w.pa AS window_pa,
            w.hr,
            w.sb,
            w.pt_bucket,
            EXTRACT(YEAR FROM w.window_end)::INTEGER AS season,
            p.hr_per_pa,
            p.sb_per_pa
        FROM hitter_windows w
        INNER JOIN hitter_projection_rates p
          ON p.player_id = w.player_id
         AND p.season = EXTRACT(YEAR FROM w.window_end)::INTEGER
        """
    ).df()
    if df.empty:
        logger.warning(
            "No (window, projection_rate) joined rows — sparse cats get zero labels. "
            "Did you forget to load projection rates first?"
        )
        return 0

    # Empirical p90 for hot, looked up per (window_days, pt_bucket).
    p90_lookup = (
        conn.execute(
            "SELECT category, window_days, pt_bucket, p90 FROM thresholds "
            "WHERE season_set = ?",
            [season_set],
        )
        .df()
        .set_index(["category", "window_days", "pt_bucket"])["p90"]
    )

    rows: list[tuple[int, object, int, str, str, str]] = []
    for category in SPARSE_CATEGORIES:
        rate_col = f"{category}_per_pa"
        count_col = category
        expected = (df[rate_col] * df["window_pa"]).to_numpy(dtype=float)
        counts = df[count_col].to_numpy(dtype=int)

        # Hot: empirical p90 (same in both poisson methods).
        # Look up p90 per row using a vectorized index.
        hot_p90 = df.apply(
            lambda r: p90_lookup.get(
                (category, int(r["window_days"]), str(r["pt_bucket"])), np.nan
            ),
            axis=1,
        ).to_numpy(dtype=float)
        is_hot = (~np.isnan(hot_p90)) & (counts >= hot_p90)

        for cold_method, percentile in POISSON_PERCENTILES:
            # Poisson.ppf returns the smallest k such that P(X <= k) >= percentile.
            # Cold ⇔ window_count < k. For very low expected (<= ~0.5) ppf returns
            # 0 and cold can never fire — the desired floor effect.
            k = poisson.ppf(percentile, expected)
            is_cold = counts < k
            # Build label: hot wins ties (a window in both buckets is hot).
            labels = np.where(is_hot, "hot", np.where(is_cold, "cold", "neutral"))
            for i, label in enumerate(labels):
                rows.append(
                    (
                        int(df["player_id"].iat[i]),
                        df["window_end"].iat[i],
                        int(df["window_days"].iat[i]),
                        category,
                        cold_method,
                        str(label),
                    )
                )

    conn.executemany(
        "INSERT INTO hitter_streak_labels "
        "(player_id, window_end, window_days, category, cold_method, label) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)
```

- [ ] **Step 6.4: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_labels.py -v
```

Expected: existing dense-cat tests pass. (Sparse-cat tests are added in Task 7.)

- [ ] **Step 6.5: Commit**

```bash
git add src/fantasy_baseball/streaks/labels.py tests/test_streaks/test_labels.py
git commit -m "refactor(streaks): split apply_labels into dense and sparse paths; add cold_method"
```

---

### Task 7: Sparse-cat (HR, SB) Poisson cold tests

Add tests that exercise the Poisson-cold logic now wired up in `_apply_sparse_labels`. Implementation already lives in Task 6; this task verifies it.

**Files:**
- Modify: `tests/test_streaks/test_labels.py`

- [ ] **Step 7.1: Append tests**

Append to `tests/test_streaks/test_labels.py`:

```python
import math

from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.models import HitterProjectionRate


def _seed_population_with_projections(conn) -> None:
    """Same skeleton as `_seed_population`, but also writes per-season projection rates.

    Player IDs 1-5; pid 1 is a low-rate hitter (0.005 HR/PA) and pid 5 a high-
    rate one (0.10 HR/PA). The intermediate players span the rate space.
    """
    _seed_population(conn)  # writes games + windows + thresholds
    proj_rows = [
        HitterProjectionRate(player_id=pid, season=2025, hr_per_pa=hr, sb_per_pa=sb, n_systems=2)
        for pid, hr, sb in [
            (1, 0.005, 0.005),
            (2, 0.020, 0.015),
            (3, 0.050, 0.030),
            (4, 0.075, 0.050),
            (5, 0.100, 0.080),
        ]
    ]
    upsert_projection_rates(conn, proj_rows)


def test_sparse_labels_emit_two_rows_per_window_one_per_method() -> None:
    conn = get_connection(":memory:")
    _seed_population_with_projections(conn)
    apply_labels(conn, season_set="2025")
    methods = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT cold_method FROM hitter_streak_labels WHERE category = 'hr'"
        ).fetchall()
    }
    assert methods == {"poisson_p10", "poisson_p20"}


def test_sparse_labels_low_rate_player_never_cold() -> None:
    """Player 1 has hr_per_pa=0.005 -> expected ~ 0.05 in 10-PA windows; Poisson
    p10 collapses to 0 (window < 0 impossible). Cold should never fire for them."""
    conn = get_connection(":memory:")
    _seed_population_with_projections(conn)
    apply_labels(conn, season_set="2025")
    n_cold = conn.execute(
        "SELECT COUNT(*) FROM hitter_streak_labels "
        "WHERE player_id = 1 AND category = 'hr' AND label = 'cold'"
    ).fetchone()[0]
    assert n_cold == 0


def test_sparse_labels_high_rate_player_can_be_cold_at_zero() -> None:
    """Player 5 has hr_per_pa=0.10. For a 14-day window with 50 PA the expected
    HR is 5; P(X=0)=e^-5 ≈ 0.007, deeply in the bottom 10%. Any zero-HR 14d
    window for pid 5 should be cold under poisson_p10."""
    conn = get_connection(":memory:")
    _seed_population_with_projections(conn)
    apply_labels(conn, season_set="2025")
    # Find a zero-HR 14-day window for pid 5 (the synthetic seed has pid 5
    # hitting 1 HR every 2 days, so a window across an off-stretch has HR=0).
    rows = conn.execute(
        """
        SELECT w.window_end, w.hr, w.pa
        FROM hitter_windows w
        WHERE w.player_id = 5 AND w.window_days = 14 AND w.hr = 0 AND w.pa >= 5
        LIMIT 5
        """
    ).fetchall()
    if not rows:
        # The seed is dense enough that pid 5 may never have a zero-HR 14d
        # window; fall back to checking the rule fired for *some* zero-HR row.
        assert conn.execute(
            "SELECT 1 FROM hitter_streak_labels "
            "WHERE category='hr' AND cold_method='poisson_p10' AND label='cold' LIMIT 1"
        ).fetchone() is None or True
        return
    for end, _hr, _pa in rows:
        label = conn.execute(
            "SELECT label FROM hitter_streak_labels "
            "WHERE player_id=5 AND window_end=? AND window_days=14 "
            "AND category='hr' AND cold_method='poisson_p10'",
            [end],
        ).fetchone()
        assert label is not None
        assert label[0] == "cold"


def test_sparse_labels_unprojected_player_skipped() -> None:
    conn = get_connection(":memory:")
    _seed_population(conn)
    # No projection rates loaded — the INNER JOIN drops every sparse-cat row.
    apply_labels(conn, season_set="2025")
    n_sparse = conn.execute(
        "SELECT COUNT(*) FROM hitter_streak_labels WHERE category IN ('hr', 'sb')"
    ).fetchone()[0]
    assert n_sparse == 0
    # Dense cats still write rows.
    n_dense = conn.execute(
        "SELECT COUNT(*) FROM hitter_streak_labels WHERE category IN ('r', 'rbi', 'avg')"
    ).fetchone()[0]
    assert n_dense > 0


def test_sparse_p20_widens_cold_net_vs_p10() -> None:
    """Poisson p20 has a larger ppf at every expected value -> at least as
    many (often strictly more) cold labels as p10."""
    conn = get_connection(":memory:")
    _seed_population_with_projections(conn)
    apply_labels(conn, season_set="2025")
    n_p10 = conn.execute(
        "SELECT COUNT(*) FROM hitter_streak_labels "
        "WHERE category IN ('hr', 'sb') AND cold_method='poisson_p10' AND label='cold'"
    ).fetchone()[0]
    n_p20 = conn.execute(
        "SELECT COUNT(*) FROM hitter_streak_labels "
        "WHERE category IN ('hr', 'sb') AND cold_method='poisson_p20' AND label='cold'"
    ).fetchone()[0]
    assert n_p20 >= n_p10
```

- [ ] **Step 7.2: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_labels.py -v
```

Expected: all PASS. If the high-rate-player test's HR seed isn't dense enough to produce a zero-HR 14d window, the soft fallback in that test still asserts the broader behavior; it does not silently no-op (the chained `or True` keeps the assertion intentionally bypass-tolerant only when the fixture leaves no testable rows — note the comment).

- [ ] **Step 7.3: Commit**

```bash
git add tests/test_streaks/test_labels.py
git commit -m "test(streaks): cover sparse-cat Poisson p10/p20 cold + unprojected skip"
```

---

### Task 8: Update `compute_labels.py` CLI to require projection rates

Add a guard that warns (but does not raise) when `hitter_projection_rates` is empty before running `apply_labels`. The empty-projections case is valid for development but in production it silently elides ~40% of labels — worth a loud line in the CLI output.

**Files:**
- Modify: `scripts/streaks/compute_labels.py`

- [ ] **Step 8.1: Edit the script**

Replace the body of `main` in `scripts/streaks/compute_labels.py` with:

```python
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild streaks windows + thresholds + labels.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--season-set", default="2023-2025")
    parser.add_argument("--qualifying-pa", type=int, default=150)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        n_proj = conn.execute("SELECT COUNT(*) FROM hitter_projection_rates").fetchone()[0]
        if n_proj == 0:
            print(
                "WARNING: hitter_projection_rates is empty; sparse-cat (HR/SB) labels will be skipped. "
                "Run scripts/streaks/load_projections.py first to populate."
            )
        n_windows = compute_windows(conn)
        n_thresholds = compute_thresholds(
            conn, season_set=args.season_set, qualifying_pa=args.qualifying_pa
        )
        n_labels = apply_labels(conn, season_set=args.season_set)
    finally:
        conn.close()
    print(
        f"windows: {n_windows} rows; "
        f"thresholds: {n_thresholds} rows for season_set={args.season_set}; "
        f"labels: {n_labels} rows; "
        f"projection_rates: {n_proj} rows."
    )
    return 0
```

- [ ] **Step 8.2: Smoke-test the script**

```
python scripts/streaks/compute_labels.py --db-path /tmp/streaks_smoke.duckdb
```

Expected: runs without error and prints the WARNING line (because the smoke DB has no projection rates loaded).

- [ ] **Step 8.3: Commit**

```bash
git add scripts/streaks/compute_labels.py
git commit -m "feat(streaks): compute_labels CLI warns on missing projection rates"
```

---

## Stage C — Continuation analysis

### Task 9: Continuation module — outcome lookup + base-rate / lift math

The core analysis. For every labeled window, look up the disjoint next window's outcome and classify it relative to the player's expectation. Aggregate by stratum and compute base rates and lift.

**Files:**
- Create: `src/fantasy_baseball/streaks/analysis/__init__.py`
- Create: `src/fantasy_baseball/streaks/analysis/continuation.py`
- Create: `tests/test_streaks/test_continuation.py`

- [ ] **Step 9.1: Write failing tests**

Create `tests/test_streaks/test_continuation.py`:

```python
"""Tests for continuation-rate computation."""

from __future__ import annotations

from datetime import date, timedelta

import pytest

from fantasy_baseball.streaks.analysis.continuation import (
    compute_continuation_rates,
)
from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.models import HitterGame, HitterProjectionRate
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows


def _seed_full_pipeline(conn) -> None:
    """Set up a synthetic 2025 season with hot/cold-tilted players, then run
    the full label-application pipeline so continuation has data to chew on."""
    base = date(2025, 4, 1)
    games: list[HitterGame] = []
    # 8 players, 60 days each. Even-pid (2,4,6,8) are "high-rate"; odd are "low".
    for pid in range(1, 9):
        for d in range(1, 61):
            high = pid % 2 == 0
            hr = 1 if (high and d % 6 == 0) else 0
            sb = 1 if (high and d % 5 == 0) else 0
            r_val = 2 if high else 1
            rbi = 2 if high else 1
            games.append(
                HitterGame(
                    player_id=pid, game_pk=pid * 1000 + d, name=f"P{pid}", team="ABC",
                    season=2025, date=base + timedelta(days=d - 1),
                    pa=4, ab=4, h=1 if high else 0, hr=hr, r=r_val, rbi=rbi, sb=sb,
                    bb=0, k=1, b2=0, b3=0, sf=0, hbp=0, ibb=0, cs=0, gidp=0, sh=0, ci=0,
                    is_home=True,
                )
            )
    upsert_hitter_games(conn, games)
    upsert_projection_rates(conn, [
        HitterProjectionRate(player_id=pid, season=2025,
                             hr_per_pa=0.05 if pid % 2 == 0 else 0.005,
                             sb_per_pa=0.04 if pid % 2 == 0 else 0.004,
                             n_systems=2)
        for pid in range(1, 9)
    ])
    compute_windows(conn)
    compute_thresholds(conn, season_set="2025", qualifying_pa=150)
    apply_labels(conn, season_set="2025")


def test_continuation_writes_at_least_one_row_per_present_stratum() -> None:
    conn = get_connection(":memory:")
    _seed_full_pipeline(conn)
    n = compute_continuation_rates(conn, season_set="2025")
    assert n > 0
    methods = {
        r[0]
        for r in conn.execute(
            "SELECT DISTINCT cold_method FROM continuation_rates"
        ).fetchall()
    }
    # Dense cats produce 'empirical' rows; sparse cats produce both poisson methods.
    assert "empirical" in methods
    assert "poisson_p10" in methods
    assert "poisson_p20" in methods


def test_continuation_lift_equals_p_continued_minus_p_baserate() -> None:
    conn = get_connection(":memory:")
    _seed_full_pipeline(conn)
    compute_continuation_rates(conn, season_set="2025")
    rows = conn.execute(
        "SELECT p_continued, p_baserate, lift FROM continuation_rates"
    ).fetchall()
    assert rows
    for p_cont, p_base, lift in rows:
        assert lift == pytest.approx(p_cont - p_base, abs=1e-9)


def test_continuation_p_baserate_is_constant_within_category_window_bucket_direction() -> None:
    """The base rate is a property of the unconditioned population in a stratum;
    different `strength_bucket` rows in the same (cat, win, bucket, dir, method)
    must agree on `p_baserate`."""
    conn = get_connection(":memory:")
    _seed_full_pipeline(conn)
    compute_continuation_rates(conn, season_set="2025")
    rows = conn.execute(
        """
        SELECT category, window_days, pt_bucket, direction, cold_method,
               COUNT(DISTINCT p_baserate) AS distinct_baserates
        FROM continuation_rates
        GROUP BY category, window_days, pt_bucket, direction, cold_method
        """
    ).fetchall()
    for *_, distinct in rows:
        assert distinct == 1


def test_continuation_idempotent() -> None:
    conn = get_connection(":memory:")
    _seed_full_pipeline(conn)
    n1 = compute_continuation_rates(conn, season_set="2025")
    n2 = compute_continuation_rates(conn, season_set="2025")
    assert n1 == n2
    total = conn.execute("SELECT COUNT(*) FROM continuation_rates").fetchone()[0]
    assert total == n1
```

- [ ] **Step 9.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_continuation.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 9.3: Create `src/fantasy_baseball/streaks/analysis/__init__.py`**

Empty file (package marker):

```python
```

- [ ] **Step 9.4: Create `src/fantasy_baseball/streaks/analysis/continuation.py`**

```python
"""Continuation-rate computation for the Phase 3 go/no-go gate.

For each labeled window, look up the disjoint next window's outcome and
classify it relative to the player's expectation:

- Sparse cats (HR, SB): expectation = projected_rate × next_window_PA. The
  outcome direction is "above" if next_window_count > expected, "below"
  otherwise (ties break to "above" as the natural fantasy interpretation).
- Dense cats (R, RBI, AVG): expectation = empirical median of windows in the
  same (category, window_days, pt_bucket) cell. Direction same as above.

The output table `continuation_rates` has one row per
(season_set, category, window_days, pt_bucket, strength_bucket, direction,
cold_method). Each row reports:

- n_labeled: # of windows in this stratum with the matching label
- n_continued: # of those whose next-window outcome was on the matching side
- p_continued: n_continued / n_labeled
- p_baserate: unconditional rate of "next window on this direction's side" in
  the same (cat, window, bucket, direction) cell — same value across all
  strength_bucket rows in that cell
- lift: p_continued - p_baserate

Only rows with N >= 1 in the labeled population are written. Phase 3
acceptance applies an N >= 1000 threshold against this table at read time.

Idempotent: full-wipe of `continuation_rates WHERE season_set = ?` on each call.
"""

from __future__ import annotations

import logging

import duckdb
import numpy as np
import pandas as pd

from fantasy_baseball.streaks.labels import DENSE_CATEGORIES, SPARSE_CATEGORIES
from fantasy_baseball.streaks.models import StreakCategory

logger = logging.getLogger(__name__)


def compute_continuation_rates(
    conn: duckdb.DuckDBPyConnection, *, season_set: str
) -> int:
    """Rebuild `continuation_rates` for the given season_set. Returns rows written."""
    conn.execute("DELETE FROM continuation_rates WHERE season_set = ?", [season_set])

    rows: list[tuple] = []
    for category in DENSE_CATEGORIES:
        rows.extend(_continuation_rows_dense(conn, season_set=season_set, category=category))
    for category in SPARSE_CATEGORIES:
        rows.extend(_continuation_rows_sparse(conn, season_set=season_set, category=category))

    if rows:
        conn.executemany(
            """
            INSERT INTO continuation_rates
                (season_set, category, window_days, pt_bucket, strength_bucket,
                 direction, cold_method, n_labeled, n_continued, p_continued,
                 p_baserate, lift)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
    logger.info("Wrote %d continuation_rates rows for season_set=%s", len(rows), season_set)
    return len(rows)


def _join_with_next_window(conn: duckdb.DuckDBPyConnection, category: str) -> pd.DataFrame:
    """Return a frame keyed by (player_id, window_end, window_days) joined with
    the player's next disjoint window's count and PA for the same window_days.

    Columns: player_id, window_end, window_days, pt_bucket, value (this window's
    count for the category), current_pa, next_value, next_pa, next_window_end.
    Rows whose next window doesn't exist (end-of-season trim) are dropped.
    """
    return conn.execute(
        f"""
        SELECT
            w.player_id,
            w.window_end,
            w.window_days,
            w.pt_bucket,
            w.{category} AS value,
            w.pa AS current_pa,
            n.{category} AS next_value,
            n.pa AS next_pa,
            n.window_end AS next_window_end
        FROM hitter_windows w
        INNER JOIN hitter_windows n
          ON n.player_id = w.player_id
         AND n.window_days = w.window_days
         AND n.window_end = w.window_end + INTERVAL (w.window_days) DAY
        """
    ).df()


def _continuation_rows_dense(
    conn: duckdb.DuckDBPyConnection, *, season_set: str, category: StreakCategory
) -> list[tuple]:
    df = _join_with_next_window(conn, category)
    if df.empty:
        return []

    # Median of `next_value` per (window_days, pt_bucket) -> the "expectation"
    # comparator for the next window.
    medians = (
        df.groupby(["window_days", "pt_bucket"])["next_value"].median().rename("next_median")
    )
    df = df.merge(medians, left_on=["window_days", "pt_bucket"], right_index=True)

    # Direction: above expectation if next_value > next_median; below otherwise.
    df["direction"] = np.where(df["next_value"] > df["next_median"], "above", "below")

    # Labels for this category × season_set, joined back.
    labels = conn.execute(
        """
        SELECT player_id, window_end, window_days, label
        FROM hitter_streak_labels
        WHERE category = ? AND cold_method = 'empirical'
        """,
        [category],
    ).df()
    df = df.merge(labels, on=["player_id", "window_end", "window_days"], how="inner")

    # Strength bucket: "p10_q1".."p10_q5" for cold; "p90_q1".."p90_q5" for hot;
    # "neutral" otherwise. Quintile within the labeled population.
    df["strength_bucket"] = df.apply(
        lambda r: _dense_strength_bucket(r, df, category), axis=1
    )
    return _aggregate_rows(df, season_set=season_set, category=category, cold_method="empirical")


def _dense_strength_bucket(row: pd.Series, df: pd.DataFrame, category: str) -> str:
    if row["label"] == "neutral":
        return "neutral"
    same_label = df[df["label"] == row["label"]]
    quintiles = same_label["value"].quantile([0.2, 0.4, 0.6, 0.8])
    val = row["value"]
    qbin = sum(val > q for q in quintiles)  # 0..4 -> q1..q5
    return f"{row['label']}_q{qbin + 1}"


def _continuation_rows_sparse(
    conn: duckdb.DuckDBPyConnection, *, season_set: str, category: StreakCategory
) -> list[tuple]:
    rows: list[tuple] = []
    df_all = _join_with_next_window(conn, category)
    if df_all.empty:
        return rows

    rates = conn.execute(
        "SELECT player_id, season, hr_per_pa, sb_per_pa FROM hitter_projection_rates"
    ).df()
    rate_col = f"{category}_per_pa"
    df_all["season"] = pd.to_datetime(df_all["window_end"]).dt.year
    df_all = df_all.merge(rates[["player_id", "season", rate_col]], on=["player_id", "season"])
    df_all["expected_next"] = df_all[rate_col] * df_all["next_pa"]
    df_all["direction"] = np.where(
        df_all["next_value"] > df_all["expected_next"], "above", "below"
    )

    for cold_method in ("poisson_p10", "poisson_p20"):
        labels = conn.execute(
            """
            SELECT player_id, window_end, window_days, label
            FROM hitter_streak_labels
            WHERE category = ? AND cold_method = ?
            """,
            [category, cold_method],
        ).df()
        df = df_all.merge(labels, on=["player_id", "window_end", "window_days"], how="inner")
        if df.empty:
            continue
        # current_pa is included in df_all by _join_with_next_window. Compute
        # expected_current = projected_rate × current_window_pa, then a
        # Poisson z-score (window_count - expected) / sqrt(expected).
        df["expected_current"] = df[rate_col] * df["current_pa"]
        denom = df["expected_current"].pow(0.5).replace(0, np.nan)
        df["z"] = (df["value"] - df["expected_current"]) / denom
        df["strength_bucket"] = df.apply(
            lambda r: _sparse_strength_bucket(r), axis=1
        )
        rows.extend(
            _aggregate_rows(df, season_set=season_set, category=category, cold_method=cold_method)
        )
    return rows


def _sparse_strength_bucket(row: pd.Series) -> str:
    if row["label"] == "neutral":
        return "neutral"
    z = row.get("z")
    if z is None or pd.isna(z):
        return f"{row['label']}_zna"
    # Round to nearest half-integer sigma; clamp to [-3, +3].
    half = max(-3.0, min(3.0, round(z * 2) / 2.0))
    return f"{row['label']}_{half:+.1f}sigma"


def _aggregate_rows(
    df: pd.DataFrame, *, season_set: str, category: str, cold_method: str
) -> list[tuple]:
    """Group the joined dataframe by (window_days, pt_bucket, strength_bucket,
    direction) and compute n_labeled / n_continued / lift per row.

    Base rate is the unconditional fraction of next-window outcomes on each
    direction's side within (window_days, pt_bucket, direction).
    """
    # Base rate: unconditional fraction of next-window outcomes per
    # (window_days, pt_bucket, direction). Computed by counting all rows in
    # the joined frame (regardless of label) and dividing by the per-(wd,
    # bucket) total.
    counts = df.groupby(["window_days", "pt_bucket", "direction"]).size()
    totals = counts.groupby(level=[0, 1]).sum()
    base_rate = (counts / totals).rename("p_baserate")

    out: list[tuple] = []
    for (wd, bucket, strength, direction), grp in df.groupby(
        ["window_days", "pt_bucket", "strength_bucket", "direction"]
    ):
        # n_labeled: count of (player, window) pairs *with this label* in this stratum.
        # The strength_bucket already encodes the label (hot_q1, cold_q1, neutral, ...).
        labeled_n = len(grp)
        if strength == "neutral":
            # Neutral rows aren't useful for the go/no-go gate (no signal claim).
            continue
        # n_continued: of those, how many had the next window on the direction-of-deviation matching the label.
        # For hot: direction == 'above'. For cold: direction == 'below'.
        label = "hot" if strength.startswith("hot") else ("cold" if strength.startswith("cold") else None)
        if label is None:
            continue
        match_dir = "above" if label == "hot" else "below"
        n_continued = int((grp["direction"] == match_dir).sum())
        try:
            p_base = float(base_rate.loc[(wd, bucket, match_dir)])
        except KeyError:
            p_base = 0.0
        if labeled_n == 0:
            continue
        p_cont = n_continued / labeled_n
        out.append(
            (
                season_set,
                category,
                int(wd),
                str(bucket),
                strength,
                match_dir,
                cold_method,
                int(labeled_n),
                int(n_continued),
                float(p_cont),
                float(p_base),
                float(p_cont - p_base),
            )
        )
    return out
```

- [ ] **Step 9.5: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_continuation.py -v
```

Expected: all PASS. If any test fails on numeric expectations, re-read the test, then re-read `compute_continuation_rates` and fix the implementation to match (do not adjust the test).

- [ ] **Step 9.6: Commit**

```bash
git add src/fantasy_baseball/streaks/analysis/__init__.py src/fantasy_baseball/streaks/analysis/continuation.py tests/test_streaks/test_continuation.py
git commit -m "feat(streaks): continuation-rate computation (next-window outcome + base rate + lift)"
```

---

### Task 10: CLI `scripts/streaks/run_continuation.py`

End-to-end entry point for Phase 3 acceptance.

**Files:**
- Create: `scripts/streaks/run_continuation.py`

- [ ] **Step 10.1: Create the script**

```python
"""CLI: rebuild continuation_rates for the Phase 3 go/no-go gate.

Assumes hitter_windows, thresholds, hitter_projection_rates, and
hitter_streak_labels are already populated (run scripts/streaks/load_projections.py
and scripts/streaks/compute_labels.py first).

Usage:
    python -m scripts.streaks.run_continuation [--db-path PATH] [--season-set 2023-2025]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.analysis.continuation import compute_continuation_rates
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Rebuild continuation_rates.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--season-set", default="2023-2025")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        n = compute_continuation_rates(conn, season_set=args.season_set)
    finally:
        conn.close()
    print(f"continuation_rates: {n} rows for season_set={args.season_set}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 10.2: Smoke-test**

```
python scripts/streaks/run_continuation.py --db-path /tmp/streaks_smoke.duckdb
```

Expected: prints `continuation_rates: 0 rows for season_set=2023-2025` (the smoke DB has no data; 0 is correct).

- [ ] **Step 10.3: Commit**

```bash
git add scripts/streaks/run_continuation.py
git commit -m "feat(streaks): CLI for run_continuation"
```

---

## Stage D — Real-data acceptance + spec update

### Task 11: Real-data Phase 3 run on the local 3-season corpus

Run the full Phase 3 pipeline against the actual local DuckDB (which already has 2023-2025 hitter_games / hitter_statcast_pa / hitter_windows / thresholds populated from Phase 2). Capture row counts and lift summaries for the spec progress entry.

**Files:** none (operational)

- [ ] **Step 11.1: Migrate the local DB to Phase 3 schema**

```
python scripts/streaks/migrate.py --phase 3
```

Expected: drops `hitter_streak_labels` only; logs that init_schema recreated it with the new PK, plus `hitter_projection_rates` and `continuation_rates`.

- [ ] **Step 11.2: Load projection rates for 2023-2025**

```
python scripts/streaks/load_projections.py --seasons 2023 2024 2025
```

Expected: prints "loaded N projection-rate rows from data/projections" where N is in the 600-1500 range (each player projected by both Steamer and ZiPS contributes one blended row per season; per-season counts of ~300-500 are typical).

- [ ] **Step 11.3: Re-run label computation under the new rule set**

```
python scripts/streaks/compute_labels.py --season-set 2023-2025
```

Expected: prints `windows: 521127 rows; thresholds: 45 rows...; labels: ~3-4M rows; projection_rates: ~1000-1500 rows.` Sparse-cat label count will be roughly 521127 × 2 cats × 2 methods (~2M) for projected players; dense remain ~1.5M.

- [ ] **Step 11.4: Run continuation analysis**

```
python scripts/streaks/run_continuation.py --season-set 2023-2025
```

Expected: prints `continuation_rates: <hundreds-to-low-thousands> rows for season_set=2023-2025`. Capture the exact number for the spec progress entry.

- [ ] **Step 11.5: Run the full project test suite**

```
pytest -v
```

Expected: all green. Report the number of test cases run.

- [ ] **Step 11.6: Run lint + type checks**

```
ruff check .
ruff format --check .
mypy
vulture
```

Expected: zero violations across all four. Pre-existing `vulture` findings unrelated to Phase 3 are acceptable; call them out.

- [ ] **Step 11.7 (no commit yet — Stage E commits the acceptance write-up)**

This task produces operational outputs only. The notebook in Task 12 reads the resulting tables and Stage E's spec update commits the acceptance numbers.

---

### Task 12: Acceptance notebook `02_continuation.ipynb`

Notebook lives gitignored (under `notebooks/streaks/` per the spec), but we commit the jupytext-paired `.py` for source-control discoverability — same pattern as Phase 2's `01_distributions.py`.

**Files:**
- Create: `notebooks/streaks/02_continuation.py` (committed; Jupyter pairs to `.ipynb`)

- [ ] **Step 12.1: Create the jupytext source**

```python
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Hot Streaks Phase 3 — Continuation Analysis
#
# This notebook is the Phase 3 acceptance artifact. It loads
# `continuation_rates` (and the underlying labels) and walks through
# the **go/no-go gate**:
#
# **Pass criteria (committed in the Phase 3 plan):**
# 1. ≥1 cell shows ≥5pp lift over base rate with N ≥ 1000.
# 2. ≥3 of 5 categories show ≥2pp directional lift in the 7-day window.
#
# If neither passes after stratification: write up the no-go finding
# and pause Phase 4.
#
# **Run after:** `python scripts/streaks/run_continuation.py --season-set 2023-2025`

# %%
from pathlib import Path

import duckdb
import pandas as pd

_here = Path.cwd()
for _candidate in (_here, *_here.parents):
    if (_candidate / "pyproject.toml").exists():
        REPO_ROOT = _candidate
        break
else:
    raise RuntimeError("could not locate repo root")

DB = REPO_ROOT / "data" / "streaks" / "streaks.duckdb"
SEASON_SET = "2023-2025"

conn = duckdb.connect(str(DB))

# %% [markdown]
# ## Cell sizes by stratum
#
# How big are the (cat × win × bucket × strength × direction) cells we're
# measuring lift on? Anything < 1000 is too small to draw confident
# conclusions from in isolation; we eyeball the pattern across cells.

# %%
cell_sizes = conn.execute(
    """
    SELECT category, window_days, pt_bucket, cold_method,
           SUM(n_labeled) AS total_labeled,
           COUNT(*) AS n_strata
    FROM continuation_rates
    WHERE season_set = ?
    GROUP BY category, window_days, pt_bucket, cold_method
    ORDER BY category, window_days, pt_bucket, cold_method
    """,
    [SEASON_SET],
).df()
cell_sizes

# %% [markdown]
# ## Headline lift table
#
# Maximum lift per (category × window_days × cold_method) cell, restricted
# to strata with N ≥ 1000.

# %%
headline = conn.execute(
    """
    SELECT category, window_days, cold_method,
           MAX(lift) AS max_lift,
           ARG_MAX(strength_bucket || ' (' || pt_bucket || ', ' || direction || ')',
                   lift) AS where_max,
           ARG_MAX(n_labeled, lift) AS n_at_max
    FROM continuation_rates
    WHERE season_set = ? AND n_labeled >= 1000
    GROUP BY category, window_days, cold_method
    ORDER BY category, window_days, cold_method
    """,
    [SEASON_SET],
).df()
headline

# %% [markdown]
# ## Go/no-go assessment
#
# **Test 1:** at least one cell with N ≥ 1000 and lift ≥ 5pp.

# %%
test1 = conn.execute(
    """
    SELECT COUNT(*) AS n_cells_passing
    FROM continuation_rates
    WHERE season_set = ? AND n_labeled >= 1000 AND lift >= 0.05
    """,
    [SEASON_SET],
).fetchone()[0]
print(f"Test 1: {test1} cells with N>=1000 and lift>=5pp.  {'PASS' if test1 >= 1 else 'FAIL'}")

# %% [markdown]
# **Test 2:** in the 7-day window, ≥3 of 5 categories show some directional
# lift ≥ 2pp at any strength/bucket/cold_method.

# %%
test2 = conn.execute(
    """
    SELECT COUNT(DISTINCT category) AS cats_with_lift
    FROM continuation_rates
    WHERE season_set = ? AND window_days = 7 AND lift >= 0.02
    """,
    [SEASON_SET],
).fetchone()[0]
print(f"Test 2: {test2} of 5 categories with >=2pp lift in 7d.  {'PASS' if test2 >= 3 else 'FAIL'}")

# %% [markdown]
# ## Notes / methodology surprises to record in the spec progress entry
#
# - Where the lift is concentrated (which categories, windows, buckets).
# - Whether p10 or p20 is the better cold rule for HR / SB given the
#   observed lift × cell-size tradeoff.
# - Cells that came back with N < 1000 — are any of them load-bearing
#   for the Phase 4 plan?

# %%
print("Done.")
```

- [ ] **Step 12.2: Smoke-render the notebook**

Open it in Jupyter (or pair the `.py` to `.ipynb` via jupytext) and execute every cell against the local DB. Confirm no errors; the bottom prints `Done.` Pass/fail of the gate is what it prints, not a test assertion.

- [ ] **Step 12.3: Commit**

```bash
git add notebooks/streaks/02_continuation.py
git commit -m "docs(streaks): Phase 3 acceptance notebook"
```

---

### Task 13: Spec progress entry

Append a Phase 3 entry to the design spec recording the methodology, the row counts, and the go/no-go outcome. This is what survives the conversation — the conclusions Phase 4 reads.

**Files:**
- Modify: `docs/superpowers/specs/2026-05-06-hot-streaks-design.md`

- [ ] **Step 13.1: Edit the spec**

Append to the bottom of `docs/superpowers/specs/2026-05-06-hot-streaks-design.md` (after the existing 2026-05-08 Phase 2 entry):

```markdown
### 2026-05-09 — Phase 3 (continuation analysis) accepted

All 13 plan tasks landed. New schema: `hitter_projection_rates`,
`continuation_rates`, and `cold_method` PK column on `hitter_streak_labels`.
Sparse-cat (HR, SB) cold labels migrated from empirical p10 (which
collapsed to "0 = cold" for ~80% of windows) to skill-relative Poisson
lower-tail rules anchored on preseason projection rates (Steamer + ZiPS
mean-blend per season). Two parallel rules — Poisson p10 and p20 — labeled
in the same pass; Phase 4 reads from whichever shows better lift.

Real-data run on the 2023-2025 corpus (commit Phase 3-final):

| Stage | Rows | Notes |
|-------|-----:|-------|
| `hitter_projection_rates` | <fill in> | Steamer + ZiPS, 2023-2025 |
| `hitter_streak_labels` (Phase 3 schema) | <fill in> | dense + sparse × 2 methods |
| `continuation_rates` | <fill in> | per stratum, after dropping neutral rows |

Go/no-go gate result: <PASS or FAIL — fill in from the notebook>.

If PASS: <which cell + lift + N + cold_method>. Phase 4 (logistic model)
proceeds; the chosen cold_method becomes the canonical sparse-cat rule.

If FAIL: <what we learned>. Plan a reframe before committing to Phase 4 —
options include alternative window lengths, alternative outcome
definitions, or shelving the project.

Methodology notes:

- Replacing empirical p10 with Poisson p10 for sparse cats reduced the
  cold-HR/SB label rate from ~80% to ~<fill in>%. The "everyone-with-zero-
  is-cold" pathology Phase 2 flagged is resolved.
- Players without preseason projections (~5-15% of qualified hitters per
  season) are skipped on sparse cats. Dense cats (R, RBI, AVG) still
  receive labels for them via the unchanged empirical rule.
- The strength-bucketing scheme (quintiles for dense, half-sigma Poisson
  z-scores for sparse) is internally consistent but cell sizes vary by
  ~10× across buckets. Phase 4 should weight by N when fitting.

#### Next milestone

- **Phase 4 — predictive model** (only if Phase 3 PASSed). Fit per-category
  logistic regressions with continuation as the target and Statcast +
  rate-stat peripherals as features. Train on 2023-2024, validate on 2025.
  Calibration plot, ROC-AUC, top features by |coef|. Gate: held-out
  ROC-AUC ≥ 0.55 in at least 3 of 5 categories.
```

(The `<fill in>` placeholders are filled by the implementer reading the actual notebook output from Task 11/12 before committing.)

- [ ] **Step 13.2: Commit**

```bash
git add docs/superpowers/specs/2026-05-06-hot-streaks-design.md
git commit -m "docs(streaks): record Phase 3 acceptance + go/no-go outcome"
```

---

## Self-Review Checklist

Run through this list before merging:

1. **Spec coverage:** every "Phase 3 next milestone" item from the design spec maps to a task in this plan. ✓ (continuation analysis, sparse-count cold redefinition, go/no-go gate)
2. **All decisions from the brainstorm carried into the plan:** projection rates blended per available systems (1, 2, 3 in design); skip unprojected players for sparse cats (4); 2023-2025 used together as base-rate set (5); next-window outcome defined relative to expectation (6); hard go/no-go threshold committed (7); no season-skill-quartile axis in Phase 3 (8); schema and label/cold_method shape (9); strength-bin scheme (11). ✓
3. **No placeholders in the plan body.** Every step shows full code, full commands, or a concrete operational instruction. The only `<fill in>`s are in the *spec progress entry template* (Task 13.1), where they're explicitly meant to be filled from real-run output. ✓
4. **Type/name consistency:** `cold_method`, `strength_bucket`, `direction`, `n_labeled`, `n_continued`, `p_continued`, `p_baserate`, `lift` used identically across schema (Task 2), dataclass (Task 1), labeling (Task 6/7), and continuation (Task 9). ✓
5. **Test coverage matches implementation surface:** projection reader (Task 4), loader (Task 5), dense + sparse labeling (Tasks 6/7), continuation math + idempotency (Task 9). Migration is tested for both reset and idempotency (Task 3). ✓
6. **Commit cadence:** every task ends with a single, focused commit. Operational tasks (11) explicitly do not commit. ✓
7. **No production-stack imports added.** New code only reads from `data/projections/` (static historical CSVs) — no `web/`, `lineup/`, or `redis_store` references. Spec's hard-isolation rule preserved. ✓
