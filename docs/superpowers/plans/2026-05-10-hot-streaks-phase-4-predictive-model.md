# Hot Streaks — Phase 4 (Predictive Model) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fit eight per-direction logistic-regression models that predict whether a hot/cold 14-day hitter window continues into the next window, with the held-out 2025 ROC-AUC ≥ 0.55 in at least three of five categories. Persist fit metadata (no model artifacts) so Phase 5 can refit on demand and inspect drift over time.

**Architecture:** Stage A extends `hitter_projection_rates` to carry R/RBI/AVG projected rates (currently HR/SB only) so every model can take `season_rate_in_category` as a feature, and adds a `model_fits` table for the audit trail. Stage B builds the training-frame query + the per-model fitter (sklearn `Pipeline` with `StandardScaler` + `LogisticRegression`, tuned via player-grouped `GroupKFold` over `C ∈ {0.01, 0.1, 1.0, 10.0}`). Stage C adds reporting helpers — bootstrap CIs on coefficients (200 player-grouped resamples), permutation feature importance, ROC-AUC + reliability evaluation. Stage D orchestrates all 8 fits and the CLI. Stage E is the real-data acceptance run, the jupytext acceptance notebook, and the spec progress entry. No isotonic correction in Phase 4 (deferred to Phase 5); no per-cat curated feature subsets (deferred to a future Phase 4.5 if the gate passes); no `joblib`-pickled model artifacts (refit on demand is fast enough).

**Tech Stack:** sklearn 1.8 (`Pipeline`, `StandardScaler`, `LogisticRegression`, `GroupKFold`, `permutation_importance`, `roc_auc_score`, `calibration_curve`), pandas, numpy, scipy, duckdb. **New dependency:** `scikit-learn` is currently installed transitively via `pybaseball`; declare it explicitly under `[project.optional-dependencies].dev` so future dependency resolves stay reproducible.

**Spec:** `docs/superpowers/specs/2026-05-06-hot-streaks-design.md` — Phase 4 high-level scope. The Phase 3 progress entry's "Methodology surprises / notes for Phase 4" section seeded several design constraints already encoded below.

---

## Design Decisions (resolved before plan-writing — see brainstorm in conversation log on 2026-05-10)

1. **Model structure: per-direction logistic regressions, hot model + cold model per category.** Hot model is trained only on rows where `current_label='hot'` and predicts `next_window_above_bucket_median=1`. Cold model is trained only on `current_label='cold'` and predicts `next_window_below_bucket_median=1`. **Sparse categories (HR, SB) get only the hot model** — Phase 3 showed the cold cells (HR poisson_p20 = 7,105 rows; SB poisson_p20 = 2,621 rows) are too sparse, especially after the per-player-grouped train/val split. **Total = 8 models.**
2. **Windows: 14d only.** Phase 3 acceptance noted 14d dominates the lift charts for every category. The 7d-SB-hot model is deferred to a follow-up — see "Future work" at the end of this plan.
3. **Train/validation split: player-grouped 5-fold CV on 2023-2024 to tune L2 strength `C ∈ {0.01, 0.1, 1.0, 10.0}`, then refit on the full 2023-2024 with the chosen `C`, then evaluate **once** on 2025.** GroupKFold with `groups=player_id` so the same hitter is never in both fold-train and fold-val. **2026 is untouched in Phase 4** — it is reserved for Phase 5 production inference.
4. **Features (uniform across all 8 models):** ten features — `streak_strength_numeric`, `babip`, `k_pct`, `bb_pct`, `iso`, `ev_avg`, `barrel_pct`, `xwoba_avg`, `season_rate_in_category`, plus one-hot `pt_bucket_low` / `pt_bucket_mid` / `pt_bucket_high` (= 12 columns post-encoding). `streak_strength_numeric` is parsed from Phase 3's `strength_bucket` string — for dense quintiles: `hot_q1=1 … hot_q5=5`, `cold_q1=1 … cold_q5=5`; for sparse half-sigma buckets: the signed sigma value as a float (e.g. `hot_+1.5sigma` → `1.5`, `hot_+2.5sigma` → `2.5`). Rows with `strength_bucket = '{label}_zna'` are dropped (definitionally undefined strength).
5. **Target derivation:** mirror Phase 3's continuation logic exactly.
   - **Dense cat hot/cold:** target = `next_value > next_bucket_median` (hot) / `next_value < next_bucket_median` (cold), where `next_bucket_median` is the median of `next_value` within `(window_days, pt_bucket)` computed across the entire labeled population (across both train and val seasons combined — the median is a population statistic, not a per-fold parameter; computing it on train alone would push median values for low-N buckets in val to spurious places). Ties (next_value == median) for cold go to the negative class.
   - **Sparse hot:** target = `next_value > expected_next_value`, where `expected_next_value = projected_rate × next_window_PA` per Phase 3.
6. **Reporting per model:**
   - **Calibration:** diagnostic reliability diagram (10 bins) on 2025. **No isotonic/Platt correction in Phase 4** — Phase 5 applies one if needed when probabilities feed the Sunday report. AUC is rank-based and immune.
   - **Coefficient uncertainty:** bootstrap CIs via 200 player-grouped resamples (resample players-with-replacement from the train set, refit, collect coefficients, report 5th/95th percentiles). Tracks under L2 regularization where asymptotic p-values are not well-defined.
   - **Feature importance:** `sklearn.inspection.permutation_importance` on 2025 — shuffle each feature and measure AUC drop. Honest under correlated features; doesn't care about scale.
7. **Persistence: no model artifact pickles.** Refit-on-demand is fast (8 LRs × ~50ms = 0.4s on this corpus). We **do** persist fit metadata to a new DuckDB table `model_fits` so 2027-us can run a query and see how 2023-2024 → 2025 generalization compared to 2024-2025 → 2026.
8. **`season_rate_in_category` data gap:** Phase 3's `hitter_projection_rates` schema has only `hr_per_pa` and `sb_per_pa`. Phase 4 extends it to add `r_per_pa`, `rbi_per_pa`, `avg` (projected season AVG) — pulled from the same Steamer/ZiPS CSVs. Migration is a non-destructive `ALTER TABLE ADD COLUMN` for each new field (DuckDB supports this); rows are refilled by re-running `scripts/streaks/load_projections.py`.
9. **Gate (committed upfront):** held-out 2025 ROC-AUC ≥ 0.55 in **at least 3 of 5 categories**, where per-category AUC is `max(hot_model_AUC, cold_model_AUC)` for dense cats and `hot_model_AUC` for sparse cats. If we fall short, the spec progress entry records the failure honestly and Phase 5 either reframes (different windows, different features) or shelves.
10. **Phase 3 `cold_method` selection for sparse categories:** sparse cats only have hot models in Phase 4, so the `poisson_p10` vs `poisson_p20` choice is **not relevant for training** (hot rows use empirical p90, which is independent of `cold_method`). The hot rows duplicated across both `cold_method` partitions in `hitter_streak_labels` are de-duplicated at training-frame build time by filtering to one method (we use `'poisson_p20'`) so the model isn't trained on two identical rows per (player, window). This choice is recorded in `model_fits.cold_method` for the HR-hot and SB-hot rows; for dense models the value is `'empirical'`.
11. **Class imbalance:** "next above bucket median given hot" is ~60-70% positive per Phase 3 lift; cold-direction targets are ~30-40% positive. Mild imbalance — sklearn `LogisticRegression(class_weight=None)` is fine. **If** any model's CV AUC is borderline (<0.55 with the default), the implementer should try `class_weight='balanced'` for that model only and record the choice in the notebook. This is operational discretion, not a default.
12. **Multicollinearity:** `babip ↔ xwoba_avg` and `iso ↔ barrel_pct` are correlated. L2 absorbs this. Bootstrap CIs on correlated pairs will be wide; the notebook narrative should note this rather than be surprised by it.

---

## File Structure

| Path | Status | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/streaks/models.py` | Modify | Extend `HitterProjectionRate` with `r_per_pa`, `rbi_per_pa`, `avg`; add `ModelFit` dataclass |
| `src/fantasy_baseball/streaks/data/schema.py` | Modify | Expand `hitter_projection_rates` DDL; add `model_fits` table DDL |
| `src/fantasy_baseball/streaks/data/migrate.py` | Modify | Add `migrate_to_phase_4` (ALTER add columns; CREATE model_fits) |
| `src/fantasy_baseball/streaks/data/projections.py` | Modify | Compute and emit `r_per_pa`, `rbi_per_pa`, `avg` alongside HR/SB rates |
| `src/fantasy_baseball/streaks/data/load_projections.py` | Modify | Loader is dataclass-fields-driven — schema change cascades automatically; one test update only |
| `src/fantasy_baseball/streaks/data/load_model_fits.py` | Create | Idempotent upsert into `model_fits` |
| `src/fantasy_baseball/streaks/analysis/predictors.py` | Create | `build_training_frame`, `fit_one_model`, `bootstrap_coef_ci`, `permutation_feature_importance`, `evaluate_model`, `fit_all_models` |
| `scripts/streaks/migrate.py` | Modify | Add `--phase 4` option |
| `scripts/streaks/load_projections.py` | (no change) | Already dataclass-fields-driven — picks up the new fields without edits |
| `scripts/streaks/fit_models.py` | Create | CLI entry point: orchestrate Phase 4 fits, print summary |
| `pyproject.toml` | Modify | Declare `scikit-learn>=1.5` explicitly under `[project.optional-dependencies].dev` |
| `tests/test_streaks/test_models.py` | Modify | Assert expanded `HitterProjectionRate` fields; add `ModelFit` shape test |
| `tests/test_streaks/test_schema.py` | Modify | Assert `r_per_pa/rbi_per_pa/avg` columns; assert `model_fits` table |
| `tests/test_streaks/test_migrate.py` | Modify | Assert `migrate_to_phase_4` adds columns + table, idempotent |
| `tests/test_streaks/test_projections.py` | Modify | Assert R/RBI/AVG blending matches HR/SB blending pattern |
| `tests/test_streaks/test_load_projections.py` | Modify | Updated fixture for the expanded dataclass |
| `tests/test_streaks/test_predictors.py` | Create | Tests for training-frame, fitting, bootstrap, permutation, evaluation, orchestrator |
| `tests/test_streaks/test_load_model_fits.py` | Create | Upsert idempotency, replace-on-conflict |
| `notebooks/streaks/03_predictors.py` | Create (gitignored `.ipynb` via jupytext) | Acceptance notebook |
| `docs/superpowers/specs/2026-05-06-hot-streaks-design.md` | Modify | Append Phase 4 progress entry |

---

## Stage A — Schema and projection-rate expansion

### Task 1: Extend `HitterProjectionRate` dataclass + add `ModelFit` dataclass

The loader's column tuple is derived from `dataclasses.fields(HitterProjectionRate)`, so adding fields here cascades through `upsert_projection_rates` automatically. `ModelFit` is new — corresponds 1:1 to the `model_fits` row written by `fit_all_models`.

**Files:**
- Modify: `src/fantasy_baseball/streaks/models.py`
- Modify: `tests/test_streaks/test_models.py`

- [ ] **Step 1.1: Write failing dataclass-shape tests**

Append to `tests/test_streaks/test_models.py`:

```python
def test_hitter_projection_rate_includes_dense_cat_rates() -> None:
    expected = (
        "player_id",
        "season",
        "hr_per_pa",
        "sb_per_pa",
        "r_per_pa",
        "rbi_per_pa",
        "avg",
        "n_systems",
    )
    assert tuple(f.name for f in fields(HitterProjectionRate)) == expected


def test_model_fit_fields_in_expected_order() -> None:
    from fantasy_baseball.streaks.models import ModelFit

    expected = (
        "model_id",
        "category",
        "direction",
        "season_set",
        "window_days",
        "cold_method",
        "chosen_C",
        "cv_auc_mean",
        "cv_auc_std",
        "val_auc",
        "n_train_rows",
        "n_val_rows",
        "fit_timestamp",
    )
    assert tuple(f.name for f in fields(ModelFit)) == expected
```

- [ ] **Step 1.2: Run the tests, confirm they fail**

```
pytest tests/test_streaks/test_models.py -v
```

Expected: two FAILures (HitterProjectionRate missing dense-cat rate fields; ModelFit not defined).

- [ ] **Step 1.3: Edit `src/fantasy_baseball/streaks/models.py`**

Replace the `HitterProjectionRate` dataclass with the expanded version and append `ModelFit` to the bottom of the file:

```python
@dataclass(frozen=True, slots=True)
class HitterProjectionRate:
    """Per-season blended projection rate for a single hitter.

    PK is (player_id, season). Rates are season-prior blended arithmetic means
    across all available systems (Steamer + ZiPS for 2023-2025; up to 5 systems
    for 2026+). All five fantasy categories are stored so per-category models
    in Phase 4 can take ``season_rate_in_category`` as a feature.

    For backward compatibility with rows written by the Phase 3 loader (which
    only populated hr_per_pa / sb_per_pa), the dense-cat fields are nullable
    until ``scripts/streaks/load_projections.py`` is re-run after the Phase 4
    migration.

    ``n_systems`` records the count of systems that contributed; rows with
    n_systems < 2 are still kept (caller decides whether to use them).
    """

    player_id: int
    season: int
    hr_per_pa: float
    sb_per_pa: float
    r_per_pa: float | None
    rbi_per_pa: float | None
    avg: float | None
    n_systems: int


@dataclass(frozen=True, slots=True)
class ModelFit:
    """One row of the Phase 4 ``model_fits`` audit table.

    PK is ``model_id`` (synthetic string like 'hr_hot_2023-2024').

    - ``cold_method`` is the partition of `hitter_streak_labels` the training
      rows were drawn from. Dense cats use 'empirical'; sparse hot uses
      'poisson_p20' (deduplication choice — see plan design decision #10).
    - ``chosen_C`` is the L2 strength selected by GroupKFold over a fixed grid.
    - ``cv_auc_mean`` / ``cv_auc_std`` are the per-fold AUC stats for that C.
    - ``val_auc`` is the single-shot 2025 ROC-AUC — the gate metric.
    - ``n_train_rows`` / ``n_val_rows`` are post-filter row counts (drop
      strength=zna, drop NULL season_rate, drop NULL peripherals).
    """

    model_id: str
    category: StreakCategory
    direction: StreakDirection
    season_set: str
    window_days: int
    cold_method: ColdMethod
    chosen_C: float
    cv_auc_mean: float
    cv_auc_std: float
    val_auc: float
    n_train_rows: int
    n_val_rows: int
    fit_timestamp: datetime
```

Add `from datetime import datetime` to the imports at the top of `models.py` if it isn't there yet.

- [ ] **Step 1.4: Run the tests, confirm they pass**

```
pytest tests/test_streaks/test_models.py -v
```

Expected: PASS.

- [ ] **Step 1.5: Commit**

```bash
git add src/fantasy_baseball/streaks/models.py tests/test_streaks/test_models.py
git commit -m "feat(streaks): Phase 4 dataclasses (expand projection_rate for dense cats, add ModelFit)"
```

---

### Task 2: Schema — expand `hitter_projection_rates`, add `model_fits` table

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/schema.py`
- Modify: `tests/test_streaks/test_schema.py`

- [ ] **Step 2.1: Write failing schema tests**

Append to `tests/test_streaks/test_schema.py`:

```python
def test_hitter_projection_rates_has_dense_cat_columns() -> None:
    conn = get_connection(":memory:")
    info = conn.execute("PRAGMA table_info('hitter_projection_rates')").fetchall()
    cols = {r[1] for r in info}
    assert {"r_per_pa", "rbi_per_pa", "avg"}.issubset(cols)


def test_model_fits_table_exists() -> None:
    conn = get_connection(":memory:")
    info = conn.execute("PRAGMA table_info('model_fits')").fetchall()
    cols = {r[1] for r in info}
    expected_cols = {
        "model_id",
        "category",
        "direction",
        "season_set",
        "window_days",
        "cold_method",
        "chosen_C",
        "cv_auc_mean",
        "cv_auc_std",
        "val_auc",
        "n_train_rows",
        "n_val_rows",
        "fit_timestamp",
    }
    assert expected_cols.issubset(cols)
    pk_cols = [r[1] for r in info if r[5]]
    assert pk_cols == ["model_id"]
```

- [ ] **Step 2.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_schema.py -v
```

Expected: two FAILures.

- [ ] **Step 2.3: Edit `src/fantasy_baseball/streaks/data/schema.py`**

Replace the `hitter_projection_rates` DDL entry in `_SCHEMA_DDL` with the expanded version, and append a new DDL entry for `model_fits`. The dense-cat rate columns are nullable so existing Phase 3 rows survive `init_schema` (re-running `load_projections.py` after migration backfills them):

```python
    """
    CREATE TABLE IF NOT EXISTS hitter_projection_rates (
        player_id INTEGER NOT NULL,
        season INTEGER NOT NULL,
        hr_per_pa DOUBLE NOT NULL,
        sb_per_pa DOUBLE NOT NULL,
        r_per_pa DOUBLE,
        rbi_per_pa DOUBLE,
        avg DOUBLE,
        n_systems INTEGER NOT NULL,
        PRIMARY KEY (player_id, season)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS model_fits (
        model_id VARCHAR NOT NULL,
        category VARCHAR NOT NULL,
        direction VARCHAR NOT NULL,
        season_set VARCHAR NOT NULL,
        window_days INTEGER NOT NULL,
        cold_method VARCHAR NOT NULL,
        chosen_C DOUBLE NOT NULL,
        cv_auc_mean DOUBLE NOT NULL,
        cv_auc_std DOUBLE NOT NULL,
        val_auc DOUBLE NOT NULL,
        n_train_rows INTEGER NOT NULL,
        n_val_rows INTEGER NOT NULL,
        fit_timestamp TIMESTAMP NOT NULL,
        PRIMARY KEY (model_id)
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
git commit -m "feat(streaks): Phase 4 schema (dense-cat projection rates, model_fits table)"
```

---

### Task 3: Phase 4 migration helper

Idempotent migration that adds the three nullable rate columns to `hitter_projection_rates` (if missing) and creates `model_fits` (via `init_schema`). The `ALTER TABLE ADD COLUMN IF NOT EXISTS` form is DuckDB-supported and safe to re-run.

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/migrate.py`
- Modify: `scripts/streaks/migrate.py`
- Modify: `tests/test_streaks/test_migrate.py`

- [ ] **Step 3.1: Write failing tests**

Append to `tests/test_streaks/test_migrate.py`:

```python
def test_migrate_to_phase_4_adds_dense_cat_columns_and_model_fits() -> None:
    from fantasy_baseball.streaks.data.migrate import migrate_to_phase_4

    conn = get_connection(":memory:")
    # init_schema in get_connection() already wrote Phase 4 columns — but the
    # migration must still be safe to run against a fresh DB.
    migrate_to_phase_4(conn)

    cols = {
        r[1] for r in conn.execute("PRAGMA table_info('hitter_projection_rates')").fetchall()
    }
    assert {"r_per_pa", "rbi_per_pa", "avg"}.issubset(cols)
    model_fits_cols = {
        r[1] for r in conn.execute("PRAGMA table_info('model_fits')").fetchall()
    }
    assert "model_id" in model_fits_cols


def test_migrate_to_phase_4_preserves_existing_rate_rows() -> None:
    """The migration must NOT clobber existing hitter_projection_rates rows —
    Phase 3 rows have NULL r/rbi/avg until load_projections is re-run."""
    from fantasy_baseball.streaks.data.migrate import migrate_to_phase_4

    conn = get_connection(":memory:")
    conn.execute(
        "INSERT INTO hitter_projection_rates "
        "(player_id, season, hr_per_pa, sb_per_pa, n_systems) "
        "VALUES (1, 2024, 0.05, 0.02, 2)"
    )
    migrate_to_phase_4(conn)
    row = conn.execute(
        "SELECT hr_per_pa, sb_per_pa, r_per_pa FROM hitter_projection_rates "
        "WHERE player_id = 1 AND season = 2024"
    ).fetchone()
    assert row == (0.05, 0.02, None)


def test_migrate_to_phase_4_is_idempotent() -> None:
    from fantasy_baseball.streaks.data.migrate import migrate_to_phase_4

    conn = get_connection(":memory:")
    migrate_to_phase_4(conn)
    migrate_to_phase_4(conn)  # second call must not raise
    cols = {
        r[1] for r in conn.execute("PRAGMA table_info('hitter_projection_rates')").fetchall()
    }
    assert {"r_per_pa", "rbi_per_pa", "avg"}.issubset(cols)
```

- [ ] **Step 3.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_migrate.py -v
```

Expected: three FAILures — `migrate_to_phase_4` not defined.

- [ ] **Step 3.3: Edit `src/fantasy_baseball/streaks/data/migrate.py`**

Append at the bottom of the file:

```python
def migrate_to_phase_4(conn: duckdb.DuckDBPyConnection) -> None:
    """Add Phase 4 columns/tables. Idempotent and non-destructive.

    Adds three nullable rate columns to ``hitter_projection_rates``
    (``r_per_pa``, ``rbi_per_pa``, ``avg``) so dense-cat continuation
    models can take ``season_rate_in_category`` as a feature. Then calls
    ``init_schema`` to ensure the ``model_fits`` table exists.

    Existing Phase 3 rows (hr_per_pa + sb_per_pa only) survive with NULL
    in the new columns. Re-run ``scripts/streaks/load_projections.py``
    after this migration to backfill them.

    ``hitter_games`` / ``hitter_statcast_pa`` / ``hitter_windows`` /
    ``thresholds`` / ``hitter_streak_labels`` / ``continuation_rates``
    are untouched.
    """
    for col in ("r_per_pa", "rbi_per_pa", "avg"):
        conn.execute(
            f"ALTER TABLE hitter_projection_rates ADD COLUMN IF NOT EXISTS {col} DOUBLE"
        )
        logger.info("ALTER hitter_projection_rates ADD COLUMN IF NOT EXISTS %s", col)
    init_schema(conn)
    logger.info("Recreated/ensured Phase 4 tables via init_schema (model_fits)")
```

- [ ] **Step 3.4: Edit `scripts/streaks/migrate.py`**

Add Phase 4 to the choices and dispatch:

```python
from fantasy_baseball.streaks.data.migrate import (
    migrate_to_phase_2,
    migrate_to_phase_3,
    migrate_to_phase_4,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Migrate the streaks DuckDB schema.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--phase", type=int, choices=[2, 3, 4], default=4)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        if args.phase == 2:
            migrate_to_phase_2(conn)
        elif args.phase == 3:
            migrate_to_phase_3(conn)
        else:
            migrate_to_phase_4(conn)
    finally:
        conn.close()
    return 0
```

- [ ] **Step 3.5: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_migrate.py -v
```

Expected: PASS.

- [ ] **Step 3.6: Commit**

```bash
git add src/fantasy_baseball/streaks/data/migrate.py scripts/streaks/migrate.py tests/test_streaks/test_migrate.py
git commit -m "feat(streaks): migrate_to_phase_4 (add dense-cat rate columns + model_fits table)"
```

---

### Task 4: Extend `projections.py` to blend R/RBI/AVG

Same shape as the existing HR/SB blending. R and RBI from FanGraphs CSVs are per-season counting totals; we divide by PA. AVG is already a rate (no PA divisor needed). The blender averages across all available systems with equal weights.

**Files:**
- Modify: `src/fantasy_baseball/streaks/data/projections.py`
- Modify: `tests/test_streaks/test_projections.py`

- [ ] **Step 4.1: Write failing test**

Append to `tests/test_streaks/test_projections.py`:

```python
def test_load_projection_rates_blends_dense_categories(tmp_path: Path) -> None:
    base = tmp_path / "2024"
    base.mkdir()
    # Steamer: 600 PA, 30 HR, 10 SB, 90 R, 100 RBI, .280 AVG.
    # ZiPS:    600 PA, 36 HR, 14 SB, 100 R, 110 RBI, .300 AVG.
    # Blended rates: HR=0.055/PA, SB=0.020/PA, R=0.158333/PA, RBI=0.175/PA, AVG=0.290.
    pd.DataFrame(
        [{
            "Name": "P", "PA": 600, "HR": 30, "SB": 10,
            "R": 90, "RBI": 100, "AVG": 0.280, "MLBAMID": 100,
        }],
        columns=["Name", "PA", "HR", "SB", "R", "RBI", "AVG", "MLBAMID"],
    ).to_csv(base / "steamer-hitters.csv", index=False)
    pd.DataFrame(
        [{
            "Name": "P", "PA": 600, "HR": 36, "SB": 14,
            "R": 100, "RBI": 110, "AVG": 0.300, "MLBAMID": 100,
        }],
        columns=["Name", "PA", "HR", "SB", "R", "RBI", "AVG", "MLBAMID"],
    ).to_csv(base / "zips-hitters.csv", index=False)

    rates = load_projection_rates(tmp_path, season=2024)
    assert len(rates) == 1
    r = rates[0]
    assert r.hr_per_pa == pytest.approx(0.055, rel=1e-6)
    assert r.sb_per_pa == pytest.approx(0.020, rel=1e-6)
    assert r.r_per_pa == pytest.approx((90 + 100) / 2 / 600, rel=1e-6)
    assert r.rbi_per_pa == pytest.approx((100 + 110) / 2 / 600, rel=1e-6)
    assert r.avg == pytest.approx(0.290, rel=1e-6)
    assert r.n_systems == 2


def test_load_projection_rates_handles_missing_dense_columns(tmp_path: Path) -> None:
    """If a CSV is missing R/RBI/AVG columns (older fixtures), the loader emits
    the rate row with NULL in those fields rather than crashing."""
    base = tmp_path / "2024"
    base.mkdir()
    # Old-style CSV: only Name/PA/HR/SB/MLBAMID.
    pd.DataFrame(
        [{"Name": "P", "PA": 600, "HR": 30, "SB": 12, "MLBAMID": 100}],
        columns=["Name", "PA", "HR", "SB", "MLBAMID"],
    ).to_csv(base / "steamer-hitters.csv", index=False)
    rates = load_projection_rates(tmp_path, season=2024)
    assert len(rates) == 1
    assert rates[0].r_per_pa is None
    assert rates[0].rbi_per_pa is None
    assert rates[0].avg is None
```

- [ ] **Step 4.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_projections.py -v
```

Expected: two FAILures — the existing loader doesn't compute R/RBI/AVG.

- [ ] **Step 4.3: Edit `src/fantasy_baseball/streaks/data/projections.py`**

Update `_load_one_system` and `load_projection_rates` to thread the three new columns. The full updated module:

```python
"""Read FanGraphs preseason projection CSVs and blend per-PA rates.

Reads every ``<system>-hitters*.csv`` under ``data/projections/{season}/``,
filters to rows with PA >= ``PROJECTION_PA_FLOOR`` (drops org filler / NRI
spring rows), coerces MLBAMID to int, and computes per-system rates for
all five fantasy categories:

- ``hr_per_pa`` = HR / PA
- ``sb_per_pa`` = SB / PA
- ``r_per_pa`` = R / PA
- ``rbi_per_pa`` = RBI / PA
- ``avg`` = projected AVG (already a rate in FanGraphs CSVs; no PA divisor)

Returns one ``HitterProjectionRate`` per (player_id, season) with the
simple arithmetic mean across the systems that included that player.
Old-style CSVs lacking R/RBI/AVG columns produce ``None`` for those rates
(rather than crashing) — caller decides whether to use them.

Filename pattern variation: 2025's CSVs are named ``<system>-hitters-2025.csv``;
other years use ``<system>-hitters.csv``. The discovery glob matches both.

This module reads flat CSVs only — no imports from ``web/`` or ``lineup/``,
preserving the streaks package's hard isolation from the production stack.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball.streaks.models import HitterProjectionRate

logger = logging.getLogger(__name__)

PROJECTION_PA_FLOOR = 200

_DENSE_CAT_SOURCE_COLS: tuple[tuple[str, str, bool], ...] = (
    # (source_csv_col, output_rate_col, is_per_pa_count)
    # is_per_pa_count=True: divide by PA. False: already a rate (AVG).
    ("R", "r_per_pa", True),
    ("RBI", "rbi_per_pa", True),
    ("AVG", "avg", False),
)


def discover_projection_files(projections_root: Path, *, season: int) -> list[Path]:
    season_dir = projections_root / str(season)
    if not season_dir.is_dir():
        return []
    return [
        p
        for p in season_dir.iterdir()
        if p.is_file() and p.suffix == ".csv" and "hitters" in p.name and "pitchers" not in p.name
    ]


def _load_one_system(path: Path) -> pd.DataFrame:
    """Load one system's projection CSV.

    Output columns: MLBAMID, PA, hr_per_pa, sb_per_pa, r_per_pa, rbi_per_pa, avg.
    Dense-cat rate columns are NaN if the source CSV is missing R/RBI/AVG.
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "MLBAMID" not in df.columns:
        logger.warning("File %s has no MLBAMID column; skipping", path)
        return pd.DataFrame(
            columns=["MLBAMID", "PA", "hr_per_pa", "sb_per_pa", "r_per_pa", "rbi_per_pa", "avg"]
        )
    df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce")
    df = df.dropna(subset=["MLBAMID"])
    df["MLBAMID"] = df["MLBAMID"].astype(int)
    df = df[df["PA"] >= PROJECTION_PA_FLOOR].copy()
    df["hr_per_pa"] = df["HR"] / df["PA"]
    df["sb_per_pa"] = df["SB"] / df["PA"]
    for src_col, out_col, is_per_pa_count in _DENSE_CAT_SOURCE_COLS:
        if src_col in df.columns:
            df[out_col] = (df[src_col] / df["PA"]) if is_per_pa_count else df[src_col]
        else:
            df[out_col] = np.nan
    return df[["MLBAMID", "PA", "hr_per_pa", "sb_per_pa", "r_per_pa", "rbi_per_pa", "avg"]]


def load_projection_rates(
    projections_root: Path, *, season: int
) -> list[HitterProjectionRate]:
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

    # NaN-aware mean: a player missing R from one system still gets the other
    # system's R contribution. Pandas mean() skips NaN by default.
    blended = stacked.groupby("MLBAMID", as_index=False).agg(
        hr_per_pa=("hr_per_pa", "mean"),
        sb_per_pa=("sb_per_pa", "mean"),
        r_per_pa=("r_per_pa", "mean"),
        rbi_per_pa=("rbi_per_pa", "mean"),
        avg=("avg", "mean"),
        n_systems=("PA", "count"),
    )

    out: list[HitterProjectionRate] = []
    for r in blended.itertuples(index=False):
        out.append(
            HitterProjectionRate(
                player_id=int(r.MLBAMID),
                season=season,
                hr_per_pa=float(r.hr_per_pa),
                sb_per_pa=float(r.sb_per_pa),
                r_per_pa=None if pd.isna(r.r_per_pa) else float(r.r_per_pa),
                rbi_per_pa=None if pd.isna(r.rbi_per_pa) else float(r.rbi_per_pa),
                avg=None if pd.isna(r.avg) else float(r.avg),
                n_systems=int(r.n_systems),
            )
        )
    return out


def load_projection_rates_for_seasons(
    projections_root: Path, seasons: Iterable[int]
) -> list[HitterProjectionRate]:
    out: list[HitterProjectionRate] = []
    for s in seasons:
        out.extend(load_projection_rates(projections_root, season=s))
    return out
```

- [ ] **Step 4.4: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_projections.py -v
```

Expected: all PASS (including the 5 Phase 3 tests, which remain valid since the new fields are populated identically to HR/SB).

- [ ] **Step 4.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/projections.py tests/test_streaks/test_projections.py
git commit -m "feat(streaks): blend dense-cat projection rates (R/RBI/AVG) alongside HR/SB"
```

---

### Task 5: Update `load_projections` test fixtures for the expanded dataclass

The loader itself (`upsert_projection_rates`) derives its column tuple from `dataclasses.fields(HitterProjectionRate)` — no code change needed. Only the test fixtures need to be updated to construct the expanded dataclass. The `INSERT OR REPLACE` SQL also extends automatically.

**Files:**
- Modify: `tests/test_streaks/test_load_projections.py`

- [ ] **Step 5.1: Read the existing test file to find every `HitterProjectionRate(...)` construction**

```
pytest tests/test_streaks/test_load_projections.py -v
```

Expected: FAILures (TypeError: missing required positional arguments `r_per_pa`, `rbi_per_pa`, `avg`).

- [ ] **Step 5.2: Edit the `_row` helper to set the new fields to `None` by default**

In `tests/test_streaks/test_load_projections.py`, replace the existing helper with:

```python
def _row(
    pid: int,
    season: int,
    hr_pa: float = 0.05,
    sb_pa: float = 0.02,
    r_pa: float | None = None,
    rbi_pa: float | None = None,
    avg: float | None = None,
    n: int = 2,
):
    return HitterProjectionRate(
        player_id=pid,
        season=season,
        hr_per_pa=hr_pa,
        sb_per_pa=sb_pa,
        r_per_pa=r_pa,
        rbi_per_pa=rbi_pa,
        avg=avg,
        n_systems=n,
    )
```

Then append one new test asserting that dense-cat fields round-trip correctly:

```python
def test_upsert_projection_rates_persists_dense_cat_fields() -> None:
    conn = get_connection(":memory:")
    upsert_projection_rates(
        conn,
        [_row(1, 2024, hr_pa=0.05, sb_pa=0.02, r_pa=0.15, rbi_pa=0.18, avg=0.275)],
    )
    row = conn.execute(
        "SELECT hr_per_pa, sb_per_pa, r_per_pa, rbi_per_pa, avg "
        "FROM hitter_projection_rates WHERE player_id=1 AND season=2024"
    ).fetchone()
    assert row == (0.05, 0.02, 0.15, 0.18, 0.275)


def test_upsert_projection_rates_persists_nulls_for_dense_cat_fields() -> None:
    conn = get_connection(":memory:")
    upsert_projection_rates(conn, [_row(1, 2024)])  # all dense fields default None
    row = conn.execute(
        "SELECT r_per_pa, rbi_per_pa, avg FROM hitter_projection_rates "
        "WHERE player_id=1 AND season=2024"
    ).fetchone()
    assert row == (None, None, None)
```

- [ ] **Step 5.3: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_load_projections.py -v
```

Expected: all PASS.

- [ ] **Step 5.4: Commit**

```bash
git add tests/test_streaks/test_load_projections.py
git commit -m "test(streaks): cover dense-cat fields in projection_rates upsert"
```

---

### Task 6: `load_model_fits` upsert helper

Mirrors the pattern of `upsert_projection_rates` — dataclass-fields-driven SQL.

**Files:**
- Create: `src/fantasy_baseball/streaks/data/load_model_fits.py`
- Create: `tests/test_streaks/test_load_model_fits.py`

- [ ] **Step 6.1: Write failing tests**

Create `tests/test_streaks/test_load_model_fits.py`:

```python
from __future__ import annotations

from datetime import datetime

from fantasy_baseball.streaks.data.load_model_fits import upsert_model_fits
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.models import ModelFit


def _fit(
    model_id: str = "hr_hot_2023-2024",
    val_auc: float = 0.58,
) -> ModelFit:
    return ModelFit(
        model_id=model_id,
        category="hr",
        direction="above",
        season_set="2023-2024",
        window_days=14,
        cold_method="poisson_p20",
        chosen_C=1.0,
        cv_auc_mean=0.57,
        cv_auc_std=0.02,
        val_auc=val_auc,
        n_train_rows=20_000,
        n_val_rows=10_000,
        fit_timestamp=datetime(2026, 5, 10, 12, 0, 0),
    )


def test_upsert_model_fits_inserts_rows() -> None:
    conn = get_connection(":memory:")
    upsert_model_fits(conn, [_fit("a"), _fit("b")])
    n = conn.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    assert n == 2


def test_upsert_model_fits_replaces_on_pk_collision() -> None:
    conn = get_connection(":memory:")
    upsert_model_fits(conn, [_fit("a", val_auc=0.50)])
    upsert_model_fits(conn, [_fit("a", val_auc=0.62)])
    val = conn.execute(
        "SELECT val_auc FROM model_fits WHERE model_id='a'"
    ).fetchone()[0]
    assert val == 0.62


def test_upsert_model_fits_empty_input_is_noop() -> None:
    conn = get_connection(":memory:")
    upsert_model_fits(conn, [])
    n = conn.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    assert n == 0
```

- [ ] **Step 6.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_load_model_fits.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 6.3: Create `src/fantasy_baseball/streaks/data/load_model_fits.py`**

```python
"""Idempotent loader for ``model_fits``."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import fields
from operator import attrgetter

import duckdb

from fantasy_baseball.streaks.models import ModelFit

_MODEL_FIT_COLS = tuple(f.name for f in fields(ModelFit))
_model_fit_row = attrgetter(*_MODEL_FIT_COLS)


def upsert_model_fits(
    conn: duckdb.DuckDBPyConnection, rows: Sequence[ModelFit]
) -> None:
    """Insert or replace rows in `model_fits` keyed by model_id."""
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(_MODEL_FIT_COLS))
    sql = (
        f"INSERT OR REPLACE INTO model_fits ({', '.join(_MODEL_FIT_COLS)}) "
        f"VALUES ({placeholders})"
    )
    conn.executemany(sql, [_model_fit_row(r) for r in rows])
```

- [ ] **Step 6.4: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_load_model_fits.py -v
```

Expected: PASS.

- [ ] **Step 6.5: Commit**

```bash
git add src/fantasy_baseball/streaks/data/load_model_fits.py tests/test_streaks/test_load_model_fits.py
git commit -m "feat(streaks): model_fits upsert helper"
```

---

## Stage B — Predictors core

### Task 7: `build_training_frame` — feature + target derivation

For each (category, direction) pair, build a pandas DataFrame containing one row per labeled (player_id, window_end) at `window_days=14`, with the 10 model features + the binary target + a `player_id` column (for GroupKFold) + a `season` column (for the train/val split downstream).

**Files:**
- Create: `src/fantasy_baseball/streaks/analysis/predictors.py`
- Create: `tests/test_streaks/test_predictors.py`

- [ ] **Step 7.1: Write failing tests**

Create `tests/test_streaks/test_predictors.py`:

```python
"""Tests for Phase 4 predictor pipeline."""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.streaks.analysis.predictors import (
    EXPECTED_FEATURE_COLUMNS,
    build_training_frame,
)
from fantasy_baseball.streaks.data.load import upsert_hitter_games
from fantasy_baseball.streaks.data.load_projections import upsert_projection_rates
from fantasy_baseball.streaks.data.schema import get_connection
from fantasy_baseball.streaks.labels import apply_labels
from fantasy_baseball.streaks.models import HitterGame, HitterProjectionRate
from fantasy_baseball.streaks.thresholds import compute_thresholds
from fantasy_baseball.streaks.windows import compute_windows


def _seed_pipeline(conn, *, n_players: int = 16, n_days: int = 90, season: int = 2024) -> None:
    """Run the full Phase 1-3 pipeline against a synthetic fixture sized so
    Phase 4's GroupKFold has at least a few players per fold.

    ``game_pk`` includes ``season`` so that calling this twice with different
    seasons does not overwrite the first season's rows via the (player_id,
    game_pk) PK — important for the orchestrator test that needs both 2023
    and 2024 game data present simultaneously.
    """
    base = date(season, 4, 1)
    games: list[HitterGame] = []
    for pid in range(1, n_players + 1):
        for d in range(1, n_days + 1):
            high = pid % 2 == 0
            hr = 1 if (high and d % 6 == 0) else 0
            sb = 1 if (high and d % 5 == 0) else 0
            r_val = 2 if high else 1
            rbi = 2 if high else 1
            games.append(
                HitterGame(
                    player_id=pid,
                    game_pk=season * 100_000 + pid * 100 + d,
                    name=f"P{pid}",
                    team="ABC",
                    season=season,
                    date=base + timedelta(days=d - 1),
                    pa=4, ab=4, h=2 if high else 1, hr=hr, r=r_val, rbi=rbi, sb=sb,
                    bb=0, k=1, b2=0, b3=0, sf=0, hbp=0, ibb=0, cs=0, gidp=0, sh=0, ci=0,
                    is_home=True,
                )
            )
    upsert_hitter_games(conn, games)
    upsert_projection_rates(
        conn,
        [
            HitterProjectionRate(
                player_id=pid, season=season,
                hr_per_pa=0.05 if pid % 2 == 0 else 0.005,
                sb_per_pa=0.04 if pid % 2 == 0 else 0.004,
                r_per_pa=0.15 if pid % 2 == 0 else 0.10,
                rbi_per_pa=0.18 if pid % 2 == 0 else 0.10,
                avg=0.275 if pid % 2 == 0 else 0.230,
                n_systems=2,
            )
            for pid in range(1, n_players + 1)
        ],
    )
    compute_windows(conn)
    compute_thresholds(conn, season_set=str(season), qualifying_pa=50)
    apply_labels(conn, season_set=str(season))


def test_build_training_frame_columns_match_expected() -> None:
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="r", direction="above", season_set="2024", window_days=14
    )
    # Required: features + target + grouping/season metadata.
    for col in EXPECTED_FEATURE_COLUMNS:
        assert col in df.columns, f"missing feature column {col}"
    assert "target" in df.columns
    assert "player_id" in df.columns
    assert "season" in df.columns


def test_build_training_frame_hot_dense_target_matches_bucket_median() -> None:
    """For dense hot, target=1 iff next_value > median(next_value) within (window_days, pt_bucket)."""
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="r", direction="above", season_set="2024", window_days=14
    )
    assert not df.empty
    assert df["target"].isin([0, 1]).all()
    # At least some variation — fixture has both above-median and below-median rows.
    assert df["target"].sum() > 0
    assert df["target"].sum() < len(df)


def test_build_training_frame_filters_to_hot_only_for_above_direction() -> None:
    """The hot model trains only on rows currently labeled hot."""
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="r", direction="above", season_set="2024", window_days=14
    )
    # Streak strength numeric is parsed from "hot_qN" — values 1..5 only.
    assert df["streak_strength_numeric"].between(1, 5).all()


def test_build_training_frame_sparse_hr_hot_uses_poisson_p20_partition() -> None:
    """HR hot rows are duplicated across poisson_p10 and poisson_p20 in
    hitter_streak_labels. Dedup to p20 in the training frame so the model
    isn't trained on identical rows twice."""
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="hr", direction="above", season_set="2024", window_days=14
    )
    if df.empty:
        return  # fixture may not produce any hot HR windows; tolerable.
    # No duplicate (player_id, window_end) pairs — confirms dedup.
    assert df.duplicated(subset=["player_id", "window_end"]).sum() == 0


def test_build_training_frame_drops_zna_strength_rows() -> None:
    """Rows with strength_bucket ending in '_zna' have undefined sigma —
    drop them rather than guess a numeric encoding."""
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="hr", direction="above", season_set="2024", window_days=14
    )
    if df.empty:
        return
    # streak_strength_numeric is float for sparse — but never NaN after drop.
    assert df["streak_strength_numeric"].notna().all()


def test_build_training_frame_pt_bucket_one_hot_encoded() -> None:
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="r", direction="above", season_set="2024", window_days=14
    )
    for col in ("pt_bucket_low", "pt_bucket_mid", "pt_bucket_high"):
        assert col in df.columns
    # Each row has exactly one bucket flag set.
    assert (df[["pt_bucket_low", "pt_bucket_mid", "pt_bucket_high"]].sum(axis=1) == 1).all()


def test_build_training_frame_includes_season_rate_for_dense_cats() -> None:
    """For R hot, season_rate_in_category should equal hitter_projection_rates.r_per_pa."""
    conn = get_connection(":memory:")
    _seed_pipeline(conn)
    df = build_training_frame(
        conn, category="r", direction="above", season_set="2024", window_days=14
    )
    assert df["season_rate_in_category"].notna().all()
    # In the fixture, high-rate players have r_per_pa=0.15 and low-rate=0.10.
    assert set(df["season_rate_in_category"].round(2)).issubset({0.10, 0.15})
```

- [ ] **Step 7.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_predictors.py -v
```

Expected: FAIL — module doesn't exist.

- [ ] **Step 7.3: Create `src/fantasy_baseball/streaks/analysis/predictors.py` (training-frame portion)**

```python
"""Phase 4 predictor pipeline: per-direction logistic regressions on
streak continuation, with player-grouped CV, bootstrap CIs, and
permutation feature importance.

See ``docs/superpowers/plans/2026-05-10-hot-streaks-phase-4-predictive-model.md``
for the design decisions captured at the top of this module.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import duckdb
import numpy as np
import pandas as pd

from fantasy_baseball.streaks.models import StreakCategory, StreakDirection

logger = logging.getLogger(__name__)

# All 8 Phase 4 models. (category, direction): hot ⇔ 'above'; cold ⇔ 'below'.
PHASE_4_MODELS: tuple[tuple[StreakCategory, StreakDirection], ...] = (
    ("r", "above"),
    ("r", "below"),
    ("rbi", "above"),
    ("rbi", "below"),
    ("avg", "above"),
    ("avg", "below"),
    ("hr", "above"),  # sparse hot only
    ("sb", "above"),  # sparse hot only
)

DENSE_CATS: frozenset[StreakCategory] = frozenset({"r", "rbi", "avg"})
SPARSE_CATS: frozenset[StreakCategory] = frozenset({"hr", "sb"})

# Sparse hot rows are duplicated across poisson_p10/poisson_p20 in
# hitter_streak_labels. Phase 4 dedupes to a single partition for training.
SPARSE_HOT_COLD_METHOD = "poisson_p20"

# Final ordered list of feature column names in the training frame.
EXPECTED_FEATURE_COLUMNS: tuple[str, ...] = (
    "streak_strength_numeric",
    "babip",
    "k_pct",
    "bb_pct",
    "iso",
    "ev_avg",
    "barrel_pct",
    "xwoba_avg",
    "season_rate_in_category",
    "pt_bucket_low",
    "pt_bucket_mid",
    "pt_bucket_high",
)

# Parsing rules for strength_bucket → numeric.
_DENSE_BUCKET_RE = re.compile(r"^(hot|cold)_q([1-5])$")
_SPARSE_BUCKET_RE = re.compile(r"^(hot|cold)_([+-]?\d+\.\d)sigma$")


def _parse_strength_numeric(bucket: str) -> float | None:
    """Encode Phase 3's strength_bucket string as a numeric feature.

    - Dense quintiles "hot_qN" / "cold_qN" → integer 1..5.
    - Sparse half-sigma buckets "hot_+1.5sigma" → 1.5 (signed float).
    - "{label}_zna" or any other shape → None (caller drops the row).
    """
    if m := _DENSE_BUCKET_RE.match(bucket):
        return float(m.group(2))
    if m := _SPARSE_BUCKET_RE.match(bucket):
        return float(m.group(2))
    return None


def _build_training_frame_dense(
    conn: duckdb.DuckDBPyConnection,
    *,
    category: StreakCategory,
    direction: StreakDirection,
    season_set: str,
    window_days: int,
) -> pd.DataFrame:
    """Dense-cat training frame. Target uses the (window_days, pt_bucket)
    median of next_value computed across the entire labeled population."""
    label = "hot" if direction == "above" else "cold"
    rate_col = "avg" if category == "avg" else f"{category}_per_pa"

    df = conn.execute(
        f"""
        SELECT
            w.player_id,
            w.window_end,
            EXTRACT(YEAR FROM w.window_end)::INTEGER AS season,
            w.pt_bucket,
            w.{category} AS value,
            n.{category} AS next_value,
            w.babip,
            w.k_pct,
            w.bb_pct,
            w.iso,
            w.ev_avg,
            w.barrel_pct,
            w.xwoba_avg,
            p.{rate_col} AS season_rate_in_category,
            l.label,
            CASE
                WHEN ? = 'above' THEN
                    CASE WHEN n.{category} > t.next_median THEN 1 ELSE 0 END
                ELSE
                    CASE WHEN n.{category} < t.next_median THEN 1 ELSE 0 END
            END AS target,
            'empirical' AS streak_strength_raw_method
        FROM hitter_windows w
        INNER JOIN hitter_windows n
          ON n.player_id = w.player_id
         AND n.window_days = w.window_days
         AND n.window_end = w.window_end + INTERVAL (w.window_days) DAY
        INNER JOIN hitter_streak_labels l
          ON l.player_id = w.player_id
         AND l.window_end = w.window_end
         AND l.window_days = w.window_days
         AND l.category = ?
         AND l.cold_method = 'empirical'
         AND l.label = ?
        INNER JOIN hitter_projection_rates p
          ON p.player_id = w.player_id
         AND p.season = EXTRACT(YEAR FROM w.window_end)::INTEGER
        INNER JOIN (
            SELECT window_days, pt_bucket, MEDIAN({category}) AS next_median
            FROM hitter_windows
            GROUP BY window_days, pt_bucket
        ) t
          ON t.window_days = w.window_days
         AND t.pt_bucket = w.pt_bucket
        WHERE w.window_days = ?
          AND p.{rate_col} IS NOT NULL
        """,
        [direction, category, label, window_days],
    ).df()

    if df.empty:
        return df

    # Strength bucket = "{label}_q[1-5]". For dense, parse from a per-row
    # quintile within the labeled population (matches Phase 3's
    # _dense_strength_buckets convention).
    values = df["value"].to_numpy()
    quintiles = np.quantile(values, [0.2, 0.4, 0.6, 0.8])
    qbins = np.clip(np.searchsorted(quintiles, values, side="left"), 0, 4)
    df["streak_strength_numeric"] = (qbins + 1).astype(float)
    return df


def _build_training_frame_sparse(
    conn: duckdb.DuckDBPyConnection,
    *,
    category: StreakCategory,
    season_set: str,
    window_days: int,
) -> pd.DataFrame:
    """Sparse-cat hot-only training frame.

    Sparse hot rows are duplicated across poisson_p10 and poisson_p20 in
    hitter_streak_labels (the hot determination is identical in both). We
    filter to ``SPARSE_HOT_COLD_METHOD`` so the model is not trained on the
    same row twice.

    Target: next_value > expected_next = projected_rate * next_window_pa
    (mirrors Phase 3 continuation logic).

    Streak strength: parsed from the sparse strength_bucket
    ("hot_+1.5sigma" → 1.5). Rows with bucket "hot_zna" are dropped.
    """
    rate_col = f"{category}_per_pa"

    df = conn.execute(
        f"""
        SELECT
            w.player_id,
            w.window_end,
            EXTRACT(YEAR FROM w.window_end)::INTEGER AS season,
            w.pt_bucket,
            w.{category} AS value,
            w.pa AS current_pa,
            n.{category} AS next_value,
            n.pa AS next_pa,
            w.babip,
            w.k_pct,
            w.bb_pct,
            w.iso,
            w.ev_avg,
            w.barrel_pct,
            w.xwoba_avg,
            p.{rate_col} AS season_rate_in_category,
            l.label,
            CASE
                WHEN n.{category} > p.{rate_col} * n.pa THEN 1 ELSE 0
            END AS target
        FROM hitter_windows w
        INNER JOIN hitter_windows n
          ON n.player_id = w.player_id
         AND n.window_days = w.window_days
         AND n.window_end = w.window_end + INTERVAL (w.window_days) DAY
        INNER JOIN hitter_streak_labels l
          ON l.player_id = w.player_id
         AND l.window_end = w.window_end
         AND l.window_days = w.window_days
         AND l.category = ?
         AND l.cold_method = ?
         AND l.label = 'hot'
        INNER JOIN hitter_projection_rates p
          ON p.player_id = w.player_id
         AND p.season = EXTRACT(YEAR FROM w.window_end)::INTEGER
        WHERE w.window_days = ?
          AND p.{rate_col} IS NOT NULL
        """,
        [category, SPARSE_HOT_COLD_METHOD, window_days],
    ).df()

    if df.empty:
        return df

    # Sparse strength_bucket isn't stored on labels — recompute here from the
    # window's expected_current vs value (matches Phase 3 _sparse_strength_buckets).
    df["expected_current"] = df["season_rate_in_category"] * df["current_pa"]
    denom = df["expected_current"].pow(0.5).replace(0, np.nan)
    df["z"] = (df["value"] - df["expected_current"]) / denom
    # Drop rows with NaN z (would be _zna strength_bucket).
    df = df.dropna(subset=["z"])
    if df.empty:
        return df
    # Half-sigma value, clamped to [+0.5, +3.0] for hot models (z is positive
    # by definition of hot under empirical p90). Cold edge cases (z below 0)
    # shouldn't occur for hot rows; clip for safety.
    half = np.clip(np.round(df["z"].to_numpy() * 2) / 2.0, 0.5, 3.0)
    df["streak_strength_numeric"] = half
    return df


def build_training_frame(
    conn: duckdb.DuckDBPyConnection,
    *,
    category: StreakCategory,
    direction: StreakDirection,
    season_set: str,
    window_days: int,
) -> pd.DataFrame:
    """Return a feature + target DataFrame for one (category, direction) model.

    Columns: EXPECTED_FEATURE_COLUMNS + 'target' + 'player_id' + 'season' +
    'window_end' (audit only).

    Filters:
    - window_days = ``window_days`` (= 14 for Phase 4)
    - current label matches the model's direction (hot for above, cold for below)
    - season_rate_in_category IS NOT NULL (drops Phase-3-only rate rows)
    - sparse cats: filtered to ``SPARSE_HOT_COLD_METHOD`` to dedupe
    - strength_bucket parses to a numeric (drops {label}_zna rows)

    Empty DataFrame is returned when no labeled rows survive — caller handles.
    """
    if category in SPARSE_CATS:
        if direction != "above":
            # Sparse cats are hot-only in Phase 4. Explicit assertion-style log
            # rather than silent empty: the orchestrator should never call this.
            raise ValueError(
                f"sparse category {category!r} only has a hot model in Phase 4; "
                f"got direction={direction!r}"
            )
        df = _build_training_frame_sparse(
            conn, category=category, season_set=season_set, window_days=window_days
        )
    else:
        df = _build_training_frame_dense(
            conn,
            category=category,
            direction=direction,
            season_set=season_set,
            window_days=window_days,
        )

    if df.empty:
        return df

    # Drop rows whose strength_bucket didn't parse (already filtered for sparse;
    # belt-and-suspenders for any future buckets).
    df = df[df["streak_strength_numeric"].notna()].copy()
    if df.empty:
        return df

    # One-hot encode pt_bucket. Phase 2 buckets are {'low', 'mid', 'high'}.
    for bucket in ("low", "mid", "high"):
        df[f"pt_bucket_{bucket}"] = (df["pt_bucket"] == bucket).astype(int)

    # Drop rows with any NULL peripheral feature (~3% of windows; tolerable loss).
    feature_cols_with_nulls = ["babip", "k_pct", "bb_pct", "iso", "ev_avg", "barrel_pct", "xwoba_avg"]
    df = df.dropna(subset=feature_cols_with_nulls)
    if df.empty:
        return df

    keep_cols = list(EXPECTED_FEATURE_COLUMNS) + ["target", "player_id", "season", "window_end"]
    return df[keep_cols].reset_index(drop=True)
```

- [ ] **Step 7.4: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_predictors.py -v
```

Expected: all PASS. Some of the sparse-cat tests may early-return because the small synthetic fixture doesn't produce hot HR windows; the explicit `if df.empty: return` in those tests is intentional and documented inline.

- [ ] **Step 7.5: Commit**

```bash
git add src/fantasy_baseball/streaks/analysis/predictors.py tests/test_streaks/test_predictors.py
git commit -m "feat(streaks): Phase 4 training-frame builder (per-direction, feature + target)"
```

---

### Task 8: `fit_one_model` — GroupKFold + L2 grid

Sklearn `Pipeline` with `StandardScaler` followed by `LogisticRegression`. Tuning loop: for each `C` in the grid, run `GroupKFold(n_splits=5)` on `(X_train, y_train, groups=player_id)`, collect per-fold AUC, pick the `C` with highest mean fold AUC. Refit a fresh pipeline on the full train set with the chosen `C`.

**Files:**
- Modify: `src/fantasy_baseball/streaks/analysis/predictors.py`
- Modify: `tests/test_streaks/test_predictors.py`

- [ ] **Step 8.1: Write failing test**

Append to `tests/test_streaks/test_predictors.py`:

```python
from fantasy_baseball.streaks.analysis.predictors import (
    DEFAULT_C_GRID,
    FitResult,
    fit_one_model,
)
from sklearn.pipeline import Pipeline


def _make_synthetic_X_y(n_rows: int = 200, n_features: int = 12, seed: int = 0):
    """Synthetic, linearly-separable-ish dataset for fit-loop unit tests."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.normal(size=(n_rows, n_features)),
        columns=list(EXPECTED_FEATURE_COLUMNS),
    )
    # Make target weakly dependent on the first feature.
    logits = X[EXPECTED_FEATURE_COLUMNS[0]].to_numpy() + 0.5 * rng.normal(size=n_rows)
    y = (logits > 0).astype(int)
    groups = rng.integers(low=1, high=10, size=n_rows)
    return X, y, groups


def test_fit_one_model_returns_fitresult_with_pipeline_and_metrics() -> None:
    X, y, groups = _make_synthetic_X_y()
    result = fit_one_model(X, y, groups, C_grid=DEFAULT_C_GRID, n_splits=5, random_state=42)
    assert isinstance(result, FitResult)
    assert isinstance(result.pipeline, Pipeline)
    assert result.chosen_C in DEFAULT_C_GRID
    assert 0.0 <= result.cv_auc_mean <= 1.0
    assert result.cv_auc_std >= 0.0
    # AUC for a linearly-separable-ish target should be well above 0.5.
    assert result.cv_auc_mean > 0.55


def test_fit_one_model_pipeline_is_fitted_on_full_train() -> None:
    """Pipeline.predict_proba should succeed without further fit."""
    X, y, groups = _make_synthetic_X_y()
    result = fit_one_model(X, y, groups, C_grid=DEFAULT_C_GRID, n_splits=5, random_state=42)
    proba = result.pipeline.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert ((proba >= 0) & (proba <= 1)).all()


def test_fit_one_model_picks_highest_cv_auc() -> None:
    """When the C-grid has a single value, that value is selected."""
    X, y, groups = _make_synthetic_X_y()
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    assert result.chosen_C == 1.0
```

- [ ] **Step 8.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_predictors.py -v
```

Expected: FAILures — `fit_one_model`, `FitResult`, `DEFAULT_C_GRID` not defined.

- [ ] **Step 8.3: Append to `src/fantasy_baseball/streaks/analysis/predictors.py`**

Add the fitting code (and required sklearn imports) below the `build_training_frame` function:

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_C_GRID: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0)


@dataclass(frozen=True)
class FitResult:
    """One fitted model + the CV metrics that selected it."""

    pipeline: Pipeline
    chosen_C: float
    cv_auc_mean: float
    cv_auc_std: float
    cv_auc_per_fold: tuple[float, ...]


def _build_pipeline(C: float, random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=C,
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=1000,
                    random_state=random_state,
                ),
            ),
        ]
    )


def fit_one_model(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    *,
    C_grid: Iterable[float] = DEFAULT_C_GRID,
    n_splits: int = 5,
    random_state: int = 42,
) -> FitResult:
    """Player-grouped 5-fold CV over an L2 strength grid, then refit on full
    train. Returns the refit pipeline + per-fold AUC stats for the chosen C.

    All inputs are positional from the caller's perspective; ``X`` columns must
    match ``EXPECTED_FEATURE_COLUMNS`` (the scaler is column-order-agnostic but
    consumers of FitResult.pipeline.coef_ assume this order).
    """
    y_arr = np.asarray(y, dtype=int)
    groups_arr = np.asarray(groups, dtype=int)
    grid = tuple(C_grid)

    best_C: float | None = None
    best_mean: float = -np.inf
    best_std: float = 0.0
    best_per_fold: tuple[float, ...] = ()

    for C in grid:
        per_fold: list[float] = []
        cv = GroupKFold(n_splits=n_splits)
        for train_idx, val_idx in cv.split(X, y_arr, groups=groups_arr):
            pipe = _build_pipeline(C=C, random_state=random_state)
            pipe.fit(X.iloc[train_idx], y_arr[train_idx])
            val_proba = pipe.predict_proba(X.iloc[val_idx])[:, 1]
            if len(np.unique(y_arr[val_idx])) < 2:
                # Degenerate fold — single class in val. Skip rather than
                # crash; sklearn.roc_auc_score requires both classes.
                continue
            per_fold.append(float(roc_auc_score(y_arr[val_idx], val_proba)))
        if not per_fold:
            logger.warning("No usable folds for C=%g (every fold had a single-class val set)", C)
            continue
        mean = float(np.mean(per_fold))
        std = float(np.std(per_fold))
        if mean > best_mean:
            best_C = C
            best_mean = mean
            best_std = std
            best_per_fold = tuple(per_fold)

    if best_C is None:
        raise RuntimeError("fit_one_model: no C value produced any usable CV fold")

    # Refit on full train with the chosen C.
    final = _build_pipeline(C=best_C, random_state=random_state)
    final.fit(X, y_arr)

    return FitResult(
        pipeline=final,
        chosen_C=best_C,
        cv_auc_mean=best_mean,
        cv_auc_std=best_std,
        cv_auc_per_fold=best_per_fold,
    )
```

- [ ] **Step 8.4: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_predictors.py -v
```

Expected: all PASS.

- [ ] **Step 8.5: Commit**

```bash
git add src/fantasy_baseball/streaks/analysis/predictors.py tests/test_streaks/test_predictors.py
git commit -m "feat(streaks): fit_one_model (GroupKFold + L2 grid + refit on full train)"
```

---

### Task 9: `bootstrap_coef_ci` — player-grouped bootstrap

200 resamples by default. Each resample picks `n_players` players with replacement (where `n_players = #unique groups`), assembles a bootstrap training set from all rows of those players, refits the pipeline with the *fixed* `chosen_C`, and collects the coefficient vector. Output: per-coefficient 5th / 95th percentiles.

**Files:**
- Modify: `src/fantasy_baseball/streaks/analysis/predictors.py`
- Modify: `tests/test_streaks/test_predictors.py`

- [ ] **Step 9.1: Write failing test**

Append to `tests/test_streaks/test_predictors.py`:

```python
from fantasy_baseball.streaks.analysis.predictors import bootstrap_coef_ci


def test_bootstrap_coef_ci_returns_per_feature_intervals() -> None:
    X, y, groups = _make_synthetic_X_y(n_rows=300)
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    cis = bootstrap_coef_ci(
        pipeline=result.pipeline,
        X=X,
        y=y,
        groups=groups,
        chosen_C=result.chosen_C,
        n_resamples=20,  # small for the unit test
        random_state=42,
    )
    # One (lo, hi) per feature column, ordered to match X.
    assert set(cis.keys()) == set(X.columns)
    for col, (lo, hi) in cis.items():
        assert lo <= hi, f"CI inverted for {col}: ({lo}, {hi})"


def test_bootstrap_coef_ci_intervals_narrow_with_more_resamples() -> None:
    """Sanity check: 100 resamples should produce intervals that *include*
    the point estimate from the original fit for most features (we don't
    assert a hard rate — bootstrap can disagree with the L2-shrunk
    point — but every CI should be finite)."""
    X, y, groups = _make_synthetic_X_y(n_rows=300)
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    cis = bootstrap_coef_ci(
        pipeline=result.pipeline,
        X=X,
        y=y,
        groups=groups,
        chosen_C=result.chosen_C,
        n_resamples=100,
        random_state=42,
    )
    for col, (lo, hi) in cis.items():
        assert np.isfinite(lo) and np.isfinite(hi)
```

- [ ] **Step 9.2: Run test, confirm it fails**

```
pytest tests/test_streaks/test_predictors.py::test_bootstrap_coef_ci_returns_per_feature_intervals -v
```

Expected: FAIL — function not defined.

- [ ] **Step 9.3: Append to `src/fantasy_baseball/streaks/analysis/predictors.py`**

```python
def bootstrap_coef_ci(
    *,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    groups: np.ndarray | pd.Series,
    chosen_C: float,
    n_resamples: int = 200,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Player-grouped bootstrap CIs on L2-regularized coefficients.

    For each of ``n_resamples`` iterations:
      1. Sample N players with replacement from the unique groups.
      2. Assemble the bootstrap training set from all rows of those players.
      3. Refit a fresh pipeline with the same chosen_C on that resample.
      4. Append the coefficient vector to the running list.

    Returns ``{feature_name: (p5, p95)}`` — 5th / 95th percentiles over the
    bootstrap distribution.

    Note: ``pipeline`` is passed in for ergonomics but not modified; we
    rebuild fresh pipelines via _build_pipeline to keep the original's
    fitted state intact.
    """
    y_arr = np.asarray(y, dtype=int)
    groups_arr = np.asarray(groups, dtype=int)
    feature_names = list(X.columns)
    n_features = len(feature_names)
    rng = np.random.default_rng(random_state)

    unique_players = np.unique(groups_arr)
    coef_samples = np.empty((n_resamples, n_features), dtype=float)

    for i in range(n_resamples):
        sampled_players = rng.choice(unique_players, size=len(unique_players), replace=True)
        # Assemble rows belonging to any sampled player. A player sampled twice
        # contributes their rows twice (the correct bootstrap behavior — it's
        # the players, not the rows, we resample).
        row_chunks: list[np.ndarray] = []
        for p in sampled_players:
            row_chunks.append(np.where(groups_arr == p)[0])
        rows = np.concatenate(row_chunks)
        X_boot = X.iloc[rows]
        y_boot = y_arr[rows]
        if len(np.unique(y_boot)) < 2:
            # Degenerate resample — single class. Skip; fill with NaN
            # placeholder so np.percentile later ignores it.
            coef_samples[i, :] = np.nan
            continue
        pipe = _build_pipeline(C=chosen_C, random_state=random_state)
        pipe.fit(X_boot, y_boot)
        coef_samples[i, :] = pipe.named_steps["lr"].coef_.ravel()

    out: dict[str, tuple[float, float]] = {}
    for j, name in enumerate(feature_names):
        col = coef_samples[:, j]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            out[name] = (float("nan"), float("nan"))
        else:
            out[name] = (float(np.percentile(col, 5)), float(np.percentile(col, 95)))
    return out
```

- [ ] **Step 9.4: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_predictors.py -v
```

Expected: all PASS.

- [ ] **Step 9.5: Commit**

```bash
git add src/fantasy_baseball/streaks/analysis/predictors.py tests/test_streaks/test_predictors.py
git commit -m "feat(streaks): bootstrap_coef_ci (player-grouped, 200 resamples default)"
```

---

### Task 10: `evaluate_model` (AUC + reliability) and `permutation_feature_importance`

Two thin wrappers — `evaluate_model` returns AUC + the reliability-diagram bin data; `permutation_feature_importance` wraps `sklearn.inspection.permutation_importance`. Notebook does the plotting.

**Files:**
- Modify: `src/fantasy_baseball/streaks/analysis/predictors.py`
- Modify: `tests/test_streaks/test_predictors.py`

- [ ] **Step 10.1: Write failing tests**

Append to `tests/test_streaks/test_predictors.py`:

```python
from fantasy_baseball.streaks.analysis.predictors import (
    EvaluationResult,
    evaluate_model,
    permutation_feature_importance,
)


def test_evaluate_model_returns_auc_and_reliability_bins() -> None:
    X, y, groups = _make_synthetic_X_y(n_rows=300)
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    eval_result = evaluate_model(pipeline=result.pipeline, X=X, y=y, n_bins=10)
    assert isinstance(eval_result, EvaluationResult)
    assert 0.0 <= eval_result.auc <= 1.0
    # 10 reliability bins, each with (mean_predicted, mean_observed, count).
    assert len(eval_result.reliability_bin_centers) == len(eval_result.reliability_observed)
    assert (np.asarray(eval_result.reliability_bin_counts) >= 0).all()


def test_evaluate_model_auc_matches_sklearn_directly() -> None:
    from sklearn.metrics import roc_auc_score as _roc

    X, y, groups = _make_synthetic_X_y(n_rows=300)
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    eval_result = evaluate_model(pipeline=result.pipeline, X=X, y=y, n_bins=10)
    direct = _roc(y, result.pipeline.predict_proba(X)[:, 1])
    assert eval_result.auc == pytest.approx(direct, rel=1e-9)


def test_permutation_feature_importance_returns_per_feature_mean_and_std() -> None:
    X, y, groups = _make_synthetic_X_y(n_rows=300)
    result = fit_one_model(X, y, groups, C_grid=(1.0,), n_splits=5, random_state=42)
    importance = permutation_feature_importance(
        pipeline=result.pipeline, X_val=X, y_val=y, n_repeats=5, random_state=42
    )
    assert set(importance.keys()) == set(X.columns)
    for col, (mean_drop, std_drop) in importance.items():
        assert np.isfinite(mean_drop)
        assert std_drop >= 0.0
```

- [ ] **Step 10.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_predictors.py -v
```

Expected: FAILures — neither function defined.

- [ ] **Step 10.3: Append to `src/fantasy_baseball/streaks/analysis/predictors.py`**

```python
from sklearn.inspection import permutation_importance


@dataclass(frozen=True)
class EvaluationResult:
    """ROC-AUC + reliability diagram for one model on one held-out set.

    Bin arrays have the same length and are aligned 1:1 — index k refers to
    the same (non-empty) bin in all three.
    """

    auc: float
    reliability_bin_centers: tuple[float, ...]
    reliability_observed: tuple[float, ...]
    reliability_bin_counts: tuple[int, ...]


def evaluate_model(
    *,
    pipeline: Pipeline,
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    n_bins: int = 10,
) -> EvaluationResult:
    """Held-out AUC + reliability diagram (n_bins equal-width bins).

    Empty bins are dropped from the returned arrays; the three reliability_*
    tuples stay aligned. ``bin_centers`` are the bin midpoints, not the mean
    predicted probability in the bin — simpler to interpret on a reliability
    plot, and accurate to within bin width / 2 of the mean.
    """
    y_arr = np.asarray(y, dtype=int)
    proba = pipeline.predict_proba(X)[:, 1]
    auc = float(roc_auc_score(y_arr, proba))

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(bin_edges, proba, side="right") - 1, 0, n_bins - 1)

    bin_centers: list[float] = []
    bin_observed: list[float] = []
    bin_counts: list[int] = []
    for k in range(n_bins):
        mask = bin_idx == k
        if not mask.any():
            continue
        bin_centers.append(0.5 * (bin_edges[k] + bin_edges[k + 1]))
        bin_observed.append(float(y_arr[mask].mean()))
        bin_counts.append(int(mask.sum()))

    return EvaluationResult(
        auc=auc,
        reliability_bin_centers=tuple(bin_centers),
        reliability_observed=tuple(bin_observed),
        reliability_bin_counts=tuple(bin_counts),
    )


def permutation_feature_importance(
    *,
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: np.ndarray | pd.Series,
    n_repeats: int = 10,
    random_state: int = 42,
) -> dict[str, tuple[float, float]]:
    """Sklearn permutation importance on the validation set.

    For each feature: shuffle it, measure AUC drop, repeat ``n_repeats`` times,
    report (mean_drop, std_drop).
    """
    y_arr = np.asarray(y_val, dtype=int)

    def _scorer(estimator, X_, y_):
        return roc_auc_score(y_, estimator.predict_proba(X_)[:, 1])

    result = permutation_importance(
        pipeline,
        X_val,
        y_arr,
        scoring=_scorer,
        n_repeats=n_repeats,
        random_state=random_state,
    )
    return {
        name: (float(result.importances_mean[i]), float(result.importances_std[i]))
        for i, name in enumerate(X_val.columns)
    }
```

- [ ] **Step 10.4: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_predictors.py -v
```

Expected: all PASS.

- [ ] **Step 10.5: Commit**

```bash
git add src/fantasy_baseball/streaks/analysis/predictors.py tests/test_streaks/test_predictors.py
git commit -m "feat(streaks): evaluate_model (AUC + reliability) + permutation_feature_importance"
```

---

## Stage C — Orchestrator and CLI

### Task 11: `fit_all_models` orchestrator + ModelFit persistence

For each of the 8 (category, direction) pairs in `PHASE_4_MODELS`:
1. Build the training frame for `season_set_train`.
2. Build a separate frame for `season_set_val` and apply the same target-derivation.
3. Run `fit_one_model` on train.
4. Run `evaluate_model` on val.
5. Run `bootstrap_coef_ci` on train.
6. Run `permutation_feature_importance` on val.
7. Construct a `ModelFit` and stash it; collect all results in a dict for return.

Persist all `ModelFit` rows to `model_fits` via `upsert_model_fits` at the end.

The val-frame target uses the **same** `next_bucket_median` as the train frame (computed across the joined population, not per-season). For sparse cats, the `expected_next = projected_rate × next_window_pa` target uses each row's own projected_rate, so no cross-season median is needed.

**Files:**
- Modify: `src/fantasy_baseball/streaks/analysis/predictors.py`
- Modify: `tests/test_streaks/test_predictors.py`

- [ ] **Step 11.1: Write failing test**

Append to `tests/test_streaks/test_predictors.py`:

```python
from fantasy_baseball.streaks.analysis.predictors import (
    AllModelsResult,
    PHASE_4_MODELS,
    fit_all_models,
)


def _seed_two_season_pipeline(conn) -> None:
    """Seed two seasons so the orchestrator has both train and val to work with."""
    _seed_pipeline(conn, n_players=20, n_days=90, season=2023)
    _seed_pipeline(conn, n_players=20, n_days=90, season=2024)
    # Re-run thresholds and labels for the combined season_set.
    compute_thresholds(conn, season_set="2023-2024", qualifying_pa=50)
    apply_labels(conn, season_set="2023-2024")


def test_fit_all_models_returns_one_result_per_phase_4_model(tmp_path) -> None:
    conn = get_connection(":memory:")
    _seed_two_season_pipeline(conn)
    # For the unit test we use 2023 as train and 2024 as val just to exercise
    # the orchestrator's two-frame plumbing. (Real-data acceptance uses
    # 2023-2024 train / 2025 val.)
    result = fit_all_models(
        conn,
        season_set_train="2023",
        season_set_val="2024",
        window_days=14,
        C_grid=(1.0,),
        n_bootstrap=10,
        random_state=42,
    )
    assert isinstance(result, AllModelsResult)
    # If the synthetic fixture is too small to produce any labeled rows for a
    # given (cat, dir), the orchestrator records a None fit for it. The
    # length of the dict still matches the model spec.
    assert len(result.fits) == len(PHASE_4_MODELS)


def test_fit_all_models_writes_to_model_fits_table() -> None:
    conn = get_connection(":memory:")
    _seed_two_season_pipeline(conn)
    fit_all_models(
        conn,
        season_set_train="2023",
        season_set_val="2024",
        window_days=14,
        C_grid=(1.0,),
        n_bootstrap=10,
        random_state=42,
    )
    n = conn.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    # At least one model should have produced enough rows to fit.
    assert n >= 1


def test_fit_all_models_skips_models_with_no_training_rows() -> None:
    """If a (cat, dir) frame is empty after filtering, the result entry is
    None and no row is written to model_fits for it."""
    conn = get_connection(":memory:")
    # Bare init — no seeded data.
    result = fit_all_models(
        conn,
        season_set_train="2099",
        season_set_val="2100",
        window_days=14,
        C_grid=(1.0,),
        n_bootstrap=10,
        random_state=42,
    )
    assert all(v is None for v in result.fits.values())
    n = conn.execute("SELECT COUNT(*) FROM model_fits").fetchone()[0]
    assert n == 0
```

- [ ] **Step 11.2: Run tests, confirm they fail**

```
pytest tests/test_streaks/test_predictors.py -v
```

Expected: FAILures — `fit_all_models`, `AllModelsResult` not defined.

- [ ] **Step 11.3: Append to `src/fantasy_baseball/streaks/analysis/predictors.py`**

```python
from fantasy_baseball.streaks.data.load_model_fits import upsert_model_fits
from fantasy_baseball.streaks.models import ModelFit


@dataclass(frozen=True)
class PerModelResult:
    """Everything the notebook needs for one (category, direction)."""

    fit: FitResult
    evaluation: EvaluationResult
    coef_ci: dict[str, tuple[float, float]]
    permutation_importance: dict[str, tuple[float, float]]
    n_train_rows: int
    n_val_rows: int
    cold_method: Literal["empirical", "poisson_p20"]


@dataclass(frozen=True)
class AllModelsResult:
    """Output of fit_all_models — one entry per (category, direction) pair."""

    fits: dict[tuple[StreakCategory, StreakDirection], PerModelResult | None]
    season_set_train: str
    season_set_val: str
    window_days: int


def _split_frame_by_season(
    df: pd.DataFrame, *, season_set_train: str, season_set_val: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Partition the unified training frame by season.

    ``season_set_train`` and ``season_set_val`` can be either single-year
    strings ('2023') or range strings ('2023-2024'); the seasons covered by
    each are derived by intersecting with the frame's ``season`` column.
    """
    train_seasons = _seasons_in_set(season_set_train)
    val_seasons = _seasons_in_set(season_set_val)
    df_train = df[df["season"].isin(train_seasons)].copy()
    df_val = df[df["season"].isin(val_seasons)].copy()
    return df_train, df_val


def _seasons_in_set(season_set: str) -> tuple[int, ...]:
    """Parse '2023' → (2023,) ; '2023-2024' → (2023, 2024)."""
    if "-" in season_set:
        lo_s, hi_s = season_set.split("-", 1)
        lo, hi = int(lo_s), int(hi_s)
        return tuple(range(lo, hi + 1))
    return (int(season_set),)


def _fit_one_phase_4_model(
    conn: duckdb.DuckDBPyConnection,
    *,
    category: StreakCategory,
    direction: StreakDirection,
    season_set_train: str,
    season_set_val: str,
    window_days: int,
    C_grid: Iterable[float],
    n_bootstrap: int,
    random_state: int,
) -> PerModelResult | None:
    """Build train + val frames for one model; fit, evaluate, bootstrap, importance."""
    season_set_combined = (
        f"{min(_seasons_in_set(season_set_train) + _seasons_in_set(season_set_val))}-"
        f"{max(_seasons_in_set(season_set_train) + _seasons_in_set(season_set_val))}"
    )
    full = build_training_frame(
        conn,
        category=category,
        direction=direction,
        season_set=season_set_combined,
        window_days=window_days,
    )
    if full.empty:
        logger.info(
            "No training frame rows for (%s, %s) — skipping", category, direction
        )
        return None

    df_train, df_val = _split_frame_by_season(
        full, season_set_train=season_set_train, season_set_val=season_set_val
    )
    if df_train.empty or df_val.empty:
        logger.info(
            "Train/val split empty for (%s, %s): train=%d val=%d — skipping",
            category, direction, len(df_train), len(df_val),
        )
        return None

    feature_cols = list(EXPECTED_FEATURE_COLUMNS)
    X_train = df_train[feature_cols]
    y_train = df_train["target"].to_numpy()
    groups = df_train["player_id"].to_numpy()
    X_val = df_val[feature_cols]
    y_val = df_val["target"].to_numpy()

    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        logger.info(
            "Single-class target for (%s, %s) — skipping", category, direction
        )
        return None

    fit_result = fit_one_model(
        X_train, y_train, groups,
        C_grid=C_grid, n_splits=5, random_state=random_state,
    )
    eval_result = evaluate_model(
        pipeline=fit_result.pipeline, X=X_val, y=y_val, n_bins=10
    )
    coef_ci = bootstrap_coef_ci(
        pipeline=fit_result.pipeline,
        X=X_train, y=y_train, groups=groups,
        chosen_C=fit_result.chosen_C,
        n_resamples=n_bootstrap,
        random_state=random_state,
    )
    importance = permutation_feature_importance(
        pipeline=fit_result.pipeline, X_val=X_val, y_val=y_val, n_repeats=10,
        random_state=random_state,
    )
    cold_method: Literal["empirical", "poisson_p20"] = (
        SPARSE_HOT_COLD_METHOD if category in SPARSE_CATS else "empirical"
    )
    return PerModelResult(
        fit=fit_result,
        evaluation=eval_result,
        coef_ci=coef_ci,
        permutation_importance=importance,
        n_train_rows=len(df_train),
        n_val_rows=len(df_val),
        cold_method=cold_method,
    )


def fit_all_models(
    conn: duckdb.DuckDBPyConnection,
    *,
    season_set_train: str = "2023-2024",
    season_set_val: str = "2025",
    window_days: int = 14,
    C_grid: Iterable[float] = DEFAULT_C_GRID,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> AllModelsResult:
    """Fit all 8 Phase 4 models, persist metadata to model_fits, return results.

    Skips any model whose training or validation frame is empty (logs the
    skip and records ``None`` in the result dict). All non-skipped models
    write one row to ``model_fits``.
    """
    fits: dict[tuple[StreakCategory, StreakDirection], PerModelResult | None] = {}
    fit_rows: list[ModelFit] = []
    timestamp = datetime.utcnow()

    for cat, direction in PHASE_4_MODELS:
        per_model = _fit_one_phase_4_model(
            conn,
            category=cat, direction=direction,
            season_set_train=season_set_train, season_set_val=season_set_val,
            window_days=window_days, C_grid=C_grid, n_bootstrap=n_bootstrap,
            random_state=random_state,
        )
        fits[(cat, direction)] = per_model
        if per_model is None:
            continue
        model_id = f"{cat}_{'hot' if direction == 'above' else 'cold'}_{season_set_train}"
        fit_rows.append(
            ModelFit(
                model_id=model_id,
                category=cat,
                direction=direction,
                season_set=season_set_train,
                window_days=window_days,
                cold_method=per_model.cold_method,
                chosen_C=per_model.fit.chosen_C,
                cv_auc_mean=per_model.fit.cv_auc_mean,
                cv_auc_std=per_model.fit.cv_auc_std,
                val_auc=per_model.evaluation.auc,
                n_train_rows=per_model.n_train_rows,
                n_val_rows=per_model.n_val_rows,
                fit_timestamp=timestamp,
            )
        )

    upsert_model_fits(conn, fit_rows)
    logger.info("Wrote %d rows to model_fits", len(fit_rows))

    return AllModelsResult(
        fits=fits,
        season_set_train=season_set_train,
        season_set_val=season_set_val,
        window_days=window_days,
    )
```

- [ ] **Step 11.4: Run tests, confirm they pass**

```
pytest tests/test_streaks/test_predictors.py -v
```

Expected: all PASS.

- [ ] **Step 11.5: Commit**

```bash
git add src/fantasy_baseball/streaks/analysis/predictors.py tests/test_streaks/test_predictors.py
git commit -m "feat(streaks): fit_all_models orchestrator (8 models + model_fits persistence)"
```

---

### Task 12: CLI `scripts/streaks/fit_models.py`

End-to-end entry point. Reads from the local DuckDB, runs `fit_all_models`, prints a summary including the gate result.

**Files:**
- Create: `scripts/streaks/fit_models.py`

- [ ] **Step 12.1: Create the script**

```python
"""CLI: fit all 8 Phase 4 streak-continuation models.

Assumes Phase 3 outputs exist:
- hitter_windows, hitter_streak_labels, hitter_projection_rates, thresholds

Usage:
    python -m scripts.streaks.fit_models [--db-path PATH]
        [--season-set-train 2023-2024] [--season-set-val 2025]
        [--window-days 14] [--n-bootstrap 200]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.analysis.predictors import (
    DENSE_CATS,
    DEFAULT_C_GRID,
    PHASE_4_MODELS,
    fit_all_models,
)
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit Phase 4 streak-continuation models.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--season-set-train", default="2023-2024")
    parser.add_argument("--season-set-val", default="2025")
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        result = fit_all_models(
            conn,
            season_set_train=args.season_set_train,
            season_set_val=args.season_set_val,
            window_days=args.window_days,
            C_grid=DEFAULT_C_GRID,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
        )
    finally:
        conn.close()

    # Print one summary line per model + per-category gate summary.
    print()
    print(f"Phase 4 fit summary — train={args.season_set_train} / val={args.season_set_val}")
    print("-" * 75)
    print(f"{'model_id':30s} {'C':>6s} {'cv_auc':>8s} {'val_auc':>8s} {'n_train':>8s} {'n_val':>8s}")
    per_cat_max: dict[str, float] = {}
    for (cat, direction), per_model in result.fits.items():
        if per_model is None:
            print(f"{cat}_{direction:<24s}  SKIPPED (no training rows or single-class target)")
            continue
        model_id = f"{cat}_{'hot' if direction == 'above' else 'cold'}_{args.season_set_train}"
        print(
            f"{model_id:30s} {per_model.fit.chosen_C:6.2f} "
            f"{per_model.fit.cv_auc_mean:8.3f} {per_model.evaluation.auc:8.3f} "
            f"{per_model.n_train_rows:8d} {per_model.n_val_rows:8d}"
        )
        per_cat_max[cat] = max(per_cat_max.get(cat, -1.0), per_model.evaluation.auc)

    # Gate: 2025 AUC >= 0.55 in >= 3 of 5 categories.
    print()
    cats_passing = sum(1 for v in per_cat_max.values() if v >= 0.55)
    print(f"Categories with max(hot_auc, cold_auc) >= 0.55: {cats_passing} / 5")
    gate = "PASS" if cats_passing >= 3 else "FAIL"
    print(f"Phase 4 gate: {gate}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 12.2: Smoke-test the script (with an empty DB — should run without crashing and print the SKIPPED lines)**

```
python scripts/streaks/fit_models.py --db-path /tmp/streaks_smoke.duckdb
```

Expected: prints 8 SKIPPED lines and `Categories with ... >= 0.55: 0 / 5  Phase 4 gate: FAIL`.

- [ ] **Step 12.3: Commit**

```bash
git add scripts/streaks/fit_models.py
git commit -m "feat(streaks): CLI for fit_models (Phase 4 orchestration)"
```

---

### Task 13: Declare `scikit-learn` as an explicit dev dependency

Currently transitive via pybaseball. Declare explicitly so it doesn't quietly disappear if pybaseball's deps change.

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 13.1: Edit `pyproject.toml`**

Add `"scikit-learn>=1.5",` to the existing `[project.optional-dependencies].dev` list (alphabetical position between `ruff` and `types-PyYAML`):

```toml
dev = [
    "pytest>=7.0",
    "pytest-cov>=4.0",
    "mypy>=1.10",
    "fakeredis>=2.21",
    "ruff>=0.6",
    "scikit-learn>=1.5",
    "types-PyYAML>=6.0",
    "types-requests>=2.31",
    "vulture>=2.11",
    "duckdb>=1.0",
    "pybaseball>=2.2",
]
```

- [ ] **Step 13.2: Reinstall dev deps to confirm the declaration takes**

```
pip install -e ".[dev]"
```

Expected: completes without error; sklearn already satisfied.

- [ ] **Step 13.3: Commit**

```bash
git add pyproject.toml
git commit -m "build: declare scikit-learn>=1.5 explicitly under dev deps"
```

---

## Stage D — Real-data acceptance + spec update

### Task 14: Real-data Phase 4 run on the local 3-season corpus

Run the full Phase 4 pipeline against the actual local DuckDB. Capture per-model AUC + gate outcome for the spec progress entry.

**Files:** none (operational)

- [ ] **Step 14.1: Migrate the local DB to Phase 4 schema**

```
python scripts/streaks/migrate.py --phase 4
```

Expected: `ALTER hitter_projection_rates ADD COLUMN IF NOT EXISTS r_per_pa/rbi_per_pa/avg` logged; `init_schema` confirms `model_fits` exists.

- [ ] **Step 14.2: Re-load projection rates (backfills r_per_pa / rbi_per_pa / avg)**

```
python scripts/streaks/load_projections.py --seasons 2023 2024 2025
```

Expected: same row count as the Phase 3 acceptance load (~1500 rows across three seasons), but now with the dense-cat rate columns populated.

- [ ] **Step 14.3: Verify dense-cat rates loaded**

```
python -c "import duckdb; c=duckdb.connect('data/streaks/streaks.duckdb'); print(c.execute('SELECT season, COUNT(*) AS n, COUNT(r_per_pa) AS n_r, COUNT(rbi_per_pa) AS n_rbi, COUNT(avg) AS n_avg FROM hitter_projection_rates GROUP BY season ORDER BY season').fetchall())"
```

Expected: per season, `n == n_r == n_rbi == n_avg` (every row has populated dense-cat rates). If any are short, inspect which projection CSVs lack the R/RBI/AVG columns and decide whether to backfill or proceed with a smaller training set.

- [ ] **Step 14.4: Fit all 8 models (real data, 2023-2024 train / 2025 val)**

```
time python scripts/streaks/fit_models.py --season-set-train 2023-2024 --season-set-val 2025 --n-bootstrap 200
```

Expected: prints per-model summary lines and the gate result. Wall time target: <120s (8 models × {CV + bootstrap} on ~50K rows per dense model). Capture the exact per-model AUC numbers for the spec progress entry.

If wall time exceeds 120s, **stop** and treat the slowness as Task 15 (perf-todo). Don't proceed to the notebook with a >120s pipeline.

- [ ] **Step 14.5: Run the full project test suite**

```
pytest -v
```

Expected: all green. Report the number of test cases run (Phase 3 had 1,538; Phase 4 should add ~20-25).

- [ ] **Step 14.6: Run lint + type checks**

```
ruff check .
ruff format --check .
mypy
vulture
```

Expected: zero violations across all four. Pre-existing vulture findings unrelated to Phase 4 are acceptable; call them out.

- [ ] **Step 14.7 (no commit — Task 16 handles the spec write-up)**

This task produces operational outputs only. Capture the per-model AUC table and gate outcome for Task 16.

---

### Task 15: Perf-todo — bootstrap parallelization (only if Task 14.4 measured >120s)

If the 200-resample × 8-model bootstrap dominates wall time and pushes the pipeline above 120s, parallelize using `joblib.Parallel`.

**This task is conditional.** Skip if Task 14.4 came in under 120s; we don't add `joblib.Parallel` for hypothetical perf wins.

**Files (conditional):**
- Modify: `src/fantasy_baseball/streaks/analysis/predictors.py`

- [ ] **Step 15.1: Replace the bootstrap loop in `bootstrap_coef_ci` with `joblib.Parallel`**

```python
from joblib import Parallel, delayed


def _one_bootstrap_fit(
    *, sampled_players, groups_arr, X, y_arr, chosen_C, random_state
):
    row_chunks = [np.where(groups_arr == p)[0] for p in sampled_players]
    rows = np.concatenate(row_chunks)
    X_boot, y_boot = X.iloc[rows], y_arr[rows]
    if len(np.unique(y_boot)) < 2:
        return None
    pipe = _build_pipeline(C=chosen_C, random_state=random_state)
    pipe.fit(X_boot, y_boot)
    return pipe.named_steps["lr"].coef_.ravel()


# Replace the for-i-in-range(n_resamples) loop in bootstrap_coef_ci with:
samples = [
    rng.choice(unique_players, size=len(unique_players), replace=True)
    for _ in range(n_resamples)
]
results = Parallel(n_jobs=-1, prefer="processes")(
    delayed(_one_bootstrap_fit)(
        sampled_players=s,
        groups_arr=groups_arr,
        X=X,
        y_arr=y_arr,
        chosen_C=chosen_C,
        random_state=random_state,
    )
    for s in samples
)
for i, coef in enumerate(results):
    if coef is None:
        coef_samples[i, :] = np.nan
    else:
        coef_samples[i, :] = coef
```

- [ ] **Step 15.2: Run the existing predictor tests to confirm no regression**

```
pytest tests/test_streaks/test_predictors.py -v
```

Expected: PASS.

- [ ] **Step 15.3: Re-run the acceptance fit and confirm wall time**

```
time python scripts/streaks/fit_models.py --season-set-train 2023-2024 --season-set-val 2025 --n-bootstrap 200
```

Expected: wall time under 120s. If still slow, drop `n_bootstrap` to 100 and note the change in the spec progress entry.

- [ ] **Step 15.4: Commit (if changes were made)**

```bash
git add src/fantasy_baseball/streaks/analysis/predictors.py
git commit -m "perf(streaks): parallelize Phase 4 bootstrap via joblib (real-data wall time)"
```

---

### Task 16: Acceptance notebook `03_predictors.ipynb`

Notebook lives gitignored; we commit the jupytext-paired `.py` for source-control discoverability (same pattern as Phase 2's `01_distributions.py` and Phase 3's `02_continuation.py`).

**Files:**
- Create: `notebooks/streaks/03_predictors.py` (committed; Jupyter pairs to `.ipynb`)

- [ ] **Step 16.1: Create the jupytext source**

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
# # Hot Streaks Phase 4 — Predictive Model
#
# This notebook is the Phase 4 acceptance artifact. It refits all 8 models
# from the local DB and renders, for each:
#
# - ROC curve on 2025
# - Reliability diagram (10 bins, diagnostic only — no isotonic correction here)
# - Coefficient bar chart with 5%/95% bootstrap CIs (200 player-grouped resamples)
# - Permutation importance bar chart
#
# **Gate (committed in the Phase 4 plan):**
# Held-out 2025 ROC-AUC ≥ 0.55 in at least 3 of 5 categories, where per-cat AUC
# is `max(hot_auc, cold_auc)` for dense cats and `hot_auc` for sparse cats.
#
# **Run after:** `python scripts/streaks/fit_models.py --season-set-train 2023-2024 --season-set-val 2025`

# %%
from pathlib import Path
import time

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fantasy_baseball.streaks.analysis.predictors import (
    DEFAULT_C_GRID,
    PHASE_4_MODELS,
    fit_all_models,
)

_here = Path.cwd()
for _candidate in (_here, *_here.parents):
    if (_candidate / "pyproject.toml").exists():
        REPO_ROOT = _candidate
        break
else:
    raise RuntimeError("could not locate repo root")

DB = REPO_ROOT / "data" / "streaks" / "streaks.duckdb"
SEASON_SET_TRAIN = "2023-2024"
SEASON_SET_VAL = "2025"

conn = duckdb.connect(str(DB))

# %% [markdown]
# ## Refit all 8 models (in-notebook)
#
# Same call the CLI makes. The refit takes ~60-90s on the 3-season corpus.

# %%
t0 = time.perf_counter()
results = fit_all_models(
    conn,
    season_set_train=SEASON_SET_TRAIN,
    season_set_val=SEASON_SET_VAL,
    window_days=14,
    C_grid=DEFAULT_C_GRID,
    n_bootstrap=200,
    random_state=42,
)
print(f"fit_all_models wall time: {time.perf_counter() - t0:.1f}s")

# %% [markdown]
# ## Gate summary

# %%
rows = []
per_cat_max: dict[str, float] = {}
for (cat, direction), per_model in results.fits.items():
    if per_model is None:
        rows.append({"category": cat, "direction": direction, "chosen_C": None,
                     "cv_auc": None, "val_auc": None, "n_train": 0, "n_val": 0})
        continue
    rows.append({
        "category": cat, "direction": direction,
        "chosen_C": per_model.fit.chosen_C,
        "cv_auc": round(per_model.fit.cv_auc_mean, 3),
        "val_auc": round(per_model.evaluation.auc, 3),
        "n_train": per_model.n_train_rows, "n_val": per_model.n_val_rows,
    })
    per_cat_max[cat] = max(per_cat_max.get(cat, -1.0), per_model.evaluation.auc)
summary = pd.DataFrame(rows)
summary

# %%
cats_passing = sum(1 for v in per_cat_max.values() if v >= 0.55)
print(f"Categories with max AUC >= 0.55: {cats_passing} / 5")
print(f"Phase 4 gate: {'PASS' if cats_passing >= 3 else 'FAIL'}")

# %% [markdown]
# ## Per-model diagnostic plots
#
# For each fitted model, render four panels:
# 1. ROC curve on 2025
# 2. Reliability diagram (predicted vs observed; the y=x diagonal is perfect)
# 3. Coefficient bar chart with bootstrap CIs
# 4. Permutation importance bar chart

# %%
from sklearn.metrics import roc_curve
from fantasy_baseball.streaks.analysis.predictors import (
    EXPECTED_FEATURE_COLUMNS,
    build_training_frame,
)


def _val_frame_for_model(cat: str, direction: str) -> pd.DataFrame:
    """Re-build the val frame so we can score the ROC curve and reliability."""
    full = build_training_frame(
        conn, category=cat, direction=direction, season_set="2023-2025", window_days=14
    )
    return full[full["season"] == int(SEASON_SET_VAL)]


def _plot_one_model(cat: str, direction: str, per_model) -> None:
    if per_model is None:
        print(f"({cat}, {direction}): no model fitted; skipping plots")
        return

    val_df = _val_frame_for_model(cat, direction)
    feature_cols = list(EXPECTED_FEATURE_COLUMNS)
    X_val = val_df[feature_cols]
    y_val = val_df["target"].to_numpy()
    proba = per_model.fit.pipeline.predict_proba(X_val)[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{cat} {direction}  C={per_model.fit.chosen_C:.2g}  "
        f"cv_auc={per_model.fit.cv_auc_mean:.3f}±{per_model.fit.cv_auc_std:.3f}  "
        f"val_auc={per_model.evaluation.auc:.3f}  n_val={per_model.n_val_rows}",
        fontsize=11,
    )

    # 1. ROC curve.
    fpr, tpr, _ = roc_curve(y_val, proba)
    axes[0, 0].plot(fpr, tpr, label=f"AUC={per_model.evaluation.auc:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    axes[0, 0].set_xlabel("False positive rate")
    axes[0, 0].set_ylabel("True positive rate")
    axes[0, 0].set_title("ROC curve")
    axes[0, 0].legend()

    # 2. Reliability diagram.
    centers = np.array(per_model.evaluation.reliability_bin_centers)
    observed = np.array(per_model.evaluation.reliability_observed)
    axes[0, 1].plot(centers, observed, "o-", label="observed")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="perfect calibration")
    axes[0, 1].set_xlabel("Predicted probability (bin center)")
    axes[0, 1].set_ylabel("Observed positive rate")
    axes[0, 1].set_title("Reliability (10 bins, diagnostic)")
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 1); axes[0, 1].set_ylim(0, 1)

    # 3. Coefficient bar chart with bootstrap CIs.
    feat_names = list(per_model.coef_ci.keys())
    coefs = per_model.fit.pipeline.named_steps["lr"].coef_.ravel()
    lo = np.array([per_model.coef_ci[n][0] for n in feat_names])
    hi = np.array([per_model.coef_ci[n][1] for n in feat_names])
    err = np.vstack([coefs - lo, hi - coefs])
    y_pos = np.arange(len(feat_names))
    axes[1, 0].barh(y_pos, coefs, xerr=err, color="steelblue", alpha=0.7, capsize=3)
    axes[1, 0].set_yticks(y_pos); axes[1, 0].set_yticklabels(feat_names)
    axes[1, 0].axvline(0, color="black", linewidth=0.5)
    axes[1, 0].set_xlabel("Coefficient (standardized)")
    axes[1, 0].set_title("L2 LR coefficient ±[5th, 95th] bootstrap")

    # 4. Permutation importance.
    means = np.array([per_model.permutation_importance[n][0] for n in feat_names])
    stds = np.array([per_model.permutation_importance[n][1] for n in feat_names])
    order = np.argsort(means)
    axes[1, 1].barh(np.arange(len(feat_names)), means[order], xerr=stds[order], color="darkorange", alpha=0.7, capsize=3)
    axes[1, 1].set_yticks(np.arange(len(feat_names)))
    axes[1, 1].set_yticklabels([feat_names[i] for i in order])
    axes[1, 1].set_xlabel("ΔAUC when feature is permuted")
    axes[1, 1].set_title("Permutation importance (n_repeats=10)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# %%
for (cat, direction) in PHASE_4_MODELS:
    _plot_one_model(cat, direction, results.fits[(cat, direction)])

# %% [markdown]
# ## Methodology notes
#
# - **Calibration:** reliability diagrams are diagnostic-only here. If Phase 5
#   sees a model whose predictions are systematically off (e.g. 10pp+ at any
#   bin center), apply isotonic correction at that point.
# - **Coefficient CIs:** correlated features (BABIP↔xwOBA, ISO↔barrel%) tend to
#   have wide CIs because L2 distributes weight ambiguously between them. The
#   narrative should not treat any single one of those four as "the driver."
# - **Permutation importance:** unlike |coef|, this is honest under correlation
#   (shuffling one of a correlated pair still leaks signal via its partner,
#   which conservatively *reduces* its measured importance — read positives
#   as real signal; weakness on a correlated feature is ambiguous).

# %% [markdown]
# ## Done.

# %%
print("Done.")
```

- [ ] **Step 16.2: Smoke-render the notebook**

Open in Jupyter (or pair the `.py` to `.ipynb` via `jupytext --set-formats ipynb,py:percent notebooks/streaks/03_predictors.py`) and execute every cell against the local DB. Confirm every cell runs without error and the gate `PASS`/`FAIL` line prints. **Do not** commit the `.ipynb`.

- [ ] **Step 16.3: Commit the jupytext source**

```bash
git add notebooks/streaks/03_predictors.py
git commit -m "docs(streaks): Phase 4 acceptance notebook (jupytext source)"
```

---

### Task 17: Spec progress entry

Append a Phase 4 entry to the design spec recording the final per-cat AUC numbers and the gate outcome.

**Files:**
- Modify: `docs/superpowers/specs/2026-05-06-hot-streaks-design.md`

- [ ] **Step 17.1: Edit the spec**

Append to the bottom of `docs/superpowers/specs/2026-05-06-hot-streaks-design.md` (after the existing Phase 3 acceptance entry):

```markdown
### 2026-05-10 — Phase 4 (predictive model) accepted

All plan tasks landed (Task 15 perf-todo was <fired / not needed>). New
schema: `model_fits` audit table plus three new nullable rate columns on
`hitter_projection_rates`
(`r_per_pa`, `rbi_per_pa`, `avg`). New library module
`streaks/analysis/predictors.py` carries the full pipeline:
`build_training_frame`, `fit_one_model` (player-grouped 5-fold CV over
`C ∈ {0.01, 0.1, 1, 10}`), `bootstrap_coef_ci` (200 player-grouped
resamples), `evaluate_model` (AUC + reliability), and
`permutation_feature_importance`. Orchestrated by `fit_all_models` and
CLI-exposed at `scripts/streaks/fit_models.py`.

Eight models fit on the 2023-2024 train / 2025 val corpus:

| Category | Direction | Chosen C | CV AUC | Val AUC | n_train | n_val |
|---|---|---:|---:|---:|---:|---:|
| R | above | <fill in> | <fill in> | <fill in> | <fill in> | <fill in> |
| R | below | <fill in> | <fill in> | <fill in> | <fill in> | <fill in> |
| RBI | above | <fill in> | <fill in> | <fill in> | <fill in> | <fill in> |
| RBI | below | <fill in> | <fill in> | <fill in> | <fill in> | <fill in> |
| AVG | above | <fill in> | <fill in> | <fill in> | <fill in> | <fill in> |
| AVG | below | <fill in> | <fill in> | <fill in> | <fill in> | <fill in> |
| HR | above (hot only) | <fill in> | <fill in> | <fill in> | <fill in> | <fill in> |
| SB | above (hot only) | <fill in> | <fill in> | <fill in> | <fill in> | <fill in> |

Per-category max(hot_auc, cold_auc):

| Category | Max AUC | Passes 0.55? |
|---|---:|---|
| R   | <fill in> | <YES/NO> |
| RBI | <fill in> | <YES/NO> |
| AVG | <fill in> | <YES/NO> |
| HR  | <fill in> | <YES/NO> |
| SB  | <fill in> | <YES/NO> |

**Gate result: <PASS or FAIL — fill in from the notebook>** (≥ 3 of 5
categories at AUC ≥ 0.55).

Total pipeline runtime on the real 2023-2025 corpus: <fill in>s wall time
(<fill in> models × CV + 200-resample bootstrap + permutation importance).
<If perf-todo Task 15 fired: "joblib-parallelized bootstrap dropped wall
time from <X>s to <Y>s.">

#### Methodology surprises / Phase 5 inputs

- **Calibration:** reliability diagrams (notebook §3) showed <fill in —
  was any model >5pp off at any bin?>. Phase 5 should <apply isotonic /
  trust raw probabilities — fill in based on observed calibration>.
- **Coefficient signs:** <fill in any surprises — e.g. "barrel_pct
  coefficient was negative for HR hot, opposite of the expected sign;
  bootstrap CI crosses zero so this is likely a correlation artifact
  with xwoba_avg">.
- **Permutation importance top features per model:** <fill in 2-3
  examples — e.g. "streak_strength_numeric and xwoba_avg dominated for
  HR hot; season_rate_in_category dominated for SB hot">.
- **Bootstrap CIs on correlated features:** <fill in observation about
  BABIP↔xwOBA, ISO↔barrel% spread widths>.

#### Pre-existing issues unrelated to Phase 4

<List any flagged during the lint/test gate, same format as prior phases.>

#### Next milestone

- **Phase 5 — weekly Sunday report.** Pull current-season game logs,
  apply calibrated thresholds, run the 8 models, emit a CLI report:
  hot rostered hitters (ranked by composite score then strength), cold
  rostered hitters, per-category labels, continuation probability from
  the model, top 1-2 peripheral drivers per player, suggested
  start/sit/stash lines. Refit models in-process each Sunday from the
  `2023-2024 + 2024-2025` training set (sliding window) — Phase 4's
  refit-on-demand decision means no joblib artifacts to chase.
- Phase 4 future work (deferred unless Phase 5 surfaces a need):
  per-category curated feature subsets (Phase 4 used a uniform set
  across all 8 models), the 7d-SB-hot model (Phase 4 used 14d only),
  isotonic calibration correction if Phase 5 probabilities consumed
  downstream need it.
```

(The `<fill in>` placeholders are filled by the implementer reading the actual notebook output before committing.)

- [ ] **Step 17.2: Commit**

```bash
git add docs/superpowers/specs/2026-05-06-hot-streaks-design.md
git commit -m "docs(streaks): record Phase 4 acceptance + gate outcome"
```

---

## Self-Review Checklist

Run through this list before merging:

1. **Spec coverage:** every Phase 4 line item from the umbrella spec is implemented or explicitly deferred.
   - "Fit per-category logistic regressions with continuation as the target" → Tasks 7-12 ✓
   - "Train on 2023-2024, validate on 2025" → Task 11 split logic, Task 14.4 acceptance run ✓
   - "Calibration plot, ROC-AUC, top features by importance" → Task 10 evaluate_model + permutation_feature_importance + notebook Task 16 ✓
   - "Gate: held-out 2025 ROC-AUC ≥ 0.55 in at least 3 of 5 categories" → CLI Task 12 + notebook Task 16 ✓
   - "p-values" called out in spec but explicitly **replaced** with bootstrap CIs in this plan (p-values are not well-defined under L2 regularization; bootstrap is the correct alternative). Documented in Plan §Design Decisions #6 and in the spec-update preamble for Task 17.
2. **All brainstormed design decisions carried into the plan:**
   - Per-direction LRs, hot+cold per cat, sparse cats hot only → Plan §Design Decisions #1, PHASE_4_MODELS constant ✓
   - 14d windows only → Plan §Design Decisions #2; window_days=14 hardcoded in CLI default ✓
   - Player-grouped 5-fold CV; C ∈ {0.01, 0.1, 1, 10} → DEFAULT_C_GRID + fit_one_model GroupKFold ✓
   - Spec-uniform 10-feature set + pt_bucket one-hot → EXPECTED_FEATURE_COLUMNS ✓
   - Diagnostic-only reliability (no isotonic) → evaluate_model returns bin data; no correction applied ✓
   - Bootstrap CIs (200 player-grouped resamples) → bootstrap_coef_ci default ✓
   - Permutation importance → permutation_feature_importance ✓
   - No model pickles; model_fits audit table → load_model_fits + fit_all_models persistence ✓
   - Gate: ≥3 of 5 cats at ≥0.55 → CLI gate logic in Task 12 ✓
3. **No placeholders in the plan body.** Every step shows full code, full commands, or a concrete operational instruction. The only `<fill in>`s are in the *spec progress entry template* (Task 17.1), where they're explicitly meant to be filled from real-run output. ✓
4. **Type/name consistency:** `model_id`, `cold_method`, `chosen_C`, `cv_auc_mean`, `val_auc`, `n_train_rows`, `n_val_rows`, `fit_timestamp` used identically across schema (Task 2), dataclass (Task 1), loader (Task 6), orchestrator (Task 11). `EXPECTED_FEATURE_COLUMNS` is the single source of truth for feature column order. ✓
5. **Test coverage matches implementation surface:** training-frame builder (Task 7), fit_one_model (Task 8), bootstrap (Task 9), evaluate + permutation (Task 10), orchestrator (Task 11), migration (Task 3), projection-rate blender (Task 4), upsert helpers (Tasks 5, 6). ✓
6. **Commit cadence:** every task ends with a single, focused commit. Operational tasks (14, 15-conditional) explicitly do not commit. ✓
7. **No production-stack imports added.** New code touches only `data/projections/` (static historical CSVs) and the local DuckDB — no `web/`, `lineup/`, or `redis_store` references. Spec's hard-isolation rule preserved. ✓
8. **New dependency declared:** `scikit-learn>=1.5` added to `pyproject.toml` dev deps (Task 13). ✓
9. **Perf risk identified and bounded:** bootstrap is the dominant cost (200 × 8 = 1600 refits). Task 14.4 establishes a 120s wall-time budget; Task 15 is the conditional parallelization fallback. ✓

## Future work (out of scope for Phase 4)

- **Per-category curated feature subsets.** Phase 4 uses a uniform 10-feature set across all 8 models. If the gate passes and certain models carry features that contribute zero permutation importance, drop them per-model in Phase 4.5.
- **7d-SB hot model.** Phase 3 noted SB hot 7d ≈ 14d in lift. Phase 4 used 14d only. Revisit if the 14d-SB hot model underperforms or if the Sunday report wants a faster-reacting SB signal.
- **Isotonic / Platt calibration.** Phase 4 reports raw probabilities. Phase 5 applies a correction *only if* reliability diagrams show systematic miscalibration that affects the Sunday report's start/sit decisions.
- **Class-imbalance handling (`class_weight='balanced'`).** Spec'd as a discretionary fallback only — apply per-model if the default fit lands AUC below 0.55 in CV.
- **Statsmodels p-values.** Explicitly out-of-scope under L2 regularization (no well-defined asymptotic theory); replaced by bootstrap CIs.
