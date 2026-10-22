"""Dataclass models for the streaks DuckDB tables, and the streak score payload.

Each table dataclass corresponds 1:1 to a DuckDB table in `streaks/data/schema.py`.
Field declaration order is the table's column order; the loaders derive their
SQL column tuples from `dataclasses.fields(...)` so this file is the single
source of truth for column names and order.

The score payload types at the bottom (`Report` and what it holds) are what the
refresh writes to `CacheKey.STREAK_SCORES` (via `streaks.payload`) and what the
Render dashboard reads back. They live here, not beside the DuckDB code that
builds them (`inference`, `reports.sunday`), because this module imports no
DuckDB: the request path can import the one definition of the payload (#175).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Literal

PtBucket = Literal["low", "mid", "high"]
StreakCategory = Literal["hr", "r", "rbi", "sb", "avg"]
StreakLabel = Literal["hot", "cold", "neutral"]
ColdMethod = Literal["empirical", "poisson_p10", "poisson_p20"]
StreakDirection = Literal["above", "below"]


@dataclass(frozen=True, slots=True)
class QualifiedHitter:
    """One entry from the MLB Stats API ≥min_pa leaderboard.

    Not persisted directly — used as the producer→orchestrator handoff for
    "which players should we fetch game logs for this season."
    """

    player_id: int
    name: str
    team: str | None
    pa: int


@dataclass(frozen=True, slots=True)
class HitterGame:
    """One game of hitter counting stats. Maps to `hitter_games` row.

    PK is (player_id, game_pk). ``game_pk`` is the MLB Stats API gamePk
    integer — unique per game, so it disambiguates doubleheaders that
    share a date. ``date`` stays as a non-PK column for query convenience.

    Captures every box-score component the streaks project needs for rate
    stats (BABIP/ISO from b2/b3/sf), refined walk rate (uBB% from ibb),
    PA-identity reconciliation (pa = ab + bb + hbp + sf + sh + ci), and
    plausible Phase 3+ context (cs for SB attempts, gidp for luck signal,
    is_home for park splits).
    """

    player_id: int
    game_pk: int
    name: str
    team: str | None
    season: int
    date: date
    pa: int
    ab: int
    h: int
    hr: int
    r: int
    rbi: int
    sb: int
    bb: int
    k: int
    b2: int
    b3: int
    sf: int
    hbp: int
    ibb: int
    cs: int
    gidp: int
    sh: int
    ci: int
    is_home: bool


@dataclass(frozen=True, slots=True)
class HitterStatcastPA:
    """One terminal-PA row from Baseball Savant. Maps to `hitter_statcast_pa` row.

    PK is (player_id, date, pa_index). ``pa_index`` is now derived after
    sorting by ``at_bat_number`` within (batter, game_date), so it is
    chronologically stable across re-fetches.
    """

    player_id: int
    date: date
    pa_index: int
    event: str | None
    launch_speed: float | None
    launch_angle: float | None
    estimated_woba_using_speedangle: float | None
    launch_speed_angle: int | None
    at_bat_number: int | None
    bb_type: str | None
    estimated_ba_using_speedangle: float | None
    hit_distance_sc: float | None


@dataclass(frozen=True, slots=True)
class HitterWindow:
    """Rolling-window aggregate. Maps to `hitter_windows` row.

    PK is (player_id, window_end, window_days). Populated in Phase 2.
    """

    player_id: int
    window_end: date
    window_days: int
    pa: int
    hr: int
    r: int
    rbi: int
    sb: int
    avg: float | None
    babip: float | None
    k_pct: float | None
    bb_pct: float | None
    iso: float | None
    ev_avg: float | None
    barrel_pct: float | None
    xwoba_avg: float | None
    pt_bucket: PtBucket


@dataclass(frozen=True, slots=True)
class Threshold:
    """One calibrated percentile threshold. Maps to `thresholds` row.

    PK is (season_set, category, window_days, pt_bucket). Populated in Phase 2.
    """

    season_set: str
    category: StreakCategory
    window_days: int
    pt_bucket: PtBucket
    p10: float
    p90: float


@dataclass(frozen=True, slots=True)
class HitterStreakLabel:
    """One hot/cold/neutral label for a (player, window, category, cold_method).

    PK is (player_id, window_end, window_days, category, cold_method).

    `cold_method` distinguishes the rule that produced the cold determination:
    - 'empirical' for dense cats (R/RBI/AVG): uses calibrated p10 from `thresholds`.
    - 'poisson_p10' / 'poisson_p20' for sparse cats (HR/SB): uses skill-relative
      `Poisson(proj_rate * window_PA).ppf(0.1 | 0.2)`.

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


@dataclass(frozen=True, slots=True)
class ModelFit:
    """One row of the Phase 4 ``model_fits`` audit + Phase B pipeline-state table.

    PK is ``model_id`` (synthetic string like 'hr_hot_2023-2024').

    - ``cold_method`` is the partition of `hitter_streak_labels` the training
      rows were drawn from. Dense cats use 'empirical'; sparse hot uses
      'poisson_p20' (deduplication choice — see plan design decision #10).
    - ``chosen_C`` is the L2 strength selected by GroupKFold over a fixed grid.
    - ``cv_auc_mean`` / ``cv_auc_std`` are the per-fold AUC stats for that C.
    - ``val_auc`` is the single-shot 2025 ROC-AUC — the gate metric.
    - ``n_train_rows`` / ``n_val_rows`` are post-filter row counts (drop
      strength=zna, drop NULL season_rate, drop NULL peripherals).

    Phase B added the persisted-pipeline fields so the dashboard refresh can
    reconstruct fitted ``StandardScaler`` + ``LogisticRegression`` pipelines
    from this table without retraining. All five are nullable for backward
    compatibility with Phase 4 rows that pre-date Phase B (which left them
    NULL); the loader treats NULL as "this row cannot be reconstructed."

    - ``feature_columns`` — the in-order feature names the model was trained
      on. Aligned 1:1 with ``coef`` / ``scaler_mean`` / ``scaler_scale``.
    - ``coef`` — the LogisticRegression coefficient vector (one entry per
      feature; binary classifier so this is a single row).
    - ``intercept`` — the LogisticRegression intercept (scalar).
    - ``scaler_mean`` / ``scaler_scale`` — StandardScaler params per feature.
    - ``dense_quintile_cutoffs`` — 4-tuple of quintile breakpoints over the
      training population's category values; required to recompute
      ``streak_strength_numeric`` at inference time for dense cats. NULL for
      sparse cats (which use a Poisson z-score formula instead).
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
    feature_columns: tuple[str, ...] | None = None
    coef: tuple[float, ...] | None = None
    intercept: float | None = None
    scaler_mean: tuple[float, ...] | None = None
    scaler_scale: tuple[float, ...] | None = None
    dense_quintile_cutoffs: tuple[float, float, float, float] | None = None


# --------------------------------------------------------------------- #
#     Streak score payload (CacheKey.STREAK_SCORES): see streaks.payload #
# --------------------------------------------------------------------- #


@dataclass(frozen=True)
class Driver:
    """One peripheral feature attributed as driving a prediction.

    ``feature`` is the column name from :data:`EXPECTED_FEATURE_COLUMNS`.
    ``z_score`` is the StandardScaler-transformed feature value — signed,
    so positive means the player's feature is above the training mean.
    """

    feature: str
    z_score: float


@dataclass(frozen=True)
class PlayerCategoryScore:
    """Scored prediction for one (player, category).

    - ``label`` is the empirical/Poisson label for the player's most recent
      14d window. For sparse cats, we use the ``poisson_p20`` cold partition
      (matches training).
    - ``probability`` is ``P(continuation)`` from the model — ``None`` when
      the label is neutral, OR when the label is non-neutral but no model
      was trained for that ``(cat, direction)`` (sparse cats are hot-only
      in Phase 4, so ``sb cold`` ends up here).
    - ``probability_baserate`` is the stratum base rate for the same event
      (from ``continuation_rates``, keyed by category/direction/pt_bucket) —
      the unconditional P(continuation) for *any* player in that stratum.
      ``probability - probability_baserate`` is the lift the streak adds;
      raw probabilities are not comparable across categories/directions
      because base rates range ~0.15-0.81. ``None`` when ``probability``
      is None or the rates table has no row for the stratum.
    - ``drivers`` is empty when ``probability`` is ``None``.
    - ``window_end`` is the date the scored 14d window ends on (most
      recent in DB). Reports surface this in the header so the reader
      knows how stale the signal is.
    """

    player_id: int
    category: StreakCategory
    label: StreakLabel
    probability: float | None
    drivers: tuple[Driver, ...]
    window_end: date | None
    probability_baserate: float | None = None


@dataclass(frozen=True)
class ReportRow:
    """One player's row in the rendered report.

    ``scores`` is keyed by category for fast cell lookup in the renderer;
    missing entries render as ``—``.

    ``days_since_last_game`` is set (an int) only when the player's
    most-recent window is stale (> STALE_TOLERANCE_DAYS before the run
    date) -- i.e. he is inactive; ``None`` means active/recent.
    """

    name: str
    positions: tuple[str, ...]
    player_id: int
    composite: int
    scores: dict[StreakCategory, PlayerCategoryScore]
    max_probability: float
    days_since_last_game: int | None = None

    @property
    def sort_key(self) -> tuple[float, float, str]:
        """Roster ordering: composite desc, max_probability desc, name asc."""
        return (-self.composite, -self.max_probability, self.name.lower())

    @property
    def fa_sort_key(self) -> tuple[float, float, str]:
        """FA ordering: |composite| desc, max_probability desc, name asc."""
        return (-abs(self.composite), -self.max_probability, self.name.lower())


@dataclass(frozen=True)
class DriverLine:
    """One driver-attribution line under the tables."""

    player_name: str
    category: StreakCategory
    label: StreakLabel
    probability: float
    drivers: tuple[Driver, ...]


@dataclass(frozen=True)
class Report:
    """Full Sunday report payload — input to both renderers."""

    report_date: date
    window_end: date | None
    team_name: str
    league_id: int
    season_set_train: str
    roster_rows: tuple[ReportRow, ...]
    fa_rows: tuple[ReportRow, ...]
    driver_lines: tuple[DriverLine, ...]
    skipped: tuple[str, ...] = field(default_factory=tuple)
