"""Pre-blend projection data quality checks."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fantasy_baseball.data.projections import HITTING_COUNTING_COLS, PITCHING_COUNTING_COLS
from fantasy_baseball.utils.name_utils import normalize_name

# Thresholds for cross-system outlier detection
EXCLUDE_THRESHOLD = 0.20  # System median < 20% of consensus → exclude
WARN_THRESHOLD = 0.50     # System median deviates > 50% from consensus → warn


@dataclass
class QualityReport:
    """Results of pre-blend projection quality checks."""
    warnings: list[str] = field(default_factory=list)
    exclusions: dict[str, set[str]] = field(default_factory=dict)
    missing_players: dict[str, list[str]] = field(default_factory=dict)


def check_projection_quality(
    system_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    roster_names: set[str] | None = None,
) -> QualityReport:
    """Run quality checks on per-system projection DataFrames before blending.

    Args:
        system_dfs: {system_name: (hitters_df, pitchers_df)} raw DataFrames.
        roster_names: Normalized names of all rostered players. If None,
            roster coverage check is skipped.

    Returns:
        QualityReport with warnings, exclusions, and missing players.
    """
    report = QualityReport()

    if len(system_dfs) < 2:
        return report

    _check_stat_outliers(system_dfs, report)
    _check_player_counts(system_dfs, report)
    if roster_names:
        _check_roster_coverage(system_dfs, roster_names, report)

    return report


def _check_stat_outliers(
    system_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    report: QualityReport,
) -> None:
    """Detect per-stat outliers across systems."""
    pass


def _check_player_counts(
    system_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    report: QualityReport,
) -> None:
    """Warn if a system has dramatically fewer players than others."""
    pass


def _check_roster_coverage(
    system_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    roster_names: set[str],
    report: QualityReport,
) -> None:
    """Warn about rostered players missing from projection systems."""
    pass
