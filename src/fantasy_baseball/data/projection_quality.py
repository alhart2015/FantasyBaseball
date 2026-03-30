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
    """Detect per-stat outliers across systems.

    For each counting stat, computes per-system medians among players who have
    >0 in that stat in at least one system. If a system's median is <20% of
    the consensus (median of medians), that stat is excluded for that system.
    If deviation is >50%, a warning is logged without exclusion.
    """
    systems = list(system_dfs.keys())

    for player_type, stat_cols, df_idx in [
        ("hitter", HITTING_COUNTING_COLS, 0),
        ("pitcher", PITCHING_COUNTING_COLS, 1),
    ]:
        sys_frames = {}
        for sys_name in systems:
            df = system_dfs[sys_name][df_idx]
            if not df.empty:
                sys_frames[sys_name] = df

        if len(sys_frames) < 2:
            continue

        for stat in stat_cols:
            present_systems = {s: df for s, df in sys_frames.items() if stat in df.columns}
            if len(present_systems) < 2:
                continue

            # Find players with >0 in this stat in ANY system (for sparse stats like SV)
            all_keys = set()
            for df in present_systems.values():
                if "fg_id" in df.columns and df["fg_id"].notna().all():
                    key_col = "fg_id"
                else:
                    key_col = "name"
                nonzero = df[df[stat].fillna(0) > 0][key_col]
                all_keys.update(nonzero.values)

            # Check for all-NaN columns before the sparse-stat filter
            # (a system with all-NaN won't contribute to all_keys but should still be flagged)
            all_nan_systems = {
                s for s, df in present_systems.items() if df[stat].isna().all()
            }

            if not all_keys and not all_nan_systems:
                continue

            # Compute per-system median among the >0 player pool
            sys_medians = {}
            for sys_name, df in present_systems.items():
                if sys_name in all_nan_systems:
                    sys_medians[sys_name] = float("nan")
                    continue
                if "fg_id" in df.columns and df["fg_id"].notna().all():
                    key_col = "fg_id"
                else:
                    key_col = "name"
                pool = df[df[key_col].isin(all_keys)]
                vals = pool[stat].fillna(0)
                sys_medians[sys_name] = float(vals.median()) if len(vals) > 0 else 0.0

            # Compute per-system consensus as the median of all OTHER systems' medians
            # (leave-one-out) so that a single outlier system doesn't skew its own reference
            valid_medians = {
                s: v for s, v in sys_medians.items() if not np.isnan(v)
            }

            for sys_name, sys_median in sys_medians.items():
                df = present_systems[sys_name]
                col_all_nan = sys_name in all_nan_systems

                if col_all_nan:
                    report.warnings.append(
                        f"EXCLUDE: {sys_name} {player_type} {stat} all NaN"
                    )
                    if sys_name not in report.exclusions:
                        report.exclusions[sys_name] = set()
                    report.exclusions[sys_name].add(stat)
                    continue

                # Consensus excludes this system (leave-one-out)
                other_medians = [v for s, v in valid_medians.items() if s != sys_name]
                if not other_medians:
                    continue
                consensus = float(np.median(other_medians))
                if consensus == 0:
                    continue

                ratio = sys_median / consensus

                if ratio < EXCLUDE_THRESHOLD:
                    report.warnings.append(
                        f"EXCLUDE: {sys_name} {player_type} {stat} "
                        f"median ({sys_median:.1f}) is <{EXCLUDE_THRESHOLD*100:.0f}% of consensus ({consensus:.1f})"
                    )
                    if sys_name not in report.exclusions:
                        report.exclusions[sys_name] = set()
                    report.exclusions[sys_name].add(stat)
                elif abs(ratio - 1.0) > WARN_THRESHOLD:
                    direction = "above" if ratio > 1 else "below"
                    report.warnings.append(
                        f"WARNING: {sys_name} {player_type} {stat} median ({sys_median:.1f}) "
                        f"is {abs(ratio - 1.0)*100:.0f}% {direction} consensus ({consensus:.1f})"
                    )


def _check_player_counts(
    system_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    report: QualityReport,
) -> None:
    """Warn if a system has dramatically fewer players than others."""
    for player_type, df_idx in [("hitter", 0), ("pitcher", 1)]:
        counts = {}
        for sys_name, dfs in system_dfs.items():
            df = dfs[df_idx]
            counts[sys_name] = len(df) if not df.empty else 0

        if not counts or all(c == 0 for c in counts.values()):
            continue

        nonzero_counts = [c for c in counts.values() if c > 0]
        if not nonzero_counts:
            continue
        median_count = float(np.median(nonzero_counts))

        for sys_name, count in counts.items():
            if count > 0 and count < median_count * 0.5:
                report.warnings.append(
                    f"WARNING: {sys_name} player count ({count} {player_type}s) is "
                    f"below 50% of median ({median_count:.0f}) — possible bad export"
                )


def _check_roster_coverage(
    system_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    roster_names: set[str],
    report: QualityReport,
) -> None:
    """Warn about rostered players missing from projection systems."""
    systems = list(system_dfs.keys())

    sys_name_sets: dict[str, set[str]] = {}
    for sys_name, (hitters, pitchers) in system_dfs.items():
        names = set()
        for df in (hitters, pitchers):
            if not df.empty and "name" in df.columns:
                names.update(df["name"].apply(normalize_name).values)
        sys_name_sets[sys_name] = names

    for player_name in sorted(roster_names):
        missing_from = [
            sys_name for sys_name in systems
            if player_name not in sys_name_sets[sys_name]
        ]
        if not missing_from:
            continue

        report.missing_players[player_name] = missing_from

        if len(missing_from) == len(systems):
            report.warnings.append(
                f"WARNING: {player_name} missing from ALL projection systems — "
                f"no projection available"
            )
        else:
            report.warnings.append(
                f"WARNING: {player_name} missing from {', '.join(missing_from)}"
            )
