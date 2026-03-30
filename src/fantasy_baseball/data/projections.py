import numpy as np
import pandas as pd
from pathlib import Path
from .fangraphs import load_projection_set, _find_file
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher

# Counting stats to blend directly (weighted average)
HITTING_COUNTING_COLS: list[str] = ["r", "hr", "rbi", "sb", "h", "ab", "pa"]
PITCHING_COUNTING_COLS: list[str] = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]


def validate_projections_dir(
    projections_dir: Path, systems: list[str]
) -> None:
    """Validate that the projections directory exists and contains expected CSV files.

    Raises FileNotFoundError with an actionable message if the directory is
    missing or if no projection files can be found for the requested systems.
    """
    if not projections_dir.exists():
        raise FileNotFoundError(
            f"Projections directory not found: {projections_dir}\n"
            f"\n"
            f"To fix this:\n"
            f"  1. Create the directory: mkdir -p {projections_dir}\n"
            f"  2. Download projection CSVs from FanGraphs:\n"
            f"     https://www.fangraphs.com/projections\n"
            f"  3. Export hitter and pitcher CSVs for each system ({', '.join(systems)})\n"
            f"  4. Save them as e.g. steamer-hitters.csv, steamer-pitchers.csv"
        )

    if not projections_dir.is_dir():
        raise FileNotFoundError(
            f"Projections path exists but is not a directory: {projections_dir}"
        )

    csv_files = list(projections_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(
            f"No CSV files found in {projections_dir}\n"
            f"\n"
            f"To fix this:\n"
            f"  1. Download projection CSVs from FanGraphs:\n"
            f"     https://www.fangraphs.com/projections\n"
            f"  2. Export hitter and pitcher CSVs for each system ({', '.join(systems)})\n"
            f"  3. Save them as e.g. steamer-hitters.csv, steamer-pitchers.csv"
        )

    # Check each requested system has at least one file (hitters or pitchers)
    missing_systems = []
    for system in systems:
        hit_file = _find_file(projections_dir, system, "hitters")
        pit_file = _find_file(projections_dir, system, "pitchers")
        if hit_file is None and pit_file is None:
            missing_systems.append(system)

    if missing_systems:
        found_files = [f.name for f in csv_files]
        raise FileNotFoundError(
            f"No projection files found for system(s): {', '.join(missing_systems)}\n"
            f"\n"
            f"Directory {projections_dir} contains: {', '.join(found_files)}\n"
            f"\n"
            f"Expected files like:\n"
            + "\n".join(
                f"  - {s}-hitters.csv / {s}-pitchers.csv"
                for s in missing_systems
            )
            + f"\n"
            f"\n"
            f"To fix this:\n"
            f"  1. Download the missing projections from FanGraphs:\n"
            f"     https://www.fangraphs.com/projections\n"
            f"  2. Select each system and export hitter + pitcher CSVs\n"
            f"  3. Save them in {projections_dir}"
        )


def blend_projections(
    projections_dir: Path,
    systems: list[str],
    weights: dict[str, float] | None = None,
    roster_names: set[str] | None = None,
    progress_cb=None,
) -> tuple[pd.DataFrame, pd.DataFrame, "QualityReport | None"]:
    """Blend multiple projection systems into weighted averages.

    Counting stats are blended directly. Rate stats (AVG, ERA, WHIP)
    are recomputed from blended component stats.

    Runs pre-blend quality checks when 2+ systems are loaded. Excludes
    stat columns flagged as outliers (e.g., a system with all-zero SV).

    Returns (hitters_df, pitchers_df, quality_report). quality_report is
    None if fewer than 2 systems were loaded.
    """
    from fantasy_baseball.data.projection_quality import check_projection_quality

    validate_projections_dir(projections_dir, systems)

    if weights is None:
        weights = {s: 1.0 / len(systems) for s in systems}

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    # Load all systems
    system_dfs: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    all_hitters: list[pd.DataFrame] = []
    all_pitchers: list[pd.DataFrame] = []

    for system in systems:
        hitters, pitchers = load_projection_set(projections_dir, system)
        system_dfs[system] = (hitters, pitchers)
        w = weights.get(system, 0)
        if not hitters.empty:
            hitters = hitters.copy()
            hitters["_weight"] = w
            hitters["_system"] = system
            all_hitters.append(hitters)
        if not pitchers.empty:
            pitchers = pitchers.copy()
            pitchers["_weight"] = w
            pitchers["_system"] = system
            all_pitchers.append(pitchers)

    # Run quality checks
    report = None
    if len(system_dfs) >= 2:
        report = check_projection_quality(system_dfs, roster_names)
        if progress_cb:
            for warning in report.warnings:
                progress_cb(f"QUALITY: {warning}")

        # Apply exclusions: zero out excluded stat columns so they don't contribute
        if report.exclusions:
            for df_list, stat_source in [
                (all_hitters, HITTING_COUNTING_COLS),
                (all_pitchers, PITCHING_COUNTING_COLS),
            ]:
                for df in df_list:
                    system = df["_system"].iloc[0] if not df.empty else None
                    if system and system in report.exclusions:
                        excluded = report.exclusions[system]
                        for stat in excluded:
                            if stat in df.columns and stat in stat_source:
                                df[stat] = float("nan")

    # Clean up _system column before blending
    for df in all_hitters + all_pitchers:
        if "_system" in df.columns:
            df.drop(columns=["_system"], inplace=True)

    blended_hitters = _blend_hitters(all_hitters)
    blended_pitchers = _blend_pitchers(all_pitchers)
    return blended_hitters, blended_pitchers, report


def _blend_players(
    dfs: list[pd.DataFrame],
    counting_cols: list[str],
    player_type: str,
) -> pd.DataFrame:
    """Vectorized projection blending shared by hitters and pitchers.

    Instead of iterating per group with pandas Series arithmetic,
    this pre-multiplies all counting stats by their normalized weight
    and uses a single groupby().sum() to aggregate.
    """
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    # Group by fg_id when available (robust against name variations
    # across systems, e.g. accented vs ASCII). Fall back to name.
    group_col = (
        "fg_id"
        if "fg_id" in combined.columns and combined["fg_id"].notna().all()
        else "name"
    )

    # Pre-multiply counting stats by normalized weight, then sum per group.
    # Per-stat NaN-aware weight normalization: if a system has NaN for a stat,
    # its weight is excluded from that stat's denominator so other systems'
    # values are not diluted. This correctly handles quality-check exclusions
    # where a bad system's stat column is NaN'd before blending.
    stat_cols = [c for c in counting_cols if c in combined.columns]
    groups = combined[group_col].values
    weights_arr = combined["_weight"].values

    weighted_parts = {}
    for stat in stat_cols:
        vals = combined[stat].values.astype(float)
        nan_mask = np.isnan(vals)
        # Per-row effective weight: 0 where stat is NaN
        eff_w = weights_arr.copy()
        eff_w[nan_mask] = 0.0
        # Normalize within each group based on non-NaN weight sum
        eff_w_series = pd.Series(eff_w, name="_eff_w")
        group_eff_w_sum = eff_w_series.groupby(groups).transform("sum").values
        denom = np.where(group_eff_w_sum > 0, group_eff_w_sum, 1.0)
        nw = np.where(group_eff_w_sum > 0, eff_w / denom, 0.0)
        weighted_parts[stat] = np.where(nan_mask, 0.0, vals) * nw

    weighted = pd.DataFrame(weighted_parts)
    weighted[group_col] = groups
    result = weighted.groupby(group_col, sort=False)[stat_cols].sum()

    # Metadata from highest-weight row per group
    idx_max = combined.groupby(group_col)["_weight"].idxmax()
    meta = combined.loc[idx_max].set_index(group_col)

    result["name"] = meta.index if group_col == "name" else meta["name"]
    result["player_type"] = player_type
    if "team" in combined.columns:
        result["team"] = meta["team"]
    if "fg_id" in combined.columns and group_col != "fg_id":
        result["fg_id"] = combined.groupby(group_col)["fg_id"].first()

    # ADP: weighted average of non-null values only
    # Use original group-normalized weights (not per-stat NaN-aware weights)
    if "adp" in combined.columns:
        group_w_sum = pd.Series(weights_arr).groupby(groups).transform("sum").values
        base_nw = np.where(group_w_sum > 0, weights_arr / group_w_sum, 0.0)
        adp_vals = combined["adp"].values.astype(float).copy()
        adp_nw = base_nw.copy()
        mask = np.isnan(adp_vals)
        adp_vals[mask] = 0.0
        adp_nw[mask] = 0.0

        adp_df = pd.DataFrame({
            group_col: combined[group_col].values,
            "_aw": adp_vals * adp_nw,
            "_nw": adp_nw,
        })
        adp_agg = adp_df.groupby(group_col).sum()
        adp_result = adp_agg["_aw"] / adp_agg["_nw"]
        result["adp"] = adp_result.replace([np.inf, -np.inf, np.nan], float("inf"))

    return result.reset_index()


def _blend_hitters(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Blend hitter projections. Recomputes AVG from blended H and AB."""
    result = _blend_players(dfs, HITTING_COUNTING_COLS, "hitter")
    if result.empty:
        return result
    result["avg"] = np.where(result["ab"] > 0, result["h"] / result["ab"], 0.0)
    return result


def _blend_pitchers(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Blend pitcher projections. Recomputes ERA and WHIP from components."""
    result = _blend_players(dfs, PITCHING_COUNTING_COLS, "pitcher")
    if result.empty:
        return result
    ip = result["ip"]
    result["era"] = np.where(ip > 0, result["er"] * 9 / ip, 0.0)
    result["whip"] = np.where(ip > 0, (result["bb"] + result["h_allowed"]) / ip, 0.0)
    return result


def match_roster_to_projections(
    roster: list[dict],
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
) -> list[dict]:
    """Match roster players to blended projections by normalized name.

    Expects ``_name_norm`` column precomputed on both DataFrames
    (call ``df["_name_norm"] = df["name"].apply(normalize_name)`` first).

    Returns a list of enriched player dicts. Each matched player gets
    ``player_type`` ("hitter"/"pitcher") and all stat columns from the
    projection row. Unmatched players are omitted.
    """
    matched = []
    for player in roster:
        name = player["name"].replace(" (Batter)", "").replace(" (Pitcher)", "")
        name_norm = normalize_name(name)
        positions = player.get("positions", [])

        proj = None
        ptype = None
        if is_hitter(positions) and not hitters_proj.empty:
            matches = hitters_proj[hitters_proj["_name_norm"] == name_norm]
            if not matches.empty:
                proj = matches.iloc[0]
                ptype = "hitter"
        if proj is None and is_pitcher(positions) and not pitchers_proj.empty:
            matches = pitchers_proj[pitchers_proj["_name_norm"] == name_norm]
            if not matches.empty:
                proj = matches.iloc[0]
                ptype = "pitcher"
        if proj is None:
            for df, pt in [(hitters_proj, "hitter"), (pitchers_proj, "pitcher")]:
                if df.empty:
                    continue
                matches = df[df["_name_norm"] == name_norm]
                if not matches.empty:
                    proj = matches.iloc[0]
                    ptype = pt
                    break

        if proj is not None:
            entry = {
                "name": name,
                "positions": positions,
                "player_type": ptype,
                "selected_position": player.get("selected_position", ""),
                "player_id": player.get("player_id", ""),
                "status": player.get("status", ""),
            }
            if ptype == "hitter":
                for col in HITTING_COUNTING_COLS:
                    entry[col] = float(proj.get(col, 0) or 0)
                entry["avg"] = float(proj.get("avg", 0) or 0)
            else:
                for col in PITCHING_COUNTING_COLS:
                    entry[col] = float(proj.get(col, 0) or 0)
                entry["era"] = float(proj.get("era", 0) or 0)
                entry["whip"] = float(proj.get("whip", 0) or 0)
            matched.append(entry)

    return matched
