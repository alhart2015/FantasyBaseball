import pandas as pd
from pathlib import Path
from .fangraphs import load_projection_set, _find_file

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
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend multiple projection systems into weighted averages.

    Counting stats are blended directly. Rate stats (AVG, ERA, WHIP)
    are recomputed from blended component stats.
    """
    validate_projections_dir(projections_dir, systems)

    if weights is None:
        weights = {s: 1.0 / len(systems) for s in systems}

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    all_hitters: list[pd.DataFrame] = []
    all_pitchers: list[pd.DataFrame] = []

    for system in systems:
        hitters, pitchers = load_projection_set(projections_dir, system)
        w = weights.get(system, 0)
        if not hitters.empty:
            hitters = hitters.copy()
            hitters["_weight"] = w
            all_hitters.append(hitters)
        if not pitchers.empty:
            pitchers = pitchers.copy()
            pitchers["_weight"] = w
            all_pitchers.append(pitchers)

    blended_hitters = _blend_hitters(all_hitters)
    blended_pitchers = _blend_pitchers(all_pitchers)
    return blended_hitters, blended_pitchers


def _blend_hitters(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Blend hitter projections. Recomputes AVG from blended H and AB."""
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    results = []
    for name, group in combined.groupby("name"):
        w = group["_weight"].values
        # Renormalize weights so players in fewer systems aren't diluted
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        row: dict = {"name": name, "player_type": "hitter"}
        for col in HITTING_COUNTING_COLS:
            if col in group.columns:
                row[col] = (group[col] * w).sum()
        # Recompute AVG from blended H and AB
        if row.get("ab", 0) > 0:
            row["avg"] = row["h"] / row["ab"]
        else:
            row["avg"] = 0.0
        if "team" in group.columns:
            row["team"] = group.loc[group["_weight"].idxmax(), "team"]
        if "fg_id" in group.columns:
            row["fg_id"] = group.iloc[0]["fg_id"]
        if "adp" in group.columns:
            adp_mask = group["adp"].notna()
            if adp_mask.any():
                adp_w = w[adp_mask.values]
                adp_w_sum = adp_w.sum()
                if adp_w_sum > 0:
                    row["adp"] = float((group.loc[adp_mask, "adp"].values * adp_w).sum() / adp_w_sum)
                else:
                    row["adp"] = float("inf")
            else:
                row["adp"] = float("inf")
        results.append(row)
    return pd.DataFrame(results)


def _blend_pitchers(dfs: list[pd.DataFrame]) -> pd.DataFrame:
    """Blend pitcher projections. Recomputes ERA and WHIP from components."""
    if not dfs:
        return pd.DataFrame()

    combined = pd.concat(dfs, ignore_index=True)
    results = []
    for name, group in combined.groupby("name"):
        w = group["_weight"].values
        # Renormalize weights so players in fewer systems aren't diluted
        w_sum = w.sum()
        if w_sum > 0:
            w = w / w_sum
        row: dict = {"name": name, "player_type": "pitcher"}
        for col in PITCHING_COUNTING_COLS:
            if col in group.columns:
                row[col] = (group[col] * w).sum()
        # Recompute ERA = ER * 9 / IP
        ip = row.get("ip", 0)
        if ip > 0:
            row["era"] = row.get("er", 0) * 9 / ip
            bb = row.get("bb", 0)
            h_allowed = row.get("h_allowed", 0)
            row["whip"] = (bb + h_allowed) / ip
        else:
            row["era"] = 0.0
            row["whip"] = 0.0
        if "team" in group.columns:
            row["team"] = group.loc[group["_weight"].idxmax(), "team"]
        if "fg_id" in group.columns:
            row["fg_id"] = group.iloc[0]["fg_id"]
        if "adp" in group.columns:
            adp_mask = group["adp"].notna()
            if adp_mask.any():
                adp_w = w[adp_mask.values]
                adp_w_sum = adp_w.sum()
                if adp_w_sum > 0:
                    row["adp"] = float((group.loc[adp_mask, "adp"].values * adp_w).sum() / adp_w_sum)
                else:
                    row["adp"] = float("inf")
            else:
                row["adp"] = float("inf")
        results.append(row)
    return pd.DataFrame(results)
