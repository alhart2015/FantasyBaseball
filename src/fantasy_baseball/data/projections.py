import pandas as pd
from pathlib import Path
from .fangraphs import load_projection_set

# Counting stats to blend directly (weighted average)
HITTING_COUNTING_COLS: list[str] = ["r", "hr", "rbi", "sb", "h", "ab", "pa"]
PITCHING_COUNTING_COLS: list[str] = ["w", "k", "sv", "ip", "er", "bb", "h_allowed"]


def blend_projections(
    projections_dir: Path,
    systems: list[str],
    weights: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Blend multiple projection systems into weighted averages.

    Counting stats are blended directly. Rate stats (AVG, ERA, WHIP)
    are recomputed from blended component stats.
    """
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
        results.append(row)
    return pd.DataFrame(results)
