import pandas as pd
from pathlib import Path
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.replacement import calculate_replacement_levels
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.name_utils import normalize_name


def build_draft_board(
    projections_dir: Path,
    positions_path: Path,
    systems: list[str],
    weights: dict[str, float] | None = None,
    sgp_overrides: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Build a ranked draft board from projections and position data."""
    hitters, pitchers = blend_projections(projections_dir, systems, weights)
    positions = load_positions_cache(positions_path)

    # Build normalized lookup for positions
    norm_positions = {normalize_name(k): v for k, v in positions.items()}

    hitters = _attach_positions(hitters, norm_positions, default_type="hitter")
    pitchers = _attach_positions(pitchers, norm_positions, default_type="pitcher")

    denoms = get_sgp_denominators(sgp_overrides)
    pool = pd.concat([hitters, pitchers], ignore_index=True)
    pool["total_sgp"] = pool.apply(
        lambda row: calculate_player_sgp(row, denoms=denoms), axis=1
    )

    replacement_levels = calculate_replacement_levels(pool)
    pool["var"] = 0.0
    pool["best_position"] = ""
    for idx, row in pool.iterrows():
        var, pos = calculate_var(row, replacement_levels, return_position=True)
        pool.at[idx, "var"] = var
        pool.at[idx, "best_position"] = pos

    # Add normalized name column for matching
    pool["name_normalized"] = pool["name"].apply(normalize_name)

    return pool.sort_values("var", ascending=False).reset_index(drop=True)


def apply_keepers(board: pd.DataFrame, keepers: list[dict]) -> pd.DataFrame:
    """Remove keeper players from the draft board.

    Uses normalized name matching to handle accented characters.
    """
    keeper_names_norm = {normalize_name(k["name"]) for k in keepers}
    return board[~board["name_normalized"].isin(keeper_names_norm)].reset_index(drop=True)


def _attach_positions(df, norm_positions, default_type):
    """Attach position eligibility using normalized name matching."""
    if df.empty:
        return df
    df = df.copy()
    default_positions = ["OF"] if default_type == "hitter" else ["SP"]
    df["positions"] = df["name"].apply(
        lambda name: norm_positions.get(normalize_name(name), default_positions)
    )
    return df
