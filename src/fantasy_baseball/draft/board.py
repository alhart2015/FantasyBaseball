import pandas as pd
from pathlib import Path
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.replacement import calculate_replacement_levels
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import compute_starters_per_position
from fantasy_baseball.utils.name_utils import normalize_name


def build_draft_board(
    projections_dir: Path,
    positions_path: Path,
    systems: list[str],
    weights: dict[str, float] | None = None,
    sgp_overrides: dict[str, float] | None = None,
    roster_slots: dict[str, int] | None = None,
    num_teams: int | None = None,
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

    starters = compute_starters_per_position(roster_slots, num_teams)
    replacement_levels = calculate_replacement_levels(pool, starters)
    pool["var"] = 0.0
    pool["best_position"] = ""
    for idx, row in pool.iterrows():
        var, pos = calculate_var(row, replacement_levels, return_position=True)
        pool.at[idx, "var"] = var
        pool.at[idx, "best_position"] = pos

    # Add normalized name column for matching
    pool["name_normalized"] = pool["name"].apply(normalize_name)

    # Unique player ID to disambiguate same-name players (e.g. Juan Soto OF vs SP)
    pool["player_id"] = pool["name"] + "::" + pool["player_type"]

    return pool.sort_values("var", ascending=False).reset_index(drop=True)


def apply_keepers(board: pd.DataFrame, keepers: list[dict]) -> pd.DataFrame:
    """Remove keeper players from the draft board.

    Uses normalized name matching to handle accented characters.
    When multiple board entries share a name (e.g. two different players
    named 'Juan Soto'), only the highest-VAR entry per keeper is removed.
    """
    ids_to_remove: set[str] = set()
    for keeper in keepers:
        norm = normalize_name(keeper["name"])
        matches = board[board["name_normalized"] == norm]
        if matches.empty:
            continue
        # Remove only the best-VAR match (the one you'd actually keep)
        best_idx = matches["var"].idxmax()
        ids_to_remove.add(board.at[best_idx, "player_id"])
    return board[~board["player_id"].isin(ids_to_remove)].reset_index(drop=True)


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
