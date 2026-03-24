import logging
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

logger = logging.getLogger(__name__)


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

    # Build normalized lookup for positions.
    # When names collide after normalization (e.g. 'José Ramírez' the 3B
    # vs 'Jose Ramirez' the minor-league P), keep the entry with more
    # eligible positions — the MLB player will have real positions while
    # the prospect typically only has a generic 'P' or 'OF'.
    norm_positions: dict[str, list[str]] = {}
    for k, v in positions.items():
        norm = normalize_name(k)
        if norm not in norm_positions or len(v) > len(norm_positions[norm]):
            norm_positions[norm] = v

    # Filter to players with meaningful projections
    if not hitters.empty:
        hitters = hitters[hitters.get("ab", pd.Series(dtype=float)).fillna(0) >= 50]
    if not pitchers.empty:
        pitchers = pitchers[pitchers.get("ip", pd.Series(dtype=float)).fillna(0) >= 10]

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

    board = pool.sort_values("var", ascending=False).reset_index(drop=True)
    _validate_top_adp_players(board, hitters, pitchers)
    return board


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


def _validate_top_adp_players(
    board: pd.DataFrame,
    unfiltered_hitters: pd.DataFrame,
    unfiltered_pitchers: pd.DataFrame,
    adp_threshold: int = 150,
) -> None:
    """Warn about top-ADP players missing from the final board.

    Compares the top *adp_threshold* players by ADP (from the unfiltered
    blended projections) against the final board.  Players filtered out
    by AB/IP minimums are logged as warnings so data problems are caught
    before draft day.
    """
    if "adp" not in board.columns:
        return

    all_blended = pd.concat([unfiltered_hitters, unfiltered_pitchers], ignore_index=True)
    if "adp" not in all_blended.columns:
        return

    top_adp = all_blended.nsmallest(adp_threshold, "adp")
    board_names = set(board["name"])
    missing = []
    for _, player in top_adp.iterrows():
        if player["name"] not in board_names:
            ab = player.get("ab", 0) or 0
            ip = player.get("ip", 0) or 0
            adp = player.get("adp", 999)
            missing.append((player["name"], player["player_type"], adp, ab, ip))

    if missing:
        logger.warning(
            "Top-%d ADP players missing from board (%d found):",
            adp_threshold, len(missing),
        )
        for name, ptype, adp, ab, ip in sorted(missing, key=lambda x: x[2]):
            stat = f"AB={ab:.0f}" if ptype == "hitter" else f"IP={ip:.0f}"
            logger.warning(
                "  ADP %3.0f: %-25s [%s] %s (filtered by minimum threshold)",
                adp, name, ptype, stat,
            )


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
