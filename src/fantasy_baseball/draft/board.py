import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.data.db import (
    get_blended_projections,
    get_positions,
    latest_blended_year,
)
from fantasy_baseball.draft import board_ranges, our_hitters, our_pitchers
from fantasy_baseball.draft.our_players import load_ours, season_mlbam
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.denominators import SgpOverrides, get_sgp_denominators
from fantasy_baseball.sgp.player_value import REPLACEMENT_RATES, calculate_player_sgp
from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import (
    DEFAULT_TEAM_AB,
    DEFAULT_TEAM_IP,
    Category,
)
from fantasy_baseball.utils.name_utils import normalize_name

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ScaleInputs:
    """The scale a board was built on (denominators, AVG / ERA / WHIP rate baselines,
    replacement floors, team volumes), so realized/estimate VAR can be scored on the
    SAME scale as the board. Returned by build_board_from_frames."""

    denoms: dict[Category, float]
    repl_rates: dict[str, Any]
    replacement_levels: dict[str, Any]
    team_ab: int
    team_ip: int


def build_draft_board(
    conn,
    sgp_overrides: SgpOverrides | None = None,
    with_ours: bool = False,
) -> pd.DataFrame:
    """Build a ranked draft board from projections and position data in SQLite.

    ``with_ours``: hitters and pitchers are the FanGraphs blend mixed with ours
    (``our_hitters.blend_board_hitters``, ``our_pitchers.blend_board_pitchers``), each
    when ours is built for the season, with each player's big-year line
    (``board_ranges.attach``: ``p90_<stat>`` columns) and his P90 and risk-adjusted VAR
    (``with_range_var``) when its ranges are built; otherwise (the default, so backtests stay FanGraphs-only) the FanGraphs blend."""
    year = latest_blended_year(conn)
    hitters, pitchers = get_blended_projections(conn, year)
    if with_ours:
        hitters, pitchers = mix_ours(conn, year, hitters, pitchers)
        hitters, pitchers = board_ranges.attach(conn, year, hitters, pitchers)
    positions = get_positions(conn)
    board, _scale = build_board_from_frames(
        hitters, pitchers, positions, sgp_overrides=sgp_overrides
    )
    return with_range_var(board)


def with_range_var(board: pd.DataFrame) -> pd.DataFrame:
    """``board`` with VARs read off each player's simulated whole seasons
    (``board_ranges``), over the same replacement level his VAR uses (``total_sgp -
    var``, his best position's):

    * ``p90_var``: his P90 season (from ``p90_sgp``).
    * ``risk_var``: his average season (from ``mean_sgp``), injuries and short seasons
      included. Held out on 2022-2026 it forecast the real season's SGP better than the
      board's line (FanGraphs' playing time), which ran ~1.2 SGP high for regulars, and
      with no bias (scripts/build_board_ranges.py --backtest). Missed time counts as
      zero, not as a replacement filling in: the same rule ``var`` applies to a short
      projected line (user's call, 2026-10-21), so the two compare directly.

    NaN without a range; a column is left out when its source is."""
    replacement = board["total_sgp"] - board["var"]
    out = board
    for source, column in (("p90_sgp", "p90_var"), ("mean_sgp", "risk_var")):
        if source in board.columns:
            out = out.assign(
                **{column: pd.to_numeric(board[source], errors="coerce") - replacement}
            )
    return out


def mix_ours(
    conn, season: int, hitters: pd.DataFrame, pitchers: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The board's FanGraphs hitters and pitchers with ours mixed in, each type when
    ours is built for ``season`` (else it stays the FanGraphs blend, with a warning)."""
    blends = (
        ("hitter", hitters, our_hitters.blend_board_hitters),
        ("pitcher", pitchers, our_pitchers.blend_board_pitchers),
    )
    out = []
    for player_type, frame, blend in blends:
        ours = load_ours(player_type, season)
        out.append(
            frame if ours is None else blend(frame, ours, season_mlbam(conn, season, player_type))
        )
    return out[0], out[1]


def build_board_from_frames(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
    positions: dict[str, list[str]],
    sgp_overrides: SgpOverrides | None = None,
    team_ab: int = DEFAULT_TEAM_AB,
    team_ip: int = DEFAULT_TEAM_IP,
    repl_rates: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, ScaleInputs]:
    """Compute a ranked draft board from already-loaded projection frames + a
    positions dict.

    Shared by build_draft_board (DB source) and the draft-value module (Apr-1
    CSV source). Returns ``(board, scale)`` where ``scale`` carries the fixed
    rate baselines + floors + team volumes so realized/estimate VAR can be scored on the SAME
    scale. Callers that only need the board (build_draft_board) discard ``scale``.

    ``repl_rates`` (AVG / ERA / WHIP) defaults to ``REPLACEMENT_RATES``, the one set
    every model scores against (#521). Only draft-value passes others: it rebuilds a
    past draft day on the rates that board was actually scored with.
    """
    # Build normalized lookup for positions.
    # When names collide after normalization (e.g. 'José Ramírez' the 3B
    # vs 'Jose Ramirez' the minor-league P), keep the entry with more
    # eligible positions -- the MLB player will have real positions while
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

    hitters = _attach_positions(hitters, norm_positions, default_type=PlayerType.HITTER)
    pitchers = _attach_positions(pitchers, norm_positions, default_type=PlayerType.PITCHER)

    denoms = get_sgp_denominators(sgp_overrides)
    pool = pd.concat([hitters, pitchers], ignore_index=True)

    # AVG / ERA / WHIP are scored against the measured waiver lines' rates, the same
    # fixed baselines every other model uses (#521), not rates derived from this pool.
    repl_rates = dict(REPLACEMENT_RATES if repl_rates is None else repl_rates)
    pool["total_sgp"] = pool.apply(
        lambda row: calculate_player_sgp(
            row,
            denoms=denoms,
            replacement_era=repl_rates["era"],
            replacement_whip=repl_rates["whip"],
            replacement_avg=repl_rates["avg"],
            team_ab=team_ab,
            team_ip=team_ip,
        ),
        axis=1,
    )

    replacement_levels = position_aware_replacement_levels(
        denoms, repl_rates, team_ab=team_ab, team_ip=team_ip
    )
    pool["var"] = 0.0
    pool["best_position"] = ""
    for idx, row in pool.iterrows():
        var, pos = calculate_var(row, replacement_levels, return_position=True)
        pool.at[idx, "var"] = var
        pool.at[idx, "best_position"] = pos

    # Add normalized name column for matching
    pool["name_normalized"] = pool["name"].apply(normalize_name)

    # Unique player ID -- use fg_id when available (handles same-name players
    # like Max Muncy LAD vs Max Muncy ATH), fall back to name::type.
    if "fg_id" in pool.columns and pool["fg_id"].notna().all():
        pool["player_id"] = pool["fg_id"].astype(str) + "::" + pool["player_type"]
    else:
        pool["player_id"] = pool["name"] + "::" + pool["player_type"]

    board = pool.sort_values("var", ascending=False).reset_index(drop=True)
    _validate_top_adp_players(board, hitters, pitchers)
    scale = ScaleInputs(
        denoms=denoms,
        repl_rates=repl_rates,
        replacement_levels=replacement_levels,
        team_ab=team_ab,
        team_ip=team_ip,
    )
    return board, scale


def rebuild_board(config_path: Path, board_path: Path) -> int:
    """Build the draft board from SQLite projections and write to JSON, its hitters
    and pitchers mixed with ours (``build_draft_board(with_ours=True)``).

    Thin I/O wrapper around :func:`build_draft_board`. Used by the web
    dashboard to refresh the cached board on new-draft, and by the
    --rebuild-board CLI flag for offline regeneration. Returns the
    number of board rows written so callers can log/report.
    """
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.db import get_connection
    from fantasy_baseball.draft.state import serialize_board, write_board

    config = load_config(config_path)
    conn = get_connection()
    try:
        board = build_draft_board(
            conn=conn,
            sgp_overrides=config.sgp_overrides,
            with_ours=True,
        )
    finally:
        conn.close()
    write_board(serialize_board(board), board_path)
    return len(board)


def apply_keepers(board: pd.DataFrame, keepers: list[dict[str, Any]]) -> pd.DataFrame:
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

    # adp may arrive as object dtype (e.g. all-NULL column from SQLite);
    # coerce to numeric and skip if no valid values remain.
    all_blended["adp"] = pd.to_numeric(all_blended["adp"], errors="coerce")
    if all_blended["adp"].isna().all():
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
            adp_threshold,
            len(missing),
        )
        for name, ptype, adp, ab, ip in sorted(missing, key=lambda x: x[2]):
            stat = f"AB={ab:.0f}" if ptype == PlayerType.HITTER else f"IP={ip:.0f}"
            logger.warning(
                "  ADP %3.0f: %-25s [%s] %s (filtered by minimum threshold)",
                adp,
                name,
                ptype,
                stat,
            )


def _attach_positions(df, norm_positions, default_type):
    """Attach position eligibility using normalized name matching."""
    if df.empty:
        return df
    df = df.copy()
    default_positions = ["OF"] if default_type == PlayerType.HITTER else ["SP"]
    df["positions"] = df["name"].apply(
        lambda name: norm_positions.get(normalize_name(name), default_positions)
    )
    return df
