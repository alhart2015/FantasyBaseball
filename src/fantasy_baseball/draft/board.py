import logging
import pandas as pd
from fantasy_baseball.data.db import get_blended_projections, get_positions
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.replacement import calculate_replacement_levels, calculate_replacement_rates
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import (
    compute_starters_per_position,
    CLOSER_SV_THRESHOLD,
    WAIVER_SP, WAIVER_RP, WAIVER_HITTER,
    HEALTHY_SP_IP, HEALTHY_CLOSER_IP, HEALTHY_HITTER_AB,
    BACKFILL_SP_THRESHOLD, BACKFILL_CLOSER_THRESHOLD, BACKFILL_HITTER_THRESHOLD,
    STARTER_IP_THRESHOLD,
    safe_float as _safe,
)
from fantasy_baseball.utils.name_utils import normalize_name

logger = logging.getLogger(__name__)


def apply_backfill_blending(pool: pd.DataFrame) -> pd.DataFrame:
    """Blend injury-prone players' stats with waiver-quality backfill.

    Players projected below a healthy baseline have their counting stats
    augmented with replacement-level stats for the gap innings/ABs.  This
    produces effective stats that reflect the true team-level cost.

    Original stats are preserved in ``orig_*`` columns for display.
    """
    pool = pool.copy()

    # Ensure counting-stat columns are float so fractional backfill
    # values can be written without pandas raising LossySetitemError.
    _float_cols = [
        "w", "k", "sv", "ip", "er", "bb", "h_allowed",
        "r", "hr", "rbi", "sb", "h", "ab",
        "era", "whip", "avg",
    ]
    for c in _float_cols:
        if c in pool.columns:
            pool[c] = pool[c].astype(float)

    for idx, row in pool.iterrows():
        if row["player_type"] == PlayerType.PITCHER:
            sv = _safe(row.get("sv", 0))
            ip = _safe(row.get("ip", 0))
            positions = row.get("positions", [])

            # Classify pitcher tier
            if sv >= CLOSER_SV_THRESHOLD:
                baseline = HEALTHY_CLOSER_IP
                threshold = BACKFILL_CLOSER_THRESHOLD
                waiver = WAIVER_RP
            elif "SP" in positions or ip >= STARTER_IP_THRESHOLD:
                baseline = HEALTHY_SP_IP
                threshold = BACKFILL_SP_THRESHOLD
                waiver = WAIVER_SP
            else:
                continue  # middle reliever — no backfill

            gap = baseline - ip
            if gap <= threshold:
                continue

            # Preserve originals
            pool.at[idx, "orig_ip"] = ip
            pool.at[idx, "orig_era"] = row.get("era", 0)
            pool.at[idx, "orig_whip"] = row.get("whip", 0)

            scale = gap / waiver["ip"]
            for col in ("w", "k", "sv", "ip", "er", "bb", "h_allowed"):
                pool.at[idx, col] = row.get(col, 0) + waiver[col] * scale

            # Recompute rate stats from blended components
            new_ip = pool.at[idx, "ip"]
            if new_ip > 0:
                pool.at[idx, "era"] = pool.at[idx, "er"] * 9 / new_ip
                pool.at[idx, "whip"] = (pool.at[idx, "bb"] + pool.at[idx, "h_allowed"]) / new_ip

        elif row["player_type"] == PlayerType.HITTER:
            ab = _safe(row.get("ab", 0))
            gap = HEALTHY_HITTER_AB - ab
            if gap <= BACKFILL_HITTER_THRESHOLD:
                continue

            pool.at[idx, "orig_ab"] = ab
            pool.at[idx, "orig_avg"] = row.get("avg", 0)

            scale = gap / WAIVER_HITTER["ab"]
            for col in ("r", "hr", "rbi", "sb", "h", "ab"):
                pool.at[idx, col] = row.get(col, 0) + WAIVER_HITTER[col] * scale

            new_ab = pool.at[idx, "ab"]
            if new_ab > 0:
                pool.at[idx, "avg"] = pool.at[idx, "h"] / new_ab

    return pool


def build_draft_board(
    conn,
    sgp_overrides: dict[str, float] | None = None,
    roster_slots: dict[str, int] | None = None,
    num_teams: int | None = None,
) -> pd.DataFrame:
    """Build a ranked draft board from projections and position data in SQLite."""
    hitters, pitchers = get_blended_projections(conn)
    positions = get_positions(conn)

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

    hitters = _attach_positions(hitters, norm_positions, default_type=PlayerType.HITTER)
    pitchers = _attach_positions(pitchers, norm_positions, default_type=PlayerType.PITCHER)

    denoms = get_sgp_denominators(sgp_overrides)
    pool = pd.concat([hitters, pitchers], ignore_index=True)

    # Apply injury backfill blending before SGP calculation
    pool = apply_backfill_blending(pool)

    # Two-pass SGP: first with defaults (for ordering), then with
    # pool-derived replacement rates (for accurate values).
    pool["total_sgp"] = pool.apply(
        lambda row: calculate_player_sgp(row, denoms=denoms), axis=1
    )

    starters = compute_starters_per_position(roster_slots, num_teams)
    repl_rates = calculate_replacement_rates(pool, starters)

    pool["total_sgp"] = pool.apply(
        lambda row: calculate_player_sgp(
            row, denoms=denoms,
            replacement_era=repl_rates["era"],
            replacement_whip=repl_rates["whip"],
            replacement_avg=repl_rates["avg"],
        ),
        axis=1,
    )

    replacement_levels = calculate_replacement_levels(pool, starters)
    pool["var"] = 0.0
    pool["best_position"] = ""
    for idx, row in pool.iterrows():
        var, pos = calculate_var(row, replacement_levels, return_position=True)
        pool.at[idx, "var"] = var
        pool.at[idx, "best_position"] = pos

    # Add normalized name column for matching
    pool["name_normalized"] = pool["name"].apply(normalize_name)

    # Unique player ID — use fg_id when available (handles same-name players
    # like Max Muncy LAD vs Max Muncy ATH), fall back to name::type.
    if "fg_id" in pool.columns and pool["fg_id"].notna().all():
        pool["player_id"] = pool["fg_id"].astype(str) + "::" + pool["player_type"]
    else:
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
            adp_threshold, len(missing),
        )
        for name, ptype, adp, ab, ip in sorted(missing, key=lambda x: x[2]):
            stat = f"AB={ab:.0f}" if ptype == PlayerType.HITTER else f"IP={ip:.0f}"
            logger.warning(
                "  ADP %3.0f: %-25s [%s] %s (filtered by minimum threshold)",
                adp, name, ptype, stat,
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
