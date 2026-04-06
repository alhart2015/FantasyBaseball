"""Recency blending for Player objects.

Bridges the gap between the recency model (which works on rate dicts and
game-log lists) and the Player/Stats dataclasses used by the lineup and
waiver modules.
"""

from __future__ import annotations

import copy
from collections import defaultdict

from fantasy_baseball.analysis.recency import predict_reliability_blend
from fantasy_baseball.models.player import (
    HitterStats, PitcherStats, Player, PlayerType,
)
from fantasy_baseball.utils.name_utils import normalize_name

# Hitter game log columns to extract from SQLite
_HITTER_LOG_COLS = ("date", "pa", "ab", "h", "r", "hr", "rbi", "sb")

# Pitcher game log columns to extract from SQLite (g synthesized as 1 per row)
_PITCHER_LOG_COLS = ("date", "ip", "k", "er", "bb", "h_allowed", "w", "sv", "gs")


def load_game_logs_by_name(
    conn,
    season: int,
) -> dict[str, list[dict]]:
    """Load per-game log entries from SQLite, keyed by normalized name.

    Returns {normalized_name: [game_dicts]} where each game dict has the
    fields expected by predict_reliability_blend.  Pitcher dicts include a
    synthesized ``g = 1`` field (each row is one game appearance).
    """
    logs: dict[str, list[dict]] = defaultdict(list)

    # Hitters
    rows = conn.execute(
        "SELECT name, date, pa, ab, h, r, hr, rbi, sb "
        "FROM game_logs WHERE season = ? AND player_type = 'hitter' "
        "ORDER BY date",
        (season,),
    ).fetchall()
    for row in rows:
        key = f"{normalize_name(row['name'])}::hitter"
        logs[key].append(
            {col: row[col] if col == "date" else (row[col] or 0) for col in _HITTER_LOG_COLS}
        )

    # Pitchers
    rows = conn.execute(
        "SELECT name, date, ip, k, er, bb, h_allowed, w, sv, gs "
        "FROM game_logs WHERE season = ? AND player_type = 'pitcher' "
        "ORDER BY date",
        (season,),
    ).fetchall()
    for row in rows:
        key = f"{normalize_name(row['name'])}::pitcher"
        entry = {col: row[col] if col == "date" else (row[col] or 0) for col in _PITCHER_LOG_COLS}
        entry["g"] = 1  # each row is one game appearance
        logs[key].append(entry)

    return dict(logs)


# ---------------------------------------------------------------------------
# Player-level blending
# ---------------------------------------------------------------------------

def blend_player_with_game_logs(
    player: Player,
    game_logs: list[dict],
    cutoff: str,
) -> Player:
    """Apply reliability-weighted recency blend to a Player's ROS stats.

    Converts the Player's ROS stats to per-PA/IP projection rates,
    runs predict_reliability_blend against game log entries, and
    returns a new Player with updated ROS stats.

    If game_logs is empty or player has no ROS stats, returns a copy unchanged.
    """
    result = copy.copy(player)
    result.positions = list(player.positions)

    if player.ros is None or not game_logs:
        return result

    if player.player_type == PlayerType.HITTER:
        result.ros = _blend_hitter(player.ros, game_logs, cutoff)
    else:
        result.ros = _blend_pitcher(player.ros, game_logs, cutoff)

    return result


def _blend_hitter(ros: HitterStats, game_logs: list[dict], cutoff: str) -> HitterStats:
    pa, ab = ros.pa, ros.ab
    if pa <= 0 or ab <= 0:
        return copy.copy(ros)

    proj_rates = {
        "hr_per_pa": ros.hr / pa,
        "r_per_pa": ros.r / pa,
        "rbi_per_pa": ros.rbi / pa,
        "sb_per_pa": ros.sb / pa,
        "avg": ros.avg,
    }
    rates = predict_reliability_blend(proj_rates, game_logs, cutoff)

    return HitterStats(
        pa=pa,
        ab=ab,
        h=rates["avg"] * ab,
        r=rates["r_per_pa"] * pa,
        hr=rates["hr_per_pa"] * pa,
        rbi=rates["rbi_per_pa"] * pa,
        sb=rates["sb_per_pa"] * pa,
        avg=rates["avg"],
    )


def _blend_pitcher(ros: PitcherStats, game_logs: list[dict], cutoff: str) -> PitcherStats:
    ip = ros.ip
    if ip <= 0:
        return copy.copy(ros)

    # Estimate G and GS from projection shape (not stored in PitcherStats).
    # Relievers: SV > 0 and W < 5 → ~1 IP/game.  Starters: ~6 IP/start.
    if ros.sv > 0 and ros.w < 5:
        est_g = max(ip, 1)
        est_gs = 0.0
    else:
        est_gs = max(ip / 6, 1)
        est_g = est_gs

    proj_rates = {
        "k_per_ip": ros.k / ip,
        "era": ros.era,
        "whip": ros.whip,
        "w_per_gs": ros.w / est_gs if est_gs > 0 else 0,
        "sv_per_g": ros.sv / est_g if est_g > 0 else 0,
    }
    rates = predict_reliability_blend(proj_rates, game_logs, cutoff)

    # Convert blended rates back to counting stats
    blended_k = rates["k_per_ip"] * ip
    blended_era = rates["era"]
    blended_whip = rates["whip"]
    blended_er = blended_era * ip / 9
    # Approximate BB/H split from WHIP (60/40 H/BB typical split)
    blended_bb = blended_whip * ip * 0.4
    blended_h_allowed = blended_whip * ip * 0.6
    blended_w = rates["w_per_gs"] * est_gs if est_gs > 0 else ros.w
    blended_sv = rates["sv_per_g"] * est_g if est_g > 0 else ros.sv

    return PitcherStats(
        ip=ip,
        w=blended_w,
        k=blended_k,
        sv=blended_sv,
        er=blended_er,
        bb=blended_bb,
        h_allowed=blended_h_allowed,
        era=blended_era,
        whip=blended_whip,
    )
