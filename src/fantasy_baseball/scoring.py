"""Roto scoring and team stat projection — shared across all modules.

Provides two core functions:
- project_team_stats: sum projected stats for a roster into a
  CategoryStats. Accepts Player dataclass objects OR flat dicts for
  backwards compatibility with draft/script callers that still build
  rosters as plain dicts.
- score_roto: assign roto points (1-N) with fractional tie-breaking
"""

from __future__ import annotations

from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS, Position
from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,  # noqa: F401
    HITTING_COUNTING,
    IL_STATUSES,
    INVERSE_STATS as INVERSE_CATS,  # noqa: F401
    PITCHING_COUNTING,
    STARTER_IP_THRESHOLD,
    safe_float as _safe,
)
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip


def _get(p, key, default=0):
    """Read a field from a Player dataclass or a plain dict."""
    if hasattr(p, key):
        return getattr(p, key)
    if isinstance(p, dict):
        return p.get(key, default)
    return default


def _stat(p, key):
    """Read a stat from a Player's ROS stats or from a flat dict."""
    # Player dataclass: stats live on the .ros attribute
    ros = getattr(p, "ros", None)
    if ros is not None and hasattr(ros, key):
        return _safe(getattr(ros, key, 0))
    # Flat dict (legacy callers, tests, draft scripts)
    if isinstance(p, dict):
        return _safe(p.get(key, 0))
    return 0.0


# ── Displacement helpers ────────────────────────────────────────────

# Generic slots that are ignored when matching positions for displacement.
_GENERIC_SLOTS: frozenset[Position] = frozenset({
    Position.P, Position.UTIL, Position.IF, Position.DH,
    Position.BN, Position.IL, Position.IL_PLUS, Position.DL, Position.DL_PLUS,
})


def _is_il(p: Player) -> bool:
    """True if the player is on the IL (by slot or by Yahoo status)."""
    if p.selected_position in IL_SLOTS:
        return True
    if p.status in IL_STATUSES:
        return True
    return False


def _is_bench(p: Player) -> bool:
    """True if the player is benched (BN slot) and NOT on the IL."""
    return p.selected_position == Position.BN and not _is_il(p)


def _playing_time(p: Player) -> float:
    """Return the playing-time measure: IP for pitchers, PA (or AB) for hitters."""
    if p.ros is None:
        return 0.0
    if p.player_type == PlayerType.PITCHER:
        return _safe(p.ros.ip)
    # Hitters: prefer PA, fall back to AB
    pa = _safe(getattr(p.ros, "pa", 0))
    if pa > 0:
        return pa
    return _safe(getattr(p.ros, "ab", 0))


def _pitcher_role(p: Player) -> str:
    """Classify a pitcher as 'SP' or 'RP' based on projected IP."""
    ip = _safe(p.ros.ip) if p.ros else 0.0
    return "SP" if ip > STARTER_IP_THRESHOLD else "RP"


def _real_positions(p: Player) -> frozenset[Position]:
    """Return the player's eligible positions minus generic/bench/IL slots."""
    return frozenset(p.positions) - _GENERIC_SLOTS


def _find_worst_match(
    il_player: Player,
    active_players: list[Player],
    already_displaced: set[str],
) -> Player | None:
    """Find the worst active player (by SGP) sharing a positional role.

    For pitchers: match SP vs RP role.
    For hitters: match on overlapping real positions; fallback to worst
    hitter overall if no position match.

    Returns None if no eligible active player exists.
    """
    candidates: list[Player] = []

    if il_player.player_type == PlayerType.PITCHER:
        role = _pitcher_role(il_player)
        for a in active_players:
            if a.name in already_displaced:
                continue
            if a.player_type != PlayerType.PITCHER:
                continue
            if _pitcher_role(a) == role:
                candidates.append(a)
    else:
        il_positions = _real_positions(il_player)
        # First pass: overlapping real positions
        for a in active_players:
            if a.name in already_displaced:
                continue
            if a.player_type != PlayerType.HITTER:
                continue
            if il_positions & _real_positions(a):
                candidates.append(a)
        # Fallback: any active hitter
        if not candidates:
            for a in active_players:
                if a.name in already_displaced:
                    continue
                if a.player_type != PlayerType.HITTER:
                    continue
                candidates.append(a)

    if not candidates:
        return None

    # Worst = lowest total SGP
    return min(candidates, key=lambda a: _player_sgp(a))


def _player_sgp(p: Player) -> float:
    """Calculate total SGP for a player, returning 0 if no ROS stats."""
    if p.ros is None:
        return 0.0
    return calculate_player_sgp(p.ros)


def _scale_stats(p: Player, factor: float) -> dict[str, float]:
    """Return a dict of scaled counting stats for the player.

    factor=1.0 means full stats; factor=0.0 means zeroed out.
    """
    result: dict[str, float] = {}
    if p.ros is None:
        return result
    if p.player_type == PlayerType.HITTER:
        for key in HITTING_COUNTING:
            result[key] = _safe(getattr(p.ros, key, 0)) * factor
    elif p.player_type == PlayerType.PITCHER:
        for key in PITCHING_COUNTING:
            result[key] = _safe(getattr(p.ros, key, 0)) * factor
    return result


def _apply_displacement(roster: list[Player]) -> list[Player | dict]:
    """Partition roster into active/bench/IL and apply displacement scaling.

    Returns a list where each entry is either an unmodified Player
    (active, unaffected) or a dict of scaled stats (active, displaced).
    Bench players are excluded. IL players are included at full scale
    (they will return and produce). The worst matching active player
    has their stats scaled down to reflect the playing time the IL
    player takes away.
    """
    # Separate players into categories
    active: list[Player] = []
    il_players: list[Player] = []

    for p in roster:
        if not isinstance(p, Player):
            # Dict-input callers: pass through unmodified
            active.append(p)
            continue
        if _is_bench(p):
            continue  # exclude bench
        if _is_il(p):
            il_players.append(p)
            continue
        active.append(p)

    # Sort IL players by descending playing time (highest PT gets first pick)
    il_players.sort(key=_playing_time, reverse=True)

    # Track which active players have already been displaced
    already_displaced: set[str] = set()
    # Map from player name to scale factor for displaced players
    displacement_factors: dict[str, float] = {}

    for il_p in il_players:
        il_pt = _playing_time(il_p)
        if il_pt <= 0:
            continue  # No playing time -> no displacement

        target = _find_worst_match(il_p, active, already_displaced)
        if target is None:
            continue

        active_pt = _playing_time(target)
        if active_pt <= 0:
            continue

        factor = max(0.0, active_pt - il_pt) / active_pt
        already_displaced.add(target.name)
        displacement_factors[target.name] = factor

    # Build output: IL players at full scale + active with displacement
    result: list = []
    # IL players contribute their full projected stats
    for p in il_players:
        result.append(p)
    # Active players: apply displacement factors to affected ones
    for p in active:
        if not isinstance(p, Player):
            result.append(p)
            continue
        if p.name in displacement_factors:
            factor = displacement_factors[p.name]
            scaled = _scale_stats(p, factor)
            scaled["player_type"] = p.player_type
            result.append(scaled)
        else:
            result.append(p)

    return result


def project_team_stats(roster, *, displacement: bool = False) -> CategoryStats:
    """Sum projected stats for a roster into a CategoryStats.

    Accepts Player dataclass objects OR plain dicts with flat stat
    keys. Rate stats (AVG, ERA, WHIP) are computed from component
    totals rather than simple sums, so the result is mathematically
    correct rather than just a naive average.

    When ``displacement=True``, bench players are excluded and IL
    players displace the worst positional match among active players,
    scaling down the displaced player's stats proportionally based on
    playing time. Only activates for Player dataclass objects — dict
    input callers are unaffected.

    The dict-input path exists for backwards compatibility with
    draft-side scripts (``scripts/simulate_draft.py``,
    ``scripts/summary.py``) that build rosters as plain dicts. Those
    scripts are explicitly out of scope for the League data model
    refactor and would need significant rework to use Player objects.
    Step 9 cleanup can revisit.
    """
    if displacement:
        roster = _apply_displacement(roster)

    r = hr = rbi = sb = h_total = ab_total = 0.0
    w = k = sv = ip_total = er_total = bb_total = ha_total = 0.0

    for p in roster:
        ptype = _get(p, "player_type")
        if ptype == PlayerType.HITTER:
            r += _stat(p, "r")
            hr += _stat(p, "hr")
            rbi += _stat(p, "rbi")
            sb += _stat(p, "sb")
            h_total += _stat(p, "h")
            ab_total += _stat(p, "ab")
        elif ptype == PlayerType.PITCHER:
            w += _stat(p, "w")
            k += _stat(p, "k")
            sv += _stat(p, "sv")
            ip_total += _stat(p, "ip")
            er_total += _stat(p, "er")
            bb_total += _stat(p, "bb")
            ha_total += _stat(p, "h_allowed")

    return CategoryStats(
        r=r, hr=hr, rbi=rbi, sb=sb,
        avg=calculate_avg(h_total, ab_total),
        w=w, k=k, sv=sv,
        era=calculate_era(er_total, ip_total),
        whip=calculate_whip(bb_total, ha_total, ip_total),
    )


def score_roto(
    all_team_stats: dict,
) -> dict[str, dict[str, float]]:
    """Assign roto points with fractional tie-breaking.

    Args:
        all_team_stats: ``{team_name: stats}`` for all teams. Each
            ``stats`` value can be either a plain ``dict[str, float]``
            (legacy callers in draft/trade code) or a
            :class:`CategoryStats` instance (callers that went through
            ``project_team_stats``). Both shapes support ``[cat]``
            indexing, which is all this function needs.

    Returns:
        ``{team_name: {cat_pts: float, ..., "total": float}}`` where
        ``cat_pts`` keys are ``"R_pts"``, ``"HR_pts"``, etc. Points
        range from 1 (worst) to N (best) for N teams.
    """
    teams = list(all_team_stats.keys())
    n = len(teams)
    results: dict[str, dict[str, float]] = {t: {} for t in teams}

    for cat in ALL_CATS:
        rev = cat not in INVERSE_CATS
        ranked = sorted(teams, key=lambda t: all_team_stats[t][cat], reverse=rev)
        i = 0
        while i < n:
            j = i + 1
            while j < n and abs(all_team_stats[ranked[j]][cat] - all_team_stats[ranked[i]][cat]) < 1e-9:
                j += 1
            avg_pts = sum(n - k for k in range(i, j)) / (j - i)
            for k in range(i, j):
                results[ranked[k]][f"{cat}_pts"] = avg_pts
            i = j

    for t in results:
        results[t]["total"] = sum(results[t].get(f"{c}_pts", 0) for c in ALL_CATS)

    return results
