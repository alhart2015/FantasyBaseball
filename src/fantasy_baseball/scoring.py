"""Roto scoring and team stat projection — shared across all modules.

Provides core functions:
- project_team_stats: sum projected stats for a roster into a
  CategoryStats. Accepts Player dataclass objects OR flat dicts for
  backwards compatibility with draft/script callers that still build
  rosters as plain dicts.
- project_team_sds: analytical team-level standard deviations for
  each category under player-independence, used to price projection
  uncertainty into score_roto.
- score_roto: assign expected roto points via pairwise Gaussian
  win-probabilities. With team_sds=None, collapses to rank-based
  scoring with averaged ties (backwards-compatible default).
"""

from __future__ import annotations

from math import erf, sqrt

from fantasy_baseball.models.player import PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import IL_SLOTS, Position
from fantasy_baseball.models.standings import CategoryStats
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.constants import (
    ALL_CATEGORIES as ALL_CATS,
)
from fantasy_baseball.utils.constants import (
    HITTING_COUNTING,
    IL_STATUSES,
    PITCHING_COUNTING,
    STARTER_IP_THRESHOLD,
    STAT_VARIANCE,
    Category,
)
from fantasy_baseball.utils.constants import (
    INVERSE_STATS as INVERSE_CATS,
)
from fantasy_baseball.utils.constants import (
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
    # Player dataclass: stats live on the .rest_of_season attribute
    ros = getattr(p, "rest_of_season", None)
    if ros is not None and hasattr(ros, key):
        return _safe(getattr(ros, key, 0))
    # Flat dict (legacy callers, tests, draft scripts)
    if isinstance(p, dict):
        return _safe(p.get(key, 0))
    return 0.0


def _prob_beats(
    mu_a: float,
    mu_b: float,
    sd_a: float,
    sd_b: float,
    *,
    higher_is_better: bool,
) -> float:
    """P(team A's category total exceeds team B's) under Gaussian independence.

    When combined SD is zero, this is a step function: 1.0 if A is ahead,
    0.0 if behind, 0.5 on exact equality. Positive combined SD smooths the
    step into a continuous sigmoid. The ``higher_is_better`` flag flips
    the direction for inverse categories (ERA, WHIP).

    This is the pairwise primitive the EV-based ``score_roto`` sums over.
    """
    diff = (mu_a - mu_b) if higher_is_better else (mu_b - mu_a)
    combined = sqrt(sd_a * sd_a + sd_b * sd_b)
    if combined == 0.0:
        if diff > 0:
            return 1.0
        if diff < 0:
            return 0.0
        return 0.5
    return 0.5 * (1.0 + erf(diff / (combined * sqrt(2.0))))


# ── Displacement helpers ────────────────────────────────────────────

# Generic slots that are ignored when matching positions for displacement.
_GENERIC_SLOTS: frozenset[Position] = frozenset(
    {
        Position.P,
        Position.UTIL,
        Position.IF,
        Position.DH,
        Position.BN,
        Position.IL,
        Position.IL_PLUS,
        Position.DL,
        Position.DL_PLUS,
    }
)


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
    if p.rest_of_season is None:
        return 0.0
    if isinstance(p.rest_of_season, PitcherStats):
        return _safe(p.rest_of_season.ip)
    # Hitters: prefer PA, fall back to AB
    pa = _safe(getattr(p.rest_of_season, "pa", 0))
    if pa > 0:
        return pa
    return _safe(getattr(p.rest_of_season, "ab", 0))


def _pitcher_role(p: Player) -> str:
    """Classify a pitcher as 'SP' or 'RP' based on projected IP."""
    ip = _safe(p.rest_of_season.ip) if isinstance(p.rest_of_season, PitcherStats) else 0.0
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
    if p.rest_of_season is None:
        return 0.0
    return calculate_player_sgp(p.rest_of_season)


def _scale_stats(p: Player, factor: float) -> dict[str, float | PlayerType]:
    """Return a dict of scaled counting stats for the player.

    factor=1.0 means full stats; factor=0.0 means zeroed out. The
    ``player_type`` key is included so callers can route the result the
    same way they would a full Player.
    """
    result: dict[str, float | PlayerType] = {}
    if p.rest_of_season is None:
        return result
    if p.player_type == PlayerType.HITTER:
        for key in HITTING_COUNTING:
            result[key] = _safe(getattr(p.rest_of_season, key, 0)) * factor
    elif p.player_type == PlayerType.PITCHER:
        for key in PITCHING_COUNTING:
            result[key] = _safe(getattr(p.rest_of_season, key, 0)) * factor
    result["player_type"] = p.player_type
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
        r=r,
        hr=hr,
        rbi=rbi,
        sb=sb,
        avg=calculate_avg(h_total, ab_total),
        w=w,
        k=k,
        sv=sv,
        era=calculate_era(er_total, ip_total),
        whip=calculate_whip(bb_total, ha_total, ip_total),
    )


def project_team_sds(
    roster,
    *,
    displacement: bool = True,
) -> dict[str, float]:
    """Aggregate per-player projection variance into team-level SDs.

    Uses ``STAT_VARIANCE`` (per-stat CV calibrated from 2022-2024
    Steamer+ZiPS vs actuals) under a player-independence assumption:

        SD_cat_team = CV_cat * sqrt(sum_over_players(stat_i^2))

    Rate stats propagate through their component totals:

        SD_AVG  = CV_h * sqrt(sum h_i^2) / sum_AB
        SD_ERA  = 9 * CV_er * sqrt(sum er_i^2) / sum_IP
        SD_WHIP = sqrt(CV_bb^2 * sum bb_i^2 + CV_ha^2 * sum ha_i^2) / sum_IP

    ``displacement`` matches :func:`project_team_stats` — bench excluded,
    IL players displace their worst active positional match.

    Returns ``{cat: sd}`` keyed by the uppercase string form (``"R"``,
    ``"HR"``, …) for every category in ``ALL_CATS``. String keys keep the
    result JSON-serializable at the cache boundary. Empty roster returns
    zeros.
    """
    if displacement:
        roster = _apply_displacement(roster)

    h_sum_sq: dict[str, float] = {k: 0.0 for k in HITTING_COUNTING}
    p_sum_sq: dict[str, float] = {k: 0.0 for k in PITCHING_COUNTING}
    total_ab = 0.0
    total_ip = 0.0

    for p in roster:
        ptype = _get(p, "player_type")
        if ptype == PlayerType.HITTER:
            for k in HITTING_COUNTING:
                v = _stat(p, k)
                h_sum_sq[k] += v * v
            total_ab += _stat(p, "ab")
        elif ptype == PlayerType.PITCHER:
            for k in PITCHING_COUNTING:
                v = _stat(p, k)
                p_sum_sq[k] += v * v
            total_ip += _stat(p, "ip")

    sds: dict[str, float] = {c.value: 0.0 for c in ALL_CATS}
    for stat_key, cat in [
        ("r", Category.R),
        ("hr", Category.HR),
        ("rbi", Category.RBI),
        ("sb", Category.SB),
    ]:
        sds[cat.value] = STAT_VARIANCE[stat_key] * sqrt(h_sum_sq[stat_key])
    for stat_key, cat in [("w", Category.W), ("k", Category.K), ("sv", Category.SV)]:
        sds[cat.value] = STAT_VARIANCE[stat_key] * sqrt(p_sum_sq[stat_key])
    if total_ab > 0:
        sds[Category.AVG.value] = STAT_VARIANCE["h"] * sqrt(h_sum_sq["h"]) / total_ab
    if total_ip > 0:
        sds[Category.ERA.value] = 9.0 * STAT_VARIANCE["er"] * sqrt(p_sum_sq["er"]) / total_ip
        whip_var = (STAT_VARIANCE["bb"] ** 2) * p_sum_sq["bb"] + (
            STAT_VARIANCE["h_allowed"] ** 2
        ) * p_sum_sq["h_allowed"]
        sds[Category.WHIP.value] = sqrt(whip_var) / total_ip
    return sds


def build_projected_standings(
    team_rosters: dict[str, list],
) -> list[dict]:
    """Build the projected_standings list written to the projections cache.

    Each entry has keys ``name``, ``team_key`` (always empty — the
    consumer fills it from standings if needed), ``rank`` (always 0 —
    ranking is computed downstream), and ``stats`` (the team's projected
    category totals from :func:`project_team_stats` with
    ``displacement=True``, serialized via ``CategoryStats.to_dict``).
    """
    return [
        {
            "name": tname,
            "team_key": "",
            "rank": 0,
            "stats": project_team_stats(roster, displacement=True).to_dict(),
        }
        for tname, roster in team_rosters.items()
    ]


def build_team_sds(
    team_rosters: dict[str, list],
    sd_scale: float,
) -> dict[str, dict[str, float]]:
    """Build the team_sds dict written to the projections cache.

    Each team's per-category SDs from :func:`project_team_sds` are
    scaled by ``sd_scale`` — typically ``sqrt(fraction_remaining)`` so
    variance damps as the season progresses and less of the roto total
    is still up for grabs.

    Keys are the uppercase string category codes (``"R"``, ``"HR"``, …)
    so the result is JSON-serializable for the projections cache. In
    memory, callers index by ``cat.value`` where ``cat: Category``.
    """
    return {
        tname: {
            cat: sd * sd_scale for cat, sd in project_team_sds(roster, displacement=True).items()
        }
        for tname, roster in team_rosters.items()
    }


def score_roto(
    all_team_stats: dict,
    *,
    team_sds: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]]:
    """Assign expected-value roto points per team per category.

    For each team in each category, points equal

        pts = 1 + Σ_{j≠me} P(me > j)

    where ``P(A > B) = Φ((μ_A - μ_B) / √(σ_A² + σ_B²))`` under Gaussian
    independence of team totals (Φ is the standard-normal CDF). When
    ``team_sds`` is ``None`` or every σ is zero, this reduces to the
    step function that recovers the standard rank-based scoring,
    including the averaged-ranks convention on exact ties.

    Args:
        all_team_stats: ``{team: stats}``. Values can be ``dict`` or
            ``CategoryStats`` — both support ``[cat]`` indexing.
        team_sds: optional ``{team: {cat: sd}}``. ``None`` disables
            uncertainty (exact-rank behavior).

    Returns:
        ``{team: {R_pts, HR_pts, ..., total}}``. All values are floats.
        Points range from 1 (last) to N (first) for N teams.
    """
    teams = list(all_team_stats.keys())
    results: dict[str, dict[str, float]] = {t: {} for t in teams}

    for cat in ALL_CATS:
        higher_is_better = cat not in INVERSE_CATS
        cat_key = cat.value
        for me in teams:
            mu_me = all_team_stats[me][cat_key]
            sd_me = team_sds.get(me, {}).get(cat_key, 0.0) if team_sds else 0.0
            pts = 1.0
            for other in teams:
                if other is me:
                    continue
                mu_o = all_team_stats[other][cat_key]
                sd_o = team_sds.get(other, {}).get(cat_key, 0.0) if team_sds else 0.0
                pts += _prob_beats(
                    mu_me,
                    mu_o,
                    sd_me,
                    sd_o,
                    higher_is_better=higher_is_better,
                )
            results[me][f"{cat.value}_pts"] = pts

    for t in results:
        results[t]["total"] = sum(results[t].get(f"{c.value}_pts", 0.0) for c in ALL_CATS)

    return results
