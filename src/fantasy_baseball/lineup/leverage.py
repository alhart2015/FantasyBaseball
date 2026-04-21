from fantasy_baseball.models.standings import StandingsSnapshot
from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS

FULL_CONFIDENCE_GAMES: int = 81


def _estimate_season_progress(standings: StandingsSnapshot) -> float:
    """Estimate season progress (0.0 to 1.0) from Redis game log data.

    Reads the season_progress Redis key (written during game log fetch).
    Returns 1.0 at FULL_CONFIDENCE_GAMES (81 games). Falls back to R-based
    estimation if Redis has no data.
    """
    try:
        from fantasy_baseball.data.kv_store import get_kv
        from fantasy_baseball.data.redis_store import get_season_progress

        progress = get_season_progress(get_kv())
        games = progress["games_elapsed"]
        if games > 0:
            return min(1.0, games / FULL_CONFIDENCE_GAMES)
    except Exception:
        pass

    # Fallback: estimate from league-average R (~4.6 R/game/team)
    if not standings.entries:
        return 0.0
    total_r = sum(e.stats.get("R", 0) for e in standings.entries)
    avg_r = total_r / len(standings.entries)
    approx_games = avg_r / 4.6
    return min(1.0, approx_games / FULL_CONFIDENCE_GAMES)


def _leverage_from_standings(
    standings: StandingsSnapshot,
    user_team_name: str,
    attack_weight: float,
    defense_weight: float,
) -> dict[str, float] | None:
    """Compute normalized leverage weights via marginal roto-point impact.

    For each category, asks: "If my stat changed by one SGP denominator,
    how many roto points would I gain or lose?"  This directly counts how
    many teams I'd pass (attack) or be passed by (defense), capturing
    packed clusters that the old single-neighbor approach missed.

    Example: if you're 1st in SB by 20 but teams 2-5 are packed within
    8 SB of each other, losing one denom (8 SB) drops you into that pack
    and costs 4 roto points — not the 1 point that single-neighbor
    leverage would predict.

    Returns None if the user team is not found.
    """
    from fantasy_baseball.sgp.player_value import get_sgp_denominators

    user_entry = None
    for entry in standings.entries:
        if entry.team_name == user_team_name:
            user_entry = entry
            break

    if user_entry is None:
        return None

    sgp_denoms = get_sgp_denominators()

    raw_leverage: dict[str, float] = {}
    for cat in ALL_CATEGORIES:
        key = cat.value
        reverse = cat not in INVERSE_STATS  # higher is better for most cats
        user_val = user_entry.stats.get(cat, 0)
        denom = sgp_denoms.get(cat, 1.0)

        other_vals = [
            entry.stats.get(cat, 0)
            for entry in standings.entries
            if entry.team_name != user_team_name
        ]

        if not other_vals:
            raw_leverage[key] = 0.0
            continue

        # Current rank: count teams better than user (0 = best)
        if reverse:
            current_rank = sum(1 for v in other_vals if v > user_val)
        else:
            current_rank = sum(1 for v in other_vals if v < user_val)

        # Attack: how many positions gained if stat improves by 1 denom?
        if reverse:
            attack_rank = sum(1 for v in other_vals if v > user_val + denom)
        else:
            attack_rank = sum(1 for v in other_vals if v < user_val - denom)
        positions_gained = current_rank - attack_rank

        # Defense: how many positions lost if stat drops by 1 denom?
        if reverse:
            defense_rank = sum(1 for v in other_vals if v > user_val - denom)
        else:
            defense_rank = sum(1 for v in other_vals if v < user_val + denom)
        positions_lost = defense_rank - current_rank

        # Weight attack vs defense, floor to small positive value
        has_attack = current_rank > 0
        has_defense = current_rank < len(other_vals)

        if has_attack and has_defense:
            w_attack = attack_weight
            w_defense = defense_weight
        elif has_attack:
            w_attack = 1.0
            w_defense = 0.0
        elif has_defense:
            w_attack = 0.0
            w_defense = 1.0
        else:
            raw_leverage[key] = 0.0
            continue

        leverage = w_attack * positions_gained + w_defense * positions_lost

        # Floor: even categories with no teams within 1 denom get a small
        # positive value so they're never completely ignored.
        raw_leverage[key] = max(leverage, 0.1)

    total = sum(raw_leverage.values())
    if total > 0:
        return {k: val / total for k, val in raw_leverage.items()}
    return None


def calculate_leverage(
    standings: StandingsSnapshot,
    user_team_name: str,
    *,
    attack_weight: float = 0.6,
    defense_weight: float = 0.4,
    season_progress: float | None = None,
    projected_standings: StandingsSnapshot | None = None,
) -> dict[str, float]:
    """Calculate leverage weights for each stat category based on standings gaps.

    For each category, ranks all teams independently and finds the
    per-category neighbors (team directly above and below the user in
    THAT category's ranking). The gap to those neighbors determines
    leverage:
      - **Attack** (team above in category): small gap = easy opportunity
        to gain a standings point.
      - **Defense** (team below in category): small gap = threat of losing
        a standings point.

    ``attack_weight`` and ``defense_weight`` control the relative importance
    of opportunities vs. threats (default 60/40 favoring attack).  When the
    user is first or last in a category, only the available neighbor is used.

    ``season_progress`` (0.0 to 1.0) controls how much weight goes to
    standings-based leverage vs. equal weights. Early season (low progress),
    leverage is mostly uniform because projections have wide error bars.
    Late season (high progress), leverage is fully standings-driven. If
    None, estimated from game logs in Redis. Ramps to 1.0 at ~81 games
    (half season).

    When ``projected_standings`` is provided, leverage gaps are computed
    from projected standings directly (they already incorporate actual
    performance + ROS projections). The uniform ramp still applies to
    reflect projection uncertainty.

    Weights are normalized to sum to 1.0.
    """
    if season_progress is None:
        season_progress = _estimate_season_progress(standings)

    uniform = {cat.value: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}

    # Use projected standings when available (they already incorporate
    # actual performance to date + ROS projections), otherwise fall back
    # to raw current standings.
    source = projected_standings if projected_standings is not None else standings
    standings_leverage = _leverage_from_standings(
        source,
        user_team_name,
        attack_weight,
        defense_weight,
    )
    if standings_leverage is None:
        return uniform

    # Blend toward uniform early in the season to reflect projection
    # uncertainty. Even projected standings have wide error bars with
    # only a few weeks of data.
    return {
        cat.value: (
            season_progress * standings_leverage[cat.value]
            + (1.0 - season_progress) * uniform[cat.value]
        )
        for cat in ALL_CATEGORIES
    }
