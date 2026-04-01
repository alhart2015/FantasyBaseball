import statistics

from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS

MAX_MEANINGFUL_GAP_MULTIPLIER: float = 3.0


def _gap_for_category(
    cat: str, user_val: float, neighbor_val: float
) -> float:
    """Return the absolute gap between user and neighbor for a category."""
    return abs(user_val - neighbor_val)


FULL_CONFIDENCE_GAMES: int = 81


def _estimate_season_progress(standings: list[dict]) -> float:
    """Estimate season progress from MLB game logs in SQLite.

    Counts distinct game dates in the game_logs table for the current season.
    Returns 0.0 to 1.0, reaching 1.0 at FULL_CONFIDENCE_GAMES (81 games).
    Falls back to R-based estimation if game_logs is unavailable.
    """
    try:
        from datetime import date
        from fantasy_baseball.data.db import get_connection
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT date) FROM game_logs WHERE season = ?",
                (date.today().year,)
            ).fetchone()
            games = row[0] if row else 0
        finally:
            conn.close()
        if games > 0:
            return min(1.0, games / FULL_CONFIDENCE_GAMES)
    except Exception:
        pass

    # Fallback: estimate from league-average R (~4.6 R/game/team)
    if not standings:
        return 0.0
    total_r = sum(t.get("stats", {}).get("R", 0) for t in standings)
    avg_r = total_r / len(standings)
    approx_games = avg_r / 4.6
    return min(1.0, approx_games / FULL_CONFIDENCE_GAMES)


def _leverage_from_standings(
    standings: list[dict],
    user_team_name: str,
    attack_weight: float,
    defense_weight: float,
) -> dict[str, float] | None:
    """Compute normalized leverage weights from a set of standings.

    Returns None if the user team is not found or has no neighbors.
    """
    sorted_teams = sorted(standings, key=lambda t: t.get("rank", 99))
    user_team = None
    user_idx = None
    for i, team in enumerate(sorted_teams):
        if team["name"] == user_team_name:
            user_team = team
            user_idx = i
            break

    if user_team is None:
        return None

    user_stats = user_team.get("stats", {})

    team_above = sorted_teams[user_idx - 1] if user_idx > 0 else None
    team_below = (
        sorted_teams[user_idx + 1]
        if user_idx < len(sorted_teams) - 1
        else None
    )

    if team_above is not None and team_below is not None:
        w_attack = attack_weight
        w_defense = defense_weight
    elif team_above is not None:
        w_attack = 1.0
        w_defense = 0.0
    elif team_below is not None:
        w_attack = 0.0
        w_defense = 1.0
    else:
        return None

    above_stats = team_above.get("stats", {}) if team_above else {}
    below_stats = team_below.get("stats", {}) if team_below else {}

    epsilon = 0.001

    raw_leverage: dict[str, float] = {}
    for cat in ALL_CATEGORIES:
        user_val = user_stats.get(cat, 0)
        leverage = 0.0

        if team_above is not None:
            above_val = above_stats.get(cat, 0)
            attack_gap = _gap_for_category(cat, user_val, above_val)
            leverage += w_attack * (1.0 / (attack_gap + epsilon))

        if team_below is not None:
            below_val = below_stats.get(cat, 0)
            defense_gap = _gap_for_category(cat, user_val, below_val)
            leverage += w_defense * (1.0 / (defense_gap + epsilon))

        raw_leverage[cat] = leverage

    # Cap outliers: near-tied categories produce extreme leverage values
    # that dominate all decisions. Clamp to MAX_MEANINGFUL_GAP_MULTIPLIER × median.
    if raw_leverage:
        med = statistics.median(raw_leverage.values())
        cap = med * MAX_MEANINGFUL_GAP_MULTIPLIER
        if cap > 0:
            raw_leverage = {cat: min(val, cap) for cat, val in raw_leverage.items()}

    total = sum(raw_leverage.values())
    if total > 0:
        return {cat: val / total for cat, val in raw_leverage.items()}
    return None


def calculate_leverage(
    standings: list[dict],
    user_team_name: str,
    *,
    attack_weight: float = 0.6,
    defense_weight: float = 0.4,
    season_progress: float | None = None,
    projected_standings: list[dict] | None = None,
) -> dict[str, float]:
    """Calculate leverage weights for each stat category based on standings gaps.

    Considers both neighbors in the standings:
      - **Attack** (team above): categories where a small gap means an easy
        opportunity to gain a standings point by overtaking them.
      - **Defense** (team below): categories where a small gap means a threat
        of losing a standings point if they catch you.

    ``attack_weight`` and ``defense_weight`` control the relative importance
    of opportunities vs. threats (default 60/40 favoring attack).  When only
    one neighbor exists (first or last place), that neighbor receives full
    weight.

    ``season_progress`` (0.0 to 1.0) controls how much weight goes to
    standings-based leverage vs. equal weights. Early season (low progress),
    leverage is mostly uniform because standings are noise. Late season
    (high progress), leverage is fully standings-driven. If None, estimated
    from the league-average runs scored in standings (proxy for games played).
    Ramps to 1.0 at ~81 games (half season).

    When ``projected_standings`` is provided, leverage is computed from a blend
    of current and projected standings (weighted by season_progress). This
    replaces the uniform ramp with forward-looking category weighting.

    Note: neighbor ordering uses current standings rank even when blending
    with projected stats. This is intentional — roto standings rank determines
    which teams you're competing with for standings points.

    Weights are normalized to sum to 1.0.
    """
    if season_progress is None:
        season_progress = _estimate_season_progress(standings)

    uniform = {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}

    if projected_standings is not None:
        # Blend current standings with projected, then compute leverage from
        # the blended view. The blend itself handles early/late season weighting.
        blended = blend_standings(standings, projected_standings, season_progress)
        result = _leverage_from_standings(
            blended, user_team_name, attack_weight, defense_weight,
        )
        return result if result is not None else uniform

    # Fallback: blend standings-based leverage with uniform weights.
    standings_leverage = _leverage_from_standings(
        standings, user_team_name, attack_weight, defense_weight,
    )
    if standings_leverage is None:
        return uniform

    return {
        cat: season_progress * standings_leverage[cat] + (1.0 - season_progress) * uniform[cat]
        for cat in ALL_CATEGORIES
    }


def blend_standings(
    current: list[dict],
    projected: list[dict],
    progress: float,
) -> list[dict]:
    """Blend current and projected standings based on season progress.

    For each stat: blended = progress * current + (1 - progress) * projected.
    At progress=0.0, result is fully projected. At progress=1.0, fully current.

    Teams matched by name. Teams appearing in only one list are included as-is.
    """
    proj_by_name = {t["name"]: t for t in projected}
    seen_names = set()
    blended = []

    for team in current:
        name = team["name"]
        seen_names.add(name)
        proj_team = proj_by_name.get(name)
        if proj_team is None:
            blended.append(team)
            continue

        blended_stats = {}
        for cat in team["stats"]:
            cur_val = team["stats"].get(cat, 0)
            proj_val = proj_team["stats"].get(cat, 0)
            blended_stats[cat] = progress * cur_val + (1.0 - progress) * proj_val

        blended.append({
            **team,
            "stats": blended_stats,
        })

    # Include projected-only teams
    for team in projected:
        if team["name"] not in seen_names:
            blended.append(team)

    return blended
