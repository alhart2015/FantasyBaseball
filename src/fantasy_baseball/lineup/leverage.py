import statistics

from fantasy_baseball.models.standings import CategoryStats, StandingsEntry, StandingsSnapshot
from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS

MAX_MEANINGFUL_GAP_MULTIPLIER: float = 3.0


def _gap_for_category(
    cat: str, user_val: float, neighbor_val: float
) -> float:
    """Return the absolute gap between user and neighbor for a category."""
    return abs(user_val - neighbor_val)


FULL_CONFIDENCE_GAMES: int = 81


def _estimate_season_progress(standings: StandingsSnapshot) -> float:
    """Estimate season progress from MLB game logs in SQLite.

    Counts distinct game dates in the game_logs table for the current season.
    Returns 0.0 to 1.0, reaching 1.0 at FULL_CONFIDENCE_GAMES (81 games).
    Falls back to R-based estimation if game_logs is unavailable.
    """
    try:
        from fantasy_baseball.data.db import get_connection
        from fantasy_baseball.utils.time_utils import local_today
        conn = get_connection()
        try:
            row = conn.execute(
                "SELECT COUNT(DISTINCT date) FROM game_logs WHERE season = ?",
                (local_today().year,)
            ).fetchone()
            games = row[0] if row else 0
        finally:
            conn.close()
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
    """Compute normalized leverage weights using per-category rank neighbors.

    For each category, ranks all teams independently and finds the teams
    directly above and below the user in THAT category. Gaps are normalized
    by SGP denominators so that a 1-run gap in R and a 0.001 AVG gap are
    compared on the same scale (both roughly "one standings point worth").

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

    user_stats = user_entry.stats
    sgp_denoms = get_sgp_denominators()
    epsilon = 0.001

    raw_leverage: dict[str, float] = {}
    for cat in ALL_CATEGORIES:
        reverse = cat not in INVERSE_STATS  # higher is better for most cats
        ranked = sorted(standings.entries, key=lambda e: e.stats.get(cat, 0), reverse=reverse)

        user_cat_idx = None
        for i, entry in enumerate(ranked):
            if entry.team_name == user_team_name:
                user_cat_idx = i
                break

        if user_cat_idx is None:
            raw_leverage[cat] = 0.0
            continue

        cat_above = ranked[user_cat_idx - 1] if user_cat_idx > 0 else None
        cat_below = (
            ranked[user_cat_idx + 1]
            if user_cat_idx < len(ranked) - 1
            else None
        )

        if cat_above is not None and cat_below is not None:
            w_attack = attack_weight
            w_defense = defense_weight
        elif cat_above is not None:
            w_attack = 1.0
            w_defense = 0.0
        elif cat_below is not None:
            w_attack = 0.0
            w_defense = 1.0
        else:
            raw_leverage[cat] = 0.0
            continue

        leverage = 0.0
        user_val = user_stats.get(cat, 0)
        denom = sgp_denoms.get(cat, 1.0)

        if cat_above is not None:
            above_val = cat_above.stats.get(cat, 0)
            raw_gap = _gap_for_category(cat, user_val, above_val)
            normalized_gap = raw_gap / denom
            leverage += w_attack * (1.0 / (normalized_gap + epsilon))

        if cat_below is not None:
            below_val = cat_below.stats.get(cat, 0)
            raw_gap = _gap_for_category(cat, user_val, below_val)
            normalized_gap = raw_gap / denom
            leverage += w_defense * (1.0 / (normalized_gap + epsilon))

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
    leverage is mostly uniform because standings are noise. Late season
    (high progress), leverage is fully standings-driven. If None, estimated
    from game logs in SQLite. Ramps to 1.0 at ~81 games (half season).

    When ``projected_standings`` is provided, leverage is computed from a blend
    of current and projected standings (weighted by season_progress). This
    replaces the uniform ramp with forward-looking category weighting.

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
    current: StandingsSnapshot,
    projected: StandingsSnapshot,
    progress: float,
) -> StandingsSnapshot:
    """Blend current and projected standings based on season progress.

    For each stat: blended = progress * current + (1 - progress) * projected.
    At progress=0.0, result is fully projected. At progress=1.0, fully current.

    Teams matched by name. Teams appearing in only one list are included as-is.
    """
    proj_by_name = {e.team_name: e for e in projected.entries}
    seen_names: set[str] = set()
    blended_entries: list[StandingsEntry] = []

    for entry in current.entries:
        name = entry.team_name
        seen_names.add(name)
        proj_entry = proj_by_name.get(name)
        if proj_entry is None:
            blended_entries.append(entry)
            continue

        blended_stats: dict[str, float] = {}
        for cat in ALL_CATEGORIES:
            cur_val = entry.stats.get(cat, 0)
            proj_val = proj_entry.stats.get(cat, 0)
            blended_stats[cat] = progress * cur_val + (1.0 - progress) * proj_val

        blended_entries.append(StandingsEntry(
            team_name=entry.team_name,
            team_key=entry.team_key,
            rank=entry.rank,
            stats=CategoryStats.from_dict(blended_stats),
        ))

    # Include projected-only teams
    for entry in projected.entries:
        if entry.team_name not in seen_names:
            blended_entries.append(entry)

    return StandingsSnapshot(
        effective_date=current.effective_date,
        entries=blended_entries,
    )
