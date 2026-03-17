from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS

MAX_MEANINGFUL_GAP_MULTIPLIER: float = 3.0


def calculate_leverage(
    standings: list[dict],
    user_team_name: str,
) -> dict[str, float]:
    """Calculate leverage weights for each stat category based on standings gaps.

    Higher leverage = smaller gap to the team above = easier to gain a standings point.
    Weights are normalized to sum to 1.0.
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
        return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}

    user_stats = user_team.get("stats", {})

    if user_idx > 0:
        target_team = sorted_teams[user_idx - 1]
    else:
        target_team = sorted_teams[user_idx + 1] if len(sorted_teams) > 1 else user_team

    target_stats = target_team.get("stats", {})

    raw_leverage: dict[str, float] = {}
    for cat in ALL_CATEGORIES:
        user_val = user_stats.get(cat, 0)
        target_val = target_stats.get(cat, 0)

        if cat in INVERSE_STATS:
            gap = abs(user_val - target_val)
        else:
            gap = abs(target_val - user_val)

        epsilon = 0.001
        raw_leverage[cat] = 1.0 / (gap + epsilon)

    total = sum(raw_leverage.values())
    if total > 0:
        return {cat: val / total for cat, val in raw_leverage.items()}
    return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}
