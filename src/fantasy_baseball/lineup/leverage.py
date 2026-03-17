from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS

MAX_MEANINGFUL_GAP_MULTIPLIER: float = 3.0


def _gap_for_category(
    cat: str, user_val: float, neighbor_val: float
) -> float:
    """Return the absolute gap between user and neighbor for a category."""
    return abs(user_val - neighbor_val)


def calculate_leverage(
    standings: list[dict],
    user_team_name: str,
    *,
    attack_weight: float = 0.6,
    defense_weight: float = 0.4,
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

    team_above = sorted_teams[user_idx - 1] if user_idx > 0 else None
    team_below = (
        sorted_teams[user_idx + 1]
        if user_idx < len(sorted_teams) - 1
        else None
    )

    # When only one neighbor exists, give it all of the weight.
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
        # Only one team in the league — equal weights everywhere.
        return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}

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

    total = sum(raw_leverage.values())
    if total > 0:
        return {cat: val / total for cat, val in raw_leverage.items()}
    return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}
