"""deltaRoto — roto-point impact metric for player swaps."""

from __future__ import annotations

from dataclasses import dataclass, field

from fantasy_baseball.utils.constants import ALL_CATEGORIES, INVERSE_STATS


# Tunable defaults (overridden by config/league.yaml delta_roto section)
FRAGILE_THRESHOLD = 1.0
EROSION_WEIGHT = 0.3
TIE_FLOOR = 0.5
EROSION_CAP = 0.5


@dataclass
class CategoryDelta:
    roto_delta: float
    defense_before: float
    defense_after: float
    score: float
    reason: str


@dataclass
class DeltaRotoResult:
    total: float
    categories: dict[str, CategoryDelta]
    before_total: float
    after_total: float

    def to_dict(self) -> dict:
        return {
            "total": round(self.total, 2),
            "before_total": round(self.before_total, 2),
            "after_total": round(self.after_total, 2),
            "categories": {
                cat: {
                    "roto_delta": round(cd.roto_delta, 2),
                    "defense_before": round(cd.defense_before, 2) if cd.defense_before < 100 else None,
                    "defense_after": round(cd.defense_after, 2) if cd.defense_after < 100 else None,
                    "score": round(cd.score, 2),
                    "reason": cd.reason,
                }
                for cat, cd in self.categories.items()
            },
        }


def score_swap(
    roto_before: dict[str, dict],
    roto_after: dict[str, dict],
    comfort_before: dict[str, float],
    comfort_after: dict[str, float],
    team_name: str,
    *,
    fragile_threshold: float = FRAGILE_THRESHOLD,
    erosion_weight: float = EROSION_WEIGHT,
    tie_floor: float = TIE_FLOOR,
    erosion_cap: float = EROSION_CAP,
) -> DeltaRotoResult:
    """Score a swap using asymmetric deltaRoto rules.

    Args:
        roto_before/after: score_roto() output — {team: {cat_pts: float, total: float}}.
        comfort_before/after: {cat: float} — defensive comfort in SGP denoms.
        team_name: user's team name in the roto dicts.

    Returns:
        DeltaRotoResult with total score and per-category breakdown.
    """
    total = 0.0
    categories: dict[str, CategoryDelta] = {}

    for cat in ALL_CATEGORIES:
        pts_b = roto_before[team_name][f"{cat}_pts"]
        pts_a = roto_after[team_name][f"{cat}_pts"]
        rd = pts_a - pts_b

        def_b = min(comfort_before[cat], 5.0)
        def_a = min(comfort_after[cat], 5.0)

        if rd < -0.01:
            cat_score = rd
            reason = "loss (full)"
        elif rd > 0.01:
            if def_a <= 0.01:
                discount = tie_floor
            elif def_a < fragile_threshold:
                discount = tie_floor + (1 - tie_floor) * (def_a / fragile_threshold)
            else:
                discount = 1.0
            cat_score = rd * discount
            reason = f"gain x{discount:.0%}"
        else:
            cat_score = 0.0
            reason = ""

        erosion = max(0, def_b - def_a)
        if erosion > 0.05 and abs(rd) <= 0.01:
            penalty = min(erosion_weight * erosion, erosion_cap)
            cat_score -= penalty
            reason += f" erosion -{penalty:.2f}" if reason else f"erosion -{penalty:.2f}"

        categories[cat] = CategoryDelta(
            roto_delta=rd,
            defense_before=def_b,
            defense_after=def_a,
            score=cat_score,
            reason=reason.strip(),
        )
        total += cat_score

    return DeltaRotoResult(
        total=total,
        categories=categories,
        before_total=roto_before[team_name]["total"],
        after_total=roto_after[team_name]["total"],
    )


def compute_delta_roto(
    drop_name: str,
    add_player: "Player",
    user_roster: "list[Player]",
    projected_standings: list[dict],
    team_name: str,
    *,
    fragile_threshold: float = FRAGILE_THRESHOLD,
    erosion_weight: float = EROSION_WEIGHT,
    tie_floor: float = TIE_FLOOR,
    erosion_cap: float = EROSION_CAP,
) -> DeltaRotoResult:
    """Compute deltaRoto for dropping one player and adding another.

    Args:
        drop_name: name of the roster player to drop.
        add_player: Player object for the player to add.
        user_roster: current roster as list of Player objects.
        projected_standings: projected end-of-season stats for all OTHER teams.
        team_name: user's team name.

    Raises:
        ValueError: if drop_name is not found on the roster.
    """
    from fantasy_baseball.scoring import project_team_stats, score_roto
    from fantasy_baseball.sgp.denominators import get_sgp_denominators

    drop_idx = None
    for i, p in enumerate(user_roster):
        if p.name == drop_name:
            drop_idx = i
            break
    if drop_idx is None:
        raise ValueError(f"Player '{drop_name}' not found on roster")

    denoms = get_sgp_denominators()

    all_before: dict[str, dict[str, float]] = {
        t["name"]: dict(t["stats"]) for t in projected_standings
    }
    all_before[team_name] = project_team_stats(user_roster).to_dict()

    roster_after = [p for i, p in enumerate(user_roster) if i != drop_idx]
    roster_after.append(add_player)
    all_after: dict[str, dict[str, float]] = {
        t["name"]: dict(t["stats"]) for t in projected_standings
    }
    all_after[team_name] = project_team_stats(roster_after).to_dict()

    roto_before = score_roto(all_before)
    roto_after = score_roto(all_after)

    comfort_before = compute_defense_comfort(all_before, team_name, denoms)
    comfort_after = compute_defense_comfort(all_after, team_name, denoms)

    return score_swap(
        roto_before, roto_after, comfort_before, comfort_after, team_name,
        fragile_threshold=fragile_threshold,
        erosion_weight=erosion_weight,
        tie_floor=tie_floor,
        erosion_cap=erosion_cap,
    )


def compute_defense_comfort(
    all_stats: dict[str, dict[str, float]],
    team_name: str,
    denoms: dict[str, float],
) -> dict[str, float]:
    """Per-category defensive comfort in SGP denominators.

    Defense = gap to the nearest team that could take a roto point from you.
    For counting stats (higher is better): gap to the best team below you.
    For inverse stats (ERA/WHIP, lower is better): gap to the best team above you
    (i.e., the team with the next-higher value that you're beating).

    Returns inf when you're in last place (nobody below to catch you — but
    also no point to lose).
    """
    comfort: dict[str, float] = {}
    for cat in ALL_CATEGORIES:
        higher_is_better = cat not in INVERSE_STATS
        my_val = all_stats[team_name][cat]
        others = [all_stats[t][cat] for t in all_stats if t != team_name]
        d = denoms[cat]

        if higher_is_better:
            worse = [v for v in others if v < my_val]
            defense = (my_val - max(worse)) / d if worse else float('inf')
        else:
            worse = [v for v in others if v > my_val]
            defense = (min(worse) - my_val) / d if worse else float('inf')

        comfort[cat] = defense
    return comfort
