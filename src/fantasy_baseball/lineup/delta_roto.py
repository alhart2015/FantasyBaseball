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
