import pandas as pd
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import (
    HITTING_CATEGORIES, PITCHING_CATEGORIES, ALL_CATEGORIES, INVERSE_STATS,
)
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

TEAM_TARGETS: dict[str, float] = {
    "R": 900, "HR": 265, "RBI": 890, "SB": 145, "AVG": 0.260,
    "W": 78, "K": 1250, "ERA": 3.80, "WHIP": 1.20, "SV": 55,
}
# NOTE: Targets derived from 10-team simulation averages (2026-03-24).
# SV=55 reflects a 2-closer team (~28 SV each).  The old SV=80 caused
# leverage to perpetually see SV as "behind pace," over-weighting saves.
# HR, RBI, SB old targets (220, 830, 100) were far below actual sim
# averages (270, 899, 151), so hitting categories were always "ahead of
# pace" — compounding the closer-heavy bias.

WARNING_THRESHOLD: float = 0.6


class CategoryBalance:
    """Track projected stat totals for a fantasy roster under construction."""

    def __init__(self):
        self._hitters: list[pd.Series] = []
        self._pitchers: list[pd.Series] = []

    def add_player(self, player: pd.Series) -> None:
        if player.get("player_type") == PlayerType.HITTER:
            self._hitters.append(player)
        elif player.get("player_type") == PlayerType.PITCHER:
            self._pitchers.append(player)

    def get_totals(self) -> dict[str, float | None]:
        totals: dict[str, float | None] = {}
        for stat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
            totals[stat] = sum(h.get(col, 0) for h in self._hitters)
        total_h = sum(h.get("h", 0) for h in self._hitters)
        total_ab = sum(h.get("ab", 0) for h in self._hitters)
        totals["AVG"] = calculate_avg(total_h, total_ab)
        for stat, col in [("W", "w"), ("K", "k"), ("SV", "sv")]:
            totals[stat] = sum(p.get(col, 0) for p in self._pitchers)
        total_ip = sum(p.get("ip", 0) for p in self._pitchers)
        if total_ip > 0:
            total_er = sum(p.get("er", 0) for p in self._pitchers)
            totals["ERA"] = calculate_era(total_er, total_ip)
            total_bb = sum(p.get("bb", 0) for p in self._pitchers)
            total_ha = sum(p.get("h_allowed", 0) for p in self._pitchers)
            totals["WHIP"] = calculate_whip(total_bb, total_ha, total_ip)
        else:
            totals["ERA"] = None
            totals["WHIP"] = None
        return totals

    def get_warnings(self) -> list[str]:
        totals = self.get_totals()
        warnings = []
        num_hitters = len(self._hitters)
        num_pitchers = len(self._pitchers)
        if num_hitters == 0 and num_pitchers == 0:
            return []
        min_hitters = 5
        min_pitchers = 3
        for cat in HITTING_CATEGORIES:
            target = TEAM_TARGETS[cat]
            if cat == "AVG":
                if num_hitters >= min_hitters and totals[cat] < target * WARNING_THRESHOLD:
                    warnings.append(f"{cat} is low ({totals[cat]:.3f}, target ~{target:.3f})")
            else:
                if num_hitters >= min_hitters and totals[cat] < target * WARNING_THRESHOLD:
                    warnings.append(f"{cat} is low ({totals[cat]:.0f}, target ~{target:.0f})")
        for cat in PITCHING_CATEGORIES:
            target = TEAM_TARGETS[cat]
            if totals[cat] is None:
                continue
            if cat in ("ERA", "WHIP"):
                if num_pitchers >= min_pitchers and totals[cat] > target / WARNING_THRESHOLD:
                    warnings.append(f"{cat} is high ({totals[cat]:.2f}, target ~{target:.2f})")
            else:
                if num_pitchers >= min_pitchers and totals[cat] < target * WARNING_THRESHOLD:
                    warnings.append(f"{cat} is low ({totals[cat]:.0f}, target ~{target:.0f})")
        return warnings


def calculate_draft_leverage(
    totals: dict[str, float | None],
    picks_made: int,
    total_picks: int,
    targets: dict[str, float] | None = None,
) -> dict[str, float]:
    """Calculate category leverage weights for draft recommendations.

    Categories where the team is behind pace get higher weight, so the
    recommender steers toward balanced rosters instead of stacking one
    player type.

    The weight is based on how far behind target pace the team is in
    each category.  Early in the draft (few picks made), weights are
    nearly equal.  As the draft progresses and imbalances grow, the
    weights diverge.

    Returns weights normalized to sum to 1.0.
    """
    if targets is None:
        targets = TEAM_TARGETS

    if total_picks <= 0 or picks_made <= 0:
        # No data yet — equal weights
        return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}

    # What fraction of the draft is complete?
    progress = min(picks_made / total_picks, 1.0)

    epsilon = 0.001
    raw: dict[str, float] = {}

    for cat in ALL_CATEGORIES:
        target = targets.get(cat, 0)
        current = totals.get(cat)
        if current is None:
            # No pitchers yet — pitching cats are maximally behind
            raw[cat] = 1.0 / epsilon
            continue

        if target == 0:
            raw[cat] = 1.0
            continue

        if cat in INVERSE_STATS or cat == "AVG":
            # Rate stats don't accumulate, so progress-based pacing
            # doesn't apply.  Instead, compare current value directly
            # to the target and boost weight when behind.
            if current is None or current < epsilon:
                # No data yet (e.g., no hitters or no pitchers) —
                # use neutral weight to avoid degenerate leverage.
                raw[cat] = 1.0
            elif cat == "AVG":
                # Lower AVG = worse.  Below target -> weight > 1.
                gap = target / current
                raw[cat] = max(gap, 0.1)
            else:
                # ERA/WHIP: higher = worse.  Above target -> weight > 1.
                gap = current / target if target > 0 else 1.0
                raw[cat] = max(gap, 0.1)
        else:
            # How far behind pace? Expected = target * progress
            expected = target * progress
            if expected < epsilon:
                raw[cat] = 1.0
                continue
            # Ratio < 1 means behind pace, > 1 means ahead
            ratio = current / expected
            # Zero in a counting category past 15% is urgent, but capped
            # to prevent any single group of categories from drowning out
            # the rest.  Without this cap, starting with 3 hitter keepers
            # and 0 pitchers makes W+K+SV+ERA+WHIP consume ~99.7% of
            # leverage weight, causing the recommender to draft nothing
            # but pitchers for 9+ rounds.
            if current < epsilon and progress > 0.15:
                raw[cat] = 10.0
            else:
                # Invert: behind pace -> high weight, ahead -> low weight
                raw[cat] = 1.0 / max(ratio, 0.1)

    total = sum(raw.values())
    if total > 0:
        return {cat: val / total for cat, val in raw.items()}
    return {cat: 1.0 / len(ALL_CATEGORIES) for cat in ALL_CATEGORIES}
