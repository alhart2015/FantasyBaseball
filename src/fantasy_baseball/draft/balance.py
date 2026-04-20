import pandas as pd

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.utils.constants import (
    HITTING_CATEGORIES,
    PITCHING_CATEGORIES,
    Category,
)
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip

TEAM_TARGETS: dict[Category, float] = {
    Category.R: 900,
    Category.HR: 265,
    Category.RBI: 890,
    Category.SB: 145,
    Category.AVG: 0.260,
    Category.W: 78,
    Category.K: 1250,
    Category.ERA: 3.80,
    Category.WHIP: 1.20,
    Category.SV: 55,
}

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
        if self._pitchers:
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
        else:
            # No pitchers: all pitching categories are None so leverage
            # treats them uniformly (avoids 100:1 asymmetry where ERA/WHIP
            # get emergency weight but W/K/SV get zero weight).
            for stat in PITCHING_CATEGORIES:
                totals[stat] = None
        return totals

    def get_avg_components(self) -> tuple[float, float]:
        """Return (total_h, total_ab) for computing projected team AVG."""
        total_h = sum(h.get("h", 0) for h in self._hitters)
        total_ab = sum(h.get("ab", 0) for h in self._hitters)
        return total_h, total_ab

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
