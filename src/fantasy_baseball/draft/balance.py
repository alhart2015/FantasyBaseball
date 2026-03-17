import pandas as pd
from fantasy_baseball.utils.constants import HITTING_CATEGORIES, PITCHING_CATEGORIES

TEAM_TARGETS: dict[str, float] = {
    "R": 850, "HR": 220, "RBI": 830, "SB": 100, "AVG": 0.265,
    "W": 75, "K": 1200, "ERA": 3.80, "WHIP": 1.20, "SV": 80,
}

WARNING_THRESHOLD: float = 0.6


class CategoryBalance:
    """Track projected stat totals for a fantasy roster under construction."""

    def __init__(self):
        self._hitters: list[pd.Series] = []
        self._pitchers: list[pd.Series] = []

    def add_player(self, player: pd.Series) -> None:
        if player.get("player_type") == "hitter":
            self._hitters.append(player)
        elif player.get("player_type") == "pitcher":
            self._pitchers.append(player)

    def get_totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for stat, col in [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb")]:
            totals[stat] = sum(h.get(col, 0) for h in self._hitters)
        total_h = sum(h.get("h", 0) for h in self._hitters)
        total_ab = sum(h.get("ab", 0) for h in self._hitters)
        totals["AVG"] = total_h / total_ab if total_ab > 0 else 0.0
        for stat, col in [("W", "w"), ("K", "k"), ("SV", "sv")]:
            totals[stat] = sum(p.get(col, 0) for p in self._pitchers)
        total_ip = sum(p.get("ip", 0) for p in self._pitchers)
        if total_ip > 0:
            total_er = sum(p.get("er", 0) for p in self._pitchers)
            totals["ERA"] = total_er * 9 / total_ip
            total_bb = sum(p.get("bb", 0) for p in self._pitchers)
            total_ha = sum(p.get("h_allowed", 0) for p in self._pitchers)
            totals["WHIP"] = (total_bb + total_ha) / total_ip
        else:
            totals["ERA"] = 0.0
            totals["WHIP"] = 0.0
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
            if cat in ("ERA", "WHIP"):
                if num_pitchers >= min_pitchers and totals[cat] > target / WARNING_THRESHOLD:
                    warnings.append(f"{cat} is high ({totals[cat]:.2f}, target ~{target:.2f})")
            else:
                if num_pitchers >= min_pitchers and totals[cat] < target * WARNING_THRESHOLD:
                    warnings.append(f"{cat} is low ({totals[cat]:.0f}, target ~{target:.0f})")
        return warnings
