import pytest
from fantasy_baseball.web.season_data import format_standings_for_display


def _sample_standings():
    """Minimal 3-team standings for tests."""
    teams = [
        ("Hart of the Order", "469.l.5652.t.3",
         {"R": 300, "HR": 90, "RBI": 290, "SB": 50, "AVG": 0.270,
          "W": 35, "K": 600, "SV": 25, "ERA": 3.50, "WHIP": 1.18}),
        ("Springfield Isotopes", "469.l.5652.t.8",
         {"R": 310, "HR": 85, "RBI": 295, "SB": 40, "AVG": 0.265,
          "W": 38, "K": 580, "SV": 30, "ERA": 3.40, "WHIP": 1.15}),
        ("SkeleThor", "469.l.5652.t.5",
         {"R": 280, "HR": 95, "RBI": 280, "SB": 55, "AVG": 0.260,
          "W": 30, "K": 620, "SV": 20, "ERA": 3.60, "WHIP": 1.22}),
    ]
    return [{"name": n, "team_key": tk, "rank": i + 1, "stats": s}
            for i, (n, tk, s) in enumerate(teams)]


class TestStandingsTeamKey:
    def test_team_key_present_in_display_data(self):
        result = format_standings_for_display(
            _sample_standings(), "Hart of the Order"
        )
        for team in result["teams"]:
            assert "team_key" in team, f"Missing team_key for {team['name']}"

    def test_team_key_values_correct(self):
        result = format_standings_for_display(
            _sample_standings(), "Hart of the Order"
        )
        isotopes = next(t for t in result["teams"] if t["name"] == "Springfield Isotopes")
        assert isotopes["team_key"] == "469.l.5652.t.8"
