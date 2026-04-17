import pytest

from fantasy_baseball.lineup.delta_roto import DeltaRotoResult, compute_delta_roto
from fantasy_baseball.models.player import (
    HitterStats, PitcherStats, Player, PlayerType,
)


def _make_hitter(name, **stats):
    """Build a Player with HitterStats for unit tests."""
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=["OF"],
        selected_position="OF",
        status="",
        rest_of_season=HitterStats(**stats),
    )


def _make_pitcher(name, **stats):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=["SP"],
        selected_position="SP",
        status="",
        rest_of_season=PitcherStats(**stats),
    )


class TestComputeDeltaRoto:
    def test_end_to_end_swap(self):
        roster = [
            _make_hitter("Hitter A", pa=632, ab=550, h=150, r=80, hr=30, rbi=90, sb=10),
            _make_pitcher("Pitcher A", w=12, k=180, sv=30, ip=60, er=20, bb=15, h_allowed=50),
        ]
        add_player = _make_hitter(
            "Hitter B", pa=632, ab=550, h=155, r=90, hr=25, rbi=85, sb=15,
        )

        standings = [
            {"name": "My Team", "team_key": "", "rank": 0,
             "stats": {"R": 800, "HR": 200, "RBI": 800, "SB": 100, "AVG": 0.260,
                       "W": 70, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.20}},
            {"name": "Rival", "team_key": "", "rank": 0,
             "stats": {"R": 810, "HR": 210, "RBI": 810, "SB": 110, "AVG": 0.265,
                       "W": 75, "K": 1250, "SV": 60, "ERA": 3.40, "WHIP": 1.15}},
        ]

        result = compute_delta_roto(
            drop_name="Hitter A",
            add_player=add_player,
            user_roster=roster,
            projected_standings=standings,
            team_name="My Team",
            team_sds=None,
        )

        assert isinstance(result, DeltaRotoResult)
        assert len(result.categories) == 10
        cat_sum = sum(cd.roto_delta for cd in result.categories.values())
        assert result.total == pytest.approx(cat_sum)
        d = result.to_dict()
        assert "total" in d
        assert "categories" in d
        assert len(d["categories"]) == 10

    def test_drop_not_found_raises(self):
        roster = [_make_hitter("Hitter A", pa=575, ab=500, h=130, r=70, hr=20, rbi=70, sb=5)]
        add_player = _make_hitter("Hitter B", pa=575, ab=500, h=130, r=70, hr=20, rbi=70, sb=5)
        standings = [{"name": "My Team", "team_key": "", "rank": 0,
                      "stats": {"R": 800, "HR": 200, "RBI": 800, "SB": 100, "AVG": 0.260,
                                "W": 70, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.20}}]
        with pytest.raises(ValueError, match="not found"):
            compute_delta_roto(
                "Nobody", add_player, roster, standings, "My Team",
                team_sds=None,
            )

    def test_team_sds_produces_small_delta_for_within_uncertainty_swap(self):
        """With wide σ, a 10-unit SB swap across two tied teams produces |ΔRoto| < 0.5,
        not the full 1.0 of a rank flip."""
        projected_standings = [
            {"name": "User", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 100, "AVG": 0,
                                        "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
            {"name": "Rival", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 99, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
        ] + [
            {"name": f"T{i}", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 10 + i, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}}
            for i in range(10)
        ]
        drop_hitter = _make_hitter("Drop", r=0, hr=0, rbi=0, sb=20, h=0, ab=100)
        add_hitter = _make_hitter("Add", r=0, hr=0, rbi=0, sb=10, h=0, ab=100)
        roster = [drop_hitter]
        team_sds = {t["name"]: {c: 0.0 for c in ["R", "HR", "RBI", "SB", "AVG",
                                                  "W", "K", "SV", "ERA", "WHIP"]}
                    for t in projected_standings}
        team_sds["User"]["SB"] = 10.0
        team_sds["Rival"]["SB"] = 10.0
        result = compute_delta_roto(
            drop_name="Drop", add_player=add_hitter, user_roster=roster,
            projected_standings=projected_standings, team_name="User",
            team_sds=team_sds,
        )
        assert abs(result.categories["SB"].roto_delta) < 0.5

    def test_team_sds_none_matches_pre_uncertainty_behavior(self):
        """team_sds=None reproduces exact-rank deltaRoto (backwards compat)."""
        projected_standings = [
            {"name": "User", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 100, "AVG": 0,
                                        "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
            {"name": "Rival", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 99, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}},
        ] + [
            {"name": f"T{i}", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 10 + i, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}}
            for i in range(10)
        ]
        drop_hitter = _make_hitter("Drop", r=0, hr=0, rbi=0, sb=20, h=0, ab=100)
        add_hitter = _make_hitter("Add", r=0, hr=0, rbi=0, sb=10, h=0, ab=100)
        result = compute_delta_roto(
            drop_name="Drop", add_player=add_hitter, user_roster=[drop_hitter],
            projected_standings=projected_standings, team_name="User",
            team_sds=None,
        )
        assert result.categories["SB"].roto_delta == pytest.approx(-1.0)

    def test_total_is_simple_subtraction_with_team_sds_none(self):
        """score_swap total == roto_after[team].total - roto_before[team].total."""
        projected_standings = [
            {"name": "User", "stats": {"R": 100, "HR": 20, "RBI": 80, "SB": 30, "AVG": 0.270,
                                        "W": 20, "K": 200, "SV": 40, "ERA": 3.80, "WHIP": 1.20}},
            {"name": "Rival", "stats": {"R": 95, "HR": 18, "RBI": 75, "SB": 25, "AVG": 0.265,
                                         "W": 18, "K": 190, "SV": 35, "ERA": 3.90, "WHIP": 1.22}},
        ]
        drop_hitter = _make_hitter("Drop", r=10, hr=5, rbi=15, sb=5, h=20, ab=80)
        add_hitter = _make_hitter("Add", r=20, hr=8, rbi=25, sb=3, h=25, ab=80)
        result = compute_delta_roto(
            drop_name="Drop", add_player=add_hitter, user_roster=[drop_hitter],
            projected_standings=projected_standings, team_name="User",
            team_sds=None,
        )
        assert result.total == pytest.approx(result.after_total - result.before_total)
