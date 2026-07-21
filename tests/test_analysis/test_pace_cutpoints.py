from fantasy_baseball.analysis.pace import (
    build_pace_deviation_payload,
    compute_pace_cutpoints,
)
from fantasy_baseball.models.player import Player
from fantasy_baseball.utils.constants import Category

DENOMS = {
    Category.R: 10.0,
    Category.HR: 5.0,
    Category.RBI: 10.0,
    Category.SB: 5.0,
    Category.AVG: 0.0015,
    Category.W: 2.0,
    Category.K: 12.0,
    Category.SV: 3.5,
    Category.ERA: 0.10,
    Category.WHIP: 0.03,
}


def test_cutpoints_twelve_values():
    # nearest-rank: index = round(q * (n-1)), n=12 -> round(q*11)
    cp = compute_pace_cutpoints([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert cp == [3, 5, 8, 10]


def test_cutpoints_exactly_min_pool():
    cp = compute_pace_cutpoints([10, 20, 30, 40, 50, 60])
    assert cp == [20, 30, 40, 50]


def test_cutpoints_below_min_pool_is_none():
    assert compute_pace_cutpoints([1, 2, 3]) is None


def test_payload_keys_and_small_pool_cutpoints():
    hitter = Player.from_dict({"name": "Test Hitter", "player_type": "hitter"})
    preseason = {
        "test hitter": Player.from_dict(
            {
                "name": "Test Hitter",
                "player_type": "hitter",
                "rest_of_season": {
                    "pa": 500,
                    "ab": 450,
                    "avg": 0.280,
                    "r": 100,
                    "hr": 30,
                    "rbi": 100,
                    "sb": 20,
                },
            }
        )
    }
    logs = {"test hitter": {"pa": 100, "ab": 90, "h": 27, "r": 25, "hr": 8, "rbi": 30, "sb": 6}}
    payload = build_pace_deviation_payload([hitter], logs, {}, preseason, DENOMS)
    assert "test hitter::hitter" in payload["deviations"]
    assert payload["deviations"]["test hitter::hitter"]["sgp_dev"] is not None
    # only one hitter -> pool below MIN_POOL_SIZE -> None cutpoints
    assert payload["cutpoints"]["hitter"] is None
    assert payload["cutpoints"]["pitcher"] is None
