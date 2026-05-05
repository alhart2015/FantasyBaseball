from datetime import date
from unittest.mock import patch

import pandas as pd

from fantasy_baseball.lineup.matchups import (
    adjust_pitcher_projection,
    calculate_matchup_factors,
    fetch_team_batting_stats,
    get_probable_starters,
    normalize_team_batting_stats,
)
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position


def test_normalize_team_batting_stats():
    """Given raw MLB API team stat dicts, produce {abbrev: {ops, k_pct}}."""
    raw = [
        {
            "team_id": 147,
            "team_name": "New York Yankees",
            "abbreviation": "NYY",
            "ops": ".787",
            "strikeouts": 1463,
            "plate_appearances": 6235,
        },
        {
            "team_id": 119,
            "team_name": "Los Angeles Dodgers",
            "abbreviation": "LAD",
            "ops": ".820",
            "strikeouts": 1300,
            "plate_appearances": 6100,
        },
    ]
    result = normalize_team_batting_stats(raw)
    assert "NYY" in result
    assert "LAD" in result
    assert abs(result["NYY"]["ops"] - 0.787) < 0.001
    assert abs(result["NYY"]["k_pct"] - 1463 / 6235) < 0.001
    assert abs(result["LAD"]["ops"] - 0.820) < 0.001


SAMPLE_STATS = {
    "NYY": {"ops": 0.750, "k_pct": 0.230},
    "COL": {"ops": 0.650, "k_pct": 0.260},
    "LAD": {"ops": 0.800, "k_pct": 0.200},
}


def test_league_average_team_gets_factor_of_one():
    same = {"A": {"ops": 0.750, "k_pct": 0.230}, "B": {"ops": 0.750, "k_pct": 0.230}}
    factors = calculate_matchup_factors(same)
    assert abs(factors["A"]["era_whip_factor"] - 1.0) < 0.001
    assert abs(factors["A"]["k_factor"] - 1.0) < 0.001


def test_weak_offense_lowers_era_factor():
    factors = calculate_matchup_factors(SAMPLE_STATS)
    assert factors["COL"]["era_whip_factor"] < 1.0
    assert factors["LAD"]["era_whip_factor"] > 1.0


def test_high_k_team_raises_k_factor():
    factors = calculate_matchup_factors(SAMPLE_STATS)
    assert factors["COL"]["k_factor"] > 1.0
    assert factors["LAD"]["k_factor"] < 1.0


def test_dampening_limits_adjustment():
    extreme = {
        "GOOD": {"ops": 0.900, "k_pct": 0.230},
        "BAD": {"ops": 0.600, "k_pct": 0.230},
    }
    factors = calculate_matchup_factors(extreme, dampening=0.5)
    assert factors["GOOD"]["era_whip_factor"] < 1.15
    assert factors["GOOD"]["era_whip_factor"] > 1.05


def _make_pitcher(name, team, era, whip, k, w, sv, ip):
    return pd.Series(
        {
            "name": name,
            "team": team,
            "player_type": "pitcher",
            "era": era,
            "whip": whip,
            "k": k,
            "w": w,
            "sv": sv,
            "ip": ip,
            "er": era * ip / 9,
            "bb": 40,
            "h_allowed": 140,
        }
    )


def test_easy_matchup_lowers_era():
    pitcher = _make_pitcher("Ace", "NYY", 3.50, 1.15, 200, 12, 0, 180)
    factors = {"era_whip_factor": 0.90, "k_factor": 1.10}
    adjusted = adjust_pitcher_projection(pitcher, factors)
    assert adjusted["era"] < 3.50
    assert adjusted["whip"] < 1.15
    assert adjusted["k"] > 200


def test_hard_matchup_raises_era():
    pitcher = _make_pitcher("Ace", "NYY", 3.50, 1.15, 200, 12, 0, 180)
    factors = {"era_whip_factor": 1.10, "k_factor": 0.90}
    adjusted = adjust_pitcher_projection(pitcher, factors)
    assert adjusted["era"] > 3.50
    assert adjusted["whip"] > 1.15
    assert adjusted["k"] < 200


def test_neutral_matchup_unchanged():
    pitcher = _make_pitcher("Ace", "NYY", 3.50, 1.15, 200, 12, 0, 180)
    factors = {"era_whip_factor": 1.0, "k_factor": 1.0}
    adjusted = adjust_pitcher_projection(pitcher, factors)
    assert abs(adjusted["era"] - 3.50) < 0.001
    assert abs(adjusted["k"] - 200) < 0.1


def test_wins_and_saves_unchanged():
    pitcher = _make_pitcher("Closer", "NYY", 2.50, 1.00, 60, 3, 35, 65)
    factors = {"era_whip_factor": 1.15, "k_factor": 0.85}
    adjusted = adjust_pitcher_projection(pitcher, factors)
    assert adjusted["w"] == 3
    assert adjusted["sv"] == 35


def test_two_start_blended_factors():
    pitcher = _make_pitcher("Horse", "NYY", 3.50, 1.15, 200, 12, 0, 180)
    matchup_list = [
        {"era_whip_factor": 0.90, "k_factor": 1.10},
        {"era_whip_factor": 1.10, "k_factor": 0.90},
    ]
    adjusted = adjust_pitcher_projection(pitcher, matchup_list)
    assert abs(adjusted["era"] - 3.50) < 0.05
    assert abs(adjusted["k"] - 200) < 1.0


@patch("fantasy_baseball.lineup.matchups.statsapi")
def test_fetch_team_batting_stats(mock_api):
    """fetch_team_batting_stats calls MLB API per team and returns normalized data."""
    mock_api.get.side_effect = [
        # First call: get teams
        {
            "teams": [
                {"id": 147, "name": "New York Yankees", "abbreviation": "NYY"},
                {"id": 119, "name": "Los Angeles Dodgers", "abbreviation": "LAD"},
            ]
        },
        # Per-team stat calls
        {
            "stats": [
                {
                    "splits": [
                        {
                            "stat": {
                                "ops": ".787",
                                "strikeOuts": 1463,
                                "plateAppearances": 6235,
                            }
                        }
                    ]
                }
            ]
        },
        {
            "stats": [
                {
                    "splits": [
                        {
                            "stat": {
                                "ops": ".820",
                                "strikeOuts": 1300,
                                "plateAppearances": 6100,
                            }
                        }
                    ]
                }
            ]
        },
    ]

    result = fetch_team_batting_stats(season=2025)
    assert "NYY" in result
    assert "LAD" in result
    assert abs(result["NYY"]["ops"] - 0.787) < 0.001


@patch("fantasy_baseball.lineup.matchups.datetime")
@patch("fantasy_baseball.lineup.matchups.statsapi")
def test_fetch_falls_back_to_prior_season_when_preseason(mock_api, mock_dt):
    """When current-year stats return empty, fall back to season-1."""
    mock_dt.now.return_value.year = 2026

    mock_api.get.side_effect = [
        # Teams list (fetched once, reused for fallback)
        {
            "teams": [
                {"id": 147, "name": "New York Yankees", "abbreviation": "NYY"},
            ]
        },
        # 2026 per-team call: empty splits (no games played yet)
        {"stats": [{"splits": []}]},
        # 2025 fallback per-team call: real data
        {
            "stats": [
                {
                    "splits": [
                        {
                            "stat": {
                                "ops": ".787",
                                "strikeOuts": 1463,
                                "plateAppearances": 6235,
                            }
                        }
                    ]
                }
            ]
        },
    ]

    result = fetch_team_batting_stats(season=None)
    assert "NYY" in result
    assert abs(result["NYY"]["ops"] - 0.787) < 0.001


def test_full_pipeline():
    """End-to-end: raw stats -> factors -> adjusted pitcher projection."""
    raw = [
        {
            "abbreviation": "COL",
            "ops": ".650",
            "strikeouts": 1600,
            "plate_appearances": 6000,
            "team_id": 115,
            "team_name": "Colorado Rockies",
        },
        {
            "abbreviation": "LAD",
            "ops": ".820",
            "strikeouts": 1200,
            "plate_appearances": 6100,
            "team_id": 119,
            "team_name": "Los Angeles Dodgers",
        },
        {
            "abbreviation": "MIL",
            "ops": ".740",
            "strikeouts": 1400,
            "plate_appearances": 6050,
            "team_id": 158,
            "team_name": "Milwaukee Brewers",
        },
    ]
    stats = normalize_team_batting_stats(raw)
    factors = calculate_matchup_factors(stats, dampening=0.5)

    pitcher = _make_pitcher("Ace", "NYY", 3.50, 1.15, 200, 12, 0, 180)

    # Facing COL (weak offense, high K%) should be favorable
    adj_easy = adjust_pitcher_projection(pitcher, factors["COL"])
    assert adj_easy["era"] < 3.50
    assert adj_easy["k"] > 200

    # Facing LAD (strong offense, low K%) should be tough
    adj_hard = adjust_pitcher_projection(pitcher, factors["LAD"])
    assert adj_hard["era"] > 3.50
    assert adj_hard["k"] < 200

    # Two-start pitcher facing both -> blended
    adj_both = adjust_pitcher_projection(pitcher, [factors["COL"], factors["LAD"]])
    assert adj_easy["era"] < adj_both["era"] < adj_hard["era"]


def _make_sched(pps):
    return {"probable_pitchers": pps}


def _pitcher(name, team="SEA"):
    p = Player(name=name, player_type=PlayerType.PITCHER, positions=[Position.SP])
    p.team = team
    return p


class TestGetProbableStartersV2:
    def test_announced_only_passes_through(self):
        sched = _make_sched(
            [
                {
                    "date": "2026-05-05",
                    "game_number": 1,
                    "away_team": "SEA",
                    "home_team": "LAD",
                    "away_pitcher": "Bryan Woo",
                    "home_pitcher": "TBD",
                },
            ]
        )
        team_stats = {"LAD": {"ops": 0.800, "k_pct": 0.20}}
        out = get_probable_starters(
            pitcher_roster=[_pitcher("Bryan Woo")],
            schedule=sched,
            matchup_factors={"LAD": {"era_whip_factor": 1.10, "k_factor": 0.95}},
            team_stats=team_stats,
            today=date(2026, 5, 5),
            window_start=date(2026, 5, 5),
            window_end=date(2026, 5, 11),
        )
        assert len(out) == 1
        assert out[0]["pitcher"] == "Bryan Woo"
        assert out[0]["starts"] == 1
        assert out[0]["matchups"][0]["announced"] is True

    def test_projected_added_when_no_announcement(self):
        # Anchor in lookback (May 1), team has no off-day; projection -> May 6.
        pps = [
            {
                "date": d,
                "game_number": 1,
                "away_team": "SEA",
                "home_team": "LAD",
                "away_pitcher": ann,
                "home_pitcher": "TBD",
            }
            for d, ann in [
                ("2026-05-01", "Bryan Woo"),  # anchor
                ("2026-05-02", ""),
                ("2026-05-03", ""),
                ("2026-05-04", ""),
                ("2026-05-05", ""),
                ("2026-05-06", ""),  # +5 -> projected
            ]
        ]
        out = get_probable_starters(
            pitcher_roster=[_pitcher("Bryan Woo")],
            schedule={"probable_pitchers": pps},
            matchup_factors={"LAD": {"era_whip_factor": 1.0, "k_factor": 1.0}},
            team_stats={"LAD": {"ops": 0.750, "k_pct": 0.22}},
            today=date(2026, 5, 5),
            window_start=date(2026, 5, 5),
            window_end=date(2026, 5, 11),
        )
        assert len(out) == 1
        assert out[0]["starts"] == 1
        assert out[0]["matchups"][0]["date"] == "2026-05-06"
        assert out[0]["matchups"][0]["announced"] is False

    def test_pitcher_with_no_starts_is_excluded(self):
        out = get_probable_starters(
            pitcher_roster=[_pitcher("Ghost Pitcher")],
            schedule={
                "probable_pitchers": [
                    {
                        "date": "2026-05-05",
                        "game_number": 1,
                        "away_team": "SEA",
                        "home_team": "LAD",
                        "away_pitcher": "TBD",
                        "home_pitcher": "TBD",
                    },
                ]
            },
            matchup_factors={},
            team_stats={},
            today=date(2026, 5, 5),
            window_start=date(2026, 5, 5),
            window_end=date(2026, 5, 11),
        )
        assert out == []

    def test_player_team_used_to_select_team_games(self):
        # Pitcher's team is NYY; their LAD game shouldn't be considered.
        sched = _make_sched(
            [
                {
                    "date": "2026-05-05",
                    "game_number": 1,
                    "away_team": "SEA",
                    "home_team": "LAD",
                    "away_pitcher": "Bryan Woo",
                    "home_pitcher": "TBD",
                },
                {
                    "date": "2026-05-05",
                    "game_number": 1,
                    "away_team": "NYY",
                    "home_team": "BOS",
                    "away_pitcher": "Gerrit Cole",
                    "home_pitcher": "TBD",
                },
            ]
        )
        out = get_probable_starters(
            pitcher_roster=[_pitcher("Gerrit Cole", team="NYY")],
            schedule=sched,
            matchup_factors={"BOS": {"era_whip_factor": 1.0, "k_factor": 1.0}},
            team_stats={"BOS": {"ops": 0.750, "k_pct": 0.22}},
            today=date(2026, 5, 5),
            window_start=date(2026, 5, 5),
            window_end=date(2026, 5, 11),
        )
        assert len(out) == 1
        assert out[0]["matchups"][0]["opponent"] == "BOS"
