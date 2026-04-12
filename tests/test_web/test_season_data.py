import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fantasy_baseball.web import season_data
from fantasy_baseball.web.season_data import read_cache, write_cache, read_meta, format_standings_for_display, format_monte_carlo_for_display, format_lineup_for_display, _standings_to_snapshot


def test_write_and_read_cache(tmp_path):
    data = {"teams": [{"name": "Hart of the Order", "total": 67}]}
    write_cache("standings", data, cache_dir=tmp_path)
    result = read_cache("standings", cache_dir=tmp_path)
    assert result == data


def test_read_cache_missing_file(tmp_path):
    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None


def test_read_cache_corrupt_json(tmp_path):
    path = tmp_path / "standings.json"
    path.write_text("not json", encoding="utf-8")
    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None


def test_read_meta_missing(tmp_path):
    result = read_meta(cache_dir=tmp_path)
    assert result == {}


def test_write_cache_overwrites(tmp_path):
    write_cache("standings", {"v": 1}, cache_dir=tmp_path)
    write_cache("standings", {"v": 2}, cache_dir=tmp_path)
    result = read_cache("standings", cache_dir=tmp_path)
    assert result == {"v": 2}


def _sample_standings():
    """10-team standings data as returned by yahoo_roster.fetch_standings()."""
    teams = [
        ("Hart of the Order", {"R": 300, "HR": 90, "RBI": 290, "SB": 50, "AVG": 0.270,
                               "W": 35, "K": 600, "SV": 25, "ERA": 3.50, "WHIP": 1.18}),
        ("SkeleThor", {"R": 310, "HR": 85, "RBI": 295, "SB": 40, "AVG": 0.265,
                       "W": 38, "K": 580, "SV": 30, "ERA": 3.40, "WHIP": 1.15}),
        ("Send in the Cavalli", {"R": 280, "HR": 95, "RBI": 280, "SB": 55, "AVG": 0.260,
                                 "W": 30, "K": 620, "SV": 20, "ERA": 3.60, "WHIP": 1.22}),
    ]
    return [{"name": n, "team_key": f"key_{i}", "rank": i + 1, "stats": s}
            for i, (n, s) in enumerate(teams)]


def test_format_standings_has_roto_points():
    data = format_standings_for_display(_standings_to_snapshot(_sample_standings()), "Hart of the Order")
    assert "teams" in data
    hart = next(t for t in data["teams"] if t["name"] == "Hart of the Order")
    assert "roto_points" in hart
    assert "total" in hart["roto_points"]


def test_format_standings_color_codes_all_teams():
    data = format_standings_for_display(_standings_to_snapshot(_sample_standings()), "Hart of the Order")
    hart = next(t for t in data["teams"] if t["name"] == "Hart of the Order")
    assert hart["is_user"] is True
    assert "color_classes" in hart
    # With 3 teams, ranks 1-2 = rank-top, 3 = rank-high
    # Hart has highest SB (50) → rank 1 → rank-top
    assert hart["color_classes"]["SB"] == "rank-top"
    # Non-user teams also get color classes
    skel = next(t for t in data["teams"] if t["name"] == "SkeleThor")
    assert skel["color_classes"]["SB"] != ""


def test_format_standings_tied_teams_same_color():
    """Teams tied in a category should get the same rank-based color class."""
    standings = _standings_to_snapshot([
        {"name": "Team A", "team_key": "a", "rank": 1,
         "stats": {"R": 100, "HR": 30, "RBI": 90, "SB": 20, "AVG": 0.260,
                   "W": 10, "K": 200, "SV": 10, "ERA": 3.50, "WHIP": 1.20}},
        {"name": "Team B", "team_key": "b", "rank": 2,
         "stats": {"R": 100, "HR": 25, "RBI": 90, "SB": 15, "AVG": 0.255,
                   "W": 8, "K": 180, "SV": 8, "ERA": 3.80, "WHIP": 1.25}},
    ])
    data = format_standings_for_display(standings, "Team A")
    a = next(t for t in data["teams"] if t["name"] == "Team A")
    b = next(t for t in data["teams"] if t["name"] == "Team B")
    # R and RBI are tied — must get the same color class
    assert a["color_classes"]["R"] == b["color_classes"]["R"]
    assert a["color_classes"]["RBI"] == b["color_classes"]["RBI"]


def _sample_monte_carlo():
    return {
        "team_results": {
            "Hart of the Order": {
                "median_pts": 68.5, "p10": 58, "p90": 76,
                "first_pct": 18.3, "top3_pct": 52.1,
            },
            "SkeleThor": {
                "median_pts": 65.0, "p10": 55, "p90": 73,
                "first_pct": 14.7, "top3_pct": 41.8,
            },
        },
        "category_risk": {
            "R": {"median_pts": 7, "p10": 5, "p90": 9, "top3_pct": 62, "bot3_pct": 8},
            "SV": {"median_pts": 4, "p10": 2, "p90": 7, "top3_pct": 22, "bot3_pct": 38},
        },
    }


def test_format_monte_carlo_sorted_by_median():
    data = format_monte_carlo_for_display(
        _sample_monte_carlo(), "Hart of the Order"
    )
    assert data["teams"][0]["name"] == "Hart of the Order"
    assert data["teams"][0]["median_pts"] == 68.5
    assert data["teams"][0]["is_user"] is True


def test_format_monte_carlo_category_risk_colors():
    data = format_monte_carlo_for_display(
        _sample_monte_carlo(), "Hart of the Order"
    )
    risk = data["category_risk"]
    sv = next(r for r in risk if r["cat"] == "SV")
    assert sv["risk_class"] == "cat-bottom"
    r_cat = next(r for r in risk if r["cat"] == "R")
    assert r_cat["risk_class"] == "cat-top"


# --- format_lineup_for_display tests ---

def _sample_roster():
    return [
        {"name": "Adley Rutschman", "positions": ["C"], "selected_position": "C",
         "player_id": "123", "status": ""},
        {"name": "Mike Trout", "positions": ["OF"], "selected_position": "OF",
         "player_id": "456", "status": "IL"},
        {"name": "Masataka Yoshida", "positions": ["OF", "UTIL"], "selected_position": "BN",
         "player_id": "789", "status": ""},
    ]


def _sample_optimal():
    return {
        "hitters": {"C": "Adley Rutschman", "OF": "Masataka Yoshida"},
        "pitchers": {},
        "moves": [
            {"action": "START", "player": "Masataka Yoshida", "slot": "OF", "reason": "wSGP: 1.9"},
            {"action": "BENCH", "player": "Mike Trout", "slot": "IL", "reason": "IL-eligible"},
        ],
    }


def test_format_lineup_separates_hitters_pitchers():
    data = format_lineup_for_display(_sample_roster(), _sample_optimal())
    assert "hitters" in data
    assert "pitchers" in data
    assert len(data["hitters"]) >= 2


def test_format_lineup_detects_suboptimal():
    data = format_lineup_for_display(_sample_roster(), _sample_optimal())
    assert data["is_optimal"] is False
    assert len(data["moves"]) == 2


def test_format_lineup_optimal_when_no_moves():
    optimal = {"hitters": {}, "pitchers": {}, "moves": []}
    data = format_lineup_for_display(_sample_roster(), optimal)
    assert data["is_optimal"] is True


def test_format_lineup_passes_ros_data_through():
    """ROS projection data on roster entries must survive format_lineup_for_display.

    After the Player-object refactor, ROS stats are flattened to top-level keys
    (e.g. entry["hr"], entry["r"]) so the Jinja2 template can access them via
    h[ros_key] — there is no longer a nested entry["ros"] dict.
    """
    roster = [
        {"name": "Aaron Judge", "positions": ["OF"], "selected_position": "OF",
         "player_id": "1", "status": "", "wsgp": 5.0,
         "ros": {"r": 90, "hr": 40, "rbi": 100, "sb": 5, "avg": 0.280}},
        {"name": "No ROS Player", "positions": ["1B"], "selected_position": "1B",
         "player_id": "2", "status": "", "wsgp": 2.0},
    ]
    data = format_lineup_for_display(roster, {"moves": []})
    judge = next(h for h in data["hitters"] if h["name"] == "Aaron Judge")
    no_ros = next(h for h in data["hitters"] if h["name"] == "No ROS Player")
    # ROS stats are flattened to top level for template h[ros_key] access
    assert judge["hr"] == 40
    assert judge["r"] == 90
    assert judge["rbi"] == 100
    assert judge["sb"] == 5
    assert abs(judge["avg"] - 0.280) < 1e-9
    # Player with no ROS data has no flat stat keys
    assert "hr" not in no_ros
    assert "r" not in no_ros


def test_roster_cache_includes_stats(tmp_path, monkeypatch):
    """After refresh, roster entries should include a 'pace' dict."""
    roster = [
        {"name": "Juan Soto", "positions": ["OF"], "selected_position": "OF",
         "player_id": "1", "status": "", "wsgp": 3.0, "player_type": "hitter",
         "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "pa": 600, "avg": 0.278,
         "pace": {
             "PA": {"actual": 102, "color_class": "stat-neutral"},
             "R": {"actual": 19, "expected": 15.3, "z_score": 1.2, "color_class": "stat-hot-2", "projection": 90},
             "HR": {"actual": 9, "expected": 5.1, "z_score": 1.6, "color_class": "stat-hot-2", "projection": 30},
             "RBI": {"actual": 18, "expected": 15.3, "z_score": 0.3, "color_class": "stat-neutral", "projection": 90},
             "SB": {"actual": 2, "expected": 1.7, "z_score": 0.2, "color_class": "stat-neutral", "projection": 10},
             "AVG": {"actual": 0.298, "expected": 0.278, "z_score": 0.7, "color_class": "stat-hot-1", "projection": 0.278},
         }},
    ]
    result = format_lineup_for_display(roster, {"moves": []})
    assert "pace" in result["hitters"][0]
    assert result["hitters"][0]["pace"]["HR"]["actual"] == 9


@pytest.fixture()
def reset_redis_singleton():
    """Reset Redis singleton state before and after each test."""
    season_data._redis_client = None
    season_data._redis_initialized = False
    yield
    season_data._redis_client = None
    season_data._redis_initialized = False


def test_get_redis_returns_none_when_unconfigured(monkeypatch, reset_redis_singleton):
    """With no env vars, _get_redis() returns None."""
    monkeypatch.delenv("UPSTASH_REDIS_REST_URL", raising=False)
    monkeypatch.delenv("UPSTASH_REDIS_REST_TOKEN", raising=False)
    result = season_data._get_redis()
    assert result is None


def test_get_redis_returns_client_when_configured(monkeypatch, reset_redis_singleton):
    """With env vars set, _get_redis() returns a Redis client."""
    monkeypatch.setenv("UPSTASH_REDIS_REST_URL", "https://fake.upstash.io")
    monkeypatch.setenv("UPSTASH_REDIS_REST_TOKEN", "fake-token")
    with patch("upstash_redis.Redis") as MockRedis:
        MockRedis.return_value = "mock-client"
        result = season_data._get_redis()
        assert result == "mock-client"
        MockRedis.assert_called_once_with(url="https://fake.upstash.io", token="fake-token")


def test_write_cache_writes_to_redis(tmp_path, monkeypatch):
    """write_cache with default cache_dir writes to Redis."""
    mock_redis = type("MockRedis", (), {"set": lambda self, k, v: None})()
    mock_redis.set = lambda k, v: setattr(mock_redis, "_last_set", (k, v))
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    data = {"teams": [1, 2, 3]}
    write_cache("standings", data, cache_dir=tmp_path)

    assert mock_redis._last_set[0] == "cache:standings"
    assert json.loads(mock_redis._last_set[1]) == data


def test_write_cache_skips_redis_non_default_dir(tmp_path, monkeypatch):
    """write_cache with non-default cache_dir does not touch Redis."""
    mock_redis = MagicMock()
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    # tmp_path != CACHE_DIR, so Redis should be skipped
    data = {"v": 1}
    write_cache("standings", data, cache_dir=tmp_path)
    mock_redis.set.assert_not_called()
    assert read_cache("standings", cache_dir=tmp_path) == data


def test_write_cache_handles_redis_error(tmp_path, monkeypatch):
    """write_cache continues if Redis raises a network error."""
    mock_redis = MagicMock()
    mock_redis.set.side_effect = ConnectionError("Upstash unreachable")
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    data = {"teams": [1, 2, 3]}
    write_cache("standings", data, cache_dir=tmp_path)
    # Local write still succeeded
    assert read_cache("standings", cache_dir=tmp_path) == data


# --- read_cache Redis fallback tests ---

def test_read_cache_falls_back_to_redis(tmp_path, monkeypatch):
    """When local disk has no file, read_cache fetches from Redis and writes back locally."""
    data = {"teams": [1, 2, 3]}
    mock_redis = type("MockRedis", (), {"get": lambda self, k: json.dumps(data)})()
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    result = read_cache("standings", cache_dir=tmp_path)
    assert result == data
    # Verify it wrote back to local disk
    local = json.loads((tmp_path / "standings.json").read_text(encoding="utf-8"))
    assert local == data


def test_read_cache_returns_none_when_both_miss(tmp_path, monkeypatch):
    """When local disk and Redis both miss, returns None."""
    mock_redis = type("MockRedis", (), {"get": lambda self, k: None})()
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None


def test_read_cache_handles_corrupt_redis_data(tmp_path, monkeypatch):
    """When Redis returns non-JSON, treat as miss."""
    mock_redis = type("MockRedis", (), {"get": lambda self, k: "not-json{{"})()
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None


def test_read_cache_skips_redis_non_default_dir(tmp_path, monkeypatch):
    """read_cache with non-default cache_dir does not touch Redis."""
    mock_redis = MagicMock()
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    # tmp_path != CACHE_DIR, so Redis should be skipped
    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None
    mock_redis.get.assert_not_called()


def test_read_cache_handles_redis_error(tmp_path, monkeypatch):
    """read_cache returns None if Redis raises a network error."""
    mock_redis = MagicMock()
    mock_redis.get.side_effect = ConnectionError("Upstash unreachable")
    monkeypatch.setattr(season_data, "_get_redis", lambda: mock_redis)
    monkeypatch.setattr(season_data, "CACHE_DIR", tmp_path)

    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None


class TestComputeComparisonStandings:
    def test_swap_changes_user_team_stats(self):
        """Swapping a hitter should change the user's projected stats and roto points."""
        from fantasy_baseball.web.season_data import compute_comparison_standings
        from fantasy_baseball.models.player import Player, HitterStats, PitcherStats

        projected_standings = [
            {"name": "My Team", "team_key": "", "rank": 0, "stats": {
                "R": 700, "HR": 200, "RBI": 700, "SB": 100, "AVG": 0.260,
                "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.20,
            }},
            {"name": "Other Team", "team_key": "", "rank": 0, "stats": {
                "R": 680, "HR": 190, "RBI": 680, "SB": 110, "AVG": 0.255,
                "W": 85, "K": 1100, "SV": 40, "ERA": 3.80, "WHIP": 1.25,
            }},
        ]

        roster = [
            Player(name="Willy Adames", player_type="hitter",
                   ros=HitterStats(pa=650, ab=567, h=133, r=80, hr=25, rbi=81, sb=11, avg=0.235)),
            Player(name="Other Hitter", player_type="hitter",
                   ros=HitterStats(pa=630, ab=550, h=150, r=90, hr=30, rbi=95, sb=5, avg=0.273)),
            Player(name="My Pitcher", player_type="pitcher",
                   ros=PitcherStats(ip=180, w=12, k=180, sv=0, er=60, bb=50, h_allowed=150,
                                    era=3.00, whip=1.11)),
        ]

        other_player = Player(
            name="Ezequiel Tovar", player_type="hitter",
            ros=HitterStats(pa=590, ab=513, h=135, r=73, hr=20, rbi=74, sb=8, avg=0.263),
        )

        result = compute_comparison_standings(
            roster_player_name="Willy Adames",
            other_player=other_player,
            user_roster=roster,
            projected_standings=projected_standings,
            user_team_name="My Team",
        )

        assert "before" in result
        assert "after" in result
        assert "categories" in result

        assert result["before"]["roto"]["My Team"]["total"] != result["after"]["roto"]["My Team"]["total"]
        assert result["before"]["stats"]["My Team"] != result["after"]["stats"]["My Team"]
        assert result["before"]["stats"]["Other Team"] == result["after"]["stats"]["Other Team"]

    def test_swap_not_found_returns_error(self):
        """If roster_player_name doesn't match anyone in user_roster, return error."""
        from fantasy_baseball.web.season_data import compute_comparison_standings
        from fantasy_baseball.models.player import Player, HitterStats

        result = compute_comparison_standings(
            roster_player_name="Nobody",
            other_player=Player(name="X", player_type="hitter",
                                ros=HitterStats(pa=0, ab=0, h=0, r=0, hr=0, rbi=0, sb=0)),
            user_roster=[Player(name="A", player_type="hitter",
                                ros=HitterStats(pa=350, ab=300, h=80, r=50, hr=10, rbi=40, sb=5))],
            projected_standings=[{"name": "My Team", "team_key": "", "rank": 0,
                                  "stats": {"R": 700, "HR": 200, "RBI": 700, "SB": 100,
                                            "AVG": 0.260, "W": 80, "K": 1200, "SV": 50,
                                            "ERA": 3.50, "WHIP": 1.20}}],
            user_team_name="My Team",
        )
        assert "error" in result


class TestComparisonConsistencyInvariant:
    """Document and test compute_comparison_standings' contract.

    These tests verify the function is internally consistent with its own
    inputs: when given a user_roster and a projected_standings entry, the
    "before" output equals project_team_stats(user_roster), and per-player
    counting-stat deltas equal team-stat deltas in a swap.

    They do NOT catch the original Arozarena/Suarez bug — that bug was
    that the refresh pipeline wrote inconsistent data into cache:roster
    (recency-blended) and cache:projections (raw), so two callers of this
    function were fed different player stats. The structural fix
    (removing recency blending so both caches derive from the same source)
    plus the grep guardrail in test_no_recency_blending.py are the actual
    protection. These tests are an additional sanity check on the
    function under test.
    """

    def _build_roster(self):
        from fantasy_baseball.models.player import Player, HitterStats, PitcherStats
        return [
            Player(name="Star Hitter", player_type="hitter",
                   ros=HitterStats(pa=650, ab=567, h=150, r=90, hr=30, rbi=95, sb=25, avg=0.265)),
            Player(name="Role Hitter", player_type="hitter",
                   ros=HitterStats(pa=500, ab=440, h=110, r=55, hr=15, rbi=55, sb=5, avg=0.250)),
            Player(name="Ace", player_type="pitcher",
                   ros=PitcherStats(ip=180, w=14, k=200, sv=0, er=60, bb=50,
                                    h_allowed=150, era=3.00, whip=1.11)),
        ]

    def test_before_stats_equal_projected_standings_entry(self):
        """The comparison "before" for the user team must equal the value stored
        in projected_standings (otherwise UI shows one number on the standings
        page and a different number on the comparison page for the same team)."""
        from fantasy_baseball.web.season_data import compute_comparison_standings
        from fantasy_baseball.models.player import Player, HitterStats
        from fantasy_baseball.scoring import project_team_stats

        roster = self._build_roster()
        # Build projected_standings so the user entry matches what
        # project_team_stats would produce — this mirrors a consistent refresh.
        user_stats = project_team_stats(roster)
        projected_standings = [
            {"name": "My Team", "team_key": "", "rank": 0, "stats": dict(user_stats)},
            {"name": "Rival", "team_key": "", "rank": 0, "stats": {
                "R": 680, "HR": 190, "RBI": 680, "SB": 110, "AVG": 0.255,
                "W": 85, "K": 1100, "SV": 40, "ERA": 3.80, "WHIP": 1.25,
            }},
        ]
        other_player = Player(
            name="Replacement", player_type="hitter",
            ros=HitterStats(pa=600, ab=530, h=140, r=75, hr=28, rbi=85, sb=3, avg=0.264),
        )

        result = compute_comparison_standings(
            roster_player_name="Star Hitter",
            other_player=other_player,
            user_roster=roster,
            projected_standings=projected_standings,
            user_team_name="My Team",
        )

        # Invariant: before user-team stats equal the projected_standings entry
        for cat, val in user_stats.items():
            assert result["before"]["stats"]["My Team"][cat] == val, (
                f"before[{cat}] diverged from projected_standings "
                f"({result['before']['stats']['My Team'][cat]} vs {val}) — "
                "this is the Arozarena/Suarez bug"
            )

    def test_swap_delta_equals_player_stat_difference(self):
        """The team-stat drop from swapping must equal the counting-stat
        difference between the two players. If it doesn't, the UI's stat
        comparison row will disagree with the team standings panel."""
        from fantasy_baseball.web.season_data import compute_comparison_standings
        from fantasy_baseball.models.player import Player, HitterStats
        from fantasy_baseball.scoring import project_team_stats

        roster = self._build_roster()
        user_stats = project_team_stats(roster)
        projected_standings = [
            {"name": "My Team", "team_key": "", "rank": 0, "stats": dict(user_stats)},
        ]

        dropped = roster[0]  # Star Hitter
        other_player = Player(
            name="Replacement", player_type="hitter",
            ros=HitterStats(pa=600, ab=530, h=140, r=75, hr=28, rbi=85, sb=3, avg=0.264),
        )

        result = compute_comparison_standings(
            roster_player_name=dropped.name,
            other_player=other_player,
            user_roster=roster,
            projected_standings=projected_standings,
            user_team_name="My Team",
        )

        for cat_attr, roto_cat in [("r", "R"), ("hr", "HR"), ("rbi", "RBI"), ("sb", "SB")]:
            player_delta = getattr(dropped.ros, cat_attr) - getattr(other_player.ros, cat_attr)
            team_delta = (
                result["before"]["stats"]["My Team"][roto_cat]
                - result["after"]["stats"]["My Team"][roto_cat]
            )
            assert abs(team_delta - player_delta) < 1e-9, (
                f"{roto_cat}: team delta {team_delta} != player delta {player_delta}"
            )

    def test_user_roster_takes_precedence_over_projected_standings_entry(self):
        """compute_comparison_standings recomputes the user team's "before"
        stats from project_team_stats(user_roster). Any pre-existing user
        entry in projected_standings is ignored. This means the refresh
        pipeline must keep cache:roster and cache:projections in sync —
        if they diverge, the comparison page silently disagrees with the
        standings page."""
        from fantasy_baseball.web.season_data import compute_comparison_standings
        from fantasy_baseball.models.player import Player, HitterStats
        from fantasy_baseball.scoring import project_team_stats

        roster = self._build_roster()
        true_user_stats = project_team_stats(roster)

        # Build projected_standings with a DELIBERATELY wrong entry for the user
        # team — this simulates what would happen if cache:roster and
        # cache:projections were written from different sources (the original
        # Arozarena/Suarez bug condition).
        wrong_user_stats = {k: v + 999 for k, v in true_user_stats.items()}
        projected_standings = [
            {"name": "My Team", "team_key": "", "rank": 0, "stats": wrong_user_stats},
        ]

        other_player = Player(
            name="Replacement", player_type="hitter",
            ros=HitterStats(pa=600, ab=530, h=140, r=75, hr=28, rbi=85, sb=3, avg=0.264),
        )

        result = compute_comparison_standings(
            roster_player_name="Star Hitter",
            other_player=other_player,
            user_roster=roster,
            projected_standings=projected_standings,
            user_team_name="My Team",
        )

        # The function recomputes from user_roster, ignoring the wrong entry.
        # If the refresh pipeline ever writes inconsistent data into the two
        # caches, this is the asymmetry that produces UI disagreement.
        assert result["before"]["stats"]["My Team"]["HR"] == true_user_stats["HR"]
        assert result["before"]["stats"]["My Team"]["HR"] != wrong_user_stats["HR"]


class TestRunFullRefreshScopingGuards:
    """Regression guards for Python scoping bugs in run_full_refresh.

    run_full_refresh is a long function that imports modules lazily
    inside conditional branches. Any local ``from X import Y`` that
    shadows a module-level name promotes that name to a local variable
    for the entire function scope (LEGB rule), causing UnboundLocalError
    when the name is referenced earlier in the function.

    Production bug landed on 2026-04-12: Step 2 of the League data
    model refactor started using ``date.fromisoformat(end_date)`` near
    the top of run_full_refresh to compute effective_date. A
    pre-existing local ``from datetime import date`` inside the
    Monte Carlo block shadowed the module-level import, so the
    function crashed with ``UnboundLocalError: cannot access local
    variable 'date' where it is not associated with a value``.
    """

    def test_no_local_datetime_import_inside_run_full_refresh(self):
        import inspect

        from fantasy_baseball.web import season_data

        src = inspect.getsource(season_data.run_full_refresh)
        assert "from datetime import date" not in src, (
            "Local `from datetime import date` found in run_full_refresh. "
            "This shadows the module-level import on line 9 and causes "
            "UnboundLocalError when date is used earlier in the function. "
            "If you need to import date inside the function, rename it "
            "(e.g. `from datetime import date as _date`) to avoid shadowing."
        )
        assert "from datetime import datetime" not in src, (
            "Local `from datetime import datetime` would shadow the "
            "module-level import for the same reason as the date case."
        )
        assert "from datetime import timedelta" not in src, (
            "Local `from datetime import timedelta` would shadow the "
            "module-level import for the same reason as the date case."
        )

    def test_module_level_date_import_is_present(self):
        """Confirms the module-level import exists so the function can
        rely on ``date`` being available without a local import."""
        import inspect

        from fantasy_baseball.web import season_data

        module_src = inspect.getsource(season_data)
        # The module-level import must appear before the run_full_refresh
        # definition, not inside it.
        first_refresh_idx = module_src.find("def run_full_refresh")
        assert first_refresh_idx > 0
        header = module_src[:first_refresh_idx]
        assert "from datetime import date" in header
