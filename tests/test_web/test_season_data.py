import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from fantasy_baseball.web import season_data
from fantasy_baseball.web.season_data import read_cache, write_cache, read_meta, format_standings_for_display, format_monte_carlo_for_display, format_lineup_for_display


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
    data = format_standings_for_display(_sample_standings(), "Hart of the Order")
    assert "teams" in data
    hart = next(t for t in data["teams"] if t["name"] == "Hart of the Order")
    assert "roto_points" in hart
    assert "total" in hart["roto_points"]


def test_format_standings_color_codes_all_teams():
    data = format_standings_for_display(_sample_standings(), "Hart of the Order")
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
    standings = [
        {"name": "Team A", "team_key": "a", "rank": 1,
         "stats": {"R": 100, "HR": 30, "RBI": 90, "SB": 20, "AVG": 0.260,
                   "W": 10, "K": 200, "SV": 10, "ERA": 3.50, "WHIP": 1.20}},
        {"name": "Team B", "team_key": "b", "rank": 2,
         "stats": {"R": 100, "HR": 25, "RBI": 90, "SB": 15, "AVG": 0.255,
                   "W": 8, "K": 180, "SV": 8, "ERA": 3.80, "WHIP": 1.25}},
    ]
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
    """ROS projection data on roster entries must survive format_lineup_for_display."""
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
    assert judge["ros"] == {"r": 90, "hr": 40, "rbi": 100, "sb": 5, "avg": 0.280}
    assert no_ros["ros"] is None


def test_roster_cache_includes_stats(tmp_path, monkeypatch):
    """After refresh, roster entries should include a 'stats' dict."""
    roster = [
        {"name": "Juan Soto", "positions": ["OF"], "selected_position": "OF",
         "player_id": "1", "status": "", "wsgp": 3.0, "player_type": "hitter",
         "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "pa": 600, "avg": 0.278,
         "stats": {
             "PA": {"actual": 102, "color_class": "stat-neutral"},
             "R": {"actual": 19, "expected": 15.3, "z_score": 1.2, "color_class": "stat-hot-2", "projection": 90},
             "HR": {"actual": 9, "expected": 5.1, "z_score": 1.6, "color_class": "stat-hot-2", "projection": 30},
             "RBI": {"actual": 18, "expected": 15.3, "z_score": 0.3, "color_class": "stat-neutral", "projection": 90},
             "SB": {"actual": 2, "expected": 1.7, "z_score": 0.2, "color_class": "stat-neutral", "projection": 10},
             "AVG": {"actual": 0.298, "expected": 0.278, "z_score": 0.7, "color_class": "stat-hot-1", "projection": 0.278},
         }},
    ]
    result = format_lineup_for_display(roster, {"moves": []})
    assert "stats" in result["hitters"][0]
    assert result["hitters"][0]["stats"]["HR"]["actual"] == 9


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
