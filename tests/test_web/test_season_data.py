import json
from pathlib import Path

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


def test_format_standings_color_codes_user_team():
    data = format_standings_for_display(_sample_standings(), "Hart of the Order")
    hart = next(t for t in data["teams"] if t["name"] == "Hart of the Order")
    assert hart["is_user"] is True
    assert "color_classes" in hart
    # With 3 teams, ranks 1 = top, 3 = bottom
    # Hart has highest SB (50) → should be cat-top
    assert hart["color_classes"]["SB"] == "cat-top"


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
