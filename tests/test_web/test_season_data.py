import json
from pathlib import Path

from fantasy_baseball.web.season_data import read_cache, write_cache, read_meta, format_standings_for_display


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
