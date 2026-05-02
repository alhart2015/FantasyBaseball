import json
from datetime import date

import pytest

from fantasy_baseball.data import kv_store
from fantasy_baseball.data.cache_keys import redis_key
from fantasy_baseball.models.standings import (
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
    Standings,
    StandingsEntry,
)
from fantasy_baseball.utils.constants import Category
from fantasy_baseball.web import season_data
from fantasy_baseball.web.season_data import (
    CacheKey,
    format_lineup_for_display,
    format_monte_carlo_for_display,
    format_standings_for_display,
    read_cache,
    read_meta,
    write_cache,
)


@pytest.fixture(autouse=True)
def _isolated_kv(tmp_path, monkeypatch):
    """Per-test isolated SQLite KV via FANTASY_LOCAL_KV_PATH.

    Required after Phase 1 of the cache refactor: read_cache/write_cache
    now route through ``kv_store.get_kv()`` instead of reading JSON files
    out of the test's ``tmp_path``. Without per-test KV isolation, every
    test would share ``data/local.db`` and stomp on each other.
    """
    monkeypatch.setenv("FANTASY_LOCAL_KV_PATH", str(tmp_path / "test.db"))
    kv_store._reset_singleton()
    yield
    kv_store._reset_singleton()


def _standings_from_raw(raw: list[dict]) -> Standings:
    """Build a Standings from a list[dict] fixture used by the old tests."""
    return Standings(
        effective_date=date(2026, 4, 1),
        entries=[
            StandingsEntry(
                team_name=t["name"],
                team_key=t.get("team_key", ""),
                rank=t.get("rank", 0),
                stats=CategoryStats.from_dict(t.get("stats", {})),
                yahoo_points_for=t.get("points_for"),
            )
            for t in raw
        ],
    )


def _projected_from_raw(raw: list[dict]) -> ProjectedStandings:
    """Build a ProjectedStandings from a list[dict] fixture."""
    return ProjectedStandings(
        effective_date=date(2026, 4, 1),
        entries=[
            ProjectedStandingsEntry(
                team_name=t["name"],
                stats=CategoryStats.from_dict(t.get("stats", {})),
            )
            for t in raw
        ],
    )


def test_write_and_read_cache():
    data = {"teams": [{"name": "Hart of the Order", "total": 67}]}
    write_cache(CacheKey.STANDINGS, data)
    assert read_cache(CacheKey.STANDINGS) == data


def test_read_cache_missing_file():
    assert read_cache(CacheKey.STANDINGS) is None


def test_read_cache_corrupt_json():
    """Corrupt JSON in the KV value is treated as a miss, not a hard error."""
    kv_store.get_kv().set(redis_key(CacheKey.STANDINGS), "not json{{")
    assert read_cache(CacheKey.STANDINGS) is None


def test_read_meta_missing():
    assert read_meta() == {}


def test_write_cache_overwrites():
    write_cache(CacheKey.STANDINGS, {"v": 1})
    write_cache(CacheKey.STANDINGS, {"v": 2})
    assert read_cache(CacheKey.STANDINGS) == {"v": 2}


def _sample_standings():
    """10-team standings data as returned by yahoo_roster.fetch_standings()."""
    teams = [
        (
            "Hart of the Order",
            {
                "R": 300,
                "HR": 90,
                "RBI": 290,
                "SB": 50,
                "AVG": 0.270,
                "W": 35,
                "K": 600,
                "SV": 25,
                "ERA": 3.50,
                "WHIP": 1.18,
            },
        ),
        (
            "SkeleThor",
            {
                "R": 310,
                "HR": 85,
                "RBI": 295,
                "SB": 40,
                "AVG": 0.265,
                "W": 38,
                "K": 580,
                "SV": 30,
                "ERA": 3.40,
                "WHIP": 1.15,
            },
        ),
        (
            "Send in the Cavalli",
            {
                "R": 280,
                "HR": 95,
                "RBI": 280,
                "SB": 55,
                "AVG": 0.260,
                "W": 30,
                "K": 620,
                "SV": 20,
                "ERA": 3.60,
                "WHIP": 1.22,
            },
        ),
    ]
    return [
        {"name": n, "team_key": f"key_{i}", "rank": i + 1, "stats": s}
        for i, (n, s) in enumerate(teams)
    ]


def test_format_standings_has_roto_points():
    data = format_standings_for_display(
        _standings_from_raw(_sample_standings()), "Hart of the Order"
    )
    assert "teams" in data
    hart = next(t for t in data["teams"] if t["name"] == "Hart of the Order")
    assert "roto_points" in hart
    # Per-category roto_points dict is now Category-enum-keyed; the total
    # lives at the top-level "roto_total" key.
    assert Category.R in hart["roto_points"]
    assert "roto_total" in hart


def test_format_standings_color_intensity_per_team():
    data = format_standings_for_display(
        _standings_from_raw(_sample_standings()), "Hart of the Order"
    )
    hart = next(t for t in data["teams"] if t["name"] == "Hart of the Order")
    skel = next(t for t in data["teams"] if t["name"] == "SkeleThor")
    cav = next(t for t in data["teams"] if t["name"] == "Send in the Cavalli")

    assert hart["is_user"] is True
    assert "color_intensity" in hart
    # SB: Hart=50, SkeleThor=40, Cavalli=55 → Cavalli leads, SkeleThor trails.
    assert cav["color_intensity"][Category.SB] == pytest.approx(1.0)
    assert skel["color_intensity"][Category.SB] == pytest.approx(-1.0)
    # Hart sits in between: (50-40)/(55-40) = 0.667, intensity = 2*0.667-1 ≈ 0.333
    assert hart["color_intensity"][Category.SB] == pytest.approx(0.333, abs=0.01)
    # Total column intensity lives at the top level, not inside color_intensity.
    assert "total_intensity" in hart


def test_format_standings_prefers_yahoo_points_for():
    """When every entry carries points_for, displayed total matches Yahoo exactly.

    Guards against the bug where display-level ties (two teams shown
    WHIP=1.03) made our score_roto diverge from Yahoo by ±0.5. The
    fix: when points_for is present, use it directly for total and
    rank instead of recomputing.
    """
    # Two teams with display-level tied WHIP (1.03) but Yahoo-reported
    # points_for that tie-breaks in favor of Team A. Our score_roto alone
    # would give each 9.5 (averaged); Yahoo gives 10 / 9.
    standings = [
        {
            "name": "Team A",
            "team_key": "a",
            "rank": 1,
            "stats": {
                "R": 100,
                "HR": 30,
                "RBI": 90,
                "SB": 20,
                "AVG": 0.260,
                "W": 10,
                "K": 200,
                "SV": 10,
                "ERA": 3.50,
                "WHIP": 1.03,
            },
            "points_for": 20.0,
        },
        {
            "name": "Team B",
            "team_key": "b",
            "rank": 2,
            "stats": {
                "R": 90,
                "HR": 25,
                "RBI": 80,
                "SB": 15,
                "AVG": 0.255,
                "W": 8,
                "K": 180,
                "SV": 8,
                "ERA": 3.80,
                "WHIP": 1.03,
            },
            "points_for": 11.0,
        },
    ]
    data = format_standings_for_display(_standings_from_raw(standings), "Team A")
    by_name = {t["name"]: t for t in data["teams"]}

    # Totals are Yahoo's, not score_roto's averaged-tie output.
    assert by_name["Team A"]["roto_total"] == 20.0
    assert by_name["Team B"]["roto_total"] == 11.0
    # score_roto's original total preserved for diagnostics (top-level key).
    assert "score_roto_total" in by_name["Team A"]
    # Ranking comes from Yahoo's rank.
    assert by_name["Team A"]["rank"] == 1
    assert by_name["Team B"]["rank"] == 2


def test_format_standings_falls_back_without_points_for():
    """Projected standings (no points_for) still use score_roto."""
    standings = [
        {
            "name": "Team A",
            "team_key": "a",
            "rank": 0,
            "stats": {
                "R": 100,
                "HR": 30,
                "RBI": 90,
                "SB": 20,
                "AVG": 0.260,
                "W": 10,
                "K": 200,
                "SV": 10,
                "ERA": 3.50,
                "WHIP": 1.20,
            },
        },
        {
            "name": "Team B",
            "team_key": "b",
            "rank": 0,
            "stats": {
                "R": 90,
                "HR": 25,
                "RBI": 80,
                "SB": 15,
                "AVG": 0.255,
                "W": 8,
                "K": 180,
                "SV": 8,
                "ERA": 3.80,
                "WHIP": 1.25,
            },
        },
    ]
    data = format_standings_for_display(_standings_from_raw(standings), "Team A")
    by_name = {t["name"]: t for t in data["teams"]}
    # Team A dominates every category → score_roto total = 2 * 10 = 20
    assert by_name["Team A"]["roto_total"] == 20.0
    assert by_name["Team B"]["roto_total"] == 10.0
    assert by_name["Team A"]["rank"] == 1
    # Without Yahoo override, roto_total == score_roto_total.
    assert by_name["Team A"]["score_roto_total"] == 20.0


def test_format_standings_tied_category_has_no_intensity():
    """When every team is tied in a category (max == min), the key is absent."""
    standings = _standings_from_raw(
        [
            {
                "name": "Team A",
                "team_key": "a",
                "rank": 1,
                "stats": {
                    "R": 100,
                    "HR": 30,
                    "RBI": 90,
                    "SB": 20,
                    "AVG": 0.260,
                    "W": 10,
                    "K": 200,
                    "SV": 10,
                    "ERA": 3.50,
                    "WHIP": 1.20,
                },
            },
            {
                "name": "Team B",
                "team_key": "b",
                "rank": 2,
                "stats": {
                    "R": 100,
                    "HR": 25,
                    "RBI": 90,
                    "SB": 15,
                    "AVG": 0.255,
                    "W": 8,
                    "K": 180,
                    "SV": 8,
                    "ERA": 3.80,
                    "WHIP": 1.25,
                },
            },
        ]
    )
    data = format_standings_for_display(standings, "Team A")
    a = next(t for t in data["teams"] if t["name"] == "Team A")
    b = next(t for t in data["teams"] if t["name"] == "Team B")
    # R and RBI are tied across all teams — the key is absent for everyone.
    assert Category.R not in a["color_intensity"]
    assert Category.R not in b["color_intensity"]
    assert Category.RBI not in a["color_intensity"]
    assert Category.RBI not in b["color_intensity"]
    # Non-tied categories still populated.
    assert Category.HR in a["color_intensity"]


def test_format_standings_era_whip_inverted():
    """Lower ERA/WHIP → higher intensity (+1.0 at the min, not the max)."""
    standings = _standings_from_raw(
        [
            {
                "name": "LowEra",
                "team_key": "a",
                "rank": 1,
                "stats": {
                    "R": 100,
                    "HR": 30,
                    "RBI": 90,
                    "SB": 20,
                    "AVG": 0.260,
                    "W": 10,
                    "K": 200,
                    "SV": 10,
                    "ERA": 2.50,
                    "WHIP": 1.00,
                },
            },
            {
                "name": "HighEra",
                "team_key": "b",
                "rank": 2,
                "stats": {
                    "R": 90,
                    "HR": 25,
                    "RBI": 80,
                    "SB": 15,
                    "AVG": 0.255,
                    "W": 8,
                    "K": 180,
                    "SV": 8,
                    "ERA": 5.00,
                    "WHIP": 1.50,
                },
            },
        ]
    )
    data = format_standings_for_display(standings, "LowEra")
    low = next(t for t in data["teams"] if t["name"] == "LowEra")
    high = next(t for t in data["teams"] if t["name"] == "HighEra")
    # Lowest ERA / WHIP should read as the leader (+1.0).
    assert low["color_intensity"][Category.ERA] == pytest.approx(1.0)
    assert low["color_intensity"][Category.WHIP] == pytest.approx(1.0)
    assert high["color_intensity"][Category.ERA] == pytest.approx(-1.0)
    assert high["color_intensity"][Category.WHIP] == pytest.approx(-1.0)


def test_format_standings_clustered_leaders_share_intensity():
    """Three teams at 99 HR (one behind 100) should get the same intensity."""
    hr_values = [100, 99, 99, 99, 70, 65, 55, 50, 45, 40]
    teams = []
    for i, hr in enumerate(hr_values):
        teams.append(
            {
                "name": f"Team{i}",
                "team_key": f"k{i}",
                "rank": i + 1,
                "stats": {
                    "R": 100,
                    "HR": hr,
                    "RBI": 90,
                    "SB": 20,
                    "AVG": 0.260,
                    "W": 10,
                    "K": 200,
                    "SV": 10,
                    "ERA": 3.50,
                    "WHIP": 1.20,
                },
            }
        )
    data = format_standings_for_display(_standings_from_raw(teams), "Team0")
    by_name = {t["name"]: t for t in data["teams"]}
    # Leader at 100 → +1.0. Trailer at 40 → -1.0.
    assert by_name["Team0"]["color_intensity"][Category.HR] == pytest.approx(1.0)
    assert by_name["Team9"]["color_intensity"][Category.HR] == pytest.approx(-1.0)
    # The three 99s share the same intensity: (99-40)/(100-40)=0.9833, intensity=0.9667.
    expected = pytest.approx(0.9667, abs=0.001)
    assert by_name["Team1"]["color_intensity"][Category.HR] == expected
    assert by_name["Team2"]["color_intensity"][Category.HR] == expected
    assert by_name["Team3"]["color_intensity"][Category.HR] == expected


def test_format_standings_total_column_intensity():
    """Total column intensity tracks distance from top/bottom total roto points."""
    data = format_standings_for_display(
        _standings_from_raw(_sample_standings()), "Hart of the Order"
    )
    teams = data["teams"]
    # Team with highest total → +1.0; lowest → -1.0.
    top = max(teams, key=lambda t: t["roto_total"])
    bot = min(teams, key=lambda t: t["roto_total"])
    assert top["total_intensity"] == pytest.approx(1.0)
    assert bot["total_intensity"] == pytest.approx(-1.0)


def _sample_monte_carlo():
    return {
        "team_results": {
            "Hart of the Order": {
                "median_pts": 68.5,
                "p10": 58,
                "p90": 76,
                "first_pct": 18.3,
                "top3_pct": 52.1,
            },
            "SkeleThor": {
                "median_pts": 65.0,
                "p10": 55,
                "p90": 73,
                "first_pct": 14.7,
                "top3_pct": 41.8,
            },
        },
        "category_risk": {
            "R": {"median_pts": 7, "p10": 5, "p90": 9, "top3_pct": 62, "bot3_pct": 8},
            "SV": {"median_pts": 4, "p10": 2, "p90": 7, "top3_pct": 22, "bot3_pct": 38},
        },
    }


def test_format_monte_carlo_sorted_by_median():
    data = format_monte_carlo_for_display(_sample_monte_carlo(), "Hart of the Order")
    assert data["teams"][0]["name"] == "Hart of the Order"
    assert data["teams"][0]["median_pts"] == 68.5
    assert data["teams"][0]["is_user"] is True


def test_format_monte_carlo_category_risk_colors():
    data = format_monte_carlo_for_display(_sample_monte_carlo(), "Hart of the Order")
    risk = data["category_risk"]
    sv = next(r for r in risk if r["cat"] == "SV")
    assert sv["risk_class"] == "cat-bottom"
    r_cat = next(r for r in risk if r["cat"] == "R")
    assert r_cat["risk_class"] == "cat-top"


# --- format_lineup_for_display tests ---


def _sample_roster():
    return [
        {
            "name": "Adley Rutschman",
            "positions": ["C"],
            "selected_position": "C",
            "player_id": "123",
            "status": "",
        },
        {
            "name": "Mike Trout",
            "positions": ["OF"],
            "selected_position": "OF",
            "player_id": "456",
            "status": "IL",
        },
        {
            "name": "Masataka Yoshida",
            "positions": ["OF", "UTIL"],
            "selected_position": "BN",
            "player_id": "789",
            "status": "",
        },
    ]


def _sample_optimal():
    return {
        "hitters": {"C": "Adley Rutschman", "OF": "Masataka Yoshida"},
        "pitchers": {},
        "moves": [
            {"action": "START", "player": "Masataka Yoshida", "slot": "OF", "reason": "SGP: 1.9"},
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
    h[rest_of_season_key] — there is no longer a nested entry["rest_of_season"] dict.
    """
    roster = [
        {
            "name": "Aaron Judge",
            "positions": ["OF"],
            "selected_position": "OF",
            "player_id": "1",
            "status": "",
            "rest_of_season": {"r": 90, "hr": 40, "rbi": 100, "sb": 5, "avg": 0.280},
        },
        {
            "name": "No ROS Player",
            "positions": ["1B"],
            "selected_position": "1B",
            "player_id": "2",
            "status": "",
        },
    ]
    data = format_lineup_for_display(roster, {"moves": []})
    judge = next(h for h in data["hitters"] if h["name"] == "Aaron Judge")
    no_ros = next(h for h in data["hitters"] if h["name"] == "No ROS Player")
    # ROS stats are flattened to top level for template h[rest_of_season_key] access
    assert judge["hr"] == 40
    assert judge["r"] == 90
    assert judge["rbi"] == 100
    assert judge["sb"] == 5
    assert abs(judge["avg"] - 0.280) < 1e-9
    # Player with no ROS data has no flat stat keys
    assert "hr" not in no_ros
    assert "r" not in no_ros


def test_roster_cache_includes_stats():
    """After refresh, roster entries should include a 'pace' dict."""
    roster = [
        {
            "name": "Juan Soto",
            "positions": ["OF"],
            "selected_position": "OF",
            "player_id": "1",
            "status": "",
            "player_type": "hitter",
            "r": 90,
            "hr": 30,
            "rbi": 90,
            "sb": 10,
            "h": 150,
            "ab": 540,
            "pa": 600,
            "avg": 0.278,
            "pace": {
                "PA": {"actual": 102, "color_class": "stat-neutral"},
                "R": {
                    "actual": 19,
                    "expected": 15.3,
                    "z_score": 1.2,
                    "color_class": "stat-hot-2",
                    "projection": 90,
                },
                "HR": {
                    "actual": 9,
                    "expected": 5.1,
                    "z_score": 1.6,
                    "color_class": "stat-hot-2",
                    "projection": 30,
                },
                "RBI": {
                    "actual": 18,
                    "expected": 15.3,
                    "z_score": 0.3,
                    "color_class": "stat-neutral",
                    "projection": 90,
                },
                "SB": {
                    "actual": 2,
                    "expected": 1.7,
                    "z_score": 0.2,
                    "color_class": "stat-neutral",
                    "projection": 10,
                },
                "AVG": {
                    "actual": 0.298,
                    "expected": 0.278,
                    "z_score": 0.7,
                    "color_class": "stat-hot-1",
                    "projection": 0.278,
                },
            },
        },
    ]
    result = format_lineup_for_display(roster, {"moves": []})
    assert "pace" in result["hitters"][0]
    assert result["hitters"][0]["pace"]["HR"]["actual"] == 9


# After Phase 1 of the cache refactor, read_cache/write_cache route
# through ``kv_store.get_kv()``. The leak-prevention invariant
# (off-Render get_kv() never reaches Upstash) is enforced by kv_store
# itself and tested in ``tests/test_data/test_kv_store.py``. The tests
# below verify the cache-layer behavior on top of the KV: round-trip,
# canonical key naming, miss handling, corrupt-payload handling, and
# graceful KV-error tolerance.


def test_write_cache_uses_canonical_redis_key():
    """write_cache stores under ``cache:<key>`` so dashboard reads on
    Render (which hit the same key in Upstash) see the same data."""
    data = {"teams": [1, 2, 3]}
    write_cache(CacheKey.STANDINGS, data)
    raw = kv_store.get_kv().get(redis_key(CacheKey.STANDINGS))
    assert raw is not None
    assert json.loads(raw) == data


def test_write_cache_swallows_kv_error(monkeypatch):
    """write_cache logs and continues if the KV write raises.

    Mirrors the pre-refactor behavior of tolerating transient Upstash
    blips during a refresh — failing the whole pipeline because one
    cache write missed would be more disruptive than the staleness it
    causes. read_cache will subsequently return None for this key,
    which the dashboard already handles as a missing-cache state.
    """

    class _RaisingKV:
        def get(self, key):
            return None

        def set(self, key, value, **_):
            raise ConnectionError("Upstash unreachable")

    monkeypatch.setattr(season_data, "get_kv", lambda: _RaisingKV())
    # No exception escapes.
    write_cache(CacheKey.STANDINGS, {"v": 1})


def test_read_cache_swallows_kv_error(monkeypatch):
    """read_cache returns None on KV error rather than propagating."""

    class _RaisingKV:
        def get(self, key):
            raise ConnectionError("Upstash unreachable")

    monkeypatch.setattr(season_data, "get_kv", lambda: _RaisingKV())
    assert read_cache(CacheKey.STANDINGS) is None


class TestComputeComparisonStandings:
    def test_swap_changes_user_team_stats(self):
        """Swapping a hitter should change the user's projected stats but not other teams'."""
        from fantasy_baseball.models.player import HitterStats, PitcherStats, Player
        from fantasy_baseball.web.season_data import compute_comparison_standings

        projected_standings = [
            {
                "name": "My Team",
                "team_key": "",
                "rank": 0,
                "stats": {
                    "R": 700,
                    "HR": 200,
                    "RBI": 700,
                    "SB": 100,
                    "AVG": 0.260,
                    "W": 80,
                    "K": 1200,
                    "SV": 50,
                    "ERA": 3.50,
                    "WHIP": 1.20,
                },
            },
            {
                "name": "Other Team",
                "team_key": "",
                "rank": 0,
                "stats": {
                    "R": 680,
                    "HR": 190,
                    "RBI": 680,
                    "SB": 110,
                    "AVG": 0.255,
                    "W": 85,
                    "K": 1100,
                    "SV": 40,
                    "ERA": 3.80,
                    "WHIP": 1.25,
                },
            },
        ]

        roster = [
            Player(
                name="Willy Adames",
                player_type="hitter",
                rest_of_season=HitterStats(
                    pa=650, ab=567, h=133, r=80, hr=25, rbi=81, sb=11, avg=0.235
                ),
            ),
            Player(
                name="Other Hitter",
                player_type="hitter",
                rest_of_season=HitterStats(
                    pa=630, ab=550, h=150, r=90, hr=30, rbi=95, sb=5, avg=0.273
                ),
            ),
            Player(
                name="My Pitcher",
                player_type="pitcher",
                rest_of_season=PitcherStats(
                    ip=180, w=12, k=180, sv=0, er=60, bb=50, h_allowed=150, era=3.00, whip=1.11
                ),
            ),
        ]

        other_player = Player(
            name="Ezequiel Tovar",
            player_type="hitter",
            rest_of_season=HitterStats(pa=590, ab=513, h=135, r=73, hr=20, rbi=74, sb=8, avg=0.263),
        )

        result = compute_comparison_standings(
            roster_player_name="Willy Adames",
            other_player=other_player,
            user_roster=roster,
            projected_standings=_projected_from_raw(projected_standings),
            user_team_name="My Team",
        )

        assert "before" in result
        assert "after" in result
        assert "categories" in result

        # User's stats must change (counting stats differ between the two players)
        assert result["before"]["stats"]["My Team"] != result["after"]["stats"]["My Team"]
        # Other teams are untouched
        assert result["before"]["stats"]["Other Team"] == result["after"]["stats"]["Other Team"]
        # Counting stat delta matches player difference: Adames 25 HR, Tovar 20 HR
        before_hr = result["before"]["stats"]["My Team"]["HR"]
        after_hr = result["after"]["stats"]["My Team"]["HR"]
        assert before_hr - after_hr == pytest.approx(25 - 20)

    def test_swap_not_found_returns_error(self):
        """If roster_player_name doesn't match anyone in user_roster, return error."""
        from fantasy_baseball.models.player import HitterStats, Player
        from fantasy_baseball.web.season_data import compute_comparison_standings

        result = compute_comparison_standings(
            roster_player_name="Nobody",
            other_player=Player(
                name="X",
                player_type="hitter",
                rest_of_season=HitterStats(pa=0, ab=0, h=0, r=0, hr=0, rbi=0, sb=0),
            ),
            user_roster=[
                Player(
                    name="A",
                    player_type="hitter",
                    rest_of_season=HitterStats(pa=350, ab=300, h=80, r=50, hr=10, rbi=40, sb=5),
                )
            ],
            projected_standings=_projected_from_raw(
                [
                    {
                        "name": "My Team",
                        "team_key": "",
                        "rank": 0,
                        "stats": {
                            "R": 700,
                            "HR": 200,
                            "RBI": 700,
                            "SB": 100,
                            "AVG": 0.260,
                            "W": 80,
                            "K": 1200,
                            "SV": 50,
                            "ERA": 3.50,
                            "WHIP": 1.20,
                        },
                    }
                ]
            ),
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
        from fantasy_baseball.models.player import HitterStats, PitcherStats, Player

        return [
            Player(
                name="Star Hitter",
                player_type="hitter",
                rest_of_season=HitterStats(
                    pa=650, ab=567, h=150, r=90, hr=30, rbi=95, sb=25, avg=0.265
                ),
            ),
            Player(
                name="Role Hitter",
                player_type="hitter",
                rest_of_season=HitterStats(
                    pa=500, ab=440, h=110, r=55, hr=15, rbi=55, sb=5, avg=0.250
                ),
            ),
            Player(
                name="Ace",
                player_type="pitcher",
                rest_of_season=PitcherStats(
                    ip=180, w=14, k=200, sv=0, er=60, bb=50, h_allowed=150, era=3.00, whip=1.11
                ),
            ),
        ]

    def test_before_stats_equal_projected_standings_entry(self):
        """The comparison "before" for the user team must equal the value stored
        in projected_standings (otherwise UI shows one number on the standings
        page and a different number on the comparison page for the same team)."""
        from fantasy_baseball.models.player import HitterStats, Player
        from fantasy_baseball.scoring import project_team_stats
        from fantasy_baseball.web.season_data import compute_comparison_standings

        roster = self._build_roster()
        user_stats = project_team_stats(roster)
        projected_standings = [
            {"name": "My Team", "team_key": "", "rank": 0, "stats": user_stats.to_dict()},
            {
                "name": "Rival",
                "team_key": "",
                "rank": 0,
                "stats": {
                    "R": 680,
                    "HR": 190,
                    "RBI": 680,
                    "SB": 110,
                    "AVG": 0.255,
                    "W": 85,
                    "K": 1100,
                    "SV": 40,
                    "ERA": 3.80,
                    "WHIP": 1.25,
                },
            },
        ]
        other_player = Player(
            name="Replacement",
            player_type="hitter",
            rest_of_season=HitterStats(pa=600, ab=530, h=140, r=75, hr=28, rbi=85, sb=3, avg=0.264),
        )

        result = compute_comparison_standings(
            roster_player_name="Star Hitter",
            other_player=other_player,
            user_roster=roster,
            projected_standings=_projected_from_raw(projected_standings),
            user_team_name="My Team",
        )

        # Invariant: "before" uses projected_standings directly — the single
        # source of truth.  No recomputation, so no drift possible.
        for cat, val in user_stats.items():
            assert result["before"]["stats"]["My Team"][cat.value] == val, (
                f"before[{cat.value}] diverged from projected_standings "
                f"({result['before']['stats']['My Team'][cat.value]} vs {val})"
            )

    def test_swap_delta_equals_player_stat_difference(self):
        """The team-stat drop from swapping must equal the counting-stat
        difference between the two players. If it doesn't, the UI's stat
        comparison row will disagree with the team standings panel."""
        from fantasy_baseball.models.player import HitterStats, Player
        from fantasy_baseball.scoring import project_team_stats
        from fantasy_baseball.web.season_data import compute_comparison_standings

        roster = self._build_roster()
        user_stats = project_team_stats(roster)
        projected_standings = [
            {"name": "My Team", "team_key": "", "rank": 0, "stats": user_stats.to_dict()},
        ]

        dropped = roster[0]  # Star Hitter
        other_player = Player(
            name="Replacement",
            player_type="hitter",
            rest_of_season=HitterStats(pa=600, ab=530, h=140, r=75, hr=28, rbi=85, sb=3, avg=0.264),
        )

        result = compute_comparison_standings(
            roster_player_name=dropped.name,
            other_player=other_player,
            user_roster=roster,
            projected_standings=_projected_from_raw(projected_standings),
            user_team_name="My Team",
        )

        for cat_attr, roto_cat in [("r", "R"), ("hr", "HR"), ("rbi", "RBI"), ("sb", "SB")]:
            player_delta = getattr(dropped.rest_of_season, cat_attr) - getattr(
                other_player.rest_of_season, cat_attr
            )
            team_delta = (
                result["before"]["stats"]["My Team"][roto_cat]
                - result["after"]["stats"]["My Team"][roto_cat]
            )
            assert abs(team_delta - player_delta) < 1e-9, (
                f"{roto_cat}: team delta {team_delta} != player delta {player_delta}"
            )

    def test_projected_standings_is_single_source_of_truth(self):
        """compute_comparison_standings uses projected_standings directly —
        it never recomputes from user_roster.  This guarantees the
        "before" on the comparison page always matches the standings page,
        regardless of whether cache:roster and cache:projections diverge."""
        from fantasy_baseball.models.player import HitterStats, Player
        from fantasy_baseball.web.season_data import compute_comparison_standings

        roster = self._build_roster()

        # projected_standings with specific values — these are the source of truth.
        canonical_stats = {
            "R": 999,
            "HR": 888,
            "RBI": 777,
            "SB": 666,
            "AVG": 0.300,
            "W": 99,
            "K": 1500,
            "SV": 55,
            "ERA": 3.50,
            "WHIP": 1.10,
        }
        projected_standings = [
            {"name": "My Team", "team_key": "", "rank": 0, "stats": canonical_stats},
        ]

        other_player = Player(
            name="Replacement",
            player_type="hitter",
            rest_of_season=HitterStats(pa=600, ab=530, h=140, r=75, hr=28, rbi=85, sb=3, avg=0.264),
        )

        result = compute_comparison_standings(
            roster_player_name="Star Hitter",
            other_player=other_player,
            user_roster=roster,
            projected_standings=_projected_from_raw(projected_standings),
            user_team_name="My Team",
        )

        # "before" must use projected_standings, not recompute from roster
        assert result["before"]["stats"]["My Team"]["HR"] == canonical_stats["HR"]
        assert result["before"]["stats"]["My Team"]["R"] == canonical_stats["R"]


class TestComparisonProjectionOverride:
    """Verify roster_player_projection overrides roster cache stats.

    When ros_projections is updated after a refresh, the roster cache
    has stale player stats.  The compare endpoint should use the
    ros_projections version so the delta matches what the browse page
    shows.
    """

    def test_override_uses_projection_stats_for_delta(self):
        """roster_player_projection's stats drive the delta, not roster cache."""
        from fantasy_baseball.models.player import HitterStats, PitcherStats, Player
        from fantasy_baseball.web.season_data import compute_comparison_standings

        # Roster cache has stale pitcher stats (sv=1)
        stale_pitcher = Player(
            name="Closer X",
            player_type="pitcher",
            rest_of_season=PitcherStats(
                ip=60, w=3, k=55, sv=1, er=25, bb=15, h_allowed=55, era=3.75, whip=1.17
            ),
        )
        roster = [
            Player(
                name="Hitter A",
                player_type="hitter",
                rest_of_season=HitterStats(
                    pa=600, ab=530, h=140, r=80, hr=25, rbi=80, sb=10, avg=0.264
                ),
            ),
            stale_pitcher,
        ]

        # ros_projections has updated stats (sv=18)
        fresh_pitcher = Player(
            name="Closer X",
            player_type="pitcher",
            rest_of_season=PitcherStats(
                ip=60, w=3, k=55, sv=18, er=25, bb=15, h_allowed=55, era=3.75, whip=1.17
            ),
        )

        projected_standings = [
            {
                "name": "My Team",
                "team_key": "",
                "rank": 0,
                "stats": {
                    "R": 700,
                    "HR": 200,
                    "RBI": 700,
                    "SB": 100,
                    "AVG": 0.260,
                    "W": 80,
                    "K": 1200,
                    "SV": 50,
                    "ERA": 3.50,
                    "WHIP": 1.20,
                },
            },
        ]

        replacement = Player(
            name="Streamer",
            player_type="pitcher",
            rest_of_season=PitcherStats(
                ip=80, w=5, k=70, sv=0, er=35, bb=25, h_allowed=75, era=3.94, whip=1.25
            ),
        )

        # Without override: uses stale sv=1 → delta = 0 - 1 = -1
        result_stale = compute_comparison_standings(
            roster_player_name="Closer X",
            other_player=replacement,
            user_roster=roster,
            projected_standings=_projected_from_raw(projected_standings),
            user_team_name="My Team",
        )
        stale_sv_delta = (
            result_stale["after"]["stats"]["My Team"]["SV"]
            - result_stale["before"]["stats"]["My Team"]["SV"]
        )
        assert stale_sv_delta == pytest.approx(-1)

        # With override: uses fresh sv=18 → delta = 0 - 18 = -18
        result_fresh = compute_comparison_standings(
            roster_player_name="Closer X",
            other_player=replacement,
            user_roster=roster,
            projected_standings=_projected_from_raw(projected_standings),
            user_team_name="My Team",
            roster_player_projection=fresh_pitcher,
        )
        fresh_sv_delta = (
            result_fresh["after"]["stats"]["My Team"]["SV"]
            - result_fresh["before"]["stats"]["My Team"]["SV"]
        )
        assert fresh_sv_delta == pytest.approx(-18)


def _refresh_run_source() -> str:
    """Concatenate the source of every RefreshRun method.

    The pre-class regression guards inspected ``run_full_refresh``
    directly. After the RefreshRun refactor, the same logic lives in
    methods on the class — this helper rebuilds the equivalent source
    blob so the guards keep working.
    """
    import inspect

    from fantasy_baseball.web import refresh_pipeline

    cls = refresh_pipeline.RefreshRun
    return "\n".join(
        inspect.getsource(getattr(cls, name))
        for name in dir(cls)
        if callable(getattr(cls, name)) and not name.startswith("__")
    )


class TestRefreshRunAttributeAccess:
    """Catch typos like config.sgp_denominators (should be sgp_overrides).

    RefreshRun methods access attributes on LeagueConfig, Player, and other
    dataclasses. A typo compiles fine but crashes at runtime — often only
    discovered after a 15-minute deploy + refresh cycle on Render.

    These tests parse the source and verify every `config.<attr>` reference
    actually exists on LeagueConfig.
    """

    def test_config_attribute_access_valid(self):
        """Every `config.<attr>` in RefreshRun methods must exist on LeagueConfig."""
        import re

        from fantasy_baseball.config import LeagueConfig

        src = _refresh_run_source()
        valid_attrs = {f.name for f in LeagueConfig.__dataclass_fields__.values()}

        # Match config.something (not config = or config: or config[)
        refs = set(re.findall(r"\bconfig\.([a-zA-Z_]\w*)", src))

        bad = refs - valid_attrs
        assert not bad, (
            f"RefreshRun references config attributes that don't exist "
            f"on LeagueConfig: {bad}. Valid attributes: {sorted(valid_attrs)}"
        )


class TestRefreshRunScopingGuards:
    """Regression guards for Python scoping bugs in RefreshRun.

    RefreshRun methods import modules lazily inside conditional
    branches. Any local ``from X import Y`` that shadows a
    module-level name promotes that name to a local variable for
    the entire function scope (LEGB rule), causing UnboundLocalError
    when the name is referenced earlier in the function.

    Production bug landed on 2026-04-12: Step 2 of the League data
    model refactor started using ``date.fromisoformat(end_date)`` near
    the top of what is now RefreshRun to compute effective_date. A
    pre-existing local ``from datetime import date`` inside the
    Monte Carlo block shadowed the module-level import, so the
    function crashed with ``UnboundLocalError: cannot access local
    variable 'date' where it is not associated with a value``.
    """

    def test_no_local_datetime_import_inside_refresh_run(self):
        src = _refresh_run_source()
        assert "from datetime import date" not in src, (
            "Local `from datetime import date` found in a RefreshRun method. "
            "This shadows the module-level import on line 9 and causes "
            "UnboundLocalError when date is used earlier in the function. "
            "If you need to import date inside the function, rename it "
            "(e.g. `from datetime import date as _date`) to avoid shadowing."
        )
        assert "from datetime import datetime" not in src, (
            "Local `from datetime import datetime` in a RefreshRun method "
            "would shadow the module-level import for the same reason as "
            "the date case."
        )
        assert "from datetime import timedelta" not in src, (
            "Local `from datetime import timedelta` in a RefreshRun method "
            "would shadow the module-level import for the same reason as "
            "the date case."
        )

    def test_module_level_date_import_is_present(self):
        """Confirms the module-level import exists so the function can
        rely on ``date`` being available without a local import."""
        import inspect

        from fantasy_baseball.web import refresh_pipeline

        module_src = inspect.getsource(refresh_pipeline)
        # The module-level import must appear before the run_full_refresh
        # definition, not inside it.
        first_refresh_idx = module_src.find("def run_full_refresh")
        assert first_refresh_idx > 0
        header = module_src[:first_refresh_idx]
        assert "from datetime import date" in header


class TestComparisonEV:
    """compute_comparison_standings with EV-based scoring."""

    def test_team_sds_none_matches_rank_based(self):
        from fantasy_baseball.models.player import HitterStats, Player
        from fantasy_baseball.web.season_data import compute_comparison_standings

        projected_standings = [
            {
                "name": "User",
                "team_key": "",
                "rank": 0,
                "stats": {
                    "R": 0,
                    "HR": 0,
                    "RBI": 0,
                    "SB": 100,
                    "AVG": 0,
                    "W": 0,
                    "K": 0,
                    "SV": 0,
                    "ERA": 0,
                    "WHIP": 0,
                },
            },
            {
                "name": "Rival",
                "team_key": "",
                "rank": 0,
                "stats": {
                    "R": 0,
                    "HR": 0,
                    "RBI": 0,
                    "SB": 99,
                    "AVG": 0,
                    "W": 0,
                    "K": 0,
                    "SV": 0,
                    "ERA": 0,
                    "WHIP": 0,
                },
            },
        ]
        drop_hitter = Player(
            name="Drop",
            player_type="hitter",
            rest_of_season=HitterStats(pa=100, ab=100, h=0, r=0, hr=0, rbi=0, sb=20),
        )
        add_hitter = Player(
            name="Add",
            player_type="hitter",
            rest_of_season=HitterStats(pa=100, ab=100, h=0, r=0, hr=0, rbi=0, sb=10),
        )
        result = compute_comparison_standings(
            roster_player_name="Drop",
            other_player=add_hitter,
            user_roster=[drop_hitter],
            projected_standings=_projected_from_raw(projected_standings),
            user_team_name="User",
            team_sds=None,
        )
        assert result["delta_roto"]["categories"]["SB"]["roto_delta"] == pytest.approx(-1.0)

    def test_team_sds_produces_fractional_delta_under_uncertainty(self):
        from fantasy_baseball.models.player import HitterStats, Player
        from fantasy_baseball.web.season_data import compute_comparison_standings

        projected_standings = [
            {
                "name": "User",
                "team_key": "",
                "rank": 0,
                "stats": {
                    "R": 0,
                    "HR": 0,
                    "RBI": 0,
                    "SB": 100,
                    "AVG": 0,
                    "W": 0,
                    "K": 0,
                    "SV": 0,
                    "ERA": 0,
                    "WHIP": 0,
                },
            },
            {
                "name": "Rival",
                "team_key": "",
                "rank": 0,
                "stats": {
                    "R": 0,
                    "HR": 0,
                    "RBI": 0,
                    "SB": 99,
                    "AVG": 0,
                    "W": 0,
                    "K": 0,
                    "SV": 0,
                    "ERA": 0,
                    "WHIP": 0,
                },
            },
        ]
        team_sds = {
            "User": {
                Category.R: 0,
                Category.HR: 0,
                Category.RBI: 0,
                Category.SB: 10.0,
                Category.AVG: 0,
                Category.W: 0,
                Category.K: 0,
                Category.SV: 0,
                Category.ERA: 0,
                Category.WHIP: 0,
            },
            "Rival": {
                Category.R: 0,
                Category.HR: 0,
                Category.RBI: 0,
                Category.SB: 10.0,
                Category.AVG: 0,
                Category.W: 0,
                Category.K: 0,
                Category.SV: 0,
                Category.ERA: 0,
                Category.WHIP: 0,
            },
        }
        drop_hitter = Player(
            name="Drop",
            player_type="hitter",
            rest_of_season=HitterStats(pa=100, ab=100, h=0, r=0, hr=0, rbi=0, sb=20),
        )
        add_hitter = Player(
            name="Add",
            player_type="hitter",
            rest_of_season=HitterStats(pa=100, ab=100, h=0, r=0, hr=0, rbi=0, sb=10),
        )
        result = compute_comparison_standings(
            roster_player_name="Drop",
            other_player=add_hitter,
            user_roster=[drop_hitter],
            projected_standings=_projected_from_raw(projected_standings),
            user_team_name="User",
            team_sds=team_sds,
        )
        assert abs(result["delta_roto"]["categories"]["SB"]["roto_delta"]) < 0.5

    def test_ev_roto_key_present_in_response(self):
        from fantasy_baseball.models.player import HitterStats, Player
        from fantasy_baseball.web.season_data import compute_comparison_standings

        projected_standings = [
            {
                "name": "User",
                "team_key": "",
                "rank": 0,
                "stats": {
                    "R": 0,
                    "HR": 0,
                    "RBI": 0,
                    "SB": 100,
                    "AVG": 0,
                    "W": 0,
                    "K": 0,
                    "SV": 0,
                    "ERA": 0,
                    "WHIP": 0,
                },
            },
            {
                "name": "Rival",
                "team_key": "",
                "rank": 0,
                "stats": {
                    "R": 0,
                    "HR": 0,
                    "RBI": 0,
                    "SB": 99,
                    "AVG": 0,
                    "W": 0,
                    "K": 0,
                    "SV": 0,
                    "ERA": 0,
                    "WHIP": 0,
                },
            },
        ]
        team_sds = {
            "User": {
                Category.R: 0,
                Category.HR: 0,
                Category.RBI: 0,
                Category.SB: 10.0,
                Category.AVG: 0,
                Category.W: 0,
                Category.K: 0,
                Category.SV: 0,
                Category.ERA: 0,
                Category.WHIP: 0,
            },
            "Rival": {
                Category.R: 0,
                Category.HR: 0,
                Category.RBI: 0,
                Category.SB: 10.0,
                Category.AVG: 0,
                Category.W: 0,
                Category.K: 0,
                Category.SV: 0,
                Category.ERA: 0,
                Category.WHIP: 0,
            },
        }
        drop_hitter = Player(
            name="Drop",
            player_type="hitter",
            rest_of_season=HitterStats(pa=100, ab=100, h=0, r=0, hr=0, rbi=0, sb=20),
        )
        add_hitter = Player(
            name="Add",
            player_type="hitter",
            rest_of_season=HitterStats(pa=100, ab=100, h=0, r=0, hr=0, rbi=0, sb=10),
        )
        result = compute_comparison_standings(
            roster_player_name="Drop",
            other_player=add_hitter,
            user_roster=[drop_hitter],
            projected_standings=_projected_from_raw(projected_standings),
            user_team_name="User",
            team_sds=team_sds,
        )
        assert "ev_roto" in result["before"]
        assert "ev_roto" in result["after"]
        ev_sb = result["before"]["ev_roto"]["User"]["SB_pts"]
        rank_sb = result["before"]["roto"]["User"]["SB_pts"]
        assert ev_sb != pytest.approx(rank_sb, abs=0.01)


class TestComputeTeamTotalsPace:
    """Regression for _compute_team_totals_pace after the canonical
    Standings cache shape landed (commit 4d88479). The cache is now
    ``{"effective_date", "teams": [...]}`` — iterating it as list[dict]
    (old shape) blew up the /lineup route with AttributeError."""

    def _canonical_standings_json(self, team_name, *, ip=None, pa=None):
        from fantasy_baseball.models.standings import (
            CategoryStats,
            Standings,
            StandingsEntry,
        )
        from fantasy_baseball.utils.constants import OpportunityStat

        extras: dict[OpportunityStat, float] = {}
        if ip is not None:
            extras[OpportunityStat.IP] = ip
        if pa is not None:
            extras[OpportunityStat.PA] = pa
        s = Standings(
            effective_date=date(2026, 4, 21),
            entries=[
                StandingsEntry(
                    team_name=team_name,
                    team_key="469.l.5652.t.4",
                    rank=2,
                    stats=CategoryStats(
                        r=146,
                        hr=44,
                        rbi=140,
                        sb=26,
                        avg=0.262,
                        w=12,
                        k=200,
                        sv=11,
                        era=4.02,
                        whip=1.18,
                    ),
                    yahoo_points_for=73.5,
                    extras=extras,
                ),
            ],
        )
        return s.to_json()

    def _patch_read_cache(self, monkeypatch, payload):
        def fake_read_cache(key, *_args, **_kwargs):
            if key == CacheKey.STANDINGS:
                return payload
            return None

        monkeypatch.setattr(season_data, "read_cache", fake_read_cache)

    def test_reads_canonical_standings_cache_without_crashing(self, monkeypatch):
        """The /lineup route calls into this helper; the canonical cache
        shape must not raise AttributeError."""
        payload = self._canonical_standings_json("Hart of the Order", ip=198.2)
        self._patch_read_cache(monkeypatch, payload)

        totals = season_data._compute_team_totals_pace(
            players=[],
            player_type="pitcher",
            team_name="Hart of the Order",
        )

        # No crash; actuals come from typed standings.
        assert totals["IP"]["actual"] == pytest.approx(198.2)
        assert totals["W"]["actual"] == pytest.approx(12)
        assert totals["K"]["actual"] == pytest.approx(200)
        assert totals["ERA"]["actual"] == pytest.approx(4.02)

    def test_missing_cache_is_silent(self, monkeypatch):
        """Empty cache path (unit tests, pre-refresh): actuals default to 0."""
        self._patch_read_cache(monkeypatch, None)

        totals = season_data._compute_team_totals_pace(
            players=[],
            player_type="hitter",
            team_name="Hart of the Order",
        )

        assert totals["PA"]["actual"] == 0
        assert totals["R"]["actual"] == 0

    def test_opportunity_stat_absent_when_yahoo_omits_it(self, monkeypatch):
        """PA isn't a standard Yahoo stat in this league's standings
        response — extras is empty, actual falls back to 0 cleanly."""
        payload = self._canonical_standings_json("Hart of the Order")
        self._patch_read_cache(monkeypatch, payload)

        totals = season_data._compute_team_totals_pace(
            players=[],
            player_type="hitter",
            team_name="Hart of the Order",
        )

        assert totals["PA"]["actual"] == 0.0
        # Counting actuals still come through from CategoryStats.
        assert totals["R"]["actual"] == pytest.approx(146)


def test_build_trends_series_empty_history(fake_redis):
    """Empty history hashes → empty payload but valid shape."""
    from fantasy_baseball.web.season_data import build_trends_series

    out = build_trends_series(fake_redis, user_team="Alpha")
    assert out["user_team"] == "Alpha"
    assert out.get("categories")
    assert out["actual"] == {"dates": [], "teams": {}}
    assert out["projected"] == {"dates": [], "teams": {}}


def test_build_trends_series_actual_only(fake_redis):
    """Standings populated, projected empty — actual fills, projected stays empty."""
    import json

    from fantasy_baseball.data import redis_store

    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-15",
        json.dumps(
            {
                "effective_date": "2026-04-15",
                "teams": [
                    {
                        "name": "Alpha",
                        "team_key": "T.1",
                        "rank": 1,
                        "stats": {
                            "R": 45,
                            "HR": 12,
                            "RBI": 40,
                            "SB": 8,
                            "AVG": 0.268,
                            "W": 3,
                            "K": 85,
                            "SV": 4,
                            "ERA": 3.21,
                            "WHIP": 1.14,
                        },
                        "yahoo_points_for": 78.5,
                        "extras": {},
                    },
                    {
                        "name": "Beta",
                        "team_key": "T.2",
                        "rank": 2,
                        "stats": {
                            "R": 38,
                            "HR": 9,
                            "RBI": 32,
                            "SB": 6,
                            "AVG": 0.255,
                            "W": 2,
                            "K": 72,
                            "SV": 3,
                            "ERA": 3.85,
                            "WHIP": 1.22,
                        },
                        "yahoo_points_for": 60.0,
                        "extras": {},
                    },
                ],
            }
        ),
    )

    from fantasy_baseball.web.season_data import build_trends_series

    out = build_trends_series(fake_redis, user_team="Alpha")
    assert out["actual"]["dates"] == ["2026-04-15"]
    assert set(out["actual"]["teams"].keys()) == {"Alpha", "Beta"}
    alpha = out["actual"]["teams"]["Alpha"]
    assert alpha["roto_points"] == [78.5]  # Yahoo authority preferred
    assert alpha["stats"]["R"] == [45]
    assert alpha["stats"]["WHIP"] == [1.14]
    assert out["projected"] == {"dates": [], "teams": {}}


def test_build_trends_series_gap_handling(fake_redis):
    """Team appears on day 1 but not day 2 → null on day 2."""
    import json

    from fantasy_baseball.data import redis_store

    base_stats = {
        "R": 0,
        "HR": 0,
        "RBI": 0,
        "SB": 0,
        "AVG": 0.0,
        "W": 0,
        "K": 0,
        "SV": 0,
        "ERA": 99.0,
        "WHIP": 99.0,
    }
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-01",
        json.dumps(
            {
                "effective_date": "2026-04-01",
                "teams": [
                    {
                        "name": "Alpha",
                        "team_key": "T.1",
                        "rank": 1,
                        "stats": {**base_stats, "R": 10},
                        "yahoo_points_for": 5.0,
                        "extras": {},
                    },
                    {
                        "name": "Beta",
                        "team_key": "T.2",
                        "rank": 2,
                        "stats": {**base_stats, "R": 5},
                        "yahoo_points_for": 4.0,
                        "extras": {},
                    },
                ],
            }
        ),
    )
    fake_redis.hset(
        redis_store.STANDINGS_HISTORY_KEY,
        "2026-04-02",
        json.dumps(
            {
                "effective_date": "2026-04-02",
                "teams": [
                    {
                        "name": "Alpha",
                        "team_key": "T.1",
                        "rank": 1,
                        "stats": {**base_stats, "R": 20},
                        "yahoo_points_for": 6.0,
                        "extras": {},
                    },
                ],
            }
        ),
    )

    from fantasy_baseball.web.season_data import build_trends_series

    out = build_trends_series(fake_redis, user_team="Alpha")
    assert out["actual"]["dates"] == ["2026-04-01", "2026-04-02"]
    beta = out["actual"]["teams"]["Beta"]
    assert beta["roto_points"][0] == 4.0
    assert beta["roto_points"][1] is None
    assert beta["stats"]["R"][0] == 5
    assert beta["stats"]["R"][1] is None


def test_build_trends_series_projected_uses_score_roto(fake_redis):
    """Projected chart has no Yahoo authority — totals come from score_roto."""
    import json

    from fantasy_baseball.data import redis_store

    base_stats = {
        "R": 800,
        "HR": 200,
        "RBI": 750,
        "SB": 80,
        "AVG": 0.260,
        "W": 70,
        "K": 1400,
        "SV": 50,
        "ERA": 3.80,
        "WHIP": 1.25,
    }
    fake_redis.hset(
        redis_store.PROJECTED_STANDINGS_HISTORY_KEY,
        "2026-04-15",
        json.dumps(
            {
                "effective_date": "2026-04-15",
                "teams": [
                    {"name": "Alpha", "stats": {**base_stats, "R": 880}},
                    {"name": "Beta", "stats": {**base_stats, "R": 820}},
                ],
            }
        ),
    )

    from fantasy_baseball.web.season_data import build_trends_series

    out = build_trends_series(fake_redis, user_team="Alpha")
    assert out["projected"]["dates"] == ["2026-04-15"]
    # With 2 teams over 10 categories, totals should be ordered Alpha > Beta (Alpha has higher R).
    alpha_total = out["projected"]["teams"]["Alpha"]["roto_points"][0]
    beta_total = out["projected"]["teams"]["Beta"]["roto_points"][0]
    assert alpha_total > beta_total
    assert out["projected"]["teams"]["Alpha"]["stats"]["R"] == [880]
