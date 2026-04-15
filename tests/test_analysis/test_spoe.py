"""Tests for current season-to-date SPoE computation."""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from fantasy_baseball.analysis.spoe import (
    build_preseason_lookup,
    compute_current_spoe,
)
from fantasy_baseball.data import redis_store
from fantasy_baseball.models.league import League


# --- Fixtures --------------------------------------------------------------

SEASON_START = "2026-03-27"
SEASON_END = "2026-09-28"
TOTAL_DAYS = (date.fromisoformat(SEASON_END) - date.fromisoformat(SEASON_START)).days


@pytest.fixture
def redis_league(fake_redis, monkeypatch):
    """Redirect ``redis_store.get_default_client()`` at the patched fake
    client so ``League.from_redis`` reads test data.
    """
    monkeypatch.setattr(redis_store, "_default_client", fake_redis)
    monkeypatch.setattr(redis_store, "_default_client_initialized", True)
    yield fake_redis


def _league_from(client, season_year: int = 2026):
    """Build a League object from the Redis test client. Mirrors how
    run_full_refresh loads the League in production (after writing
    rosters via write_roster_snapshot)."""
    return League.from_redis(season_year)


def _snapshot_roster(client, snapshot_date: str, team: str, players: list[dict]):
    """Write a single team's roster snapshot into weekly_rosters_history.

    ``players`` is a list of ``{"name": str, "selected_position": str,
    "positions": list[str]}`` dicts (matching what the refresh pipeline
    receives). We map this to the ``{slot, player_name, positions, ...}``
    shape that ``write_roster_snapshot`` serializes.
    """
    entries = [
        {
            "slot": p["selected_position"],
            "player_name": p["name"],
            "positions": ", ".join(p.get("positions", [])),
            "status": p.get("status") or "",
            "yahoo_id": p.get("player_id") or "",
        }
        for p in players
    ]
    redis_store.write_roster_snapshot(client, snapshot_date, team, entries)


def _hitter_preseason(name: str, **overrides) -> dict:
    base = {
        "name": name, "player_type": "hitter",
        "pa": 650, "ab": 580, "h": 150, "r": 85, "hr": 25,
        "rbi": 80, "sb": 10, "avg": 0.259,
    }
    base.update(overrides)
    return base


def _pitcher_preseason(name: str, **overrides) -> dict:
    base = {
        "name": name, "player_type": "pitcher",
        "ip": 180, "w": 12, "k": 180, "sv": 0,
        "er": 70, "bb": 50, "h_allowed": 150,
        "era": 3.50, "whip": 1.11,
    }
    base.update(overrides)
    return base


def _preseason_lookup_from(*players) -> dict:
    hitters = pd.DataFrame([p for p in players if p.get("player_type") == "hitter"])
    pitchers = pd.DataFrame([p for p in players if p.get("player_type") == "pitcher"])
    return build_preseason_lookup(hitters, pitchers)


# --- build_preseason_lookup tests ------------------------------------------

class TestBuildPreseasonLookup:
    def test_normalized_name_is_key(self):
        hitters = pd.DataFrame([_hitter_preseason("Juan Soto")])
        lookup = build_preseason_lookup(hitters, pd.DataFrame())
        assert "juan soto" in lookup

    def test_accent_stripped(self):
        hitters = pd.DataFrame([_hitter_preseason("Julio Rodríguez", hr=30)])
        lookup = build_preseason_lookup(hitters, pd.DataFrame())
        assert "julio rodriguez" in lookup
        assert lookup["julio rodriguez"]["hr"] == 30

    def test_player_type_populated(self):
        hitters = pd.DataFrame([_hitter_preseason("Hitter A")])
        pitchers = pd.DataFrame([_pitcher_preseason("Pitcher B")])
        lookup = build_preseason_lookup(hitters, pitchers)
        assert lookup["hitter a"]["player_type"] == "hitter"
        assert lookup["pitcher b"]["player_type"] == "pitcher"

    def test_empty_dataframes_return_empty_lookup(self):
        lookup = build_preseason_lookup(pd.DataFrame(), pd.DataFrame())
        assert lookup == {}


# --- compute_current_spoe tests --------------------------------------------

class TestComputeCurrentSpoe:
    def test_returns_empty_results_before_season_start(self, redis_league):
        conn = redis_league
        lookup = _preseason_lookup_from(_hitter_preseason("Player A"))

        # today BEFORE season_start → days_elapsed = 0 but season_fraction=0
        result = compute_current_spoe(
            _league_from(conn), standings=[], preseason_lookup=lookup,
            season_start=SEASON_START, season_end=SEASON_END,
            today=date.fromisoformat("2026-03-01"),
        )
        assert result["season_fraction"] == 0.0
        assert result["results"] == []

    def test_one_team_one_week_ownership_credits_proportional(self, redis_league):
        conn = redis_league
        lookup = _preseason_lookup_from(
            _hitter_preseason("Star", hr=30, r=90, rbi=90, sb=10,
                              h=150, ab=580, pa=650),
        )

        # Snapshot team A owning Star on the second Tuesday of the season
        _snapshot_roster(conn, "2026-03-31", "Team A", [
            {"name": "Star", "selected_position": "OF", "positions": ["OF"]},
        ])
        _snapshot_roster(conn, "2026-03-31", "Team B", [])

        standings = [
            {"name": "Team A", "stats": {
                "R": 5, "HR": 2, "RBI": 4, "SB": 0, "AVG": 0.250,
                "W": 0, "K": 0, "SV": 0, "ERA": 99.0, "WHIP": 99.0,
            }},
            {"name": "Team B", "stats": {
                "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0.0,
                "W": 0, "K": 0, "SV": 0, "ERA": 99.0, "WHIP": 99.0,
            }},
        ]

        # today = 2026-04-07 → snapshot covers 7 days (full week)
        result = compute_current_spoe(
            _league_from(conn), standings=standings, preseason_lookup=lookup,
            season_start=SEASON_START, season_end=SEASON_END,
            today=date.fromisoformat("2026-04-07"),
        )

        # Expected HR contribution for Team A:
        # preseason HR (30) * (7 days / TOTAL_DAYS)
        expected_hr = 30 * 7 / TOTAL_DAYS
        hr_row = next(
            r for r in result["results"]
            if r["team"] == "Team A" and r["category"] == "HR"
        )
        assert hr_row["projected_stat"] == pytest.approx(expected_hr, rel=1e-9)

        # Team B has no roster so projected_stat == 0
        hr_row_b = next(
            r for r in result["results"]
            if r["team"] == "Team B" and r["category"] == "HR"
        )
        assert hr_row_b["projected_stat"] == 0.0

    def test_player_owned_for_two_weeks_gets_double_credit(self, redis_league):
        conn = redis_league
        lookup = _preseason_lookup_from(
            _hitter_preseason("Star", sb=20),
        )

        _snapshot_roster(conn, "2026-03-31", "Team A", [
            {"name": "Star", "selected_position": "OF", "positions": ["OF"]},
        ])
        _snapshot_roster(conn, "2026-04-07", "Team A", [
            {"name": "Star", "selected_position": "OF", "positions": ["OF"]},
        ])

        standings = [
            {"name": "Team A", "stats": {
                "R": 0, "HR": 0, "RBI": 0, "SB": 3, "AVG": 0.250,
                "W": 0, "K": 0, "SV": 0, "ERA": 99.0, "WHIP": 99.0,
            }},
            {"name": "Team B", "stats": {
                "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0.0,
                "W": 0, "K": 0, "SV": 0, "ERA": 99.0, "WHIP": 99.0,
            }},
        ]

        # today = 2026-04-14 → first snapshot covers 7 days, second covers 7 days
        result = compute_current_spoe(
            _league_from(conn), standings=standings, preseason_lookup=lookup,
            season_start=SEASON_START, season_end=SEASON_END,
            today=date.fromisoformat("2026-04-14"),
        )

        expected_sb = 20 * (14 / TOTAL_DAYS)
        sb_row = next(
            r for r in result["results"]
            if r["team"] == "Team A" and r["category"] == "SB"
        )
        assert sb_row["projected_stat"] == pytest.approx(expected_sb, rel=1e-9)

    def test_mid_season_ownership_change_splits_credit(self, redis_league):
        """Player on Team A for week 1, Team B for week 2. Each gets 1 week of credit.

        With the per-team ownership_periods architecture, we need explicit
        snapshots for each team to mark ownership boundaries. Team A must have
        a snapshot on 2026-04-07 showing a different roster (without Traded)
        to end the ownership period.
        """
        conn = redis_league
        lookup = _preseason_lookup_from(
            _hitter_preseason("Traded", hr=40),
        )

        _snapshot_roster(conn, "2026-03-31", "Team A", [
            {"name": "Traded", "selected_position": "OF", "positions": ["OF"]},
        ])
        _snapshot_roster(conn, "2026-04-07", "Team A", [
            {"name": "UnknownPlayer", "selected_position": "OF", "positions": ["OF"]},
        ])
        _snapshot_roster(conn, "2026-04-07", "Team B", [
            {"name": "Traded", "selected_position": "OF", "positions": ["OF"]},
        ])

        standings = [
            {"name": "Team A", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 99, "WHIP": 99}},
            {"name": "Team B", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 99, "WHIP": 99}},
        ]

        result = compute_current_spoe(
            _league_from(conn), standings=standings, preseason_lookup=lookup,
            season_start=SEASON_START, season_end=SEASON_END,
            today=date.fromisoformat("2026-04-14"),
        )

        # Team A owns Traded from 2026-03-31 to 2026-04-07 (7 days)
        hr_a = next(r for r in result["results"] if r["team"] == "Team A" and r["category"] == "HR")
        assert hr_a["projected_stat"] == pytest.approx(40 * 7 / TOTAL_DAYS, rel=1e-9)

        # Team B owns Traded from 2026-04-07 to 2026-04-14 (7 days)
        hr_b = next(r for r in result["results"] if r["team"] == "Team B" and r["category"] == "HR")
        assert hr_b["projected_stat"] == pytest.approx(40 * 7 / TOTAL_DAYS, rel=1e-9)

    def test_partial_current_week(self, redis_league):
        """Current snapshot's period extends only to today, not a full 7 days."""
        conn = redis_league
        lookup = _preseason_lookup_from(
            _hitter_preseason("Star", r=150),
        )
        _snapshot_roster(conn, "2026-04-07", "Team A", [
            {"name": "Star", "selected_position": "OF", "positions": ["OF"]},
        ])
        standings = [
            {"name": "Team A", "stats": {"R": 5, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 99, "WHIP": 99}},
            {"name": "Team B", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 99, "WHIP": 99}},
        ]

        # today = 2026-04-10 → 3 days since snapshot, not 7
        result = compute_current_spoe(
            _league_from(conn), standings=standings, preseason_lookup=lookup,
            season_start=SEASON_START, season_end=SEASON_END,
            today=date.fromisoformat("2026-04-10"),
        )

        expected_r = 150 * 3 / TOTAL_DAYS
        r_row = next(r for r in result["results"] if r["team"] == "Team A" and r["category"] == "R")
        assert r_row["projected_stat"] == pytest.approx(expected_r, rel=1e-9)

    def test_missing_preseason_skips_player_silently(self, redis_league):
        conn = redis_league
        lookup = _preseason_lookup_from()  # empty

        _snapshot_roster(conn, "2026-03-31", "Team A", [
            {"name": "Unknown Player", "selected_position": "OF", "positions": ["OF"]},
        ])
        _snapshot_roster(conn, "2026-03-31", "Team B", [])

        standings = [
            {"name": "Team A", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 99, "WHIP": 99}},
            {"name": "Team B", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 99, "WHIP": 99}},
        ]

        result = compute_current_spoe(
            _league_from(conn), standings=standings, preseason_lookup=lookup,
            season_start=SEASON_START, season_end=SEASON_END,
            today=date.fromisoformat("2026-04-14"),
        )

        # Team A contributed nothing despite owning the player — he wasn't in preseason lookup
        hr_a = next(r for r in result["results"] if r["team"] == "Team A" and r["category"] == "HR")
        assert hr_a["projected_stat"] == 0.0

    def test_preseason_snapshot_does_not_credit_days_before_season_start(self, redis_league):
        """Regression: snapshots dated before season_start should not
        contribute days from before the season began.

        If a pre-season snapshot dated 2026-03-24 exists (3 days before
        season_start 2026-03-27), the walk from that snapshot to the next
        one should only credit days from 2026-03-27 onward — not 2026-03-24.
        """
        conn = redis_league
        lookup = _preseason_lookup_from(
            _hitter_preseason("Star", hr=185),  # 1 HR per day full season
        )

        # Pre-season snapshot 3 days before season_start
        _snapshot_roster(conn, "2026-03-24", "Team A", [
            {"name": "Star", "selected_position": "OF", "positions": ["OF"]},
        ])
        # Regular-season snapshot (next Tuesday)
        _snapshot_roster(conn, "2026-03-31", "Team A", [
            {"name": "Star", "selected_position": "OF", "positions": ["OF"]},
        ])
        _snapshot_roster(conn, "2026-03-31", "Team B", [])

        standings = [
            {"name": "Team A", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 99, "WHIP": 99}},
            {"name": "Team B", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 99, "WHIP": 99}},
        ]

        result = compute_current_spoe(
            _league_from(conn), standings=standings, preseason_lookup=lookup,
            season_start=SEASON_START, season_end=SEASON_END,
            today=date.fromisoformat("2026-04-07"),
        )

        # The 2026-03-24 snapshot covers the span 2026-03-24 → 2026-03-31.
        # Season starts 2026-03-27, so only 4 of those 7 days should count
        # (2026-03-27, 28, 29, 30).
        # Then 2026-03-31 → 2026-04-07 contributes 7 more days.
        # Total: 11 days of ownership, NOT 14.
        expected_hr = 185 * 11 / TOTAL_DAYS
        hr_a = next(r for r in result["results"] if r["team"] == "Team A" and r["category"] == "HR")
        assert hr_a["projected_stat"] == pytest.approx(expected_hr, rel=1e-9)

        # Sanity check: if the bug were still present, projected_stat would be
        # 185 * 14 / TOTAL_DAYS instead, which differs meaningfully.
        buggy_value = 185 * 14 / TOTAL_DAYS
        assert abs(hr_a["projected_stat"] - buggy_value) > 1e-6, (
            "Pre-season clipping bug reintroduced — projection matches "
            "the 14-day (un-clipped) calculation instead of 11 days."
        )

    def test_result_shape_matches_luck_template(self, redis_league):
        """Backward-compatibility guard: the returned dict must have the
        keys the luck.html template expects."""
        conn = redis_league
        lookup = _preseason_lookup_from(_hitter_preseason("Star"))
        _snapshot_roster(conn, "2026-03-31", "Team A", [
            {"name": "Star", "selected_position": "OF", "positions": ["OF"]},
        ])
        _snapshot_roster(conn, "2026-03-31", "Team B", [])
        standings = [
            {"name": "Team A", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 99, "WHIP": 99}},
            {"name": "Team B", "stats": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                                         "W": 0, "K": 0, "SV": 0, "ERA": 99, "WHIP": 99}},
        ]
        result = compute_current_spoe(
            _league_from(conn), standings=standings, preseason_lookup=lookup,
            season_start=SEASON_START, season_end=SEASON_END,
            today=date.fromisoformat("2026-04-07"),
        )

        assert "snapshot_date" in result
        assert "results" in result
        assert any(r["category"] == "total" for r in result["results"])

        # Each category row must have these keys for luck.html
        for row in result["results"]:
            assert "team" in row
            assert "category" in row
            assert "projected_pts" in row
            assert "actual_pts" in row
            assert "spoe" in row
