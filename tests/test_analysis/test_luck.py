"""Tests for analysis/luck.py: luck with each team's decisions held fixed."""

from __future__ import annotations

from datetime import date

import pytest

from fantasy_baseball.analysis.luck import (
    IdResolver,
    Projection,
    Snapshot,
    compute_luck,
    projection_on,
    stint_luck,
)
from fantasy_baseball.data import redis_store
from fantasy_baseball.models.league import League
from fantasy_baseball.models.standings import CategoryStats, Standings, StandingsEntry
from fantasy_baseball.sgp.denominators import get_sgp_denominators

START, END = date(2026, 3, 27), date(2026, 9, 28)
HITTER = 1001  # MLBAM ids in this file's own made-up league
OTHER = 1002


@pytest.fixture
def kv(fake_redis, monkeypatch):
    from fantasy_baseball.data import kv_store

    monkeypatch.setattr(kv_store, "get_kv", lambda: fake_redis)
    return fake_redis


def _roster(client, day: str, team: str, players: list[tuple[str, str, str]]) -> None:
    """``players``: (name, slot, positions)."""
    redis_store.write_roster_snapshot(
        client,
        day,
        team,
        [
            {"slot": s, "player_name": n, "positions": pos, "status": "", "yahoo_id": ""}
            for n, s, pos in players
        ],
    )


def _hitter(r_per_pa: float, pa_per_day: float = 4.0) -> Projection:
    rates = {"r": r_per_pa, "hr": 0.04, "rbi": 0.12, "sb": 0.01, "h": 0.250}
    return Projection(rates=rates, per_day=pa_per_day, ab_pa=0.9)


def _game(day: str, pa: int = 4, r: int = 1) -> dict:
    return {"date": day, "pa": pa, "ab": pa - 1, "h": 1, "r": r, "hr": 0, "rbi": 0, "sb": 0}


def _resolver() -> IdResolver:
    return IdResolver({"hitter": {"joe bat": {HITTER}, "sam stick": {OTHER}}}, {}, {})


def test_performance_counts_active_games_only_at_the_playing_time_he_got(kv) -> None:
    _roster(kv, "2026-04-07", "A", [("Joe Bat", "OF", "OF")])
    _roster(kv, "2026-04-14", "A", [("Joe Bat", "BN", "OF")])
    snaps = [Snapshot(START, {HITTER: _hitter(0.10)}, {})]
    games = {"hitter": {HITTER: [_game("2026-04-08", pa=5, r=2), _game("2026-04-15", pa=4, r=3)]}}
    stints, unresolved = stint_luck(
        League.from_redis(2026), snaps, games, _resolver(), START, END, date(2026, 4, 21)
    )
    assert unresolved == [] and len(stints) == 1
    st = stints[0]
    # Performance: only the active week's game, expected at the PA he got (5 x 0.10).
    assert st.actual["r"] == 2 and st.expected["r"] == pytest.approx(0.5)
    # Playing time: every rostered day (both weeks, bench included) vs 4 PA a day.
    assert st.pt_got == 9 and st.pt_proj == pytest.approx(4.0 * 14)


def test_the_projection_is_the_one_on_the_day_he_was_acquired_held_all_stint(kv) -> None:
    _roster(kv, "2026-04-07", "A", [("Joe Bat", "OF", "OF")])
    _roster(kv, "2026-06-09", "A", [("Joe Bat", "OF", "OF")])
    _roster(kv, "2026-06-09", "B", [("Sam Stick", "OF", "OF")])
    snaps = [
        Snapshot(START, {HITTER: _hitter(0.10), OTHER: _hitter(0.10)}, {}),
        Snapshot(date(2026, 6, 1), {HITTER: _hitter(0.30), OTHER: _hitter(0.30)}, {}),
    ]
    games = {"hitter": {HITTER: [_game("2026-06-10")], OTHER: [_game("2026-06-10")]}}
    stints, _ = stint_luck(
        League.from_redis(2026), snaps, games, _resolver(), START, END, date(2026, 6, 16)
    )
    by = {st.team: st for st in stints}
    # A got him in April: the June game is still judged by April's projection.
    assert by["A"].expected["r"] == pytest.approx(4 * 0.10)
    # B got Sam in June: June's projection.
    assert by["B"].expected["r"] == pytest.approx(4 * 0.30)
    assert by["A"].proj_as_of == START and by["B"].proj_as_of == date(2026, 6, 1)


def test_dropped_and_re_added_is_a_new_stint_on_the_new_projection(kv) -> None:
    _roster(kv, "2026-04-07", "A", [("Joe Bat", "OF", "OF")])
    _roster(kv, "2026-04-14", "A", [("Sam Stick", "OF", "OF")])  # Joe dropped
    _roster(kv, "2026-06-09", "A", [("Joe Bat", "OF", "OF")])
    snaps = [
        Snapshot(START, {HITTER: _hitter(0.10)}, {}),
        Snapshot(date(2026, 6, 1), {HITTER: _hitter(0.30)}, {}),
    ]
    stints, _ = stint_luck(
        League.from_redis(2026),
        snaps,
        {"hitter": {HITTER: [], OTHER: []}},
        _resolver(),
        START,
        END,
        date(2026, 6, 16),
    )
    assert [(st.acquired, st.proj_as_of) for st in stints if st.mid == HITTER] == [
        (date(2026, 4, 7), START),
        (date(2026, 6, 9), date(2026, 6, 1)),
    ]


def test_a_projection_weeks_older_than_the_pickup_is_marked_stale(kv) -> None:
    _roster(kv, "2026-05-05", "A", [("Joe Bat", "OF", "OF")])
    _roster(kv, "2026-05-05", "B", [("Sam Stick", "OF", "OF")])
    snaps = [
        Snapshot(START, {HITTER: _hitter(0.1)}, {}),
        Snapshot(date(2026, 4, 28), {OTHER: _hitter(0.1)}, {}),
    ]
    assert projection_on(snaps, HITTER, "hitter", date(2026, 5, 5)) == (_hitter(0.1), START)
    stints, _ = stint_luck(
        League.from_redis(2026), snaps, {"hitter": {}}, _resolver(), START, END, date(2026, 5, 12)
    )
    by = {st.team: st for st in stints}
    assert by["A"].stale  # picked up May 5 on the March 27 projection
    assert not by["B"].stale  # a week-old projection is current enough


def test_team_luck_points_re_score_the_whole_league_and_sum_to_zero(kv) -> None:
    _roster(kv, "2026-04-07", "A", [("Joe Bat", "OF", "OF")])
    _roster(kv, "2026-04-07", "B", [("Sam Stick", "OF", "OF")])
    # Projected to hit exactly what the games below show, except runs (0.10 a PA).
    runs_only = Projection(
        rates={"r": 0.10, "hr": 0.0, "rbi": 0.0, "sb": 0.0, "h": 1 / 3}, per_day=4.0, ab_pa=0.75
    )
    snaps = [Snapshot(START, {HITTER: runs_only, OTHER: runs_only}, {})]
    # Joe scores 10 runs in 40 PA (4 expected): A beat projection by 6 R; B by none.
    games = {
        "hitter": {
            HITTER: [_game(f"2026-04-{d:02d}", pa=4, r=1) for d in range(8, 18)],
            OTHER: [_game(f"2026-04-{d:02d}", pa=4, r=0) for d in range(8, 18)],
        }
    }

    def entry(name: str, r: float) -> StandingsEntry:
        stats = {
            "R": r,
            "HR": 10,
            "RBI": 30,
            "SB": 5,
            "AVG": 0.25,
            "W": 3,
            "K": 50,
            "SV": 2,
            "ERA": 4.0,
            "WHIP": 1.3,
        }
        return StandingsEntry(
            team_name=name, team_key="", rank=0, stats=CategoryStats.from_dict(stats)
        )

    # Real standings: A leads R by 3 -- but without his luck (6 R) A would trail.
    standings = Standings(
        effective_date=date(2026, 4, 21), entries=[entry("A", 33), entry("B", 30)]
    )
    out = compute_luck(
        League.from_redis(2026),
        standings,
        snaps,
        games,
        _resolver(),
        get_sgp_denominators(),
        START,
        END,
        date(2026, 4, 21),
    )
    teams = {t["team"]: t for t in out["teams"]}
    assert teams["A"]["categories"]["R"] == 1.0 and teams["B"]["categories"]["R"] == -1.0
    assert sum(t["luck_points"] for t in out["teams"]) == 0
    joe = next(p for p in out["players"] if p["name"] == "Joe Bat")
    assert joe["actual"]["R"] == 10 and joe["expected"]["R"] == pytest.approx(4.0)
    assert joe["perf_sgp"] > 0


def test_a_shared_name_is_settled_by_mlb_team_and_never_guessed() -> None:
    resolve = IdResolver(
        {"hitter": {"max muncy": {571970, 691777}}},
        {("Spacemen", "max muncy"): "LAD"},
        {("max muncy", "LAD"): 571970, ("max muncy", "ATH"): 691777},
    )
    assert resolve("Spacemen", "max muncy", "hitter") == 571970
    assert resolve("Other", "max muncy", "hitter") is None  # no MLB team to settle it
    # No game log at all (out all season): a unique projection name.
    out_all_year = IdResolver({"hitter": {}}, {}, {("ryan pepiot", "TBR"): 686752})
    assert out_all_year("X", "ryan pepiot", "pitcher") == 686752
