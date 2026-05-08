"""Tests for streaks-specific game log parsing and per-season fetch."""

from datetime import date
from unittest.mock import Mock, patch

from fantasy_baseball.streaks.data.game_logs import (
    fetch_hitter_season_game_logs,
    pa_identity_gap,
    parse_hitter_game_log_full,
)
from fantasy_baseball.streaks.models import HitterGame


def _split(date="2024-04-01", game_pk=745444, **stat_overrides):
    stat = {
        "plateAppearances": 4,
        "atBats": 3,
        "hits": 1,
        "homeRuns": 1,
        "runs": 1,
        "rbi": 2,
        "stolenBases": 0,
        "baseOnBalls": 1,
        "strikeOuts": 1,
    }
    stat.update(stat_overrides)
    return {"date": date, "stat": stat, "game": {"gamePk": game_pk}}


def test_parse_hitter_game_log_full_extracts_all_columns():
    row = parse_hitter_game_log_full(
        _split(),
        player_id=660271,
        name="Mike Trout",
        team="LAA",
        season=2024,
    )
    # _split() omits b2/b3/sf/hbp/ibb/cs/gidp/sh/ci and isHome, so the parser
    # defaults those to 0 / True per the missing-field contract.
    assert row == HitterGame(
        player_id=660271,
        game_pk=745444,
        name="Mike Trout",
        team="LAA",
        season=2024,
        date=date(2024, 4, 1),
        pa=4,
        ab=3,
        h=1,
        hr=1,
        r=1,
        rbi=2,
        sb=0,
        bb=1,
        k=1,
        b2=0,
        b3=0,
        sf=0,
        hbp=0,
        ibb=0,
        cs=0,
        gidp=0,
        sh=0,
        ci=0,
        is_home=True,
    )


def test_parse_hitter_game_log_full_defaults_missing_stats_to_zero():
    row = parse_hitter_game_log_full(
        {"date": "2024-04-01", "stat": {}, "game": {"gamePk": 1}},
        player_id=1,
        name="X",
        team=None,
        season=2024,
    )
    assert row.pa == 0
    assert row.bb == 0


def test_parse_hitter_game_log_full_disambiguates_doubleheaders():
    """Two splits sharing a date but with distinct game_pks must produce
    distinct HitterGame instances — that's the whole point of the PK fix."""
    g1 = parse_hitter_game_log_full(
        _split(date="2024-07-04", game_pk=746001, homeRuns=1),
        player_id=660271,
        name="Mike Trout",
        team="LAA",
        season=2024,
    )
    g2 = parse_hitter_game_log_full(
        _split(date="2024-07-04", game_pk=746002, homeRuns=2),
        player_id=660271,
        name="Mike Trout",
        team="LAA",
        season=2024,
    )
    assert g1.date == g2.date
    assert g1.game_pk != g2.game_pk
    assert g1.hr == 1 and g2.hr == 2


def test_fetch_hitter_season_game_logs_returns_one_row_per_split():
    fake_resp = Mock()
    fake_resp.raise_for_status = Mock()
    fake_resp.json = Mock(
        return_value={
            "stats": [
                {
                    "splits": [
                        _split(date="2024-04-01", game_pk=745444),
                        _split(date="2024-04-02", game_pk=745445, homeRuns=0),
                    ]
                }
            ]
        }
    )
    with patch("fantasy_baseball.streaks.data.game_logs.requests.get", return_value=fake_resp):
        rows = fetch_hitter_season_game_logs(
            player_id=660271, name="Mike Trout", team="LAA", season=2024
        )
    assert len(rows) == 2
    assert rows[0].date == date(2024, 4, 1)
    assert rows[0].hr == 1
    assert rows[1].hr == 0
    assert all(r.player_id == 660271 for r in rows)
    assert all(r.season == 2024 for r in rows)


def test_fetch_hitter_season_game_logs_handles_empty_splits():
    fake_resp = Mock()
    fake_resp.raise_for_status = Mock()
    fake_resp.json = Mock(return_value={"stats": [{"splits": []}]})
    with patch("fantasy_baseball.streaks.data.game_logs.requests.get", return_value=fake_resp):
        rows = fetch_hitter_season_game_logs(player_id=1, name="X", team=None, season=2024)
    assert rows == []


def _make_split(
    stat: dict, *, is_home: bool = True, game_pk: int = 1, date: str = "2025-04-01"
) -> dict:
    return {
        "game": {"gamePk": game_pk},
        "date": date,
        "isHome": is_home,
        "stat": stat,
    }


def test_parse_captures_new_fields() -> None:
    split = _make_split(
        {
            "plateAppearances": 5,
            "atBats": 4,
            "hits": 2,
            "homeRuns": 1,
            "runs": 1,
            "rbi": 2,
            "stolenBases": 0,
            "baseOnBalls": 1,
            "strikeOuts": 1,
            "doubles": 1,
            "triples": 0,
            "sacFlies": 0,
            "hitByPitch": 0,
            "intentionalWalks": 0,
            "caughtStealing": 0,
            "groundIntoDoublePlay": 0,
            "sacBunts": 0,
            "catchersInterference": 0,
        },
        is_home=False,
    )
    g = parse_hitter_game_log_full(split, player_id=1, name="X", team="ABC", season=2025)
    assert g.b2 == 1
    assert g.b3 == 0
    assert g.sf == 0
    assert g.hbp == 0
    assert g.ibb == 0
    assert g.cs == 0
    assert g.gidp == 0
    assert g.sh == 0
    assert g.ci == 0
    assert g.is_home is False


def test_parse_treats_missing_fields_as_zero() -> None:
    # Older API responses or partial splits may omit columns. Default to 0
    # so the row still loads and the identity check catches genuine drift.
    split = _make_split(
        {
            "plateAppearances": 1,
            "atBats": 1,
            "hits": 0,
            "homeRuns": 0,
            "runs": 0,
            "rbi": 0,
            "stolenBases": 0,
            "baseOnBalls": 0,
            "strikeOuts": 1,
        }
    )
    g = parse_hitter_game_log_full(split, player_id=1, name="X", team=None, season=2025)
    assert g.b2 == 0 and g.b3 == 0 and g.sf == 0 and g.hbp == 0
    assert g.ibb == 0 and g.cs == 0 and g.gidp == 0 and g.sh == 0 and g.ci == 0
    assert g.is_home is True  # default


def test_pa_identity_gap_zero_for_clean_row() -> None:
    g = HitterGame(
        player_id=1,
        game_pk=1,
        name="X",
        team=None,
        season=2025,
        date=date(2025, 4, 1),
        pa=5,
        ab=3,
        h=1,
        hr=0,
        r=0,
        rbi=0,
        sb=0,
        bb=1,
        k=1,
        b2=0,
        b3=0,
        sf=1,
        hbp=0,
        ibb=0,
        cs=0,
        gidp=0,
        sh=0,
        ci=0,
        is_home=True,
    )
    # 5 == 3 + 1 + 0 + 1 + 0 + 0
    assert pa_identity_gap(g) == 0


def test_pa_identity_gap_detects_drift() -> None:
    g = HitterGame(
        player_id=1,
        game_pk=1,
        name="X",
        team=None,
        season=2025,
        date=date(2025, 4, 1),
        pa=5,
        ab=3,
        h=1,
        hr=0,
        r=0,
        rbi=0,
        sb=0,
        bb=1,
        k=1,
        b2=0,
        b3=0,
        sf=0,
        hbp=0,
        ibb=0,
        cs=0,
        gidp=0,
        sh=0,
        ci=0,
        is_home=True,
    )
    # PA=5, components sum to 4 -> gap of +1
    assert pa_identity_gap(g) == 1


def test_pa_identity_gap_detects_overcount() -> None:
    g = HitterGame(
        player_id=1,
        game_pk=1,
        name="X",
        team=None,
        season=2025,
        date=date(2025, 4, 1),
        pa=4,
        ab=3,
        h=1,
        hr=0,
        r=0,
        rbi=0,
        sb=0,
        bb=1,
        k=1,
        b2=0,
        b3=0,
        sf=1,
        hbp=0,
        ibb=0,
        cs=0,
        gidp=0,
        sh=0,
        ci=0,
        is_home=True,
    )
    # PA=4, components sum to 5 (e.g. parser double-counted SF) -> gap of -1
    assert pa_identity_gap(g) == -1
