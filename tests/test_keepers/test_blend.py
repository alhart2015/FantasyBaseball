from __future__ import annotations

import pandas as pd
import pytest

from fantasy_baseball.keepers.actuals import HITTER_PT, HITTER_RATES, PITCHER_PT, PITCHER_RATES
from fantasy_baseball.keepers.blend import parse_blend

_HITTER = {
    "mlbam_id": 11.0,
    "pa": 600.0,
    "ab": 540.0,
    "h": 162.0,
    "hr": 30.0,
    "r": 90.0,
    "rbi": 100.0,
    "sb": 12.0,
    "name": "Test Hitter",
}
_PITCHER = {
    "mlbam_id": 22.0,
    "ip": 180.0,
    "k": 200.0,
    "w": 15.0,
    "sv": 3.0,
    "er": 60.0,
    "bb": 45.0,
    "h_allowed": 150.0,
    "name": "Test Pitcher",
}


def _payload(hitters=None, pitchers=None) -> dict:
    return {
        "hitters": [dict(_HITTER)] if hitters is None else hitters,
        "pitchers": [dict(_PITCHER)] if pitchers is None else pitchers,
    }


def test_parse_blend_emits_the_canonical_hitter_schema() -> None:
    out = parse_blend(_payload(), "hitter")
    assert list(out.columns) == [HITTER_PT, *HITTER_RATES]
    row = out.loc[11]
    assert row["pa"] == pytest.approx(600.0)
    assert row["ab_pa"] == pytest.approx(540 / 600)
    assert row["h_ab"] == pytest.approx(162 / 540)
    assert row["hr_pa"] == pytest.approx(30 / 600)


def test_parse_blend_emits_the_canonical_pitcher_schema() -> None:
    out = parse_blend(_payload(), "pitcher")
    assert list(out.columns) == [PITCHER_PT, *PITCHER_RATES]
    row = out.loc[22]
    assert row["k_ip"] == pytest.approx(200 / 180)
    assert row["sv_ip"] == pytest.approx(3 / 180)
    # `h_allowed`, not `h` -- a pitcher record's `h` would read as hits taken.
    assert row["h_ip"] == pytest.approx(150 / 180)


def test_parse_blend_drops_records_without_an_mlbam_id() -> None:
    rows = [dict(_HITTER), {**_HITTER, "mlbam_id": None, "name": "No Id"}]
    assert list(parse_blend(_payload(hitters=rows), "hitter").index) == [11]


def test_parse_blend_keeps_the_higher_volume_row_for_a_duplicate_id() -> None:
    """A traded player appears twice; the short stint must not win."""
    rows = [{**_HITTER, "pa": 120.0, "ab": 100.0}, {**_HITTER, "pa": 600.0, "ab": 540.0}]
    out = parse_blend(_payload(hitters=rows), "hitter")
    assert len(out) == 1
    assert out.loc[11, "pa"] == pytest.approx(600.0)


def test_parse_blend_yields_nan_not_zero_for_a_zero_volume_line() -> None:
    rows = [{**_HITTER, "pa": 0.0, "ab": 0.0, "h": 0.0}]
    out = parse_blend(_payload(hitters=rows), "hitter")
    assert out.loc[11, "pa"] == 0.0
    for col in HITTER_RATES:
        assert pd.isna(out.loc[11, col])


def test_parse_blend_raises_on_a_missing_field_rather_than_emitting_nan() -> None:
    rows = [{k: v for k, v in _HITTER.items() if k != "sb"}]
    with pytest.raises(KeyError, match="sb"):
        parse_blend(_payload(hitters=rows), "hitter")


def test_parse_blend_raises_on_an_empty_pool() -> None:
    with pytest.raises(ValueError, match="no 'hitters' records"):
        parse_blend({"hitters": [], "pitchers": [dict(_PITCHER)]}, "hitter")


def test_parse_blend_rejects_an_unknown_player_type() -> None:
    with pytest.raises(ValueError, match="hitter"):
        parse_blend(_payload(), "batter")
