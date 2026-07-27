import pandas as pd
import pytest

from fantasy_baseball.keepers.actuals import (
    HITTER_PT,
    HITTER_RATES,
    PITCHER_PT,
    PITCHER_RATES,
    coerce_numeric,
    innings_to_float,
    normalize_hitting,
    normalize_pitching,
)
from tests.test_keepers.conftest import mlb_hitting, mlb_pitching


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5.1", 5 + 1 / 3),
        ("5.2", 5 + 2 / 3),
        ("7.0", 7.0),
        ("0.1", 1 / 3),
        ("12", 12.0),
        (0, 0.0),
        (None, 0.0),
        ("", 0.0),
    ],
)
def test_innings_to_float(raw: object, expected: float) -> None:
    assert innings_to_float(raw) == pytest.approx(expected)


def test_innings_rejects_impossible_outs() -> None:
    # Only .0/.1/.2 are legal; .3 would silently become a third of an inning too many.
    with pytest.raises(ValueError, match="baseball-notation"):
        innings_to_float("5.3")


def test_coerce_numeric_handles_api_junk() -> None:
    assert coerce_numeric("3.45") == pytest.approx(3.45)
    assert coerce_numeric(None) == 0.0
    assert coerce_numeric("-.--") == 0.0
    assert coerce_numeric("") == 0.0
    assert coerce_numeric("abc") == 0.0


def test_normalize_hitting_builds_rates_on_correct_denominators() -> None:
    raw = mlb_hitting()
    out = normalize_hitting(raw)
    row = out.loc[1]
    assert row["pa"] == 600.0
    assert row["ab_pa"] == pytest.approx(540 / 600)
    assert row["h_ab"] == pytest.approx(162 / 540)  # AB, not PA
    assert row["hr_pa"] == pytest.approx(30 / 600)
    assert row["sb_pa"] == pytest.approx(12 / 600)
    assert row["r_pa"] == pytest.approx(90 / 600)
    assert row["rbi_pa"] == pytest.approx(100 / 600)


def test_normalize_pitching_converts_innings_notation() -> None:
    raw = mlb_pitching(**{"stat.inningsPitched": "180.1"})
    out = normalize_pitching(raw)
    row = out.loc[7]
    assert row["ip"] == pytest.approx(180 + 1 / 3)
    assert row["er_ip"] == pytest.approx(60 / (180 + 1 / 3))
    assert row["k_ip"] == pytest.approx(200 / (180 + 1 / 3))


def test_normalize_drops_rows_without_an_mlbam_id() -> None:
    raw = mlb_hitting(**{"player.id": [1, None]})
    assert list(normalize_hitting(raw).index) == [1]


def test_rate_constants_match_the_emitted_column_order() -> None:
    # A positional zip() over a normalized frame must not mis-pair columns.
    assert list(normalize_hitting(mlb_hitting()).columns) == [HITTER_PT, *HITTER_RATES]
    assert list(normalize_pitching(mlb_pitching()).columns) == [PITCHER_PT, *PITCHER_RATES]


def test_normalize_pitching_covers_every_rate() -> None:
    raw = mlb_pitching()
    row = normalize_pitching(raw).loc[7]
    assert row["w_ip"] == pytest.approx(15 / 180)
    assert row["bb_ip"] == pytest.approx(45 / 180)
    assert row["h_ip"] == pytest.approx(150 / 180)


def test_zero_playing_time_yields_nan_on_every_hitter_rate() -> None:
    # 0/0 must be NaN so the gate can see "no information", NOT 0.0 which reads
    # as "a real observation of zero rate" and would crush the fold.
    raw = mlb_hitting(
        **{
            "player.id": 3,
            "stat.plateAppearances": 0,
            "stat.atBats": 0,
            "stat.hits": 0,
            "stat.runs": 0,
            "stat.homeRuns": 0,
            "stat.rbi": 0,
            "stat.stolenBases": 0,
        }
    )
    out = normalize_hitting(raw)
    assert out.loc[3, "pa"] == 0.0
    for col in HITTER_RATES:
        assert pd.isna(out.loc[3, col])


def test_zero_innings_yields_nan_on_every_pitcher_rate() -> None:
    raw = mlb_pitching(
        **{
            "player.id": 8,
            "stat.inningsPitched": "0.0",
            "stat.earnedRuns": 0,
            "stat.baseOnBalls": 0,
            "stat.hits": 0,
            "stat.strikeOuts": 0,
            "stat.wins": 0,
        }
    )
    out = normalize_pitching(raw)
    assert out.loc[8, "ip"] == 0.0
    for col in PITCHER_RATES:
        assert pd.isna(out.loc[8, col])


def test_normalize_pitching_drops_rows_without_an_mlbam_id() -> None:
    raw = mlb_pitching(**{"player.id": [7, None]})
    assert list(normalize_pitching(raw).index) == [7]
