import pandas as pd
import pytest

from fantasy_baseball.keepers.actuals import (
    coerce_numeric,
    innings_to_float,
    normalize_hitting,
    normalize_pitching,
)


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
    raw = pd.DataFrame(
        {
            "player.id": [1],
            "stat.plateAppearances": [600],
            "stat.atBats": [540],
            "stat.hits": [162],
            "stat.runs": [90],
            "stat.homeRuns": [30],
            "stat.rbi": [100],
            "stat.stolenBases": [12],
        }
    )
    out = normalize_hitting(raw)
    row = out.loc[1]
    assert row["pa"] == 600.0
    assert row["ab_pa"] == pytest.approx(540 / 600)
    assert row["h_ab"] == pytest.approx(162 / 540)  # AB, not PA
    assert row["hr_pa"] == pytest.approx(30 / 600)
    assert row["sb_pa"] == pytest.approx(12 / 600)


def test_normalize_pitching_converts_innings_notation() -> None:
    raw = pd.DataFrame(
        {
            "player.id": [7],
            "stat.inningsPitched": ["180.1"],
            "stat.earnedRuns": [60],
            "stat.baseOnBalls": [45],
            "stat.hits": [150],
            "stat.strikeOuts": [200],
            "stat.wins": [15],
        }
    )
    out = normalize_pitching(raw)
    row = out.loc[7]
    assert row["ip"] == pytest.approx(180 + 1 / 3)
    assert row["er_ip"] == pytest.approx(60 / (180 + 1 / 3))
    assert row["k_ip"] == pytest.approx(200 / (180 + 1 / 3))


def test_zero_playing_time_yields_nan_rates_not_zeros() -> None:
    # 0/0 must be NaN so the gate can see "no information", NOT 0.0 which reads
    # as "a real observation of zero rate" and would crush the fold.
    raw = pd.DataFrame(
        {
            "player.id": [3],
            "stat.plateAppearances": [0],
            "stat.atBats": [0],
            "stat.hits": [0],
            "stat.runs": [0],
            "stat.homeRuns": [0],
            "stat.rbi": [0],
            "stat.stolenBases": [0],
        }
    )
    out = normalize_hitting(raw)
    assert out.loc[3, "pa"] == 0.0
    assert pd.isna(out.loc[3, "hr_pa"])


def test_normalize_drops_rows_without_an_mlbam_id() -> None:
    raw = pd.DataFrame(
        {
            "player.id": [1, None],
            "stat.plateAppearances": [600, 100],
            "stat.atBats": [540, 90],
            "stat.hits": [162, 27],
            "stat.runs": [90, 10],
            "stat.homeRuns": [30, 2],
            "stat.rbi": [100, 9],
            "stat.stolenBases": [12, 1],
        }
    )
    assert list(normalize_hitting(raw).index) == [1]
