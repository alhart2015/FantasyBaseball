from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.keepers.actuals import (
    HITTER_PT,
    HITTER_RATES,
    PITCHER_PT,
    PITCHER_RATES,
    normalize_hitting,
    normalize_pitching,
)
from fantasy_baseball.keepers.vintages import decompose_hitters, decompose_pitchers, load_vintage


def test_decompose_hitters_uses_own_denominators() -> None:
    df = pd.DataFrame(
        {
            "MLBAMID": [11],
            "PA": [600],
            "AB": [540],
            "H": [162],
            "R": [90],
            "HR": [30],
            "RBI": [100],
            "SB": [12],
        }
    )
    row = decompose_hitters(df).loc[11]
    assert row["h_ab"] == pytest.approx(162 / 540)
    assert row["hr_pa"] == pytest.approx(30 / 600)
    assert row["r_pa"] == pytest.approx(90 / 600)
    assert row["rbi_pa"] == pytest.approx(100 / 600)
    assert row["ab_pa"] == pytest.approx(540 / 600)


def test_decompose_pitchers_splits_bb_and_hits() -> None:
    # BB and H_allowed must stay separate: calculate_replacement_rates needs them
    # as distinct columns, so folding (BB+H)/IP as one unit is not acceptable.
    df = pd.DataFrame(
        {"MLBAMID": [22], "IP": [180.0], "ER": [60], "BB": [45], "H": [150], "SO": [200], "W": [15]}
    )
    row = decompose_pitchers(df).loc[22]
    assert row["bb_ip"] == pytest.approx(45 / 180)
    assert row["h_ip"] == pytest.approx(150 / 180)
    assert row["k_ip"] == pytest.approx(200 / 180)
    assert row["w_ip"] == pytest.approx(15 / 180)
    assert row["er_ip"] == pytest.approx(60 / 180)
    assert "whip" not in row.index


def test_zero_denominator_zips_rows_yield_nan() -> None:
    # ZiPS 2028 has 14 pitcher rows at IP=0 and 3 hitter rows at PA=0.
    df = pd.DataFrame(
        {"MLBAMID": [33], "IP": [0.0], "ER": [0], "BB": [0], "H": [0], "SO": [0], "W": [0]}
    )
    out = decompose_pitchers(df)
    assert out.loc[33, "ip"] == 0.0
    for col in ("k_ip", "w_ip", "er_ip", "bb_ip", "h_ip"):
        assert pd.isna(out.loc[33, col])


def test_zero_pa_hitter_rows_yield_nan() -> None:
    df = pd.DataFrame(
        {
            "MLBAMID": [44],
            "PA": [0],
            "AB": [0],
            "H": [0],
            "R": [0],
            "HR": [0],
            "RBI": [0],
            "SB": [0],
        }
    )
    out = decompose_hitters(df)
    assert out.loc[44, "pa"] == 0.0
    assert pd.isna(out.loc[44, "hr_pa"])
    assert pd.isna(out.loc[44, "h_ab"])


def test_decompose_drops_rows_without_an_mlbam_id() -> None:
    df = pd.DataFrame(
        {
            "MLBAMID": [11, None],
            "PA": [600, 400],
            "AB": [540, 360],
            "H": [162, 90],
            "R": [90, 50],
            "HR": [30, 10],
            "RBI": [100, 45],
            "SB": [12, 3],
        }
    )
    assert list(decompose_hitters(df).index) == [11]


def test_load_vintage_reads_real_files() -> None:
    root = Path(__file__).resolve().parents[2] / "data" / "projections"
    hitters, pitchers = load_vintage(2026, root)
    assert len(hitters) > 1000
    assert len(pitchers) > 1000
    assert hitters.index.name == "mlbam_id"
    assert pitchers.index.name == "mlbam_id"


def test_both_loaders_decompose_to_the_same_rates() -> None:
    """The ZiPS side and the MLB-actuals side must agree, rate for rate.

    build_pairs computes `actual_Y[cols] - zips[cols]`. A mismatched column NAME
    would raise, but a mismatched DENOMINATOR under a matching name is silent and
    corrupts every residual. Nothing else in the suite compares the two loaders.
    """
    counts = {"pa": 600, "ab": 540, "h": 162, "r": 90, "hr": 30, "rbi": 100, "sb": 12}
    zips_row = pd.DataFrame(
        {
            "MLBAMID": [5],
            "PA": [counts["pa"]],
            "AB": [counts["ab"]],
            "H": [counts["h"]],
            "R": [counts["r"]],
            "HR": [counts["hr"]],
            "RBI": [counts["rbi"]],
            "SB": [counts["sb"]],
        }
    )
    mlb_row = pd.DataFrame(
        {
            "player.id": [5],
            "stat.plateAppearances": [counts["pa"]],
            "stat.atBats": [counts["ab"]],
            "stat.hits": [counts["h"]],
            "stat.runs": [counts["r"]],
            "stat.homeRuns": [counts["hr"]],
            "stat.rbi": [counts["rbi"]],
            "stat.stolenBases": [counts["sb"]],
        }
    )
    pd.testing.assert_frame_equal(decompose_hitters(zips_row), normalize_hitting(mlb_row))

    pitch = {"ip": 180.0, "er": 60, "bb": 45, "h": 150, "so": 200, "w": 15}
    zips_p = pd.DataFrame(
        {
            "MLBAMID": [6],
            "IP": [pitch["ip"]],
            "ER": [pitch["er"]],
            "BB": [pitch["bb"]],
            "H": [pitch["h"]],
            "SO": [pitch["so"]],
            "W": [pitch["w"]],
        }
    )
    mlb_p = pd.DataFrame(
        {
            "player.id": [6],
            "stat.inningsPitched": ["180.0"],
            "stat.earnedRuns": [pitch["er"]],
            "stat.baseOnBalls": [pitch["bb"]],
            "stat.hits": [pitch["h"]],
            "stat.strikeOuts": [pitch["so"]],
            "stat.wins": [pitch["w"]],
        }
    )
    pd.testing.assert_frame_equal(decompose_pitchers(zips_p), normalize_pitching(mlb_p))


def test_vintage_columns_match_the_rate_constants() -> None:
    # Same invariant the actuals loader is held to, so a positional zip() over
    # either side of the residual cannot mis-pair.
    hitters = pd.DataFrame(
        {
            "MLBAMID": [1],
            "PA": [600],
            "AB": [540],
            "H": [162],
            "R": [90],
            "HR": [30],
            "RBI": [100],
            "SB": [12],
        }
    )
    pitchers = pd.DataFrame(
        {"MLBAMID": [2], "IP": [180.0], "ER": [60], "BB": [45], "H": [150], "SO": [200], "W": [15]}
    )
    assert list(decompose_hitters(hitters).columns) == [HITTER_PT, *HITTER_RATES]
    assert list(decompose_pitchers(pitchers).columns) == [PITCHER_PT, *PITCHER_RATES]


def test_load_vintage_falls_back_to_a_year_suffixed_filename(tmp_path: Path) -> None:
    # data/projections/2025 uses the year-suffixed form; the glob fallback is the
    # only thing that would find it, and nothing else exercises that branch.
    year = tmp_path / "2099"
    year.mkdir()
    pd.DataFrame(
        {
            "MLBAMID": [1],
            "PA": [600],
            "AB": [540],
            "H": [162],
            "R": [90],
            "HR": [30],
            "RBI": [100],
            "SB": [12],
        }
    ).to_csv(year / "zips-hitters-2099.csv", index=False)
    pd.DataFrame(
        {"MLBAMID": [2], "IP": [180.0], "ER": [60], "BB": [45], "H": [150], "SO": [200], "W": [15]}
    ).to_csv(year / "zips-pitchers-2099.csv", index=False)
    hitters, pitchers = load_vintage(2099, tmp_path)
    assert list(hitters.index) == [1]
    assert list(pitchers.index) == [2]


def test_load_vintage_raises_when_no_export_exists(tmp_path: Path) -> None:
    (tmp_path / "2098").mkdir()
    with pytest.raises(FileNotFoundError, match="no ZiPS hitters export"):
        load_vintage(2098, tmp_path)
