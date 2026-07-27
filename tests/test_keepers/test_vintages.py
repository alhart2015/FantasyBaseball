from pathlib import Path

import pandas as pd
import pytest

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
