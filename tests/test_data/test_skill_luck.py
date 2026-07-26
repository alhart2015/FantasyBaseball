from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.data import skill_luck


def test_fetch_or_cache_refuses_empty_and_reuses(tmp_path: Path):
    calls = {"n": 0}
    good = pd.DataFrame({"a": [1, 2]})

    def _fetch_good():
        calls["n"] += 1
        return good

    p = tmp_path / "x.csv"
    out = skill_luck.fetch_or_cache(p, _fetch_good)
    assert list(out["a"]) == [1, 2] and calls["n"] == 1
    # cache hit: fetcher not called again
    skill_luck.fetch_or_cache(p, _fetch_good)
    assert calls["n"] == 1
    # empty fetch to a fresh path raises and writes nothing
    q = tmp_path / "y.csv"
    with pytest.raises(RuntimeError):
        skill_luck.fetch_or_cache(q, lambda: pd.DataFrame())
    assert not q.exists()


def test_load_mlb_hitters_derives_rates_and_drops_zero_pa(tmp_path: Path):
    raw = pd.DataFrame(
        {
            "mlbam": [665742, 700],
            "plateAppearances": [600, 0],  # second row: no PA -> dropped
            "atBats": [500, 400],
            "hits": [150, 100],
            "homeRuns": [30, 10],
            "runs": [100, 50],
            "rbi": [100, 45],
            "stolenBases": [10, 5],
            "avg": [0.300, 0.250],
            "baseOnBalls": [90, 30],
            "strikeOuts": [120, 90],
            "sacFlies": [5, 2],
        }
    )
    out = skill_luck.load_mlb_hitters(tmp_path, 2024, fetcher=lambda: raw)

    assert len(out) == 1  # pa<=0 row dropped
    row = out.iloc[0]
    assert row["mlbam"] == 665742
    assert row["k_pct"] == pytest.approx(120 / 600)  # SO/PA
    assert row["bb_pct"] == pytest.approx(90 / 600)  # BB/PA
    # BABIP = (H-HR) / (AB-SO-HR+SF)
    expected_babip = (150 - 30) / (500 - 120 - 30 + 5)
    assert row["babip"] == pytest.approx(expected_babip)


def test_build_hitter_skill_luck_joins_and_reports_coverage(tmp_path: Path):
    mlb = pd.DataFrame(
        {
            "mlbam": [665742, 800],
            "plateAppearances": [600, 550],
            "atBats": [500, 480],
            "hits": [150, 140],
            "homeRuns": [30, 10],
            "runs": [100, 60],
            "rbi": [100, 55],
            "stolenBases": [10, 5],
            "avg": [0.300, 0.290],
            "baseOnBalls": [90, 40],
            "strikeOuts": [120, 100],
            "sacFlies": [5, 3],
        }
    )
    sc_x = pd.DataFrame(
        {
            "player_id": [665742],
            "woba": [0.400],
            "est_woba": [0.360],
            "ba": [0.300],
            "est_ba": [0.270],
            "slg": [0.600],
            "est_slg": [0.520],
        }
    )
    sc_brl = pd.DataFrame({"player_id": [665742], "brl_percent": [14.0]})

    rows, cov = skill_luck.build_hitter_skill_luck(
        tmp_path,
        2024,
        fetchers={
            "mlb": lambda: mlb,
            "sc_x": lambda: sc_x,
            "sc_brl": lambda: sc_brl,
        },
    )
    soto = rows[665742]
    assert soto.mlbam == 665742 and soto.woba == 0.400 and soto.xwoba == 0.360
    assert soto.barrel_pct == 0.14 and soto.k_pct == pytest.approx(120 / 600)
    # mlbam 800 has no statcast row -> present but xStats None
    assert rows[800].xwoba is None and rows[800].woba is None
    assert cov.matched == 1 and cov.no_xstats == 1


def test_build_pitcher_skill_luck_joins_and_reports_coverage(tmp_path: Path):
    mlb = pd.DataFrame(
        {
            "mlbam": [543037, 900],
            "inningsPitched": ["180.0", "150.0"],
            "battersFaced": [720, 620],
            "wins": [15, 10],
            "saves": [0, 0],
            "strikeOuts": [216, 130],
            "baseOnBalls": [50, 55],
            "era": [3.20, 4.10],
            "whip": [1.10, 1.30],
        }
    )
    sc_x = pd.DataFrame({"player_id": [543037], "woba": [0.290], "est_woba": [0.300]})

    rows, cov = skill_luck.build_pitcher_skill_luck(
        tmp_path,
        2024,
        fetchers={"mlb": lambda: mlb, "sc_x": lambda: sc_x},
    )
    ace = rows[543037]
    assert ace.mlbam == 543037 and ace.woba == 0.290 and ace.xwoba == 0.300
    assert ace.k_pct == pytest.approx(216 / 720) and ace.bb_pct == pytest.approx(50 / 720)
    assert ace.ip == 180.0
    assert ace.pa == 0.0
    assert ace.barrel_pct is None and ace.xslg is None and ace.slg is None
    assert ace.xba is None and ace.ba is None and ace.babip is None
    # mlbam 900 has no statcast row -> present but xwoba/woba None
    assert rows[900].xwoba is None and rows[900].woba is None
    assert cov.matched == 1 and cov.no_xstats == 1
