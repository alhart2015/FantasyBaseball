"""#442: score the fantasy-relevant hitters (top N by FanGraphs' projected value) too."""

import pandas as pd
import pytest

from fantasy_baseball.hitter_ros import backtest
from fantasy_baseball.hitter_ros.evaluate import fantasy_value
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import DEFAULT_TEAM_AB, Category

DENOMS = get_sgp_denominators()


def _projection(**cols):
    base = {"pa": [600.0], "ab": [540.0], "r": [0.0], "hr": [0.0], "rbi": [0.0], "sb": [0.0]}
    base.update(cols)
    base.setdefault("avg", [0.250])
    return pd.DataFrame(base, index=pd.Index([1], name="player_id"))


def test_value_is_sgp_of_the_counts_and_marginal_hits():
    # 30 HR over 600 PA, nothing else, a replacement-level AVG.
    p = _projection(hr=[30 / 600])
    assert fantasy_value(p, DENOMS).iloc[0] == pytest.approx(30 / DENOMS[Category.HR])
    # A .300 hitter on 540 AB: 27 hits over replacement.
    p = _projection(avg=[0.300])
    expect = 0.05 * 540 / (DENOMS[Category.AVG] * DEFAULT_TEAM_AB)
    assert fantasy_value(p, DENOMS).iloc[0] == pytest.approx(expect)


def _systems():
    ids = pd.Index([1, 2, 3], name="player_id")
    star = {"pa": 650.0, "ab": 580.0, "r": 0.16, "hr": 0.06, "rbi": 0.16, "sb": 0.03, "avg": 0.290}
    good = {**star, "hr": 0.04, "avg": 0.270}
    bench = {**star, "pa": 200.0, "ab": 180.0, "hr": 0.02, "avg": 0.230}
    a = pd.DataFrame([star, good, bench], index=ids)
    # The second system rates player 2 below the bench bat; on average he's still second.
    b = a.copy()
    b.loc[2, "pa"] = 150.0
    b.loc[2, "ab"] = 135.0
    return {"steamer": a, "zips": b}


def test_relevant_players_rank_by_mean_value_over_systems():
    assert list(backtest.relevant_players(_systems(), DENOMS, top=2)) == [1, 2]
    assert list(backtest.relevant_players(_systems(), DENOMS, top=1)) == [1]


def test_rows_are_tagged_and_untagged_without_fangraphs(monkeypatch):
    monkeypatch.setattr(backtest, "RELEVANT_TOP", 1)
    scored = pd.DataFrame({"player_id": [1, 3, 3], "system": ["ours"] * 3})
    tagged = backtest._tag_relevant(scored, _systems(), DENOMS)
    assert tagged["relevant"].dtype == "boolean"
    assert tagged["relevant"].tolist() == [True, False, False]
    assert backtest._tag_relevant(scored, {}, DENOMS)["relevant"].isna().all()


def _scored(unit):
    rows = []
    for u in ("2025", "2026"):
        for pid in range(1, 9):
            for system in ("ours", "fg_blend"):
                for stat in ("r", "hr", "rbi", "sb", "avg"):
                    actual = pid / 100
                    proj = actual if system == "ours" else -actual  # blend orders backwards
                    rows.append(
                        {
                            unit: u,
                            "player_id": pid,
                            "system": system,
                            "stat": stat,
                            "projected": proj,
                            "actual": actual,
                            "pa": 300.0,
                            "abs_err": abs(proj - actual),
                            "lf_err": abs(proj - actual),
                            "career_pa": 100.0 if pid <= 2 else 2000.0,
                            "relevant": pid <= 5,
                        }
                    )
    df = pd.DataFrame(rows)
    return df.assign(relevant=df["relevant"].astype("boolean"))


def test_summary_scores_the_relevant_hitters_among_themselves():
    md = "\n".join(backtest._relevant_blocks(_scored("snapshot"), "snapshot"))
    assert f"top {backtest.RELEVANT_TOP}" in md
    assert "5 of them scored per snapshot" in md
    assert "Of them, under 700 MLB PA when projected, 2 per snapshot" in md
    # Frames scored before the tag: no block.
    assert backtest._relevant_blocks(_scored("snapshot").drop(columns="relevant"), "season") == []


def test_compare_reports_the_relevant_gap():
    from scripts.compare_hitter_ros_runs import _relevant_gaps

    row: dict[str, object] = {}
    _relevant_gaps(row, _scored("season"), "pre")
    # Ours orders the relevant five perfectly, the blend backwards: +100 points.
    assert row["pre_top_pairw_gap_hr"] == pytest.approx(100.0)
    untagged: dict[str, object] = {}
    _relevant_gaps(untagged, _scored("season").drop(columns="relevant"), "pre")
    assert untagged == {}


def test_league_denominators_fall_back_to_the_defaults(tmp_path):
    assert backtest.league_denominators(tmp_path / "missing.yaml") == DENOMS
