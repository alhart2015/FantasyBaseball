import importlib.util
from pathlib import Path

from fantasy_baseball.analysis import breakout

_SPEC = importlib.util.spec_from_file_location(
    "backtest_hr_level",
    Path(__file__).resolve().parents[2] / "scripts" / "backtest_hr_level.py",
)
bhl = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(bhl)


def _row(mlbam, pa, slg, xslg, brl_pa, xhr):
    return breakout.SkillLuckRow(
        mlbam=mlbam,
        player_type="hitter",
        pa=pa,
        ip=0.0,
        age=27.0,
        barrel_pct=None,
        xslg=xslg,
        slg=slg,
        xba=None,
        ba=None,
        babip=None,
        xwoba=None,
        woba=None,
        k_pct=None,
        bb_pct=None,
        brl_pa=brl_pa,
        xhr=xhr,
    )


def _entry(hr, next_hr, brl_pa, mlbam):
    surface = {"pa": 600.0, "hr": hr, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}
    actual_next = breakout.line_rates(
        {"pa": 600.0, "hr": next_hr, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}, "hitter"
    )
    prior = {"pa": 600.0, "hr": 8, "r": 60, "rbi": 60, "sb": 5, "avg": 0.250}
    hist = [
        (2018, breakout.line_rates(prior, "hitter")),
        (2019, breakout.line_rates(prior, "hitter")),
    ]
    return (surface, _row(mlbam, 600.0, 0.520, 0.470, brl_pa, hr - 4), actual_next, hist, None)


def test_run_level_gate_wellformed():
    def year():
        return {1000 + i: _entry(26 + i, 26 + i - 4, 0.04 + 0.004 * i, 1000 + i) for i in range(12)}

    corpus = {2020: year(), 2021: year()}
    res = bhl.run_level_gate(corpus, fit_years=[2020], report_years=[2021])
    assert res["n_report"] == 12
    assert "gate_ci" in res and len(res["gate_ci"]) == 2
    assert res["gate_clears"] in (True, False)
    assert res["weight_form"] in ("flat", "rel")
    assert set(res["direct_level_spearman"]) >= {"surface", "barrel", "xhr"}
    assert 0.0 <= res["prod_constants"]["w_s"] <= 1.0
    assert 0.0 <= res["prod_constants"]["cw"] <= 1.0
