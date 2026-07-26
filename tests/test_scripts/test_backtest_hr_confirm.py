from fantasy_baseball.analysis import breakout, hr_confirm


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


_PA = 600.0
# Low-HR prior history so every current surface is a clear mover
# (|surface-prior| >> HR_MOVE_MIN); avoids any tuning-to-pass on the filter.
_PRIOR_HR = 8


def _entry(hr, next_hr, brl_pa, xhr, mlbam):
    surface = {"pa": _PA, "hr": hr, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}
    actual_next = breakout.line_rates(
        {"pa": _PA, "hr": next_hr, "r": 80, "rbi": 80, "sb": 5, "avg": 0.270}, "hitter"
    )
    prior_line = {"pa": _PA, "hr": _PRIOR_HR, "r": 60, "rbi": 60, "sb": 5, "avg": 0.250}
    hist = [
        (2018, breakout.line_rates(prior_line, "hitter")),
        (2019, breakout.line_rates(prior_line, "hitter")),
    ]
    return (surface, _row(mlbam, _PA, 0.520, 0.470, brl_pa, xhr), actual_next, hist, None)


def test_run_produces_wellformed_verdicts():
    # 12 hitters/year, HR spread 26..37 (all clear movers vs the low prior); brl_pa
    # varies so the barrel calibration is non-degenerate; overperformers regress
    # next year (next_hr = hr - 4), and xhr tracks the regressed level.
    def year():
        data = {}
        for i in range(12):
            hr = 26 + i
            brl_pa = 0.04 + 0.004 * i
            data[1000 + i] = _entry(hr, hr - 4, brl_pa, (hr - 4), 1000 + i)
        return data

    corpus = {2020: year(), 2021: year()}
    res = hr_confirm.run(corpus, fit_years=[2020], report_years=[2021])
    assert res["n_report"] == 12  # all 12 clear the PA floor + mover filter
    assert set(res["verdicts"]) == {"barrel", "xhr"}
    for cand in ("barrel", "xhr"):
        assert res["verdicts"][cand]["verdict"] in {
            "no (CI includes 0)",
            "level-confounded -- do not wire in",
            "CI-positive but MAE-inconsistent",
            "wire-in eligible",
            "inconclusive (thin sample for level-control)",
        }
    # deterministic: identical CI bounds on a second run (seeded bootstrap)
    res2 = hr_confirm.run(corpus, fit_years=[2020], report_years=[2021])
    assert res["verdicts"]["xhr"]["ci_vs_xslg"] == res2["verdicts"]["xhr"]["ci_vs_xslg"]
