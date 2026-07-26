"""HR-confirmation backtest logic (issue #262): does barrel rate or park-adjusted
xHR confirm next-year HR better than the SLG-vs-xSLG proxy w_for_stat uses? Pure
(no I/O); the corpus is built by scripts/backtest_hr_confirm.py and passed in.

See docs/superpowers/specs/2026-07-26-hr-confirmation-backtest-design.md.
"""

from __future__ import annotations

from collections.abc import Collection
from statistics import fmean

from fantasy_baseball.analysis.breakout import (
    _confirm_gap,  # reused so the xslg candidate is byte-identical to w_for_stat's HR branch
    _reliability,
    line_rates,
)
from fantasy_baseball.analysis.breakout_backtest import (
    Corpus,
    _bootstrap_diff,
    _league_mean,
    _rates_to_line,
    _spearman,
    marcel_prior,
    rate_mae,
)

HrRecord = dict[str, float]
BarrelCalib = tuple[float, float]  # (slope, intercept) for HR/PA ~ brl_pa

CANDIDATES: tuple[str, ...] = ("xslg", "barrel", "xhr")
PA_FLOOR = 150.0
HR_MOVE_MIN = 0.005  # HR/PA move to count as a candidate (~3 HR / 600 PA)
SEED = 12345
MAE_EPS = 0.0005  # HR/PA tolerance for the MAE consistency flag (~0.3 HR / 600 PA)
SHIPPED_XSLG_SCALE = 0.150
HR_STABILIZE = 120.0  # shipped stat_stabilize["hr"]
CONFIRM_WEIGHT = 0.5  # shipped confirm_weight
HRPA_SCALE_GRID = [0.010 + 0.005 * i for i in range(11)]  # 0.010 .. 0.060
SLG_SCALE_GRID = [0.075 + 0.025 * i for i in range(8)]  # 0.075 .. 0.250


def build_hr_records(
    corpus: Corpus,
    years: Collection[int],
    *,
    pa_floor: float = PA_FLOOR,
    hr_move_min: float = HR_MOVE_MIN,
) -> list[HrRecord]:
    """Per-hitter-season HR records on the common support (all four confirm
    ingredients present), filtered to the moved-HR candidate population."""
    recs: list[HrRecord] = []
    for year in years:
        year_data = corpus[year]
        lg = _league_mean(year_data)
        for surface, sl, actual_next, hist, _zips in year_data.values():
            if sl.slg is None or sl.xslg is None or sl.brl_pa is None or sl.xhr is None:
                continue  # off common support
            pa = float(sl.pa)
            if pa < pa_floor:
                continue
            proj_line = {**surface, **_rates_to_line(marcel_prior(hist, lg, sl.age), surface)}
            prior_hr = line_rates(proj_line, "hitter")["hr"]
            surface_hr = line_rates(surface, "hitter")["hr"]
            if abs(surface_hr - prior_hr) < hr_move_min:
                continue  # HR did not move -- not a mirage/breakout candidate
            recs.append(
                {
                    "mlbam": float(sl.mlbam),
                    "prior_hr": prior_hr,
                    "surface_hr": surface_hr,
                    "actual_hr": actual_next["hr"],
                    "pa": pa,
                    "slg": sl.slg,
                    "xslg": sl.xslg,
                    "brl_pa": sl.brl_pa,
                    "xhr_rate": sl.xhr / pa if pa > 0 else 0.0,
                }
            )
    return recs


def fit_barrel_calibration(records: Collection[HrRecord]) -> BarrelCalib:
    """OLS slope/intercept of HR/PA ~ brl_pa (barrels are a rate skill; this maps
    them to an expected HR/PA). Scale-invariant: the slope absorbs brl_pa's units."""
    xs = [r["brl_pa"] for r in records]
    ys = [r["surface_hr"] for r in records]
    n = len(xs)
    if n < 2:
        return 0.0, fmean(ys) if ys else 0.0
    mx, my = fmean(xs), fmean(ys)
    var = sum((x - mx) ** 2 for x in xs)
    if var == 0.0:
        return 0.0, my
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / var
    return slope, my - slope * mx


def expected_hr_rate(candidate: str, rec: HrRecord, calib: BarrelCalib) -> float:
    """The candidate's expected HR/PA (for barrel/xhr). Unused for xslg (its gap is
    in SLG units); callers guard on candidate."""
    if candidate == "barrel":
        slope, intercept = calib
        return intercept + slope * rec["brl_pa"]
    if candidate == "xhr":
        return rec["xhr_rate"]
    raise ValueError(candidate)


def confirm(candidate: str, rec: HrRecord, calib: BarrelCalib, *, scale: float) -> float:
    """confirm in [0,1], reusing breakout._confirm_gap so xslg == w_for_stat's HR branch."""
    if candidate == "xslg":
        return _confirm_gap(rec["slg"], rec["xslg"], scale)
    return _confirm_gap(rec["surface_hr"], expected_hr_rate(candidate, rec, calib), scale)


def forward_hr(
    rec: HrRecord,
    confirm_value: float,
    *,
    cw: float = CONFIRM_WEIGHT,
    hr_stabilize: float = HR_STABILIZE,
) -> float:
    """prior + w*(surface-prior), w = reliability*((1-cw)+cw*confirm) -- the exact
    w_for_stat blend, HR branch."""
    reliability = _reliability(rec["pa"], hr_stabilize)
    w = reliability * ((1.0 - cw) + cw * confirm_value)
    return rec["prior_hr"] + w * (rec["surface_hr"] - rec["prior_hr"])


def _scale_grid(candidate: str) -> list[float]:
    return SLG_SCALE_GRID if candidate == "xslg" else HRPA_SCALE_GRID


def _forwards(
    records: Collection[HrRecord], candidate: str, calib: BarrelCalib, scale: float
) -> list[float]:
    return [forward_hr(r, confirm(candidate, r, calib, scale=scale)) for r in records]


def tune_scale(fit_records: list[HrRecord], candidate: str, calib: BarrelCalib) -> float:
    """Grid-search the confirm-gap scale on FIT records, maximizing forward-line
    Spearman (the same metric the verdict gates on)."""
    actual = [r["actual_hr"] for r in fit_records]
    best_scale, best_rho = _scale_grid(candidate)[0], -2.0
    for scale in _scale_grid(candidate):
        rho = _spearman(_forwards(fit_records, candidate, calib, scale), actual)
        if rho > best_rho:
            best_scale, best_rho = scale, rho
    return best_scale


def signed_gap(candidate: str, rec: HrRecord, calib: BarrelCalib) -> float:
    """The candidate's own over/under-performance signal (positive = overperformed)."""
    if candidate == "xslg":
        return rec["slg"] - rec["xslg"]
    return rec["surface_hr"] - expected_hr_rate(candidate, rec, calib)


def level_control(records: list[HrRecord], candidate: str, calib: BarrelCalib) -> list[float]:
    """Within prior-HR/PA terciles, Spearman(signed_gap, next-year change). A real
    luck signal stays negative in each tier (overperformance -> next-year decline)."""
    ordered = sorted(records, key=lambda r: r["prior_hr"])
    k = len(ordered) // 3
    if k < 2:
        return []
    tiers = [ordered[:k], ordered[k : 2 * k], ordered[2 * k :]]
    out: list[float] = []
    for tier in tiers:
        gaps = [signed_gap(candidate, r, calib) for r in tier]
        change = [r["actual_hr"] - r["surface_hr"] for r in tier]
        out.append(_spearman(gaps, change))
    return out


def _mean_mae(forwards: list[float], actual: list[float]) -> float:
    return fmean([rate_mae({"hr": f}, {"hr": a}) for f, a in zip(forwards, actual, strict=True)])


def _verdict_for(*, ci: tuple[float, float], mae_delta: float, tier_signs: list[int]) -> str:
    """mae_delta = MAE(candidate) - MAE(xslg). tier_signs = sign of each tier's
    level-control Spearman (expected: negative)."""
    if ci[0] <= 0:
        return "no (CI includes 0)"
    if len(tier_signs) < 3:
        return "inconclusive (thin sample for level-control)"
    if sum(1 for s in tier_signs if s < 0) < 2:
        return "level-confounded -- do not wire in"
    if mae_delta > MAE_EPS:
        return "CI-positive but MAE-inconsistent"
    return "wire-in eligible"


def run(
    corpus: Corpus, *, fit_years: Collection[int], report_years: Collection[int], seed: int = SEED
) -> dict:
    """Full backtest: calibrate + tune scales on fit_years, score all candidates on
    the held-out report_years, level-control, verdict per challenger vs xslg."""
    # Barrel HR/PA ~ brl_pa is calibrated on the FULL fit-year skill range (PA floor +
    # common support, NO mover filter) per the spec, so the slope is unbiased; scale
    # tuning uses the mover population `fit` (matching what report scoring evaluates).
    calib = fit_barrel_calibration(build_hr_records(corpus, fit_years, hr_move_min=0.0))
    fit = build_hr_records(corpus, fit_years)
    report = build_hr_records(corpus, report_years)
    actual = [r["actual_hr"] for r in report]

    scales = {c: tune_scale(fit, c, calib) for c in CANDIDATES}
    scales["xslg_shipped"] = SHIPPED_XSLG_SCALE
    forwards = {
        "xslg": _forwards(report, "xslg", calib, scales["xslg"]),
        "xslg_shipped": _forwards(report, "xslg", calib, SHIPPED_XSLG_SCALE),
        "barrel": _forwards(report, "barrel", calib, scales["barrel"]),
        "xhr": _forwards(report, "xhr", calib, scales["xhr"]),
        "surface": [r["surface_hr"] for r in report],
        "prior": [r["prior_hr"] for r in report],
    }
    spearman = {k: _spearman(v, actual) for k, v in forwards.items()}
    mae = {k: _mean_mae(v, actual) for k, v in forwards.items()}

    verdicts: dict[str, dict] = {}
    for cand in ("barrel", "xhr"):
        ci = _bootstrap_diff(forwards[cand], forwards["xslg"], actual, seed=seed)
        tiers = level_control(report, cand, calib)
        tier_signs = [1 if t >= 0 else -1 for t in tiers]
        verdicts[cand] = {
            "ci_vs_xslg": ci,
            "tier_spearman": tiers,
            "verdict": _verdict_for(
                ci=ci, mae_delta=mae[cand] - mae["xslg"], tier_signs=tier_signs
            ),
        }
    return {
        "n_fit": len(fit),
        "n_report": len(report),
        "barrel_calib": calib,
        "scales": scales,
        "spearman": spearman,
        "mae": mae,
        "verdicts": verdicts,
    }
