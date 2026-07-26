from __future__ import annotations

import random
from collections.abc import Collection
from statistics import fmean
from typing import Any

from fantasy_baseball.analysis.breakout import (
    DEFAULT_WMAP,
    SkillLuckRow,
    WMapParams,
    adjust_line,
    line_rates,
)

_RECENCY = {0: 5.0, 1: 4.0, 2: 3.0}  # most-recent .. 3rd
_REGRESS_W = 4.0  # league-mean pseudo-weight

# SGP-calibrated per-PA-rate weights, matched to breakout.py:LABEL_WEIGHTS so no
# single category (previously avg, at ~96% of the composite) dominates the
# spread. era/whip are NEGATIVE so sgp_on_ruler stays a plain dot product where
# lower is better.
DEFAULT_RULER = {
    "hr": 65.0,
    "r": 33.0,
    "rbi": 33.0,
    "sb": 85.0,
    "avg": 40.0,
    "k": 10.0,
    "w": 100.0,
    "sv": 90.0,
    "era": -2.0,
    "whip": -60.0,
}

CANDIDATE_DEVIATION = (
    0.15  # raw surface-vs-prior deviation to count as a breakout/decline candidate
)
_TUNE_GRID: dict[str, list[float]] = {
    "confirm_weight": [0.3, 0.5, 0.7],
    "hr": [80.0, 120.0, 200.0],
    "avg": [600.0, 800.0, 1200.0],
}

# A counting/rate line keyed by stat name (e.g. {"pa": 600.0, "hr": 40.0, "avg": 0.320}).
Line = dict[str, float]
# corpus[year][fg_id] = (surface_line, skill_luck_row, actual_next_rates, history, zips_line)
CorpusEntry = tuple[Line, SkillLuckRow, Line, list[tuple[int, Line]], Line | None]
CorpusYear = dict[int, CorpusEntry]
Corpus = dict[int, CorpusYear]
# Per-candidate scored record: floats (surface/skill/zips/actual/believed scores,
# zips may be None) plus the rate-line dicts _label_lift/rate_mae need.
Record = dict[str, Any]


def marcel_prior(
    history: list[tuple[int, dict[str, float]]], league_mean: dict[str, float], age: float | None
) -> dict[str, float]:
    history = sorted(history, key=lambda t: t[0], reverse=True)[:3]
    stats = set().union(*[set(d) for _, d in history]) if history else set(league_mean)
    prior = {}
    for s in stats:
        num = _REGRESS_W * league_mean.get(s, 0.0)
        den = _REGRESS_W
        for i, (_, line) in enumerate(history):
            wt = _RECENCY.get(i, 0.0)
            if line.get(s) is not None:
                num += wt * line[s]
                den += wt
        val = num / den if den > 0 else league_mean.get(s, 0.0)
        if age is not None:
            val *= 1.0 - 0.003 * (age - 27.0)  # mild peak-27 age curve
        prior[s] = val
    return prior


def rate_mae(pred_rates: dict[str, float], actual_rates: dict[str, float]) -> float:
    keys = set(pred_rates) & set(actual_rates)
    if not keys:
        return 0.0
    return sum(abs(pred_rates[k] - actual_rates[k]) for k in keys) / len(keys)


def sgp_on_ruler(rates: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(s, 0.0) * v for s, v in rates.items())


def _league_mean(year_data: CorpusYear) -> Line:
    rows = [line_rates(s, "hitter") for s, *_ in year_data.values()]
    keys: set[str] = set().union(*[set(r) for r in rows]) if rows else set()
    return {k: fmean([r.get(k, 0.0) for r in rows]) for k in keys}


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2:
        return 0.0

    def ranks(v: list[float]) -> list[float]:
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = float(pos)
        return rk

    rx, ry = ranks(xs), ranks(ys)
    mx, my = fmean(rx), fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0


def _params(confirm_weight: float, hr_stab: float, avg_stab: float) -> WMapParams:
    base = dict(DEFAULT_WMAP.stat_stabilize)
    base.update({"hr": hr_stab, "avg": avg_stab})
    return WMapParams(confirm_weight=confirm_weight, stat_stabilize=base)


def _rates_to_line(rate_line: Line, pt_source: Line) -> Line:
    # rebuild a counting line at the PT-source's playing time so adjust_line sees
    # a projection line in counting shape. Rate-only stats pass through.
    pt = float(pt_source.get("pa", 0.0))
    line = dict(pt_source)
    for s, v in rate_line.items():
        line[s] = v if s in ("avg", "era", "whip") else v * pt
    return line


def _records(
    corpus: Corpus, years: Collection[int], params: WMapParams, ruler: dict[str, float]
) -> list[Record]:
    """Per-candidate scored records for `years` under `params` (hitters only)."""
    recs: list[Record] = []
    for year in years:
        year_data = corpus[year]
        lg = _league_mean(year_data)
        for surface, sl, actual_next, hist, zips_line in year_data.values():
            proj_line = {**surface, **_rates_to_line(marcel_prior(hist, lg, sl.age), surface)}
            res = adjust_line(
                surface,
                proj_line,
                sl,
                "hitter",
                params=params,
                deviation_threshold=CANDIDATE_DEVIATION,
            )
            if abs(res.surface_deviation) < CANDIDATE_DEVIATION:
                continue  # not a breakout/decline candidate this year
            surface_rates = line_rates(surface, "hitter")
            adjusted_rates = line_rates(res.adjusted_line, "hitter")
            recs.append(
                {
                    "surface": sgp_on_ruler(surface_rates, ruler),
                    "skill": sgp_on_ruler(adjusted_rates, ruler),
                    "zips": (
                        sgp_on_ruler(line_rates(zips_line, "hitter"), ruler)
                        if zips_line is not None
                        else None
                    ),
                    "actual": sgp_on_ruler(actual_next, ruler),
                    "believed": res.believed_deviation,
                    "surface_rates": surface_rates,
                    "adjusted_rates": adjusted_rates,
                    "prior_rates": line_rates(proj_line, "hitter"),
                    "actual_rates": actual_next,
                }
            )
    return recs


def tune_wmap(
    corpus: Corpus, fit_years: Collection[int], *, ruler: dict[str, float] = DEFAULT_RULER
) -> WMapParams:
    """Grid-search w-params on fit_years ONLY; never reads report_years."""
    best, best_rho = DEFAULT_WMAP, -2.0
    for cw in _TUNE_GRID["confirm_weight"]:
        for hr in _TUNE_GRID["hr"]:
            for avg in _TUNE_GRID["avg"]:
                p = _params(cw, hr, avg)
                recs = _records(corpus, fit_years, p, ruler)
                rho = _spearman([r["skill"] for r in recs], [r["actual"] for r in recs])
                if rho > best_rho:
                    best, best_rho = p, rho
    return best


_RETENTION_CLAMP = 2.0  # a near-zero (sr-pr) gap must not blow up into a dominant outlier


def _retention(rec: Record) -> float:
    holds: list[float] = []
    for s, sr in rec["surface_rates"].items():
        pr = rec["prior_rates"].get(s, sr)
        ar = rec["actual_rates"].get(s)
        if ar is None or abs(sr - pr) < 1e-9:
            continue
        r = (ar - pr) / (sr - pr)  # 1.0 = fully held, 0.0 = fully regressed
        holds.append(max(-_RETENTION_CLAMP, min(_RETENTION_CLAMP, r)))
    return fmean(holds) if holds else 0.0


def _label_lift(recs: list[Record]) -> float:
    if len(recs) < 3:
        return 0.0
    ordered = sorted(recs, key=lambda r: r["believed"])
    k = len(ordered) // 3
    return fmean([_retention(r) for r in ordered[-k:]]) - fmean(
        [_retention(r) for r in ordered[:k]]
    )


def _bootstrap_diff(
    a: list[float], b: list[float], actual: list[float], *, iters: int = 2000, seed: int = 0
) -> tuple[float, float]:
    # 95% CI on spearman(a, actual) - spearman(b, actual) via seeded resampling.
    rng = random.Random(seed)
    n = len(actual)
    if n < 2:
        return 0.0, 0.0
    diffs: list[float] = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        da = _spearman([a[i] for i in idx], [actual[i] for i in idx])
        db = _spearman([b[i] for i in idx], [actual[i] for i in idx])
        diffs.append(da - db)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]


def run_backtest(
    corpus: Corpus,
    *,
    fit_years: Collection[int],
    report_years: Collection[int],
    params: WMapParams | None = None,
    ruler: dict[str, float] = DEFAULT_RULER,
) -> dict[str, Any]:
    """Hitters-only v1. Tunes w on fit_years (unless params given), evaluates the
    fixed params on the candidate population of report_years across three estimators.
    Pitcher backtest is a named follow-up (thinner Savant pitcher xStats coverage)."""
    if params is None:
        params = tune_wmap(corpus, fit_years, ruler=ruler)
    recs = _records(corpus, report_years, params, ruler)
    actual = [r["actual"] for r in recs]
    zrecs = [r for r in recs if r["zips"] is not None]
    spearman = {
        "surface": _spearman([r["surface"] for r in recs], actual),
        "skill_adjusted": _spearman([r["skill"] for r in recs], actual),
        "pure_zips": _spearman([r["zips"] for r in zrecs], [r["actual"] for r in zrecs]),
    }
    ci_vs_surface = _bootstrap_diff(
        [r["skill"] for r in recs], [r["surface"] for r in recs], actual
    )
    ci_vs_zips = (
        _bootstrap_diff(
            [r["skill"] for r in zrecs], [r["zips"] for r in zrecs], [r["actual"] for r in zrecs]
        )
        if len(zrecs) >= 2
        else (0.0, 0.0)
    )
    clears = ci_vs_surface[0] > 0 and ci_vs_zips[0] > 0

    def _mean_rate_mae(key: str) -> float:
        return fmean([rate_mae(r[key], r["actual_rates"]) for r in recs]) if recs else 0.0

    return {
        "spearman": spearman,
        "ci_skill_vs_surface": ci_vs_surface,
        "ci_skill_vs_zips": ci_vs_zips,
        "rate_mae": {
            "surface": _mean_rate_mae("surface_rates"),
            "skill_adjusted": _mean_rate_mae("adjusted_rates"),
        },
        "label_lift": _label_lift(recs),
        "verdict": "clears gate" if clears else "not good enough",
        "n": len(recs),
        "n_zips": len(zrecs),  # pure_zips spearman is over this subset, not `n`
    }
