"""Calibrate the closer role-switch mixture curves (SV variance) from realized data.

Fits a K-component mean-1 mixture by maximum likelihood of realized SV:
  SV ~ sum_k p_k(s) * NegBin(s*a_k(s), r),  a_k = w_k/p_k,
with p(s) and w(s) each a K-way softmax over K-1 free logit lines in s, and r fixed
at STAT_DISPERSION['sv']. Reuses backtest_sd_calibration.build_year so the fit
population matches the backtest (projected IP >= P_IP_MIN, 2022-2025).

Read-only. Emits the SV_ROLE_MIXTURE dict literal to paste into constants.py, then
validate with `python scripts/backtest_sd_calibration.py` (the authoritative gate).
"""

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from scipy.stats import nbinom

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backtest_sd_calibration as bt

from fantasy_baseball.utils.constants import STAT_DISPERSION

R = float(STAT_DISPERSION["sv"])
EPS = 1e-9
K = 3  # components


def load_pairs():
    """(projected SV, realized SV) for the backtest pitcher population, 2022-2025."""
    s_all, y_all = [], []
    for year in bt.YEARS:
        _, pm, p_cats = bt.build_year(year)
        if pm is None or not any(acol == "SV" for acol, _ in p_cats):
            continue
        s = pm["SV_p"].to_numpy(dtype=float)
        y = pm["SV_a"].to_numpy(dtype=float)
        m = ~np.isnan(y)
        s_all.append(s[m])
        y_all.append(y[m])
    return np.concatenate(s_all), np.concatenate(y_all)


def _softmax(curves, s):
    logits = [b0 + b1 * s for b0, b1 in curves]
    logits.append(np.zeros_like(s))
    a = np.stack(logits, axis=-1)
    a = a - a.max(axis=-1, keepdims=True)
    e = np.exp(a)
    return e / e.sum(axis=-1, keepdims=True)


def _unpack(params):
    n = K - 1
    p_curves = [[params[2 * i], params[2 * i + 1]] for i in range(n)]
    w_curves = [[params[2 * (n + i)], params[2 * (n + i) + 1]] for i in range(n)]
    return p_curves, w_curves


def components(params, s):
    p_curves, w_curves = _unpack(params)
    p = _softmax(p_curves, s)
    w = _softmax(w_curves, s)
    return p, w / p


def nll(params, s, y):
    p, a = components(params, s)
    mu = np.maximum(s[:, None] * a, EPS)
    lp = nbinom.logpmf(y[:, None], R, R / (R + mu))
    return -float(np.sum(logsumexp(np.log(p) + lp, axis=1)))


def main():
    s, y = load_pairs()
    print(
        f"pitchers: {len(s)}  proj SV {s.min():.1f}-{s.max():.1f}  "
        f"realized 0-{y.max():.0f}  (mean proj {s.mean():.2f} vs realized {y.mean():.2f})"
    )
    rng = np.random.default_rng(0)
    best = None
    starts = [np.zeros(4 * (K - 1))] + [rng.normal(0, 1.0, 4 * (K - 1)) for _ in range(24)]
    for x0 in starts:
        res = minimize(
            nll,
            x0,
            args=(s, y),
            method="Nelder-Mead",
            options={"maxiter": 60000, "maxfev": 60000, "xatol": 1e-7, "fatol": 1e-7},
        )
        if best is None or res.fun < best.fun:
            best = res
    print(f"fit nll={best.fun:.1f} success={best.success}")

    print(f"\n{'s':>6}{'n':>6}   " + "  ".join(f"p{k} a{k}" for k in range(K)) + "   real_mu")
    edges = np.quantile(s, np.linspace(0, 1, 11))
    # also force a closer bucket so the thin high-s tail is visible
    for lo, hi in [*itertools.pairwise(edges), (edges[-1] * 0.0 + 10.0, 100.0)]:
        m = (s >= lo) & (s < hi)
        if m.sum() < 5:
            continue
        sm = float(s[m].mean())
        p, a = components(best.x, np.array([sm]))
        cells = "  ".join(f"{p[0, k]:.2f} {a[0, k]:.2f}" for k in range(K))
        print(f"{sm:6.1f}{int(m.sum()):6d}   {cells}   {y[m].mean():.2f}")

    p_curves, w_curves = _unpack(best.x)
    print("\nSV_ROLE_MIXTURE: dict[str, list[list[float]]] = {")
    print(f'    "p_logits": {[[round(float(b), 4) for b in c] for c in p_curves]},')
    print(f'    "w_logits": {[[round(float(b), 4) for b in c] for c in w_curves]},')
    print("}")


if __name__ == "__main__":
    main()
