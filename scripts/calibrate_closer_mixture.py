"""Calibrate the closer role-switch mixture curves (SV variance) from realized data.

Fits two logistic curves -- the modal probability q(s) and the surprise mean-share
f(s) -- by maximum likelihood of realized SV under the mean-1 mixture
  SV ~ q * NegBin(s*a_m, r) + (1-q) * NegBin(s*a_s, r),  a_s=f/(1-q), a_m=(1-f)/q,
with r fixed at STAT_DISPERSION['sv']. Reuses backtest_sd_calibration.build_year so
the fit population matches the backtest (projected IP >= P_IP_MIN, 2022-2025).

Read-only. Emits the SV_ROLE_MIXTURE dict literal to paste into constants.py, then
validate with `python scripts/backtest_sd_calibration.py` (the authoritative gate).
"""

import itertools
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import backtest_sd_calibration as bt

from fantasy_baseball.utils.constants import STAT_DISPERSION

R = float(STAT_DISPERSION["sv"])
EPS = 1e-9


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


def components(params, s):
    b0, b1, g0, g1 = params
    q = np.clip(1.0 / (1.0 + np.exp(-(b0 + b1 * s))), 1e-3, 1 - 1e-3)
    f = np.clip(1.0 / (1.0 + np.exp(-(g0 + g1 * s))), 1e-9, 1 - 1e-6)
    a_s = f / (1.0 - q)
    a_m = (1.0 - f) / q
    return q, a_m, a_s


def nll(params, s, y):
    q, a_m, a_s = components(params, s)
    mu_m = np.maximum(s * a_m, EPS)
    mu_s = np.maximum(s * a_s, EPS)
    lp_m = nbinom.logpmf(y, R, R / (R + mu_m))
    lp_s = nbinom.logpmf(y, R, R / (R + mu_s))
    ll = np.logaddexp(np.log(q) + lp_m, np.log(1.0 - q) + lp_s)
    return -float(np.sum(ll))


def main():
    s, y = load_pairs()
    print(
        f"pitchers: {len(s)}  proj SV {s.min():.1f}-{s.max():.1f}  "
        f"realized 0-{y.max():.0f}  (mean proj {s.mean():.2f} vs realized {y.mean():.2f})"
    )
    best = None
    for x0 in [
        np.array([0.2, 0.05, 1.0, -0.1]),
        np.array([3.0, -0.05, 0.5, -0.05]),
        np.array([1.0, 0.0, 0.0, -0.1]),
    ]:
        res = minimize(
            nll,
            x0,
            args=(s, y),
            method="Nelder-Mead",
            options={"maxiter": 40000, "xatol": 1e-7, "fatol": 1e-7},
        )
        if best is None or res.fun < best.fun:
            best = res
    b0, b1, g0, g1 = best.x
    print(f"fit nll={best.fun:.1f} success={best.success}")
    print(f"\n{'s':>6}{'n':>6}{'q':>8}{'a_m':>8}{'a_s':>8}{'model_mu':>10}{'real_mu':>9}")
    edges = np.quantile(s, np.linspace(0, 1, 11))
    for lo, hi in itertools.pairwise(edges):
        m = (s >= lo) & ((s < hi) if hi < edges[-1] else (s <= hi))
        if m.sum() == 0:
            continue
        sm = float(s[m].mean())
        q, a_m, a_s = components(best.x, np.array([sm]))
        model_mu = sm * (q[0] * a_m[0] + (1 - q[0]) * a_s[0])
        print(
            f"{sm:6.1f}{int(m.sum()):6d}{q[0]:8.3f}{a_m[0]:8.3f}{a_s[0]:8.3f}"
            f"{model_mu:10.2f}{y[m].mean():9.2f}"
        )
    print("\nSV_ROLE_MIXTURE: dict[str, list[float]] = {")
    print(f'    "q_logit": [{b0:.4f}, {b1:.4f}],')
    print(f'    "f_logit": [{g0:.4f}, {g1:.4f}],')
    print("}")


if __name__ == "__main__":
    main()
