# NegBin Counting-Stat Sampler Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the bias-inducing clipped-Gaussian performance multiplier in the Monte Carlo with a Gaussian-copula Negative-Binomial sampler whose dispersion is calibrated from 2022-2024 actuals.

**Architecture:** Phase 1 adds an offline calibration script that fits a per-stat NegBin dispersion `r` from projection-vs-actual residuals (conditional on realized playing time) and emits a `STAT_DISPERSION` constant, validated by leave-one-season-out interval coverage. Phase 2 rewrites only `_apply_variance` in `simulation.py` to draw correlated NegBin counts via a Gaussian copula, scaling variance by `fraction_remaining` through an effective dispersion `r_eff` with a Poisson-floor clamp. Both Monte Carlo callers are otherwise untouched.

**Tech Stack:** Python 3.12, numpy, pandas, scipy (`scipy.stats.nbinom`/`norm`/`poisson`, `scipy.optimize.minimize_scalar`), pytest.

**Source spec:** `docs/superpowers/specs/2026-06-14-negbin-stat-sampler-design.md`

---

## Background the engineer needs

- **Player stat keys** (`src/fantasy_baseball/utils/constants.py:118-119`):
  `HITTING_COUNTING = ["r","hr","rbi","sb","h","ab"]`,
  `PITCHING_COUNTING = ["w","k","sv","ip","er","bb","h_allowed"]`.
  The correlated subset (what the copula touches) excludes the playing-time-only
  `ab`/`ip`: `HITTER_CORR_STATS = ["r","hr","rbi","sb","h"]`,
  `PITCHER_CORR_STATS = ["w","k","sv","er","bb","h_allowed"]`
  (`constants.py:269,283`).
- **Correlation matrices** already exist as list-of-lists:
  `HITTER_CORRELATION` (`constants.py:270`), `PITCHER_CORRELATION` (`:284`).
- **NegBin mean/dispersion -> scipy:** for mean `mu` and dispersion `r`
  (`var = mu + mu^2/r`), `scipy.stats.nbinom(n=r, p=r/(r+mu))`. As `r -> inf` this
  is Poisson(`mu`); we represent the Poisson floor with the sentinel
  `r = float("inf")` and branch to `scipy.stats.poisson`.
- **Verified CSV mapping** (read all CSVs with `encoding="utf-8-sig"`, join on
  `MLBAMID`): hitter proj/actual keys = `R/HR/RBI/SB/H/AB/PA`; pitcher keys =
  `W/SO/SV/ER/BB/H/IP` where **`k`=`SO`** and **`h_allowed`=`H`**. Usable years:
  **2022, 2023, 2024** only (no 2025 projections; 2025 actuals are rate-style).
- **Reuse** the loader helpers in `scripts/calibrate_playing_time.py`
  (`_load:98`, `_find_proj:116`, `_blend:125`) -- same pattern, extended to
  counting columns.
- **`CLOSER_SV_THRESHOLD = 20`** (`constants.py:115`) for the saves role rule.

---

## Phase 1: Calibration script (additive -- no runtime behavior change)

New file: `scripts/calibrate_stat_dispersion.py`.
Tests: `tests/test_simulation/test_stat_dispersion_calibration.py`.

### Task 1.1: NegBin dispersion MLE fit (+ Poisson-floor clamp)

**Files:**
- Create: `scripts/calibrate_stat_dispersion.py`
- Test: `tests/test_simulation/test_stat_dispersion_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulation/test_stat_dispersion_calibration.py
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from calibrate_stat_dispersion import POISSON_SENTINEL, fit_dispersion


def test_fit_dispersion_recovers_known_r():
    rng = np.random.default_rng(0)
    mu = np.full(20_000, 12.0)
    true_r = 2.5
    p = true_r / (true_r + mu)
    x = rng.negative_binomial(true_r, p)
    r_hat = fit_dispersion(x, mu)
    assert abs(r_hat - true_r) < 0.3


def test_fit_dispersion_clamps_to_poisson_when_underdispersed():
    rng = np.random.default_rng(1)
    mu = np.full(20_000, 8.0)
    x = rng.poisson(mu)  # var == mean -> no overdispersion
    assert fit_dispersion(x, mu) == POISSON_SENTINEL
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'calibrate_stat_dispersion'`

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/calibrate_stat_dispersion.py
"""Calibrate per-stat Negative-Binomial dispersion from projection-vs-actual
residuals, conditional on realized playing time (2022-2024).

Mirrors scripts/calibrate_playing_time.py's data handling. Emits a
STAT_DISPERSION dict (per-stat r, with a Poisson sentinel) for paste into
src/fantasy_baseball/utils/constants.py, plus a leave-one-season-out
interval-coverage table that gates the shipped values. Performance dispersion
is measured conditional on realized PT so it does NOT double-count the
playing-time model's variance.

Usage:
    python scripts/calibrate_stat_dispersion.py
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import nbinom

# Sentinel for "no overdispersion -> use Poisson" (NegBin r -> inf).
POISSON_SENTINEL = float("inf")

# Bounds for the log-r search. exp(13.8) ~ 1e6: at the upper bound the NegBin is
# indistinguishable from Poisson, so we treat hitting it as the Poisson floor.
_LOG_R_LO = np.log(1e-3)
_LOG_R_HI = np.log(1e6)


def fit_dispersion(x: np.ndarray, mu: np.ndarray) -> float:
    """MLE of a single NegBin dispersion r for counts x with per-obs means mu.

    Each observation is x_i ~ NegBin(mean=mu_i, dispersion=r) with a shared r
    (heteroscedastic means, one dispersion). Returns POISSON_SENTINEL when the
    data is not overdispersed (the optimizer pins r at the upper bound).
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    mask = mu > 0
    x, mu = x[mask], mu[mask]

    def nll(log_r: float) -> float:
        r = np.exp(log_r)
        p = r / (r + mu)
        return -float(np.sum(nbinom.logpmf(x, r, p)))

    res = minimize_scalar(nll, bounds=(_LOG_R_LO, _LOG_R_HI), method="bounded")
    r_hat = float(np.exp(res.x))
    if r_hat >= 1e5:  # pinned high -> effectively Poisson
        return POISSON_SENTINEL
    return r_hat
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_stat_dispersion.py tests/test_simulation/test_stat_dispersion_calibration.py
git commit -m "feat(sim): NegBin dispersion MLE with Poisson-floor clamp"
```

### Task 1.2: Interval-coverage helper (the acceptance metric)

**Files:**
- Modify: `scripts/calibrate_stat_dispersion.py`
- Test: `tests/test_simulation/test_stat_dispersion_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_simulation/test_stat_dispersion_calibration.py
from calibrate_stat_dispersion import interval_coverage


def test_interval_coverage_matches_nominal_when_model_is_correct():
    rng = np.random.default_rng(2)
    mu = np.full(50_000, 15.0)
    r = 3.0
    p = r / (r + mu)
    x = rng.negative_binomial(r, p)
    cov80 = interval_coverage(x, mu, r, level=0.80)
    assert abs(cov80 - 0.80) < 0.03


def test_interval_coverage_handles_poisson_sentinel():
    rng = np.random.default_rng(3)
    mu = np.full(50_000, 6.0)
    x = rng.poisson(mu)
    cov50 = interval_coverage(x, mu, POISSON_SENTINEL, level=0.50)
    assert abs(cov50 - 0.50) < 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -k coverage -v`
Expected: FAIL with `ImportError: cannot import name 'interval_coverage'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/calibrate_stat_dispersion.py
from scipy.stats import poisson


def interval_coverage(x: np.ndarray, mu: np.ndarray, r: float, level: float) -> float:
    """Fraction of x inside the central `level` predictive interval of the model.

    r == POISSON_SENTINEL uses Poisson(mu); otherwise NegBin(mean=mu, disp=r).
    """
    x = np.asarray(x, dtype=float)
    mu = np.asarray(mu, dtype=float)
    lo_q, hi_q = (1.0 - level) / 2.0, (1.0 + level) / 2.0
    if r == POISSON_SENTINEL:
        lo = poisson.ppf(lo_q, mu)
        hi = poisson.ppf(hi_q, mu)
    else:
        p = r / (r + mu)
        lo = nbinom.ppf(lo_q, r, p)
        hi = nbinom.ppf(hi_q, r, p)
    return float(np.mean((x >= lo) & (x <= hi)))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -k coverage -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_stat_dispersion.py tests/test_simulation/test_stat_dispersion_calibration.py
git commit -m "feat(sim): NegBin predictive-interval coverage helper"
```

### Task 1.3: Build the conditional-on-PT residual table

**Files:**
- Modify: `scripts/calibrate_stat_dispersion.py`
- Test: `tests/test_simulation/test_stat_dispersion_calibration.py`

This loads projections + actuals per year, joins on `MLBAMID`, and produces, per
(stat, year), arrays of `actual_count` and the conditional mean
`mu = (proj_count / proj_PT) * actual_PT`. Population filter: include rows with
`actual_PT > 0` (the performance population; the PT-loss tail belongs to the
playing-time model). `proj_PT` is `PA` for hitters, `IP` for pitchers.

- [ ] **Step 1: Write the failing test** (uses a tiny synthetic CSV fixture)

```python
# append to tests/test_simulation/test_stat_dispersion_calibration.py
import pandas as pd

from calibrate_stat_dispersion import build_residuals


def _write_csv(path, rows, header):
    path.write_text(header + "\n" + "\n".join(rows) + "\n", encoding="utf-8")


def test_build_residuals_conditions_on_realized_pt(tmp_path, monkeypatch):
    # One hitter: proj 600 PA / 30 SB (rate 0.05/PA); realized 300 PA, 18 SB.
    # mu = 0.05 * 300 = 15.0; actual_count = 18.
    proj_dir = tmp_path / "projections" / "2022"
    proj_dir.mkdir(parents=True)
    stats_dir = tmp_path / "stats"
    stats_dir.mkdir()
    # Proj CSVs must carry ALL counting columns -- pandas merge only suffixes
    # overlapping columns, so a partial proj would leave R/HR/RBI/H unsuffixed
    # and build_residuals' m["R_proj"] lookup would KeyError.
    proj_header = "PA,R,HR,RBI,SB,H,MLBAMID"
    proj_row = "600,90,25,85,30,150,111"
    _write_csv(proj_dir / "steamer-hitters.csv", [proj_row], proj_header)
    _write_csv(proj_dir / "zips-hitters.csv", [proj_row], proj_header)
    h_header = "Name,Team,G,PA,AB,H,HR,R,RBI,SB,AVG,MLBAMID"
    _write_csv(stats_dir / "hitters-2022.csv",
               ["A,X,75,300,270,80,12,45,40,18,.296,111"], h_header)

    monkeypatch.setattr("calibrate_stat_dispersion.PROJ_DIR", tmp_path / "projections")
    monkeypatch.setattr("calibrate_stat_dispersion.STATS_DIR", stats_dir)
    monkeypatch.setattr("calibrate_stat_dispersion.YEARS", [2022])

    res = build_residuals("hitters")
    # All five correlated hitter keys must be produced (proves the loop covers
    # every key, not just sb).
    assert set(res) == {"r", "hr", "rbi", "sb", "h"}
    sb = res["sb"]
    assert sb["year"].tolist() == [2022]
    assert sb["actual"].tolist() == [18.0]
    assert abs(sb["mu"].iloc[0] - 15.0) < 1e-6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -k residuals -v`
Expected: FAIL with `ImportError: cannot import name 'build_residuals'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/calibrate_stat_dispersion.py
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

PROJ_DIR = PROJECT_ROOT / "data" / "projections"
STATS_DIR = PROJECT_ROOT / "data" / "stats"
YEARS = [2022, 2023, 2024]

# model stat key -> CSV column name (verified). k=SO, h_allowed=H.
HITTER_COLS = {"r": "R", "hr": "HR", "rbi": "RBI", "sb": "SB", "h": "H"}
PITCHER_COLS = {"w": "W", "k": "SO", "sv": "SV", "er": "ER", "bb": "BB", "h_allowed": "H"}


def _read(path: Path, cols: list[str]) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "MLBAMID" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["MLBAMID"]).copy()
    df["MLBAMID"] = df["MLBAMID"].astype(int)
    keep = ["MLBAMID", *[c for c in cols if c in df.columns]]
    return df[keep]


def _find_proj(year: int, system: str, kind: str) -> Path | None:
    d = PROJ_DIR / str(year)
    cand = d / f"{system}-{kind}.csv"
    if cand.exists():
        return cand
    matches = sorted(d.glob(f"{system}-{kind}*.csv"))
    return matches[0] if matches else None


def _blend_proj(year: int, kind: str, cols: list[str]) -> pd.DataFrame:
    s = _read(_find_proj(year, "steamer", kind) or Path("x"), cols)
    z = _read(_find_proj(year, "zips", kind) or Path("x"), cols)
    if s.empty or z.empty:
        return pd.DataFrame()
    m = s.merge(z, on="MLBAMID", suffixes=("_s", "_z"))
    out = pd.DataFrame({"MLBAMID": m["MLBAMID"]})
    for c in cols:
        cs, cz = f"{c}_s", f"{c}_z"
        if cs in m.columns and cz in m.columns:
            out[c] = (m[cs] + m[cz]) / 2.0
    return out


def build_residuals(kind: str) -> dict[str, pd.DataFrame]:
    """Per-stat DataFrame of {year, actual, mu} conditioned on realized PT.

    mu = (proj_count / proj_PT) * actual_PT. Rows with actual_PT <= 0 (the
    PT-loss tail owned by the playing-time model) are excluded.
    """
    is_hitter = kind == "hitters"
    pt = "PA" if is_hitter else "IP"
    colmap = HITTER_COLS if is_hitter else PITCHER_COLS
    proj_cols = [pt, *colmap.values()]
    actual_cols = [pt, *colmap.values()]

    per_stat: dict[str, list[pd.DataFrame]] = {k: [] for k in colmap}
    for year in YEARS:
        proj = _blend_proj(year, kind, proj_cols)
        actual = _read(STATS_DIR / f"{kind}-{year}.csv", actual_cols)
        if proj.empty or actual.empty:
            print(f"  {kind} {year}: projection or actuals missing, skipping")
            continue
        m = proj.merge(actual, on="MLBAMID", suffixes=("_proj", "_act"))
        m = m[m[f"{pt}_act"] > 0]
        for key, col in colmap.items():
            # Guard: a projection system may lack a column; merge only suffixes
            # overlapping names, so f"{col}_proj" can be absent. Skip rather than
            # KeyError (and real data has all columns, so this rarely triggers).
            if f"{col}_proj" not in m.columns or f"{col}_act" not in m.columns:
                continue
            proj_rate = m[f"{col}_proj"] / m[f"{pt}_proj"]
            mu = proj_rate * m[f"{pt}_act"]
            df = pd.DataFrame({"year": year, "actual": m[f"{col}_act"].astype(float),
                               "mu": mu.astype(float)})
            df = df[df["mu"] > 0]
            per_stat[key].append(df)
    return {k: pd.concat(v, ignore_index=True) if v else pd.DataFrame()
            for k, v in per_stat.items()}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -k residuals -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_stat_dispersion.py tests/test_simulation/test_stat_dispersion_calibration.py
git commit -m "feat(sim): build conditional-on-PT residual table for dispersion fit"
```

### Task 1.4: LOSO coverage + main(); emit STAT_DISPERSION

**Files:**
- Modify: `scripts/calibrate_stat_dispersion.py`
- Test: `tests/test_simulation/test_stat_dispersion_calibration.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_simulation/test_stat_dispersion_calibration.py
from calibrate_stat_dispersion import bucket_diagnostic, loso_coverage


def test_loso_coverage_returns_per_fold_table():
    rng = np.random.default_rng(4)
    frames = []
    for yr in (2022, 2023, 2024):
        mu = rng.uniform(5, 25, 4000)
        r = 3.0
        x = rng.negative_binomial(r, r / (r + mu))
        frames.append(pd.DataFrame({"year": yr, "actual": x.astype(float), "mu": mu}))
    data = pd.concat(frames, ignore_index=True)
    table = loso_coverage(data, levels=(0.50, 0.80))
    assert set(table["held_out"]) == {2022, 2023, 2024}
    assert abs(table["cov_0.80"].mean() - 0.80) < 0.05


def test_bucket_diagnostic_flags_scalar_r_as_adequate_when_true():
    # Data generated with a single r across all mu buckets -> the per-observation
    # Pearson statistic is ~1.0 in every bucket (immune to within-bucket mu spread).
    rng = np.random.default_rng(5)
    mu = rng.uniform(2, 40, 30_000)
    r = 3.0
    x = rng.negative_binomial(r, r / (r + mu))
    diag = bucket_diagnostic(pd.DataFrame({"actual": x.astype(float), "mu": mu}), r, n_bins=4)
    assert diag["pearson"].between(0.85, 1.15).all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -k loso -v`
Expected: FAIL with `ImportError: cannot import name 'loso_coverage'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/calibrate_stat_dispersion.py
def loso_coverage(data: pd.DataFrame, levels=(0.50, 0.80)) -> pd.DataFrame:
    """Leave-one-season-out coverage: fit r on the other years, test on the held-out."""
    rows = []
    years = sorted(data["year"].unique())
    for held in years:
        train = data[data["year"] != held]
        test = data[data["year"] == held]
        if train.empty or test.empty:
            continue
        r = fit_dispersion(train["actual"].to_numpy(), train["mu"].to_numpy())
        row = {"held_out": held, "r": r, "n": len(test)}
        for lv in levels:
            row[f"cov_{lv:.2f}"] = interval_coverage(
                test["actual"].to_numpy(), test["mu"].to_numpy(), r, lv
            )
        rows.append(row)
    return pd.DataFrame(rows)


def bucket_diagnostic(df: pd.DataFrame, r: float, n_bins: int = 4) -> pd.DataFrame:
    """Per projected-count bucket: Pearson dispersion of standardized residuals.

    For each row the implied conditional variance is mu + mu^2/r (mu for Poisson).
    The Pearson statistic mean((actual - mu)^2 / implied_var) is ~1.0 when the
    dispersion fits that bucket's mean range. It uses PER-OBSERVATION mu, so it is
    immune to the spread of mu WITHIN a bucket (a raw observed-vs-implied variance
    comparison would be inflated by that spread and falsely fail at low mu). A
    statistic systematically >1 at low mu and <1 at high mu (or vice versa) means
    a single scalar r does not fit and a mean-dependent dispersion is warranted.
    """
    d = df.copy()
    d["bin"] = pd.qcut(d["mu"], q=n_bins, labels=False, duplicates="drop")
    out = []
    for b, g in d.groupby("bin"):
        implied = g["mu"] if r == POISSON_SENTINEL else g["mu"] + g["mu"] ** 2 / r
        pearson = float((((g["actual"] - g["mu"]) ** 2) / implied).mean())
        out.append({"bin": int(b), "n": len(g), "mu_med": float(g["mu"].median()),
                    "pearson": pearson})
    return pd.DataFrame(out)


def main() -> None:
    from fantasy_baseball.utils.constants import (
        HITTER_CORR_STATS,
        PITCHER_CORR_STATS,
    )

    print("=" * 72)
    print("STAT DISPERSION CALIBRATION (NegBin r), years:", YEARS)
    dispersion: dict[str, float] = {}
    for kind, keys in (("hitters", HITTER_CORR_STATS), ("pitchers", PITCHER_CORR_STATS)):
        res = build_residuals(kind)
        for key in keys:
            df = res.get(key, pd.DataFrame())
            if df.empty:
                print(f"  {key}: no data, skipping")
                continue
            r = fit_dispersion(df["actual"].to_numpy(), df["mu"].to_numpy())
            dispersion[key] = r
            cov = loso_coverage(df)
            diag = bucket_diagnostic(df, r)
            r_disp = "Poisson" if r == POISSON_SENTINEL else f"{r:.3f}"
            print(f"\n  {key}: r={r_disp}  n={len(df)}")
            print(cov.to_string(index=False))
            print("  bucket diagnostic (observed vs implied variance):")
            print(diag.to_string(index=False))

    print("\nSTAT_DISPERSION = {")
    for k, v in dispersion.items():
        rv = 'float("inf")' if v == POISSON_SENTINEL else f"{v:.3f}"
        print(f'    "{k}": {rv},')
    print("}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -k "loso or bucket" -v`
Expected: PASS (both `loso` and `bucket_diagnostic` tests)

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_stat_dispersion.py tests/test_simulation/test_stat_dispersion_calibration.py
git commit -m "feat(sim): LOSO coverage + bucket diagnostic + calibration main"
```

### Task 1.5: Saves role-stable conditioning

**Files:**
- Modify: `scripts/calibrate_stat_dispersion.py`
- Test: `tests/test_simulation/test_stat_dispersion_calibration.py`

For `sv`, restrict the residual rows to role-stable closers: projected
`SV >= CLOSER_SV_THRESHOLD` AND realized `SV >= CLOSER_SV_THRESHOLD`. This needs
the pitcher merge to also carry `SV_proj`/`SV_act` (already present, since `sv` is
in `PITCHER_COLS`). Add a `_role_stable_mask` applied only to the `sv` stat, and
fit `sv` on the POOLED data (no LOSO gate -- informational coverage only).

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_simulation/test_stat_dispersion_calibration.py
from calibrate_stat_dispersion import CLOSER_SV_THRESHOLD, role_stable_sv


def test_role_stable_sv_keeps_only_stable_closers():
    df = pd.DataFrame({
        "year": [2022, 2022, 2022],
        "actual": [30.0, 2.0, 28.0],       # realized SV
        "mu": [29.0, 25.0, 27.0],          # proj-rate * realized IP (proxy proj SV)
        "proj_sv": [30.0, 28.0, 26.0],
        "actual_sv": [30.0, 2.0, 28.0],
    })
    out = role_stable_sv(df)
    # row 1 (proj 30 / act 30) and row 3 (proj 26 / act 28) are stable closers;
    # row 2 (proj 28 / act 2) lost the job -> excluded.
    assert out["actual"].tolist() == [30.0, 28.0]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -k role_stable -v`
Expected: FAIL with `ImportError: cannot import name 'role_stable_sv'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to scripts/calibrate_stat_dispersion.py; import the threshold near the top:
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD  # noqa: E402


def role_stable_sv(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only pitcher-seasons that were closers in BOTH projection and reality."""
    stable = (df["proj_sv"] >= CLOSER_SV_THRESHOLD) & (df["actual_sv"] >= CLOSER_SV_THRESHOLD)
    return df[stable].reset_index(drop=True)
```

Then in `build_residuals`, for the `sv` key only, attach `proj_sv`/`actual_sv`
columns to the per-row frame:

```python
            if key == "sv":
                df["proj_sv"] = m[f"{col}_proj"].astype(float)
                df["actual_sv"] = m[f"{col}_act"].astype(float)
```

And in `main`, after building pitcher residuals, replace the `sv` frame with
`role_stable_sv(res["sv"])` before fitting, and skip its LOSO gate (print
in-sample coverage instead):

```python
            if key == "sv":
                df = role_stable_sv(df)
                r = fit_dispersion(df["actual"].to_numpy(), df["mu"].to_numpy())
                dispersion["sv"] = r
                in_sample = {lv: interval_coverage(df["actual"].to_numpy(),
                             df["mu"].to_numpy(), r, lv) for lv in (0.50, 0.80)}
                print(f"\n  sv (role-stable, pooled): r="
                      f"{'Poisson' if r == POISSON_SENTINEL else f'{r:.3f}'} "
                      f"n={len(df)} in-sample cov={in_sample}")
                continue
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -k role_stable -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/calibrate_stat_dispersion.py tests/test_simulation/test_stat_dispersion_calibration.py
git commit -m "feat(sim): role-stable saves conditioning for dispersion fit"
```

### Task 1.6: Run calibration, add STAT_DISPERSION constant (alongside STAT_VARIANCE)

**Files:**
- Modify: `src/fantasy_baseball/utils/constants.py` (add `STAT_DISPERSION`)

- [ ] **Step 1: Run the calibration script and capture output**

Run: `python scripts/calibrate_stat_dispersion.py`
Expected: a printed `STAT_DISPERSION = {...}` block, per-stat LOSO coverage
tables, and per-stat bucket-diagnostic tables. **Acceptance gates:**
1. Coverage: for every stat except `sv`, the averaged held-out 50% and 80%
   coverage are within +/- 10 pp of nominal.
2. Bucket diagnostic: each stat's per-bucket Pearson statistic is roughly within
   [0.8, 1.2] across buckets. If a stat's statistic trends systematically (e.g.
   >1.2 at low mu, <0.8 at high mu), a single scalar r is inadequate -- escalate
   THAT stat to a mean-keyed dispersion (extend
   `STAT_DISPERSION[stat]` to a list of `{mu_threshold, r}` bands and have the
   sampler's `r_mat` fill look up the band by `mu`; the `_negbin_copula_counts`
   interface is unchanged since `r` is already per-element).
If either gate fails for a stat, do NOT hand-tune r -- record the failure and
STOP for human review (the model may be wrong for that stat).

- [ ] **Step 2: Add the emitted constant to `constants.py`**

Paste the emitted dict directly below `STAT_VARIANCE` (`constants.py:135-151`),
keeping `STAT_VARIANCE` in place (it is removed in Phase 2):

```python
# Per-stat Negative-Binomial dispersion r (var = mu + mu^2/r), calibrated from
# 2022-2024 projection-vs-actual residuals conditional on realized PT (see
# scripts/calibrate_stat_dispersion.py). float("inf") == Poisson floor (no
# overdispersion). sv is fit on the role-stable closer population.
STAT_DISPERSION: dict[str, float] = {
    # <paste emitted values>
}
```

- [ ] **Step 3: Verify the constant imports**

Run: `python -c "from fantasy_baseball.utils.constants import STAT_DISPERSION; print(STAT_DISPERSION)"`
Expected: prints the dict, no error.

- [ ] **Step 4: Lint + commit**

Run: `python -m ruff check scripts/calibrate_stat_dispersion.py src/fantasy_baseball/utils/constants.py && python -m ruff format --check scripts/calibrate_stat_dispersion.py`
Expected: clean.

```bash
git add src/fantasy_baseball/utils/constants.py
git commit -m "feat(sim): add calibrated STAT_DISPERSION constant (2022-2024)"
```

**Phase 1 verification:** `python -m pytest tests/test_simulation/test_stat_dispersion_calibration.py -v` (all pass), `python -m ruff check .` (clean), `python -m vulture` (no new findings). Runtime behavior unchanged (STAT_DISPERSION is not yet consumed).

---

## Phase 2: Sampler rewrite (internal to _apply_variance)

Tests: `tests/test_simulation/test_negbin_sampler.py` (new) + existing MC tests.

### Task 2.1: Correlation-matrix constants + copula draw helper

**Files:**
- Modify: `src/fantasy_baseball/simulation.py` (add helper + corr-matrix arrays)
- Test: `tests/test_simulation/test_negbin_sampler.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_simulation/test_negbin_sampler.py
import numpy as np

from fantasy_baseball.simulation import _negbin_copula_counts


def test_mean_is_unbiased_and_no_zero_spike():
    rng = np.random.default_rng(0)
    mu = np.full(200_000, 19.0)         # Cruz-like SB
    r = np.full(200_000, 2.2)
    z = rng.standard_normal(200_000)
    x = _negbin_copula_counts(mu, r, z, fraction_remaining=1.0)
    assert abs(x.mean() - 19.0) / 19.0 < 0.01          # no upward bias
    assert np.mean(x == 0) < 0.02                       # no ~9% zero-spike
    assert np.all(x >= 0)
    assert np.all(x == np.floor(x))                     # integer draws


def test_finite_under_extreme_latent():
    mu = np.array([19.0, 4.0])
    r = np.array([2.2, 3.8])
    z = np.array([40.0, -40.0])        # Phi(z) -> 1 / 0
    x = _negbin_copula_counts(mu, r, z, fraction_remaining=1.0)
    assert np.all(np.isfinite(x))


def test_fraction_remaining_scales_variance_linearly():
    rng = np.random.default_rng(1)
    mu = np.full(300_000, 30.0)
    r = np.full(300_000, 2.2)
    z = rng.standard_normal(300_000)
    var_full = (_negbin_copula_counts(mu, r, z, 1.0)).var()
    var_half = (_negbin_copula_counts(mu, r, z, 0.5)).var()
    assert abs(var_half / var_full - 0.5) < 0.06       # supra-floor regime


def test_poisson_sentinel_draws_poisson():
    rng = np.random.default_rng(2)
    mu = np.full(100_000, 8.0)
    r = np.full(100_000, np.inf)
    z = rng.standard_normal(100_000)
    x = _negbin_copula_counts(mu, r, z, fraction_remaining=1.0)
    assert abs(x.var() / x.mean() - 1.0) < 0.05        # Poisson: var == mean
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_simulation/test_negbin_sampler.py -v`
Expected: FAIL with `ImportError: cannot import name '_negbin_copula_counts'`

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/fantasy_baseball/simulation.py (imports at top)
from scipy.stats import nbinom, norm, poisson

# add near HITTER_COV/PITCHER_COV (correlation matrices as unit-variance arrays
# for the copula latent; replaces the sigma-scaled covariance):
HITTER_CORR_MATRIX = np.array(HITTER_CORRELATION)
PITCHER_CORR_MATRIX = np.array(PITCHER_CORRELATION)

_U_EPS = 1e-9


def _negbin_copula_counts(
    mu: np.ndarray,
    r: np.ndarray,
    z: np.ndarray,
    fraction_remaining: float,
) -> np.ndarray:
    """Map correlated standard-normal latents z to NegBin counts via a copula.

    mu: per-element mean (base * scale). r: per-element calibrated dispersion
    (np.inf == Poisson floor). Variance is scaled by fraction_remaining through
    an effective dispersion r_eff; an element whose target variance falls to/below
    its mean (the Poisson floor) is drawn Poisson. u is clamped to [eps, 1-eps] so
    nbinom/poisson ppf never returns inf.
    """
    mu = np.asarray(mu, dtype=float)
    r = np.asarray(r, dtype=float)
    u = np.clip(norm.cdf(z), _U_EPS, 1.0 - _U_EPS)

    out = np.zeros_like(mu)
    pos = mu > 0
    if not np.any(pos):
        return out

    mu_p = mu[pos]
    r_p = r[pos]
    u_p = u[pos]

    # var_full = mu + mu^2/r (r=inf -> var_full=mu, the Poisson case)
    with np.errstate(divide="ignore"):
        var_full = mu_p + np.where(np.isinf(r_p), 0.0, mu_p**2 / r_p)
    var_target = fraction_remaining * var_full

    # supra-floor: var_target > mu -> NegBin with r_eff; else Poisson(mu)
    supra = var_target > mu_p
    res = np.empty_like(mu_p)

    if np.any(supra):
        r_eff = mu_p[supra] ** 2 / (var_target[supra] - mu_p[supra])
        p_eff = r_eff / (r_eff + mu_p[supra])
        res[supra] = nbinom.ppf(u_p[supra], r_eff, p_eff)
    if np.any(~supra):
        res[~supra] = poisson.ppf(u_p[~supra], mu_p[~supra])

    out[pos] = res
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_simulation/test_negbin_sampler.py -v`
Expected: PASS (all four)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/simulation.py tests/test_simulation/test_negbin_sampler.py
git commit -m "feat(sim): Gaussian-copula NegBin count sampler with r_eff scaling"
```

### Task 2.2: Correlation-preservation test

**Files:**
- Test: `tests/test_simulation/test_negbin_sampler.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_simulation/test_negbin_sampler.py
from fantasy_baseball.simulation import PITCHER_CORR_MATRIX
from fantasy_baseball.utils.constants import PITCHER_CORR_STATS


def test_copula_preserves_sign_of_correlation():
    rng = np.random.default_rng(7)
    n = 100_000
    z = rng.multivariate_normal(np.zeros(len(PITCHER_CORR_STATS)), PITCHER_CORR_MATRIX, size=n)
    i_sv = PITCHER_CORR_STATS.index("sv")
    i_er = PITCHER_CORR_STATS.index("er")
    mu_sv = np.full(n, 25.0); r_sv = np.full(n, 2.0)
    mu_er = np.full(n, 65.0); r_er = np.full(n, 4.0)
    sv = _negbin_copula_counts(mu_sv, r_sv, z[:, i_sv], 1.0)
    er = _negbin_copula_counts(mu_er, r_er, z[:, i_er], 1.0)
    # PITCHER_CORRELATION has sv vs er at -0.341 -> realized correlation negative.
    assert np.corrcoef(sv, er)[0, 1] < -0.1
```

- [ ] **Step 2: Run test to verify it fails (or passes immediately)**

Run: `python -m pytest tests/test_simulation/test_negbin_sampler.py -k correlation -v`
Expected: PASS if `PITCHER_CORR_MATRIX` already exported in Task 2.1; if the
import fails, export it. (This test pins behavior; no new product code if it
already passes.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_simulation/test_negbin_sampler.py
git commit -m "test(sim): copula preserves inter-stat correlation sign"
```

### Task 2.3: Rewrite `_apply_variance` to use the copula sampler

**Files:**
- Modify: `src/fantasy_baseball/simulation.py:474-541` (`_apply_variance`)
- Modify: `src/fantasy_baseball/simulation.py` (remove `_build_cov_matrix`,
  `HITTER_COV`, `PITCHER_COV` usage)

- [ ] **Step 1: Read the current function**

Run: open `src/fantasy_baseball/simulation.py:474-541` and confirm the loop shape
(it currently does `perf = max(0, 1.0 + draws[idx_map[col]])`).

- [ ] **Step 2: Replace the draw + per-stat mapping**

Replace the covariance draw and per-column loop body. New body (keep the
`scales`, `frac_missed`, injuries, and `repl` logic verbatim):

```python
    is_hitter = player_type == PlayerType.HITTER
    counting_cols = HITTING_COUNTING if is_hitter else PITCHING_COUNTING
    corr_matrix = HITTER_CORR_MATRIX if is_hitter else PITCHER_CORR_MATRIX
    idx_map = HITTER_IDX if is_hitter else PITCHER_IDX
    disp = STAT_DISPERSION  # stat -> r (np.inf == Poisson)
    n_corr = len(idx_map)
    mean = np.zeros(n_corr)

    n = len(players)
    if n == 0:
        return []

    scales = _playing_time_scales(players, player_type, rng, fraction_remaining)
    # Unit-variance correlated latents (the copula's Gaussian layer).
    all_z = rng.multivariate_normal(mean, corr_matrix, size=n)

    # Build per-(player, corr-stat) mu and r arrays, then map through the copula.
    corr_keys = [c for c in counting_cols if c in idx_map]
    mu_mat = np.zeros((n, n_corr))
    r_mat = np.full((n, n_corr), np.inf)
    for i, p in enumerate(players):
        scale = float(scales[i])
        for col in corr_keys:
            j = idx_map[col]
            mu_mat[i, j] = float(p.get(col, 0) or 0) * scale
            r_mat[i, j] = disp.get(col, np.inf)
    counts = np.empty((n, n_corr))
    for j in range(n_corr):
        counts[:, j] = _negbin_copula_counts(
            mu_mat[:, j], r_mat[:, j], all_z[:, j], fraction_remaining
        )

    adjusted = []
    for i, p in enumerate(players):
        scale = float(scales[i])
        frac_missed = max(0.0, 1.0 - scale)
        if frac_missed >= _NOTABLE_PT_LOSS:
            injuries_out.append((p.get("name", "?"), frac_missed))
        repl = _replacement_line(p, is_hitter)
        row = {}
        for col in counting_cols:
            repl_contrib = repl.get(col, 0) * frac_missed
            if col in idx_map:
                row[col] = counts[i, idx_map[col]] + repl_contrib
            else:
                row[col] = float(p.get(col, 0) or 0) * scale + repl_contrib
        row["name"] = p.get("name", "?")
        row["player_type"] = player_type
        adjusted.append(row)
    return adjusted
```

Update the imports/constants at the top of `simulation.py`: import
`STAT_DISPERSION` from constants; remove the `STAT_VARIANCE` import and the
`HITTER_COV`/`PITCHER_COV`/`_build_cov_matrix` definitions (lines ~46-66).

- [ ] **Step 3: Run the sampler + structural MC tests**

Run: `python -m pytest tests/test_simulation/ -v`
Expected: sampler tests PASS; some value-pinned MC golden tests may FAIL (handled
in Task 2.4). Structural tests (no-crash, keys present) PASS.

- [ ] **Step 4: Remove STAT_VARIANCE from constants**

Delete `STAT_VARIANCE` (`constants.py:135-151`) now that nothing reads it.
Run: `python -m pytest -q --ignore=tests/test_draft/test_simulate_draft.py tests/ 2>&1 | tail -5`
and `python -c "import fantasy_baseball.simulation"` (Expected: imports clean,
no `STAT_VARIANCE` reference errors).

- [ ] **Step 5: Add a caller-unchanged guard test**

This proves the in-season caller's team-level subtraction/recombination still
holds after the sampler rewrite (the spec's "callers unchanged" obligation).

```python
# append to tests/test_simulation/test_negbin_sampler.py
# (np already imported at the top of this file)
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.simulation import simulate_remaining_season


def test_simulate_remaining_preserves_actuals_floor_and_finite_rates():
    rng = np.random.default_rng(11)
    roster = {
        "T1": [
            {"player_type": PlayerType.HITTER, "name": "H1", "pa": 600, "ab": 540,
             "r": 90, "hr": 25, "rbi": 85, "sb": 15, "h": 150},
            {"player_type": PlayerType.PITCHER, "name": "P1", "ip": 180, "w": 12,
             "k": 200, "sv": 0, "er": 65, "bb": 45, "h_allowed": 150,
             "positions": ["SP"]},
        ]
    }
    actuals = {"T1": {"R": 40, "HR": 12, "RBI": 38, "SB": 7, "AVG": .270,
                      "W": 6, "K": 95, "SV": 1, "ERA": 3.50, "WHIP": 1.15,
                      "AB": 250, "IP": 85}}
    stats, _ = simulate_remaining_season(actuals, roster, 0.5, rng)
    s = stats["T1"]
    # final counting totals never drop below banked actuals; rates finite/sane.
    for cat in ("R", "HR", "RBI", "SB", "W", "K", "SV"):
        assert s[cat] >= actuals["T1"][cat]
    assert 0 < s["AVG"] < 1 and 0 < s["ERA"] < 30 and 0 < s["WHIP"] < 5
```

Run: `python -m pytest tests/test_simulation/test_negbin_sampler.py -k remaining -v`
Expected: PASS (the caller logic is untouched, so this guards against an
accidental regression to its subtraction/recombination).

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/simulation.py src/fantasy_baseball/utils/constants.py tests/test_simulation/test_negbin_sampler.py
git commit -m "feat(sim): rewrite _apply_variance to copula NegBin; drop STAT_VARIANCE"
```

### Task 2.4: Re-bless value-pinned MC golden tests

**Files:**
- Modify: whichever tests pin specific MC output values (find them in Step 1)

- [ ] **Step 1: Find value-pinned MC tests that now fail**

Run: `python -m pytest tests/test_simulation/ tests/test_draft/ -q -p no:cacheprovider --ignore=tests/test_draft/test_simulate_draft.py 2>&1 | tail -30`
Identify failures that assert specific simulated stat totals/standings values.

- [ ] **Step 2: For each, confirm the change is explained by the model switch**

For each failing assertion, verify the new value is consistent with the
NegBin/copula change (means roughly preserved, variance/zero-spike changed) and
NOT a structural break (missing keys, crashes, wrong roster). Document the reason
in the commit. Do NOT loosen assertions to hide a structural break -- if a
failure is not explained by the model switch, STOP and investigate.

- [ ] **Step 3: Re-pin the expected values**

Update each value-pinned expected number to the new model's output. Keep
tolerances as tight as before (re-pin the number; don't widen the band).

- [ ] **Step 4: Run the suites green**

Run: `python -m pytest tests/test_simulation/ tests/test_draft/ tests/test_web/ -n auto -q -p no:cacheprovider --ignore=tests/test_draft/test_simulate_draft.py`
Expected: all pass. (Note: `test_simulate_draft.py` has a PRE-EXISTING native
pandas segfault unrelated to this change -- run it isolated to confirm it still
behaves as before; do not attribute its crash to this work.)

- [ ] **Step 5: Commit**

```bash
git add tests/
git commit -m "test(sim): re-bless MC goldens for NegBin/copula sampler"
```

### Task 2.5: Performance benchmark

**Files:**
- (No source change unless a regression is found)

- [ ] **Step 1: Benchmark draft + dashboard MC before/after**

Run a representative MC (e.g. `python scripts/simulate_draft.py -s immediate
--scoring-mode vona --monte-carlo` or the dashboard refresh path) and time it,
comparing against the pre-change commit. Record wall-clock.

- [ ] **Step 2: Check against budgets**

Acceptance: dashboard MC refresh within 1.25x baseline AND under the existing
free-tier budget; draft MC within 1.5x baseline. If exceeded, optimize the hot
path (the per-stat `nbinom.ppf` loop in `_negbin_copula_counts`): precompute
masks once and avoid recomputing `norm.cdf` redundantly; consider drawing all
stats' counts in fewer ppf calls.

- [ ] **Step 3: Record numbers**

Note the before/after wall-clock in the final PR description.

**Phase 2 verification (FORCED CHECKLIST):**
- `python -m pytest tests/test_simulation/ tests/test_draft/ tests/test_web/ -n auto -q --ignore=tests/test_draft/test_simulate_draft.py` -- all pass
- `python -m ruff check .` -- zero violations
- `python -m ruff format --check .` -- clean
- `python -m vulture` -- no NEW findings
- `python -m mypy` IF `simulation.py`/`constants.py` are in `[tool.mypy].files` -- check the list first

---

## Self-review notes (author)

- **Spec coverage:** calibration (Tasks 1.1-1.6) <- spec Architecture #1 +
  calibration methodology; copula sampler + r_eff + Poisson floor + tail clamp
  (Task 2.1) <- spec Architecture #2 + Math appendix + fraction_remaining; corr
  preservation (Task 2.2) <- spec decision #2; constant swap additive-then-remove
  (1.6, 2.3-2.4) <- spec Architecture #3 + phasing; out-of-sample coverage (1.4,
  1.6 gate) <- spec Testing acceptance; saves role-stable (1.5) <- spec saves
  decision; regression re-bless (2.4) <- spec regression policy; perf (2.5) <-
  spec Performance budgets.
- **Type consistency:** `_negbin_copula_counts(mu, r, z, fraction_remaining)`,
  `fit_dispersion(x, mu)`, `interval_coverage(x, mu, r, level)`,
  `build_residuals(kind) -> dict[str, DataFrame{year,actual,mu}]`,
  `loso_coverage(data, levels)`, `role_stable_sv(df)` -- names match across tasks.
- **Known caveat carried from spec:** `test_simulate_draft.py` pre-existing
  segfault is out of scope; do not let it block the suite (it is `--ignore`d in
  the runner above and tracked separately).

---

## Phase 1b addendum: Banded (mean-dependent) dispersion for SB/SV

**Why (measured, not hypothetical):** Running `main()` (Task 1.4) on real
2022-2024 data, the bucket diagnostic returned Pearson ~1.0 for R/HR/RBI/H/W/K/
BB/H-allowed (scalar `r` fits) but **~6-9 at low projected counts for SB and SV**
(strong low-`mu` under-dispersion). SV's is largely the closer job-change tail
(addressed by Task 1.5 role-stable conditioning -- re-check after); SB's is
genuine. Decision (user-approved): ship **banded** dispersion for SB (and SV if
still flagged after role-stable conditioning); all other stats stay scalar.

Representation: a `STAT_DISPERSION` value is EITHER a scalar `float` OR a list of
`(mu_upper, r)` bands sorted ascending, last `mu_upper == float("inf")`. One
shared resolver maps a value + projected means to a per-element `r`; BOTH the
calibration coverage/diagnostic and the runtime sampler call it.

### Task 1.5b: shared dispersion resolver

**Files:**
- Create: `src/fantasy_baseball/utils/dispersion.py`
- Test: `tests/test_simulation/test_dispersion_resolver.py`

- [ ] **Step 1: Write failing test**

```python
import numpy as np

from fantasy_baseball.utils.dispersion import resolve_dispersion_r


def test_scalar_broadcasts():
    out = resolve_dispersion_r(3.0, np.array([1.0, 50.0, 200.0]))
    assert np.allclose(out, [3.0, 3.0, 3.0])


def test_inf_scalar_passes_through():
    out = resolve_dispersion_r(float("inf"), np.array([5.0, 20.0]))
    assert np.all(np.isinf(out))


def test_banded_lookup_picks_first_band_with_upper_ge_mu():
    bands = [(5.0, 0.6), (12.0, 1.5), (float("inf"), 3.5)]
    out = resolve_dispersion_r(bands, np.array([3.0, 5.0, 5.1, 12.0, 100.0]))
    # mu<=5 -> 0.6; 5<mu<=12 -> 1.5; mu>12 -> 3.5
    assert np.allclose(out, [0.6, 0.6, 1.5, 1.5, 3.5])
```

- [ ] **Step 2: Run, confirm ImportError fail.**
Run: `python -m pytest tests/test_simulation/test_dispersion_resolver.py -v`

- [ ] **Step 3: Implement**

```python
"""Resolve a STAT_DISPERSION value (scalar r or (mu_upper, r) bands) to per-element r."""

from __future__ import annotations

import numpy as np


def resolve_dispersion_r(value: float | list[tuple[float, float]], mu) -> np.ndarray:
    """Per-element NegBin dispersion r for projected means mu.

    value is a scalar float (one r for all mu) or a list of (mu_upper, r) bands
    sorted ascending with the final mu_upper == inf. Each element takes the r of
    the first band whose mu_upper >= its mu (np.searchsorted, side="left").
    float("inf") as an r marks the Poisson floor and passes through unchanged.
    """
    mu = np.asarray(mu, dtype=float)
    if isinstance(value, (int, float)):
        return np.full(mu.shape, float(value))
    bounds = np.array([b for b, _ in value], dtype=float)
    rs = np.array([r for _, r in value], dtype=float)
    idx = np.clip(np.searchsorted(bounds, mu, side="left"), 0, len(rs) - 1)
    return rs[idx]
```

- [ ] **Step 4: Run, confirm 3 passed.**
- [ ] **Step 5: Commit** `src/fantasy_baseball/utils/dispersion.py` + the test:
"feat(sim): shared scalar/banded dispersion resolver"

### Task 1.5c: banded fit + per-element bucket diagnostic + main integration

**Files:**
- Modify: `scripts/calibrate_stat_dispersion.py`
- Test: `tests/test_simulation/test_stat_dispersion_calibration.py`

- [ ] **Step 1: Write failing tests** (append)

```python
from calibrate_stat_dispersion import fit_banded_dispersion


def test_fit_banded_dispersion_lowers_pearson_for_mean_dependent_stat():
    # Construct mean-dependent overdispersion: low mu -> small r (heavy spread),
    # high mu -> large r (tight). A single scalar r cannot fit both; bands can.
    rng = np.random.default_rng(9)
    lo_mu = rng.uniform(1, 5, 8000)
    hi_mu = rng.uniform(20, 40, 8000)
    lo = rng.negative_binomial(0.5, 0.5 / (0.5 + lo_mu))
    hi = rng.negative_binomial(8.0, 8.0 / (8.0 + hi_mu))
    df = pd.DataFrame({
        "actual": np.concatenate([lo, hi]).astype(float),
        "mu": np.concatenate([lo_mu, hi_mu]),
    })
    bands = fit_banded_dispersion(df, n_bands=4)
    assert bands[-1][0] == float("inf")
    # banded Pearson within tolerance across buckets; scalar would not be
    from calibrate_stat_dispersion import bucket_diagnostic
    from fantasy_baseball.utils.dispersion import resolve_dispersion_r
    r_elem = resolve_dispersion_r(bands, df["mu"].to_numpy())
    diag = bucket_diagnostic(df, r_elem, n_bins=4)
    assert diag["pearson"].between(0.6, 1.6).all()
```

- [ ] **Step 2: Run, confirm fail.**

- [ ] **Step 3: Implement** -- add `fit_banded_dispersion`, generalize
`bucket_diagnostic` to accept a per-element `r` array (scalar still broadcasts),
and make `main()` choose scalar-vs-banded per stat by the diagnostic gate.

```python
def fit_banded_dispersion(df, n_bands: int = 4):
    """Fit a separate r per qcut(mu) band; return [(mu_upper, r), ...] last=inf."""
    d = df.copy()
    d["bin"] = pd.qcut(d["mu"], q=n_bands, labels=False, duplicates="drop")
    bins = sorted(d["bin"].unique())
    bands = []
    for i, b in enumerate(bins):
        g = d[d["bin"] == b]
        r = fit_dispersion(g["actual"].to_numpy(), g["mu"].to_numpy())
        upper = float("inf") if i == len(bins) - 1 else float(g["mu"].max())
        bands.append((upper, r))
    return bands
```

Generalize `bucket_diagnostic` so `r` may be a scalar OR a per-element array
(aligned to df rows); compute implied variance per row, treating inf as Poisson:

```python
def bucket_diagnostic(df, r, n_bins: int = 4):
    d = df.copy()
    d["r_elem"] = r  # scalar broadcasts; array aligns positionally
    d["bin"] = pd.qcut(d["mu"], q=n_bins, labels=False, duplicates="drop")
    out = []
    for b, g in d.groupby("bin"):
        implied = np.where(
            np.isinf(g["r_elem"]), g["mu"], g["mu"] + g["mu"] ** 2 / g["r_elem"]
        )
        pearson = float((((g["actual"] - g["mu"]) ** 2) / implied).mean())
        out.append({"bin": int(b), "n": len(g), "mu_med": float(g["mu"].median()),
                    "pearson": pearson})
    return pd.DataFrame(out)
```

In `main()`, for each stat: fit scalar `r`, compute scalar bucket diagnostic; if
any bucket Pearson is outside [0.75, 1.35] (systematic misfit), refit banded via
`fit_banded_dispersion`, recompute the diagnostic with the resolved per-element
`r` (import `resolve_dispersion_r`), and emit the band list instead of the
scalar. Print both the chosen kind and the post-fix diagnostic.

- [ ] **Step 4: Run** the new + existing calibration tests (scalar
bucket_diagnostic test still passes because scalar broadcasts).
- [ ] **Step 5: Commit** the two files:
"feat(sim): banded dispersion fit + per-element bucket diagnostic"

### Adjustment to Task 1.6 (emit mixed STAT_DISPERSION)

`STAT_DISPERSION` now has type `dict[str, float | list[tuple[float, float]]]`.
Paste the emitted block verbatim (scalar stats as floats, SB/SV as band lists,
e.g. `"sb": [(4.0, 0.6), (10.0, 1.4), (float("inf"), 3.1)]`). The acceptance gate
adds: each banded stat's post-fix per-bucket Pearson is within ~[0.7, 1.3]. Keep
the gate's spirit one-sided on UNDER-coverage (Pearson >> 1 is the dangerous
direction); benign over-coverage (Pearson < 1 from discreteness) does not block.

### Adjustment to Task 2.3 (sampler r-fill uses the resolver)

In the rewritten `_apply_variance`, fill `r_mat` per correlated stat via the
shared resolver instead of a scalar dict lookup:

```python
from fantasy_baseball.utils.dispersion import resolve_dispersion_r
...
    for col in corr_keys:
        j = idx_map[col]
        mu_col = mu_mat[:, j]
        r_mat[:, j] = resolve_dispersion_r(STAT_DISPERSION[col], mu_col)
```
(`mu_mat[:, j]` is `base*scale` per player for that stat -- the projected mean
used both as the NegBin mean and as the band key.) `_negbin_copula_counts` is
unchanged (it already takes a per-element `r` array). Add a property test that a
banded `r` restores per-bucket Pearson ~1.0 through the full sampler (i.e. the
escalation actually fixes the low-`mu` under-dispersion end to end).
