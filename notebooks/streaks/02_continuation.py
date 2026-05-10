# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Hot Streaks Phase 3 — Continuation Analysis
#
# This notebook is the Phase 3 acceptance artifact. It loads
# `continuation_rates` (and the underlying labels) and walks through
# the **go/no-go gate**:
#
# **Pass criteria (committed in the Phase 3 plan):**
# 1. ≥1 cell shows ≥5pp lift over base rate with N ≥ 1000.
# 2. ≥3 of 5 categories show ≥2pp directional lift in the 7-day window.
#
# If neither passes after stratification: write up the no-go finding
# and pause Phase 4.
#
# **Run after:** `python scripts/streaks/run_continuation.py --season-set 2023-2025`

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

_here = Path.cwd()
for _candidate in (_here, *_here.parents):
    if (_candidate / "pyproject.toml").exists():
        REPO_ROOT = _candidate
        break
else:
    raise RuntimeError("could not locate repo root")

DB = REPO_ROOT / "data" / "streaks" / "streaks.duckdb"
SEASON_SET = "2023-2025"

conn = duckdb.connect(str(DB))

# %% [markdown]
# ## Cell sizes by stratum
#
# How big are the (cat × win × bucket × strength × direction) cells we're
# measuring lift on? Anything < 1000 is too small to draw confident
# conclusions from in isolation; we eyeball the pattern across cells.

# %%
cell_sizes = conn.execute(
    """
    SELECT category, window_days, pt_bucket, cold_method,
           SUM(n_labeled) AS total_labeled,
           COUNT(*) AS n_strata
    FROM continuation_rates
    WHERE season_set = ?
    GROUP BY category, window_days, pt_bucket, cold_method
    ORDER BY category, window_days, pt_bucket, cold_method
    """,
    [SEASON_SET],
).df()
cell_sizes

# %% [markdown]
# ## Headline lift table
#
# Maximum lift per (category × window_days × cold_method) cell, restricted
# to strata with N ≥ 1000.

# %%
headline = conn.execute(
    """
    SELECT category, window_days, cold_method,
           MAX(lift) AS max_lift,
           ARG_MAX(strength_bucket || ' (' || pt_bucket || ', ' || direction || ')',
                   lift) AS where_max,
           ARG_MAX(n_labeled, lift) AS n_at_max
    FROM continuation_rates
    WHERE season_set = ? AND n_labeled >= 1000
    GROUP BY category, window_days, cold_method
    ORDER BY category, window_days, cold_method
    """,
    [SEASON_SET],
).df()
headline

# %% [markdown]
# ## Go/no-go assessment
#
# **Test 1:** at least one cell with N ≥ 1000 and lift ≥ 5pp.

# %%
test1 = conn.execute(
    """
    SELECT COUNT(*) AS n_cells_passing
    FROM continuation_rates
    WHERE season_set = ? AND n_labeled >= 1000 AND lift >= 0.05
    """,
    [SEASON_SET],
).fetchone()[0]
print(f"Test 1: {test1} cells with N>=1000 and lift>=5pp.  {'PASS' if test1 >= 1 else 'FAIL'}")

# %% [markdown]
# **Test 2:** in the 7-day window, ≥3 of 5 categories show some directional
# lift ≥ 2pp at any strength/bucket/cold_method.

# %%
test2 = conn.execute(
    """
    SELECT COUNT(DISTINCT category) AS cats_with_lift
    FROM continuation_rates
    WHERE season_set = ? AND window_days = 7 AND lift >= 0.02
    """,
    [SEASON_SET],
).fetchone()[0]
print(f"Test 2: {test2} of 5 categories with >=2pp lift in 7d.  {'PASS' if test2 >= 3 else 'FAIL'}")

# %% [markdown]
# ## Poisson distribution calibration check
#
# The sparse-cat (HR / SB) cold rule assumes window counts follow a Poisson
# distribution with rate = projected_rate × window_PA. The math is principled
# (each PA is approximately Bernoulli; sum-of-Bernoullis = Binomial; Binomial
# → Poisson when N is large and p is small — Var differs by ~5-7% at our
# parameters). But the calibration deserves an empirical eyeball before we
# trust it for the cold-label rule.
#
# We bin windows by expected HR count, plot the empirical PMF against
# Poisson(λ=bin_center).pmf, and check that the bottom-decile / bottom-
# quintile thresholds the Poisson rule is using line up with the empirical
# bottom-decile / bottom-quintile of actual outcomes.

# %%
import numpy as np
from scipy.stats import poisson as poisson_dist

calib = conn.execute(
    """
    SELECT
        w.pa AS window_pa,
        w.hr,
        w.sb,
        p.hr_per_pa * w.pa AS expected_hr,
        p.sb_per_pa * w.pa AS expected_sb
    FROM hitter_windows w
    INNER JOIN hitter_projection_rates p
      ON p.player_id = w.player_id
     AND p.season = EXTRACT(YEAR FROM w.window_end)::INTEGER
    WHERE w.window_days = 7
    """
).df()
print(f"Total 7-day projected windows: {len(calib):,}")

# Pick a workable bin for HR (most informative range — lower bins are too
# noisy because both empirical and Poisson are dominated by zeros, higher
# bins have too few rows).
bin_lo, bin_hi, lam_center = 1.5, 2.5, 2.0
sub = calib[(calib["expected_hr"] >= bin_lo) & (calib["expected_hr"] < bin_hi)]
print(f"Calibration bin (HR): expected ∈ [{bin_lo}, {bin_hi}), N = {len(sub):,} windows")

max_hr = max(int(sub["hr"].max()), 6)
emp_pmf = (
    sub["hr"].value_counts(normalize=True).reindex(range(max_hr + 1), fill_value=0.0).sort_index()
)
poiss_pmf = pd.Series(
    poisson_dist.pmf(np.arange(max_hr + 1), lam_center), index=range(max_hr + 1)
)

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(max_hr + 1)
ax.bar(x - 0.2, emp_pmf.to_numpy(), width=0.4, label="empirical", color="steelblue")
ax.bar(x + 0.2, poiss_pmf.to_numpy(), width=0.4, label=f"Poisson(λ={lam_center})", color="orange")
ax.set_xlabel("HR in 7-day window")
ax.set_ylabel("P(count)")
ax.set_title(f"Calibration: empirical vs Poisson, expected_HR ∈ [{bin_lo}, {bin_hi})")
ax.legend()
plt.show()

emp_p10 = sub["hr"].quantile(0.10)
emp_p20 = sub["hr"].quantile(0.20)
poiss_p10 = int(poisson_dist.ppf(0.10, lam_center))
poiss_p20 = int(poisson_dist.ppf(0.20, lam_center))
print(f"  empirical bottom-10th HR: {emp_p10}    Poisson p10 (rule threshold): {poiss_p10}")
print(f"  empirical bottom-20th HR: {emp_p20}    Poisson p20 (rule threshold): {poiss_p20}")
print()
print("Interpretation:")
print("  If empirical bottom-10th < Poisson p10: real-world counts are overdispersed — our")
print("    cold rule is conservative (under-labels cold).")
print("  If empirical bottom-10th > Poisson p10: counts are underdispersed — rule over-labels.")
print("  If they match within ±1 count: Poisson is calibrated and the rule is well-tuned.")

# %% [markdown]
# ## Notes / methodology surprises to record in the spec progress entry
#
# - Where the lift is concentrated (which categories, windows, buckets).
# - Whether p10 or p20 is the better cold rule for HR / SB given the
#   observed lift × cell-size tradeoff.
# - Cells that came back with N < 1000 — are any of them load-bearing
#   for the Phase 4 plan?
# - Poisson calibration result above — log the empirical-vs-rule
#   comparison so Phase 4 knows whether to trust the rule unmodified or
#   add an overdispersion correction.

# %%
print("Done.")
