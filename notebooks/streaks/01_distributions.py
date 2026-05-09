# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Hot Streaks Phase 2 — Distributions & Threshold Sanity Check
#
# This is the Phase-2 acceptance notebook. It loads the calibrated
# `thresholds` table and the underlying `hitter_windows` data, and
# confirms that the empirical p10/p90 thresholds make intuitive sense
# for each category × window × playing-time bucket.
#
# **Run after:** `python scripts/streaks/compute_labels.py --season-set 2023-2025`

# %%
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

# Walk up from the notebook to find the repo root (the directory containing
# `pyproject.toml`), so the notebook works from any cwd.
_here = Path.cwd()
for _candidate in (_here, *_here.parents):
    if (_candidate / "pyproject.toml").exists():
        REPO_ROOT = _candidate
        break
else:
    raise RuntimeError("could not locate repo root (no pyproject.toml above cwd)")

DB = REPO_ROOT / "data" / "streaks" / "streaks.duckdb"
SEASON_SET = "2023-2025"

conn = duckdb.connect(str(DB))

# %% [markdown]
# ## Threshold table
#
# All p10 / p90 thresholds calibrated from 2023-2025 windows for hitters
# with ≥150 PA in the relevant season.

# %%
thresholds = conn.execute(
    "SELECT category, window_days, pt_bucket, p10, p90 "
    "FROM thresholds WHERE season_set = ? "
    "ORDER BY category, window_days, pt_bucket",
    [SEASON_SET],
).df()
thresholds

# %% [markdown]
# ## Distribution histograms — counting categories
#
# For each (category, window_days, pt_bucket), plot the empirical
# distribution and overlay p10/p90 vertical lines. Bucket order: low / mid / high.

# %%
windows = conn.execute(
    """
    SELECT player_id, window_end, window_days, pt_bucket,
           pa, hr, r, rbi, sb, avg
    FROM hitter_windows
    """
).df()

for cat in ("hr", "r", "rbi", "sb"):
    fig, axes = plt.subplots(3, 3, figsize=(14, 10), sharey=False)
    for col, wd in enumerate((3, 7, 14)):
        for row, bucket in enumerate(("low", "mid", "high")):
            ax = axes[row, col]
            sub = windows[(windows["window_days"] == wd) & (windows["pt_bucket"] == bucket)]
            ax.hist(sub[cat], bins=range(int(sub[cat].max()) + 2), edgecolor="black")
            t = thresholds[
                (thresholds["category"] == cat)
                & (thresholds["window_days"] == wd)
                & (thresholds["pt_bucket"] == bucket)
            ]
            if not t.empty:
                ax.axvline(t.iloc[0]["p10"], color="blue", linestyle="--", label=f"p10={t.iloc[0]['p10']:.1f}")
                ax.axvline(t.iloc[0]["p90"], color="red", linestyle="--", label=f"p90={t.iloc[0]['p90']:.1f}")
                ax.legend(fontsize=8)
            ax.set_title(f"{cat.upper()} — {wd}d × {bucket}")
    fig.suptitle(f"Distribution + thresholds: {cat.upper()}", fontsize=14)
    fig.tight_layout()
    plt.show()

# %% [markdown]
# ## Distribution — AVG (rate)

# %%
fig, axes = plt.subplots(3, 3, figsize=(14, 10))
for col, wd in enumerate((3, 7, 14)):
    for row, bucket in enumerate(("low", "mid", "high")):
        ax = axes[row, col]
        sub = windows[(windows["window_days"] == wd) & (windows["pt_bucket"] == bucket)]
        ax.hist(sub["avg"].dropna(), bins=40, edgecolor="black")
        t = thresholds[
            (thresholds["category"] == "avg")
            & (thresholds["window_days"] == wd)
            & (thresholds["pt_bucket"] == bucket)
        ]
        if not t.empty:
            ax.axvline(t.iloc[0]["p10"], color="blue", linestyle="--", label=f"p10={t.iloc[0]['p10']:.3f}")
            ax.axvline(t.iloc[0]["p90"], color="red", linestyle="--", label=f"p90={t.iloc[0]['p90']:.3f}")
            ax.legend(fontsize=8)
        ax.set_title(f"AVG — {wd}d × {bucket}")
fig.suptitle("Distribution + thresholds: AVG", fontsize=14)
fig.tight_layout()
plt.show()

# %% [markdown]
# ## Label distribution sanity check
#
# For HR / 7d / high (the canonical "is this player on a home-run tear"
# question), how do labels split across hot / cold / neutral?

# %%
label_split = conn.execute(
    """
    SELECT label, COUNT(*) AS n
    FROM hitter_streak_labels l
    JOIN hitter_windows w
      ON w.player_id = l.player_id
     AND w.window_end = l.window_end
     AND w.window_days = l.window_days
    WHERE l.category = 'hr' AND w.window_days = 7 AND w.pt_bucket = 'high'
    GROUP BY label
    ORDER BY label
    """
).df()
label_split

# %% [markdown]
# ## Eyeball checklist
#
# Confirm before signing off:
#
# 1. **HR / 7d / high** p90 ≈ 2-3 (a "hot HR week" is 2-3+ HR)
# 2. **AVG / 14d / high** p90 ∈ [.340, .420]
# 3. **AVG / 14d / high** p10 ∈ [.150, .190]
# 4. **SB / 7d / high** p90 ≈ 2 (rare to steal 2+ in a week as a top-PA hitter)
# 5. No category has p10 > p90 (this would be a bug)
# 6. Bucket monotonicity: for counting cats, p90 should rise as bucket
#    moves low → mid → high (more PA = more counts). Visual check.
#
# **Methodology note:** for sparse counting categories (HR, SB), p10
# collapses to 0 because most weeks have zero events even for high-PA
# hitters. The "cold" label therefore covers any window with zero
# events in that category — a much wider net than "below the 10th
# percentile of nonzero counts." Phase 3 may revisit the symmetric-
# percentile approach for sparse counts (e.g., a per-category lower
# bound like "0 means cold only if PA ≥ N").

# %%
print("Notebook done.")
