# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
# ---

# %% [markdown]
# # Hot Streaks Phase 4 — Predictive Model
#
# This notebook is the Phase 4 acceptance artifact. It refits all 8 models
# from the local DB and renders, for each:
#
# - ROC curve on 2025
# - Reliability diagram (10 bins, diagnostic only — no isotonic correction here)
# - Coefficient bar chart with 5%/95% bootstrap CIs (200 player-grouped resamples)
# - Permutation importance bar chart
#
# **Gate (committed in the Phase 4 plan):**
# Held-out 2025 ROC-AUC >= 0.55 in at least 3 of 5 categories, where per-cat AUC
# is `max(hot_auc, cold_auc)` for dense cats and `hot_auc` for sparse cats.
#
# **Run after:** `python scripts/streaks/fit_models.py --season-set-train 2023-2024 --season-set-val 2025`

# %%
import time
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

from fantasy_baseball.streaks.analysis.predictors import (
    _LR_STEP_NAME,
    DEFAULT_C_GRID,
    EXPECTED_FEATURE_COLUMNS,
    PHASE_4_MODELS,
    build_training_frame,
    fit_all_models,
)

_here = Path.cwd()
for _candidate in (_here, *_here.parents):
    if (_candidate / "pyproject.toml").exists():
        REPO_ROOT = _candidate
        break
else:
    raise RuntimeError("could not locate repo root")

DB = REPO_ROOT / "data" / "streaks" / "streaks.duckdb"
SEASON_SET_TRAIN = "2023-2024"
SEASON_SET_VAL = "2025"

conn = duckdb.connect(str(DB))

# %% [markdown]
# ## Refit all 8 models (in-notebook)
#
# Same call the CLI makes. The refit takes ~60s on the 3-season corpus.

# %%
t0 = time.perf_counter()
results = fit_all_models(
    conn,
    season_set_train=SEASON_SET_TRAIN,
    season_set_val=SEASON_SET_VAL,
    window_days=14,
    C_grid=DEFAULT_C_GRID,
    n_bootstrap=200,
    random_state=42,
)
print(f"fit_all_models wall time: {time.perf_counter() - t0:.1f}s")

# %% [markdown]
# ## Gate summary

# %%
rows = []
per_cat_max: dict[str, float] = {}
for (cat, direction), per_model in results.fits.items():
    if per_model is None:
        rows.append(
            {
                "category": cat,
                "direction": direction,
                "chosen_C": None,
                "cv_auc": None,
                "val_auc": None,
                "n_train": 0,
                "n_val": 0,
            }
        )
        continue
    rows.append(
        {
            "category": cat,
            "direction": direction,
            "chosen_C": per_model.fit.chosen_C,
            "cv_auc": round(per_model.fit.cv_auc_mean, 3),
            "val_auc": round(per_model.evaluation.auc, 3),
            "n_train": per_model.n_train_rows,
            "n_val": per_model.n_val_rows,
        }
    )
    per_cat_max[cat] = max(per_cat_max.get(cat, -1.0), per_model.evaluation.auc)
summary = pd.DataFrame(rows)
summary

# %%
cats_passing = sum(1 for v in per_cat_max.values() if v >= 0.55)
print(f"Categories with max AUC >= 0.55: {cats_passing} / 5")
print(f"Phase 4 gate: {'PASS' if cats_passing >= 3 else 'FAIL'}")

# %% [markdown]
# ## Per-model diagnostic plots
#
# For each fitted model, render four panels:
# 1. ROC curve on 2025
# 2. Reliability diagram (predicted vs observed; the y=x diagonal is perfect)
# 3. Coefficient bar chart with bootstrap CIs
# 4. Permutation importance bar chart


# %%
def _val_frame_for_model(cat: str, direction: str) -> pd.DataFrame:
    """Re-build the val frame so we can score the ROC curve and reliability."""
    full = build_training_frame(
        conn, category=cat, direction=direction, season_set="2023-2025", window_days=14
    )
    return full[full["season"] == int(SEASON_SET_VAL)]


def _plot_one_model(cat: str, direction: str, per_model) -> None:
    if per_model is None:
        print(f"({cat}, {direction}): no model fitted; skipping plots")
        return

    val_df = _val_frame_for_model(cat, direction)
    feature_cols = list(EXPECTED_FEATURE_COLUMNS)
    X_val = val_df[feature_cols]
    y_val = val_df["target"].to_numpy()
    proba = per_model.fit.pipeline.predict_proba(X_val)[:, 1]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(
        f"{cat} {direction}  C={per_model.fit.chosen_C:.2g}  "
        f"cv_auc={per_model.fit.cv_auc_mean:.3f}±{per_model.fit.cv_auc_std:.3f}  "
        f"val_auc={per_model.evaluation.auc:.3f}  n_val={per_model.n_val_rows}",
        fontsize=11,
    )

    # 1. ROC curve.
    fpr, tpr, _ = roc_curve(y_val, proba)
    axes[0, 0].plot(fpr, tpr, label=f"AUC={per_model.evaluation.auc:.3f}")
    axes[0, 0].plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
    axes[0, 0].set_xlabel("False positive rate")
    axes[0, 0].set_ylabel("True positive rate")
    axes[0, 0].set_title("ROC curve")
    axes[0, 0].legend()

    # 2. Reliability diagram.
    centers = np.array(per_model.evaluation.reliability_bin_centers)
    observed = np.array(per_model.evaluation.reliability_observed)
    axes[0, 1].plot(centers, observed, "o-", label="observed")
    axes[0, 1].plot([0, 1], [0, 1], "--", color="gray", alpha=0.5, label="perfect calibration")
    axes[0, 1].set_xlabel("Predicted probability (bin center)")
    axes[0, 1].set_ylabel("Observed positive rate")
    axes[0, 1].set_title("Reliability (10 bins, diagnostic)")
    axes[0, 1].legend()
    axes[0, 1].set_xlim(0, 1)
    axes[0, 1].set_ylim(0, 1)

    # 3. Coefficient bar chart with bootstrap CIs.
    feat_names = list(per_model.coef_ci.keys())
    coefs = per_model.fit.pipeline.named_steps[_LR_STEP_NAME].coef_.ravel()
    lo = np.array([per_model.coef_ci[n][0] for n in feat_names])
    hi = np.array([per_model.coef_ci[n][1] for n in feat_names])
    err = np.vstack([coefs - lo, hi - coefs])
    y_pos = np.arange(len(feat_names))
    axes[1, 0].barh(y_pos, coefs, xerr=err, color="steelblue", alpha=0.7, capsize=3)
    axes[1, 0].set_yticks(y_pos)
    axes[1, 0].set_yticklabels(feat_names)
    axes[1, 0].axvline(0, color="black", linewidth=0.5)
    axes[1, 0].set_xlabel("Coefficient (standardized)")
    axes[1, 0].set_title("L2 LR coefficient +/-[5th, 95th] bootstrap")

    # 4. Permutation importance.
    means = np.array([per_model.permutation_importance[n][0] for n in feat_names])
    stds = np.array([per_model.permutation_importance[n][1] for n in feat_names])
    order = np.argsort(means)
    axes[1, 1].barh(
        np.arange(len(feat_names)),
        means[order],
        xerr=stds[order],
        color="darkorange",
        alpha=0.7,
        capsize=3,
    )
    axes[1, 1].set_yticks(np.arange(len(feat_names)))
    axes[1, 1].set_yticklabels([feat_names[i] for i in order])
    axes[1, 1].set_xlabel("delta-AUC when feature is permuted")
    axes[1, 1].set_title("Permutation importance (n_repeats=10)")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# %%
for cat, direction in PHASE_4_MODELS:
    _plot_one_model(cat, direction, results.fits[(cat, direction)])

# %% [markdown]
# ## Methodology notes
#
# - **Calibration:** reliability diagrams are diagnostic-only here. If Phase 5
#   sees a model whose predictions are systematically off (e.g. 10pp+ at any
#   bin center), apply isotonic correction at that point.
# - **Coefficient CIs:** correlated features (BABIP <-> xwOBA, ISO carrying
#   power signal alongside xwOBA) tend to have wide CIs because L2 distributes
#   weight ambiguously between them. The narrative should not treat any single
#   one of those features as "the driver." (Note: ``barrel_pct`` was dropped
#   from the Phase 4 feature set because the local Statcast corpus has it NULL
#   on every row — see the Task 14 acceptance entry in the spec progress log.)
# - **Permutation importance:** unlike |coef|, this is honest under correlation
#   (shuffling one of a correlated pair still leaks signal via its partner,
#   which conservatively *reduces* its measured importance — read positives
#   as real signal; weakness on a correlated feature is ambiguous).
# - **Sparse-cat AUCs (HR/SB hot) hover near coin-flip.** The Phase 3 lift
#   came from streak_strength alone; peripherals don't add much for HR/SB
#   hot continuation. Worth flagging in Phase 5's Sunday report — sparse-cat
#   probabilities should be consumed conservatively.

# %% [markdown]
# ## Done.

# %%
print("Done.")
