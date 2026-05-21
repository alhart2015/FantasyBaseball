"""Offline A/B: do delta-from-prior-window features help streak continuation?

Hypothesis: snapshot peripherals miss trajectory. Adames diagnostic (May
2026) showed K% collapsing 41.7% -> 27.1% across a cold stretch while the
chip was still "cold" -- the static 14d window couldn't see the recovery.

Test: for each (player, window_end) with window_days=14, look up the
14d window ending 7 days earlier (so windows half-overlap and the
delta captures "what's changed in the last week vs the first week of
the window"). Add features:
  k_pct_delta_7d, xwoba_avg_delta_7d, ev_avg_delta_7d,
  barrel_pct_delta_7d, babip_delta_7d

Compare CV + held-out AUC vs the baseline 12-feature model. If lift is
real (>=~0.01 val AUC) on cold-continuation models, wire it into the
production training + inference path under TDD.

Method mirrors scripts/streaks/eval_zscore_features.py exactly so the two
results are directly comparable.
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.analysis.predictors import (
    EXPECTED_FEATURE_COLUMNS,
    PHASE_4_MODELS,
    build_training_frame,
)

DB = PROJECT_ROOT / "data" / "streaks" / "streaks.duckdb"

DELTA_PERIPHERALS = ("k_pct", "xwoba_avg", "ev_avg", "barrel_pct", "babip")
DELTA_FEATURE_NAMES = tuple(f"{p}_delta_7d" for p in DELTA_PERIPHERALS)


def _load_prior_window_deltas(
    conn: duckdb.DuckDBPyConnection,
) -> pd.DataFrame:
    """For every 14d window, compute the delta vs the 14d window ending 7 days earlier.

    Returns one row per (player_id, window_end) with one column per peripheral
    delta. Windows with no 7-day-prior counterpart are absent (caller will
    left-join and drop).
    """
    df = conn.execute(
        """
        SELECT
            cur.player_id,
            cur.window_end,
            cur.k_pct - prior.k_pct           AS k_pct_delta_7d,
            cur.xwoba_avg - prior.xwoba_avg   AS xwoba_avg_delta_7d,
            cur.ev_avg - prior.ev_avg         AS ev_avg_delta_7d,
            cur.barrel_pct - prior.barrel_pct AS barrel_pct_delta_7d,
            cur.babip - prior.babip           AS babip_delta_7d
        FROM hitter_windows cur
        INNER JOIN hitter_windows prior
          ON  prior.player_id = cur.player_id
         AND prior.window_days = 14
         AND prior.window_end = cur.window_end - INTERVAL '7' DAY
        WHERE cur.window_days = 14
        """
    ).df()
    return df


def _augment_with_deltas(df: pd.DataFrame, deltas: pd.DataFrame) -> pd.DataFrame:
    """Left-join deltas onto df, drop rows missing the prior window."""
    out = df.merge(deltas, on=["player_id", "window_end"], how="left")
    n_pre = len(out)
    delta_cols = list(DELTA_FEATURE_NAMES)
    out = out.dropna(subset=delta_cols).copy()
    n_dropped = n_pre - len(out)
    print(
        f"  dropped {n_dropped}/{n_pre} rows ({n_dropped / n_pre * 100:.1f}%) "
        f"with no 7-day-prior window"
    )
    return out


def _fit_and_score(
    df_train: pd.DataFrame, df_val: pd.DataFrame, feature_cols: list[str], label: str
) -> dict[str, object]:
    """Same shape as the z-score A/B harness for direct comparability."""
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train["target"].to_numpy()
    groups = df_train["player_id"].to_numpy()
    X_val = df_val[feature_cols].to_numpy()
    y_val = df_val["target"].to_numpy()

    C_grid = (0.01, 0.1, 1.0, 10.0)
    best: dict[str, object] = {"C": None, "mean": -np.inf, "std": 0.0}
    for C in C_grid:
        cv = GroupKFold(n_splits=5)
        per_fold = []
        for tr, va in cv.split(X_train, y_train, groups=groups):
            pipe = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("lr", LogisticRegression(C=C, solver="lbfgs", max_iter=1000)),
                ]
            )
            pipe.fit(X_train[tr], y_train[tr])
            if len(np.unique(y_train[va])) < 2:
                continue
            per_fold.append(roc_auc_score(y_train[va], pipe.predict_proba(X_train[va])[:, 1]))
        if not per_fold:
            continue
        m = float(np.mean(per_fold))
        if m > best["mean"]:  # type: ignore[operator]
            best = {"C": C, "mean": m, "std": float(np.std(per_fold))}

    final = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=best["C"], solver="lbfgs", max_iter=1000)),
        ]
    )
    final.fit(X_train, y_train)
    val_auc = (
        roc_auc_score(y_val, final.predict_proba(X_val)[:, 1])
        if len(np.unique(y_val)) >= 2
        else float("nan")
    )

    coef = final.named_steps["lr"].coef_.ravel()
    coefs = {name: float(c) for name, c in zip(feature_cols, coef, strict=True)}
    return {
        "label": label,
        "n_train": len(df_train),
        "n_val": len(df_val),
        "chosen_C": best["C"],
        "cv_auc_mean": best["mean"],
        "cv_auc_std": best["std"],
        "val_auc": val_auc,
        "coefs": coefs,
    }


def evaluate_one(
    conn: duckdb.DuckDBPyConnection,
    deltas: pd.DataFrame,
    *,
    category: str,
    direction: str,
    season_set_train: str = "2023-2024",
    season_set_val: str = "2025",
) -> None:
    print()
    print("=" * 78)
    print(f"MODEL: {category} {direction}")
    print("=" * 78)

    train_lo, train_hi = (int(s) for s in season_set_train.split("-"))
    train_seasons = list(range(train_lo, train_hi + 1))
    val_seasons = [int(season_set_val)]
    all_seasons = sorted(set(train_seasons + val_seasons))
    combined = f"{min(all_seasons)}-{max(all_seasons)}"

    full = build_training_frame(
        conn,
        category=category,  # type: ignore[arg-type]
        direction=direction,  # type: ignore[arg-type]
        season_set=combined,
        window_days=14,
    )
    if full.empty:
        print("  empty training frame, skipping")
        return

    augmented = _augment_with_deltas(full, deltas)
    if augmented.empty:
        print("  no rows survived delta join, skipping")
        return

    df_train = augmented[augmented["season"].isin(train_seasons)].copy()
    df_val = augmented[augmented["season"].isin(val_seasons)].copy()
    if df_train.empty or df_val.empty:
        print(f"  empty split: train={len(df_train)} val={len(df_val)}, skipping")
        return

    baseline_features = list(EXPECTED_FEATURE_COLUMNS)
    augmented_features = baseline_features + list(DELTA_FEATURE_NAMES)

    r_base = _fit_and_score(df_train, df_val, baseline_features, "baseline")
    r_aug = _fit_and_score(df_train, df_val, augmented_features, "augmented")

    print()
    print(f"  {'metric':<20} {'baseline':>10} {'augmented':>10} {'delta':>10}")
    print("  " + "-" * 52)
    cv_b = r_base["cv_auc_mean"]
    cv_a = r_aug["cv_auc_mean"]
    val_b = r_base["val_auc"]
    val_a = r_aug["val_auc"]
    assert isinstance(cv_b, float) and isinstance(cv_a, float)
    assert isinstance(val_b, float) and isinstance(val_a, float)
    print(f"  {'CV AUC mean':<20} {cv_b:>10.4f} {cv_a:>10.4f} {cv_a - cv_b:>+10.4f}")
    print(f"  {'CV AUC std':<20} {r_base['cv_auc_std']:>10.4f} {r_aug['cv_auc_std']:>10.4f}")
    print(f"  {'val AUC (2025)':<20} {val_b:>10.4f} {val_a:>10.4f} {val_a - val_b:>+10.4f}")
    print(f"  chosen C: baseline={r_base['chosen_C']}  augmented={r_aug['chosen_C']}")
    print(f"  rows:    train={r_base['n_train']}  val={r_base['n_val']}")

    print()
    print("  Delta feature coefficients in augmented model:")
    coefs_aug = r_aug["coefs"]
    assert isinstance(coefs_aug, dict)
    for f in DELTA_FEATURE_NAMES:
        coef = coefs_aug.get(f, float("nan"))
        print(f"    {f:>22}: {coef:>+8.4f}")
    print("  (positive coef => positive delta increases P(continuation))")


def main() -> None:
    conn = duckdb.connect(str(DB), read_only=True)
    print("Loading 7-day prior window deltas (single query)...")
    deltas = _load_prior_window_deltas(conn)
    print(f"  loaded {len(deltas)} (player, window_end) pairs with deltas")

    for cat, direction in PHASE_4_MODELS:
        try:
            evaluate_one(conn, deltas, category=cat, direction=direction)
        except Exception as e:
            print(f"  ERROR on ({cat}, {direction}): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
