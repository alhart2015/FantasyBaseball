"""Offline A/B: do per-player baseline z-score peripherals help streak models?

Hypothesis: the production model uses absolute peripheral values (k_pct,
xwoba_avg, ev_avg, barrel_pct). "K% is 27%" reads differently from "K% is
27% for a hitter whose career baseline is 18%." A z-score vs the player's
own baseline should let the model see this.

Method:
  1. Pull the same training frame the production code uses, for one (cat,
     direction) at a time -- start with ('r','below') since that's the
     Adames cold case.
  2. Compute per-player baseline = median of each peripheral across the
     player's windows in PRIOR seasons only (leak-free).
  3. Drop rows whose player has no prior-season data (cold-start hitters).
  4. Add 4 z-score features: (current - baseline) / pop_sd  for k_pct,
     xwoba_avg, ev_avg, barrel_pct.
  5. Train baseline-only and baseline+z-score logistic regressions with
     the same player-grouped 5-fold CV the production code uses; report
     CV AUC delta + held-out AUC delta on 2025.

If the lift is real (~>0.01 AUC), wire it into the production train/score
path under TDD. If not, document the null result and move on.
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

# Peripherals we'll z-score. The other features (streak_strength_numeric,
# season_rate_in_category, pt_bucket_*) are not contact-quality signals
# in the same sense -- z-scoring them would be conceptually different.
Z_SCORE_PERIPHERALS = ("k_pct", "xwoba_avg", "ev_avg", "barrel_pct")
Z_SCORE_FEATURE_NAMES = tuple(f"{p}_z" for p in Z_SCORE_PERIPHERALS)


def _player_prior_season_baselines(
    conn: duckdb.DuckDBPyConnection,
    *,
    seasons: list[int],
) -> pd.DataFrame:
    """For each (player_id, season), compute per-peripheral medians across
    that player's windows in PRIOR seasons only.

    Returns: one row per (player_id, season) with one column per peripheral
    holding the player's median across all their windows in earlier seasons.
    Players with no prior-season data are absent from the frame.
    """
    sql = f"""
        SELECT player_id, EXTRACT(YEAR FROM window_end)::INTEGER AS year,
               k_pct, xwoba_avg, ev_avg, barrel_pct
        FROM hitter_windows
        WHERE window_days = 14
          AND EXTRACT(YEAR FROM window_end)::INTEGER
              IN ({", ".join(str(s) for s in seasons)})
    """
    df = conn.execute(sql).df()
    if df.empty:
        return pd.DataFrame(
            columns=["player_id", "season", *(f"baseline_{p}" for p in Z_SCORE_PERIPHERALS)]
        )

    rows: list[dict] = []
    # For each requested season, compute medians over (player's windows in years < season).
    for s in seasons:
        prior = df[df["year"] < s]
        if prior.empty:
            continue
        grouped = prior.groupby("player_id", sort=False)
        agg = grouped[list(Z_SCORE_PERIPHERALS)].median()
        agg = agg.rename(columns={p: f"baseline_{p}" for p in Z_SCORE_PERIPHERALS})
        agg["season"] = s
        agg = agg.reset_index()
        rows.append(agg)

    if not rows:
        return pd.DataFrame(
            columns=["player_id", "season", *(f"baseline_{p}" for p in Z_SCORE_PERIPHERALS)]
        )
    return pd.concat(rows, ignore_index=True)


def _population_sds(conn: duckdb.DuckDBPyConnection) -> dict[str, float]:
    """Population SD per peripheral across all 14d windows.

    Used as the z-score denominator so the score is unit-free and the
    feature scaling stays interpretable.
    """
    sql = f"""
        SELECT {", ".join(f"STDDEV_POP({p}) AS sd_{p}" for p in Z_SCORE_PERIPHERALS)}
        FROM hitter_windows
        WHERE window_days = 14
    """
    row = conn.execute(sql).fetchone()
    assert row is not None
    return {p: float(row[i]) for i, p in enumerate(Z_SCORE_PERIPHERALS)}


def _augment_with_zscores(
    df: pd.DataFrame, baselines: pd.DataFrame, sds: dict[str, float]
) -> pd.DataFrame:
    """Left-join baselines onto df, then compute z-scores. Drop rows missing
    any baseline (player had no prior-season windows)."""
    out = df.merge(baselines, on=["player_id", "season"], how="left")
    n_pre = len(out)
    baseline_cols = [f"baseline_{p}" for p in Z_SCORE_PERIPHERALS]
    out = out.dropna(subset=baseline_cols).copy()
    n_dropped = n_pre - len(out)
    print(
        f"  dropped {n_dropped}/{n_pre} rows ({n_dropped / n_pre * 100:.1f}%) "
        f"with no prior-season baseline"
    )
    for p in Z_SCORE_PERIPHERALS:
        out[f"{p}_z"] = (out[p] - out[f"baseline_{p}"]) / sds[p]
    return out


def _fit_and_score(
    df_train: pd.DataFrame, df_val: pd.DataFrame, feature_cols: list[str], label: str
) -> dict[str, float]:
    """Player-grouped 5-fold CV on df_train + held-out eval on df_val.

    Mirrors fit_one_model's structure but inlined so the script stays
    self-contained and uses the same C_grid + GroupKFold.
    """
    X_train = df_train[feature_cols].to_numpy()
    y_train = df_train["target"].to_numpy()
    groups = df_train["player_id"].to_numpy()
    X_val = df_val[feature_cols].to_numpy()
    y_val = df_val["target"].to_numpy()

    C_grid = (0.01, 0.1, 1.0, 10.0)
    best = {"C": None, "mean": -np.inf, "std": 0.0}
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
        if m > best["mean"]:
            best = {"C": C, "mean": m, "std": float(np.std(per_fold))}

    # Refit on full train at chosen C
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

    # Build the full frame the production code would build.
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

    baselines = _player_prior_season_baselines(conn, seasons=all_seasons)
    sds = _population_sds(conn)
    print(f"  population SDs: {sds}")
    augmented = _augment_with_zscores(full, baselines, sds)
    if augmented.empty:
        print("  no rows survived baseline join, skipping")
        return

    # Same train/val split as production
    df_train = augmented[augmented["season"].isin(train_seasons)].copy()
    df_val = augmented[augmented["season"].isin(val_seasons)].copy()
    if df_train.empty or df_val.empty:
        print(f"  empty split: train={len(df_train)} val={len(df_val)}, skipping")
        return

    baseline_features = list(EXPECTED_FEATURE_COLUMNS)
    augmented_features = baseline_features + list(Z_SCORE_FEATURE_NAMES)

    r_base = _fit_and_score(df_train, df_val, baseline_features, "baseline")
    r_aug = _fit_and_score(df_train, df_val, augmented_features, "augmented")

    print()
    print(f"  {'metric':<20} {'baseline':>10} {'augmented':>10} {'delta':>10}")
    print("  " + "-" * 52)
    print(
        f"  {'CV AUC mean':<20} "
        f"{r_base['cv_auc_mean']:>10.4f} "
        f"{r_aug['cv_auc_mean']:>10.4f} "
        f"{r_aug['cv_auc_mean'] - r_base['cv_auc_mean']:>+10.4f}"
    )
    print(f"  {'CV AUC std':<20} {r_base['cv_auc_std']:>10.4f} {r_aug['cv_auc_std']:>10.4f}")
    print(
        f"  {'val AUC (2025)':<20} "
        f"{r_base['val_auc']:>10.4f} "
        f"{r_aug['val_auc']:>10.4f} "
        f"{r_aug['val_auc'] - r_base['val_auc']:>+10.4f}"
    )
    print(f"  chosen C: baseline={r_base['chosen_C']}  augmented={r_aug['chosen_C']}")
    print(f"  rows:    train={r_base['n_train']}  val={r_base['n_val']}")

    print()
    print("  Z-score feature coefficients in augmented model:")
    for f in Z_SCORE_FEATURE_NAMES:
        coef = r_aug["coefs"].get(f, float("nan"))
        print(f"    {f:>20}: {coef:>+8.4f}")
    print("  (positive coef => higher z increases P(continuation))")
    print()
    print("  Raw-peripheral coefficient shift baseline -> augmented:")
    for p in Z_SCORE_PERIPHERALS:
        base_c = r_base["coefs"].get(p, float("nan"))
        aug_c = r_aug["coefs"].get(p, float("nan"))
        print(f"    {p:>20}: {base_c:>+8.4f} -> {aug_c:>+8.4f}")
    print(
        "  (if raw coef shrinks toward 0 in augmented, the z-score is absorbing the signal cleanly)"
    )


def main() -> None:
    conn = duckdb.connect(str(DB), read_only=True)

    # Hit every Phase 4 model so we can see broad lift vs single-model fluke.
    for cat, direction in PHASE_4_MODELS:
        try:
            evaluate_one(conn, category=cat, direction=direction)
        except Exception as e:
            print(f"  ERROR on ({cat}, {direction}): {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
