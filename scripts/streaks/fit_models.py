"""CLI: fit all 8 Phase 4 streak-continuation models.

Assumes Phase 3 outputs exist:
- hitter_windows, hitter_streak_labels, hitter_projection_rates, thresholds

Usage:
    python -m scripts.streaks.fit_models [--db-path PATH]
        [--season-set-train 2023-2024] [--season-set-val 2025]
        [--window-days 14] [--n-bootstrap 200]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.streaks.analysis.predictors import (
    DEFAULT_C_GRID,
    fit_all_models,
)
from fantasy_baseball.streaks.data.schema import DEFAULT_DB_PATH, get_connection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Fit Phase 4 streak-continuation models.")
    parser.add_argument("--db-path", type=Path, default=DEFAULT_DB_PATH)
    parser.add_argument("--season-set-train", default="2023-2024")
    parser.add_argument("--season-set-val", default="2025")
    parser.add_argument("--window-days", type=int, default=14)
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    conn = get_connection(args.db_path)
    try:
        result = fit_all_models(
            conn,
            season_set_train=args.season_set_train,
            season_set_val=args.season_set_val,
            window_days=args.window_days,
            C_grid=DEFAULT_C_GRID,
            n_bootstrap=args.n_bootstrap,
            random_state=args.random_state,
        )
    finally:
        conn.close()

    # Print one summary line per model + per-category gate summary.
    print()
    print(f"Phase 4 fit summary — train={args.season_set_train} / val={args.season_set_val}")
    print("-" * 75)
    print(f"{'model_id':30s} {'C':>6s} {'cv_auc':>8s} {'val_auc':>8s} {'n_train':>8s} {'n_val':>8s}")
    per_cat_max: dict[str, float] = {}
    for (cat, direction), per_model in result.fits.items():
        if per_model is None:
            print(f"{cat}_{direction:<24s}  SKIPPED (no training rows or single-class target)")
            continue
        model_id = f"{cat}_{'hot' if direction == 'above' else 'cold'}_{args.season_set_train}"
        print(
            f"{model_id:30s} {per_model.fit.chosen_C:6.2f} "
            f"{per_model.fit.cv_auc_mean:8.3f} {per_model.evaluation.auc:8.3f} "
            f"{per_model.n_train_rows:8d} {per_model.n_val_rows:8d}"
        )
        per_cat_max[cat] = max(per_cat_max.get(cat, -1.0), per_model.evaluation.auc)

    # Gate: 2025 AUC >= 0.55 in >= 3 of 5 categories.
    print()
    cats_passing = sum(1 for v in per_cat_max.values() if v >= 0.55)
    print(f"Categories with max(hot_auc, cold_auc) >= 0.55: {cats_passing} / 5")
    gate = "PASS" if cats_passing >= 3 else "FAIL"
    print(f"Phase 4 gate: {gate}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
