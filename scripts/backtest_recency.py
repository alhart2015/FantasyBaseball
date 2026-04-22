"""Backtest recency-weighting models against 2025 actual game logs.

Runs 5 prediction models at 5 monthly checkpoints, evaluating next-week and
rest-of-season predictions against actual MLB game log data.

Usage:
    python scripts/backtest_recency.py
"""

import csv
import json
import sys
from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd
from backtest_2025 import DRAFT_2025

from fantasy_baseball.analysis.game_logs import fetch_all_game_logs
from fantasy_baseball.analysis.recency import (
    HITTER_STAT_KEYS,
    PITCHER_STAT_KEYS,
    _aggregate_hitter_games,
    _aggregate_pitcher_games,
    _parse_date,
    predict_exponential_decay,
    predict_fixed_blend,
    predict_preseason,
    predict_reliability_blend,
    predict_season_to_date,
)
from fantasy_baseball.utils.name_utils import normalize_name

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"
GAME_LOG_CACHE = PROJECT_ROOT / "data" / "stats" / "game_logs_2025.json"
RESULTS_CSV = PROJECT_ROOT / "data" / "stats" / "recency_backtest_results.csv"

CHECKPOINTS = [
    "2025-05-01",
    "2025-06-01",
    "2025-07-01",
    "2025-08-01",
    "2025-09-01",
]
NEXT_WEEK_DAYS = 7

HITTER_STATS = ["hr_per_pa", "r_per_pa", "rbi_per_pa", "sb_per_pa", "avg"]
PITCHER_STATS = ["k_per_ip", "era", "whip", "w_per_gs", "sv_per_g"]

# Minimum data thresholds to include a player in evaluation
HITTER_MIN_NEXTWEEK_PA = 10
HITTER_MIN_REST_OF_SEASON_PA = 50
PITCHER_MIN_NEXTWEEK_IP = 3
PITCHER_MIN_REST_OF_SEASON_IP = 20

# Minimum preseason PA/IP to include a player in projections
PROJ_HITTER_MIN_PA = 50
PROJ_PITCHER_MIN_IP = 10

MODELS = [
    ("preseason", predict_preseason),
    ("season_to_date", predict_season_to_date),
    ("fixed_blend", predict_fixed_blend),
    ("reliability_blend", predict_reliability_blend),
    ("exponential_decay", predict_exponential_decay),
]

SEASON_END = date(2025, 9, 28)  # approximate end of 2025 regular season


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------


def strip_suffixes(name: str) -> str:
    """Remove common name suffixes for looser matching."""
    for suffix in [" jr.", " jr", " sr.", " sr", " ii", " iii"]:
        if name.endswith(suffix):
            name = name[: -len(suffix)].strip()
    return name


# ---------------------------------------------------------------------------
# Step 1: Load preseason projections
# ---------------------------------------------------------------------------


def load_preseason_projections() -> dict:
    """Load Steamer + ZiPS 2025 projections, average them, return per-PA/IP rates.

    Returns:
        dict keyed by MLBAMID (int):
            {
                'name': str,
                'type': 'hitter' | 'pitcher',
                'proj': {stat_key: float, ...}
            }
    """
    hitter_files = [
        PROJECTIONS_DIR / "2025" / "steamer-hitters-2025.csv",
        PROJECTIONS_DIR / "2025" / "zips-hitters-2025.csv",
    ]
    pitcher_files = [
        PROJECTIONS_DIR / "2025" / "steamer-pitchers-2025.csv",
        PROJECTIONS_DIR / "2025" / "zips-pitchers-2025.csv",
    ]

    projections = {}

    # -- Hitters --
    hitter_dfs = []
    for fpath in hitter_files:
        df = pd.read_csv(fpath)
        df.columns = df.columns.str.strip()
        # Normalize column names: Steamer uses "MLBAMID", ZiPS same
        df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce")
        df = df.dropna(subset=["MLBAMID"])
        df["MLBAMID"] = df["MLBAMID"].astype(int)
        hitter_dfs.append(df[["Name", "MLBAMID", "PA", "AB", "H", "HR", "R", "RBI", "SB"]])

    # Combine and average by MLBAMID
    hitter_combined = pd.concat(hitter_dfs)
    hitter_avg = (
        hitter_combined.groupby("MLBAMID")[["PA", "AB", "H", "HR", "R", "RBI", "SB"]]
        .mean()
        .reset_index()
    )
    # Keep first name per MLBAMID
    hitter_names = hitter_combined.drop_duplicates("MLBAMID")[["MLBAMID", "Name"]].set_index(
        "MLBAMID"
    )["Name"]
    hitter_avg["Name"] = hitter_avg["MLBAMID"].map(hitter_names)

    for _, row in hitter_avg.iterrows():
        mid = int(row["MLBAMID"])
        pa = float(row["PA"])
        ab = float(row["AB"])
        if pa < PROJ_HITTER_MIN_PA:
            continue
        projections[mid] = {
            "name": row["Name"],
            "type": "hitter",
            "proj": {
                "hr_per_pa": float(row["HR"]) / pa,
                "r_per_pa": float(row["R"]) / pa,
                "rbi_per_pa": float(row["RBI"]) / pa,
                "sb_per_pa": float(row["SB"]) / pa,
                "avg": float(row["H"]) / ab if ab > 0 else 0.0,
            },
        }

    # -- Pitchers --
    pitcher_dfs = []
    for fpath in pitcher_files:
        df = pd.read_csv(fpath)
        df.columns = df.columns.str.strip()
        df["MLBAMID"] = pd.to_numeric(df["MLBAMID"], errors="coerce")
        df = df.dropna(subset=["MLBAMID"])
        df["MLBAMID"] = df["MLBAMID"].astype(int)
        pitcher_dfs.append(
            df[["Name", "MLBAMID", "IP", "W", "SO", "SV", "ER", "BB", "H", "GS", "G"]]
        )

    pitcher_combined = pd.concat(pitcher_dfs)
    pitcher_avg = (
        pitcher_combined.groupby("MLBAMID")[["IP", "W", "SO", "SV", "ER", "BB", "H", "GS", "G"]]
        .mean()
        .reset_index()
    )
    pitcher_names = pitcher_combined.drop_duplicates("MLBAMID")[["MLBAMID", "Name"]].set_index(
        "MLBAMID"
    )["Name"]
    pitcher_avg["Name"] = pitcher_avg["MLBAMID"].map(pitcher_names)

    for _, row in pitcher_avg.iterrows():
        mid = int(row["MLBAMID"])
        ip = float(row["IP"])
        gs = float(row["GS"])
        g = float(row["G"])
        if ip < PROJ_PITCHER_MIN_IP:
            continue
        projections[mid] = {
            "name": row["Name"],
            "type": "pitcher",
            "proj": {
                "k_per_ip": float(row["SO"]) / ip,
                "era": float(row["ER"]) * 9 / ip,
                "whip": (float(row["BB"]) + float(row["H"])) / ip,
                "w_per_gs": float(row["W"]) / gs if gs > 0 else 0.0,
                "sv_per_g": float(row["SV"]) / g if g > 0 else 0.0,
            },
        }

    print(
        f"Loaded projections for {len(projections)} players "
        f"({sum(1 for v in projections.values() if v['type'] == 'hitter')} hitters, "
        f"{sum(1 for v in projections.values() if v['type'] == 'pitcher')} pitchers)"
    )
    return projections


# ---------------------------------------------------------------------------
# Step 2: Match DRAFT_2025 names to MLBAMIDs
# ---------------------------------------------------------------------------


def match_draft_to_projections(projections: dict) -> list[dict]:
    """Match draft player names to MLBAM IDs from projection data.

    Returns:
        List of dicts: {'mlbam_id': int, 'name': str, 'type': str, 'proj': dict}
    """
    # Build normalized name -> mlbam_id lookup from projections
    name_lookup: dict[str, int] = {}
    for mid, data in projections.items():
        key = normalize_name(data["name"])
        name_lookup[key] = mid
        # Also index stripped suffix version
        stripped = strip_suffixes(key)
        if stripped != key:
            name_lookup.setdefault(stripped, mid)

    matched_players = []
    seen_ids: set[int] = set()
    matched = 0
    missed = 0
    missed_names = []

    # Deduplicate draft list (same player may appear multiple times, e.g. Ohtani)
    seen_draft_names: set[str] = set()

    for _round, player_name, _team in DRAFT_2025:
        norm = normalize_name(player_name)
        if norm in seen_draft_names:
            continue
        seen_draft_names.add(norm)

        # Try exact match
        mid = name_lookup.get(norm)

        # Try with suffixes stripped
        if mid is None:
            stripped = strip_suffixes(norm)
            mid = name_lookup.get(stripped)

        if mid is not None and mid not in seen_ids:
            seen_ids.add(mid)
            data = projections[mid]
            matched_players.append(
                {
                    "mlbam_id": mid,
                    "name": data["name"],
                    "type": data["type"],
                    "proj": data["proj"],
                }
            )
            matched += 1
        elif mid is None:
            missed += 1
            missed_names.append(player_name)

    if missed_names:
        print(f"Unmatched players ({missed}): {', '.join(missed_names)}")
    print(f"Matched {matched}/{matched + missed} draft players to projection IDs")
    return matched_players


# ---------------------------------------------------------------------------
# Step 3: Fetch game logs
# ---------------------------------------------------------------------------


def fetch_game_logs(matched_players: list[dict]) -> dict:
    """Fetch (or load cached) game logs for all matched players."""
    print(f"Fetching game logs for {len(matched_players)} players (cache: {GAME_LOG_CACHE})...")
    return fetch_all_game_logs(matched_players, season=2025, cache_path=GAME_LOG_CACHE)


# ---------------------------------------------------------------------------
# Step 4: Compute actual rates for a time window
# ---------------------------------------------------------------------------


def compute_actual_rates(
    games: list[dict], player_type: str, start: date, end: date
) -> dict | None:
    """Compute actual per-PA/IP rates for games in [start, end).

    Returns None if insufficient data.
    """
    window_games = [g for g in games if start <= _parse_date(g["date"]) < end]

    if player_type == "hitter":
        agg = _aggregate_hitter_games(window_games)
        return agg if agg["pa"] > 0 else None
    else:
        agg = _aggregate_pitcher_games(window_games)
        return agg if agg["ip"] > 0 else None


def has_sufficient_data(agg: dict, player_type: str, target: str) -> bool:
    """Check if aggregated data meets minimum threshold for evaluation."""
    if player_type == "hitter":
        min_pa = HITTER_MIN_NEXTWEEK_PA if target == "next_week" else HITTER_MIN_REST_OF_SEASON_PA
        return agg["pa"] >= min_pa
    else:
        min_ip = PITCHER_MIN_NEXTWEEK_IP if target == "next_week" else PITCHER_MIN_REST_OF_SEASON_IP
        return agg["ip"] >= min_ip


# ---------------------------------------------------------------------------
# Step 5: Evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(matched_players: list[dict], game_logs: dict) -> list[dict]:
    """Run the full evaluation loop.

    Returns:
        List of result dicts with keys: checkpoint, model, target, stat, mae, n_players
    """
    results = []

    for checkpoint_str in CHECKPOINTS:
        checkpoint = date.fromisoformat(checkpoint_str)
        next_week_end = checkpoint + timedelta(days=NEXT_WEEK_DAYS)

        print(f"\n=== Checkpoint: {checkpoint_str} ===")

        for model_name, model_fn in MODELS:
            # Accumulate errors per (target, stat)
            errors: dict[tuple[str, str], list[float]] = {}

            for player in matched_players:
                mid = player["mlbam_id"]
                ptype = player["type"]
                proj = player["proj"]
                stat_keys = HITTER_STATS if ptype == "hitter" else PITCHER_STATS

                log_data = game_logs.get(mid)
                if log_data is None:
                    continue
                games = log_data["games"]

                # Get prediction from model (using games before checkpoint)
                prediction = model_fn(proj, games, checkpoint)

                for target, start, end in [
                    ("next_week", checkpoint, next_week_end),
                    ("rest_of_season", checkpoint, SEASON_END + timedelta(days=1)),
                ]:
                    actual = compute_actual_rates(games, ptype, start, end)
                    if actual is None:
                        continue
                    if not has_sufficient_data(actual, ptype, target):
                        continue

                    for stat in stat_keys:
                        pred_val = prediction.get(stat, 0.0)
                        act_val = actual.get(stat, 0.0)
                        err = abs(pred_val - act_val)
                        key = (target, stat)
                        errors.setdefault(key, []).append(err)

            # Compute MAE per (target, stat)
            for (target, stat), errs in errors.items():
                mae = sum(errs) / len(errs) if errs else 0.0
                results.append(
                    {
                        "checkpoint": checkpoint_str,
                        "model": model_name,
                        "target": target,
                        "stat": stat,
                        "mae": round(mae, 6),
                        "n_players": len(errs),
                    }
                )

            n_total = sum(len(v) for v in errors.values())
            print(f"  {model_name:<22} {n_total} total (target, stat, player) evals")

    return results


# ---------------------------------------------------------------------------
# Step 6: Save results CSV
# ---------------------------------------------------------------------------


def save_results(results: list[dict]) -> None:
    """Write evaluation results to CSV."""
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["checkpoint", "model", "target", "stat", "mae", "n_players"]
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nResults saved to {RESULTS_CSV}")


# ---------------------------------------------------------------------------
# Step 7: Print human-readable summary
# ---------------------------------------------------------------------------


def print_summary(results: list[dict]) -> None:
    """Print summary answering the four key questions."""
    if not results:
        print("No results to summarize.")
        return

    # Group results
    from collections import defaultdict

    # (model, target) -> list of mae values
    model_target_maes: dict[tuple, list] = defaultdict(list)
    # (model, target, stat) -> list of mae values
    model_stat_maes: dict[tuple, list] = defaultdict(list)
    # (checkpoint, model, target) -> list of mae values
    cp_model_maes: dict[tuple, list] = defaultdict(list)

    for row in results:
        mt = (row["model"], row["target"])
        model_target_maes[mt].append(row["mae"])
        mts = (row["model"], row["target"], row["stat"])
        model_stat_maes[mts].append(row["mae"])
        cmt = (row["checkpoint"], row["model"], row["target"])
        cp_model_maes[cmt].append(row["mae"])

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("inf")

    print("\n" + "=" * 80)
    print("RECENCY WEIGHTING BACKTEST SUMMARY")
    print("=" * 80)

    # --- Question 1: Overall best model ---
    print("\n1. OVERALL: Which model has lowest average MAE?\n")
    for target in ("next_week", "rest_of_season"):
        rankings = sorted(
            [m for m in [mn for mn, _ in MODELS]],
            key=lambda m: avg(model_target_maes.get((m, target), [])),
        )
        print(f"   Target: {target}")
        for rank, model in enumerate(rankings, 1):
            mae_avg = avg(model_target_maes.get((model, target), []))
            print(f"     {rank}. {model:<22} avg MAE = {mae_avg:.5f}")
        print()

    # --- Question 2: Per-stat benefit ---
    print("2. PER-STAT: Stats that benefit most from recency weighting\n")
    all_stats = HITTER_STATS + PITCHER_STATS
    for target in ("next_week", "rest_of_season"):
        print(f"   Target: {target}")
        for stat in all_stats:
            baseline = avg(model_stat_maes.get(("preseason", target, stat), []))
            best_mae = float("inf")
            best_model = "preseason"
            for model, _ in MODELS:
                m = avg(model_stat_maes.get((model, target, stat), []))
                if m < best_mae:
                    best_mae = m
                    best_model = model
            if baseline > 0:
                pct_improvement = (baseline - best_mae) / baseline * 100
                print(
                    f"     {stat:<14} baseline={baseline:.5f}  "
                    f"best={best_mae:.5f} ({best_model}, {pct_improvement:+.1f}%)"
                )
        print()

    # --- Question 3: Early vs late season ---
    print("3. PER-CHECKPOINT: Does recency help more early vs late season?\n")
    for target in ("next_week", "rest_of_season"):
        print(f"   Target: {target}")
        print(f"   {'Checkpoint':<12}", end="")
        for model, _ in MODELS:
            print(f"  {model[:10]:>10}", end="")
        print()
        for cp in CHECKPOINTS:
            print(f"   {cp:<12}", end="")
            for model, _ in MODELS:
                m = avg(cp_model_maes.get((cp, model, target), []))
                print(f"  {m:>10.5f}", end="")
            print()
        print()

    # --- Question 4: Conclusion ---
    print("4. CONCLUSION\n")
    for target in ("next_week", "rest_of_season"):
        best_model = min(
            [mn for mn, _ in MODELS],
            key=lambda m: avg(model_target_maes.get((m, target), [])),
        )
        best_mae = avg(model_target_maes.get((best_model, target), []))
        baseline_mae = avg(model_target_maes.get(("preseason", target), []))
        pct = (baseline_mae - best_mae) / baseline_mae * 100 if baseline_mae > 0 else 0
        print(
            f"   {target.upper():<10}: Best model = {best_model} "
            f"(MAE {best_mae:.5f}, {pct:+.1f}% vs preseason)"
        )

    recommendation = "\n   Recommendation: "
    nw_best = min(
        [mn for mn, _ in MODELS],
        key=lambda m: avg(model_target_maes.get((m, "next_week"), [])),
    )
    ros_best = min(
        [mn for mn, _ in MODELS],
        key=lambda m: avg(model_target_maes.get((m, "rest_of_season"), [])),
    )
    if nw_best == ros_best:
        recommendation += f"Build recency weighting using '{nw_best}' for both targets."
    else:
        recommendation += (
            f"Use '{nw_best}' for short-term (next week) and "
            f"'{ros_best}' for long-term (ROS) predictions."
        )
    print(recommendation)
    print()


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Recency Weighting Backtest — 2025 Season")
    print("=" * 60)

    # Step 1: Load projections
    projections = load_preseason_projections()

    # Step 2: Match draft names
    matched_players = match_draft_to_projections(projections)

    # Step 3: Fetch game logs
    game_logs = fetch_game_logs(matched_players)

    # Step 4 + 5: Run evaluation
    results = run_evaluation(matched_players, game_logs)

    # Step 6: Save CSV
    save_results(results)

    # Step 7: Print summary
    print_summary(results)

    print("Done.")


if __name__ == "__main__":
    main()
