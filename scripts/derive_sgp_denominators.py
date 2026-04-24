"""Derive empirical SGP denominators from this league's historical standings.

Reads end-of-season standings from ``data/historical_standings.json`` and
computes the average adjacent-rank gap per category across multiple years.
This is the ground-truth methodology: "given how THIS league has actually
played, how many stats does it take to move one standings spot?"

Usage:
    python scripts/derive_sgp_denominators.py
    python scripts/derive_sgp_denominators.py --years 2024 2025

The output is a YAML snippet suitable for pasting into ``config/league.yaml``
under ``sgp_denominators:``. No files are written; review and apply manually.

A Monte Carlo fallback is available via --mc for seasons without historical
data (e.g., a new league's first year), but historical standings are
preferred when available.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.data.db import get_connection
from fantasy_baseball.draft.board import build_draft_board
from fantasy_baseball.utils.constants import ALL_CATEGORIES, STAT_VARIANCE, Category

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
STANDINGS_PATH = PROJECT_ROOT / "data" / "historical_standings.json"

# Per-team closer count used in the Monte Carlo fallback. Sorting pitchers
# by VAR alone under-picks closers because their low IP depresses counting-
# stat VAR; we split the pool so each team gets a realistic share.
#
# Floor of 10 SV (not utils.constants.CLOSER_SV_THRESHOLD of 20) because the
# pool needs enough closer candidates to cover CLOSERS_PER_TEAM * num_teams;
# a stricter floor risks under-filling the closer pool in years where
# projected SV is fragmented across committee setups.
CLOSERS_PER_TEAM = 2
CLOSER_SV_FLOOR = 10


def _adjacent_gap(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    ordered = sorted(values)
    return (ordered[-1] - ordered[0]) / (len(values) - 1)


def _format_denom(cat: Category, value: float) -> str:
    if cat is Category.AVG:
        return f"{value:.4f}"
    if cat in {Category.ERA, Category.WHIP}:
        return f"{value:.3f}"
    return f"{round(value):d}"


# ---------- Historical-standings derivation (preferred) ----------


def derive_from_history(years: list[str]) -> dict[Category, tuple[float, list[float]]]:
    """Compute per-category gap averaged across `years`.

    Returns ``{cat: (mean_gap, [per_year_gaps])}``.
    """
    with open(STANDINGS_PATH) as f:
        data = json.load(f)

    gaps_by_cat: dict[Category, list[float]] = {cat: [] for cat in ALL_CATEGORIES}
    for year in years:
        standings = data[year]["standings"]
        for cat in ALL_CATEGORIES:
            vals = [t["stats"][cat.value] for t in standings]
            gaps_by_cat[cat].append(_adjacent_gap(vals))
    return {cat: (float(np.mean(gs)), gs) for cat, gs in gaps_by_cat.items()}


# ---------- Monte Carlo fallback ----------


def _active_hitter_slots(roster_slots: dict[str, int]) -> int:
    return sum(v for k, v in roster_slots.items() if k not in {"P", "SP", "RP", "BN", "IL", "NA"})


def _active_pitcher_slots(roster_slots: dict[str, int]) -> int:
    return sum(v for k, v in roster_slots.items() if k in {"P", "SP", "RP"})


def _serpentine_deal(players, num_teams: int, picks_per_team: int):
    teams: list[list] = [[] for _ in range(num_teams)]
    for pick_num in range(num_teams * picks_per_team):
        rnd = pick_num // num_teams
        pos = pick_num % num_teams
        team_idx = pos if rnd % 2 == 0 else (num_teams - 1 - pos)
        teams[team_idx].append(players[pick_num])
    return teams


def _sample_factor(rng: np.random.Generator, stat_key: str) -> float:
    cv = STAT_VARIANCE.get(stat_key, 0.0)
    if cv <= 0:
        return 1.0
    return max(0.0, float(rng.normal(1.0, cv)))


def _sum_stats(roster, keys: list[str], rng: np.random.Generator | None = None) -> dict[str, float]:
    totals = {k: 0.0 for k in keys}
    for p in roster:
        for k in keys:
            raw = float(p.get(k, 0) or 0)
            factor = _sample_factor(rng, k) if rng is not None else 1.0
            totals[k] += raw * factor
    return totals


def _compute_team_categories(hitters, pitchers, rng=None) -> dict[Category, float]:
    h = _sum_stats(hitters, ["r", "hr", "rbi", "sb", "h", "ab"], rng)
    p = _sum_stats(pitchers, ["w", "k", "sv", "ip", "er", "bb", "h_allowed"], rng)
    avg = h["h"] / h["ab"] if h["ab"] > 0 else 0.0
    era = 9.0 * p["er"] / p["ip"] if p["ip"] > 0 else 0.0
    whip = (p["bb"] + p["h_allowed"]) / p["ip"] if p["ip"] > 0 else 0.0
    return {
        Category.R: h["r"],
        Category.HR: h["hr"],
        Category.RBI: h["rbi"],
        Category.SB: h["sb"],
        Category.AVG: avg,
        Category.W: p["w"],
        Category.K: p["k"],
        Category.SV: p["sv"],
        Category.ERA: era,
        Category.WHIP: whip,
    }


def derive_from_mc(trials: int, seed: int) -> dict[str, float]:
    config = load_config(CONFIG_PATH)
    conn = get_connection()
    board = build_draft_board(
        conn=conn,
        sgp_overrides=config.sgp_overrides or None,
        roster_slots=config.roster_slots or None,
        num_teams=config.num_teams,
    )
    conn.close()

    hitters_df = board[board["player_type"] == "hitter"]
    pitchers_df = board[board["player_type"] == "pitcher"]

    num_teams = config.num_teams
    h_slots = _active_hitter_slots(config.roster_slots)
    p_slots = _active_pitcher_slots(config.roster_slots)

    closer_picks = num_teams * CLOSERS_PER_TEAM
    non_closer_picks_per_team = p_slots - CLOSERS_PER_TEAM
    non_closer_picks = num_teams * non_closer_picks_per_team
    closer_mask = pitchers_df["sv"] >= CLOSER_SV_FLOOR
    closer_pool = pitchers_df[closer_mask].sort_values("sv", ascending=False).head(closer_picks)
    non_closer_pool = (
        pitchers_df[~closer_mask].sort_values("var", ascending=False).head(non_closer_picks)
    )
    hitter_pool = hitters_df.sort_values("var", ascending=False).head(num_teams * h_slots)

    hitter_rows = [row for _, row in hitter_pool.iterrows()]
    closer_rows = [row for _, row in closer_pool.iterrows()]
    non_closer_rows = [row for _, row in non_closer_pool.iterrows()]
    hitter_teams = _serpentine_deal(hitter_rows, num_teams, h_slots)
    closer_teams = _serpentine_deal(closer_rows, num_teams, CLOSERS_PER_TEAM)
    non_closer_teams = _serpentine_deal(non_closer_rows, num_teams, non_closer_picks_per_team)
    pitcher_teams = [closer_teams[i] + non_closer_teams[i] for i in range(num_teams)]

    rng = np.random.default_rng(seed)
    gap_accum: dict[Category, list[float]] = {cat: [] for cat in ALL_CATEGORIES}
    for _ in range(trials):
        team_cats = [
            _compute_team_categories(hitter_teams[i], pitcher_teams[i], rng)
            for i in range(num_teams)
        ]
        for cat in ALL_CATEGORIES:
            gap_accum[cat].append(_adjacent_gap([t[cat] for t in team_cats]))
    return {cat: float(np.mean(gap_accum[cat])) for cat in ALL_CATEGORIES}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--years",
        nargs="+",
        default=["2023", "2024", "2025"],
        help="Years to average over (default: 2023 2024 2025)",
    )
    parser.add_argument(
        "--mc",
        action="store_true",
        help="Use Monte Carlo projection-based derivation instead of history",
    )
    parser.add_argument("--trials", type=int, default=1000, help="MC trials (default 1000)")
    parser.add_argument("--seed", type=int, default=42, help="MC RNG seed (default 42)")
    args = parser.parse_args()

    if args.mc:
        print(f"# Monte Carlo derivation: {args.trials} trials")
        print("# NOTE: projection-based MC captures only projection-CV variance.")
        print("# Injury/role-change/streaming variance is NOT captured.")
        print("# Prefer historical-standings derivation when available.")
        print()
        denoms = derive_from_mc(args.trials, args.seed)
    else:
        print(f"# Derived from historical standings: years {args.years}")
        print(f"# Source: {STANDINGS_PATH.relative_to(PROJECT_ROOT)}")
        print(f"# Method: mean(adjacent-rank gap) across {len(args.years)} seasons")
        print()
        history = derive_from_history(args.years)
        print("# Per-year gaps:")
        for cat in ALL_CATEGORIES:
            mean, per_year = history[cat]
            per_year_str = " ".join(
                f"{y}={g:.3f}" for y, g in zip(args.years, per_year, strict=True)
            )
            print(f"#   {cat.value:>4}: {per_year_str}  -> mean {mean:.4f}")
        print()
        denoms = {cat: history[cat][0] for cat in ALL_CATEGORIES}

    print("sgp_denominators:")
    for cat in ALL_CATEGORIES:
        print(f"  {cat.value}: {_format_denom(cat, denoms[cat])}")


if __name__ == "__main__":
    main()
