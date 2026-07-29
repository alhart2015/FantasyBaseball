"""Rank keeper candidates by blending 2026 value, true-talent skills and age.

Reads what `fetch_keeper_skills.py` cached, computes each player's actual roto
value (SGP) for the season, and blends the three families in percentile space
using the weights fitted in `keepers/composite.py`.

Writes `data/cache/keeper_skills/keeper_rankings_{year}.csv` with, per player:

    value_pct   percentile of actual SGP this season
    skill_pct   percentile of the true-talent stats
    age_pct     percentile of age, younger better
    composite   the fitted blend -- the ranking column
    reg_gap     value_pct - skill_pct; positive = outran his peripherals
    zips_pct    percentile of ZiPS out-year projected SGP, UNVALIDATED (below)

Hitters and pitchers are ranked in separate pools, so a percentile compares a
player to his own kind and never across.

The ZiPS column is deliberately outside the composite. The out-year files are
snapshots generated before the season started, so they carry no information
about it, and there is no historical out-year vintage to backtest a weight
against. Read it as a second opinion, not as a fitted input.

Usage:
    python scripts/keeper_rankings.py
    python scripts/keeper_rankings.py --backtest        # regenerate the study
    python scripts/keeper_rankings.py --year 2025
"""

from __future__ import annotations

import argparse
import sys
from itertools import product
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.keepers.actuals import index_by_mlbam, innings_to_float
from fantasy_baseball.keepers.composite import (
    FITTED_WEIGHTS,
    composite,
    percentile,
    regression_gap,
    skill_percentile,
)
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.name_utils import normalize_name

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
SKILLS_DIR = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"

# Below these a percentile is noise: the skills module regresses nothing, so a
# 3-inning pitcher can post the league's best ERA-. See its module docstring.
MIN_PT = {"hitter": 250, "pitcher": 50}
BACKTEST_FIT_YEARS = (2022, 2023)
BACKTEST_HOLDOUT = 2024


def _raw(year: int, name: str) -> pd.DataFrame:
    return pd.read_csv(SKILLS_DIR / f"raw_{year}" / name)


def _sgp(lines: pd.DataFrame, denoms) -> pd.Series:
    return lines.apply(lambda row: calculate_player_sgp(row, denoms=denoms), axis=1)


def season_value(year: int, kind: str, denoms) -> pd.DataFrame:
    """Actual roto value, age and playing time for `year`, indexed by mlbam_id.

    Taken from the same BBRef pulls the skills come from, so both sides of the
    join share one provenance and one id.
    """
    if kind == "hitter":
        frame = index_by_mlbam(_raw(year, f"bref_batting_{year}.v2.csv"), "mlbID")
        pt = pd.to_numeric(frame["PA"], errors="coerce")
        lines = pd.DataFrame(
            {
                "r": pd.to_numeric(frame["R"], errors="coerce"),
                "hr": pd.to_numeric(frame["HR"], errors="coerce"),
                "rbi": pd.to_numeric(frame["RBI"], errors="coerce"),
                "sb": pd.to_numeric(frame["SB"], errors="coerce"),
                "ab": pd.to_numeric(frame["AB"], errors="coerce"),
                "avg": pd.to_numeric(frame["BA"], errors="coerce"),
            }
        ).fillna(0.0)
        lines["player_type"] = PlayerType.HITTER
    else:
        frame = index_by_mlbam(_raw(year, f"bref_pitching_{year}.v2.csv"), "mlbID")
        pt = frame["IP"].map(innings_to_float)
        lines = pd.DataFrame(
            {
                "w": pd.to_numeric(frame["W"], errors="coerce"),
                "k": pd.to_numeric(frame["SO"], errors="coerce"),
                "sv": pd.to_numeric(frame["SV"], errors="coerce"),
                "ip": pt,
                "era": pd.to_numeric(frame["ERA"], errors="coerce"),
                "whip": pd.to_numeric(frame["WHIP"], errors="coerce"),
            }
        ).fillna(0.0)
        lines["player_type"] = PlayerType.PITCHER

    return pd.DataFrame(
        {
            "age": pd.to_numeric(frame["Age"], errors="coerce"),
            "pt": pt,
            "sgp": _sgp(lines, denoms),
        },
        index=frame.index,
    )


def zips_out_year_sgp(year: int, kind: str, denoms) -> pd.Series:
    """SGP of the ZiPS projection for `year`, read straight from the export.

    Not routed through `keepers.vintages`: that decomposition drops SV, which
    would leave every closer looking replacement-level.
    """
    stem = "hitters" if kind == "hitter" else "pitchers"
    matches = sorted((PROJECTIONS_DIR / str(year)).glob(f"zips-{stem}*.csv"))
    if not matches:
        return pd.Series(dtype=float)
    frame = index_by_mlbam(pd.read_csv(matches[-1], encoding="utf-8-sig"), "MLBAMID")
    numeric = {c: pd.to_numeric(frame[c], errors="coerce") for c in frame.columns if c != "Name"}
    if kind == "hitter":
        ab = numeric["AB"]
        lines = pd.DataFrame(
            {
                "r": numeric["R"],
                "hr": numeric["HR"],
                "rbi": numeric["RBI"],
                "sb": numeric["SB"],
                "ab": ab,
                "avg": (numeric["H"] / ab.where(ab > 0)),
            }
        ).fillna(0.0)
        lines["player_type"] = PlayerType.HITTER
    else:
        ip = numeric["IP"]
        lines = pd.DataFrame(
            {
                "w": numeric["W"],
                "k": numeric["SO"],
                "sv": numeric["SV"],
                "ip": ip,
                "era": (9.0 * numeric["ER"] / ip.where(ip > 0)),
                "whip": ((numeric["BB"] + numeric["H"]) / ip.where(ip > 0)),
            }
        ).fillna(0.0)
        lines["player_type"] = PlayerType.PITCHER
    return _sgp(lines, denoms)


def build(year: int, kind: str, denoms, keepers: dict[str, str]) -> pd.DataFrame:
    value = season_value(year, kind, denoms)
    skills = pd.read_csv(SKILLS_DIR / f"{kind}_skills_{year}.csv").set_index("mlbam_id")
    frame = value.join(skills, how="inner", rsuffix="_sk")

    qualified = frame[frame["pt"] >= MIN_PT[kind]].copy()
    qualified["value_pct"] = percentile(qualified["sgp"])
    qualified["skill_pct"] = skill_percentile(qualified, kind)
    qualified["age_pct"] = percentile(qualified["age"], higher_is_better=False)
    qualified["composite"] = composite(
        qualified["value_pct"], qualified["skill_pct"], qualified["age_pct"], kind
    )
    qualified["reg_gap"] = regression_gap(qualified["value_pct"], qualified["skill_pct"])

    # Rank the projection WITHIN the qualified pool, not within all ~1900 ZiPS
    # rows. Most of that file is minor leaguers, so ranking there would put every
    # established regular above the 90th percentile and say nothing.
    zips = zips_out_year_sgp(year + 1, kind, denoms).reindex(qualified.index)
    qualified["zips_pct"] = percentile(zips)

    qualified["keeper_of"] = [keepers.get(normalize_name(str(n)), "") for n in qualified["name"]]
    qualified["rank"] = qualified["composite"].rank(ascending=False, method="min").astype(int)
    return qualified.sort_values("composite", ascending=False)


# --- backtest -------------------------------------------------------------


def _transition(year: int, kind: str, denoms) -> pd.DataFrame:
    """Features observed in `year` against the SGP percentile realized in year+1."""

    def observed(y: int) -> pd.DataFrame:
        value = season_value(y, kind, denoms)
        skills = pd.read_csv(SKILLS_DIR / f"{kind}_skills_{y}.csv").set_index("mlbam_id")
        return value.join(skills, how="inner", rsuffix="_sk")

    now, nxt = observed(year), observed(year + 1)
    feat = now[now["pt"] >= MIN_PT[kind]].copy()
    feat["value_pct"] = percentile(feat["sgp"])
    feat["skill_pct"] = skill_percentile(feat, kind)
    feat["age_pct"] = percentile(feat["age"], higher_is_better=False)
    # A player who does not appear next season scores 0 rather than dropping
    # out: vanishing is the outcome a keeper decision most wants to avoid.
    feat["target"] = percentile(nxt["sgp"]).reindex(feat.index).fillna(0.0)
    return feat.dropna(subset=["value_pct", "skill_pct", "age_pct"])


def _rho(frame: pd.DataFrame, weights: tuple[float, float, float], kind: str) -> float:
    blended = composite(
        frame["value_pct"], frame["skill_pct"], frame["age_pct"], kind, weights=weights
    )
    return float(blended.corr(frame["target"], method="spearman"))


def run_backtest(denoms) -> None:
    for kind in ("hitter", "pitcher"):
        fit = [_transition(y, kind, denoms) for y in BACKTEST_FIT_YEARS]
        hold = _transition(BACKTEST_HOLDOUT, kind, denoms)
        print(
            f"\n{'=' * 70}\n{kind.upper()}  fit={list(BACKTEST_FIT_YEARS)} holdout={BACKTEST_HOLDOUT}"
        )

        grid = [round(i / 10, 1) for i in range(11)]
        rows = []
        for w_value, w_skill in product(grid, grid):
            w_age = round(1.0 - w_value - w_skill, 3)
            if not 0.0 <= w_age <= 0.5:
                continue
            weights = (w_value, w_skill, w_age)
            rows.append(
                {
                    "w_value": w_value,
                    "w_skill": w_skill,
                    "w_age": w_age,
                    "fit_rho": sum(_rho(f, weights, kind) for f in fit) / len(fit),
                    "holdout_rho": _rho(hold, weights, kind),
                }
            )
        table = pd.DataFrame(rows).sort_values("fit_rho", ascending=False)
        print(table.head(5).round(4).to_string(index=False))
        print("\n  baselines (holdout):")
        for label, weights in [
            ("shipped weights", FITTED_WEIGHTS[kind]),
            ("value only", (1.0, 0.0, 0.0)),
            ("skill only", (0.0, 1.0, 0.0)),
        ]:
            print(f"    {label:16s} rho = {_rho(hold, weights, kind):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, help="defaults to config season_year")
    parser.add_argument("--backtest", action="store_true", help="regenerate the weight study")
    parser.add_argument("--top", type=int, default=20, help="rows to print per pool")
    args = parser.parse_args()

    config = load_config(CONFIG_PATH)
    year = args.year or config.season_year
    denoms = get_sgp_denominators(getattr(config, "sgp_overrides", None))

    if args.backtest:
        run_backtest(denoms)
        return 0

    keepers = {normalize_name(k["name"]): k["team"] for k in (config.keepers or [])}
    for kind in ("hitter", "pitcher"):
        table = build(year, kind, denoms, keepers)
        out_path = SKILLS_DIR / f"keeper_rankings_{kind}_{year}.csv"
        table.to_csv(out_path)
        shown = [
            "rank",
            "name",
            "age",
            "pt",
            "sgp",
            "value_pct",
            "skill_pct",
            "composite",
            "reg_gap",
            "zips_pct",
            "keeper_of",
        ]
        print(
            f"\n=== {kind.upper()} ({len(table)} qualified, >= {MIN_PT[kind]}) -> {out_path.name}"
        )
        print(table[shown].head(args.top).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
