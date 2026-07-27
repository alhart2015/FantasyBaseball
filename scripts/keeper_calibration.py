"""Run the #266 keeper-value calibration study.

Measures how much of a season's performance surprise should carry forward into a
projection that has not seen that season:

    updated_2027 = ZiPS_2027 + k * shrink * (actual_2026 - ZiPS_2026)

One coefficient per folded quantity, fit by leave-one-pair-out over the three
usable year pairs and scored against the two endpoints (k=0, ignore the season;
k=1, transfer the whole shrunk surprise). Metric, gates and shrink constants are
pre-registered in docs/superpowers/keeper-calibration-finding-2026-07-27.md.

Writes data/analysis/keeper_calibration_<player_type>.csv plus an n0 sensitivity
sweep. MLB pulls cache under data/cache/keeper_calibration/ (gitignored).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fantasy_baseball.keepers.actuals import (
    HITTER_PT,
    HITTER_RATES,
    PITCHER_PT,
    PITCHER_RATES,
)
from fantasy_baseball.keepers.calibration import (
    PAIR_YEARS,
    Fitted,
    FullTransfer,
    ShrunkTransfer,
    YearPair,
    ZeroTransfer,
    build_pairs,
    gated,
    leave_one_out,
    survivorship,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "keeper_calibration"
PROJECTIONS_ROOT = REPO_ROOT / "data" / "projections"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"

CHOSEN = ShrunkTransfer.name
ENDPOINT_NAMES = (ZeroTransfer.name, FullTransfer.name)


@dataclass(frozen=True)
class Settings:
    """Per-player-type study policy, pre-registered in the finding document.

    Section A.3 fixes the gates, A.4 the shrink constants and the sensitivity
    grid. A frozen dataclass rather than a dict so the heterogeneous field types
    survive to the reader and the type checker.
    """

    columns: tuple[str, ...]
    pt_col: str
    gate: float
    n0: float
    n0_grid: tuple[float, ...]
    survivorship_thresholds: tuple[float, ...]


# PAIR_YEARS tops out at 2024 and targets 2025, so NO 2026 pull ever happens --
# which matters because fetch_or_cache never invalidates and would freeze an
# in-progress season permanently (spec 6.5). Do not extend `years` past the last
# complete season without giving 2026 a date-stamped cache path.
SETTINGS = {
    "hitter": Settings(
        columns=(*HITTER_RATES, HITTER_PT),
        pt_col=HITTER_PT,
        gate=100.0,
        n0=200.0,
        n0_grid=(100.0, 200.0, 400.0),
        survivorship_thresholds=(100.0, 300.0),
    ),
    "pitcher": Settings(
        columns=(*PITCHER_RATES, PITCHER_PT),
        pt_col=PITCHER_PT,
        gate=50.0,
        n0=50.0,
        n0_grid=(25.0, 50.0, 100.0),
        survivorship_thresholds=(50.0, 100.0),
    ),
}


def build_report(
    pairs: list[YearPair],
    columns: tuple[str, ...],
    n0: float,
    *,
    pt_col: str | None = None,
    gate: float = 0.0,
) -> pd.DataFrame:
    """One row per (estimator, coefficient), with a per-coefficient verdict.

    Acceptance is per coefficient and by majority of held-out pairs, never pooled
    across coefficients and never by mean error (finding A.2, spec 6.6). Section
    6.3 already predicts that one coefficient -- SB/PA, across the 2023 rules
    break -- will misbehave for a known exogenous reason, and a pooled verdict
    would either sink ten good coefficients or smuggle that one through.
    """
    rows: list[dict[str, object]] = []
    # The headline fit uses the SAME gated sample leave_one_out scores, or the
    # shipped number would be fit on rows the study excluded. Gate once: it
    # depends on neither the column nor the estimator.
    kept = [gated(p, gate) for p in pairs]
    for column in columns:
        # The playing-time coefficient is fit and scored unshrunk and unweighted.
        # Shrinking it would damp an injury signal in proportion to the playing
        # time the injury suppressed (spec 5.3); weighting it by target playing
        # time would delete every non-survivor (finding A.1).
        rate_like = column != pt_col
        folds = {
            est.name: leave_one_out(
                est, pairs, column, n0, gate=gate, shrunk=rate_like, weighted=rate_like
            )
            for est in (ShrunkTransfer(), ZeroTransfer(), FullTransfer())
        }
        chosen_full = ShrunkTransfer().fit(kept, column, n0, shrunk=rate_like, weighted=rate_like)
        verdict, wins = _verdict(folds)
        for name, fold in folds.items():
            rows.append(_summarize(name, column, fold, verdict, wins, chosen_full))
    return pd.DataFrame(rows)


def _verdict(folds: dict[str, pd.DataFrame]) -> tuple[str, dict[str, int]]:
    """`pass` if the chosen estimator beats BOTH endpoints on a majority of pairs.

    Otherwise fall back to whichever endpoint won more pairs head-to-head, with
    mean held-out error as the tie-break (spec 6.6).
    """
    errors = {name: fold.set_index("held_out_year")["error"] for name, fold in folds.items()}
    chosen = errors[CHOSEN]
    n_pairs = len(chosen)
    wins = {name: int((chosen < errors[name]).sum()) for name in ENDPOINT_NAMES}
    if all(w * 2 > n_pairs for w in wins.values()):
        return "pass", wins
    zero_wins = int((errors["k=0"] < errors["k=1"]).sum())
    if zero_wins * 2 > n_pairs:
        best = "k=0"
    elif zero_wins * 2 < n_pairs:
        best = "k=1"
    else:
        best = min(ENDPOINT_NAMES, key=lambda name: float(errors[name].mean()))
    return f"fallback:{best}", wins


def _summarize(
    name: str,
    column: str,
    fold: pd.DataFrame,
    verdict: str,
    wins: dict[str, int],
    chosen_full: Fitted,
) -> dict[str, object]:
    """One report row for one (estimator, coefficient)."""
    row: dict[str, object] = {
        "estimator": name,
        "column": column,
        "verdict": verdict,
        "mean_error": float(fold["error"].mean()),
        "wins_vs_k0": wins["k=0"],
        "wins_vs_k1": wins["k=1"],
        "n_pairs": len(fold),
    }
    is_chosen = name == CHOSEN
    for _, f in fold.iterrows():
        year = int(f["held_out_year"])
        row[f"err_{year}"] = float(f["error"])
        if is_chosen:
            # Stability across the three pairs (spec 6.2 requirement 4): the
            # per-fold raw fits, reported rather than averaged.
            row[f"k_raw_ex_{year}"] = float(f["param_k_raw"])
    if not is_chosen:
        row["k_full"] = float(fold["param_k"].iloc[0])
        return row
    params, raw = chosen_full.params, fold["param_k_raw"]
    row.update(
        {
            "k_full": params["k"],
            "k_raw_full": params["k_raw"],
            "ci_lo_full": params["ci_lo"],
            "ci_hi_full": params["ci_hi"],
            "n_fit_full": params["n_fit"],
            # The calibration-only level term (spec 6.2 requirement 12). Its
            # PRODUCTION value is 0 and `predict` never applies it; it is
            # reported so the size of the level offset it absorbed is visible.
            "c_fit_full": params["c_fit"],
            "k_raw_min": float(raw.min()),
            "k_raw_max": float(raw.max()),
            "k_raw_spread": float(raw.max() - raw.min()),
            # Requirement 7: an amplifying fit is flagged, never silent.
            "out_of_range": bool(params["k_raw"] < 0.0 or params["k_raw"] > 1.0),
        }
    )
    return row


def n0_sweep(
    pairs: list[YearPair],
    columns: tuple[str, ...],
    grid: tuple[float, ...],
    *,
    pt_col: str | None = None,
    gate: float = 0.0,
) -> pd.DataFrame:
    """The pre-registered sensitivity grid (finding A.4). Reported, never used to
    select the headline coefficient -- `k` and the shrink enter multiplicatively,
    so `k` is only interpretable against a stated n0."""
    frames = []
    for n0 in grid:
        report = build_report(pairs, columns, n0, pt_col=pt_col, gate=gate)
        chosen = report.loc[report["estimator"] == CHOSEN].copy()
        chosen.insert(0, "n0", n0)
        frames.append(chosen[["n0", "column", "k_full", "k_raw_full", "mean_error", "verdict"]])
    return pd.concat(frames, ignore_index=True)


def _print_table(title: str, frame: pd.DataFrame) -> None:
    print()
    print(title)
    print("-" * len(title))
    print(frame.to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player-type", choices=("hitter", "pitcher", "both"), default="both")
    parser.add_argument("--skip-sweep", action="store_true", help="omit the n0 sensitivity grid")
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    types = ("hitter", "pitcher") if args.player_type == "both" else (args.player_type,)
    survivorship_frames = []

    for player_type in types:
        cfg = SETTINGS[player_type]
        pairs = build_pairs(player_type, CACHE_DIR, PROJECTIONS_ROOT, years=PAIR_YEARS)

        for threshold in cfg.survivorship_thresholds:
            table = survivorship(pairs, threshold)
            _print_table(f"{player_type}: survivorship at threshold {threshold}", table)
            table.insert(0, "threshold", threshold)
            table.insert(0, "player_type", player_type)
            survivorship_frames.append(table)

        report = build_report(pairs, cfg.columns, cfg.n0, pt_col=cfg.pt_col, gate=cfg.gate)
        out = ANALYSIS_DIR / f"keeper_calibration_{player_type}.csv"
        report.to_csv(out, index=False)
        _print_table(f"{player_type}: leave-one-pair-out (gate {cfg.gate}, n0 {cfg.n0})", report)
        print(f"wrote {out}")

        if not args.skip_sweep:
            sweep = n0_sweep(pairs, cfg.columns, cfg.n0_grid, pt_col=cfg.pt_col, gate=cfg.gate)
            sweep_out = ANALYSIS_DIR / f"keeper_calibration_{player_type}_n0_sweep.csv"
            sweep.to_csv(sweep_out, index=False)
            _print_table(f"{player_type}: n0 sensitivity", sweep)
            print(f"wrote {sweep_out}")

    # The finding's Part A cites this table as the evidence that pair membership
    # is not preconditioned on survival (0.755/0.777/0.795, not 0.84-0.87). It
    # must be regenerable by the script, not assembled by hand.
    survivorship_out = ANALYSIS_DIR / "keeper_calibration_survivorship.csv"
    pd.concat(survivorship_frames, ignore_index=True).to_csv(survivorship_out, index=False)
    print(f"wrote {survivorship_out}")


if __name__ == "__main__":
    main()
