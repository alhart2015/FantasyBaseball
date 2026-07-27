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
    FittedK,
    FullTransfer,
    ShrunkTransfer,
    YearPair,
    ZeroTransfer,
    build_pairs,
    gated,
    leave_one_out,
    survivorship,
)
from fantasy_baseball.keepers.coefficients import (
    CHOSEN_ESTIMATOR,
    FALLBACK_PREFIX,
    PASS_VERDICT,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache" / "keeper_calibration"
PROJECTIONS_ROOT = REPO_ROOT / "data" / "projections"
ANALYSIS_DIR = REPO_ROOT / "data" / "analysis"

# The library owns this string: it is the value `policy_from_study` matches on
# when reading the report back. `test_chosen_estimator_name_matches_the_library`
# binds it to ShrunkTransfer.name so the two cannot drift.
CHOSEN = CHOSEN_ESTIMATOR
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
    pt_col: str | None,
    gate: float,
) -> pd.DataFrame:
    """One row per (estimator, coefficient), with a per-coefficient verdict.

    `pt_col` and `gate` are REQUIRED, with no defaults, deliberately. Omitting
    them used to be possible and silently reproduced both bugs this study already
    fixed once: `gate=0.0` fits on rows the study excludes, and `pt_col=None`
    fits the playing-time column shrunk and weighted, which zero-weights every
    non-survivor. Pass `pt_col=None` explicitly if the frames genuinely carry no
    playing-time column.

    Acceptance is per coefficient and by majority of held-out pairs, never pooled
    across coefficients and never by mean error (finding A.2, spec 6.6). Section
    6.3 already predicts that one coefficient -- SB/PA, across the 2023 rules
    break -- will misbehave for a known exogenous reason, and a pooled verdict
    would either sink ten good coefficients or smuggle that one through.
    """
    rows: list[dict[str, object]] = []
    # Gate once. Every consumer below sees the same rows, and `leave_one_out` no
    # longer re-derives the gate per column per estimator.
    kept = [gated(p, gate) for p in pairs]
    for column in columns:
        # The playing-time coefficient is fit and scored unshrunk and unweighted.
        # Shrinking it would damp an injury signal in proportion to the playing
        # time the injury suppressed (spec 5.3); weighting it by target playing
        # time would delete every non-survivor (finding A.1).
        rate_like = column != pt_col
        folds = {
            est.name: leave_one_out(est, kept, column, n0, shrunk=rate_like, weighted=rate_like)
            for est in (ShrunkTransfer(), ZeroTransfer(), FullTransfer())
        }
        chosen_full = ShrunkTransfer().fit(kept, column, n0, shrunk=rate_like, weighted=rate_like)
        verdict, wins = _verdict(folds)
        for name, fold in folds.items():
            row = _summarize(name, column, fold, verdict, wins, chosen_full)
            # The coefficients are only interpretable against the n0 and gate they
            # were fit under, so the artifact records them rather than leaving the
            # shipped constants and the study free to drift apart silently.
            row["n0"], row["gate"] = n0, gate
            rows.append(row)
    return pd.DataFrame(rows)


def _verdict(folds: dict[str, pd.DataFrame]) -> tuple[str, dict[str, int]]:
    """`pass` if the chosen estimator beats BOTH endpoints on a majority of pairs.

    Otherwise fall back to whichever endpoint won more pairs head-to-head, with
    mean held-out error as the tie-break (spec 6.6).
    """
    zero_name, full_name = ENDPOINT_NAMES
    errors = {name: fold.set_index("held_out_year")["error"] for name, fold in folds.items()}
    chosen = errors[CHOSEN]
    n_pairs = len(chosen)
    wins = {name: int((chosen < errors[name]).sum()) for name in ENDPOINT_NAMES}
    if all(w * 2 > n_pairs for w in wins.values()):
        return PASS_VERDICT, wins
    zero_wins = int((errors[zero_name] < errors[full_name]).sum())
    if zero_wins * 2 > n_pairs:
        best = zero_name
    elif zero_wins * 2 < n_pairs:
        best = full_name
    else:
        best = min(ENDPOINT_NAMES, key=lambda name: float(errors[name].mean()))
    return FALLBACK_PREFIX + best.removeprefix("k="), wins


def _summarize(
    name: str,
    column: str,
    fold: pd.DataFrame,
    verdict: str,
    wins: dict[str, int],
    chosen_full: FittedK,
) -> dict[str, object]:
    """One report row for one (estimator, coefficient)."""
    row: dict[str, object] = {
        "estimator": name,
        "column": column,
        "verdict": verdict,
        "mean_error": float(fold["error"].mean()),
        "wins_vs_k0": wins[ENDPOINT_NAMES[0]],
        "wins_vs_k1": wins[ENDPOINT_NAMES[1]],
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
            # Requirement 7: an amplifying fit is flagged, never silent. This must
            # consider EVERY fold, not just the full-sample fit -- k_ip's ex-2023
            # fold fits 1.024 and clamps while the full-sample fit sits at 0.970,
            # which a full-sample-only flag reported as clean.
            "out_of_range": bool(
                params["k_raw"] < 0.0
                or params["k_raw"] > 1.0
                or (raw < 0.0).any()
                or (raw > 1.0).any()
            ),
            "folds_clamped": int(((raw < 0.0) | (raw > 1.0)).sum()),
        }
    )
    return row


def n0_sweep(
    pairs: list[YearPair],
    columns: tuple[str, ...],
    grid: tuple[float, ...],
    *,
    pt_col: str,
    gate: float,
) -> pd.DataFrame:
    """The pre-registered sensitivity grid (finding A.4). Reported, never used to
    select the headline coefficient -- `k` and the shrink enter multiplicatively,
    so `k` is only interpretable against a stated n0."""
    frames = []
    for n0 in grid:
        report = build_report(pairs, columns, n0, pt_col=pt_col, gate=gate)
        chosen = report.loc[report["estimator"] == CHOSEN]
        # `build_report` already emits the n0 it was run at -- do not re-insert it.
        frames.append(chosen[["n0", "column", "k_full", "k_raw_full", "mean_error", "verdict"]])
    return pd.concat(frames, ignore_index=True)


def level_term_diagnostic(
    pairs: list[YearPair], pt_col: str, n0: float, gate: float
) -> pd.DataFrame:
    """Evidence for finding B.3: how much of the playing-time error is a LEVEL.

    Scores the shipped form against a variant that applies the fitted intercept.
    The intercept's production value is 0 (it corrects the ZiPS_Y-targets-year-Y
    gap, which ZiPS_2027 does not have), so this is a diagnostic and not a shipped
    option -- but it is the number that explains why `pa` falls back to k=1, and
    B.3 quotes it, so the script has to be able to regenerate it.
    """
    kept = [gated(p, gate) for p in pairs]
    rows: list[dict[str, object]] = []
    for est in (ZeroTransfer(), FullTransfer(), ShrunkTransfer(), _WithIntercept()):
        loo = leave_one_out(est, kept, pt_col, n0, shrunk=False, weighted=False)
        rows.append(
            {
                "column": pt_col,
                "estimator": est.name,
                "mean_error": float(loo["error"].mean()),
                **{f"err_{int(r.held_out_year)}": float(r.error) for r in loo.itertuples()},
            }
        )
    return pd.DataFrame(rows)


def playing_time_levels(pairs: list[YearPair], pt_col: str, gate: float) -> pd.DataFrame:
    """The systematic LEVEL in the playing-time residual (finding B.3).

    Separate from `level_term_diagnostic` because it answers a different question
    with a disjoint column set; merging them padded 44% of the artifact with
    blanks and made `estimator` carry a magic 'levels' value.
    """
    kept = [gated(p, gate) for p in pairs]
    base = pd.concat([p.base[pt_col] for p in kept])
    realized = pd.concat([p.realized_pt for p in kept])
    target_pt = pd.concat([p.target_pt for p in kept])
    return pd.DataFrame(
        [
            {
                "column": pt_col,
                "mean_zips_pt": float(base.mean()),
                "mean_actual_pt": float(realized.mean()),
                "mean_residual": float((realized - base).mean()),
                "mean_target_minus_base": float((target_pt - base).mean()),
                "frac_non_survivors": float((target_pt == 0).mean()),
            }
        ]
    )


class _WithIntercept(ShrunkTransfer):
    """Diagnostic only: applies the calibration intercept at predict time.

    Never shipped -- see `level_term_diagnostic`. Exists so B.3's numbers come out
    of the committed script rather than an ad-hoc session.
    """

    name = "fitted-k+c"

    def fit(
        self,
        pairs: list[YearPair],
        column: str,
        n0: float,
        *,
        shrunk: bool = True,
        weighted: bool = True,
    ) -> FittedK:
        fitted = super().fit(pairs, column, n0, shrunk=shrunk, weighted=weighted)
        return _ShiftedK(fitted.params)


class _ShiftedK(FittedK):
    """`FittedK` that DOES apply `c_fit`. Diagnostic only."""

    def predict(self, base: pd.Series, residual: pd.Series, weight: pd.Series) -> pd.Series:
        return super().predict(base, residual, weight) + self.params["c_fit"]


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

        diag = level_term_diagnostic(pairs, cfg.pt_col, cfg.n0, cfg.gate)
        levels = playing_time_levels(pairs, cfg.pt_col, cfg.gate)
        diag_out = ANALYSIS_DIR / f"keeper_calibration_{player_type}_level_term.csv"
        levels_out = ANALYSIS_DIR / f"keeper_calibration_{player_type}_pt_levels.csv"
        diag.to_csv(diag_out, index=False)
        levels.to_csv(levels_out, index=False)
        _print_table(f"{player_type}: playing-time level term (finding B.3)", diag)
        _print_table(f"{player_type}: playing-time levels (finding B.3)", levels)
        print(f"wrote {diag_out}")
        print(f"wrote {levels_out}")

        if not args.skip_sweep:
            sweep = n0_sweep(pairs, cfg.columns, cfg.n0_grid, pt_col=cfg.pt_col, gate=cfg.gate)
            sweep_out = ANALYSIS_DIR / f"keeper_calibration_{player_type}_n0_sweep.csv"
            sweep.to_csv(sweep_out, index=False)
            _print_table(f"{player_type}: n0 sensitivity", sweep)
            print(f"wrote {sweep_out}")

    # The finding's Part A cites this table as the evidence that pair membership
    # is not preconditioned on survival (0.755/0.777/0.795, not 0.84-0.87). It
    # must be regenerable by the script, not assembled by hand -- but a partial
    # run would otherwise overwrite the committed both-types artifact with one
    # player type's rows, silently shrinking the evidence Part A quotes.
    survivorship_out = ANALYSIS_DIR / "keeper_calibration_survivorship.csv"
    if len(types) == 2:
        pd.concat(survivorship_frames, ignore_index=True).to_csv(survivorship_out, index=False)
        print(f"wrote {survivorship_out}")
    else:
        print(f"skipped {survivorship_out.name}: needs both player types (--player-type both)")


if __name__ == "__main__":
    main()
