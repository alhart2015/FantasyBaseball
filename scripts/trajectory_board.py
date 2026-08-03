"""Top-N career trajectories across the whole player pool (#311).

`player_trajectory.py` answers "what about this player". This ranks everyone at once,
which is the question a keeper decision actually asks: of all the players I could hold,
which are worth the most over the years I would hold them?

Ranked on TOTAL VAR over the horizon -- value above the position-aware waiver floor,
summed across the projected years. Raw SGP would silently penalise every catcher and
reliever, and a single year would not be a keeper question. The two differ enough to
flip the order: Mason Miller is 8.9 VAR over three years to Zack Wheeler's 6.9, while on
raw SGP Wheeler leads him every single year.

    python scripts/trajectory_board.py --top 25
    python scripts/trajectory_board.py --pool pitcher --top 15 --horizon 5
    python scripts/trajectory_board.py --top 10 --min-sgp 8

The band is p10..p90 from the empirical outcome distribution, NOT a multiple of a
standard deviation -- see `PathPoint.p10`. Read it: at three years out the interval is
most of the story, especially for pitchers, where the point estimate carries little.

Build the panel first (one time, ~1 minute):
    python scripts/build_pt_panel.py --start 2000 --end 2026 --out-dir data/trajectory
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

from fantasy_baseball.config import load_config
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
from fantasy_baseball.trajectory.board import (
    BoardRow,
    board_inputs,
    player_names,
    season_slots,
)
from fantasy_baseball.trajectory.era import era_normalize
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR, load_scored_panel
from fantasy_baseball.trajectory.shape import prepare, shape_trajectory

#: Bootstrap refits per query. The board reports no SE column, and `se` enters the band
#: only through `spread`, which moves by 0.0006 between 250 draws and 1000 -- so the
#: sweep buys a 4x speedup for precision it does not print. The single-player CLI keeps
#: the higher default, where the SE IS a printed column.
SWEEP_DRAWS = 250

#: Fitting weight that must sit near the query's own current season before its row is
#: ranked rather than flagged. Shape has no kernel on `down`, so a query can be matched
#: to a cohort it sits entirely outside and priced by extrapolating that cohort's line.
#:
#: 10% separates the two failure modes cleanly on the real board -- measured, not chosen:
#:
#:     Sal Stewart      16.8 now / 1.5 prior     1.8%   extrapolated
#:     Kevin McGonigle  13.6 now / 0.0 prior     4.6%   extrapolated
#:     CJ Abrams        20.9 now / 14.0 prior    6.3%   extrapolated
#:     ---
#:     Crow-Armstrong   20.2 now / 17.2 prior   16.1%   supported
#:     Hunter Goodman   16.1 now / 12.9 prior   20.5%   supported
#:     Juan Soto        12.9 now / 21.5 prior   33.1%   supported
#:     Bobby Witt Jr.   15.5 now / 18.9 prior   40.1%   supported
#:
#: The obvious gauge, `sgp - mean_start`, cannot make that split: it reads "above this
#: cohort's mean" and "outside this cohort" the same way, and flagged 11 of the top 20.
MIN_LOCAL_SUPPORT = 0.10


def score(
    rows: list[BoardRow], panel: pd.DataFrame, kind: str, horizons: tuple[int, ...]
) -> list[dict]:
    """Fit every row against one prepared state.

    The whole point of #311: `build_history` and the forward-value lookup depend on the
    panel, not the player, so they are hoisted out of the loop. One state per pool --
    a hitter-fitted state cannot price a pitcher and will refuse to try.
    """
    prepared = prepare(panel, kind=kind, horizons=horizons)
    scored = []
    for row in rows:
        traj, _ = shape_trajectory(
            prepared,
            kind=kind,
            age=row.age,
            sgp=row.sgp,
            peak=row.prior_sgp,
            horizons=horizons,
            replacement=row.floor,
            slot=row.slot,
            bootstrap_draws=SWEEP_DRAWS,
        )
        if not traj.observable:
            continue
        scored.append(
            {
                "name": row.name,
                "pool": row.pool,
                "age": row.age,
                "slot": row.slot,
                "now": row.sgp,
                "prior": row.prior_sgp,
                "total": traj.total,
                # Summed across horizons, so the band describes the TOTAL rather than
                # any one year. This assumes the years move together, which overstates
                # the width if they do not -- stated rather than hidden, and it is the
                # conservative direction for a keep-or-cut call.
                "p10": sum(p.p10 for p in traj.observable),
                "p90": sum(p.p90 for p in traj.observable),
                "years": len(traj.observable),
                "n_eff": min(p.n_effective for p in traj.observable),
                "support": traj.local_support,
            }
        )
    return scored


def render(scored: list[dict], top: int, horizons: tuple[int, ...], levels: dict) -> None:
    scored.sort(key=lambda r: r["total"], reverse=True)
    span = f"{len(horizons)}-year"
    floors = "  ".join(f"{s} {levels[s]:.2f}" for s in sorted(levels, key=lambda s: levels[s]))
    print(f"\nTOP {min(top, len(scored))} by {span} TOTAL VAR   (floors: {floors})")
    print(f"{len(scored)} players scored\n")
    print(
        f"{'rank':>4}  {'player':<24} {'age':>3} {'slot':>4} {'now':>6} {'prior':>6} "
        f"{'total':>7}  {'p10..p90':>16} {'yrs':>3} {'n_eff':>6} {'supp':>5}"
    )
    for i, r in enumerate(scored[:top], start=1):
        band = f"{r['p10']:6.1f}..{r['p90']:<6.1f}"
        # (!) is a warning, not decoration: below the threshold the fitted line was
        # evaluated outside its own support, so the estimate AND the band describe
        # players unlike this one.
        flag = " (!)" if r["support"] < MIN_LOCAL_SUPPORT else ""
        print(
            f"{i:4d}  {r['name'][:24]:<24} {r['age']:3d} {r['slot']:>4} {r['now']:6.1f} "
            f"{r['prior']:6.1f} {r['total']:7.1f}  {band:>16} {r['years']:3d} {r['n_eff']:6.0f} "
            f"{r['support']:5.0%}{flag}"
        )
    if any(r["support"] < MIN_LOCAL_SUPPORT for r in scored[:top]):
        print(
            f"\n  (!) under {MIN_LOCAL_SUPPORT:.0%} of the fitting weight sits near this"
            "\n      player's own current season. `peak` is kernel-weighted, `down` is not,"
            "\n      so a season that far outruns its prior is priced by extrapolating a"
            "\n      line fitted on players unlike him -- and the band, being that cohort's"
            "\n      scatter, comes out NARROW rather than wide. Do not rank on these."
            "\n      Estimator fix is #310; --min-support drops them entirely."
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", choices=("hitter", "pitcher", "both"), default="both")
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--horizon", type=int, default=3, help="years forward to project")
    parser.add_argument(
        "--min-sgp",
        type=float,
        default=0.0,
        help="skip players below this current-season pace, to cut the fringe",
    )
    parser.add_argument(
        "--min-support",
        type=float,
        default=0.0,
        help=(
            "drop rows whose fitting weight near the query's own current season falls "
            f"below this (try {MIN_LOCAL_SUPPORT}); by default they are flagged, not removed"
        ),
    )
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.top < 1:
        parser.error("--top must be at least 1")
    if not args.panel_dir.is_absolute():
        args.panel_dir = PROJECT_ROOT / args.panel_dir

    overrides = load_config(PROJECT_ROOT / "config" / "league.yaml").sgp_overrides
    levels = position_aware_replacement_levels(get_sgp_denominators(overrides))
    horizons = tuple(range(1, args.horizon + 1))
    pools = ["hitter", "pitcher"] if args.pool == "both" else [args.pool]

    def load(kind: str, include_partial: bool) -> pd.DataFrame:
        return era_normalize(
            load_scored_panel(
                kind,
                panel_dir=args.panel_dir,
                sgp_overrides=overrides,
                include_partial=include_partial,
            ),
            kind,
            sgp_overrides=overrides,
        )

    # Dating the in-progress season is a league fact and must come off the HITTER panel
    # even when pricing pitchers -- pitcher `games` counts appearances, not team games.
    calendar = load("hitter", True)
    cache = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
    names = player_names(cache)
    season = int(calendar["season"].max())
    eligibility = season_slots(cache, season)

    started = time.perf_counter()
    scored: list[dict] = []
    for kind in pools:
        live = calendar if kind == "hitter" else load(kind, True)
        rows = [
            r
            for r in board_inputs(
                live,
                kind=kind,
                names=names,
                replacement_levels=levels,
                eligibility=eligibility,
                calendar=calendar,
                season=season,
            )
            if r.sgp >= args.min_sgp
        ]
        print(f"  {kind}: {len(rows)} players with a {season} line", flush=True)
        # The comp pool must NOT contain the in-progress season: a two-thirds year would
        # be averaged in as though it were a full one.
        scored += score(rows, load(kind, False), kind, horizons)

    if args.min_support > 0:
        dropped = [r for r in scored if r["support"] < args.min_support]
        scored = [r for r in scored if r["support"] >= args.min_support]
        # Say what was dropped. A silently shortened board reads as "these are the best
        # players", when it is "these are the ones the model can speak to".
        print(f"  dropped {len(dropped)} rows below --min-support {args.min_support:.0%}")
    if not scored:
        print("\nnothing scored -- check --min-sgp and that the panel covers this season")
        return 1
    render(scored, args.top, horizons, levels)
    print(f"\n  scored in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
