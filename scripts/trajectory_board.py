"""Top-N career trajectories across the whole player pool (#311).

`player_trajectory.py` answers "what about this player". This ranks everyone at once,
which is the question a keeper decision actually asks: of all the players I could hold,
which are worth the most over the years I would hold them?

Ranked on TOTAL VAR over the horizon -- value above the position-aware waiver floor,
summed across the projected years. Raw SGP would silently penalise every catcher and
reliever, and a single year would not be a keeper question. The two differ enough to
flip the order: Mason Miller is 8.9 VAR over three years to Zack Wheeler's 6.9, while on
raw SGP Wheeler leads him every single year.

THE ONE THAT ANSWERS MOST QUESTIONS -- top 50 league-wide, your whole roster, the best
five on every other team, and a CSV of all 551 rows to slice afterwards:

    python scripts/trajectory_board.py --top 50 --min-sgp 4 --by-team --csv board.csv

Then answer follow-ups from `board.csv` rather than re-running: the sweep is ~17s, and
two answers pulled from one file cannot disagree with each other the way two sweeps can.

    python scripts/trajectory_board.py --top 25                    # league board only
    python scripts/trajectory_board.py --team "Hello Peanuts!"     # one roster, in full
    python scripts/trajectory_board.py --pool pitcher --horizon 5
    python scripts/trajectory_board.py --by-team --min-support 0.1 # drop extrapolations

`--min-sgp 4` trims the fringe without touching anyone rankable; `--by-team` and `--team`
read LIVE rosters from Upstash, so they need `.env` credentials and a network.

The band is p10..p90 from the empirical outcome distribution, NOT a multiple of a
standard deviation -- see `PathPoint.p10`. Read it: at three years out the interval is
most of the story, especially for pitchers, where the point estimate carries little.

`(!)` marks a row whose fitted line was evaluated outside its own support. The BAND is
honest on those and is what to read; the point estimate is the part still assuming the
line holds out there. See `MIN_LOCAL_SUPPORT`.

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
from fantasy_baseball.data.rosters import RosterSpot, live_rosters
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
from fantasy_baseball.trajectory.board import board_inputs, player_names, season_slots
from fantasy_baseball.trajectory.comps import MIN_LOCAL_SUPPORT
from fantasy_baseball.trajectory.era import era_normalize
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR, load_scored_panel
from fantasy_baseball.trajectory.sweep import RANK_MOVE, add_ranks, sweep_pool, totals
from fantasy_baseball.utils.name_utils import normalize_name


def assign_teams(scored: list[dict], spots: list[RosterSpot]) -> dict[str, list[str]]:
    """Stamp each scored row with its owning team, and return who never matched.

    Joined on (normalized name, player_type) because roster blobs carry no mlbam_id
    (#284). The unmatched are returned rather than swallowed: a silently shortened team
    reads as "he has nobody else worth listing".
    """
    by_key = {(s.normalized, s.player_type): s for s in spots}
    for row in scored:
        spot = by_key.get((normalize_name(row["name"]), row["pool"]))
        row["team"] = spot.team if spot else None
        row["status"] = spot.status if spot else ""

    scored_keys = {(normalize_name(r["name"]), r["pool"]) for r in scored}
    missing: dict[str, list[str]] = {}
    for s in spots:
        if (s.normalized, s.player_type) not in scored_keys:
            missing.setdefault(s.team, []).append(s.name)
    return missing


def by_team(
    scored: list[dict],
    spots: list[RosterSpot],
    missing: dict[str, list[str]],
    my_team: str,
    per_team: int,
    base: int,
    horizons: tuple[int, ...],
    only: str | None = None,
) -> None:
    """Every player on your team, then the best `per_team` on each other team.

    Ranks shown are LEAGUE ranks, carried from `add_ranks`, not ranks within the team --
    otherwise every team's best player reads as a 1.
    """
    one, span = _header(base, horizons)

    def block(team: str, limit: int | None, note: str = "") -> None:
        """One team's block. Takes the TEAM, never a decorated title -- `missing` is keyed
        on the raw name, and passing a label like "X -- YOUR TEAM" made the lookup miss, so
        the unmatched-player warning fired for every opposing team and silently never for
        your own: the one roster a keep-or-cut call is made from."""
        rows = sorted(
            (r for r in scored if r["team"] == team), key=lambda r: r["total"], reverse=True
        )
        shown = rows if limit is None else rows[:limit]
        total = sum(r["total"] for r in rows)
        head = f"{team}{note}  ({len(rows)} scored, {total:.1f} total {span} VAR)"
        print(f"\n{head}\n{'-' * len(head)}")
        for r in shown:
            band = f"{r['p10']:5.1f}..{r['p90']:<5.1f}"
            flag = " (!!)" if r["band_fell_back"] else (" (!)" if r["extrapolated"] else "    ")
            hurt = f" [{r['status']}]" if r["status"] else ""
            print(
                f"  #{r['rank_total']:<4d} #{r['rank_next']:<4d} {r['name'][:22]:<22} "
                f"{r['age']:3d} {r['slot']:>4} {r['total']:6.1f} {r['next']:5.1f}  "
                f"{band:>14}{flag}{hurt}"
            )
        if team in missing:
            print(f"  not scored: {', '.join(sorted(missing[team]))}")

    print(f"\n\n{'=' * 78}\nPER-TEAM  (#{span} and #{one} are LEAGUE ranks)\n{'=' * 78}")
    # Teams come from the ROSTERS, not from the scored rows. A team whose players were all
    # filtered out -- by --min-sgp, by --min-support, or by the join failing wholesale --
    # has no scored rows at all, so deriving the list from `scored` dropped it and its
    # entire `missing` list with it, leaving nothing on screen to say it existed.
    rostered = {s.team for s in spots}
    if only is not None:
        # One team, in full. `--team` exists so asking about somebody else's roster does
        # not mean re-running the sweep and reading past nine other blocks.
        if only not in rostered:
            print(f"\n  no team named {only!r}. Teams: {', '.join(sorted(rostered))}")
            return
        block(only, None, note="  -- all players")
        return
    block(my_team, None, note="  -- YOUR TEAM, all players")
    for team in sorted(rostered - {my_team}):
        block(team, per_team)
    if not rostered:
        print("\n  no rosters read -- see the join note above.")


def _header(base: int, horizons: tuple[int, ...]) -> tuple[str, str]:
    """Column labels as SEASONS, since "+1" and "3-year" are not what a keeper thinks in."""
    return f"{base + 1}", f"{base + min(horizons)}-{str(base + max(horizons))[-2:]}"


def render(
    scored: list[dict], top: int, horizons: tuple[int, ...], levels: dict, base: int
) -> None:
    scored.sort(key=lambda r: r["total"], reverse=True)
    one, span = _header(base, horizons)
    floors = "  ".join(f"{s} {levels[s]:.2f}" for s in sorted(levels, key=lambda s: levels[s]))
    print(f"\nTOP {min(top, len(scored))} by {span} TOTAL VAR   (floors: {floors})")
    print(f"{len(scored)} players scored\n")
    print(
        f"{'#' + span:>8} {'#' + one:>6}  {'player':<24} {'age':>3} {'slot':>4} {'now':>6} "
        f"{'prior':>6} {span + ' VAR':>10} {one + ' VAR':>9}  {'p10..p90':>16} {'yrs':>4} {'supp':>5}"
    )
    for r in scored[:top]:
        band = f"{r['p10']:6.1f}..{r['p90']:<6.1f}"
        # (!) is a warning, not decoration: below the threshold the fitted line was
        # evaluated outside its own support, so the band is wide because the model is
        # extrapolating rather than because this player is genuinely volatile.
        flag = " (!!)" if r["band_fell_back"] else (" (!)" if r["extrapolated"] else "")
        # The MOVE between the two ranks is the keeper signal: a player far better over
        # three years than next year is who you hold rather than who you start.
        shift = r["rank_next"] - r["rank_total"]
        arrow = f"{shift:+d}" if abs(shift) >= RANK_MOVE else ""
        print(
            f"{r['rank_total']:8d} {r['rank_next']:6d}  {r['name'][:24]:<24} {r['age']:3d} "
            f"{r['slot']:>4} {r['now']:6.1f} {r['prior']:6.1f} {r['total']:10.1f} "
            f"{r['next']:9.1f}  {band:>16} {r['years']:4d} {r['support']:5.0%}{flag}{arrow:>5}"
        )
    if any(r["extrapolated"] for r in scored[:top]):
        print(
            f"\n  (!) under {MIN_LOCAL_SUPPORT:.0%} of the fitting weight sits near this"
            "\n      player's own current season -- LAST season is kernel-weighted and THIS"
            "\n      one is not, so a season that far outruns its prior is priced by"
            "\n      extrapolating a line fitted on players unlike him."
            "\n"
            "\n      The BAND already accounts for this and is wide on these rows, so read it"
            "\n      rather than the point estimate: measured on breakouts the interval is"
            "\n      calibrated for hitters (12%/11% against a nominal 10%/10%) and still"
            "\n      optimistic for pitchers three years out (24% below p10). What stays"
            "\n      unguarded is the estimate itself, which leans on the fitted line holding"
            "\n      outside its own data -- locally unbiased where checked, but assumed."
            "\n      Estimator fix is #310; --min-support drops these rows entirely."
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
    parser.add_argument(
        "--by-team",
        action="store_true",
        help="also break the board down by fantasy team, reading live rosters from Upstash",
    )
    parser.add_argument("--per-team", type=int, default=5, help="rows per opposing team")
    parser.add_argument(
        "--team",
        help="show ONE team in full instead of the per-team breakdown (implies --by-team)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help=(
            "write every scored row here -- ranks, band, support and owning team. The "
            "sweep takes ~17s, so slicing a saved board beats re-running it, and two "
            "answers taken from one file cannot disagree with each other."
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

    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    overrides = config.sgp_overrides
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
    swept = []
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
        # be averaged in as though it were a full one. DERIVED from `live` rather than
        # loaded again -- a second `load()` re-reads a 4.7MB CSV and runs two more
        # full-panel `apply` passes for a frame that is this one minus its partial rows.
        # Verified identical on both pools: same ids, same seasons, max |sgp diff| 0.0.
        complete = live[~live["partial_season"]].reset_index(drop=True)
        # VAR only. The web board (#321) also caches a raw-SGP fit for its second column,
        # but that is a SECOND fit per player -- asking for it here would double a 17s run
        # to serve a column this CLI does not print.
        swept += sweep_pool(rows, complete, kind, horizons, scales=("var",))

    scored = totals(swept, horizons, scale="var")

    # RANK FIRST, then filter. `add_ranks` documents itself as ranking "over the whole
    # scored pool", and the web board does exactly that -- so ranking the filtered subset
    # here renumbered 1..N and gave the same player a different number on the two
    # surfaces. The visible consequence is that the # column now has GAPS when
    # --min-support drops a row: a rank is a position among everyone the model could
    # price, not among whatever survived a display filter.
    add_ranks(scored)
    if args.min_support > 0:
        dropped = [r for r in scored if r["support"] < args.min_support]
        scored = [r for r in scored if r["support"] >= args.min_support]
        # Say what was dropped. A silently shortened board reads as "these are the best
        # players", when it is "these are the ones the model can speak to".
        print(f"  dropped {len(dropped)} rows below --min-support {args.min_support:.0%}")
    if not scored:
        print("\nnothing scored -- check --min-sgp and that the panel covers this season")
        return 1
    render(scored, args.top, horizons, levels, season)
    show_teams = bool(args.by_team or args.team)
    if show_teams or args.csv:
        # Live Upstash, not the local mirror: roster membership is exactly the kind of
        # state that goes stale silently, and a trade since the last sync would show a
        # player on the wrong team with no indication anything was wrong.
        #
        # `--csv` alone must not DIE on a missing network. The roster read only adds an
        # owner column there, and the docstring promises credentials are needed for
        # `--by-team`/`--team` -- so failing the documented headline invocation offline,
        # after the 17s sweep has already run and printed, costs the whole run for a
        # column. The team views genuinely cannot proceed, so those still fail loudly.
        try:
            spots = live_rosters(config.team_name)
            print(f"\n  {len(spots)} roster spots read from Upstash")
            missing = assign_teams(scored, spots)
        except Exception as exc:
            if show_teams:
                raise
            print(f"\n  NOTE: rosters unavailable ({type(exc).__name__}); CSV has no team column.")
            spots, missing = [], {}
        if show_teams:
            by_team(
                scored, spots, missing, config.team_name, args.per_team, season, horizons, args.team
            )
    if args.csv:
        pd.DataFrame(scored).sort_values("rank_total").to_csv(args.csv, index=False)
        print(f"\n  wrote {len(scored)} rows to {args.csv}")
    print(f"\n  scored in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
