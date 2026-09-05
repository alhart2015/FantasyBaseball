"""Top-N career trajectories across the whole player pool (#311).

`player_trajectory.py` answers "what about this player". This ranks everyone at once,
which is the question a keeper decision actually asks: of all the players I could hold,
which are worth the most over the years I would hold them?

Ranked on TOTAL VAR over the horizon -- value above the position-aware waiver floor,
summed across the projected years. Raw SGP would silently penalise every catcher and
reliever, and a single year would not be a keeper question. The two differ enough to
flip the order (2027-29, measured 2026-08-06): Mason Miller is 4.5 VAR to Zack Wheeler's
-0.0, while on raw SGP Wheeler leads 27.9 to 26.8. Wheeler is the better pitcher and the
worse KEEPER, because an SP floor of 9.29 is a much higher bar than an RP's 7.42.

Those figures replace pre-#331 ones (8.9 and 6.9) that no VAR this tool computes can
reproduce: VAR was a clamped refit then and is `raw - years * floor` now. The old text
also claimed Wheeler led "every single year", which is no longer true either -- he leads
years one and two, 11.7 to 10.6 and 9.3 to 8.7, and trails in year three, 6.8 to 7.5.

THE ONE THAT ANSWERS MOST QUESTIONS -- top 50 league-wide, your whole roster, the best
five on every other team, and a CSV of all 551 rows to slice afterwards:

    python scripts/trajectory_board.py --top 50 --min-sgp 4 --by-team --csv board.csv

Then answer follow-ups from `board.csv` rather than re-running: the sweep is ~17s, and
two answers pulled from one file cannot disagree with each other the way two sweeps can.

    python scripts/trajectory_board.py --top 25                    # league board only
    python scripts/trajectory_board.py --team "Hello Peanuts!"     # one roster, in full
    python scripts/trajectory_board.py --pool pitcher --horizon 5
    python scripts/trajectory_board.py --by-team --min-support 0.3 # thin rows dropped

`--min-sgp 4` trims the fringe without touching anyone rankable; `--by-team` and `--team`
read LIVE rosters from Upstash, so they need `.env` credentials and a network.

The band is p10..p90 from the empirical outcome distribution, NOT a multiple of a
standard deviation -- see `PathPoint.p10`. Read it: at three years out the interval is
most of the story, especially for pitchers, where the point estimate carries little.

THE BAND IS CALIBRATED. Each tail holds its nominal 10%, measured by rolling origin on
held-out seasons -- see `trajectory.calibration` and
`docs/trajectory-band-calibration-2026-09-04.md`. Read it as it prints; there is no
support level or horizon at which it needs discounting.

`supp` is the share of fitting weight sitting near the query's own current season. It is
no longer a warning -- the band already accounts for it, and the correction it earns is
larger below 30% -- but it is still what separates a number the model has seen many
players make from one it is reaching for.

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
from fantasy_baseball.data.rosters import RosterSpot, live_rosters, manual_store_active
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
from fantasy_baseball.trajectory.board import board_inputs, player_names, season_slots
from fantasy_baseball.trajectory.calibration import MAX_HORIZON, span_target
from fantasy_baseball.trajectory.panel import DEFAULT_PANEL_DIR
from fantasy_baseball.trajectory.ros_anchor import load_anchored_panels
from fantasy_baseball.trajectory.roster_join import RosterIndex, index_rosters
from fantasy_baseball.trajectory.sweep import add_ranks, rank_move, sweep_pool, totals
from fantasy_baseball.trajectory.value_bars import load_shipped_bars
from fantasy_baseball.utils.name_utils import normalize_name


def by_team(
    scored: list[dict],
    spots: list[RosterSpot],
    index: RosterIndex,
    my_team: str,
    per_team: int,
    base: int,
    horizons: tuple[int, ...],
    only: str | None = None,
    *,
    keep: int,
    detail: bool = False,
) -> None:
    """Every player on your team, then the best `per_team` on each other team.

    Ranks shown are LEAGUE ranks, carried from `add_ranks`, not ranks within the team --
    otherwise every team's best player reads as a 1.

    `per_team` is how many rows to PRINT; `keep` is how many a team may actually keep,
    and the headline sums the latter. Required, with no default: a headline that
    ASSERTS a league rule must not be able to invent one. `shown` is floored at `keep`
    so the printed rows can always account for the number beside them. The two are different questions and the web view
    already separated them -- this surface had kept headlining the best `per_team`, so
    the same board ranked teams differently depending on which surface you read.
    """
    one, span = _header(base, horizons)

    def rows_for(team: str) -> list[dict]:
        """This team's scored rows, strongest first.

        MEMBERSHIP, matching the web board: a key rostered by two teams appears under
        both. `r["team"]` is the single winning team and is what the CSV carries, but
        it is not the ownership test.
        """
        # ORDERED BY P(keeper), matching the league board. The block's headline VAR sum
        # still uses the mean -- that is a value total, not a ranking -- but the rows a
        # reader scans are ordered by the number the decision is actually made on.
        return sorted(
            (r for r in scored if team in r["teams"]),
            key=lambda r: r["p_keeper"] if r.get("p_keeper") is not None else -1,
            reverse=True,
        )

    def keeper_strength(team: str) -> float:
        """The headline number, as an ordering key. Same rule, one definition."""
        rows = rows_for(team)
        return sum(r["total"] for r in rows[: min(keep, len(rows))])

    def block(team: str, limit: int | None, note: str = "") -> None:
        """One team's block. Takes the TEAM, never a decorated title -- the missing list is keyed
        on the raw name, and passing a label like "X -- YOUR TEAM" made the lookup miss, so
        the unmatched-player warning fired for every opposing team and silently never for
        your own: the one roster a keep-or-cut call is made from."""
        rows = rows_for(team)
        shown = rows if limit is None else rows[: max(limit, keep)]
        # The best `per_team`, NOT the roster and NOT `shown`. Three separate reasons.
        #
        # Not the roster: VAR is a shift rather than a clamp as of #331, so a below-
        # replacement player carries a real negative instead of 0.0 and a whole-roster
        # sum is mostly tail. Measured at the 2027-29 range on live rosters, 93.5% of
        # scored players are negative and each team's tail runs -62 to -196 against a
        # best-5 signal of 15 to 73. It ranked Boston Estrellas last in the league on
        # the depth of its junk while its best five were 4th. Rosters are all 23-26
        # scored players, so the tail was not even measuring roster SIZE -- it was
        # measuring how bad the bottom was, which is not a keeper signal at three
        # keepers a team. Nobody keeps their 20th man.
        #
        # Not `shown`: your own block passes `limit=None` to list every player you own,
        # so summing what is displayed would put your headline on 24 players and every
        # opponent's on 5 -- and read as a deficit, since your extra rows are the
        # negative ones. The cap belongs to the number, not the list.
        #
        # `keep`, NOT `per_team`. The extra rows are a scouting POOL -- an opponent
        # without this model may not keep his best three, so the fourth and fifth names
        # are worth seeing -- but only three of them can be retained, so summing five
        # counts two players nobody keeps. That is not a smaller signal, it is a
        # different ordering: on 2026-08-22 the best-5 leader TRAILED on best-3.
        #
        # `shown` is floored at `keep` for the same reason `build_teams_board` floors
        # `per_team`: `--per-team 2` under a three-keeper rule would otherwise print two
        # players over a header naming three, a number the visible rows cannot make.
        # `kept`, not `keep`, in the slice -- a team with two scored players sums two,
        # and the label has to be the number that was actually summed.
        kept = min(keep, len(rows))
        total = sum(r["total"] for r in rows[:kept])
        head = (
            f"{team}{note}  ({len(rows)} scored, "
            f"{total:.1f} total {span} VAR from the best {kept} they may keep)"
        )
        print(f"\n{head}\n{'-' * len(head)}")
        cols = f"  {'player':<22} {'age':>3} {'slot':>4} {'elite':>7} {'keeper':>7} {'bust':>7}"
        if detail:
            cols += f"   {'#mean':<5} {span:>6} {one:>5} {'p10..p90':>14} {'supp':>5}"
        print(cols)
        for r in shown:
            hurt = f" [{r['status']}]" if r["status"] else ""
            line = (
                f"  {r['name'][:22]:<22} {r['age']:3d} {r['slot']:>4} "
                f"{pct(r.get('p_elite')):>7} {pct(r.get('p_keeper')):>7} "
                f"{pct(r.get('p_bust')):>7}"
            )
            if detail:
                band = f"{r['p10']:5.1f}..{r['p90']:<5.1f}"
                line += (
                    f"   #{r['rank_total']:<4d} {r['total']:6.1f} {r['next']:5.1f} "
                    f"{band:>14} {r['support']:4.0%}"
                )
            print(line + hurt)
        missing = index.unscored_for(team)
        if missing:
            print(f"  not scored: {', '.join(missing)}")

    print(f"\n\n{'=' * 78}\nPER-TEAM  (#{span} and #{one} are LEAGUE ranks)\n{'=' * 78}")
    # Teams come from the ROSTERS, not from the scored rows. A team whose players were all
    # filtered out -- by --min-sgp, by --min-support, or by the join failing wholesale --
    # has no scored rows at all, so deriving the list from `scored` dropped it and its
    # entire missing list with it, leaving nothing on screen to say it existed.
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
    # Ordered by keeper strength, like the web view -- not alphabetically. The point of
    # the per-team view is comparing teams, and a list sorted by name buries that.
    #
    # Name is the tie-break, not decoration: two teams with nothing scored both sum to
    # 0.0, and leaving that to set order makes the output reorder between runs.
    for team in sorted(rostered - {my_team}, key=lambda t: (-keeper_strength(t), t)):
        block(team, per_team)
    if not rostered:
        print("\n  no rosters read -- see the join note above.")


def _header(base: int, horizons: tuple[int, ...]) -> tuple[str, str]:
    """Column labels as SEASONS, since "+1" and "3-year" are not what a keeper thinks in."""
    return f"{base + 1}", f"{base + min(horizons)}-{str(base + max(horizons))[-2:]}"


def pct(value: float | None) -> str:
    """A probability, or `--` when the span has no measured bar to compute one against."""
    return "  --" if value is None else f"{value:.0%}"


def bar_note(horizons: tuple[int, ...]) -> str:
    """The realized bars this board's probabilities are measured against, named on screen.

    PRINTED, not assumed. These are realized VAR totals from completed seasons, not
    quantiles of the projected pool -- a distinction that cost two whole tiers when it was
    got wrong -- so the numbers and the window count are on the page rather than in a
    docstring somebody would have to go find.
    """
    bars, target = load_shipped_bars(), span_target(horizons)
    if bars is None or target is None or not bars.bars.get(target):
        return "  no realized bars for this range -- probabilities unavailable"
    starts = bars.windows.get(target, [])
    named = "  ".join(
        f"{name} {bars.bar(target, name):.1f}"
        for name in ("elite", "keeper", "bust")
        if bars.bar(target, name) is not None
    )
    window = f"{len(starts)} window{'s' if len(starts) != 1 else ''} ({min(starts)}-{max(starts)})"
    return f"  bars, realized VAR over {window}:  {named}"


def render(
    scored: list[dict],
    top: int,
    horizons: tuple[int, ...],
    levels: dict,
    base: int,
    ranked: int,
    *,
    detail: bool = False,
) -> None:
    """The league board. Probabilities lead; the projection behind them is `--detail`.

    SORTED BY P(keeper), not by the projected mean. The mean is an input to the
    probability, and the two order the pool differently wherever the bands differ -- a
    thin projection with a wide band can out-mean a supported one and still be less likely
    to clear the bar. Ranking on the number the decision reads is the point of the
    feature.
    """
    scored.sort(key=lambda r: r["p_keeper"] if r.get("p_keeper") is not None else -1, reverse=True)
    one, span = _header(base, horizons)
    print(f"\nTOP {min(top, len(scored))} by P(keeper) over {span}")
    print(bar_note(horizons))
    if ranked != len(scored):
        print(f"  {len(scored)} scored after --min-support, ranked against all {ranked}")
    else:
        print(f"  {len(scored)} players scored")
    if detail:
        floors = "  ".join(f"{s} {levels[s]:.2f}" for s in sorted(levels, key=lambda s: levels[s]))
        print(f"  floors: {floors}")

    head = (
        f"\n  {'#':>3}  {'player':<24} {'age':>3} {'slot':>4} "
        f"{'elite':>7} {'keeper':>7} {'bust':>7}"
    )
    if detail:
        head += (
            f"   {'now':>6} {'prior':>6} {span + ' VAR':>10} {one + ' VAR':>9} "
            f"{'p10..p90':>16} {'supp':>5} {'#mean':>6}"
        )
    print(head)
    for i, r in enumerate(scored[:top], start=1):
        line = (
            f"  {i:>3}  {r['name'][:24]:<24} {r['age']:3d} {r['slot']:>4} "
            f"{pct(r.get('p_elite')):>7} {pct(r.get('p_keeper')):>7} {pct(r.get('p_bust')):>7}"
        )
        if detail:
            band = f"{r['p10']:6.1f}..{r['p90']:<6.1f}"
            # The MOVE between the two mean-ranks is the old keeper signal, kept in detail
            # because it says something the probabilities do not: whether a player is
            # worth HOLDING rather than STARTING.
            shift = rank_move(r)
            arrow = f"{shift:+d}" if shift else ""
            line += (
                f"   {r['now']:6.1f} {r['prior']:6.1f} {r['total']:10.1f} {r['next']:9.1f} "
                f"{band:>16} {r['support']:5.0%} {r['rank_total']:>6}{arrow:>5}"
            )
        print(line)


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
            "below this; the band is calibrated at every level, so this trims the pool "
            "rather than hiding anything untrustworthy"
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
    parser.add_argument(
        "--detail",
        action="store_true",
        help=(
            "add the projection behind the probabilities -- mean VAR, band, support and "
            "the mean-based rank. Hidden by default because the probability is the "
            "decision and the mean is one of its inputs"
        ),
    )
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    args = parser.parse_args()
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    if args.horizon > MAX_HORIZON:
        # REFUSED, not clamped silently. Past this there is no `s{k}` multiplier, so the
        # board would print a band nothing measured -- and it would look exactly like a
        # calibrated one. `build_band_calibration.py` fits 1..MAX_HORIZON; raising this
        # means raising that.
        parser.error(f"--horizon max is {MAX_HORIZON} (the calibrated range)")
    if args.top < 1:
        parser.error("--top must be at least 1")
    if not args.panel_dir.is_absolute():
        args.panel_dir = PROJECT_ROOT / args.panel_dir

    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    overrides = config.sgp_overrides
    levels = position_aware_replacement_levels(get_sgp_denominators(overrides))
    horizons = tuple(range(1, args.horizon + 1))
    pools = ["hitter", "pitcher"] if args.pool == "both" else [args.pool]

    # THE SAME loader the pushed board uses, so the CLI and the web board cannot end up
    # anchoring the in-progress season on two different things. It injects season-to-date
    # plus a rest-of-season projection before era-normalizing -- see `ros_anchor`.
    loaded = load_anchored_panels(
        systems=config.projection_systems,
        weights={s: config.projection_weights[s] for s in config.projection_systems},
        panel_dir=args.panel_dir,
        sgp_overrides=overrides,
    )
    cache = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
    names = player_names(cache)
    season = loaded.season
    eligibility = season_slots(cache, season)
    anchor = (
        f"anchored on the {loaded.snapshot_date} rest-of-season projection"
        if loaded.snapshot_date
        else f"{season} is complete, so nothing is projected"
    )
    print(f"  {season}: {anchor}", flush=True)

    started = time.perf_counter()
    swept = []
    for kind in pools:
        live = loaded.panels[kind]
        rows = [
            r
            for r in board_inputs(
                live,
                kind=kind,
                names=names,
                replacement_levels=levels,
                eligibility=eligibility,
                season=season,
            )
            if r.sgp >= args.min_sgp
        ]
        print(f"  {kind}: {len(rows)} players with a {season} line", flush=True)
        if loaded.no_ros[kind]:
            print(
                f"    {len(loaded.no_ros[kind])} dropped: a {season} line but no row in "
                f"the {loaded.snapshot_date} rest-of-season snapshot",
                flush=True,
            )
        # The comp pool must NOT contain the in-progress season: a two-thirds year would
        # be averaged in as though it were a full one. DERIVED from `live` rather than
        # loaded again -- a second `load()` re-reads a 4.7MB CSV and runs two more
        # full-panel `apply` passes for a frame that is this one minus its partial rows.
        # Verified identical on both pools: same ids, same seasons, max |sgp diff| 0.0.
        complete = live[~live["partial_season"]].reset_index(drop=True)
        # ONE fit, serving both scales. This used to ask for `scales=("var",)` to avoid a
        # second fit per player the CLI does not print; since #331 VAR is the raw fit
        # minus the slot's floor, so there is no second fit left to decline.
        swept += sweep_pool(rows, complete, kind, horizons)

    scored = totals(swept, horizons, scale="var")

    # RANK FIRST, then filter. `add_ranks` documents itself as ranking "over the whole
    # scored pool", and the web board does exactly that -- so ranking the filtered subset
    # here renumbered 1..N and gave the same player a different number on the two
    # surfaces. The visible consequence is that the # column now has GAPS when
    # --min-support drops a row: a rank is a position among everyone the model could
    # price, not among whatever survived a display filter.
    add_ranks(scored)
    ranked = len(scored)
    if args.min_support > 0:
        dropped = [r for r in scored if r["support"] < args.min_support]
        scored = [r for r in scored if r["support"] >= args.min_support]
        # Say what was dropped. A silently shortened board reads as "these are the best
        # players", when it is "these are the ones the model can speak to".
        print(f"  dropped {len(dropped)} rows below --min-support {args.min_support:.0%}")
    if not scored:
        print("\nnothing scored -- check --min-sgp and that the panel covers this season")
        return 1
    render(scored, args.top, horizons, levels, season, ranked, detail=args.detail)
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
            # NAME THE SOURCE. `live_rosters` serves prod Upstash normally and the
            # hand-transcribed manual store in manual mode, and this line is what an
            # operator reads to confirm which vintage is on screen -- so hardcoding
            # "Upstash" printed a false provenance claim in exactly the mode where
            # vintage is the entire concern.
            source = "the manual store" if manual_store_active() else "Upstash"
            print(f"\n  {len(spots)} roster spots read from {source}")
            # The index is READ-ONLY, so the stamping the CLI needs happens here
            # rather than inside it -- these rows are the CLI's own and mutating
            # them is safe, which is not true of the web's cached rows.
            index = index_rosters(scored, spots, config.team_name)
            for row in scored:
                # One lookup, one normalize. `team_for`/`status_for` each normalize
                # again, so routing through both cost three NFKD passes per row to
                # read two attributes off one entry.
                key = (normalize_name(row["name"]), row["pool"])
                spot = index.spot_of.get(key)
                row["team"] = spot.team if spot else None
                row["status"] = spot.status if spot else ""
                row["teams"] = index.owners_of.get(key, frozenset())
        except Exception as exc:
            if show_teams:
                raise
            print(f"\n  NOTE: rosters unavailable ({type(exc).__name__}); CSV has no team column.")
            spots, index = [], index_rosters([], [], config.team_name)
        if show_teams:
            by_team(
                scored,
                spots,
                index,
                config.team_name,
                args.per_team,
                season,
                horizons,
                args.team,
                keep=config.keepers_per_team,
                detail=args.detail,
            )
    if args.csv:
        pd.DataFrame(scored).sort_values("rank_total").to_csv(args.csv, index=False)
        print(f"\n  wrote {len(scored)} rows to {args.csv}")
    print(f"\n  scored in {time.perf_counter() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
