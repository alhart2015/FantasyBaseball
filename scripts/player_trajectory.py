"""Career SGP trajectory from historical comparables (#303).

A player produced X SGP last year and is on pace for Y this one, at age A. What does
the rest of his career look like?

STANDALONE. Reads `data/trajectory/` and touches nothing in the keeper pipeline.

Three matchers, selected with `--match`. **`shape` is the default**: it beats level
matching out of sample on every elite slice, and by ~21% RMSE on the case a keeper
decision actually turns on -- a star coming off a down year, where level matching
under-predicts by 3.31 SGP a year. Re-measure with `scripts/backtest_trajectory.py`;
those numbers come from there and nowhere else, and they are HITTERS ONLY (#313).
Shape needs BOTH of the player's last two seasons, which `--player` looks up for you.

    shape     fit forward SGP on both anchors, kernel-weighted age and level (#310)
    current   comps matched on this season's level alone -- the original estimator
    track     current, plus a hard band on the prior season too (#305)

Usage:
    python scripts/player_trajectory.py --player "Juan Soto"
    python scripts/player_trajectory.py --pool hitter --age 25 --sgp 13 --prior-sgp 18
    python scripts/player_trajectory.py --player "Juan Soto" --show-anchors
    python scripts/player_trajectory.py --pool hitter --age 25 --sgp 13 --match current
    python scripts/player_trajectory.py --player "Bobby Witt Jr." --match current --show-comps 15

Build the panel first (one time, ~1 minute):
    python scripts/build_pt_panel.py --start 2000 --end 2026 --out-dir data/trajectory
"""

from __future__ import annotations

import argparse
import sys
from functools import lru_cache
from pathlib import Path

import pandas as pd

# --show-comps prints real MLBAM names; 369 of them are non-ASCII and mangle (or, off
# cp1252, raise UnicodeEncodeError mid-table) under the Windows default.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.config import load_config
from fantasy_baseball.trajectory.comps import (
    DEFAULT_BAND,
    Trajectory,
    comp_trajectory,
)
from fantasy_baseball.trajectory.era import era_normalize
from fantasy_baseball.trajectory.panel import (
    DEFAULT_PANEL_DIR,
    load_scored_panel,
    prorate_partial,
    season_elapsed_fraction,
)
from fantasy_baseball.trajectory.shape import shape_trajectory
from fantasy_baseball.utils.name_utils import normalize_name

PEOPLE_CACHE = PROJECT_ROOT / "data" / "cache" / "keeper_skills"

#: Below this many comps the path is directional at best. Not a statistical threshold --
#: a floor at which the printed precision stops being honest.
THIN_COMPS = 20


@lru_cache(maxsize=1)
def _people() -> pd.DataFrame:
    """Every MLBAM people cache in the shared directory, unioned, for name lookup.

    UNIONED rather than ranked, because no single-file rule is stable here. This
    directory is shared with the keeper pipeline, whose `--end` defaults to the current
    year, so any 2027 keeper rebuild drops `mlb_people_all_2010_2027.csv` beside the
    trajectory build's `..._2000_2026.csv`. Ranking on (end, -start) then prefers the
    2027 file and silently loses the 2000-2009 players -- 2060 of them, 1923 present in
    the trajectory panel -- exactly the regression a filename sort caused before, just
    re-armed on a timer. A plain string sort is worse still ("2010" > "2000").

    A union has no such failure mode: ids are stable and a name never disagrees between
    caches, so more files can only mean better coverage. Deduplicated on `id`, keeping
    the newest file's spelling.
    """
    caches = sorted(PEOPLE_CACHE.glob("mlb_people_all_*.csv"))
    if not caches:
        raise FileNotFoundError(
            f"no people cache in {PEOPLE_CACHE}; run scripts/build_pt_panel.py first"
        )
    people = pd.concat(
        [pd.read_csv(path, usecols=["id", "fullName"]) for path in caches],
        ignore_index=True,
    ).drop_duplicates(subset="id", keep="last")
    people["norm"] = people["fullName"].map(normalize_name)
    return people


def _names() -> pd.Series:
    """mlbam_id -> full name, so a comp list is readable rather than a column of ids."""
    people = _people()
    return people.set_index("id")["fullName"]


def _resolve_player(
    name: str,
    panels: dict[str, pd.DataFrame],
    calendar: pd.DataFrame,
    mlbam_id: int | None = None,
) -> list[tuple[str, int, int, float]]:
    """One (pool, mlbam_id, age, sgp) per pool the player was used in, pace-adjusted.

    Returns a LIST, because this league drafts and scores a two-way player as two
    separate assets -- a hitter and a pitcher -- so each half gets its own trajectory
    rather than one of them being picked and the other silently dropped. `panel._in_role`
    has already decided what counts as being used in a role, so appearing in both pools
    here means genuinely two-way, not a position player's one mop-up inning.

    `calendar` is the HITTER panel including partial seasons, used only to date the
    in-progress season -- see `season_elapsed_fraction` for why that must not come from
    the pitcher panel.

    A normalized name is NOT unique (58 hitters share one with another player: two Chris
    Youngs, two Chris Carters, accent-collapsed pairs like Angel Sanchez). Rather than
    pooling their seasons and taking whichever namesake played most recently -- which
    silently prints one player's trajectory under the other's name -- an ambiguous name
    lists the candidates and stops. `mlbam_id` is the way through.
    """
    people = _people()
    hits = people[people["norm"] == normalize_name(name)]
    if hits.empty:
        raise SystemExit(f"no player named {name!r} in the people cache")

    named = {int(i) for i in hits["id"]}
    if mlbam_id is not None and mlbam_id not in named:
        # Without this check the id silently WINS over the name and the trajectory is
        # printed under a player it does not belong to -- reintroducing, through the
        # disambiguation flag itself, exactly the mix-up the flag exists to prevent.
        actual = people.set_index("id")["fullName"].get(mlbam_id, "an unknown player")
        raise SystemExit(
            f"--mlbam-id {mlbam_id} is {actual}, not {name!r}. "
            f"Ids for that name: {', '.join(str(i) for i in sorted(named))}"
        )
    ids = {mlbam_id} if mlbam_id is not None else named
    found = []
    for pool, panel in panels.items():
        rows = panel[panel["mlbam_id"].isin(ids)]
        if not rows.empty:
            found.append((pool, rows))
    if not found:
        who = f"mlbam id {mlbam_id}" if mlbam_id is not None else name
        raise SystemExit(f"{who} is in the people cache but has no observed season in the panel")

    present = sorted({int(i) for _, rows in found for i in rows["mlbam_id"]})
    if len(present) > 1:
        names = people.set_index("id")["fullName"]
        lines = []
        for pid in present:
            spans = [rows[rows["mlbam_id"] == pid] for _, rows in found]
            last = max(int(s["season"].max()) for s in spans if not s.empty)
            lines.append(f"    --mlbam-id {pid}   {names.get(pid, '?')}, through {last}")
        raise SystemExit(
            f"{name!r} matches {len(present)} different players; pick one:\n" + "\n".join(lines)
        )

    if len(found) > 1:
        print(f"{name} was used in both roles; scoring each separately.")

    resolved = []
    for pool, rows in found:
        row = rows.loc[rows["season"].idxmax()]
        season = int(row["season"])
        sgp = float(row["sgp"])
        if bool(row["partial_season"]):
            fraction = season_elapsed_fraction(calendar, season)
            paced = prorate_partial(sgp, fraction)
            print(
                f"{name} ({pool}): {season} is {fraction:.0%} complete -- "
                f"{sgp:.1f} SGP so far, pacing to {paced:.1f}"
            )
            sgp = paced
        resolved.append((pool, int(row["mlbam_id"]), int(row["age"]), sgp))
    return resolved


def _prior_for(panel: pd.DataFrame, mlbam_id: int, args: argparse.Namespace) -> float:
    """The player's own SGP in the season before the one being scored.

    Explicit --prior-sgp wins. Otherwise it is read off the panel, and a player who was
    not in the league that year scores 0 -- the same convention comps get, and for a
    young player the normal case rather than an error.
    """
    if args.prior_sgp is not None:
        return args.prior_sgp
    rows = panel[panel["mlbam_id"] == mlbam_id]
    current = int(rows["season"].max())
    previous = rows[rows["season"] == current - 1]
    prior = float(previous["sgp"].sum()) if not previous.empty else 0.0
    print(f"  prior season ({current - 1}): {prior:.1f} SGP")
    return prior


def _no_support(traj: Trajectory) -> bool:
    """True when nothing could be scored, whatever the reason.

    Tests `observable`, NOT `n_comps`. Keying on the cohort size left a window --
    enough seasons to form a cohort, too few EFFECTIVE to fit any horizon -- in which
    every row printed "--" and the footer still read "total over 0 years: 0.0 SGP",
    a fabricated forecast of no future value for a player the model could not score.
    """
    return not traj.observable


def _warn_if_thin(traj: Trajectory) -> None:
    """Say so when the support is too thin for the printed precision.

    Reads the EFFECTIVE size, not the row count. Under kernel weighting those diverge:
    a 41-row shape fit carrying an effective 15 cleared a raw-count threshold while
    fitting `on_peak` at -1.03 -- more production last year predicting less next year --
    and printed unqualified to two decimals.
    """
    support = min((p.n_effective for p in traj.observable), default=0.0)
    if support >= THIN_COMPS:
        return
    unit = "effective fitting seasons" if traj.mode == "shape" else "comps"
    widen = "the kernels" if traj.mode == "shape" else "--band/--prior-band"
    print(
        f"  *** THIN: {support:.0f} {unit} at the weakest horizon. The numbers below are"
        f" directional only -- widen {widen}, or read a different mode. ***"
    )


def _print_total(traj: Trajectory) -> None:
    covered, asked = len(traj.observable), len(traj.path)
    note = "" if covered == asked else f"  (only {covered} of {asked} are observable)"
    print(f"\n   total over {covered} years: {traj.total:.1f} SGP{note}")


def render(traj: Trajectory, show_comps: int) -> None:
    span = f"{traj.seasons[0]}-{traj.seasons[1]}" if traj.seasons else "n/a"
    print(f"\n{traj.kind.upper()}: {traj.sgp:.1f} SGP in an age-{traj.age} season")
    if traj.mode == "shape":
        # A fitted prediction, not an average over a handful of careers: `n` counts the
        # rows the relationship was fit on, and nothing was excluded on a cliff.
        print(f"  SHAPE: {traj.prior_sgp:.1f} last year -> {traj.sgp:.1f} now")
        if _no_support(traj):
            # NOT "not yet observable" -- nothing was censored here. Either no season is
            # near enough to enter the kernels at all, or too few carry enough weight to
            # fit. Both must stop before the table, which would otherwise print NaNs
            # under a "total over 0 years: 0.0 SGP" that reads as a real forecast.
            if traj.n_comps == 0:
                print(
                    "  NO FIT -- no season is close enough in age and level to score "
                    "this query. Widen the kernels, or check the age and --prior-sgp."
                )
            else:
                print(
                    f"  NO FIT -- {traj.n_comps} season(s) are near enough to weigh, but "
                    "none of the horizons reach the effective-support floor. Widen the "
                    "kernels, or try --match current."
                )
            return
        print(f"  fit on {traj.n_comps} weighted seasons, {span}")
        print(
            f"  their average shape: {traj.mean_prior:.1f} -> {traj.mean_start:.1f} SGP "
            "(kernel-weighted)"
        )
        _warn_if_thin(traj)
        # Weighted survival against the EFFECTIVE size, so every column in the row
        # describes the same population the fit used. A raw count beside a weighted
        # median invited the reader to take both as properties of the prediction.
        print("\n   age    pred   +/-SE  +/-spread   median   played (of eff)   if played")
        for p in traj.path:
            if p.n == 0:
                print(f"   {p.age:3d}        --      --         --       --   (not fittable)")
                continue
            print(
                f"   {p.age:3d}   {p.mean:7.2f}   {p.se:5.2f}    {p.spread:7.2f}  {p.median:7.2f}"
                f"     {p.survival:5.0%} (of {p.n_effective:5.0f})  {p.mean_if_survived:6.2f}"
            )
        _print_total(traj)
        if show_comps:
            # The block below reads sgp0/hN, which a shape frame does not have -- and a
            # shape fit has no per-query comps to list, only a weighted population.
            print(
                f"\n   (--show-comps {show_comps} lists individual comps, which only the "
                "comp matchers have; add --match current, or --show-anchors for the "
                "fitted coefficients)"
            )
        return

    if traj.prior_sgp is not None:
        print(f"  matched on TRACK RECORD: {traj.prior_sgp:.1f} -> {traj.sgp:.1f} SGP")
    # `n_comps` and `span` describe the NEAREST horizon's cohort; later horizons see
    # fewer, which is why the per-row n is printed rather than left to this header.
    print(f"  {traj.n_comps} comps at +1 within +/-{traj.band} SGP, age-{traj.age} seasons {span}")
    if _no_support(traj):
        print("  NO COMPS -- widen --band/--prior-band or check the age")
        return
    # `mean_prior` skips comps whose own prior sits before the panel begins, so it can
    # describe a SMALLER cohort than `mean_start` beside it -- and is NaN when every
    # comp is censored, which rendered as "after nan the year before". Say what it
    # covers rather than implying both halves of the sentence share a denominator.
    known = int(traj.comps["sgp_prior"].notna().sum()) if "sgp_prior" in traj.comps else 0
    if known == 0:
        prior_note = "; no comp's prior year is inside the panel"
    elif known < traj.n_comps:
        prior_note = f", after {traj.mean_prior:.1f} the year before ({known} of {traj.n_comps})"
    else:
        prior_note = f", after {traj.mean_prior:.1f} the year before"
    print(f"  comps started from {traj.mean_start:.1f} SGP on average{prior_note}")
    _warn_if_thin(traj)

    print("\n   age   exp SGP    +/-SE  +/-spread   median   still playing   if playing")
    for p in traj.path:
        if p.n == 0:
            print(f"   {p.age:3d}        --        --         --       --   (not yet observable)")
            continue
        print(
            f"   {p.age:3d}   {p.mean:7.2f}    {p.se:5.2f}    {p.spread:7.2f}  {p.median:7.2f}"
            f"    {p.survivors:5d}/{p.n} ({p.survival:4.0%})  {p.mean_if_survived:6.2f}"
        )
    _print_total(traj)

    if show_comps:
        # Ranked by closeness to the query, not by sgp0 -- "show me the comps" means the
        # ones actually driving the average, and nlargest would only ever show the top
        # edge of the band.
        top = traj.comps.assign(gap=(traj.comps["sgp0"] - traj.sgp).abs()).nsmallest(
            show_comps, "gap"
        )
        top = top.assign(player=top["mlbam_id"].map(_names()).fillna(top["mlbam_id"].astype(str)))
        cols = ["player", "season", "sgp0"] + [f"h{p.horizon}" for p in traj.path]
        print(f"\n   {len(top)} closest comps (0 = did not play, -- = season not played yet):")
        print(top[cols].to_string(index=False, na_rep="--", float_format=lambda v: f"{v:6.2f}"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--player", help="look the age and current pace up by name")
    parser.add_argument(
        "--mlbam-id",
        type=int,
        help="disambiguate --player when a name matches more than one player",
    )
    parser.add_argument("--pool", choices=("hitter", "pitcher"))
    parser.add_argument("--age", type=int, help="the player's age in the season he is producing")
    parser.add_argument("--sgp", type=float, help="full-season SGP pace")
    parser.add_argument(
        "--band",
        type=float,
        # default=None, NOT DEFAULT_BAND: the validation below must distinguish
        # "the user asked for a band" from "nobody mentioned one".
        default=None,
        help=f"comp width in SGP for --match current/track (default {DEFAULT_BAND})",
    )
    parser.add_argument(
        "--match",
        choices=("shape", "track", "current"),
        default="shape",
        help=(
            "'shape' (default) fits forward SGP on both anchors -- last year and this "
            "year -- with kernel-weighted age and level, excluding nobody on a cliff "
            "(#310); 'current' matches comps on this season's level alone; 'track' "
            "adds a hard band on the prior season too (#305). --band/--prior-band "
            "apply to the two comp modes only; shape has no band"
        ),
    )
    parser.add_argument(
        "--prior-sgp",
        type=float,
        help=(
            "prior-season SGP, required by --match shape/track without --player "
            "(looked up automatically with --player); 0 means he was not in the majors"
        ),
    )
    parser.add_argument(
        "--prior-band",
        type=float,
        help="comp width on the prior season (defaults to --band)",
    )
    parser.add_argument("--horizon", type=int, default=5, help="years forward to project")
    parser.add_argument("--show-comps", type=int, default=0, metavar="N")
    parser.add_argument(
        "--show-anchors",
        action="store_true",
        help="with --match shape, print the fitted coefficients behind each prediction",
    )
    parser.add_argument(
        "--no-era-adjust",
        action="store_true",
        help="pool raw SGP without restating each season's run environment",
    )
    parser.add_argument(
        "--panel-dir",
        type=Path,
        default=DEFAULT_PANEL_DIR,
        help="where to read panels from (default data/trajectory/); relative to the repo",
    )
    args = parser.parse_args()

    if not args.player and not (args.pool and args.age is not None and args.sgp is not None):
        parser.error("pass --player, or all of --pool/--age/--sgp")
    if args.horizon < 1:
        parser.error("--horizon must be at least 1")
    # Every flag that cannot affect the chosen mode is refused, not ignored. Accepting
    # one silently lets a user tune a parameter, see a byte-identical answer, and
    # conclude the parameter does nothing -- or worse, believe an input was honoured
    # that was dropped. --prior-band was already refused; the rest were not.
    if args.prior_band is not None and args.match != "track":
        parser.error(
            f"--prior-band applies to --match track; {args.match} "
            f"{'uses kernels, not bands' if args.match == 'shape' else 'has no prior band'}"
        )
    if args.band is not None and args.match == "shape":
        parser.error("--band applies to --match current/track; shape uses kernels, not bands")
    if args.prior_sgp is not None and args.match == "current":
        parser.error(
            "--prior-sgp applies to --match shape/track; --match current scores on this "
            "season alone and would discard it"
        )
    if args.show_anchors and args.match != "shape":
        parser.error("--show-anchors applies to --match shape; the comp matchers fit no anchors")

    # Anchor to the REPO, mirroring build_pt_panel._anchor on the write side. The
    # documented build command passes a RELATIVE --out-dir, so a reader resolving the
    # same string against the cwd would glob `<cwd>/data/trajectory`, find nothing, and
    # tell the user to build a panel that is already built.
    if not args.panel_dir.is_absolute():
        args.panel_dir = PROJECT_ROOT / args.panel_dir

    # league.yaml's denominators are what every other valuation in this repo prices in,
    # so the trajectory is quoted in the same units as a draft board or a keeper line.
    overrides = load_config(PROJECT_ROOT / "config" / "league.yaml").sgp_overrides

    def load(kind: str, include_partial: bool) -> pd.DataFrame:
        panel = load_scored_panel(
            kind,
            panel_dir=args.panel_dir,
            sgp_overrides=overrides,
            include_partial=include_partial,
        )
        if args.no_era_adjust:
            return panel
        return era_normalize(panel, kind, sgp_overrides=overrides)

    if args.player:
        # The query needs the in-progress season; the comp pool must not have it.
        wanted = [args.pool] if args.pool else ["hitter", "pitcher"]
        live = {k: load(k, True) for k in wanted}
        # Dating the season is a league fact and must come off the hitter panel even for
        # a pitcher query -- pitcher `games` counts appearances, not team games.
        calendar = live.get("hitter") if "hitter" in live else load("hitter", True)
        queries = [
            # `is not None`, never `or`: --sgp 0 and --age 0 are falsy but meaningful.
            (
                pool,
                age if args.age is None else args.age,
                sgp if args.sgp is None else args.sgp,
                _prior_for(live[pool], pid, args) if args.match in ("track", "shape") else None,
            )
            for pool, pid, age, sgp in _resolve_player(args.player, live, calendar, args.mlbam_id)
        ]
        if args.sgp is not None:
            # The looked-up pace was printed above but is not what gets scored; say so
            # rather than leaving two different numbers on screen with no indication of
            # which one drove the table.
            print(f"  (--sgp {args.sgp} overrides the pace above)")
    else:
        if args.match in ("track", "shape") and args.prior_sgp is None:
            # Never guess it. Assuming last year equalled this year is a real modelling
            # claim -- it says the season is representative -- and it would silently
            # move the answer for exactly the players (breakouts, collapses) the two
            # anchors exist to tell apart.
            parser.error(
                f"--match {args.match} needs the player's PRIOR season too; pass "
                "--prior-sgp N (use 0 if he was not in the majors), or --player NAME "
                "to look it up, or --match current to score on this season alone"
            )
        # Gate the prior on the MODE, exactly as the --player branch does. Passing it
        # through unconditionally made `--match current --prior-sgp N` run the track
        # estimator instead -- silently overriding the mode the user asked for.
        queries = [
            (args.pool, args.age, args.sgp, args.prior_sgp if args.match != "current" else None)
        ]

    horizons = tuple(range(1, args.horizon + 1))
    for pool, age, sgp, prior in queries:
        if args.match == "shape":
            traj, anchors = shape_trajectory(
                load(pool, False),
                kind=pool,
                age=age,
                sgp=sgp,
                peak=prior,
                horizons=horizons,
            )
            if args.show_anchors:
                print("\n   fitted anchors (forward = intercept + a*now + b*last year):")
                print("     h  intercept   a(now)  b(last)   n_fit   n_eff")
                for a in anchors:
                    print(
                        f"     {a.horizon}   {a.intercept:8.2f} {a.on_down:8.3f} "
                        f"{a.on_peak:8.3f} {a.n_fit:7d} {a.n_effective:7.0f}"
                    )
        else:
            traj = comp_trajectory(
                load(pool, False),
                kind=pool,
                age=age,
                sgp=sgp,
                band=DEFAULT_BAND if args.band is None else args.band,
                prior_sgp=prior,
                prior_band=args.prior_band,
                horizons=horizons,
            )
        render(traj, args.show_comps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
