"""Sweep the whole player pool and push the trajectory board to prod Upstash (#321).

This runs OFFLINE, on a machine that has the panel. It is not part of the refresh
pipeline and cannot be: the fit reads `data/trajectory/*.csv` and
`data/cache/keeper_skills`, both gitignored, so neither exists on Render. The season
dashboard is a pure reader of what this writes.

    python scripts/push_trajectory_board.py                 # sweep + push to prod
    python scripts/push_trajectory_board.py --dry-run       # sweep, report sizes, no write
    python scripts/push_trajectory_board.py --max-horizon 3 # shorter dropdown, faster

TWO KEYS, ONE VINTAGE (#344). `cache:trajectory_board` is the board every view reads;
`cache:trajectory_chart_data` is the per-player career history and comps that ONLY the
player chart reads. They were one blob, and carrying the extras inline more than doubled
what the two default views had to fetch and parse to render rows that never show them.
Both are stamped with the SAME `generated_at`, computed once in `build_payload`, and the
player view refuses to draw extras whose stamp disagrees with the board's -- a stale
career line under a fresh projection renders perfectly and is simply wrong. The chart
data is written FIRST, so a successful board write implies its extras are already there.

The board is only as fresh as the last run of this script, which is why the payload
carries its own vintage and the page prints it. TWO vintages now (#348): the panel it
was fitted on, and the rest-of-season snapshot the in-progress season is anchored to.
Re-run it when the panel is rebuilt (`scripts/build_pt_panel.py`) or when a fresh ROS
export lands (`scripts/ingest_ros_export.py`) -- either one moves every current-season
figure the fits are built on.

WHY ONE SWEEP AT THE LONGEST HORIZON. The end-year dropdown needs no refit -- see
`fantasy_baseball.trajectory.sweep`, where the reasoning and the measurement live.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

#: Longest projected year offered. Five forward years is also what the player chart
#: draws (#324).
#:
#: Cost, re-measured 2026-08-06 on the 2000-2026 panel: 33.5s at three years, ~52s at
#: five, so roughly 4-5s per pool per extra year. The figure here used to be quoted "per
#: pool per scale" -- there is only one scale now (#331 made VAR a shift, so the second
#: fit went), and the old per-scale number does not reconcile with what one scale costs
#: today, so this is the measurement rather than the old one halved.
DEFAULT_MAX_HORIZON = 5

#: Current-season pace below which a player is not fitted. THE SAME CUT
#: `trajectory_board.py --min-sgp` defaults to, and it is a `>=` test rather than a
#: no-op: 171 of the 1,340 players with a 2026 line pace NEGATIVE, and the CLI has never
#: scored them. Every analysis on this model ran at this cut, so the web board matching
#: it is what makes the two comparable.
#:
#: Deliberately not a control and deliberately not raised. It cannot change any fitted
#: number -- the fitting pool is `complete`, which the query rows never enter -- but it
#: does change RANKS, and a young player pacing low is exactly the ambiguous keeper call.
MIN_SGP = 0.0

#: Comps stored per player -- the SAME constant the view clamps `n` against, imported
#: rather than re-declared here. See its docstring in `career_comps`; a second literal
#: beside this one is what the deleted parity test existed to police.
from fantasy_baseball.trajectory.career_comps import MAX_COMPS, closest_careers, match_pool
from fantasy_baseball.trajectory.model import DEFAULT_LOOKBACK


def _career_by_age(seasons) -> dict[int, float]:
    """One player's REALIZED SGP keyed by age, with holes left as holes.

    The raw reading of the stored frame, shared by everything that needs it: `_arc`
    fills the holes to draw a line, and the `career` dict `build_payload` assembles for
    `player_comps` must not fill them at all. They were
    one comprehension doing both jobs, and the fill is exactly the thing backward
    matching cannot tolerate -- see `Prepared.back`.

    Ages within one player are distinct: `collapse_split_seasons` gives one row per
    (mlbam_id, season) and age advances with the season, so nothing is collapsed here.
    """
    if seasons is None:
        return {}
    return {int(a): round(float(s), 4) for a, s in zip(seasons["age"], seasons["sgp"], strict=True)}


def _arc(seasons) -> list[list[float]]:
    """One player's stored career as ``[[age, sgp], ...]``, ascending, ``[]`` when absent.

    ONE spelling for the two arcs a chart pairs on a single axis -- the subject's own
    history and a comp's whole career. They came off the same `by_id` frame through two
    hand-copied comprehensions, one of which sorted and one of which did not, so the
    blob's two `[[age, sgp]]` lists carried different ordering guarantees for no reason
    a reader could find.

    **A MISSED SEASON IS A ZERO HERE, AND A GAP IN THE MATCH.** This is a DRAWING, and a
    line that skips an age draws straight through a year the rest of the page shows as
    nothing, on a card whose whole job is showing where the busts sit in the arc. So the
    hole is filled with the 0 a roster slot actually got.

    That is the opposite of what `closest_careers` does with the same hole, deliberately
    (#358): matching on a filled 0 makes an injured star and a journeyman look alike.
    The two conventions live one function apart on purpose -- `Prepared.back` and
    `Prepared.forward` carry the same split for the same reason.

    Filled only BETWEEN observed seasons. Before his debut and after his last year he was
    not a zero, he was absent, and the career span is what the arc is about.

    Sorted (implicitly, by iterating the span) even though `groupby` happens to emit
    ascending: nothing in the panel contract promises it, and `_netted` on the read side
    sorts for the same reason.
    """
    scored = _career_by_age(seasons)
    if not scored:
        return []
    return [[age, scored.get(age, 0.0)] for age in range(min(scored), max(scored) + 1)]


def _playing_time(frame, volume: str) -> dict[tuple[int, int], float]:
    """``(mlbam_id, season) -> PA or IP`` for every row in ``frame``, absent where unknown.

    ON A 162-GAME FOOTING, not as the box score printed it. `_scale_short_schedules`
    multiplies this column by `162 / scheduled_games` on load, so a 2020 season arrives at
    2.7x its literal count -- Alvarez's nine 2020 plate appearances are 24 here. That is
    the RIGHT number for the question the column is asked ("was this a part-time
    season?"), because the SGP printed beside it was scaled by the very same factor; an
    unscaled volume against a scaled SGP is the pair that actually misleads. It does have
    to be SAID, which the page does -- a column headed "PA/IP" showing 24 for a 9-PA
    season is a wrong number until it is labelled.

    SUMMED over a split season, like `collapse_split_seasons` sums its SGP. A traded
    player's two rows are half a season each, and taking either one alone would print a
    300-PA year beside the full-season SGP the same row was matched on.
    `collapse_split_seasons` cannot supply this: it keeps only sgp and age.

    A GROUP HOLDING ANY NaN IS DROPPED, and the two-part test is what makes that true.
    `Series.sum()` defaults to `skipna=True, min_count=0`, so an ALL-NaN group sums to a
    real `0.0` that sails through `notna` -- a single-part guard could never fire, and it
    would store `pt: 0.0`, printing "PA/IP: 0" beside a real SGP figure and inverting the
    exact hurt-versus-finished signal the column exists for (#357). A PARTLY-NaN group is
    worse than useless too: it sums only the half it can see and understates a split
    season. Both become absent, which the table renders as a dash.

    A FUNCTION, not four lines inline in `build_payload`, for one reason: the test that
    covers it was written inline against a local frame and therefore asserted pandas'
    behaviour rather than this rule, so reverting the guard would have left it green.
    A named seam is what lets the test drive the real thing.

    Vectorized, because the per-group Python alternative would run once per
    (player, season) over ~16k rows per pool.
    """
    if volume not in frame.columns:
        return {}
    keys = [frame["mlbam_id"], frame["season"]]
    totals = frame[volume].groupby(keys).sum(min_count=1)
    totals = totals.where(~frame[volume].isna().groupby(keys).any())
    return {
        (int(i), int(season)): round(float(v), 1)
        for (i, season), v in totals.items()
        if pd.notna(v)
    }


def player_comps(prepared, player, career: dict[int, float], names: dict, pt: dict) -> list[dict]:
    """One player's stored comp block -- the careers that looked like his, and what
    happened to them. ``[]`` when nothing historical resembles him closely enough.

    TAKES THE PLAYER, not his `age` and `mlbam_id` as separate arguments. It was briefly
    split into the two fields it reads, on the theory that a function should ask for the
    narrowest thing it needs. That was wrong twice over, and both showed up immediately:

    * Two same-typed ints sit adjacent in the signature, so `player_comps(prepared,
      player.mlbam_id, player.age, ...)` -- the natural id-then-age ordering -- type-checks
      under mypy, runs without error, and returns `[]` for every player, because no
      candidate sits at `age == 592885`. The result is a comp-less board that renders
      perfectly. The object made that unrepresentable.
    * It silently gutted two tests. `_swept(2)` and `_swept(3)` differ ONLY in path
      length, so once the function stopped reading the path, both tests were calling it
      twice with identical arguments and comparing the result to itself -- including
      `test_a_comp_is_selected_on_the_career_never_on_the_prediction`, the named pin for
      this issue's whole premise, which `closest_careers`'s docstring cites by name.

    The object is the narrowest thing it needs: it is what guarantees the age and the id
    describe one player.

    MATCHED BACKWARD (#358). ``career`` is what he has actually DONE, and the comps are
    chosen on that alone; the fitted path never enters. The forward matcher this replaced
    chose them by how close their realized path landed to our PREDICTION, which made the
    comp set a redrawing of the forecast rather than evidence about it.

    That also removed a failure mode rather than moving it. The old matcher raised when
    ``len(predicted) != len(prepared.horizons)`` -- reachable for a player observable at
    h=1..3 but not h=4..5, whom ``sweep_pool`` keeps -- and one such player discarded a
    whole ~52s sweep and pushed nothing. There is no ``predicted`` here, so a short fit
    is no longer a length contract to violate: he gets full-length comp paths and the
    view truncates them to the horizons he was actually fitted at.

    An EMPTY list means no candidate shared enough of his ages (`required_overlap`) --
    the honest answer for a career the panel has no parallel for, and the page already
    renders it with an explanation.

    Names and playing time are attached HERE, not in ``closest_careers``: naming needs
    the people cache and volume needs the panel frame, and keeping both out of that
    module is what lets it be tested with no data files. An unknown id renders as its id
    rather than vanishing -- a comp is still a comp.
    """
    return [
        {
            # THE JOIN KEY for the stored career map, not decoration. `chart_key(id,
            # pool)` is how a per-comp card finds the arc it draws; a display name
            # cannot do it (#284), and two players sharing one normalized name is
            # common enough to matter.
            "id": c.mlbam_id,
            "name": names.get(c.mlbam_id, str(c.mlbam_id)),
            "season": c.season,
            "rmse": round(c.rmse, 3),
            # HOW MUCH CAREER THE MATCH ACTUALLY SAW. Stored rather than derived on read:
            # the reader has no way to recompute it, and a 3-age match and an 8-age match
            # at the same RMSE are not the same claim.
            "overlap": c.overlap,
            # PA (hitters) or IP (pitchers) in the anchor season, so a reader can tell a
            # comp who declined from one who was hurt (#357). None when the pool frame
            # carried no volume column.
            "pt": pt.get((c.mlbam_id, c.season)),
            "path": [round(v, 3) for v in c.path],
        }
        for c in closest_careers(
            prepared,
            career,
            age=player.age,
            n=MAX_COMPS,
            # He is not his own comp. The forward-observability mask usually removes him
            # anyway, but only because his anchor season is the newest one -- a fact
            # about this panel, not a rule. See `closest_careers`.
            exclude_id=player.mlbam_id,
        )
    ]


class EmptyPoolError(RuntimeError):
    """Raised when a pool scores zero players, so the push is refused.

    Pushing anyway overwrites a complete ``cache:trajectory_board`` with a
    half board -- and it renders perfectly: the page reports "Showing 50 of
    563 scored" under "Of everyone you could hold", with every pitcher
    silently gone and only the panel filename in the vintage line to say so.

    The reachable cause is a panel-vintage mismatch. ``panel_path`` picks the
    newest panel for each kind INDEPENDENTLY, while the base season comes off
    the HITTER panel alone, so a hitter panel rebuilt through 2026 against a
    pitcher panel still ending 2025 makes ``board_inputs(season=2026)`` find
    ``current.empty`` and return no rows.

    Same shape as the 2026-06-04 ROS incident -- a failed fetch overwriting a
    good blob with a degraded one -- and guarded the same way: abort before
    any KV read or write, keeping the last-good board.
    """


def _require_scored_pool(kind: str, rows: list, season: int, panel_name: str) -> None:
    """Abort the push unless ``kind`` scored at least one player.

    Runs during payload assembly, BEFORE any KV read or write, so a refused
    run leaves the deployed board untouched.

    Raises:
        EmptyPoolError: the pool produced no scorable rows.
    """
    if rows:
        return
    raise EmptyPoolError(
        f"Refusing to push: the {kind} pool scored 0 players for {season}. "
        f"Its panel is {panel_name} -- most likely it predates {season} while the "
        f"hitter panel (which sets the base season) does not. Pushing would replace "
        f"the deployed board with a {kind}-less one that still renders normally. "
        f"Keeping the last-good board -- rebuild the {kind} panel through {season} "
        f"and re-run."
    )


def build_payload(max_horizon: int, panel_dir: Path) -> tuple[dict, dict, int]:
    """Sweep both pools, once each. Returns the board, the chart data, and rows scored.

    ONE fit per player, on raw SGP; VAR is derived on read (#331). It said "on both
    scales" because it used to store both, and a reader who believes that is a reader
    who adds back a `scales` argument or a second `shape_trajectory` call that the
    payload no longer has anywhere to put.

    TWO payloads, ONE `generated_at`, computed here and threaded into both. Calling
    `local_now()` twice would stamp them seconds apart, and the player view's pairing
    check is an equality test -- it would refuse every chart this script writes.
    """
    from fantasy_baseball.config import load_config
    from fantasy_baseball.sgp.denominators import get_sgp_denominators
    from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
    from fantasy_baseball.trajectory.board import board_inputs, player_names, season_slots
    from fantasy_baseball.trajectory.model import collapse_split_seasons
    from fantasy_baseball.trajectory.panel import panel_path, season_elapsed_fraction
    from fantasy_baseball.trajectory.ros_anchor import load_anchored_panels
    from fantasy_baseball.trajectory.shape import prepare
    from fantasy_baseball.trajectory.sweep import (
        chart_key,
        sweep_pool,
        to_chart_payload,
        to_payload,
    )
    from fantasy_baseball.utils.time_utils import local_now

    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    overrides = config.sgp_overrides
    levels = position_aware_replacement_levels(get_sgp_denominators(overrides))
    horizons = tuple(range(1, max_horizon + 1))

    # BOTH pools, era-normalized, with the in-progress season anchored on season-to-date
    # plus a rest-of-season projection. The ordering rules that make that legitimate --
    # inject before normalizing, take the base season's era factor off the ACTUAL rows --
    # live in `load_anchored_panels`, spelled once for all three entry points.
    loaded = load_anchored_panels(
        systems=config.projection_systems,
        weights={s: config.projection_weights[s] for s in config.projection_systems},
        panel_dir=panel_dir,
        sgp_overrides=overrides,
    )
    season = loaded.season
    # Dating the in-progress season is a league fact and must come off the HITTER panel
    # even when pricing pitchers -- pitcher `games` counts appearances, not team games.
    calendar = loaded.panels["hitter"]
    cache = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
    names = player_names(cache)
    eligibility = season_slots(cache, season)
    # PROVENANCE, not an input to any fit -- the anchor replaced the pace adjustment that
    # used to consume it (#348). Still stamped and still printed: how far into the season
    # a board was built says how much of its base year is projection rather than record.
    # `games` is deliberately left un-projected by the anchor, so this is still a fact
    # about what has been played.
    elapsed = season_elapsed_fraction(calendar, season)

    swept = []
    # KEYED ON `(mlbam_id, pool)`, never the bare id -- `chart_key` serializes the pair.
    # This dict lives OUTSIDE the pool loop, and a two-way player is produced once per
    # pool, so on a bare id the pitcher pass overwrote the hitter's entry and Ohtani's
    # hitter row rendered his pitching career line and pitcher comps. See
    # `SweptPlayer.mlbam_id`.
    extras: dict[tuple[int, str], dict] = {}
    # Deduped comp careers, keyed by `chart_key(id, pool)` -- see `to_chart_payload`.
    # Outside the pool loop like `extras`, because the key already carries the pool and
    # a two-way comp gets one entry per pool under distinct keys.
    careers: dict[str, list] = {}
    excluded = {"low_sgp": 0, "no_current_line": 0, "no_ros_projection": 0}
    for kind in ("hitter", "pitcher"):
        live = loaded.panels[kind]
        candidates = list(
            board_inputs(
                live,
                kind=kind,
                names=names,
                replacement_levels=levels,
                eligibility=eligibility,
                season=season,
            )
        )
        rows = [r for r in candidates if r.sgp >= MIN_SGP]
        # Who this pool leaves out, so the page can say so. Three separate exclusions:
        # the min-SGP gate; players who had a line LAST season and none this one, who
        # are never candidates at all; and players with a current line that no
        # rest-of-season projection covers, whom the anchor drops.
        no_ros = {int(pid) for pid in loaded.no_ros[kind]}
        excluded["low_sgp"] += len(candidates) - len(rows)
        excluded["no_ros_projection"] += len(no_ros)
        # MINUS the ones the anchor already removed. They are gone from `live`'s current
        # season by construction, so without this subtraction every no-ROS player who
        # played last season is counted twice and the total overstates the gap.
        excluded["no_current_line"] += len(
            {int(pid) for pid in live.loc[live["season"] == season - 1, "mlbam_id"]}
            - {int(pid) for pid in live.loc[live["season"] == season, "mlbam_id"]}
            - no_ros
        )
        print(f"  {kind}: {len(rows)} players with a {season} line", flush=True)
        if no_ros:
            # SAID OUT LOUD, like the short-comp note below. A player who leaves the
            # board without a word looks to a reader exactly like one the model priced
            # low, which is the reading this whole exclusion block exists to prevent.
            print(
                f"    {len(no_ros)} dropped: a {season} line but no row in the "
                f"{loaded.snapshot_date} rest-of-season snapshot",
                flush=True,
            )
        started = time.perf_counter()
        # The comp pool must NOT contain the in-progress season: a two-thirds year would
        # be averaged in as though it were a full one.
        complete = live[~live["partial_season"]].reset_index(drop=True)
        produced = sweep_pool(rows, complete, kind, horizons)
        # Guard on what the sweep PRODUCED, not on what it was handed. sweep_pool
        # independently drops every player whose fitted path has no observable point, so
        # a panel whose complete seasons do not span horizon 1 yields hundreds of
        # candidates and zero scored rows -- passing an input-side check and pushing
        # the pool-less board this guard exists to refuse.
        _require_scored_pool(kind, produced, season, panel_path(kind, panel_dir).name)
        swept += produced

        # `sweep_pool` builds its own prepared state and does not return it. Preparing a
        # second time costs one vectorized reindex per pool -- cheap next to the sweep,
        # and far cheaper than widening `sweep_pool`'s signature, which the CLI and its
        # tests also call.
        # `lookback=` is what turns the backward window on. It is opt-in (see `prepare`)
        # because this is its only consumer, and `sweep_pool` above just called `prepare`
        # without it -- so the sweep no longer pays for a window it never reads.
        prepared = prepare(complete, kind=kind, horizons=horizons, lookback=DEFAULT_LOOKBACK)
        # THROUGH THE SHARED COLLAPSE, like every other reader of this panel. A
        # mid-season trade can put two rows on one player-year, and `collapse_split_seasons`
        # is where that rule lives -- `collapsed_index`'s docstring names it as the single
        # site precisely so the fitting side and a lookup side cannot spell it differently.
        # The career line drawn here was a third reader that skipped it.
        #
        # A NO-OP TODAY, and it must stay one: measured 2026-08-07 against the live
        # 2000-2026 panels, ZERO duplicate (mlbam_id, season) pairs in either complete
        # frame (16,408 hitter / 17,947 pitcher rows), and the function early-returns the
        # panel unchanged when there are none. This is consistency insurance against a
        # future panel build, not a fix for anything currently wrong.
        by_id = {int(i): g for i, g in collapse_split_seasons(complete).groupby("mlbam_id")}
        # Anchor-season volume for every candidate comp, so a card can say whether a
        # collapse was decline or absence (#357). PA for hitters, IP for pitchers -- one
        # column, named by pool, because the two pools never share a frame here.
        volume = "pa" if kind == "hitter" else "ip"
        pt = _playing_time(complete, volume)

        no_comps: dict[str, int] = {}
        for player in produced:
            history = _arc(by_id.get(player.mlbam_id))
            # WHAT HE HAS ACTUALLY DONE, which is what the comps are matched on (#358).
            # His complete seasons out of the same `by_id` the career line is drawn from,
            # plus the base season at his current age -- which `complete` does not carry
            # while it is in progress, and which is `now`: season-to-date plus the
            # rest-of-season blend, re-scored (`ros_anchor`). Without it the newest and
            # most decision-relevant year of his career is missing from the match.
            #
            # UNFILLED. `_career_by_age`, not `_arc`: a year he did not play must stay
            # absent here, or an injured star matches a journeyman.
            career = {**_career_by_age(by_id.get(player.mlbam_id)), player.age: player.now}
            comps = player_comps(prepared, player, career, names, pt)
            if not comps:
                # Only on the empty branch: `match_pool` repeats the candidate mask and
                # the overlap count, and paying that for the ~85% of players who DO get
                # comps would be a second full pass over the pool for a number nothing
                # reads.
                reason = match_pool(prepared, career, player.age, exclude_id=player.mlbam_id).reason
                no_comps[reason] = no_comps.get(reason, 0) + 1
            else:
                # THE SAME `by_id` the subject's own history comes from, so a comp's arc
                # and the subject overlay drawn on top of it are on one scale by
                # construction. Resolved per pool, because `by_id` is per pool: a hitter
                # comp looked up in the pitcher frame would draw the wrong career.
                #
                # An id `by_id` does not hold writes nothing: comps come from the same
                # `prepared` this frame built, so absence is a defect, and an empty card
                # is the honest rendering of one rather than a fabricated arc.
                for comp in comps:
                    key = chart_key(comp["id"], kind)
                    if key not in careers and (arc := _arc(by_id.get(comp["id"]))):
                        careers[key] = arc
            extras[(player.mlbam_id, player.pool)] = {"history": history, "comps": comps}
        print(f"    swept in {time.perf_counter() - started:.1f}s", flush=True)
        if no_comps:
            # SAID OUT LOUD, AND BY CAUSE. A push that quietly drops comps for part of
            # the pool renders exactly like one that did not. Broken out because the
            # three causes point at different things: "no career shares enough of his
            # ages" is about the player and is expected for the very young, while
            # "too recent to follow forward" and "no player of this age" are facts about
            # the PANEL, and an operator told the first when the truth is the third goes
            # looking in the wrong place entirely.
            total = sum(no_comps.values())
            print(
                f"    {total} got no comps (their rows and career lines are intact):",
                flush=True,
            )
            for reason, count in sorted(no_comps.items(), key=lambda kv: -kv[1]):
                print(f"      {count:5} -- {reason}", flush=True)

    # ONE stamp for both blobs -- see this function's docstring.
    generated_at = local_now().isoformat(timespec="seconds")
    payload = to_payload(
        swept,
        base_season=season,
        max_horizon=max_horizon,
        min_sgp=MIN_SGP,
        season_elapsed=round(elapsed, 4),
        # Whether `season` was still in progress when this panel was built. Read off
        # the panel rather than the calendar date: `_live_seasons` in build_pt_panel.py
        # flags a season partial iff `year >= today.year`, so a panel rebuilt in
        # January un-flags the season that just ended -- and the reader must follow the
        # panel it was actually built from, not today's date.
        base_season_partial=bool(
            calendar.loc[calendar["season"] == season, "partial_season"].any()
        ),
        generated_at=generated_at,
        # The vintage a reader must be shown. Filenames, not a timestamp: the panel is a
        # build artifact whose span is what identifies it.
        panel_vintage={k: panel_path(k, panel_dir).name for k in ("hitter", "pitcher")},
        # The SECOND vintage (#348). Every base-season figure on this board is part
        # projection, and which projection is a property of the snapshot it came from --
        # the FanGraphs fetch is Cloudflare-403 blocked, so a snapshot can sit for a
        # while and two boards built a week apart can share a panel and not an anchor.
        # None when `base_season_partial` is False: a complete base season has no
        # remainder, so nothing was injected and there is no snapshot to name.
        ros_snapshot=loaded.snapshot_date.isoformat() if loaded.snapshot_date else None,
        floors={slot: round(v, 4) for slot, v in sorted(levels.items())},
        excluded={**excluded, "total": sum(excluded.values())},
    )
    return (
        payload,
        to_chart_payload(extras, careers=careers, generated_at=generated_at),
        len(swept),
    )


def _target_store(*, local: bool):
    """Where this push writes -- decided by the FLAG, on both branches.

    Neither branch consults ``RENDER``. The prod client is explicit already
    (``build_explicit_upstash_kv`` builds one regardless of the env), and routing the
    local branch through ``get_kv()`` made the destination depend on an environment
    variable this process does not own: ``is_remote()`` reads RENDER at call time, so
    ``--local`` in a shell where it was already "true" wrote the board to PRODUCTION
    while reporting the local mirror.

    The script also used to SET RENDER on the prod path and never clear it. That was
    copied from ``scripts/refresh_remote.py``, where it IS load-bearing because that
    script then runs code resolving through ``get_kv()``; here nothing does. What it did
    do is invert the provenance: with RENDER set, ``_code_sha()`` takes the is_remote()
    branch, skips the ``git rev-parse`` fallback, finds no RENDER_GIT_COMMIT on a laptop,
    and stamps ``_sha: "unknown"`` -- so the PROD blob, the one an operator has to date
    when the board disagrees with the CLI, recorded "unknown" while ``--local`` recorded
    the real commit.
    """
    from fantasy_baseball.data.kv_store import (
        build_explicit_sqlite_kv,
        build_explicit_upstash_kv,
    )

    return build_explicit_sqlite_kv() if local else build_explicit_upstash_kv()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-horizon", type=int, default=DEFAULT_MAX_HORIZON)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="sweep and report the payload size without writing to prod",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "write to the local SQLite store instead of prod, so the dashboard can be "
            "checked against a real board before anything reaches Render"
        ),
    )
    parser.add_argument("--panel-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.max_horizon < 1:
        parser.error("--max-horizon must be at least 1")
    panel_dir = args.panel_dir
    if panel_dir is not None and not panel_dir.is_absolute():
        panel_dir = PROJECT_ROOT / panel_dir

    started = time.perf_counter()
    payload, chart, scored = build_payload(args.max_horizon, panel_dir)
    body = json.dumps(payload, separators=(",", ":"))
    chart_body = json.dumps(chart, separators=(",", ":"))
    print(
        f"\n  {scored} players scored to {args.max_horizon} years "
        f"in {time.perf_counter() - started:.1f}s"
    )
    # SEPARATELY. They are two keys with two very different read profiles -- every view
    # pays the board, only the player chart pays the extras -- and one combined figure
    # hides exactly the number #344 was opened about.
    print(f"  board      {len(body) / 1024:.0f} KB  (base {payload['base_season']})")
    print(f"  chart data {len(chart_body) / 1024:.0f} KB  ({len(chart['players'])} players)")

    if args.dry_run:
        print("\n  --dry-run: nothing written to prod")
        return 0

    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.web.season_data import unwrap_cache_envelope, write_cache_to

    target = _target_store(local=args.local)
    # CHART DATA FIRST. The two blobs are paired by `generated_at` and the player view
    # refuses a pair that disagrees, so the ordering decides which way a half-finished
    # push degrades: extras-then-board means a written board always has its extras
    # already stored, and a crash between the two leaves the OLD board beside the new
    # extras -- a mismatch the reader catches. The reverse order would leave the new
    # board beside stale extras, which is the same mismatch but with a window where the
    # page's only correct behaviour is to refuse a chart it could have drawn.
    write_cache_to(target, CacheKey.TRAJECTORY_CHART_DATA, chart)
    write_cache_to(target, CacheKey.TRAJECTORY_BOARD, payload)

    # Read them back. A push that silently wrote nothing leaves the dashboard on a stale
    # board that still renders, which is the failure this project keeps re-learning.
    # BOTH keys, and both stamps printed: the pairing is what the player view checks, so
    # an operator has to be able to see it agree without opening the KV.
    stored = json.loads(target.get(redis_key(CacheKey.TRAJECTORY_BOARD)))
    data = unwrap_cache_envelope(stored)
    stored_chart = json.loads(target.get(redis_key(CacheKey.TRAJECTORY_CHART_DATA)))
    chart_data = unwrap_cache_envelope(stored_chart)
    where = "local SQLite" if args.local else "prod Upstash"
    print(
        f"\n  wrote to {where}: "
        f"{len(data['players'])} players, generated {data['generated_at']}, "
        f"panel {data['panel_vintage']}, ROS {data['ros_snapshot']}"
    )
    print(
        f"  chart data: {len(chart_data['players'])} players, "
        f"generated {chart_data['generated_at']}"
    )
    if chart_data["generated_at"] != data["generated_at"]:
        print(
            "  WARNING: the two stamps disagree, so the player view will refuse to draw "
            "career lines and comps. Re-run this script."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
