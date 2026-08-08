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
carries its own vintage and the page prints it. Re-run it when the panel is rebuilt
(`scripts/build_pt_panel.py`) or when the season has moved enough that the paced
current-season figure every fit is built on has meaningfully changed.

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
#: rather than re-declared here. See its docstring in `comp_paths`; a second literal
#: beside this one is what the deleted parity test existed to police.
from fantasy_baseball.trajectory.comp_paths import MAX_COMPS, closest_paths


def _arc(seasons) -> list[list[float]]:
    """One player's stored career as ``[[age, sgp], ...]``, ascending, ``[]`` when absent.

    ONE spelling for the two arcs a chart pairs on a single axis -- the subject's own
    history and a comp's whole career. They came off the same `by_id` frame through two
    hand-copied comprehensions, one of which sorted and one of which did not, so the
    blob's two `[[age, sgp]]` lists carried different ordering guarantees for no reason
    a reader could find.

    **A MISSED SEASON IS A ZERO, NOT A GAP.** `load_scored_panel` drops a season the
    player did not play (`observed` False, `pa <= 0`) and `_in_role` drops one he played
    in the other role, so a career with a lost year arrives here as a frame with a hole
    in it. `shape.prepare` takes the opposite convention on the very same data -- it
    reindexes and `np.nan_to_num(..., nan=0.0)`, commented "a missing key means he was
    out of the league that year -- a real 0" -- and that zero is part of the RMSE that
    made him a comp and is what the comps table prints. An arc that skipped the age drew
    a straight line through a year the rest of the page shows as nothing, on a card whose
    whole job is showing where the busts sit in the arc.

    Filled only BETWEEN observed seasons. Before his debut and after his last year he was
    not a zero, he was absent, and the career span is what the arc is about.

    Sorted (implicitly, by iterating the span) even though `groupby` happens to emit
    ascending: nothing in the panel contract promises it, and `_netted` on the read side
    sorts for the same reason.
    """
    if seasons is None:
        return []
    # Ages within one player are distinct: `collapse_split_seasons` gives one row per
    # (mlbam_id, season) and age advances with the season, so nothing is collapsed here.
    scored = {
        int(a): round(float(s), 4) for a, s in zip(seasons["age"], seasons["sgp"], strict=True)
    }
    if not scored:
        return []
    return [[age, scored.get(age, 0.0)] for age in range(min(scored), max(scored) + 1)]


def player_comps(prepared, player, horizons: tuple[int, ...], names: dict) -> list[dict] | None:
    """One player's stored comp block, or ``None`` when his path is too short to match.

    ``player.sgp`` is ``traj.observable`` -- the points with ``n > 0`` -- and the
    candidate mask ``seasons + h <= last`` shrinks as the horizon grows, so a player can
    be observable at h=1..3 and not at h=4..5. ``sweep_pool`` keeps him: it drops a
    player only when the path is ENTIRELY empty. ``closest_paths`` then raises on
    ``len(predicted) != len(prepared.horizons)``, and nothing caught it -- one such
    player discarded the whole ~52s sweep and pushed nothing. Latent on the default
    horizon, directly reachable via ``--max-horizon``.

    SKIPPED, NEVER PADDED. ``forward`` stores a real 0.0 for "out of the league", so
    padding the short tail with zeros would match him against the cohort that stopped
    playing -- a confident wrong answer in place of an honest gap. The page already
    renders an empty comps list with an explanation, so an absent block degrades.

    Names are attached HERE, not in ``closest_paths``: naming needs the people cache, and
    keeping it out of that module is what lets it be tested with no data files. An
    unknown id renders as its id rather than vanishing -- a comp is still a comp.
    """
    if len(player.sgp) != len(horizons):
        return None
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
            "path": [round(v, 3) for v in c.path],
        }
        for c in closest_paths(
            prepared, [point.mean for point in player.sgp], age=player.age, n=MAX_COMPS
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
    from fantasy_baseball.trajectory.comps import collapse_split_seasons
    from fantasy_baseball.trajectory.era import era_normalize
    from fantasy_baseball.trajectory.panel import (
        load_scored_panel,
        panel_path,
        season_elapsed_fraction,
    )
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

    def load(kind: str) -> pd.DataFrame:
        return era_normalize(
            load_scored_panel(
                kind, panel_dir=panel_dir, sgp_overrides=overrides, include_partial=True
            ),
            kind,
            sgp_overrides=overrides,
        )

    # Dating the in-progress season is a league fact and must come off the HITTER panel
    # even when pricing pitchers -- pitcher `games` counts appearances, not team games.
    calendar = load("hitter")
    season = int(calendar["season"].max())
    cache = PROJECT_ROOT / "data" / "cache" / "keeper_skills"
    names = player_names(cache)
    eligibility = season_slots(cache, season)
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
    excluded = {"low_sgp": 0, "no_current_line": 0}
    for kind in ("hitter", "pitcher"):
        live = calendar if kind == "hitter" else load(kind)
        candidates = list(
            board_inputs(
                live,
                kind=kind,
                names=names,
                replacement_levels=levels,
                eligibility=eligibility,
                calendar=calendar,
                season=season,
            )
        )
        rows = [r for r in candidates if r.sgp >= MIN_SGP]
        # Who this pool leaves out, so the page can say so. Two separate exclusions:
        # the min-SGP gate, and -- larger and entirely silent -- players who had a line
        # LAST season and none this one, who are never candidates at all.
        excluded["low_sgp"] += len(candidates) - len(rows)
        excluded["no_current_line"] += len(
            set(live.loc[live["season"] == season - 1, "mlbam_id"])
            - set(live.loc[live["season"] == season, "mlbam_id"])
        )
        print(f"  {kind}: {len(rows)} players with a {season} line", flush=True)
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
        prepared = prepare(complete, kind=kind, horizons=horizons)
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

        short_paths = 0
        for player in produced:
            history = _arc(by_id.get(player.mlbam_id))
            # None means his observable path is shorter than the swept horizons, so no
            # honest match exists -- see `player_comps`. He keeps his row and his career
            # line; only the comps go.
            comps = player_comps(prepared, player, horizons, names)
            if comps is None:
                short_paths += 1
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
            extras[(player.mlbam_id, player.pool)] = {
                "history": history,
                "comps": comps if comps is not None else [],
            }
        print(f"    swept in {time.perf_counter() - started:.1f}s", flush=True)
        if short_paths:
            # SAID OUT LOUD. A push that quietly drops comps for part of the pool
            # renders exactly like one that did not, and the page's "none stored" note
            # reads as "this player has no comps" rather than "this run skipped them".
            print(
                f"    {short_paths} observable at fewer than {len(horizons)} horizons; "
                f"comps skipped for those (their rows and career lines are intact)",
                flush=True,
            )

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
        floors={slot: round(v, 4) for slot, v in sorted(levels.items())},
        excluded={**excluded, "total": excluded["low_sgp"] + excluded["no_current_line"]},
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
        f"panel {data['panel_vintage']}"
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
