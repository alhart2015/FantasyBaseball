"""Sweep the whole player pool and push the trajectory board to prod Upstash (#321).

This runs OFFLINE, on a machine that has the panel. It is not part of the refresh
pipeline and cannot be: the fit reads `data/trajectory/*.csv` and
`data/cache/keeper_skills`, both gitignored, so neither exists on Render. The season
dashboard is a pure reader of what this writes.

    python scripts/push_trajectory_board.py                 # sweep + push to prod
    python scripts/push_trajectory_board.py --dry-run       # sweep, report size, no write
    python scripts/push_trajectory_board.py --max-horizon 3 # shorter dropdown, faster

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
from fantasy_baseball.trajectory.comp_paths import MAX_COMPS


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


def build_payload(max_horizon: int, panel_dir: Path) -> tuple[dict, int]:
    """Sweep both pools, once each. Returns the payload and the rows scored.

    ONE fit per player, on raw SGP; VAR is derived on read (#331). It said "on both
    scales" because it used to store both, and a reader who believes that is a reader
    who adds back a `scales` argument or a second `shape_trajectory` call that the
    payload no longer has anywhere to put.
    """
    from fantasy_baseball.config import load_config
    from fantasy_baseball.sgp.denominators import get_sgp_denominators
    from fantasy_baseball.sgp.replacement import position_aware_replacement_levels
    from fantasy_baseball.trajectory.board import board_inputs, player_names, season_slots
    from fantasy_baseball.trajectory.comp_paths import closest_paths
    from fantasy_baseball.trajectory.comps import collapse_split_seasons
    from fantasy_baseball.trajectory.era import era_normalize
    from fantasy_baseball.trajectory.panel import (
        load_scored_panel,
        panel_path,
        season_elapsed_fraction,
    )
    from fantasy_baseball.trajectory.shape import prepare
    from fantasy_baseball.trajectory.sweep import sweep_pool, to_payload
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
    extras: dict[int, dict] = {}
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

        for player in produced:
            seasons = by_id.get(player.mlbam_id)
            history = (
                [
                    [int(a), round(float(s), 4)]
                    for a, s in zip(seasons["age"], seasons["sgp"], strict=True)
                ]
                if seasons is not None
                else []
            )
            comps = closest_paths(
                prepared,
                [point.mean for point in player.sgp],
                age=player.age,
                n=MAX_COMPS,
            )
            extras[player.mlbam_id] = {
                "history": history,
                "comps": [
                    {
                        # Named HERE, not in `closest_paths`: naming needs the people
                        # cache, and keeping it out of that module is what lets it be
                        # tested with no data files. An unknown id renders as its id
                        # rather than vanishing -- a comp is still a comp.
                        "name": names.get(c.mlbam_id, str(c.mlbam_id)),
                        "season": c.season,
                        "rmse": round(c.rmse, 3),
                        "path": [round(v, 3) for v in c.path],
                    }
                    for c in comps
                ],
            }
        print(f"    swept in {time.perf_counter() - started:.1f}s", flush=True)

    payload = to_payload(
        swept,
        extras=extras,
        base_season=season,
        max_horizon=max_horizon,
        min_sgp=MIN_SGP,
        season_elapsed=round(elapsed, 4),
        generated_at=local_now().isoformat(timespec="seconds"),
        # The vintage a reader must be shown. Filenames, not a timestamp: the panel is a
        # build artifact whose span is what identifies it.
        panel_vintage={k: panel_path(k, panel_dir).name for k in ("hitter", "pitcher")},
        floors={slot: round(v, 4) for slot, v in sorted(levels.items())},
        excluded={**excluded, "total": excluded["low_sgp"] + excluded["no_current_line"]},
    )
    return payload, len(swept)


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
    payload, scored = build_payload(args.max_horizon, panel_dir)
    body = json.dumps(payload, separators=(",", ":"))
    print(
        f"\n  {scored} players scored to {args.max_horizon} years "
        f"in {time.perf_counter() - started:.1f}s"
    )
    print(f"  payload {len(body) / 1024:.0f} KB  (base {payload['base_season']})")

    if args.dry_run:
        print("\n  --dry-run: nothing written to prod")
        return 0

    from fantasy_baseball.data.cache_keys import CacheKey, redis_key
    from fantasy_baseball.web.season_data import unwrap_cache_envelope, write_cache_to

    target = _target_store(local=args.local)
    write_cache_to(target, CacheKey.TRAJECTORY_BOARD, payload)

    # Read it back. A push that silently wrote nothing leaves the dashboard on a stale
    # board that still renders, which is the failure this project keeps re-learning.
    stored = json.loads(target.get(redis_key(CacheKey.TRAJECTORY_BOARD)))
    data = unwrap_cache_envelope(stored)
    print(
        f"\n  wrote to {'local SQLite' if args.local else 'prod Upstash'}: "
        f"{len(data['players'])} players, generated {data['generated_at']}, "
        f"panel {data['panel_vintage']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
