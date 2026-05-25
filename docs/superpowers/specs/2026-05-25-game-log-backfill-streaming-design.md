# Game-Log Backfill Streaming Fix -- Design

- Date: 2026-05-25
- Status: Approved (pending spec review)
- Surface: `src/fantasy_baseball/data/mlb_game_logs.py` (sync engine; box-score fetch/parse)
- Follow-up to: `docs/superpowers/specs/2026-05-24-per-player-game-logs-design.md` (the feature this fixes)

## Problem

The first post-deploy refresh of the per-player game-log pipeline (live on Render since
2026-05-24 20:39 UTC, deploy `dep-d89m2sp9rddc73apb0u0`) runs the one-time **backfill** --
~805 regular-season games for 2026 so far. It does not complete: the worker is killed
mid-fetch and the backfill writes nothing.

The cause is that `_fetch_boxscores` holds **every** fetched box score in memory at once:

```python
# mlb_game_logs.py (current)
def _fetch_boxscores(games, progress_cb):
    results: dict[int, dict[str, Any]] = {}
    ...
    results[gp] = box          # accumulates ALL raw box-score JSONs
    ...
    return results, failed     # _sync parses them in a SECOND pass afterward
```

A full MLB box score is hundreds of KB of JSON (~1-2 MB as a Python dict). Holding ~805 of
them simultaneously is ~1 GB, which blows the **512 MB free Render instance**. The process is
killed around the point the logs show it die. (The ~1-2 MB/box figure is an estimate, not
measured against a live box score; the death point tracking ~650/805 rather than a wall-clock
cutoff corroborates a memory ceiling.)

There are actually **two** references pinning the box scores, and the second is subtle:

1. `results[gp] = box` -- the explicit accumulator.
2. The `futures` list. `_fetch_boxscores` does `futures = [pool.submit(_one, g) for g in games]`
   and iterates `as_completed(futures)`. A `concurrent.futures.Future` keeps its result in
   `Future._result` **even after `.result()` is called** -- `result()` does not clear it. As long
   as the `futures` list is alive, every `Future` (and through it every box-score dict) stays
   referenced. Since `results[gp]` and `future._result` point at the *same* dict object, peak is
   ~1x all box scores (not 2x), but **both** references must be gone for a box to be freed.

This second point is load-bearing for the fix (see Design): dropping only the `results`
accumulator while still binding `futures` to a surviving list frees nothing.

Because `_sync` writes the per-player keys, the derived rollup, and the watermark only **after**
the entire fetch loop finishes (`_upsert_and_roll` runs post-fetch), an interruption during the
fetch persists nothing. And since the watermark is never set, **every** subsequent refresh takes
the backfill branch again, re-fetches all ~805 games, and dies again -- a stuck loop the pipeline
cannot escape on its own.

## Evidence (Render + Upstash, 2026-05-25)

- App logs ~01:49-01:50 UTC: `fantasy_baseball.web.refresh_pipeline Box scores: 500/805 ...
  650/805` (a message emitted only by the new `_fetch_boxscores`), immediately followed by a
  worker restart (`Running 'gunicorn wsgi:app ...'` at 01:50:46, `Starting gunicorn` at 01:51:11).
  The restart lands mid-fetch at ~650/805, before the write phase.
- Upstash: `SCAN game_logs:*` is empty; `game_logs:2026:fetched_through_utc` is null; `DBSIZE`
  185. The old-shape `game_log_totals:*` rollups and `season_progress` remain (last good
  old-code run), so the dashboard still serves stale-but-valid data.

The death point (~650/805) tracks a memory ceiling far more than a clean wall-clock cutoff, and
the memory math above is conclusive: peak memory is the **raw box scores**, not the compact
parsed rows. (Note: the earlier /simplify efficiency review missed this -- it assumed the raw
JSON was dropped after parsing, but `_fetch_boxscores` returns it all for a second-pass parse.
A naive streaming refactor misses the futures-list pin in the same spirit: it looks like the box
is dropped at the end of each iteration, but the `futures` list keeps it alive.)

## Goals

- Make the backfill (and incremental sync) memory-safe on the 512 MB free instance so the
  backfill completes in one run, sets the watermark, and populates the `game_logs:*` keys.
- Preserve all verified behavior exactly: Ohtani two-way in both rollups, the position-player
  pitching filter, gamePk-keyed correction self-heal, watermark advanced only on a clean pass,
  dates / season_progress.
- Keep the change small and localized to the fetch/parse seam.

## Non-goals

- No change to the incremental path's semantics, the watermark model, or the redis_store schema.
- No change to box-score parsing logic (`mlb_boxscore.py`) or the rollup derivation
  (`_upsert_and_roll`).
- Batched/checkpointed backfill is **out of scope for this fix** -- it is the documented fallback
  below, pursued only if streaming proves insufficient.

## Design: stream the parse (primary fix)

Parse each box score into its compact per-player rows **as it completes** in the threadpool, and
drop the raw box-score JSON immediately. Peak memory becomes the accumulated compact rows (a few
MB for a full season) plus at most ~pool-size raw box scores in flight, instead of all ~805 at
once.

This only works if the box score is *actually* released after parsing. Two structural rules make
that true, and **both are required** (see "Why the futures list must not survive the loop" below):

1. Do **not** bind the futures to a named list that outlives the loop. Pass the submission
   comprehension straight into `as_completed(...)`; capture `len(games)` separately for the
   progress denominator since the list is no longer around to measure.
2. `del future, box` at the end of each iteration so neither the parsed box nor its `Future`
   lingers in a loop variable while the next one is awaited.

Replace `_fetch_boxscores` and the second-pass parse loop in `_sync` with a single collector:

```python
def _collect_player_rows(
    games: list[dict[str, Any]], progress_cb
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], set[str], set[int]]:
    """Fetch box scores in parallel and parse each into compact per-game rows as it
    completes, never retaining the raw box-score JSON. Returns
    (hitting, pitching, dates, failed_gamePks)."""
    ctx = {g["gamePk"]: _game_context(g) for g in games}
    hitting: dict[str, dict[str, Any]] = {}
    pitching: dict[str, dict[str, Any]] = {}
    dates: set[str] = set()
    failed: set[int] = set()

    def _one(game: dict[str, Any]) -> tuple[int, dict[str, Any] | None]:
        gp = game["gamePk"]
        try:
            return gp, _fetch_boxscore(gp)
        except Exception:
            return gp, None

    total = len(games)
    with ThreadPoolExecutor(max_workers=15) as pool:
        # Do NOT bind the submissions to a named list: a Future retains its result
        # (the raw box score) until the Future itself is released, and a surviving
        # `futures` list would pin all ~total box scores for the whole loop. Passing
        # the comprehension straight in lets as_completed drop each Future as it is
        # yielded, so only the in-flight + just-yielded box scores stay live.
        for i, future in enumerate(
            as_completed([pool.submit(_one, g) for g in games]), 1
        ):
            gp, box = future.result()
            if box is None:
                failed.add(gp)
            else:
                _pk, gnum, date = ctx[gp]
                dates.add(date)
                for mlbam_id, name, batting, pitch in iter_boxscore_players(box):
                    if batting:
                        h = hitting.setdefault(mlbam_id, {"name": name, "rows": {}})
                        h["name"] = name or h["name"]
                        h["rows"][gp] = boxscore_hitter_row(batting, gp, gnum, date)
                    if pitch:
                        p = pitching.setdefault(mlbam_id, {"name": name, "rows": {}})
                        p["name"] = name or p["name"]
                        p["rows"][gp] = boxscore_pitcher_row(pitch, gp, gnum, date)
            del future, box  # release the Future and its box score before the next one
            if progress_cb and i % 50 == 0:
                progress_cb(f"Box scores: {i}/{total}...")
    return hitting, pitching, dates, failed
```

`_sync` shrinks to consume the collector, with everything downstream unchanged:

```python
def _sync(client, season, games, now_utc, progress_cb) -> None:
    hitting, pitching, dates, failed = _collect_player_rows(games, progress_cb)
    all_ok = not failed

    positions = _resolve_positions(client, season, list(pitching.keys()))
    kept_pitching = {}
    for mlbam_id, payload in pitching.items():
        if mlbam_id not in positions:
            all_ok = False
            continue
        if should_record_pitching(positions[mlbam_id]):
            kept_pitching[mlbam_id] = payload

    _upsert_and_roll(client, season, "hitting", hitting)
    _upsert_and_roll(client, season, "pitching", kept_pitching)

    known_dates = set(get_game_log_dates(client, season))
    if dates:
        known_dates |= dates
        set_game_log_dates(client, season, list(known_dates))
    set_season_progress(
        client, games_elapsed=len(known_dates), total=162, as_of=local_today().isoformat()
    )

    if all_ok:
        set_game_logs_watermark(client, season, now_utc.isoformat())
    if progress_cb:
        progress_cb(f"Game logs synced: {len(games)} games (clean={all_ok})")
```

### Why this is enough

- Position resolution, the pitching filter, the upsert, the rollup derivation, dates, and the
  watermark are byte-for-byte the same -- only *where* parsing happens moves (into the
  `as_completed` loop), so behavior is identical and the existing tests still pin it.
- Peak memory drops from ~all-805 raw box scores (~1 GB) to: the compact row accumulators
  (~a few MB for a full season -- ~805 games x ~30 players x ~150 bytes) plus the raw box scores
  for futures that have completed but not yet been parsed. Parsing is microseconds versus a
  ~0.3 s fetch, so completed results are consumed promptly and only ~pool-size (15) raw box
  scores are live at once (~15-30 MB transient).

### Why the futures list must not survive the loop

The memory win above comes **entirely** from not retaining the futures, not from moving the parse.
Moving the parse into the loop is what *lets* each box be dropped after it is consumed; not
keeping the `futures` list (rule 1) is what *actually* drops it. Verified empirically with
weakrefs on CPython 3.11:

- **Named `futures` list kept** (the shape of both the current code and a naive streaming
  refactor): after consuming 15 of 20 results, **all 15 are still alive** -- pinned via
  `Future._result`. Setting the local `box = None` changes nothing.
- **Comprehension passed straight to `as_completed`** (rule 1): after consuming 15 of 20, **0-1
  are alive** -- `as_completed` drops each Future as it yields it, so only the current loop
  variable can linger (and `del future, box` removes even that).

A naive refactor that drops the `results` dict but still writes `futures = [pool.submit(...)]`
would therefore have **the same ~1x-all-box-scores peak as today** and very likely OOM again at
~650/805 -- which would falsely look like "streaming is insufficient, escalate to the batching
fallback" when the real miss is the one-line futures-list pin.

### Verify-after-deploy

- **Resolved (not just suspected): the refresh runs off the gunicorn request thread.**
  `season_routes.py` spawns `threading.Thread(target=run_full_refresh, daemon=True)` and the route
  returns immediately while the UI polls `/api/refresh-status`. `render.yaml` runs
  `gunicorn wsgi:app --timeout 120` with **no `--workers`** flag (a single sync worker), so the
  120 s `--timeout` governs the worker's accept-loop heartbeat -- which the background thread does
  not block -- not the refresh. The worker restart in the evidence is the OOM killer taking the
  whole process (and the daemon thread with it), not a request timeout. So the killer is memory,
  the duration risk is low, and the memory budget is a single process. Still worth watching the
  backfill run past ~2 minutes to completion as a sanity check.
- After deploy + one refresh: `SCAN game_logs:2026:*` is non-empty, `game_logs:2026:fetched_through_utc`
  is set, Ohtani (`game_logs:2026:660271:*`) is present if he has played, and the Render memory
  metric stays under the 512 MB cap during the backfill.

## Fallback: batched + checkpointed backfill (only if streaming is insufficient)

If, after the streaming fix, the backfill still gets killed -- because the compact accumulators
for a full season are still too large, OR because the killer is a hard time limit (gunicorn
timeout / free-tier cap) rather than memory -- escalate to batching:

- Process the backfill `games` in fixed-size batches (e.g., 100). For each batch:
  collect rows (streaming) -> resolve positions -> `_upsert_and_roll` that batch -> record
  progress. Peak memory becomes one batch, and each batch's writes **persist** before the next.
- Persist a backfill cursor (e.g., `game_logs:{season}:backfill_cursor` -- a count or the set of
  completed gamePks) so a re-run **skips completed games** and makes durable forward progress
  across crashes, eventually finishing and setting the watermark. This is what breaks the stuck
  loop under a hard time limit (each run advances the cursor instead of redoing the same prefix).
- Incremental cycles (~dozens of changed games) need neither batching nor a cursor.

A lightweight middle ground, if only a mild memory spike remains: submit futures in waves of
~50 instead of all at once, bounding completed-but-unparsed buffering. This is a one-line change
to the collector and avoids the full cursor machinery.

Batching is more invasive (it threads partial state through the rollup-derivation path and adds a
cursor key), so it is deliberately deferred unless the streaming fix is measured to be
insufficient.

## Testing

- The existing `tests/test_data/test_mlb_game_logs_sync.py` suite must stay green unchanged -- it
  pins the behaviors this refactor must preserve (backfill records two-way + filters mop-up,
  incremental correction overwrites by gamePk, watermark withheld on position-unresolved and on
  box-score-fetch-failure). They patch `_fetch_boxscore` (singular) and call `sync_game_logs`, so
  the internal `_fetch_boxscores` -> `_collect_player_rows` refactor does not touch their seams.
- Add a unit test for `_collect_player_rows` directly (patching `_fetch_boxscore`): given two
  games and canned box scores, it returns the expected hitting/pitching rows and dates; when
  `_fetch_boxscore` raises for one gamePk, that gamePk lands in `failed` and the other game is
  still parsed.
- Note the limit of that test: a "return holds only compact rows" assertion is **always** true
  (the function returns compact rows regardless of how many box scores it pinned mid-loop), so it
  does **not** catch the futures-retention regression. The thing that matters -- transient peak
  memory during the loop -- is not reliably unit-testable: with a small fixture every box fits "in
  flight," so retention vs. release is indistinguishable. The futures-list structure (rule 1) is
  therefore guarded by code review and the explanatory comment in the collector, not by a test.
- Memory is not unit-testable here; rely on the structural change plus the post-deploy Render
  memory metric.

## Rollout

Branch `fix/game-log-backfill-oom` -> PR -> deploy. First post-deploy refresh should complete the
now-memory-safe backfill, set the watermark, and populate `game_logs:*`. Verify via the Upstash
SCAN and the Render memory metric above. If it still dies, capture whether it ran past ~2 minutes
(time-limit signal) or spiked memory (size signal) and escalate to the batching fallback
accordingly.

## Risks

- **Retaining the futures defeats the fix:** the single most likely way to ship this and still
  OOM is to write `futures = [pool.submit(...)]` (the natural shape) and pin every box score
  through `Future._result`. Guarded by rule 1, the `del future, box`, and the collector comment;
  there is no test that would catch a regression here, so it must hold in review.
- **Wrong root cause (duration, not memory):** ruled out -- the refresh runs on a daemon thread,
  not the request handler, so gunicorn's 120 s timeout does not apply (see Verify-after-deploy).
  If a future change moved the refresh back into the request path, the batching + cursor fallback
  (durable per-batch progress) would be required instead.
- **Completed-future buffering:** if the network is much faster than parsing, completed futures
  could buffer more than ~pool-size raw box scores. Mitigated by the wave-submission middle
  ground if observed.
