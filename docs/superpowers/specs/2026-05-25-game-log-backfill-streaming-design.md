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
killed around the point the logs show it die.

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
JSON was dropped after parsing, but `_fetch_boxscores` returns it all for a second-pass parse.)

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

    with ThreadPoolExecutor(max_workers=15) as pool:
        futures = [pool.submit(_one, g) for g in games]
        for i, future in enumerate(as_completed(futures), 1):
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
            # `box` goes out of scope here; the raw JSON is released before the next future.
            if progress_cb and i % 50 == 0:
                progress_cb(f"Box scores: {i}/{len(games)}...")
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

### Verify-after-deploy

- Confirm the refresh runs **off the gunicorn request thread** (the UI polls
  `/api/refresh-status`, which implies a background worker; gunicorn `--timeout 120` would not
  apply to a background thread). If the refresh is in fact synchronous in the request handler,
  the killer is the 120 s timeout, not memory -- in which case streaming alone will not fix it
  and the fallback (with checkpointing) is required. Confirm by watching whether the backfill now
  runs past ~2 minutes to completion.
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
  still parsed. Assert the return holds only compact rows (no raw box-score dicts) -- the
  structural guarantee that raw JSON is not retained.
- Memory is not unit-testable here; rely on the structural change plus the post-deploy Render
  memory metric.

## Rollout

Branch `fix/game-log-backfill-oom` -> PR -> deploy. First post-deploy refresh should complete the
now-memory-safe backfill, set the watermark, and populate `game_logs:*`. Verify via the Upstash
SCAN and the Render memory metric above. If it still dies, capture whether it ran past ~2 minutes
(time-limit signal) or spiked memory (size signal) and escalate to the batching fallback
accordingly.

## Risks

- **Wrong root cause (duration, not memory):** if the backfill is synchronous in the request
  thread, gunicorn's 120 s timeout kills it regardless of memory. Streaming would not fix that;
  the batching + cursor fallback (durable per-batch progress) would. The verify-after-deploy step
  distinguishes the two.
- **Completed-future buffering:** if the network is much faster than parsing, completed futures
  could buffer more than ~pool-size raw box scores. Mitigated by the wave-submission middle
  ground if observed.
