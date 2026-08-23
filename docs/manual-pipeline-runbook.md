# Manual (Yahoo-free) pipeline runbook

Yahoo's API app is locked out, so league standings and the ten rosters are typed
by hand into `data/manual/*.yaml` and everything downstream runs the normal
pipeline against an **isolated KV store**, `data/manual.db`. Nothing in manual
mode writes the Yahoo baseline `data/local.db` (it is read once, read-only, to
create the copy) and nothing writes production Upstash.

Commands first. Reasoning is at the bottom.

---

## 1. Run it

From the repository root, in a **fresh terminal** (see [Do not do
this](#3-do-not-do-this) for why a reused one is dangerous):

```bash
# once, ever -- create the isolated store as a consistent copy of the baseline
python scripts/bootstrap_manual_kv.py

# every time -- validate the transcriptions without opening any KV store
python scripts/run_manual_refresh.py --dry-run

# every time -- the real run: seeds the manual store, runs the pipeline,
# prints the roster audit and writes it to data/manual/audit-<date>.txt
python scripts/run_manual_refresh.py
```

That is the whole flow. `run_manual_refresh.py` sets
`FANTASY_LOCAL_KV_PATH=data/manual.db` and `FB_SKIP_YAHOO=1` **for itself**,
before its first `fantasy_baseball` import -- you do not export anything, and
you should not.

It prints the resolved absolute KV path in its header block, every time, before
any guard runs. Read it:

```
========================================================================
MANUAL PIPELINE -- hand-transcribed inputs, Yahoo DISABLED
  KV store : C:\Users\HartAlden\FantasyBaseball\data\manual.db
  mode     : LIVE (writes the store above)
========================================================================
```

If that path does not end in `data\manual.db`, stop.

Useful flags:

```bash
python scripts/run_manual_refresh.py --skip-game-logs   # reuse the store's MLB game logs
python scripts/run_manual_refresh.py --skip-blend       # reuse the cached ROS blend (fast re-run)
python scripts/run_manual_refresh.py --report-out out.txt
```

Exit codes: `0` ok, `1` started then failed, `2` refused before touching
anything.

## 2. Update the transcriptions

Three files, all under `data/manual/`:

| File | What it holds |
|---|---|
| `standings.yaml` | The Yahoo standings page: one row per team, category totals |
| `rosters.yaml` | All ten rosters, `snapshot_date`, one entry per player |
| `fa_exclusions.yaml` | Names to drop from the synthesized free-agent pool |

Edit, then `--dry-run` to validate, then run for real. Yahoo team keys are
**looked up** from the store's existing `cache:standings`, never typed into the
YAML.

## 3. Do not do this

**(a) Never run a sync with `FANTASY_LOCAL_KV_PATH` exported.**

`kv_sync.sync_remote_to_local()` wipes its destination -- literally
`DELETE FROM kv; DELETE FROM hash_kv;` -- and then refills it from Upstash. It
picks that destination with `get_kv()`, which honours
`FANTASY_LOCAL_KV_PATH`. Both of these call it with no explicit destination:

```bash
python scripts/run_season_dashboard.py   # syncs on startup unless --no-sync
python scripts/refresh_remote.py         # syncs Upstash -> local at the end
```

Run either from a shell where `FANTASY_LOCAL_KV_PATH=data/manual.db` is still
exported and the hand-transcribed store is destroyed and replaced with the last
Yahoo snapshot, with no error and no prompt.

Both now refuse when the resolved KV path is not `data/local.db`: they print the
path, exit `2`, and delete nothing. `refresh_remote.py` checks at the very top of
its run, so a refused launch has not written production Upstash either. The
reliable habit is still a fresh terminal per mode -- the guard is a backstop, not
a licence.

**(b) Never run `scripts/ingest_ros_export.py` without `--no-push`.**

It sets `RENDER=true` in-process (`scripts/ingest_ros_export.py`, in
`_push_to_prod`) and writes **production Upstash**, then triggers a full prod
refresh. That is correct for its normal job and completely wrong while the
league's live state is hand-typed. If you need fresh ROS projections during
manual mode, stage only:

```bash
python scripts/ingest_ros_export.py --no-push
```

**(c) Never point `--kv-path` at `local.db`.**

```bash
python scripts/run_manual_refresh.py --kv-path data/local.db   # refuses, exit 2
```

The script refuses any target named `local.db` or equal to the repo's
`data/local.db`, and `manual.seed.assert_isolated_store` refuses again at the
write. Both are backstops, not permission to try it: `data/local.db` is the only
copy of the pre-outage Yahoo history.

## 4. Open the dashboard against the manual store

`--no-sync` is mandatory here; without it the launcher refuses (that is (a)
above).

PowerShell:

```powershell
$env:FANTASY_LOCAL_KV_PATH = "C:\Users\HartAlden\FantasyBaseball\data\manual.db"
$env:FB_SKIP_YAHOO = "1"
python scripts/run_season_dashboard.py --no-sync
```

bash / Git Bash:

```bash
FANTASY_LOCAL_KV_PATH="$PWD/data/manual.db" FB_SKIP_YAHOO=1 \
  python scripts/run_season_dashboard.py --no-sync
```

Use an **absolute** path. `kv_store` resolves this variable against the current
working directory, not the repo root, so `data/manual.db` from the wrong
directory silently creates a second empty store. The launcher's first line of
output is the resolved absolute path -- confirm it before using the numbers:

```
KV store: C:\Users\HartAlden\FantasyBaseball\data\manual.db
```

**Do not press the dashboard's Refresh button in this mode.** `POST /api/refresh`
runs the full pipeline against whatever store the process is bound to. Re-run
`scripts/run_manual_refresh.py` instead. `FB_SKIP_YAHOO=1` above is a seatbelt
for a stray click, not a licence to click.

Close that terminal when you are done with it.

## 5. Revert, when Yahoo access comes back

```bash
# 1. New terminal. Confirm neither variable is set.
#    PowerShell:  $env:FANTASY_LOCAL_KV_PATH ; $env:FB_SKIP_YAHOO
#    bash:        echo "[$FANTASY_LOCAL_KV_PATH] [$FB_SKIP_YAHOO]"

# 2. Prove Yahoo works, locally and non-destructively for prod: this is the
#    full refresh against data/local.db only. The first Yahoo call re-prompts
#    on the console if the OAuth token in config/oauth.json is stale.
python scripts/run_lineup.py

# 3. Then the normal refresh: writes prod Upstash, then syncs back down to
#    data/local.db.
python scripts/refresh_remote.py

# 4. Normal dashboard: syncs on startup, prints "KV store: ...\data\local.db".
python scripts/run_season_dashboard.py
```

Then retire the manual store:

```powershell
# also removes the -wal / -shm sidecars
Remove-Item data/manual.db*
```

Nothing else needs undoing. The manual pipeline only ever wrote `data/manual.db`;
`data/local.db` and Upstash were never opened for writing, so the baseline needs
no repair and prod needs no rollback. Keep `data/manual/*.yaml` and the
`data/manual/audit-*.txt` reports -- they are the record of what the league
looked like during the outage. `data/manual.db` is derived and can be rebuilt at
any time with `scripts/bootstrap_manual_kv.py`.

---

## Why it is built this way

**Isolation is by whole store, not by key prefix.** A manual run writes
hand-typed data into `cache:standings`, `weekly_rosters_history`,
`standings_history` and the whole `cache:*` family -- the same keys the Yahoo
pipeline owns. There is no prefix that separates them, so the separation is the
file: every manual process exports `FANTASY_LOCAL_KV_PATH` before its first
`fantasy_baseball` import, `get_kv()` builds `SqliteKVStore(data/manual.db)`, and
every read and write in that process lands there.

**The env var is process-wide and sticky.** `get_kv()` is a singleton that
captures the path on its first call, and a shell export outlives the command you
set it for. That is the entire hazard surface: the variable is invisible, the
sync is destructive, and the failure is silent. Hence the two rules that actually
matter -- one terminal per mode, and read the `KV store:` line.

**`data/manual.db` is a copy of the baseline, not an empty store.** The blended
projections, the frozen `cache:positions` eligibility map and the existing
`game_logs:*` watermarks all have to be there for the pipeline to run at all;
`bootstrap_manual_kv.py` takes a transactionally consistent SQLite backup (WAL
sidecars included) with the source opened read-only.

**Related:** `docs/stale-data-refresh-runbook.md` covers `FB_SKIP_YAHOO=1` on its
own -- stale-data mode, which reuses the last persisted league state instead of
hand-transcribed input. Manual mode is that plus transcription plus store
isolation.
