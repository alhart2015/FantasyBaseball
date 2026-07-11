# Daily Summary Email -- Design

**Issue:** #200 (backlog, in-season)
**Date:** 2026-07-11
**Status:** Approved design, pre-plan

## Purpose

Send an automated morning email summarizing the overnight state of the fantasy
team, so the manager starts each day with results, trends, standings movement,
recommended lineup moves, injury news, and upcoming matchups in one place --
without opening the dashboard.

Audience: the manager (single recipient) today, with a plausible extension to
leaguemates later. This drives two decisions: use a real transactional email
service (not personal SMTP), and make the recipient a configurable list.

## Scope

All six sections from issue #200 are in v1. Each maps to a real data source in
the stack, but the section descriptions below are corrected to match what the
data *actually* is -- an earlier draft mischaracterized three of them.

| Section | Source | Reality check |
|---|---|---|
| Last night's results (box-score lines for rostered players) | per-player MLB game logs (KV) + name->MLBAM crosswalk | Requires per-player resolution + per-date filter; see Data access. |
| Hot/cold streaks | `STREAK_SCORES` (`Report`) | **Single 14-day window, hitters-only, percentile-threshold + continuation-probability** -- NOT "7d & 14d vs projection." No pitcher streaks exist. |
| Standings changes (overnight movement) | `STANDINGS` + new snapshot | Rank + total roto points diff directly; per-category place-point movement is a **recomputation** from stored raw totals (see Standings snapshot). |
| Recommended lineup changes | `LINEUP_OPTIMAL.moves` | Week/ROS roto active-vs-bench recommendations off the next-lock roster -- **not** per-day/per-game start-sit. |
| Injury news affecting roster | live Yahoo `fetch_injuries()` -- `injury_note` carries news text | Requires a live Yahoo `league` handle + resolved `team_key` (see Data access). |
| Upcoming probable pitcher matchups | `PROBABLE_STARTERS` (KV, roster-mapped) | `required=False` in the refresh -- may be absent; treat absence as an empty section. |

Key finding that holds up: **injury news needs no external news feed.** Yahoo's
per-player `injury_note` / `status_full` fields (extracted by
`lineup/yahoo_roster.py::fetch_injuries` / `parse_injuries_raw`,
`yahoo_roster.py:141-215`) carry the free-text injury update.

## Delivery

**Resend** transactional email API. Rationale over alternatives:

- **Gmail/personal SMTP** -- zero-cost and trivial for a single self-recipient,
  but a dead end the moment leaguemates are added (deliverability, HTML quirks,
  per-account sending limits). Rejected because the audience is "me now,
  leaguemates later."
- **SendGrid / SES** -- viable but more boilerplate (SendGrid) or heavier setup
  and domain verification (SES) than warranted for current scale.
- **Resend** -- clean Python/HTTP API, free tier covers current volume, good
  HTML rendering, scales to a recipient list with no rewrite. Chosen.

## Architecture

Approach chosen: **summary module + thin cron script** (over a single standalone
script or a refresh-pipeline step). It matches the repo's module conventions,
keeps the three concerns independently testable, and stays decoupled from the
refresh path (a summary bug cannot break the refresh; the summary does not fire
on every refresh).

```
src/fantasy_baseball/summary/
  models.py     # DailySummary + per-section dataclasses (typed payload)
  assemble.py   # build_daily_summary(...) -> DailySummary
  render.py     # render_html(summary) -> str; render_text(summary) -> str
  send.py       # send_email(html, text, subject, recipients) via Resend
scripts/send_daily_summary.py   # thin orchestrator: assemble -> render -> send
```

Three independently testable units:

- **assemble** -- reads the KV that the morning refresh already populated, AND
  stands up a live Yahoo session/league/team_key to fetch injuries (this is real
  coupling, not "just read the KV" -- see Data access), and returns a typed
  `DailySummary`. No rendering, no email I/O.
- **render** -- pure `DailySummary -> str`. Produces HTML (primary) and a plain
  text fallback. No data access.
- **send** -- Resend client wrapper. No knowledge of summary content beyond the
  rendered strings.

### Data model

`DailySummary` is a frozen dataclass with one typed sub-object per section, so
`assemble` and `render` never pass dicts around:

```python
@dataclass(frozen=True)
class DailySummary:
    as_of: date                         # MLB officialDate of "last night"
    last_night: list[PlayerLine]        # box-score lines for rostered players
    unmatched: list[str]                # rostered players we couldn't resolve
    streaks: list[StreakItem]           # hitters-only, single-window hot/cold
    standings_delta: StandingsDelta     # rank + total-points + per-category moves
    lineup_moves: list[LineupMove]      # week/ROS active-vs-bench recs
    injuries: list[InjuryItem]          # status + Yahoo injury_note text
    probables: list[ProbableMatchup]    # roster arms' upcoming starts (may be [])
    section_errors: list[str]           # names of builders that raised
```

Each builder is a pure function `(inputs) -> SectionModel`. A section with no
data (no games last night, no injuries, first-run standings baseline, absent
`PROBABLE_STARTERS`) yields an empty list / sentinel, and `render` omits that
block. `section_errors` records builders that *raised* (distinct from
legitimately empty) so `render` can note them and the empty-summary guard can
tell a failure from a quiet night.

### Data access details (the three under-specified sections)

**Last night's results.** Game logs are stored per-player-per-game keyed by
**MLBAM integer id** (`data/mlb_game_logs.py::get_player_game_log`), while
rosters are keyed by Yahoo `player_id` / `name::player_type`. So `build_last_night`
must, per rostered player: (1) resolve Yahoo name -> MLBAM via a crosswalk, (2)
read that player's game log, (3) filter to yesterday's MLB `officialDate`. The
existing `build_name_to_mlbam_map` (`streaks/pipeline.py:233`) is built from
**hitter** projection CSVs only; this feature must extend it to build from BOTH
hitter and pitcher projection CSVs (`{system}-hitters.csv` /
`{system}-pitchers.csv`, both carry `MLBAMID`) so pitcher lines resolve too.
Unresolved players are **not silently dropped**: they go into
`DailySummary.unmatched` and `render` lists them ("N players unmatched") so a
crosswalk gap is visible, not invisible.

**Injuries.** `fetch_injuries(league, team_key)` issues a raw Yahoo API call, so
`assemble` must build the Yahoo `league` handle (via the same
`get_yahoo_session()` / `get_league()` path the lineup scripts use) and resolve
the user's `team_key` (`fetch_teams()` + `find_user_team_key()`). `fetch_injuries`
is currently exercised only by tests; wiring it into a live path is new
integration work, not a no-op reuse. The plan must budget for it.

**Probables.** Read `PROBABLE_STARTERS` from the KV (already mapped to roster
arms by the refresh). It is written with `required=False`, so a KV miss is an
expected empty section, not an error. Do **not** read `data/weekly_schedule.json`
-- it is a raw league-wide, team-abbrev-keyed, refresh-written artifact (and the
checked-in copy is stale), the wrong layer for this.

### Streaks

`STREAK_SCORES` is a `streaks.models.Report` over a single window
(`window_days=14`), **hitters-only** (`resolve_hitters` drops pitchers,
`sunday.py:169-182`), categorized by calibrated percentile thresholds plus
Poisson continuation-probability models -- not a projection delta.
`build_streaks` reuses the existing classification (no new statistical model)
but must **aggregate** the per-player/per-category/per-method `Report` rows into
a compact hot/cold digest. The email will not contain pitcher streaks; the
section is explicitly hitters-only.

### Standings snapshot

The KV holds only *current* `standings`, so overnight movement requires diffing
against a stored baseline.

- New cache key `STANDINGS_SNAPSHOT` holding `{date, standings}` (full
  `Standings.to_json()` payload). It must be added to the `CacheKey` StrEnum and
  routed through `redis_key()` like every other key; it is written/read by the
  summary job itself and lives **outside** the refresh's write set (and outside
  `kv_sync`'s enumerated keys).
- `Standings.to_json()` stores per-team `rank`, `yahoo_points_for` (total roto
  points), and raw `CategoryStats` totals -- but **not** per-category place
  points. Therefore:
  - **Rank** and **total roto points** diff directly between snapshots.
  - **Per-category movement** (who gained the SB point overnight) requires
    re-scoring category rankings from the stored raw totals for *both* the prior
    snapshot and current standings via the existing `score_roto` machinery. All
    inputs are present in `to_json`, so this is feasible; it is computation, not
    a field lookup, and `build_standings_delta` owns it.
- Delta basis = **the last summary run** (not a fixed calendar day): "since you
  last looked" semantics, robust to a missed/failed run (the next email spans a
  longer window).
- **Staleness guard.** The delta is only meaningful if `STANDINGS` was refreshed
  since the last snapshot. `build_standings_delta` compares the current
  standings' `effective_date` (and/or the `META` refresh timestamp) against the
  stored snapshot's date; if they match (the morning refresh has not run, or the
  summary fired before it), it renders "standings not yet refreshed today"
  rather than a misleading "no movement."
- The snapshot is written back **only after a successful send** (see error
  handling), so a failed run does not corrupt the next delta baseline. If the
  snapshot write itself fails after a successful send, the run exits non-zero and
  logs loudly so the stale baseline is surfaced, not silently carried.
- First-ever run (no prior snapshot): the section renders "baseline established
  -- deltas start next run" rather than erroring.

## Operations

### Scheduling

A new Render cron job runs `scripts/send_daily_summary.py`. **The cron schedule
string (UTC) is the single source of truth for when the job runs** -- it is set
to fire shortly after the morning refresh cron so the KV is fresh. There is no
separate `send_hour` config that could contradict it; the script does not
self-gate on wall-clock time. The script sets `RENDER=true` before importing the
pipeline / first cache read (mirroring `scripts/refresh_remote.py:30-43`), then
fetches live injuries from Yahoo.

Ordering note: the summary depends on the morning refresh having run. Scheduling
"shortly after" is a timing convention, not a guarantee; the standings staleness
guard above is what actually protects correctness if the refresh is late or
failed.

### Config & secrets

The Render cron needs every credential the assemble step touches -- not just the
email key:

- `RESEND_API_KEY` -- Resend send auth.
- `YAHOO_OAUTH_JSON` -- headless Yahoo OAuth blob (written to a temp file, then
  `yahoo_oauth.OAuth2` auto-refreshes the access token), same mechanism the
  refresh cron already uses. Required for the live injury fetch.
- `UPSTASH_*` -- the Upstash REST creds; `RENDER=true` only flips the gate, the
  creds must be present to read the KV.

All are Render env vars + `.env` (gitignored). `config/league.yaml` gains a
`summary` block for non-secret settings:
- `recipients`: list of email addresses (one entry today, extensible).
- `from_address`: verified Resend sender.

Timezone is **not** a new config field. Reuse the codebase-wide
`utils/time_utils.LOCAL_TZ` (`America/New_York`) and `local_today()`; "last
night" is pinned to MLB `officialDate` (how game logs are already dated) so the
day boundary matches the data, not a separately-configured tz that could drift.

### Error handling

- Each of the six builders is wrapped independently: a builder that raises logs
  the error, appends its name to `section_errors`, and yields an empty section,
  so one bad section never kills the email. `render` notes which sections failed.
- **Live-auth failure mode.** The Yahoo OAuth refresh token can expire or be
  revoked; a headless cron cannot complete the browser consent flow. If the
  Yahoo session fails to build, the injuries builder fails (logged in
  `section_errors`) but the KV-sourced sections still send; the log makes the
  re-consent need visible.
- Send failure (Resend down / API error) logs loudly and exits non-zero so the
  Render cron surfaces it.
- The standings snapshot is written **only after a successful send**; a failed
  send does not advance the delta baseline.
- **Empty-summary guard, precisely defined.** The script skips sending only when
  assemble could not read the KV at all -- i.e. every KV-sourced builder raised
  (total cache miss), tracked via `section_errors` -- and exits non-zero. A
  legitimately quiet night (KV read fine, but no games / no injuries / no moves)
  still sends an email that says so; it is not suppressed.

### Testing

- Each builder: unit-tested against fixture inputs -- hot streak, no games last
  night, first-run standings baseline, injured player with a note, unresolved
  (unmatched) player, absent `PROBABLE_STARTERS`.
- Crosswalk extension: a pitcher name resolves to its MLBAM id from a pitcher
  projection CSV fixture (guards the hitter-only regression).
- `build_standings_delta`: rank/total-points diff; per-category recomputation
  from raw totals; staleness guard fires when `effective_date` is unchanged;
  first-run baseline path.
- `render_html` / `render_text`: snapshot test on a fully-populated
  `DailySummary`, one with empty sections (blocks omitted), and one with
  `section_errors` set (failure note rendered).
- `send.py`: Resend client mocked -- assert payload shape; never hit the network.
- Snapshot round-trip: first run establishes baseline, second run computes a
  correct delta; a failed send does not advance the baseline.
- Empty-summary guard: all-builders-raised suppresses send (non-zero exit);
  quiet-night still sends.

## Out of scope (v1)

- Pitcher hot/cold streaks (the streaks subsystem is hitters-only; not adding a
  pitcher streak model here).
- Multi-user delivery beyond a static recipient list (no per-user rosters /
  auth). True multi-tenant is issue #204 (wontfix).
- Configurable per-recipient section preferences.
- Historical standings trend charts (only the single overnight delta).
- Retry/queue on send failure beyond the cron's own next-day run.
- Automated recovery from Yahoo OAuth re-consent (surfaced via logs; manual).

## Conventions to honor

- ASCII-only in all strings that may hit `print()` / logs (Windows cp1252).
  Player names pulled from data may be non-ASCII; the entry-point script must
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` if it prints
  them, and the HTML email is UTF-8.
- Player IDs are `name::player_type`; never key on bare names.
- No `x or default` for numeric fields (0.0 is falsy).
- Read Upstash (not local SQLite) for live season state: set `RENDER=true`
  before the first cache read / pipeline import.
- Reuse existing functions (`fetch_injuries`, streak `Report`, `LINEUP_OPTIMAL`
  moves, `score_roto`, `build_name_to_mlbam_map`, `LOCAL_TZ`/`local_today`)
  rather than recomputing.
