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

All six sections from issue #200 are in v1. Investigation showed every section
is low-cost because the data already exists in the stack:

| Section | Source |
|---|---|
| Last night's results (box-score lines for rostered players) | game logs / rosters in KV |
| Hot/cold streaks (7d & 14d vs projection) | `STREAK_SCORES` (already computed) |
| Standings changes (roto points gained/lost overnight) | `STANDINGS` + new dated snapshot |
| Recommended lineup changes for today | `LINEUP_OPTIMAL` + `ROSTER_AUDIT` |
| Injury news affecting roster | live Yahoo `fetch_injuries()` -- `injury_note` carries the news text |
| Upcoming probable pitcher matchups | `PROBABLE_STARTERS` / `weekly_schedule.json` |

Key finding: **injury news needs no external news feed.** Yahoo's per-player
`injury_note` / `status_full` fields (already extracted by
`lineup/yahoo_roster.py::fetch_injuries` / `parse_injuries_raw`) carry the
free-text injury update. Probables are likewise already fetched.

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

- **assemble** -- reads the KV that the morning refresh already populated, plus
  live injury fetch, and returns a typed `DailySummary`. No rendering, no I/O to
  email.
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
    as_of: date
    last_night: list[PlayerLine]        # box-score lines for rostered players
    streaks: StreakDigest               # hot/cold, 7d & 14d vs projection
    standings_delta: StandingsDelta     # roto points gained/lost overnight
    lineup_moves: list[LineupMove]      # today's recommended changes
    injuries: list[InjuryItem]          # status + Yahoo injury_note text
    probables: list[ProbableMatchup]    # upcoming SP matchups for roster arms
```

Each builder is a pure function `(cache_data, ...) -> SectionModel`. A section
with no data (no games last night, no injuries, first-run standings baseline)
yields an empty list / sentinel, and `render` omits that block. One empty or
failed section never blocks the others.

### Section builders

| Builder | Source | Logic |
|---|---|---|
| `build_last_night` | game logs + rosters | Yesterday's box-score line per rostered player (H, HR, RBI, R, SB, AVG for hitters; IP, ER, K, W/SV for pitchers). Omit players who did not play. |
| `build_streaks` | `STREAK_SCORES` | Reuse existing 7d/14d-vs-projection scores; classify hot/cold by existing thresholds. No new math. |
| `build_lineup_moves` | `LINEUP_OPTIMAL` + `ROSTER_AUDIT` | Diff optimal vs. current active lineup -> today's start/sit/swap recs. |
| `build_injuries` | live `fetch_injuries()` | Yahoo `status`, `status_full`, `injury_note`. Injury news comes from `injury_note`. |
| `build_probables` | `PROBABLE_STARTERS` / `weekly_schedule.json` | Roster arms' next starts + opponent. |
| `build_standings_delta` | `STANDINGS` + new snapshot | See below. |

### Standings snapshot

The KV holds only *current* `standings`, so computing "roto points gained/lost
since the last email" requires diffing against a stored snapshot.

- New cache key `STANDINGS_SNAPSHOT` holding `{date, standings}`.
- `build_standings_delta` reads the prior snapshot and diffs category-roto
  points vs. current standings.
- Delta basis is **the last summary run** (not a fixed calendar day): "since you
  last looked" semantics, robust to a missed/failed run (the next email just
  spans a longer window).
- The snapshot is written back **only after a successful send** (see error
  handling), so a failed run does not corrupt the next delta baseline.
- First-ever run (no prior snapshot): the section renders "baseline established
  -- deltas start next run" rather than erroring.

The snapshot is self-maintaining inside the summary job -- no extra cron, no
dependency on the refresh writing history.

## Operations

### Scheduling

A new Render cron job runs `scripts/send_daily_summary.py`, scheduled shortly
after the morning refresh cron so the KV is fresh. The script sets `RENDER=true`
before the first cache read (reads Upstash, per the cross-cutting convention)
and fetches live injuries from Yahoo. Send time and timezone are config-driven
(see below), not encoded solely in the cron string.

### Config & secrets

- `RESEND_API_KEY` -- Render env var + `.env` (gitignored).
- `config/league.yaml` gains a `summary` block:
  - `recipients`: list of email addresses (one entry today, extensible).
  - `send_hour` + timezone: informs the intended send slot.
  - `from_address`: verified Resend sender.

### Error handling

- Each of the six builders is wrapped independently: a builder that raises logs
  the error and yields an empty section, so one bad section never kills the
  email. The email notes which section failed to build.
- Send failure (Resend down / API error) logs loudly and exits non-zero so the
  Render cron surfaces it.
- The standings snapshot is written **only after a successful send**, so a failed
  run does not advance the delta baseline.
- If assembly produces a completely empty summary (total KV miss), the script
  skips sending rather than mailing an empty shell (and exits non-zero).

### Testing

- Each builder: unit-tested against fixture KV payloads -- hot streak, no games
  last night, first-run standings baseline, injured player with a note.
- `render_html` / `render_text`: snapshot test on a fully-populated
  `DailySummary` and on one with empty sections (blocks omitted cleanly).
- `send.py`: Resend client mocked -- assert payload shape; never hit the network
  in tests.
- Snapshot round-trip: first run establishes baseline, second run computes a
  correct delta; failed send does not advance the baseline.

## Out of scope (v1)

- Multi-user delivery beyond a static recipient list (no per-user rosters /
  auth). The recipient list is a config array; true multi-tenant is issue #204
  (wontfix).
- Configurable per-recipient section preferences.
- Historical standings trend charts (only the single overnight delta).
- Retry/queue on send failure beyond the cron's own next-day run.

## Conventions to honor

- ASCII-only in all strings that may hit `print()` / logs (Windows cp1252).
  Player names pulled from data may be non-ASCII; the entry-point script must
  `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` if it prints
  them, and the HTML email is UTF-8.
- Player IDs are `name::player_type`; never key on bare names.
- No `x or default` for numeric fields (0.0 is falsy).
- Read Upstash (not local SQLite) for live season state: set `RENDER=true`
  before the first cache read.
- Reuse existing functions (`fetch_injuries`, streak scores, lineup optimizer
  output) rather than recomputing.
