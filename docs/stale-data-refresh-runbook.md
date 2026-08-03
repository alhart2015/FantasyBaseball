# Stale-data refresh (running without Yahoo)

When Yahoo OAuth is broken, the refresh pipeline aborts on its first step and
nothing downstream runs -- the dashboard freezes entirely, including the parts
that depend only on MLB game logs and projections.

Stale-data mode keeps those parts moving. It skips every step that pulls from
Yahoo and runs the rest against the last league state the pipeline persisted.

## Turning it on

**Production (Render):** set `FB_SKIP_YAHOO=1` in the Render dashboard. The
refresh cron POSTs `/api/refresh` with no arguments, so an env var is the only
switch that works without a code deploy -- and unsetting it is how you turn the
mode back off. No restart of the cron is needed; the next run picks it up.

**Locally:**

```bash
python scripts/refresh_remote.py --skip-yahoo
```

`FB_SKIP_YAHOO` also works locally. Accepted truthy values: `1`, `true`, `yes`,
`on` (case-insensitive). Anything else, including unset, means normal mode.

## What still refreshes

Everything whose inputs are MLB game logs, projections, or the persisted league
state:

- MLB game-log totals and season progress
- Blended / ROS / full-season projections
- Projected standings, team SDs, standings breakdown
- Pace, pace deviations, rankings
- Lineup optimizer and lineup moves
- Probable starters
- Leverage (league-wide and per-team)
- ROS Monte Carlo and SPoE
- Draft value

## What goes stale

| Skipped step | Effect |
|---|---|
| Yahoo auth, team lookup | No live league handle |
| Standings fetch | Reuses `cache:standings`; YTD category totals freeze |
| Roster fetch (user + opponents) | Reuses the newest `weekly_rosters_history` snapshot |
| Snapshot writes | `weekly_rosters_history` / `standings_history` stop growing |
| Free agents | Roster audit, stash board, and positions map keep their previous values |
| Transactions | Transaction feed and analyzer keep their previous values |
| Streaks | Cache keeps its previous value |

Pending moves are the exception: they are a diff of today's vs the effective
date's *live* roster, so they are cleared to `[]` rather than left stale --
republishing the old diff would keep showing moves that have already resolved.

`cache:meta` records `yahoo_skipped: true` and `rosters_as_of: <date>` so a
consumer can tell the underlying league state is stale even though
`last_refresh` is current.

## Known limitations

- **The numbers drift.** Projected standings advance with new game results
  while the YTD standings they build on stay frozen at the last live fetch.
  The longer the mode runs, the more the two disagree. Treat projected
  standings, SPoE, and Monte Carlo win probabilities as directional, not exact.
- **Roster changes are invisible.** Adds, drops, and trades made during the
  outage do not appear until the first live refresh, so the optimizer will
  recommend lineups from an out-of-date roster.
- **The daily summary cron is not covered.** `scripts/send_daily_summary.py`
  authenticates with Yahoo after its freshness gate passes. Because stale-data
  mode keeps writing a fresh `cache:meta`, that gate now passes and the script
  fails at auth instead of skipping. Expect no summary email during the outage.

## Turning it off

Unset `FB_SKIP_YAHOO` (or set it to `0`) once Yahoo auth works again. The next
refresh fetches live rosters and standings, writes new snapshots, and the
transaction analyzer picks up everything it missed on that run.

## Failure modes

Stale-data mode raises `RuntimeError` if there is no prior state to run
against -- no `cache:standings`, or no stored roster snapshot for the
configured team. That is deliberate: there is nothing to refresh, and a
dashboard built on nothing is worse than a failed cron.
