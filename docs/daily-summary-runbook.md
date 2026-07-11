# Daily Summary Email -- Operations Runbook

The daily summary email is an automated morning digest of overnight team state
(last night's results, hot/cold streaks, standings movement, lineup
recommendations, injuries, upcoming probables). It is sent via Resend from a
Render cron job that runs shortly after the morning data refresh.

Entry point: `scripts/send_daily_summary.py` (function `main`).

## How it works

1. Flip to remote mode: `main()` sets `os.environ["RENDER"] = "true"` at runtime
   so all `get_kv()` reads hit Upstash (the source of truth for live state), not
   the local SQLite cache. This is set inside `main()` -- not at module import
   time -- so importing the module has no global side effect.
2. Freshness gate: read the `META` cache entry's provenance `_written_at`
   timestamp (set by `write_cache` on every write, converted to local time). If
   `META` is absent, or `_written_at` is not from today, log an error and exit
   non-zero WITHOUT sending (a stale/absent refresh means the numbers would be
   wrong). Note: the gate deliberately does NOT parse the free-text
   `META.last_refresh` payload field -- production writes it in an inconsistent,
   sometimes date-less form (e.g. `"9:00 AM"`), so `_written_at` is the reliable
   signal.
3. Assemble the summary from cached KV data, render HTML + text, and send via
   Resend.
4. Write the standings snapshot (`STANDINGS_SNAPSHOT`) ONLY after a successful
   send, so a failed run never corrupts tomorrow's standings-delta baseline.

## Render cron job

Configure a Render cron job (Dashboard -> New -> Cron Job, or add a
`type: cron` service in `render.yaml`) with:

- Command: `python scripts/send_daily_summary.py`
- Schedule (UTC): approximately +15 minutes after the morning refresh cron.
  The refresh cron is configured in the Render dashboard (it is not checked into
  this repo). Look up the exact UTC time of the refresh job and set this cron to
  run 15 minutes later, giving the refresh time to finish writing to Upstash
  before the summary reads it. The freshness gate (below) is the backstop if the
  refresh has not completed.
- Runtime / build: same as the web service -- `pip install -e .`, Python 3.11.

### Required environment variables

Set these on the cron job (most mirror the web service's env):

- `RESEND_API_KEY` -- Resend API key used to send the email.
- `YAHOO_OAUTH_JSON` -- Yahoo OAuth credentials JSON (used to resolve the league
  and the user's team key).
- `UPSTASH_REDIS_REST_URL` -- Upstash REST URL (live KV source of truth).
- `UPSTASH_REDIS_REST_TOKEN` -- Upstash REST token.

Note: the script also sets `RENDER=true` itself at runtime, so you do not need
to set `RENDER` on the cron -- but setting it does no harm.

## Resend setup

- Verify the sender domain in Resend. The `from_address` used to send the email
  must be on a domain you have verified in the Resend dashboard, or Resend will
  reject the send.
- Recipients and the from-address live in `config/league.yaml` under the
  `summary` block:

  ```yaml
  summary:
    recipients:
      - "you@example.com"
    from_address: "digest@your-verified-domain.com"
  ```

  If `summary.recipients` or `summary.from_address` is missing, the script logs
  an error and exits non-zero without sending.

## Failure semantics

The script exits non-zero on any failure, so the Render cron marks the run
failed and surfaces it:

- Exit 1: refresh not fresh -- `META` is missing or its `_written_at` is not from today
  (the morning refresh did not run / has not finished). Nothing is sent.
- Exit 2: configuration missing -- `RESEND_API_KEY` unset, or
  `summary.recipients` / `summary.from_address` not configured. Nothing is sent.
- Exit 3: Resend send failed -- the email did not go out. The standings snapshot
  is NOT advanced, so the next run's delta baseline is preserved.
- Exit 0: sent successfully and snapshot written.

## Manual local test

To send a real email from your machine against the live Upstash data (the script
reads Upstash and sends a real message):

```bash
RENDER=true RESEND_API_KEY=... python scripts/send_daily_summary.py
```

This requires local Yahoo OAuth (same as `run_lineup.py`) and the Upstash
credentials available in the environment or `.env`. Point `summary.recipients`
in `config/league.yaml` at your own address first so the test email goes to you.
