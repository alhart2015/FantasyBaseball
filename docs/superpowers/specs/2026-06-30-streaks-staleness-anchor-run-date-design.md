# Streaks: anchor the "current streak" on the run date

**Status:** approved design (2026-06-30); hardened via spec-review (2026-07-01)
**Scope:** the streaks staleness bug only. The separate "surface continuation
probability on the lineup chip" item is an independent, rendering-only follow-up
and is out of scope here.

## Problem

On 2026-06-30 Oneil Cruz displayed as **hot** on the /streaks page and the
lineup chip despite not having played in ~a month (on the IL). The hot/cold
label is a *frozen, stale* one, not a current signal.

Root cause is recency, not a bad calculation:

- `windows._build_per_day_frames` reindexes each player's rolling-window
  calendar only to `[first_played, last_played]` (`windows.py:72-74`). An
  injured player's most recent `window_end` is therefore frozen at his **last
  game** (~a month ago), and the trailing 14-day window there still captures his
  pre-injury stretch.
- `compute_windows` drops any window with `pa < 5` (`windows.py:347`), so an
  injured player's zero-PA "as of today" windows are never created — his most
  recent *surviving* window is the pre-injury one. (This is why extending the
  calendar to today does not by itself fix the bug: the zero-PA windows that
  would carry the "no recent games" signal get filtered out here.)
- `inference.score_player_windows` selects each player's most recent window
  `<= run_date` with **no lower bound** (`_load_most_recent_windows`,
  `inference.py:586-598`, `QUALIFY ROW_NUMBER() ... ORDER BY window_end DESC =
  1`). A month-old window is thus selected and scored as if it were the current
  streak.
- The /streaks **web page** (`web/templates/season/streaks.html` +
  `_streaks_row.html`, rendered by `web/season_routes.py:609-618` from the
  serialized `cache:streak_scores` dict) and the lineup chip (`indicator.py` via
  `build_indicator`, called from `season_routes.py:652` on `/lineup` and `:697`
  on `/lineup/tbodies`) both read that same payload, so both show the stale
  label. (The continuation-probability chip is a separate, out-of-scope
  future item — it does not currently consume this payload.)

## Goal

Anchor the "current streak" on the **run date**, not the player's last game. A
window only counts as a current streak if it ends within a small tolerance of
the run date. A player whose most recent window is staler than that is forced to
**neutral / inactive** (no hot/cold tone, no continuation probability). Roster
players are kept in the grid and annotated "inactive - N days"; inactive free
agents collapse to `composite == 0` and drop out of the FA section (see
"Free-agent behavior" below).

## Decisions (settled during brainstorming + spec-review)

- **Tolerance:** `STALE_TOLERANCE_DAYS = 4`. Absorbs Statcast's 1-2 day
  publication lag plus normal off-days, while catching any multi-week IL stint.
  Boundary is strict `>`: exactly 4 days stale still counts as current.
- **Stale behavior:** roster players are shown as **neutral/inactive** (kept in
  the grid, forced neutral, annotated) — *not* dropped from the roster section.
- **Run date:** `local_today()` (America/New_York calendar date;
  `utils/time_utils.py`), already passed as `build_report(today=...)` via
  `pipeline.compute_streak_report` (`today or local_today()`). No new clock
  source needed.
- **Staleness is window-based, not game-based** (see "Staleness metric" — this
  is a deliberate, load-bearing choice).

## Staleness metric: why window_end, not the true last-game date

The anchor uses `days_since = (run_date - window_end).days`, where `window_end`
is the end of the player's most-recent *surviving* window (trailing PA >= 5).
This is **not** always the player's literal last-game date, because
`compute_windows` drops `pa < 5` windows:

- For the headline case (a player who was playing regularly, then went out for
  weeks) the last game's trailing-14 window easily clears `pa >= 5`, so
  `window_end` == last-game date and the count is exactly "days since last game."
- For a just-returned player who has played only 1-2 games (< 5 trailing PA),
  the window ending on his actual last game is filtered out, so `window_end`
  falls back to a pre-absence qualifying window. Such a player is therefore
  forced inactive.

**Both are correct behavior for this feature.** The question the anchor answers
is "do we have a *scoreable* current 14-day window?", not "did the player suit
up yesterday." A player with < 5 PA over the last two weeks has no meaningful
14-day streak to display, so suppressing hot/cold for him is right. Because of
this, the annotation is worded as **"inactive - N days"** (idle for N days) and
must **not** claim "since last game" — for the common IL case they coincide, but
we do not display a "days since last game" number we cannot guarantee. Sourcing
the exact last-game date from `hitter_games` is a possible future refinement, not
part of this change.

## Design

### 1. Anchor point (correctness fix) — `inference.score_player_windows`

The single point both surfaces derive from. After the existing per-player
window selection:

- New module constant `STALE_TOLERANCE_DAYS = 4` in `inference.py`.
- New parameter `stale_after_days: int | None = STALE_TOLERANCE_DAYS`.
  `None` disables staleness handling (preserves the old behavior; used by
  analysis paths and by tests that want the pre-change semantics). In production
  it is always the default 4 — `build_report`/`compute_streak_report` do not
  expose it and it is not a `league.yaml` knob.
- For each player, compute `days_since = (run_date - window_end).days` where
  `run_date` is the existing `window_end_on_or_before` argument (which
  `build_report` passes as `today`). If `stale_after_days is not None` and
  `days_since > stale_after_days`, the player is **inactive**: force *every* one
  of that player's category scores to neutral — `label="neutral"`,
  `probability=None`, `drivers=()` — while keeping `window_end` intact so the
  day-count remains recoverable downstream.

Effect: a stale window can never surface as hot/cold from `score_player_windows`
again. Because `reports/sunday._composite` counts `#hot - #cold` over **labels**
(regardless of whether a probability was computed — so sparse-cat cold is
counted too), forcing every label to `neutral` collapses an inactive player's
`composite` to `0`, which makes both the report row and the chip neutral
automatically. Forcing `label="neutral"` (not merely `probability=None`) is
therefore load-bearing.

Players with *no* window at all remain a `no_window` skip (unchanged) — that is
a distinct "we have no data" case, not "played but not recently."

### 2. Inactive annotation — report layer

Distinguish "neutral because inactive" from "neutral because genuinely no
streak," and carry the day-count to the renderers and the chip without giving
the chip a clock.

- Add `days_since_last_game: int | None = None` to `reports/sunday.ReportRow`,
  **as the last field with an explicit `= None` default** (the dataclass is
  frozen and currently has zero defaulted fields; every existing `ReportRow(...)`
  construction — in `_row_from_scores` and across the test suite — must keep
  compiling, so the new field cannot be positional-without-default). `None` ⇒
  active/recent; an `int` ⇒ inactive, and the value is the day count.
- `_row_from_scores` gains a `today: date` parameter. It reads `window_end` from
  the player's scores (identical across the player's categories) and sets
  `days_since_last_game = (today - window_end).days` **only when** that exceeds
  `STALE_TOLERANCE_DAYS` (else `None`). So `days_since_last_game is not None`
  is exactly "is inactive." Both `_row_from_scores` and step 1 read the same
  `STALE_TOLERANCE_DAYS` constant, keeping the neutral-forcing and the
  annotation in lockstep; if the tolerance ever becomes a `build_report`
  parameter, it must be threaded to both call sites together.
- `build_report` threads its `today` into both `_row_from_scores` calls
  (roster + FAs). `_row_from_scores` has exactly two callers (both in
  `build_report`) and no direct test callers, so adding the param is contained.
- `streaks/dashboard.py` serializes `days_since_last_game` in the row payload
  (JSON `int` or `null`) and reads it back with `p.get("days_since_last_game")`
  (tolerant of already-cached payloads written before this field existed, and of
  the round-trip test fixture).

### 3. Rendering

There is **no HTML renderer of the `Report` dataclass.** The surfaces are:

- **Developer reports** — `reports/sunday.render_markdown` (`sunday.py:469`) and
  `reports/sunday.render_terminal` (`sunday.py:576`, terminal, not HTML). Both
  consume the `Report` dataclass. When `row.days_since_last_game is not None`,
  **append the marker into the player-name cell** (e.g. `"Oneil Cruz (inactive -
  30 days)"`) rather than adding a new column. Both renderers have a fixed
  header + a `len(headers)`-derived separator/column-width layout, so an
  in-cell append avoids header, separator-row, and `_column_widths` churn in
  both renderers.
- **The /streaks web page** (the surface where the bug was observed) — the Jinja
  templates `web/templates/season/streaks.html` + `_streaks_row.html`, rendered
  from the **serialized dict** (`read_cache_dict(CacheKey.STREAK_SCORES)`), not
  the `Report` object. The tone fix reaches this page for free (neutral labels
  make the per-category chips vanish and `composite` render `+0`), but the
  annotation requires a template edit in `_streaks_row.html` (and/or
  `streaks.html`) to render "inactive - N days" for rows whose serialized
  `days_since_last_game` is non-null. This is why step 2 serializes the field.
- **Lineup chip** (`indicator.py`): an inactive row already resolves to neutral
  via `composite == 0`. Enhance the neutral tooltip so that when
  `row.get("days_since_last_game")` is set it reads `"Inactive - N days"`
  instead of the generic `"composite=0 (no active streaks)"`. `indicator.py`
  stays duckdb-free (a pure payload consumer) — it only reads the new field.

All new marker text is **ASCII-only** (repo convention): use a hyphen, not an en
dash. (Pre-existing non-ASCII glyphs in `sunday.py`/`indicator.py` are not in
scope and are left as-is.)

The template marker must be guarded (`{% if row.days_since_last_game %}`). The
golden HTML snapshot `tests/test_web/snapshots/streaks.html`
(`tests/test_web/test_streaks_snapshot.py`) seeds an **active** row, so a
properly-guarded marker may leave the snapshot unchanged; regenerate it as part
of this work and eyeball the diff to confirm only intended marker/tone changes
appear (a no-op diff is acceptable and expected for an all-active fixture).

### 4. Free-agent behavior

`build_report` drops `composite == 0` rows from the **FA** section
(`sunday.py:333`). An inactive FA collapses to `composite == 0`, so it is
removed from `fa_rows` rather than shown annotated — desirable (an idle FA is not
a pickup signal). The "inactive - N days" annotation therefore appears only on
**roster** rows. The "kept in the grid" decision applies to roster players.

### 5. Edge cases

- Boundary: strict `>` — `days_since == 4` is still current.
- Statcast lag: active players sit at `days_since` 1-2, safely under 4.
- League-wide no-game stretches (e.g. the ~4-day All-Star break): once the gap
  plus lag exceeds 4 days, all players correctly read inactive until games
  resume — no one has a current streak mid-break. This is expected/correct, not
  a false positive; it self-corrects within 1-2 days of games resuming.
- `window_end is None`: only occurs for a player with no window, who is already a
  `no_window` skip and never reaches the annotation path.
- `stale_after_days=None`: no staleness filtering; `score_player_windows` behaves
  exactly as before.

## Components touched

- `streaks/inference.py` — `STALE_TOLERANCE_DAYS`, `stale_after_days` param,
  neutral-forcing in `score_player_windows`.
- `streaks/reports/sunday.py` — `ReportRow.days_since_last_game` (defaulted,
  last field), `_row_from_scores(today=...)`, `build_report` threading, the
  `render_markdown` and `render_terminal` renderers.
- `streaks/dashboard.py` — payload serialize/deserialize of the new field
  (`.get` on read). Update the module docstring's "mirrors the dataclass fields
  1:1 - round-trip equality holds" note to record the tolerant `.get()`
  back-compat for `days_since_last_game`.
- `streaks/indicator.py` — inactive tooltip.
- `web/templates/season/_streaks_row.html` (and/or `streaks.html`) — render the
  inactive marker on the web page.
- `tests/test_web/snapshots/streaks.html` — regenerated golden snapshot.

## Testing

- `score_player_windows`:
  - forces neutral for a player whose only window is > 4 days before `run_date`;
  - preserves hot/cold for a player with a window within 4 days;
  - `stale_after_days=None` disables the forcing (old behavior).
  - Note: the existing `score_player_windows` suite anchors
    `window_end_on_or_before = scoring_end + 1 day` (so `days_since == 1`),
    meaning default-on staleness does not flip those tests to all-neutral;
    confirm they still pass unchanged.
- `_row_from_scores`: sets `days_since_last_game` when the window is stale,
  `None` when recent.
- `build_report` integration (`test_sunday_report.py`): this test anchors
  `today = latest_end` (the global `MAX(window_end)` across all seeded players),
  so under default-on staleness any fixture player whose latest window predates
  that global max by > 4 days is now forced neutral. Confirm it still passes —
  its assertions (roster row count, rows kept-not-dropped, `len(fa_rows) <=
  len(fas)`) are robust to this. If a player intended to be active flips to
  neutral, adjust the fixture so intended-active players' windows sit within 4
  days of `latest_end`.
- `dashboard` payload: `days_since_last_game` round-trips through
  serialize/deserialize. The round-trip fixture must include a row with
  `days_since_last_game` set to an **explicit non-None int**. This field is a
  *stored* `ReportRow` attribute (defaulting to `None`) computed only inside
  `_row_from_scores`, never at `ReportRow(...)` construction — so the fixture
  must pass `days_since_last_game=<int>` explicitly. (Merely moving
  `report_date`/`window_end` apart does NOT auto-populate it; the existing
  `_example_report` row leaves it at the `None` default, which would mask a
  forgotten `_deserialize_row` update.)
- `indicator`: inactive tooltip appears when the field is present; active chips
  are unchanged.
- Web: `test_streaks_snapshot.py` regenerated; `test_streaks_route.py` /
  `test_season_routes.py` still pass with the new field present in the payload.
- Regression: a recent hot player still reads hot (tolerance does not
  over-suppress an active player).

## Out of scope

- Surfacing continuation probability on the chip (separate rendering-only item).
- Sourcing the exact last-game date from `hitter_games` for the annotation
  (possible future refinement; this change uses the window-based count).
- Changing `compute_windows`' `pa >= 5` training-window filter or the
  `[first_played, last_played]` calendar (the fix is at the scoring/report layer,
  leaving training-window generation untouched).
