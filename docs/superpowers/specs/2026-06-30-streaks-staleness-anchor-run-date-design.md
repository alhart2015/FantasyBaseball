# Streaks: anchor the "current streak" on the run date

**Status:** approved design (2026-06-30)
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
  would carry the "no recent games" signal get filtered out.)
- `inference.score_player_windows` selects each player's most recent window
  `<= run_date` with **no lower bound** (`_load_most_recent_windows`,
  `inference.py:586-598`, `QUALIFY ROW_NUMBER() ... ORDER BY window_end DESC =
  1`). A month-old window is thus scored as if it were the current streak.
- The /streaks report (`reports/sunday.build_report`) and the lineup + continuation
  chips (`indicator.py`) both derive from the same `cache:streak_scores` payload,
  so both show the stale label.

## Goal

Anchor the "current streak" on the **run date**, not the player's last game. A
window only counts as a current streak if it ends within a small tolerance of
the run date. A player whose most recent window is staler than that is shown as
**neutral / inactive** with an "N days since last game" annotation — kept in the
grid, but with no hot/cold tone and no continuation probability.

## Decisions (settled during brainstorming)

- **Tolerance:** `STALE_TOLERANCE_DAYS = 4`. Absorbs Statcast's 1-2 day
  publication lag plus normal off-days, while catching any multi-week IL stint.
  Boundary is strict `>`: exactly 4 days stale still counts as current.
- **Stale behavior:** shown as **neutral/inactive** (kept in the grid, forced
  neutral, annotated "N days since last game") — *not* dropped from the report.
- **Run date:** `local_today()` (the real calendar date), already passed as
  `build_report(today=...)` via `pipeline.compute_streak_report`
  (`today or local_today()`). No new clock source needed.

## Design

### 1. Anchor point (correctness fix) — `inference.score_player_windows`

The single point both surfaces derive from. After the existing per-player
window selection:

- New module constant `STALE_TOLERANCE_DAYS = 4` in `inference.py`.
- New parameter `stale_after_days: int | None = STALE_TOLERANCE_DAYS`.
  `None` disables staleness handling (preserves the old behavior for
  analysis/back-compat and makes the feature explicit in tests).
- For each player, compute `days_since = (run_date - window_end).days` where
  `run_date` is the existing `window_end_on_or_before` argument (which
  `build_report` passes as `today`). If `stale_after_days is not None` and
  `days_since > stale_after_days`, the player is **inactive**: force *every* one
  of that player's category scores to neutral — `label="neutral"`,
  `probability=None`, `drivers=()` — while keeping `window_end` intact so the
  day-count remains recoverable downstream.

Effect: a stale window can never surface as hot/cold from `score_player_windows`
again. Because `reports/sunday._composite` counts `#hot - #cold` over labels,
an inactive player's `composite` collapses to `0`, which makes both the report
row and the chip neutral automatically.

Players with *no* window at all remain a `no_window` skip (unchanged) — that is
a distinct "we have no data" case, not "played but not recently."

### 2. Inactive annotation — report layer

Distinguish "neutral because inactive" from "neutral because genuinely no
streak," and carry the day-count to the renderers and the chip without giving
the chip a clock.

- Add `days_since_last_game: int | None` to `reports/sunday.ReportRow`
  (`None` ⇒ active/recent; an `int` ⇒ inactive, and the value is the day count).
- `_row_from_scores` gains a `today: date` parameter. It reads `window_end` from
  the player's scores (identical across the player's categories) and sets
  `days_since_last_game = (today - window_end).days` **only when** that exceeds
  `STALE_TOLERANCE_DAYS` (else `None`). So `days_since_last_game is not None`
  is exactly "is inactive." The shared constant keeps this in lockstep with the
  neutral-forcing in step 1.
- `build_report` threads its `today` into both `_row_from_scores` calls
  (roster + FAs).
- `streaks/dashboard.py` serializes `days_since_last_game` in the row payload
  (JSON `int` or `null`) and reads it back in the round-trip deserializer.

### 3. Rendering

- **/streaks report** (both the markdown and HTML renderers that consume
  `Report`): show an "inactive - N days since last game" marker on any row where
  `days_since_last_game is not None`. ASCII-only (repo convention): use a hyphen,
  not an en dash.
- **Lineup chip** (`indicator.py`): an inactive row already resolves to neutral
  via `composite == 0`. Enhance the neutral tooltip so that when
  `row.get("days_since_last_game")` is set it reads
  `"Inactive - N days since last game"` instead of the generic
  `"composite=0 (no active streaks)"`. `indicator.py` stays duckdb-free (it is a
  pure payload consumer) — it only reads the new field.

### 4. Edge cases

- Boundary: strict `>` — `days_since == 4` is still current.
- Statcast lag: active players sit at `days_since` 1-2, safely under 4.
- `window_end is None`: only occurs for a player with no window, who is already a
  `no_window` skip and never reaches the annotation path.
- `stale_after_days=None`: no staleness filtering; `score_player_windows` behaves
  exactly as before (used by analysis paths / tests).

## Components touched

- `streaks/inference.py` — `STALE_TOLERANCE_DAYS`, `stale_after_days` param,
  neutral-forcing in `score_player_windows`.
- `streaks/reports/sunday.py` — `ReportRow.days_since_last_game`, `_row_from_scores(today=...)`,
  `build_report` threading, the two report renderers.
- `streaks/dashboard.py` — payload serialize/deserialize of the new field.
- `streaks/indicator.py` — inactive tooltip.

## Testing

- `score_player_windows`:
  - forces neutral for a player whose only window is > 4 days before `run_date`;
  - preserves hot/cold for a player with a window within 4 days;
  - `stale_after_days=None` disables the forcing (old behavior).
- `_row_from_scores`: sets `days_since_last_game` when the window is stale,
  `None` when recent.
- `dashboard` payload: `days_since_last_game` round-trips through
  serialize/deserialize (including the `None` case).
- `indicator`: inactive tooltip appears when the field is present; active chips
  are unchanged.
- Regression: a recent hot player still reads hot (tolerance does not
  over-suppress an active player).

## Out of scope

- Surfacing continuation probability on the chip (separate rendering-only item).
- Changing `compute_windows`' `pa >= 5` training-window filter or the
  `[first_played, last_played]` calendar (the fix is at the scoring/report layer,
  leaving training-window generation untouched).
