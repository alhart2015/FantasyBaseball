# Upcoming Pitcher Starts

## Overview

Replace the current "Probable Starters" table with a forward-looking view of every roster starting pitcher's projected starts during the upcoming scoring week. MLB only announces probable starters 2–3 days ahead, so by mid-week the existing table is mostly empty and "kinda useless." The fix: where MLB has announced a starter, use it; otherwise project from a 5-team-game rotation anchored on the pitcher's most recent start.

## Problem

`get_probable_starters` in `lineup/matchups.py` cross-references the roster against MLB's `probable_pitchers` for the Yahoo scoring week. Because MLB only populates probables ~2–3 days ahead:

- On Monday, the table shows Mon–Wed starters and a lot of "TBD" for Thu–Sun.
- On Wednesday, the table is even sparser — most of the visible week has already happened.
- The user can't see "who pitches Friday" or "do I have a 2-start guy?" until the day before.

The user wants every roster SP's full week of projected starts visible on Monday morning.

## Scope

**Pitcher universe:** Roster pitchers where:

1. Yahoo eligibility includes `SP`, AND
2. Blended projection has `gs > 0`.

This excludes pure RPs/closers (no SP eligibility) and SP-eligible swingmen who are projected as relievers (`gs == 0`). Opponent rosters are out of scope for this iteration — keep the current roster-only behavior.

**Window:** Yahoo's Mon–Sun scoring week (`start_date`/`end_date` from `_fetch_standings_and_roster`). Same as today.

**Lookback:** 14 days before `start_date`, used only to find each pitcher's most recent start (the "anchor"). 14 days covers typical IL/skip rotation gaps without ballooning the schedule fetch.

## Data sources

### MLB schedule (extended)

`fetch_week_schedule(start_date, end_date)` becomes `fetch_week_schedule(start_date, end_date, lookback_days=0)`. The refresh pipeline passes `lookback_days=14`, so the actual statsapi call covers `(start_date − 14d) .. end_date` in one request. The returned `probable_pitchers` list now includes both:

- **Past games** (within the lookback): MLB populates `home_probable_pitcher` / `away_probable_pitcher` with the *actual* starting pitcher for completed games. These are the anchor points.
- **Future games** (within the scoring week): some are MLB-announced (real names), some are `"TBD"`.

The `games_per_team` and other returned fields keep their current semantics (counts only the scoring week, not the lookback) to avoid affecting other consumers of the cache.

### Pitcher → MLB team

Pulled from the projection row's `team` field during the SP filter step. Falls back to "the team they last appeared as a starter for in the lookback" if `team` is missing — handles mid-season trades where the projection is stale.

### Team batting stats / matchup factors

Unchanged. `get_team_batting_stats` and `calculate_matchup_factors` keep their current shape and feed the per-start `matchup_quality` + `detail` payload exactly as today.

## Anchor + projection logic

For each in-scope SP:

1. **Build the team game index.** Filter the extended `probable_pitchers` list to the pitcher's team. Sort chronologically by `(date, game_number)` so doubleheaders are two distinct entries. This is the team's game stream.
2. **Find the anchor.** Walk the team game index backwards looking for the most recent game where this pitcher's name appears as `away_probable_pitcher` or `home_probable_pitcher` AND the game date is strictly before `today` (`local_today()`). Normalized-name matching (existing `normalize_name`), since both sides use FanGraphs/MLB name spellings. The anchor's index in the team stream is the rotation reference point. Using `today` (not `start_date`) as the cutoff matters when the refresh runs mid-week — a pitcher who already started this past Monday should anchor on that Monday, not on the prior week.
3. **Project forward.** From `anchor_index + 5`, step by `+5` until past `end_date`. Each step lands on a specific team-game (date + opponent), so the projected start carries that game's date and opponent. If the team has an off-day during the week, the calendar gap stretches automatically — the rotation is in team-game space, not calendar-day space.
4. **Merge with announced starts.** Any future game in the scoring week where this pitcher is the announced probable becomes an entry with `announced=True`. Projected entries get `announced=False`.
5. **Resolve collisions.** If a projected entry lands on a team-game where MLB has announced a *different* pitcher, drop the projection (MLB's announcement is authoritative for that game). If the projection lands on a game with no announcement, keep it as projected.
6. **No anchor → no projections.** If a pitcher has no GS in the lookback window, they get whatever MLB has announced for them in the scoring week (possibly nothing). Acceptable corner case for IL returnees with mid-week MLB announcements.
7. **Pitchers with zero starts in the window are not included** in the output.

Per-start payload, after the matchup adjustment pipeline runs:

```json
{
  "date": "2026-05-11",
  "day": "Mon",
  "opponent": "LAD",
  "indicator": "@",
  "announced": true,
  "matchup_quality": "Tough",
  "detail": {"ops": 0.789, "ops_rank": 4, "k_pct": 22.1, "k_rank": 18}
}
```

Per-pitcher rollup keeps the existing fields (`pitcher`, `starts`, `matchups`, summary `matchup_quality`) so the template doesn't need a wholesale rewrite — only the per-start chip rendering and the new `announced` flag are new.

## Code organization

- **New module `src/fantasy_baseball/lineup/upcoming_starts.py`** with pure functions:
  - `build_team_game_index(probable_pitchers, team_abbrev) -> list[GameSlot]` — chronological list of team games with `(date, opponent, indicator, game_number, announced_starter)`.
  - `find_anchor_index(team_games, pitcher_name, before_date) -> int | None`.
  - `project_start_indices(anchor_index, total_games, end_date_index, step=5) -> list[int]`.
  - `compose_pitcher_entries(pitcher, team_games, projections, matchup_factors, team_stats) -> list[StartEntry]`.
- **`data/mlb_schedule.py`:** add `lookback_days: int = 0` parameter to `fetch_week_schedule` and `get_week_schedule`. The actual statsapi call uses `(start_date − lookback_days) .. end_date`. The `start_date`/`end_date` fields in the returned dict still reflect the scoring week, not the lookback span.
- **`lineup/matchups.py::get_probable_starters`:** delegated to the new module. Keep the OPS-rank/quality computation in `matchups.py` (existing concern). The `get_probable_starters` signature gains an `sp_filter_predicate` callable parameter; default keeps current behavior so existing callers/tests stay green during rollout.
- **`web/refresh_pipeline.py::_fetch_probable_starters`:** pre-filter `pitcher_roster_for_schedule` to SP-eligible AND `gs > 0`, pass `lookback_days=14` through to `get_week_schedule`. Otherwise unchanged.
- **Template `season/lineup.html`:** the "Day | Opponent | Matchup | Starts" row format becomes one row per pitcher with a chip stack: each chip shows `{day} {indicator}{opponent} · OPS rank ({rank})`. Background color carries Great/Fair/Tough. Projected chips get a dotted border; tooltip says e.g. `"Projected from rotation; last start Sun @LAD"`. The `Matchup` column collapses into the chips. The `Starts` column keeps its 2-start badge.

## Testing

Unit tests in `tests/test_lineup/test_upcoming_starts.py` with synthetic schedule fixtures:

- **Simple 5-day rotation, no off-day.** Anchor Mon → projected Sat. ✓
- **Off-day in middle of week.** Mon start, off-day Wed → projected start lands Sun (5 team games later, calendar = +6 days). ✓
- **Two-start week from lookback.** Anchor was last Wed → first projected start Mon, second Sat. Both in window.
- **MLB-announced start matches projection.** Anchor Mon, MLB has the same pitcher announced for Sat → entry is `announced=True`, no duplicate.
- **Announced different pitcher collides with projection.** Our projection says SP-X starts Friday; MLB announced SP-Y for that game → SP-X projection is dropped.
- **No anchor in lookback.** Pitcher has no recent GS → empty entry list (unless MLB announces them mid-week).
- **Doubleheader.** Team plays two games on one date; rotation steps over both correctly via game_number tie-break.

Existing `tests/test_web/test_refresh_pipeline.py` updated to seed both lookback and scoring-week schedule data and assert the new payload shape.

## Rollout

Cache key (`PROBABLE_STARTERS`) shape is a superset of the current shape: existing fields are preserved, new fields are added. A single refresh repopulates the cache. No flag, no migration, no dual-write path.

Manual verification: run `scripts/run_season_dashboard.py` locally against the live league before merging — the user's standing rule for refresh-path changes.

## Out of scope

- Opponent SP coverage (current behavior is roster-only; keep that).
- Handedness-aware lineup strength (vs LHP/RHP wOBA splits) — needs a new data source; deferred.
- 6-man rotation handling.
- Pitcher swap detection (rotation skips beyond MLB's announcement).
- "Started in the last 7 days" recency filter — explicitly dropped during design; revisit if the projection produces obviously wrong results.
