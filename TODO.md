# TODO — Code Audit Findings (2026-03-18)

## Critical (will break draft day)

- [x] **Crash on ERA/WHIP display before pitchers drafted** — `run_draft.py:221` does `f'ERA:{totals["ERA"]:.2f}'` but `balance.get_totals()` returns `None` for ERA/WHIP when no pitchers are drafted yet. Crashes on first hitter pick.

- [x] **Failing test has wrong assertion** — `test_leverage.py:45` asserts `leverage["R"] > leverage["SB"]` but the test data has a tiny SB defense gap (5) which correctly produces higher SB leverage. Fix the test, not the code.

- [x] **Dashboard roster grid shows players in wrong positions** — `dashboard.html` `renderRoster` splices the flat `user_roster` list (draft order) across positions using filled counts. Names end up under wrong position labels.

- [x] **Duplicate name collisions remove wrong players** — 68 names appear twice on the board (e.g., "Juan Soto" OF and "Juan Soto" SP are different people). `apply_keepers` and `board[~board["name"].isin(drafted)]` remove both. Same issue during draft when filtering available players.

## High (will give wrong advice)

- [x] **`build_draft_board` ignores custom roster slots for replacement levels** — `board.py:35` calls `calculate_replacement_levels(pool)` without passing `starters_per_position` from config. Uses hardcoded 10-team defaults. Player VAR rankings may be wrong.

- [x] **Lineup optimizer ignores custom roster slots** — `optimizer.py:9-14` builds `HITTER_SLOTS` at module import from `DEFAULT_ROSTER_SLOTS`. `optimize_hitter_lineup` doesn't accept roster slots as a parameter.

- [x] **Dead fallback code in `run_lineup.py:141-150`** — Fallback loop finds a projection match for unmatched roster players but never appends to `roster_hitters` or `roster_pitchers`. Players silently dropped from lineup optimization.

- [x] **Recommender never flags IF or UTIL as position needs** — `recommender.py:58` `_get_unfilled_positions` skips IF and UTIL. Empty IF/UTIL slots never get `[NEED]` flags. That's 3 slots (1 IF + 2 UTIL) invisible to the need detector.

- [x] **Number selection picks from stale recommendations** — `run_draft.py:260-264` regenerates recs without `picks_until_next` when user types a number. Order can differ from what was displayed, causing wrong player selection.

- [x] **Scarcity note overwrites position need note** — `recommender.py:42-46` overwrites `note` when scarcity condition is met, losing the position need explanation while `need_flag` stays True.

## Medium

- [x] **Projection dilution for players in fewer systems** — Blending weights stats by system weight even when a player appears in only 1 of 3 systems. Their counting stats get multiplied by ~0.33. Affects 2,265 hitters (mostly fringe, none with AB > 200).

- [x] **`get_filled_positions` uses VAR-optimal position, not actual roster slot** — `recommender.py:77` counts a SS/2B player as filling SS (their best VAR position) even if you'd roster them at 2B. Can misidentify which positions are filled.

- [x] **Scarcity check uses tiny window** — `recommender.py:44` counts `remaining_at_pos` in top `n * 3 = 15` players only. Produces false scarcity warnings when depth exists further down the board.

- [x] **Dashboard shows team number instead of team name** — `dashboard.html:204` `renderStatus` shows `state.picking_team` (integer) instead of the team name from config. User sees "8" instead of "Hart of the Order".

- [x] **No graceful Ctrl+C/EOF handling during draft** — Accidental Ctrl+C or Ctrl+D crashes the script with no recovery path. State is preserved in JSON but restart requires manual context recovery.

- [x] **9,278-player board is excessive** — Board includes every player from all projection systems, including minor leaguers with 0 AB. Consider filtering to meaningful projections (AB > 50 or IP > 10).

## Low

- [x] **Dashboard XSS via innerHTML** — Player names inserted via `innerHTML` with template literals. Low risk (names from CSV) but not sanitized.

- [x] **No board search/sort in dashboard** — Position filters exist but no search box or column sorting. Hard to find specific players during a live draft.

- [x] **Flask dev server for dashboard** — Uses `app.run()` (single-threaded dev server). Fine for one user but could lag under rapid polling.
