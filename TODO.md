# TODO — In-Season Enhancements

- [ ] **Show games this week on lineup page** — The lineup template reads `games_this_week` but it's never populated (always 0). Cross-reference each roster player with the weekly MLB schedule to show how many games they play this week. Useful for start/sit decisions — a player with 7 games is more valuable than one with 4. The schedule data is already fetched in Step 9 (`get_week_schedule`); just need to count games per team and attach to each Player.

- [ ] **Pitcher streaming tool** — Score free agent SPs by matchup quality to identify streamers (pick up a mediocre pitcher facing a terrible offense for one start, then drop). Builds on the matchup adjustment system.

- [ ] **Browser-based OAuth flow for season dashboard** — Add Yahoo OAuth redirect flow directly in the dashboard so it can be used from a phone without CLI re-auth. When the token is expired, redirect to Yahoo login, handle the callback, store the refreshed token. Required before remote hosting.

- [ ] **Batch MLB roster fetch** — `fetch_and_load_game_logs` calls `statsapi.get("team_roster")` 30 times (once per team) to build the player list. The MLB API supports `/sports/1/players` to get all players in one call. Also, `statsapi.get("teams")` is called in 3 separate modules (`db.py`, `mlb_schedule.py`, `matchups.py`) — extract a shared utility.

- [ ] **Incremental game log fetch with startDate** — Currently fetches the full season game log for each player and filters client-side. The MLB Stats API accepts `startDate`/`endDate` params on the gameLog endpoint. Track the last sync timestamp (e.g., 2026-03-29 01:23 EST) and only pull game logs since then. Query the `game_logs` table for `MAX(date)` per player (or globally) and pass `startDate=last_date+1` to skip downloading games we already have.

- [ ] **Yahoo fantasy and mlbapi mcp servers** — Do they exist?

- [ ] **Add section for hot waiver pickups** — This is the opposite of buy-low. These are people outperforming expectations who are on a hot streak and could be picked up to ride the hot hand.

- [ ] **Sort buy-low candidates by wSGP** — Buy-low candidates are currently sorted by largest underperformance (gap between pace and projection). Sort by wSGP instead so the most impactful pickup opportunities surface first.

- [ ] **Standings page visual redesign** — The standings page shows the data but doesn't make the important context visually obvious. Brainstorm display options to highlight: (1) the gap to the team directly ahead and behind in each category (e.g., "you have 93 HR, 1 behind 1st, 4 ahead of 3rd"), (2) which categories are closest to gaining or losing a roto point (the leverage concept we already compute), (3) overall roto point trajectory. The goal is to glance at standings and immediately see where the opportunities and threats are, not just a grid of numbers.

- [ ] **Multi-player trades and draft pick deals** — Extend trade recommender to support 2-for-2 swaps and draft pick trades. For draft picks: identify teams out of contention who might trade current-year players for next-year picks (contender/rebuilder dynamic). If Hart is competing, propose "my 2027 3rd-round pick for your closer" style deals to out-of-contention teams. Requires modeling draft pick value and team contention status from standings.

- [ ] **Keeper value in all decisions** — Factor multi-year keeper value into trade recommendations, waiver pickups, and draft strategy. Pull future-year projected stats (FanGraphs has age curves and multi-year projections) to estimate whether a player will be a keeper candidate next year. Young breakout players (e.g., a 23-year-old having a great season) should be valued higher because they'll be kept — trading them away costs future value, not just current-year production. Conversely, aging veterans on decline curves are worth less than their current stats suggest because they won't be kept. This affects: (1) trade recommender — don't trade away future keepers for a marginal current-year upgrade, and flag opponents' aging stars as buy-low targets; (2) waiver wire — prioritize young upside players over veteran rentals; (3) draft strategy — weight keeper-eligible players higher in later rounds. Requires modeling: age curves, keeper eligibility rules (league-specific), and a "keeper probability" score per player.


- [ ] **Harden data ingest: position and name matching** — The projection-to-roster matching pipeline has had repeated bugs around edge cases: Julio Rodriguez (accent encoding), Mason Miller (name collisions between hitter and pitcher), Shohei Ohtani (dual hitter/pitcher split). Audit `match_roster_to_projections`, name normalization, and position collision resolution end-to-end. Add targeted test cases for each known problem player and fix any remaining fragility.

- [ ] **Clean up "default for backwards compatibility" and other code smells** — Grep for remaining `# default for backwards compat` comments and similar shims. These were added during refactors and should be resolved — either the new behavior is correct (remove the comment and dead path) or the migration is incomplete (finish it).

- [ ] **Fix AVG tie resolution beyond 3 decimal places** — Roto standings ties in AVG are resolved by comparing to 3 decimal places, but actual ties at .001 granularity are common. Compare to full precision (or at least 5+ digits) to properly break ties. Check ERA and WHIP for the same issue.

- [ ] **Daily summary email** — Automated morning email with: last night's results for all roster players (hits, HRs, RBI, etc.), hot/cold streaks (rolling 7-day and 14-day performance vs projection), standings changes (gained/lost roto points overnight), recommended lineup changes for today, add/drop suggestions based on waiver wire, injury news affecting roster, and upcoming probable pitcher matchups. Could use the existing leverage and waiver modules as the analytical backbone.

- [ ] **Show player ranking on other teams' rosters** — The roster view for opponents doesn't show player VAR/ranking info. Add the same ranking context shown for the user's own roster so you can quickly assess trade targets and opponent strengths.

- [ ] **Round IP display to 1 decimal place** — IP values show excessive precision (e.g., 2.3333333). Round to 1 decimal place for display. Note: baseball IP is measured in thirds (6.1 = 6⅓ innings), so the display should ideally convert to baseball notation (6.1, 6.2, 7.0) rather than naive rounding.


- [ ] **Simplify suggested fixes** — Larger refactors identified by codebase-wide simplify review (2026-03-28):
  - `draft/projections.py` has a ~100-line duplicate `simulate_season()` that diverges from the canonical `simulation.py` version (no batched draws, no correlated variance). Likely dead code — investigate and remove or delegate.
  - Per-category SGP computation is repeated in `recommender._vona_leverage_weight`, `weighted_sgp.calculate_weighted_sgp`, and `waivers._category_sgp`. Extract a shared `compute_category_sgp_dict()` helper.
  - Strategy functions all take `(board, full_board, tracker, balance, config, team_filled, **kwargs)` — introduce a `DraftContext` dataclass to clean up ~15 function signatures and eliminate `kwargs.get()` boilerplate.
  - `simulate_draft.py` opponent ADP loop uses `iterrows()` per pick (~54K Series allocations per sim). Convert `adp_board` to a list of dicts before the draft loop.
  - `replacement.py:_get_eligible_players` recomputes position masks via `.apply(lambda)` on every pick in the recommender. Pre-compute a `{position: bool_array}` dict once and maintain incrementally.

# TODO — Postseason / Offseason

- [ ] **Post-draft Monte Carlo analysis** — Run Monte Carlo simulations on the actual completed draft results (real rosters from Yahoo) to assess win probability and category risk. Currently `simulate_draft.py --monte-carlo` only works on simulated drafts. Need a script or mode that takes real Yahoo rosters post-draft and feeds them through the shared `simulate_season` engine. Low priority since `summary.py` covers this during the season.

- [ ] **Evaluate pitcher VAR overstatement with actual draft outcomes** — The unified P replacement level makes every decent SP show high VAR. Compare projected VAR rankings vs actual roto contribution for pitchers drafted in rounds 1-5. Track how much SP production came from streaming vs drafted starters. If drafted SPs underperformed their VAR relative to drafted hitters, consider a `PITCHER_VAR_DISCOUNT` (~0.85) or separate SP/RP replacement levels. Also measure the gap between VAR-only and VONA rankings against actual outcomes to quantify how much correction VONA provides.

- [ ] **Calibrate closer replacement quality from waiver data** — The injury backfill model uses rough guesses for waiver-quality closer stats (~4.50 ERA, ~1.35 WHIP, ~5 SV). Track which closers were available on waivers throughout the season and their actual stats. Update `WAIVER_RP` in constants.py to match reality. Check whether the 60 IP baseline and 10 IP threshold for closer backfill produced reasonable adjustments.

- [ ] **Reconsider hitter backfill threshold: AB vs PA** — The 600 AB baseline penalizes durable high-walk hitters like Soto (projected 536 AB but ~640 PA). Should use PA (~650) instead since it better captures "played a full season" regardless of plate discipline. Check which durable hitters were incorrectly flagged as injury risks under the AB-based threshold, and compare AB-based vs PA-based backfill against actual outcomes.

- [ ] **Validate SGP denominators against actual roto outcomes** — Current defaults may undervalue high-AVG hitters and overvalue speed (AVG denominator 0.005 vs league-specific 0.0025, SB 8 vs 11.9). Track how high-AVG and high-SB players perform relative to their VAR rankings throughout the season. If high-AVG types consistently outperform their ranking and speed specialists underperform, adjust denominators (or blend with league-specific values) before next draft. Need more years of league standings data to separate signal from noise.

- [ ] **Add tests for draft dashboard delta protocol** — `compute_delta`, `read_delta`, and the Flask `/api/state?since=<version>` endpoint have zero test coverage. Test cases needed: delta with None old_state, changed vs unchanged field detection, version inclusion, multi-version-behind fallback. This is the real-time update mechanism — a bug here silently serves stale data during a live draft.

- [ ] **Fix league.yaml strategy/scoring_mode before next draft** — Config has `strategy: three_closers` and `scoring_mode: var`, but prior analysis (after fixing 4 bugs) validated `two_closers` + `vona` as correct. `three_closers` forces closers at rounds 5/9/13 where most have negative VAR. Update config before next year's draft simulations.

- [ ] **Refactor draft pipeline to use Player dataclass** — The draft pipeline (`board.py`, `replacement.py`, `var.py`, `rankings.py`) operates end-to-end on pandas DataFrames with string-keyed column access. Now that the in-season SGP functions accept `HitterStats`/`PitcherStats` directly, extend the same pattern to the draft pipeline: build `Player` objects from blended projections, compute SGP/VAR on the dataclass, and only convert to a DataFrame for the final ranked board output. This eliminates the split where in-season code uses typed dataclasses but draft code uses untyped Series.
