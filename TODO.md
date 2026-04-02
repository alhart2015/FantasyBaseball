# TODO — In-Season Enhancements

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

- [ ] **Player identity: use fg_id as primary key instead of name** — Rankings, roster matching, and other lookups currently key by `normalized_name::player_type`. This breaks when two players share a name AND type (e.g., two pitchers named Mason Miller — the real closer gets overwritten by the fringe prospect). FanGraphs `fg_id` is unique and already available in projection tables. Refactor to use `fg_id` as the primary identifier throughout the pipeline: projection matching, ranking lookups, game log joins, and cache keys. This also fixes edge cases with accent mismatches (José vs Jose) and name format differences between Yahoo and FanGraphs.

- [ ] **Extract rate stat utility functions** — AVG (`h/ab`), ERA (`er*9/ip`), and WHIP (`(bb+h_allowed)/ip`) are computed inline in 7+ places: `projections.py` blend, `pace.py`, `board.py` backfill, `rankings.py`, `replacement.py`, `draft/projections.py`, and now `models/player.py`. Extract `calculate_avg()`, `calculate_era()`, `calculate_whip()` into `utils/rate_stats.py` and replace all inline computations.

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
