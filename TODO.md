# TODO — In-Season Enhancements

- [ ] **Validate ROS projections account for injuries** — Verify that FanGraphs ROS projections properly reduce stats for players on IL (e.g., Strider's ROS K projection should reflect missed time, not a full healthy season). If not, the injury management system should scale projections by expected games remaining.

- [ ] **Pitcher streaming tool** — Score free agent SPs by matchup quality to identify streamers (pick up a mediocre pitcher facing a terrible offense for one start, then drop). Builds on the matchup adjustment system.

- [ ] **Automate ROS projection download** — If the manual FanGraphs CSV download gets tedious, automate it. FanGraphs doesn't have a public API so this would require browser automation (Playwright) or finding an unofficial data source. Low priority unless the manual process is a pain.

- [ ] **Browser-based OAuth flow for season dashboard** — Add Yahoo OAuth redirect flow directly in the dashboard so it can be used from a phone without CLI re-auth. When the token is expired, redirect to Yahoo login, handle the callback, store the refreshed token. Required before remote hosting.

- [ ] **Multi-player trades and draft pick deals** — Extend trade recommender to support 2-for-2 swaps and draft pick trades. For draft picks: identify teams out of contention who might trade current-year players for next-year picks (contender/rebuilder dynamic). If Hart is competing, propose "my 2027 3rd-round pick for your closer" style deals to out-of-contention teams. Requires modeling draft pick value and team contention status from standings.

- [ ] **Keeper value in all decisions** — Factor multi-year keeper value into trade recommendations, waiver pickups, and draft strategy. Pull future-year projected stats (FanGraphs has age curves and multi-year projections) to estimate whether a player will be a keeper candidate next year. Young breakout players (e.g., a 23-year-old having a great season) should be valued higher because they'll be kept — trading them away costs future value, not just current-year production. Conversely, aging veterans on decline curves are worth less than their current stats suggest because they won't be kept. This affects: (1) trade recommender — don't trade away future keepers for a marginal current-year upgrade, and flag opponents' aging stars as buy-low targets; (2) waiver wire — prioritize young upside players over veteran rentals; (3) draft strategy — weight keeper-eligible players higher in later rounds. Requires modeling: age curves, keeper eligibility rules (league-specific), and a "keeper probability" score per player.

# TODO — Postseason / Offseason

- [ ] **Post-draft Monte Carlo analysis** — Run Monte Carlo simulations on the actual completed draft results (real rosters from Yahoo) to assess win probability and category risk. Currently `simulate_draft.py --monte-carlo` only works on simulated drafts. Need a script or mode that takes real Yahoo rosters post-draft and feeds them through the shared `simulate_season` engine. Low priority since `summary.py` covers this during the season.

- [ ] **Evaluate pitcher VAR overstatement with actual draft outcomes** — The unified P replacement level makes every decent SP show high VAR. Compare projected VAR rankings vs actual roto contribution for pitchers drafted in rounds 1-5. Track how much SP production came from streaming vs drafted starters. If drafted SPs underperformed their VAR relative to drafted hitters, consider a `PITCHER_VAR_DISCOUNT` (~0.85) or separate SP/RP replacement levels. Also measure the gap between VAR-only and VONA rankings against actual outcomes to quantify how much correction VONA provides.

- [ ] **Calibrate closer replacement quality from waiver data** — The injury backfill model uses rough guesses for waiver-quality closer stats (~4.50 ERA, ~1.35 WHIP, ~5 SV). Track which closers were available on waivers throughout the season and their actual stats. Update `WAIVER_RP` in constants.py to match reality. Check whether the 60 IP baseline and 10 IP threshold for closer backfill produced reasonable adjustments.

- [ ] **Reconsider hitter backfill threshold: AB vs PA** — The 600 AB baseline penalizes durable high-walk hitters like Soto (projected 536 AB but ~640 PA). Should use PA (~650) instead since it better captures "played a full season" regardless of plate discipline. Check which durable hitters were incorrectly flagged as injury risks under the AB-based threshold, and compare AB-based vs PA-based backfill against actual outcomes.
