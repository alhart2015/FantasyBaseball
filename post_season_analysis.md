# Post-Season Analysis

Items to investigate after the 2026 season using actual results.

## TODO

### 1. Evaluate pitcher VAR overstatement with actual draft outcomes

The unified P replacement level (91st pitcher, SGP ~7.97) is 2-4 SGP below hitter position replacement levels, making every decent SP show high VAR. The math is correct — pitcher depth really is greater — but VAR treats all value as fungible when it isn't. In roto, SP production is partially replaceable via in-season streaming (grab a hot starter, drop, repeat), while hitter value requires consistent AB and can't be replicated from waivers the same way.

**What to check after the season:**
- Compare projected VAR rankings vs actual roto contribution for pitchers drafted in rounds 1-5. Did they deliver value proportional to their draft cost, or did later-round SPs and waiver pickups produce similar results?
- Track how much SP production came from streaming vs drafted starters across the league.
- If drafted SPs underperformed their VAR relative to drafted hitters, consider one of:
  - A `PITCHER_VAR_DISCOUNT` factor (~0.85) applied to all pitcher VAR
  - Separate SP/RP replacement levels instead of the unified P pool
- VONA already compensates (Woo drops from VAR=6.15 to VONA=0.25), so measure the gap between VAR-only and VONA rankings against actual outcomes to quantify how much correction VONA provides and whether more is needed.

### 2. Calibrate closer replacement quality from waiver data

The injury backfill model uses waiver-quality replacement stats for SPs (~4.20 ERA, ~1.30 WHIP) and closers (~4.50 ERA, ~1.35 WHIP, ~5 SV). The closer values are rough guesses — actual waiver closers in a 10-team league may be better.

**What to check after the season:**
- Track which closers were available on waivers throughout the season and their actual stats
- Compute the average ERA, WHIP, and SV of closers picked up on waivers across the league
- Update `WAIVER_RP` replacement stats in constants.py to match reality
- Check whether the 60 IP baseline and 10 IP threshold for closer backfill produced reasonable VAR adjustments — did the model correctly identify which closers were injury risks vs full-season contributors?

### 3. Reconsider hitter backfill threshold: AB vs PA

The hitter backfill baseline is 600 AB, but this penalizes durable high-walk hitters like Juan Soto (projected 536 AB but ~640 PA — a full healthy season). Soto triggered backfill despite being one of the most durable players in baseball, because his walk rate converts PA to AB at a lower rate than average.

**What to investigate:**
- Should the baseline use PA (~650) instead of AB (~600)? PA better captures "this player played a full season" regardless of plate discipline
- If switching to PA: need to verify that PA is available in blended projections (it's in `HITTING_COUNTING_COLS` but may not propagate through all paths)
- Check which other durable hitters were incorrectly flagged as injury risks under the AB-based threshold
- Compare: which produces better VAR rankings against actual outcomes — AB-based or PA-based backfill?
