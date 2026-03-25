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
