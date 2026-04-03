# League-Specific SGP Denominators

**Computed:** 2026-04-03
**Data:** 3 seasons of final standings (2023-2025), 10 teams each
**Sample:** 27 inter-team gaps per category (9 per year x 3 years)

## Method

For each category, sort teams by that stat, compute the average gap between adjacent teams, then average across years. This gives the empirical "how much of this stat separates one standings place."

## Results

| Cat  | Default | League | Diff   | Notes                          |
|------|---------|--------|--------|--------------------------------|
| R    | 20.0    | 21.0   | +5%    | Close                          |
| HR   | 9.0     | 7.3    | -19%   | HR tighter than typical        |
| RBI  | 20.0    | 21.3   | +6%    | Close                          |
| SB   | 8.0     | 11.9   | +49%   | SB spread wider, less valuable |
| AVG  | 0.005   | 0.0025 | -50%   | AVG very tight, more valuable  |
| W    | 3.0     | 2.4    | -19%   | Pitching wins tight            |
| K    | 30.0    | 33.3   | +11%   | Close                          |
| SV   | 7.0     | 8.1    | +16%   | Slightly less valuable         |
| ERA  | 0.15    | 0.121  | -20%   | ERA tighter than typical       |
| WHIP | 0.015   | 0.018  | +19%   | WHIP slightly less valuable    |

## Per-Year Breakdown

| Cat  | 2023  | 2024  | 2025  |
|------|-------|-------|-------|
| R    | 19.1  | 23.1  | 20.9  |
| HR   | 5.3   | 8.0   | 8.4   |
| RBI  | 18.7  | 29.0  | 16.2  |
| SB   | 12.7  | 15.6  | 7.6   |
| AVG  | .0027 | .0020 | .0029 |
| W    | 3.3   | 1.7   | 2.3   |
| K    | 46.0  | 19.1  | 34.9  |
| SV   | 7.4   | 7.8   | 9.1   |
| ERA  | .174  | .083  | .104  |
| WHIP | .023  | .013  | .017  |

## Reliability Assessment

**Not recommended for direct use yet.** Year-to-year variance is too high for 27 data points:

- K ranges from 19.1 to 46.0 (2.4x)
- RBI ranges from 16.2 to 29.0 (1.8x)
- W ranges from 1.7 to 3.3 (2.0x)
- ERA ranges from 0.08 to 0.17 (2.1x)

Single outlier teams (someone punting a category, a team running away with it) create gaps that dominate the small sample. With 9 gaps per year, one outlier is ~11% of the sample.

**Directional takeaways** (probably real):
- AVG is genuinely tighter in this league than defaults assume. High-AVG hitters are undervalued.
- SB is genuinely more spread out. Speed-first players are overvalued by the defaults.
- Pitching (W, ERA) is tighter than defaults. Pitching is undervalued.

**Possible future action:**
- Accumulate more years before switching (5+ years would give ~45 gaps per category)
- Consider a blended approach: weight league-specific data with defaults (e.g., 50/50)
- Could also use the `sgp_denominators` override in `league.yaml` to experiment with specific categories (e.g., just override SB and AVG)
