# Trajectory board: prorated vs ROS-anchored (#348)

**A SANITY CHECK, NOT EVIDENCE.** It says which players the anchor moved and by how much.
It says nothing about whether the new number is closer to the truth. Per Hart's call
(2026-08-08) a historical backtest is explicitly not a gate on this work, and it is not
tractable anyway: `data/projections/*/rest_of_season/` exists for 2026 only and prod
Upstash overwrites ROS daily, so there is no historical ROS to replay. The measurement
that WOULD be evidence -- players who missed 20+ games, with a standard error -- needs
the synthetic Marcel-style reconstruction the issue describes.

Run 2026-08-09 against `hitter_pt_panel_2000_2026.csv` / `pitcher_pt_panel_2000_2026.csv`
and the `2026-07-21` ROS snapshot. Horizons 1-3, `min_sgp` 0.0, VAR scale.

## Method

Two boards, same panel, same sweep, differing only in the base-season anchor:

- **before** -- `era_normalize(load_scored_panel(include_partial=True))`, then the retired
  proration: `sgp / season_elapsed_fraction` on partial rows, one league-wide fraction
  (0.698 on this panel).
- **after** -- `ros_anchor.load_anchored_panels`: season-to-date plus the blended
  rest-of-season line, combined on the STAT LINE, injected before `era_normalize`, with
  the 2026 era factor taken from the actual rows.

Compared on `now` (the anchor itself) and on 3-year total VAR.

## What moved

1,169 players on both boards. **90 are on the anchored board only** -- they paced
negative and were cut by `min_sgp`, and the projection's regression toward the mean lifts
them back over 0.0. None went the other way; no player was dropped for want of an ROS
row (`excluded.no_ros_projection` is 0, and coverage on this snapshot is 4747/4748 hitter
rows and 749/749 pitcher rows).

### Biggest risers, 3-year VAR

| player | pool | games | now before | now after | dVAR |
|---|---|---|---|---|---|
| Ronald Acuna Jr. | hitter | 60 | -1.57 | 2.64 | +6.71 |
| Hunter Greene | pitcher | 5 | -7.37 | -2.89 | +6.40 |
| Francisco Lindor | hitter | 56 | -4.14 | 0.11 | +5.87 |
| Luis Robert Jr. | hitter | 35 | -7.78 | -4.46 | +5.28 |
| Colt Emerson | hitter | 59 | -5.57 | -3.44 | +4.89 |
| Wyatt Langford | hitter | 60 | -1.13 | 1.82 | +4.63 |
| Hunter Brown | pitcher | 10 | -4.35 | -1.19 | +4.55 |
| Mookie Betts | hitter | 74 | -2.11 | 0.85 | +4.49 |
| Vladimir Guerrero Jr. | hitter | 106 | 1.06 | 3.85 | +4.10 |

**This is the population the change was opened about.** A full-time player has ~113 games
at 69.8% elapsed; the risers are at 5, 10, 35, 56, 60, 74. Under proration their depressed
counting stats were divided by the LEAGUE's elapsed fraction, so the six weeks they missed
were baked in as their rate and they were priced as worse players rather than as healthy
players who lost time. Acuna at 60 games moves from below replacement to above it.

### Biggest fallers, 3-year VAR

The largest fall is -2.63 (Keider Montero), against +6.71 at the top. They are
over-performers being regressed: Jacob Misiorowski 13.50 -> 12.29, Cam Schlittler
11.02 -> 9.49, Liam Hicks 4.81 -> 3.69. A projection does not extend a two-thirds-season
hot streak at the pace it was run.

## What this does NOT show

The pooled slice does not separate cleanly, and it should not be quoted as though it did.

| slice (hitters, 30+ games) | n | mean d_now |
|---|---|---|
| 30-60 games | 128 | +0.53 |
| 60-80 games | 88 | +0.62 |
| 80-100 games | 91 | +0.68 |
| 100-113 games | 116 | +0.92 |

That is the OPPOSITE ordering from the injury story, and holding production roughly fixed
(`now_before` in -3..+3) does not fix it either -- the buckets run +1.41, +1.00, +0.66,
+1.04, which is not monotone at n=7/25/50/64. Two confounds are mixed in and this design
cannot pull them apart:

1. **Regression to the mean**, which moves every player and is largest in the middle of
   the distribution, not at the injured end. Bucketed by `now_before` quintile the mean
   d_now runs +0.46 / +0.42 / +0.82 / +0.99 / +0.78 -- a regression signature, not a
   playing-time one.
2. **A 19-day-old snapshot.** The panel's YTD runs through ~today and the ROS remainder
   was projected from 2026-07-21, so `YTD + ROS` double-counts those team games for
   everyone. Mean d_now over the whole board is +0.43, and a full-time player carries the
   most of that because his remainder is the largest. Hart's call was to take the blend at
   face value -- the snapshot is stale only because of an unrelated Yahoo auth problem --
   but it does mean this table's LEVELS are inflated and its ordering by playing time is
   not readable.

The named movers are the finding. The aggregate is not.
