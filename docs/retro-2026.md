# 2026 season retrospective

Hart of the Order won the league: 92.5 points, 13 ahead of Hello Peanuts! (79.5).
This is the look-back asked for in #381. Each section links to its sub-issue.

**Read every number with the sample size in mind.** One season, ten teams. Where a
finding could plausibly be noise, it says so.

## Data used

| What | Source | Check |
|---|---|---|
| Final standings | `data/manual/standings.yaml` (2026-09-28) | Stats and Points tabs agree in every category |
| Player actuals | MLB per-game logs (`game_logs:2026:*`) | Match `data/stats/*-2026.csv` exactly: 534/534 hitters, 649/649 pitchers |
| Preseason projections | `data/projections/2026/*.csv`, equal-weight 5-system blend | Same `blend_projections` the app uses |
| In-season projections | `data/projections/2026/rest_of_season/<date>/` (14 vintages) | Nearest vintage on or before each date; none between 03-30 and 06-04 |
| Moves | Yahoo API ledger to 07-26, then `data/manual/transactions-2026-07-22-to-09-22.csv` | Replays to all 7 manual roster snapshots with 0 mismatches |
| Ownership | Moves, capped at the first saved roster that no longer lists the player | Needed because the API ledger has no trades |

Actuals come from the game logs, not the season stat files, because the stat files drop
anyone under 50 PA / 10 IP -- exactly the injured players projections miss on.

## Part 1: projection accuracy (#235)

**Rate stats were accurate; counting stats ran 10-18% hot for draftable players.**

Draft pool = top 300 by blended ADP (175 hitters, 125 pitchers):

| Stat | Bias (actual vs projected) | | Stat | Bias |
|---|---|---|---|---|
| PA | -8% | | IP | -15% |
| R | -11% | | W | -15% |
| HR | -13% | | K | -15% |
| RBI | -14% | | SV | -26% |
| SB | -18% | | ERA / WHIP | -0.5% / -0.2% |
| AVG | -2% | | | |

Two causes:

1. **Playing time.** A quarter of drafted hitters got under 75% of their projected PA;
   a third of drafted pitchers got under 75% of their projected IP.
2. **Projections are too spread out.** It was not a low-offense year: league HR/PA, R/PA
   and AVG were normal for 2022-2026. Among regulars, the top fifth by projected rate
   came in about 11% short on HR/PA and RBI/PA; the bottom fifth came in slightly over.
   The slope of actual rate on projected rate was 0.61-0.88 for R/HR/RBI/SB (1.0 =
   calibrated). **The same direction showed up in 2025** (steamer+zips, slope 0.70-0.96),
   so this is not a one-year fluke.

**No system was best across the board.** On the same draft pool, most stats' mean
absolute errors sat within about 10% across systems. The bigger gaps: ZiPS and OOPSY had
about 17% more RBI error than Steamer, OOPSY about 19% more SB error than Steamer, and
Steamer and OOPSY about 12% more K error than THE BAT X. The blend was best or
near-best in most stats. One season is not enough to justify changing the equal weights.

**Team level.** We were projected 1st in every weekly snapshot from 03-24 on. The
preseason projection missed teams' final points by 16.7 on average (rank correlation
0.47); by June the miss was about 4-6 (correlation 0.93 or better).

## Part 2: were the uncertainty ranges honest? (#383)

Measured as SD(z), where z = (actual - projected) / model SD. 1.0 = honest, above 1 =
ranges too narrow. Errors are centered on the league-wide miss each week, because roto
only cares about gaps between teams.

**Preseason ranges were honest: 0.95** (69 team-category cells, 03-30 vintage).

**In-season ranges were about 2x too narrow: 2.15 overall, 3-4 in September.**

| Vintage | Share of season left | Today's model | Without the extra sqrt shrink |
|---|---|---|---|
| 06-08 | 0.61 | 1.70 | 1.32 |
| 07-14 | 0.41 | 1.53 | 0.98 |
| 08-25 | 0.18 | 1.48 | 0.63 |
| 09-08 | 0.11 | 3.09 | 1.02 |
| 09-14 | 0.08 | 3.97 | 1.09 |
| **All in-season** | | **2.15** | **0.94** |

Cause: in-season, each player's variance is priced on his **rest-of-season** mean, which
already shrinks as the season goes on, and then `build_team_sds` multiplies the team SD
by `sqrt(fraction_remaining)` again (`refresh_pipeline.py:1129`, `scoring.py:1406`). Removing
the second shrink takes the in-season number from 2.15 to 0.94. Filed as #388.

The 2022-2025 SD backtest could not catch this: it checks full-season ranges, where the
fraction left is 1 and the extra shrink does nothing.

Other findings:

- **Saves are too narrow either way** (1.48 even without the extra shrink). Closer jobs
  change hands more than the role model allows.
- **The frozen preseason Monte Carlo's points ranges caught only 2 of 10 teams** in their
  p10-p90 band (3 of 10 in the "with management" version); expected about 8. Final
  scores ran 26-92.5, much wider than the simulated medians (45-72). That clashes with the
  honest preseason category ranges above, so part of it is probably in-season
  management -- waivers alone moved teams by up to 36 SGP (Part 3). Open question.

Caveats: 10 teams; weekly cells for the same team are strongly correlated; realized
totals include each owner's later moves, which the model does not try to predict.

## Part 3: waiver moves (#384)

Scored as **swap value**: SGP the added player actually produced while the team owned
him, minus SGP the dropped player actually produced over the same days. No projections
are involved. A swap within +/-0.05 SGP counts as even.

**Us: +21.6 SGP over 41 adds (16 won, 11 lost, 14 even). 3rd in the league.**

| Team | Adds | Net swap SGP | Won | Lost | Even | Final points |
|---|---|---|---|---|---|---|
| Hello Peanuts! | 48 | +36.5 | 29 | 12 | 7 | 79.5 |
| Jon's Underdogs | 81 | +26.1 | 38 | 25 | 18 | 74.0 |
| **Hart of the Order** | **41** | **+21.6** | **16** | **11** | **14** | **92.5** |
| SkeleThor | 23 | +20.3 | 12 | 4 | 7 | 59.5 |
| Springfield Isotopes | 25 | +19.8 | 13 | 10 | 2 | 69.0 |
| Work in Progress | 14 | +15.0 | 9 | 3 | 2 | 43.0 |
| Boston Estrellas | 75 | +11.5 | 28 | 22 | 25 | 33.5 |
| Tortured Baseball Department | 6 | +10.8 | 4 | 0 | 2 | 31.5 |
| Send in the Cavalli | 5 | +7.5 | 4 | 1 | 0 | 26.0 |
| Spacemen | 41 | +2.8 | 17 | 9 | 15 | 41.5 |

Three adds are left out because the name could not be tied to one MLB player (a "Y. Diaz"
and a "Max Muncy") or the player never appeared in a game (Jordan Westburg).

Our best moves: Ceddanne Rafaela for Trevor Story (+7.0), Jose Soriano (+6.0, no drop),
Otto Lopez for Matt McLain (+5.9), Dominic Canzone (+2.5), Emilio Pagan for Yoendrys
Gomez (+2.5). Worst: Jake Burger for Munetaka Murakami (-2.1), Wyatt Langford for Cole
Carrigg (-1.2), Vinnie Pasquantino for Jake Burger (-1.1), Nathan Eovaldi for Payton
Tolle (-1.0). The 08-05 trade (Kyle Tucker + Rnd 5 for Julio Rodriguez + Rnd 12) came
out about even on 2026 stats (+0.3).

**The title came from the roster we started with more than from waivers.** We were
projected 1st from March on; waivers added a solid but third-best +21.6.

Caveat: "while owned" counts bench days on both sides, so it is an upper bound on what
the swap added to the standings.

## Part 4: missed opportunities (#385)

**Free agents.** At each saved roster date, every unowned MLB player was compared, over
the next 4 weeks, to our worst active player of the same type.

- Almost every week, some free agent beat our worst player by more than 1 SGP. Nearly all
  of those were hindsight: the projection at the time did not prefer them.
- When the projection **clearly** preferred a free agent (by more than 0.5 SGP over 4
  weeks), the free agent did better 87% of the time (20 of 23 cases). Those are the real
  misses: 7 players, 10 player-weeks.
- **Almost all of them say the same thing: drop Yoendrys Gomez sooner.** Aaron Nola,
  Will Warren, Sean Manaea, Dustin May, Griffin Jax and Ian Seymour were each clearly
  better projected than Gomez between 08-22 and 09-08. We dropped him 09-14.

**The waiver tool.** Six of its seven reports survive (the 09-22 one was overwritten; see
#385).

- Its #1 recommendation helped in 5 of 6 reports, but by little (+0.1 to +0.8 SGP).
  Gomez was the #1 drop in 5 of the 6. We followed 2 of 6.
- The rest of its list was noise: across all 46 recommendations only 37% helped,
  the average was -0.3 SGP, and projected gain barely predicted real gain (correlation
  0.16).

## What to change for 2027

1. **Fix the in-season SD double shrink** (#388). In-season ranges drive leverage, waiver
   scoring and trade checks; today they are about 2x too confident.
2. **Shrink projections toward the mean before valuing players.** Two seasons in a row, the
   top tier underdelivered on HR/RBI/R relative to the bottom. Needs its own backtest
   before changing draft values (#389).
3. **Trust the waiver tool's #1 pick; ignore the rest of its list.** When the projection
   clearly says a free agent beats your worst player, it was right 87% of the time.
4. **Keep equal blend weights for now.** No system was best across the board; recheck
   the RBI, SB and K gaps next year before down-weighting anyone.
5. **Save the ROS projection every week, all season.** The April-May gap forced Part 4
   to judge spring free agents with the 03-30 projection.
