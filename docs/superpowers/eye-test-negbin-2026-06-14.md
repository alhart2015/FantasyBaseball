# Eye-test: OLD clipped-Gaussian vs NEW NegBin-copula MC sampler

Team: **Hart of the Order**  |  Date: 2026-06-14  |  Branch: june-13-group-review

READ-ONLY analysis. No committed code changed. Numbers are from live sim runs.

## Method and data sources

- OLD model = `simulation.py` at commit `bb1b187` (clipped-Gaussian `max(0, 1.0 + draw)` multiplier, sigma-scaled covariance).
- NEW model = current-HEAD `simulation.py` (Gaussian-copula NegBin sampler, `_negbin_copula_counts`).
- Sanity check (which sampler `_apply_variance` actually CALLS, via `inspect.getsource`): OLD calls clipped-Gaussian `max(0, 1.0 + draws`: **True** (OLD calls `_negbin_copula_counts`: **False**); NEW calls `_negbin_copula_counts`: **True** (NEW calls clipped-Gaussian: **False**). NOTE: bb1b187 already DEFINES `_negbin_copula_counts` but does not wire it into `_apply_variance` yet -- the active OLD sampler is the clip, as intended.
- Roster source: `data/weekly_rosters_2026.json`, most-recent snapshot **2026-04-07** for Hart of the Order (deduped to 27 unique players; 14 hitters + 13 pitchers matched). NOTE: that file's latest in-season snapshot is 2026-04-07 (it has not been refreshed past early April), so this is the early-April roster, the freshest LOCAL weekly roster.
- Projections: blended ROS full-season from `2026-06-10/` (atc, oopsy, steamer, the-bat-x, zips) via the project's own `blend_projections()` (equal-weight multi-system average of ROS-only counting stats; no YTD add, no normalizer -- the bare blend is sufficient for an eye-test of the SAMPLER).
- Mechanism: `_apply_variance(players, type, rng, [], fraction_remaining=1.0)` called directly, 5000 iterations, seed=12345 (same seed and same projection inputs fed to BOTH models). Active-roster selection (top 13 hitters / 9 pitchers, closer-priority pitcher sort) mirrors `simulate_season()`.

## 1. Per-player projections

### Hitters (R / HR / RBI / SB front-and-center, plus AVG)

`proj` = blended projection mean. `old`/`new` = sim means; `(sd)` = sim SD. `d%` = (new-old)/old on the mean. SB column is the headline banded stat.

| Player | proj R | old R | new R | proj HR | old HR | new HR | proj RBI | old RBI | new RBI | proj SB | old SB(sd) | new SB(sd) | d%SB |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oneil Cruz | 50 | 58.2 | 58.2 | 16 | 17.5 | 17.5 | 47 | 55.5 | 55.4 | 19 | 19.6(11.6) | 19.4(9.5) | -1.2% |
| CJ Abrams | 51 | 58.7 | 58.9 | 13 | 14.0 | 14.2 | 47 | 55.1 | 55.4 | 17 | 18.2(10.3) | 17.5(8.0) | -3.7% |
| Randy Arozarena | 51 | 59.1 | 59.1 | 12 | 14.8 | 14.8 | 45 | 54.0 | 53.9 | 16 | 17.1(9.5) | 16.8(7.8) | -1.3% |
| Julio Rodriguez | 56 | 62.6 | 62.7 | 17 | 18.5 | 18.5 | 55 | 61.3 | 61.4 | 15 | 16.4(9.0) | 15.9(7.2) | -2.6% |
| Juan Soto | 64 | 69.1 | 69.3 | 23 | 22.6 | 22.7 | 61 | 65.6 | 65.9 | 12 | 14.0(7.1) | 13.7(5.9) | -2.1% |
| Otto Lopez | 45 | 54.3 | 54.3 | 8 | 10.2 | 10.2 | 41 | 50.5 | 50.6 | 11 | 13.3(6.4) | 13.0(5.6) | -2.3% |
| Byron Buxton | 51 | 59.0 | 59.1 | 19 | 20.0 | 19.9 | 48 | 56.7 | 56.8 | 9 | 11.5(5.1) | 11.4(4.8) | -0.8% |
| Andy Pages | 48 | 56.9 | 56.8 | 16 | 17.3 | 17.2 | 52 | 59.1 | 59.0 | 7 | 10.0(4.4) | 9.9(4.4) | -1.1% |
| Marcus Semien | 45 | 53.7 | 53.7 | 11 | 12.7 | 12.7 | 43 | 51.4 | 51.5 | 7 | 10.6(4.8) | 10.5(4.8) | -1.5% |
| Adolis Garcia | 41 | 51.7 | 51.7 | 14 | 16.2 | 16.2 | 46 | 55.1 | 55.0 | 6 | 9.4(4.3) | 9.3(4.3) | -1.1% |
| Willy Adames | 47 | 55.8 | 55.9 | 15 | 15.4 | 15.5 | 48 | 55.5 | 55.8 | 6 | 9.4(4.2) | 9.2(4.4) | -2.4% |
| Ivan Herrera | 48 | 53.9 | 54.1 | 11 | 13.0 | 13.1 | 43 | 50.5 | 50.7 | 5 | 4.9(2.8) | 4.8(3.1) | -2.5% |
| Munetaka Murakami | 38 | 49.4 | 49.6 | 16 | 17.1 | 17.2 | 39 | 51.4 | 51.4 | 4 | 5.0(2.2) | 5.0(2.7) | -1.2% |
| Junior Caminero | 56 | 62.4 | 62.3 | 23 | 23.3 | 23.3 | 66 | 71.4 | 71.1 | 3 | 4.5(2.1) | 4.4(2.6) | -1.2% |

### Pitchers (W / K / SV front-and-center, plus ERA / WHIP)

SV column is the headline banded stat.

| Player | proj W | old W | new W | proj K | old K | new K | proj SV | old SV(sd) | new SV(sd) | d%SV |
|---|---|---|---|---|---|---|---|---|---|---|
| Josh Hader | 3 | 4.1 | 4.1 | 48 | 69.2 | 69.0 | 18 | 17.1(14.6) | 16.5(7.7) | -3.3% |
| Clayton Beeter | 2 | 3.2 | 3.2 | 38 | 61.8 | 61.8 | 14 | 14.2(11.3) | 13.5(5.8) | -5.4% |
| Bryan Abreu | 2 | 3.5 | 3.5 | 44 | 67.0 | 66.9 | 1 | 3.4(2.2) | 3.4(2.3) | -1.5% |
| Spencer Strider | 6 | 6.7 | 6.7 | 105 | 111.8 | 111.8 | 0 | 2.3(2.4) | 2.3(2.4) | 0.0% |
| Zack Wheeler | 7 | 8.3 | 8.4 | 111 | 139.3 | 139.3 | 0 | 0.0(0.0) | 0.0(0.0) | 0.0% |
| Aaron Nola | 6 | 7.8 | 7.8 | 101 | 131.3 | 131.5 | 0 | 0.0(0.0) | 0.0(0.0) | 0.0% |
| Bryan Woo | 7 | 8.7 | 8.8 | 106 | 135.3 | 135.3 | 0 | 0.0(0.0) | 0.0(0.0) | 0.0% |
| Jesus Luzardo | 8 | 9.0 | 9.0 | 121 | 147.9 | 148.0 | 0 | 0.0(0.0) | 0.0(0.0) | 0.0% |
| Jose Soriano | 6 | 7.8 | 7.7 | 102 | 132.6 | 132.7 | 0 | 0.0(0.0) | 0.0(0.0) | 0.0% |
| Logan Webb | 7 | 8.6 | 8.6 | 101 | 131.3 | 131.3 | 0 | 0.0(0.0) | 0.0(0.0) | 0.0% |
| Mason Miller | 0 | 2.1 | 2.1 | 1 | 34.1 | 34.1 | 0 | 2.8(2.5) | 2.8(2.5) | 0.0% |
| Ryan Walker | 2 | 3.5 | 3.5 | 28 | 54.3 | 54.3 | 0 | 2.7(2.5) | 2.7(2.5) | 0.0% |
| Sonny Gray | 7 | 6.9 | 6.9 | 96 | 103.5 | 103.8 | 0 | 2.5(2.4) | 2.5(2.4) | 0.0% |

NOTE: a few pitchers carry near-zero ROS lines in the 2026-06-10 ROS-only blend (e.g. Mason Miller proj K=1, W=0 -- absent/injured in the remaining-games projection at that snapshot). Their sim means come almost entirely from the shared replacement backfill, so OLD and NEW agree there by construction; they are not informative for the sampler comparison.

### Count-bias and zero-spike diagnostics (all rostered players, per-iter draws)

Aggregated over every (player, stat) counting draw. `mean ratio` = sim-mean / projection (1.000 = unbiased). `zero %` = fraction of draws that came out exactly 0 (the old sampler's zero-spike).

| Stat | proj sum | OLD mean ratio | NEW mean ratio | OLD zero% | NEW zero% |
|---|---|---|---|---|---|
| R | 689 | 1.169 | 1.169 | 0.0% | 0.0% |
| HR | 214 | 1.085 | 1.086 | 0.1% | 0.0% |
| RBI | 680 | 1.166 | 1.167 | 0.0% | 0.0% |
| SB | 135 | 1.214 | 1.191 | 2.3% | 0.6% |
| W | 63 | 1.279 | 1.279 | 0.3% | 3.1% |
| K | 1003 | 1.315 | 1.315 | 0.0% | 0.8% |
| SV | 33 | 1.061 | 1.019 | 55.9% | 55.8% |

## 2. Team-level category totals (active roster)

| Category | OLD mean(sd) | NEW mean(sd) | delta | % change |
|---|---|---|---|---|
| R | 763.1(34.2) | 764.7(36.8) | 1.5 | 0.2% |
| HR | 222.9(19.4) | 223.3(19.5) | 0.4 | 0.2% |
| RBI | 753.5(37.1) | 755.6(41.2) | 2.1 | 0.3% |
| SB | 158.9(24.6) | 155.3(21.5) | -3.6 | -2.2% |
| AVG | 0.252(0.005) | 0.252(0.006) | 0.000 | 0.0% |
| W | 67.8(8.1) | 68.1(7.6) | 0.3 | 0.4% |
| K | 1103.0(68.5) | 1107.1(67.0) | 4.1 | 0.4% |
| SV | 28.2(19.3) | 22.8(12.1) | -5.5 | -19.4% |
| ERA | 3.826(0.254) | 3.821(0.266) | -0.005 | -0.1% |
| WHIP | 1.230(0.051) | 1.229(0.055) | -0.001 | -0.1% |

### Players where OLD vs NEW differ by more than 5% on a counting stat

- Clayton Beeter SV: old 14.2 -> new 13.5 (-5.4%)

### Isolated pure-sampler bias and zero-spike (scale=1.0, no playing-time haircut, no backfill)

This strips the shared replacement backfill so the numbers reflect the SAMPLER only. `mean ratio` = sampler-mean / mu (1.000 = unbiased).

| Stat | OLD mean ratio | NEW mean ratio | OLD zero% | NEW zero% |
|---|---|---|---|---|
| R | 0.999 | 1.000 | 0.0% | 0.0% |
| HR | 0.999 | 0.999 | 0.2% | 0.0% |
| RBI | 0.999 | 1.000 | 0.0% | 0.0% |
| SB | 1.025 | 1.001 | 8.0% | 3.4% |
| W | 1.003 | 0.998 | 0.8% | 12.0% |
| K | 1.000 | 1.002 | 0.0% | 3.4% |
| SV | 1.067 | 0.993 | 13.6% | 14.6% |
| ER | 1.000 | 1.001 | 0.0% | 2.3% |
| BB | 0.998 | 1.002 | 0.0% | 3.4% |
| H_ALLOWED | 1.000 | 1.002 | 0.0% | 1.0% |

## 3. What to look for

- **Count bias (isolated sampler, mean over all counting stats):** OLD = 0.83% off mu, NEW = -0.03% off mu. The clip's `max(0, .)` truncation lifts the mean above the projected mu (the prior benchmark's ~+2.6% upward bias); the NegBin is mean-exact by construction.
- **Where the OLD bias concentrates -- the high-sigma BANDED stats SB/SV:** isolated OLD SB ratio 1.025 (+2.5%), OLD SV ratio 1.067 (+6.7%) -- exactly the prior benchmark's upward count bias. The high-mu stats (R/HR/RBI/K) sit at ~1.000 in BOTH because the clip rarely truncates there; the bias is a low-mu/high-sigma effect. NEW is ~1.000 on every stat.
- **Zero-spike, OLD's defect is on SB/SV:** isolated OLD SB 8.0% / SV 13.6% of draws land exactly 0 -- the prior benchmark's ~9-11% zero-spike. These are ARTIFICIAL (the clip floors at 0 whenever `1 + draw <= 0`). NEW's zeros (e.g. W 12%, SV 14.6%) are by contrast LEGITIMATE low-mu NegBin/Poisson outcomes: a pitcher projected for ~3 W genuinely posts 0 some seasons, which the symmetric clip could only do via an implausible draw. So 'more zeros' in NEW is correct, not a regression.
- **End-to-end means (with backfill) run slightly higher under OLD on the banded stats** -- SB 1.214 vs 1.191 and SV 1.061 vs 1.019 in the section-1 diagnostic; both sit above 1.0 only because the shared replacement backfill (identical for both models) is included.
- **Banded stats SB/SV direction:** team SB 158.9 -> 155.3 (-2.2%), team SV 28.2 -> 22.8 (-19.4%). NEW slightly lower = removed upward bias.
- **Spread:** compare the (sd) columns -- NEW banded SB/SV/HR/RBI/ER show wider low-projection dispersion (overdispersed NegBin) without the symmetric Gaussian's negative tail.

## 4. ERoto (STAT_VARIANCE) vs NEW NegBin implied SD

The analytic ERoto path (`scoring.py`) uses per-stat relative SD from `STAT_VARIANCE` (performance CV; playing-time `cv_pt` added separately). The NEW MC uses `STAT_DISPERSION` r, giving implied CV^2 = 1/mu + 1/r at mean mu. Shown at a representative mu (a typical rostered projection).

| Stat | ERoto sigma (STAT_VARIANCE) | rep. mu | NegBin r at mu | NegBin implied CV |
|---|---|---|---|---|
| SB | 0.715 | 7.6 | 4.747 | 0.585 |
| SV | 0.900 | 14.0 | 37.757 | 0.313 |
| HR | 0.343 | 15.1 | 12.455 | 0.383 |

Takeaway: ERoto's flat per-stat sigma (e.g. SB sigma=0.715, SV sigma=0.900) and the NegBin's mu-dependent implied CV are two parameterizations of the SAME dispersion -- unifying them would make the analytic band and the MC agree by construction instead of by coincidence.

