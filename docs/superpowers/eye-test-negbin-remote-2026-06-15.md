# Eye-test (REMOTE PROD): OLD clipped-Gaussian vs NEW NegBin-copula MC

Team: **Hart of the Order**  |  Date: 2026-06-15  |  Branch: june-13-group-review

READ-ONLY analysis against **prod Upstash** (the source of truth). No writes,
no `.set()`, no `write_cache`, no refresh pipeline, no `_push_to_prod`,
`RENDER` never set. Prod was read via `build_explicit_upstash_kv()` with the
`_meta` envelope hand-unwrapped (same logic as `season_data._read_enveloped`)
so reads never routed through the off-Render `get_kv()` SQLite trap. All
numbers below are from live sim runs on the freshly-read prod blobs.

## Method and data sources

- **OLD model** = `simulation.py` at commit `bb1b187` (clipped-Gaussian
  `max(0, 1.0 + draw)` multiplier), loaded from the git blob via
  `git show bb1b187:...` and `exec`'d into a `sim_old` module.
- **NEW model** = current-HEAD `simulation.py` (Gaussian-copula NegBin sampler,
  `_negbin_copula_counts`).
- **Sampler sanity check** (the active wiring inside each model's
  `_apply_variance`): OLD uses clipped-Gaussian `max(0, 1.0 + draws`: **True**
  / OLD wires `counts = _negbin_copula_counts`: **False**. NEW wires
  `_negbin_copula_counts`: **True** / NEW uses clipped-Gaussian: **False**.
- **Inputs (all from prod Upstash, read-only):**
  - `cache:roster` (Hart's matched roster), `_written_at` 2026-06-14T10:01:39Z,
    `_sha` 399090c, `_job` refresh. 25 players, projections already attached
    (`full_season_projection` + `rest_of_season` nested per player).
  - `cache:opp_rosters` (9 opponents), `_written_at` 2026-06-14T10:01:15Z.
  - `cache:standings` (actual YTD), `effective_date` **2026-06-16**,
    `_written_at` 2026-06-14T10:00:45Z. Hart YTD: R473 HR145 RBI432 SB97
    AVG.269 W40 K645 ERA3.75 WHIP1.17 **SV24**; extras IP=616.
  - `cache:ros_projections` / `cache:full_season_projections`,
    `_ros_snapshot_date` **2026-06-10** (the Mason-Miller correctness source).
- **fraction_remaining = 0.5730** (season 2026-03-27..2026-09-28, today
  2026-06-14), via the pipeline's own `compute_fraction_remaining`.
- **Active slots:** h_slots=12, p_slots=9 (from `config.roster_slots`, same
  `BENCH_SLOTS | {"P"}` exclusion the pipeline uses).
- **Two faithful paths run for BOTH models with the SAME inputs + seed:**
  1. `run_ros_monte_carlo(...)` for the full 10-team league (seed=42, 1000
     iters) -- the exact production MC entry point. READ-only (it does not
     cache; only the pipeline's surrounding code writes).
  2. `simulate_remaining_season(...)` repeated for the per-player /
     team-category tables (Hart full-roster aggregation: 5000 iters seed=12345
     for per-player + full-season team totals; 1000 iters seed=42 for the
     YTD-blended final totals).

## 1. Mason Miller correctness check (the key fix vs the LOCAL eye-test)

The prior LOCAL eye-test ran on an early-April weekly-roster snapshot whose
"Mason Miller" matched the **wrong** player -- a KCR scrub with proj SV=0,
so SV was invisible/uninformative. On **prod**, Hart's roster entry resolves
to the **correct SDP closer**:

| Source | name | team | fg_id | mlbam | ROS SV | season SV | season IP | season K |
|---|---|---|---|---|---|---|---|---|
| Hart roster (prod) | Mason Miller | SDP | (player_id 12505) | -- | 19.05 | 37.05 | 66.0 | 117 |
| ros_projections | Mason Miller | SDP | 31757 | 695243 | **19.05** | -- | 37.4 | 61 |
| full_season_projections | Mason Miller | SDP | 31757 | 695243 | -- | **37.05** | 66.0 | 117 |
| (the wrong one) ros/full | Mason Miller | KCR | sa3023658 | 692223 | 0.0 | 0.0 | ~1.8 | ~1 |

Confirmed: the SDP closer (ROS SV ~19 / season ~37) is present in the prod
projection tables, and **Hart's roster entry resolves to him**, not the KCR
scrub. This is the load-bearing correctness check for the remote re-run.

## 2. Hart of the Order CURRENT roster (prod `cache:roster`)

As-of: roster `_written_at` 2026-06-14; standings `effective_date` 2026-06-16.
All 25 players matched a projection (every entry carries a non-empty
`full_season_projection`). Slot is Yahoo `selected_position`.

| Player | type | team | slot | season R/W | season HR/K | season SB/SV |
|---|---|---|---|---|---|---|
| Ivan Herrera | hitter | STL | C | 94 | 20 | 8 |
| Freddie Freeman | hitter | LAD | 1B | 91 | 24 | 6 |
| Otto Lopez | hitter | MIA | 2B | 88 | 13 | 24 |
| Junior Caminero | hitter | TBR | 3B | 97 | 37 | 3 |
| CJ Abrams | hitter | WSN | SS | 96 | 27 | 27 |
| Willy Adames | hitter | SFG | IF | 81 | 26 | 7 |
| Julio Rodriguez | hitter | SEA | OF | 95 | 30 | 24 |
| Randy Arozarena | hitter | SEA | OF | 98 | 19 | 35 |
| Juan Soto | hitter | NYM | OF | 95 | 38 | 18 |
| Byron Buxton | hitter | MIN | OF | 98 | 41 | 15 |
| Ceddanne Rafaela | hitter | BOS | UTIL | 74 | 17 | 19 |
| Vinnie Pasquantino | hitter | KCR | UTIL | 72 | 20 | 4 |
| Oneil Cruz | hitter | PIT | IL | 95 | 30 | 40 |
| Bryan Woo | pitcher | SEA | P | 12 | 185 | 0 |
| Mason Miller | pitcher | SDP | P | 3 | 117 | **37** |
| Jesus Luzardo | pitcher | PHI | P | 13 | 209 | 0 |
| Nathan Eovaldi | pitcher | TEX | P | 11 | 174 | 0 |
| Sonny Gray | pitcher | BOS | P | 15 | 147 | 0 |
| Spencer Strider | pitcher | ATL | P | 10 | 151 | 0 |
| Logan Webb | pitcher | SFG | P | 10 | 159 | 0 |
| Josh Hader | pitcher | HOU | P | 4 | 55 | **20** |
| Zack Wheeler | pitcher | PHI | P | 12 | 164 | 0 |
| Jose Soriano | pitcher | LAA | BN | 14 | 194 | 0 |
| Edwin Diaz | pitcher | LAD | BN | 2 | 31 | 4 |
| Blake Snell | pitcher | LAD | IL | 4 | 64 | 0 |

Match rate: **25/25** players matched a projection.

## 3. Before/after MC -- `run_ros_monte_carlo` (production path, seed=42, 1000 iters)

Roto win-points / standings outcome for Hart (full 10-team league, prod
rosters + prod actual standings YTD blend):

| Metric | OLD (clipped-Gaussian) | NEW (NegBin copula) |
|---|---|---|
| median roto pts | 77.0 | **78.0** |
| p10 / p90 | 67 / 87 | 67 / 87 |
| first_pct (win league) | 52.1% | **54.8%** |
| top3_pct | 90.5% | 93.2% |

Category risk (Hart), the two banded stats:

| Cat | OLD median (p10/p90, top3%/bot3%) | NEW median (p10/p90, top3%/bot3%) |
|---|---|---|
| SV | 5.0 (2/9, 21.5% / 38.7%) | **4.5** (2/8, 13.4% / 33.6%) |
| SB | 10.0 (7/10, 87.4% / 0.4%) | 10.0 (8/10, 94.8% / 0.0%) |

Read: NEW nudges the overall win read **up** (median 77->78, first 52->55%)
while pulling **SV risk down** -- the NegBin removes the clip's upward SV
inflation, so Hart's SV category looks weaker (median 5->4.5 roto pts, top-3
odds 21.5%->13.4%) but his floor on the other categories tightens enough that
the net standings position improves. SB stays pinned near the top in both.

## 4. Team-level OLD vs NEW category totals (Hart)

Two views. **(a) Full-season sim totals** (`mean(sd)`, what the sampler
produces before the YTD subtract/add): the cleanest sampler comparison.
**(b) YTD-blended final totals** (the production blend: actual YTD + simulated
remainder): directly comparable to the prior local eye-test's team table.

### (a) Full-season sim totals (5000 iters, seed=12345)

| Cat | OLD mean(sd) | NEW mean(sd) | delta | % |
|---|---|---|---|---|
| R | 1119.6(43.6) | 1119.9(42.4) | 0.3 | 0.0% |
| HR | 331.6(26.7) | 330.5(22.3) | -1.2 | -0.4% |
| RBI | 1052.9(45.6) | 1052.4(45.1) | -0.5 | -0.1% |
| SB | 226.6(39.5) | 224.2(29.7) | -2.4 | -1.0% |
| AVG | 0.267(0.006) | 0.267(0.006) | 0.000 | 0.0% |
| W | 102.1(11.4) | 101.8(10.8) | -0.3 | -0.3% |
| K | 1541.3(72.4) | 1540.0(70.2) | -1.3 | -0.1% |
| **SV** | **46.2(30.5)** | **43.5(17.9)** | **-2.7** | **-5.8%** |
| ERA | 3.670(0.228) | 3.663(0.210) | -0.007 | -0.2% |
| WHIP | 1.186(0.047) | 1.184(0.047) | -0.002 | -0.2% |

### (b) YTD-blended final totals (1000 iters, seed=42; production path)

| Cat | OLD mean(sd) | NEW mean(sd) | delta | % |
|---|---|---|---|---|
| R | 1119.8(43.0) | 1121.0(42.0) | 1.2 | 0.1% |
| HR | 331.9(27.0) | 331.6(21.9) | -0.3 | -0.1% |
| RBI | 1054.9(45.5) | 1054.3(44.3) | -0.5 | -0.1% |
| SB | 227.3(38.4) | 225.4(29.6) | -1.9 | -0.8% |
| AVG | 0.267(0.006) | 0.267(0.006) | 0.000 | 0.0% |
| W | 102.0(11.9) | 101.7(11.0) | -0.2 | -0.2% |
| K | 1540.7(77.2) | 1539.5(72.9) | -1.2 | -0.1% |
| **SV** | **49.4(26.6)** | **44.1(17.0)** | **-5.3** | **-10.7%** |
| ERA | 3.668(0.218) | 3.661(0.212) | -0.007 | -0.2% |
| WHIP | 1.185(0.047) | 1.184(0.046) | -0.001 | -0.1% |

(Both full-season views run high vs a 9-pitcher/12-hitter league's eventual
*active* lines because they sum the projected full-season totals of the best
12 hitters / 9 pitchers selected each iter; the win-points in section 3 are
the decision-relevant output.)

The banded stats are exactly where OLD and NEW diverge: **SV -5.8% (full) /
-10.7% (blended)** and **SB -0.8% to -1.0%**. The clip's `max(0, .)`
truncation biases low-mu/high-sigma counts upward; the NegBin is mean-exact,
so NEW sits lower on SB/SV. The other 8 categories move <0.5% (high-mu stats
where the clip rarely truncates). Note the **SD collapse on SV** (full-season
30.5 -> 17.9; blended 26.6 -> 17.0): the clipped-Gaussian's symmetric spread
on a multi-closer team produced an implausibly fat SV tail; the NegBin's
overdispersion is tighter and more realistic.

## 5. Per-player table (active-roster contributions, 5000 iters)

`n` = iterations the player made the active roster (out of 5000). `o`/`n` =
OLD/NEW sim mean; `(sd)` on the banded stat. Headline banded stats: **SB**
(hitters), **SV** (pitchers).

### Hitters -- SB front-and-center

| Player | n | R o/n | HR o/n | RBI o/n | SB o(sd) -> n(sd) | d%SB |
|---|---|---|---|---|---|---|
| Oneil Cruz | 4978 | 96.2/96.0 | 30.4/30.2 | 92.4/92.1 | 40.44(20.81) -> 40.22(15.13) | -0.5% |
| Randy Arozarena | 4930 | 100.0/99.5 | 20.2/20.0 | 80.4/79.8 | 35.52(17.61) -> 34.77(12.72) | -2.1% |
| CJ Abrams | 4977 | 97.2/97.1 | 27.0/26.8 | 101.0/100.9 | 27.86(14.06) -> 27.39(10.23) | -1.7% |
| Otto Lopez | 4356 | 91.0/91.0 | 13.4/13.5 | 75.1/75.0 | 25.63(11.67) -> 24.89(8.79) | -2.9% |
| Julio Rodriguez | 4946 | 96.3/96.2 | 30.7/30.6 | 90.5/90.4 | 24.33(12.33) -> 24.12(9.24) | -0.9% |
| Ceddanne Rafaela | 3927 | 78.7/79.0 | 18.5/18.7 | 78.7/79.1 | 21.67(9.52) -> 20.69(7.11) | -4.5% |
| Juan Soto | 4972 | 97.0/96.7 | 38.0/37.8 | 94.7/94.1 | 18.59(9.22) -> 18.37(6.88) | -1.2% |
| Byron Buxton | 4944 | 99.3/99.1 | 41.5/40.9 | 84.6/84.2 | 15.34(7.45) -> 15.39(5.76) | +0.3% |
| Willy Adames | 4230 | 84.6/84.9 | 27.2/26.9 | 80.6/80.8 | 8.14(3.59) -> 8.10(3.38) | -0.5% |
| Ivan Herrera | 4315 | 97.4/97.2 | 21.5/21.4 | 77.9/77.8 | 8.04(3.90) -> 7.97(3.44) | -0.9% |
| Freddie Freeman | 4776 | 92.6/92.8 | 25.3/25.3 | 93.0/93.2 | 6.66(3.20) -> 6.62(3.33) | -0.6% |
| Vinnie Pasquantino | 3713 | 76.7/77.2 | 22.4/22.2 | 88.9/89.3 | 5.04(2.23) -> 5.14(2.60) | +2.0% |
| Junior Caminero | 4936 | 98.4/98.5 | 37.7/37.7 | 98.0/98.1 | 3.52(1.65) -> 3.50(2.01) | -0.6% |

### Pitchers -- SV front-and-center (Mason Miller now CORRECT)

| Player | n | W o/n | K o/n | SV o(sd) -> n(sd) | d%SV |
|---|---|---|---|---|---|
| **Mason Miller** | 4799 | 3.9/3.9 | 123.0/122.0 | **37.86(24.06) -> 34.76(11.41)** | **-8.2%** |
| Josh Hader | 1947 | 4.6/4.6 | 69.0/69.5 | 31.48(11.47) -> 26.00(5.29) | -17.4% |
| Jose Soriano | 4984 | 14.2/14.2 | 199.0/198.9 | 0.0 -> 0.0 | 0.0% |
| Jesus Luzardo | 4997 | 12.8/12.8 | 212.2/212.6 | 0.0 -> 0.0 | 0.0% |
| Bryan Woo | 4972 | 12.6/12.6 | 191.2/190.7 | 0.0 -> 0.0 | 0.0% |
| Zack Wheeler | 4770 | 12.4/12.4 | 174.3/174.5 | 0.0 -> 0.0 | 0.0% |
| Nathan Eovaldi | 4899 | 11.8/11.8 | 180.5/180.9 | 0.0 -> 0.0 | 0.0% |
| Logan Webb | 4648 | 10.7/10.7 | 167.4/167.7 | 0.0 -> 0.0 | 0.0% |
| Spencer Strider | 4533 | 11.1/11.2 | 169.7/170.1 | 0.0 -> 0.0 | 0.0% |
| Sonny Gray | 4445 | 14.8/14.8 | 159.1/159.2 | 0.0 -> 0.0 | 0.0% |
| Blake Snell | 6 | 7.8/8.5 | 113.4/128.5 | 0.36 -> 0.00 | n/a (IL, n=6) |
| Edwin Diaz | 0 | -- | -- | -- | benched all 5000 iters |

Notes:
- **Mason Miller is now the headline SV player and resolves correctly**: his
  SV mean shifts 37.9 -> 34.8 (-8.2%) and his SD collapses 24.1 -> 11.4. The
  clipped-Gaussian gave a closer projected for ~37 SV an SD of ~24 (a ~+/-65%
  swing, with the high tail propped up by the clip); the NegBin's ~11 SD is far
  more credible for a closer's save total. This is the single biggest
  per-player correction in the whole eye-test, and it was **invisible in the
  local run** (where Miller had SV=0).
- Josh Hader makes the active 9 only ~39% of iters (the SP corps + Miller
  crowd him out); when he does, SV 31.5 -> 26.0 with SD 11.5 -> 5.3.
- **Edwin Diaz never makes the active roster** (a 3rd/4th closer behind a deep
  SP staff + two higher-SV closers) -- correct active-roster selection, not a
  matching failure. Blake Snell (IL) appears in only 6 of 5000 iters.

## 6. Comparison to the prior LOCAL eye-test

| Quantity | LOCAL (2026-06-14) | REMOTE/PROD (this run) |
|---|---|---|
| Roster source | `weekly_rosters_2026.json`, 2026-04-07 snapshot (early April) | prod `cache:roster`, 2026-06-14 (current) |
| Mason Miller | matched **wrong** KCR scrub, proj SV=0 (uninformative) | **correct SDP closer**, season SV 37 |
| Projection scope | ROS-only blend (no YTD add) | full-season + YTD-blended (production) |
| Team SV (OLD -> NEW) | 28.2 -> 22.8 (-19.4%), ROS-only, no real closer | full-season **46.2 -> 43.5 (-5.8%)**; blended **49.4 -> 44.1 (-10.7%)** |
| Headline SV player | (none -- Miller was SV=0) | **Mason Miller 37.9 -> 34.8 (SD 24.1 -> 11.4)** |

The **corrected team SV** is the headline change: the prior local "28 -> 23"
ran on a roster with **no real closer in the Miller slot**, so its team SV was
built almost entirely on Hader + small backfill. With prod resolving Miller to
the genuine SDP closer (season SV 37), Hart's team SV roughly **doubles** in
absolute level and the OLD->NEW direction holds -- NEW lower than OLD on the
banded SV stat (full-season -5.8%, blended -10.7%), but the dominant story is
the **variance**: the NegBin tightens the SV SD dramatically (team SV SD
26.6->17.0 blended; Miller 24.1->11.4) rather than just shifting the mean.
The local run's larger -19.4% mean drop was an artifact of a tiny,
backfill-heavy SV base; on the real roster the mean move is smaller and the
SD correction is the real signal.

The qualitative conclusion from the local run still holds and is now on
correct data: **NEW removes the clip's upward bias and fat symmetric tail on
the low-mu/high-sigma banded stats (SB, SV), leaves the high-mu stats
essentially unchanged, and slightly improves Hart's overall win read
(median 77->78, first 52.1%->54.8%).**
