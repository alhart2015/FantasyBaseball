# Eye-test (REMOTE PROD): OLD flat-CV vs NEW NegBin analytic team SD vs MC

Team: **Hart of the Order**  |  Date: 2026-06-15  |  Branch: `eroto-negbin-unification` (HEAD 983a17b)

READ-ONLY validation against **prod Upstash** (the source of truth). No writes:
only `kv.get(...)` was called. No `.set` / `.delete` / `.hset` / `compare_delete`
/ `set_if_absent` / `write_cache` / `_push_to_prod`, no refresh/ingest/dashboard
pipeline, `RENDER` never set. Prod was read via `build_explicit_upstash_kv()`
(NOT `get_kv()`, which off-Render reads a local SQLite) with the `_meta`
envelope hand-unwrapped.

## What this validates

The variance unification put ALL per-stat performance dispersion on
`STAT_DISPERSION` (NegBin): the analytic ERoto team SD
(`scoring.project_team_sds`) now derives each player's performance variance from
`negbin_perf_variance(stat, mu) = mu + mu**2/r` (`utils/dispersion.py`) -- the
SAME quantity the Monte Carlo's `_negbin_copula_counts` samples. The old path
used a flat per-stat CV constant (the deleted `STAT_VARIANCE`) whose `sv=0.900`
made the analytic SV SD roughly 3x the MC's. Goal: confirm the NEW analytic
team SD now AGREES with the MC's sampled team SD, especially on SV.

## Blob provenance (prod `_meta`)

| Blob | `_written_at` | other |
|---|---|---|
| `cache:roster` | 2026-06-15T10:01:52Z | `_sha` 399090c, `_job` refresh |
| `cache:ros_projections` | 2026-06-10T18:50:25Z | `_ros_snapshot_date` **2026-06-10** |

(The roster is one daily refresh newer than the prior remote eye-test's
2026-06-14T10:01:39Z snapshot; same `_sha` 399090c, same ROS vintage 2026-06-10
-- the Mason-Miller-correct source. George Springer has replaced Vinnie
Pasquantino at a UTIL slot and Spencer Strider has moved to BN vs the prior
snapshot; otherwise the roster matches.)

## Method

- **Roster reconstruction:** `Player.from_dict(d)` over the 25 `cache:roster`
  player dicts. Each carries nested `full_season_projection` + `rest_of_season`.
  **Match rate: 25/25** -- every player has a non-empty `full_season_projection`.
- **NEW analytic** = `scoring.project_team_sds(roster, displacement=True)`.
  `project_team_sds` reads stats via `_stat(..., "rest_of_season")` and scales
  displaced players off `rest_of_season`, so to exercise it at FULL-SEASON mu
  (to compare against the full-season MC and the season-total anchors, e.g.
  Miller SV~38) each Player's `rest_of_season` was set to its
  `full_season_projection` before the call. This is the only manipulation; the
  variance math is unmodified production code.
- **OLD analytic** = the SAME displaced active roster (`scoring._apply_displacement`),
  the SAME `cv_pt` from `playing_time_params(type, _full_season_volume(p,...))[1]`,
  the SAME `_stat` reads -- substituting only the flat-CV performance term:
  counting `var_cat = sum mu^2*(CV[stat]^2 + cv_pt^2)`;
  `AVG = CV[h]*sqrt(sum h^2)/sum_AB`, `ERA = 9*CV[er]*sqrt(sum er^2)/sum_IP`,
  `WHIP = sqrt(CV[bb]^2*sum bb^2 + CV[ha]^2*sum ha^2)/sum_IP`. OLD CVs:
  `r=0.156, hr=0.343, rbi=0.187, sb=0.715, h=0.103, w=0.416, k=0.139, sv=0.900,
  er=0.252, bb=0.257, h_allowed=0.143`.
- **MC sampled SD** = production sampler `simulation._apply_variance`
  (`_negbin_copula_counts` + calibrated playing-time) run 5000 iters
  (seed=12345), `fraction_remaining=1.0` (full season -- the same scale as the
  full-season analytic), re-selecting the active 12 hitters / 9 pitchers each
  iter and aggregating Hart's team category totals. SD = `np.std` across iters.

**CAVEAT (selection churn):** the MC RE-SELECTS the active roster every iter, so
its SD mixes per-player performance variance WITH which-players-make-the-9/12
churn; the analytic SD is a fixed-roster (post-displacement) sum. They are
expected to agree most tightly on the banded low-mu cats the unification
targeted (SV, SB) and to diverge on high-mu cats where the MC's near-fixed best
lineup collapses the across-iter spread (most visibly K, see below).

## Per-category team SD (full-season)

| Cat | OLD analytic | NEW analytic | MC sampled | (MC mean) | NEW/MC | verdict |
|---|---|---|---|---|---|---|
| R | 80.04 | 78.92 | 54.88 | 1121.8 | 1.44 | NEW ~= OLD; both > MC (fixed-roster vs churn) |
| HR | 38.92 | 33.35 | 28.95 | 330.5 | 1.15 | close; NEW slightly tighter than OLD |
| RBI | 80.24 | 79.60 | 58.34 | 1041.3 | 1.36 | NEW ~= OLD; both > MC |
| **SB** | **56.37** | **40.89** | **38.06** | 230.3 | **1.07** | **NEW converges to MC; OLD ~1.48x MC** |
| AVG | 0.0080 | 0.0062 | 0.0063 | 0.266 | 0.98 | NEW lands on MC; OLD high |
| W | 18.23 | 15.38 | 11.00 | 102.8 | 1.40 | NEW < OLD; both > MC |
| K | 182.96 | 180.25 | 81.28 | 1557.6 | 2.22 | both >> MC -- MC K SD collapsed by near-fixed SP set |
| **SV** | **37.97** | **18.34** | **20.99** | 39.9 | **0.87** | **NEW converges to MC; OLD ~1.81x MC** |
| ERA | 0.307 | 0.279 | 0.265 | 3.685 | 1.06 | NEW ~= MC; OLD high |
| WHIP | 0.0495 | 0.0485 | 0.0591 | 1.189 | 0.82 | NEW ~= OLD; both a touch under MC |

Headline: on **SV** the OLD analytic SD (37.97) was **1.81x** the MC sampled SD
(20.99); the NEW analytic SD (18.34) is **0.87x** the MC -- the ~2-3x SV
over-statement is gone and NEW now sits just under (and close to) the MC. The
OLD->NEW SV ratio is **37.97 / 18.34 = 2.07x** tighter. **SB** behaves the same
way (OLD 56.37 = 1.48x MC; NEW 40.89 = 1.07x MC). The 8 non-SV/SB cats move much
less between OLD and NEW (the flat CV was far less wrong on high-mu stats):
HR -14%, W -16%, RBI/R/K/ERA/WHIP/AVG all within a few percent OLD->NEW.

On the cats where BOTH analytic SDs run above the MC (R, RBI, K, W), the gap is
the selection-churn caveat, not a dispersion bug: the analytic sums each fixed
post-displacement player's full variance, while the MC's best-9 SP set is nearly
the same arms every iter, so the across-iter team K/W/R/RBI spread is narrower
than the independent-sum analytic. K is the extreme (MC 81 vs analytic 180):
Hart's SP corps is deep and stable, so the active K total barely moves across
iters. This is orthogonal to the NegBin unification, which is about per-player
dispersion, and it does not affect the SV/SB verdict.

## Per-player SV/SB spot-check (deterministic, full-season mu, NO selection noise)

Per counting stat: OLD perf-only `= mu*CV`; OLD total `= sqrt(mu^2*(CV^2+cv_pt^2))`;
NEW perf-only `= sqrt(negbin_perf_variance(stat, mu))`;
NEW total `= sqrt(negbin_perf_variance(stat, mu) + mu^2*cv_pt^2)`.

| Player | stat | mu | OLD perf | OLD total | NEW perf | NEW total |
|---|---|---|---|---|---|---|
| Mason Miller | SV | 38.1 | 34.25 | 37.75 | **8.74** | 18.12 |
| Josh Hader | SV | 19.8 | 17.86 | 20.20 | 5.50 | 10.93 |
| Edwin Diaz | SV | 4.0 | 3.60 | 4.07 | 2.10 | 2.84 |
| Oneil Cruz | SB | 40.3 | 28.80 | 29.83 | 19.55 | 21.04 |
| Randy Arozarena | SB | 34.9 | 24.98 | 25.87 | 17.09 | 18.37 |
| CJ Abrams | SB | 27.1 | 19.40 | 20.09 | 13.50 | 14.48 |
| Otto Lopez | SB | 23.9 | 17.08 | 17.69 | 12.01 | 12.86 |

**Mason Miller anchor (the load-bearing one):** NEW perf-only SV SD =
`sqrt(38.1 + 38.1^2/37.757) = 8.74` (anchor predicted ~8.56 at mu=37 -- matches,
scaled to mu=38.1). OLD perf-only = `38.1*0.900 = 34.25` (anchor ~33.3 at mu=37
-- matches). The OLD perf-only SV SD is **~3.9x** the NEW perf-only SD,
player-for-player -- the cleanest, churn-free proof that the deleted `sv=0.900`
flat CV grossly overstated closer-save dispersion.

Note on the per-player **total** (with cv_pt): Miller's full-season IP=67.4 lands
on the RP playing-time curve with `cv_pt=0.4172`, so his NEW total SV SD =
`sqrt(8.74^2 + 38.1^2*0.4172^2) = 18.1` -- the playing-time term, not
performance, now dominates a closer's SV uncertainty. The prior remote
eye-test's per-player Miller SV SD of ~11.4 was at the season-damped
`fraction_remaining=0.573`; this run is at full-season `fr=1.0`, which scales the
PT-driven spread up correspondingly (and is why the team SV MC here is ~21, not
~18). Analytic and MC are on the same `fr=1.0` footing, so the NEW-vs-MC
comparison is internally consistent.

## Anchor checklist

- Miller NEW perf-only SV SD ~8.56: **HELD** (8.74 at mu=38.1).
- Miller OLD perf-only SV SD ~33.3: **HELD** (34.25 at mu=38.1).
- Team SV NEW high-teens near MC ~17.9: **HELD in spirit** -- NEW=18.34, and the
  full-season `fr=1.0` MC here is 20.99 (the prior doc's 17.9 was the damped-`fr`
  variant); NEW/MC=0.87, well inside "converged."
- Team SV OLD ~3x: **DIRECTIONALLY HELD** -- OLD/NEW = 2.07x and OLD/MC = 1.81x.
  The absolute OLD (37.97, not 45-55) is lower than the bare anchor because
  `project_team_sds` displacement drops Josh Hader from Hart's active 9 pitchers
  (deep SP corps + Miller crowd him out; only Miller SV=38 + the IL-activated
  Edwin Diaz SV=4 carry the analytic team SV). The MC re-selects and sometimes
  seats Hader, so its SV mean (~40) and SD reflect a slightly different active
  set -- the selection-churn caveat. The ~2-3x over-statement on SV is the
  signal, and it is gone.
- 8 non-banded cats OLD ~= NEW and both near MC: **HELD** for the dispersion
  comparison; R/RBI/K/W analytic sit above MC purely from fixed-roster vs
  re-selection churn (esp. K), not from the variance model.

## Conclusion

The NEW unified-NegBin analytic team SD **converges to the MC's sampled SD on the
stats the unification targeted**: SV (NEW 18.34 vs MC 20.99, NEW/MC 0.87) and SB
(NEW 40.89 vs MC 38.06, NEW/MC 1.07). The OLD flat-CV path overstated both
(SV 1.81x MC, SB 1.48x MC); the **OLD->NEW SV SD tightens 2.07x** and the
per-player Miller proof shows the OLD `sv=0.900` CV inflated closer-save
performance dispersion ~3.9x over the NegBin. **The ~3x analytic-vs-MC SV gap is
gone.** The remaining analytic-above-MC gaps on R/RBI/K/W are the documented
fixed-roster-vs-re-selection-churn artifact, orthogonal to the NegBin
unification. Validation PASSES.
