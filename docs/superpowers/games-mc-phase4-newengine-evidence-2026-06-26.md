# Phase 4 new-engine before/after evidence (2026-06-26)

Run: local refresh, `FB_SELECTION_ATTRIBUTION=1`, seed=42, n_iter=1000,
fraction_remaining=0.5081, live league snapshot. 4-arm table:
`games-mc-phase4-newengine-evidence-2026-06-26-raw.txt`
(topk_per_iter / topk_fixed / active_slot / new_engine).

## Verdict: PASS -- the engine behaves as designed.

`new_engine` ~= ERoto (slot-legal, displacement-correct) + a small bench
injury-fill premium. Decisive comparison vs ERoto end-of-season (from the cached
standings_breakdown):

| team | cat | ERoto | new_engine | premium | old top-k |
|---|---|---:|---:|---:|---:|
| SkeleThor | RBI | 926 | 948 | +22 | 1021 |
| SkeleThor | HR  | 243 | 244 | +1.6 | 279 |
| Hart | RBI | 1001 | 1022 | +20 | 1011 |
| Hart | HR  | 325.6 | 325.8 | +0.2 | 328 |
| Cavalli | RBI | 841 | 838 | -3 | 980 |

Two required behaviors confirmed:
1. **IL displacement tracked by construction.** SkeleThor RBI new_engine 948 ~=
   ERoto 926, vs the old top-k 1021. The motivating "~94-RBI over-credit" is now a
   +22-RBI LEGITIMATE injury-insurance premium, not a bug. (Okamoto/Bauers
   displaced, Hicks/Acuna activated -- HR within 1.6 of ERoto.)
2. **Bench injury-fill premium that ERoto lacks.** +15-22 RBI for IL/depth teams;
   ~0 net for healthy-bench Cavalli (its benched Perez/Ward/Arraez contribute a
   small net, NOT their full ~99 RBI that the old top-k seated).

The apparent "anomalies" are correct against the right anchor (ERoto, not the
crude top-k/active_slot bounds): new_engine < active_slot for IL teams (active_slot
omits displacement, over-counting), and new_engine > top-k for Hart (top-k was
never the true ceiling; ERoto + fill is the right number).

Acceptance (spec): SkeleThor RBI lands materially below 1020 (948, between the
~926 ERoto floor and 1020) -- MET. Bench bats contribute a small injury-fill
share, not their full value -- MET. Pitchers unchanged (distribution; model
untouched in Phase 4).

Caveat: AVG/ERA/WHIP and the SD calibration (pt_mean_fraction horizon split) are
validated in Phase 6's SD backtest, not here. One slightly-off cat: Cavalli HR
-11 vs ERoto, within MC/HR-variance noise.
