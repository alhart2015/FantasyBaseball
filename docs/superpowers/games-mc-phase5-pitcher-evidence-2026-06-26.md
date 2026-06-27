# Phase 5 pitcher evidence (2026-06-26) -- PASS (new_engine ~= ERoto)

Run: local refresh, FB_SELECTION_ATTRIBUTION=1, seed=42, n_iter=1000,
frac=0.5081. 4-arm table (now with pitchers ROS-direct in the new_engine arm):
games-mc-phase5-pitcher-evidence-2026-06-26-raw.txt.

new_engine pitcher categories match ERoto end-of-season to within rounding
(ERoto = team_ytd + sum(pitcher contribution_stats), from standings_breakdown):

| team | cat | ERoto | old top-k | new_engine | NE-ERoto |
|---|---|---:|---:|---:|---:|
| Hart | W | 89.7 | 95.0 | 90.0 | +0.3 |
| Hart | K | 1413.7 | 1477.8 | 1417.1 | +3.4 |
| Hart | SV | 72.2 | 55.5 | 72.0 | -0.2 |
| SkeleThor | W | 84.5 | 97.4 | 85.0 | +0.5 |
| SkeleThor | K | 1304.0 | 1413.7 | 1305.0 | +1.0 |
| SkeleThor | SV | 76.8 | 59.4 | 76.0 | -0.8 |
| Cavalli | W | 61.3 | 66.7 | 61.0 | -0.3 |
| Cavalli | K | 980.6 | 1089.4 | 982.0 | +1.4 |
| Cavalli | SV | 90.6 | 82.1 | 91.0 | +0.4 |

Confirms:
1. **pt_mean_fraction=0 (no mean haircut) is correct** -- pitcher means == ERoto, NOT
   the ~20% deflation that pt_mean_fraction=1.0 (plan-review CRITICAL) would have
   caused. (Pitchers get no bench-fill premium, unlike hitters -- the accepted
   asymmetry; pitchers exactly mirror ERoto.)
2. **Both top-k errors fixed.** Old top-k OVER-credited K (seated bench pitchers:
   Hart 1478 vs ERoto 1414) and UNDER-credited SV (its w+k+sv key drops low-IP
   closers: Hart 55 vs ERoto 72). The slot-based active set (active-slot pitchers +
   pool-displaced IL) restores both to ERoto.

Full engine now validated: HITTERS new_engine ~= ERoto + small bench-injury-fill
premium; PITCHERS new_engine ~= ERoto (no premium). Exactly the spec's model.
