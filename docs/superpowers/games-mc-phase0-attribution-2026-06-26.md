# Phase 0 attribution gate -- decision note (2026-06-26)

Run conditions: `seed=42`, `n_iter=1000`, `fraction_remaining=0.5081` (the live
in-season snapshot, local full refresh with `FB_SELECTION_ATTRIBUTION=1`). Raw
table: `phase0_attribution.txt` (per team, all 10 categories, three arms).

Three selection arms, sharing one seed (differ only in which players are summed):
- `topk_per_iter` -- today's MC: top-k by raw stats, RE-SELECTED every iteration
  (best-ball churn), bench AND IL eligible to be seated.
- `topk_fixed` -- top-k fixed ONCE on the projected mean (removes churn; still
  lets bench/IL into the pool).
- `active_slot` -- only the manager's active-slotted players (healthy bench AND
  IL excluded), via the canonical `_classify_roster`.

Decomposition of the per-iter-top-k over-credit, per category:
`gap = topk_per_iter - active_slot` ; `churn = topk_per_iter - topk_fixed` ;
`seating = topk_fixed - active_slot` (bench/IL seating).

## RBI (the category the project was motivated on)

| team | gap | churn | seating | seating% |
|---|---:|---:|---:|---:|
| Boston Estrellas | 25.1 | 1.2 | 23.9 | 95% |
| Hart of the Order | 28.3 | 17.9 | 10.4 | 37% |
| Hello Peanuts! | 31.8 | 10.3 | 21.4 | 67% |
| Jon's Underdogs | 32.7 | -2.9 | 35.6 | 109% |
| Send in the Cavalli | 139.5 | 10.7 | 128.8 | 92% |
| SkeleThor | 20.2 | 18.2 | 2.1 | 10% |
| Spacemen | 76.3 | 11.4 | 64.9 | 85% |
| Springfield Isotopes | 56.5 | 16.7 | 39.8 | 70% |
| Tortured Baseball Dept | 13.7 | -0.9 | 14.5 | 106% |
| Work in Progress | 31.8 | 2.1 | 29.7 | 93% |
| **TOTAL** | **455.8** | **84.8** | **371.0** | **81%** |

## Verdict against the literal gate criterion: STOP

The plan's gate was defined on SkeleThor RBI: proceed iff bench-seating accounts
for >= 50% of the SkeleThor gap. **SkeleThor seating_share = 10.3% -> STOP.**

SkeleThor's bench-seating is only ~2 RBI. The "~94-RBI" figure that motivated the
whole project was MC (1020) vs ERoto (926) -- but the slot-legal `active_slot` MC
arm is **999.73**, far above ERoto's 926. So SkeleThor's MC-vs-ERoto gap splits as:
~18 RBI churn + ~2 RBI bench-seating + ~74 RBI `active_slot`-vs-ERoto. That last
~74 RBI is NOT a selection problem -- it is the ROS playing-time-haircut /
displacement / mean-vs-variance modeling difference between the engines (a
separate, already-open TODO). **The motivating SkeleThor diagnosis was a
misattribution.**

## But the literal gate is misleading: league-wide, seating DOMINATES

SkeleThor is the atypical team. Across the league, bench/IL-seating is the
DOMINANT selection effect: **8 of 10 teams have RBI seating_share > 50%**, and the
league totals are **371 RBI of seating vs 85 RBI of churn (81% seating).** Send in
the Cavalli alone carries 129 RBI of seating (and ~184 R; its `active_slot`
totals sit far below its top-k). So the games-based engine is NOT solving a
non-problem -- it addresses a real, large, league-wide effect. The one team the
original eyeball-read fixated on is the single team where it barely matters.

## Critical caveat: how much of "seating" is IL vs healthy bench?

`active_slot` excludes BOTH healthy bench AND IL-slotted players. IL players DO
return and play part of the ROS, so for IL-heavy teams `active_slot` is an
artificially low FLOOR and `topk_per_iter` (which seats IL at FULL, immediately)
is a high CEILING. The truth is in between -- exactly where both ERoto's existing
IL-displacement AND the proposed games-fill engine operate. The huge-gap teams
(Send in the Cavalli 129, Spacemen 65, Springfield 40) are plausibly IL/injury
driven, not healthy-bench-depth driven. This matters for WHICH fix is right:

- If most seating is **IL players seated at full**: the cheap fix is to make the
  MC exclude/displace IL like ERoto already does -- the "mirror ERoto" approach we
  originally set aside -- which captures most of the 371 RBI without the full
  games-availability build.
- If a meaningful share is **healthy bench depth**: that is the injury-insurance
  band the full games-fill engine is uniquely for.

We have NOT yet split seating into IL vs healthy-bench. That split is a cheap
follow-up (the `active_slot`/`topk_fixed` arms already exist; add an
"active+IL-at-full" or "active+IL-displaced" arm).

## Three separable effects, three different fixes

1. **Churn (~85 RBI league-wide):** per-iteration best-ball re-selection. Cheap
   fix -- freeze selection (`topk_fixed`-style, once on the mean). Worth doing
   regardless of the engine decision.
2. **IL seated at full (unknown share of the 371):** the MC is IL-status-blind;
   ERoto already displaces IL. Medium fix -- mirror that in the MC.
3. **Healthy-bench-as-injury-insurance (the remainder):** the full games-based
   availability engine (Phases 1-6 of the spec).

## Decision: GO (full games engine), churn-freeze folded in

A per-player ROS-RBI dig (Hart, SkeleThor, Send in the Cavalli; from the cached
`standings_breakdown`) resolved the IL-vs-healthy-bench question by inspection and
corrected the model:

- **IL players** (Hart's Cruz; SkeleThor's Hicks, Acuna): their ROS projection
  already bakes in the injury, so they are counted at FULL ROS and displace ONE
  eligible active body by the IL player's expected ROS playing time -- a
  one-for-one, slot-PT-conserving swap. ERoto already does this correctly (Cruz
  ~200 ROS AB displaces Springer, who drops ~186 AB; Riley is untouched in ERoto
  -- Riley's 0 was the today's-MC raw-top-k column, not ERoto). The current MC
  does NOT do this (it seats IL at full AND keeps the displaced body).
- **Healthy bench** (Cavalli's Perez/Ward/Arraez): must NOT be zeroed. The MC's
  playing-time sampling IS the injury simulation; when an active starter draws a
  low-PT stretch, an eligible bench bat fills those simulated missed player-games
  at a nonnegative rate, capped at one body, replacement-level only when no bench
  body is free. This is the injury-insurance value the engine exists for --
  neither today's MC (full, no injury needed) nor ERoto/active-slot (zero) models
  it. Bench-exclusion was therefore the WRONG "cheap fix": it under-credits depth
  exactly as much as top-k over-credits it.

Corrected target model the engine must build:
1. IL = full (injury-baked) ROS + one eligible active body displaced by the IL
   player's expected ROS playing time (mirror ERoto; the MC lacks this today).
2. Healthy bench = fill the per-iteration SIMULATED missed games of eligible
   active starters; nonnegative, one-body cap, replacement-level last.
3. Churn = freeze the per-iteration re-selection (a real ~85-RBI league-wide
   artifact). Folded INTO the engine work, not shipped separately (user: "we
   don't need the cheap win on its own").

What the gate accomplished: it killed the WRONG framing (SkeleThor's gap is
IL-displacement, not bench-seating) and the per-player dig pinned the RIGHT model
(above). Motivation is re-framed away from the refuted SkeleThor 94-RBI story
toward league-wide bench-injury-insurance + correct IL displacement.

Raw three-arm table: `games-mc-phase0-attribution-2026-06-26-raw.txt`.
