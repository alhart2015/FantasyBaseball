# /loop-review case study: PR #280 (feat/277 decompose-luck) -- 2026-07-31

Data point #2 for issue **#274** ("Investigate /loop-review latency"). This run is
worse and *differently shaped* than the #272 run already documented there, so it
isolates a failure mode the existing hypotheses don't cover.

- **Branch / PR:** `feat/277-decompose-luck` -> PR #280 (stacked on `feat/273-league-keeper-board`)
- **Diff at start of loop:** ~10 files, +1703/-137 (the committed #277 five-family keeper composite) plus the accumulating loop-review edits
- **Passes run:** **10 (hit the `--max 10` cap; did NOT reach a clean pass)**
- **Wall clock:** ~an overnight session (multiple hours). Each `/code-review` pass = one workflow of 10-17 subagents.
- **Session transcript:** `~/.claude/projects/C--Users-HartAlden-FantasyBaseball/d2b8a652-4156-4dec-96fb-f08689a6dbcf.jsonl`
- **Workflow journals:** `.../subagents/workflows/wf_<runId>/journal.jsonl` (runIds in the ledger below)

---

## TL;DR diagnosis

The #272 run in #274 had a clean shape: findings decayed 3 -> 2 -> 0 and the loop
terminated on a zero-finding pass. **This run never decayed.** It found real,
CONFIRMED issues on almost every pass and hit the cap without a clean pass. Both of
the operator's a-priori hypotheses were true *at once*, and they compounded:

1. **Fix-induced churn ("having to fix last pass").** Multiple passes found bugs that
   the *previous pass's own fix* introduced or unmasked. At least 5 of the ~10 passes
   were reacting to the prior pass, not to the original diff.
2. **One concern discovered one facet per pass ("should have found on pass 1").** Roughly
   passes 4-10 were all facets of a *single* question -- "what happens when a keeper
   pool / family / board is empty, absent, all-NaN, or half-broken?" A systematic
   edge-case enumeration of that one concern on pass 1 would have surfaced most of them
   together.

The mechanism tying them together: **the loop does local, salience-driven review.**
Each `/code-review` finds the single most salient issue in the current diff, a local
band-aid is applied, and the band-aid shifts the problem to the adjacent facet, which
the next 25-30 min pass then "discovers." There is no trigger that says "you have now
found 3 findings in the same subsystem across 3 passes -- stop whacking facets and
enumerate the whole space with one coherent design."

This is **not** the `/simplify` <-> `/code-review` oscillation the cap is designed to
catch. `/simplify` was clean or near-clean every pass; the churn was inside the
correctness dimension, driven by the fixes.

---

## Per-pass ledger

`/simplify` = 4 parallel cleanup agents; `/code-review` = one xhigh workflow. "Tokens"
and "ms" are the `/code-review` workflow's own subagent totals (from the task
notification), excluding the 4 `/simplify` agents (~250-350k tokens/pass) and the
orchestrator's own fixing/verification between phases.

| pass | code-review runId | agents | subagent tok | ms | reported | refuted | headline finding | fix-induced? |
|---|---|---|---|---|---|---|---|---|
| 3 | wkylh9eye | 17 | 1,394,749 | 1,753,997 | 7 | 0 | strict-guard gaps, `_weighted_rho`/`_best_weights`, deleted null floor | no (original diff) |
| 4 | wpifg7u20 | 13 | 1,148,297 | 1,770,804 | 2 | 2 | live board silently drops all-NaN `batted_ball` | partly (guard added pass 3-area) |
| 5 | w6ji23k79 | 16 | 1,332,810 | 1,862,340 | 6 | 1 | guard covers only 2 of 5 families (age can go all-NaN too) | **yes** (pass-4 guard too narrow) |
| 6 | wf_aff6b396 | 11 | 1,095,906 | 1,633,369 | 1 | 0 | empty pool misreported as "skill all-NaN" | yes (pass-5 generalized guard) |
| 7 | wf_c226a300 | 15 | 1,366,222 | 1,824,650 | 2 | 1 | relocating the empty guard crashes --study/--backtest early-season | **yes** (pass-7 /simplify relocation) |
| 8 | wf_ef115978 | 13 | 1,153,748 | 1,404,912 | 2 | 2 | #277's all-NaN drop broke composite's base empty-tolerance; broken-join now silent | **yes** (pass-8 empty-tolerance flip side) + latent-in-#277 |
| 9 | wf_6a85cdc6 | 15 | 1,299,992 | ~20,612,663 (?) | 2 | 2 | partial one-pool board slips the whole-board guard; typo'd family_order -> opaque KeyError | **yes** (pass-8 fail-loud too coarse) |
| 10 | w3y9139tc | 10 | 777,617 | 995,768 | 1 | 0 | pass-9 test used `match="sklll"` which also matches the opaque error it was meant to exclude | **yes** (pass-9 test) |

- **Code-review subagent tokens, passes 3-10: ~9.57M.** Add `/simplify` (~2M) and passes 1-2
  (pre-session) and the review machinery alone is ~12M+ subagent tokens for this branch.
- Pass 9's `ms` (~20.6M = 5.7h) is almost certainly a measurement artifact (background
  queueing / launch-to-completion gap), not agent compute; the agent work looked like the
  others. Flag for #274 to check how workflow `duration_ms` is measured.
- **Refuted counts matter:** passes 4/5/7/8/9 each refuted 1-2 candidates. The verify stage
  is doing real work (killing plausible-but-wrong findings), which is part of the per-pass cost.

---

## The fix-induced chain (the core evidence)

This is the sequence that should have been one coherent design pass, expanded into seven:

1. **(latent, in committed #277)** `composite()` gained an all-NaN-family *drop*
   (`if series.isna().all(): continue`). On a 0-row pool every family is *vacuously*
   all-NaN, so all drop and it hits the "no weighted family" raise -- silently breaking
   the base `feat/273` behavior of blending an empty pool to an empty result. Nobody
   noticed for several passes because mid-season pools are never empty.
2. **pass 4** added a live-board guard for all-NaN `batted_ball` only.
3. **pass 5** found that guard covers 2 of 5 families -> generalized it to all families.
   Same pass, the orchestrator's own docstring "fix" wrongly claimed only 2 families could
   go all-NaN; pass-5 code-review caught that too.
4. **pass 6** found the generalized guard misreports an *empty* pool as "skill all-NaN"
   (`.isna().all()` is vacuously true on empty) -> added an empty-pool branch.
5. **pass 7** the orchestrator's `/simplify` *relocated* that empty branch to the source
   (`_qualified_families`) on altitude advice -> immediately crashed `--study`/`--backtest`
   early-season, because those diagnostics intentionally build empty sub-pools. Reverted.
6. **pass 8** traced the real root (item 1): restored composite's empty-tolerance, and made
   the live-board commands fail loud on an empty board. That fail-loud *silenced the
   broken-join case* (a mid-season 0-row join now printed "0 qualified" with exit 0).
7. **pass 9** added the broken-join / partial-pool fail-loud, and (separately) found the
   pass-3 unknown-family guard was unreachable through the production wrapper.
8. **pass 10** found the pass-9 test's `match` string was too loose to pin its own fix.

Every arrow is "the previous fix created or exposed the next finding." A single pass-1
prompt of the form *"enumerate every state in which a pool/family/board can be empty,
absent, all-NaN, or partially-joined, across the live board, the 5 CLI diagnostics, and
composite; specify the intended behavior of each; verify"* would have produced items 1-8
as one design, not seven 25-min discovery cycles.

---

## Why this run and not #272's

- #272 diff: 12 files, +2055, mostly *new* logic (np.partition, role_ip routing, dedupe).
  Findings were independent bugs in independent code -> they decayed to zero.
- #280 diff: the feature (five-family blend) was correct by pass 3. The remaining surface
  was a *cross-cutting invariant* (empty/degraded-pool handling) touching composite,
  `_qualified_families`, `_require_mandatory_families`, `projected`, `build`, 3 live
  commands, and 4 diagnostics. Cross-cutting invariants are exactly what local
  salience-driven review discovers one facet at a time.

So the shape of the diff predicts the loop's failure mode: **independent bugs decay;
cross-cutting invariants churn.** #274's hypotheses 3/4 (decay + scope) describe the
first; this run needs a hypothesis 6.

---

## Proposed hypothesis 6 (new) + remedies

**H6: Local, salience-driven review turns a single cross-cutting invariant into N serial
passes, and reactive local fixes generate the next pass's findings.**

Remedies, cheapest first:

1. **Cluster detection -> theme enumeration.** If M consecutive passes (M>=2) report
   findings whose files/symbols overlap, the loop should escalate from "find the next
   issue" to "enumerate this concern's whole state space and fix it once." Cheap to detect
   (Jaccard over finding file:line sets); high leverage. This is the single biggest win
   for *this* run's failure mode, orthogonal to #274's gate-caching win for #272's.
2. **Fix-diff review, not branch-diff review.** After pass 1, each pass should hard-focus
   the *delta since the last pass* (the fix just applied) plus its blast radius, because
   the empirical finding source shifted from "the original diff" to "the last fix" by pass
   4. (This is #274's hypothesis 4, confirmed with stronger evidence: here the last fix was
   the finding source in >=5 of 10 passes.)
3. **Orchestrator discipline: stop band-aiding.** A large share of the churn was the
   orchestrator applying the *narrowest* fix each pass (add a guard for one family; relocate
   one branch; document-not-fix a partial-board gap that pass 9 then had to actually fix).
   A "when you see the 2nd finding in a subsystem, step back and design the whole thing"
   rule in the *skill* (not just in operator judgment) would have collapsed passes 4-10.
4. Everything in #274 still applies (shared gates, decay early-exit). Note the decay
   early-exit would NOT have helped here -- findings never decayed, so a decay rule would
   have run all 10 passes anyway. That's the point: #274's remedies target the #272 shape;
   H6 targets this one.

---

## Raw material for deeper analysis

- **This session's transcript** (all 10 passes, orchestrator reasoning, every fix):
  `~/.claude/projects/C--Users-HartAlden-FantasyBaseball/d2b8a652-4156-4dec-96fb-f08689a6dbcf.jsonl`
  (the pre-compaction portion is in the prior session file referenced at its top).
- **Per-pass code-review workflow journals** (one `{"type":"result",...}` line per subagent
  with its full finding/verify output):
  `.../subagents/workflows/wf_<runId>/journal.jsonl` for each runId in the ledger.
- **The shipped diff the churn produced** (the coherent end state, for "what should pass 1
  have targeted"): commit `0adf00d1` on `feat/277-decompose-luck`, files
  `src/fantasy_baseball/keepers/composite.py` and `scripts/keeper_rankings.py` -- the guard
  cluster (`check_known_families`, `_require_mandatory_families`, `_fail_if_empty_board`,
  composite empty-return + strict) is the answer key.
- **Cost tallies** are in the ledger; the `duration_ms` anomaly on pass 9 is worth
  reconciling against how the workflow runtime clocks background time.
