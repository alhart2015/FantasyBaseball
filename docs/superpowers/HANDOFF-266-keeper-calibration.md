# Handoff: #266 keeper-value calibration study

Written 2026-07-27. Resume from here in a plain Claude Code session -- **no superpowers
skills needed**. Everything below is self-contained.

## TL;DR

**Increment 1 is COMPLETE.** All 8 tasks are implemented, run for real, and committed on
`feat/266-keeper-calibration` (nothing pushed, no PR).

**Read the finding, not this file:**
`docs/superpowers/keeper-calibration-finding-2026-07-27.md`. Part A is the pre-registration
(metric, gates, shrink constants, committed before any fitting code); Part B is the result.

**Headline:** `k` is roughly 0.4-0.7 and decisively not zero. Ten of twelve coefficients pass
the per-coefficient bar; the two fallbacks land on `k=1`, never `k=0`; every coefficient beats
the stale-baseline endpoint on all three held-out pairs. Coefficients are conditional on
`n0 = 200 PA` / `n0 = 50 IP` -- the product `k*w` is identified, `k` alone is not.

**The three things increment 2 must not miss** (full list in the finding's B.10):
1. Use `fold.gate_ramp`, not `gate_mask`, on the serve path. The shipped playing-time
   coefficients make the hard gate a 78.7% (hitter) / 44.6% (pitcher) cliff across two plate
   appearances.
2. Resolve the playing-time level term. The fit sample's mean PA residual is **-91, not the
   spec's +58** -- ZiPS over-projects playing time on this population. An unshipped intercept
   would cut hitter-PA error 19%; whether it belongs in production is the open question.
3. Re-fit `sb_pa` first when the 2025->2026 pair opens. It is the least stable coefficient
   (spread 0.257, CI [0.480, 0.793]) and is reported as provisional.

Everything below is the original handoff, kept for context on how the work was framed. Its
"resume at Task 3" instructions are spent.

## What this feature is

Yahoo 5x5 roto keeper league. Every team keeps exactly **3 players, mandatory**, and the
keepers consume draft rounds 1-3. The tool must project 2027 value to decide which 3, and
compare the owner's trio against the other nine teams'.

**The problem:** the ZiPS 2027/2028 projections on disk were generated 2026-03-25 and know
nothing about the 2026 season. FanGraphs has not regenerated them (verified: a fresh
2026-07-27 download is identical). So we must fold 2026 into a stale baseline ourselves.

**The whole feature reduces to one number**: how much of a season's surprise carries
forward? `updated_2027 = ZiPS_2027 + k * shrink * (actual_2026 - ZiPS_2026)`. `k = 0` ignores
2026; `k = 1` transfers everything. **Increment 1 (this work) measures `k`.** Increment 2
builds the actual keeper rankings on top of it.

## Where the work lives

| Path | What |
|---|---|
| `docs/superpowers/specs/2026-07-27-keeper-value-definition-design.md` | The design. Read sections 5.2-5.5 and 6 before touching the estimator. |
| `docs/superpowers/plans/2026-07-27-keeper-calibration-study.md` | **The instruction set.** 8 tasks, literal code, TDD steps. |
| `.superpowers/sdd/2026-07-27-keeper-calibration-study/progress.md` | Ledger of what's done + deferred minor findings. Gitignored. |
| `src/fantasy_baseball/keepers/` | Where the code goes. |
| `tests/test_keepers/` | Where the tests go. |

## Done so far

**Task 1** (`e35cf7a3`) -- `keepers/actuals.py`: `innings_to_float`, `coerce_numeric`.
The MLB Stats API returns innings as a **baseball-notation string**: `"5.1"` means 5 1/3
innings, not 5.1. Naive `float()` is wrong. Verified against a live pull.

**Task 2** (`df0d3dd8`) -- `keepers/actuals.py`: `normalize_hitting`, `normalize_pitching`.
Turns a raw API frame into the canonical schema below. 14 tests pass; full suite 2520 pass.

Both tasks passed an independent code review (spec compliance + quality), with no Critical or
Important findings. Task 2's review verified the three subtle-failure points empirically rather
than by eye: `h_ab` really divides by AB (not PA), a zero denominator really produces NaN and
cannot produce 0.0, and id-less rows are dropped rather than coerced. It also confirmed the
gates independently (`ruff`, `ruff format`, `mypy`, `vulture` all clean).

**So Tasks 1-2 are trustworthy to build on.** Deferred minor findings are listed at the bottom
of this file and in the ledger.

## Remaining: Tasks 3-8

Work straight through the plan file. Summary of what each does:

3. **`keepers/vintages.py`** -- load a ZiPS CSV vintage from disk, decompose to the canonical
   schema. Also renames `_safe_ratio` -> `safe_ratio` in `actuals.py` (**11 call sites**).
4. **`keepers/fold.py`** -- `shrink`, `gate_mask`, `fold_rates`, `reconstruct_hitter`,
   `reconstruct_pitcher`. Pure functions. **Increment 2 reuses this module unchanged.**
5. **`keepers/calibration.py`** part 1 -- `YearPair`, `build_pairs`, `survivorship`. Then
   run the measurement step for BOTH hitters and pitchers; the spec flags the pitcher side
   as an unmeasured gap.
6. **`keepers/calibration.py`** part 2 -- pre-register the metric in writing FIRST, then
   `Estimator`/`Fitted` protocols, `ZeroTransfer`/`FullTransfer` endpoints, `weighted_mse`,
   `leave_one_out`. Requires adding an `ARG002` per-file-ignore to `pyproject.toml`.
7. **Choose and implement the estimator.** THE DELIVERABLE -- see below.
8. **`scripts/keeper_calibration.py`** + run it for real + write the finding document.

## The canonical schema (everything depends on it)

Frames indexed by `mlbam_id` (int), carrying rate columns plus one playing-time column:

```
hitters: pa   + hr_pa, r_pa, rbi_pa, sb_pa, h_ab, ab_pa
pitchers: ip  + k_ip, w_ip, er_ip, bb_ip, h_ip
```

Both MLB actuals and ZiPS vintages decompose to this, which is what makes them differenceable.

## Task 7 is deliberately open -- read this before doing it

**Choosing the estimator IS the deliverable.** The spec deliberately does NOT specify the
regression form. Three earlier attempts to pin it down in prose were each found broken by
review, which is why it was descoped to "decide against real data."

The plan's Task 7 lists 12 requirements the estimator must satisfy. The three that killed
previous attempts:

- **Requirement 1 -- calibration and production must apply the SAME functional form.** A term
  that exists only in calibration must be justified and its production value stated. One
  earlier attempt added a free scale term `a`; because `a*Z + k*(A-Z) = (a-k)*Z + k*A`, a free
  `a` made `k` degenerate into the plain OLS slope on `actual_Y`, destroying the meaning of
  the `k=0` / `k=1` endpoints and the acceptance test with it.
- **Requirement 12 -- the playing-time residual has a large systematic MEAN that is not
  surprise.** ZiPS hedges playing time pool-wide (2025 regulars ran +58 mean PA versus
  projection). A single multiplicative coefficient cannot separate a level offset from signal.
  Both failed estimators died here.
- **Requirement 7 -- do not ship a coefficient that amplifies residuals** on the strength of
  held-out error alone.

Also: **playing time is the twelfth coefficient**, fit via
`leave_one_out(..., column=PT_COL[player_type], shrunk=False)`. The `shrunk=False` is not
optional -- the shrink damps noisy RATE observations, and applying it to the PT residual would
damp an injury signal in proportion to the playing time the injury suppressed.

## Traps -- all of these are verified, do not re-derive

**Data facts:**
- Usable year-pairs are exactly **2022->2023, 2023->2024, 2024->2025**. A 2025 pair needs a
  complete 2026 season (it is July 2026); a 2021 pair has no ZiPS vintage on disk.
- ZiPS 2027/2028 have **`SV` populated in 0 of 1838 rows.** Saves do not exist in the
  out-years. `safe_float` silently coerces NaN to 0.0, so this fails silently. Relievers are
  therefore excluded from the out-year ranking entirely (increment 2's problem, not increment 1's).
- **95 of 1838 pitchers** in ZiPS 2027 have no ZiPS 2026 counterpart. All 1901 hitters do.
- **No `Age` column** in any of the seven ZiPS vintages (2022-2028).
- Mean `AB/PA` in ZiPS 2027 is **0.8977**.
- 2028 has 14 pitcher rows at `IP=0` and 3 hitter rows at `PA=0`; `0/0` guards are required.

**Correctness traps:**
1. **The calibration base must NOT already know year Y.** Every ZiPS file is a preseason
   projection for its own year, so `ZiPS_{Y+1}` already absorbed year Y. Fitting against it
   would drive `k` toward zero *by construction* and ship the conclusion that 2026 tells us
   nothing about 2027 -- the opposite of the truth. **Use `ZiPS_Y` as the base.** This is the
   single most important idea in the whole design.
2. **`build_pairs` membership must be `zips INTERSECT actual_Y` only** -- do NOT also intersect
   year Y+1. Intersecting all three preconditions the sample on having survived. Verified:
   the three-way version yields survival 0.842/0.870/0.868, the correct two-way version yields
   **0.755/0.777/0.795**. If you measure 84-87%, your membership is wrong.
3. **Each rate multiplies its OWN denominator.** `H_2027 = AB_2027 * (H/AB)`, not
   `PA * (H/AB)`. Getting this wrong inflates AVG by 1/0.8977 = 1.114 -- a .250 hitter scores
   .278. Derive `AB` from `PA` first so `AB <= PA` is structural.
4. **Zero playing time must yield NaN rates, never 0.0.** A 0.0 rate reads downstream as a real
   observation of zero. And note `0 * NaN = NaN`, so a zero shrink does NOT rescue a NaN rate --
   guard explicitly.
5. **Absence from the MLB leaderboard is not zero playing time.** `playerPool=all` returns MLB
   players only, so a player who spent the year in AAA is *missing*, not present with 0 PA.
   Absence must pass through unfolded, never become a large negative residual.
6. **Do not route an in-progress season through `fetch_or_cache`** (`keepers/cache.py`) -- it
   never invalidates, so the first mid-season pull freezes permanently.

**Repo rules (from CLAUDE.md):**
- **ASCII only** in source, comments, and anything reaching `print()`. Windows box, cp1252
  stdout, one non-ASCII glyph raises `UnicodeEncodeError`. This bit me writing the plan itself.
- `src/fantasy_baseball/keepers/` is under **mypy** with `warn_return_any` -- annotate locals
  holding untyped pandas returns (`result: pd.Series = ...`). See `keepers/mlb_stats.py`.
- Gates before every commit: `pytest`, `ruff check .`, `ruff format --check .`, `mypy`, `vulture`.
- Increment 1 is **standalone**: no imports from `fantasy_baseball.sgp`, `.draft`, `.models`,
  or `.data`. (Importing `fantasy_baseball.keepers.*` is fine.)
- Tests are the guardrail -- fix the code, not the test.
- Always work on a feature branch; ask before merging to main.

## Known measurements already taken

Hitters at 100 PA, correct (non-preconditioned) membership:

```
2022->2023  n_matched ~525  n_in_year 465  survival 0.755
2023->2024  n_matched ~510  n_in_year 458  survival 0.777
2024->2025  n_matched ~535  n_in_year 454  survival 0.795
```

Pitchers (measured with the BUGGY preconditioned membership, so treat as upper bounds and
re-measure): ~560-575 matched per pair; at `IP >= 50` about 311-313 in-year, survival ~69%;
at `IP >= 100` about 116-129 in-year, survival ~61-63%. So roughly **930 pitcher rows across
three pairs at IP>=50 versus ~360 at IP>=100** -- the threshold choice is the live decision,
and the plan makes reporting it a blocking first step.

## Deferred minor findings (nothing blocking)

- `innings_to_float` validates only the first fractional digit, so `"5.15"` parses as 5 1/3
  rather than raising. Contained today because the ZiPS path uses `astype(float)`, never this
  function. Worth tightening if a decimal-IP source is ever routed through it.
- `HITTER_RATES`'s tuple order does not match the column order `normalize_hitting` emits. Safe
  while consumers index by name (which the plan does), but a positional `zip()` would silently
  mis-pair. Keep in mind at Task 3, which consumes these constants.
- Test coverage gaps inherited from the plan's own test code: `r_pa`/`rbi_pa` are never asserted
  for hitters, `w_ip`/`bb_ip`/`h_ip` never for pitchers, the zero-PT test asserts NaN only on
  `hr_pa`, and there is no pitcher-side drop-row or zero-IP test. The shared helpers make the
  risk low and the reviewer confirmed the uncovered cases behave correctly at runtime, but if
  you touch this schema, widen the assertions.

**Already checked -- do not re-litigate:** duplicate `mlbam_id` from traded players is
impossible. The MLB season endpoint aggregates a traded player into one split; verified on a
live 2025 pull (765 hitter rows / 765 unique ids, 873 / 873 for pitchers). No dedup needed.

## How to resume

```bash
cd C:/Users/alden/FantasyBaseball
git checkout feat/266-keeper-calibration
pytest tests/test_keepers/ -v          # should be green
```

Then open the plan file and start at Task 3. The plan contains the literal code for Tasks 3-6;
Task 7 is the open one; Task 8 wires it together and runs it.

Nothing is pushed and no PR exists yet. Increment 2 (the actual keeper rankings, par curve,
cross-team table) is a separate future effort -- spec sections 2, 5.1 and 7 cover it, including
a scoring-path bug it must fix (`calculate_var` routes the SP/RP floor by projected IP, so 418
of 727 ZiPS-2027 starters would be scored against the reliever floor; pass `role_ip`).
