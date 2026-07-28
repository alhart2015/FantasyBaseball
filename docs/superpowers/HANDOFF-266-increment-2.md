# Handoff: #266 increment 2 (keeper rankings)

Written 2026-07-28 by the session that finished increment 1. Everything here is either
(a) a decision that session made which is not written down anywhere else, or (b) a fact
it established by running code. The design documents cover the rest; this file exists so
you do not re-derive or re-litigate what is already settled.

## READ THIS FIRST: where the documents are

**If PR #268 is unmerged, the finding document is not on `main`.**

```bash
git checkout feat/266-keeper-calibration    # or merge #268 first
```

| Path | What |
|---|---|
| `docs/superpowers/specs/2026-07-27-keeper-value-definition-design.md` | The design. Sections 2, 5.1, 5.6 and 7 are increment 2's scope. |
| `docs/superpowers/keeper-calibration-finding-2026-07-27.md` | Increment 1's result. **Section B.10 is your task list.** Part A is the pre-registration; Part B the measured results. |
| `docs/superpowers/HANDOFF-266-keeper-calibration.md` | Increment 1's handoff. Historical now; its "resume at Task 3" is spent. |
| `src/fantasy_baseball/keepers/coefficients.py` | The shipped coefficients. Import `POLICIES`; do not retype them. |
| `src/fantasy_baseball/keepers/fold.py` | The fold primitives. Reused unchanged. |

Note B.10's items are misnumbered `1,2,3,4,5,7,6` -- item 7 (saves) was inserted late.
There are seven items, not six.

## The state of things

- Increment 1 is complete, reviewed through five `/loop-review` iterations to a clean
  pass, and open as **PR #268** (29 commits, unmerged as of writing).
- **Nothing in `main` imports `keepers.coefficients`.** Merging #268 changes no runtime
  behaviour. Increment 2 is what makes it do anything.
- **Issue #269** is an open bug in the *existing* keeper ranker that increment 2 will
  hit immediately. See "The duplicate-player problem" below.
- There is **no GitHub issue for increment 2**. #266's body describes the definition
  deliverable (now done) and says "implementation of that approach is a follow-up".

## Do this first, before writing any ranking code

**B.10 item 5 is a gate, not a task.** It is listed fifth and reads as equal-weight; it
is not. The study established that the fold beats *ignoring the season entirely*
(`k=0`). It never compared against what `main` actually ships.

Since PR #259, `src/fantasy_baseball/analysis/keeper_value.py` computes out-years as:

```
line = anchor x (ZiPS_{Y+1} / ZiPS_Y)                     # _scale_line
line = (1 - lam) * line + lam * ZiPS_{Y+1}                # lam = DEFAULT_OUT_YEAR_REGRESSION = 0.6
```

with `DEFAULT_PT_HEAL_CAP = 2.0` (up-only playing-time heal) and the script defaulting to
`--anchor current`. That incumbent already carries roughly 40% of the realized-season
signal per stat.

**If the fold does not beat that, most of B.10 is moot work.** Measure it before
building the par curve or the cross-team table.

## Pre-register the acceptance bar BEFORE the first comparison

Increment 1's single highest-value process decision was writing the metric, the weights
and the gates into the finding document and committing them *before* any fitting code
existed. That discipline caught a real bug: it exposed that `build_pairs` was leaving the
target playing-time column NaN, which silently deleted every non-survivor from the
playing-time fit.

Increment 2 has no equivalent and needs one. Spec 6.6 says explicitly that increment 1's
bar was measured on the rate/PT scale while the feature consumes VAR **rank**, and that
rate error can improve while ranking degrades. So write down, and commit, before looking
at any result:

- which rank metric (Spearman on the full pool? top-N overlap? sum of |rank delta| over
  the 243 rostered players? something that weights the top of the board more?)
- over which universe (full 3,739-row pool, the 243 rostered players, or the top N)
- against which baselines (at minimum: the #259 incumbent, and raw ZiPS 2027 unfolded)
- what "passing" is, per player type, and whether it is per-coefficient like increment 1
  or global

Do not pick these after seeing a result you like.

## Undocumented findings from the prior session

### 1. Which per-stat coefficient differences are real

The finding reports thirteen coefficients with confidence intervals but never says which
differences are statistically distinguishable. Measured afterwards, from the committed
CSVs (weighted heterogeneity / Q test, and pairwise z on the reported CIs):

```
                 Q      df    p          I^2    verdict
hitters        24.4      6    0.00043     75%   single pooled k REJECTED
pitchers       95.4      5    4.9e-19     95%   single pooled k REJECTED
```

**So per-stat coefficients are correct and pooling to one number would be wrong.** But
the individual values are not all precise -- typical SE is ~0.05, so each `k` carries
about +/- 0.10:

- **Hitters are roughly three tiers, not six distinct values.** `ab_pa` (0.687) is high,
  `h_ab` (0.428) is low, and `hr_pa` / `r_pa` / `rbi_pa` / `sb_pa` are not individually
  distinguishable from one another. Treating 0.494 vs 0.531 as a meaningful difference
  is reading noise.
- **Pitchers genuinely separate.** `k_ip` (0.970) differs from all five others;
  `er_ip` (0.343) vs `bb_ip` (0.697) is real.

Caveat that runs in your favour: all coefficients for a player type are fit on the same
rows, so the estimates are positively correlated and the pairwise test is conservative.
Pairs it calls distinguishable are safe; pairs it does not may separate under a proper
covariance treatment.

**External validity check worth knowing:** the coefficient ordering reproduces known
year-over-year stat reliability -- strikeout rate persists most, ERA least; `AB/PA`
persists, `H/AB` (BABIP-driven) regresses hard. That is independent evidence the per-stat
structure is signal, not noise-fitting.

If you want to be conservative, partial-pooling the four indistinguishable hitter rates
toward their common mean is defensible. Shipping them as-is is also defensible. What is
**not** defensible is claiming each of the six hitter rates has its own measured optimum.

### 2. The existing keeper path, as actually run

Increment 2 will either extend or replace `scripts/keeper_value.py`. Facts established by
running it:

- Entry point is `scripts/keeper_value.py`; the library is
  `src/fantasy_baseball/analysis/keeper_value.py`.
- `build_results(...)` returns `(results, candidate_ids)`. Introspect its signature rather
  than assuming; it takes anchor / horizon / base_year / out_year_regression / pt_heal_cap.
- **`KeeperValueResult.player_id` is `fg_id::player_type`, NOT `name::player_type`.**
  This is a trap -- the repo's convention elsewhere is `name::player_type`, and assuming
  it here produces a silent zero-match join. Example value: `'27769::hitter'`.
- Fields are `player_id, name, per_year_var, total, flags, pct_from_out_years,
  pct_from_saves`. There is no `player_type` attribute; split it off `player_id`.
- `--anchor current` reads `cache:full_season_projections` from Upstash and is
  **read-only** (only `kv.get`, verified -- no writes anywhere in the script or library).
  `build_explicit_upstash_kv()` works without setting `RENDER=true`.
- The run prints `[keeper-value] 1738 current-blob players absent from the preseason
  board (skipped; see spec follow-up)`. Pre-existing, already flagged in the code.
- The `*` markers in its output are a hardcoded `CANDIDATES` list in the script, **not**
  roster membership.

### 3. Joining to rosters

Spec section 7 wants the owner's 25 plus nine opponents (243 players). Shapes verified
against the live cache:

- `cache:roster` -> a **list** of 25 dicts, each with `name`, `player_type`, `positions`,
  `team`, `player_id` (a **Yahoo** id, e.g. 11836), `rest_of_season`,
  `full_season_projection`.
- `cache:opp_rosters` -> a **dict** of 9 team names to rosters. Team names as of writing:
  Boston Estrellas, Hello Peanuts!, Jon's Underdogs, Send in the Cavalli, SkeleThor,
  Spacemen, Springfield Isotopes, Tortured Baseball Department, Work in Progress.

**The join must be by normalized name, not id.** Roster carries Yahoo ids; the board
carries FanGraphs ids. Use `fantasy_baseball.sgp.rankings.rank_key(name, player_type)`,
which lowercases and strips accents -- `rank_key("Julio Rodriguez", "hitter")` and the
accented spelling both give `'julio rodriguez::hitter'`. With that key, all 25 roster
players matched the board with zero misses.

### 4. The duplicate-player problem (issue #269)

You will hit this the moment you build a cross-team table. Measured on the 3,709-row pool:
3,690 distinct `normalized_name::player_type`, so **19 duplicated names**. They are two
different problems and must not be treated the same way:

- **6 are one player under two ids** -- the real bug. Identical values; signature is a
  real FanGraphs id paired with an `sa`-prefixed prospect placeholder (1,934 of 3,709
  rows carry an `sa` id). Three are in keeper range: Mason Miller (`31757` /
  `sa3023658`), Cade Smith (`27867` / `sa3023371`), Miguel Vargas (`20178` /
  `sa3016193`).
- **13 are genuinely different people sharing a normalized name.** Do NOT merge these.
  The clearest is `max muncy::hitter` -- `29779` (Athletics) and `13301` (Dodgers) are
  two real players.

Spec 5.5 already requires dedupe by MLBAMID for trio selection and the par ordinal; that
is the right axis, because it separates the two groups by identity rather than by name.
A fix that collapses on normalized name alone merges two real players.

Related and already documented in spec 5.6: `cache:positions` is keyed by **bare
normalized name** (706 entries against a 3,739-row pool, ~18.8% match), so both groups
collide there too and can route a row to the wrong replacement floor.

### 5. Smaller things spotted, not filed

- `pct_from_out_years` renders as **-49%** for Sonny Gray. A share cannot be negative;
  the calculation appears not to handle a negative out-year value. Increment 2 touches
  this code. Not filed as an issue -- decide whether it is news before opening one.
- `fold.reconstruct_pitcher` emits no `sv`, but `analysis.keeper_value.PITCHER_FIELDS`
  scores `sv`. A missing column reads as zero saves. This is B.10 item 7.

## Traps -- verified, do not re-derive

**From increment 1, still live:**

- The coefficients are conditional on `n0` = 200 PA / 50 IP. Only the product `k * w` is
  identified; `k` rises with `n0` (hr_pa 0.407 at n0=100, 0.494 at 200, 0.655 at 400)
  while held-out error does not move. Using these `k` with a different `n0` is meaningless.
- Use `FoldPolicy.serve_weights`, never hand-assembled weights. It is the only place three
  rules become mechanical: ramp not hard gate; rates shrunk but playing time not; shrink
  uses the policy's own `n0`.
- `pa` ships as a `k=1` fallback standing in for an unshipped -83 PA level term. It is
  **not** a settled result -- the fitted value is 0.646 with a CI excluding 1.0. B.3 is
  the analysis; this is the largest open modelling question.
- `sb_pa` is provisional -- widest CI of any coefficient, across the 2023 rules break.
  Re-fit it first when the 2025->2026 pair opens after the 2026 season.

**Process traps this session actually fell into:**

- **Do not redirect stderr to `/dev/null` when running the study.** A run crashed
  partway, wrote one file, and looked successful; it was caught only by noticing that
  the other artifacts' mtimes were stale. Check file mtimes after any pipeline run.
- **Compare CSVs by value, not by column order.** Reordering emitted columns produces a
  frame-level diff that looks like a regression and is not. Use
  `assert_frame_equal(a[cols], b[cols], check_like=True)` over the shared columns.
- **Beware inverted slice indices when patching files with Python.** `t[start:end]` with
  `end < start` yields `""`, and `t.replace("", new)` inserts `new` between every
  character, destroying the file. Assert the slice is non-empty first.
- **`sys.stdout.reconfigure(encoding="utf-8", errors="replace")`** is required in any
  ad-hoc script that prints player names. This box is cp1252 and accented names
  (Rodriguez, Hernandez, Acuna) otherwise mangle or raise.

## Open decisions the prior session did NOT make

These are genuinely open. Do not assume a default:

1. Whether the -83 PA level term ships. Applying it beat every shipped option by ~15% on
   hitter PA held-out error, but doing so requires establishing that the level is a
   persistent ZiPS playing-time hedge (which ZiPS 2027 would also carry) rather than the
   year-Y-to-Y+1 aging gap that ZiPS 2027 has already priced. Increment 1 could not
   separate them -- it would need a vintage pair where the base is aged forward, which
   does not exist on disk.
2. Whether to partial-pool the four indistinguishable hitter rate coefficients.
3. Whether the serve-time gate should equal the fit-sample threshold. Spec 5.4 says they
   are different objects and may legitimately differ; increment 1 set both to the same
   value (100 PA / 50 IP) and said so, but did not argue they must match.
4. What happens to pool rows with no ZiPS rest-of-season line. Spec 6.4 notes roughly
   2,382 of 3,739 rows lack one; increment 1 was not required to decide.

## How to verify you have not broken increment 1

```bash
python scripts/keeper_calibration.py     # regenerates 9 artifacts under data/analysis/
git diff --stat data/analysis/           # must be empty; they are byte-identical
pytest -n auto                           # 2596 passing at handoff
ruff check . && ruff format --check . && mypy && vulture
```

`tests/test_keepers/test_coefficients.py` binds the shipped constants to the study CSVs
and **skips** if those CSVs are absent -- so if you ever untrack them, the guard vanishes
silently rather than failing. `data/analysis/.gitignore` exists to keep them tracked while
ignoring unrelated scratch output.
