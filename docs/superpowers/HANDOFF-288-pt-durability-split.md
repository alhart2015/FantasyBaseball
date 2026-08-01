# HANDOFF: split playing-time/durability out of the keeper `luck` family

- **Issue:** #288 (https://github.com/alhart2015/FantasyBaseball/issues/288)
- **Date:** 2026-08-01
- **Status:** SUPERSEDED 2026-08-01 by
  `docs/superpowers/specs/2026-08-01-pt-durability-split-design.md` (approved
  design, ready for planning). Read the spec, not this file -- it corrects three
  things recorded below (decision 1's presentation reading, Soto's PA framing,
  and why option C is dead). This file is kept only for the deep-dive raw data
  and the how-we-got-here narrative.

## How we got here

Hart asked for the league-wide keeper top-30 at out-year (`future`) weights
0.2/0.4/0.6/0.8, then deep-dived six players. Three results struck him as model
artifacts rather than truths, and he was right:

| player | board rank | luck-OFF rank | move | why it smells |
| --- | --- | --- | --- | --- |
| Juan Soto | 21 | **4** | -17 | perennial MVP bat rated outside the top 20 |
| Sal Stewart | 3 | **19** | +16 | top-3 keeper on a 116 wRC+ and mediocre out-year ZiPS |
| CJ Abrams | 11 | **30** | +19 | monster year built on +.031 AVG-over-xBA |
| Junior Caminero | 7 | 7 | 0 | control -- luck-neutral, rank is real |
| James Wood | 2 | 2 | 0 | control -- luck-neutral, rank is real |
| Julio Rodriguez | 24 | 26 | +2 | low rank is REAL (66 skill pct, 106 wRC+); only ZiPS 17.0 saves him |

"luck-OFF" = re-rank with the `luck` family weight zeroed. All three surprises are
`luck`-family artifacts. Raw evidence for each player is in the deep-dive card below.

**The smoking gun: Soto had 359 PA in 2026.** His SGP (8.8, value_pct 80) is
depressed by missed time, so `luck = value_pct - skill_pct` = **-16** despite a
96th-percentile skill line (.297 xBA, .422 xwOBA, 157 wRC+). The model charges him
for an injury as if it were a permanent demerit.

Hart's framing, which drove this whole thread:

> "Surely being injury prone is real signal but that seems like its own model and
> not something we should lump together with luck."

## Finding 1: the 0.8 `luck` weight is an algebra artifact, not a claim about luck

`luck = value_pct - skill_pct` **exactly**, so `skill + 0.8*luck` is a rewrite of:

```
0.8 * value  +  0.2 * skill
```

The shipped hitter composite `(1.0*skill + 0.8*luck - 0.2*batted_ball + 0.2*future + 0.3*age) / 2.1`
is therefore really:

```
(0.8*value + 0.2*skill - 0.2*batted_ball + 0.2*future + 0.3*age) / 2.1
```

**38% of a keeper's grade is last year's raw roto line; only 10% is his peripherals.**
SGP is a counting-stat total, so that 38% is rate-times-volume welded together. Soto
is graded 38% on a 359-PA counting line. This is structural, not a tuning problem.

## Finding 2: raw playing time is a BETTER durability signal than `luck`

From `python scripts/keeper_rankings.py --study` (mean Spearman rho over the 3
transitions 2022->23, 2023->24, 2024->25), run 2026-08-01:

```
HITTERS            -> next SGP   -> next PT   -> next RATE
  last-yr value        0.654       0.639        0.514
  skills               0.482       0.464        0.381
  luck                 0.372       0.376        0.282
  playing time         0.575       0.607        0.377
  batted-ball          0.049       0.049        0.030
  age (younger)        0.190       0.164        0.208
  future (stale)       0.521       0.507        0.415

PITCHERS           -> next SGP   -> next PT   -> next RATE
  last-yr value        0.464       0.412        0.342
  skills               0.300       0.081        0.464
  luck                 0.241       0.383       -0.041
  playing time         0.309       0.465       -0.008
  batted-ball         -0.019      -0.017       -0.004
  age (younger)        0.056       0.081        0.004
  future (stale)       0.352       0.374        0.149
```

Read: `luck` is a **degraded proxy for playing time**. Raw PT beats it at predicting
next-year PT in both pools (0.607 vs 0.376 hitters; 0.465 vs 0.383 pitchers). For
pitchers `luck` predicts next-year RATE at **-0.041** -- literally zero. The
`composite.py` docstring's "mostly a volume signal wearing a misleading name" is
measurably true.

## Finding 3: "just add a `pt` family" was already tried and LOST -- don't repeat it

`scripts/keeper_rankings.py` `CANDIDATES` and
`docs/superpowers/specs/2026-07-30-decompose-luck-design.md` (issue #277) already
ran this bake-off:

```
baseline           skill, luck, future, age
A: pt+luck         skill, pt, luck, future, age      <- LOST holdout
B: pt+batted_ball  skill, pt, batted_ball, future, age  <- LOST holdout
C: luck-batted_ball skill, luck, batted_ball, future, age  <- SHIPPED
```

A lost because `pt` and `luck` are the same signal -- bolting one next to the other
is collinear and buys nothing. **`pt_pct` is already materialized on every row**
(`_qualified_families`, keeper_rankings.py:331) and `"pt"` is already in
`KNOWN_FAMILIES` (composite.py:102). The machinery exists; the naive use of it is
spent.

## The proposed direction (NOT yet approved by Hart)

**Reparameterize, do not add.** Swap `{skill, luck}` for `{rate, durability}`:

- **`rate`** -- PT-neutral talent. SGP-per-PA / SGP-per-IP, or lean harder on the
  existing `skill` peripherals. An injury-shortened elite-rate season stops being
  punished.
- **`durability`** -- its own forward-looking availability model, kept as a
  SEPARATE dimension (see "Decisions" below).

Nothing collinear, and each term means what its name says.

**Why this has headroom the #277 bake-off never had:** same-season PT predicts
next-season PT at only **0.607 (hitters) / 0.465 (pitchers)** -- one noisy
observation, no regression to the mean, no history. That is the ceiling the current
model is pinned at. A multi-season durability estimate is genuinely new information.
It is also exactly what fixes Soto: five years of 650-PA seasons say "durable bat
who missed time," where one year of 359 PA says "declining asset."

Predicted effects on the three surprises (hypotheses to verify, not results):
- **Soto** rises -- elite rate, and multi-season history says he is durable.
- **Stewart** falls -- his value was volume plus role on a 74th-pct skill line;
  one season of PT is thin evidence at age 22.
- **Abrams** falls -- rate inflated by +.031 AVG-over-xBA, which `batted_ball`
  already claws back and a rate-based term would not reward in the first place.

## Decisions taken so far

1. **Availability stays a SEPARATE dimension from talent** -- two numbers shown
   side by side (e.g. "elite bat, ~60% to clear 140 games"), NOT folded into one
   blended keeper value via a PT multiplier. Hart: "They should be separate."
   Rationale: durability risk is a different KIND of thing than talent and you may
   weigh it differently for a win-now keeper vs a rebuild stash.

## OPEN QUESTION -- this is exactly where the discussion stopped

**What is the durability/availability model built from?** Three options were about
to be put to Hart; he stopped to ask for clarification first, so the question may
need reframing before it is re-asked. Do not assume he has seen these.

- **(A) Revealed PT + projections [was my recommendation].** Recency-weighted
  multi-season PA/IP from the BBRef pulls already cached, regressed to a pool mean,
  blended with ZiPS projected PA/IP so rookies and role changes are not judged on
  absent history. Age/debut-aware. No new data source. Argument: PA *is* the
  outcome of injuries, so it measures durability directly rather than modeling its
  mechanism.
- **(B) Revealed PT + real IL history.** Adds actual IL stints -- days lost, injury
  type, recurrence. Closer to a true injury-propensity model (the deferred
  #251/#253 thread), but needs a new scraped source with its own ingest, caching
  and backfill, and its marginal value over "how many PA did he actually get" is
  unproven.
- **(C) Projected PT only.** Use ZiPS projected PA/IP and skip history. Cheapest,
  but the out-year ZiPS files are two years forward from their information set
  (2027 ZiPS has never seen 2026) -- which is why `future` is already
  weight-discounted for staleness.

### Known traps for whichever option wins

- **The rookie trap.** A player with no MLB history looks maximally fragile under
  naive multi-season PT. Sal Stewart is the live example. Needs debut/service-time
  awareness or a projection blend -- this is the main argument for (A) over a
  history-only design.
- **Durability vs role.** Low PT can mean fragile OR benched-for-being-bad. Both
  cut next-year PT, so for a keeper board it may not matter, but it matters for how
  the number is LABELED to the user given decision 1 above.
- **Backtest lookback cost.** Transitions fit on 2022, 2023; holdout 2024
  (`BACKTEST_FIT_YEARS`, `BACKTEST_HOLDOUT`). Cached BBRef seasons are **2022-2026
  only** (`data/cache/keeper_skills/raw_2022 .. raw_2026`). A 3-year lookback
  feature for the 2022 transition needs 2019-2021, which is NOT cached. Either
  fetch them (`keepers/bref.py` wraps `pybaseball.batting_stats_bref(year)` /
  `pitching_stats_bref(year)` behind `fetch_or_cache` -- one call per year), or
  shorten the lookback, or lose fit years. **Fetching 2019-2021 also drags in
  COVID-shortened 2020** (60 games), which will wreck any raw PA-history feature
  unless explicitly rescaled or excluded.
- **`MIN_PT` floor hides the worst cases.** `MIN_PT = {"hitter": 250, "pitcher": 50}`.
  A star who missed most of a season never reaches the board at all, so the
  durability model cannot be evaluated on the players it most needs to get right.
- **`--backtest` must decide the winner, not a docstring.** The repo's stated rule
  (#272 history): regenerable evidence via a script flag, never an asserted
  constant. Any new parameterization ships only if it beats baseline holdout by
  more than the pool noise AND clears the skill-only null floor.

## Code map

- `src/fantasy_baseball/keepers/composite.py` -- `FITTED_WEIGHTS`
  (hitter `(1.0, 0.8, -0.2, 0.2, 0.3)`), `FAMILIES`, `KNOWN_FAMILIES` (already
  contains `"pt"`), `luck()`, `batted_ball()`, `composite()`. The module docstring
  is the argument-of-record for every family; read it first.
- `scripts/keeper_rankings.py` -- 1562 lines. Key entry points:
  - `_qualified_families` (:314) materializes `value_pct`, `skill_pct`, `luck_pct`,
    `pt_pct`, `batted_ball_pct`, `age_pct`. **`pt_pct` already exists.**
  - `_transition` (:554) builds features in year T against year T+1, and ALREADY
    computes `target_pt` and `target_rate` alongside `target`. A durability model
    can be fit against `target_pt` with no new plumbing.
  - `_FAMILY_GRID` / `CANDIDATES` (:581-597) -- the bake-off. `_GRID_PT` exists.
  - `run_backtest` (:680), `run_study` (:1169), `run_fit` (:869).
  - Constants: `MIN_PT` (:154), `BACKTEST_FIT_YEARS`/`BACKTEST_HOLDOUT` (:179-180),
    `SKILLS_DIR` (:149).
- `src/fantasy_baseball/keepers/bref.py` -- the BBRef fetchers, for extending years.
- `docs/superpowers/specs/2026-07-30-decompose-luck-design.md` -- the #277 spec that
  produced the current `batted_ball` claw-back. Read it before proposing anything;
  it is the direct predecessor and it already burned the obvious approach.
- `docs/superpowers/keeper-277-verdict-2026-07-30.md` -- that bake-off's verdict.

## Reproducing the evidence

```bash
cd C:/Users/HartAlden/FantasyBaseball
python scripts/keeper_rankings.py --study      # the correlation tables above
python scripts/keeper_rankings.py --backtest   # the candidate bake-off + watchlist
```

Scratchpad scripts from this session (session-scoped temp dir, copy them out if
still needed):

```
C:\Users\HARTAL~1\AppData\Local\Temp\claude\C--Users-HartAlden-FantasyBaseball\2caeab62-89c5-4c61-a759-7def60ba8e6c\scratchpad\
  keeper_deepdive.py        # per-player cards + luck-OFF counterfactual rank
  keeper_outyear_sweep.py   # top-30 boards at future weight 0.2/0.4/0.6/0.8
  caminero_etal_sweep.py    # same sweep, narrowed to the six players
```

`keeper_deepdive.py` is the one worth keeping -- it is how the luck-OFF table above
was produced. Note the board column is `rank`, not `rk`.

### Deep-dive raw data (2026 hitter board, 233 qualified, median SGP 6.7)

```
Juan Soto        OF  age 27  PA 359  SGP  8.8 (value_pct 80)  AVG .283 xBA .297 (-.014)  wRC+ 157  xwOBA .422
  skill 96  luck -16  batted_ball 25  future 98  age 58    ZiPS 2027 17.6  2028 16.9
  contributions: skill +0.96  luck -0.13  bb -0.05  future +0.20  age +0.17
  rank 21  ->  luck-OFF rank 4

Sal Stewart      1B  age 22  PA 463  SGP 10.8 (value_pct 96)  AVG .253 xBA .249 (+.004)  wRC+ 116
  skill 74  luck +22  batted_ball 56  future 90  age 97     ZiPS 2027 12.9  2028 13.5
  rank 3  ->  luck-OFF rank 19

CJ Abrams        SS  age 25  PA 449  SGP 14.1 (value_pct 100) AVG .292 xBA .261 (+.031)  wRC+ 151
  skill 79  luck +21  batted_ball 93  future 93  age 79     ZiPS 2027 14.4  2028 14.2
  contributions: bb -0.19  luck +0.17
  rank 11  ->  luck-OFF rank 30
```

## Next steps for the picking-up session

1. Re-open the brainstorming skill and resume at "ask clarifying questions."
2. Ask Hart what he wanted to clarify about the durability-source question above --
   he rejected the multiple-choice framing to add context first. Let him talk before
   re-asking.
3. Settle the durability source (A/B/C or something he proposes).
4. Then settle the `rate` side: is it SGP-per-PA, or just lean on the existing
   `skill` peripherals? These are close but not identical -- `skill` is peripherals
   only, SGP-per-PA is production-per-opportunity and still carries batted-ball luck.
5. Then how the two dimensions are PRESENTED, given decision 1 (two columns? a
   talent rank plus a durability tier/flag?).
6. Write the spec to `docs/superpowers/specs/2026-XX-XX-pt-durability-split-design.md`,
   commit, self-review, get Hart's review, then invoke `writing-plans`.

Expect this to be several PRs, not one -- Hart said up front: "I expect this to take
a while to fully solve."
