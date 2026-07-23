# Keeper-Trade Generator (Consolidation Trades)

**Date:** 2026-07-23
**Status:** Approved
**Builds on:** the keeper-asset-value metric
(`docs/superpowers/specs/2026-07-22-keeper-value-design.md`, PR #254).

## Problem

The keeper metric ranks every player's multi-year keeper value, and the
leaguewide sweep shows each team's best-3 "trio-sum." The manager wants to turn
that landscape into **actionable trade suggestions**: specifically, trades that
acquire a rival's stud to upgrade his own top-3 while paying with players that
are worthless to him as keepers. There is a trade *evaluator*
(`trades/multi_trade.py::evaluate_multi_trade`) that scores a *given* trade's
2026 roto impact, but nothing *generates* candidates from the keeper landscape.

## The economic model (what the generator exploits)

In a **flat keep-3** league, only a team's **top-3 keeper values count**;
everything below its 3rd keeper is **trapped** -- unkeepable, worth 0 in keeper
terms. Two consequences drive every suggestion:

1. **The displaced keeper is free to give.** The moment Hart acquires a stud G
   that cracks his top-3, his old #3 keeper drops out of the top-3 and becomes
   trapped -- worth ~0 to Hart's keeper pool whether he trades it or not. So
   Hart can include his old #3 in the package at **zero keeper cost**, which
   makes the package far more attractive to the opponent.

2. **Hart's keeper gain is fixed by (G, his #3); the package is just the price.**
   `my_gain = value(G) - value(my #3)`, independent of what surplus Hart adds.
   So the generator gives the *minimum* trapped package that lifts the
   opponent's trio, and a 2026 guardrail stops Hart from overpaying with players
   he needs this year.

The result is a genuine win-win arbitrage: Hart converts trapped depth into a
top-end upgrade; the top-heavy opponent converts one unpairable stud into two
keepers that lift its weak 2nd/3rd slots. **Both top-3 pools improve.**

Worked example: acquire Judge (16.9) from Spacemen for Caminero (14.4) + Woo
(12.1). Hart: 49.1 -> 51.6 (+2.5). Spacemen: 33.1 -> 35.2 (+2.1).

## Objective and acceptance (decided during brainstorming)

- **Objective:** maximize Hart's top-3 keeper gain, **subject to a 2026
  guardrail** -- reject/flag any suggestion that meaningfully dents his title
  run (he leads ~98.6% ROS).
- **Acceptance model:** **keeper-mutual only.** Suggest trades where *both*
  top-3 keeper pools strictly improve. No opponent-contention / win-now
  modeling -- the manager judges who would realistically deal.
- **Trade shape:** **consolidation only** -- Hart *receives exactly one* stud
  (must crack his top-3) and *gives a package* (his displaced #3 + trapped
  surplus), package size configurable (`--max-give`, default 3).

## Algorithm

Per opponent team `T`, for each candidate stud `G` on `T`'s roster:

```
me         = rosters[my_team]                        # RosterPlayer w/ keeper_value
my_sorted  = me sorted by keeper_value desc
my_third   = my_sorted[2].keeper_value
protect    = {my_sorted[0].id, my_sorted[1].id}      # top-2 always kept
giveable   = [p in me if p.id not in protect]        # old #3 + everything below

for T, roster in rosters (T != my_team):
    T_top3_before = sum(top-3 keeper_value of roster)
    for G in roster with keeper_value(G) > my_third:      # G upgrades my top-3
        my_gain   = keeper_value(G) - my_third            # fixed
        my_after  = my_sorted[0].kv + my_sorted[1].kv + keeper_value(G)
        # keeper-viable candidate packages first (cheap; no evaluator call),
        # in preference order (fewest players, then least keeper given):
        for pkg in keeper_viable_packages(G, roster, giveable, T_top3_before):
            verdict = guardrail(give=pkg, receive=G)      # injected; see guardrail
            if verdict.ok:
                their_after = top3_sum((roster - G) + pkg)
                emit TradeSuggestion(minimal, ..., verdict)
                if sweetener: try one more giveable, emit sweetened if guardrail ok
                break                                     # first passing pkg wins
        # no keeper-viable pkg passes the guardrail -> no suggestion for (T, G)
rank all suggestions by my_gain desc
```

`keeper_viable_packages(G, roster, giveable, T_top3_before)`: yields the subsets
of `giveable` (at most `max_give`) for which the opponent's trio **strictly
improves** -- `top3_sum((roster - G) + pkg) > T_top3_before` -- ordered by fewest
players, then least total `keeper_value` given (protect Hart's better surplus for
the next deal). `giveable` is small (~5-8), `max_give` ~3, so enumeration is
trivial. The **guardrail is applied per package, in that order**, and the first
package that passes is emitted (bounding evaluator calls).

**Sweetened variant:** the emitted minimal package **+ the single next-best
giveable player**, kept only if the guardrail still passes -- a more generous
offer the opponent is likelier to accept, at a known extra 2026 cost.

## The 2026 guardrail (full `evaluate_multi_trade`, category-aware)

The guardrail keeps Hart from denting his 2026 title run for keeper gain. v1 uses
the **real** `trades/multi_trade.py::evaluate_multi_trade` -- **not** a VAR proxy
-- because roto value is category-relative: a gain in a category Hart has already
locked is worth ~0, and a loss in a contested one is full pain. A category-blind
VAR delta can mis-call exactly the borderline trades this guardrail exists for,
and category position is the toolkit's core philosophy, so correctness wins over
the small extra integration.

**Guardrail verdict** for a candidate: run `evaluate_multi_trade(...)` and return
`GuardrailResult(legal=result.legal, delta_total=result.delta_total, ok=(result.legal
and result.delta_total >= -threshold))` (default threshold `2.0` roto points).
`result.delta_total` is Hart's projected 2026 roto-point change, standings-aware.

**Input assembly (reuse the dashboard's trade evaluator).** The
`/api/evaluate-trade` route (`web/season_routes.py:1031-1099`) already assembles
every input `evaluate_multi_trade` needs from cached blobs. **Extract that
assembly into a shared helper** -- `trades/eval_inputs.py::load_trade_eval_context(
kv, config) -> TradeEvalContext` returning `hart_roster`, `opp_rosters`,
`waiver_pool` (via `trades/multi_trade.py::build_waiver_pool(hart_roster,
opp_rosters, ros_cache)`), `projected_standings`, `team_sds`, `fraction_remaining`,
`roster_slots` -- and have **both** the route and the generator use it (the
route's `_projected_from_cache` / `_team_sds_from_cache` helpers move into the
shared module). Blobs: `CacheKey.ROSTER`, `OPP_ROSTERS`, `PROJECTIONS`
(`projected_standings` / `team_sds` / `fraction_remaining`), `ROS_PROJECTIONS`.
The generator reads them **live from Upstash** (`build_explicit_upstash_kv`,
unwrap the `_data` envelope) like `injury_stress.load_mc_inputs_from_upstash`.

**Roster-legal proposal (the one real cost of "correct").** A 1-for-N
consolidation unbalances rosters, so each candidate builds a legal `TradeProposal`:
`send` = package keys, `receive` = `[G key]`, and to rebalance --
`my_adds` = Hart's `N-1` best waiver refills from `waiver_pool` (so his roster
stays full and the eval credits a replacement, not an empty slot), `opp_drops` =
the opponent's `N-1` lowest keeper-value players (whom they'd cut to fit the
package). `my_active_ids` / `opp_active_ids` are derived from the roster blobs'
active slots (verify the exact field at implementation; default to the evaluator's
own handling if absent). A candidate that can't be made legal is skipped.

**Bounding evaluator calls.** `evaluate_multi_trade` recomputes league standings,
so the pure generator receives the guardrail as an **injected callable** and
invokes it only on **keeper-viable** packages, in preference order (fewest
players / least keeper given), taking the first that passes -- ~1-2 evaluator
calls per `(T, G)`, a few dozen per run. The keeper-side viability (does the
opponent's trio improve) is decided first, cheaply, with no evaluator call.

## Components / files

- **Create** `src/fantasy_baseball/analysis/keeper_trades.py` -- **pure** logic:
  the data model, `generate_consolidation_trades(...)`, keeper-viable-package
  enumeration, top-3 helpers. Takes rosters (keeper_value only) and a **guardrail
  callable**; **no I/O**. Fully unit-testable with synthetic leagues + a mock
  guardrail.
- **Create** `src/fantasy_baseball/trades/eval_inputs.py` -- shared assembly of
  `evaluate_multi_trade`'s inputs (`load_trade_eval_context`), extracted from the
  `/api/evaluate-trade` route (including its `_projected_from_cache` /
  `_team_sds_from_cache` helpers). **Modify** `web/season_routes.py` to call it,
  so the route and the generator share one assembly path (DRY).
- **Create** `scripts/keeper_trades.py` -- I/O + orchestration: keeper values via
  the `scripts/keeper_value.py` build path, **live** Upstash blobs
  (`build_explicit_upstash_kv`, `_data`-unwrapped, as
  `injury_stress.load_mc_inputs_from_upstash` does), player<->keeper-value
  matching, and the **real `evaluate_multi_trade` guardrail** (builds a
  roster-legal proposal with auto refills/drops per candidate), ASCII render.
- **Create** `tests/test_analysis/test_keeper_trades.py` (pure generator, mock
  guardrail) and `tests/test_trades/test_eval_inputs.py` (assembly round-trips a
  fixture blob into the shape `evaluate_multi_trade` expects).

## Data model (pure module)

```python
@dataclass(frozen=True)
class RosterPlayer:
    player_id: str
    name: str
    keeper_value: float          # discounted multi-year VAR at the chosen discount

@dataclass(frozen=True)
class GuardrailResult:
    legal: bool
    delta_total: float           # Hart's projected 2026 roto-point change
    ok: bool                     # legal AND delta_total >= -threshold

# Given (package to send, player to receive), the injected guardrail returns a
# verdict. The script closes over the real evaluate_multi_trade + the threshold;
# tests inject a mock. The threshold thus lives in the callable, not the generator.
Guardrail = Callable[[list[RosterPlayer], RosterPlayer], GuardrailResult]

@dataclass(frozen=True)
class TradeSuggestion:
    target_team: str
    acquire: RosterPlayer
    give: tuple[RosterPlayer, ...]
    variant: str                 # "minimal" | "sweetened"
    my_top3_before: float
    my_top3_after: float
    my_gain: float
    their_top3_before: float
    their_top3_after: float
    their_gain: float
    guardrail: GuardrailResult

def generate_consolidation_trades(
    my_team: str,
    rosters: Mapping[str, list[RosterPlayer]],
    guardrail: Guardrail,
    *,
    max_give: int = 3,
    sweetener: bool = True,
) -> list[TradeSuggestion]: ...
```

## Player <-> keeper-value matching

Roster players (Upstash `Player` objects) are matched to keeper-value results by
**`fg_id` when available, else normalized `name::player_type`** -- the same
`sgp/rankings.py::fg_key`/`rank_key`/`lookup_rank` path the keeper report uses,
so same-name players don't collide. A roster player absent from the keeper
universe (filtered by the board's AB>=50 / IP>=10 minimums, or not in the
preseason blend) gets **`keeper_value = 0.0`** -- a legal throw-in that never
cracks a top-3. The script logs how many roster players were unmatched.

## Output (ranked by Hart's keeper gain)

```
ACQUIRE Aaron Judge (#6, kv 16.9) from Spacemen         (illustrative numbers)
  give [minimal]:  Junior Caminero (kv 14.4) + Bryan Woo (kv 12.1)
    YOU:   top-3 49.1 -> 51.6  (+2.5)   [Soto, JRod, Judge]
    THEM:  top-3 33.1 -> 35.2  (+2.1)   [Caminero, Woo, Garcia]
    2026:  roto delta -1.4  (evaluate_multi_trade; guardrail OK, threshold -2.0)
           driver: +HR/R/RBI ~0 (already maxed) vs -W/-K from shipping Woo
  give [sweetened]: + James Wood (kv 9.3)
    2026:  roto delta -3.1  (guardrail FAIL < -2.0)
```
The 2026 delta is `evaluate_multi_trade`'s standings-aware `delta_total`, not a
raw talent sum -- so gains in categories Hart has already locked count for ~0 and
losses in contested ones count in full. That's why the minimal package can pass
while the sweetener (one more shipped contributor) tips it over the threshold.

CLI: `--discount R` (single rate for the keeper ranking, default 0.80),
`--horizon N` (default 3), `--guardrail-threshold X` (default 2.0),
`--max-give N` (default 3), `--no-sweetener`. ASCII-only;
`sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the entry point.

## Edge cases

- **No stud above Hart's #3 on a team** -> that team yields no suggestions.
- **No viable package** (can't lift the opponent's trio within `max_give`, or
  every package fails the guardrail) -> skip that (T, G); do not emit.
- **Roster legality:** a 1-for-N trade unbalances rosters, so the guardrail
  builds a legal proposal (auto refills/drops, see guardrail). If no legal,
  guardrail-passing package exists for a `(T, G)`, it is skipped -- not emitted
  as illegal.
- **Unmatched roster players:** `keeper_value = 0.0` (see matching); counted and
  logged, never silently dropped.
- **Fewer than 3 keepers on a team:** top-3 sum uses however many exist; handled
  by the top-3 helper (no index error).
- **Guardrail numeric trap:** `delta_total` compared with `>=`; a `delta_total ==
  0.0` verdict is a real value, never falsy-sunk. `keeper_value == 0.0` likewise.

## Testing (pure generator, synthetic leagues)

- **Consolidation-found test:** a synthetic league where team B has one stud +
  two scrubs and Hart has surplus depth -> the generator emits "acquire stud,
  give displaced-#3 + surplus," and **both top-3 sums strictly improve**.
- **Displaced-keeper-is-free test:** `my_gain == value(G) - value(my #3)`
  regardless of which surplus fills the package.
- **Minimal-package test:** the emitted minimal package is the fewest-player
  subset that lifts the opponent's trio; adding fewer players would leave them
  not-improved.
- **Guardrail test (injected mock):** a mock guardrail that fails the first
  keeper-viable package but passes the next -> the generator skips the first and
  emits the second; a mock that fails all packages for a `(T, G)` -> no
  suggestion. Confirms the generator consults the guardrail and stops at the
  first pass (bounding calls).
- **No-target test:** a team whose best player is below Hart's #3 yields no
  suggestions.
- **Ranking test:** suggestions sort by `my_gain` desc.
- **Matching/zero-value test:** an unmatched roster player is treated as
  keeper_value 0 and never appears in a top-3.
- **`eval_inputs` round-trip test** (`tests/test_trades/`): `load_trade_eval_context`
  turns a fixture cache blob into the exact inputs `evaluate_multi_trade` accepts
  (a smoke test that the extraction preserves the route's behavior). The real
  end-to-end guardrail is exercised by running the script against live Upstash --
  not a unit test (it needs live season state), stated as such.

## Non-goals (v1)

- **Opponent acceptance / contention modeling** (keeper-mutual only, by decision).
- **Trade shapes other than consolidation** (no get-2, no general N-for-M).
- **Multi-team trades.**
- **Auto-execution** of a suggested trade (suggest only; the manager proposes it
  in Yahoo).
- **Risk modeling** (injury/role) -- still the separate deferred follow-up from
  the keeper-value spec.

## What is reused vs modified

- **Reused, not modified:** `analysis/keeper_value.py`, `sgp/rankings.py`,
  `trades/multi_trade.py` (`evaluate_multi_trade` + `build_waiver_pool`),
  `analysis/injury_stress.py` (its Upstash `_data`-unwrap recipe). The generator
  composes them.
- **Modified:** `web/season_routes.py` -- the `/api/evaluate-trade` input
  assembly is extracted into `trades/eval_inputs.py::load_trade_eval_context` and
  the route is refactored to call it, so the route and the generator share one
  path (no duplicated assembly). Behavior of the route is unchanged (the
  round-trip test guards this).
