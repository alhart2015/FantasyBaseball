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
me         = rosters[my_team] annotated with keeper_value + var_2026
my_sorted  = me sorted by keeper_value desc
my_third   = my_sorted[2].keeper_value
protect    = {my_sorted[0].id, my_sorted[1].id}      # top-2 always kept
giveable   = [p in me if p.id not in protect]        # old #3 + everything below

for T, roster in rosters (T != my_team):
    T_top3_before = sum(top-3 keeper_value of roster)
    for G in roster with keeper_value(G) > my_third:      # G upgrades my top-3
        my_gain   = keeper_value(G) - my_third            # fixed
        my_after  = my_sorted[0].kv + my_sorted[1].kv + keeper_value(G)
        pkg = minimal_package(G, roster, giveable)        # see Package selection
        if pkg is None: continue                          # no viable offer
        their_after = top3_sum((roster - G) + pkg)
        roto_delta  = var_2026(G) - sum(var_2026(p) for p in pkg)   # guardrail proxy
        emit TradeSuggestion(minimal); if sweetener: emit TradeSuggestion(sweetened)
rank all suggestions by my_gain desc
```

`minimal_package(G, roster, giveable)`: the **fewest-player** subset of
`giveable` (at most `max_give`) such that **(a)** `top3_sum((roster - G) + pkg)
> T_top3_before` (opponent strictly improves) **and (b)** the guardrail holds
(`roto_delta >= -threshold`). Ties broken by **least total keeper_value given**
(protect Hart's better surplus for the next deal). `giveable` is small (~5-8),
`max_give` ~3, so exhaustive subset search is trivial. Returns `None` if no
subset satisfies both.

**Sweetened variant:** `minimal package + the single next-best giveable player`,
emitted only if the guardrail still holds -- a more generous offer the opponent
is likelier to accept, at a known extra 2026 cost.

## The 2026 guardrail (DECISION -- please confirm in review)

The guardrail keeps Hart from wrecking his 2026 title run for keeper gain.

- **v1: VAR-based proxy (recommended).** `roto_delta_2026 = var_2026(G) -
  sum(var_2026(p) for p in pkg)`, using each player's `per_year_var[2026]` (VAR
  = value over replacement in standings-points, already computed by the keeper
  metric). VAR already nets against replacement, so a dropped-and-waiver-refilled
  body contributes ~`-VAR`; the proxy therefore models a 1-for-N consolidation's
  net 2026 roto impact directly, with **zero extra integration**. `guardrail_ok
  = roto_delta_2026 >= -threshold` (default threshold `2.0` standings-points).
- **Upgrade path: full `evaluate_multi_trade`.** Category-aware, leverage-aware
  `delta_total` for Hart's team. More accurate (captures category shape), but
  requires assembling `projected_standings`, `team_sds`, `waiver_pool`,
  `fraction_remaining`, `roster_slots` from the live season state. Wire it as a
  drop-in replacement for the proxy once those inputs are plumbed; the pure
  generator takes the guardrail as an injected callable so the swap touches only
  the script.

**Why v1 uses the proxy:** it needs data already in hand and keeps the build
small; the proxy's error (ignoring category shape) is second-order for a
go/no-go guardrail. The brainstorm named `evaluate_multi_trade`; this spec
proposes deferring it to keep v1 shippable -- **confirm or override.**

## Components / files

- **Create** `src/fantasy_baseball/analysis/keeper_trades.py` -- **pure** logic:
  the data model, `generate_consolidation_trades(...)`, `minimal_package(...)`,
  top-3 helpers. Takes rosters annotated with keeper_value + var_2026 and a
  guardrail callable; **no I/O**. Fully unit-testable with synthetic leagues.
- **Create** `scripts/keeper_trades.py` -- I/O + orchestration: keeper values
  via `analysis/keeper_value.build_results` path (reuse `scripts/keeper_value.py`
  helpers), **live rosters** from Upstash (ROSTER + OPP_ROSTERS via
  `build_explicit_upstash_kv`, exactly as `injury_stress.load_mc_inputs_from_upstash`
  does), player<->keeper-value matching, the VAR-proxy guardrail, ASCII render.
- **Create** `tests/test_analysis/test_keeper_trades.py`.

## Data model (pure module)

```python
@dataclass(frozen=True)
class RosterPlayer:
    player_id: str
    name: str
    keeper_value: float     # discounted multi-year VAR at the chosen discount
    var_2026: float         # per_year_var[base_year], for the guardrail proxy

@dataclass(frozen=True)
class TradeSuggestion:
    target_team: str
    acquire: RosterPlayer
    give: tuple[RosterPlayer, ...]
    variant: str                    # "minimal" | "sweetened"
    my_top3_before: float
    my_top3_after: float
    my_gain: float
    their_top3_before: float
    their_top3_after: float
    their_gain: float
    roto_delta_2026: float
    guardrail_ok: bool

def generate_consolidation_trades(
    my_team: str,
    rosters: Mapping[str, list[RosterPlayer]],
    *,
    guardrail_threshold: float = 2.0,
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
preseason blend) gets **`keeper_value = 0.0`, `var_2026 = 0.0`** -- a legal
throw-in that never cracks a top-3. The script logs how many roster players were
unmatched.

## Output (ranked by Hart's keeper gain)

```
ACQUIRE Aaron Judge (#6, kv 16.9) from Spacemen         (illustrative numbers)
  give [minimal]:  Junior Caminero (kv 14.4) + Bryan Woo (kv 12.1)
    YOU:   top-3 49.1 -> 51.6  (+2.5)   [Soto, JRod, Judge]
    THEM:  top-3 33.1 -> 35.2  (+2.1)   [Caminero, Woo, Garcia]
    2026:  roto delta -2.0     var_2026: Judge 9 - (Caminero 6 + Woo 5); guardrail OK (>= -2.0)
  give [sweetened]: + James Wood (kv 9.3, var_2026 4)
    THEM:  top-3 33.1 -> 35.2  (+2.1)   2026: roto delta -6.0  guardrail FAIL (< -2.0)
```
(Note: this real case sits right at the threshold -- giving two 2026 contributors
for one bat genuinely costs ~2 standings points -- which is exactly the kind of
call the guardrail exists to surface.)

CLI: `--discount R` (single rate for the keeper ranking, default 0.80),
`--horizon N` (default 3), `--guardrail-threshold X` (default 2.0),
`--max-give N` (default 3), `--no-sweetener`. ASCII-only;
`sys.stdout.reconfigure(encoding="utf-8", errors="replace")` at the entry point.

## Edge cases

- **No stud above Hart's #3 on a team** -> that team yields no suggestions.
- **No viable package** (can't lift the opponent's trio within `max_give`, or
  every package fails the guardrail) -> skip that (T, G); do not emit.
- **Roster legality:** a 1-for-N trade shrinks Hart's roster by N-1 (refilled
  from waivers) and grows the opponent's; v1 does **not** enforce Yahoo
  slot-legality (that lives in `evaluate_multi_trade`), it only reports package
  size. Note this in output.
- **Unmatched roster players:** `keeper_value = 0.0` (see matching); counted and
  logged, never silently dropped.
- **Fewer than 3 keepers on a team:** top-3 sum uses however many exist; handled
  by the top-3 helper (no index error).
- **Guardrail numeric trap:** `roto_delta` compared with `>=`; `var_2026 == 0.0`
  is a real value, never falsy-sunk.

## Testing (pure generator, synthetic leagues)

- **Consolidation-found test:** a synthetic league where team B has one stud +
  two scrubs and Hart has surplus depth -> the generator emits "acquire stud,
  give displaced-#3 + surplus," and **both top-3 sums strictly improve**.
- **Displaced-keeper-is-free test:** `my_gain == value(G) - value(my #3)`
  regardless of which surplus fills the package.
- **Minimal-package test:** the emitted minimal package is the fewest-player
  subset that lifts the opponent's trio; adding fewer players would leave them
  not-improved.
- **Guardrail test:** with an injected guardrail that fails a package, that
  package is dropped (minimal) or flagged `guardrail_ok=False` (sweetened);
  a package within threshold passes.
- **No-target test:** a team whose best player is below Hart's #3 yields no
  suggestions.
- **Ranking test:** suggestions sort by `my_gain` desc.
- **Matching/zero-value test:** an unmatched roster player is treated as
  keeper_value 0 and never appears in a top-3.

## Non-goals (v1)

- **Opponent acceptance / contention modeling** (keeper-mutual only, by decision).
- **Trade shapes other than consolidation** (no get-2, no general N-for-M).
- **Multi-team trades.**
- **Yahoo roster-legality enforcement / auto-execution.**
- **Full `evaluate_multi_trade` guardrail** -- deferred behind the VAR proxy
  (upgrade path documented above).
- **Risk modeling** (injury/role) -- still the separate deferred follow-up from
  the keeper-value spec.

## What doesn't change / reuse

- `analysis/keeper_value.py`, `sgp/rankings.py`, `trades/multi_trade.py`,
  `analysis/injury_stress.py` (its Upstash roster-load recipe) are **reused, not
  modified**. The generator composes them.
