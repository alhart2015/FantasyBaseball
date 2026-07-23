# Keeper-Trade Generator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-23-keeper-trade-generator-design.md`

**Goal:** Generate keeper-mutual "consolidation" trade suggestions (acquire one stud that cracks your top-3, pay with your displaced #3 + trapped surplus), each cleared by a category-aware 2026 `evaluate_multi_trade` guardrail.

**Architecture:** A **pure** module `analysis/keeper_trades.py` holds the data model, the generation loop (`generate_consolidation_trades`, guardrail injected as a callable), keeper-viable-package enumeration, and `build_consolidation_proposal` (constructs a roster-legal `TradeProposal`). A shared `trades/eval_inputs.py` extracts the `/api/evaluate-trade` input assembly so the route and the generator share it. A thin script `scripts/keeper_trades.py` wires offline keeper values + live Upstash rosters + the real `evaluate_multi_trade` guardrail and renders the report.

**Tech Stack:** Python 3, pandas, pytest. Reuses `analysis/keeper_value` (via `scripts/keeper_value.py`), `trades/multi_trade` (`evaluate_multi_trade`, `build_waiver_pool`, `TradeProposal`, `_current_active_set`), `sgp/rankings`, `sgp/player_value.calculate_player_sgp`, `data/kv_store.build_explicit_upstash_kv`, `models/player.Player`.

## Global Constraints

- **ASCII-only** in all source, logs, format strings, report output. `sigma`, `--`, `->`, straight quotes.
- **Never key on bare names.** Identities are `Player.player_key` = `"name::player_type"` (`models/player.make_player_key`); keeper-value lookup uses `sgp/rankings.rank_key`/`fg_key`/`lookup_rank`.
- **No `x or default` for numeric defaults;** `v if v is not None else default`. `0.0`/`keeper_value == 0.0`/`delta_total == 0.0` are real values.
- **Pure module does no I/O.** `analysis/keeper_trades.py` imports only stdlib + `trades.multi_trade.TradeProposal` + `models.player`; the guardrail is an injected callable.
- **Keeper values are offline** (local DB board via `scripts/keeper_value.build_results`, `keeper_value = discounted_total(r.per_year_var, base_year, discount, horizon)`); **rosters + guardrail inputs are live Upstash**.

---

## File Structure

- **Create** `src/fantasy_baseball/analysis/keeper_trades.py` -- pure: data model, `top3_sum`, `keeper_viable_packages`, `generate_consolidation_trades`, `build_consolidation_proposal`.
- **Create** `src/fantasy_baseball/trades/eval_inputs.py` -- `TradeEvalContext` + `load_trade_eval_context(...)` (extracted from the route).
- **Modify** `src/fantasy_baseball/web/season_routes.py` -- `/api/evaluate-trade` calls `load_trade_eval_context`.
- **Create** `scripts/keeper_trades.py` -- orchestration + render (importable; guarded `main()`).
- **Create** `tests/test_analysis/test_keeper_trades.py`, `tests/test_trades/test_eval_inputs.py`.

---

### Task 1: Data model + top-3 helper

**Files:**
- Create: `src/fantasy_baseball/analysis/keeper_trades.py`
- Test: `tests/test_analysis/test_keeper_trades.py`

**Interfaces:**
- Produces: `RosterPlayer(player_id: str, name: str, keeper_value: float)`; `GuardrailResult(legal: bool, delta_total: float, ok: bool)`; `TradeSuggestion(...)`; `Guardrail = Callable[[list[RosterPlayer], RosterPlayer], GuardrailResult]`; `top3_sum(players: Iterable[RosterPlayer]) -> float` (sum of the 3 highest `keeper_value`, or all if fewer than 3).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis/test_keeper_trades.py
from fantasy_baseball.analysis import keeper_trades as kt


def rp(name, kv):
    return kt.RosterPlayer(player_id=f"{name}::hitter", name=name, keeper_value=kv)


def test_top3_sum_takes_three_highest():
    players = [rp("a", 10), rp("b", 8), rp("c", 6), rp("d", 4)]
    assert kt.top3_sum(players) == 24.0          # 10 + 8 + 6, ignores d


def test_top3_sum_handles_fewer_than_three():
    assert kt.top3_sum([rp("a", 10), rp("b", 8)]) == 18.0
    assert kt.top3_sum([]) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_keeper_trades.py -v`
Expected: FAIL (`ModuleNotFoundError` / `AttributeError`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/analysis/keeper_trades.py
"""Keeper-trade generator: keeper-mutual consolidation trades. Pure math -- the
2026 guardrail is injected as a callable. See
docs/superpowers/specs/2026-07-23-keeper-trade-generator-design.md.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class RosterPlayer:
    player_id: str          # Player.player_key = "name::player_type"
    name: str
    keeper_value: float     # discounted multi-year VAR at the chosen discount


@dataclass(frozen=True)
class GuardrailResult:
    legal: bool
    delta_total: float      # Hart's projected 2026 roto-point change
    ok: bool                # legal AND delta_total >= -threshold


Guardrail = Callable[[Sequence["RosterPlayer"], "RosterPlayer"], GuardrailResult]


@dataclass(frozen=True)
class TradeSuggestion:
    target_team: str
    acquire: RosterPlayer
    give: tuple[RosterPlayer, ...]
    variant: str            # "minimal" | "sweetened"
    my_top3_before: float
    my_top3_after: float
    my_gain: float
    their_top3_before: float
    their_top3_after: float
    their_gain: float
    guardrail: GuardrailResult


def top3_sum(players: Iterable[RosterPlayer]) -> float:
    return float(sum(sorted((p.keeper_value for p in players), reverse=True)[:3]))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_keeper_trades.py -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/keeper_trades.py tests/test_analysis/test_keeper_trades.py
git commit -m "feat(keeper-trades): data model + top3_sum helper"
```

---

### Task 2: `keeper_viable_packages` enumeration

**Files:**
- Modify: `src/fantasy_baseball/analysis/keeper_trades.py`
- Test: `tests/test_analysis/test_keeper_trades.py`

**Interfaces:**
- Produces: `keeper_viable_packages(acquire: RosterPlayer, opp_roster: Sequence[RosterPlayer], giveable: Sequence[RosterPlayer], opp_top3_before: float, max_give: int) -> Iterator[tuple[RosterPlayer, ...]]` -- yields subsets of `giveable` (size 1..max_give) for which the opponent's trio strictly improves after removing `acquire` and adding the package, ordered by fewest players then least total keeper_value given.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis/test_keeper_trades.py (add)
def test_keeper_viable_packages_ordered_and_improving():
    # opp: stud G(16) + scrubs s1(3), s2(2)  -> top3_before = 21
    G = rp("G", 16)
    opp = [G, rp("s1", 3), rp("s2", 2)]
    # giveable: d3(14), sur(12), low(1)
    giveable = [rp("d3", 14), rp("sur", 12), rp("low", 1)]
    pkgs = list(kt.keeper_viable_packages(G, opp, giveable, kt.top3_sum(opp), max_give=3))
    # After removing G, opp keeps s1(3),s2(2); a package must lift top3 above 21.
    # d3+sur -> top3 = 14+12+3 = 29 > 21 (viable, 2 players).
    # single players can't reach 21 (14+3+2=19), so 2-player packages come first.
    assert pkgs, "expected at least one viable package"
    assert all(kt.top3_sum([p for p in opp if p is not G] + list(pkg)) > 21 for pkg in pkgs)
    # ordered fewest-players-first: no 3-player pkg precedes a viable 2-player one
    sizes = [len(pkg) for pkg in pkgs]
    assert sizes == sorted(sizes)
    # {d3,sur} present and beats {d3,low} on the tie (more keeper given is NOT preferred:
    # among equal-size viable packages, least total keeper_value given comes first)
    two_player = [pkg for pkg in pkgs if len(pkg) == 2]
    assert two_player[0] == (giveable[0], giveable[1]) or set(two_player[0]) == {giveable[0], giveable[1]}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_keeper_trades.py -k viable -v`
Expected: FAIL (`AttributeError: ... keeper_viable_packages`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/analysis/keeper_trades.py (add imports)
from itertools import combinations
from collections.abc import Iterator
```

```python
# src/fantasy_baseball/analysis/keeper_trades.py (add)
def keeper_viable_packages(
    acquire: RosterPlayer,
    opp_roster: Sequence[RosterPlayer],
    giveable: Sequence[RosterPlayer],
    opp_top3_before: float,
    max_give: int,
) -> Iterator[tuple[RosterPlayer, ...]]:
    """Packages (subsets of giveable) that strictly lift the opponent's trio once
    `acquire` leaves and the package arrives. Ordered fewest-players, then least
    total keeper_value given (protect Hart's better surplus)."""
    opp_without = [p for p in opp_roster if p.player_id != acquire.player_id]
    candidates: list[tuple[int, float, tuple[RosterPlayer, ...]]] = []
    for size in range(1, max_give + 1):
        for combo in combinations(giveable, size):
            if top3_sum([*opp_without, *combo]) > opp_top3_before:
                candidates.append((size, sum(p.keeper_value for p in combo), combo))
    candidates.sort(key=lambda c: (c[0], c[1]))   # fewest players, then least kv given
    for _size, _cost, combo in candidates:
        yield combo
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_keeper_trades.py -k viable -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/keeper_trades.py tests/test_analysis/test_keeper_trades.py
git commit -m "feat(keeper-trades): keeper-viable package enumeration"
```

---

### Task 3: `generate_consolidation_trades` (main loop, injected guardrail)

**Files:**
- Modify: `src/fantasy_baseball/analysis/keeper_trades.py`
- Test: `tests/test_analysis/test_keeper_trades.py`

**Interfaces:**
- Consumes: `top3_sum`, `keeper_viable_packages`, the `Guardrail` callable.
- Produces: `generate_consolidation_trades(my_team, rosters, guardrail, *, max_give=3, sweetener=True) -> list[TradeSuggestion]`. For each opponent stud `G` with `keeper_value(G) > my_third`, walks keeper-viable packages in order, and emits the first whose guardrail `ok`; if `sweetener`, also emits a "sweetened" suggestion (minimal + the next unused giveable) when its guardrail still `ok`. Sorted by `my_gain` desc.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis/test_keeper_trades.py (add)
def _league():
    # Hart: top-2 protected (soto 18, jrod 16), #3 cam 14, surplus woo 12, wood 9
    hart = [rp("soto", 18), rp("jrod", 16), rp("cam", 14), rp("woo", 12), rp("wood", 9)]
    # Spacemen: stud judge 17 + scrubs g(8), t(7)
    spacemen = [rp("judge", 17), rp("g", 8), rp("t", 7)]
    return {"Hart": hart, "Spacemen": spacemen}


def _pass_all(give, receive):
    return kt.GuardrailResult(legal=True, delta_total=-1.0, ok=True)


def test_consolidation_found_both_trios_improve():
    league = _league()
    out = kt.generate_consolidation_trades("Hart", league, _pass_all, sweetener=False)
    assert out, "expected a suggestion"
    s = next(x for x in out if x.acquire.name == "judge")
    assert s.my_top3_after > s.my_top3_before          # you improve
    assert s.their_top3_after > s.their_top3_before     # they improve
    assert s.my_gain == 17 - 14                         # value(G) - your #3


def test_displaced_keeper_is_free_gain_is_fixed():
    league = _league()
    out = kt.generate_consolidation_trades("Hart", league, _pass_all, sweetener=False)
    s = next(x for x in out if x.acquire.name == "judge")
    assert s.my_gain == 3.0                             # independent of the package


def test_guardrail_skips_first_package_takes_next():
    league = _league()
    seen = []

    def gr(give, receive):
        seen.append(tuple(p.name for p in give))
        ok = len(give) >= 2 and any(p.name == "wood" for p in give)  # fail until 'wood' in pkg
        return kt.GuardrailResult(legal=True, delta_total=-1.0, ok=ok)

    out = kt.generate_consolidation_trades("Hart", league, gr, sweetener=False)
    s = next(x for x in out if x.acquire.name == "judge")
    assert any(p.name == "wood" for p in s.give)        # emitted the first passing pkg
    assert len(seen) >= 2                                # consulted guardrail more than once


def test_no_target_when_no_stud_above_my_third():
    league = {"Hart": [rp("soto", 18), rp("jrod", 16), rp("cam", 14)],
              "Weak": [rp("x", 10), rp("y", 5), rp("z", 3)]}
    assert kt.generate_consolidation_trades("Hart", league, _pass_all) == []


def test_suggestions_sorted_by_my_gain_desc():
    # two studs of different value -> the bigger upgrade ranks first
    league = {
        "Hart": [rp("soto", 18), rp("jrod", 16), rp("cam", 14), rp("woo", 12), rp("wood", 9)],
        "A": [rp("big", 20), rp("a1", 3), rp("a2", 2)],      # gain 20-14 = 6
        "B": [rp("mid", 15), rp("b1", 3), rp("b2", 2)],      # gain 15-14 = 1
    }
    out = kt.generate_consolidation_trades("Hart", league, _pass_all, sweetener=False)
    gains = [s.my_gain for s in out]
    assert gains == sorted(gains, reverse=True)
    assert out[0].acquire.name == "big"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_keeper_trades.py -k "consolidation or displaced or guardrail_skips or no_target" -v`
Expected: FAIL (`AttributeError: ... generate_consolidation_trades`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/analysis/keeper_trades.py (add)
def generate_consolidation_trades(
    my_team: str,
    rosters: Mapping[str, Sequence[RosterPlayer]],
    guardrail: Guardrail,
    *,
    max_give: int = 3,
    sweetener: bool = True,
) -> list[TradeSuggestion]:
    me = sorted(rosters[my_team], key=lambda p: p.keeper_value, reverse=True)
    if len(me) < 3:
        return []
    my_top2, my_third = me[:2], me[2].keeper_value
    my_top3_before = top3_sum(me)
    protect = {me[0].player_id, me[1].player_id}
    giveable = [p for p in me if p.player_id not in protect]

    out: list[TradeSuggestion] = []
    for team, roster in rosters.items():
        if team == my_team:
            continue
        opp_top3_before = top3_sum(roster)
        for g in roster:
            if g.keeper_value <= my_third:
                continue
            my_gain = g.keeper_value - my_third
            my_top3_after = my_top2[0].keeper_value + my_top2[1].keeper_value + g.keeper_value
            for pkg in keeper_viable_packages(g, roster, giveable, opp_top3_before, max_give):
                verdict = guardrail(pkg, g)
                if not verdict.ok:
                    continue
                out.append(_suggestion(
                    team, g, pkg, "minimal", my_top3_before, my_top3_after, my_gain,
                    roster, opp_top3_before, verdict,
                ))
                if sweetener:
                    extra = next((p for p in giveable if p not in pkg), None)
                    if extra is not None:
                        spkg = (*pkg, extra)
                        sv = guardrail(spkg, g)
                        if sv.ok:
                            out.append(_suggestion(
                                team, g, spkg, "sweetened", my_top3_before, my_top3_after,
                                my_gain, roster, opp_top3_before, sv,
                            ))
                break   # first passing minimal package wins for this (team, g)
    out.sort(key=lambda s: s.my_gain, reverse=True)
    return out


def _suggestion(team, g, pkg, variant, my_before, my_after, my_gain,
                roster, opp_before, verdict) -> TradeSuggestion:
    their_after = top3_sum([p for p in roster if p.player_id != g.player_id] + list(pkg))
    return TradeSuggestion(
        target_team=team, acquire=g, give=tuple(pkg), variant=variant,
        my_top3_before=my_before, my_top3_after=my_after, my_gain=my_gain,
        their_top3_before=opp_before, their_top3_after=their_after,
        their_gain=their_after - opp_before, guardrail=verdict,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_keeper_trades.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/keeper_trades.py tests/test_analysis/test_keeper_trades.py
git commit -m "feat(keeper-trades): generate_consolidation_trades with injected guardrail"
```

---

### Task 4: `build_consolidation_proposal` (pure roster-legal proposal)

**Files:**
- Modify: `src/fantasy_baseball/analysis/keeper_trades.py`
- Test: `tests/test_analysis/test_keeper_trades.py`

**Interfaces:**
- Consumes: `trades.multi_trade.TradeProposal`, `trades.multi_trade._current_active_set`, `models.player.Player`.
- Produces: `build_consolidation_proposal(opponent, hart_players, package_keys, receive_key, my_adds_keys, opp_drop_keys) -> TradeProposal`. Pure assembly: `send`/`receive`/`my_adds`/`opp_drops` from the given keys, and `my_active_ids = (current active keys) - set(send) | {receive} | set(my_adds)`, `opp_active_ids = set()`. The ROS-ranked `my_adds_keys` / `opp_drop_keys` are chosen by the caller (see Task 6 helper `_ros_refills_and_drops`) so this function stays free of SGP/denoms.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis/test_keeper_trades.py (add)
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import Position


def _pl(name, pos):
    return Player(name=name, player_type=PlayerType.HITTER, selected_position=pos)


def test_build_consolidation_proposal_balances_and_sets_active():
    # Hart active: soto(OF), jrod(OF), cam(3B); bench: woo(BN)
    hart = [_pl("soto", Position.OF), _pl("jrod", Position.OF),
            _pl("cam", Position.THIRD_BASE), _pl("woo", Position.BN)]
    prop = kt.build_consolidation_proposal(
        opponent="Spacemen",
        hart_players=hart,
        package_keys=["cam::hitter", "woo::hitter"],   # send 2
        receive_key="judge::hitter",                    # get 1
        my_adds_keys=["fa1::hitter"],                   # refill N-1 = 1
        opp_drop_keys=["scrub::hitter"],                # opp drops N-1 = 1
    )
    assert prop.send == ["cam::hitter", "woo::hitter"]
    assert prop.receive == ["judge::hitter"]
    assert prop.my_adds == ["fa1::hitter"]
    assert prop.opp_drops == ["scrub::hitter"]
    # cam was active and is sent -> leaves active; judge + fa1 enter; soto/jrod stay
    assert prop.my_active_ids == {"soto::hitter", "jrod::hitter", "judge::hitter", "fa1::hitter"}
    assert prop.opp_active_ids == set()          # empty -> evaluator opp fallback
    assert prop.my_active_ids                     # regression: NEVER empty
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_keeper_trades.py -k build_consolidation -v`
Expected: FAIL (`AttributeError: ... build_consolidation_proposal`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/analysis/keeper_trades.py (add imports)
from fantasy_baseball.models.player import Player
from fantasy_baseball.trades.multi_trade import TradeProposal, _current_active_set
```

```python
# src/fantasy_baseball/analysis/keeper_trades.py (add)
def build_consolidation_proposal(
    *,
    opponent: str,
    hart_players: Sequence[Player],
    package_keys: Sequence[str],
    receive_key: str,
    my_adds_keys: Sequence[str],
    opp_drop_keys: Sequence[str],
) -> TradeProposal:
    """Roster-legal 1-for-N consolidation proposal. `my_active_ids` is the post-trade
    active set -- REQUIRED, or evaluate_multi_trade zeroes Hart's active roster
    (multi_trade.py:200-213). opp_active_ids stays empty (opp fallback handles it)."""
    current_active = _current_active_set(hart_players)
    my_active = (current_active - set(package_keys)) | {receive_key} | set(my_adds_keys)
    return TradeProposal(
        opponent=opponent,
        send=list(package_keys),
        receive=[receive_key],
        my_drops=[],
        opp_drops=list(opp_drop_keys),
        my_adds=list(my_adds_keys),
        my_active_ids=my_active,
        opp_active_ids=set(),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_keeper_trades.py -k build_consolidation -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/keeper_trades.py tests/test_analysis/test_keeper_trades.py
git commit -m "feat(keeper-trades): build_consolidation_proposal (pure, roster-legal)"
```

---

### Task 5: Extract `trades/eval_inputs.py` + refactor the route

**Files:**
- Create: `src/fantasy_baseball/trades/eval_inputs.py`
- Modify: `src/fantasy_baseball/web/season_routes.py` (the `/api/evaluate-trade` route)
- Test: `tests/test_trades/test_eval_inputs.py`

**Interfaces:**
- Produces: `TradeEvalContext(hart_name, hart_roster, opp_rosters, waiver_pool, projected_standings, team_sds, fraction_remaining, roster_slots)` and `load_trade_eval_context(*, hart_name, roster_raw, opp_rosters_raw, proj_cache, ros_cache, roster_slots) -> TradeEvalContext`. Both the route and the generator read their blobs (route: local cache; generator: Upstash) then call this with the raw dicts.

**Regression first (per spec):** confirm `tests/test_web/test_season_routes.py` covers `/api/evaluate-trade`; if it does not pin the response, add a characterization test there BEFORE editing the route.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_trades/test_eval_inputs.py
from fantasy_baseball.trades.eval_inputs import load_trade_eval_context


def _fixture_blobs():
    roster_raw = [{"name": "A B", "player_type": "hitter", "hr": 20, "ab": 400}]
    opp_raw = {"Opp": [{"name": "C D", "player_type": "pitcher", "ip": 100, "w": 8}]}
    proj_cache = {
        # ProjectedStandings.from_json shape: effective_date + teams[{name, stats}];
        # stats {} -> CategoryStats defaults (models/standings.py:363-381).
        "projected_standings": {
            "effective_date": "2026-07-23",
            "teams": [
                {"name": "Hart of the Order", "stats": {}},
                {"name": "Opp", "stats": {}},
            ],
        },
        "team_sds": None,
        "fraction_remaining": 0.4,
    }
    ros_cache = {"hitters": [], "pitchers": []}
    return roster_raw, opp_raw, proj_cache, ros_cache


def test_load_trade_eval_context_shapes_inputs():
    roster_raw, opp_raw, proj_cache, ros_cache = _fixture_blobs()
    ctx = load_trade_eval_context(
        hart_name="Hart of the Order", roster_raw=roster_raw, opp_rosters_raw=opp_raw,
        proj_cache=proj_cache, ros_cache=ros_cache, roster_slots={"OF": 4, "P": 9, "BN": 2},
    )
    assert ctx.hart_name == "Hart of the Order"
    assert [p.name for p in ctx.hart_roster] == ["A B"]
    assert set(ctx.opp_rosters) == {"Opp"}
    assert ctx.fraction_remaining == 0.4
    assert isinstance(ctx.waiver_pool, dict)          # keyed by player_key
    assert ctx.roster_slots["OF"] == 4
    # projected_standings is the typed object with both teams present
    assert any(e.team_name == "Hart of the Order" for e in ctx.projected_standings.entries)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trades/test_eval_inputs.py -v`
Expected: FAIL (`ModuleNotFoundError: ... trades.eval_inputs`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/trades/eval_inputs.py
"""Shared assembly of evaluate_multi_trade's inputs from cached blobs. Used by the
/api/evaluate-trade route (local cache) and the keeper-trade generator (Upstash)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fantasy_baseball.models.player import Player
from fantasy_baseball.models.standings import ProjectedStandings
from fantasy_baseball.scoring import team_sds_from_json
from fantasy_baseball.trades.multi_trade import build_waiver_pool
from fantasy_baseball.utils.constants import Category


@dataclass(frozen=True)
class TradeEvalContext:
    hart_name: str
    hart_roster: list[Player]
    opp_rosters: dict[str, list[Player]]
    waiver_pool: dict[str, Player]
    projected_standings: ProjectedStandings
    team_sds: dict[str, dict[Category, float]] | None
    fraction_remaining: float
    roster_slots: dict[str, int]


def load_trade_eval_context(
    *,
    hart_name: str,
    roster_raw: list[dict[str, Any]],
    opp_rosters_raw: dict[str, list[dict[str, Any]]],
    proj_cache: dict[str, Any],
    ros_cache: dict[str, Any],
    roster_slots: dict[str, int],
) -> TradeEvalContext:
    hart_roster = [Player.from_dict(p) for p in roster_raw]
    opp_rosters = {n: [Player.from_dict(p) for p in ps] for n, ps in opp_rosters_raw.items()}
    waiver_pool = build_waiver_pool(hart_roster, opp_rosters, ros_cache)
    raw_ps = proj_cache.get("projected_standings")
    if not raw_ps:
        raise ValueError("proj_cache missing 'projected_standings'")
    sds_raw = proj_cache.get("team_sds")
    fr = proj_cache.get("fraction_remaining")
    return TradeEvalContext(
        hart_name=hart_name,
        hart_roster=hart_roster,
        opp_rosters=opp_rosters,
        waiver_pool=waiver_pool,
        projected_standings=ProjectedStandings.from_json(raw_ps),
        team_sds=team_sds_from_json(sds_raw) if sds_raw else None,
        fraction_remaining=1.0 if fr is None else float(fr),
        roster_slots=roster_slots,
    )
```

(Confirmed paths: `ProjectedStandings` is in `models.standings`; `team_sds_from_json` is in `scoring` -- as imported above.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trades/test_eval_inputs.py -v`
Expected: PASS.

- [ ] **Step 5: Refactor the route to use it**

In `web/season_routes.py::api_evaluate_trade`, replace the inline assembly (reading `ROSTER`/`OPP_ROSTERS`/`PROJECTIONS`/`ROS_PROJECTIONS`, building `hart_roster`/`opp_rosters`/`waiver_pool`/`projected_standings`/`team_sds`/`fr`) with:

```python
from fantasy_baseball.trades.eval_inputs import load_trade_eval_context
# ... after the roster/opponent-presence guards (keep those + the 404s):
ctx = load_trade_eval_context(
    hart_name=config.team_name,
    roster_raw=roster_raw,
    opp_rosters_raw=opp_rosters_raw,
    proj_cache=proj_cache,
    ros_cache=read_cache_dict(CacheKey.ROS_PROJECTIONS) or {},
    roster_slots=config.roster_slots,
)
```

Then pass `ctx.*` into `evaluate_multi_trade(...)`. Keep the route's existing 404 guards (no roster / no projected_standings) and JSON serialization unchanged.

- [ ] **Step 6: Run route + eval-inputs tests**

Run: `pytest tests/test_trades/test_eval_inputs.py tests/test_web/test_season_routes.py -q`
Expected: PASS (route behavior unchanged).

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/trades/eval_inputs.py src/fantasy_baseball/web/season_routes.py tests/test_trades/test_eval_inputs.py
git commit -m "refactor(trades): extract shared load_trade_eval_context; route uses it"
```

---

### Task 6: `scripts/keeper_trades.py` (orchestration + render)

**Files:**
- Create: `scripts/keeper_trades.py`
- Test: `tests/test_scripts/test_keeper_trades_script.py` (matching + ROS refill/drop helper)

**Interfaces:**
- Consumes: `analysis.keeper_trades` (`generate_consolidation_trades`, `build_consolidation_proposal`, `RosterPlayer`, `GuardrailResult`); `scripts/keeper_value.py` (`build_results` path); `trades.eval_inputs.load_trade_eval_context`; `trades.multi_trade.evaluate_multi_trade`; `data.kv_store.build_explicit_upstash_kv`; `sgp.rankings` + `sgp.player_value.calculate_player_sgp`.
- Produces (importable functions; guarded `main()`):
  - `_ros_value(player, denoms) -> float` -- `calculate_player_sgp(player.rest_of_season, denoms)` or `-inf` if no ROS line.
  - `_ros_refills_and_drops(waiver_pool, opp_players, denoms, n) -> (list[str], list[str])` -- top-`n` waiver keys by ROS value; `opp_players`' bottom-`n` keys by ROS value (`opp_players` already excludes the received stud).
  - `to_roster_players(players, keeper_by_key) -> list[RosterPlayer]` -- attach keeper_value **fg_id-primary** (`fg_key(p.fg_id, type)` when present) with `rank_key(name, type)` fallback; unmatched -> `0.0`. `keeper_by_key` is indexed by both the board `player_id` (fg-based when available) and `rank_key(name)`.
  - `make_guardrail(ctx, denoms, threshold) -> Guardrail` -- resolves the opponent from `receive`, calls `_ros_refills_and_drops` + `build_consolidation_proposal` + `evaluate_multi_trade`.
  - `render(suggestions) -> str`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scripts/test_keeper_trades_script.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

import keeper_trades as script  # noqa: E402
from fantasy_baseball.analysis.keeper_trades import RosterPlayer  # noqa: E402


def test_to_roster_players_attaches_keeper_value_and_zero_for_unmatched():
    from fantasy_baseball.models.player import Player, PlayerType
    from fantasy_baseball.sgp.rankings import fg_key, rank_key

    players = [
        Player(name="Juan Soto", player_type=PlayerType.HITTER),                 # name match
        Player(name="Two Names", player_type=PlayerType.HITTER, fg_id="999"),     # fg_id match
        Player(name="Nobody Here", player_type=PlayerType.HITTER),               # unmatched
    ]
    keeper_by_key = {
        rank_key("Juan Soto", PlayerType.HITTER): 18.0,
        fg_key("999", PlayerType.HITTER): 12.0,                 # fg-based board id
        rank_key("Two Names", PlayerType.HITTER): 1.0,          # decoy: fg_id must win
    }
    out = script.to_roster_players(players, keeper_by_key)
    kv = {p.name: p.keeper_value for p in out}
    assert kv["Juan Soto"] == 18.0
    assert kv["Two Names"] == 12.0                  # fg_id-primary beats the name entry
    assert kv["Nobody Here"] == 0.0                 # unmatched -> 0.0, never dropped
    assert all(isinstance(p, RosterPlayer) for p in out)


def test_ros_refills_and_drops_picks_top_and_bottom(monkeypatch):
    from fantasy_baseball.models.player import Player, PlayerType

    def pl(name):
        return Player(name=name, player_type=PlayerType.HITTER)

    vals = {"fa_hi": 9.0, "fa_mid": 5.0, "fa_lo": 1.0, "opp_a": 8.0, "opp_b": 2.0, "opp_c": 3.0}
    monkeypatch.setattr(script, "_ros_value", lambda p, denoms: vals[p.name])
    waiver = {pl(n).player_key: pl(n) for n in ("fa_hi", "fa_mid", "fa_lo")}
    opp = [pl("opp_a"), pl("opp_b"), pl("opp_c")]
    adds, drops = script._ros_refills_and_drops(waiver, opp, denoms=None, n=1)
    assert adds == ["fa_hi::hitter"]                # top-1 refill by ROS value
    assert drops == ["opp_b::hitter"]               # bottom-1 drop (2.0 is lowest)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_keeper_trades_script.py -v`
Expected: FAIL (`ModuleNotFoundError: keeper_trades`).

- [ ] **Step 3: Write minimal implementation**

```python
# scripts/keeper_trades.py
"""Suggest keeper-mutual consolidation trades. Keeper values are offline (local
board); rosters + the 2026 guardrail inputs are live Upstash. See
docs/superpowers/specs/2026-07-23-keeper-trade-generator-design.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import keeper_value as kv_script  # scripts/keeper_value.py (build_results, discounted_total, BASE_YEAR)
from fantasy_baseball.analysis.keeper_trades import (
    GuardrailResult,
    RosterPlayer,
    build_consolidation_proposal,
    generate_consolidation_trades,
)
from fantasy_baseball.analysis.keeper_value import discounted_total
from fantasy_baseball.config import load_config
from fantasy_baseball.data.cache_keys import CacheKey, redis_key
from fantasy_baseball.data.kv_store import build_explicit_upstash_kv
from fantasy_baseball.models.player import Player
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.rankings import fg_key, rank_key
from fantasy_baseball.trades.eval_inputs import load_trade_eval_context
from fantasy_baseball.trades.multi_trade import evaluate_multi_trade

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "config" / "league.yaml"


def _ros_value(player: Player, denoms) -> float:
    if player.rest_of_season is None:
        return float("-inf")
    return calculate_player_sgp(player.rest_of_season, denoms)


def to_roster_players(players, keeper_by_key) -> list[RosterPlayer]:
    """Attach keeper_value fg_id-primary, name fallback (spec + CLAUDE.md 'never key
    on bare names'). keeper_by_key holds BOTH fg-based board ids and rank_key(name)."""
    out = []
    for p in players:
        val = None
        if p.fg_id:
            val = keeper_by_key.get(fg_key(str(p.fg_id), p.player_type))
        if val is None:
            val = keeper_by_key.get(rank_key(p.name, p.player_type))
        out.append(RosterPlayer(
            player_id=p.player_key, name=p.name,
            keeper_value=val if val is not None else 0.0,
        ))
    return out
```

Then (same file) the Upstash blob reader, keeper-value load, guardrail wiring, and render:

```python
def _cache(kv, key):
    raw = kv.get(redis_key(key))
    if raw is None:
        raise RuntimeError(f"Upstash missing {key}; run a refresh first.")
    o = json.loads(raw) if isinstance(raw, str) else raw
    return o["_data"] if isinstance(o, dict) and "_data" in o else o


def _ros_refills_and_drops(waiver_pool, opp_players, denoms, n):
    """(top-n waiver refill keys by ROS value, bottom-n opp drop keys by ROS value).
    `opp_players` already excludes the received stud. Pure given Players + denoms."""
    adds = sorted(waiver_pool.values(), key=lambda p: _ros_value(p, denoms), reverse=True)
    drops = sorted(opp_players, key=lambda p: _ros_value(p, denoms))
    return [p.player_key for p in adds[:n]], [p.player_key for p in drops[:n]]


def make_guardrail(ctx, denoms, threshold):
    """Injected guardrail: resolves the opponent from `receive`, builds a
    roster-legal proposal, and returns evaluate_multi_trade's verdict for Hart."""

    def _owning_team(receive_key: str) -> str:
        for team, players in ctx.opp_rosters.items():
            if any(p.player_key == receive_key for p in players):
                return team
        raise KeyError(f"{receive_key} is not on any opponent roster")

    def guardrail(give, receive):
        package_keys = [p.player_id for p in give]   # RosterPlayer.player_id == player_key
        n = max(0, len(package_keys) - 1)
        opp_name = _owning_team(receive.player_id)
        opp_players = [p for p in ctx.opp_rosters[opp_name] if p.player_key != receive.player_id]
        my_adds, opp_drops = _ros_refills_and_drops(ctx.waiver_pool, opp_players, denoms, n)
        proposal = build_consolidation_proposal(
            opponent=opp_name, hart_players=ctx.hart_roster, package_keys=package_keys,
            receive_key=receive.player_id, my_adds_keys=my_adds, opp_drop_keys=opp_drops,
        )
        r = evaluate_multi_trade(
            proposal=proposal, hart_name=ctx.hart_name, hart_roster=ctx.hart_roster,
            opp_rosters=ctx.opp_rosters, waiver_pool=ctx.waiver_pool,
            projected_standings=ctx.projected_standings, team_sds=ctx.team_sds,
            roster_slots=ctx.roster_slots, fraction_remaining=ctx.fraction_remaining,
        )
        return GuardrailResult(legal=r.legal, delta_total=r.delta_total,
                               ok=r.legal and r.delta_total >= -threshold)

    return guardrail
```

`main()`:

```python
def _parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Suggest keeper-mutual consolidation trades.")
    ap.add_argument("--discount", type=float, default=0.80)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--guardrail-threshold", type=float, default=2.0)
    ap.add_argument("--max-give", type=int, default=3)
    ap.add_argument("--no-sweetener", action="store_true")
    return ap.parse_args(argv)


def main(argv=None):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    args = _parse_args(argv)
    config = load_config(CONFIG_PATH)
    denoms = get_sgp_denominators(config.sgp_overrides)

    # 1. offline keeper values -> {rank_key: keeper_value}
    results, _ = kv_script.build_results(base_year=kv_script.BASE_YEAR, horizon=args.horizon)
    keeper_by_key: dict[str, float] = {}
    for r in results:
        ptype = r.player_id.rsplit("::", 1)[-1]
        val = discounted_total(r.per_year_var, kv_script.BASE_YEAR, args.discount, args.horizon)
        # index by BOTH the board player_id (== fg_key form when the board is
        # fg-based) and rank_key(name) so roster lookup can prefer fg_id.
        for k in (r.player_id, rank_key(r.name, ptype)):
            if k not in keeper_by_key or val > keeper_by_key[k]:
                keeper_by_key[k] = val

    # 2. live Upstash: assemble the eval context + rosters
    kv = build_explicit_upstash_kv()
    roster_raw = _cache(kv, CacheKey.ROSTER)
    opp_raw = _cache(kv, CacheKey.OPP_ROSTERS)
    proj_cache = _cache(kv, CacheKey.PROJECTIONS)
    ros_cache = _cache(kv, CacheKey.ROS_PROJECTIONS)
    ctx = load_trade_eval_context(
        hart_name=config.team_name, roster_raw=roster_raw, opp_rosters_raw=opp_raw,
        proj_cache=proj_cache, ros_cache=ros_cache, roster_slots=config.roster_slots,
    )

    # 3. annotate rosters with keeper_value
    rosters = {config.team_name: to_roster_players(ctx.hart_roster, keeper_by_key)}
    for team, players in ctx.opp_rosters.items():
        rosters[team] = to_roster_players(players, keeper_by_key)

    # 4. generate
    guardrail = make_guardrail(ctx, denoms, args.guardrail_threshold)
    suggestions = generate_consolidation_trades(
        config.team_name, rosters, guardrail,
        max_give=args.max_give, sweetener=not args.no_sweetener,
    )
    print(render(suggestions))


if __name__ == "__main__":
    main()
```

`render(suggestions)` -- group minimal+sweetened under the same `(team, acquire)`, in ranked order:

```python
def render(suggestions) -> str:
    if not suggestions:
        return "No keeper-mutual consolidation trades found."
    order: list[tuple[str, str]] = []
    groups: dict[tuple[str, str], list] = {}
    for s in suggestions:
        key = (s.target_team, s.acquire.player_id)
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(s)
    lines = ["Keeper-mutual consolidation trades (ranked by your keeper gain)", ""]
    for key in order:
        rows = groups[key]
        a = rows[0].acquire
        lines.append(f"ACQUIRE {a.name} (kv {a.keeper_value:.1f}) from {key[0]}")
        for s in rows:
            give = " + ".join(f"{p.name} ({p.keeper_value:.1f})" for p in s.give)
            g = s.guardrail
            lines.append(f"  give [{s.variant}]: {give}")
            lines.append(f"    YOU:  top-3 {s.my_top3_before:.1f} -> {s.my_top3_after:.1f} (+{s.my_gain:.1f})")
            lines.append(f"    THEM: top-3 {s.their_top3_before:.1f} -> {s.their_top3_after:.1f} (+{s.their_gain:.1f})")
            lines.append(f"    2026: roto delta {g.delta_total:+.1f}  guardrail {'OK' if g.ok else 'FAIL'}")
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scripts/test_keeper_trades_script.py -v`
Expected: PASS.

- [ ] **Step 5: Live smoke (manual, not a unit test)**

Run: `python scripts/keeper_trades.py --max-give 2 2>&1 | head -40`
Expected: a ranked list of consolidation suggestions (needs live Upstash + the local keeper board + ZiPS files). If Upstash blobs are missing, it errors with the missing-key message -- expected. Do NOT fabricate blobs to force it.

- [ ] **Step 6: Commit**

```bash
git add scripts/keeper_trades.py tests/test_scripts/test_keeper_trades_script.py
git commit -m "feat(keeper-trades): orchestration script (offline keeper + live guardrail)"
```

---

### Task 7: End-of-effort verification

- [ ] **Step 1: Touched-area tests**

Run: `pytest tests/test_analysis/test_keeper_trades.py tests/test_trades/test_eval_inputs.py tests/test_scripts/test_keeper_trades_script.py -v`
Expected: all PASS.

- [ ] **Step 2: Regression (the route refactor is the risk)**

Run: `pytest tests/test_web/test_season_routes.py tests/test_trades -q`
Expected: no new failures (route behavior preserved).

- [ ] **Step 3: Lint / format / dead-code**

Run: `python -m ruff check . && python -m ruff format --check . && python -m vulture src/fantasy_baseball/analysis/keeper_trades.py src/fantasy_baseball/trades/eval_inputs.py scripts/keeper_trades.py`
Expected: zero ruff violations; no format drift; no NEW vulture findings. (Note: `_current_active_set` is a `_`-prefixed import from `multi_trade`; if vulture/ruff object, add a `# noqa`/whitelist or promote it to a public name in `multi_trade` in the same commit.)

- [ ] **Step 4: mypy (covered files)**

Run: `python -m mypy src/fantasy_baseball/analysis/keeper_trades.py src/fantasy_baseball/trades/eval_inputs.py`
Expected: clean. (`analysis/` and `trades/multi_trade.py` are under `[tool.mypy].files`; confirm `trades/eval_inputs.py` is covered and fix findings.)

- [ ] **Step 5: Commit any fixes**

```bash
git add -A -- src/fantasy_baseball/analysis/keeper_trades.py src/fantasy_baseball/trades/ scripts/keeper_trades.py tests/
git commit -m "chore(keeper-trades): end-of-effort verification fixes"
```

---

## Notes for the implementer

- **Importing `_current_active_set` (a private) from `multi_trade`** is deliberate reuse of the evaluator's own active-set definition (so the proposal's active math matches the evaluator's). If ruff/vulture object to importing a private, promote it to `current_active_set` in `multi_trade.py` (same commit) and update both call sites.
- **`_ros_value` returns `-inf` for no-ROS players** so they sink to the bottom for refills (never picked) and to the bottom for drops (dropped first) -- consistent. `denoms=None` in the helper's unit test is fine because that test stubs `_ros_value`; the live path passes real `denoms`.
- **Roster `Player.fg_id` is usually absent** (Yahoo rosters), so keeper matching falls to `rank_key` (normalized name) -- the same path the leaguewide keeper analysis used successfully.
- **You cannot fully run `main()` without live Upstash** + the local keeper board + ZiPS 2027/2028 CSVs. The unit tests do not need any of that; the live smoke does.
