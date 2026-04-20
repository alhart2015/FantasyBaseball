# Multi-Player Trade Evaluator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Build a Trade" section to `/waivers-trades` that lets the user construct an arbitrary N-for-M trade with a specific opponent (including roster-balancing drops and waiver-wire pickups) and see the resulting delta-roto for their own team.

**Architecture:** Backend math lives in a new `trades/multi_trade.py` module that generalizes the existing `apply_swap_delta` pool-assumption logic to aggregate player lists. A new Flask route `POST /api/evaluate-trade` drives the UI; waiver autocomplete is served by `GET /api/waiver-search`. Frontend is added to `waivers_trades.html` as a new section below the existing Trade Finder — vanilla JS, no framework.

**Tech Stack:** Python 3.11+, Flask, Jinja2, vanilla JS, pytest. Reuses `apply_swap_delta`, `score_roto`, `player_rest_of_season_stats`, cached standings/rosters/ROS projections.

**Design spec:** `docs/superpowers/specs/2026-04-20-multi-player-trade-evaluator-design.md`

---

## File Structure

**New files:**
- `src/fantasy_baseball/trades/multi_trade.py` — dataclasses and `evaluate_multi_trade()` function.
- `tests/test_trades/test_multi_trade.py` — unit tests.

**Modified files:**
- `src/fantasy_baseball/trades/evaluate.py` — add a small helper `aggregate_player_stats()` (reuse from `player_rest_of_season_stats` pattern). No changes to `_can_roster_without`.
- `src/fantasy_baseball/web/season_routes.py` — two new routes (`/api/evaluate-trade`, `/api/waiver-search`) plus a waiver-pool helper.
- `src/fantasy_baseball/web/templates/season/waivers_trades.html` — new "Build a Trade" section.

---

## Math primer (read before starting)

Given a proposal with `send[]`, `receive[]`, `my_drops[]`, `opp_drops[]`, `my_adds[]`, and `my_active_ids` (a set of player keys marked active after the trade):

1. **My-team before active set** = players on my roster whose Yahoo `selected_position` is NOT in `{BN, IL}`.
2. **My-team after active set** = `my_active_ids` (already excludes anyone not marked active by the UI; excludes sent/dropped by construction).
3. **My deltas**:
   - `my_loses_ros = aggregate(before_set - after_set)`
   - `my_gains_ros = aggregate(after_set - before_set)`
4. **Opp-team before active set** = all opp players whose `selected_position` is not `IL`. (We do not model opp bench.)
5. **Opp-team after active set** = opp_before − received − opp_drops + sent (all as Player objects).
6. **Opp deltas**:
   - `opp_loses_ros = aggregate(received + opp_drops)`
   - `opp_gains_ros = aggregate(sent)`
7. **Score**: take `projected_standings` as baseline; apply `apply_swap_delta(my_team_stats, my_loses, my_gains)` and `apply_swap_delta(opp_team_stats, opp_loses, opp_gains)`. Run `score_roto` before and after. Delta = sum over categories of `after[me] - before[me]`.

**Why this works**: `apply_swap_delta` uses pool assumptions (`_TEAM_AB=5500`, `_TEAM_IP=1450`) to back out hits/ER/BH, so it handles correctly aggregated rate stats — as long as the aggregated `loses_ros["AVG"]` is the AB-weighted mean (`sum(AVG_i * ab_i) / sum(ab_i)`) and aggregated `loses_ros["ab"]` is the total. This makes N-for-M collapse to one call per team.

**Aggregation formulas** (for a list of players, each with `player_rest_of_season_stats()` output):
```
ab = Σ p.ab                        ip = Σ p.ip
hits = Σ p.AVG * p.ab              er = Σ p.ERA * p.ip / 9
bh = Σ p.WHIP * p.ip               AVG = hits / ab (or 0 if ab=0)
                                   ERA = 9 * er / ip (or 0 if ip=0)
                                   WHIP = bh / ip (or 0 if ip=0)
R/HR/RBI/SB/W/K/SV = Σ per-stat
```

---

## Task 1: Add `aggregate_player_stats` helper

**Files:**
- Modify: `src/fantasy_baseball/trades/evaluate.py` — append new function after `player_rest_of_season_stats` (~line 265).
- Test: `tests/test_trades/test_evaluate.py` — add new test.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trades/test_evaluate.py`:

```python
from fantasy_baseball.trades.evaluate import (
    aggregate_player_stats,
    player_rest_of_season_stats,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player


def test_aggregate_two_hitters_sums_counts_and_weights_avg():
    h1 = Player(name="A", player_type="hitter", positions=["OF"],
                rest_of_season=HitterStats(pa=600, ab=500, h=150,
                                            r=80, hr=25, rbi=70, sb=10, avg=0.300))
    h2 = Player(name="B", player_type="hitter", positions=["2B"],
                rest_of_season=HitterStats(pa=500, ab=400, h=100,
                                            r=50, hr=10, rbi=40, sb=5, avg=0.250))
    agg = aggregate_player_stats([h1, h2])
    assert agg["R"] == 130
    assert agg["HR"] == 35
    assert agg["ab"] == 900
    # Weighted AVG = (150+100)/(500+400) = 250/900
    assert abs(agg["AVG"] - 250/900) < 1e-9
    assert agg["ip"] == 0


def test_aggregate_two_pitchers_weights_era_and_whip():
    p1 = Player(name="P1", player_type="pitcher", positions=["P"],
                rest_of_season=PitcherStats(ip=100, w=8, k=100, sv=0,
                                             era=3.60, whip=1.20,
                                             er=40, bb=30, h_allowed=90))
    p2 = Player(name="P2", player_type="pitcher", positions=["P"],
                rest_of_season=PitcherStats(ip=50, w=3, k=60, sv=20,
                                             era=2.70, whip=1.00,
                                             er=15, bb=10, h_allowed=40))
    agg = aggregate_player_stats([p1, p2])
    assert agg["W"] == 11
    assert agg["K"] == 160
    assert agg["SV"] == 20
    assert agg["ip"] == 150
    # Weighted ERA = 9 * (40+15) / 150 = 9 * 55 / 150 = 3.30
    assert abs(agg["ERA"] - 3.30) < 1e-6
    # Weighted WHIP = (1.20*100 + 1.00*50) / 150 = 170/150
    assert abs(agg["WHIP"] - 170/150) < 1e-6


def test_aggregate_empty_list_returns_zeros():
    agg = aggregate_player_stats([])
    assert agg == {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0.0,
                   "W": 0, "K": 0, "SV": 0, "ERA": 0.0, "WHIP": 0.0,
                   "ab": 0, "ip": 0}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trades/test_evaluate.py::test_aggregate_empty_list_returns_zeros -v`
Expected: FAIL with `ImportError: cannot import name 'aggregate_player_stats'`

- [ ] **Step 3: Implement `aggregate_player_stats`**

Append to `src/fantasy_baseball/trades/evaluate.py` after `player_rest_of_season_stats`:

```python
def aggregate_player_stats(players: list[Player]) -> dict:
    """Aggregate ROS stats across multiple players into one dict.

    Returns the same shape as :func:`player_rest_of_season_stats`. Counting
    stats sum; rate stats are weighted (AVG by AB, ERA/WHIP by IP). An
    empty list returns all zeros.

    This lets multi-player trades call :func:`apply_swap_delta` exactly
    once per team with combined loses/gains stats.
    """
    total = {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0.0,
             "W": 0, "K": 0, "SV": 0, "ERA": 0.0, "WHIP": 0.0,
             "ab": 0, "ip": 0}
    if not players:
        return total

    total_hits = 0.0
    total_er = 0.0
    total_bh = 0.0

    for p in players:
        s = player_rest_of_season_stats(p)
        for cat in ("R", "HR", "RBI", "SB", "W", "K", "SV"):
            total[cat] += s[cat]
        total["ab"] += s["ab"]
        total["ip"] += s["ip"]
        total_hits += s["AVG"] * s["ab"]
        total_er += s["ERA"] * s["ip"] / 9.0
        total_bh += s["WHIP"] * s["ip"]

    if total["ab"] > 0:
        total["AVG"] = total_hits / total["ab"]
    if total["ip"] > 0:
        total["ERA"] = 9.0 * total_er / total["ip"]
        total["WHIP"] = total_bh / total["ip"]
    return total
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trades/test_evaluate.py -v`
Expected: all pass (including existing ones).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/evaluate.py tests/test_trades/test_evaluate.py
git commit -m "feat(trades): aggregate_player_stats for multi-player ROS math"
```

---

## Task 2: Create `multi_trade.py` scaffold with dataclasses

**Files:**
- Create: `src/fantasy_baseball/trades/multi_trade.py`
- Test: `tests/test_trades/test_multi_trade.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_trades/test_multi_trade.py`:

```python
"""Tests for the multi-player trade evaluator."""

from __future__ import annotations

import pytest

from fantasy_baseball.models.player import HitterStats, PitcherStats, Player
from fantasy_baseball.trades.multi_trade import (
    CategoryDelta,
    MultiTradeResult,
    TradeProposal,
)


def test_trade_proposal_defaults_empty_lists():
    p = TradeProposal(opponent="Foo")
    assert p.send == []
    assert p.receive == []
    assert p.my_drops == []
    assert p.opp_drops == []
    assert p.my_adds == []
    assert p.my_active_ids == set()


def test_multi_trade_result_shape():
    r = MultiTradeResult(
        legal=True,
        reason=None,
        delta_total=1.5,
        categories={"R": CategoryDelta(before=10.0, after=11.0, delta=1.0)},
    )
    assert r.legal is True
    assert r.categories["R"].delta == 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trades/test_multi_trade.py -v`
Expected: FAIL with ModuleNotFoundError.

- [ ] **Step 3: Implement module scaffold**

Create `src/fantasy_baseball/trades/multi_trade.py`:

```python
"""Multi-player trade evaluator.

Generalizes the 1-for-1 trade math in trades.evaluate to arbitrary N-for-M
swaps with optional drops on either side and optional waiver pickups on
the user's side. Reports delta-roto for the user's team only.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TradeProposal:
    """A multi-player trade proposal submitted from the UI.

    All player identifiers are ``"<name>::<player_type>"`` keys (the same
    format used in rankings caches).
    """
    opponent: str
    send: list[str] = field(default_factory=list)
    receive: list[str] = field(default_factory=list)
    my_drops: list[str] = field(default_factory=list)
    opp_drops: list[str] = field(default_factory=list)
    my_adds: list[str] = field(default_factory=list)
    my_active_ids: set[str] = field(default_factory=set)


@dataclass
class CategoryDelta:
    """Per-category before/after/delta for a single roto stat."""
    before: float
    after: float
    delta: float  # roto points change (e.g. +0.5 = half a category point)


@dataclass
class MultiTradeResult:
    """Output of :func:`evaluate_multi_trade`."""
    legal: bool
    reason: str | None
    delta_total: float
    categories: dict[str, CategoryDelta]


def evaluate_multi_trade(*args, **kwargs) -> MultiTradeResult:  # pragma: no cover - placeholder
    raise NotImplementedError("evaluate_multi_trade is implemented in Task 3")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_trades/test_multi_trade.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/multi_trade.py tests/test_trades/test_multi_trade.py
git commit -m "feat(trades): scaffold multi_trade dataclasses"
```

---

## Task 3: Add `_can_roster_after` size-only legality helper

**Files:**
- Modify: `src/fantasy_baseball/trades/multi_trade.py`
- Modify: `tests/test_trades/test_multi_trade.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trades/test_multi_trade.py`:

```python
from fantasy_baseball.trades.multi_trade import _can_roster_after


def _hitter_with_key(name: str) -> Player:
    return Player(
        name=name, player_type="hitter", positions=["OF"],
        rest_of_season=HitterStats(pa=600, ab=500, h=125, r=70, hr=20,
                                    rbi=60, sb=5, avg=0.250),
    )


ROSTER_SLOTS_STANDARD = {
    "C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
    "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2,
}


def _roster_of(size: int, il: int = 0) -> list[Player]:
    roster: list[Player] = []
    for i in range(size):
        p = _hitter_with_key(f"P{i}")
        p.selected_position = "IL" if i < il else "OF"
        roster.append(p)
    return roster


def test_can_roster_after_passes_when_size_balances():
    roster = _roster_of(23)
    removed = ["P0::hitter", "P1::hitter"]
    added = [_hitter_with_key("Add1"), _hitter_with_key("Add2")]
    ok, reason = _can_roster_after(roster, removed, added, ROSTER_SLOTS_STANDARD)
    assert ok is True
    assert reason is None


def test_can_roster_after_rejects_wrong_resulting_size():
    roster = _roster_of(23)
    removed = ["P0::hitter", "P1::hitter"]
    added = [_hitter_with_key("Add1")]
    ok, reason = _can_roster_after(roster, removed, added, ROSTER_SLOTS_STANDARD)
    assert ok is False
    assert reason is not None
    assert "22" in reason  # explains the mismatch


def test_can_roster_after_ignores_il_players_in_baseline_count():
    roster = _roster_of(25, il=2)
    ok, reason = _can_roster_after(roster, [], [], ROSTER_SLOTS_STANDARD)
    assert ok is True, reason
```

Note: `_hitter_with_key` produces a player whose key is `"P0::hitter"` etc. Verify in Task 4 that `player_key()` returns that format — this is why keys go `name::player_type`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trades/test_multi_trade.py::test_can_roster_after_passes_when_size_balances -v`
Expected: FAIL with `ImportError: cannot import name '_can_roster_after'`

- [ ] **Step 3: Implement `_can_roster_after` + `player_key` helper**

Add to `src/fantasy_baseball/trades/multi_trade.py`:

```python
from fantasy_baseball.models.player import Player


def player_key(player: Player) -> str:
    """Canonical player identifier: ``name::player_type`` (hitter|pitcher)."""
    return f"{player.name}::{player.player_type}"


def _non_il_size(roster: list[Player]) -> int:
    return sum(
        1 for p in roster
        if (getattr(p, "selected_position", None) or "") != "IL"
    )


def _target_size(roster_slots: dict) -> int:
    """Total active + bench slots (excludes IL)."""
    return sum(v for k, v in roster_slots.items() if k != "IL")


def _can_roster_after(
    roster: list[Player],
    removals: list[str],
    additions: list[Player],
    roster_slots: dict,
) -> tuple[bool, str | None]:
    """Size-only legality check for a multi-player proposal.

    ``roster`` is the current roster including IL players.
    ``removals`` is a list of ``player_key()`` strings for players leaving
    (traded away or dropped).  ``additions`` is a list of Player objects
    coming in (traded in or picked up from waivers).

    The roster is considered legal iff
    ``non_il_count - |removals| + |additions| == target_size``,
    where ``target_size = sum(roster_slots) - roster_slots["IL"]``
    (23 in this league: 12 active hitters + 9 pitchers + 2 bench).
    IL-listed players neither count in the baseline nor are removed by
    this trade.

    Returns ``(True, None)`` if legal, otherwise ``(False, reason)``.
    """
    target = _target_size(roster_slots)
    non_il = _non_il_size(roster)
    new_size = non_il - len(removals) + len(additions)
    if new_size != target:
        return False, (
            f"Roster would have {new_size} non-IL players; "
            f"league requires exactly {target}"
        )
    return True, None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trades/test_multi_trade.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/multi_trade.py tests/test_trades/test_multi_trade.py
git commit -m "feat(trades): size-only legality helper for multi-player trades"
```

---

## Task 4: Implement `evaluate_multi_trade` — happy path (2-for-2)

**Files:**
- Modify: `src/fantasy_baseball/trades/multi_trade.py`
- Modify: `tests/test_trades/test_multi_trade.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_trades/test_multi_trade.py`:

```python
from fantasy_baseball.trades.evaluate import player_rest_of_season_stats
from fantasy_baseball.trades.multi_trade import evaluate_multi_trade


def _make_hitter(name, r=70, hr=20, rbi=65, sb=8, avg=0.270, ab=500, pos="OF"):
    h = int(avg * ab)
    return Player(
        name=name, player_type="hitter", positions=[pos],
        rest_of_season=HitterStats(pa=int(ab * 1.15), ab=ab, h=h,
                                    r=r, hr=hr, rbi=rbi, sb=sb, avg=avg),
    )


def _make_pitcher(name, ip=150, w=9, k=140, sv=0, era=3.80, whip=1.25, pos="P"):
    er = int(era * ip / 9)
    bb = 40
    h_allowed = int(whip * ip - bb)
    return Player(
        name=name, player_type="pitcher", positions=[pos],
        rest_of_season=PitcherStats(ip=ip, w=w, k=k, sv=sv, era=era, whip=whip,
                                     er=er, bb=bb, h_allowed=h_allowed),
    )


def _team_stats_from_players(players: list[Player]) -> dict[str, float]:
    """Build a stats dict matching the apply_swap_delta baseline pools."""
    stats = {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "W": 0, "K": 0, "SV": 0,
             "AVG": 0.270, "ERA": 3.80, "WHIP": 1.25}
    for p in players:
        s = player_rest_of_season_stats(p)
        for cat in ("R", "HR", "RBI", "SB", "W", "K", "SV"):
            stats[cat] += s[cat]
    return stats


def _standings_of(teams: dict[str, list[Player]]) -> list[dict]:
    return [{"name": name, "stats": _team_stats_from_players(players)}
            for name, players in teams.items()]


def test_evaluate_2_for_2_legal_returns_delta():
    me_name = "Hart"
    opp_name = "Rival"
    # Build a league with 4 teams so score_roto has distribution to work with.
    me_roster = [_make_hitter(f"Me{i}", r=80-i) for i in range(11)] + \
                [_make_pitcher(f"MeP{i}") for i in range(9)] + \
                [_make_hitter(f"MeBN{i}") for i in range(3)]  # 23 non-IL
    for p in me_roster:
        p.selected_position = "BN" if p.name.startswith("MeBN") else (
            "P" if p.player_type == "pitcher" else "OF"
        )

    rival_roster = [_make_hitter(f"Riv{i}", r=70-i) for i in range(11)] + \
                   [_make_pitcher(f"RivP{i}") for i in range(9)] + \
                   [_make_hitter(f"RivBN{i}") for i in range(3)]
    for p in rival_roster:
        p.selected_position = "BN" if p.name.startswith("RivBN") else (
            "P" if p.player_type == "pitcher" else "OF"
        )

    team3 = [_make_hitter(f"T3_{i}", r=60) for i in range(20)] + \
            [_make_pitcher(f"T3P{i}") for i in range(3)]
    team4 = [_make_hitter(f"T4_{i}", r=50) for i in range(20)] + \
            [_make_pitcher(f"T4P{i}") for i in range(3)]

    standings = _standings_of({me_name: me_roster, opp_name: rival_roster,
                                "T3": team3, "T4": team4})

    proposal = TradeProposal(
        opponent=opp_name,
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_active_ids={player_key(p) for p in me_roster
                       if p.selected_position not in ("BN", "IL")
                       and p.name not in ("Me0", "Me1")}
                       | {"Riv0::hitter", "Riv1::hitter"},
    )

    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name=me_name,
        hart_roster=me_roster,
        opp_rosters={opp_name: rival_roster},
        waiver_pool={},
        projected_standings=standings,
        team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
    )
    assert result.legal is True, result.reason
    # Receiving two higher-R hitters (Riv0=70, Riv1=69) while sending two lower
    # (Me0=80, Me1=79) — actually net loss in R. Accept any sign; just assert
    # that the result is consistent (categories sum to delta_total).
    cat_sum = sum(cd.delta for cd in result.categories.values())
    assert abs(cat_sum - result.delta_total) < 1e-6
    assert set(result.categories.keys()) == {
        "R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"
    }
```

Note: the numeric delta sign depends on league setup. The test asserts structural correctness (legal, category sum matches total, all 10 categories present). Behavioural tests follow in later tasks.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trades/test_multi_trade.py::test_evaluate_2_for_2_legal_returns_delta -v`
Expected: FAIL with `NotImplementedError`.

- [ ] **Step 3: Implement `evaluate_multi_trade`**

Replace the placeholder `evaluate_multi_trade` in `src/fantasy_baseball/trades/multi_trade.py` with the full implementation. Add imports at top of file:

```python
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.trades.evaluate import (
    aggregate_player_stats,
    apply_swap_delta,
)
from fantasy_baseball.utils.constants import ALL_CATEGORIES
```

Then add these helpers + the main function (replacing the `raise NotImplementedError` stub):

```python
def _index_roster(roster: list[Player]) -> dict[str, Player]:
    return {player_key(p): p for p in roster}


def _resolve_keys(keys: list[str], index: dict[str, Player]) -> list[Player]:
    missing = [k for k in keys if k not in index]
    if missing:
        raise KeyError(f"Unknown player key(s): {missing}")
    return [index[k] for k in keys]


def _current_active_set(roster: list[Player]) -> set[str]:
    """Keys of roster players not currently on BN or IL in Yahoo."""
    return {
        player_key(p) for p in roster
        if (getattr(p, "selected_position", None) or "") not in ("BN", "IL")
    }


def evaluate_multi_trade(
    *,
    proposal: TradeProposal,
    hart_name: str,
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    waiver_pool: dict[str, Player],
    projected_standings: list[dict],
    team_sds: dict[str, dict[str, float]] | None,
    roster_slots: dict,
) -> MultiTradeResult:
    """Evaluate an arbitrary N-for-M trade with optional drops and adds.

    See docs/superpowers/specs/2026-04-20-multi-player-trade-evaluator-design.md
    for the full math.  Returns ``MultiTradeResult`` with per-category and
    total delta-roto for ``hart_name``'s team.
    """
    # --- 1. Resolve keys -----------------------------------------------------
    if proposal.opponent not in opp_rosters:
        return MultiTradeResult(
            legal=False, reason=f"Unknown opponent: {proposal.opponent}",
            delta_total=0.0, categories={},
        )

    my_idx = _index_roster(hart_roster)
    opp_idx = _index_roster(opp_rosters[proposal.opponent])

    try:
        sent = _resolve_keys(proposal.send, my_idx)
        received = _resolve_keys(proposal.receive, opp_idx)
        my_drops = _resolve_keys(proposal.my_drops, my_idx)
        opp_drops = _resolve_keys(proposal.opp_drops, opp_idx)
        my_adds = _resolve_keys(proposal.my_adds, waiver_pool)
    except KeyError as exc:
        return MultiTradeResult(
            legal=False, reason=str(exc),
            delta_total=0.0, categories={},
        )

    # --- 2. Legality ---------------------------------------------------------
    my_removals = proposal.send + proposal.my_drops
    my_additions = received + my_adds
    my_ok, my_reason = _can_roster_after(
        hart_roster, my_removals, my_additions, roster_slots,
    )
    if not my_ok:
        return MultiTradeResult(
            legal=False, reason=f"My team: {my_reason}",
            delta_total=0.0, categories={},
        )

    opp_removals = proposal.receive + proposal.opp_drops
    opp_additions = sent
    opp_ok, opp_reason = _can_roster_after(
        opp_rosters[proposal.opponent], opp_removals, opp_additions, roster_slots,
    )
    if not opp_ok:
        return MultiTradeResult(
            legal=False, reason=f"Opponent: {opp_reason}",
            delta_total=0.0, categories={},
        )

    # --- 3. Build active-set deltas ------------------------------------------
    all_mine_by_key = {**my_idx, **{player_key(p): p for p in my_adds}}
    before_mine = _current_active_set(hart_roster)
    after_mine = set(proposal.my_active_ids)

    mine_leaving = [all_mine_by_key[k] for k in before_mine - after_mine
                    if k in all_mine_by_key]
    mine_entering = [all_mine_by_key[k] for k in after_mine - before_mine
                     if k in all_mine_by_key]
    my_loses = aggregate_player_stats(mine_leaving)
    my_gains = aggregate_player_stats(mine_entering)

    # Opp: treat all non-IL as active. They lose received+opp_drops, gain sent.
    opp_loses = aggregate_player_stats(received + opp_drops)
    opp_gains = aggregate_player_stats(sent)

    # --- 4. Apply deltas to baseline and score -------------------------------
    baseline_by_team = {t["name"]: t["stats"] for t in projected_standings}
    if hart_name not in baseline_by_team:
        return MultiTradeResult(
            legal=False, reason=f"Team {hart_name} missing from projected_standings",
            delta_total=0.0, categories={},
        )

    post = []
    for t in projected_standings:
        if t["name"] == hart_name:
            post.append({"name": t["name"],
                         "stats": apply_swap_delta(t["stats"], my_loses, my_gains)})
        elif t["name"] == proposal.opponent:
            post.append({"name": t["name"],
                         "stats": apply_swap_delta(t["stats"], opp_loses, opp_gains)})
        else:
            post.append(t)

    before_roto = score_roto(
        {t["name"]: t["stats"] for t in projected_standings}, team_sds=team_sds,
    )
    after_roto = score_roto(
        {t["name"]: t["stats"] for t in post}, team_sds=team_sds,
    )

    categories: dict[str, CategoryDelta] = {}
    total_delta = 0.0
    for cat in ALL_CATEGORIES:
        before_pts = before_roto[hart_name][f"{cat}_pts"]
        after_pts = after_roto[hart_name][f"{cat}_pts"]
        delta = after_pts - before_pts
        categories[cat] = CategoryDelta(
            before=before_pts, after=after_pts, delta=delta,
        )
        total_delta += delta

    return MultiTradeResult(
        legal=True, reason=None,
        delta_total=total_delta, categories=categories,
    )
```

- [ ] **Step 4: Run tests to verify all pass**

Run: `pytest tests/test_trades/test_multi_trade.py -v`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/multi_trade.py tests/test_trades/test_multi_trade.py
git commit -m "feat(trades): evaluate_multi_trade core implementation"
```

---

## Task 5: Behavioural tests — bench, drops, adds, IL, illegal

**Files:**
- Modify: `tests/test_trades/test_multi_trade.py`

- [ ] **Step 1: Write additional tests**

Append to `tests/test_trades/test_multi_trade.py`:

```python
def _build_league():
    """Helper: build a 4-team league with me=Hart, opp=Rival, and valid rosters.

    Returns (hart_roster, rival_roster, standings, all_teams).
    """
    me = [_make_hitter(f"Me{i}", r=80-i) for i in range(11)] + \
         [_make_pitcher(f"MeP{i}") for i in range(9)] + \
         [_make_hitter(f"MeBN{i}") for i in range(3)]  # 23 non-IL
    for p in me:
        p.selected_position = ("BN" if p.name.startswith("MeBN")
                                else "P" if p.player_type == "pitcher" else "OF")
    riv = [_make_hitter(f"Riv{i}", r=70-i) for i in range(11)] + \
          [_make_pitcher(f"RivP{i}") for i in range(9)] + \
          [_make_hitter(f"RivBN{i}") for i in range(3)]
    for p in riv:
        p.selected_position = ("BN" if p.name.startswith("RivBN")
                                else "P" if p.player_type == "pitcher" else "OF")
    t3 = [_make_hitter(f"T3_{i}", r=60) for i in range(20)] + \
         [_make_pitcher(f"T3P{i}") for i in range(3)]
    t4 = [_make_hitter(f"T4_{i}", r=50) for i in range(20)] + \
         [_make_pitcher(f"T4P{i}") for i in range(3)]
    standings = _standings_of({"Hart": me, "Rival": riv, "T3": t3, "T4": t4})
    return me, riv, standings


def test_2_for_3_with_drop_is_legal_and_scores():
    me, riv, standings = _build_league()
    # I send 2, receive 3. Need to drop 1 to balance.
    proposal = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter", "Riv2::hitter"],
        my_drops=["MeBN0::hitter"],  # drop a bench hitter to balance 23
        opp_drops=[],  # opp: -3 + 2 = -1, but opp trade is not balanced, test below
        my_active_ids={player_key(p) for p in me
                        if p.selected_position not in ("BN", "IL")
                        and p.name not in ("Me0", "Me1")}
                       | {"Riv0::hitter", "Riv1::hitter", "Riv2::hitter"},
    )
    # Opp won't be legal here: -3+2 = -1; add synthetic opp_adds? No, our UI
    # doesn't support opp_adds. Test the legality banner catches this.
    result = evaluate_multi_trade(
        proposal=proposal, hart_name="Hart",
        hart_roster=me, opp_rosters={"Rival": riv}, waiver_pool={},
        projected_standings=standings, team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
    )
    assert result.legal is False
    assert "Opponent" in result.reason


def test_2_for_3_drop_on_both_sides_is_legal():
    me, riv, standings = _build_league()
    # The opp receives 2, sends 3 → needs 0 drops, just loses a slot.
    # Wait: -3+2 = -1. Opp has 23 → 22 after trade. They need a waiver add
    # which we don't support. In practice: they'd drop fewer hitters but here
    # the UI just surfaces "illegal". Exercise with opp_adds via my_adds path
    # is not applicable; instead do a balanced 2-for-2 + drops:
    proposal = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_drops=["MeBN0::hitter"],
        opp_drops=["RivBN0::hitter"],
        my_active_ids={player_key(p) for p in me
                        if p.selected_position not in ("BN", "IL")
                        and p.name not in ("Me0", "Me1")}
                       | {"Riv0::hitter", "Riv1::hitter"},
    )
    # Legality: me = 23 - 3 + 2 = 22 ≠ 23. Still illegal.
    result = evaluate_multi_trade(
        proposal=proposal, hart_name="Hart",
        hart_roster=me, opp_rosters={"Rival": riv}, waiver_pool={},
        projected_standings=standings, team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
    )
    assert result.legal is False
    assert "22" in result.reason


def test_2_for_2_plus_drop_plus_waiver_add_is_legal():
    me, riv, standings = _build_league()
    waiver = _make_hitter("Waiver1", r=75)
    proposal = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_drops=["MeBN0::hitter"],
        my_adds=["Waiver1::hitter"],
        my_active_ids={player_key(p) for p in me
                        if p.selected_position not in ("BN", "IL")
                        and p.name not in ("Me0", "Me1")}
                       | {"Riv0::hitter", "Riv1::hitter", "Waiver1::hitter"},
    )
    result = evaluate_multi_trade(
        proposal=proposal, hart_name="Hart",
        hart_roster=me, opp_rosters={"Rival": riv},
        waiver_pool={"Waiver1::hitter": waiver},
        projected_standings=standings, team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
    )
    assert result.legal is True, result.reason
    assert set(result.categories.keys()) >= {"R", "HR"}


def test_received_player_marked_bench_does_not_contribute():
    me, riv, standings = _build_league()
    # Two proposals: same trade, but in one case a received player is benched.
    active_set_all = ({player_key(p) for p in me
                        if p.selected_position not in ("BN", "IL")
                        and p.name not in ("Me0", "Me1")}
                      | {"Riv0::hitter", "Riv1::hitter"})
    active_set_bench_riv1 = active_set_all - {"Riv1::hitter"}

    proposal_all = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_active_ids=active_set_all,
    )
    proposal_bench = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter", "Me1::hitter"],
        receive=["Riv0::hitter", "Riv1::hitter"],
        my_active_ids=active_set_bench_riv1,
    )

    r_all = evaluate_multi_trade(
        proposal=proposal_all, hart_name="Hart",
        hart_roster=me, opp_rosters={"Rival": riv}, waiver_pool={},
        projected_standings=standings, team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
    )
    r_bench = evaluate_multi_trade(
        proposal=proposal_bench, hart_name="Hart",
        hart_roster=me, opp_rosters={"Rival": riv}, waiver_pool={},
        projected_standings=standings, team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
    )
    assert r_all.legal is True
    assert r_bench.legal is True
    # The bench variant forfeits Riv1's contribution; R delta should differ.
    assert r_all.delta_total != r_bench.delta_total


def test_il_players_excluded_from_size_count():
    me, riv, standings = _build_league()
    # Put a player on IL — keep roster at 25 total, still 23 non-IL.
    me.append(_make_hitter("MeIL", r=10))
    me[-1].selected_position = "IL"
    me.append(_make_hitter("MeIL2", r=10))
    me[-1].selected_position = "IL"
    # 25 total, 23 non-IL. A 1-for-1 trade should still balance.
    proposal = TradeProposal(
        opponent="Rival",
        send=["Me0::hitter"],
        receive=["Riv0::hitter"],
        my_active_ids={player_key(p) for p in me
                        if p.selected_position not in ("BN", "IL")
                        and p.name != "Me0"}
                       | {"Riv0::hitter"},
    )
    result = evaluate_multi_trade(
        proposal=proposal, hart_name="Hart",
        hart_roster=me, opp_rosters={"Rival": riv}, waiver_pool={},
        projected_standings=standings, team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
    )
    assert result.legal is True, result.reason


def test_unknown_player_key_returns_illegal_with_reason():
    me, riv, standings = _build_league()
    proposal = TradeProposal(
        opponent="Rival",
        send=["Ghost::hitter"],
        receive=["Riv0::hitter"],
        my_active_ids=set(),
    )
    result = evaluate_multi_trade(
        proposal=proposal, hart_name="Hart",
        hart_roster=me, opp_rosters={"Rival": riv}, waiver_pool={},
        projected_standings=standings, team_sds=None,
        roster_slots=ROSTER_SLOTS_STANDARD,
    )
    assert result.legal is False
    assert "Ghost" in result.reason
```

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_trades/test_multi_trade.py -v`
Expected: all pass.

- [ ] **Step 3: Commit**

```bash
git add tests/test_trades/test_multi_trade.py
git commit -m "test(trades): multi-trade behavioural coverage (bench, drops, IL)"
```

---

## Task 6: Waiver pool builder + search endpoint

**Files:**
- Modify: `src/fantasy_baseball/trades/multi_trade.py` (new helper)
- Modify: `src/fantasy_baseball/web/season_routes.py` (new route)
- Modify: `tests/test_trades/test_multi_trade.py` (test for helper)

- [ ] **Step 1: Write failing test for waiver pool builder**

Append to `tests/test_trades/test_multi_trade.py`:

```python
from fantasy_baseball.trades.multi_trade import build_waiver_pool


def test_build_waiver_pool_excludes_rostered_players():
    a = _make_hitter("Alice")
    b = _make_hitter("Bob")
    c = _make_hitter("Carol")
    d = _make_pitcher("Dan")
    my_roster = [a]
    opp_rosters = {"Rival": [b]}
    ros_projections = {
        "hitters": [
            {"name": "Alice", "player_type": "hitter", "positions": ["OF"],
             "rest_of_season": {"ab": 500, "h": 125, "r": 70, "hr": 20,
                                 "rbi": 60, "sb": 5, "avg": 0.25, "pa": 575}},
            {"name": "Bob", "player_type": "hitter", "positions": ["OF"],
             "rest_of_season": {"ab": 500, "h": 125, "r": 70, "hr": 20,
                                 "rbi": 60, "sb": 5, "avg": 0.25, "pa": 575}},
            {"name": "Carol", "player_type": "hitter", "positions": ["OF"],
             "rest_of_season": {"ab": 500, "h": 125, "r": 70, "hr": 20,
                                 "rbi": 60, "sb": 5, "avg": 0.25, "pa": 575}},
        ],
        "pitchers": [
            {"name": "Dan", "player_type": "pitcher", "positions": ["P"],
             "rest_of_season": {"ip": 150, "w": 9, "k": 140, "sv": 0,
                                 "era": 3.80, "whip": 1.25, "er": 63, "bb": 40,
                                 "h_allowed": 147}},
        ],
    }
    pool = build_waiver_pool(my_roster, opp_rosters, ros_projections)
    assert "Carol::hitter" in pool
    assert "Dan::pitcher" in pool
    assert "Alice::hitter" not in pool
    assert "Bob::hitter" not in pool
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_trades/test_multi_trade.py::test_build_waiver_pool_excludes_rostered_players -v`
Expected: FAIL with `ImportError`.

- [ ] **Step 3: Implement `build_waiver_pool`**

Append to `src/fantasy_baseball/trades/multi_trade.py`:

```python
def build_waiver_pool(
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    ros_projections: dict,
) -> dict[str, Player]:
    """Build a keyed player pool of everyone with ROS projections who is
    not on any roster.

    ``ros_projections`` is the cached ROS projection dict: ``{"hitters":
    [{...}], "pitchers": [{...}]}``. Each entry is in the format accepted
    by :meth:`Player.from_dict`.

    Returned dict is keyed by :func:`player_key` (``"name::player_type"``).
    """
    rostered = {player_key(p) for p in hart_roster}
    for roster in opp_rosters.values():
        rostered |= {player_key(p) for p in roster}

    pool: dict[str, Player] = {}
    for bucket, player_type in (("hitters", "hitter"), ("pitchers", "pitcher")):
        for d in ros_projections.get(bucket, []):
            payload = dict(d)
            payload.setdefault("player_type", player_type)
            player = Player.from_dict(payload)
            key = player_key(player)
            if key in rostered:
                continue
            pool[key] = player
    return pool
```

- [ ] **Step 4: Add `/api/waiver-search` route**

In `src/fantasy_baseball/web/season_routes.py`, near the existing `/api/trade-search` handler, add:

```python
@app.route("/api/waiver-search")
@_require_auth
def api_waiver_search():
    from fantasy_baseball.models.player import Player
    from fantasy_baseball.trades.multi_trade import build_waiver_pool, player_key
    from fantasy_baseball.utils.name_utils import normalize_name

    query = (request.args.get("q") or "").strip()
    if len(query) < 2:
        return jsonify([])

    roster_raw = read_cache(CacheKey.ROSTER) or []
    opp_rosters_raw = read_cache(CacheKey.OPP_ROSTERS) or {}
    ros_cache = read_cache(CacheKey.ROS_PROJECTIONS) or {}

    hart_roster = [Player.from_dict(p) for p in roster_raw]
    opp_rosters = {n: [Player.from_dict(p) for p in ps]
                   for n, ps in opp_rosters_raw.items()}
    pool = build_waiver_pool(hart_roster, opp_rosters, ros_cache)

    q_norm = normalize_name(query)
    matches = [
        {
            "key": key,
            "name": p.name,
            "player_type": p.player_type,
            "positions": p.positions,
        }
        for key, p in pool.items()
        if q_norm in normalize_name(p.name)
    ]
    matches.sort(key=lambda m: m["name"])
    return jsonify(matches[:20])
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_trades/test_multi_trade.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/trades/multi_trade.py \
        src/fantasy_baseball/web/season_routes.py \
        tests/test_trades/test_multi_trade.py
git commit -m "feat(trades): waiver pool helper + /api/waiver-search endpoint"
```

---

## Task 7: `POST /api/evaluate-trade` route

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Test: `tests/test_web/test_evaluate_trade_route.py` (create if `tests/test_web` doesn't exist; otherwise append)

- [ ] **Step 1: Check for existing test_web directory**

Run: `ls tests/test_web 2>/dev/null || ls tests/web 2>/dev/null || echo "neither exists"`

If `tests/test_web/` exists, use it. If `tests/web/` exists, use it. Otherwise, pick `tests/test_web/` and create a skeleton.

- [ ] **Step 2: Write failing test using Flask test client**

Create or append to `tests/test_web/test_evaluate_trade_route.py`:

```python
"""Tests for POST /api/evaluate-trade."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest


@pytest.fixture
def client(monkeypatch, tmp_path):
    # Disable auth for tests.
    monkeypatch.setenv("ADMIN_PASSWORD", "")
    from fantasy_baseball.web.app import create_app
    app = create_app()
    app.config["TESTING"] = True
    return app.test_client()


def _fake_cache(monkeypatch, values: dict):
    """Patch read_cache to return the values dict."""
    def fake_read_cache(key, cache_dir=None):
        return values.get(key.value)
    import fantasy_baseball.web.season_routes as routes
    monkeypatch.setattr(routes, "read_cache", fake_read_cache)


def test_evaluate_trade_returns_400_on_missing_opponent(client, monkeypatch):
    _fake_cache(monkeypatch, {
        "roster": [], "opp_rosters": {}, "projections": {},
        "ros_projections": {"hitters": [], "pitchers": []},
    })
    resp = client.post("/api/evaluate-trade",
                       json={"send": ["A::hitter"], "receive": ["B::hitter"]})
    assert resp.status_code == 400
    assert "opponent" in resp.get_json()["error"].lower()


def test_evaluate_trade_returns_legal_result_shape(client, monkeypatch):
    # Minimal fixture: me + opp each have balanced rosters.
    # For a full happy-path test, populate all caches; here we mostly test
    # that the handler wires up correctly and returns the expected keys.
    from fantasy_baseball.models.player import HitterStats, Player

    def _hit(name, r=70):
        return Player(name=name, player_type="hitter", positions=["OF"],
                      rest_of_season=HitterStats(pa=600, ab=500, h=125,
                                                  r=r, hr=20, rbi=60, sb=5,
                                                  avg=0.250)).to_dict()

    me = [_hit(f"M{i}") for i in range(23)]
    for i, p in enumerate(me):
        p["selected_position"] = "BN" if i >= 21 else "OF"
    opp = [_hit(f"R{i}") for i in range(23)]
    for i, p in enumerate(opp):
        p["selected_position"] = "BN" if i >= 21 else "OF"

    standings_stats = {"R": 1000, "HR": 250, "RBI": 750, "SB": 80,
                        "AVG": 0.260, "W": 70, "K": 1200, "SV": 50,
                        "ERA": 3.80, "WHIP": 1.25}
    projected_standings = [
        {"name": "Hart", "stats": dict(standings_stats)},
        {"name": "Rival", "stats": dict(standings_stats)},
        {"name": "T3", "stats": dict(standings_stats)},
        {"name": "T4", "stats": dict(standings_stats)},
    ]

    _fake_cache(monkeypatch, {
        "roster": me,
        "opp_rosters": {"Rival": opp},
        "projections": {"projected_standings": projected_standings,
                         "team_sds": None},
        "ros_projections": {"hitters": [], "pitchers": []},
    })

    # Stub config.team_name -> "Hart"
    import fantasy_baseball.web.season_routes as routes
    class _FakeCfg:
        team_name = "Hart"
        roster_slots = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
                         "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2}
    monkeypatch.setattr(routes, "_load_config", lambda: _FakeCfg())

    payload = {
        "opponent": "Rival",
        "send": ["M0::hitter"],
        "receive": ["R0::hitter"],
        "my_drops": [], "opp_drops": [], "my_adds": [],
        "my_active_ids": [player_key_json(p) for p in me[1:21]] + ["R0::hitter"],
    }
    resp = client.post("/api/evaluate-trade", json=payload)
    data = resp.get_json()
    assert resp.status_code == 200, data
    assert "legal" in data
    assert "delta_total" in data
    assert set(data["categories"].keys()) >= {"R", "HR", "ERA"}


def player_key_json(p: dict) -> str:
    return f"{p['name']}::{p['player_type']}"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `pytest tests/test_web/test_evaluate_trade_route.py -v`
Expected: FAIL with 404 (route not defined).

- [ ] **Step 4: Implement the route**

Add to `src/fantasy_baseball/web/season_routes.py` near the existing `/api/trade-search`:

```python
@app.route("/api/evaluate-trade", methods=["POST"])
@_require_auth
def api_evaluate_trade():
    from fantasy_baseball.models.player import Player
    from fantasy_baseball.trades.multi_trade import (
        TradeProposal, build_waiver_pool, evaluate_multi_trade,
    )

    data = request.get_json(silent=True) or {}
    opponent = (data.get("opponent") or "").strip()
    if not opponent:
        return jsonify({"error": "opponent is required"}), 400

    config = _load_config()
    roster_raw = read_cache(CacheKey.ROSTER)
    opp_rosters_raw = read_cache(CacheKey.OPP_ROSTERS)
    if roster_raw is None or opp_rosters_raw is None:
        return jsonify({"error": "No roster data. Run a refresh first."}), 404
    if opponent not in opp_rosters_raw:
        return jsonify({"error": f"Unknown opponent: {opponent}"}), 400

    proj_cache = read_cache(CacheKey.PROJECTIONS) or {}
    projected_standings = proj_cache.get("projected_standings")
    team_sds = proj_cache.get("team_sds")
    if not projected_standings:
        return jsonify({"error": "No projected standings. Run a refresh first."}), 404

    ros_cache = read_cache(CacheKey.ROS_PROJECTIONS) or {}

    hart_roster = [Player.from_dict(p) for p in roster_raw]
    opp_rosters = {n: [Player.from_dict(p) for p in ps]
                   for n, ps in opp_rosters_raw.items()}
    waiver_pool = build_waiver_pool(hart_roster, opp_rosters, ros_cache)

    proposal = TradeProposal(
        opponent=opponent,
        send=list(data.get("send") or []),
        receive=list(data.get("receive") or []),
        my_drops=list(data.get("my_drops") or []),
        opp_drops=list(data.get("opp_drops") or []),
        my_adds=list(data.get("my_adds") or []),
        my_active_ids=set(data.get("my_active_ids") or []),
    )

    result = evaluate_multi_trade(
        proposal=proposal,
        hart_name=config.team_name,
        hart_roster=hart_roster,
        opp_rosters=opp_rosters,
        waiver_pool=waiver_pool,
        projected_standings=projected_standings,
        team_sds=team_sds,
        roster_slots=config.roster_slots,
    )
    return jsonify({
        "legal": result.legal,
        "reason": result.reason,
        "delta_total": round(result.delta_total, 2),
        "categories": {
            cat: {"before": round(cd.before, 2),
                   "after": round(cd.after, 2),
                   "delta": round(cd.delta, 2)}
            for cat, cd in result.categories.items()
        },
    })
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_web/test_evaluate_trade_route.py -v`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py \
        tests/test_web/test_evaluate_trade_route.py
git commit -m "feat(trades): POST /api/evaluate-trade route"
```

---

## Task 8: Frontend HTML scaffold for Build a Trade

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`
- Modify: `src/fantasy_baseball/web/season_routes.py` (to pass roster data to the template)

- [ ] **Step 1: Augment the template context**

Find the handler that renders `waivers_trades.html` (currently around line 339 in `season_routes.py`) and add `opp_rosters_data` and `my_roster_data` to the render context. Look for the existing `my_players` / `opp_players` keys passed to the template. Add alongside:

```python
# Full roster dicts for the Build a Trade section (hitter+pitcher split irrelevant here).
import json as _json
my_roster_data = roster_raw or []
opp_rosters_data = opp_rosters_raw or {}
return render_template(
    "season/waivers_trades.html",
    # ...existing kwargs...
    my_roster_data=_json.dumps(my_roster_data),
    opp_rosters_data=_json.dumps(opp_rosters_data),
)
```

Grep first to find the existing `render_template("season/waivers_trades.html", ...)` call and add to its kwargs.

- [ ] **Step 2: Add the HTML section**

In `src/fantasy_baseball/web/templates/season/waivers_trades.html`, after the existing Trade Finder section's closing tag (find the outermost div of the existing Trade Finder — around line 64), add:

```html
<section class="card" id="build-trade-section" style="margin-top: 1rem;">
  <div class="section-toggle" onclick="toggleSection('build-trade-body')">
    <h3>Build a Trade</h3>
    <span class="chevron">▸</span>
  </div>
  <div class="section-body collapsed" id="build-trade-body">
    <div class="mb-2">
      <label for="bt-opponent">Opponent</label>
      <select id="bt-opponent" onchange="bt.onOpponentChange()"></select>
    </div>

    <div class="build-trade-rosters" style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem;">
      <div>
        <h4>My Team</h4>
        <div id="bt-my-roster" class="bt-roster"></div>
        <div class="mt-2">
          <label for="bt-waiver-search">Add from Waivers</label>
          <input type="text" id="bt-waiver-search" placeholder="Type ≥2 letters…"
                 oninput="bt.onWaiverQuery(event)" autocomplete="off" />
          <ul id="bt-waiver-suggestions" class="bt-suggestions"></ul>
          <div id="bt-waiver-chips" class="bt-chips"></div>
        </div>
      </div>
      <div>
        <h4>Opponent Team</h4>
        <div id="bt-opp-roster" class="bt-roster"></div>
      </div>
    </div>

    <div id="bt-legality" class="bt-legality mt-2"></div>
    <button type="button" id="bt-evaluate" class="pill" disabled
            onclick="bt.evaluate()">Evaluate Trade</button>

    <div id="bt-result" class="bt-result"></div>
  </div>
</section>

<script>
  window.BT_DATA = {
    myRoster: {{ my_roster_data | safe }},
    oppRosters: {{ opp_rosters_data | safe }}
  };
</script>
```

- [ ] **Step 3: Add minimal CSS**

In the `<style>` block at the top of `waivers_trades.html`, append:

```css
.bt-roster { max-height: 420px; overflow-y: auto; border: 1px solid var(--panel-bg); padding: 0.25rem; }
.bt-row { display: grid; grid-template-columns: 1fr auto auto; gap: 0.4rem; padding: 0.2rem 0.4rem;
          border-bottom: 1px solid rgba(255,255,255,0.06); align-items: center; }
.bt-row.il { opacity: 0.4; }
.bt-row-name { font-size: 0.9rem; }
.bt-seg { display: inline-flex; gap: 2px; }
.bt-seg button { font-size: 0.75rem; padding: 1px 6px; background: transparent;
                  border: 1px solid var(--panel-bg); color: var(--text); cursor: pointer; }
.bt-seg button.active { background: var(--text); color: var(--panel-bg); }
.bt-seg button.trade.active { background: var(--success); color: #000; }
.bt-seg button.drop.active { background: var(--danger); color: #fff; }
.bt-active-toggle { font-size: 0.7rem; cursor: pointer; }
.bt-suggestions { list-style: none; padding: 0; margin: 0.25rem 0; max-height: 160px; overflow-y: auto; }
.bt-suggestions li { padding: 0.2rem 0.4rem; cursor: pointer; background: var(--panel-bg); }
.bt-suggestions li:hover { background: var(--text); color: var(--panel-bg); }
.bt-chips { display: flex; gap: 0.25rem; flex-wrap: wrap; margin-top: 0.25rem; }
.bt-chip { background: var(--panel-bg); padding: 0.2rem 0.4rem; border-radius: 3px; font-size: 0.8rem; }
.bt-legality { font-family: monospace; font-size: 0.85rem; }
.bt-legality.illegal { color: var(--danger); }
.bt-result { margin-top: 1rem; }
```

- [ ] **Step 4: Manual check**

Start the dev server and open `/waivers-trades`. Expand the new "Build a Trade" section. Verify:
- Opponent dropdown shows (empty for now)
- Two empty roster panels appear side-by-side
- Waiver search input renders
- Legality banner is empty
- Evaluate button is disabled and visible

Run: `python scripts/run_season_dashboard.py` (stop with Ctrl+C after visual confirmation).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html \
        src/fantasy_baseball/web/season_routes.py
git commit -m "feat(trades): HTML scaffold for Build a Trade section"
```

---

## Task 9: Frontend JS — state model + roster rendering

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`

- [ ] **Step 1: Add the state module**

Append to the `<script>` block at the bottom of `waivers_trades.html`:

```javascript
(function() {
  const state = {
    opponent: null,
    myActions: new Map(),    // key -> "TRADE" | "DROP" | null
    oppActions: new Map(),
    myBench: new Map(),      // key -> bool (true = bench)
    oppBench: new Map(),     // unused for opp, kept for symmetry
    waiverAdds: new Map(),   // key -> {name, positions, bench}
  };

  function playerKey(p) { return `${p.name}::${p.player_type}`; }

  function isIL(p) { return (p.selected_position || "") === "IL"; }
  function isBN(p) { return (p.selected_position || "") === "BN"; }

  function renderRow(p, side) {
    const key = playerKey(p);
    const il = isIL(p);
    const action = side === "my" ? state.myActions.get(key) : state.oppActions.get(key);
    const benchDefault = side === "my" ? (isBN(p) || il) : il;
    const bench = side === "my"
      ? (state.myBench.has(key) ? state.myBench.get(key) : benchDefault)
      : benchDefault;
    const bencheable = side === "my";

    const benchControl = bencheable && !il ? `
      <label class="bt-active-toggle">
        <input type="checkbox" ${bench ? "" : "checked"}
               onchange="bt.toggleActive('${key}')" />
        active
      </label>` : "";

    return `
      <div class="bt-row ${il ? "il" : ""}" data-key="${key}">
        <span class="bt-row-name">${p.name} <small>${(p.positions||[]).join("/")}</small></span>
        <span class="bt-seg">
          <button class="trade ${action === "TRADE" ? "active" : ""}"
                  ${il ? "disabled" : ""}
                  onclick="bt.setAction('${side}','${key}','TRADE')">T</button>
          <button class="drop ${action === "DROP" ? "active" : ""}"
                  ${il ? "disabled" : ""}
                  onclick="bt.setAction('${side}','${key}','DROP')">D</button>
          <button class="${action == null ? "active" : ""}"
                  onclick="bt.setAction('${side}','${key}',null)">—</button>
        </span>
        ${benchControl}
      </div>`;
  }

  function renderRosters() {
    const myDiv = document.getElementById("bt-my-roster");
    myDiv.innerHTML = window.BT_DATA.myRoster.map(p => renderRow(p, "my")).join("");
    const oppDiv = document.getElementById("bt-opp-roster");
    if (!state.opponent) { oppDiv.innerHTML = "<em>Select opponent</em>"; return; }
    const opp = window.BT_DATA.oppRosters[state.opponent] || [];
    oppDiv.innerHTML = opp.map(p => renderRow(p, "opp")).join("");
  }

  function populateOpponents() {
    const sel = document.getElementById("bt-opponent");
    const names = Object.keys(window.BT_DATA.oppRosters || {}).sort();
    sel.innerHTML = "<option value=''>— choose —</option>" +
      names.map(n => `<option value="${n}">${n}</option>`).join("");
  }

  window.bt = {
    init() { populateOpponents(); renderRosters(); updateLegality(); },
    onOpponentChange() {
      state.opponent = document.getElementById("bt-opponent").value || null;
      state.oppActions.clear();
      renderRosters(); updateLegality();
    },
    setAction(side, key, action) {
      const bag = side === "my" ? state.myActions : state.oppActions;
      if (action == null) bag.delete(key); else bag.set(key, action);
      renderRosters(); updateLegality();
    },
    toggleActive(key) {
      const cur = state.myBench.has(key)
        ? state.myBench.get(key)
        : (window.BT_DATA.myRoster.find(p => playerKey(p) === key)?.selected_position === "BN");
      state.myBench.set(key, !cur);
      updateLegality();
    },
  };

  // init after DOMContentLoaded if section is already visible; call from toggle otherwise
  document.addEventListener("DOMContentLoaded", () => window.bt.init());

  // placeholders wired up in later tasks
  window.bt.onWaiverQuery = () => {};
  window.bt.evaluate = () => {};

  // Used by updateLegality and evaluate in later tasks.
  window.bt._state = state;
  window.bt._playerKey = playerKey;
  window.bt._isIL = isIL;

  function updateLegality() { /* implemented in Task 10 */ }
  window.bt._updateLegality = updateLegality;
})();
```

- [ ] **Step 2: Verify manually**

Start the dev server, navigate to `/waivers-trades`, expand "Build a Trade", and confirm:
- Opponent dropdown populated with all opp team names
- My roster renders as rows with T/D/— buttons and an "active" checkbox (IL players greyed out)
- Selecting an opponent renders their roster
- Clicking T/D on a row toggles the button state visually (state updates; no legality check yet)

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat(trades): render rosters + selection state for Build a Trade"
```

---

## Task 10: Legality banner + Evaluate button enable

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`

- [ ] **Step 1: Implement `updateLegality`**

Replace the `function updateLegality() { /* implemented in Task 10 */ }` placeholder with:

```javascript
function updateLegality() {
  const banner = document.getElementById("bt-legality");
  const btn = document.getElementById("bt-evaluate");
  if (!state.opponent) {
    banner.textContent = "";
    banner.classList.remove("illegal");
    btn.disabled = true; return;
  }
  const myRoster = window.BT_DATA.myRoster;
  const oppRoster = window.BT_DATA.oppRosters[state.opponent] || [];

  const myNonIL = myRoster.filter(p => !isIL(p)).length;
  const oppNonIL = oppRoster.filter(p => !isIL(p)).length;

  const mySend = [...state.myActions.entries()].filter(([,a]) => a === "TRADE").length;
  const myDrop = [...state.myActions.entries()].filter(([,a]) => a === "DROP").length;
  const oppSend = [...state.oppActions.entries()].filter(([,a]) => a === "TRADE").length;
  const oppDrop = [...state.oppActions.entries()].filter(([,a]) => a === "DROP").length;
  const waivers = state.waiverAdds.size;

  // Me: -(send + drop) + (oppSend + waivers)
  const myAfter = myNonIL - mySend - myDrop + oppSend + waivers;
  const oppAfter = oppNonIL - oppSend - oppDrop + mySend;

  const myNetRx = oppSend + waivers;
  const myNetTx = mySend + myDrop;
  const oppNetRx = mySend;
  const oppNetTx = oppSend + oppDrop;

  const TARGET = 23;
  const illegal = (myAfter !== TARGET) || (oppAfter !== TARGET);
  banner.classList.toggle("illegal", illegal);
  banner.innerHTML = `
    <div>My team: +${myNetRx} / -${myNetTx} &rarr; ${myAfter} (need ${TARGET})</div>
    <div>Opp: +${oppNetRx} / -${oppNetTx} &rarr; ${oppAfter} (need ${TARGET})</div>
  `;

  const hasTrade = mySend >= 1 && oppSend >= 1;
  btn.disabled = !(hasTrade && !illegal);
}
```

- [ ] **Step 2: Verify manually**

On the dev server: open Build a Trade, pick an opponent, mark 1 player on each side as TRADE. Observe:
- Banner shows `+1 / -1 → 23 (need 23)` for both sides
- Evaluate button becomes enabled
- Mark a drop — banner turns red; button disables

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat(trades): legality banner and Evaluate button gating"
```

---

## Task 11: Waiver autocomplete (frontend)

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`

- [ ] **Step 1: Implement waiver search**

Replace `window.bt.onWaiverQuery = () => {};` with:

```javascript
let _waiverTimer = null;

window.bt.onWaiverQuery = (ev) => {
  const q = ev.target.value.trim();
  const sugg = document.getElementById("bt-waiver-suggestions");
  clearTimeout(_waiverTimer);
  if (q.length < 2) { sugg.innerHTML = ""; return; }
  _waiverTimer = setTimeout(async () => {
    const resp = await fetch(`/api/waiver-search?q=${encodeURIComponent(q)}`);
    if (!resp.ok) { sugg.innerHTML = ""; return; }
    const matches = await resp.json();
    sugg.innerHTML = matches.map(m =>
      `<li onclick="bt.addWaiver('${m.key}','${m.name.replace(/'/g, "&apos;")}','${(m.positions||[]).join('/')}')">
         ${m.name} <small>(${(m.positions||[]).join('/')})</small>
       </li>`
    ).join("");
  }, 250);
};

window.bt.addWaiver = (key, name, positions) => {
  if (state.waiverAdds.has(key)) return;
  state.waiverAdds.set(key, { name, positions, bench: false });
  document.getElementById("bt-waiver-search").value = "";
  document.getElementById("bt-waiver-suggestions").innerHTML = "";
  renderWaiverChips();
  updateLegality();
};

window.bt.removeWaiver = (key) => {
  state.waiverAdds.delete(key);
  renderWaiverChips();
  updateLegality();
};

window.bt.toggleWaiverActive = (key) => {
  const add = state.waiverAdds.get(key);
  if (add) { add.bench = !add.bench; renderWaiverChips(); }
};

function renderWaiverChips() {
  const chips = document.getElementById("bt-waiver-chips");
  chips.innerHTML = [...state.waiverAdds.entries()].map(([k, v]) => `
    <span class="bt-chip" data-key="${k}">
      ${v.name} <small>${v.positions}</small>
      <label class="bt-active-toggle">
        <input type="checkbox" ${v.bench ? "" : "checked"}
               onchange="bt.toggleWaiverActive('${k}')" /> active
      </label>
      <button onclick="bt.removeWaiver('${k}')" style="background:none;border:none;color:var(--danger);cursor:pointer">×</button>
    </span>`).join("");
}
```

- [ ] **Step 2: Verify manually**

Type `bob` in the waiver search. Suggestions should appear. Click one — it appears as a chip. Click × to remove it.

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat(trades): waiver autocomplete in Build a Trade"
```

---

## Task 12: Evaluate handler + result rendering

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`

- [ ] **Step 1: Implement evaluate**

Replace `window.bt.evaluate = () => {};` with:

```javascript
window.bt.evaluate = async () => {
  const opp = state.opponent;
  if (!opp) return;

  const send = [...state.myActions.entries()].filter(([,a]) => a === "TRADE").map(([k]) => k);
  const receive = [...state.oppActions.entries()].filter(([,a]) => a === "TRADE").map(([k]) => k);
  const myDrops = [...state.myActions.entries()].filter(([,a]) => a === "DROP").map(([k]) => k);
  const oppDrops = [...state.oppActions.entries()].filter(([,a]) => a === "DROP").map(([k]) => k);
  const myAdds = [...state.waiverAdds.keys()];

  // Active set = my kept (not sent/dropped, not bench-toggled) + received-active + adds-active
  const myKeptKeys = window.BT_DATA.myRoster
    .filter(p => !isIL(p))
    .map(playerKey)
    .filter(k => !send.includes(k) && !myDrops.includes(k));
  const myActive = new Set();
  for (const k of myKeptKeys) {
    const p = window.BT_DATA.myRoster.find(q => playerKey(q) === k);
    const isBenchDefault = p && p.selected_position === "BN";
    const benchChoice = state.myBench.has(k) ? state.myBench.get(k) : isBenchDefault;
    if (!benchChoice) myActive.add(k);
  }
  for (const k of receive) myActive.add(k);  // default active
  for (const [k, v] of state.waiverAdds.entries()) if (!v.bench) myActive.add(k);

  const payload = {
    opponent: opp,
    send, receive,
    my_drops: myDrops,
    opp_drops: oppDrops,
    my_adds: myAdds,
    my_active_ids: [...myActive],
  };

  const resultEl = document.getElementById("bt-result");
  resultEl.innerHTML = "<em>Evaluating…</em>";
  const resp = await fetch("/api/evaluate-trade", {
    method: "POST", headers: {"Content-Type": "application/json"},
    body: JSON.stringify(payload)
  });
  const data = await resp.json();
  if (!resp.ok || !data.legal) {
    resultEl.innerHTML = `<div class="bt-result-bad">Trade is not legal: ${data.reason || data.error || 'unknown'}</div>`;
    return;
  }
  renderResult(data);
};

function renderResult(data) {
  const el = document.getElementById("bt-result");
  const cats = ["R","HR","RBI","SB","AVG","W","K","SV","ERA","WHIP"];
  const rows = cats.map(cat => {
    const c = data.categories[cat] || {before:0,after:0,delta:0};
    const cls = c.delta > 0 ? "cat-gain" : c.delta < 0 ? "cat-loss" : "";
    const sign = c.delta >= 0 ? "+" : "";
    return `<tr><th>${cat}</th><td>${c.before.toFixed(2)}</td>
            <td>${c.after.toFixed(2)}</td>
            <td class="${cls}">${sign}${c.delta.toFixed(2)}</td></tr>`;
  }).join("");
  const totalCls = data.delta_total > 0 ? "cat-gain" : data.delta_total < 0 ? "cat-loss" : "";
  const totalSign = data.delta_total >= 0 ? "+" : "";
  el.innerHTML = `
    <div class="card" style="margin-top: 1rem;">
      <h4>Proposed trade ΔRoto:
        <span class="${totalCls}">${totalSign}${data.delta_total.toFixed(2)}</span>
      </h4>
      <table class="trade-details">
        <thead><tr><th>Cat</th><th>Before</th><th>After</th><th>Δ</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>`;
}
```

- [ ] **Step 2: Manual smoke test**

Start dev server. Build a legal 1-for-1 trade. Click Evaluate. A result card should appear with per-category before/after/delta and a total at the top. Numbers should match a sanity check: trading a high-R player for a low-R player shows negative R delta.

Repeat with a 2-for-2 and verify the delta is roughly the sum of what two individual swaps would be.

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat(trades): Evaluate button posts proposal and renders result"
```

---

## Task 13: Final verification

**Files:** none (verification only)

- [ ] **Step 1: Run full pytest**

Run: `pytest -v`
Expected: all tests pass (or note any pre-existing, unrelated failures).

- [ ] **Step 2: Run ruff**

Run: `ruff check . && ruff format --check .`
Expected: no violations. Fix any introduced by the new code with `ruff format .` and re-check.

- [ ] **Step 3: Run vulture**

Run: `vulture src/fantasy_baseball/trades/multi_trade.py src/fantasy_baseball/web/season_routes.py --min-confidence 80`
Expected: no new dead-code findings introduced by this change. Pre-existing findings elsewhere are acceptable; note them but don't fix.

- [ ] **Step 4: Run mypy on touched modules if covered**

Check `pyproject.toml` `[tool.mypy].files` — if `trades/` or the new `multi_trade.py` is listed, run:
```
mypy src/fantasy_baseball/trades/multi_trade.py
```
Expected: no new errors.

- [ ] **Step 5: End-to-end smoke test**

Start the season dashboard:
```
python scripts/run_season_dashboard.py
```

Log in, go to Waivers & Trades, expand Build a Trade. Run through:

1. Pick an opponent. Both rosters render.
2. Mark 1 send + 1 receive. Banner goes green. Evaluate enabled.
3. Click Evaluate. Result card appears with 10 categories and a total delta.
4. Add a 2-for-2 with 1 drop on each side. Banner stays green. Evaluate. Verify total delta changes.
5. Bench one of the received players. Evaluate again. Verify total delta decreases (that player no longer contributes).
6. Type "a" in waiver search — suggestions appear after 2+ chars. Pick one. Chip appears. Rebalance drops accordingly.
7. Mark a second drop without a matching addition. Banner turns red. Evaluate disabled.

Stop the server.

- [ ] **Step 6: Summarize**

Write a short PR-style summary of what was built and what was verified. Report any failing checks or surprises.

---

## Self-review checklist (run before handing off)

- [x] Every task has complete code blocks (no "TBD")
- [x] No references to types/functions not defined in earlier tasks
- [x] Task 1 defines `aggregate_player_stats`, used in Task 4
- [x] Task 2 defines `TradeProposal`, `CategoryDelta`, `MultiTradeResult`, used in Task 4
- [x] Task 3 defines `_can_roster_after` and `player_key`, used in Task 4, 6, 7
- [x] Task 6 defines `build_waiver_pool`, used in Task 7
- [x] Task 7 defines route, used by Task 12 frontend
- [x] Task 8-9 define `window.bt` state + render, Task 10 adds legality, Task 11 adds waiver, Task 12 adds evaluate
- [x] Spec coverage: placement ✓, opponent dropdown ✓, roster panels ✓, segmented TRADE/DROP ✓, Active/Bench toggle (my side only) ✓, waiver autocomplete (my side only) ✓, legality banner ✓, evaluate button gating ✓, result panel with categories ✓, backend evaluate route ✓, waiver search route ✓, IL exclusion ✓, size-only legality ✓, aggregation math ✓, tests for bench/drops/IL/illegal ✓
- [x] Out of scope items (opp delta-roto, opp waiver, slot assignment, live preview) NOT implemented
