# Games-based Availability MC -- Phase 0 (Attribution Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cheap, reproducible diagnostic that determines whether the in-season MC's over-credit of deep rosters comes from bench-seating (build the games-fill engine) or from per-iteration top-k re-selection churn (just freeze selection) -- and gate the rest of the project on the answer.

**Architecture:** Add a fixed-column override to `simulate_remaining_season_batch` so a caller can replace the per-iteration top-k active-roster pick with a fixed active set. Build two helpers that compute those fixed columns -- one from the manager's actual active slots (via the existing Player-typed `_classify_roster`), one from a once-on-the-mean top-k. An orchestrator runs three selection "arms" (per-iteration top-k = today; fixed top-k = churn removed; active-slot = churn AND bench removed) and reports per-team category medians. A driver runs it on a real cached league snapshot and writes the decision note.

**Tech Stack:** Python, NumPy, pytest. Reuses `fantasy_baseball.simulation`, `fantasy_baseball.scoring._classify_roster`, `fantasy_baseball.models.player`.

## Global Constraints

- ASCII-only in all source, log, and report strings (Windows cp1252 stdout). No non-ASCII glyphs.
- Player identity keys on `yahoo_id` / object identity, never bare names (collision risk).
- Numeric defaults use `is not None`, never `x or default` (0/0.0 falsy footgun).
- This is the spec's Phase 0 (`docs/superpowers/specs/2026-06-26-games-based-availability-mc-design.md`). It adds NO games-data plumbing, NO dataclass changes, NO fill engine. Only: the batch override + selection helpers + diagnostic.
- PASS CRITERION for the gate: bench-exclusion (active-slot arm) alone closes >= 50% of the re-measured SkeleThor RBI gap vs the per-iteration top-k arm, measured under recorded seed/fraction_remaining/iterations. The eyeballed "1020 vs 926" is NOT the baseline -- re-measure it.
- Real cached data only (Upstash/Render source of truth; never stale local cache). Use `--no-sync` so the local run does not get clobbered by an Upstash sync.

---

### Task 1: Fixed-column override on `simulate_remaining_season_batch`

**Files:**
- Modify: `src/fantasy_baseball/simulation.py` (`simulate_remaining_season_batch`, ~714-806)
- Test: `tests/test_simulation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `simulate_remaining_season_batch(..., active_cols: dict[str, dict[str, np.ndarray]] | None = None)`. When `active_cols` is `None` (default), behavior is unchanged (per-iteration top-k). When `active_cols[team] = {"h": ndarray[int], "p": ndarray[int]}` is present, the team's hitter/pitcher contributions are summed over exactly those fixed column indices (into the team's hitter and pitcher sublists, in roster order) for every iteration, instead of the per-iteration top-k pick.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulation.py
def test_active_cols_override_sums_fixed_columns():
    """With active_cols pinning a subset, the batch sums exactly those players
    every iteration -- equivalent to making them the whole roster under top-k."""
    import numpy as np
    from fantasy_baseball.simulation import simulate_remaining_season_batch

    rosters, actuals = _build_batch_equiv_scenario()
    team = "Team A"  # 3 hitters, 2 pitchers in this fixture
    frac, h_slots, p_slots, n = 0.45, 3, 2, 2000

    # Pin hitter cols {0, 2} (drop hitter index 1) and both pitchers.
    active = {team: {"h": np.array([0, 2]), "p": np.array([0, 1])}}
    for t in rosters:
        if t != team:
            n_h = sum(1 for p in rosters[t] if p["player_type"] == "hitter")
            n_p = sum(1 for p in rosters[t] if p["player_type"] == "pitcher")
            active[t] = {"h": np.arange(n_h), "p": np.arange(n_p)}

    rng = np.random.default_rng(7)
    pinned = simulate_remaining_season_batch(
        actuals, rosters, frac, rng, h_slots, p_slots, n_iter=n, active_cols=active
    )

    # Reference: same RNG, but a roster whose only hitters ARE cols 0 and 2,
    # with h_slots large enough that top-k selects all of them (no churn).
    ref_rosters = dict(rosters)
    hitters = [p for p in rosters[team] if p["player_type"] == "hitter"]
    pitchers = [p for p in rosters[team] if p["player_type"] == "pitcher"]
    ref_rosters[team] = [hitters[0], hitters[2], *pitchers]
    rng2 = np.random.default_rng(7)
    ref = simulate_remaining_season_batch(
        actuals, ref_rosters, frac, rng2, 99, 99, n_iter=n
    )

    # Counting medians for the pinned team must match the reference within noise.
    for c in ("R", "HR", "RBI", "SB"):
        assert abs(np.median(pinned[team][c]) - np.median(ref[team][c])) <= 2.0, c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation.py::test_active_cols_override_sums_fixed_columns -v`
Expected: FAIL -- `simulate_remaining_season_batch() got an unexpected keyword argument 'active_cols'`.

- [ ] **Step 3: Add the parameter and override the selection**

In `simulate_remaining_season_batch`, add `active_cols: dict | None = None` to the signature (after `n_iter`). Replace the hitter and pitcher selection blocks so a provided fixed-column set bypasses top-k:

```python
    for team, players in team_rosters.items():
        actuals = actual_standings.get(team, {})
        hitters = [p for p in players if p.get("player_type") == PlayerType.HITTER]
        pitchers = [p for p in players if p.get("player_type") == PlayerType.PITCHER]

        hb = _apply_variance_batch(hitters, PlayerType.HITTER, rng, fraction_remaining, n_iter)
        pb = _apply_variance_batch(pitchers, PlayerType.PITCHER, rng, fraction_remaining, n_iter)

        team_cols = active_cols.get(team) if active_cols is not None else None

        if hitters:
            if team_cols is not None:
                cols = team_cols["h"]
                h_idx = np.broadcast_to(cols, (n_iter, cols.shape[0]))
            else:
                h_idx = _topk_indices(hb["r"] + hb["hr"] + hb["rbi"] + hb["sb"], h_slots)
            sim_r = _gather_sum(hb["r"], h_idx)
            sim_hr = _gather_sum(hb["hr"], h_idx)
            sim_rbi = _gather_sum(hb["rbi"], h_idx)
            sim_sb = _gather_sum(hb["sb"], h_idx)
            sim_h = _gather_sum(hb["h"], h_idx)
            sim_ab = _gather_sum(hb["ab"], h_idx)
        else:
            sim_r = sim_hr = sim_rbi = sim_sb = sim_h = sim_ab = zeros

        if pitchers:
            if team_cols is not None:
                cols = team_cols["p"]
                p_idx = np.broadcast_to(cols, (n_iter, cols.shape[0]))
            else:
                pkey = (pb["sv"] >= CLOSER_SV_THRESHOLD).astype(float) * _CLOSER_RANK_BONUS + (
                    pb["w"] + pb["k"] + pb["sv"]
                )
                p_idx = _topk_indices(pkey, p_slots)
            sim_w = _gather_sum(pb["w"], p_idx)
            sim_k = _gather_sum(pb["k"], p_idx)
            sim_sv = _gather_sum(pb["sv"], p_idx)
            sim_ip = _gather_sum(pb["ip"], p_idx)
            sim_er = _gather_sum(pb["er"], p_idx)
            sim_bb = _gather_sum(pb["bb"], p_idx)
            sim_ha = _gather_sum(pb["h_allowed"], p_idx)
        else:
            sim_w = sim_k = sim_sv = sim_ip = sim_er = sim_bb = sim_ha = zeros
```

(The rest of the per-team body -- the YTD blend and the `out[team]` assignment -- is unchanged.)

Update the docstring to document `active_cols` and note that an empty column array yields a zero contribution for that stat group.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation.py::test_active_cols_override_sums_fixed_columns -v`
Expected: PASS.

- [ ] **Step 5: Run the existing batch test to confirm no regression**

Run: `pytest tests/test_simulation.py -k batch -v`
Expected: PASS (default `active_cols=None` path is byte-unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/simulation.py tests/test_simulation.py
git commit -m "feat(sim): add active_cols override to simulate_remaining_season_batch

Lets a caller pin a fixed active set (bypassing per-iteration top-k) for the
Phase 0 selection-attribution diagnostic. Default None preserves behavior."
```

---

### Task 2: Active-set column helpers

**Files:**
- Create: `src/fantasy_baseball/mc_selection.py`
- Test: `tests/test_mc_selection.py`

**Interfaces:**
- Consumes: `simulate_remaining_season_batch`'s expectation that `active_cols[team]["h"]/["p"]` are int ndarrays indexing the hitter/pitcher sublists in roster order; `scoring._classify_roster(list[Player]) -> (active, il, bench)`.
- Produces:
  - `compute_active_slot_cols(players: list) -> dict[str, np.ndarray]` -- columns of the active-slot players (healthy bench AND IL excluded), via `_classify_roster`, keyed by object identity. Operates on Player objects.
  - `compute_fixed_topk_cols(flat_players: list[dict], h_slots: int, p_slots: int) -> dict[str, np.ndarray]` -- columns of the top-`h_slots` hitters by mean `r+hr+rbi+sb` and top-`p_slots` pitchers by `closer-bonus + w+k+sv`, computed once on the projected means (no per-iteration churn). Operates on the flattened dicts so its stat basis matches the batch's top-k.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mc_selection.py
import numpy as np
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.mc_selection import compute_active_slot_cols, compute_fixed_topk_cols


def _hitter(name, slot, r=80):
    return Player(
        name=name, player_type=PlayerType.HITTER, positions=[Position.OF],
        selected_position=slot, rest_of_season=HitterStats(r=r, hr=20, rbi=70, sb=5, h=150, ab=550),
    )


def _pitcher(name, slot, k=150):
    return Player(
        name=name, player_type=PlayerType.PITCHER, positions=[Position.P],
        selected_position=slot, rest_of_season=PitcherStats(w=10, k=k, ip=180, er=70, bb=50, h_allowed=150),
    )


def test_active_slot_cols_excludes_healthy_bench_and_il():
    players = [
        _hitter("H_active", Position.OF),     # hitter col 0 -> active
        _hitter("H_bench", Position.BN),      # hitter col 1 -> excluded (bench)
        _pitcher("P_active", Position.P),     # pitcher col 0 -> active
        _pitcher("P_il", Position.IL),        # pitcher col 1 -> excluded (IL)
    ]
    cols = compute_active_slot_cols(players)
    assert cols["h"].tolist() == [0]
    assert cols["p"].tolist() == [0]


def test_fixed_topk_cols_picks_highest_mean_stats():
    flat = [
        {"player_type": "hitter", "r": 100, "hr": 30, "rbi": 100, "sb": 10},  # col 0 best
        {"player_type": "hitter", "r": 50, "hr": 10, "rbi": 40, "sb": 2},     # col 1 worst
        {"player_type": "hitter", "r": 80, "hr": 25, "rbi": 80, "sb": 8},     # col 2 mid
        {"player_type": "pitcher", "w": 12, "k": 200, "sv": 0, "ip": 190},    # col 0
        {"player_type": "pitcher", "w": 5, "k": 90, "sv": 0, "ip": 70},       # col 1
    ]
    cols = compute_fixed_topk_cols(flat, h_slots=2, p_slots=1)
    assert sorted(cols["h"].tolist()) == [0, 2]   # top 2 hitters
    assert cols["p"].tolist() == [0]              # top 1 pitcher
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mc_selection.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'fantasy_baseball.mc_selection'`.

- [ ] **Step 3: Implement the helpers**

```python
# src/fantasy_baseball/mc_selection.py
"""Active-set column helpers for the Phase 0 selection-attribution diagnostic.

Produce the fixed column indices consumed by
``simulation.simulate_remaining_season_batch(active_cols=...)``: which hitter /
pitcher sublist positions form the active set under a given selection rule.
"""

from __future__ import annotations

import numpy as np

from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.utils.constants import CLOSER_SV_THRESHOLD
from fantasy_baseball.simulation import _CLOSER_RANK_BONUS


def _is_hitter(p) -> bool:
    ptype = p.player_type if isinstance(p, Player) else p.get("player_type")
    return ptype == PlayerType.HITTER


def compute_active_slot_cols(players: list) -> dict[str, np.ndarray]:
    """Columns of the active-slot players (healthy bench AND IL excluded).

    Classification is the canonical, Player-typed ``_classify_roster`` -- NOT a
    flat-dict reimplementation -- so slot/IL semantics cannot drift. Identity is
    by object, so same-name players never collide.
    """
    from fantasy_baseball.scoring import _classify_roster

    active, _il, _bench = _classify_roster([p for p in players if isinstance(p, Player)])
    active_ids = {id(p) for p in active}

    h_cols: list[int] = []
    p_cols: list[int] = []
    hi = pi = 0
    for p in players:
        if _is_hitter(p):
            if id(p) in active_ids:
                h_cols.append(hi)
            hi += 1
        else:
            if id(p) in active_ids:
                p_cols.append(pi)
            pi += 1
    return {"h": np.array(h_cols, dtype=int), "p": np.array(p_cols, dtype=int)}


def compute_fixed_topk_cols(
    flat_players: list[dict], h_slots: int, p_slots: int
) -> dict[str, np.ndarray]:
    """Top-k columns by projected mean stats, fixed once (no per-iteration churn).

    Mirrors the batch's per-iteration keys -- hitters by ``r+hr+rbi+sb``,
    pitchers by ``closer-bonus + w+k+sv`` -- but evaluated on the projected
    means so the selection is identical across iterations.
    """
    h_keys: list[tuple[float, int]] = []
    p_keys: list[tuple[float, int]] = []
    hi = pi = 0
    for p in flat_players:
        if _is_hitter(p):
            key = p.get("r", 0) + p.get("hr", 0) + p.get("rbi", 0) + p.get("sb", 0)
            h_keys.append((key, hi))
            hi += 1
        else:
            sv = p.get("sv", 0)
            bonus = _CLOSER_RANK_BONUS if sv >= CLOSER_SV_THRESHOLD else 0.0
            key = bonus + p.get("w", 0) + p.get("k", 0) + sv
            p_keys.append((key, pi))
            pi += 1

    def _top(keys: list[tuple[float, int]], k: int) -> np.ndarray:
        chosen = [idx for _, idx in sorted(keys, reverse=True)[:k]]
        return np.array(sorted(chosen), dtype=int)

    return {"h": _top(h_keys, h_slots), "p": _top(p_keys, p_slots)}
```

(If `_CLOSER_RANK_BONUS` is not importable from `simulation`, read `simulation.py` to confirm its name/location and import accordingly; it is a module-level constant near the top-k pitcher key.)

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mc_selection.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/mc_selection.py tests/test_mc_selection.py
git commit -m "feat(mc): active-slot and fixed-topk column helpers for Phase 0 diagnostic"
```

---

### Task 3: Three-arm attribution orchestrator

**Files:**
- Modify: `src/fantasy_baseball/mc_selection.py`
- Test: `tests/test_mc_selection.py`

**Interfaces:**
- Consumes: `compute_active_slot_cols`, `compute_fixed_topk_cols`, `simulate_remaining_season_batch`, `simulation._flatten_full_season`.
- Produces: `run_selection_attribution(team_rosters, actual_standings, fraction_remaining, h_slots, p_slots, n_iter, seed) -> dict[str, dict[str, dict[str, float]]]` returning `{arm: {team: {category: median}}}` for the three arms `"topk_per_iter"`, `"topk_fixed"`, `"active_slot"`. `team_rosters` is `dict[str, list[Player]]` (Player objects, pre-flatten).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_mc_selection.py  (add)
from fantasy_baseball.mc_selection import run_selection_attribution


def test_run_selection_attribution_three_arms_and_ordering():
    """active_slot excludes the bench bat, so its counting totals are <=
    topk_per_iter, which (best-ball) is >= topk_fixed for a deep roster."""
    deep = [
        _hitter("Star", Position.OF, r=100),
        _hitter("Reg", Position.OF, r=80),
        _hitter("BenchMasher", Position.BN, r=95),  # benched but high-stat
        _pitcher("Ace", Position.P),
    ]
    rosters = {"Deep": deep}
    actuals = {"Deep": {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                        "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}}
    res = run_selection_attribution(rosters, actuals, 1.0, h_slots=2, p_slots=1,
                                    n_iter=2000, seed=3)
    assert set(res) == {"topk_per_iter", "topk_fixed", "active_slot"}
    # active_slot seats only the 2 active OF (Star, Reg); top-k can seat the
    # bench masher over a starter, so active_slot R median is the lowest.
    assert res["active_slot"]["Deep"]["R"] <= res["topk_per_iter"]["Deep"]["R"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_mc_selection.py::test_run_selection_attribution_three_arms_and_ordering -v`
Expected: FAIL -- `cannot import name 'run_selection_attribution'`.

- [ ] **Step 3: Implement the orchestrator**

```python
# src/fantasy_baseball/mc_selection.py  (add)
import numpy as np

from fantasy_baseball.simulation import (
    _flatten_full_season,
    simulate_remaining_season_batch,
)
from fantasy_baseball.utils.constants import ALL_CATS

_CATS = [c.value for c in ALL_CATS]


def run_selection_attribution(
    team_rosters: dict,
    actual_standings: dict,
    fraction_remaining: float,
    h_slots: int,
    p_slots: int,
    n_iter: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run the MC under three selection arms and return per-team category medians.

    Arms: ``topk_per_iter`` (today's behavior), ``topk_fixed`` (top-k fixed once
    on the mean -> isolates re-selection churn), ``active_slot`` (manager's
    active slots, bench+IL excluded -> isolates bench seating on top of churn).
    """
    flat = {t: [_flatten_full_season(p) for p in players] for t, players in team_rosters.items()}
    active = {t: compute_active_slot_cols(players) for t, players in team_rosters.items()}
    topk = {t: compute_fixed_topk_cols(flat[t], h_slots, p_slots) for t in flat}

    arms: dict[str, dict | None] = {
        "topk_per_iter": None,
        "topk_fixed": topk,
        "active_slot": active,
    }
    out: dict[str, dict[str, dict[str, float]]] = {}
    for arm, cols in arms.items():
        rng = np.random.default_rng(seed)
        batch = simulate_remaining_season_batch(
            actual_standings, flat, fraction_remaining, rng, h_slots, p_slots, n_iter,
            active_cols=cols,
        )
        out[arm] = {t: {c: float(np.median(batch[t][c])) for c in _CATS} for t in flat}
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_mc_selection.py -v`
Expected: PASS.

- [ ] **Step 5: Full local checks**

Run: `pytest tests/test_mc_selection.py tests/test_simulation.py -q && ruff check src/fantasy_baseball/mc_selection.py && ruff format --check src/fantasy_baseball/mc_selection.py`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/mc_selection.py tests/test_mc_selection.py
git commit -m "feat(mc): three-arm selection-attribution orchestrator (Phase 0)"
```

---

### Task 4: Run on real cached data and write the gate decision

**Files:**
- Create: `scripts/compare_mc_active_selection.py`
- Create: `docs/superpowers/games-mc-phase0-attribution-2026-06-26.md` (the decision note + captured numbers)

**Interfaces:**
- Consumes: `run_selection_attribution`; the refresh pipeline's roster hydration (the only place real Player rosters + actual standings + `fraction_remaining` + slot counts exist together).

This task is a drive-the-pipeline-and-observe step, not a unit test. It produces the gate evidence.

- [ ] **Step 1: Identify the data source.** Read `src/fantasy_baseball/web/refresh_pipeline.py` around `_run_ros_monte_carlo` (~1350-1400) and `run()` (~490-520). Confirm the locals available there: `rest_of_season_mc_rosters` (dict team -> list[Player]), `actual_standings_dict`, `self.fraction_remaining`, `h_slots`, `p_slots`. These are exactly `run_selection_attribution`'s inputs.

- [ ] **Step 2: Write the driver script.** `scripts/compare_mc_active_selection.py` injects `src/` into `sys.path` (mirror other scripts), constructs a `RefreshPipeline`, runs the pipeline steps up to and including `_hydrate_rosters` (and whatever earlier steps those depend on -- follow the ordered calls in `run()`), then builds `actual_standings_dict` and slot counts exactly as `_run_ros_monte_carlo` does, and calls:

```python
res = run_selection_attribution(
    rest_of_season_mc_rosters, actual_standings_dict, fraction_remaining,
    h_slots, p_slots, n_iter=1000, seed=42,
)
```

Then print, for every team and all ten categories, a 3-column table (`topk_per_iter`, `topk_fixed`, `active_slot`) plus the ERoto figure if readily available from `self.projected_standings`. Print the run header: `seed=42`, the actual `fraction_remaining`, `n_iter=1000`, and the effective date. ASCII only.

Run it against real cache with sync disabled:

Run: `python scripts/compare_mc_active_selection.py --no-sync`
(If the pipeline constructor/auth differs, follow `scripts/run_season_dashboard.py` for the exact construction + `--no-sync` wiring -- reuse its setup rather than inventing one.)

- [ ] **Step 3: Compute the gate.** From the printed table, take SkeleThor RBI: `gap = topk_per_iter - active_slot` (the re-measured gap, replacing the eyeballed 94). Compute `churn_share = (topk_per_iter - topk_fixed) / gap` and `seating_share = (topk_fixed - active_slot) / gap`. Record all three SkeleThor RBI numbers and the Hart RBI rank under each arm.

- [ ] **Step 4: Write the decision note.** In `docs/superpowers/games-mc-phase0-attribution-2026-06-26.md`, record: run conditions; the full 3-arm table; SkeleThor RBI `gap`, `churn_share`, `seating_share`; whether the Hart 1st->3rd re-rank reproduces and under which arm it resolves; and the VERDICT against the gate:
  - `seating_share >= 0.50` (bench-exclusion closes >= 50% of the gap) -> **GO**: bench seating is a real driver; proceed to plan Phases 1-6 (the games-fill engine). Note that the active_slot arm *over*-corrects (removes legitimate injury-insurance), which is exactly what the fill engine adds back.
  - `seating_share < 0.50` (churn dominates) -> **STOP**: the cheap fix is to freeze selection (ship `topk_fixed`, or its slot-aware equivalent) instead of building the engine. Surface this to the user.

- [ ] **Step 5: Commit the evidence.**

```bash
git add scripts/compare_mc_active_selection.py docs/superpowers/games-mc-phase0-attribution-2026-06-26.md
git commit -m "feat(mc): Phase 0 selection-attribution run + gate decision note"
```

- [ ] **Step 6: Surface the verdict to the user.** Report the table, the shares, and the GO/STOP verdict. Do NOT auto-proceed to Phases 1-6 -- the gate decision is the user's to confirm, because STOP means the whole games-fill engine is unnecessary.

---

## Self-Review

**Spec coverage (Phase 0 only):** The spec's Phase 0 (gate) is fully covered -- reuse existing `_classify_roster` (Task 2), sum-vs-top-k toggle (Task 1), three arms to separate seating from churn (Task 3), real-cached-data run with recorded conditions and the >=50% criterion (Task 4). NO games plumbing / dataclass changes / fill engine appear, per the spec's Phase 0 scope. Phases 1-6 are intentionally out of scope until the gate returns GO.

**Refinement vs spec:** The spec's Phase 0 named a two-arm comparison (top-k vs bench-exclusion); this plan uses THREE arms (adds `topk_fixed`) because two arms cannot separate bench-seating from re-selection churn -- and that separation is the entire point of the gate. This strictly strengthens Phase 0 within its stated intent; the spec's Phase 0 wording will be updated to match when Phases 1-6 are planned.

**Placeholder scan:** No TBD/TODO/"handle edge cases". Task 4 is an observe-and-decide task with concrete inputs, commands, computations, and a binary verdict rule -- appropriate detail for an integration/measurement step.

**Type consistency:** `active_cols[team] = {"h": ndarray, "p": ndarray}` is produced by `compute_active_slot_cols`/`compute_fixed_topk_cols` and consumed by `simulate_remaining_season_batch` (Task 1) and `run_selection_attribution` (Task 3) with identical shape/keys throughout. `run_selection_attribution` returns `{arm: {team: {cat: float}}}`, consumed by Task 4's table/gate.
