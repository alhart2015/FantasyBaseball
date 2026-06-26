# Games-based Availability MC -- Phase 0 (Attribution Gate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cheap, reproducible diagnostic that determines whether the in-season MC's over-credit of deep rosters comes from bench-seating (build the games-fill engine) or from per-iteration top-k re-selection churn (just freeze selection) -- and gate the rest of the project on the answer.

**Architecture:** Add a fixed-column override to `simulate_remaining_season_batch` so a caller can replace the per-iteration top-k active-roster pick with a fixed active set. A new `mc_selection.py` module builds those fixed columns (one set from the manager's actual active slots via the existing Player-typed `_classify_roster`, one from a once-on-the-mean top-k), runs three selection "arms," and formats the comparison. An env-gated hook inside the refresh pipeline's `_run_ros_monte_carlo` runs it on the real, correctly-built league snapshot and writes the decision note.

**Tech Stack:** Python, NumPy, pytest. Reuses `fantasy_baseball.simulation`, `fantasy_baseball.scoring._classify_roster`, `fantasy_baseball.models.player`, `fantasy_baseball.web.refresh_pipeline`.

## Global Constraints

- ASCII-only in all source, log, and report strings (Windows cp1252 stdout). No non-ASCII glyphs.
- Player identity keys on `yahoo_id` / object identity, never bare names (collision risk).
- Numeric defaults use `is not None`, never `x or default` (0/0.0 falsy footgun).
- This is the spec's Phase 0 (`docs/superpowers/specs/2026-06-26-games-based-availability-mc-design.md`). It adds NO games-data plumbing, NO dataclass changes, NO fill engine. Only: the batch override + selection helpers + the gated diagnostic hook.
- All imports go at the TOP of each module. `pyproject.toml` selects ruff `E` (incl. E402) and `I` (isort) with no E402/isort exemption for `src/**`; mid-file imports fail lint.
- PASS CRITERION for the gate: bench-exclusion (active-slot arm) accounts for >= 50% of the re-measured SkeleThor RBI gap vs the per-iteration top-k arm, measured under recorded seed/fraction_remaining/iterations. The eyeballed "1020 vs 926" is NOT the baseline -- re-measure it.
- Real cached data only (Upstash/Render source of truth). Drive the real refresh pipeline (`run_full_refresh`), do not fabricate rosters.

---

### Task 1: Fixed-column override on `simulate_remaining_season_batch`

**Files:**
- Modify: `src/fantasy_baseball/simulation.py` (`simulate_remaining_season_batch`, ~714-806)
- Test: `tests/test_simulation.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `simulate_remaining_season_batch(..., active_cols: dict[str, dict[str, np.ndarray]] | None = None)`. When `active_cols` is `None` (default), behavior is unchanged (per-iteration top-k). When `active_cols[team] = {"h": ndarray[int], "p": ndarray[int]}` is present, the team's hitter/pitcher contributions are summed over exactly those fixed column indices (into the team's hitter and pitcher sublists, in roster order) for every iteration, instead of the per-iteration top-k pick. A team absent from `active_cols` falls back to top-k. Empty index arrays yield a zero contribution for that stat group.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_simulation.py
def test_active_cols_override_sums_fixed_columns():
    """With active_cols pinning a subset of hitters, the batch sums exactly those
    every iteration. Statistically equivalent to a roster whose only hitters ARE
    those columns under a top-k that selects all of them -- the two runs use the
    same seed but different roster shapes, so per-iteration draws are NOT aligned;
    the assertion is therefore on medians within a 2.0 tolerance, not exact."""
    import numpy as np
    from fantasy_baseball.simulation import simulate_remaining_season_batch

    rosters, actuals = _build_batch_equiv_scenario()  # "Team A": 5 hitters, 4 pitchers
    team = "Team A"
    frac, h_slots, p_slots, n = 0.45, 3, 2, 2000

    # Build per-team col sets: pin Team A hitters {0,2} (drop hitter 1) and ALL of
    # its pitchers (so the pitcher arm matches the reference's all-pitchers run).
    def _counts(t):
        nh = sum(1 for p in rosters[t] if p["player_type"] == "hitter")
        npc = sum(1 for p in rosters[t] if p["player_type"] == "pitcher")
        return nh, npc

    nh_a, np_a = _counts(team)
    active = {team: {"h": np.array([0, 2]), "p": np.arange(np_a)}}
    for t in rosters:
        if t != team:
            nh, npc = _counts(t)
            active[t] = {"h": np.arange(nh), "p": np.arange(npc)}

    rng = np.random.default_rng(7)
    pinned = simulate_remaining_season_batch(
        actuals, rosters, frac, rng, h_slots, p_slots, n_iter=n, active_cols=active
    )

    # Reference: a roster whose only Team A hitters are cols 0 and 2 (all pitchers
    # kept), with h_slots/p_slots large enough that top-k selects everyone.
    ref_rosters = dict(rosters)
    hitters = [p for p in rosters[team] if p["player_type"] == "hitter"]
    pitchers = [p for p in rosters[team] if p["player_type"] == "pitcher"]
    ref_rosters[team] = [hitters[0], hitters[2], *pitchers]
    rng2 = np.random.default_rng(7)
    ref = simulate_remaining_season_batch(
        actuals, ref_rosters, frac, rng2, 99, 99, n_iter=n
    )

    for c in ("R", "HR", "RBI", "SB"):
        assert abs(np.median(pinned[team][c]) - np.median(ref[team][c])) <= 2.0, c
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation.py::test_active_cols_override_sums_fixed_columns -v`
Expected: FAIL -- `simulate_remaining_season_batch() got an unexpected keyword argument 'active_cols'`.

- [ ] **Step 3: Add the parameter and override the selection**

In `simulate_remaining_season_batch`, add `active_cols: dict | None = None` to the signature (after `n_iter`). Replace the hitter and pitcher selection blocks (currently `simulation.py:749-774`) so a provided fixed-column set bypasses top-k:

```python
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

Notes for the implementer:
- The line just above this block (`hb = ...`, `pb = ...`) and everything below it (the YTD blend and `out[team]` assignment) are UNCHANGED. Read the current 749-774 block first and confirm the variable names (`hb`, `pb`, `zeros`, `sim_*`, `_gather_sum`, `_topk_indices`, `pkey`, `_CLOSER_RANK_BONUS`, `CLOSER_SV_THRESHOLD`) match what you are replacing.
- `cols` must be int dtype (the helpers in Task 2 build `dtype=int`); `np.broadcast_to(cols, (n_iter, cols.shape[0]))` then yields a valid index for `np.take_along_axis`. An empty `cols` (shape `(n_iter, 0)`) sums to zeros -- verified safe, no error.

Update the docstring to document `active_cols`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_simulation.py::test_active_cols_override_sums_fixed_columns -v`
Expected: PASS.

- [ ] **Step 5: Confirm no regression on the default path**

Run: `pytest tests/test_simulation.py -q`
Expected: PASS (default `active_cols=None` path is byte-unchanged).

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/simulation.py tests/test_simulation.py
git commit -m "feat(sim): add active_cols override to simulate_remaining_season_batch

Lets a caller pin a fixed active set (bypassing per-iteration top-k) for the
Phase 0 selection-attribution diagnostic. Default None preserves behavior."
```

---

### Task 2: Selection module -- helpers, orchestrator, and report formatter

**Files:**
- Create: `src/fantasy_baseball/mc_selection.py`
- Test: `tests/test_mc_selection.py`

This is one task (not three) so that every import lands at the module top in a single commit -- avoiding E402/I001/F811 from appending imports mid-file.

**Interfaces:**
- Consumes: `simulation.simulate_remaining_season_batch(active_cols=...)`, `simulation._flatten_full_season`, `simulation._CLOSER_RANK_BONUS`, `scoring._classify_roster(list[Player]) -> (active, il, bench)`, `utils.constants.CLOSER_SV_THRESHOLD`, `utils.constants.ALL_CATEGORIES`.
- Produces:
  - `compute_active_slot_cols(players: list) -> dict[str, np.ndarray]`
  - `compute_fixed_topk_cols(flat_players: list[dict], h_slots: int, p_slots: int) -> dict[str, np.ndarray]`
  - `run_selection_attribution(team_rosters, actual_standings, fraction_remaining, h_slots, p_slots, n_iter, seed) -> dict[str, dict[str, dict[str, float]]]` returning `{arm: {team: {category: median}}}` for arms `"topk_per_iter"`, `"topk_fixed"`, `"active_slot"`. `team_rosters` is `dict[str, list[Player]]` (Player objects, pre-flatten).
  - `format_attribution_table(res: dict, teams: list[str] | None = None) -> str` -- ASCII table of the three arms per team/category.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_mc_selection.py
import numpy as np
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.positions import Position
from fantasy_baseball.mc_selection import (
    compute_active_slot_cols,
    compute_fixed_topk_cols,
    run_selection_attribution,
    format_attribution_table,
)


def _hitter(name, slot, r=80):
    return Player(
        name=name, player_type=PlayerType.HITTER, positions=[Position.OF],
        selected_position=slot,
        rest_of_season=HitterStats(r=r, hr=20, rbi=70, sb=5, h=150, ab=550),
    )


def _pitcher(name, slot, k=150):
    return Player(
        name=name, player_type=PlayerType.PITCHER, positions=[Position.P],
        selected_position=slot,
        rest_of_season=PitcherStats(w=10, k=k, ip=180, er=70, bb=50, h_allowed=150),
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
    assert sorted(cols["h"].tolist()) == [0, 2]
    assert cols["p"].tolist() == [0]


def test_run_selection_attribution_three_arms_and_ordering():
    """active_slot seats only the 2 active OF, so its R median is <= the
    per-iteration top-k arm, which can seat the benched masher over a starter."""
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
    assert res["active_slot"]["Deep"]["R"] <= res["topk_per_iter"]["Deep"]["R"]
    table = format_attribution_table(res)
    assert "Deep" in table and "active_slot" in table
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mc_selection.py -v`
Expected: FAIL -- `ModuleNotFoundError: No module named 'fantasy_baseball.mc_selection'`.

- [ ] **Step 3: Implement the module (all imports at top)**

```python
# src/fantasy_baseball/mc_selection.py
"""Active-set selection helpers for the Phase 0 selection-attribution diagnostic.

Produce the fixed column indices consumed by
``simulation.simulate_remaining_season_batch(active_cols=...)``, run the MC under
three selection arms (per-iteration top-k / fixed top-k / active-slot), and
format the comparison. This is diagnostic-only: NO games plumbing, NO fill engine.
"""

from __future__ import annotations

import numpy as np

from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.scoring import _classify_roster
from fantasy_baseball.simulation import (
    _CLOSER_RANK_BONUS,
    _flatten_full_season,
    simulate_remaining_season_batch,
)
from fantasy_baseball.utils.constants import ALL_CATEGORIES, CLOSER_SV_THRESHOLD

_CATS = [c.value for c in ALL_CATEGORIES]


def _is_hitter(p) -> bool:
    ptype = p.player_type if isinstance(p, Player) else p.get("player_type")
    return ptype == PlayerType.HITTER


def compute_active_slot_cols(players: list) -> dict[str, np.ndarray]:
    """Columns of the active-slot players (healthy bench AND IL excluded).

    Uses the canonical, Player-typed ``_classify_roster`` -- NOT a flat-dict
    reimplementation -- so slot/IL semantics cannot drift. Identity is by object,
    so same-name players never collide.
    """
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
    pitchers by ``closer-bonus + w+k+sv`` -- evaluated on the projected means.
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


def run_selection_attribution(
    team_rosters: dict,
    actual_standings: dict,
    fraction_remaining: float,
    h_slots: int,
    p_slots: int,
    n_iter: int,
    seed: int,
) -> dict[str, dict[str, dict[str, float]]]:
    """Run the MC under three selection arms; return per-team category medians.

    Arms: ``topk_per_iter`` (today), ``topk_fixed`` (top-k fixed once on the mean
    -> isolates re-selection churn), ``active_slot`` (manager's active slots,
    bench+IL excluded -> isolates bench seating on top of churn). All arms share
    one seed, so they differ only in which columns are summed.
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


def format_attribution_table(res: dict, teams: list[str] | None = None) -> str:
    """ASCII table: per team, per category, the three arms' median totals."""
    arms = ["topk_per_iter", "topk_fixed", "active_slot"]
    if teams is None:
        teams = sorted(next(iter(res.values())).keys())
    lines: list[str] = []
    for team in teams:
        lines.append(f"== {team} ==")
        lines.append(f"{'cat':<6}" + "".join(f"{a:>16}" for a in arms))
        for c in _CATS:
            row = "".join(f"{res[a][team][c]:>16.2f}" for a in arms)
            lines.append(f"{c:<6}{row}")
        lines.append("")
    return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mc_selection.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Lint + format the new module**

Run: `ruff check src/fantasy_baseball/mc_selection.py tests/test_mc_selection.py && ruff format --check src/fantasy_baseball/mc_selection.py tests/test_mc_selection.py`
Expected: all green (all imports are top-of-file, so no E402/I001).

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/mc_selection.py tests/test_mc_selection.py
git commit -m "feat(mc): selection-attribution helpers, orchestrator, report (Phase 0)"
```

---

### Task 3: Run on real cached data via a gated pipeline hook, and write the gate decision

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py` (`_run_ros_monte_carlo`, ~1350-1400)
- Create: `docs/superpowers/games-mc-phase0-attribution-2026-06-26.md` (decision note + captured numbers)

**Why a gated hook (not a standalone driver):** the inputs `run_selection_attribution` needs -- `rest_of_season_mc_rosters` (dict team -> list[Player]), `actual_standings_dict`, `self.fraction_remaining`, `h_slots`, `p_slots` -- are all built INSIDE `_run_ros_monte_carlo` (refresh_pipeline.py ~1359-1388). `self.fraction_remaining` and the AB/IP-enriched `actual_standings_dict` are NOT available after `_hydrate_rosters` alone (fraction_remaining is set later, in `_build_projected_standings`), so a partial-pipeline driver would pass `None`. Reusing this hook gives correct inputs for free.

**Interfaces:**
- Consumes: `mc_selection.run_selection_attribution`, `mc_selection.format_attribution_table`.

This is a drive-the-pipeline-and-observe step, not a unit test. It produces the gate evidence.

- [ ] **Step 1: Add the gated hook.** In `refresh_pipeline._run_ros_monte_carlo`, AFTER `actual_standings_dict` is fully built (after the `for e in self.standings.entries:` loop, ~line 1388) and `h_slots`/`p_slots` are computed, insert:

```python
            import os

            if os.environ.get("FB_SELECTION_ATTRIBUTION"):
                from fantasy_baseball.mc_selection import (
                    format_attribution_table,
                    run_selection_attribution,
                )

                assert self.fraction_remaining is not None
                attr = run_selection_attribution(
                    rest_of_season_mc_rosters,
                    actual_standings_dict,
                    self.fraction_remaining,
                    h_slots,
                    p_slots,
                    n_iter=1000,
                    seed=42,
                )
                header = (
                    f"seed=42 n_iter=1000 fraction_remaining={self.fraction_remaining:.4f}\n"
                )
                self._progress("Selection-attribution diagnostic written")
                with open("phase0_attribution.txt", "w", encoding="ascii", errors="replace") as fh:
                    fh.write(header)
                    fh.write(format_attribution_table(attr))
```

(The `import os` is local to keep the diagnostic self-contained and removable; this is a temporary gated hook to be deleted if the gate says STOP, or folded into the real before/after artifact if GO.)

- [ ] **Step 2: Run a local full refresh with the diagnostic enabled.** `run_full_refresh()` is driven by `scripts/refresh_remote.py` (refresh only) and `scripts/run_lineup.py` (refresh + lineup); both run the real pipeline locally and need Yahoo auth. Run, from the repo root, with the env var set (this is interactive/auth-bound -- if you cannot complete OAuth, ask the user to run it via the session `!` prefix):

Run: `FB_SELECTION_ATTRIBUTION=1 python scripts/refresh_remote.py`
Expected: the refresh completes and writes `phase0_attribution.txt` with the header + the three-arm table for every team and all ten categories.

- [ ] **Step 3: Compute the gate.** From `phase0_attribution.txt`, read SkeleThor RBI under the three arms. Compute:
  - `gap = topk_per_iter - active_slot` (the re-measured gap; replaces the eyeballed 94)
  - `churn_share = (topk_per_iter - topk_fixed) / gap`
  - `seating_share = (topk_fixed - active_slot) / gap`
Record all three SkeleThor RBI numbers and (from the same table) the Hart RBI value under each arm.

- [ ] **Step 4: Write the decision note** at `docs/superpowers/games-mc-phase0-attribution-2026-06-26.md`: run conditions; the full three-arm table; SkeleThor RBI `gap`/`churn_share`/`seating_share`; whether the Hart 1st->3rd re-rank reproduces and which arm resolves it; and the VERDICT:
  - `seating_share >= 0.50` -> **GO**: bench seating is a real driver; proceed to plan Phases 1-6 (the games-fill engine). Note the active_slot arm *over*-corrects (removes legitimate injury-insurance) -- exactly what the fill engine adds back.
  - `seating_share < 0.50` -> **STOP**: churn dominates; the cheap fix is to freeze selection (ship `topk_fixed` or a slot-aware equivalent) instead of building the engine.

- [ ] **Step 5: Commit the evidence.**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py docs/superpowers/games-mc-phase0-attribution-2026-06-26.md
git commit -m "feat(mc): Phase 0 selection-attribution hook + gate decision note"
```

- [ ] **Step 6: Surface the verdict to the user.** Report the table, the shares, and the GO/STOP verdict. Do NOT auto-proceed to Phases 1-6 -- the gate decision is the user's to confirm, because STOP means the games-fill engine is unnecessary.

---

## Self-Review

**Spec coverage (Phase 0 only):** The spec's Phase 0 gate is fully covered -- reuse existing `_classify_roster` (Task 2), sum-vs-top-k toggle (Task 1), three arms to separate seating from churn (Task 2), a real-cached-data run with recorded conditions and the >=50% criterion (Task 3). NO games plumbing / dataclass changes / fill engine, per the spec's Phase 0 scope. Phases 1-6 are intentionally out of scope until the gate returns GO.

**Refinement vs spec:** The spec's Phase 0 named a two-arm comparison; this plan uses THREE arms (adds `topk_fixed`) because two arms cannot separate bench-seating from re-selection churn -- the whole point of the gate (spec names churn as the competing explanation). Strictly stronger, within intent; the spec's Phase 0 wording will be reconciled when Phases 1-6 are planned.

**Placeholder scan:** No TBD/TODO/"handle edge cases". Task 3 is an observe-and-decide step with concrete inputs, the exact hook location, a concrete run command, the gate arithmetic, and a binary verdict rule.

**Type consistency:** `active_cols[team] = {"h": ndarray, "p": ndarray}` (int dtype) is produced by `compute_active_slot_cols`/`compute_fixed_topk_cols`, consumed by `simulate_remaining_season_batch` (Task 1) and `run_selection_attribution` (Task 2). `run_selection_attribution` returns `{arm: {team: {cat: float}}}`, consumed by `format_attribution_table` and Task 3's gate math. Imports verified: `ALL_CATEGORIES` and `CLOSER_SV_THRESHOLD` live in `utils.constants`; `_CLOSER_RANK_BONUS` and `_flatten_full_season` in `simulation`; the pipeline class is `RefreshRun` driven by `run_full_refresh`.
