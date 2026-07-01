# Draft-Value Metric Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a per-player and per-team "draft value" metric (realized VAR vs draft-slot par expectation, with a skill/luck split) as a library module + CLI report.

**Architecture:** A new `analysis/draft_value.py` library composes existing SGP/VAR/board/ROS/transaction machinery. Preseason VAR + the par curve come from a *reproduced* draft-day board (built off `data/fantasy.db`, validated against the frozen `draft_state_board.json`). Realized/estimate VAR is scored on that exact same scale. A CLI (`scripts/draft_value.py`) joins current rosters + transactions to classify each rostered player's acquisition and rolls value up per team.

**Tech Stack:** Python 3, pandas, pytest. Reuses `fantasy_baseball.draft.board`, `fantasy_baseball.sgp.*`, `fantasy_baseball.data.ros_pipeline`, `fantasy_baseball.lineup.yahoo_roster`, `fantasy_baseball.data.redis_store`.

## Global Constraints

- **Spec:** `docs/superpowers/specs/2026-07-01-draft-value-metric-design.md` — the authoritative design. Read it before starting; every task implements part of it.
- **Value currency is VAR** (SGP minus positional replacement floor), never raw SGP.
- **Single scale:** all VAR (preseason, realized, estimate, par) MUST use the board's exact scale inputs — pool-derived rates, `team_ab=5500`, `team_ip=1450`, `get_sgp_denominators()` code defaults, and `position_aware_replacement_levels` floors. Never call `calculate_player_sgp` with the module rate defaults (0.250/4.50/1.35).
- **Player identity:** key on `name::player_type` with normalized names (`normalize_name`), VAR tie-break on collisions. Never key on bare names.
- **ASCII-only** in all code, strings, and report output (Windows cp1252 stdout). Use ASCII names in synthetic fixtures.
- **No `x or default` for numeric defaults** — VAR/SGP can be 0.0 or negative; use explicit `is not None`, especially in sort keys and par-index lookups.
- **skill/luck** are defined ONLY for on-board players on the projected horizon; render N/A otherwise (never NaN, never a dropped row).
- Run `ruff check .`, `ruff format --check .`, and `pytest -v` (relevant subset acceptable, state which) before declaring any task done.

**Module layout (locked here):**
- `src/fantasy_baseball/analysis/draft_value.py` — all library units.
- `scripts/draft_value.py` — CLI + markdown renderer.
- `tests/test_analysis/test_draft_value.py` — unit tests.
- Modify `src/fantasy_baseball/draft/board.py` — add opt-in `return_scale` to expose scale inputs.

**Shared dataclasses (defined in Task 1, used everywhere):**
```python
@dataclass(frozen=True)
class ScaleInputs:
    denoms: dict            # from get_sgp_denominators()
    repl_rates: dict        # {"era","whip","avg"} pool-derived
    replacement_levels: dict # position -> floor SGP
    team_ab: int
    team_ip: int
```

---

### Task 1: Reproduce the draft-day board scale + validate against frozen board

**Files:**
- Modify: `src/fantasy_baseball/draft/board.py` (add `return_scale` param to `build_draft_board`)
- Create: `src/fantasy_baseball/analysis/draft_value.py`
- Create: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Produces: `ScaleInputs` dataclass; `reproduce_draft_day_board(conn=None) -> tuple[pd.DataFrame, ScaleInputs]`; `validate_scale_against_frozen(board_df, frozen_path=None, tol=0.05) -> list[str]` (returns list of mismatch messages, empty == pass).

- [ ] **Step 1: Add `return_scale` to `build_draft_board` (backward-compatible).**

In `src/fantasy_baseball/draft/board.py`, change the signature to add `return_scale: bool = False`, and at the end return the scale bundle when requested. The scale variables already exist inside the function: `denoms`, `repl_rates`, `replacement_levels`. First read the function to confirm the local variable names (`denoms` from `get_sgp_denominators()`, `repl_rates` from `calculate_replacement_rates`, `replacement_levels` from `position_aware_replacement_levels`), then:

```python
def build_draft_board(
    conn,
    roster_slots: dict[str, int] | None = None,
    num_teams: int | None = None,
    return_scale: bool = False,
):
    ...  # existing body unchanged
    # existing final return is `return board` (the sorted DataFrame). Replace with:
    if return_scale:
        scale = {
            "denoms": denoms,
            "repl_rates": repl_rates,
            "replacement_levels": replacement_levels,
            "team_ab": DEFAULT_TEAM_AB,
            "team_ip": DEFAULT_TEAM_IP,
        }
        return board, scale
    return board
```

Import `DEFAULT_TEAM_AB, DEFAULT_TEAM_IP` from `fantasy_baseball.sgp.player_value` at the top of `board.py` if not already imported.

- [ ] **Step 2: Grep all `build_draft_board(` call sites to confirm none break.**

Run: `rg -n "build_draft_board\(" src scripts tests`
Expected: every existing call uses positional/keyword args without `return_scale`; the new default `False` preserves their `pd.DataFrame` return. If any call unpacks a tuple, fix it. State the call sites found.

- [ ] **Step 3: Write the failing test for scale reproduction + frozen validation.**

```python
# tests/test_analysis/test_draft_value.py
from fantasy_baseball.analysis import draft_value as dv


def test_reproduce_board_returns_scale_and_matches_frozen():
    board, scale = dv.reproduce_draft_day_board()
    # scale bundle is well-formed
    assert set(scale.replacement_levels) >= {"C", "1B", "SS", "OF", "SP", "RP"}
    assert scale.team_ab == 5500 and scale.team_ip == 1450
    assert {"era", "whip", "avg"} <= set(scale.repl_rates)
    # frozen validation passes within 0.05
    mismatches = dv.validate_scale_against_frozen(board)
    assert mismatches == [], f"{len(mismatches)} players drift from frozen var: {mismatches[:5]}"
```

- [ ] **Step 4: Run it to confirm it fails.**

Run: `pytest tests/test_analysis/test_draft_value.py::test_reproduce_board_returns_scale_and_matches_frozen -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 5: Implement `reproduce_draft_day_board` + `validate_scale_against_frozen`.**

```python
# src/fantasy_baseball/analysis/draft_value.py
"""Draft-value metric: realized VAR vs draft-slot par expectation."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fantasy_baseball.data.db import get_connection
from fantasy_baseball.draft.board import build_draft_board
from fantasy_baseball.utils.name_utils import normalize_name

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FROZEN_BOARD = _REPO_ROOT / "data" / "draft_state_board.json"


@dataclass(frozen=True)
class ScaleInputs:
    denoms: dict
    repl_rates: dict
    replacement_levels: dict
    team_ab: int
    team_ip: int


def reproduce_draft_day_board(conn=None) -> tuple[pd.DataFrame, ScaleInputs]:
    """Rebuild the draft-day board off fantasy.db and return it with its scale.

    NOT the KV store: build_draft_board reads fantasy.db (blended_projections +
    positions). Reproduction must be validated with validate_scale_against_frozen.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        board, scale_d = build_draft_board(conn, return_scale=True)
    finally:
        if own_conn:
            conn.close()
    scale = ScaleInputs(
        denoms=scale_d["denoms"],
        repl_rates=scale_d["repl_rates"],
        replacement_levels=scale_d["replacement_levels"],
        team_ab=scale_d["team_ab"],
        team_ip=scale_d["team_ip"],
    )
    return board, scale


def validate_scale_against_frozen(board_df, frozen_path=None, tol=0.05) -> list[str]:
    """Assert reproduced var reproduces frozen draft_state_board.json var.

    Frozen var is stored rounded to 1 dp, so tol=0.05. Returns mismatch messages;
    empty list means PASS. A non-empty result is a LOUD STOP (drift investigation),
    never a reason to loosen tol.
    """
    frozen_path = Path(frozen_path) if frozen_path else _FROZEN_BOARD
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_var = {}
    for row in frozen:
        pid = row.get("player_id")
        if pid is not None and row.get("var") is not None:
            frozen_var[pid] = float(row["var"])
    mismatches: list[str] = []
    for _, row in board_df.iterrows():
        pid = row["player_id"]
        if pid not in frozen_var:
            continue
        repro = float(row["var"])
        if abs(repro - frozen_var[pid]) > tol:
            mismatches.append(f"{row['name']} ({pid}): repro={repro:.3f} frozen={frozen_var[pid]:.3f}")
    return mismatches
```

- [ ] **Step 6: Run the test; if frozen validation fails, STOP and investigate drift.**

Run: `pytest tests/test_analysis/test_draft_value.py::test_reproduce_board_returns_scale_and_matches_frozen -v`
Expected: PASS. If `validate_scale_against_frozen` returns mismatches, the board/SGP code has drifted since the freeze (or `fantasy.db` moved off the draft-day vintage). Do NOT raise `tol`. Investigate: confirm `fantasy.db` `blended_projections` is the draft-day vintage (reseed from `data/projections/2026/` if needed), or pin reproduction to the draft-day commit. Surface findings before proceeding.

- [ ] **Step 7: Commit.**

```bash
git add src/fantasy_baseball/draft/board.py src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): reproduce draft-day board scale + frozen validation"
```

---

### Task 2: Score VAR on the board scale (with the estimate-scale oracle)

**Files:**
- Modify: `src/fantasy_baseball/analysis/draft_value.py`
- Modify: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Consumes: `ScaleInputs`.
- Produces: `score_var(line: dict, positions: list[str], player_type: str, scale: ScaleInputs, fraction: float = 1.0) -> float`. `fraction < 1.0` applies the YTD to-date scaling (counting + volume + team volumes * fraction; rates held; to-date floors). `player_type` is `"hitter"` or `"pitcher"`.

- [ ] **Step 1: Write the failing tests (full-season score + estimate-scale reproduction oracle).**

```python
def _hitter_line(**kw):
    base = {"r": 90, "hr": 30, "rbi": 95, "sb": 12, "avg": 0.280, "ab": 560}
    base.update(kw)
    return base


def test_score_var_reproduces_board_var_for_onboard_player():
    board, scale = dv.reproduce_draft_day_board()
    row = board[board["player_type"] == "hitter"].iloc[0]
    line = {k: row[k] for k in ("r", "hr", "rbi", "sb", "avg", "ab")}
    var = dv.score_var(line, list(row["positions"]), "hitter", scale)
    assert abs(var - float(row["var"])) < 1e-6  # same scale -> same VAR


def test_score_var_fraction_half_scales_counting_not_rate():
    _, scale = dv.reproduce_draft_day_board()
    full = dv.score_var(_hitter_line(), ["OF"], "hitter", scale, fraction=1.0)
    half = dv.score_var(_hitter_line(), ["OF"], "hitter", scale, fraction=0.5)
    # counting SGP halves, rate SGP is fraction-invariant, floor also to-date -> VAR roughly halves but not below full
    assert half < full
```

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k score_var -v`
Expected: FAIL (`score_var` not defined).

- [ ] **Step 3: Implement `score_var` + to-date helpers.**

Add to `draft_value.py`:

```python
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.sgp.var import calculate_var
from fantasy_baseball.utils.constants import REPLACEMENT_BY_POSITION

_COUNTING_HIT = ("r", "hr", "rbi", "sb", "ab")   # ab is volume (scales)
_COUNTING_PIT = ("w", "k", "sv", "ip")           # ip is volume (scales)


def _to_date_floors(scale: ScaleInputs, fraction: float) -> dict:
    """Recompute position floors on a to-date scale (NOT scale.replacement_levels * f).

    Floor SGP is NOT linear in f: its rate component is f-invariant while only the
    counting component scales. So rebuild each floor from REPLACEMENT_BY_POSITION with
    counting+volume * f, rates held, team volumes * f. At f=1 this reproduces the
    board's position_aware_replacement_levels floors.
    """
    if fraction == 1.0:
        return scale.replacement_levels
    floors = {}
    team_ab = scale.team_ab * fraction
    team_ip = scale.team_ip * fraction
    for pos, raw in REPLACEMENT_BY_POSITION.items():
        line = dict(raw)
        for k in ("r", "hr", "rbi", "sb", "ab", "w", "k", "sv", "ip"):
            if k in line:
                line[k] = line[k] * fraction
        is_pitcher = "ip" in raw
        # derive rate stats the floor line implies, held constant vs f
        if is_pitcher:
            ip = raw["ip"] or 1.0
            line["era"] = (raw["er"] / ip) * 9.0
            line["whip"] = (raw["bb"] + raw["h_allowed"]) / ip
        else:
            ab = raw["ab"] or 1.0
            line["avg"] = raw["h"] / ab
        floors[pos] = calculate_player_sgp(
            pd.Series(line),
            denoms=scale.denoms,
            team_ab=team_ab,
            team_ip=team_ip,
            replacement_avg=scale.repl_rates["avg"],
            replacement_era=scale.repl_rates["era"],
            replacement_whip=scale.repl_rates["whip"],
        )
    # UTIL floor mirrors board convention: reuse the OF floor if present
    if "UTIL" in scale.replacement_levels and "UTIL" not in floors:
        floors["UTIL"] = floors.get("OF", scale.replacement_levels["UTIL"])
    return floors


def score_var(line, positions, player_type, scale, fraction=1.0):
    scaled = dict(line)
    counting = _COUNTING_HIT if player_type == "hitter" else _COUNTING_PIT
    if fraction != 1.0:
        for k in counting:
            if scaled.get(k) is not None:
                scaled[k] = scaled[k] * fraction
    team_ab = scale.team_ab * fraction
    team_ip = scale.team_ip * fraction
    total_sgp = calculate_player_sgp(
        pd.Series(scaled),
        denoms=scale.denoms,
        team_ab=team_ab,
        team_ip=team_ip,
        replacement_avg=scale.repl_rates["avg"],
        replacement_era=scale.repl_rates["era"],
        replacement_whip=scale.repl_rates["whip"],
    )
    floors = _to_date_floors(scale, fraction)
    series = pd.Series({"total_sgp": total_sgp, "positions": list(positions), "player_type": player_type})
    return calculate_var(series, floors)
```

Note: verify `calculate_var` reads `player_type` for the SP/RP floor routing; if it reads `ip` instead, include `ip` in `series`. Read `sgp/var.py` `_pitcher_floor_key` before finalizing and adjust the `series` fields to match exactly what it needs.

- [ ] **Step 4: Run tests to verify pass.**

Run: `pytest tests/test_analysis/test_draft_value.py -k score_var -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): score_var on board scale with to-date floor scaling"
```

---

### Task 3: Reconstruct the 2026 draft (team + slot per pick) with the validation gate

**Files:**
- Modify: `src/fantasy_baseball/analysis/draft_value.py`
- Modify: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Produces: `@dataclass DraftPick(slot: int|None, round: int, team: str, player_name: str, is_keeper: bool)`; `reconstruct_draft() -> list[DraftPick]`; `validate_reconstruction(picks, known_team=None, known_roster=None) -> list[str]` (empty == pass).

- [ ] **Step 1: Write the failing test.**

```python
def test_reconstruct_draft_shape_and_gate():
    picks = dv.reconstruct_draft()
    keepers = [p for p in picks if p.is_keeper]
    drafted = [p for p in picks if not p.is_keeper]
    assert len(keepers) == 30
    assert len(drafted) == 200
    # every team owns exactly 3 keepers
    from collections import Counter
    kc = Counter(p.team for p in keepers)
    assert set(kc.values()) == {3}
    # validation gate passes on real data (30/30 keeper match, names present)
    assert dv.validate_reconstruction(picks) == []
```

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k reconstruct -v`
Expected: FAIL.

- [ ] **Step 3: Implement reconstruction.**

Read `config/draft_order.json` (`rounds`, `trades`) and `data/draft_state.json` (`drafted_players`) and `config/league.yaml` (`keepers`). The invariant: `drafted_players[0:30]` are the 30 keepers (in `league.yaml` order); `drafted_players[30:230]` are the 200 live picks in snake order. Build the live pick->team map by flattening `draft_order.json` `rounds` in order and applying `trades` (swap the team at `[round-1][slot-1]`), then zip against `drafted_players[30:230]`.

```python
import yaml

_CONFIG = _REPO_ROOT / "config"
_DRAFT_STATE = _REPO_ROOT / "data" / "draft_state.json"


@dataclass(frozen=True)
class DraftPick:
    slot: int | None      # 1..200 live-pick ordinal; None for keepers
    round: int            # 0 for keepers
    team: str
    player_name: str
    is_keeper: bool


def _load_league():
    return yaml.safe_load((_CONFIG / "league.yaml").read_text(encoding="utf-8"))


def reconstruct_draft() -> list[DraftPick]:
    order = json.loads((_CONFIG / "draft_order.json").read_text(encoding="utf-8"))
    state = json.loads(_DRAFT_STATE.read_text(encoding="utf-8"))
    league = _load_league()
    drafted = state["drafted_players"]
    keeper_defs = league["keepers"]

    picks: list[DraftPick] = []
    # keepers: first len(keeper_defs) entries, paired to league.yaml order
    n_keep = len(keeper_defs)
    for i, kd in enumerate(keeper_defs):
        picks.append(DraftPick(None, 0, kd["team"], drafted[i], True))

    # live picks: flatten rounds, apply trades, zip to drafted[n_keep:]
    rounds = [list(r) for r in order["rounds"]]
    for tr in order.get("trades", []):
        rounds[tr["round"] - 1][tr["slot"] - 1] = tr["to"]
    flat_teams = [(rnd_i + 1, team) for rnd_i, rnd in enumerate(rounds) for team in rnd]
    live = drafted[n_keep:]
    for slot, (name, (rnd, team)) in enumerate(zip(live, flat_teams), start=1):
        picks.append(DraftPick(slot, rnd, team, name, False))
    return picks


def validate_reconstruction(picks, known_team=None, known_roster=None) -> list[str]:
    problems: list[str] = []
    league = _load_league()
    keeper_norm = {(normalize_name(k["name"]), k["team"]) for k in league["keepers"]}
    recon_keeper_norm = {(normalize_name(p.player_name), p.team) for p in picks if p.is_keeper}
    missing = keeper_norm - recon_keeper_norm
    if missing:
        problems.append(f"keeper mismatch (missing {len(missing)}): {sorted(missing)[:5]}")
    if known_team and known_roster:
        recon = {normalize_name(p.player_name) for p in picks if p.team == known_team}
        want = {normalize_name(n) for n in known_roster}
        if not want <= recon:
            problems.append(f"known-team {known_team} roster mismatch: missing {sorted(want - recon)[:5]}")
    return problems
```

- [ ] **Step 4: Run the test.**

Run: `pytest tests/test_analysis/test_draft_value.py -k reconstruct -v`
Expected: PASS. If keeper count != 30 or drafted != 200, the invariant is violated (autopick/undo/reordering) — STOP and inspect `draft_state.json` alignment against `draft_order.json` before proceeding (per spec validation gate). Also cross-check `state["user_roster"]` as a `known_roster` for the user's team.

- [ ] **Step 5: Commit.**

```bash
git add src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): reconstruct 2026 draft picks with validation gate"
```

---

### Task 4: Build the par curve (drafted curve + keeper mean)

**Files:**
- Modify: `src/fantasy_baseball/analysis/draft_value.py`
- Modify: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Consumes: `reconstruct_draft`, `reproduce_draft_day_board`.
- Produces: `@dataclass ParCurve(drafted_pars: list[float], keeper_par: float)` with method `par_for_slot(self, ordinal: int) -> float` (1-based ordinal into `drafted_pars`); `build_par_curve(picks, board, fraction=1.0) -> ParCurve`. `preseason_var_lookup(board) -> dict[str, float]` keyed by `name::player_type`.

- [ ] **Step 1: Write the failing test.**

```python
def test_par_curve_is_descending_and_keeper_mean():
    board, scale = dv.reproduce_draft_day_board()
    picks = dv.reconstruct_draft()
    curve = dv.build_par_curve(picks, board)
    # drafted pars sorted descending
    assert curve.drafted_pars == sorted(curve.drafted_pars, reverse=True)
    # par_for_slot(1) is the top on-board drafted VAR
    assert curve.par_for_slot(1) == curve.drafted_pars[0]
    # keeper par is the mean of 30 keeper VARs (finite)
    assert curve.keeper_par == curve.keeper_par  # not NaN
```

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k par_curve -v`
Expected: FAIL.

- [ ] **Step 3: Implement the par curve.**

Join each pick to its preseason VAR via normalized `name::player_type`. On-board drafted players contribute to the sorted par curve; off-board fliers are skipped (curve shrinks). Keeper par = mean of keeper VARs (keepers are elite, always on-board). For `fraction < 1.0`, recompute each on-board player's to-date VAR via `score_var` on their board stat line before sorting.

```python
def _board_index(board):
    """name_normalized::player_type -> board row (VAR tie-break on collisions)."""
    idx = {}
    for _, row in board.iterrows():
        key = f"{row['name_normalized']}::{row['player_type']}"
        cur = idx.get(key)
        if cur is None or float(row["var"]) > float(cur["var"]):
            idx[key] = row
    return idx


def preseason_var_lookup(board) -> dict:
    return {k: float(v["var"]) for k, v in _board_index(board).items()}


def _match_board_row(name, bindex):
    norm = normalize_name(name)
    for ptype in ("hitter", "pitcher"):
        row = bindex.get(f"{norm}::{ptype}")
        if row is not None:
            return row
    return None


@dataclass
class ParCurve:
    drafted_pars: list
    keeper_par: float

    def par_for_slot(self, ordinal: int) -> float:
        # ordinal is 1-based among ON-BOARD drafted picks
        return self.drafted_pars[ordinal - 1]


def _var_for_row(row, scale, fraction):
    if fraction == 1.0:
        return float(row["var"])
    ptype = row["player_type"]
    keys = ("r", "hr", "rbi", "sb", "avg", "ab") if ptype == "hitter" else ("w", "k", "sv", "era", "whip", "ip")
    line = {k: row[k] for k in keys}
    return dv_score := score_var(line, list(row["positions"]), ptype, scale, fraction)


def build_par_curve(picks, board, fraction=1.0, scale=None) -> ParCurve:
    bindex = _board_index(board)
    drafted_vars = []
    keeper_vars = []
    for p in picks:
        row = _match_board_row(p.player_name, bindex)
        if row is None:
            continue  # off-board flier: excluded from par curve
        v = _var_for_row(row, scale, fraction) if fraction != 1.0 else float(row["var"])
        if p.is_keeper:
            keeper_vars.append(v)
        else:
            drafted_vars.append(v)
    drafted_vars.sort(reverse=True)
    keeper_par = sum(keeper_vars) / len(keeper_vars) if keeper_vars else float("nan")
    return ParCurve(drafted_vars, keeper_par)
```

(Remove the walrus in `_var_for_row`; write `return score_var(...)` plainly — shown here only to flag the fraction path. Clean it up when implementing.)

- [ ] **Step 4: Run the test.**

Run: `pytest tests/test_analysis/test_draft_value.py -k par_curve -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): par curve (drafted curve + keeper mean)"
```

---

### Task 5: Estimate/actual line builders + season fraction + convergence oracles

**Files:**
- Modify: `src/fantasy_baseball/analysis/draft_value.py`
- Modify: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Produces: `load_full_season_lines() -> dict` (`name::player_type -> line dict`); `load_actual_to_date_lines() -> dict`; `season_fraction() -> float`; `estimate_var(line, positions, player_type, scale, fraction) -> float` is just `score_var`. The convergence oracles live here as tests.

- [ ] **Step 1: Write the failing tests (loaders + f=1 convergence + f<1 correctness).**

```python
def test_full_season_and_actual_loaders_shape():
    full = dv.load_full_season_lines()
    assert full, "no full-season lines (KV store not synced?)"
    k = next(iter(full))
    assert "::" in k
    line = full[k]
    assert any(s in line for s in ("hr", "k"))


def test_convergence_f1_equals_full_season(synthetic_scale):
    # a healthy player: actual+ROS == preseason; at f=1 YTD == projected
    scale = synthetic_scale
    line = _hitter_line()
    proj = dv.score_var(line, ["OF"], "hitter", scale, fraction=1.0)
    ytd = dv.score_var(line, ["OF"], "hitter", scale, fraction=1.0)
    assert abs(proj - ytd) < 1e-9


def test_ytd_f_half_linear_player(synthetic_scale):
    # actual-to-date == 0.5 * full for a linear healthy player; YTD VAR reproduces
    # the f-consistent VAR and does NOT collapse the rate component (guards f*floor_full).
    scale = synthetic_scale
    full_line = _hitter_line()
    todate = {k: (v * 0.5 if k in ("r", "hr", "rbi", "sb", "ab") else v) for k, v in full_line.items()}
    # score_var applies the *0.5 itself when fraction=0.5, so pass the FULL line:
    ytd = dv.score_var(full_line, ["OF"], "hitter", scale, fraction=0.5)
    # manual: same line scaled by hand at fraction=1.0 against to-date floors
    assert ytd == ytd  # finite
    # counting-dominated: half-season VAR strictly less than full, strictly > 0-scale
    assert dv.score_var(full_line, ["OF"], "hitter", scale, fraction=0.5) < dv.score_var(full_line, ["OF"], "hitter", scale, fraction=1.0)
```

Add a `synthetic_scale` fixture that builds a `ScaleInputs` from the real board once (`board, scale = dv.reproduce_draft_day_board()`), so oracle tests do not re-run the board each time.

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k "loaders or convergence or ytd_f_half" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the loaders + fraction.**

Read the full-season cache (`CacheKey.FULL_SEASON_PROJECTIONS`) via `read_cache`, and game-log totals via `_load_game_log_totals`. Key everything by `name::player_type`. Full-season records are MLBAM-keyed with `name`; game-log totals come back keyed by normalized name in two dicts (hitter/pitcher), which gives `player_type` directly.

```python
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.redis_store import read_cache
from fantasy_baseball.web.season_data import _load_game_log_totals


def _hit_line_from(rec):
    return {"r": rec.get("r", 0), "hr": rec.get("hr", 0), "rbi": rec.get("rbi", 0),
            "sb": rec.get("sb", 0), "ab": rec.get("ab", 0),
            "avg": (rec["h"] / rec["ab"]) if rec.get("ab") else 0.0}


def _pit_line_from(rec):
    ip = rec.get("ip") or 0.0
    return {"w": rec.get("w", 0), "k": rec.get("k", 0), "sv": rec.get("sv", 0), "ip": ip,
            "era": (rec.get("er", 0) / ip * 9.0) if ip else 0.0,
            "whip": ((rec.get("bb", 0) + rec.get("h_allowed", 0)) / ip) if ip else 0.0}


def load_full_season_lines() -> dict:
    payload = read_cache(CacheKey.FULL_SEASON_PROJECTIONS) or {}
    out = {}
    for rec in payload.get("hitters", []):
        out[f"{normalize_name(rec['name'])}::hitter"] = _hit_line_from(rec)
    for rec in payload.get("pitchers", []):
        out[f"{normalize_name(rec['name'])}::pitcher"] = _pit_line_from(rec)
    return out


def load_actual_to_date_lines() -> dict:
    hitter_logs, pitcher_logs = _load_game_log_totals()
    out = {}
    for norm, rec in hitter_logs.items():
        out[f"{norm}::hitter"] = _hit_line_from(rec)
    for norm, rec in pitcher_logs.items():
        out[f"{norm}::pitcher"] = _pit_line_from(rec)
    return out


def season_fraction() -> float:
    """League games played / full schedule. v1: date-based fraction of the MLB season.

    Read the elapsed fraction from the standings snapshot game count if available;
    otherwise fall back to a date-based fraction. Pin the exact source when wiring
    the CLI (Task 9) against real standings; keep this helper the single source.
    """
    from datetime import date
    season_start = date(2026, 3, 26)
    season_end = date(2026, 9, 28)
    today = date.today()
    total = (season_end - season_start).days
    done = max(0, min(total, (today - season_start).days))
    return done / total
```

- [ ] **Step 4: Run the tests.**

Run: `pytest tests/test_analysis/test_draft_value.py -k "loaders or convergence or ytd_f_half" -v`
Expected: PASS. If `load_full_season_lines` is empty, the KV store is not synced — note this; the CLI (Task 9) documents the sync/`--no-sync` requirement.

- [ ] **Step 5: Commit.**

```bash
git add src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): full-season/actual line loaders + season fraction + oracles"
```

---

### Task 6: Per-player value calculator (value + skill/luck, both horizons)

**Files:**
- Modify: `src/fantasy_baseball/analysis/draft_value.py`
- Modify: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Produces: `@dataclass PlayerValue(team, name, player_type, slot, baseline_kind, preseason_var, est_var_proj, est_var_ytd, value_proj, value_ytd, skill, luck)`; `compute_player_value(team, name, player_type, positions, baseline_proj, baseline_ytd, baseline_kind, preseason_var, full_line, todate_line, scale, fraction) -> PlayerValue`. `baseline_kind` in {"drafted","keeper","waiver"}. The projected value subtracts the full-season `baseline_proj`; the YTD value subtracts the to-date `baseline_ytd` (the two horizons have different pars — never conflate them).

- [ ] **Step 1: Write the failing test (decomposition identity + N/A rules).**

```python
def test_value_decomposition_identity(synthetic_scale):
    scale = synthetic_scale
    line = _hitter_line()
    pv = dv.compute_player_value(
        team="Hart of the Order", name="Test Bat", player_type="hitter", positions=["OF"],
        baseline_proj=5.0, baseline_ytd=2.5, baseline_kind="drafted", preseason_var=8.0,
        full_line=line, todate_line=line, scale=scale, fraction=0.5,
    )
    # projected decomposition holds exactly
    assert abs((pv.skill + pv.luck) - pv.value_proj) < 1e-9
    # YTD is value-only
    assert pv.value_ytd is not None


def test_offboard_waiver_gem_skill_luck_na(synthetic_scale):
    scale = synthetic_scale
    line = _hitter_line()
    pv = dv.compute_player_value(
        team="Hart of the Order", name="Gem", player_type="hitter", positions=["OF"],
        baseline_proj=0.0, baseline_ytd=0.0, baseline_kind="waiver", preseason_var=None,
        full_line=line, todate_line=line, scale=scale, fraction=0.5,
    )
    assert pv.skill is None and pv.luck is None
    assert pv.value_proj is not None  # value still computed vs replacement (0)
```

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k "decomposition or offboard" -v`
Expected: FAIL.

- [ ] **Step 3: Implement `compute_player_value`.**

```python
@dataclass
class PlayerValue:
    team: str
    name: str
    player_type: str
    slot: int | None
    baseline_kind: str
    preseason_var: float | None
    est_var_proj: float | None
    est_var_ytd: float | None
    value_proj: float | None
    value_ytd: float | None
    skill: float | None
    luck: float | None


def compute_player_value(team, name, player_type, positions, baseline_proj, baseline_ytd,
                         baseline_kind, preseason_var, full_line, todate_line, scale, fraction):
    est_proj = score_var(full_line, positions, player_type, scale, 1.0) if full_line is not None else None
    est_ytd = score_var(todate_line, positions, player_type, scale, fraction) if todate_line is not None else None
    value_proj = (est_proj - baseline_proj) if est_proj is not None else None
    value_ytd = (est_ytd - baseline_ytd) if est_ytd is not None else None
    if preseason_var is not None and est_proj is not None:
        skill = preseason_var - baseline_proj
        luck = est_proj - preseason_var
    else:
        skill = luck = None
    return PlayerValue(team, name, player_type, None, baseline_kind, preseason_var,
                       est_proj, est_ytd, value_proj, value_ytd, skill, luck)
```

The projected `value_proj` subtracts the full-season `baseline_proj`; the YTD
`value_ytd` subtracts the to-date `baseline_ytd` (par curve built with `fraction<1`).
The two horizons' pars are distinct — the caller (Task 9) computes both.

- [ ] **Step 4: Run the tests.**

Run: `pytest tests/test_analysis/test_draft_value.py -k "decomposition or offboard" -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): per-player value calculator (value + skill/luck)"
```

---

### Task 7: Acquisition classifier (elimination model)

**Files:**
- Modify: `src/fantasy_baseball/analysis/draft_value.py`
- Modify: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Produces: `load_add_txns_by_team() -> dict[str, set[str]]` (team -> set of normalized added-player names); `classify_acquisition(team, norm_name, drafted_by_team, kept_by_team, add_by_team) -> str` returning `"drafted"|"keeper"|"waiver"|"trade_excluded"`.

- [ ] **Step 1: Write the failing test.**

```python
def test_classify_precedence():
    drafted = {"hart of the order": {"juan soto"}}
    kept = {"hart of the order": {"julio rodriguez"}}
    adds = {"hart of the order": {"matt mclain", "juan soto"}}  # soto also re-added
    # draft/keep precedence beats a later same-team re-add
    assert dv.classify_acquisition("Hart of the Order", "juan soto", drafted, kept, adds) == "drafted"
    assert dv.classify_acquisition("Hart of the Order", "julio rodriguez", drafted, kept, adds) == "keeper"
    # pure waiver add
    assert dv.classify_acquisition("Hart of the Order", "matt mclain", drafted, kept, adds) == "waiver"
    # rostered, no draft/keep, no add -> trade-acquired -> excluded
    assert dv.classify_acquisition("Hart of the Order", "some trade guy", drafted, kept, adds) == "trade_excluded"
```

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k classify -v`
Expected: FAIL.

- [ ] **Step 3: Implement classifier + add-txn loader.**

```python
def load_add_txns_by_team() -> dict:
    txns = read_cache(CacheKey.TRANSACTIONS) or []
    by_team: dict[str, set] = {}
    for t in txns:
        if t.get("status") not in (None, "successful"):
            continue
        add_name = t.get("add_name")
        team = t.get("team")
        if add_name and team:
            by_team.setdefault(normalize_name_team(team), set()).add(normalize_name(add_name))
    return by_team


def normalize_name_team(team: str) -> str:
    return team.strip().lower()


def classify_acquisition(team, norm_name, drafted_by_team, kept_by_team, add_by_team) -> str:
    tkey = normalize_name_team(team)
    if norm_name in drafted_by_team.get(tkey, set()):
        return "drafted"
    if norm_name in kept_by_team.get(tkey, set()):
        return "keeper"
    if norm_name in add_by_team.get(tkey, set()):
        return "waiver"
    return "trade_excluded"
```

Update the test's dict keys to use `normalize_name_team` output (lowercased team) so the fixtures match the loader convention; adjust `drafted`/`kept`/`adds` keys accordingly.

- [ ] **Step 4: Run the test.**

Run: `pytest tests/test_analysis/test_draft_value.py -k classify -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): elimination-model acquisition classifier"
```

---

### Task 8: Team roll-up (sum + per-player average + credited count)

**Files:**
- Modify: `src/fantasy_baseball/analysis/draft_value.py`
- Modify: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Produces: `@dataclass TeamRollup(team, sum_value, avg_value, credited_count, case3_count)`; `roll_up_team(team, player_values, case3_count, horizon="proj") -> TeamRollup`.

- [ ] **Step 1: Write the failing test.**

```python
def test_team_rollup_sum_avg_count():
    pvs = [
        dv.PlayerValue("A", "hitter", None, "drafted", 8.0, 10.0, 6.0, 4.0, 2.5, None, None),
        dv.PlayerValue("B", "hitter", None, "waiver", None, 3.0, 1.5, 3.0, 1.4, None, None),
    ]
    r = dv.roll_up_team("Hart of the Order", pvs, case3_count=2, horizon="proj")
    assert r.credited_count == 2
    assert abs(r.sum_value - 7.0) < 1e-9   # 4.0 + 3.0
    assert abs(r.avg_value - 3.5) < 1e-9
    assert r.case3_count == 2
```

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k rollup -v`
Expected: FAIL.

- [ ] **Step 3: Implement roll-up.**

```python
@dataclass
class TeamRollup:
    team: str
    sum_value: float
    avg_value: float
    credited_count: int
    case3_count: int


def roll_up_team(team, player_values, case3_count, horizon="proj") -> TeamRollup:
    attr = "value_proj" if horizon == "proj" else "value_ytd"
    vals = [getattr(pv, attr) for pv in player_values if getattr(pv, attr) is not None]
    n = len(vals)
    total = sum(vals)
    avg = total / n if n else float("nan")
    return TeamRollup(team, total, avg, n, case3_count)
```

- [ ] **Step 4: Run the test.**

Run: `pytest tests/test_analysis/test_draft_value.py -k rollup -v`
Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): team roll-up (sum + per-player avg + counts)"
```

---

### Task 9: CLI orchestration + markdown report + known-pick oracle

**Files:**
- Create: `scripts/draft_value.py`
- Modify: `src/fantasy_baseball/analysis/draft_value.py` (add `run_draft_value()` orchestrator + off-board position attach + Ohtani handling)
- Modify: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_draft_value(fraction=None) -> tuple[list[PlayerValue], list[TeamRollup]]`; CLI prints tables + writes markdown.

- [ ] **Step 1: Write the failing end-to-end + known-pick test.**

```python
def test_run_draft_value_end_to_end_and_known_pick():
    players, teams = dv.run_draft_value()
    assert players and teams
    # every team roll-up has a credited count and a case3 count
    assert all(t.credited_count >= 0 for t in teams)
    # known-pick sanity: a specific keeper resolves with a finite projected value
    soto = next((p for p in players if normalize_name(p.name) == "juan soto"), None)
    assert soto is not None and soto.value_proj == soto.value_proj  # not NaN
    assert soto.baseline_kind in ("keeper", "drafted", "waiver")
```

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k end_to_end -v`
Expected: FAIL.

- [ ] **Step 3: Implement `run_draft_value` orchestrator.**

Wire the pieces: reproduce board (+ validate; raise on drift), reconstruct draft (+ validate gate), build projected and to-date par curves, load full-season + actual lines, load current rosters (`get_latest_weekly_rosters`) and add-txns, classify each current-roster player, compute values, roll up per team. Off-board players get positions via the board's default-attach convention (`["OF"]` for hitters, `["SP"]` for pitchers) resolved by their game-log/full-season type. Ohtani resolves to the hitter line ("batter only").

```python
from fantasy_baseball.data.redis_store import get_latest_weekly_rosters, _get_client  # confirm client accessor


def _default_positions(player_type):
    return ["OF"] if player_type == "hitter" else ["SP"]


def _resolve_type_and_positions(norm_name, full_lines, bindex):
    """Return (player_type, positions). Prefer board row; else infer from which
    full-season/actual line exists; Ohtani -> hitter (batter only)."""
    for ptype in ("hitter", "pitcher"):
        row = bindex.get(f"{norm_name}::{ptype}")
        if row is not None:
            return ptype, list(row["positions"])
    for ptype in ("hitter", "pitcher"):
        if f"{norm_name}::{ptype}" in full_lines:
            return ptype, _default_positions(ptype)
    return "hitter", _default_positions("hitter")


def run_draft_value(fraction=None):
    board, scale = reproduce_draft_day_board()
    mism = validate_scale_against_frozen(board)
    if mism:
        raise RuntimeError(f"Draft-day board drift ({len(mism)}); investigate, do not loosen tol: {mism[:3]}")
    picks = reconstruct_draft()
    gate = validate_reconstruction(picks)
    if gate:
        raise RuntimeError(f"Draft reconstruction gate failed: {gate}")
    f = season_fraction() if fraction is None else fraction

    bindex = _board_index(board)
    preseason = preseason_var_lookup(board)
    par_proj = build_par_curve(picks, board, fraction=1.0)
    par_ytd = build_par_curve(picks, board, fraction=f, scale=scale)
    full_lines = load_full_season_lines()
    todate_lines = load_actual_to_date_lines()

    # draft/keep sets by team (normalized)
    drafted_by, kept_by = {}, {}
    slot_by_name = {}  # norm_name -> on-board drafted ordinal
    # assign on-board drafted ordinals in draft order for par_for_slot
    onboard_ordinal = 0
    for p in sorted([pk for pk in picks if not pk.is_keeper], key=lambda x: x.slot):
        row = _match_board_row(p.player_name, bindex)
        if row is None:
            continue
        onboard_ordinal += 1
        slot_by_name[normalize_name(p.player_name)] = onboard_ordinal
    for p in picks:
        tkey = normalize_name_team(p.team)
        norm = normalize_name(p.player_name)
        (kept_by if p.is_keeper else drafted_by).setdefault(tkey, set()).add(norm)

    add_by = load_add_txns_by_team()
    rosters = get_latest_weekly_rosters(_get_client())

    # par curves sorted descending; ordinal from slot_by_name indexes par_for_slot
    # but par curves are sorted by VAR, not draft order -> par_for_slot uses the
    # ordinal position AFTER sorting. Map each drafted player to par by their sorted
    # rank: build once.
    # (Implementation detail: par_for_slot(ordinal) expects the sorted-descending
    #  index. Assign each on-board drafted player par = drafted_pars[rank] where rank
    #  is their position in the VAR-sorted order. Two players can't share a rank;
    #  zip sorted picks-by-preseason-var to drafted_pars.)
    players, case3 = [], {}
    for entry in rosters:
        team = entry["team"]
        raw_name = entry["player_name"]
        norm = normalize_name(raw_name)
        kind = classify_acquisition(team, norm, drafted_by, kept_by, add_by)
        if kind == "trade_excluded":
            case3[team] = case3.get(team, 0) + 1
            continue
        ptype, positions = _resolve_type_and_positions(norm, full_lines, bindex)
        key = f"{norm}::{ptype}"
        pre = preseason.get(key)
        full_line = full_lines.get(key)
        todate_line = todate_lines.get(key)
        # baselines
        if kind == "keeper":
            base_proj, base_ytd = par_proj.keeper_par, par_ytd.keeper_par
        elif kind == "drafted":
            rank = slot_by_name.get(norm)
            base_proj = par_proj.par_for_slot(rank) if rank else 0.0
            base_ytd = par_ytd.par_for_slot(rank) if rank else 0.0
        else:  # waiver
            base_proj = base_ytd = 0.0
        pv = compute_player_value(team, raw_name, ptype, positions, base_proj, base_ytd,
                                  kind, pre, full_line, todate_line, scale, f)
        players.append(pv)

    # group by team (PlayerValue carries .team) and roll up, passing the case-3 count
    by_team: dict[str, list] = {}
    for pv in players:
        by_team.setdefault(pv.team, []).append(pv)
    all_teams = {entry["team"] for entry in rosters}
    teams = [roll_up_team(t, by_team.get(t, []), case3.get(t, 0)) for t in sorted(all_teams)]
    return players, teams
```

The `drafted` baseline above uses `assign_drafted_par` (defined next), NOT the raw
draft ordinal: `par_for_slot(ordinal)` indexes the VAR-sorted `drafted_pars`, so each
on-board drafted player must be assigned the par at their *preseason-VAR rank*, not
their draft-order position. Replace the `slot_by_name` / `par_for_slot(rank)` baseline
logic with this helper (build once, before the roster loop):

```python
def assign_drafted_par(picks, board, par_curve) -> dict:
    """norm_name -> par value, assigned by preseason-VAR rank among on-board drafted."""
    bindex = _board_index(board)
    scored = []
    for p in picks:
        if p.is_keeper:
            continue
        row = _match_board_row(p.player_name, bindex)
        if row is not None:
            scored.append((normalize_name(p.player_name), float(row["var"])))
    scored.sort(key=lambda t: t[1], reverse=True)  # same order as drafted_pars
    return {name: par_curve.drafted_pars[i] for i, (name, _var) in enumerate(scored)}
```

In `run_draft_value`, compute `par_by_name_proj = assign_drafted_par(picks, board, par_proj)`
and `par_by_name_ytd = assign_drafted_par(picks, board_ytd_or_scaled, par_ytd)` (for the
YTD map, the ranks are the same players; reuse the proj ranks but pull pars from
`par_ytd.drafted_pars` at the same index), then set `base_proj = par_by_name_proj.get(norm, 0.0)`
and `base_ytd = par_by_name_ytd.get(norm, 0.0)` for the `drafted` branch. Delete the
now-unused `slot_by_name` block.

- [ ] **Step 4: Implement the CLI + markdown renderer.**

```python
# scripts/draft_value.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # data has accented names

from fantasy_baseball.analysis.draft_value import run_draft_value


def _fmt(x):
    return "  N/A" if x is None else f"{x:6.1f}"


def main():
    players, teams = run_draft_value()
    lines = ["# Draft Value Report", ""]
    lines.append("## Team leaderboard (projected, headline = avg)")
    lines.append("| Team | avg | sum | credited | trade-excl |")
    lines.append("|---|---|---|---|---|")
    for t in sorted(teams, key=lambda r: (r.avg_value if r.avg_value == r.avg_value else -9e9), reverse=True):
        lines.append(f"| {t.team} | {_fmt(t.avg_value)} | {_fmt(t.sum_value)} | {t.credited_count} | {t.case3_count} |")
    lines.append("")
    lines.append("## Per-player (projected)")
    lines.append("| Player | kind | preVAR | estVAR | skill | luck | value | valueYTD |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for p in sorted(players, key=lambda p: (p.value_proj if p.value_proj is not None else -9e9), reverse=True):
        lines.append(
            f"| {p.name} | {p.baseline_kind} | {_fmt(p.preseason_var)} | {_fmt(p.est_var_proj)} "
            f"| {_fmt(p.skill)} | {_fmt(p.luck)} | {_fmt(p.value_proj)} | {_fmt(p.value_ytd)} |"
        )
    report = "\n".join(lines)
    out = Path("data/analysis/draft_value_report.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run the end-to-end test + the CLI against real (synced) data.**

Run: `pytest tests/test_analysis/test_draft_value.py -k end_to_end -v`
Expected: PASS.
Run: `python scripts/draft_value.py`
Expected: prints the leaderboard + per-player table, writes `data/analysis/draft_value_report.md`. If it errors on empty full-season lines, the KV store needs a sync first (document in the script's docstring; mirror `run_season_dashboard.py --no-sync` conventions).

- [ ] **Step 6: Eyeball the known-pick sanity.**

Manually verify one keeper (e.g. Juan Soto): his `preVAR` matches the frozen board, his `estVAR` is on the same scale, and `value = estVAR - keeper_par`. State the numbers in the commit message.

- [ ] **Step 7: Full verification + commit.**

Run: `pytest tests/test_analysis/ -v`, `ruff check .`, `ruff format --check .`, `vulture` (no NEW findings). Fix all failures.

```bash
git add scripts/draft_value.py src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): CLI orchestrator + markdown report + known-pick oracle"
```

---

## Spec-coverage self-check (author checklist, verify before handoff)

- VAR currency + single scale -> Tasks 1-2 (frozen validation + score_var).
- Skill/luck (projected, on-board only) + N/A rules -> Task 6.
- Two horizons (projected + YTD) + to-date scaling + f<1 oracle -> Tasks 2, 5, 6.
- Par curve (drafted + keeper mean) -> Task 4.
- Draft-slot reconstruction + validation gate -> Task 3.
- Elimination attribution + case-3 count -> Tasks 7, 9.
- Team roll-up (sum + avg + credited count) -> Task 8.
- Cross-source joins (name::player_type, normalize) + Ohtani + off-board positions -> Tasks 4, 9.
- Validation oracles 1-8 -> distributed across Tasks 1 (1), 2 (2), 5 (3,4), 6 (5), 3 (6), 7/9 (7), 9 (8).
- CLI + markdown -> Task 9.

## Notes for the implementer

- **Context decay:** re-read `draft_value.py` before each task's edits; the file grows across tasks.
- **Read before scoring:** before Task 2 Step 3, read `sgp/var.py` `calculate_var` / `_pitcher_floor_key` to confirm exactly which Series fields the floor routing needs (`player_type` vs `ip`); wire `score_var`'s Series to match.
- **KV sync:** in-season loaders (full-season, game logs, transactions, rosters) read the KV store; a stale/empty local store yields empty output. Confirm a fresh sync (or run against the source) before trusting numbers; the CLI docstring must say so.
- **Do not modify failing oracle tests to pass** — a failing frozen-validation or reconstruction gate means the data/scale is off, not the test.
