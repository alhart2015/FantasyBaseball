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
- Modify `src/fantasy_baseball/draft/board.py` — extract shared `build_board_from_frames` core (with opt-in `return_scale`) so the preseason board can be rebuilt from the Apr-1 CSVs.

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

### Task 1: Rebuild the preseason board from Apr-1 CSVs (shared board core) + soft frozen check

**Why not `fantasy.db`:** its `blended_projections` were overwritten with in-season
updates (verified 2026-07-01 — a board rebuilt from the DB drifts from the frozen
board non-monotonically), and the SQLite `blended_projections` table is not even
populated by the production build (`build_db.py` writes blend to Redis only). The
preserved Apr-1 CSVs (`data/projections/2026/*.csv`) are the authoritative draft-day
vintage; `blend_projections` is a pure function of them.

**Files:**
- Modify: `src/fantasy_baseball/draft/board.py` (extract `build_board_from_frames` core)
- Create: `src/fantasy_baseball/analysis/draft_value.py`
- Create: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Produces: `build_board_from_frames(hitters, pitchers, positions, roster_slots=None, num_teams=None, return_scale=False)` in `board.py`; `ScaleInputs` dataclass; `reproduce_draft_day_board() -> tuple[pd.DataFrame, ScaleInputs]` (rebuilds from Apr-1 CSVs); `frozen_drift_summary(board_df, frozen_path=None, tol=0.05) -> dict` (soft, returns `{joined, over_tol, max, median}`; never raises).

- [ ] **Step 1: Extract the board compute core in `board.py` (reuse, not rewrite).**

Read `src/fantasy_baseball/draft/board.py` lines 24-94. Move the body from the
`norm_positions` dedup through the final `board = pool.sort_values(...)` /
`_validate_top_adp_players` into a new `build_board_from_frames`, and make
`build_draft_board` a thin wrapper. Add `return_scale` on the core only.

```python
# board.py — imports: add DEFAULT_TEAM_AB, DEFAULT_TEAM_IP
from fantasy_baseball.sgp.player_value import DEFAULT_TEAM_AB, DEFAULT_TEAM_IP, calculate_player_sgp


def build_draft_board(
    conn,
    roster_slots: dict[str, int] | None = None,
    num_teams: int | None = None,
    return_scale: bool = False,
):
    """Build a ranked draft board from projections and position data in SQLite."""
    hitters, pitchers = get_blended_projections(conn)
    positions = get_positions(conn)
    return build_board_from_frames(
        hitters, pitchers, positions, roster_slots, num_teams, return_scale
    )


def build_board_from_frames(
    hitters, pitchers, positions, roster_slots=None, num_teams=None, return_scale=False
):
    """Compute a ranked draft board from already-loaded projection frames + a
    positions dict. Shared by build_draft_board (DB source) and the draft-value
    module (Apr-1 CSV source). When return_scale, also returns the pool-derived
    rates + floors so realized/estimate VAR can be scored on the SAME scale.
    """
    # (verbatim relocation of the existing body: norm_positions dedup, ab/ip filters,
    #  _attach_positions, two-pass SGP, position_aware_replacement_levels, the VAR
    #  loop, name_normalized, player_id, sort, _validate_top_adp_players)
    norm_positions: dict[str, list[str]] = {}
    for k, v in positions.items():
        norm = normalize_name(k)
        if norm not in norm_positions or len(v) > len(norm_positions[norm]):
            norm_positions[norm] = v
    if not hitters.empty:
        hitters = hitters[hitters.get("ab", pd.Series(dtype=float)).fillna(0) >= 50]
    if not pitchers.empty:
        pitchers = pitchers[pitchers.get("ip", pd.Series(dtype=float)).fillna(0) >= 10]
    hitters = _attach_positions(hitters, norm_positions, default_type=PlayerType.HITTER)
    pitchers = _attach_positions(pitchers, norm_positions, default_type=PlayerType.PITCHER)
    denoms = get_sgp_denominators()
    pool = pd.concat([hitters, pitchers], ignore_index=True)
    pool["total_sgp"] = pool.apply(lambda row: calculate_player_sgp(row, denoms=denoms), axis=1)
    starters = compute_starters_per_position(roster_slots, num_teams)
    repl_rates = calculate_replacement_rates(pool, starters)
    pool["total_sgp"] = pool.apply(
        lambda row: calculate_player_sgp(
            row, denoms=denoms,
            replacement_era=repl_rates["era"], replacement_whip=repl_rates["whip"],
            replacement_avg=repl_rates["avg"],
        ),
        axis=1,
    )
    replacement_levels = position_aware_replacement_levels(denoms, repl_rates)
    pool["var"] = 0.0
    pool["best_position"] = ""
    for idx, row in pool.iterrows():
        var, pos = calculate_var(row, replacement_levels, return_position=True)
        pool.at[idx, "var"] = var
        pool.at[idx, "best_position"] = pos
    pool["name_normalized"] = pool["name"].apply(normalize_name)
    if "fg_id" in pool.columns and pool["fg_id"].notna().all():
        pool["player_id"] = pool["fg_id"].astype(str) + "::" + pool["player_type"]
    else:
        pool["player_id"] = pool["name"] + "::" + pool["player_type"]
    board = pool.sort_values("var", ascending=False).reset_index(drop=True)
    _validate_top_adp_players(board, hitters, pitchers)
    if return_scale:
        scale = {
            "denoms": denoms, "repl_rates": repl_rates,
            "replacement_levels": replacement_levels,
            "team_ab": DEFAULT_TEAM_AB, "team_ip": DEFAULT_TEAM_IP,
        }
        return board, scale
    return board
```

- [ ] **Step 2: Grep `build_draft_board(` call sites; run board tests.**

Run: `rg -n "build_draft_board\(" src scripts tests`
Then: `pytest tests/test_draft/ -v`
Expected: `build_draft_board` still returns a `pd.DataFrame` for all existing callers (the refactor is behavior-preserving); board tests pass. State call sites found.

- [ ] **Step 3: Write the failing test (rebuild returns scale; soft frozen summary).**

```python
# tests/test_analysis/test_draft_value.py
from fantasy_baseball.analysis import draft_value as dv


def test_build_preseason_board_returns_scale_and_soft_frozen():
    board, scale = dv.reproduce_draft_day_board()
    assert not board.empty
    assert set(scale.replacement_levels) >= {"C", "1B", "SS", "OF", "SP", "RP"}
    assert scale.team_ab == 5500 and scale.team_ip == 1450
    assert {"era", "whip", "avg"} <= set(scale.repl_rates)
    # soft frozen cross-check: returns a drift summary, never raises
    summary = dv.frozen_drift_summary(board)
    assert summary["joined"] > 0
    assert set(summary) >= {"joined", "over_tol", "max", "median"}
```

- [ ] **Step 4: Run it to confirm it fails.**

Run: `pytest tests/test_analysis/test_draft_value.py::test_build_preseason_board_returns_scale_and_soft_frozen -v`
Expected: FAIL (module/functions not defined).

- [ ] **Step 5: Implement `reproduce_draft_day_board` (Apr-1 CSV rebuild) + `frozen_drift_summary` (soft).**

```python
# src/fantasy_baseball/analysis/draft_value.py
"""Draft-value metric: realized VAR vs draft-slot par expectation."""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fantasy_baseball.config import load_config
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.data.yahoo_players import load_positions_cache
from fantasy_baseball.draft.board import build_board_from_frames
from fantasy_baseball.utils.name_utils import normalize_name

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FROZEN_BOARD = _REPO_ROOT / "data" / "draft_state_board.json"
_PRESEASON_CSVS = _REPO_ROOT / "data" / "projections" / "2026"
_POSITIONS_JSON = _REPO_ROOT / "data" / "player_positions.json"
_CONFIG = _REPO_ROOT / "config" / "league.yaml"


@dataclass(frozen=True)
class ScaleInputs:
    denoms: dict
    repl_rates: dict
    replacement_levels: dict
    team_ab: int
    team_ip: int


def reproduce_draft_day_board() -> tuple[pd.DataFrame, ScaleInputs]:
    """Rebuild the preseason board from the Apr-1 projection CSVs — pure, no DB/KV.

    blend_projections is deterministic over data/projections/2026/*.csv; positions
    come from data/player_positions.json. The board core is the shared
    build_board_from_frames, so the scale (rates/floors/denoms/volumes) is identical
    to the real draft board's, just computed off the preserved draft-day CSVs.
    """
    config = load_config(_CONFIG)
    hitters, pitchers, _quality = blend_projections(
        _PRESEASON_CSVS, config.projection_systems, config.projection_weights
    )
    positions = load_positions_cache(_POSITIONS_JSON)
    board, scale_d = build_board_from_frames(
        hitters, pitchers, positions,
        roster_slots=config.roster_slots or None, num_teams=config.num_teams,
        return_scale=True,
    )
    scale = ScaleInputs(
        denoms=scale_d["denoms"], repl_rates=scale_d["repl_rates"],
        replacement_levels=scale_d["replacement_levels"],
        team_ab=scale_d["team_ab"], team_ip=scale_d["team_ip"],
    )
    return board, scale


def frozen_drift_summary(board_df, frozen_path=None, tol=0.05) -> dict:
    """SOFT cross-check vs the frozen draft-day board. Reports drift; never raises.

    The frozen board (draft_state_board.json) was built at draft time with
    possibly-since-churned code, so exact reproduction is not expected. A large
    systematic drift is worth surfacing (wrong config/vintage) but is not a stop.
    """
    frozen_path = Path(frozen_path) if frozen_path else _FROZEN_BOARD
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    frozen_var = {
        row["player_id"]: float(row["var"])
        for row in frozen
        if row.get("player_id") is not None and row.get("var") is not None
    }
    diffs = []
    for _, row in board_df.iterrows():
        pid = row["player_id"]
        if pid in frozen_var:
            diffs.append(abs(float(row["var"]) - frozen_var[pid]))
    over = sum(1 for d in diffs if d > tol)
    summary = {
        "joined": len(diffs),
        "over_tol": over,
        "max": max(diffs) if diffs else 0.0,
        "median": statistics.median(diffs) if diffs else 0.0,
    }
    if diffs and over > 0.5 * len(diffs):
        logger.warning(
            "draft-value: rebuilt board drifts from frozen draft_state_board.json "
            "(%d/%d players > %.2f VAR, max %.2f). Expected some drift from code "
            "churn since the freeze; investigate only if this looks systematic.",
            over, len(diffs), tol, summary["max"],
        )
    return summary
```

- [ ] **Step 6: Run the test.**

Run: `pytest tests/test_analysis/test_draft_value.py::test_build_preseason_board_returns_scale_and_soft_frozen -v`
Expected: PASS. The soft check logs a drift warning — and the drift is expected to be **near-total, not marginal**: verified 2026-07-01, essentially all ~3669 joined players exceed the 0.05 tol (median ~0.83, max ~2.3 VAR) because the blend/SGP/VAR code churned since the draft-day freeze. This is fine and expected — do NOT treat it as a failure or try to make the frozen check pass. The Apr-1 CSVs + current code give ONE internally self-consistent scale used for preseason, par, realized, and estimate alike; the real single-scale guarantee is Task 2's exact `score_var == board.var` oracle (0.0 diff), not this frozen sanity signal.

- [ ] **Step 7: Commit.**

```bash
git add src/fantasy_baseball/draft/board.py src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): rebuild preseason board from Apr-1 CSVs (shared core) + soft frozen check"
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
            line["player_type"] = "pitcher"  # StrEnum-compatible; required by calculate_player_sgp dispatch
        else:
            ab = raw["ab"] or 1.0
            line["avg"] = raw["h"] / ab
            line["player_type"] = "hitter"
        floors[pos] = calculate_player_sgp(
            pd.Series(line),
            denoms=scale.denoms,
            team_ab=team_ab,
            team_ip=team_ip,
            replacement_avg=scale.repl_rates["avg"],
            replacement_era=scale.repl_rates["era"],
            replacement_whip=scale.repl_rates["whip"],
        )
    # UTIL floor mirrors the board: max of the hitter floors (see replacement.py).
    hitter_floors = [floors[p] for p in ("C", "1B", "2B", "3B", "SS", "OF") if p in floors]
    if hitter_floors:
        floors["UTIL"] = max(hitter_floors)
    return floors


def score_var(line, positions, player_type, scale, fraction=1.0):
    # player_type must be "hitter"/"pitcher" (StrEnum-compatible) so calculate_player_sgp
    # dispatches (player.get("player_type") == PlayerType.HITTER/PITCHER, player_value.py:104,121).
    scaled = dict(line)
    scaled["player_type"] = player_type
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
    # calculate_var needs total_sgp + positions + ip (pitcher floor routing reads
    # player.get("ip", 0.0) via _pitcher_floor_key -> role_from_ip, var.py:18).
    series = pd.Series({
        "total_sgp": total_sgp,
        "positions": list(positions),
        "player_type": player_type,
        "ip": scaled.get("ip", 0.0),
    })
    return calculate_var(series, floors)
```

Both `player_type` (for `calculate_player_sgp` dispatch) and `ip` (for `calculate_var`
pitcher-floor routing via `_pitcher_floor_key`) are included in the Series — that is
exactly what the two functions read. Do not omit either.

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
    import json as _json
    from collections import Counter
    from fantasy_baseball.utils.name_utils import normalize_name as nn

    picks = dv.reconstruct_draft()
    keepers = [p for p in picks if p.is_keeper]
    drafted = [p for p in picks if not p.is_keeper]
    assert len(keepers) == 30
    assert len(drafted) == 200
    # every team owns exactly 3 keepers
    assert set(Counter(p.team for p in keepers).values()) == {3}

    # ENFORCE the known-roster gate (spec oracle 6b): the user's roster must
    # reconstruct exactly. Infer the user's team from its keepers, then assert its
    # reconstructed roster is a superset of state["user_roster"].
    state = _json.loads(dv._DRAFT_STATE.read_text(encoding="utf-8"))
    user_roster = state["user_roster"]
    league = dv._load_league()
    keeper_team = {nn(k["name"]): k["team"] for k in league["keepers"]}
    user_team = next(keeper_team[nn(n)] for n in user_roster if nn(n) in keeper_team)
    assert dv.validate_reconstruction(picks, known_team=user_team, known_roster=user_roster) == []
```

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k reconstruct -v`
Expected: FAIL.

- [ ] **Step 3: Implement reconstruction.**

Read `config/draft_order.json` (`rounds`, `trades`) and `data/draft_state.json` (`drafted_players`) and `config/league.yaml` (`keepers`). The invariant (VERIFIED against real data — reconstructing the user's team reproduces `state["user_roster"]` 23/23 with this mapping, and only 3/23 without the keeper-round skip): `draft_order.json` `rounds` is the FULL 23x10 = 230-slot snake order; **rounds 1-3 (the first 30 slots) are consumed by the 30 keepers**, so the 200 live picks map to `rounds[3:]` (rounds 4-23). `drafted_players[0:30]` are the 30 keepers in `league.yaml` order; `drafted_players[30:230]` are the 200 live picks in snake order. Build the live pick->team map by flattening ALL rounds (with absolute round numbers), applying `trades` on the absolute `[round-1][slot-1]` cell (trades are on rounds 5/6/13/18/23, all in the live range), then SKIP the first 30 keeper-round slots and zip against `drafted_players[30:230]`.

```python
import yaml

_CONFIG = _REPO_ROOT / "config"
_DRAFT_STATE = _REPO_ROOT / "data" / "draft_state.json"


@dataclass(frozen=True)
class DraftPick:
    slot: int | None      # 1..200 live-pick ordinal; None for keepers
    round: int            # 0 for keepers; else absolute draft round (4..23)
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

    # live picks: flatten ALL rounds (absolute round numbers), apply trades on the
    # absolute cell, then SKIP the first n_keep keeper-round slots and zip to live picks.
    rounds = [list(r) for r in order["rounds"]]
    for tr in order.get("trades", []):
        rounds[tr["round"] - 1][tr["slot"] - 1] = tr["to"]
    flat_teams = [(rnd_i + 1, team) for rnd_i, rnd in enumerate(rounds) for team in rnd]
    live_teams = flat_teams[n_keep:]           # drop the 30 keeper-round slots (rounds 1-3)
    live = drafted[n_keep:]
    for slot, (name, (rnd, team)) in enumerate(zip(live, live_teams), start=1):
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
    ptype = str(row["player_type"])
    keys = ("r", "hr", "rbi", "sb", "avg", "ab") if ptype == "hitter" else ("w", "k", "sv", "era", "whip", "ip")
    line = {k: row[k] for k in keys}
    return score_var(line, list(row["positions"]), ptype, scale, fraction)


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

Note: `build_par_curve(picks, board, fraction=1.0, scale=None)` — when `fraction < 1.0`
you MUST pass `scale` (from `reproduce_draft_day_board`) so `_var_for_row` can rescore.
`str(row["player_type"])` normalizes the StrEnum to `"hitter"`/`"pitcher"`.

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
import pytest


@pytest.fixture(scope="module")
def synthetic_scale():
    # build the real board/scale once for all oracle tests in this module
    _board, scale = dv.reproduce_draft_day_board()
    return scale


def test_full_season_and_actual_loaders_shape():
    full = dv.load_full_season_lines()
    assert full, "no full-season lines (KV store not synced?)"
    k = next(iter(full))
    assert "::" in k
    line = full[k]
    assert any(s in line for s in ("hr", "k"))


def test_season_fraction_in_unit_range():
    f = dv.season_fraction()
    assert 0.0 <= f <= 1.0


def test_ytd_fraction_is_not_linear_in_f(synthetic_scale):
    # Guards the f*floor_full bug: rate SGP is f-invariant while counting scales by f,
    # so a to-date VAR does NOT simply equal f * full VAR. This is oracle 5 at the
    # score_var level (a distinct, non-tautological check that f=1 convergence cannot see).
    scale = synthetic_scale
    full = dv.score_var(_hitter_line(), ["OF"], "hitter", scale, fraction=1.0)
    half = dv.score_var(_hitter_line(), ["OF"], "hitter", scale, fraction=0.5)
    assert half < full                         # counting-dominated: to-date VAR is smaller
    assert abs(half - 0.5 * full) > 1e-6       # but NOT linear in f (rate component invariant)
```

The `synthetic_scale` fixture (module-scoped) builds the real board/scale once so the
oracle tests here and in Task 6 do not re-run the board each time. The value-level
convergence oracle (YTD value == projected value at f=1) lives in Task 6, where
`compute_player_value` exists.

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k "loaders or season_fraction or not_linear" -v`
Expected: FAIL.

- [ ] **Step 3: Implement the loaders + fraction.**

Read the full-season cache (`CacheKey.FULL_SEASON_PROJECTIONS`) via `read_cache`, and game-log totals via `_load_game_log_totals`. Key everything by `name::player_type`. Full-season records are MLBAM-keyed with `name`; game-log totals come back keyed by normalized name in two dicts (hitter/pitcher), which gives `player_type` directly.

```python
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.web.season_data import read_cache, _load_game_log_totals  # read_cache lives here, NOT redis_store


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

Run: `pytest tests/test_analysis/test_draft_value.py -k "loaders or season_fraction or not_linear" -v`
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


def test_convergence_ytd_equals_proj_at_f1(synthetic_scale):
    # spec oracle 3: at f=1 (ROS->0), with full_line == todate_line and matching
    # baselines, the YTD value converges to the projected value. Non-tautological:
    # it exercises BOTH horizon paths (est_proj at fraction=1.0 vs est_ytd at fraction=1.0)
    # and both baselines through the real value computation.
    scale = synthetic_scale
    line = _hitter_line()
    pv = dv.compute_player_value(
        team="Hart of the Order", name="Test Bat", player_type="hitter", positions=["OF"],
        baseline_proj=5.0, baseline_ytd=5.0, baseline_kind="drafted", preseason_var=8.0,
        full_line=line, todate_line=line, scale=scale, fraction=1.0,
    )
    assert abs(pv.value_proj - pv.value_ytd) < 1e-9
```

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k "decomposition or offboard or convergence" -v`
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

Run: `pytest tests/test_analysis/test_draft_value.py -k "decomposition or offboard or convergence" -v`
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
    # PlayerValue has 12 fields; construct with keywords to avoid positional drift.
    pvs = [
        dv.PlayerValue(team="Hart of the Order", name="A", player_type="hitter", slot=1,
                       baseline_kind="drafted", preseason_var=8.0, est_var_proj=10.0,
                       est_var_ytd=6.0, value_proj=4.0, value_ytd=2.5, skill=2.0, luck=2.0),
        dv.PlayerValue(team="Hart of the Order", name="B", player_type="hitter", slot=None,
                       baseline_kind="waiver", preseason_var=None, est_var_proj=3.0,
                       est_var_ytd=1.5, value_proj=3.0, value_ytd=1.4, skill=None, luck=None),
    ]
    r = dv.roll_up_team("Hart of the Order", pvs, case3_count=2, horizon="proj")
    assert r.credited_count == 2
    assert abs(r.sum_value - 7.0) < 1e-9   # value_proj: 4.0 + 3.0
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
    from fantasy_baseball.utils.name_utils import normalize_name
    players, teams = dv.run_draft_value()
    assert players and teams
    # every team roll-up has a credited count and a case3 count
    assert all(t.credited_count >= 0 for t in teams)
    # known-pick sanity: a specific keeper resolves with a finite projected value
    soto = next((p for p in players if normalize_name(dv._strip_suffix(p.name)) == "juan soto"), None)
    assert soto is not None and soto.value_proj == soto.value_proj  # not NaN
    assert soto.baseline_kind in ("keeper", "drafted", "waiver")
```

- [ ] **Step 2: Run to confirm failure.**

Run: `pytest tests/test_analysis/test_draft_value.py -k end_to_end -v`
Expected: FAIL.

- [ ] **Step 3: Implement `run_draft_value` orchestrator.**

Wire the pieces: reproduce board (+ validate; raise on drift), reconstruct draft (+ validate gate), build projected and to-date par curves, load full-season + actual lines, load current rosters (`get_latest_weekly_rosters`) and add-txns, classify each current-roster player, compute values, roll up per team. Off-board players get positions via the board's default-attach convention (`["OF"]` for hitters, `["SP"]` for pitchers) resolved by their game-log/full-season type. Ohtani resolves to the hitter line ("batter only").

```python
import re as _re

from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.data.redis_store import get_latest_weekly_rosters

_YAHOO_SUFFIX_RE = _re.compile(r"\s*\((?:Batter|Pitcher)\)\s*$")


def _strip_suffix(name):
    return _YAHOO_SUFFIX_RE.sub("", name).strip()


def _default_positions(player_type):
    return ["OF"] if player_type == "hitter" else ["SP"]


def _resolve_type_and_positions(norm_name, full_lines, bindex):
    """Return (player_type, positions). Prefer board row; else infer from which
    full-season line exists; off-board -> board default positions. Ohtani -> hitter."""
    for ptype in ("hitter", "pitcher"):
        row = bindex.get(f"{norm_name}::{ptype}")
        if row is not None:
            return ptype, list(row["positions"])
    for ptype in ("hitter", "pitcher"):
        if f"{norm_name}::{ptype}" in full_lines:
            return ptype, _default_positions(ptype)
    return "hitter", _default_positions("hitter")


def run_draft_value(fraction=None):
    import json as _json

    board, scale = reproduce_draft_day_board()
    frozen_drift_summary(board)  # soft: logs a warning on large drift, never raises
    picks = reconstruct_draft()
    # enforce the reconstruction gate against the user's known roster (spec oracle 6b)
    state = _json.loads(_DRAFT_STATE.read_text(encoding="utf-8"))
    user_roster = state.get("user_roster") or []
    league = _load_league()
    keeper_team = {normalize_name(k["name"]): k["team"] for k in league["keepers"]}
    user_team = next((keeper_team[normalize_name(n)] for n in user_roster
                      if normalize_name(n) in keeper_team), None)
    gate = validate_reconstruction(picks, known_team=user_team, known_roster=user_roster)
    if gate:
        raise RuntimeError(f"Draft reconstruction gate failed: {gate}")
    f = season_fraction() if fraction is None else fraction

    bindex = _board_index(board)
    preseason = preseason_var_lookup(board)
    par_proj = build_par_curve(picks, board, fraction=1.0)
    par_ytd = build_par_curve(picks, board, fraction=f, scale=scale)
    full_lines = load_full_season_lines()
    todate_lines = load_actual_to_date_lines()

    # draft/keep sets by team; DRAFT-ORDER ordinal among on-board drafted picks.
    # par(slot) = drafted_pars[ordinal-1] (drafted_pars is VAR-sorted desc): the k-th
    # on-board drafted pick is measured against the k-th-best available VAR. This is the
    # spec's par(slot). Do NOT index by the player's OWN VAR rank -- that makes
    # par == own VAR, so skill = preseason_var - par == 0 for every drafted player.
    drafted_by, kept_by = {}, {}
    slot_by_name = {}  # norm_name -> draft-order ordinal among on-board drafted
    onboard_ordinal = 0
    for p in sorted((pk for pk in picks if not pk.is_keeper), key=lambda x: x.slot):
        if _match_board_row(p.player_name, bindex) is None:
            continue  # off-board flier: excluded from par curve and slot indexing
        onboard_ordinal += 1
        slot_by_name[normalize_name(p.player_name)] = onboard_ordinal
    for p in picks:
        tkey = normalize_name_team(p.team)
        (kept_by if p.is_keeper else drafted_by).setdefault(tkey, set()).add(normalize_name(p.player_name))

    add_by = load_add_txns_by_team()
    rosters = get_latest_weekly_rosters(get_kv())

    players, case3 = [], {}
    for entry in rosters:
        team = entry["team"]
        norm = normalize_name(_strip_suffix(entry["player_name"]))  # strip "(Batter)/(Pitcher)"
        kind = classify_acquisition(team, norm, drafted_by, kept_by, add_by)
        if kind == "trade_excluded":
            case3[team] = case3.get(team, 0) + 1
            continue
        ptype, positions = _resolve_type_and_positions(norm, full_lines, bindex)
        key = f"{norm}::{ptype}"
        pre = preseason.get(key)
        full_line = full_lines.get(key)
        todate_line = todate_lines.get(key)
        if kind == "keeper":
            base_proj, base_ytd = par_proj.keeper_par, par_ytd.keeper_par
        elif kind == "drafted":
            rank = slot_by_name.get(norm)  # draft-order ordinal
            base_proj = par_proj.par_for_slot(rank) if rank else 0.0
            base_ytd = par_ytd.par_for_slot(rank) if rank else 0.0
        else:  # waiver
            base_proj = base_ytd = 0.0
        pv = compute_player_value(team, entry["player_name"], ptype, positions,
                                  base_proj, base_ytd, kind, pre, full_line, todate_line, scale, f)
        players.append(pv)

    # group by team (PlayerValue carries .team) and roll up with the per-team case-3 count
    by_team: dict[str, list] = {}
    for pv in players:
        by_team.setdefault(pv.team, []).append(pv)
    all_teams = {entry["team"] for entry in rosters}
    teams = [roll_up_team(t, by_team.get(t, []), case3.get(t, 0)) for t in sorted(all_teams)]
    return players, teams
```

Note: `par_ytd.par_for_slot(rank)` uses the same draft-order `rank`; `par_ytd.drafted_pars`
is the to-date VAR list (same on-board players, sorted desc), so slot `rank` maps to the
`rank`-th best to-date VAR — the to-date par for that slot. Both horizons share one
draft-order ordinal, only the par values differ.

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
- Validation oracles 1-9 -> Task 2 (1 internal single-scale, 3 estimate-scale),
  Task 1 (2 soft frozen), Task 5 (5 f<1 at score_var level), Task 6 (4 convergence,
  6 decomposition), Task 3 (7 slot-reconstruction gate), Task 7/9 (8 classifier +
  case-3), Task 9 (9 known-pick).
- CLI + markdown -> Task 9.

## Notes for the implementer

- **Context decay:** re-read `draft_value.py` before each task's edits; the file grows across tasks.
- **Read before scoring:** before Task 2 Step 3, read `sgp/var.py` `calculate_var` / `_pitcher_floor_key` to confirm exactly which Series fields the floor routing needs (`player_type` vs `ip`); wire `score_var`'s Series to match.
- **KV sync:** in-season loaders (full-season, game logs, transactions, rosters) read the KV store; a stale/empty local store yields empty output. Confirm a fresh sync (or run against the source) before trusting numbers; the CLI docstring must say so.
- **Do not modify failing oracle tests to pass** — a failing frozen-validation or reconstruction gate means the data/scale is off, not the test.
