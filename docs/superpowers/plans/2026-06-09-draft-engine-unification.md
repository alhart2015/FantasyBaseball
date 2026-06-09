# Draft Engine Unification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-06-09-draft-engine-unification-design.md`

**Goal:** Collapse the two parallel draft recommenders (VAR/VONA and deltaRoto) behind one `recommend()` seam with a uniform `RankedPick` row, make strategies orthogonal overlays, merge the two simulators into one, and point the league config at the verdict winner (`deltaroto_immediate`).

**Architecture:** A new `draft/recommend.py` owns the `RankedPick` dataclass, the two row adapters, and the `recommend(scoring_mode, strategy, ...)` dispatch that ranks (var/vona via `recommender.get_recommendations`, deltaroto_* via `eroto_recs.rank_candidates`), applies a strategy overlay, and slot-gates via the existing `select_from_ranked`. The dashboard, the single consolidated simulator, and `compare_strategies.py` all call `recommend()`. Each phase is guarded by golden-master parity tests so scoring never silently regresses.

**Tech Stack:** Python 3.12, pandas, pytest, Flask, ruff, mypy, vulture.

---

## File Structure

- **Create** `src/fantasy_baseball/draft/recommend.py` -- `RankedPick`, `from_recommendation`, `from_recrow`, `recommend()`, position-string serializer.
- **Modify** `src/fantasy_baseball/draft/strategy.py` -- `pick_*` become overlays on `list[RankedPick]`; `STRATEGIES` becomes the overlay registry; `rec.var` -> `rec.score`.
- **Modify** `src/fantasy_baseball/draft/recs_integration.py` -- expose its input assembly to `recommend()` (no duplication).
- **Modify** `src/fantasy_baseball/web/app.py` -- `/api/recs` calls `recommend()`; serializer preserves `immediate_delta`/`value_of_picking_now`/`per_category`.
- **Modify** `src/fantasy_baseball/config.py` -- `VALID_SCORING_MODES` gains the two deltaRoto modes.
- **Modify** `config/league.yaml` -- `scoring_mode: deltaroto_immediate`, `strategy: default` (final phase).
- **Modify** `scripts/simulate_draft.py` -- becomes the single `--scoring-mode` sim, decomposed into harness/field/reporting; routes picks through `recommend()`.
- **Delete** `scripts/sim_deltaroto.py` -- folded into `simulate_draft.py`.
- **Modify** `scripts/compare_strategies.py`, `scripts/replay_picks.py` -- iterate/route via the seam.
- **Modify** `src/fantasy_baseball/draft/CLAUDE.md` -- describe the single seam.
- **Create** `tests/test_draft/test_recommend.py`, `tests/test_draft/test_ranked_pick.py`, `tests/test_draft/test_strategy_overlays.py`, `tests/test_draft/test_parity_golden.py` -- new test modules.

**Key existing signatures (verified, do not guess):**
- `recommender.get_recommendations(board, drafted, user_roster, n=5, filled_positions=None, picks_until_next=None, roster_slots=None, num_teams=None, scoring_mode="var") -> list[Recommendation]` (`recommender.py:153`).
- `Recommendation(name, var, score, best_position, positions, player_type, need_flag=False, note="")` -- `positions: list[Position]`, `__post_init__` parses str->Position (`recommender.py:33`).
- `eroto_recs.rank_candidates(*, candidates, replacements, team_name, projected_standings, team_sds, picks_until_next_turn=0, adp_table=None, user_rp_filled=0) -> list[RecRow]` (`eroto_recs.py:99`).
- `RecRow(player_id, name, positions: list[str], immediate_delta, value_of_picking_now, per_category)` (`eroto_recs.py:88`).
- `recs_integration.compute_rec_inputs(state, board_path, league_yaml) -> RecInputs` with fields `candidates, replacements, projected_standings, team_sds, adp_table, rp_filled_by_team` (`recs_integration.py:274`).
- `strategy.select_from_ranked(ranked, open_starters, pick_rank)` -- duck-types on `.positions` (`strategy.py:83`).
- Imports: `from fantasy_baseball.models.positions import Position`; `from fantasy_baseball.models.player import PlayerType`.

---

## Phase 0: Step-0 cleanup

### Task 0: Dead-code sweep on touched modules

**Files:**
- Modify (as needed): `src/fantasy_baseball/draft/recommender.py`, `eroto_recs.py`, `recs_integration.py`, `strategy.py`

- [ ] **Step 1: Find dead code in the modules this plan touches**

Run:
```bash
ruff check --select F,I src/fantasy_baseball/draft/recommender.py src/fantasy_baseball/draft/eroto_recs.py src/fantasy_baseball/draft/recs_integration.py src/fantasy_baseball/draft/strategy.py
vulture src/fantasy_baseball/draft/recommender.py src/fantasy_baseball/draft/eroto_recs.py src/fantasy_baseball/draft/strategy.py
```
Expected: a list of unused imports / unreferenced helpers, if any.

- [ ] **Step 2: Remove only what the tools flag as unused in these files**

Delete unused imports and unreferenced private helpers the tools report. Do NOT touch anything still referenced. If the tools report nothing, record that and skip to commit.

- [ ] **Step 3: Verify the suite still passes**

Run: `pytest tests/test_draft -q`
Expected: PASS (no behavior change).

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "chore(draft): step-0 dead-code sweep before engine unification"
```

---

## Phase 1: RankedPick + adapters (no behavior change)

### Task 1: `RankedPick` dataclass

**Files:**
- Create: `src/fantasy_baseball/draft/recommend.py`
- Test: `tests/test_draft/test_ranked_pick.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft/test_ranked_pick.py
from fantasy_baseball.draft.recommend import RankedPick
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.player import PlayerType


def test_ranked_pick_holds_core_fields_and_defaults():
    rp = RankedPick(
        player_id="123",
        name="Test Player",
        positions=[Position.SS, Position.OF],
        player_type=PlayerType.HITTER,
        score=4.2,
    )
    assert rp.score == 4.2
    assert rp.metrics == {}
    assert rp.per_category == {}
    assert rp.note == ""
    assert rp.need_flag is False


def test_position_strings_serializes_enum_values():
    rp = RankedPick(
        player_id="1", name="P", positions=[Position.SS, Position.OF],
        player_type=PlayerType.HITTER, score=0.0,
    )
    assert rp.position_strings() == ["SS", "OF"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_ranked_pick.py -v`
Expected: FAIL with `ModuleNotFoundError: ... draft.recommend`.

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/draft/recommend.py
from __future__ import annotations

from dataclasses import dataclass, field

from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import Position


@dataclass
class RankedPick:
    """One ranked draft candidate, uniform across every scoring mode.

    ``score`` is the active mode's primary metric. ``metrics`` carries every
    mode-native metric (deltaRoto modes populate both ``immediate_delta`` and
    ``value_of_picking_now`` so the dashboard can toggle between them).
    """

    player_id: str
    name: str
    positions: list[Position]
    player_type: PlayerType
    score: float
    metrics: dict[str, float] = field(default_factory=dict)
    per_category: dict[str, float] = field(default_factory=dict)
    note: str = ""
    need_flag: bool = False

    def position_strings(self) -> list[str]:
        """Position codes as plain strings (for JSON / display)."""
        return [p.value if isinstance(p, Position) else str(p) for p in self.positions]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_ranked_pick.py -v`
Expected: PASS. If `Position.SS.value` is not `"SS"`, read `models/positions.py` for the actual enum value spelling and fix the test's expected strings to match the real codes (do not change the enum).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/recommend.py tests/test_draft/test_ranked_pick.py
git commit -m "feat(draft): add RankedPick uniform recommendation row"
```

### Task 2: `from_recommendation` adapter

**Files:**
- Modify: `src/fantasy_baseball/draft/recommend.py`
- Test: `tests/test_draft/test_ranked_pick.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_draft/test_ranked_pick.py
from fantasy_baseball.draft.recommend import from_recommendation
from fantasy_baseball.draft.recommender import Recommendation


def test_from_recommendation_maps_var_to_score():
    rec = Recommendation(
        name="Slugger",
        var=6.5,
        score=6.5,
        best_position="OF",
        positions=["OF"],
        player_type=PlayerType.HITTER,
        need_flag=True,
        note="need OF",
    )
    rp = from_recommendation(rec, player_id="999")
    assert rp.score == 6.5
    assert rp.metrics == {"var": 6.5}
    assert rp.name == "Slugger"
    assert rp.need_flag is True
    assert rp.note == "need OF"
    assert rp.position_strings() == ["OF"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_ranked_pick.py::test_from_recommendation_maps_var_to_score -v`
Expected: FAIL with `ImportError: cannot import name 'from_recommendation'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/fantasy_baseball/draft/recommend.py
from fantasy_baseball.draft.recommender import Recommendation


def from_recommendation(rec: Recommendation, *, player_id: str) -> RankedPick:
    """Adapt a VAR/VONA ``Recommendation`` into a ``RankedPick``.

    ``Recommendation`` carries no player_id, so callers pass it in (the
    board lookup already has it).
    """
    return RankedPick(
        player_id=player_id,
        name=rec.name,
        positions=list(rec.positions),
        player_type=rec.player_type,
        score=rec.var,
        metrics={"var": rec.var},
        note=rec.note,
        need_flag=rec.need_flag,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_ranked_pick.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/recommend.py tests/test_draft/test_ranked_pick.py
git commit -m "feat(draft): from_recommendation adapter -> RankedPick"
```

### Task 3: `from_recrow` adapter

**Files:**
- Modify: `src/fantasy_baseball/draft/recommend.py`
- Test: `tests/test_draft/test_ranked_pick.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_draft/test_ranked_pick.py
import pytest
from fantasy_baseball.draft.recommend import from_recrow
from fantasy_baseball.draft.eroto_recs import RecRow


def _recrow():
    return RecRow(
        player_id="42",
        name="Closer",
        positions=["RP"],
        immediate_delta=3.1,
        value_of_picking_now=2.4,
        per_category={"SV": 1.5, "ERA": 0.6},
    )


def test_from_recrow_immediate_metric_is_score():
    rp = from_recrow(_recrow(), metric="immediate_delta", player_type=PlayerType.PITCHER)
    assert rp.score == 3.1
    assert rp.metrics == {"immediate_delta": 3.1, "value_of_picking_now": 2.4}
    assert rp.per_category == {"SV": 1.5, "ERA": 0.6}
    assert rp.position_strings() == ["RP"]


def test_from_recrow_vopn_metric_is_score():
    rp = from_recrow(_recrow(), metric="value_of_picking_now", player_type=PlayerType.PITCHER)
    assert rp.score == 2.4


def test_from_recrow_rejects_unknown_metric():
    with pytest.raises(ValueError):
        from_recrow(_recrow(), metric="bogus", player_type=PlayerType.PITCHER)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_ranked_pick.py -k from_recrow -v`
Expected: FAIL with `ImportError: cannot import name 'from_recrow'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/fantasy_baseball/draft/recommend.py
from fantasy_baseball.draft.eroto_recs import RecRow

_DELTAROTO_METRICS = ("immediate_delta", "value_of_picking_now")


def from_recrow(row: RecRow, *, metric: str, player_type: PlayerType) -> RankedPick:
    """Adapt a deltaRoto ``RecRow`` into a ``RankedPick``.

    ``metric`` selects which native metric becomes ``score``; both are kept
    in ``metrics`` so the dashboard can display/toggle both. ``RecRow``
    carries position strings; ``__post_init__``-free RankedPick keeps them as
    parsed ``Position`` enums for overlay slot logic.
    """
    if metric not in _DELTAROTO_METRICS:
        raise ValueError(f"metric must be one of {_DELTAROTO_METRICS}, got {metric!r}")
    metrics = {
        "immediate_delta": row.immediate_delta,
        "value_of_picking_now": row.value_of_picking_now,
    }
    return RankedPick(
        player_id=row.player_id,
        name=row.name,
        positions=[Position.parse(p) for p in row.positions],
        player_type=player_type,
        score=metrics[metric],
        metrics=metrics,
        per_category=dict(row.per_category),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_ranked_pick.py -k from_recrow -v`
Expected: PASS. (If `Position.parse` is not the parse entry point, read `models/positions.py` and use the real constructor.)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/recommend.py tests/test_draft/test_ranked_pick.py
git commit -m "feat(draft): from_recrow adapter -> RankedPick"
```

---

## Phase 2: The recommend() seam + dashboard rewire

### Task 4: Capture the `/api/recs` golden master

**Files:**
- Create: `tests/test_draft/test_parity_golden.py`
- Create: `tests/test_draft/fixtures/recs_golden_state.json` (a small fixed draft state)

- [ ] **Step 1: Write a test that records and re-asserts the current `/api/recs` output**

```python
# tests/test_draft/test_parity_golden.py
"""Golden-master parity guard.

Pins the pre-refactor /api/recs payload so every phase proves the deltaRoto
path through recommend() reproduces it byte-for-byte. Run with team_sds active
(the production path) per the standing meta-lesson that variance-free scoring
flips verdicts.
"""
import json
from pathlib import Path

from fantasy_baseball.web.app import create_app

GOLDEN = Path(__file__).parent / "fixtures" / "recs_golden.json"
STATE = Path(__file__).parent / "fixtures" / "recs_golden_state.json"


def _get_recs(tmp_path):
    app = create_app(state_path=STATE)
    client = app.test_client()
    resp = client.get("/api/recs?team=Hart%20of%20the%20Order")
    assert resp.status_code == 200, resp.get_data(as_text=True)
    return resp.get_json()


def test_recs_match_golden(tmp_path):
    rows = _get_recs(tmp_path)
    if not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(rows, indent=2, sort_keys=True))
    expected = json.loads(GOLDEN.read_text())
    assert rows == expected
```

- [ ] **Step 2: Create the fixture state**

Build `tests/test_draft/fixtures/recs_golden_state.json` from a real seeded draft: run `python scripts/run_draft_dashboard.py --rebuild-board` once, POST a `/api/new-draft` then a handful of `/api/pick`s (or copy an existing `data/draft_state.json` after a few picks), and save the resulting state JSON to the fixture path. The board file must be reachable by `create_app`; point the fixture's sibling `_board.json` alongside it or rely on the default board path. Document in the test file's docstring exactly how the fixture was generated.

- [ ] **Step 3: Run to generate + assert the golden**

Run: `pytest tests/test_draft/test_parity_golden.py -v`
Expected: PASS (first run writes `recs_golden.json`, asserts equal to itself). Inspect the written `recs_golden.json` and confirm rows contain `player_id, name, positions, immediate_delta, value_of_picking_now, per_category`.

- [ ] **Step 4: Commit**

```bash
git add tests/test_draft/test_parity_golden.py tests/test_draft/fixtures/
git commit -m "test(draft): golden-master parity guard for /api/recs"
```

### Task 4b: Capture the simulator golden master (var + deltaRoto pick sequences)

**Files:**
- Create: `tests/test_draft/test_parity_sim_golden.py`
- Create: `tests/test_draft/fixtures/sim_golden_var.json`, `tests/test_draft/fixtures/sim_golden_deltaroto.json`

Spec sec 6 requires a fixed-seed simulator golden run WITH `team_sds`, so the
var/vona path through `recommend()` can be proven to reproduce pre-refactor
picks exactly. `/api/recs` only exercises `deltaroto_immediate`+`default`, so
this is the ONLY guard for the strategy-overlay (P3) and sim-consolidation (P4)
changes. The two modes currently live in two entry points (`simulate_draft.py`
for var/vona, `sim_deltaroto.py` for deltaRoto); capture each from its current
entry now, then re-point at the consolidated entry in Task 14.

- [ ] **Step 1: Find the deterministic knob in each sim**

Read `simulate_draft.py` and `sim_deltaroto.py` for a seed / `strategy_noise`
control. Confirm `seed=7` + `strategy_noise=0.0` yields a deterministic draft
(the user's own pick has no randomness; only the field uses noise). If a sim
lacks a seed, thread one into its field RNG (small test-facing change) before
capturing.

- [ ] **Step 2: Provide a deterministic user-pick-sequence wrapper in each sim**

If absent, add to each sim a thin function returning the user team's drafted
`player_id`s in pick order (a list of strings -- JSON-stable) for one draft:

```python
# simulate_draft.py
def run_user_pick_sequence(*, scoring_mode, strategy, seed, strategy_noise=0.0):
    """Run one deterministic draft; return the user team's [player_id, ...]."""
    result = simulate_one_draft(  # or the real entry found in Step 1
        config=_default_config(), scoring_mode=scoring_mode, strategy=strategy,
        seed=seed, strategy_noise=strategy_noise,
    )
    return [p["player_id"] for p in result.user_picks]
```

Add the analogous wrapper to `sim_deltaroto.py` (its deltaRoto entry). Both must
run the PRE-refactor pick path -- this task lands before Task 5 changes anything.

- [ ] **Step 3: Write the golden capture+assert test**

```python
# tests/test_draft/test_parity_sim_golden.py
"""Pre-refactor simulator golden. Pins the user team's pick sequence for a
fixed seed WITH team_sds so P3/P4 prove the seam reproduces pre-refactor picks.
Generated against simulate_draft.py (var) + sim_deltaroto.py (deltaRoto) before
Task 5; re-pointed at the consolidated sim in Task 14."""
import json
from pathlib import Path

import pytest

from scripts.simulate_draft import run_user_pick_sequence as varvona_seq

FIX = Path(__file__).parent / "fixtures"


def _assert_golden(seq, name):
    g = FIX / name
    if not g.exists():
        g.write_text(json.dumps(seq, indent=2))
    assert seq == json.loads(g.read_text())


def test_sim_var_picks_match_golden():
    _assert_golden(varvona_seq(scoring_mode="var", strategy="default", seed=7), "sim_golden_var.json")


def test_sim_deltaroto_picks_match_golden():
    from scripts.sim_deltaroto import run_user_pick_sequence as dr_seq
    _assert_golden(dr_seq(scoring_mode="deltaroto_immediate", strategy="default", seed=7),
                   "sim_golden_deltaroto.json")
```

- [ ] **Step 4: Generate + assert**

Run: `pytest tests/test_draft/test_parity_sim_golden.py -v`
Expected: PASS (first run writes both goldens). Inspect each JSON: a list of
`player_id` strings of length = the user team's pick count. Confirm a second run
is byte-identical (determinism holds).

- [ ] **Step 5: Commit**

```bash
git add tests/test_draft/test_parity_sim_golden.py tests/test_draft/fixtures/sim_golden_*.json scripts/simulate_draft.py scripts/sim_deltaroto.py
git commit -m "test(draft): pre-refactor simulator golden (var + deltaRoto pick sequences)"
```

### Task 5: `RecommendContext` + deltaRoto ranking

**Files:**
- Modify: `src/fantasy_baseball/draft/recommend.py`
- Test: `tests/test_draft/test_recommend.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft/test_recommend.py
import pytest

from fantasy_baseball.draft.recommend import RecommendContext, rank_for_mode


def test_rank_for_mode_deltaroto_immediate_returns_ranked_picks(deltaroto_ctx):
    picks = rank_for_mode(deltaroto_ctx(scoring_mode="deltaroto_immediate"))
    assert picks, "expected at least one ranked pick"
    assert picks[0].score == picks[0].metrics["immediate_delta"]
    scores = [p.metrics["immediate_delta"] for p in picks]
    assert scores == sorted(scores, reverse=True)


def test_rank_for_mode_vopn_sorts_by_vopn(deltaroto_ctx):
    picks = rank_for_mode(deltaroto_ctx(scoring_mode="deltaroto_vopn"))
    assert picks[0].score == picks[0].metrics["value_of_picking_now"]
    vopn = [p.metrics["value_of_picking_now"] for p in picks]
    assert vopn == sorted(vopn, reverse=True)


def test_rank_for_mode_deltaroto_requires_inputs():
    ctx = RecommendContext(
        scoring_mode="deltaroto_immediate", team_name="X", picks_until_next=8, inputs=None
    )
    with pytest.raises(ValueError, match="requires inputs"):
        rank_for_mode(ctx)
```

Add a `deltaroto_ctx` fixture factory to `tests/test_draft/conftest.py` that builds a `RecommendContext` from the golden fixture state: `inputs = recs_integration.compute_rec_inputs(STATE_DICT, BOARD_PATH, LEAGUE_YAML)`, then returns `lambda *, scoring_mode: RecommendContext(scoring_mode=scoring_mode, team_name="Hart of the Order", picks_until_next=8, inputs=inputs)`. Reuse the `STATE`/board path from Task 4.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_recommend.py -v`
Expected: FAIL with `ImportError: cannot import name 'rank_for_mode'`.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/fantasy_baseball/draft/recommend.py
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import pandas as pd

from fantasy_baseball.draft import eroto_recs

if TYPE_CHECKING:
    from fantasy_baseball.draft.recs_integration import RecInputs

_DELTAROTO_MODES = {
    "deltaroto_immediate": "immediate_delta",
    "deltaroto_vopn": "value_of_picking_now",
}
_VARVONA_MODES = ("var", "vona")


@dataclass
class RecommendContext:
    """Everything either ranker needs for one pick.

    deltaRoto modes use ``inputs`` (a ``RecInputs``); var/vona modes use the
    pandas ``board`` + ``drafted`` + ``filled_positions`` + ``config``. The
    caller fills whichever the active mode requires; ``rank_for_mode`` validates.
    """

    scoring_mode: str
    team_name: str
    picks_until_next: int
    inputs: "RecInputs | None" = None
    board: pd.DataFrame | None = None
    drafted: list[str] = field(default_factory=list)
    filled_positions: dict[str, int] | None = None
    config: Any = None


def _rank_deltaroto(ctx: RecommendContext) -> list[RankedPick]:
    if ctx.inputs is None:
        raise ValueError(f"scoring_mode {ctx.scoring_mode!r} requires inputs (RecInputs)")
    metric = _DELTAROTO_MODES[ctx.scoring_mode]
    rows = eroto_recs.rank_candidates(
        candidates=ctx.inputs.candidates,
        replacements=ctx.inputs.replacements,
        team_name=ctx.team_name,
        projected_standings=ctx.inputs.projected_standings,
        team_sds=ctx.inputs.team_sds,
        picks_until_next_turn=ctx.picks_until_next,
        adp_table=ctx.inputs.adp_table,
        user_rp_filled=ctx.inputs.rp_filled_by_team.get(ctx.team_name, 0),
    )
    type_by_id = {c.yahoo_id: c.player_type for c in ctx.inputs.candidates}
    picks: list[RankedPick] = []
    for r in rows:
        pt = type_by_id.get(r.player_id)
        if pt is None:
            # Fail loud rather than mislabel a pitcher as a hitter in overlays.
            raise KeyError(f"candidate id {r.player_id!r} ({r.name}) absent from board candidates")
        picks.append(from_recrow(r, metric=metric, player_type=pt))
    if metric == "value_of_picking_now":
        picks.sort(key=lambda p: p.score, reverse=True)
    return picks


def rank_for_mode(ctx: RecommendContext) -> list[RankedPick]:
    """Single dispatcher: rank the candidate pool for ``ctx.scoring_mode``."""
    if ctx.scoring_mode in _DELTAROTO_MODES:
        return _rank_deltaroto(ctx)
    raise ValueError(f"unknown scoring_mode {ctx.scoring_mode!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_recommend.py -v`
Expected: PASS. Confirm `Player.yahoo_id` / `Player.player_type` are attributes (read `models/player.py`); if `player_type` is a method, call it.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/recommend.py tests/test_draft/test_recommend.py tests/test_draft/conftest.py
git commit -m "feat(draft): RecommendContext + deltaRoto ranking dispatch"
```

### Task 6: Add var/vona to the `rank_for_mode` dispatcher

**Files:**
- Modify: `src/fantasy_baseball/draft/recommend.py`
- Test: `tests/test_draft/test_recommend.py`, `tests/test_draft/conftest.py`

This makes `rank_for_mode` serve ALL FOUR modes through one signature -- the
core of the spec's "one shared seam". var/vona use the same `RecommendContext`,
just populating `board`/`drafted`/`filled_positions`/`config` instead of `inputs`.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_draft/test_recommend.py
@pytest.mark.parametrize("mode", ["var", "vona"])
def test_rank_for_mode_var_vona_scores_present(varvona_ctx, mode):
    picks = rank_for_mode(varvona_ctx(scoring_mode=mode))
    assert picks
    assert picks[0].metrics[mode] == picks[0].score


def test_rank_for_mode_var_vona_requires_board():
    ctx = RecommendContext(scoring_mode="var", team_name="X", picks_until_next=8, board=None)
    with pytest.raises(ValueError, match="requires board"):
        rank_for_mode(ctx)


def test_rank_for_mode_rejects_unknown_mode(varvona_ctx):
    with pytest.raises(ValueError, match="unknown scoring_mode"):
        rank_for_mode(varvona_ctx(scoring_mode="nope"))
```

Add a `varvona_ctx` fixture factory to `conftest.py` returning `lambda *, scoring_mode: RecommendContext(scoring_mode=scoring_mode, team_name="Hart of the Order", picks_until_next=8, board=pd.DataFrame(recs_integration.load_board_rows(BOARD_PATH)), drafted=<drafted ids from golden state>, filled_positions=<filled for the team>, config=<real LeagueConfig>)`.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_recommend.py -k "var_vona or unknown or requires_board" -v`
Expected: FAIL -- var/vona currently hits the `unknown scoring_mode` raise.

- [ ] **Step 3: Write minimal implementation**

```python
# add to src/fantasy_baseball/draft/recommend.py
from fantasy_baseball.draft.recommender import get_recommendations


def _rank_var_vona(ctx: RecommendContext) -> list[RankedPick]:
    if ctx.board is None:
        raise ValueError(f"scoring_mode {ctx.scoring_mode!r} requires board (DataFrame)")
    recs = get_recommendations(
        ctx.board,
        drafted=ctx.drafted,
        user_roster=[],
        n=15,
        filled_positions=ctx.filled_positions,
        picks_until_next=ctx.picks_until_next,
        roster_slots=ctx.config.roster_slots,
        num_teams=ctx.config.num_teams,
        scoring_mode=ctx.scoring_mode,
    )
    id_by_name = dict(zip(ctx.board["name"], ctx.board["player_id"], strict=False))
    out: list[RankedPick] = []
    for rec in recs:
        rp = from_recommendation(rec, player_id=str(id_by_name.get(rec.name, rec.name)))
        if ctx.scoring_mode == "vona":
            vona = rec.score if rec.score is not None else rec.var
            rp.metrics = {"vona": vona}
            rp.score = vona
        out.append(rp)
    return out
```

Then extend the dispatcher (replace the body of `rank_for_mode` from Task 5):

```python
def rank_for_mode(ctx: RecommendContext) -> list[RankedPick]:
    """Single dispatcher: rank the candidate pool for ``ctx.scoring_mode``."""
    if ctx.scoring_mode in _DELTAROTO_MODES:
        return _rank_deltaroto(ctx)
    if ctx.scoring_mode in _VARVONA_MODES:
        return _rank_var_vona(ctx)
    raise ValueError(f"unknown scoring_mode {ctx.scoring_mode!r}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_recommend.py -k "var_vona or unknown or requires" -v`
Expected: PASS. If vona's metric lives on `Recommendation.score` vs `.var`, read `recommender.py:210-247` and map the correct field.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/recommend.py tests/test_draft/test_recommend.py tests/test_draft/conftest.py
git commit -m "feat(draft): var/vona branch -- rank_for_mode serves all four modes"
```

### Task 7: Rewire `/api/recs` through the seam (golden must hold)

**Files:**
- Modify: `src/fantasy_baseball/web/app.py:369-393`
- Modify: `src/fantasy_baseball/draft/recommend.py` (add `to_recs_json`)
- Test: `tests/test_draft/test_parity_golden.py` (unchanged -- must still pass)

- [ ] **Step 1: Add the serializer + a test for it**

```python
# append to tests/test_draft/test_ranked_pick.py
def test_to_recs_json_preserves_dashboard_keys():
    from fantasy_baseball.draft.recommend import to_recs_json
    rp = RankedPick(
        player_id="42", name="Closer", positions=[Position.RP],
        player_type=PlayerType.PITCHER, score=3.1,
        metrics={"immediate_delta": 3.1, "value_of_picking_now": 2.4},
        per_category={"SV": 1.5},
    )
    d = to_recs_json(rp)
    assert d == {
        "player_id": "42",
        "name": "Closer",
        "positions": ["RP"],
        "immediate_delta": 3.1,
        "value_of_picking_now": 2.4,
        "per_category": {"SV": 1.5},
    }
```

```python
# add to src/fantasy_baseball/draft/recommend.py
def to_recs_json(pick: RankedPick) -> dict:
    """Serialize a deltaRoto-mode RankedPick into the exact /api/recs shape
    the dashboard JS expects (immediate_delta + value_of_picking_now top-level)."""
    return {
        "player_id": pick.player_id,
        "name": pick.name,
        "positions": pick.position_strings(),
        "immediate_delta": pick.metrics["immediate_delta"],
        "value_of_picking_now": pick.metrics["value_of_picking_now"],
        "per_category": pick.per_category,
    }
```

- [ ] **Step 2: Run the serializer test (red, then green)**

Run: `pytest tests/test_draft/test_ranked_pick.py::test_to_recs_json_preserves_dashboard_keys -v`
Expected: PASS after adding `to_recs_json`.

- [ ] **Step 3: Rewire the endpoint**

Replace the body of `recs()` in `web/app.py:369-393` so it builds picks via the seam and serializes with `to_recs_json`:

```python
    @app.get("/api/recs")
    def recs():
        from dataclasses import replace

        from fantasy_baseball.draft.recommend import RecommendContext, rank_for_mode, to_recs_json

        team = request.args.get("team")
        if not team:
            return jsonify({"error": "missing team parameter"}), 400
        league_yaml = _load_league_yaml()
        state = draft_controller.resume_or_init(app.config[CFG_STATE_PATH])
        try:
            inputs = _build_rec_inputs(app, state, league_yaml)
        except RuntimeError as e:
            return jsonify({"error": str(e)}), 503
        picks_until_next = _picks_until_next_turn(state, team, league_yaml)
        # The live draft always serves a deltaRoto mode (it owns both metrics
        # the dashboard toggles between); immediate is the verdict winner.
        # replace() makes a non-mutating shallow copy so the app-cached inputs
        # are not clobbered by the candidate-pool slice.
        ctx = RecommendContext(
            scoring_mode="deltaroto_immediate",
            team_name=team,
            picks_until_next=picks_until_next,
            inputs=replace(inputs, candidates=inputs.candidates[:RECS_CANDIDATE_POOL_SIZE]),
        )
        picks = rank_for_mode(ctx)
        return jsonify([to_recs_json(p) for p in picks[:10]])
```

Note: `RecInputs` must be a dataclass for `replace()` to work (it is). The slice
must happen before ranking so the O(N*score_roto) pass stays bounded by
`RECS_CANDIDATE_POOL_SIZE`.

- [ ] **Step 4: Run the golden parity test**

Run: `pytest tests/test_draft/test_parity_golden.py -v`
Expected: PASS -- the payload through the seam equals the pre-refactor golden. If positions serialize differently (enum value vs raw string), fix `position_strings()`/`Position.parse` round-trip until equal. Do NOT regenerate the golden to make it pass.

- [ ] **Step 5: Run the full draft + web suite**

Run: `pytest tests/test_draft tests/test_web -q`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/draft/recommend.py src/fantasy_baseball/web/app.py tests/test_draft/test_ranked_pick.py
git commit -m "refactor(draft-web): /api/recs serves through recommend() seam"
```

---

## Phase 3: Strategies as orthogonal overlays

### Task 8: Define the overlay protocol + port `default`

**Files:**
- Modify: `src/fantasy_baseball/draft/strategy.py`
- Modify: `src/fantasy_baseball/draft/recommend.py` (add `recommend()` that composes rank + overlay + select)
- Test: `tests/test_draft/test_strategy_overlays.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_draft/test_strategy_overlays.py
from fantasy_baseball.draft.recommend import RankedPick
from fantasy_baseball.draft.strategy import OVERLAYS
from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.player import PlayerType


def _pick(name, score, pos=Position.OF):
    return RankedPick(player_id=name, name=name, positions=[pos],
                      player_type=PlayerType.HITTER, score=score,
                      metrics={"immediate_delta": score})


def test_default_overlay_defers_to_slot_gate():
    # default applies NO constraint -- it returns None so recommend()'s
    # select_from_ranked makes the position-aware greedy pick (verdict winner).
    ranked = [_pick("A", 5.0), _pick("B", 3.0)]
    assert OVERLAYS["default"](ranked, roster_state=None, config=None) is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_strategy_overlays.py -v`
Expected: FAIL with `ImportError: cannot import name 'OVERLAYS'`.

- [ ] **Step 3: Implement the overlay registry with `default`**

Add to `strategy.py` an `OVERLAYS` dict whose values have signature
`overlay(ranked: list[RankedPick], *, roster_state, config, **kwargs) -> RankedPick | None`.
`default` applies no constraint and DEFERS (returns `None`), so `recommend()`
falls through to `select_from_ranked` -- the position-aware greedy selection that
reproduces the pre-refactor `pick_default` / `_choose_rec` behavior and is the
verdict winner. (An overlay returns a `RankedPick` only when it actively
overrides the slot-gated greedy choice.)

```python
# strategy.py
def overlay_default(ranked, *, roster_state=None, config=None, **kwargs):
    """No-constraint overlay: defer to recommend()'s slot-gated selection."""
    return None


OVERLAYS = {
    "default": overlay_default,
}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_strategy_overlays.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/strategy.py tests/test_draft/test_strategy_overlays.py
git commit -m "feat(draft): overlay registry with default defer overlay"
```

### Task 9: `recommend()` composes rank + overlay + slot-gate

**Files:**
- Modify: `src/fantasy_baseball/draft/recommend.py`
- Test: `tests/test_draft/test_recommend.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_draft/test_recommend.py
from fantasy_baseball.draft.recommend import recommend


def test_recommend_deltaroto_default_picks_top_immediate(deltaroto_ctx):
    chosen = recommend(
        deltaroto_ctx(scoring_mode="deltaroto_immediate"),
        strategy="default",
        open_starters=set(),
    )
    assert chosen is not None
    assert chosen.score == chosen.metrics["immediate_delta"]


@pytest.mark.parametrize("mode", ["var", "vona"])
def test_recommend_var_vona_runs_through_same_seam(varvona_ctx, mode):
    # Proves recommend() serves all four modes through one entry (spec sec 2).
    chosen = recommend(varvona_ctx(scoring_mode=mode), strategy="default", open_starters=set())
    assert chosen is not None
    assert chosen.score == chosen.metrics[mode]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_draft/test_recommend.py::test_recommend_deltaroto_default_picks_top_immediate -v`
Expected: FAIL with `ImportError: cannot import name 'recommend'`.

- [ ] **Step 3: Implement `recommend()`**

```python
# add to src/fantasy_baseball/draft/recommend.py
from fantasy_baseball.draft.strategy import OVERLAYS, select_from_ranked


def recommend(
    ctx: RecommendContext,
    *,
    strategy: str,
    open_starters: set,
    roster_state=None,
    pick_rank: int = 0,
) -> RankedPick | None:
    """Rank for ctx.scoring_mode, apply the strategy overlay, slot-gate.

    Serves all four modes because rank_for_mode(ctx) does the dispatch; the
    overlay and slot-gate are mode-agnostic (they consume RankedPick)."""
    ranked = rank_for_mode(ctx)
    if strategy not in OVERLAYS:
        raise ValueError(f"unknown strategy {strategy!r}; valid: {sorted(OVERLAYS)}")
    chosen = OVERLAYS[strategy](ranked, roster_state=roster_state, config=ctx.config)
    if chosen is not None:
        return chosen
    # Overlay deferred -> plain slot-gated selection.
    return select_from_ranked(ranked, open_starters, pick_rank)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_draft/test_recommend.py::test_recommend_deltaroto_default_picks_top_immediate -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/recommend.py tests/test_draft/test_recommend.py
git commit -m "feat(draft): recommend() composes rank + overlay + slot-gate"
```

### Task 10: Port the closer family overlays (`two_closers`, `three_closers`, `four_closers`, `nonzero_sv`)

**Files:**
- Modify: `src/fantasy_baseball/draft/strategy.py`
- Test: `tests/test_draft/test_strategy_overlays.py`

- [ ] **Step 1: Read the existing closer strategies**

Read `strategy.py` `pick_two_closers`/`pick_n_closers` (`strategy.py:285`+) and `pick_nonzero_sv` (`:146`). Identify what each reads: round number (`tracker.current_round` / `pick_rank`), candidate save projection (`per_category["SV"]` or board `sv` column), and the closer-timing thresholds. These are the constraints to reimplement against `RankedPick`.

- [ ] **Step 2: Write the failing test (one representative behavior per overlay)**

```python
# append to tests/test_draft/test_strategy_overlays.py
def _closer(name, score, sv):
    return RankedPick(player_id=name, name=name, positions=[Position.RP],
                      player_type=PlayerType.PITCHER, score=score,
                      metrics={"immediate_delta": score}, per_category={"SV": sv})


def test_nonzero_sv_skips_zero_save_relievers_for_closer_slot():
    ranked = [_closer("MiddleReliever", 9.0, 0.0), _closer("Closer", 4.0, 30.0)]
    chosen = OVERLAYS["nonzero_sv"](ranked, roster_state=None, config=None,
                                    closer_slots_open=1)
    assert chosen.name == "Closer"
```

(Write one analogous test pinning the round-gating behavior for `two_closers`: e.g. before the configured closer round it defers, at/after it forces the best save-projected reliever. Use the thresholds read in Step 1.)

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_draft/test_strategy_overlays.py -k closer -v`
Expected: FAIL with `KeyError: 'nonzero_sv'`.

- [ ] **Step 4: Implement the closer overlays against RankedPick**

Reimplement each closer strategy as an overlay that filters/orders `ranked` by `per_category["SV"]` and the round gate, returning the chosen `RankedPick` or `None` to defer. Register them in `OVERLAYS`. Reuse a shared `_save_projection(pick)` helper reading `pick.per_category.get("SV", 0.0)` (per the CLAUDE.md numeric-default rule, NOT `or 0`).

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/test_draft/test_strategy_overlays.py -k closer -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/draft/strategy.py tests/test_draft/test_strategy_overlays.py
git commit -m "feat(draft): closer-family strategies as RankedPick overlays"
```

### Task 11: Port the no-punt + AVG overlays (`no_punt`, `no_punt_opp`, `no_punt_stagger`, `no_punt_cap3`, `avg_hedge`, `avg_anchor`, `closers_avg`, `balanced`, `anti_fragile`)

**Files:**
- Modify: `src/fantasy_baseball/draft/strategy.py`
- Test: `tests/test_draft/test_strategy_overlays.py`

- [ ] **Step 1: Read each remaining strategy and list the board fields it reads**

For each `pick_*` in `STRATEGIES` not yet ported, read its body and record which signals it needs: category gaps (leverage), AVG floor, opponent modeling. Anything sourced from a pandas board column must come from `RankedPick.per_category` / `metrics` instead. Record any signal NOT available on `RankedPick` -- that is the trigger for the spec's "overlay-where-cheap" fallback.

- [ ] **Step 2: Write one failing behavioral test per overlay**

Write one focused test per strategy on a synthetic `list[RankedPick]`, pinning the single constraint that distinguishes it from `default`. Confirm the exact threshold/round numbers against the source read in Step 1 (the table states the behavior, not the magic numbers):

| Strategy | Defining behavior to pin |
| --- | --- |
| `no_punt` | Never lets any 5x5 category's projected team total fall below the punt floor: given a candidate that would zero a still-thin category vs one that doesn't, picks the protecting one even at lower score. |
| `no_punt_opp` | Same protection but keyed off the largest *opponent-relative* category gap (leverage), not the absolute total. |
| `no_punt_stagger` | Spreads category fills across rounds: will defer a second pick in an already-addressed category when another category is unprotected. |
| `no_punt_cap3` | Like `no_punt` but stops protecting a category once 3 contributors are rostered (caps over-investment). |
| `avg_hedge` | Once AVG-risk threshold is crossed, prefers the higher-AVG of two near-equal-score hitters. |
| `avg_anchor` | Early (before the configured round) forces a high-AVG anchor bat over a higher-score low-AVG bat. |
| `closers_avg` | Combines the closer-round gate (from Task 10) with the AVG floor: takes a closer on closer rounds, else applies the AVG hedge. |
| `balanced` | Maximizes the minimum across category contributions (picks the candidate that lifts the weakest category) rather than raw score. |
| `anti_fragile` | Down-weights candidates whose value concentrates in one category; prefers spread contributors at near-equal score. |

Each test builds 2-3 `RankedPick`s with `per_category` values that make the constraint bite, asserts the overlay returns the protecting/preferred one, and asserts it falls back to top-score when the constraint is not triggered.

- [ ] **Step 3: Run to verify red**

Run: `pytest tests/test_draft/test_strategy_overlays.py -v`
Expected: FAIL with missing `OVERLAYS` keys.

- [ ] **Step 4: Implement each overlay; trip the fallback explicitly if a signal is missing**

Port each strategy. If a strategy depends on a signal not present on `RankedPick` and plumbing it through is more than a small change, STOP and apply the spec's fallback: leave that strategy as `default` behavior under deltaRoto, keep its full behavior for var/vona by reading the board inside the overlay via an optional `board=` kwarg, and log a one-line note in the plan's task record naming the strategy and the missing signal. Do not fabricate the signal.

- [ ] **Step 5: Run to verify green**

Run: `pytest tests/test_draft/test_strategy_overlays.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/draft/strategy.py tests/test_draft/test_strategy_overlays.py
git commit -m "feat(draft): no-punt + AVG strategies as overlays"
```

### Task 12: Keep OVERLAYS and STRATEGIES key-aligned (additive)

The destructive cleanup -- deleting the legacy `pick_*` API and aliasing
`STRATEGIES = OVERLAYS` -- is DEFERRED to Task 15b, because `simulate_draft.py`
and `compare_strategies.py` still call the old `pick_*(board, full_board,
tracker, ...)` signature until Tasks 13-15 reroute them. Aliasing now would
break those callers mid-plan. This task only guarantees the two registries stay
key-aligned so `config.py` validation (`strategy in STRATEGIES`) accepts exactly
the overlay names.

**Files:**
- Modify: `src/fantasy_baseball/draft/strategy.py`
- Test: `tests/test_draft/test_strategy_overlays.py`

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_draft/test_strategy_overlays.py
from fantasy_baseball.draft.strategy import OVERLAYS, STRATEGIES


def test_overlays_cover_every_strategy_name():
    # Every legacy strategy name must have an overlay so config validation and
    # the unified seam agree on the valid set.
    assert set(OVERLAYS) == set(STRATEGIES)
```

- [ ] **Step 2: Run to verify red, then green**

Run: `pytest tests/test_draft/test_strategy_overlays.py::test_overlays_cover_every_strategy_name -v`
Expected: FAIL if any of the 14 strategy keys (default, nonzero_sv, avg_hedge, two_closers, three_closers, four_closers, no_punt, no_punt_opp, no_punt_stagger, no_punt_cap3, avg_anchor, closers_avg, balanced, anti_fragile) is missing from OVERLAYS. Add any missing overlay from Tasks 8/10/11 until green.

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/draft/strategy.py tests/test_draft/test_strategy_overlays.py
git commit -m "test(draft): OVERLAYS covers every STRATEGIES key"
```

---

## Phase 4: Consolidate the simulators

> **File-budget note (CLAUDE.md <=5 files/phase):** Phase 4 as a whole touches
> more than 5 files, so the *Task* (not the Phase) is the checkpoint unit here --
> each of Tasks 13, 14, 15, 15b is a single commit touching <=5 files, with
> tests green at each boundary. Execute and review one task at a time.

### Task 13: Route the simulator's user pick through `recommend()`

**Files:**
- Modify: `scripts/simulate_draft.py`
- Test: `tests/test_draft/test_simulate_draft.py` (existing; extend)

- [ ] **Step 1: Read the current pick paths**

Read `simulate_draft.py` `_simulate`/pick functions (around `:304-330`) and how it builds per-pick inputs. Identify where the user pick and each field pick are chosen.

- [ ] **Step 2: Write a failing test pinning mode-routing**

```python
# tests/test_draft/test_simulate_draft.py (extend)
def test_simulate_accepts_deltaroto_mode_and_runs(tiny_league_config):
    from scripts.simulate_draft import simulate_one_draft
    result = simulate_one_draft(
        config=tiny_league_config,
        scoring_mode="deltaroto_immediate",
        strategy="default",
        seed=7,
    )
    assert result.rosters  # a full draft completed
```

(If the public sim entry has a different name, use the real one found in Step 1; the assertion is that a deltaRoto mode runs end-to-end.)

Define `tiny_league_config` in `tests/test_draft/conftest.py`: a `LeagueConfig` for a small league (e.g. `num_teams=4`, the real `roster_slots`, two keepers) loaded via `config.load_config` from a fixture `league.yaml`, or constructed directly. Reuse `minimal_league_yaml` from Task 16 if it lands first; otherwise add the fixture here and have Task 16 reuse it.

- [ ] **Step 3: Run to verify red**

Run: `pytest tests/test_draft/test_simulate_draft.py -k deltaroto -v`
Expected: FAIL (mode not handled / function missing).

- [ ] **Step 4: Implement seam routing for the user pick**

Replace the user-pick selection with a `recommend(ctx, strategy=..., open_starters=...)` call. `recommend()` already serves all four modes as of Task 9 -- this task only builds the `RecommendContext` per pick and calls it; no new ranker is added here. Build the context from the in-progress draft state:
- deltaRoto modes: `inputs = recs_integration.compute_rec_inputs(state, board_path, league_yaml)` (the sim advances a state dict; reuse it), set `ctx.inputs`.
- var/vona modes: set `ctx.board` (the sim's board DataFrame), `ctx.drafted`, `ctx.filled_positions`, `ctx.config`.

Set only the input the mode needs; `rank_for_mode` raises a clear error if the required one is absent. Compute `open_starters` from the user roster as the sim already does.

- [ ] **Step 5: Run to verify green + existing sim tests**

Run: `pytest tests/test_draft/test_simulate_draft.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add scripts/simulate_draft.py tests/test_draft/test_simulate_draft.py
git commit -m "refactor(sim): user pick routes through recommend() seam"
```

### Task 14: Fold deltaRoto field model in; decompose the monolith

**Files:**
- Modify: `scripts/simulate_draft.py` (split into harness/field/reporting within the file or sibling modules under `src/fantasy_baseball/draft/sim/`)
- Modify: `scripts/sim_deltaroto.py` (port unique pieces, then delete in Task 15)
- Test: `tests/test_draft/test_simulate_draft.py`

- [ ] **Step 1: Diff the two sims' field models**

Read `sim_deltaroto.py` `STRATEGY_SPECS`/field loop (`:73-178`) and `simulate_draft.py`'s opponent path. List behaviors unique to each (ADP noise model, pick_rank variance, position-aware gate). The seam + overlays now cover selection; what remains unique is the field-assignment + variance harness.

- [ ] **Step 2: Write a test pinning the variance harness**

Pin that opponents use per-team pick_rank/ADP variance (fixed seed -> deterministic rosters; two seeds -> different rosters). Assert reproducibility under a fixed seed.

- [ ] **Step 3: Run red**

Run: `pytest tests/test_draft/test_simulate_draft.py -k variance -v`
Expected: FAIL until the consolidated harness exists.

- [ ] **Step 4: Implement the single harness**

Extract three units (free functions or a small `sim/` package): **harness** (draft loop + snake order), **field** (opponent strategy assignment + variance), **reporting** (standings/roto/keeper summary). Each opponent and the user pick call `recommend()`. Port any deltaRoto-only field behavior worth keeping. Keep functions small and individually testable.

Preserve `run_user_pick_sequence` (Task 4b) on the consolidated `simulate_draft`, now serving all four modes, so the var sim golden's import is unchanged.

- [ ] **Step 5: Run green + the var sim golden**

Run: `pytest tests/test_draft/test_simulate_draft.py tests/test_draft/test_parity_sim_golden.py::test_sim_var_picks_match_golden -v`
Expected: PASS -- the var/vona path through the consolidated harness reproduces the pre-refactor pick sequence captured in Task 4b. If it diverges, fix the harness; do NOT regenerate the golden.

- [ ] **Step 6: Commit**

```bash
git add scripts/simulate_draft.py src/fantasy_baseball/draft/sim/ tests/test_draft/test_simulate_draft.py
git commit -m "refactor(sim): single scoring_mode-driven harness (field + reporting split)"
```

### Task 15: Delete `sim_deltaroto.py`; rewire `compare_strategies.py` + `replay_picks.py`

**Files:**
- Delete: `scripts/sim_deltaroto.py`
- Modify: `scripts/compare_strategies.py:88` (mode list), and its sim invocation
- Modify: `scripts/replay_picks.py:98`

- [ ] **Step 1: Re-point the deltaRoto sim golden, then confirm nothing imports sim_deltaroto**

In `tests/test_draft/test_parity_sim_golden.py`, change the deltaRoto import from `from scripts.sim_deltaroto import run_user_pick_sequence` to `from scripts.simulate_draft import run_user_pick_sequence` (the consolidated entry from Task 14 now serves deltaRoto modes). Run `pytest tests/test_draft/test_parity_sim_golden.py -v` -- BOTH goldens must still match (the consolidated sim reproduces sim_deltaroto's pre-refactor deltaRoto picks). Then:

Run: `grep -rn "sim_deltaroto" src scripts tests docs config`
Expected: no remaining imports (only docs references). If anything else imports it, migrate first.

- [ ] **Step 2: Extend the mode grid in compare_strategies**

Change `scoring_modes = ["vona", "var"]` (`compare_strategies.py:88`) to `["var", "vona", "deltaroto_immediate", "deltaroto_vopn"]` and route each combo through the consolidated sim entry from Task 14.

- [ ] **Step 3: Repoint replay_picks**

Update `replay_picks.py:98` (`scoring_mode="vona"`) to accept a `--scoring-mode` arg defaulting to `deltaroto_immediate`, routing through the seam.

- [ ] **Step 4: Delete the redundant sim**

```bash
git rm scripts/sim_deltaroto.py
```

- [ ] **Step 5: Smoke-run the consolidated tooling**

Run:
```bash
python scripts/simulate_draft.py -s default --scoring-mode deltaroto_immediate --iters 2
python scripts/replay_picks.py --scoring-mode deltaroto_immediate
pytest tests/test_draft/test_parity_sim_golden.py -v
```
Expected: the scripts run without error (small iter count); both sim goldens PASS through the single consolidated entry.

- [ ] **Step 6: Commit**

```bash
git add scripts/compare_strategies.py scripts/replay_picks.py tests/test_draft/test_parity_sim_golden.py
git commit -m "refactor(sim): delete sim_deltaroto; one sim drives all modes"
```

### Task 15b: Drop the legacy pick_* API; alias STRATEGIES = OVERLAYS

PREREQUISITE: Tasks 13-15 complete -- no caller (the consolidated sim,
`compare_strategies.py`, `replay_picks.py`, the dashboard) invokes the old
`pick_*(board, full_board, tracker, ...)` API any more. This task removes that
now-dead path (the cleanup deferred from Task 12).

**Files:**
- Modify: `src/fantasy_baseball/draft/strategy.py`
- Test: existing `tests/test_draft` strategy tests

- [ ] **Step 1: Confirm nothing calls the legacy API**

Run:
```bash
grep -rn "get_recommendations\|_choose_rec\|_get_recs\|pick_default\|pick_two_closers\|pick_no_punt\|pick_avg\|pick_closers\|pick_balanced\|pick_anti_fragile\|pick_nonzero" src scripts tests
```
Expected: no production/script references to the legacy `pick_*`/`_get_recs`/`_choose_rec`/`get_recommendations`-inside-strategy remain (only `OVERLAYS` and the overlay functions). If a test still calls a legacy `pick_*`, migrate it to `OVERLAYS[name](ranked, ...)` first (translate the assertion, do not drop coverage).

- [ ] **Step 2: Remove the dead path and alias the registry**

Delete the legacy `pick_*` functions, `_get_recs`, `_choose_rec`, and the `get_recommendations` import from `strategy.py` (this also removes the lone `rec.var` read at the old `strategy.py:833` -- overlays read `rec.score`/`per_category` instead). Set `STRATEGIES = OVERLAYS`. Keep `select_from_ranked` (used by `recommend()`).

- [ ] **Step 3: Run the draft + sim suites + both parity goldens**

Run: `pytest tests/test_draft -q`
Expected: PASS, including `test_parity_golden.py` and `test_parity_sim_golden.py` (the removal changes no picks).

- [ ] **Step 4: Verify no dead code / lint**

Run: `vulture src/fantasy_baseball/draft/strategy.py` and `ruff check src/fantasy_baseball/draft/strategy.py`
Expected: no NEW dead code, zero lint violations.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/draft/strategy.py tests/test_draft
git commit -m "refactor(draft): drop legacy pick_* API; STRATEGIES = OVERLAYS"
```

---

## Phase 5: Config + docs

### Task 16: Accept the deltaRoto modes in config validation

**Files:**
- Modify: `src/fantasy_baseball/config.py:38`
- Test: `tests/test_config.py` (create if absent)

- [ ] **Step 1: Write the failing test**

```python
# tests/test_config.py
import pytest
from fantasy_baseball.config import load_config


@pytest.mark.parametrize("mode", ["var", "vona", "deltaroto_immediate", "deltaroto_vopn"])
def test_valid_scoring_modes_accepted(tmp_path, mode, minimal_league_yaml):
    path = minimal_league_yaml(scoring_mode=mode, strategy="default")
    cfg = load_config(path)
    assert cfg.scoring_mode == mode


def test_invalid_scoring_mode_rejected(tmp_path, minimal_league_yaml):
    path = minimal_league_yaml(scoring_mode="bogus", strategy="default")
    with pytest.raises(ValueError, match="Unknown scoring_mode"):
        load_config(path)
```

Add a `minimal_league_yaml` fixture writing a valid minimal `league.yaml` with overridable `scoring_mode`/`strategy`.

- [ ] **Step 2: Run to verify red**

Run: `pytest tests/test_config.py -v`
Expected: FAIL -- `deltaroto_immediate` rejected by current `VALID_SCORING_MODES`.

- [ ] **Step 3: Widen the valid set**

In `config.py:38`:
```python
    VALID_SCORING_MODES = {"var", "vona", "deltaroto_immediate", "deltaroto_vopn"}
```

- [ ] **Step 4: Run to verify green**

Run: `pytest tests/test_config.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/config.py tests/test_config.py
git commit -m "feat(config): accept deltaroto_immediate/deltaroto_vopn scoring modes"
```

### Task 17: Flip `league.yaml` to the verdict winner

**Files:**
- Modify: `config/league.yaml:11-12`

- [ ] **Step 1: Update the config**

```yaml
draft:
  strategy: default
  scoring_mode: deltaroto_immediate
```

- [ ] **Step 2: Verify config loads and the dashboard still serves recs**

Run:
```bash
python -c "from pathlib import Path; from fantasy_baseball.config import load_config; print(load_config(Path('config/league.yaml')).scoring_mode)"
pytest tests/test_draft/test_parity_golden.py -v
```
Expected: prints `deltaroto_immediate`; golden parity still PASS.

- [ ] **Step 3: Commit**

```bash
git add config/league.yaml
git commit -m "config: draft scoring_mode -> deltaroto_immediate (verdict winner)"
```

### Task 18: Refresh `draft/CLAUDE.md`

**Files:**
- Modify: `src/fantasy_baseball/draft/CLAUDE.md:11-22` (Scoring modes + Strategy sections)

- [ ] **Step 1: Rewrite the Scoring modes + Strategy sections**

Replace the "Two ranking modes" framing with the unified seam: four `scoring_mode` values (`var`, `vona`, `deltaroto_immediate`, `deltaroto_vopn`), `recommend(scoring_mode, strategy)` as the single entry, strategies as orthogonal overlays in `OVERLAYS`/`STRATEGIES`, and that the dashboard + simulator + compare_strategies all route through `recommend()`. Note `deltaroto_immediate` is the validated default. Keep it ASCII-only.

- [ ] **Step 2: Verify no stale references remain**

Run: `grep -n "Two ranking modes\|VONA was tested\|position-level" src/fantasy_baseball/draft/CLAUDE.md`
Expected: update or remove any line that no longer matches the unified design.

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/draft/CLAUDE.md
git commit -m "docs(draft): document the unified recommend() seam"
```

---

## Final verification (end-of-effort checklist, per CLAUDE.md)

- [ ] **Run the full suite**

Run: `pytest -n auto`
Expected: all PASS. Confirm `tests/test_draft/test_parity_golden.py` is green (no silent scoring regression).

- [ ] **Lint + format + dead code + types**

Run:
```bash
ruff check .
ruff format --check .
vulture
mypy
```
Expected: zero lint violations, no format drift, no NEW vulture findings, mypy clean for any touched file under `[tool.mypy].files`. Paste a concise summary of each into the final report.

- [ ] **Manual dashboard smoke**

Run: `python scripts/run_draft_dashboard.py --rebuild-board`, open `http://localhost:5050`, confirm the recs panel renders, the immediate/VOPN toggle works, and click-to-pick advances the draft.

---

## Self-review notes (filled during writing)

- **Spec coverage:** RankedPick+metrics (Tasks 1-3); unified seam serving ALL FOUR modes via `RecommendContext`+`rank_for_mode`+`recommend` (Tasks 5,6,9 -- closes review finding #1); overlays orthogonal (8,10-12,15b); one sim (13-15); config+yaml (16-17); docs (18); parity guard -- `/api/recs` golden (4,7,17) AND simulator golden for var+deltaRoto pick sequences (4b, asserted at 14/15/15b -- closes review finding #2).
- **Type consistency:** `RankedPick.positions` is `list[Position]`; `position_strings()` is the only place enums become strings (used by `to_recs_json`). `score`/`metrics` populated by both adapters. `rank_for_mode(ctx)` and `recommend(ctx, ...)` share the single `RecommendContext`. `OVERLAYS` keys == `STRATEGIES` keys == `config.py` valid strategies (pinned by Task 12's test).
- **Ordering guard:** the destructive `STRATEGIES=OVERLAYS` + legacy `pick_*` removal is Task 15b (after the sim/compare/replay callers reroute in 13-15), not Task 12 -- aliasing earlier would break the still-legacy sim.
- **Known fallback:** Task 11 Step 4 is the single sanctioned place to invoke the spec's "overlay-where-cheap" path; it requires a logged note naming the strategy + missing signal, never silent.
