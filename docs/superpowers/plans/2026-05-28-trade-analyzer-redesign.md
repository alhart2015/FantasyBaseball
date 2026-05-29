# Trade Analyzer Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Build-a-Trade redesign: click-to-select-and-swap replaces SortableJS drag-drop for a mobile-usable interaction, and the evaluator surfaces the opponent's deltaroto so trades can be pitched to the other manager.

**Architecture:** Backend extends `MultiTradeResult` with parallel `opp_*` fields (roto / ev_roto / stat_totals / categories / band) computed against the opponent's baseline; `TradeProposal` gains `opp_active_ids` (with backwards-compat fallback). The Flask routes return the new fields additively and the optimize route gains a `side` param. Frontend keeps the `state.placements` map but replaces SortableJS with a tiny click state machine (`state.selection` + `tap(target)`), restructures the panel DOM for stacked mobile-first layout with sticky team headers, and renders two result cards (My / Their) each carrying the existing three-mode toggle.

**Tech Stack:** Python 3.12 / pytest / Flask / vanilla JS (no framework) / CSS (no preprocessor). Existing helpers reused: `apply_swap_delta`, `aggregate_player_stats`, `score_roto_dict`, `compute_delta_roto_band`.

**Branch:** `feat/trade-analyzer-click-swap` (already created from `main`, with the design spec committed as `a78ef86`).

**Design spec:** `docs/superpowers/specs/2026-05-28-trade-analyzer-redesign-design.md`

---

## File Structure

Files this plan creates or modifies (no unrelated refactors):

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/trades/multi_trade.py` | Modify | Extend `TradeProposal` (add `opp_active_ids`); extend `MultiTradeResult` (add `opp_*` fields); extend `evaluate_multi_trade` to compute opp views + band. |
| `src/fantasy_baseball/web/season_routes.py` | Modify | `/api/evaluate-trade` serializes new `opp_*` fields; `/api/optimize-trade-lineup` accepts `side` param ("my" / "opp" / "both", default "both") with partial-failure handling. |
| `src/fantasy_baseball/web/templates/season/waivers_trades.html` | Modify | Replace SortableJS event layer with click state machine; new stacked DOM structure with sticky team headers + selection indicator strip; two-card result panel; rename Optimize button; include `opp_active_ids` in payload. |
| `src/fantasy_baseball/web/static/season.css` | Modify | `.page-trades` layout: mobile-first stacked panels, sticky team headers, selected-row styling (amber border + glow), bottom indicator strip, two-card result grid. |
| `tests/test_trades/test_multi_trade.py` | Modify | Add tests for `opp_active_ids`, opp views, opp band, opp_active_ids fallback, regression guard for existing my-side values. |
| `tests/test_web/test_evaluate_trade_route.py` | Modify | Assert new `opp_*` fields in response; existing fields unchanged. |
| `tests/test_web/test_optimize_trade_lineup_route.py` | Create | New test file: matrix of `side` param values (my / opp / both / invalid) + partial-failure path. |

No files are deleted.

---

## Task 1: TradeProposal gains `opp_active_ids`

**Files:**
- Modify: `src/fantasy_baseball/trades/multi_trade.py:25-39`
- Test: `tests/test_trades/test_multi_trade.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_trades/test_multi_trade.py`:

```python
def test_trade_proposal_has_opp_active_ids_default_empty_set():
    p = TradeProposal(opponent="Foo")
    assert p.opp_active_ids == set()


def test_trade_proposal_accepts_opp_active_ids():
    p = TradeProposal(opponent="Foo", opp_active_ids={"Cade Smith::pitcher"})
    assert p.opp_active_ids == {"Cade Smith::pitcher"}
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_trades/test_multi_trade.py::test_trade_proposal_has_opp_active_ids_default_empty_set -v
```

Expected: FAIL with `AttributeError` or `TypeError` (no such field).

- [ ] **Step 3: Add field to dataclass**

In `src/fantasy_baseball/trades/multi_trade.py`, the `TradeProposal` block (lines 25-39) — append after `my_active_ids`:

```python
    opp_active_ids: set[str] = field(default_factory=set)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_trades/test_multi_trade.py::test_trade_proposal_has_opp_active_ids_default_empty_set tests/test_trades/test_multi_trade.py::test_trade_proposal_accepts_opp_active_ids -v
```

Expected: both PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/multi_trade.py tests/test_trades/test_multi_trade.py
git commit -m "feat(trades): add opp_active_ids to TradeProposal"
```

---

## Task 2: MultiTradeResult gains `opp_*` fields

**Files:**
- Modify: `src/fantasy_baseball/trades/multi_trade.py:68-82`
- Test: `tests/test_trades/test_multi_trade.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_trades/test_multi_trade.py`:

```python
def test_multi_trade_result_has_opp_view_blocks_and_categories():
    from fantasy_baseball.trades.multi_trade import (
        MultiTradeResult,
        ViewBlock,
        CategoryDelta,
    )

    empty_view = ViewBlock(delta_total=0.0, categories={})
    r = MultiTradeResult(
        legal=True,
        reason=None,
        delta_total=0.0,
        categories={},
        roto=empty_view,
        ev_roto=empty_view,
        stat_totals=empty_view,
        band=None,
        opp_delta_total=0.0,
        opp_categories={"R": CategoryDelta(before=10.0, after=11.0, delta=1.0)},
        opp_roto=empty_view,
        opp_ev_roto=empty_view,
        opp_stat_totals=empty_view,
        opp_band=None,
    )
    assert r.opp_delta_total == 0.0
    assert r.opp_categories["R"].delta == 1.0
    assert r.opp_roto.delta_total == 0.0
    assert r.opp_ev_roto.delta_total == 0.0
    assert r.opp_stat_totals.delta_total == 0.0
    assert r.opp_band is None


def test_multi_trade_result_opp_fields_have_safe_defaults():
    """All opp_* fields must be constructable without explicit args (parity with my-side)."""
    r = MultiTradeResult(legal=True, reason=None, delta_total=0.0, categories={})
    assert r.opp_delta_total == 0.0
    assert r.opp_categories == {}
    assert r.opp_roto.delta_total == 0.0
    assert r.opp_ev_roto.delta_total == 0.0
    assert r.opp_stat_totals.delta_total == 0.0
    assert r.opp_band is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
pytest tests/test_trades/test_multi_trade.py::test_multi_trade_result_has_opp_view_blocks_and_categories tests/test_trades/test_multi_trade.py::test_multi_trade_result_opp_fields_have_safe_defaults -v
```

Expected: FAIL with `TypeError: __init__() got an unexpected keyword argument 'opp_*'`.

- [ ] **Step 3: Extend dataclass**

In `src/fantasy_baseball/trades/multi_trade.py`, modify the `MultiTradeResult` block (lines 68-82). The full new block:

```python
@dataclass
class MultiTradeResult:
    """Output of :func:`evaluate_multi_trade`."""

    legal: bool
    reason: str | None
    delta_total: float
    categories: dict[str, CategoryDelta]
    roto: ViewBlock = field(default_factory=lambda: ViewBlock(delta_total=0.0, categories={}))
    ev_roto: ViewBlock = field(default_factory=lambda: ViewBlock(delta_total=0.0, categories={}))
    stat_totals: ViewBlock = field(
        default_factory=lambda: ViewBlock(delta_total=0.0, categories={})
    )
    band: dict[str, float | str] | None = None
    # --- Opponent-side parallel fields -------------------------------------
    opp_delta_total: float = 0.0
    opp_categories: dict[str, CategoryDelta] = field(default_factory=dict)
    opp_roto: ViewBlock = field(default_factory=lambda: ViewBlock(delta_total=0.0, categories={}))
    opp_ev_roto: ViewBlock = field(
        default_factory=lambda: ViewBlock(delta_total=0.0, categories={})
    )
    opp_stat_totals: ViewBlock = field(
        default_factory=lambda: ViewBlock(delta_total=0.0, categories={})
    )
    opp_band: dict[str, float | str] | None = None
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_trades/test_multi_trade.py::test_multi_trade_result_has_opp_view_blocks_and_categories tests/test_trades/test_multi_trade.py::test_multi_trade_result_opp_fields_have_safe_defaults -v
```

Expected: both PASS. Also run the whole multi_trade test file to confirm no regression:

```bash
pytest tests/test_trades/test_multi_trade.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/multi_trade.py tests/test_trades/test_multi_trade.py
git commit -m "feat(trades): add opp_* fields to MultiTradeResult"
```

---

## Task 3: Compute opponent active-set deltas (with fallback)

**Files:**
- Modify: `src/fantasy_baseball/trades/multi_trade.py` (inside `evaluate_multi_trade`, around lines 188-223)
- Test: `tests/test_trades/test_multi_trade.py`

Today the opp side uses a roster-level delta:

```python
opp_loses = aggregate_player_stats(received + opp_drops)
opp_gains = aggregate_player_stats(sent)
```

When `opp_active_ids` is provided, this needs to become an active-set delta mirroring my-side. When not provided, fall back to today's roster-level math (preserves regression and matches the docstring's "treat all non-IL as active").

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trades/test_multi_trade.py`:

```python
def test_opp_active_ids_fallback_matches_current_roster_level_math():
    """Without opp_active_ids, opp views must equal today's roster-level computation."""
    # This is a regression guard: a 1-for-1 trade with no drops produces the
    # same opp stat totals whether computed roster-level or active-set, because
    # both swapped players were active. So the fallback path must equal the
    # non-fallback path in this specific shape.
    from fantasy_baseball.trades.multi_trade import (
        TradeProposal,
        evaluate_multi_trade,
    )
    proposal_no_aids = _make_simple_1for1_proposal()
    proposal_with_aids = _make_simple_1for1_proposal()
    proposal_with_aids.opp_active_ids = _opp_active_set_for_simple_fixture()

    fixture = _eval_fixture()
    r_fallback = evaluate_multi_trade(proposal=proposal_no_aids, **fixture)
    r_explicit = evaluate_multi_trade(proposal=proposal_with_aids, **fixture)

    assert r_fallback.opp_stat_totals.categories["R"].after == \
        r_explicit.opp_stat_totals.categories["R"].after


def test_opp_active_ids_changes_opp_stat_totals_when_lineup_differs():
    """When opp_active_ids excludes a player who was active before the trade,
    opp_stat_totals must drop that player's contribution."""
    from fantasy_baseball.trades.multi_trade import (
        TradeProposal,
        evaluate_multi_trade,
    )
    fixture = _eval_fixture()
    full_active = _opp_active_set_for_simple_fixture()
    bench_one = set(full_active)
    bench_one.discard(next(iter(full_active)))  # bench one player

    p_full = _make_simple_1for1_proposal()
    p_full.opp_active_ids = full_active
    p_bench = _make_simple_1for1_proposal()
    p_bench.opp_active_ids = bench_one

    r_full = evaluate_multi_trade(proposal=p_full, **fixture)
    r_bench = evaluate_multi_trade(proposal=p_bench, **fixture)

    # Benching any active hitter must reduce opp R/HR/RBI/SB after-totals.
    assert r_bench.opp_stat_totals.categories["R"].after < \
        r_full.opp_stat_totals.categories["R"].after
```

> **Note for the engineer:** Your fixture helpers (`_make_simple_1for1_proposal`, `_opp_active_set_for_simple_fixture`, `_eval_fixture`) need to set up a minimal 23-player roster on each side, projected standings with both team names, and one 1-for-1 trade. Look at the existing fixtures starting around `tests/test_trades/test_multi_trade.py:87` (`_hitter_with_key`) and reuse the same pattern. Build these helpers as module-level functions at the top of the test file alongside `_hitter_with_key`.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_trades/test_multi_trade.py::test_opp_active_ids_fallback_matches_current_roster_level_math tests/test_trades/test_multi_trade.py::test_opp_active_ids_changes_opp_stat_totals_when_lineup_differs -v
```

Expected: FAIL — either `opp_stat_totals` is empty (default) or values don't match because we haven't built the opp view yet. The point is to anchor the contract.

- [ ] **Step 3: Implement opp active-set computation**

In `src/fantasy_baseball/trades/multi_trade.py`, replace the `# --- 3. Build active-set deltas ---` block (lines 188-207) with:

```python
    # --- 3. Build active-set deltas ------------------------------------------
    # MY side (unchanged): use proposal.my_active_ids; fall back to current
    # active set if it's empty (legacy callers).
    all_mine_by_key = {
        **my_idx,
        **{player_key(p): p for p in received},
        **{player_key(p): p for p in my_adds},
    }
    before_mine = _current_active_set(hart_roster)
    after_mine = set(proposal.my_active_ids)

    mine_leaving = [all_mine_by_key[k] for k in before_mine - after_mine if k in all_mine_by_key]
    mine_entering = [all_mine_by_key[k] for k in after_mine - before_mine if k in all_mine_by_key]
    my_loses = aggregate_player_stats(mine_leaving)
    my_gains = aggregate_player_stats(mine_entering)

    # OPP side: prefer active-set delta when opp_active_ids is provided;
    # otherwise fall back to the roster-level computation (today's behavior).
    all_opp_by_key = {
        **opp_idx,
        **{player_key(p): p for p in sent},
    }
    before_opp = _current_active_set(opp_rosters[proposal.opponent])

    if proposal.opp_active_ids:
        after_opp = set(proposal.opp_active_ids)
        opp_leaving = [
            all_opp_by_key[k] for k in before_opp - after_opp if k in all_opp_by_key
        ]
        opp_entering = [
            all_opp_by_key[k] for k in after_opp - before_opp if k in all_opp_by_key
        ]
        opp_loses = aggregate_player_stats(opp_leaving)
        opp_gains = aggregate_player_stats(opp_entering)
    else:
        # Legacy fallback: roster-level (matches today's docstring "treat all
        # non-IL as active"). Track before/after player lists for the band call
        # in Task 4 by deriving them from the same assumption.
        opp_loses = aggregate_player_stats(received + opp_drops)
        opp_gains = aggregate_player_stats(sent)
        # Derive synthetic active sets so the band code in Task 4 can use a
        # single code path: assume sent slides into a vacated active slot,
        # opp_drops vacate an active slot (matches roster-level semantics).
        received_keys = {player_key(p) for p in received}
        sent_keys = {player_key(p) for p in sent}
        opp_drop_keys = set(proposal.opp_drops)
        after_opp = (before_opp - received_keys - opp_drop_keys) | sent_keys
        opp_leaving = [
            all_opp_by_key[k] for k in before_opp - after_opp if k in all_opp_by_key
        ]
        opp_entering = [
            all_opp_by_key[k] for k in after_opp - before_opp if k in all_opp_by_key
        ]
```

> **Why both branches build `opp_leaving` / `opp_entering`:** Task 4 needs Player lists for the band call. Doing it once here keeps the band code symmetric with my-side.

- [ ] **Step 4: Run the fallback regression test only**

```bash
pytest tests/test_trades/test_multi_trade.py::test_opp_active_ids_fallback_matches_current_roster_level_math -v
```

Expected: still FAIL (we have the math but haven't built the view yet). Move on to Task 4 — these two tests will both pass after Task 4.

- [ ] **Step 5: Commit (intermediate)**

```bash
git add src/fantasy_baseball/trades/multi_trade.py tests/test_trades/test_multi_trade.py
git commit -m "feat(trades): compute opp active-set deltas with fallback"
```

---

## Task 4: Build opp views (roto, ev_roto, stat_totals, categories) + band

**Files:**
- Modify: `src/fantasy_baseball/trades/multi_trade.py` (inside `evaluate_multi_trade`, the view-building section around lines 209-288)
- Test: `tests/test_trades/test_multi_trade.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trades/test_multi_trade.py`:

```python
def test_opp_roto_view_is_built_against_opp_baseline():
    """opp_roto.delta_total is the roto-point change for the opponent's team."""
    from fantasy_baseball.trades.multi_trade import evaluate_multi_trade

    p = _make_simple_1for1_proposal()
    p.opp_active_ids = _opp_active_set_for_simple_fixture()
    r = evaluate_multi_trade(proposal=p, **_eval_fixture())

    assert r.legal is True
    # opp_roto must have categories filled and a delta_total that's the sum of cat deltas.
    assert set(r.opp_roto.categories) == {c.value for c in ALL_CATEGORIES_LOCAL}
    summed = sum(cv.delta for cv in r.opp_roto.categories.values())
    assert abs(r.opp_roto.delta_total - summed) < 1e-9


def test_opp_ev_roto_and_stat_totals_built():
    from fantasy_baseball.trades.multi_trade import evaluate_multi_trade

    p = _make_simple_1for1_proposal()
    p.opp_active_ids = _opp_active_set_for_simple_fixture()
    r = evaluate_multi_trade(proposal=p, **_eval_fixture())

    assert r.opp_ev_roto.categories  # non-empty
    assert r.opp_stat_totals.categories  # non-empty
    # stat_totals delta_total is conventionally 0.0
    assert r.opp_stat_totals.delta_total == 0.0


def test_opp_band_is_present_when_team_sds_provided():
    from fantasy_baseball.trades.multi_trade import evaluate_multi_trade

    p = _make_simple_1for1_proposal()
    p.opp_active_ids = _opp_active_set_for_simple_fixture()
    r = evaluate_multi_trade(proposal=p, **_eval_fixture())  # fixture includes team_sds

    assert r.opp_band is not None
    assert "mean" in r.opp_band
    assert "sd" in r.opp_band
    assert "p_positive" in r.opp_band


def test_my_side_results_unchanged_after_opp_additions():
    """Regression guard: when opp_active_ids matches today's implicit
    roster-level set, my-side roto/ev_roto/stat_totals must be byte-equal
    to the legacy result.
    """
    # Use the fallback path (no opp_active_ids) so my-side computation is
    # exactly today's code path. Pin the expected numeric values.
    from fantasy_baseball.trades.multi_trade import evaluate_multi_trade

    p = _make_simple_1for1_proposal()  # no opp_active_ids set
    r = evaluate_multi_trade(proposal=p, **_eval_fixture())

    # These specific values come from running the test once after Task 4
    # implementation lands. Update the literals to whatever the test prints
    # on the first run, then assert on them as a regression guard.
    assert r.legal is True
    # placeholder asserts: replace with actual values from a green run.
    # e.g. assert r.roto.delta_total == pytest.approx(...)
    #      assert r.ev_roto.categories["R"].after == pytest.approx(...)
```

Add at the top of the test file alongside other imports:

```python
from fantasy_baseball.utils.constants import ALL_CATEGORIES as ALL_CATEGORIES_LOCAL
```

> **For the engineer:** The fourth test (`test_my_side_results_unchanged_after_opp_additions`) is the regression guard. Run it once after Task 4 implementation lands, copy the printed values out, and replace the placeholder asserts with real `pytest.approx` calls. This pins the my-side numbers so future opp-side refactors can't drift them silently.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_trades/test_multi_trade.py -k "opp_roto_view or opp_ev_roto or opp_band or my_side_results_unchanged" -v
```

Expected: FAIL — opp views are still empty defaults.

- [ ] **Step 3: Implement opp view construction**

In `src/fantasy_baseball/trades/multi_trade.py`, modify the `# --- 4. Apply deltas to baseline and score ---` block (lines 209-262) to compute opp views alongside my-side. Replace from the `def _build_view` line through the `categories` dict construction with:

```python
    # --- 4. Apply deltas to baseline and score -------------------------------
    if not any(e.team_name == hart_name for e in projected_standings.entries):
        return MultiTradeResult(
            legal=False,
            reason=f"Team {hart_name} missing from projected_standings",
            delta_total=0.0,
            categories={},
        )

    before_stats = {e.team_name: e.stats.to_dict() for e in projected_standings.entries}
    after_stats = dict(before_stats)
    after_stats[hart_name] = apply_swap_delta(before_stats[hart_name], my_loses, my_gains)
    after_stats[proposal.opponent] = apply_swap_delta(
        before_stats[proposal.opponent], opp_loses, opp_gains
    )

    roto_before = score_roto_dict(before_stats)
    roto_after = score_roto_dict(after_stats)
    ev_roto_before = score_roto_dict(before_stats, team_sds=team_sds)
    ev_roto_after = score_roto_dict(after_stats, team_sds=team_sds)

    def _build_view(team_name: str, before_pts, after_pts) -> ViewBlock:
        cats: dict[str, CategoryView] = {}
        total = 0.0
        for cat in ALL_CATEGORIES:
            b = before_pts[team_name][f"{cat.value}_pts"]
            a = after_pts[team_name][f"{cat.value}_pts"]
            cats[cat.value] = CategoryView(before=b, after=a, delta=a - b)
            total += a - b
        return ViewBlock(delta_total=total, categories=cats)

    def _build_stat_totals(team_name: str) -> ViewBlock:
        cats: dict[str, CategoryView] = {}
        for cat in ALL_CATEGORIES:
            b = float(before_stats[team_name][cat.value])
            a = float(after_stats[team_name][cat.value])
            cats[cat.value] = CategoryView(before=b, after=a, delta=a - b)
        return ViewBlock(delta_total=0.0, categories=cats)

    def _build_categories(team_name: str) -> tuple[dict[str, CategoryDelta], float]:
        cats: dict[str, CategoryDelta] = {}
        total = 0.0
        for cat in ALL_CATEGORIES:
            b = ev_roto_before[team_name][f"{cat.value}_pts"]
            a = ev_roto_after[team_name][f"{cat.value}_pts"]
            cats[cat.value] = CategoryDelta(before=b, after=a, delta=a - b)
            total += a - b
        return cats, total

    roto_view = _build_view(hart_name, roto_before, roto_after)
    ev_roto_view = _build_view(hart_name, ev_roto_before, ev_roto_after)
    stat_totals_view = _build_stat_totals(hart_name)
    categories, total_delta = _build_categories(hart_name)

    opp_name = proposal.opponent
    opp_roto_view = _build_view(opp_name, roto_before, roto_after)
    opp_ev_roto_view = _build_view(opp_name, ev_roto_before, ev_roto_after)
    opp_stat_totals_view = _build_stat_totals(opp_name)
    opp_categories, opp_total_delta = _build_categories(opp_name)
```

Then replace the `# --- 5. Monte-Carlo confidence band ---` block (lines 263-277) with:

```python
    # --- 5. Monte-Carlo confidence bands (my + opp) --------------------------
    from fantasy_baseball.lineup.delta_roto import compute_delta_roto_band

    field_stats = projected_standings.field_stats(hart_name)
    before_players = [my_idx[k] for k in before_mine if k in my_idx]
    after_players = [all_mine_by_key[k] for k in after_mine if k in all_mine_by_key]
    band_result = compute_delta_roto_band(
        before_players,
        after_players,
        field_stats,
        hart_name,
        fraction_remaining,
        projected_standings=projected_standings,
        team_sds=team_sds,
    )

    opp_field_stats = projected_standings.field_stats(opp_name)
    opp_before_players = [opp_idx[k] for k in before_opp if k in opp_idx]
    opp_after_players = [all_opp_by_key[k] for k in after_opp if k in all_opp_by_key]
    try:
        opp_band_result = compute_delta_roto_band(
            opp_before_players,
            opp_after_players,
            opp_field_stats,
            opp_name,
            fraction_remaining,
            projected_standings=projected_standings,
            team_sds=team_sds,
        )
        opp_band_dict = opp_band_result.to_dict()
    except Exception:
        # Opp band is best-effort. If sampling fails (e.g. zero variance for
        # that team), fall back to None and let the UI show the plain delta.
        opp_band_dict = None
```

Then replace the final `return MultiTradeResult(...)` (lines 279-288) with:

```python
    return MultiTradeResult(
        legal=True,
        reason=None,
        delta_total=total_delta,
        categories=categories,
        roto=roto_view,
        ev_roto=ev_roto_view,
        stat_totals=stat_totals_view,
        band=band_result.to_dict(),
        opp_delta_total=opp_total_delta,
        opp_categories=opp_categories,
        opp_roto=opp_roto_view,
        opp_ev_roto=opp_ev_roto_view,
        opp_stat_totals=opp_stat_totals_view,
        opp_band=opp_band_dict,
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_trades/test_multi_trade.py -v
```

Expected: all PASS. The regression-guard test from Step 1 needs its placeholder asserts replaced with real values — copy them from the first green run.

- [ ] **Step 5: Pin the regression-guard values**

Read the output of the regression-guard test (it will print or you can add a temporary `print(r.roto.delta_total)`), then update the placeholder asserts in `test_my_side_results_unchanged_after_opp_additions` to real `pytest.approx(...)` values. Re-run:

```bash
pytest tests/test_trades/test_multi_trade.py::test_my_side_results_unchanged_after_opp_additions -v
```

Expected: PASS with concrete pinned values.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/trades/multi_trade.py tests/test_trades/test_multi_trade.py
git commit -m "feat(trades): compute opp views + band in evaluate_multi_trade"
```

---

## Task 5: Wire `/api/evaluate-trade` to surface opp fields

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py:871-888`
- Test: `tests/test_web/test_evaluate_trade_route.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web/test_evaluate_trade_route.py`:

```python
def test_evaluate_trade_response_includes_opp_fields(client_with_trade_fixture):
    """Response JSON must include opp_roto, opp_ev_roto, opp_stat_totals,
    opp_categories, opp_delta_total, opp_band."""
    client, payload = client_with_trade_fixture
    resp = client.post("/api/evaluate-trade", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["legal"] is True
    for field_name in (
        "opp_delta_total",
        "opp_categories",
        "opp_roto",
        "opp_ev_roto",
        "opp_stat_totals",
        "opp_band",
    ):
        assert field_name in data, f"missing {field_name}"
    # opp views have the same shape as my-side views
    assert "delta_total" in data["opp_roto"]
    assert "categories" in data["opp_roto"]


def test_evaluate_trade_response_preserves_existing_fields(client_with_trade_fixture):
    """My-side keys must still be present and at the expected nesting."""
    client, payload = client_with_trade_fixture
    resp = client.post("/api/evaluate-trade", json=payload)
    data = resp.get_json()
    for field_name in ("legal", "reason", "delta_total", "categories",
                       "roto", "ev_roto", "stat_totals", "band"):
        assert field_name in data, f"missing {field_name}"
```

> **For the engineer:** `client_with_trade_fixture` should be a pytest fixture in the same file (or `conftest.py`) that yields a Flask test client and a valid trade payload. Look at existing tests in `tests/test_web/test_evaluate_trade_route.py` for the fixture pattern already in use, and add `opp_active_ids` to the payload if the new field is exercised. If a similar fixture exists under a different name, reuse it.

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_web/test_evaluate_trade_route.py::test_evaluate_trade_response_includes_opp_fields -v
```

Expected: FAIL with `KeyError` or `AssertionError: missing opp_*`.

- [ ] **Step 3: Update the route**

In `src/fantasy_baseball/web/season_routes.py`, the `return jsonify(...)` block at lines 871-888. Append the opp fields to the returned dict — full new block:

```python
        return jsonify(
            {
                "legal": result.legal,
                "reason": result.reason,
                "delta_total": round(result.delta_total, 2),
                "categories": {
                    cat: {
                        "before": round(cd.before, 2),
                        "after": round(cd.after, 2),
                        "delta": round(cd.delta, 2),
                    }
                    for cat, cd in result.categories.items()
                },
                "roto": _serialize_view(result.roto),
                "ev_roto": _serialize_view(result.ev_roto),
                "stat_totals": _serialize_view(result.stat_totals),
                "band": result.band,
                # Opp-side parallel fields.
                "opp_delta_total": round(result.opp_delta_total, 2),
                "opp_categories": {
                    cat: {
                        "before": round(cd.before, 2),
                        "after": round(cd.after, 2),
                        "delta": round(cd.delta, 2),
                    }
                    for cat, cd in result.opp_categories.items()
                },
                "opp_roto": _serialize_view(result.opp_roto),
                "opp_ev_roto": _serialize_view(result.opp_ev_roto),
                "opp_stat_totals": _serialize_view(result.opp_stat_totals),
                "opp_band": result.opp_band,
            }
        )
```

Also accept the new `opp_active_ids` field in the request payload. In the route around lines 836-844, the `TradeProposal(...)` construction — append:

```python
            opp_active_ids=set(data.get("opp_active_ids") or []),
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_web/test_evaluate_trade_route.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py tests/test_web/test_evaluate_trade_route.py
git commit -m "feat(api): /api/evaluate-trade surfaces opp_* delta fields"
```

---

## Task 6: Add `side` param to `/api/optimize-trade-lineup`

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py` (the `api_optimize_trade_lineup` route starting at line 891)
- Test: `tests/test_web/test_optimize_trade_lineup_route.py` (create)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_web/test_optimize_trade_lineup_route.py`:

```python
"""Tests for the /api/optimize-trade-lineup route's `side` parameter."""

from __future__ import annotations

import pytest

# Reuse the same test client fixture used by test_evaluate_trade_route.py.
# If that fixture lives at module level there, import it; otherwise lift it
# into conftest.py and import from there.
from tests.test_web.test_evaluate_trade_route import client_with_trade_fixture  # noqa: F401


def test_optimize_lineup_default_side_is_both(client_with_trade_fixture):
    client, payload = client_with_trade_fixture
    # No 'side' key in payload -> default is 'both'.
    resp = client.post("/api/optimize-trade-lineup", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert data.get("ok") is True
    assert "my_slots" in data
    assert "opp_slots" in data


def test_optimize_lineup_side_my_returns_only_my_slots(client_with_trade_fixture):
    client, payload = client_with_trade_fixture
    payload = dict(payload)
    payload["side"] = "my"
    resp = client.post("/api/optimize-trade-lineup", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "my_slots" in data
    assert "opp_slots" not in data or data["opp_slots"] in ({}, None)


def test_optimize_lineup_side_opp_returns_only_opp_slots(client_with_trade_fixture):
    client, payload = client_with_trade_fixture
    payload = dict(payload)
    payload["side"] = "opp"
    resp = client.post("/api/optimize-trade-lineup", json=payload)
    assert resp.status_code == 200
    data = resp.get_json()
    assert "opp_slots" in data
    assert "my_slots" not in data or data["my_slots"] in ({}, None)


def test_optimize_lineup_rejects_invalid_side(client_with_trade_fixture):
    client, payload = client_with_trade_fixture
    payload = dict(payload)
    payload["side"] = "nonsense"
    resp = client.post("/api/optimize-trade-lineup", json=payload)
    assert resp.status_code == 400
    data = resp.get_json()
    assert "side must be" in (data.get("error") or "")
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_web/test_optimize_trade_lineup_route.py -v
```

Expected: FAIL — route ignores `side` or only does my-side.

- [ ] **Step 3: Update the route**

In `src/fantasy_baseball/web/season_routes.py`, read the existing `api_optimize_trade_lineup` (starts at line 891). The change is structural — gate the my-side and opp-side optimizer calls on the `side` value. Skeleton of the new control flow (adapt to the existing variable names in your local read):

```python
        SIDE_CHOICES = {"my", "opp", "both"}
        side = (data.get("side") or "both").strip().lower()
        if side not in SIDE_CHOICES:
            return jsonify({"error": "side must be 'my', 'opp', or 'both'"}), 400

        result: dict[str, object] = {"ok": True}
        partial = False

        if side in ("my", "both"):
            try:
                my_slots = _run_optimizer_for_my_side(...)  # existing logic
                result["my_slots"] = my_slots
            except Exception as exc:
                if side == "my":
                    raise
                partial = True
                result["my_slots_error"] = str(exc)

        if side in ("opp", "both"):
            try:
                opp_slots = _run_optimizer_for_opp_side(...)
                result["opp_slots"] = opp_slots
            except Exception as exc:
                if side == "opp":
                    raise
                partial = True
                result["opp_slots_error"] = str(exc)

        if partial:
            result["partial"] = True

        return jsonify(result)
```

> **For the engineer:** Extract the existing my-side optimizer logic into a local helper `_run_optimizer_for_my_side`. Mirror it for opp using the same hitter/pitcher optimizers from `fantasy_baseball.lineup.optimizer` but with the opp's post-trade roster (built from `opp_rosters[opponent]` plus `sent`, minus `received` and `opp_drops`). Reuse the legality helpers (`_can_roster_after`, `player_key`) already imported.

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_web/test_optimize_trade_lineup_route.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py tests/test_web/test_optimize_trade_lineup_route.py
git commit -m "feat(api): /api/optimize-trade-lineup accepts side=my|opp|both"
```

---

## Task 7: Click state machine replaces SortableJS

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html` (the SortableJS block at lines 643-700 and the `bt.init()` function)

This task swaps the event layer only — `state.placements`, `placeAt`, `renderPanels`, and the legality / payload logic all stay untouched.

- [ ] **Step 1: Plan the state shape**

In the IIFE near the top of the `<script>` block (around line 324), the `state` object currently has `opponent`, `placements`, `initialPlacements`, `waiverMeta`. Add:

```javascript
    selection: null,  // null | { key, fromZone, fromSide, origin }
```

The `selection` object captures enough to render the indicator strip and to revert if the user replaces the selection with a waiver suggestion (origin: 'waiver').

- [ ] **Step 2: Implement `tap()` dispatch**

Add this new function inside the IIFE (place it just above the existing `placeAt` function):

```javascript
  function tap(target) {
    // target = { kind: "player"|"slot"|"waiver"|"void",
    //            key?: string,           // for kind == "player" or "waiver"
    //            side?: "my"|"opp",      // for kind == "slot"
    //            zone?: string }         // for kind == "slot" (slotId|"BN"|"IL"|"DROP")

    // Deselect rules.
    if (!state.selection) {
      if (target.kind === "player") {
        const place = state.placements.get(target.key);
        if (!place || place.zone === "IL") return;  // IL is read-only
        state.selection = { key: target.key, fromZone: place.zone, fromSide: place.currentSide, origin: place.origin };
        renderPanels();
        return;
      }
      if (target.kind === "waiver") {
        // Seed a waiver-origin placement on tap, like SortableJS did onStart.
        state.waiverMeta.set(target.key, { name: target.waiverName, positions: target.waiverPositions || "" });
        state.placements.set(target.key, { currentSide: "my", zone: "BN", origin: "waiver" });
        state.selection = { key: target.key, fromZone: "BN", fromSide: "my", origin: "waiver" };
        renderPanels();
        return;
      }
      return;  // idle + non-actionable target = no-op
    }

    // selected(A). Now handle all second-tap cases.
    const sel = state.selection;

    if (target.kind === "void") {
      state.selection = null;
      renderPanels();
      return;
    }

    if (target.kind === "player") {
      if (target.key === sel.key) {
        // Tap same player -> deselect.
        state.selection = null;
        renderPanels();
        return;
      }
      // Tap a different player. Swap their zones/sides.
      const otherPlace = state.placements.get(target.key);
      if (!otherPlace || otherPlace.zone === "IL") return;  // can't tap IL
      // Swap selection <-> other.
      placeAt(sel.key, otherPlace.currentSide, otherPlace.zone);
      placeAt(target.key, sel.fromSide, sel.fromZone);
      state.selection = null;
      return;
    }

    if (target.kind === "waiver") {
      // Replace selection with a fresh waiver suggestion. Revert sel to its from-place if it was a waiver-origin first-pick.
      if (sel.origin === "waiver") {
        state.placements.delete(sel.key);
        state.waiverMeta.delete(sel.key);
      }
      state.waiverMeta.set(target.key, { name: target.waiverName, positions: target.waiverPositions || "" });
      state.placements.set(target.key, { currentSide: "my", zone: "BN", origin: "waiver" });
      state.selection = { key: target.key, fromZone: "BN", fromSide: "my", origin: "waiver" };
      renderPanels();
      return;
    }

    if (target.kind === "slot") {
      if (target.zone === "IL") return;  // IL is read-only
      placeAt(sel.key, target.side, target.zone);
      state.selection = null;
      return;
    }
  }
```

- [ ] **Step 3: Replace SortableJS bindings with click handlers**

Replace `attachSortables()` (lines 663-700) with `attachClickHandlers()`:

```javascript
  function attachClickHandlers() {
    // Click on a player chip -> tap player.
    document.querySelectorAll("#bt-my-slots .bt-chip-player, #bt-opp-slots .bt-chip-player, .bt-zone .bt-chip-player").forEach(chip => {
      chip.addEventListener("click", (e) => {
        e.stopPropagation();
        const key = chip.dataset.btDragKey;
        if (!key) return;  // IL chips have no key
        tap({ kind: "player", key });
      });
    });

    // Click on an empty slot or zone -> tap slot.
    document.querySelectorAll("[data-bt-side][data-bt-zone]").forEach(el => {
      el.addEventListener("click", (e) => {
        if (e.target.closest(".bt-chip-player")) return;  // chip handled it
        const side = el.dataset.btSide;
        const zone = el.dataset.btZone;
        if (!side || !zone) return;
        tap({ kind: "slot", side, zone });
      });
    });

    // Click on a waiver suggestion -> tap waiver.
    document.querySelectorAll("#bt-waiver-suggestions li[data-bt-drag-key]").forEach(li => {
      li.addEventListener("click", (e) => {
        e.stopPropagation();
        tap({
          kind: "waiver",
          key: li.dataset.btDragKey,
          waiverName: li.dataset.btWaiverName,
          waiverPositions: li.dataset.btWaiverPositions || "",
        });
      });
    });
  }
```

In `renderPanels()` (line 564), replace the call `attachSortables()` with `attachClickHandlers()`.

Also: add a global click handler that catches taps on non-actionable areas (deselect):

```javascript
  document.addEventListener("click", (e) => {
    if (!state.selection) return;
    // If the click was inside the trade panel but not on any actionable element, deselect.
    const inBuilder = e.target.closest("#section-build-trade");
    const onActionable = e.target.closest(".bt-chip-player, [data-bt-side][data-bt-zone], #bt-waiver-suggestions li");
    if (inBuilder && !onActionable) tap({ kind: "void" });
  });
```

In `bt.init()` (line 790), replace `attachSortables()` with `attachClickHandlers()`. Also delete the `Sortable.min.js` `<script>` import at line 161.

- [ ] **Step 4: Expose debug hooks for QA**

After the existing `window.bt._state` / `_playerKey` / `_isIL` exports (lines 891-894), add:

```javascript
  window.bt._selection = () => state.selection;
  window.bt._tap = tap;
```

- [ ] **Step 5: Manual QA**

Start the dashboard:

```bash
python scripts/run_season_dashboard.py
```

Open the Trades page, expand Build a Trade, pick an opponent. Open browser devtools console and run:

```javascript
bt._state.placements.size              // should equal 23 + 23 = 46 + IL count
bt._selection()                        // null
bt._tap({kind:"player", key: <some opp key from the rendered roster>})
bt._selection()                        // shows {key, fromZone, fromSide, origin}
bt._tap({kind:"player", key: <some my key>})
bt._selection()                        // null again (swap committed)
```

Confirm the chips visibly moved between sides. If they did, click handlers + tap dispatch + placeAt all wired correctly.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat(trade-ui): click state machine replaces SortableJS"
```

---

## Task 8: Mobile-first layout + selection styling

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html` (DOM structure around lines 99-150)
- Modify: `src/fantasy_baseball/web/static/season.css` (`.page-trades` block)

- [ ] **Step 1: Restructure the Build-a-Trade DOM**

In `waivers_trades.html` around line 104, replace the existing `<div class="build-trade-rosters" ...>` block with a stacked structure. Full replacement:

```html
    <div class="bt-rosters">
        <div class="bt-panel" id="bt-panel-my">
            <div class="bt-panel-header">
                <span class="bt-team-name">{{ (config.team_name if config else "My Team") }}</span>
                <span class="bt-active-count" id="bt-my-count">--/23</span>
            </div>
            <div id="bt-my-slots" class="bt-slot-grid" data-bt-side="my"></div>
            <div class="bt-zone-label">Bench</div>
            <div id="bt-my-bench" class="bt-zone" data-bt-side="my" data-bt-zone="BN"></div>
            <div class="bt-zone-label">IL (read-only)</div>
            <div id="bt-my-il" class="bt-zone" data-bt-side="my"></div>
            <div class="bt-zone-label">Drops</div>
            <div id="bt-my-drops" class="bt-zone bt-drops-zone" data-bt-side="my" data-bt-zone="DROP"></div>
            <div class="bt-waiver-block">
                <label for="bt-waiver-search">Add from Waivers</label>
                <input type="text" id="bt-waiver-search" placeholder="Type &ge;2 letters&hellip;" autocomplete="off" />
                <ul id="bt-waiver-suggestions" class="bt-suggestions"></ul>
            </div>
        </div>
        <div class="bt-panel" id="bt-panel-opp">
            <div class="bt-panel-header">
                <span class="bt-team-name" id="bt-opp-name">Opponent</span>
                <span class="bt-active-count" id="bt-opp-count">--/23</span>
            </div>
            <div id="bt-opp-slots" class="bt-slot-grid" data-bt-side="opp"></div>
            <div class="bt-zone-label">Bench</div>
            <div id="bt-opp-bench" class="bt-zone" data-bt-side="opp" data-bt-zone="BN"></div>
            <div class="bt-zone-label">IL (read-only)</div>
            <div id="bt-opp-il" class="bt-zone" data-bt-side="opp"></div>
            <div class="bt-zone-label">Drops</div>
            <div id="bt-opp-drops" class="bt-zone bt-drops-zone" data-bt-side="opp" data-bt-zone="DROP"></div>
        </div>
    </div>

    <div id="bt-selection-strip" class="bt-selection-strip" hidden>
        <span class="bt-selection-text"></span>
        <a href="#" class="bt-selection-cancel">cancel</a>
    </div>
```

> The `bt-zone-label` divs are siblings of the zones now (the old code put them inside each zone's container). Make sure the JS `renderPanels` no longer writes labels into `.bt-zone` divs - it should only fill them with chips. Remove the lines in `renderPanels()` around line 559 that write `<div class="bt-zone-label">...</div>` into zone innerHTML.

- [ ] **Step 2: Update `renderPanels()` to skip the label injection**

In the loop over `["BN", "IL", "DROP"]` inside `renderPanels()` (lines 548-562), remove the `const label = ...` line and the leading `<div class="bt-zone-label">${label}</div>` from the innerHTML assignment. The zone innerHTML should only contain chips now.

Also update the opp panel header and active counts in `renderPanels()`. After the existing roster-row loop, add:

```javascript
      // Update team header active counts.
      let activeCount = 0;
      for (const [, pl] of state.placements) {
        if (pl.currentSide === side && pl.zone !== "IL" && pl.zone !== "DROP") activeCount++;
      }
      const countEl = document.getElementById(`bt-${side}-count`);
      if (countEl) countEl.textContent = `${activeCount}/23`;
      if (side === "opp") {
        const nameEl = document.getElementById("bt-opp-name");
        if (nameEl) nameEl.textContent = state.opponent || "Opponent";
      }
```

- [ ] **Step 3: Add selection strip controller**

After `attachClickHandlers()`, add a small render helper:

```javascript
  function renderSelectionStrip() {
    const strip = document.getElementById("bt-selection-strip");
    if (!strip) return;
    if (!state.selection) {
      strip.hidden = true;
      return;
    }
    const p = playerByKey(state.selection.key);
    const name = p ? p.name : state.selection.key.split("::")[0];
    strip.hidden = false;
    strip.querySelector(".bt-selection-text").textContent = `Selected: ${name} -- tap a slot to swap or move`;
  }
```

Call `renderSelectionStrip()` at the end of `renderPanels()`, and in `tap()` after any state change. Wire the cancel link:

```javascript
  document.addEventListener("click", (e) => {
    const cancel = e.target.closest(".bt-selection-cancel");
    if (!cancel) return;
    e.preventDefault();
    tap({ kind: "void" });
  });
```

- [ ] **Step 4: Add CSS for the new layout**

In `src/fantasy_baseball/web/static/season.css`, find the existing `.page-trades` block and add (or replace, if a partial exists):

```css
.page-trades .bt-rosters {
    display: flex;
    flex-direction: column;
    gap: 16px;
}
.page-trades .bt-panel {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    overflow: hidden;
}
.page-trades .bt-panel-header {
    position: sticky;
    top: 0;
    background: var(--panel-header-bg, #1a3a2e);
    padding: 10px 12px;
    font-size: 13px;
    font-weight: 600;
    display: flex;
    justify-content: space-between;
    align-items: center;
    z-index: 2;
}
#bt-panel-opp .bt-panel-header { background: #3a1e2a; }
.page-trades .bt-team-name { font-weight: 700; }
.page-trades .bt-active-count { color: var(--text-secondary); font-size: 11px; }
.page-trades .bt-slot-grid {
    display: flex;
    flex-direction: column;
}
.page-trades .bt-slot-row {
    display: grid;
    grid-template-columns: 32px 1fr;
    gap: 8px;
    padding: 8px 12px;
    align-items: center;
    border-bottom: 1px solid var(--panel-border);
    min-height: 36px;
    cursor: pointer;
}
.page-trades .bt-slot-label { font-size: 11px; color: var(--text-secondary); }
.page-trades .bt-chip-player {
    display: inline-block;
    cursor: pointer;
    user-select: none;
    max-width: 100%;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    padding: 4px 6px;
}
.page-trades .bt-chip-player.il { cursor: default; color: var(--text-secondary); }
.page-trades .bt-chip-player.foreign { color: var(--accent, #fc4); font-style: italic; }
.page-trades .bt-chip-player.selected {
    background: rgba(255, 204, 68, 0.15);
    box-shadow: inset 4px 0 0 #fc4, inset 0 0 0 1px #fc4;
    font-weight: 600;
}
.page-trades .bt-slot-row:has(.bt-chip-player.selected) {
    background: rgba(255, 204, 68, 0.08);
}
.page-trades .bt-zone {
    min-height: 36px;
    padding: 4px 12px;
    border-bottom: 1px solid var(--panel-border);
}
.page-trades .bt-zone-label {
    padding: 6px 12px 2px;
    font-size: 11px;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.page-trades .bt-selection-strip {
    position: sticky;
    bottom: 0;
    background: #fc4;
    color: #1a1a1a;
    padding: 10px 14px;
    font-weight: 600;
    font-size: 13px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 -3px 8px rgba(0, 0, 0, 0.4);
    z-index: 5;
}
.page-trades .bt-selection-strip[hidden] { display: none; }
.page-trades .bt-selection-cancel { color: #1a1a1a; text-decoration: underline; }

@media (min-width: 768px) {
    .page-trades .bt-rosters {
        flex-direction: row;
    }
    .page-trades .bt-panel { flex: 1; min-width: 0; }
}
```

> Drop the existing `<style>` block at the top of `waivers_trades.html` (lines 6-53) since these rules supersede the inline ones. The original block was layout-only per its own comment.

- [ ] **Step 5: Update `renderChip` to apply selected class**

In `renderChip` (line 517), update the class list to include `selected` when this chip is the current selection:

```javascript
    const isSelected = state.selection && state.selection.key === key;
    if (isSelected) cls.push("selected");
```

- [ ] **Step 6: Manual QA**

Re-run the dashboard. Build-a-Trade should now stack vertically with sticky team headers. Tap a player and confirm:
- The row gets the amber halo and the chip text bolds.
- The bottom strip appears with "Selected: <name> -- tap a slot to swap or move" and a cancel link.
- Tapping the same player again clears the halo and hides the strip.
- Tapping a slot on the other side moves the player; halo + strip clear.
- Resize the window to >=768px and confirm the two panels go side-by-side.

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html src/fantasy_baseball/web/static/season.css
git commit -m "feat(trade-ui): mobile-first stacked layout + selection styling"
```

---

## Task 9: Two-card result panel

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html` (the `renderResult` function around line 727)
- Modify: `src/fantasy_baseball/web/static/season.css`

- [ ] **Step 1: Replace `renderResult`**

In `waivers_trades.html` around line 727-754, replace the existing `renderResult` function:

```javascript
  function renderResult(data) {
    const el = document.getElementById("bt-result");
    if (!el) return;
    el.innerHTML =
      '<div id="bt-result-moves"></div>' +
      '<div class="bt-result-cards">' +
        '<div class="card bt-result-card bt-result-card-my">' +
          '<div class="bt-result-card-head">' +
            '<span class="bt-result-card-team">' + esc((window.BT_DATA && window.BT_DATA.myTeamName) || "My Team") + '</span>' +
            '<span id="bt-result-band-my"></span>' +
          '</div>' +
          '<div id="bt-result-my-deltas"></div>' +
        '</div>' +
        '<div class="card bt-result-card bt-result-card-opp">' +
          '<div class="bt-result-card-head">' +
            '<span class="bt-result-card-team">' + esc(state.opponent || "Opponent") + '</span>' +
            '<span id="bt-result-band-opp"></span>' +
          '</div>' +
          '<div id="bt-result-opp-deltas"></div>' +
        '</div>' +
      '</div>' +
      '<div id="bt-result-tagline" class="bt-result-tagline"></div>';

    const myTarget = document.getElementById("bt-result-my-deltas");
    const oppTarget = document.getElementById("bt-result-opp-deltas");
    if (data.band) {
      document.getElementById("bt-result-band-my").innerHTML = bandLineHTML(data.band, 'span', 'font-weight: bold;');
    }
    if (data.opp_band) {
      document.getElementById("bt-result-band-opp").innerHTML = bandLineHTML(data.opp_band, 'span', 'font-weight: bold;');
    }

    const myPayload = {
      roto: data.roto || {delta_total: 0, categories: {}},
      ev_roto: data.ev_roto || {delta_total: 0, categories: {}},
      stat_totals: data.stat_totals || {delta_total: 0, categories: {}},
    };
    const oppPayload = {
      roto: data.opp_roto || {delta_total: 0, categories: {}},
      ev_roto: data.opp_ev_roto || {delta_total: 0, categories: {}},
      stat_totals: data.opp_stat_totals || {delta_total: 0, categories: {}},
    };
    if (typeof window.renderThreeModeDeltaView === "function") {
      window.renderThreeModeDeltaView(myTarget, myPayload, {initialMode: "ev_roto"});
      window.renderThreeModeDeltaView(oppTarget, oppPayload, {initialMode: "ev_roto"});
    } else {
      myTarget.textContent = "Failed to load delta view component.";
    }

    // Tagline interprets the pair.
    const myPos = (data.delta_total || 0) > 0;
    const oppPos = (data.opp_delta_total || 0) > 0;
    const tag = document.getElementById("bt-result-tagline");
    if (myPos && oppPos) tag.textContent = "Both positive -- pitch this trade";
    else if (myPos && !oppPos) tag.textContent = "Good for you, bad for them -- unlikely to accept";
    else if (!myPos && oppPos) tag.textContent = "Bad for you, good for them -- skip";
    else tag.textContent = "Neither side wins -- skip";

    renderMovesSummary();
  }
```

- [ ] **Step 2: Add CSS for the two-card layout**

In `season.css`, append to `.page-trades`:

```css
.page-trades .bt-result-cards {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 12px;
}
.page-trades .bt-result-card {
    padding: 12px;
}
.page-trades .bt-result-card-my { border-left: 4px solid var(--success, #4d8); }
.page-trades .bt-result-card-opp { border-left: 4px solid var(--accent, #fc4); }
.page-trades .bt-result-card-head {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 8px;
    font-size: 13px;
    font-weight: 600;
}
.page-trades .bt-result-tagline {
    text-align: center;
    padding: 10px;
    color: var(--text-secondary);
    font-style: italic;
    font-size: 12px;
}

@media (min-width: 768px) {
    .page-trades .bt-result-cards {
        flex-direction: row;
    }
    .page-trades .bt-result-card { flex: 1; min-width: 0; }
}
```

- [ ] **Step 3: Manual QA**

Stage a 1-for-1 trade in the UI (tap a player on opp, tap a player on yours), click Evaluate. Confirm:
- Two cards render with team names and band lines.
- Each card has its own three-mode toggle (ev_roto / roto / stat totals).
- The tagline below interprets correctly (try both directions: a "+/+" trade and a "+/-" trade).

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html src/fantasy_baseball/web/static/season.css
git commit -m "feat(trade-ui): two-card result panel with opp deltaroto"
```

---

## Task 10: "Optimize Both" + side=both payload

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html` (button label at line 142-143 and `optimizeLineup` function around line 834)

- [ ] **Step 1: Rename button**

Find:

```html
    <button type="button" id="bt-optimize" class="pill" disabled
            onclick="bt.optimizeLineup()">Optimize Lineup</button>
```

Replace label only:

```html
    <button type="button" id="bt-optimize" class="pill" disabled
            onclick="bt.optimizeLineup()">Optimize Both</button>
```

- [ ] **Step 2: Send side=both and apply both-side slot updates**

In `optimizeLineup` (around line 834-887), the payload currently goes from `buildPayload()`. Add the `side` field:

```javascript
      const payload = buildPayload();
      payload.side = "both";
```

The existing code already applies `data.my_slots` and `data.opp_slots` (lines 875-883), so no further change to the apply loop. Update the in-flight label:

```javascript
        optBtn.textContent = "Optimizing both...";
```

Surface partial-failure if present (just below the existing apply loops, before `restoreButton()`):

```javascript
      if (data.partial) {
        alert("Couldn't optimize one side -- using their current arrangement.");
      }
```

- [ ] **Step 3: Manual QA**

Stage a trade, click Optimize Both. Confirm both panels' slot arrangements shift. If the server returns `partial: true`, confirm the alert fires.

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat(trade-ui): rename to Optimize Both, send side=both"
```

---

## Task 11: `buildPayload` includes `opp_active_ids`

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html` (`buildPayload` around line 702)

- [ ] **Step 1: Update buildPayload**

Find `buildPayload` (line 702). Add `oppActive` accumulation in the same loop that builds `myActive`:

```javascript
  function buildPayload() {
    const send = [], receive = [], myDrops = [], oppDrops = [], myAdds = [], myActive = [], oppActive = [];
    for (const [k, pl] of state.placements) {
      if (pl.zone === "DROP") {
        if (pl.origin === "my") myDrops.push(k);
        else if (pl.origin === "opp") oppDrops.push(k);
        continue;
      }
      if (pl.zone === "IL") continue;
      if (pl.currentSide === "opp" && pl.origin === "my") send.push(k);
      if (pl.currentSide === "my" && pl.origin === "opp") receive.push(k);
      if (pl.currentSide === "my" && pl.origin === "waiver") myAdds.push(k);
      if (pl.currentSide === "my" && pl.zone !== "BN") myActive.push(k);
      if (pl.currentSide === "opp" && pl.zone !== "BN") oppActive.push(k);
    }
    return {
      opponent: state.opponent,
      send, receive,
      my_drops: myDrops,
      opp_drops: oppDrops,
      my_adds: myAdds,
      my_active_ids: myActive,
      opp_active_ids: oppActive,
    };
  }
```

- [ ] **Step 2: Manual QA**

Stage a trade where you also bench an opp player by tapping them onto opp's bench zone. Click Evaluate. In devtools Network tab, confirm the request payload includes `opp_active_ids` with the benched player NOT in the array.

Verify in the response: `opp_stat_totals.categories.R.after` is lower than it would be if the benched player were still active (compare against the same trade without the bench by hitting Reset and re-evaluating).

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat(trade-ui): payload carries opp_active_ids"
```

---

## Task 12: Full verification + QA checklist

This is the FORCED VERIFICATION pass from `CLAUDE.md`. No code changes — just run all checks and walk the QA checklist before opening the PR.

- [ ] **Step 1: Run full test suite**

```bash
pytest -v
```

Expected: all PASS. If anything red is unrelated to this branch, call it out in the PR description.

- [ ] **Step 2: Run lint**

```bash
ruff check .
```

Expected: zero violations.

- [ ] **Step 3: Run format check**

```bash
ruff format --check .
```

Expected: no drift. If there's drift, run `ruff format .` and amend the most recent commit (or commit as a separate cleanup).

- [ ] **Step 4: Run vulture**

```bash
vulture
```

Expected: no NEW dead-code findings introduced by this branch. Pre-existing findings unrelated to trades are acceptable; note them in the PR description.

- [ ] **Step 5: Run mypy if covered**

Check whether any file in this branch's diff appears under `[tool.mypy].files` in `pyproject.toml`. If yes:

```bash
mypy
```

Expected: no NEW errors.

- [ ] **Step 6: Walk the QA checklist**

Run the dashboard (`python scripts/run_season_dashboard.py`), open the Trades page, expand Build a Trade, pick an opponent, and execute every step:

1. Tap player -> indicator strip shows their name; tap again -> cleared.
2. Tap player A (mine), tap player B (opp) -> swap committed, indicator clears.
3. Same-side tap-tap -> intra-team reshuffle.
4. Tap empty bench slot with selection -> move there.
5. Tap drops zone with selection -> drop committed.
6. Tap IL slot with selection -> no-op.
7. Tap waiver suggestion with active selection -> selection replaced.
8. Opponent dropdown change with active selection -> selection cleared, rosters rebuilt.
9. Evaluate with valid trade -> both team cards render with band + cat deltas.
10. Optimize Both -> both rosters update; press Evaluate again -> cards refresh.
11. Mobile viewport (375px via devtools) -- sticky headers, touch targets >=36px, bottom strip visible.
12. Desktop viewport (>=768px) -- side-by-side rosters and result cards.

Any failure becomes a follow-up commit on this branch.

- [ ] **Step 7: Push and open the PR**

```bash
git push -u origin feat/trade-analyzer-click-swap
gh pr create --base main --title "feat(trade-analyzer): click-to-swap UX + two-sided deltaroto" --body "$(cat <<'EOF'
Replaces SortableJS drag-drop in the Build-a-Trade panel with a click-to-select-and-swap state machine and adds the opponent's deltaroto so trades can be pitched to the other manager.

Design: `docs/superpowers/specs/2026-05-28-trade-analyzer-redesign-design.md`
Plan: `docs/superpowers/plans/2026-05-28-trade-analyzer-redesign.md`

QA checklist run -- all 12 items pass.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Self-Review

**Spec coverage:**

| Spec section | Tasks |
|--------------|-------|
| Click interaction model (state machine, transitions) | Task 7 (state + `tap`), Task 8 (selection styling) |
| Layout (stacked panels, sticky headers, selection strip, two-card result) | Task 8, Task 9 |
| Backend `MultiTradeResult` opp fields | Task 2 |
| Backend `TradeProposal.opp_active_ids` + fallback | Task 1, Task 3 |
| Backend opp views (roto/ev_roto/stat_totals/categories/band) | Task 4 |
| `/api/evaluate-trade` opp fields + opp_active_ids ingest | Task 5 |
| `/api/optimize-trade-lineup` side param + partial failure | Task 6 |
| `Optimize Both` rename + side=both payload | Task 10 |
| `opp_active_ids` in JS payload | Task 11 |
| Error handling (rapid taps, stale chips, IL no-op, dropdown clear, band failure fallback) | Task 4 (band try/except), Task 7 (IL no-op + key-based selection), Task 8 (renderPanels reset path) |
| Testing (backend pytest cases) | Tasks 1, 2, 3, 4, 5, 6 each include their own tests |
| QA checklist (frontend manual) | Task 12 |

No spec section is uncovered.

**Placeholder scan:** I searched the plan for TBD/TODO/XXX -- none found. The `placeholder asserts` in Task 4's regression-guard test are explicitly documented as a "pin on first green run" step; that's a deliberate procedure for a regression-guard test, not an unfilled gap.

**Type consistency:** Method/field names checked across tasks: `opp_active_ids` (Tasks 1, 3, 5, 11), `opp_delta_total` / `opp_categories` / `opp_roto` / `opp_ev_roto` / `opp_stat_totals` / `opp_band` (Tasks 2, 4, 5, 9), `tap()` / `state.selection` / `window.bt._tap` (Tasks 7, 8, 9), `side` param values `{"my","opp","both"}` (Tasks 6, 10). All consistent.
