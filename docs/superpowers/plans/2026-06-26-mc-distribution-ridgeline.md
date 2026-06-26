# MC Distribution Ridgeline View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Distributions" view to the season standings page that renders a ridgeline plot of each team's Monte Carlo outcome distribution -- overall roto points and per-category (raw totals + discrete roto-points) -- replacing the unhelpful Category Bars tab.

**Architecture:** `run_ros_monte_carlo()` already computes per-iteration samples and discards them. We retain them transiently, collapse them into compact KDE curves (continuous metrics) and exact PMFs (discrete category roto-points) via a new pure module, and ride them inside the existing `cache:monte_carlo` payload. A new formatter reshapes them for the template; a raw-canvas JS renderer draws the ridgeline. The old Category Bars code is deleted.

**Tech Stack:** Python 3.11, numpy (no scipy.stats -- hand-rolled Gaussian KDE), Flask + Jinja, raw HTML5 canvas 2D (no new JS deps).

## Global Constraints

- **ASCII-only** in all source, log messages, format strings, and report renderers (Windows cp1252 stdout). No true minus, sigma, em/en dash, smart quotes, arrows. Use `-`, `sigma`, `--`, `"`, `'`, `->`.
- **No `x or default` for numeric defaults.** Use `x if x is not None else default`. `0`/`0.0` are falsy.
- **numpy helper style:** functions returning arrays annotate `-> np.ndarray` and wrap the final return in `cast(np.ndarray, ...)` to satisfy mypy `warn_return_any` (see `_poisson_ppf_fast` in `simulation.py`). All values that reach the cache must be plain Python `float`/`int`/`list`/`dict` -- **no numpy types in the payload** (they break JSON serialization and the Redis round-trip).
- **Categories:** import `from fantasy_baseball.utils.constants import ALL_CATEGORIES`. Iterate enum members; use `c.value` for the string code (`"R"`, `"HR"`, ...). Canonical order: R, HR, RBI, SB, AVG, W, K, ERA, WHIP, SV. ERA/WHIP are inverse (lower is better).
- **Player/team identity:** teams are keyed by name at the simulation layer (that is the identifier there); the frontend keys highlight off a server-set `is_user` boolean, never off a name comparison.
- **Verification gates (per repo CLAUDE.md):** before declaring done, run `pytest -v` (or a stated subset), `ruff check .`, `ruff format --check .`, `vulture` (no NEW findings), and `mypy` if any touched file is under `[tool.mypy].files` in `pyproject.toml` (check the list). Paste output as evidence.
- **Spec:** `docs/superpowers/specs/2026-06-26-mc-distribution-ridgeline-design.md`. This plan implements it; if a conflict surfaces, the spec wins -- stop and reconcile.

## File Structure

- **Create** `src/fantasy_baseball/distributions.py` -- pure distribution-builder: Gaussian KDE helper, continuous-metric builder, discrete-PMF builder, and the `build_distributions()` orchestrator. numpy-only, no Flask/sim imports. One responsibility: turn per-iteration samples into compact JSON-ready curves.
- **Create** `tests/test_distributions.py` -- unit tests for the above.
- **Modify** `src/fantasy_baseball/simulation.py` -- `run_ros_monte_carlo()`: accumulate per-category points for all teams, derive `category_risk` from the user's slice (drop `user_cat_pts`), call `build_distributions()`, add `distributions` + `user_team` to the return.
- **Modify** `tests/test_simulation.py` -- assert the new `distributions` key/shape.
- **Modify** `src/fantasy_baseball/web/season_data.py` -- add `format_distributions_for_display()`; delete the Category Bars builders.
- **Modify** `tests/test_web/test_season_data.py` -- add formatter tests; delete Category Bars tests.
- **Modify** `src/fantasy_baseball/web/season_routes.py` -- wire `distributions` into the standings route; delete Category Bars plumbing.
- **Modify** `tests/test_web/test_season_routes.py` -- add a route embed test; delete Category Bars route tests.
- **Modify** `src/fantasy_baseball/web/templates/season/standings.html` -- add the Distributions view + selector + embedded JSON; delete the Category Bars view, its CSS, and the error-bars CDN/JS includes.
- **Create** `src/fantasy_baseball/web/static/season_distributions.js` -- the raw-canvas ridgeline renderer.
- **Delete** `src/fantasy_baseball/web/static/season_category_bars.js`.

---

## Task 1: Gaussian KDE + continuous-metric builder

**Files:**
- Create: `src/fantasy_baseball/distributions.py`
- Test: `tests/test_distributions.py`

**Interfaces:**
- Produces:
  - `GRID_POINTS: int = 60`, `BW_FLOOR_FRACTION: float = 0.01`
  - `_clean_samples(samples, sentinel: float | None) -> np.ndarray`
  - `_silverman_bandwidth(samples: np.ndarray) -> float`
  - `_gaussian_kde_curve(samples: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray`
  - `build_continuous_metric(team_samples: dict[str, "ArrayLike"], sentinel: float | None = None) -> dict` returning `{"x": list[float], "teams": {name: {"y": list[float], "median": float}}}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_distributions.py`:

```python
"""Unit tests for the MC distribution-builder module."""

import json

import numpy as np

from fantasy_baseball.distributions import (
    GRID_POINTS,
    _silverman_bandwidth,
    build_continuous_metric,
)


def test_continuous_metric_shared_grid_and_shape():
    rng = np.random.default_rng(0)
    team_samples = {
        "A": rng.normal(100.0, 8.0, 500),
        "B": rng.normal(130.0, 8.0, 500),
    }
    out = build_continuous_metric(team_samples)
    assert len(out["x"]) == GRID_POINTS
    assert set(out["teams"]) == {"A", "B"}
    for name in ("A", "B"):
        assert len(out["teams"][name]["y"]) == GRID_POINTS
    # x is the single shared grid (one list, every team sampled on it).
    assert isinstance(out["x"], list)
    # Medians track the samples.
    assert abs(out["teams"]["A"]["median"] - 100.0) < 5.0
    assert abs(out["teams"]["B"]["median"] - 130.0) < 5.0


def test_continuous_metric_density_integrates_to_one():
    rng = np.random.default_rng(1)
    out = build_continuous_metric({"A": rng.normal(50.0, 5.0, 800)})
    x = np.array(out["x"])
    y = np.array(out["teams"]["A"]["y"])
    assert abs(float(np.trapezoid(y, x)) - 1.0) < 0.05


def test_continuous_metric_is_json_serializable_plain_floats():
    rng = np.random.default_rng(2)
    out = build_continuous_metric({"A": rng.normal(0.0, 1.0, 100)})
    json.dumps(out)  # raises TypeError if any numpy types leaked
    assert isinstance(out["x"][0], float)
    assert isinstance(out["teams"]["A"]["y"][0], float)
    assert isinstance(out["teams"]["A"]["median"], float)


def test_near_constant_input_is_finite_and_normalized():
    # One near-deterministic team plus a spread team (so pooled range > 0 and
    # the metric-relative bandwidth floor is positive). A near-constant team
    # SHOULD render tight; the contract here is "finite, normalized, no NaN",
    # not artificial width.
    samples = {
        "tight": np.full(200, 50.0),
        "wide": np.linspace(40.0, 60.0, 200),
    }
    out = build_continuous_metric(samples)
    y = np.array(out["teams"]["tight"]["y"])
    assert np.all(np.isfinite(y))
    x = np.array(out["x"])
    assert abs(float(np.trapezoid(y, x)) - 1.0) < 0.1
    # Peak sits at the constant value.
    assert abs(float(x[int(np.argmax(y))]) - 50.0) < 2.0


def test_sentinel_values_are_dropped_before_grid():
    # ERA-like samples with a few 99.0 zero-IP sentinels; they must not stretch
    # the grid or add mass near 99.
    samples = {"A": np.array([3.1, 3.2, 3.3, 3.0, 3.4, 99.0, 99.0])}
    out = build_continuous_metric(samples, sentinel=99.0)
    assert max(out["x"]) < 10.0  # 99 dropped, grid stays near the real data


def test_silverman_bandwidth_zero_for_constant_input():
    assert _silverman_bandwidth(np.full(10, 5.0)) == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'fantasy_baseball.distributions'`

- [ ] **Step 3: Write minimal implementation**

Create `src/fantasy_baseball/distributions.py`:

```python
"""Compact outcome-distribution curves for the Monte Carlo standings ridgeline.

``run_ros_monte_carlo`` retains per-iteration samples transiently; these helpers
collapse them into small KDE curves (continuous metrics) and exact PMFs (discrete
category roto-points) that are cheap to cache and ship to the browser. Every
output is plain Python floats -- JSON-serializable, no numpy types in the payload.
"""

from typing import Any, cast

import numpy as np

# Grid resolution and the metric-relative bandwidth floor. Tunable; see the spec
# "Open questions" -- these are visual knobs, not data-contract values.
GRID_POINTS = 60
BW_FLOOR_FRACTION = 0.01  # floor = 1% of a metric's pooled (post-sentinel) range


def _clean_samples(samples: Any, sentinel: float | None) -> np.ndarray:
    """Drop non-finite values (and an optional sentinel) from a sample array.

    ERA/WHIP carry a ``99.0`` zero-IP sentinel from the batch simulation; left in,
    it would stretch the grid and paint a phantom tail near 99.
    """
    arr = np.asarray(samples, dtype=float)
    arr = arr[np.isfinite(arr)]
    if sentinel is not None:
        arr = arr[arr != sentinel]
    return arr


def _silverman_bandwidth(samples: np.ndarray) -> float:
    """Silverman's rule-of-thumb bandwidth; 0.0 for <2 points or zero spread."""
    n = samples.size
    if n < 2:
        return 0.0
    std = float(np.std(samples, ddof=1))
    q75, q25 = np.percentile(samples, [75, 25])
    iqr = float(q75 - q25)
    spread = min(std, iqr / 1.349) if iqr > 0 else std
    return 0.9 * spread * n ** (-0.2)


def _gaussian_kde_curve(samples: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    """Gaussian KDE of ``samples`` sampled on ``grid``, normalized to integrate ~1.

    ``bw <= 0`` (a degenerate, zero-variance team) collapses to a single spike at
    the median rather than dividing by zero. Defensive: ``build_continuous_metric``
    always applies a positive metric-relative floor, so the real callers never pass
    ``bw <= 0`` -- this guard only protects direct/standalone use of the helper.
    """
    if bw <= 0.0:
        y = np.zeros_like(grid)
        y[int(np.argmin(np.abs(grid - float(np.median(samples)))))] = 1.0
        return cast(np.ndarray, y)
    z = (grid[:, None] - samples[None, :]) / bw
    dens = np.exp(-0.5 * z * z).sum(axis=1) / (samples.size * bw * np.sqrt(2.0 * np.pi))
    area = float(np.trapezoid(dens, grid))  # np.trapz removed in numpy 2.0+
    if area > 0.0:
        dens = dens / area
    return cast(np.ndarray, dens)


def build_continuous_metric(team_samples: dict[str, Any], sentinel: float | None = None) -> dict:
    """Build a shared-grid KDE ridgeline payload for one continuous metric.

    ``team_samples`` is ``{team: sample_array}``. Returns
    ``{"x": [...], "teams": {team: {"y": [...], "median": float}}}`` where ``x`` is
    one grid shared by every team (so ridgeline rows are horizontally comparable),
    bandwidth is per-team (Silverman with a metric-relative floor), and the grid is
    padded by ``3 * bw_max`` so no team's tails clip. Teams with no usable samples
    are omitted.
    """
    cleaned = {}
    for name, raw in team_samples.items():
        arr = _clean_samples(raw, sentinel)
        if arr.size > 0:
            cleaned[name] = arr
    if not cleaned:
        return {"x": [], "teams": {}}

    pooled = np.concatenate(list(cleaned.values()))
    lo = float(pooled.min())
    hi = float(pooled.max())
    span = hi - lo
    bw_floor = max(1e-9, BW_FLOOR_FRACTION * span)
    bws = {name: max(_silverman_bandwidth(arr), bw_floor) for name, arr in cleaned.items()}
    bw_max = max(bws.values())
    grid = np.linspace(lo - 3.0 * bw_max, hi + 3.0 * bw_max, GRID_POINTS)

    teams = {}
    for name, arr in cleaned.items():
        y = _gaussian_kde_curve(arr, grid, bws[name])
        teams[name] = {
            "y": [float(v) for v in y],
            "median": float(np.median(arr)),
        }
    return {"x": [float(v) for v in grid], "teams": teams}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distributions.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/distributions.py tests/test_distributions.py
git commit -m "feat(distributions): Gaussian KDE + continuous-metric ridgeline builder"
```

---

## Task 2: Discrete-PMF builder

**Files:**
- Modify: `src/fantasy_baseball/distributions.py`
- Test: `tests/test_distributions.py`

**Interfaces:**
- Consumes: nothing new.
- Produces: `build_discrete_metric(team_samples: dict[str, "ArrayLike"]) -> dict` returning `{"x": list[float], "teams": {name: {"p": list[float], "mean": float}}}` -- `x` is the shared sorted union of observed point values; each team's `p` aligns to it (zeros at unobserved), sums to 1.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_distributions.py`:

```python
from fantasy_baseball.distributions import build_discrete_metric


def test_discrete_metric_shared_support_and_pmf():
    team_samples = {
        "A": [11, 11, 12, 12, 12],
        "B": [1, 2, 2, 3],
    }
    out = build_discrete_metric(team_samples)
    # Shared x = sorted union of observed values across BOTH teams.
    assert out["x"] == [1.0, 2.0, 3.0, 11.0, 12.0]
    for name in ("A", "B"):
        p = out["teams"][name]["p"]
        assert len(p) == len(out["x"])
        assert abs(sum(p) - 1.0) < 1e-9
    # A never realized 1/2/3 -> zeros there.
    assert out["teams"]["A"]["p"][:3] == [0.0, 0.0, 0.0]
    # mean = sum(x * p): A is (11*2 + 12*3)/5 = 11.6
    assert abs(out["teams"]["A"]["mean"] - 11.6) < 1e-9


def test_discrete_metric_half_integer_support_from_ties():
    # A tie produces a 0.5-step point value; it must appear in the shared support.
    out = build_discrete_metric({"A": [11.5, 11.5, 12.0], "B": [1.0, 1.0, 1.0]})
    assert 11.5 in out["x"]
    assert out["x"] == sorted(out["x"])


def test_discrete_metric_json_serializable():
    out = build_discrete_metric({"A": [1, 2, 3], "B": [3, 3, 3]})
    json.dumps(out)
    assert isinstance(out["x"][0], float)
    assert isinstance(out["teams"]["A"]["p"][0], float)
    assert isinstance(out["teams"]["A"]["mean"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributions.py -k discrete -v`
Expected: FAIL with `ImportError: cannot import name 'build_discrete_metric'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/fantasy_baseball/distributions.py`:

```python
def build_discrete_metric(team_samples: dict[str, Any]) -> dict:
    """Build a shared-support PMF ridgeline payload for one discrete metric.

    ``team_samples`` is ``{team: point_value_array}`` (category roto points, which
    are half-integers under tie-splitting). Returns
    ``{"x": [...], "teams": {team: {"p": [...], "mean": float}}}`` where ``x`` is
    the sorted union of distinct point values observed across ALL teams and each
    team's ``p`` is aligned to that shared ``x`` (0 at unobserved values), so a
    ridgeline can stack the rows on one axis. Teams with no samples are omitted.
    """
    cleaned = {}
    for name, raw in team_samples.items():
        arr = _clean_samples(raw, None)
        if arr.size > 0:
            # Snap to the nearest 0.5 so tie-split values compare exactly.
            cleaned[name] = np.round(arr * 2.0) / 2.0
    if not cleaned:
        return {"x": [], "teams": {}}

    support = np.unique(np.concatenate(list(cleaned.values())))
    teams = {}
    for name, arr in cleaned.items():
        counts = np.array([np.count_nonzero(arr == v) for v in support], dtype=float)
        p = counts / counts.sum()
        teams[name] = {
            "p": [float(v) for v in p],
            "mean": float(np.sum(support * p)),
        }
    return {"x": [float(v) for v in support], "teams": teams}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distributions.py -v`
Expected: PASS (9 tests total)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/distributions.py tests/test_distributions.py
git commit -m "feat(distributions): shared-support discrete PMF builder"
```

---

## Task 3: `build_distributions` orchestrator

**Files:**
- Modify: `src/fantasy_baseball/distributions.py`
- Test: `tests/test_distributions.py`

**Interfaces:**
- Consumes: `build_continuous_metric`, `build_discrete_metric`.
- Produces: `build_distributions(all_totals: dict[str, list[float]], batch: dict[str, dict[str, np.ndarray]], all_cat_pts: dict[str, dict[str, list[float]]], cats: list[str], user_team: str) -> dict` returning `{"overall", "category_totals", "category_points", "user_team"}` exactly as the spec's return shape.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_distributions.py`:

```python
from fantasy_baseball.distributions import build_distributions


def test_build_distributions_shape_and_serializable():
    rng = np.random.default_rng(7)
    cats = ["R", "ERA"]
    teams = ["A", "B"]
    all_totals = {t: list(rng.normal(80.0, 6.0, 100)) for t in teams}
    batch = {
        "A": {"R": rng.normal(800.0, 30.0, 100), "ERA": rng.normal(3.5, 0.2, 100)},
        "B": {"R": rng.normal(750.0, 30.0, 100), "ERA": rng.normal(3.8, 0.2, 100)},
    }
    all_cat_pts = {
        "A": {"R": [10] * 100, "ERA": [9] * 100},
        "B": {"R": [2] * 100, "ERA": [3] * 100},
    }
    out = build_distributions(all_totals, batch, all_cat_pts, cats, user_team="A")

    assert set(out) == {"overall", "category_totals", "category_points", "user_team"}
    assert out["user_team"] == "A"
    # Overall covers all teams.
    assert set(out["overall"]["teams"]) == {"A", "B"}
    # One entry per category, all teams present.
    assert set(out["category_totals"]) == {"R", "ERA"}
    assert set(out["category_points"]) == {"R", "ERA"}
    assert set(out["category_totals"]["R"]["teams"]) == {"A", "B"}
    assert set(out["category_points"]["ERA"]["teams"]) == {"A", "B"}
    # No numpy types anywhere.
    json.dumps(out)


def test_build_distributions_drops_era_sentinel():
    cats = ["ERA"]
    batch = {"A": {"ERA": np.array([3.1, 3.2, 99.0, 3.0, 99.0, 3.3])}}
    out = build_distributions(
        all_totals={"A": [50.0, 51.0, 52.0]},
        batch=batch,
        all_cat_pts={"A": {"ERA": [8, 8, 9]}},
        cats=cats,
        user_team="A",
    )
    assert max(out["category_totals"]["ERA"]["x"]) < 10.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_distributions.py -k build_distributions -v`
Expected: FAIL with `ImportError: cannot import name 'build_distributions'`

- [ ] **Step 3: Write minimal implementation**

Append to `src/fantasy_baseball/distributions.py`:

```python
# ERA/WHIP carry a 99.0 zero-IP sentinel from simulate_remaining_season_batch.
_SENTINEL_CATS = {"ERA", "WHIP"}
_SENTINEL_VALUE = 99.0


def build_distributions(
    all_totals: dict[str, list[float]],
    batch: dict[str, dict[str, np.ndarray]],
    all_cat_pts: dict[str, dict[str, list[float]]],
    cats: list[str],
    user_team: str,
) -> dict:
    """Assemble the full ``distributions`` payload from the MC's transient arrays.

    - ``overall``: KDE of each team's total roto points (``all_totals``).
    - ``category_totals``: KDE of each team's raw stat total per category
      (``batch``); ERA/WHIP drop the 99.0 sentinel.
    - ``category_points``: exact PMF of each team's roto points per category
      (``all_cat_pts``).
    ``user_team`` is carried through for the formatter to mark ``is_user``.
    """
    overall = build_continuous_metric({name: np.asarray(v, dtype=float) for name, v in all_totals.items()})

    category_totals = {}
    category_points = {}
    for cat in cats:
        sentinel = _SENTINEL_VALUE if cat in _SENTINEL_CATS else None
        category_totals[cat] = build_continuous_metric(
            {name: batch[name][cat] for name in batch}, sentinel=sentinel
        )
        category_points[cat] = build_discrete_metric(
            {name: all_cat_pts[name][cat] for name in all_cat_pts}
        )

    return {
        "overall": overall,
        "category_totals": category_totals,
        "category_points": category_points,
        "user_team": user_team,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_distributions.py -v`
Expected: PASS (11 tests total)

- [ ] **Step 5: Verify lint/format/types on the new module**

Run: `ruff check src/fantasy_baseball/distributions.py tests/test_distributions.py && ruff format --check src/fantasy_baseball/distributions.py tests/test_distributions.py`
Expected: no violations. Then check whether `distributions.py` is under `[tool.mypy].files` in `pyproject.toml`; if the mypy file list uses globs that include `src/fantasy_baseball/`, run `mypy src/fantasy_baseball/distributions.py` and fix any findings.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/distributions.py tests/test_distributions.py
git commit -m "feat(distributions): build_distributions orchestrator"
```

---

## Task 4: Retain distributions in `run_ros_monte_carlo`

**Files:**
- Modify: `src/fantasy_baseball/simulation.py:929-988`
- Test: `tests/test_simulation.py`

**Interfaces:**
- Consumes: `build_distributions` (Task 3).
- Produces: `run_ros_monte_carlo(...)` return dict gains `"distributions"` (the Task 3 shape) and the `category_risk` derivation now reads the user's slice of an all-team `all_cat_pts` accumulator. `team_results` and `category_risk` values are unchanged.

**Notes:** `run_ros_monte_carlo` is at `simulation.py:891-988`. The per-iteration loop is 949-963; `all_totals` (all teams) is built at 956; `user_cat_pts` (user only) at 961-963; `category_risk` at 977-986. `batch` (from `simulate_remaining_season_batch`, `{team: {cat_str: ndarray}}`) is in scope from line 945. `cats = [c.value for c in ALL_CATS]` exists at 936.

**CRITICAL -- non-unique edit anchors.** `run_monte_carlo` (lines 807-888) contains **byte-identical** copies of the `user_cat_pts` init, the accumulation loop, the `category_risk` block, and the `return {...}` line that this task edits. A bare `old_string` for any of them matches **twice** and the Edit tool will refuse (or worse, edit the wrong function). Every edit below is anchored on content that is **unique to `run_ros_monte_carlo`**: Step 4 includes the unique "Vectorized: one batched simulation" comment; Step 5 is one contiguous replacement starting at the unique `sim_stats = {... batch[name][cat][i] ...}` line. Before each edit, `git grep -n` the first line of your `old_string` to confirm it appears exactly once.

- [ ] **Step 1: Write the failing test**

First read the existing `run_ros_monte_carlo` test in `tests/test_simulation.py` (`test_returns_expected_format`, ~lines 495-509) to confirm the exact fixture helpers and call convention. They are: helper functions `_build_two_team_rosters()` and `_build_actual_standings()` (call them inline), with literals `fraction_remaining=0.5, h_slots=3, p_slots=2, user_team_name="Team A"`, and exactly **two** teams ("Team A", "Team B"). Reuse those (do not fabricate a new roster shape). Append:

```python
def test_run_ros_monte_carlo_returns_distributions():
    # Reuse the same fixtures as test_returns_expected_format.
    result = run_ros_monte_carlo(
        team_rosters=_build_two_team_rosters(),
        actual_standings=_build_actual_standings(),
        fraction_remaining=0.5,
        h_slots=3,
        p_slots=2,
        user_team_name="Team A",
        n_iterations=50,
        seed=1,
    )
    dist = result["distributions"]
    team_names = set(result["team_results"])
    assert team_names == {"Team A", "Team B"}

    # Documented top-level shape.
    assert set(dist) == {"overall", "category_totals", "category_points", "user_team"}
    assert dist["user_team"] == "Team A"

    # overall + every category cover all teams (category_points populated for ALL
    # teams, not just the user -- guards the all-team accumulation change).
    assert set(dist["overall"]["teams"]) == team_names
    for cat in ("R", "HR", "RBI", "SB", "AVG", "W", "K", "ERA", "WHIP", "SV"):
        assert set(dist["category_points"][cat]["teams"]) == team_names
        assert set(dist["category_totals"][cat]["teams"]) == team_names

    # Whole payload survives the cache round-trip (no numpy types).
    import json

    json.dumps(result)

    # category_risk preserved (same keys, same fields as before this change).
    assert set(result["category_risk"]) == {
        "R", "HR", "RBI", "SB", "AVG", "W", "K", "ERA", "WHIP", "SV"
    }
    assert "top3_pct" in result["category_risk"]["R"]
```

(If the helper names differ from `_build_two_team_rosters` / `_build_actual_standings` when you read the file, substitute the real names -- but they are functions called inline, not module constants.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulation.py::test_run_ros_monte_carlo_returns_distributions -v`
Expected: FAIL with `KeyError: 'distributions'`

- [ ] **Step 3: Add the import**

At the top of `simulation.py`, add the import in isort order. First-party `fantasy_baseball.*` imports are alphabetical: `distributions` sorts **before** `models.player` (line 14), so insert it immediately **before** the `from fantasy_baseball.models.player import PlayerType` line:

```python
from fantasy_baseball.distributions import build_distributions
from fantasy_baseball.models.player import PlayerType
```

(Placing it after `scoring` on line 15 would be out of order and `ruff check` will flag the `I` rule.)

- [ ] **Step 4: Replace the user-only accumulator with an all-team one**

In `run_ros_monte_carlo`, replace the `user_cat_pts` init. To disambiguate from the identical line in `run_monte_carlo`, the `old_string` includes the following `cats` line and the unique "Vectorized" comment block. Replace:

```python
    user_cat_pts: dict[str, list[float]] = {c.value: [] for c in ALL_CATS}
    cats = [c.value for c in ALL_CATS]

    # Vectorized: one batched simulation of all iterations replaces the former
    # per-iteration simulate_remaining_season call. Roto scoring stays per
    # iteration (it ranks teams within each draw), reading column i from the batch.
    # The batch below is the heavy step (the scoring loop is cheap), so signal the
    # MC phase before it rather than letting the prior step's message linger.
```

with:

```python
    all_cat_pts: dict[str, dict[str, list[float]]] = {
        name: {c.value: [] for c in ALL_CATS} for name in team_names
    }
    cats = [c.value for c in ALL_CATS]

    # Vectorized: one batched simulation of all iterations replaces the former
    # per-iteration simulate_remaining_season call. Roto scoring stays per
    # iteration (it ranks teams within each draw), reading column i from the batch.
    # The batch below is the heavy step (the scoring loop is cheap), so signal the
    # MC phase before it rather than letting the prior step's message linger.
```

- [ ] **Step 5: One contiguous edit -- all-team accumulation, derived category_risk, distributions, return**

This single replacement starts at the `sim_stats = {... batch[name][cat][i] ...}` line, which is **unique to `run_ros_monte_carlo`** (it reads `batch`; `run_monte_carlo` does not), so the whole span is unambiguous. The middle `team_results` block is unchanged but included verbatim so the `old_string` matches. Replace:

```python
        sim_stats = {name: {cat: float(batch[name][cat][i]) for cat in cats} for name in team_names}
        sim_roto = score_roto_dict(sim_stats)
        ranked = sorted(sim_roto.items(), key=lambda x: x[1]["total"], reverse=True)
        for rank, (name, pts) in enumerate(ranked, 1):
            all_totals[name].append(pts["total"])
            if rank == 1:
                mc_wins[name] += 1
            if rank <= 3:
                mc_top3[name] += 1
            if name == user_team_name:
                for c in ALL_CATS:
                    user_cat_pts[c.value].append(pts.get(f"{c.value}_pts", 0))

    n = n_iterations
    team_results = {}
    for name in team_names:
        arr = np.array(all_totals[name])
        team_results[name] = {
            "median_pts": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10))),
            "p90": round(float(np.percentile(arr, 90))),
            "first_pct": round(mc_wins[name] / n * 100, 1),
            "top3_pct": round(mc_top3[name] / n * 100, 1),
        }

    category_risk = {}
    for c in ALL_CATS:
        arr = np.array(user_cat_pts[c.value])
        category_risk[c.value] = {
            "median_pts": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "top3_pct": round(float((arr >= 8).sum()) / n * 100, 1),
            "bot3_pct": round(float((arr <= 3).sum()) / n * 100, 1),
        }

    return {"team_results": team_results, "category_risk": category_risk}
```

with:

```python
        sim_stats = {name: {cat: float(batch[name][cat][i]) for cat in cats} for name in team_names}
        sim_roto = score_roto_dict(sim_stats)
        ranked = sorted(sim_roto.items(), key=lambda x: x[1]["total"], reverse=True)
        for rank, (name, pts) in enumerate(ranked, 1):
            all_totals[name].append(pts["total"])
            if rank == 1:
                mc_wins[name] += 1
            if rank <= 3:
                mc_top3[name] += 1
            team_cat_pts = all_cat_pts[name]
            for c in ALL_CATS:
                team_cat_pts[c.value].append(pts.get(f"{c.value}_pts", 0))

    n = n_iterations
    team_results = {}
    for name in team_names:
        arr = np.array(all_totals[name])
        team_results[name] = {
            "median_pts": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10))),
            "p90": round(float(np.percentile(arr, 90))),
            "first_pct": round(mc_wins[name] / n * 100, 1),
            "top3_pct": round(mc_top3[name] / n * 100, 1),
        }

    category_risk = {}
    # The user's slice of the all-team accumulator. .get fallback preserves the
    # prior soft-degrade (empty arrays -> nan) if the user team is absent from the
    # rosters, rather than a hard KeyError.
    user_cat_pts = all_cat_pts.get(user_team_name, {c.value: [] for c in ALL_CATS})
    for c in ALL_CATS:
        arr = np.array(user_cat_pts[c.value])
        category_risk[c.value] = {
            "median_pts": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "top3_pct": round(float((arr >= 8).sum()) / n * 100, 1),
            "bot3_pct": round(float((arr <= 3).sum()) / n * 100, 1),
        }

    distributions = build_distributions(
        all_totals, batch, all_cat_pts, cats, user_team_name
    )

    return {
        "team_results": team_results,
        "category_risk": category_risk,
        "distributions": distributions,
    }
```

- [ ] **Step 6: Run the new test + the existing simulation suite**

Run: `pytest tests/test_simulation.py -v`
Expected: PASS, including the new test and all pre-existing `run_ros_monte_carlo` tests (category_risk values unchanged).

- [ ] **Step 7: Lint/format/types**

Run: `ruff check src/fantasy_baseball/simulation.py && ruff format --check src/fantasy_baseball/simulation.py`
Expected: no violations. `simulation.py` **is** under `[tool.mypy].files` -- run `mypy src/fantasy_baseball/simulation.py` and fix findings (the `build_distributions` import is followed; the typed helper signatures from Tasks 1-3 keep this clean, but verify, do not assume).

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/simulation.py tests/test_simulation.py
git commit -m "feat(sim): retain compact outcome distributions in run_ros_monte_carlo"
```

---

## Task 5: `format_distributions_for_display`

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py` (add a new function; mirror `format_monte_carlo_for_display` at 671-718)
- Test: `tests/test_web/test_season_data.py`

**Interfaces:**
- Consumes: the `distributions` dict from Task 4.
- Produces: `format_distributions_for_display(distributions: dict | None) -> dict` returning:
  ```
  {
    "overall": {"x": [...], "rows": [{"team", "is_user", "y", "median"}, ... sorted]},
    "category_totals": {CAT: {"x": [...], "rows": [{"team","is_user","y","median"}]}},
    "category_points": {CAT: {"x": [...], "rows": [{"team","is_user","p","mean"}]}},
  }
  ```
  Rows sorted best-on-top: by `median`/`mean` descending, except ERA/WHIP `category_totals` ascending (lower raw is better). `is_user` set from the payload's `user_team`; the raw `user_team` string is dropped. Missing/empty input -> all three keys present with empty `rows`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_web/test_season_data.py`:

```python
from fantasy_baseball.web.season_data import format_distributions_for_display


def _distributions_payload():
    return {
        "overall": {
            "x": [10.0, 20.0],
            "teams": {
                "Me": {"y": [0.1, 0.2], "median": 90.0},
                "Rival": {"y": [0.2, 0.1], "median": 95.0},
            },
        },
        "category_totals": {
            "R": {
                "x": [1.0, 2.0],
                "teams": {
                    "Me": {"y": [0.3, 0.1], "median": 800.0},
                    "Rival": {"y": [0.1, 0.3], "median": 760.0},
                },
            },
            "ERA": {
                "x": [3.0, 4.0],
                "teams": {
                    "Me": {"y": [0.2, 0.2], "median": 3.50},
                    "Rival": {"y": [0.2, 0.2], "median": 3.20},
                },
            },
        },
        "category_points": {
            "R": {
                "x": [1.0, 2.0],
                "teams": {
                    "Me": {"p": [0.0, 1.0], "mean": 2.0},
                    "Rival": {"p": [1.0, 0.0], "mean": 1.0},
                },
            },
        },
        "user_team": "Me",
    }


def test_format_distributions_marks_is_user_and_drops_user_team():
    out = format_distributions_for_display(_distributions_payload())
    assert "user_team" not in out
    me = next(r for r in out["overall"]["rows"] if r["team"] == "Me")
    assert me["is_user"] is True
    rival = next(r for r in out["overall"]["rows"] if r["team"] == "Rival")
    assert rival["is_user"] is False


def test_format_distributions_sorts_overall_by_median_desc():
    out = format_distributions_for_display(_distributions_payload())
    teams = [r["team"] for r in out["overall"]["rows"]]
    assert teams == ["Rival", "Me"]  # 95 before 90


def test_format_distributions_era_sorts_ascending_best_on_top():
    out = format_distributions_for_display(_distributions_payload())
    # ERA lower is better -> Rival (3.20) on top of Me (3.50).
    teams = [r["team"] for r in out["category_totals"]["ERA"]["rows"]]
    assert teams == ["Rival", "Me"]


def test_format_distributions_points_sorts_by_mean_desc():
    out = format_distributions_for_display(_distributions_payload())
    teams = [r["team"] for r in out["category_points"]["R"]["rows"]]
    assert teams == ["Me", "Rival"]  # mean 2.0 before 1.0
    assert out["category_points"]["R"]["rows"][0]["mean"] == 2.0


def test_format_distributions_handles_none():
    out = format_distributions_for_display(None)
    assert out == {"overall": {"x": [], "rows": []}, "category_totals": {}, "category_points": {}}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_data.py -k distributions -v`
Expected: FAIL with `ImportError: cannot import name 'format_distributions_for_display'`

- [ ] **Step 3: Write minimal implementation**

Add to `src/fantasy_baseball/web/season_data.py` (near `format_monte_carlo_for_display`; `INVERSE_CATS` is already imported at the top of the file).

IMPORTANT: `INVERSE_CATS` is a `frozenset` of `Category` **enum members** (`{Category.ERA, Category.WHIP}`), but the distributions payload keys categories by their **string** code (`"ERA"`). `"ERA" in INVERSE_CATS` is always `False`. Compare against a string set built once:

```python
# Payload category keys are strings (c.value); INVERSE_CATS holds enum members.
_INVERSE_CAT_VALUES = {c.value for c in INVERSE_CATS}


def _distribution_rows(metric: dict, user_team: str, value_key: str, sort_key: str, ascending: bool) -> dict:
    """Reshape one metric's ``teams`` map into sorted, is_user-marked rows."""
    rows = [
        {
            "team": name,
            "is_user": name == user_team,
            value_key: entry[value_key],
            sort_key: entry[sort_key],
        }
        for name, entry in metric.get("teams", {}).items()
    ]
    rows.sort(key=lambda r: r[sort_key], reverse=not ascending)
    return {"x": metric.get("x", []), "rows": rows}


def format_distributions_for_display(distributions: dict | None) -> dict:
    """Reshape the MC ``distributions`` payload into a template-ready ridgeline dict.

    Marks each row ``is_user`` server-side (dropping the raw ``user_team`` string)
    and sorts rows best-on-top: by ``median``/``mean`` descending, except ERA/WHIP
    raw totals ascending (lower is better). Mirrors ``format_*_for_display``.
    """
    empty = {"overall": {"x": [], "rows": []}, "category_totals": {}, "category_points": {}}
    if not distributions or "overall" not in distributions:
        return empty

    user_team = distributions.get("user_team", "")
    overall = _distribution_rows(distributions["overall"], user_team, "y", "median", ascending=False)

    category_totals = {}
    for cat, metric in distributions.get("category_totals", {}).items():
        category_totals[cat] = _distribution_rows(
            metric, user_team, "y", "median", ascending=cat in _INVERSE_CAT_VALUES
        )

    category_points = {}
    for cat, metric in distributions.get("category_points", {}).items():
        category_points[cat] = _distribution_rows(
            metric, user_team, "p", "mean", ascending=False
        )

    return {
        "overall": overall,
        "category_totals": category_totals,
        "category_points": category_points,
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_web/test_season_data.py -k distributions -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py
git commit -m "feat(season-data): format_distributions_for_display ridgeline formatter"
```

---

## Task 6: Wire distributions into the standings route

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py:567-593`
- Test: `tests/test_web/test_season_routes.py`

**Interfaces:**
- Consumes: `format_distributions_for_display` (Task 5).
- Produces: the standings route passes `distributions=` to the template. The read happens inside the existing `if raw_mc.get("rest_of_season"):` guard; the module-scope default covers the no-MC path.

**Notes:** The route renders the full-page embed node only after Task 8 adds it to the template. So Task 6's test asserts the `distributions` **kwarg** reaches `render_template` (by patching it), which fails-first and passes after this task -- independent of Task 8. The full-page `#distributions-data` embed-content assertion lives in Task 8.

- [ ] **Step 1: Write the failing test**

Model the cache seeding + config patching on the existing `test_standings_passes_baseline_meta_to_template` (~lines 767-791), which seeds `CacheKey.MONTE_CARLO` with a `rest_of_season` blob and GETs `/standings` under a patched `_load_config().team_name`. (Do NOT model on the category-bars tests -- those seed `PROJECTIONS`, the wrong cache key.) Append:

```python
from unittest.mock import patch


def test_standings_passes_distributions_to_template(...):  # reuse the harness's client + cache fixtures
    # Seed CacheKey.MONTE_CARLO with rest_of_season carrying team_results,
    # category_risk, AND a distributions block (overall + one category) whose
    # user_team matches the patched config team name. Then:
    with patch("fantasy_baseball.web.season_routes.render_template", return_value="") as rendered:
        client.get("/standings")
    dist = rendered.call_args.kwargs["distributions"]
    assert "overall" in dist
    assert dist["overall"]["rows"]
    assert any(r["is_user"] for r in dist["overall"]["rows"])  # is_user marked server-side
    assert "user_team" not in dist  # raw string dropped


def test_standings_distributions_empty_without_mc(...):
    # No MONTE_CARLO cache seeded -> the module-scope default empty-state reaches
    # the template; the route must not crash.
    with patch("fantasy_baseball.web.season_routes.render_template", return_value="") as rendered:
        client.get("/standings")
    dist = rendered.call_args.kwargs["distributions"]
    assert dist == {"overall": {"x": [], "rows": []}, "category_totals": {}, "category_points": {}}
```

(Fill the client/cache-seeding/config-patch wiring from `test_standings_passes_baseline_meta_to_template`. `render_template` is imported into `season_routes` from flask, so patch it at `fantasy_baseball.web.season_routes.render_template`.)

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py -k distributions -v`
Expected: FAIL (no `distributions-data` node in the rendered page).

- [ ] **Step 3: Add the import at module scope**

The module-scope default in Step 4 calls `format_distributions_for_display` **outside** the lazy `if raw_standings:` import block, so it must be imported at module top -- NOT lazily (a lazy-only import would `NameError`). `season_data` is already imported at module top (lines 27-34); add the new name to that block in isort order (after `format_category_bars_for_display`):

```python
from fantasy_baseball.web.season_data import (
    CacheKey,
    coerce_basis,
    format_category_bars_for_display,
    format_distributions_for_display,
    read_cache_dict,
    read_cache_list,
    read_meta,
)
```

(`format_category_bars_for_display` is removed from this block in Task 7; `format_distributions_for_display` stays.)

- [ ] **Step 4: Default the local, then read distributions inside the ROS guard**

Add the module-scope default near the other `None` inits (after line 514 `raw_breakdown = None`) -- now resolvable because of Step 3's module-top import:

```python
        distributions = format_distributions_for_display(None)
```

Then populate it inside the existing ROS guard. Change lines 567-575 from:

```python
            raw_mc = read_cache_dict(CacheKey.MONTE_CARLO)
            if raw_mc:
                baseline_meta = raw_mc.get("baseline_meta")
                if raw_mc.get("base"):
                    mc_data = format_monte_carlo_for_display(raw_mc["base"], config.team_name)
                if raw_mc.get("rest_of_season"):
                    rest_of_season_mc_data = format_monte_carlo_for_display(
                        raw_mc["rest_of_season"], config.team_name
                    )
```

to:

```python
            raw_mc = read_cache_dict(CacheKey.MONTE_CARLO)
            if raw_mc:
                baseline_meta = raw_mc.get("baseline_meta")
                if raw_mc.get("base"):
                    mc_data = format_monte_carlo_for_display(raw_mc["base"], config.team_name)
                if raw_mc.get("rest_of_season"):
                    rest_of_season_mc_data = format_monte_carlo_for_display(
                        raw_mc["rest_of_season"], config.team_name
                    )
                    # rest_of_season may predate this feature; .get() yields None
                    # and the formatter returns the empty-state shape.
                    distributions = format_distributions_for_display(
                        raw_mc["rest_of_season"].get("distributions")
                    )
```

(The read is safely inside `if raw_mc.get("rest_of_season"):`, so `raw_mc["rest_of_season"]` is never `None` at the `.get("distributions")` call.)

- [ ] **Step 5: Pass it to the template**

In the `render_template("season/standings.html", ...)` call, add a kwarg alongside the existing ones (e.g. after `rest_of_season_mc=rest_of_season_mc_data,` at line 589):

```python
            distributions=distributions,
```

- [ ] **Step 6: Run test to verify it passes**

Run: `pytest tests/test_web/test_season_routes.py -k distributions -v`
Expected: PASS (both tests). These patch `render_template` and assert on the `distributions` kwarg, so they do NOT depend on Task 8's template node. The full-page `#distributions-data` embed-content assertion is added in Task 8.

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py tests/test_web/test_season_routes.py
git commit -m "feat(season-routes): pass distributions to the standings template"
```

---

## Task 7: Delete Category Bars (backend + frontend + tests, atomic)

**Why atomic:** the route's `category_bars` kwarg and the template's `{{ category_bars }}` reference must disappear together, and the `format_category_bars_for_display` function and its route call must disappear together -- otherwise `/standings` 500s in the intermediate state. So all Category Bars surface (backend, frontend, tests) is removed in this one task. This runs **before** the additive Distributions view (Task 8); in between, the standings page simply has three tabs (Current/Projected/Monte Carlo) and renders cleanly. The route already passes `distributions=` (Task 6); with no template node yet, that kwarg is harmlessly unused.

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py` (delete line 13, lines 416-485)
- Modify: `src/fantasy_baseball/web/season_routes.py` (delete lines 30, 577, 590)
- Modify: `src/fantasy_baseball/web/templates/season/standings.html` (delete the nav button, view block, toggle branch, CSS, error-bars CDN)
- Delete: `src/fantasy_baseball/web/static/season_category_bars.js`
- Modify: `tests/test_web/test_season_data.py`, `tests/test_web/test_season_routes.py` (delete category-bars tests)

**Interfaces:** none produced; removes dead surface. `INVERSE_CATS` import in `season_data.py` STAYS (used at line 1183). All `format_monte_carlo_for_display` and `format_distributions_for_display` references STAY.

- [ ] **Step 1: Delete the season_data.py functions and orphaned import**

Delete the `category_finish_odds` import line 13 (`from fantasy_baseball.category_odds import category_finish_odds`) and the three contiguous functions at lines 416-485 (`_category_odds`, `_category_bars_one_flavor`, `format_category_bars_for_display`). Do NOT touch the `INVERSE_CATS` import (lines 30-32) -- it is still used at line 1183. Keep the standalone `category_odds` module and `tests/test_category_odds.py`.

- [ ] **Step 2: Delete the season_routes.py plumbing**

Remove `format_category_bars_for_display,` from the module-top import block (line 30 -- leave `format_distributions_for_display` added in Task 6), the `category_bars = format_category_bars_for_display(...)` line (577), and the `category_bars=category_bars,` render kwarg (590).

- [ ] **Step 3: Delete the template Category Bars surface**

In `src/fantasy_baseball/web/templates/season/standings.html`, delete: the Category Bars nav button (line 24, `data-view="categorybars"`); the entire `#view-categorybars` block (234-261, including its `category-bars-data` JSON node and `catbars-*` sub-toggles); the `categorybars` line(s) in `toggleTopView` (353-354, the `style.display` line and the `if (v === 'categorybars' && window.renderCategoryBars)` line); the category-bars CSS rules (664-682: `.catbars-wrapper`, `#catbars-cat-toggle`, `#catbars-cat-toggle .pill:first-child`, `.catbars-odds`, `.catbars-odds strong`); the `chartjs-chart-error-bars` CDN include (703); and the `season_category_bars.js` include (704).

- [ ] **Step 4: Decide the Chart.js CDN line**

The `chart.js` CDN (line 702) was used on this page only by Category Bars (`scatterWithErrorBars`); the Task-8 renderer uses raw canvas. Grep for remaining Chart.js usage in the template:

Run: `git grep -n -e "new Chart" -e "Chart(" -e "chart.js" -- src/fantasy_baseball/web/templates/season/standings.html`
Expected: zero hits. If zero, delete the Chart.js CDN line (702) too (no dead includes, per repo convention). If any hit remains, keep it.

- [ ] **Step 5: Delete the JS file**

`git rm src/fantasy_baseball/web/static/season_category_bars.js`

- [ ] **Step 6: Delete the tests**

In `tests/test_web/test_season_data.py`: remove the `format_category_bars_for_display,` import (line 19), the `_bars_display_dict` helper (2497-2519), and the 7 category-bars test functions (2522-2608). In `tests/test_web/test_season_routes.py`: remove `test_standings_category_bars_empty_without_projections` (1364-1388) and `test_standings_embeds_category_bars_data` (1391-1483). (Lift any reusable cache-seeding helper into the Task-6 tests rather than duplicating.)

- [ ] **Step 7: Grep to confirm nothing dangles**

Run: `git grep -n -e category_bars -e catbars -e _category_odds -e category_finish_odds -e season_category_bars -e renderCategoryBars -- src/ tests/`
Expected: zero hits except the independent standalone `category_odds` module and `tests/test_category_odds.py` (which stay). No hits in `season_data.py`, `season_routes.py`, the template, or the JS static dir.

- [ ] **Step 8: Run the affected suites + vulture, then commit**

Run: `pytest tests/test_web/ -v && vulture src/fantasy_baseball/web/season_data.py`
Expected: PASS (no import errors; `/standings` still renders -- the page now has three tabs), no NEW vulture findings.

```bash
git add -A src/fantasy_baseball/web/ tests/test_web/
git rm src/fantasy_baseball/web/static/season_category_bars.js
git commit -m "refactor(web): delete Category Bars (replaced by Distributions)"
```

---

## Task 8: Distributions view -- template + canvas renderer (additive)

Category Bars was fully removed in Task 7, so every step here is purely additive -- there is nothing to "replace."

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/standings.html` (nav button, new `#view-distributions` block, `toggleTopView` hook, script include)
- Create: `src/fantasy_baseball/web/static/season_distributions.js`
- Modify: `tests/test_web/test_season_routes.py` (full-page embed test, Step 7)

**Interfaces:**
- Consumes: the `distributions` template var (Task 6) and the `all_categories` template var (already passed by the route).
- Produces: a working Distributions tab.

- [ ] **Step 1: Add the nav button**

In `standings.html`, add a Distributions button to the `#standings-top-toggle` group (the group now ends at the Monte Carlo button, since the Category Bars button was removed in Task 7). Add it as the last button before the closing `</div>`:

```html
    <button class="pill" data-view="distributions" onclick="toggleTopView(this)">Distributions</button>
```

- [ ] **Step 2: Add the view block + embedded JSON**

Insert a new `#view-distributions` block (model placement/markup on `#view-montecarlo` at 144-232). Put it right after the Monte Carlo view block:

```html
{# -- Distributions (ridgeline of per-team MC outcome spreads) -- #}
<div id="view-distributions" style="display: none;">
    <div class="pill-group" id="dist-metric-toggle">
        <button class="pill active" data-distmetric="overall" onclick="distSetMetric(this)">Overall</button>
        {% for cat in all_categories %}
        <button class="pill" data-distmetric="{{ cat.value }}" onclick="distSetMetric(this)">{{ cat.value }}</button>
        {% endfor %}
    </div>

    <div class="pill-group" id="dist-mode-toggle" style="display: none;">
        <button class="pill active" data-distmode="totals" onclick="distSetMode(this)">Totals</button>
        <button class="pill" data-distmode="points" onclick="distSetMode(this)">Points</button>
    </div>

    <div class="dist-wrapper"><canvas id="distributions-canvas"></canvas></div>
    <p id="dist-empty" class="placeholder-text" style="display: none;">
        No Monte Carlo distributions available. Click "Refresh Data" first.
    </p>

    <script type="application/json" id="distributions-data">{{ distributions | tojson }}</script>
</div>
```

- [ ] **Step 3: Add the CSS**

In the page's `<style>` block (the one that ran ~638-701 before the catbars rules were removed in Task 7), add:

```css
.dist-wrapper { position: relative; height: 520px; margin-top: 0.75rem; }
#dist-metric-toggle {
    display: grid;
    grid-template-columns: repeat(6, max-content);
    column-gap: 6px;
    row-gap: 4px;
    border-bottom: none;
}
```

- [ ] **Step 4: Hook the render-on-show in `toggleTopView`**

In `toggleTopView` (the `categorybars` branch was removed in Task 7), add a distributions branch before the closing `}`:

```javascript
    document.getElementById('view-distributions').style.display = v === 'distributions' ? '' : 'none';
    if (v === 'distributions' && window.renderDistributions) window.renderDistributions();
```

- [ ] **Step 5: Add the script include**

Add the renderer include at the bottom of the content block where the page's other `<script src=...>` includes live (the Category Bars include and error-bars CDN were removed in Task 7):

```html
<script src="{{ url_for('static', filename='season_distributions.js') }}"></script>
```

- [ ] **Step 6: Write the renderer**

Create `src/fantasy_baseball/web/static/season_distributions.js`:

```javascript
/* Distributions: ridgeline of per-team Monte Carlo outcome distributions for the
 * standings "Distributions" view. Reads the JSON embedded by standings.html
 * (#distributions-data) and draws into #distributions-canvas with the raw 2D
 * canvas API -- one density row per team on a shared x-axis, user row highlighted,
 * a central-tendency tick per row. Re-renders on metric/mode change and when the
 * tab is shown (the canvas has zero size while its view is display:none).
 *
 * Formatted data shape (from format_distributions_for_display):
 *   { overall: {x:[...], rows:[{team,is_user,y:[...],median}]},
 *     category_totals: {CAT: {x:[...], rows:[{team,is_user,y:[...],median}]}},
 *     category_points: {CAT: {x:[...], rows:[{team,is_user,p:[...],mean}]}} }
 */
(function () {
  "use strict";

  var USER_COLOR = "#e15759";
  var OTHER_COLOR = "#4e79a7";
  var RATE_CATS = { AVG: 3, ERA: 2, WHIP: 2 };

  var state = { metric: "overall", mode: "totals" };
  var payload = null;

  function loadPayload() {
    var node = document.getElementById("distributions-data");
    if (!node) return null;
    try { return JSON.parse(node.textContent); } catch (e) { return null; }
  }

  function fmtTick(value) {
    var cat = state.metric;
    if (state.metric !== "overall" && state.mode === "totals" && RATE_CATS[cat] != null) {
      return value.toFixed(RATE_CATS[cat]);
    }
    return String(Math.round(value * 10) / 10);
  }

  // Resolve the active metric into a uniform shape:
  // {x:[...], rows:[{team,is_user,curve:[...],center:float}], discrete:bool, label}
  function currentMetric() {
    if (!payload) return null;
    if (state.metric === "overall") {
      return adapt(payload.overall, "y", "median", false, "Total roto points");
    }
    var cat = state.metric;
    if (state.mode === "points") {
      var cp = (payload.category_points || {})[cat];
      return adapt(cp, "p", "mean", true, cat + " roto points");
    }
    var ct = (payload.category_totals || {})[cat];
    return adapt(ct, "y", "median", false, cat + " total");
  }

  function adapt(metric, curveKey, centerKey, discrete, label) {
    if (!metric || !metric.rows || !metric.rows.length) return null;
    return {
      x: metric.x,
      discrete: discrete,
      label: label,
      rows: metric.rows.map(function (r) {
        return { team: r.team, is_user: r.is_user, curve: r[curveKey], center: r[centerKey] };
      })
    };
  }

  function showEmpty(canvas, empty, isEmpty) {
    if (canvas) canvas.style.display = isEmpty ? "none" : "";
    if (empty) empty.style.display = isEmpty ? "" : "none";
  }

  function render() {
    if (payload == null) payload = loadPayload();
    var canvas = document.getElementById("distributions-canvas");
    var empty = document.getElementById("dist-empty");
    if (!canvas) return;

    var data = currentMetric();
    if (!data) { showEmpty(canvas, empty, true); return; }
    showEmpty(canvas, empty, false);

    // Size the backing store to the CSS box * devicePixelRatio for crisp lines.
    var dpr = window.devicePixelRatio || 1;
    var cssW = canvas.clientWidth || canvas.parentNode.clientWidth || 800;
    var cssH = canvas.clientHeight || canvas.parentNode.clientHeight || 520;
    canvas.width = Math.round(cssW * dpr);
    canvas.height = Math.round(cssH * dpr);
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, cssW, cssH);

    drawRidgeline(ctx, cssW, cssH, data);
  }

  function drawRidgeline(ctx, W, H, data) {
    var padL = 110, padR = 24, padT = 16, padB = 34;
    var plotW = W - padL - padR;
    var plotH = H - padT - padB;
    var rows = data.rows;
    var n = rows.length;

    var xs = data.x;
    var xMin = xs[0], xMax = xs[xs.length - 1];
    var xSpan = xMax - xMin || 1;
    function sx(v) { return padL + ((v - xMin) / xSpan) * plotW; }

    // Per-row vertical band; curves overlap upward by ~1.7 bands for the
    // classic ridgeline look. Peak height is normalized per row (each row's own
    // max), so a tight row reads as tall-and-narrow, a wide row as low-and-broad.
    var band = plotH / n;
    var overlap = 1.7;

    ctx.font = "12px system-ui, sans-serif";
    ctx.textBaseline = "middle";

    // x-axis label.
    ctx.fillStyle = "#888";
    ctx.textAlign = "center";
    ctx.fillText(data.label, padL + plotW / 2, H - 12);

    // Draw back-to-front (bottom rows first) so upper rows overlap correctly.
    for (var i = n - 1; i >= 0; i--) {
      var row = rows[i];
      var baseY = padT + (i + 1) * band;
      var curve = row.curve;
      var cMax = 0;
      for (var k = 0; k < curve.length; k++) if (curve[k] > cMax) cMax = curve[k];
      if (cMax <= 0) cMax = 1;
      var amp = band * overlap;

      function cy(idx) { return baseY - (curve[idx] / cMax) * amp; }

      var stroke = row.is_user ? USER_COLOR : OTHER_COLOR;
      var fill = row.is_user ? "rgba(225,87,89,0.35)" : "rgba(78,121,167,0.22)";

      if (data.discrete) {
        // Stems at each support value.
        ctx.strokeStyle = stroke;
        ctx.lineWidth = row.is_user ? 2.5 : 1.5;
        for (var s = 0; s < xs.length; s++) {
          if (curve[s] <= 0) continue;
          var px = sx(xs[s]);
          ctx.beginPath();
          ctx.moveTo(px, baseY);
          ctx.lineTo(px, cy(s));
          ctx.stroke();
        }
      } else {
        // Filled density path.
        ctx.beginPath();
        ctx.moveTo(sx(xs[0]), baseY);
        for (var j = 0; j < xs.length; j++) ctx.lineTo(sx(xs[j]), cy(j));
        ctx.lineTo(sx(xs[xs.length - 1]), baseY);
        ctx.closePath();
        ctx.fillStyle = fill;
        ctx.fill();
        ctx.strokeStyle = stroke;
        ctx.lineWidth = row.is_user ? 2.5 : 1.5;
        ctx.stroke();
      }

      // Central-tendency tick (median for continuous, mean for discrete).
      var tx = sx(row.center);
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(tx, baseY);
      ctx.lineTo(tx, baseY - amp);
      ctx.stroke();
      ctx.setLineDash([]);

      // Row label (team name, bold for user) + center value.
      ctx.fillStyle = row.is_user ? USER_COLOR : "#ccc";
      ctx.font = (row.is_user ? "bold " : "") + "12px system-ui, sans-serif";
      ctx.textAlign = "right";
      ctx.fillText(row.team, padL - 8, baseY - band * 0.35);
      ctx.fillStyle = "#888";
      ctx.font = "11px system-ui, sans-serif";
      ctx.fillText(fmtTick(row.center), padL - 8, baseY - band * 0.35 + 14);
    }
  }

  window.renderDistributions = render;

  function setActivePill(groupSelector, stateKey, dataAttr, el) {
    document.querySelectorAll(groupSelector + " .pill").forEach(function (p) {
      p.classList.remove("active");
    });
    el.classList.add("active");
    state[stateKey] = el.dataset[dataAttr];
  }

  window.distSetMetric = function (el) {
    setActivePill("#dist-metric-toggle", "metric", "distmetric", el);
    // Totals|Points only applies to a specific category, not Overall.
    var modeToggle = document.getElementById("dist-mode-toggle");
    if (modeToggle) modeToggle.style.display = state.metric === "overall" ? "none" : "";
    render();
  };

  window.distSetMode = function (el) {
    setActivePill("#dist-mode-toggle", "mode", "distmode", el);
    render();
  };

  // Re-render on resize so the canvas stays crisp and correctly sized.
  window.addEventListener("resize", function () {
    if (document.getElementById("view-distributions") &&
        document.getElementById("view-distributions").style.display !== "none") {
      render();
    }
  });
})();
```

- [ ] **Step 7: Add the full-page embed-content test (now that the node exists)**

The `#distributions-data` node exists only now, so add the full-page embed assertion here (deferred from Task 6). Append to `tests/test_web/test_season_routes.py`, modeling cache seeding on `test_standings_passes_baseline_meta_to_template` (seed `MONTE_CARLO` with a `rest_of_season` carrying a `distributions` block whose `user_team` matches the patched config team name):

```python
def test_standings_embeds_distributions_node(...):  # reuse the harness's client + cache fixtures
    body = client.get("/standings").get_data(as_text=True)
    match = re.search(
        r'<script type="application/json" id="distributions-data">(.*?)</script>',
        body,
        re.DOTALL,
    )
    assert match is not None, "distributions-data script tag not found"
    dist = json.loads(match.group(1))
    assert dist["overall"]["rows"]
    assert any(r["is_user"] for r in dist["overall"]["rows"])
    assert "user_team" not in dist
```

Run: `pytest tests/test_web/test_season_routes.py -k distributions -v`
Expected: PASS (this test plus the two from Task 6).

- [ ] **Step 8: Manual smoke check (no unit test for canvas draw)**

The ridgeline draw is not unit-tested (consistent with existing chart JS) -- it is verified visually in Task 9's local refresh. For now, confirm the template renders without a Jinja error by exercising the route once (the Step 7 test already does this) and confirm the JS file has no syntax error (e.g. `node --check src/fantasy_baseball/web/static/season_distributions.js` if node is available; otherwise eyeball the braces).

- [ ] **Step 9: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/standings.html src/fantasy_baseball/web/static/season_distributions.js tests/test_web/test_season_routes.py
git commit -m "feat(web): Distributions ridgeline view (template + canvas renderer)"
```

---

## Task 9: Full verification + local refresh

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run: `pytest -n auto`
Expected: all pass. If any pre-existing test references the deleted Category Bars or the old `user_cat_pts`, investigate per the repo's "don't modify failing tests without justification" rule -- a failure here likely means a missed reference, not a wrong test.

- [ ] **Step 2: Lint, format, dead code**

Run: `ruff check . && ruff format --check . && vulture`
Expected: zero ruff violations, no formatting drift, no NEW vulture findings (pre-existing unrelated findings are acceptable -- call them out).

- [ ] **Step 3: Types**

`[tool.mypy].files` is an explicit file list (not globs). `simulation.py`, `season_data.py`, and `season_routes.py` ARE in it (mypy mandatory); `distributions.py` is NOT in the list -- but `simulation.py` imports it, so mypy follows it. Run `mypy` over the configured set (or at least `mypy src/fantasy_baseball/simulation.py src/fantasy_baseball/web/season_data.py src/fantasy_baseball/web/season_routes.py`) and fix findings. Do not assume clean -- verify.

- [ ] **Step 4: Local refresh -- the real data populates the cache**

Per the repo's "run refresh before merge" rule, exercise the refresh path locally so the new `distributions` actually get computed and cached, then eyeball the view:

Run: `python scripts/run_season_dashboard.py --no-sync`
(`--no-sync` so it does not wipe local SQLite via Upstash sync while verifying not-yet-deployed code.) Trigger a refresh, open `/standings`, click the **Distributions** tab, and verify: Overall ridgeline renders with your row highlighted; the category pills switch metrics; the Totals|Points toggle appears for a category and swaps the curve for stems; ERA/WHIP have no phantom spike near 99; no console errors.

- [ ] **Step 5: Final evidence**

Paste the outputs of Steps 1-3 (or concise summaries) and a one-line confirmation of Step 4's visual check into the completion message. Do not claim done without this evidence.

---

## Self-Review

**Spec coverage:**
- Retain compact distributions (overall/category_totals/category_points) -> Tasks 1-4.
- Per-team bandwidth, shared grid padded by 3*bw_max, metric-relative floor, sentinel drop -> Task 1 + Task 3.
- Discrete shared-support PMF, half-integer ties -> Task 2.
- All-team category-points accumulation + derive category_risk, drop user_cat_pts -> Task 4.
- `format_distributions_for_display` (is_user server-side, drop user_team, sort incl. ERA/WHIP ascending via value-string set) -> Task 5.
- Route plumbing inside the rest_of_season guard, absence empty-state, module-top import -> Task 6.
- Delete Category Bars (backend + frontend + tests + CDN + CSS + JS file), atomic -> Task 7.
- Ridgeline view + selector + Totals|Points + canvas render -> Task 8.
- Tests for KDE/builder/discrete/integration/formatter/route -> Tasks 1-6, 8.
- Verification gates + local refresh -> Task 9.
- Deferred (preseason toggle, p10-p90 band, points-swing annotations) -> intentionally absent. Confirmed.

**Ordering:** Task 7 (atomic Category Bars deletion) runs before Task 8 (additive Distributions add); between them the page renders three tabs cleanly. No task removes a route kwarg whose template reference still exists, or a function whose call site still exists -- those pairs are deleted together in Task 7.

**Type consistency:** `build_distributions` (Task 3) is called in Task 4 with `(all_totals, batch, all_cat_pts, cats, user_team_name)` -- matches its signature. `format_distributions_for_display` returns the `{overall, category_totals, category_points}` shape consumed by the JS `adapt()` (`y`/`median` for continuous, `p`/`mean` for discrete) in Task 8 -- consistent. Embedded node id `distributions-data` matches between Task 8 (template), the Task 6/Task 8 route tests, and the JS `loadPayload()`.

**Environment-specific:** numpy 2.4.4 -- use `np.trapezoid` (not the removed `np.trapz`). `INVERSE_CATS` holds `Category` enum members, so the formatter compares against `{c.value for c in INVERSE_CATS}` (string keys). `run_monte_carlo` (807-888) duplicates the blocks Task 4 edits, so Task 4 anchors on the unique `batch`/`sim_stats`/comment lines. `simulation.py`/`season_data.py`/`season_routes.py` are under mypy; `distributions.py` is not (but is imported by `simulation.py`, so its typed signatures matter).

**Placeholder scan:** Task 4's test uses the real fixture **functions** `_build_two_team_rosters()` / `_build_actual_standings()` (called inline, 2 teams, user "Team A"); Task 6's route test reuses `test_standings_passes_baseline_meta_to_template`'s MONTE_CARLO-seeding harness. Both name a concrete reuse target, not an unfilled placeholder. All code steps contain complete code.
