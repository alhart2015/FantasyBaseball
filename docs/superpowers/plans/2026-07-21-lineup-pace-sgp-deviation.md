# Lineup SGP-Deviation Pace Coloring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-21-lineup-pace-sgp-deviation-design.md`

**Goal:** Replace the lineup page's overall-pace color (equal-weight mean of per-category z-scores) with a leaguewide percentile of each player's "SGP delivered vs preseason-expected" deviation, hitters ranked among hitters and pitchers among pitchers.

**Architecture:** A new pure metric (`compute_sgp_deviation`) sums per-category `actual_sgp - expected_sgp` (expected = preseason projection prorated to actual playing time). A new refresh-pipeline step computes this for every rostered player, derives tercile/sextile cutpoints per player-type, and caches both under `CacheKey.PACE_DEVIATIONS`. The reshaped `compute_overall_pace` buckets a player's cached deviation against those cutpoints; the three `season_data` display sites and the two lineup templates consume the new shape.

**Tech Stack:** Python 3.11+, pytest, Flask/Jinja2 templates, the existing SGP/denominator + pace modules.

## Global Constraints

- **ASCII-only** in all source, log strings, and template text that may hit `print()` (Windows cp1252 stdout). Use `-`, `sigma`, `--`, straight quotes.
- **Player IDs are `name::player_type`**; keys in the deviations map use `normalize_name(name)::player_type_value`.
- **Display only:** this metric must not feed roster/trade/waiver/MC/standings logic.
- **Numeric defaults:** never `x or default` for numbers; use explicit `is None` checks or `dict.get(k, default)`.
- **Do not modify per-category cell coloring** (`_z_to_color` and the per-stat `z_score`/`color_class`) - only the overall Slot color changes.
- **Final gate:** `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` for any touched file in `[tool.mypy].files`. Show outputs.

---

### Task 1: SGP-deviation metric (pure)

**Files:**
- Modify: `src/fantasy_baseball/analysis/pace.py` (add imports, add `_prorated_expected`, refactor `compute_player_pace` to use it, add `compute_sgp_deviation`)
- Test: `tests/test_analysis/test_sgp_deviation.py` (create)

**Interfaces:**
- Produces:
  - `_prorated_expected(proj: float, actual_opp: float, proj_opp: float) -> float`
  - `compute_sgp_deviation(actual_stats: dict[str, Any], projected_stats: dict[str, Any], player_type: str, denoms: dict[Category, float]) -> dict[str, Any]` returning `{"sgp_dev": float | None, "actual_sgp": float | None, "expected_sgp": float | None}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis/test_sgp_deviation.py`:

```python
import pytest

from fantasy_baseball.analysis.pace import compute_sgp_deviation
from fantasy_baseball.utils.constants import Category

# Explicit denominators so expected values are hand-computable.
DENOMS = {
    Category.R: 10.0,
    Category.HR: 5.0,
    Category.RBI: 10.0,
    Category.SB: 5.0,
    Category.AVG: 0.0015,
    Category.W: 2.0,
    Category.K: 12.0,
    Category.SV: 3.5,
    Category.ERA: 0.10,
    Category.WHIP: 0.03,
}


def test_hitter_full_deviation():
    # 100/500 PA -> proration factor 0.2. Overperforming across the board.
    actual = {"pa": 100, "ab": 90, "h": 27, "r": 25, "hr": 8, "rbi": 30, "sb": 6}
    projected = {"pa": 500, "ab": 450, "avg": 0.280, "r": 100, "hr": 30, "rbi": 100, "sb": 20}
    out = compute_sgp_deviation(actual, projected, "hitter", DENOMS)
    # counting dev: R (25-20)/10=.5, HR (8-6)/5=.4, RBI (30-20)/10=1.0, SB (6-4)/5=.4 -> 2.3
    # AVG dev: (0.300-0.280)*90/(0.0015*5500) = 1.8/8.25 = 0.2182
    assert out["sgp_dev"] == pytest.approx(2.518, abs=0.005)
    assert out["actual_sgp"] == pytest.approx(8.845, abs=0.005)
    assert out["expected_sgp"] == pytest.approx(6.327, abs=0.005)
    # replacement cancels in the delta
    assert out["sgp_dev"] == pytest.approx(out["actual_sgp"] - out["expected_sgp"], abs=1e-6)


def test_hitter_rate_gated_out_below_30_pa():
    # 20 PA: counting colored (>=10), AVG NOT colored (<30) -> AVG excluded.
    actual = {"pa": 20, "ab": 18, "h": 8, "r": 10, "hr": 4, "rbi": 12, "sb": 2}
    projected = {"pa": 200, "ab": 180, "avg": 0.300, "r": 100, "hr": 40, "rbi": 100, "sb": 20}
    out = compute_sgp_deviation(actual, projected, "hitter", DENOMS)
    # factor 0.1: R exp10 act10 ->0, HR exp4 act4 ->0, RBI exp10 act12 ->0.2, SB exp2 act2 ->0
    assert out["sgp_dev"] == pytest.approx(0.2, abs=1e-6)
    # actual_sgp counts only R/HR/RBI/SB (no AVG term): 1.0+0.8+1.2+0.4 = 3.4
    assert out["actual_sgp"] == pytest.approx(3.4, abs=1e-6)


def test_below_counting_gate_returns_none():
    actual = {"pa": 5, "ab": 4, "h": 1, "r": 1, "hr": 0, "rbi": 1, "sb": 0}
    projected = {"pa": 500, "ab": 450, "avg": 0.280, "r": 100, "hr": 30, "rbi": 100, "sb": 20}
    out = compute_sgp_deviation(actual, projected, "hitter", DENOMS)
    assert out["sgp_dev"] is None


def test_no_projection_returns_none():
    actual = {"pa": 100, "ab": 90, "h": 27, "r": 25, "hr": 8, "rbi": 30, "sb": 6}
    out = compute_sgp_deviation(actual, {}, "hitter", DENOMS)
    assert out["sgp_dev"] is None


def test_pitcher_outperforming_positive_dev():
    # Lower ERA/WHIP than projected -> positive deviation (inverse stats).
    actual = {"ip": 50, "k": 60, "w": 5, "sv": 0, "er": 15, "bb": 12, "h_allowed": 38}
    projected = {"ip": 180, "k": 200, "w": 12, "sv": 0, "era": 3.50, "whip": 1.10}
    out = compute_sgp_deviation(actual, projected, "pitcher", DENOMS)
    # actual ERA 2.70 < 3.50, actual WHIP 1.00 < 1.10, K/W ahead of pace
    assert out["sgp_dev"] > 0


def test_pitcher_underperforming_negative_dev():
    actual = {"ip": 50, "k": 30, "w": 1, "sv": 0, "er": 35, "bb": 25, "h_allowed": 60}
    projected = {"ip": 180, "k": 200, "w": 12, "sv": 0, "era": 3.50, "whip": 1.10}
    out = compute_sgp_deviation(actual, projected, "pitcher", DENOMS)
    # actual ERA 6.30 > 3.50, actual WHIP 1.70 > 1.10, K/W behind pace
    assert out["sgp_dev"] < 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_sgp_deviation.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_sgp_deviation'`

- [ ] **Step 3: Add imports and the proration helper to `pace.py`**

In `src/fantasy_baseball/analysis/pace.py`, extend the constants import (currently `HITTER_PROJ_KEYS, INVERSE_STATS, PITCHER_PROJ_KEYS, Category`) to also import the team-volume constants, and add the replacement-rate import:

```python
from fantasy_baseball.utils.constants import (
    DEFAULT_TEAM_AB,
    DEFAULT_TEAM_IP,
    HITTER_PROJ_KEYS,
    INVERSE_STATS,
    PITCHER_PROJ_KEYS,
    Category,
)
from fantasy_baseball.sgp.player_value import (
    REPLACEMENT_AVG,
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
)
```

Add the shared proration helper near the top (after `_z_to_color`):

```python
def _prorated_expected(proj: float, actual_opp: float, proj_opp: float) -> float:
    """Projected counting stat scaled to actual playing time so far.

    Returns 0.0 when the projection has no opportunity (proj_opp <= 0) or
    projects zero of the stat, matching the guard compute_player_pace uses.
    """
    if proj_opp > 0 and proj > 0:
        return proj * (actual_opp / proj_opp)
    return 0.0
```

- [ ] **Step 4: Refactor `compute_player_pace` to use the helper**

In `compute_player_pace`, replace the inline proration:

```python
        if proj_opp > 0 and proj > 0:
            expected = proj * (actual_opp / proj_opp)
        else:
            expected = 0.0
```

with:

```python
        expected = _prorated_expected(proj, actual_opp, proj_opp)
```

- [ ] **Step 5: Add `compute_sgp_deviation`**

Append to `pace.py`:

```python
def compute_sgp_deviation(
    actual_stats: dict[str, Any],
    projected_stats: dict[str, Any],
    player_type: str,
    denoms: dict[Category, float],
) -> dict[str, Any]:
    """SGP delivered vs preseason-expected, prorated to actual playing time.

    Returns {"sgp_dev", "actual_sgp", "expected_sgp"} in roto-point (SGP)
    units. ``sgp_dev`` is None when the player has no games above the counting
    gate or no projection. Counting categories use ``stat / denom``; rate
    categories (AVG, ERA/WHIP) use the same marginal-value-over-actual-volume
    formulas as ``sgp.player_value`` (the innings divisor cancels for ERA/WHIP).
    The replacement baseline cancels in the delta but is kept in the returned
    ``actual_sgp`` / ``expected_sgp`` so the tooltip reads as value-over-
    replacement. Sample-size gates mirror ``compute_player_pace``.
    """
    none_result = {"sgp_dev": None, "actual_sgp": None, "expected_sgp": None}
    if not projected_stats:
        return none_result

    if player_type == PlayerType.HITTER:
        opp_key, counting = "pa", HITTER_COUNTING
        min_counting, min_rates = HITTER_MIN_COUNTING, HITTER_MIN_RATES
    else:
        opp_key, counting = "ip", PITCHER_COUNTING
        min_counting, min_rates = PITCHER_MIN_COUNTING, PITCHER_MIN_RATES

    actual_opp = actual_stats.get(opp_key, 0) or 0
    proj_opp = projected_stats.get(opp_key, 0) or 0
    if actual_opp < min_counting:
        return none_result

    actual_sgp = 0.0
    expected_sgp = 0.0

    for stat in counting:
        cat = Category(stat.upper())
        denom = denoms.get(cat)
        if not denom:
            continue
        actual = actual_stats.get(stat, 0) or 0
        proj = projected_stats.get(stat, 0) or 0
        expected = _prorated_expected(proj, actual_opp, proj_opp)
        actual_sgp += actual / denom
        expected_sgp += expected / denom

    if actual_opp >= min_rates:
        if player_type == PlayerType.HITTER:
            actual_ab = actual_stats.get("ab", 0) or 0
            denom = denoms.get(Category.AVG)
            if denom and actual_ab > 0:
                actual_avg = calculate_avg(
                    actual_stats.get("h", 0) or 0, actual_ab, default=0.0
                )
                proj_avg = projected_stats.get("avg", 0.0) or 0.0
                scale = actual_ab / (denom * DEFAULT_TEAM_AB)
                actual_sgp += (actual_avg - REPLACEMENT_AVG) * scale
                expected_sgp += (proj_avg - REPLACEMENT_AVG) * scale
        else:
            actual_ip = actual_stats.get("ip", 0) or 0
            if actual_ip > 0:
                # divisor cancels between marginal and one_sgp, so ERA and WHIP
                # share the form (repl - rate) * ip / (denom * team_ip).
                for cat, repl, actual_rate, proj_rate in (
                    (
                        Category.ERA,
                        REPLACEMENT_ERA,
                        calculate_era(
                            actual_stats.get("er", 0) or 0, actual_ip, default=0.0
                        ),
                        projected_stats.get("era", 0.0) or 0.0,
                    ),
                    (
                        Category.WHIP,
                        REPLACEMENT_WHIP,
                        calculate_whip(
                            actual_stats.get("bb", 0) or 0,
                            actual_stats.get("h_allowed", 0) or 0,
                            actual_ip,
                            default=0.0,
                        ),
                        projected_stats.get("whip", 0.0) or 0.0,
                    ),
                ):
                    denom = denoms.get(cat)
                    if not denom:
                        continue
                    scale = actual_ip / (denom * DEFAULT_TEAM_IP)
                    actual_sgp += (repl - actual_rate) * scale
                    expected_sgp += (repl - proj_rate) * scale

    return {
        "sgp_dev": round(actual_sgp - expected_sgp, 3),
        "actual_sgp": round(actual_sgp, 3),
        "expected_sgp": round(expected_sgp, 3),
    }
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `pytest tests/test_analysis/test_sgp_deviation.py -v`
Expected: PASS (6 tests)

- [ ] **Step 7: Run existing pace tests to confirm the refactor is safe**

Run: `pytest tests/test_analysis/ tests/test_web/test_season_data.py -q`
Expected: PASS (the `_prorated_expected` refactor is behavior-preserving).

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/analysis/pace.py tests/test_analysis/test_sgp_deviation.py
git commit -m "feat(pace): add compute_sgp_deviation metric + shared proration helper"
```

---

### Task 2: Cutpoints + leaguewide payload builder (pure)

**Files:**
- Modify: `src/fantasy_baseball/analysis/pace.py` (add `MIN_POOL_SIZE`, `compute_pace_cutpoints`, `build_pace_deviation_payload`)
- Test: `tests/test_analysis/test_pace_cutpoints.py` (create)

**Interfaces:**
- Consumes: `compute_sgp_deviation` (Task 1), `normalize_name`, `HITTER_PROJ_KEYS`/`PITCHER_PROJ_KEYS`, `PlayerType`.
- Produces:
  - `compute_pace_cutpoints(devs: list[float]) -> list[float] | None` (`[q16, q33, q66, q83]`, or None if `len(devs) < MIN_POOL_SIZE`).
  - `build_pace_deviation_payload(players, hitter_logs, pitcher_logs, preseason_lookup, denoms) -> dict` with shape `{"deviations": {"<norm>::<type>": {...}}, "cutpoints": {"hitter": [...] | None, "pitcher": [...] | None}}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_analysis/test_pace_cutpoints.py`:

```python
from fantasy_baseball.analysis.pace import (
    build_pace_deviation_payload,
    compute_pace_cutpoints,
)
from fantasy_baseball.models.player import Player
from fantasy_baseball.utils.constants import Category

DENOMS = {
    Category.R: 10.0,
    Category.HR: 5.0,
    Category.RBI: 10.0,
    Category.SB: 5.0,
    Category.AVG: 0.0015,
    Category.W: 2.0,
    Category.K: 12.0,
    Category.SV: 3.5,
    Category.ERA: 0.10,
    Category.WHIP: 0.03,
}


def test_cutpoints_twelve_values():
    # nearest-rank: index = round(q * (n-1)), n=12 -> round(q*11)
    cp = compute_pace_cutpoints([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    assert cp == [3, 5, 8, 10]


def test_cutpoints_exactly_min_pool():
    cp = compute_pace_cutpoints([10, 20, 30, 40, 50, 60])
    assert cp == [20, 30, 40, 50]


def test_cutpoints_below_min_pool_is_none():
    assert compute_pace_cutpoints([1, 2, 3]) is None


def test_payload_keys_and_small_pool_cutpoints():
    hitter = Player.from_dict({"name": "Test Hitter", "player_type": "hitter"})
    preseason = {
        "test hitter": Player.from_dict(
            {
                "name": "Test Hitter",
                "player_type": "hitter",
                "rest_of_season": {
                    "pa": 500, "ab": 450, "avg": 0.280,
                    "r": 100, "hr": 30, "rbi": 100, "sb": 20,
                },
            }
        )
    }
    logs = {"test hitter": {"pa": 100, "ab": 90, "h": 27, "r": 25, "hr": 8, "rbi": 30, "sb": 6}}
    payload = build_pace_deviation_payload([hitter], logs, {}, preseason, DENOMS)
    assert "test hitter::hitter" in payload["deviations"]
    assert payload["deviations"]["test hitter::hitter"]["sgp_dev"] is not None
    # only one hitter -> pool below MIN_POOL_SIZE -> None cutpoints
    assert payload["cutpoints"]["hitter"] is None
    assert payload["cutpoints"]["pitcher"] is None
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_pace_cutpoints.py -v`
Expected: FAIL with `ImportError: cannot import name 'compute_pace_cutpoints'`

- [ ] **Step 3: Implement the cutpoints + payload functions**

Append to `pace.py` (add the constant near the other thresholds, e.g. below `Z_LIGHT`):

```python
# Minimum qualified players in a pool before percentile bucketing is meaningful.
MIN_POOL_SIZE = 6
```

```python
def compute_pace_cutpoints(devs: list[float]) -> list[float] | None:
    """Return [q16, q33, q66, q83] nearest-rank cutpoints for a pool of
    SGP deviations, or None when the pool is smaller than MIN_POOL_SIZE.

    Nearest-rank index = round(q * (n - 1)) over the ascending-sorted list.
    """
    if len(devs) < MIN_POOL_SIZE:
        return None
    ordered = sorted(devs)
    n = len(ordered)
    return [ordered[round(q * (n - 1))] for q in (1 / 6, 1 / 3, 2 / 3, 5 / 6)]


def build_pace_deviation_payload(
    players: list[Any],
    hitter_logs: dict[str, dict[str, Any]],
    pitcher_logs: dict[str, dict[str, Any]],
    preseason_lookup: dict[str, Any],
    denoms: dict[Category, float],
) -> dict[str, Any]:
    """Compute the leaguewide SGP-deviation map + per-type cutpoints.

    Iterates rostered ``players``, builds each one's YTD actuals (from the
    game logs) and preseason projection (from ``preseason_lookup``), calls
    :func:`compute_sgp_deviation`, keys the result by
    ``normalize_name(name)::player_type``, and derives hitter/pitcher
    cutpoints over the players with a defined ``sgp_dev``.
    """
    deviations: dict[str, dict[str, Any]] = {}
    hitter_devs: list[float] = []
    pitcher_devs: list[float] = []

    for player in players:
        norm = normalize_name(player.name)
        if player.player_type == PlayerType.HITTER:
            actuals = hitter_logs.get(norm, {})
            proj_keys = HITTER_PROJ_KEYS
        else:
            actuals = pitcher_logs.get(norm, {})
            proj_keys = PITCHER_PROJ_KEYS
        pre = preseason_lookup.get(norm)
        if pre is not None and pre.rest_of_season is not None:
            projected = {k: getattr(pre.rest_of_season, k, 0) for k in proj_keys}
        else:
            projected = {}

        summary = compute_sgp_deviation(actuals, projected, player.player_type, denoms)
        deviations[f"{norm}::{player.player_type.value}"] = summary
        if summary["sgp_dev"] is not None:
            if player.player_type == PlayerType.HITTER:
                hitter_devs.append(summary["sgp_dev"])
            else:
                pitcher_devs.append(summary["sgp_dev"])

    return {
        "deviations": deviations,
        "cutpoints": {
            "hitter": compute_pace_cutpoints(hitter_devs),
            "pitcher": compute_pace_cutpoints(pitcher_devs),
        },
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_analysis/test_pace_cutpoints.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/pace.py tests/test_analysis/test_pace_cutpoints.py
git commit -m "feat(pace): add cutpoints + leaguewide deviation payload builder"
```

---

### Task 3: Cache key + refresh-pipeline step

**Files:**
- Modify: `src/fantasy_baseball/data/cache_keys.py` (add `PACE_DEVIATIONS`)
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py` (add `_compute_pace_deviations`, register it after `_compute_pace`)
- Modify: `tests/test_web/test_refresh_pipeline.py` (add `CacheKey.PACE_DEVIATIONS` to the expected-keys assertion)

**Interfaces:**
- Consumes: `build_pace_deviation_payload` (Task 2), `self.roster_players`, `self.opp_rosters`, `self.hitter_logs`, `self.pitcher_logs`, `self.preseason_lookup`, `self._league_denoms()`, `write_cache`, `CacheKey`.
- Produces: `CacheKey.PACE_DEVIATIONS` cache payload (see Task 2 shape).

- [ ] **Step 1: Add the cache key**

In `src/fantasy_baseball/data/cache_keys.py`, add to the `CacheKey` enum (after `STANDINGS_SNAPSHOT`):

```python
    PACE_DEVIATIONS = "pace_deviations"
```

- [ ] **Step 2: Write the failing integration assertion**

In `tests/test_web/test_refresh_pipeline.py`, add `CacheKey.PACE_DEVIATIONS` to the list inside `test_all_expected_cache_files_written` (the `expected_keys` block around line 50-67):

```python
            CacheKey.PACE_DEVIATIONS,
```

- [ ] **Step 3: Run test to verify it fails**

Run: `pytest tests/test_web/test_refresh_pipeline.py::TestRunFullRefresh::test_all_expected_cache_files_written -v`
Expected: FAIL with `Missing cache key: pace_deviations`

(If the test class name differs, run `pytest tests/test_web/test_refresh_pipeline.py -k all_expected_cache_files_written -v`.)

- [ ] **Step 4: Implement the pipeline step**

In `src/fantasy_baseball/web/refresh_pipeline.py`, add the method right after `_compute_pace` (which ends at the `attach_pace_to_roster(...)` call):

```python
    def _compute_pace_deviations(self):
        from fantasy_baseball.analysis.pace import build_pace_deviation_payload

        assert self.roster_players is not None
        assert self.opp_rosters is not None
        assert self.hitter_logs is not None
        assert self.pitcher_logs is not None
        assert self.preseason_lookup is not None

        self._progress("Computing leaguewide pace deviations...")
        all_rostered = list(self.roster_players)
        for roster in self.opp_rosters.values():
            all_rostered.extend(roster)

        payload = build_pace_deviation_payload(
            all_rostered,
            self.hitter_logs,
            self.pitcher_logs,
            self.preseason_lookup,
            self._league_denoms(),
        )
        write_cache(CacheKey.PACE_DEVIATIONS, payload, required=False)
        h = payload["cutpoints"]["hitter"]
        p = payload["cutpoints"]["pitcher"]
        self._progress(
            f"Pace deviations: {len(payload['deviations'])} players, "
            f"hitter cutpoints {'set' if h else 'n/a'}, "
            f"pitcher cutpoints {'set' if p else 'n/a'}"
        )
```

- [ ] **Step 5: Register the step in the pipeline**

In `_run_pipeline_steps`, add the call immediately after `self._compute_pace()`:

```python
        self._compute_pace()
        self._compute_pace_deviations()
```

- [ ] **Step 6: Run the integration test to verify it passes**

Run: `pytest tests/test_web/test_refresh_pipeline.py -k all_expected_cache_files_written -v`
Expected: PASS (`pace_deviations` now written).

- [ ] **Step 7: Run the full refresh-pipeline test file**

Run: `pytest tests/test_web/test_refresh_pipeline.py -q`
Expected: PASS (no other assertions regress).

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/data/cache_keys.py src/fantasy_baseball/web/refresh_pipeline.py tests/test_web/test_refresh_pipeline.py
git commit -m "feat(pace): compute + cache leaguewide pace deviations in refresh pipeline"
```

---

### Task 4: Reshape `compute_overall_pace` + wire display sites

**Files:**
- Modify: `src/fantasy_baseball/analysis/pace.py` (reshape `compute_overall_pace`)
- Modify: `src/fantasy_baseball/web/season_data.py` (3 call sites: `format_lineup_for_display`, and the matched + unmatched branches of `build_opponent_lineup`)
- Test: `tests/test_analysis/test_overall_pace.py` (rewrite)
- Modify: `tests/test_web/test_opponent_lineup.py` (update the mocked `overall_pace` shape)

**Interfaces:**
- Consumes: `CacheKey.PACE_DEVIATIONS` payload (Task 3), `normalize_name`, `read_cache_dict`.
- Produces: `compute_overall_pace(sgp_summary: dict[str, Any] | None, cutpoints: list[float] | None) -> dict[str, Any]` returning `{"color_class", "sgp_dev", "actual_sgp", "expected_sgp"}`.

- [ ] **Step 1: Rewrite the overall-pace unit test**

Replace the entire contents of `tests/test_analysis/test_overall_pace.py` with:

```python
from fantasy_baseball.analysis.pace import compute_overall_pace

CUTPOINTS = [3.0, 5.0, 8.0, 10.0]  # q16, q33, q66, q83


def _summary(dev):
    return {"sgp_dev": dev, "actual_sgp": dev, "expected_sgp": 0.0}


def test_bright_green_top_sixth():
    assert compute_overall_pace(_summary(11.0), CUTPOINTS)["color_class"] == "stat-hot-2"


def test_boundary_q83_is_bright_green():
    assert compute_overall_pace(_summary(10.0), CUTPOINTS)["color_class"] == "stat-hot-2"


def test_light_green():
    assert compute_overall_pace(_summary(9.0), CUTPOINTS)["color_class"] == "stat-hot-1"


def test_neutral_middle_third():
    assert compute_overall_pace(_summary(6.0), CUTPOINTS)["color_class"] == "stat-neutral"


def test_light_red():
    assert compute_overall_pace(_summary(4.0), CUTPOINTS)["color_class"] == "stat-cold-1"


def test_bright_red_bottom_sixth():
    assert compute_overall_pace(_summary(2.0), CUTPOINTS)["color_class"] == "stat-cold-2"


def test_boundary_q16_is_light_red():
    assert compute_overall_pace(_summary(3.0), CUTPOINTS)["color_class"] == "stat-cold-1"


def test_none_dev_is_neutral():
    out = compute_overall_pace(_summary(None), CUTPOINTS)
    assert out["color_class"] == "stat-neutral"
    assert out["sgp_dev"] is None


def test_missing_cutpoints_is_neutral():
    out = compute_overall_pace(_summary(11.0), None)
    assert out["color_class"] == "stat-neutral"
    assert out["sgp_dev"] == 11.0  # value preserved for the tooltip


def test_missing_summary_is_neutral():
    out = compute_overall_pace(None, CUTPOINTS)
    assert out["color_class"] == "stat-neutral"
    assert out["sgp_dev"] is None


def test_passthrough_fields():
    out = compute_overall_pace(
        {"sgp_dev": -2.4, "actual_sgp": 5.1, "expected_sgp": 7.5}, CUTPOINTS
    )
    assert out["sgp_dev"] == -2.4
    assert out["actual_sgp"] == 5.1
    assert out["expected_sgp"] == 7.5
    assert out["color_class"] == "stat-cold-2"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_overall_pace.py -v`
Expected: FAIL (old signature took one arg; several assertions error).

- [ ] **Step 3: Reshape `compute_overall_pace`**

Replace the existing `compute_overall_pace` in `pace.py` with:

```python
def compute_overall_pace(
    sgp_summary: dict[str, Any] | None,
    cutpoints: list[float] | None,
) -> dict[str, Any]:
    """Bucket a player's cached SGP deviation against its pool cutpoints.

    ``sgp_summary`` is one entry from the ``PACE_DEVIATIONS`` deviations map
    ({"sgp_dev", "actual_sgp", "expected_sgp"}); ``cutpoints`` is
    ``[q16, q33, q66, q83]`` for the player's type (or None). Renders neutral
    when the deviation is undefined, cutpoints are missing, or the pool was
    too small (cutpoints None). The tooltip values pass through unchanged.
    """
    dev = sgp_summary.get("sgp_dev") if sgp_summary else None
    result = {
        "color_class": "stat-neutral",
        "sgp_dev": dev,
        "actual_sgp": sgp_summary.get("actual_sgp") if sgp_summary else None,
        "expected_sgp": sgp_summary.get("expected_sgp") if sgp_summary else None,
    }
    if dev is None or not cutpoints:
        return result

    q16, q33, q66, q83 = cutpoints
    if dev >= q83:
        result["color_class"] = "stat-hot-2"
    elif dev >= q66:
        result["color_class"] = "stat-hot-1"
    elif dev >= q33:
        result["color_class"] = "stat-neutral"
    elif dev >= q16:
        result["color_class"] = "stat-cold-1"
    else:
        result["color_class"] = "stat-cold-2"
    return result
```

- [ ] **Step 4: Run the unit test to verify it passes**

Run: `pytest tests/test_analysis/test_overall_pace.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Wire the user-lineup call site (`format_lineup_for_display`)**

In `src/fantasy_baseball/web/season_data.py`, inside `format_lineup_for_display`, add ONLY the `normalize_name` import (the function already imports `compute_overall_pace` and `Player` at its top - do not re-add those or ruff will flag a redundant import), and read the cache once before the `for p in roster:` loop:

```python
    from fantasy_baseball.utils.name_utils import normalize_name

    pace_dev = read_cache_dict(CacheKey.PACE_DEVIATIONS) or {}
    deviations = pace_dev.get("deviations", {})
    cutpoints = pace_dev.get("cutpoints", {})
```

Then replace the entry's overall_pace line:

```python
            "overall_pace": compute_overall_pace(player.pace),
```

with a lookup by `(normalized name, type)`:

```python
            "overall_pace": compute_overall_pace(
                deviations.get(f"{normalize_name(player.name)}::{player.player_type.value}"),
                cutpoints.get(player.player_type.value),
            ),
```

- [ ] **Step 6: Wire the opponent-lineup call sites (`build_opponent_lineup`)**

In `build_opponent_lineup`, add the cache read once, right after the existing `rankings = read_cache_dict(CacheKey.RANKINGS) or {}` line:

```python
    pace_dev = read_cache_dict(CacheKey.PACE_DEVIATIONS) or {}
    deviations = pace_dev.get("deviations", {})
    cutpoints = pace_dev.get("cutpoints", {})
```

Replace the matched-player line (`entry["overall_pace"] = compute_overall_pace(entry["pace"])`, inside the `for player in matched:` loop where `norm` and `player` are in scope) with:

```python
        entry["overall_pace"] = compute_overall_pace(
            deviations.get(f"{norm}::{player.player_type.value}"),
            cutpoints.get(player.player_type.value),
        )
```

Replace the unmatched-player line (`entry["overall_pace"] = compute_overall_pace(entry["pace"])`, in the `for raw_player in roster:` loop) with a neutral render (no matched Player / summary):

```python
            entry["overall_pace"] = compute_overall_pace(None, None)
```

- [ ] **Step 7: Update the opponent-lineup test fixture**

In `tests/test_web/test_opponent_lineup.py`, replace the mocked `overall_pace` value:

```python
                        "overall_pace": {"avg_z": None, "color_class": "stat-neutral"},
```

with the new shape:

```python
                        "overall_pace": {
                            "sgp_dev": None,
                            "actual_sgp": None,
                            "expected_sgp": None,
                            "color_class": "stat-neutral",
                        },
```

- [ ] **Step 8: Run the affected tests**

Run: `pytest tests/test_analysis/test_overall_pace.py tests/test_web/test_opponent_lineup.py tests/test_web/test_season_data.py -q`
Expected: PASS.

- [ ] **Step 9: Commit**

```bash
git add src/fantasy_baseball/analysis/pace.py src/fantasy_baseball/web/season_data.py tests/test_analysis/test_overall_pace.py tests/test_web/test_opponent_lineup.py
git commit -m "feat(pace): bucket overall pace by leaguewide SGP-deviation percentile"
```

---

### Task 5: Templates + end-to-end verification

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/_lineup_hitters_tbody.html`
- Modify: `src/fantasy_baseball/web/templates/season/_lineup_pitchers_tbody.html`

**Interfaces:**
- Consumes: the reshaped `overall_pace` dict (`sgp_dev`, `actual_sgp`, `expected_sgp`, `color_class`).

- [ ] **Step 1: Update the hitters tbody tooltip**

In `_lineup_hitters_tbody.html`, the Slot `<td>` currently guards on `overall_pace.avg_z` and shows an "Avg z-score" row. Replace the guarded tooltip block:

```html
            {% if p.overall_pace and p.overall_pace.avg_z is not none %}
            <div class="tooltip">
                <div style="font-weight: 600; margin-bottom: 6px;">{{ p.name }} - Overall Pace</div>
                <div class="tooltip-row"><span class="tooltip-label">Avg z-score</span><span class="tooltip-val">{{ '%+.1f'|format(p.overall_pace.avg_z) }}</span></div>
            </div>
            {% endif %}
```

with the SGP-pace tooltip (guard on `sgp_dev`):

```html
            {% if p.overall_pace and p.overall_pace.sgp_dev is not none %}
            <div class="tooltip">
                <div style="font-weight: 600; margin-bottom: 6px;">{{ p.name }} - Overall Pace</div>
                <div class="tooltip-row"><span class="tooltip-label">SGP pace</span><span class="tooltip-val">{{ '%+.1f'|format(p.overall_pace.sgp_dev) }}</span></div>
                <div class="tooltip-row"><span class="tooltip-label">Delivered / expected</span><span class="tooltip-val">{{ '%.1f'|format(p.overall_pace.actual_sgp) }} / {{ '%.1f'|format(p.overall_pace.expected_sgp) }}</span></div>
            </div>
            {% endif %}
```

(Note: the label text uses ASCII `-`, not an em-dash, per the global ASCII constraint. If the existing file uses a non-ASCII dash, replace it with `-` in this edit.)

- [ ] **Step 2: Update the pitchers tbody tooltip**

Apply the identical replacement in `_lineup_pitchers_tbody.html` (same block, `p.overall_pace.avg_z` -> `p.overall_pace.sgp_dev` plus the delivered/expected row).

- [ ] **Step 3: Verify the templates render (no regressions in web tests)**

Run: `pytest tests/test_web/ -q`
Expected: PASS.

- [ ] **Step 4: End-to-end check with the run/verify skill**

Use the `run` skill (or `python scripts/run_season_dashboard.py --no-sync`) to load the lineup page against the current local cache. Confirm:
- The Slot cell shows green/red/neutral colors (not all-neutral).
- Hovering a colored player shows "SGP pace <signed number>" and "Delivered / expected".
- A gated/undefined player (e.g. a just-added bench player) shows neutral with no tooltip.
- Open an opponent lineup (the opponent view uses the same `overall_pace` shape,
  whose `avg_z` field was removed) and confirm its Slot cells still color and
  tooltip correctly - no console errors, no all-neutral column.

If the local cache predates this change (no `PACE_DEVIATIONS` key), run a refresh first (`python scripts/refresh_remote.py`, or trigger the dashboard refresh) so the pipeline writes the new payload, then reload.

- [ ] **Step 5: Full verification gate**

Run each and fix any failure:

```bash
pytest -v
ruff check .
ruff format --check .
vulture
mypy src/fantasy_baseball/analysis/pace.py  # + any other touched files listed in [tool.mypy].files
```

Expected: all green. Report the exact commands and results in the final message.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/_lineup_hitters_tbody.html src/fantasy_baseball/web/templates/season/_lineup_pitchers_tbody.html
git commit -m "feat(lineup): show SGP-pace tooltip on the overall Slot cell"
```

---

## Notes for the implementer

- The old avg-z behavior stays live until Task 4's atomic swap; Tasks 1-3 add code without changing the displayed color.
- `compute_sgp_deviation` is called **only** from `build_pace_deviation_payload` (pipeline). The display path is a pure cache lookup - never recompute deviations at request time, or the user/opponent bases can drift.
- Every deviation-map key is `normalize_name(name)::player_type_value`. Use the same form on both the write (Task 2) and read (Task 4) sides.
- Per-category cell coloring (`compute_player_pace` / `_z_to_color`) is intentionally unchanged.
