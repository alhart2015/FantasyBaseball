# Shape vs keeper-value historical backtest (PR 1 of #325)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-08-08-retire-keeper-value-design.md`

**Goal:** Race the `keeper_forecast` -> `keeper_value` chain against `shape` out of sample on historical seasons, on the keeper decision rather than on pool-wide rank, so that retiring it in PR 2 and PR 3 rests on measurement rather than on argument.

**Architecture:** The keeper chain is currently hardwired to base year 2026 and a live Upstash blend. This plan lifts four hardwirings so it can be run for a historical base year Y, puts both estimators' output on one SGP scale through `trajectory.panel.score()`, and adds decision-shaped slices to `scripts/backtest_trajectory.py`. Everything except `era_factors()` and `normalize_frame()` is deliberately temporary and is deleted in PR 3.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. No new dependencies.

## Global Constraints

- **ASCII only** in source, log messages, and anything reaching `print()`. Windows stdout is cp1252. Use `-`, `->`, `sigma`, straight quotes. Scripts that print player names must `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` first (both scripts touched here already do).
- **Player identity is `mlbam_id`.** Never key on a bare name. Any name in this plan is resolved to an id by a lookup in the same code path, never typed as a literal.
- **No `x or default` for numeric defaults.** `0`, `0.0` and `""` are falsy; use `x if x is not None else default`.
- **`src/fantasy_baseball/keepers/` and `src/fantasy_baseball/trajectory/` are both under `[tool.mypy].files`** (`pyproject.toml:80`, `pyproject.toml:95`). Every library change in this plan needs full type annotations and a clean `mypy` run. `scripts/` is not under mypy.
- **Tests are the guardrail.** Do not loosen an assertion to make it pass.
- **Verification gate for every commit:** `pytest -v` (or the stated subset), `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` when a mypy-covered file was touched.
- Outcome years stop at **2025**. 2026 is in progress and is never an outcome year.
- Base years in scope: **2022, 2023, 2024** at +1; **2022, 2023** at +2.

## File Structure

| file | responsibility | survives PR 3? |
|---|---|---|
| `src/fantasy_baseball/trajectory/era.py` | `era_factors()` extracted from `era_normalize`; `normalize_frame()` applies a factor table to any rate frame | **yes** |
| `src/fantasy_baseball/keepers/vintages.py` | `load_vintage` gains an optional `factors` argument | no (deleted PR 3) |
| `scripts/keeper_persistence.py` | `load_rates` gains `factors`; `TRANSITIONS` becomes a default, not a constant | no |
| `scripts/keeper_forecast.py` | `BASE_YEAR` becomes a parameter; `forecast_pool` takes an `observed` frame and a transition list | no |
| `scripts/backtest_trajectory.py` | historical head-to-head, slices, censoring, bootstrap | no |
| `tests/test_trajectory/test_era.py` | characterization + `normalize_frame` tests | yes |
| `tests/test_scripts/test_backtest_historical.py` | censoring, regret, roster join, historical-mode guards | no |

---

### Task 1: Extract `era_factors()` without changing `era_normalize`'s behaviour

**Files:**
- Modify: `src/fantasy_baseball/trajectory/era.py:88-133`
- Test: `tests/test_trajectory/test_era.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `era_factors(df: pd.DataFrame, kind: str, *, reference_seasons: tuple[int, ...] = REFERENCE_SEASONS) -> pd.DataFrame` — rows are seasons, columns are the rate columns of `RATE_DENOMINATORS[kind]`, values are the multiplicative factor for that season. Raises the same `ValueError` as `era_normalize` when a reference season is missing.

- [ ] **Step 1: Write the characterization test**

Add to `tests/test_trajectory/test_era.py`:

```python
def test_era_factors_reproduces_the_scaling_era_normalize_applies() -> None:
    """era_factors is an EXTRACTION, not a reimplementation.

    era_normalize's own output is the contract. If these two ever disagree, the
    keeper-value side of the backtest is normalized onto a different reference than
    the shape side and every comparison in it is silently wrong.
    """
    panel = _hitters(
        [
            {"mlbam_id": 1, "season": 2022, "hr_pa": 0.040},
            {"mlbam_id": 2, "season": 2023, "hr_pa": 0.050},
            {"mlbam_id": 3, "season": 2024, "hr_pa": 0.060},
            {"mlbam_id": 4, "season": 2025, "hr_pa": 0.055},
        ]
    )
    factors = era_factors(panel, "hitter")
    normalized = era_normalize(panel, "hitter")

    for season in (2022, 2023, 2024, 2025):
        raw_row = panel.loc[panel["season"] == season].iloc[0]
        norm_row = normalized.loc[normalized["season"] == season].iloc[0]
        assert norm_row["hr_pa"] == pytest.approx(raw_row["hr_pa"] * factors.loc[season, "hr_pa"])
        assert norm_row["era_factor_hr_pa"] == pytest.approx(factors.loc[season, "hr_pa"])


def test_era_factors_raises_on_a_missing_reference_season_like_era_normalize() -> None:
    panel = _hitters([{"mlbam_id": 1, "season": 2022}])
    with pytest.raises(ValueError, match="reference seasons"):
        era_factors(panel, "hitter")
```

Add `era_factors` to the import at the top of the file.

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_trajectory/test_era.py -k era_factors -v`
Expected: FAIL with `ImportError: cannot import name 'era_factors'`.

- [ ] **Step 3: Extract the function**

In `src/fantasy_baseball/trajectory/era.py`, split the factor computation out of `era_normalize` and have `era_normalize` call it. Replace lines 88-133 with:

```python
def era_factors(
    df: pd.DataFrame,
    kind: str,
    *,
    reference_seasons: tuple[int, ...] = REFERENCE_SEASONS,
) -> pd.DataFrame:
    """``season -> {rate_column: multiplicative factor}`` into the reference environment.

    Split out of `era_normalize` so the keeper-value chain can restate its own inputs
    onto the same reference the trajectory panel uses. Two independent computations of
    "what is a 2022 home run worth in 2023-2025 terms" is exactly the disagreement this
    subsystem cannot afford, so there is one and both callers use it.
    """
    rates = league_rates(df, kind)
    missing = [s for s in reference_seasons if s not in rates.index]
    if missing:
        # ALL of them, not merely one. Accepting a partial window silently restates
        # every season onto a reference that is not the one league.yaml's denominators
        # were calibrated against -- normalizing onto 2023 alone rather than the
        # 2023-2025 mean shifts every era factor, and therefore every historical SGP,
        # into units the output never mentions. If a narrower window is genuinely
        # wanted, say so by passing `reference_seasons` explicitly.
        raise ValueError(
            f"reference seasons {missing} are not in the panel "
            f"({int(rates.index.min())}-{int(rates.index.max())}), so the run "
            f"environment would be defined by only {sorted(set(reference_seasons) - set(missing))}"
            " -- not the window league.yaml's SGP denominators were calibrated on. "
            "Rebuild the panel to cover them, or pass reference_seasons explicitly."
        )
    reference = rates.loc[list(reference_seasons)].mean()
    # A season whose pool rate is 0 gives inf, and NaN propagates from an empty pool.
    # Both mean "no usable adjustment for this season"; neutralize to 1.0 rather than
    # blanking the column or, worse, multiplying a real rate by inf.
    # np.nan, not pd.NA: pd.NA makes the column object-dtype and the later astype(float)
    # raises on it.
    return (reference / rates).replace([np.inf, -np.inf], np.nan)


def era_normalize(
    df: pd.DataFrame,
    kind: str,
    *,
    reference_seasons: tuple[int, ...] = REFERENCE_SEASONS,
    sgp_overrides: SgpOverrides | None = None,
) -> pd.DataFrame:
    """Rescale every season's category rates into the reference run environment.

    Returns a copy with the rate columns adjusted and `sgp` re-scored. `era_factor_*`
    columns are kept so a surprising comp can be traced back to its adjustment rather
    than taken on faith.
    """
    factors = era_factors(df, kind, reference_seasons=reference_seasons)
    out = df.copy()
    for rate in RATE_DENOMINATORS[kind]:
        factor = out["season"].map(factors[rate]).astype(float).fillna(1.0)
        out[f"era_factor_{rate}"] = factor
        out[rate] = out[rate] * factor
    return score(out, kind, sgp_overrides)
```

- [ ] **Step 4: Run the full era test module to verify nothing else moved**

Run: `pytest tests/test_trajectory/test_era.py -v`
Expected: PASS, including every pre-existing test. Any pre-existing failure means the extraction changed behaviour — fix the extraction, not the test.

- [ ] **Step 5: Run the wider trajectory suite and the type checker**

Run: `pytest tests/test_trajectory/ -v && mypy && ruff check . && ruff format --check .`
Expected: all clean.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/trajectory/era.py tests/test_trajectory/test_era.py
git commit -m "trajectory: extract era_factors from era_normalize (#325)"
```

---

### Task 2: `normalize_frame()` and both keeper-chain loaders

**Files:**
- Modify: `src/fantasy_baseball/trajectory/era.py` (add `normalize_frame`)
- Modify: `src/fantasy_baseball/keepers/vintages.py:73-100` (`load_vintage`)
- Modify: `scripts/keeper_persistence.py:90-110` (`load_rates`)
- Test: `tests/test_trajectory/test_era.py`, `tests/test_keepers/test_vintages.py`

**Interfaces:**
- Consumes: `era_factors` from Task 1.
- Produces: `normalize_frame(frame: pd.DataFrame, season: int, kind: str, factors: pd.DataFrame) -> pd.DataFrame` — returns a copy with each rate column of `RATE_DENOMINATORS[kind]` multiplied by `factors.loc[season, rate]`, falling back to 1.0 for a season absent from the table. `load_vintage(year, projections_root, player_type, *, factors=None)` and `load_rates(year, kind, *, source, factors=None)` both apply it when `factors` is given.

**Why both loaders.** `keeper_persistence.load_rates` feeds the persistence fit; `keepers.vintages.load_vintage` feeds `forecast_pool` directly at `scripts/keeper_forecast.py:258-259`. Normalizing only one leaves `drift` — an additive term — fit in one unit system and applied in another, which biases every forecast by a fixed amount with no symptom.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_trajectory/test_era.py`:

```python
def test_normalize_frame_scales_each_rate_by_its_season_factor() -> None:
    panel = _hitters(
        [
            {"mlbam_id": 1, "season": 2022, "hr_pa": 0.040},
            {"mlbam_id": 2, "season": 2023, "hr_pa": 0.050},
            {"mlbam_id": 3, "season": 2024, "hr_pa": 0.060},
            {"mlbam_id": 4, "season": 2025, "hr_pa": 0.055},
        ]
    )
    factors = era_factors(panel, "hitter")
    frame = pd.DataFrame({"pa": [600.0], "hr_pa": [0.040]}, index=pd.Index([99], name="mlbam_id"))

    out = normalize_frame(frame, 2022, "hitter", factors)

    assert out["hr_pa"].iloc[0] == pytest.approx(0.040 * factors.loc[2022, "hr_pa"])
    # Volume is never era-normalized -- a 600-PA season is 600 PA in any year.
    assert out["pa"].iloc[0] == 600.0
    # The input is not mutated; callers hold on to raw frames.
    assert frame["hr_pa"].iloc[0] == 0.040


def test_normalize_frame_leaves_an_unknown_season_alone() -> None:
    """1.0, not KeyError. A season with no factor means no usable adjustment, which is
    what era_normalize's own fillna(1.0) already decided."""
    panel = _hitters(
        [
            {"mlbam_id": 1, "season": 2023},
            {"mlbam_id": 2, "season": 2024},
            {"mlbam_id": 3, "season": 2025},
        ]
    )
    factors = era_factors(panel, "hitter")
    frame = pd.DataFrame({"pa": [600.0], "hr_pa": [0.040]}, index=pd.Index([99], name="mlbam_id"))

    out = normalize_frame(frame, 1998, "hitter", factors)

    assert out["hr_pa"].iloc[0] == 0.040
```

Add to `tests/test_keepers/test_vintages.py`:

```python
def test_load_vintage_and_load_rates_agree_on_the_same_season(tmp_path) -> None:
    """The two loaders are separate code paths into the same chain (spec: 'There are
    TWO loaders and both must normalize'). If they disagree, the persistence fit and
    the fold it feeds run in different unit systems and `drift` is silently biased.
    """
    import sys

    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
    from keeper_persistence import load_rates

    from fantasy_baseball.trajectory.era import era_factors, normalize_frame

    panel = _scored_reference_panel()  # 2023-2025 present so era_factors can be built
    factors = era_factors(panel, "hitter")

    raw = load_vintage(2024, PROJECTIONS, "hitter")
    via_vintage = load_vintage(2024, PROJECTIONS, "hitter", factors=factors)
    via_rates = load_rates(2024, "hitter", source="projection", factors=factors)

    expected = normalize_frame(raw, 2024, "hitter", factors)
    common = via_vintage.index.intersection(via_rates.index)
    assert len(common) > 0
    for rate in ("hr_pa", "r_pa", "sb_pa"):
        pd.testing.assert_series_equal(
            via_vintage.loc[common, rate], expected.loc[common, rate], check_names=False
        )
        pd.testing.assert_series_equal(
            via_rates.loc[common, rate], expected.loc[common, rate], check_names=False
        )
```

Write `_scored_reference_panel()` in that test module as a small helper building a hitter frame with seasons 2023, 2024 and 2025 through `trajectory.panel.score`, mirroring `tests/test_trajectory/test_era.py::_hitters`.

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_trajectory/test_era.py -k normalize_frame tests/test_keepers/test_vintages.py -k agree -v`
Expected: FAIL — `normalize_frame` does not exist, `load_vintage` takes no `factors`.

- [ ] **Step 3: Add `normalize_frame` to `era.py`**

```python
def normalize_frame(
    frame: pd.DataFrame, season: int, kind: str, factors: pd.DataFrame
) -> pd.DataFrame:
    """Restate one season's rate frame into the reference run environment.

    For frames that are ABOUT a single season and do not carry a `season` column --
    a ZiPS vintage, an actuals export -- where `era_normalize` needs a panel. Volume
    (`pa`/`ip`) and the structural `ab_pa` ratio are left alone for the same reasons
    `era_normalize` leaves them alone; see the module docstring.
    """
    out = frame.copy()
    for rate in RATE_DENOMINATORS[kind]:
        if rate not in out.columns:
            continue
        factor = factors[rate].get(season, 1.0)
        out[rate] = out[rate] * (1.0 if pd.isna(factor) else float(factor))
    return out
```

- [ ] **Step 4: Thread `factors` through both loaders**

In `src/fantasy_baseball/keepers/vintages.py`, change the signature and apply at the end, before returning:

```python
def load_vintage(
    year: int,
    projections_root: Path,
    player_type: str,
    *,
    factors: pd.DataFrame | None = None,
) -> pd.DataFrame:
```

and immediately before `return frame`:

```python
    if factors is not None:
        # Historical backtests only (#325). The live board runs raw, because its
        # baseline and its target are both in the current run environment.
        from fantasy_baseball.trajectory.era import normalize_frame

        frame = normalize_frame(frame, year, player_type, factors)
```

In `scripts/keeper_persistence.py`, `load_rates` gains the same keyword and applies `normalize_frame(frame, year, kind, factors)` after the dedupe, returning the normalized frame.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_trajectory/test_era.py tests/test_keepers/test_vintages.py -v`
Expected: PASS.

- [ ] **Step 6: Full gate and commit**

Run: `pytest -v && mypy && ruff check . && ruff format --check . && vulture`

```bash
git add src/fantasy_baseball/trajectory/era.py src/fantasy_baseball/keepers/vintages.py scripts/keeper_persistence.py tests/
git commit -m "keepers: normalize rate frames at both loader entry points (#325)"
```

---

### Task 3: Parameterize the keeper chain's base year and observed frame

**Files:**
- Modify: `scripts/keeper_forecast.py` — `BASE_YEAR` (line 104), `volume_forecast` (lines 146-215), `forecast_pool` (lines 250-300), `main`
- Test: `tests/test_scripts/test_backtest_historical.py` (create)

**Interfaces:**
- Consumes: Task 2's `factors` plumbing.
- Produces: `forecast_pool(kind: str, base_year: int, target_year: int, observed: pd.DataFrame, args, *, transitions=None, factors=None) -> pd.DataFrame` — same output schema as today (index `mlbam_id`, columns `[pt, *rates]` plus `{col}_gap`). `volume_forecast(kind, base_year, target_year, observed, include_exits=True)`.

**The change is mechanical but one detail is not.** `volume_forecast` currently reads `BASE_YEAR` from module scope in five places (`series_for(BASE_YEAR - 1, ...)`, `BASE_YEAR - 2`, the `age` fallback, and the `seasons` list). All five become `base_year`. Missing one leaves a 2026 lag in a 2022 forecast, which produces a plausible number rather than an error.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scripts/test_backtest_historical.py`:

```python
"""Guards on the historical mode added for #325.

These assert the things whose failure produces a PLAUSIBLE WRONG NUMBER rather than
an exception -- a 2026 lag inside a 2022 forecast, a panel that still contains the
future, a query player who can match himself.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def test_volume_forecast_reads_no_season_after_its_base_year(monkeypatch) -> None:
    """BASE_YEAR was a module constant read in five places inside volume_forecast.
    Threading four of the five leaves a 2026 lag in a 2022 forecast and no symptom."""
    import keeper_forecast

    seen: list[int] = []
    real = pd.DataFrame.loc

    panel = pd.DataFrame(
        {
            "mlbam_id": [1, 1, 1, 1],
            "season": [2020, 2021, 2022, 2026],
            "pa": [500.0, 550.0, 600.0, 610.0],
            "games": [140, 145, 150, 152],
            "age": [25, 26, 27, 31],
            "partial_season": [False, False, False, False],
        }
    )
    monkeypatch.setattr(keeper_forecast.pd, "read_csv", lambda *_a, **_k: panel)
    monkeypatch.setattr(keeper_forecast, "_panel_path", lambda kind: Path("fake.csv"))

    def spy(year: int, column: str):
        seen.append(year)
        return pd.Series(dtype=float)

    observed = pd.Series([600.0], index=pd.Index([1], name="mlbam_id"))
    keeper_forecast.volume_forecast("hitter", 2022, 2023, observed)

    assert seen, "volume_forecast never consulted the panel"
    assert max(seen) <= 2022, f"read seasons after the base year: {sorted(set(seen))}"
```

Note: the spy needs `volume_forecast` to route its per-season reads through a named
helper. Part of this task is extracting the inline `series_for` closure into a
module-level `_series_for(panel, year, column, index)` so it can be observed; that is
required for the test, not incidental.

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_scripts/test_backtest_historical.py -v`
Expected: FAIL — `volume_forecast` takes 3 positional args, not 4.

- [ ] **Step 3: Thread `base_year` through**

- Change `BASE_YEAR = 2026` to `DEFAULT_BASE_YEAR = 2026` and keep a module-level `BASE_YEAR` nowhere.
- `volume_forecast(kind, base_year, target_year, observed, include_exits=True)`; replace all five `BASE_YEAR` reads with `base_year`; `steps = target_year - base_year`.
- Extract `series_for` to `_series_for(panel: pd.DataFrame, year: int, column: str, index: pd.Index) -> pd.Series`.
- `forecast_pool(kind, base_year, target_year, observed, args, *, transitions=None, factors=None)`; `load_vintage(base_year, PROJECTIONS, kind, factors=factors)` and `load_vintage(target_year, ..., factors=factors)`; drop the internal `parse_blend` call and take `observed` as a frame.
- `main()` passes `DEFAULT_BASE_YEAR` and `parse_blend(payload, kind)`, so the live path is unchanged.

- [ ] **Step 4: Run the test and the live path**

Run: `pytest tests/test_scripts/test_backtest_historical.py -v`
Expected: PASS.

Run: `python scripts/keeper_forecast.py --pool hitter --top 5`
Expected: the same top-5 table as before the change. This needs `.env` and network; if unavailable, say so in the commit message rather than claiming it was verified.

- [ ] **Step 5: Commit**

```bash
git add scripts/keeper_forecast.py tests/test_scripts/test_backtest_historical.py
git commit -m "keeper_forecast: make the base year and observed frame parameters (#325)"
```

---

### Task 4: Transition-list parameterization, LOTO and the causal variant

**Files:**
- Modify: `scripts/keeper_persistence.py:57` (`TRANSITIONS`), `scripts/keeper_forecast.py` (`load_shares`)
- Test: `tests/test_scripts/test_backtest_historical.py`

**Interfaces:**
- Produces: `transitions_for(base_year: int, mode: str) -> tuple[tuple[int, int], ...]` in `scripts/backtest_trajectory.py`. `mode` is `"loto"` or `"causal"`. `load_shares(kind, args, transitions=None, factors=None)`.

**The counts this must produce**, from the spec — get these wrong and the leakage disclosure is wrong:

| base | `loto` | future transitions in it | `causal` |
|---|---|---|---|
| 2022 | (2023,2024), (2024,2025) | 2 | **empty** |
| 2023 | (2022,2023), (2024,2025) | 1 | (2022,2023) |
| 2024 | (2022,2023), (2023,2024) | 0 | (2022,2023), (2023,2024) |

- [ ] **Step 1: Write the failing test**

```python
def test_transitions_for_matches_the_counts_the_spec_discloses() -> None:
    """The leakage disclosure in the PR body is computed from these. Base 2024 is the
    year where loto and causal COINCIDE -- which is why the sensitivity check runs on
    2023, not 2024."""
    from backtest_trajectory import transitions_for

    assert transitions_for(2022, "loto") == ((2023, 2024), (2024, 2025))
    assert transitions_for(2023, "loto") == ((2022, 2023), (2024, 2025))
    assert transitions_for(2024, "loto") == ((2022, 2023), (2023, 2024))

    assert transitions_for(2022, "causal") == ()
    assert transitions_for(2023, "causal") == ((2022, 2023),)
    assert transitions_for(2024, "causal") == ((2022, 2023), (2023, 2024))

    # The sensitivity check is only meaningful where the two differ.
    assert transitions_for(2024, "loto") == transitions_for(2024, "causal")
    assert transitions_for(2023, "loto") != transitions_for(2023, "causal")


def test_transitions_for_refuses_a_base_year_with_no_causal_data() -> None:
    from backtest_trajectory import transitions_for

    assert transitions_for(2022, "causal") == ()
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_scripts/test_backtest_historical.py -k transitions -v`
Expected: FAIL — `cannot import name 'transitions_for'`.

- [ ] **Step 3: Implement**

In `scripts/backtest_trajectory.py`:

```python
def transitions_for(base_year: int, mode: str) -> tuple[tuple[int, int], ...]:
    """Which (year, year+1) transitions the persistence fit may use for `base_year`.

    `loto` drops only the transition being predicted. It does NOT make the fit causal:
    for base 2022 both survivors are LATER than the transition predicted. That is
    disclosed in the writeup as a third advantage keeper-value keeps, rather than
    silently corrected, because a causal rule leaves base 2022 with nothing to fit on
    and base 2023 with one transition -- which would delete the +2 horizon.

    `causal` is the sensitivity variant. It is only informative for base 2023: base
    2022 returns empty and base 2024 returns exactly what `loto` returns.
    """
    if mode not in {"loto", "causal"}:
        raise ValueError(f"mode must be 'loto' or 'causal', got {mode!r}")
    predicted = (base_year, base_year + 1)
    if mode == "loto":
        return tuple(t for t in ALL_TRANSITIONS if t != predicted)
    return tuple(t for t in ALL_TRANSITIONS if t[1] <= base_year)
```

with `from keeper_persistence import TRANSITIONS as ALL_TRANSITIONS`.

In `scripts/keeper_forecast.py`, `load_shares` takes `transitions` (defaulting to `keeper_persistence.TRANSITIONS`) and `factors`, and passes both down to `build_transition` / `build_volume_transition` / `load_rates`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_scripts/test_backtest_historical.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/keeper_forecast.py scripts/keeper_persistence.py scripts/backtest_trajectory.py tests/test_scripts/test_backtest_historical.py
git commit -m "keeper_forecast: parameterize the persistence transition list (#325)"
```

---

### Task 5: Censor the playing-time panel and account for the fallbacks

**Files:**
- Modify: `scripts/keeper_forecast.py` (`volume_forecast`)
- Test: `tests/test_scripts/test_backtest_historical.py`

**Interfaces:**
- Produces: `volume_forecast` returns `tuple[pd.Series | None, FallbackReport]` where `FallbackReport` is a `dataclass(frozen=True)` with `whole_pool: bool`, `per_player: int`, `total: int`.

**Rules from the spec:** per-player fallbacks are tolerated to 25% of the pool; past that the base year is excluded from the headline. A whole-pool fallback fails the base year outright rather than producing a number.

- [ ] **Step 1: Write the failing tests**

```python
def test_volume_forecast_censors_the_panel_to_the_base_year(monkeypatch) -> None:
    """Training on seasons after Y would let the curve learn the future it is being
    asked to predict."""
    import keeper_forecast

    captured: dict[str, pd.DataFrame] = {}
    monkeypatch.setattr(
        keeper_forecast, "lag_panel", lambda panel, kind, **kw: captured.setdefault("panel", panel)
    )
    # ... panel fixture spanning 2018-2026, base year 2022 ...
    assert int(captured["panel"]["season"].max()) <= 2022


def test_fallback_report_counts_per_player_misses() -> None:
    report = keeper_forecast.FallbackReport(whole_pool=False, per_player=30, total=100)
    assert report.share == pytest.approx(0.30)
    assert report.exceeds_headline_threshold is True


def test_fallback_report_flags_a_whole_pool_miss_regardless_of_share() -> None:
    report = keeper_forecast.FallbackReport(whole_pool=True, per_player=0, total=100)
    assert report.share == 0.0
    assert report.exceeds_headline_threshold is True
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_scripts/test_backtest_historical.py -k "censors or fallback" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
HEADLINE_FALLBACK_SHARE = 0.25


@dataclass(frozen=True)
class FallbackReport:
    """How much of a base year's forecast came from the gap model rather than the curve.

    A gap-model fallback still counts as keeper-value -- it is what the engine does when
    the data is thin, and substituting something better would be scoring an engine that
    does not exist. But a base year mostly produced by it is not evidence about the
    curve, so it is reported separately instead of folded into the headline.
    """

    whole_pool: bool
    per_player: int
    total: int

    @property
    def share(self) -> float:
        return self.per_player / self.total if self.total else 0.0

    @property
    def exceeds_headline_threshold(self) -> bool:
        return self.whole_pool or self.share > HEADLINE_FALLBACK_SHARE
```

In `volume_forecast`, filter `panel = panel[panel["season"] <= base_year]` immediately after `pd.read_csv`, and return the report alongside the projection.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_scripts/test_backtest_historical.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/keeper_forecast.py tests/test_scripts/test_backtest_historical.py
git commit -m "keeper_forecast: censor the PT panel to the base year and report fallbacks (#325)"
```

---

### Task 6: Score a keeper-value forecast onto the panel's SGP and VAR scale

**Files:**
- Modify: `scripts/backtest_trajectory.py`
- Test: `tests/test_scripts/test_backtest_historical.py`

**Interfaces:**
- Produces: `keeper_value_sgp(frame: pd.DataFrame, kind: str, sgp_overrides) -> pd.Series` — mlbam_id -> SGP, scored by `trajectory.panel.score`. `var_for(sgp_by_id: pd.Series, kind: str, base_year: int, cache_dir: Path) -> pd.Series` — SGP minus the year-`base_year` position-aware floor.

**Why this works at all:** `keepers.actuals.HITTER_PT == "pa"`, `PITCHER_PT == "ip"`, and `HITTER_RATES`/`PITCHER_RATES` are character-for-character the columns `trajectory.panel.score` reconstructs from. `forecast_pool`'s output frame is already in that schema, so it can be handed to the panel's own scorer with no translation. Do **not** route through `keeper_forecast.to_counting`, which renames to `PA`/`IP` and finishes `AVG`/`ERA`/`WHIP`.

- [ ] **Step 1: Write the failing test**

```python
def test_keeper_value_sgp_uses_the_panels_own_scorer() -> None:
    """One scorer, or the two estimators are not on one scale. A forecast frame is
    already in the panel's rate schema (keepers.actuals.HITTER_PT == 'pa'), so this is
    a hand-off, not a translation."""
    from backtest_trajectory import keeper_value_sgp

    from fantasy_baseball.trajectory.panel import score

    frame = pd.DataFrame(
        {
            "pa": [600.0],
            "ab_pa": [0.90],
            "h_ab": [0.280],
            "hr_pa": [0.05],
            "r_pa": [0.15],
            "rbi_pa": [0.14],
            "sb_pa": [0.02],
        },
        index=pd.Index([12345], name="mlbam_id"),
    )
    expected = score(frame.reset_index(), "hitter")["sgp"].iloc[0]
    assert keeper_value_sgp(frame, "hitter", None).loc[12345] == pytest.approx(expected)


def test_var_uses_year_Y_eligibility_not_the_outcome_years(tmp_path) -> None:
    """A catcher who stops catching in the outcome year must still be priced against the
    catcher floor -- that is the information the keeper decision had."""
    from backtest_trajectory import var_for

    (tmp_path / "mlb_fielding_2023.csv").write_text(
        "player.id,position.abbreviation,stat.games\n12345,C,100\n", encoding="utf-8"
    )
    (tmp_path / "mlb_fielding_2024.csv").write_text(
        "player.id,position.abbreviation,stat.games\n12345,1B,100\n", encoding="utf-8"
    )
    sgp = pd.Series([10.0], index=pd.Index([12345], name="mlbam_id"))

    as_catcher = var_for(sgp, "hitter", 2023, tmp_path)
    as_first_baseman = var_for(sgp, "hitter", 2024, tmp_path)

    # The catcher floor is the lowest, so the same SGP is worth MORE as a catcher.
    assert as_catcher.loc[12345] > as_first_baseman.loc[12345]
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_scripts/test_backtest_historical.py -k "scorer or eligibility" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`keeper_value_sgp` calls `panel.score(frame.reset_index(), kind, sgp_overrides)` and returns the `sgp` column indexed by `mlbam_id`. `var_for` reuses `trajectory.board`'s existing eligibility reader (the `cache_dir / f"mlb_fielding_{season}.csv"` path with its degrade-to-UTIL fallback) and `trajectory.value.resolve_slots` / `best_floor` — do not write a second eligibility path.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_scripts/test_backtest_historical.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_trajectory.py tests/test_scripts/test_backtest_historical.py
git commit -m "backtest: score keeper-value forecasts on the panel's SGP scale (#325)"
```

---

### Task 7: Realized outcomes and the two views

**Files:**
- Modify: `scripts/backtest_trajectory.py`
- Test: `tests/test_scripts/test_backtest_historical.py`

**Interfaces:**
- Produces: `Outcome` dataclass with `mlbam_id`, `sgp_by_year: dict[int, float]`, `volume_by_year: dict[int, float]`, `anchor_volume: float`; and `censored(outcome: Outcome, years: Sequence[int], threshold: float = 0.5) -> bool`.

**Rules from the spec, all of which have a test below:** the ratio is against year **Y**, never Y+1; zero volume is censored; a missing outcome row is zero volume, scoring 0 SGP in ALL and censored in the injury view; a player censored in **either** outcome year leaves the multi-year metric entirely.

- [ ] **Step 1: Write the failing tests**

```python
CENSOR_CASES = [
    # (anchor_volume, outcome_volumes, expected_censored, why)
    (600.0, {2024: 600.0, 2025: 600.0}, False, "healthy both years"),
    (600.0, {2024: 300.0, 2025: 600.0}, False, "exactly 50% is NOT under the cut"),
    (600.0, {2024: 299.0, 2025: 600.0}, True, "just under 50% in one year censors both"),
    (600.0, {2024: 0.0, 2025: 600.0}, True, "zero volume is censored, by explicit decision"),
    (600.0, {2025: 600.0}, True, "a MISSING row is zero volume, not a skipped year"),
]


@pytest.mark.parametrize("anchor,volumes,expected,why", CENSOR_CASES)
def test_censoring_boundaries(anchor, volumes, expected, why) -> None:
    from backtest_trajectory import Outcome, censored

    outcome = Outcome(
        mlbam_id=1, sgp_by_year={}, volume_by_year=volumes, anchor_volume=anchor
    )
    assert censored(outcome, [2024, 2025]) is expected, why


def test_the_ratio_is_against_the_anchor_year_not_the_previous_outcome() -> None:
    """A wrecked Y+1 must not redefine 'normal' for Y+2. Against Y+1 (100 PA) a
    500-PA Y+2 would look like a 5x recovery and pass; against the 600-PA anchor it is
    the anchor that decides, and Y+1 is censored either way."""
    from backtest_trajectory import Outcome, censored

    outcome = Outcome(
        mlbam_id=1,
        sgp_by_year={},
        volume_by_year={2024: 100.0, 2025: 500.0},
        anchor_volume=600.0,
    )
    assert censored(outcome, [2024, 2025]) is True
    assert censored(outcome, [2025]) is False


def test_a_missing_outcome_row_scores_zero_in_the_ALL_view() -> None:
    from backtest_trajectory import Outcome

    outcome = Outcome(mlbam_id=1, sgp_by_year={}, volume_by_year={}, anchor_volume=600.0)
    assert outcome.realized([2024, 2025]) == 0.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_scripts/test_backtest_historical.py -k "censor or ratio or missing" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
@dataclass(frozen=True)
class Outcome:
    """What a player actually did in the outcome years, and how much he played.

    `sgp_by_year` and `volume_by_year` are sparse: a season the player did not appear
    in is ABSENT, not zero, so the distinction between "played badly" and "was not
    there" survives into the censoring rule. `realized` collapses it to the 0 a
    vanished player is worth to a roster slot; `censored` treats it as zero volume.
    """

    mlbam_id: int
    sgp_by_year: dict[int, float]
    volume_by_year: dict[int, float]
    anchor_volume: float

    def realized(self, years: Sequence[int]) -> float:
        return sum(self.sgp_by_year.get(y, 0.0) for y in years)


def censored(outcome: Outcome, years: Sequence[int], threshold: float = 0.5) -> bool:
    """True if ANY outcome year falls under `threshold` of the ANCHOR year's volume.

    Any, not all: a one-year sum and a two-year sum are not the same target, so a
    player wrecked in one of two years leaves the multi-year metric entirely rather
    than contributing a shorter one.
    """
    if outcome.anchor_volume <= 0:
        return True
    return any(
        outcome.volume_by_year.get(year, 0.0) < threshold * outcome.anchor_volume
        for year in years
    )
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_scripts/test_backtest_historical.py -v`
Expected: PASS, all 5 parametrized cases included.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_trajectory.py tests/test_scripts/test_backtest_historical.py
git commit -m "backtest: realized outcomes and the injury-excluded view (#325)"
```

---

### Task 8: Resolve drafted names to ids

**Files:**
- Modify: `scripts/backtest_trajectory.py`
- Test: `tests/test_scripts/test_backtest_historical.py`

**Interfaces:**
- Produces: `RosterResolution` dataclass with `by_team: dict[str, list[int]]`, `unresolved: list[tuple[str, str]]`, `ambiguous: list[tuple[str, str]]`; and `resolve_draft(draft: list[dict], people: pd.DataFrame, pool_by_id: dict[int, str]) -> RosterResolution`.

**This is the risk in the whole plan.** `data/historical_drafts_resolved.json` carries bare names; everything else is `mlbam_id`. `CLAUDE.md` names bare-name joins as a defect class and `trajectory/roster_join.py` records that `(normalized_name, pool)` is not unique — the live board has two hitters called Max Muncy. Unresolved names are **reported, never dropped silently**: a silent drop thins roster pools toward the fringe, which flatters both estimators.

- [ ] **Step 1: Write the failing tests**

```python
def test_resolve_draft_reports_an_unresolved_name_instead_of_dropping_it() -> None:
    from backtest_trajectory import resolve_draft

    people = pd.DataFrame({"id": [1], "fullName": ["Yordan Alvarez"]})
    result = resolve_draft(
        [
            {"team": "Hart of the Order", "player": "Yordan Alvarez"},
            {"team": "Hart of the Order", "player": "Nobody At All"},
        ],
        people,
        {1: "hitter"},
    )
    assert result.by_team["Hart of the Order"] == [1]
    assert ("Hart of the Order", "Nobody At All") in result.unresolved


def test_resolve_draft_reports_an_ambiguous_name_rather_than_picking_one() -> None:
    """Two hitters called Max Muncy is a real case (roster_join.py). Picking one
    silently means a roster is scored against the wrong player's career."""
    from backtest_trajectory import resolve_draft

    people = pd.DataFrame({"id": [1, 2], "fullName": ["Max Muncy", "Max Muncy"]})
    result = resolve_draft(
        [{"team": "Spacemen", "player": "Max Muncy"}], people, {1: "hitter", 2: "hitter"}
    )
    assert result.by_team.get("Spacemen", []) == []
    assert ("Spacemen", "Max Muncy") in result.ambiguous


def test_resolve_draft_normalizes_accents() -> None:
    from backtest_trajectory import resolve_draft

    people = pd.DataFrame({"id": [7], "fullName": ["Jesus Luzardo"]})
    result = resolve_draft(
        [{"team": "Spacemen", "player": "Jesus Luzardo"}], people, {7: "pitcher"}
    )
    assert result.by_team["Spacemen"] == [7]
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_scripts/test_backtest_historical.py -k resolve_draft -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Use `fantasy_baseball.utils.name_utils.normalize_name` and `trajectory.board.people` for the id -> name union. Group people by normalized name; a name mapping to more than one id that the panel can score is `ambiguous` and contributes no id; a name mapping to none is `unresolved`.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_scripts/test_backtest_historical.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_trajectory.py tests/test_scripts/test_backtest_historical.py
git commit -m "backtest: resolve drafted names to mlbam ids, reporting misses (#325)"
```

---

### Task 9: Keeper-triple regret and the agreement rate

**Files:**
- Modify: `scripts/backtest_trajectory.py`
- Test: `tests/test_scripts/test_backtest_historical.py`

**Interfaces:**
- Produces: `TripleResult` with `team: str`, `picked: tuple[int, ...]`, `regret: float`; and `triple_regret(candidates: Sequence[int], forecast: Mapping[int, float], realized: Mapping[int, float], keep: int = 3) -> tuple[tuple[int, ...], float]`.

**Rules:** picks come from forecast VAR; regret is the realized shortfall against the ex-post best `keep`. In the injury view, censored players leave the candidate pool **and** the optimum, so each estimator re-picks from the reduced roster — that keeps both triples the same size, which is what makes regret coherent.

- [ ] **Step 1: Write the failing tests**

```python
def test_triple_regret_is_zero_when_the_forecast_picks_the_ex_post_best() -> None:
    from backtest_trajectory import triple_regret

    forecast = {1: 30.0, 2: 20.0, 3: 10.0, 4: 5.0, 5: 1.0}
    realized = {1: 30.0, 2: 20.0, 3: 10.0, 4: 5.0, 5: 1.0}
    picked, regret = triple_regret([1, 2, 3, 4, 5], forecast, realized)
    assert picked == (1, 2, 3)
    assert regret == pytest.approx(0.0)


def test_triple_regret_is_the_realized_shortfall_not_the_forecast_error() -> None:
    """Ranking wrong only costs what it actually cost."""
    from backtest_trajectory import triple_regret

    forecast = {1: 30.0, 2: 20.0, 3: 10.0, 4: 9.0, 5: 1.0}
    realized = {1: 5.0, 2: 20.0, 3: 10.0, 4: 25.0, 5: 1.0}
    picked, regret = triple_regret([1, 2, 3, 4, 5], forecast, realized)
    assert picked == (1, 2, 3)
    # best available was 4 + 2 + 3 = 55; picked 5 + 20 + 10 = 35
    assert regret == pytest.approx(20.0)


def test_agreement_rate_counts_identical_triples() -> None:
    """The likeliest outcome is that both estimators name the same three. Those rows
    contribute zero to the difference while counting toward n, so a bootstrap over 20
    decisions of which 18 agree reports a tight interval around zero that MEANS the
    slice had two informative rows."""
    from backtest_trajectory import agreement_rate

    shape = [(1, 2, 3), (1, 2, 3), (4, 5, 6)]
    keeper = [(1, 2, 3), (1, 2, 3), (7, 8, 9)]
    assert agreement_rate(shape, keeper) == pytest.approx(2 / 3)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_scripts/test_backtest_historical.py -k "triple or agreement" -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

`triple_regret` sorts candidates by `forecast` descending (tie-break on `mlbam_id` for determinism), takes `keep`, and returns the picks plus `sum(top-keep realized) - sum(picked realized)`. `agreement_rate` is the share of index positions where the two pick tuples are equal as sets.

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_scripts/test_backtest_historical.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_trajectory.py tests/test_scripts/test_backtest_historical.py
git commit -m "backtest: keeper-triple regret and estimator agreement rate (#325)"
```

---

### Task 10: The bootstrap, with the right resampling unit per slice

**Files:**
- Modify: `scripts/backtest_trajectory.py`
- Test: `tests/test_scripts/test_backtest_historical.py`

**Interfaces:**
- Produces: `bootstrap_difference(a: Sequence[float], b: Sequence[float], *, draws: int = 10_000, seed: int = 7) -> tuple[float, float, float]` returning `(lo, hi, share_a_better)`.

**Resampling unit:** team-decisions (cluster) for keeper-triple regret; players for top-of-board and breakout. Resampling players inside a roster would change the roster, which changes the ex-post optimum and leaves regret undefined.

- [ ] **Step 1: Write the failing tests**

```python
def test_bootstrap_difference_separates_an_obvious_gap() -> None:
    from backtest_trajectory import bootstrap_difference

    a = [1.0] * 40
    b = [5.0] * 40
    lo, hi, share = bootstrap_difference(a, b)
    assert hi < 0
    assert share == pytest.approx(1.0)


def test_bootstrap_difference_reports_a_null_as_straddling_zero() -> None:
    from backtest_trajectory import bootstrap_difference

    rng = np.random.default_rng(0)
    values = rng.normal(size=40)
    lo, hi, _ = bootstrap_difference(list(values), list(values))
    assert lo <= 0 <= hi


def test_bootstrap_difference_is_deterministic_for_a_seed() -> None:
    from backtest_trajectory import bootstrap_difference

    a, b = [1.0, 2.0, 3.0, 4.0], [2.0, 2.0, 2.0, 9.0]
    assert bootstrap_difference(a, b, seed=3) == bootstrap_difference(a, b, seed=3)
```

- [ ] **Step 2: Run to verify they fail**

Run: `pytest tests/test_scripts/test_backtest_historical.py -k bootstrap -v`
Expected: FAIL.

- [ ] **Step 3: Implement**

Paired resample of indices with `np.random.default_rng(seed)`; the statistic is `mean(a[idx]) - mean(b[idx])`; return the 2.5/97.5 percentiles and the share of draws where `a` is lower (for regret, lower is better — document the direction in the docstring, since it inverts between regret and error).

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_scripts/test_backtest_historical.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_trajectory.py tests/test_scripts/test_backtest_historical.py
git commit -m "backtest: paired bootstrap on the estimator difference (#325)"
```

---

### Task 11: Wire the historical mode into the CLI and run it

**Files:**
- Modify: `scripts/backtest_trajectory.py` (`main`)
- Test: manual run; no new unit test

**Interfaces:**
- Consumes: everything above.
- Produces: `python scripts/backtest_trajectory.py --historical --base-year 2023 --pool hitter`.

- [ ] **Step 1: Add the arguments and the report**

New flags: `--historical`, `--base-year` (repeatable, defaults to 2022 2023 2024), `--horizon-max` (1 or 2), `--causal-check`, `--censor-threshold` (default 0.5), `--draws` (default 10000). The report prints, per base year and pool:

- coverage: players each estimator alone could score, and the intersection size
- fallback report and the future-transition count
- the censored list (name, anchor volume, outcome volume), split zero vs non-zero, at 0.5 and 0.2
- keeper-triple regret at both horizons, both views, with agreement rate and the disagreeing-subset difference
- top-of-board and breakout, both views
- bootstrap interval and win share on every headline

Per-view roster counts are printed for the triple slice, and any roster present in one view but not the other is named — otherwise a difference between views confounds injury exclusion with a changed roster set.

- [ ] **Step 2: Run for one base year to smoke it out**

Run: `python scripts/backtest_trajectory.py --historical --base-year 2023 --pool hitter`
Expected: a complete report with no traceback. Investigate any base year that reports a whole-pool fallback before proceeding.

- [ ] **Step 3: Run the full matrix**

```bash
python scripts/backtest_trajectory.py --historical --pool hitter  --out data/analysis/backtest_hitter.csv
python scripts/backtest_trajectory.py --historical --pool pitcher --out data/analysis/backtest_pitcher.csv
python scripts/backtest_trajectory.py --historical --base-year 2023 --causal-check --pool hitter
python scripts/backtest_trajectory.py --historical --base-year 2023 --causal-check --pool pitcher
```

- [ ] **Step 4: Full gate**

Run: `pytest -v && ruff check . && ruff format --check . && vulture && mypy`
Expected: all clean. Report exactly what each returned; do not summarize as "checks pass".

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_trajectory.py
git commit -m "backtest: historical shape-vs-keeper-value mode (#325)"
```

---

### Task 12: Write the verdict

**Files:**
- No code. Output only.

- [ ] **Step 1: Assemble the tables**

Every table from Task 11, both views, both pools, with counts and intervals.

- [ ] **Step 2: State the verdict against the pre-registered criterion**

From the spec, unchanged and not to be reinterpreted after seeing the numbers:

> A verdict of "cannot separate" is acceptable and does not block PR 2 or PR 3. The only result that **blocks** deletion is keeper-value beating shape on the keeper-triple or top-of-board slice by a margin the bootstrap separates from zero, in either view.

If the agreement rate is high, say so plainly: it means the two engines do not disagree where the decision is made, which is grounds for deleting one on simplicity with no accuracy claim attached.

- [ ] **Step 3: Post it**

```bash
gh issue comment 325 --body-file <verdict.md>
```

and open the PR with the same tables in the body.

- [ ] **Step 4: Record the measurement commit**

Note the commit sha of Task 11 — PR 3 cites it for numbers produced by this run, and cites PR 2's parent for the older shape-vs-`current` tables (which came from a tree that still contained `comps.py`).

---

## Self-Review

**Spec coverage.** One scale -> Tasks 1, 2, 6. Chain runnable out of sample -> Tasks 3, 4, 5. Three advantages / leakage disclosure -> Task 4. Target and eligibility -> Task 6. Two views -> Task 7. Slices -> Tasks 8, 9, 11. Coverage and low-support -> Task 11. Noise floor -> Task 10. PR 1 verification list -> characterization (Task 1), censor boundaries (Task 7), historical-mode guards (Tasks 3, 5), regret (Task 9), multi-year censoring (Task 7). Verdict -> Task 12.

**Gap found and left open deliberately:** the spec's "the fit inputs and the fold inputs carry the same era factors" test is Task 2's cross-loader test, which requires real ZiPS files on disk. If `data/projections/2024/` is absent in a given checkout that test must skip with a stated reason rather than being deleted.

**Type consistency.** `era_factors` returns a season-indexed frame in Tasks 1 and 2. `normalize_frame(frame, season, kind, factors)` keeps that argument order at every call site. `volume_forecast` returns a tuple from Task 5 onward — Task 3's signature change and Task 5's return change touch the same function, so Task 5 must update Task 3's call sites, and its test asserts the tuple shape.
