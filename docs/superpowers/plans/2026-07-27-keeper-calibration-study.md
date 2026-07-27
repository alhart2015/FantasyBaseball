# Keeper-value calibration study (increment 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure how much of a season's performance surprise should carry forward into a
projection that has not seen that season, so #266's keeper metric can fold 2026 into the stale
ZiPS 2027 baseline.

**Architecture:** Four new modules under `src/fantasy_baseball/keepers/` plus one script. Raw
MLB actuals are normalized to a canonical rate/playing-time frame; ZiPS vintages are decomposed
to the same schema; a shared fold module applies shrink, gating and reconstruction (reused
verbatim by increment 2); a calibration module assembles year-pairs, measures sample size and
survivorship, and runs a pluggable estimator through leave-one-pair-out evaluation against two
fixed endpoints.

**Tech Stack:** Python 3.11, pandas, numpy, pytest. No network at import time; all fetchers take
an injectable `fetcher` callable so tests never hit the API.

**Spec:** `docs/superpowers/specs/2026-07-27-keeper-value-definition-design.md` (sections 5.2-5.5,
6.1-6.6, 9).

## Global Constraints

- **ASCII only** in all source, log messages, and printed output. This is a Windows box; stdout
  is cp1252 and a non-ASCII glyph crashes the script. Use `-`, `--`, `'`, `"`, `->`, `sigma`.
- **`keepers/` is under mypy coverage** (`pyproject.toml` line 79). Every new module must pass
  `mypy` clean, including `warn_return_any` -- annotate locals holding untyped pandas returns.
- **Player identity is MLBAM id** for actuals-to-ZiPS joins, `PlayerId` (FanGraphs) across ZiPS
  vintages. Never key on bare names.
- **No SGP, VAR, board, or cache imports.** Increment 1 is standalone: ZiPS CSVs on disk plus the
  MLB Stats API only. A `from fantasy_baseball.sgp...` import in any file this plan creates is a
  defect.
- **Scored categories** are R, HR, RBI, SB, AVG (hitters) and W, K, SV, ERA, WHIP (pitchers). SV
  is out of scope for the out-years (spec 5.1) and is not folded.
- **Do not cache in-progress seasons.** `fetch_or_cache` never invalidates
  (`keepers/cache.py:23-38`), so a 2026 pull must not go through it.
- **Run at the end of every task:** `pytest -v <the task's test file>`, `ruff check .`,
  `ruff format --check .`, `mypy`. All must be clean before commit.

## File Structure

| File | Responsibility |
|---|---|
| `src/fantasy_baseball/keepers/actuals.py` | Convert a raw MLB Stats API frame to the canonical rate/PT schema. Owns the baseball-notation innings conversion and numeric coercion. |
| `src/fantasy_baseball/keepers/vintages.py` | Load a ZiPS vintage from disk and decompose it to the same canonical schema. |
| `src/fantasy_baseball/keepers/fold.py` | The fold itself: shrink, gate, reconstruction, clamps. Pure functions, no I/O. **Reused unchanged by increment 2.** |
| `src/fantasy_baseball/keepers/calibration.py` | Year-pair assembly, sample-size and survivorship measurement, estimator protocol, leave-one-pair-out evaluation, per-coefficient acceptance. |
| `scripts/keeper_calibration.py` | CLI entry point; writes the results table to `data/analysis/`. |
| `tests/test_keepers/test_actuals.py` | |
| `tests/test_keepers/test_vintages.py` | |
| `tests/test_keepers/test_fold.py` | |
| `tests/test_keepers/test_calibration.py` | |

**Canonical schema** (both actuals and ZiPS decompose to this; every task depends on it):

```python
HITTER_RATES = ("hr_pa", "r_pa", "rbi_pa", "sb_pa", "h_ab", "ab_pa")
PITCHER_RATES = ("k_ip", "w_ip", "er_ip", "bb_ip", "h_ip")
HITTER_PT = "pa"
PITCHER_PT = "ip"
```

Frames are indexed by `mlbam_id` (int) and carry exactly the rate columns plus the PT column.

---

### Task 1: Baseball-notation innings and numeric coercion

**Files:**
- Create: `src/fantasy_baseball/keepers/actuals.py`
- Test: `tests/test_keepers/test_actuals.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `innings_to_float(value: object) -> float`, `coerce_numeric(value: object) -> float`.

Spec 6.5: `stat.inningsPitched` is a **string in baseball notation** -- `"5.1"` means 5 1/3
innings, not 5.1. Verified against a live 2025 pull. ZiPS IP is decimal, so differencing without
conversion is invalid. `stat.era` and `stat.whip` are also strings.

- [ ] **Step 1: Write the failing test**

```python
import pytest

from fantasy_baseball.keepers.actuals import coerce_numeric, innings_to_float


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5.1", 5 + 1 / 3),
        ("5.2", 5 + 2 / 3),
        ("7.0", 7.0),
        ("0.1", 1 / 3),
        ("12", 12.0),
        (0, 0.0),
        (None, 0.0),
        ("", 0.0),
    ],
)
def test_innings_to_float(raw: object, expected: float) -> None:
    assert innings_to_float(raw) == pytest.approx(expected)


def test_innings_rejects_impossible_outs() -> None:
    # Only .0/.1/.2 are legal; .3 would silently become a third of an inning too many.
    with pytest.raises(ValueError):
        innings_to_float("5.3")


def test_coerce_numeric_handles_api_junk() -> None:
    assert coerce_numeric("3.45") == pytest.approx(3.45)
    assert coerce_numeric(None) == 0.0
    assert coerce_numeric("-.--") == 0.0
    assert coerce_numeric("") == 0.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_keepers/test_actuals.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'fantasy_baseball.keepers.actuals'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Normalize raw MLB Stats API season frames to the canonical rate/PT schema.

The API returns innings as a baseball-notation STRING ("5.1" = 5 1/3), and ERA/WHIP
as strings, so every numeric field is coerced explicitly. See spec section 6.5.
"""

from __future__ import annotations

_NULLISH = {"", "nan", "none", "-", "-.--", ".---"}


def coerce_numeric(value: object) -> float:
    """Best-effort float for an MLB Stats API scalar; nullish/unparseable -> 0.0."""
    if value is None:
        return 0.0
    text = str(value).strip()
    if text.lower() in _NULLISH:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def innings_to_float(value: object) -> float:
    """Convert baseball-notation innings to decimal innings.

    The fractional digit counts OUTS, not tenths: "5.1" is 5 1/3 innings. Only .0,
    .1 and .2 are legal; anything else means the input was not baseball notation
    and is raised rather than silently mis-scaled.
    """
    if value is None:
        return 0.0
    text = str(value).strip()
    if text.lower() in _NULLISH:
        return 0.0
    if "." not in text:
        return coerce_numeric(text)
    whole, _, frac = text.partition(".")
    outs_text = frac[:1] or "0"
    if outs_text not in {"0", "1", "2"}:
        raise ValueError(f"not baseball-notation innings: {value!r}")
    return coerce_numeric(whole) + int(outs_text) / 3.0
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_keepers/test_actuals.py -v`
Expected: PASS (11 tests)

- [ ] **Step 5: Run the gates and commit**

```bash
ruff check . && ruff format --check . && mypy
git add src/fantasy_baseball/keepers/actuals.py tests/test_keepers/test_actuals.py
git commit -m "feat(keepers): baseball-notation innings + numeric coercion (#266)"
```

---

### Task 2: Normalize MLB actuals to the canonical schema

**Files:**
- Modify: `src/fantasy_baseball/keepers/actuals.py`
- Test: `tests/test_keepers/test_actuals.py`

**Interfaces:**
- Consumes: `innings_to_float`, `coerce_numeric` from Task 1.
- Produces: `normalize_hitting(raw: pd.DataFrame) -> pd.DataFrame` and
  `normalize_pitching(raw: pd.DataFrame) -> pd.DataFrame`, each returning a frame indexed by
  `mlbam_id` with the canonical rate columns plus the PT column.

Field names verified against a live 2025 pull (spec section 11): hitters carry
`stat.plateAppearances`, `stat.atBats`, `stat.hits`, `stat.runs`, `stat.homeRuns`, `stat.rbi`,
`stat.stolenBases`; pitchers carry `stat.earnedRuns`, `stat.baseOnBalls`, `stat.hits`,
`stat.inningsPitched`, `stat.strikeOuts`, `stat.wins`.

Zero-PT rows produce `NaN` rates (`0/0`), which spec 5.5 requires be guarded explicitly rather
than left to propagate -- `0 * NaN` is `NaN`, so a zero shrink does **not** rescue them.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from fantasy_baseball.keepers.actuals import normalize_hitting, normalize_pitching


def test_normalize_hitting_builds_rates_on_correct_denominators() -> None:
    raw = pd.DataFrame(
        {
            "player.id": [1],
            "stat.plateAppearances": [600],
            "stat.atBats": [540],
            "stat.hits": [162],
            "stat.runs": [90],
            "stat.homeRuns": [30],
            "stat.rbi": [100],
            "stat.stolenBases": [12],
        }
    )
    out = normalize_hitting(raw)
    row = out.loc[1]
    assert row["pa"] == 600.0
    assert row["ab_pa"] == pytest.approx(540 / 600)
    assert row["h_ab"] == pytest.approx(162 / 540)  # AB, not PA
    assert row["hr_pa"] == pytest.approx(30 / 600)
    assert row["sb_pa"] == pytest.approx(12 / 600)


def test_normalize_pitching_converts_innings_notation() -> None:
    raw = pd.DataFrame(
        {
            "player.id": [7],
            "stat.inningsPitched": ["180.1"],
            "stat.earnedRuns": [60],
            "stat.baseOnBalls": [45],
            "stat.hits": [150],
            "stat.strikeOuts": [200],
            "stat.wins": [15],
        }
    )
    out = normalize_pitching(raw)
    row = out.loc[7]
    assert row["ip"] == pytest.approx(180 + 1 / 3)
    assert row["er_ip"] == pytest.approx(60 / (180 + 1 / 3))
    assert row["k_ip"] == pytest.approx(200 / (180 + 1 / 3))


def test_zero_playing_time_yields_nan_rates_not_zeros() -> None:
    # 0/0 must be NaN so the gate can see "no information", NOT 0.0 which reads
    # as "a real observation of zero rate" and would crush the fold.
    raw = pd.DataFrame(
        {
            "player.id": [3],
            "stat.plateAppearances": [0],
            "stat.atBats": [0],
            "stat.hits": [0],
            "stat.runs": [0],
            "stat.homeRuns": [0],
            "stat.rbi": [0],
            "stat.stolenBases": [0],
        }
    )
    out = normalize_hitting(raw)
    assert out.loc[3, "pa"] == 0.0
    assert pd.isna(out.loc[3, "hr_pa"])


def test_normalize_drops_rows_without_an_mlbam_id() -> None:
    raw = pd.DataFrame(
        {
            "player.id": [1, None],
            "stat.plateAppearances": [600, 100],
            "stat.atBats": [540, 90],
            "stat.hits": [162, 27],
            "stat.runs": [90, 10],
            "stat.homeRuns": [30, 2],
            "stat.rbi": [100, 9],
            "stat.stolenBases": [12, 1],
        }
    )
    assert list(normalize_hitting(raw).index) == [1]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_keepers/test_actuals.py -v`
Expected: FAIL, `ImportError: cannot import name 'normalize_hitting'`

- [ ] **Step 3: Write minimal implementation**

Append to `actuals.py` (add `import numpy as np` and `import pandas as pd` at the top):

```python
HITTER_RATES = ("hr_pa", "r_pa", "rbi_pa", "sb_pa", "h_ab", "ab_pa")
PITCHER_RATES = ("k_ip", "w_ip", "er_ip", "bb_ip", "h_ip")
HITTER_PT = "pa"
PITCHER_PT = "ip"


def _safe_ratio(numer: pd.Series, denom: pd.Series) -> pd.Series:
    """Elementwise numer/denom with 0/0 -> NaN (never 0.0).

    NaN is the honest answer for "no observation"; 0.0 would read downstream as a
    real observation of a zero rate. Spec 5.5.
    """
    result: pd.Series = numer.divide(denom.where(denom > 0, other=np.nan))
    return result


def _indexed(raw: pd.DataFrame) -> pd.DataFrame:
    frame = raw.loc[raw["player.id"].notna()].copy()
    frame["mlbam_id"] = frame["player.id"].astype(int)
    return frame.set_index("mlbam_id")


def normalize_hitting(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _indexed(raw)
    num = {
        col: frame[f"stat.{col}"].map(coerce_numeric)
        for col in ("plateAppearances", "atBats", "hits", "runs", "homeRuns", "rbi", "stolenBases")
    }
    pa, ab = num["plateAppearances"], num["atBats"]
    out = pd.DataFrame(
        {
            HITTER_PT: pa,
            "ab_pa": _safe_ratio(ab, pa),
            "h_ab": _safe_ratio(num["hits"], ab),
            "hr_pa": _safe_ratio(num["homeRuns"], pa),
            "r_pa": _safe_ratio(num["runs"], pa),
            "rbi_pa": _safe_ratio(num["rbi"], pa),
            "sb_pa": _safe_ratio(num["stolenBases"], pa),
        },
        index=frame.index,
    )
    return out


def normalize_pitching(raw: pd.DataFrame) -> pd.DataFrame:
    frame = _indexed(raw)
    ip = frame["stat.inningsPitched"].map(innings_to_float)
    num = {
        col: frame[f"stat.{col}"].map(coerce_numeric)
        for col in ("earnedRuns", "baseOnBalls", "hits", "strikeOuts", "wins")
    }
    out = pd.DataFrame(
        {
            PITCHER_PT: ip,
            "k_ip": _safe_ratio(num["strikeOuts"], ip),
            "w_ip": _safe_ratio(num["wins"], ip),
            "er_ip": _safe_ratio(num["earnedRuns"], ip),
            "bb_ip": _safe_ratio(num["baseOnBalls"], ip),
            "h_ip": _safe_ratio(num["hits"], ip),
        },
        index=frame.index,
    )
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_keepers/test_actuals.py -v`
Expected: PASS. Add `import pytest` to the test file if not already present.

- [ ] **Step 5: Run the gates and commit**

```bash
ruff check . && ruff format --check . && mypy
git add src/fantasy_baseball/keepers/actuals.py tests/test_keepers/test_actuals.py
git commit -m "feat(keepers): normalize MLB actuals to canonical rate/PT schema (#266)"
```

---

### Task 3: Decompose ZiPS vintages to the same schema

**Files:**
- Create: `src/fantasy_baseball/keepers/vintages.py`
- Test: `tests/test_keepers/test_vintages.py`

**Interfaces:**
- Consumes: `HITTER_RATES`, `PITCHER_RATES`, `HITTER_PT`, `PITCHER_PT`, `_safe_ratio` from
  `actuals.py`. Re-export `_safe_ratio` as `safe_ratio` from `actuals.py` in this task so
  `vintages.py` does not import a private name.
- Produces: `decompose_hitters(df: pd.DataFrame) -> pd.DataFrame`,
  `decompose_pitchers(df: pd.DataFrame) -> pd.DataFrame`,
  `load_vintage(year: int, projections_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]`.

ZiPS CSVs carry `MLBAMID` in every vintage (verified 2022-2028), so the join to actuals is by id
with no name matching. `load_projection_set` (`data/fangraphs.py`) already handles the
year-suffixed filename variants via its glob fallback, but it lowercases columns -- this task
reads the raw CSV directly to keep `MLBAMID`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.keepers.vintages import decompose_hitters, decompose_pitchers, load_vintage


def test_decompose_hitters_uses_own_denominators() -> None:
    df = pd.DataFrame(
        {
            "MLBAMID": [11],
            "PA": [600],
            "AB": [540],
            "H": [162],
            "R": [90],
            "HR": [30],
            "RBI": [100],
            "SB": [12],
        }
    )
    row = decompose_hitters(df).loc[11]
    assert row["h_ab"] == pytest.approx(162 / 540)
    assert row["hr_pa"] == pytest.approx(30 / 600)


def test_decompose_pitchers_splits_bb_and_hits() -> None:
    # BB and H_allowed must stay separate: calculate_replacement_rates needs them
    # as distinct columns, so folding (BB+H)/IP as one unit is not acceptable.
    df = pd.DataFrame(
        {"MLBAMID": [22], "IP": [180.0], "ER": [60], "BB": [45], "H": [150], "SO": [200], "W": [15]}
    )
    row = decompose_pitchers(df).loc[22]
    assert row["bb_ip"] == pytest.approx(45 / 180)
    assert row["h_ip"] == pytest.approx(150 / 180)
    assert "whip" not in row.index


def test_zero_denominator_zips_rows_yield_nan(tmp_path: Path) -> None:
    # ZiPS 2028 has 14 pitcher rows at IP=0 and 3 hitter rows at PA=0.
    df = pd.DataFrame({"MLBAMID": [33], "IP": [0.0], "ER": [0], "BB": [0], "H": [0], "SO": [0], "W": [0]})
    assert pd.isna(decompose_pitchers(df).loc[33, "k_ip"])


def test_load_vintage_reads_real_files() -> None:
    hitters, pitchers = load_vintage(2026, Path("data/projections"))
    assert len(hitters) > 1000
    assert len(pitchers) > 1000
    assert hitters.index.name == "mlbam_id"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_keepers/test_vintages.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'fantasy_baseball.keepers.vintages'`

- [ ] **Step 3: Write minimal implementation**

First, in `actuals.py`, rename `_safe_ratio` to `safe_ratio` (update its two call sites in that
file). Then create `vintages.py`:

```python
"""Load a ZiPS vintage from disk and decompose it to the canonical rate/PT schema.

Reads the raw CSV rather than going through data.fangraphs.load_projection_set,
because that lowercases and remaps columns and drops MLBAMID -- which is the join
key to the MLB actuals. Filename variants (year-suffixed, proj-from-dated) are
resolved by glob here, matching the loader's own fallback behaviour.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.actuals import HITTER_PT, PITCHER_PT, safe_ratio


def _find(directory: Path, player_type: str) -> Path:
    exact = directory / f"zips-{player_type}.csv"
    if exact.exists():
        return exact
    matches = sorted(directory.glob(f"zips-{player_type}-*.csv"))
    if not matches:
        raise FileNotFoundError(f"no ZiPS {player_type} export under {directory}")
    return matches[-1]


def _indexed(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.loc[df["MLBAMID"].notna()].copy()
    frame["mlbam_id"] = frame["MLBAMID"].astype(int)
    return frame.set_index("mlbam_id")


def decompose_hitters(df: pd.DataFrame) -> pd.DataFrame:
    frame = _indexed(df)
    pa, ab = frame["PA"].astype(float), frame["AB"].astype(float)
    return pd.DataFrame(
        {
            HITTER_PT: pa,
            "ab_pa": safe_ratio(ab, pa),
            "h_ab": safe_ratio(frame["H"].astype(float), ab),
            "hr_pa": safe_ratio(frame["HR"].astype(float), pa),
            "r_pa": safe_ratio(frame["R"].astype(float), pa),
            "rbi_pa": safe_ratio(frame["RBI"].astype(float), pa),
            "sb_pa": safe_ratio(frame["SB"].astype(float), pa),
        },
        index=frame.index,
    )


def decompose_pitchers(df: pd.DataFrame) -> pd.DataFrame:
    frame = _indexed(df)
    ip = frame["IP"].astype(float)
    return pd.DataFrame(
        {
            PITCHER_PT: ip,
            "k_ip": safe_ratio(frame["SO"].astype(float), ip),
            "w_ip": safe_ratio(frame["W"].astype(float), ip),
            "er_ip": safe_ratio(frame["ER"].astype(float), ip),
            "bb_ip": safe_ratio(frame["BB"].astype(float), ip),
            "h_ip": safe_ratio(frame["H"].astype(float), ip),
        },
        index=frame.index,
    )


def load_vintage(year: int, projections_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    directory = projections_root / str(year)
    hitters = pd.read_csv(_find(directory, "hitters"))
    pitchers = pd.read_csv(_find(directory, "pitchers"))
    return decompose_hitters(hitters), decompose_pitchers(pitchers)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_keepers/test_vintages.py tests/test_keepers/test_actuals.py -v`
Expected: PASS. Both files, since `safe_ratio` was renamed.

- [ ] **Step 5: Run the gates and commit**

```bash
ruff check . && ruff format --check . && mypy
git add src/fantasy_baseball/keepers/vintages.py src/fantasy_baseball/keepers/actuals.py tests/test_keepers/
git commit -m "feat(keepers): decompose ZiPS vintages to canonical schema (#266)"
```

---

### Task 4: The fold -- shrink, gate, reconstruction

**Files:**
- Create: `src/fantasy_baseball/keepers/fold.py`
- Test: `tests/test_keepers/test_fold.py`

**Interfaces:**
- Consumes: the canonical schema constants from `actuals.py`.
- Produces: `shrink(n: pd.Series, n0: float) -> pd.Series`,
  `gate_mask(realized_pt: pd.Series, threshold: float) -> pd.Series`,
  `fold_rates(base, residual, weight, k) -> pd.DataFrame`,
  `reconstruct_hitter(rates: pd.DataFrame, pa: pd.Series) -> pd.DataFrame`,
  `reconstruct_pitcher(rates: pd.DataFrame, ip: pd.Series) -> pd.DataFrame`.

**This module is reused unchanged by increment 2.** It is pure -- no I/O, no config reads.

Spec requirements encoded here: the shrink is bounded at or below 1 (5.3); the shrink applies to
**rate** residuals only, never to playing time (5.3); each rate multiplies its **own** denominator
with AB derived from PA first (5.2); reconstructed rates are floored at 0 and `0/0` guarded on the
output side, because a negative `er_ip` would produce a negative ERA which the scoring path rewards
without bound (5.2).

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.keepers.fold import (
    fold_rates,
    gate_mask,
    reconstruct_hitter,
    reconstruct_pitcher,
    shrink,
)


def test_shrink_is_bounded_and_monotone() -> None:
    n = pd.Series([0.0, 50.0, 200.0, 600.0, 5000.0])
    w = shrink(n, n0=200.0)
    assert w.iloc[0] == 0.0
    assert w.iloc[2] == pytest.approx(0.5)
    assert (w < 1.0).all()          # never amplifies
    assert w.is_monotonic_increasing


def test_gate_excludes_low_playing_time() -> None:
    realized = pd.Series([0.0, 30.0, 200.0], index=[1, 2, 3])
    mask = gate_mask(realized, threshold=50.0)
    assert list(mask) == [False, False, True]


def test_gate_treats_missing_as_ungated() -> None:
    # Absence from the MLB leaderboard means AAA, not zero PA -- it must fall
    # through to passthrough, never to a large negative residual. Spec 5.4.
    realized = pd.Series([np.nan], index=[9])
    assert list(gate_mask(realized, threshold=50.0)) == [False]


def test_fold_rates_endpoints() -> None:
    base = pd.DataFrame({"hr_pa": [0.04]})
    resid = pd.DataFrame({"hr_pa": [0.02]})
    weight = pd.Series([1.0])
    assert fold_rates(base, resid, weight, k=0.0)["hr_pa"].iloc[0] == pytest.approx(0.04)
    assert fold_rates(base, resid, weight, k=1.0)["hr_pa"].iloc[0] == pytest.approx(0.06)
    assert fold_rates(base, resid, weight, k=0.5)["hr_pa"].iloc[0] == pytest.approx(0.05)


def test_fold_rates_floors_at_zero() -> None:
    base = pd.DataFrame({"sb_pa": [0.01]})
    resid = pd.DataFrame({"sb_pa": [-0.30]})
    out = fold_rates(base, resid, pd.Series([1.0]), k=1.0)
    assert out["sb_pa"].iloc[0] == 0.0     # never negative


def test_reconstruct_hitter_uses_ab_for_hits_and_pa_for_counting() -> None:
    rates = pd.DataFrame({"ab_pa": [0.9], "h_ab": [0.300], "hr_pa": [0.05],
                          "r_pa": [0.15], "rbi_pa": [0.16], "sb_pa": [0.02]})
    out = reconstruct_hitter(rates, pd.Series([600.0]))
    assert out["ab"].iloc[0] == pytest.approx(540.0)
    assert out["h"].iloc[0] == pytest.approx(540.0 * 0.300)   # AB, not PA
    assert out["avg"].iloc[0] == pytest.approx(0.300)          # not inflated by 1/0.9
    assert out["hr"].iloc[0] == pytest.approx(30.0)
    assert out["ab"].iloc[0] <= out["pa"].iloc[0]              # structural


def test_reconstruct_hitter_guards_zero_ab() -> None:
    rates = pd.DataFrame({"ab_pa": [0.9], "h_ab": [0.300], "hr_pa": [0.05],
                          "r_pa": [0.15], "rbi_pa": [0.16], "sb_pa": [0.02]})
    out = reconstruct_hitter(rates, pd.Series([0.0]))
    assert out["avg"].iloc[0] == 0.0      # 0/0 guarded, not NaN


def test_reconstruct_pitcher_builds_era_and_whip_from_components() -> None:
    rates = pd.DataFrame({"k_ip": [1.0], "w_ip": [0.08], "er_ip": [0.35],
                          "bb_ip": [0.25], "h_ip": [0.80]})
    out = reconstruct_pitcher(rates, pd.Series([180.0]))
    assert out["era"].iloc[0] == pytest.approx(9 * 0.35)
    assert out["whip"].iloc[0] == pytest.approx(0.25 + 0.80)
    assert out["k"].iloc[0] == pytest.approx(180.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_keepers/test_fold.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'fantasy_baseball.keepers.fold'`

- [ ] **Step 3: Write minimal implementation**

```python
"""The fold: shrink a rate residual, gate on realized playing time, reconstruct a line.

Pure functions -- no I/O, no config. Increment 2 reuses this module unchanged.

Two rules here are load-bearing and were both wrong in earlier spec drafts:
  * The shrink applies to RATE residuals only. Applying it to playing time would
    damp an injury signal in proportion to the playing time the injury suppressed.
  * Each rate multiplies its OWN denominator, with AB derived from PA first.
    Multiplying H/AB by PA inflates AVG by 1/0.8977 (a .250 hitter scores .278).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def shrink(n: pd.Series, n0: float) -> pd.Series:
    """Sample-size shrink in [0, 1): n / (n + n0). Never amplifies.

    `n` is REALIZED playing time -- only observed opportunities carry sampling
    noise, so a blended full-season figure would understate it (spec 5.3).
    """
    filled = n.fillna(0.0).clip(lower=0.0)
    result: pd.Series = filled / (filled + n0)
    return result


def gate_mask(realized_pt: pd.Series, threshold: float) -> pd.Series:
    """True where the player has enough realized MLB playing time to be folded.

    NaN (absent from the MLB leaderboard -- i.e. in the minors) is False: absence
    is not an observation of zero, and folding it would gut the projection.
    """
    result: pd.Series = realized_pt.fillna(0.0) >= threshold
    return result


def fold_rates(
    base: pd.DataFrame, residual: pd.DataFrame, weight: pd.Series, k: float
) -> pd.DataFrame:
    """base + k * weight * residual, per rate column, floored at 0."""
    out = base.copy()
    for col in base.columns:
        moved = base[col] + k * weight * residual[col].fillna(0.0)
        out[col] = moved.clip(lower=0.0)
    return out


def _guarded(numer: pd.Series, denom: pd.Series) -> pd.Series:
    result: pd.Series = numer.divide(denom.where(denom > 0, other=np.nan)).fillna(0.0)
    return result


def reconstruct_hitter(rates: pd.DataFrame, pa: pd.Series) -> pd.DataFrame:
    ab = (pa * rates["ab_pa"]).clip(lower=0.0)
    h = ab * rates["h_ab"]
    return pd.DataFrame(
        {
            "pa": pa,
            "ab": ab,
            "h": h,
            "avg": _guarded(h, ab),
            "hr": pa * rates["hr_pa"],
            "r": pa * rates["r_pa"],
            "rbi": pa * rates["rbi_pa"],
            "sb": pa * rates["sb_pa"],
        },
        index=rates.index,
    )


def reconstruct_pitcher(rates: pd.DataFrame, ip: pd.Series) -> pd.DataFrame:
    er = ip * rates["er_ip"]
    bb = ip * rates["bb_ip"]
    hits = ip * rates["h_ip"]
    return pd.DataFrame(
        {
            "ip": ip,
            "er": er,
            "bb": bb,
            "h_allowed": hits,
            "era": _guarded(9.0 * er, ip),
            "whip": _guarded(bb + hits, ip),
            "k": ip * rates["k_ip"],
            "w": ip * rates["w_ip"],
        },
        index=rates.index,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_keepers/test_fold.py -v`
Expected: PASS (9 tests)

- [ ] **Step 5: Run the gates and commit**

```bash
ruff check . && ruff format --check . && mypy
git add src/fantasy_baseball/keepers/fold.py tests/test_keepers/test_fold.py
git commit -m "feat(keepers): fold primitives -- shrink, gate, reconstruction (#266)"
```

---

### Task 5: Year-pair assembly, sample size and survivorship -- including pitchers

**Files:**
- Create: `src/fantasy_baseball/keepers/calibration.py`
- Test: `tests/test_keepers/test_calibration.py`

**Interfaces:**
- Consumes: `load_vintage` (Task 3), `normalize_hitting`/`normalize_pitching` (Task 2),
  `fetch_mlb_season` (`keepers/mlb_stats.py`), `gate_mask` (Task 4).
- Produces: `YearPair` dataclass with fields `year`, `base`, `residual`, `target`,
  `realized_pt`; `build_pairs(...) -> list[YearPair]`;
  `survivorship(pairs, threshold) -> pd.DataFrame`.

**Spec 6.3 makes this the mandatory first measurement.** Every sample-size and survivorship
figure in the spec is hitters-only, while six of the twelve coefficients are pitcher-side. The
threshold choice swings the pitcher sample fourfold (ZiPS 2023 has 1881 pitchers, 1316 at IP>=50
but only 328 at IP>=100). This task must report both player types on the same footing, and report
whether the pitcher study is adequately powered.

Usable pairs are **2022->2023, 2023->2024, 2024->2025** only. A 2025 pair needs a complete 2026
season (`season_end: 2026-09-28`); a 2021 pair is impossible because `data/projections/` starts at
2022.

- [ ] **Step 1: Write the failing test**

```python
import pandas as pd

from fantasy_baseball.keepers.calibration import PAIR_YEARS, YearPair, survivorship


def test_pair_years_are_the_three_usable_ones() -> None:
    # 2025->2026 needs a complete 2026 season; 2021->2022 has no ZiPS 2021 on disk.
    assert PAIR_YEARS == (2022, 2023, 2024)


def test_survivorship_counts_players_who_kept_playing() -> None:
    pair = YearPair(
        year=2022,
        base=pd.DataFrame({"hr_pa": [0.04, 0.03]}, index=[1, 2]),
        residual=pd.DataFrame({"hr_pa": [0.01, -0.01]}, index=[1, 2]),
        target=pd.DataFrame({"hr_pa": [0.045, float("nan")]}, index=[1, 2]),
        realized_pt=pd.Series([500.0, 300.0], index=[1, 2]),
        target_pt=pd.Series([550.0, 0.0], index=[1, 2]),
    )
    out = survivorship([pair], threshold=100.0)
    row = out.iloc[0]
    assert row["n_in_year"] == 2
    assert row["n_survived"] == 1
    assert row["survival_rate"] == 0.5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_keepers/test_calibration.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'fantasy_baseball.keepers.calibration'`

- [ ] **Step 3: Write minimal implementation**

```python
"""Assemble ZiPS-vs-actual year pairs and measure the fit sample.

The one non-negotiable methodological constraint (spec 6.1): the base must be
ZiPS_Y, built knowing only through Y-1, so it has NOT already absorbed year Y --
mirroring production, where ZiPS 2027 has never seen 2026. Using ZiPS_{Y+1} as
the base would fit how much surprise ZiPS already absorbed and drive the
coefficient to zero by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.actuals import normalize_hitting, normalize_pitching
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.keepers.vintages import load_vintage

# Year Y of each usable (Y, Y+1) pair. 2025 needs a complete 2026 season; 2021 has
# no ZiPS vintage on disk (data/projections starts at 2022).
PAIR_YEARS = (2022, 2023, 2024)


@dataclass(frozen=True)
class YearPair:
    """One (Y, Y+1) observation set, already aligned on mlbam_id."""

    year: int
    base: pd.DataFrame       # ZiPS_Y rates
    residual: pd.DataFrame   # actual_Y rates - ZiPS_Y rates
    target: pd.DataFrame     # actual_{Y+1} rates
    realized_pt: pd.Series   # actual_Y playing time (drives shrink and gate)
    target_pt: pd.Series     # actual_{Y+1} playing time


def build_pairs(
    player_type: str,
    cache_dir: Path,
    projections_root: Path,
    years: tuple[int, ...] = PAIR_YEARS,
) -> list[YearPair]:
    group = "hitting" if player_type == "hitter" else "pitching"
    normalize = normalize_hitting if player_type == "hitter" else normalize_pitching
    pt_col = "pa" if player_type == "hitter" else "ip"
    pairs: list[YearPair] = []
    for year in years:
        zips_h, zips_p = load_vintage(year, projections_root)
        zips = zips_h if player_type == "hitter" else zips_p
        act_y = normalize(fetch_mlb_season(cache_dir, year, group))
        act_next = normalize(fetch_mlb_season(cache_dir, year + 1, group))
        ids = zips.index.intersection(act_y.index).intersection(act_next.index)
        rate_cols = [c for c in zips.columns if c != pt_col]
        pairs.append(
            YearPair(
                year=year,
                base=zips.loc[ids, rate_cols],
                residual=act_y.loc[ids, rate_cols] - zips.loc[ids, rate_cols],
                target=act_next.loc[ids, rate_cols],
                realized_pt=act_y.loc[ids, pt_col],
                target_pt=act_next.loc[ids, pt_col],
            )
        )
    return pairs


def survivorship(pairs: list[YearPair], threshold: float) -> pd.DataFrame:
    """Per pair: how many cleared `threshold` in year Y, and how many again in Y+1.

    Fitting on survivors alone measures persistence GIVEN continued play, which
    biases the playing-time coefficient upward. Spec 6.3 requires this measured on
    the actual fit sample, not on the wider MLB population.
    """
    rows = []
    for pair in pairs:
        in_year = pair.realized_pt >= threshold
        survived = in_year & (pair.target_pt >= threshold)
        n_in, n_sur = int(in_year.sum()), int(survived.sum())
        rows.append(
            {
                "year": pair.year,
                "n_matched": int(len(pair.base)),
                "n_in_year": n_in,
                "n_survived": n_sur,
                "survival_rate": (n_sur / n_in) if n_in else float("nan"),
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_keepers/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Measure the real samples for BOTH player types**

This is the spec-mandated measurement, not an optional check. Run:

```bash
python -c "
import sys; sys.path.insert(0,'src')
from pathlib import Path
from fantasy_baseball.keepers.calibration import build_pairs, survivorship
cache = Path('data/analysis/keeper_calibration_cache'); cache.mkdir(parents=True, exist_ok=True)
for ptype, thresholds in (('hitter',(100.0,300.0)), ('pitcher',(50.0,100.0))):
    pairs = build_pairs(ptype, cache, Path('data/projections'))
    for t in thresholds:
        print(ptype, 'threshold', t); print(survivorship(pairs, t).to_string(index=False))
"
```

Record the output. Expected for hitters at 100 PA: roughly 350 matched per pair with survival
near 75-80%, matching the spec. **The pitcher figures are unknown and are the point of this
step.** If the pitcher sample at the chosen threshold is too small to fit twelve coefficients,
that is a finding to report, not a reason to proceed quietly.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/keepers/calibration.py tests/test_keepers/test_calibration.py
git commit -m "feat(keepers): year-pair assembly + sample/survivorship measurement (#266)"
```

---

### Task 6: Estimator protocol, endpoints, and leave-one-pair-out evaluation

**Files:**
- Modify: `src/fantasy_baseball/keepers/calibration.py`
- Test: `tests/test_keepers/test_calibration.py`

**Interfaces:**
- Consumes: `YearPair` (Task 5), `shrink` (Task 4).
- Produces: `Estimator` protocol with `name: str` and
  `fit(pairs: list[YearPair], column: str, n0: float) -> Fitted`; `Fitted` protocol with
  `params: dict[str, float]` and `predict(base: pd.Series, residual: pd.Series, weight: pd.Series) -> pd.Series`;
  concrete `ZeroTransfer` and `FullTransfer`; `weighted_mse(pred, actual, weight) -> float`;
  `leave_one_out(estimator, pairs, column, n0) -> pd.DataFrame`.

The two endpoints are the things any chosen estimator must beat (spec 6.2 requirement 3):
`ZeroTransfer` ignores the season entirely -- today's stale baseline -- and `FullTransfer`
moves the full shrunk surprise.

- [ ] **Step 1: Write the failing test**

```python
from fantasy_baseball.keepers.calibration import (
    FullTransfer,
    ZeroTransfer,
    leave_one_out,
    weighted_mse,
)


def _pair(year: int) -> YearPair:
    idx = [1, 2, 3]
    return YearPair(
        year=year,
        base=pd.DataFrame({"hr_pa": [0.030, 0.040, 0.050]}, index=idx),
        residual=pd.DataFrame({"hr_pa": [0.010, 0.000, -0.010]}, index=idx),
        target=pd.DataFrame({"hr_pa": [0.040, 0.040, 0.040]}, index=idx),
        realized_pt=pd.Series([600.0, 600.0, 600.0], index=idx),
        target_pt=pd.Series([600.0, 600.0, 600.0], index=idx),
    )


def test_weighted_mse_weights_by_playing_time() -> None:
    pred = pd.Series([0.0, 0.0])
    actual = pd.Series([1.0, 0.0])
    assert weighted_mse(pred, actual, pd.Series([1.0, 1.0])) == pytest.approx(0.5)
    assert weighted_mse(pred, actual, pd.Series([3.0, 1.0])) == pytest.approx(0.75)


def test_endpoints_predict_as_documented() -> None:
    pair = _pair(2022)
    zero = ZeroTransfer().fit([pair], "hr_pa", n0=200.0)
    full = FullTransfer().fit([pair], "hr_pa", n0=200.0)
    w = pd.Series([1.0, 1.0, 1.0], index=pair.base.index)
    b, r = pair.base["hr_pa"], pair.residual["hr_pa"]
    assert list(zero.predict(b, r, w)) == pytest.approx(list(b))
    assert list(full.predict(b, r, w)) == pytest.approx(list(b + r))


def test_leave_one_out_holds_out_each_pair() -> None:
    pairs = [_pair(2022), _pair(2023), _pair(2024)]
    out = leave_one_out(ZeroTransfer(), pairs, "hr_pa", n0=200.0)
    assert sorted(out["held_out_year"]) == [2022, 2023, 2024]
    assert out["error"].notna().all()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_keepers/test_calibration.py -v`
Expected: FAIL, `ImportError: cannot import name 'ZeroTransfer'`

- [ ] **Step 3: Write minimal implementation**

Append to `calibration.py` (add `from typing import Protocol` and
`from fantasy_baseball.keepers.fold import shrink` at the top):

```python
class Fitted(Protocol):
    params: dict[str, float]

    def predict(
        self, base: pd.Series, residual: pd.Series, weight: pd.Series
    ) -> pd.Series: ...


class Estimator(Protocol):
    name: str

    def fit(self, pairs: list[YearPair], column: str, n0: float) -> Fitted: ...


class _FixedK:
    """Endpoint estimator: predict = base + k * weight * residual, k not fitted."""

    def __init__(self, k: float) -> None:
        self.params = {"k": k}

    def predict(self, base: pd.Series, residual: pd.Series, weight: pd.Series) -> pd.Series:
        result: pd.Series = base + self.params["k"] * weight * residual.fillna(0.0)
        return result


class ZeroTransfer:
    """k = 0: ignore the season entirely. This is today's stale-baseline behaviour."""

    name = "k=0"

    def fit(self, pairs: list[YearPair], column: str, n0: float) -> Fitted:
        return _FixedK(0.0)


class FullTransfer:
    """k = 1: move the full (shrunk) surprise."""

    name = "k=1"

    def fit(self, pairs: list[YearPair], column: str, n0: float) -> Fitted:
        return _FixedK(1.0)


def weighted_mse(pred: pd.Series, actual: pd.Series, weight: pd.Series) -> float:
    """Playing-time-weighted MSE, so a 20-PA player's rate cannot dominate."""
    mask = actual.notna() & pred.notna() & (weight > 0)
    if not mask.any():
        return float("nan")
    err = (pred[mask] - actual[mask]) ** 2
    return float((err * weight[mask]).sum() / weight[mask].sum())


def leave_one_out(
    estimator: Estimator, pairs: list[YearPair], column: str, n0: float
) -> pd.DataFrame:
    """Fit on all pairs but one, evaluate on the held-out pair. Spec 6.3."""
    rows = []
    for held in pairs:
        train = [p for p in pairs if p.year != held.year]
        fitted = estimator.fit(train, column, n0)
        weight = shrink(held.realized_pt, n0)
        pred = fitted.predict(held.base[column], held.residual[column], weight)
        rows.append(
            {
                "estimator": estimator.name,
                "column": column,
                "held_out_year": held.year,
                "n": int(len(held.base)),
                "error": weighted_mse(pred, held.target[column], held.target_pt),
                **{f"param_{k}": v for k, v in fitted.params.items()},
            }
        )
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_keepers/test_calibration.py -v`
Expected: PASS

- [ ] **Step 5: Run the gates and commit**

```bash
ruff check . && ruff format --check . && mypy
git add src/fantasy_baseball/keepers/calibration.py tests/test_keepers/test_calibration.py
git commit -m "feat(keepers): estimator protocol, endpoints, leave-one-pair-out (#266)"
```

---

### Task 7: Choose, implement and document the estimator

**Files:**
- Modify: `src/fantasy_baseball/keepers/calibration.py`
- Test: `tests/test_keepers/test_calibration.py`
- Create: `docs/superpowers/findings/2026-XX-XX-keeper-calibration-finding.md`

**Interfaces:**
- Consumes: everything from Tasks 5-6.
- Produces: one or more concrete `Estimator` implementations alongside `ZeroTransfer` /
  `FullTransfer`.

**This is the deliverable, and the spec deliberately does not tell you the answer.** Section 6.2
lists twelve requirements the chosen estimator must satisfy. Three earlier attempts to specify a
form in prose were each found broken; the form is to be chosen against real data and written up.

Before writing any estimator, re-read spec section 6.2. The requirements most likely to sink a
candidate:

- **Requirement 1:** calibration and production must apply the *same* functional form. A term
  present only in calibration must be justified and its production value stated. The known gap:
  `ZiPS_Y` targets year Y while the target is Y+1, whereas `ZiPS_2027` is already aged to 2027.
- **Requirement 12:** the playing-time residual has a large systematic *mean* that is not
  surprise -- ZiPS hedges playing time pool-wide (2025 regulars ran +58 mean PA versus
  projection). A single multiplicative coefficient cannot separate a level offset from signal.
  **Both estimators that failed in earlier rounds died here** -- a free scale term made `k`
  degenerate into the OLS slope on `actual_Y`, and a normalization term proved unidentified and
  uncomputable at serve time.
- **Requirement 7:** a coefficient that would amplify residuals in production must not ship
  silently on held-out error alone.

- [ ] **Step 1: Pin the error metric and the gate thresholds, and write them down**

Spec 6.2 requirements 9 and 11: the error metric and the gate thresholds must be recorded
**before** the first fit, so the increment that must clear the gate does not also pick the gate's
yardstick after seeing results. Create the finding document with an "Advance decisions" section
stating: the metric (`weighted_mse` on the rate scale, weighted by target-year playing time),
how it aggregates across coefficients (per coefficient, not pooled -- acceptance is per-category
per spec 6.6), and the chosen hitter (PA) and pitcher (IP) gate thresholds with their
justification from Task 5's measured samples. Commit this before proceeding.

- [ ] **Step 2: Write a failing test for the chosen estimator**

Write a test that plants a known coefficient in synthetic data and asserts the estimator recovers
it within tolerance. This is the test the spec's section 9 requires ("recovery of a known planted
coefficient on synthetic data"), and it is what proves the fit code is correct independently of
whether the real-data answer is interesting:

```python
def test_estimator_recovers_a_planted_coefficient() -> None:
    rng = np.random.default_rng(0)
    n, k_true = 500, 0.6
    base = pd.Series(rng.uniform(0.02, 0.06, n))
    resid = pd.Series(rng.normal(0.0, 0.01, n))
    weight = pd.Series(np.ones(n))
    target = base + k_true * weight * resid + pd.Series(rng.normal(0.0, 1e-4, n))
    pair = YearPair(
        year=2022,
        base=base.to_frame("hr_pa"),
        residual=resid.to_frame("hr_pa"),
        target=target.to_frame("hr_pa"),
        realized_pt=pd.Series(np.full(n, 600.0)),
        target_pt=pd.Series(np.full(n, 600.0)),
    )
    fitted = <YourEstimator>().fit([pair], "hr_pa", n0=200.0)
    assert fitted.params["k"] == pytest.approx(k_true, abs=0.05)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `pytest tests/test_keepers/test_calibration.py -k planted -v`
Expected: FAIL

- [ ] **Step 4: Implement the estimator and make it pass**

Implement against the `Estimator` / `Fitted` protocols from Task 6 so it drops into
`leave_one_out` with no changes. Keep the class docstring explicit about which of the twelve
requirements the design satisfies and how -- especially requirements 1 and 12.

- [ ] **Step 5: Run the gates and commit**

```bash
pytest tests/test_keepers/ -v && ruff check . && ruff format --check . && mypy
git add src/fantasy_baseball/keepers/calibration.py tests/test_keepers/test_calibration.py
git commit -m "feat(keepers): calibration estimator (#266)"
```

---

### Task 8: The study script and the written finding

**Files:**
- Create: `scripts/keeper_calibration.py`
- Test: `tests/test_scripts/test_keeper_calibration_script.py`
- Modify: `docs/superpowers/findings/2026-XX-XX-keeper-calibration-finding.md`
- Modify: `pyproject.toml` (vulture `ignore_names`)

**Interfaces:**
- Consumes: everything above.
- Produces: a CLI writing `data/analysis/keeper_calibration_<player_type>.csv`.

- [ ] **Step 1: Write the failing test**

```python
from pathlib import Path

from scripts.keeper_calibration import build_report


def test_build_report_covers_every_coefficient_and_estimator(tmp_path: Path) -> None:
    # Uses the synthetic pair helper so the test never touches the network.
    report = build_report(_synthetic_pairs(), columns=("hr_pa", "sb_pa"), n0=200.0)
    assert set(report["column"]) == {"hr_pa", "sb_pa"}
    assert {"k=0", "k=1"} <= set(report["estimator"])
    # Acceptance is per coefficient, not pooled -- spec 6.6.
    assert "verdict" in report.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_scripts/test_keeper_calibration_script.py -v`
Expected: FAIL, `ModuleNotFoundError: No module named 'scripts.keeper_calibration'`

- [ ] **Step 3: Write the script**

Follow the repo convention of injecting `src/` into `sys.path` (see `scripts/keeper_value.py`).
`build_report` must:
- run every estimator (both endpoints plus the chosen one) through `leave_one_out` for every
  coefficient;
- aggregate per `(estimator, column)` to mean held-out error;
- emit a `verdict` column per **coefficient**: `pass` when the chosen estimator beats both
  endpoints on a majority of held-out pairs, otherwise `fallback:k=0` or `fallback:k=1` naming
  whichever endpoint won (spec 6.6);
- flag any fitted coefficient outside [0, 1] (spec 6.2 requirement 7);
- print an ASCII-only summary table.

The CLI writes both player types and must **not** route the in-progress 2026 season through
`fetch_or_cache`. Since `PAIR_YEARS` tops out at 2024 and targets 2025, no 2026 pull happens --
assert this explicitly in the script with a comment so a future edit does not silently add one.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_scripts/test_keeper_calibration_script.py -v`
Expected: PASS

- [ ] **Step 5: Run the study for real and complete the finding**

```bash
python scripts/keeper_calibration.py
```

Fill in the finding document with everything spec 6.2 requires: fitted coefficients with
confidence intervals and out-of-range flags; leave-one-pair-out error against both endpoints;
stability across the three pairs, with SB read against the 2023 rules break rather than averaged
away; survivorship on the fit sample for both player types; the conditioning statement (the
coefficients are conditional on the chosen `n0` and gate thresholds); the analytic train/serve
attenuation implied by the current ~35% rest-of-season share; and the gate discontinuity
magnitude or the ramp chosen.

- [ ] **Step 6: Drop the stale vulture suppression**

`fetch_mlb_season` now has a real caller, so remove it from `ignore_names` in `pyproject.toml`
(the entry exists only because the fetchers were callerless after PR #267). Leave the four Savant
entries -- they remain callerless until a later increment.

- [ ] **Step 7: Full gates and commit**

```bash
pytest -v && ruff check . && ruff format --check . && mypy && vulture
git add scripts/keeper_calibration.py tests/test_scripts/ docs/superpowers/findings/ pyproject.toml
git commit -m "feat(keepers): calibration study script + written finding (#266)"
```

---

## Self-Review

**Spec coverage.** Section 5.2 decomposition -> Tasks 2-4. Section 5.3 shrink (bounded, rates
only, realized PT) -> Task 4. Section 5.4 gate, including absence-is-not-zero and the
discontinuity -> Tasks 4, 5, 7. Section 5.5 NaN guards on realized, ZiPS-base and reconstructed
sides -> Tasks 2, 3, 4. Section 6.1 base-must-not-know-year-Y -> Task 5 (encoded in the module
docstring and in `PAIR_YEARS`). Section 6.2's twelve requirements -> Task 7, with 9 and 11 pinned
in Step 1 before any fit. Section 6.3 three pairs, SB break, survivorship, pitcher-side gap ->
Tasks 5 and 8. Section 6.4 train/serve -> Task 8 finding. Section 6.5 ingest hazards -> Tasks 1
and 8. Section 6.6 per-coefficient acceptance -> Task 8. Section 9 test list -> distributed
across Tasks 1-4 and 7-8.

**Deliberately out of scope:** section 5.1's `role_ip` routing fix, sections 2 and 7's par and
cross-team work, and section 5.6's position-eligibility problems are all increment 2 -- they need
VAR, which increment 1 does not build.

**Known gap the plan carries forward:** Task 7 Step 4 cannot show final code, because choosing the
estimator *is* the deliverable. This is the one place the plan intentionally departs from the
no-placeholders rule, and the mitigation is that its protocol, its test, its acceptance rule and
its twelve requirements are all fully specified -- only the body is open.

**Type consistency check.** `safe_ratio` is renamed from `_safe_ratio` in Task 3 and both call
sites in `actuals.py` are updated in the same task. `YearPair` gains `target_pt` in Task 5 and it
is used by `weighted_mse` in Task 6 and `survivorship` in Task 5 -- consistent. `Estimator.fit`
takes `(pairs, column, n0)` in Tasks 6 and 7 identically. `shrink(n, n0)` is keyword-compatible
across Tasks 4, 6 and 7.
