# Keeper Breakout/Mirage Diagnostic Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md`

**Goal:** Ship a backtested breakout/mirage keeper resource: source season Statcast xStats + FanGraphs rates + age, regress the luck out of the current anchor into a skill-adjusted true-talent line, rank players by forward keeper value, and prove (or disprove) that ranking beats surface-believed and pure-ZiPS baselines year-over-year.

**Architecture:** Four isolated units. A fetch+cache+join data layer (`data/skill_luck.py`) produces a per-player-season frame of rates/xStats/age. A pure classifier (`analysis/breakout.py`) turns (surface line, projection line, underlying signal) into a skill-adjusted counting line + label + reason. A report script reuses `scripts/keeper_value.py`'s board/scale/anchor machinery and calls `keeper_value()` twice per player (surface vs adjusted). A backtest script scores three estimators year-over-year on a fixed yardstick. Only the classifier and scoring math are unit-tested against fixtures; network is always mocked.

**Tech Stack:** Python 3.12, pandas, pybaseball (installed), pytest. Reuses `fantasy_baseball.analysis.keeper_value`, `.draft_value`, `.draft.board`, `.utils.name_utils`, and `scripts/keeper_value.py` helpers.

## Global Constraints

- **ASCII-only** in all source, log, and report strings (Windows cp1252 stdout). No true minus, sigma, em/en dash, smart quotes, arrows. Use `-`, `sigma`, `--`, `->`.
- **Never `x or default` for numeric fields** (`0`/`0.0` are falsy). Use `x if x is not None else default` or `dict.get(k, default)`.
- **Player IDs are `name::player_type`**; never key on bare names. Cross-source joins use `normalize_name` (accent-stripped, lowercased) with VAR tie-break.
- **No network in tests.** Mock `pybaseball` calls. Fetch code must be import-safe (no fetch at import time).
- **Cache fail-safe:** never overwrite a good cache file with an empty or errored fetch result.
- **Pure classifier:** `analysis/breakout.py` does no I/O (no file, network, or KV reads).
- **This plan does not modify `src/fantasy_baseball/analysis/keeper_value.py`.** It only calls its public functions.
- **End-of-effort verification** (per CLAUDE.md): `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` for any touched file listed under `[tool.mypy].files` (`src/fantasy_baseball/analysis/` is covered -> `analysis/breakout.py` needs mypy-clean).

---

## Shared data shapes (defined once, referenced by all tasks)

These live in `src/fantasy_baseball/analysis/breakout.py` and are imported by the data layer and scripts.

```python
# The joined per-player-season underlying signal (one row). All optional except pa/player_type:
# a player may be missing xStats (insufficient batted balls) or be a pitcher.
@dataclass(frozen=True)
class SkillLuckRow:
    mlbam: int
    player_type: str            # "hitter" | "pitcher"
    pa: float                   # hitter plate appearances (0.0 for pitchers)
    ip: float                   # pitcher innings (0.0 for hitters)
    age: float | None
    # hitter confirmations
    barrel_pct: float | None    # brl_percent (share, e.g. 0.12)
    xslg: float | None
    slg: float | None
    xba: float | None
    ba: float | None
    babip: float | None
    xwoba: float | None
    woba: float | None
    k_pct: float | None
    bb_pct: float | None
    # pitcher confirmations (K-BB, xwOBA-against reuse xwoba/woba/k_pct/bb_pct above)

# The classifier output for one player.
@dataclass(frozen=True)
class BreakoutResult:
    adjusted_line: dict[str, float]   # counting line, same keys keeper_value consumes
    label: str                        # one of LABELS
    reason: str                       # short ASCII driver string
    w_by_stat: dict[str, float]       # believed-fraction per adjusted rate, for the report/backtest
    confidence: str                   # "full" | "low"
    surface_deviation: float          # raw signed aggregate surface-vs-projection deviation
    believed_deviation: float         # w-weighted signed deviation (drives the label)

LABELS = ("real breakout", "lucky mirage", "real decline", "slump", "stable")
```

---

## Phase 0: Shared shapes

### Task 0: Create breakout.py with the shared dataclasses

Ordering matters: the Phase-1 data layer (Task 3) imports `SkillLuckRow` from
`breakout.py`, so the shapes module must exist first. This task creates
`breakout.py` containing ONLY the dataclasses and `LABELS` from "Shared data
shapes" above; later tasks (4-7) add functions to the same file.

**Files:**
- Create: `src/fantasy_baseball/analysis/breakout.py`
- Test: `tests/test_analysis/test_breakout.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis/test_breakout.py
from fantasy_baseball.analysis import breakout

def test_shapes_exist():
    row = breakout.SkillLuckRow(mlbam=1, player_type="hitter", pa=600, ip=0.0, age=27.0,
        barrel_pct=None, xslg=None, slg=None, xba=None, ba=None, babip=None,
        xwoba=None, woba=None, k_pct=None, bb_pct=None)
    assert row.player_type == "hitter"
    assert "real breakout" in breakout.LABELS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_breakout.py::test_shapes_exist -v`
Expected: FAIL (module `breakout` not found).

- [ ] **Step 3: Write minimal implementation**

Create `src/fantasy_baseball/analysis/breakout.py` with a module docstring, `from __future__ import annotations`, `from dataclasses import dataclass, field`, and the `SkillLuckRow`, `BreakoutResult` dataclasses and `LABELS` tuple exactly as in "Shared data shapes" above.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_breakout.py::test_shapes_exist -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/breakout.py tests/test_analysis/test_breakout.py
git commit -m "feat(breakout): shared SkillLuckRow/BreakoutResult shapes"
```

---

## Phase 1: Data layer + identity

### Task 1: Chadwick MLBAM <-> FanGraphs id map (fetch + cache)

**Files:**
- Create: `src/fantasy_baseball/data/skill_luck.py`
- Test: `tests/test_data/test_skill_luck.py`

**Interfaces:**
- Consumes: nothing.
- Produces: `load_id_map(cache_dir: Path, *, fetcher: Callable[[], pd.DataFrame] | None = None) -> pd.DataFrame` returning a frame with columns `["key_mlbam", "key_fangraphs"]` (ints, NaN rows dropped). `fetcher` defaults to `pybaseball.chadwick_register` and exists so tests inject a fake.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_data/test_skill_luck.py
from pathlib import Path
import pandas as pd
from fantasy_baseball.data import skill_luck

def _fake_register():
    return pd.DataFrame({
        "key_mlbam": [665742, 700, float("nan")],   # Soto, junk, unmatched
        "key_fangraphs": [20123, float("nan"), 55],
        "name_first": ["Juan", "No", "No"],
        "name_last": ["Soto", "Fg", "Mlbam"],
    })

def test_load_id_map_drops_unmatched_and_caches(tmp_path: Path):
    m = skill_luck.load_id_map(tmp_path, fetcher=_fake_register)
    # only the fully-identified row survives
    assert list(m["key_mlbam"]) == [665742]
    assert list(m["key_fangraphs"]) == [20123]
    # second call with a fetcher that would raise must hit the cache, not the network
    def _boom():
        raise AssertionError("must not refetch")
    m2 = skill_luck.load_id_map(tmp_path, fetcher=_boom)
    assert list(m2["key_mlbam"]) == [665742]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_skill_luck.py::test_load_id_map_drops_unmatched_and_caches -v`
Expected: FAIL (`module 'skill_luck' has no attribute 'load_id_map'`).

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/data/skill_luck.py
"""Season-level skill/luck data layer: FanGraphs rates + age, Statcast xStats,
and the MLBAM<->FanGraphs id map, cached to data/skill_luck/. Fetch-on-miss,
never overwrite a good cache with an empty/failed pull. See
docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd

ID_MAP_FILE = "id_map_mlbam_fg.csv"


def _read_cached(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df = pd.read_csv(path)
        if not df.empty:
            return df
    return None


def load_id_map(
    cache_dir: Path, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    path = cache_dir / ID_MAP_FILE
    cached = _read_cached(path)
    if cached is not None:
        return cached
    if fetcher is None:
        from pybaseball import chadwick_register  # local import: keeps module import-safe

        fetcher = chadwick_register
    reg = fetcher()
    out = reg[["key_mlbam", "key_fangraphs"]].dropna().astype({"key_mlbam": int, "key_fangraphs": int})
    if out.empty:
        raise RuntimeError("chadwick_register returned no usable id rows; refusing to cache empty")
    cache_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_skill_luck.py::test_load_id_map_drops_unmatched_and_caches -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/skill_luck.py tests/test_data/test_skill_luck.py
git commit -m "feat(skill-luck): cached MLBAM<->FanGraphs id map via chadwick register"
```

---

### Task 2: Season stat fetch + cache with fail-safe overwrite guard

**Files:**
- Modify: `src/fantasy_baseball/data/skill_luck.py`
- Test: `tests/test_data/test_skill_luck.py`

**Interfaces:**
- Consumes: `_read_cached` (Task 1).
- Produces:
  - `fetch_or_cache(path: Path, fetcher: Callable[[], pd.DataFrame]) -> pd.DataFrame` -- generic cache-first loader that refuses to write an empty/failed result.
  - `load_fg_hitters(cache_dir, year, *, fetcher=None) -> pd.DataFrame` and `load_statcast_hitters(cache_dir, year, *, xstats_fetcher=None, barrels_fetcher=None) -> pd.DataFrame`, each returning a per-player frame keyed by an id column. Column names are normalized to the `SkillLuckRow` field names via an explicit rename map with a fail-loud assertion when an expected source column is absent (guards against pybaseball schema drift).

- [ ] **Step 1: Write the failing test**

```python
def test_fetch_or_cache_refuses_empty_and_reuses(tmp_path: Path):
    from fantasy_baseball.data import skill_luck
    calls = {"n": 0}
    good = pd.DataFrame({"a": [1, 2]})
    def _fetch_good():
        calls["n"] += 1
        return good
    p = tmp_path / "x.csv"
    out = skill_luck.fetch_or_cache(p, _fetch_good)
    assert list(out["a"]) == [1, 2] and calls["n"] == 1
    # cache hit: fetcher not called again
    skill_luck.fetch_or_cache(p, _fetch_good)
    assert calls["n"] == 1
    # empty fetch to a fresh path raises and writes nothing
    import pytest
    q = tmp_path / "y.csv"
    with pytest.raises(RuntimeError):
        skill_luck.fetch_or_cache(q, lambda: pd.DataFrame())
    assert not q.exists()

def test_load_fg_hitters_renames_and_fails_loud_on_schema_drift(tmp_path: Path):
    from fantasy_baseball.data import skill_luck
    src = pd.DataFrame({"IDfg": [20123], "Age": [26.0], "K%": [0.20], "BB%": [0.15],
                        "BABIP": [0.34], "HR/FB": [0.18], "Contact%": [0.78], "PA": [600]})
    out = skill_luck.load_fg_hitters(tmp_path, 2024, fetcher=lambda: src)
    row = out.iloc[0]
    assert row["key_fangraphs"] == 20123 and row["k_pct"] == 0.20 and row["age"] == 26.0
    import pytest
    with pytest.raises(KeyError):
        skill_luck.load_fg_hitters(tmp_path, 2025, fetcher=lambda: pd.DataFrame({"IDfg": [1]}))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_skill_luck.py -k "fetch_or_cache or fails_loud" -v`
Expected: FAIL (`fetch_or_cache` / `load_fg_hitters` undefined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/fantasy_baseball/data/skill_luck.py

def fetch_or_cache(path: Path, fetcher: Callable[[], pd.DataFrame]) -> pd.DataFrame:
    cached = _read_cached(path)
    if cached is not None:
        return cached
    df = fetcher()
    if df is None or df.empty:
        raise RuntimeError(f"fetch for {path.name} returned empty; refusing to overwrite/write cache")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df


# Explicit source-column -> SkillLuckRow-field maps. If a source key is missing we
# raise KeyError (fail loud) rather than silently emit a NaN column.
_FG_HIT_RENAME = {
    "IDfg": "key_fangraphs", "Age": "age", "K%": "k_pct", "BB%": "bb_pct",
    "BABIP": "babip", "HR/FB": "hr_fb", "Contact%": "contact_pct", "PA": "pa",
}


def _rename_strict(df: pd.DataFrame, rename: dict[str, str]) -> pd.DataFrame:
    missing = [c for c in rename if c not in df.columns]
    if missing:
        raise KeyError(f"source frame missing expected columns {missing}; got {list(df.columns)}")
    return df[list(rename)].rename(columns=rename)


def load_fg_hitters(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    def _default() -> pd.DataFrame:
        from pybaseball import batting_stats

        return batting_stats(year, qual=1)

    raw = fetch_or_cache(cache_dir / f"fg_h_{year}.csv", fetcher or _default)
    return _rename_strict(raw, _FG_HIT_RENAME)
```

Statcast hitter loader (xStats + barrels), same pattern:

```python
_STATCAST_XHIT_RENAME = {
    "player_id": "mlbam", "woba": "woba", "est_woba": "xwoba",
    "ba": "ba", "est_ba": "xba", "slg": "slg", "est_slg": "xslg",
}
_STATCAST_BARREL_RENAME = {"player_id": "mlbam", "brl_percent": "barrel_pct"}


def load_statcast_hitters(
    cache_dir: Path, year: int, *,
    xstats_fetcher: Callable[[], pd.DataFrame] | None = None,
    barrels_fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    def _x() -> pd.DataFrame:
        from pybaseball import statcast_batter_expected_stats

        return statcast_batter_expected_stats(year, minPA=1)

    def _b() -> pd.DataFrame:
        from pybaseball import statcast_batter_exitvelo_barrels

        return statcast_batter_exitvelo_barrels(year, minBBE=1)

    x = _rename_strict(fetch_or_cache(cache_dir / f"sc_x_h_{year}.csv", xstats_fetcher or _x),
                       _STATCAST_XHIT_RENAME)
    b = _rename_strict(fetch_or_cache(cache_dir / f"sc_brl_h_{year}.csv", barrels_fetcher or _b),
                       _STATCAST_BARREL_RENAME)
    # barrel_pct arrives as a percent (0-100) on Savant; normalize to a share.
    b = b.assign(barrel_pct=b["barrel_pct"].astype(float) / 100.0)
    return x.merge(b, on="mlbam", how="left")
```

> Note for the implementer: the exact Savant column spellings (`est_woba`, `brl_percent`, `player_id`) are current as of pybaseball 2.2.x but Savant renames occasionally. The `_rename_strict` assertion turns any drift into a loud `KeyError` at first fetch -- if that fires, print `raw.columns.tolist()` and update the rename map; do NOT silence it.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_skill_luck.py -k "fetch_or_cache or fails_loud" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/skill_luck.py tests/test_data/test_skill_luck.py
git commit -m "feat(skill-luck): season FanGraphs+Statcast fetch/cache with fail-loud rename"
```

---

### Task 3: Join to per-player-season SkillLuckRow frame + coverage report

**Files:**
- Modify: `src/fantasy_baseball/data/skill_luck.py`
- Test: `tests/test_data/test_skill_luck.py`

**Interfaces:**
- Consumes: `load_id_map`, `load_fg_hitters`, `load_statcast_hitters`.
- Produces: `build_hitter_skill_luck(cache_dir, year, *, fetchers=None) -> tuple[dict[int, SkillLuckRow], CoverageReport]` where the dict is keyed by `key_fangraphs` (the id the board carries as `fg_id`) and `CoverageReport` is a small dataclass `(matched: int, fg_only: int, unmatched_fg: list[int])`. Statcast is joined via the id map (fg->mlbam); players with FG rates but no Statcast get a row with xStats fields `None` (confidence downgraded later).

- [ ] **Step 1: Write the failing test**

```python
def test_build_hitter_skill_luck_joins_and_reports_coverage(tmp_path: Path):
    from fantasy_baseball.data import skill_luck
    id_map = pd.DataFrame({"key_mlbam": [665742, 700], "key_fangraphs": [20123, 800]})
    fg = pd.DataFrame({"IDfg": [20123, 800], "Age": [26.0, 30.0], "K%": [0.2, 0.3],
                       "BB%": [0.15, 0.05], "BABIP": [0.34, 0.28], "HR/FB": [0.2, 0.1],
                       "Contact%": [0.78, 0.7], "PA": [600, 550]})
    sc = pd.DataFrame({"player_id": [665742], "woba": [0.400], "est_woba": [0.360],
                       "ba": [0.300], "est_ba": [0.270], "slg": [0.600], "est_slg": [0.520]})
    brl = pd.DataFrame({"player_id": [665742], "brl_percent": [14.0]})
    rows, cov = skill_luck.build_hitter_skill_luck(
        tmp_path, 2024,
        fetchers={"id_map": lambda: id_map, "fg": lambda: fg,
                  "sc_x": lambda: sc, "sc_brl": lambda: brl},
    )
    soto = rows[20123]
    assert soto.mlbam == 665742 and soto.woba == 0.400 and soto.xwoba == 0.360
    assert soto.barrel_pct == 0.14 and soto.k_pct == 0.2
    # fg 800 has no statcast -> present but xStats None
    assert rows[800].xwoba is None and rows[800].woba is None
    assert cov.matched == 1 and cov.fg_only == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_data/test_skill_luck.py::test_build_hitter_skill_luck_joins_and_reports_coverage -v`
Expected: FAIL (`build_hitter_skill_luck` undefined).

- [ ] **Step 3: Write minimal implementation**

```python
# append to skill_luck.py -- HOIST these two imports into the module's top import
# block (ruff E402): `from dataclasses import dataclass` and the SkillLuckRow import.
from dataclasses import dataclass

from fantasy_baseball.analysis.breakout import SkillLuckRow  # shared shape from Task 0


@dataclass(frozen=True)
class CoverageReport:
    matched: int        # FG rows with Statcast xStats joined
    fg_only: int        # FG rows without Statcast
    unmatched_fg: list[int]  # FG ids absent from the id map


def _f(v) -> float | None:
    return None if v is None or pd.isna(v) else float(v)


def build_hitter_skill_luck(cache_dir, year, *, fetchers=None):
    fetchers = fetchers or {}
    id_map = load_id_map(cache_dir, fetcher=fetchers.get("id_map"))
    fg = load_fg_hitters(cache_dir, year, fetcher=fetchers.get("fg"))
    sc = load_statcast_hitters(cache_dir, year,
                               xstats_fetcher=fetchers.get("sc_x"),
                               barrels_fetcher=fetchers.get("sc_brl"))
    fg2m = dict(zip(id_map["key_fangraphs"].astype(int), id_map["key_mlbam"].astype(int)))
    sc_by_mlbam = {int(r.mlbam): r for r in sc.itertuples(index=False)}
    rows: dict[int, SkillLuckRow] = {}
    matched = fg_only = 0
    unmatched: list[int] = []
    for r in fg.itertuples(index=False):
        fgid = int(r.key_fangraphs)
        mlbam = fg2m.get(fgid)
        if mlbam is None:
            unmatched.append(fgid)
            continue
        s = sc_by_mlbam.get(mlbam)
        if s is not None:
            matched += 1
        else:
            fg_only += 1
        rows[fgid] = SkillLuckRow(
            mlbam=mlbam, player_type="hitter", pa=float(r.pa), ip=0.0, age=_f(r.age),
            barrel_pct=_f(getattr(s, "barrel_pct", None)) if s else None,
            xslg=_f(getattr(s, "xslg", None)) if s else None,
            slg=_f(getattr(s, "slg", None)) if s else None,
            xba=_f(getattr(s, "xba", None)) if s else None,
            ba=_f(getattr(s, "ba", None)) if s else None,
            babip=_f(r.babip),
            xwoba=_f(getattr(s, "xwoba", None)) if s else None,
            woba=_f(getattr(s, "woba", None)) if s else None,
            k_pct=_f(r.k_pct), bb_pct=_f(r.bb_pct),
        )
    return rows, CoverageReport(matched, fg_only, unmatched)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_data/test_skill_luck.py::test_build_hitter_skill_luck_joins_and_reports_coverage -v`
Expected: PASS. Then run the whole file: `pytest tests/test_data/test_skill_luck.py -v`.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/skill_luck.py tests/test_data/test_skill_luck.py
git commit -m "feat(skill-luck): join FG+Statcast into per-season SkillLuckRow + coverage report"
```

> Pitcher parity (`build_pitcher_skill_luck`, `load_fg_pitchers`, `load_statcast_pitchers`) follows the identical pattern with pitcher rename maps (`IP`, `K%`, `BB%`, `est_woba` from `statcast_pitcher_expected_stats`). Implement it as a sibling function in this same task's file once the hitter path is green; test it with a pitcher fixture mirroring Step 1. Kept in one task because it is the same seam, not a separable deliverable.

---

## Phase 2: Classifier (pure)

### Task 4: Rate extraction (line_rates)

**Files:**
- Modify: `src/fantasy_baseball/analysis/breakout.py` (created in Task 0)
- Test: `tests/test_analysis/test_breakout.py`

**Interfaces:**
- Consumes: the `SkillLuckRow` / `LABELS` shapes from Task 0.
- Produces: `line_rates(line: Mapping[str, Any], player_type: str) -> dict[str, float]` converting a counting line to the per-PA/per-IP rates the classifier compares. Hitter rates: `hr, r, rbi, sb` per PA and `avg` (= H/AB). Pitcher rates: `k` per IP, `w, sv` per IP, `era`, `whip`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_analysis/test_breakout.py
from fantasy_baseball.analysis import breakout

def test_line_rates_hitter():
    line = {"pa": 600, "ab": 540, "h": 162, "hr": 30, "r": 90, "rbi": 100, "sb": 20, "avg": 0.300}
    rates = breakout.line_rates(line, "hitter")
    assert abs(rates["hr"] - 30 / 600) < 1e-9
    assert abs(rates["avg"] - 0.300) < 1e-9
    assert abs(rates["sb"] - 20 / 600) < 1e-9

def test_line_rates_zero_pa_is_safe():
    rates = breakout.line_rates({"pa": 0, "hr": 0, "avg": 0.0}, "hitter")
    assert rates["hr"] == 0.0  # no ZeroDivision, no NaN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_breakout.py -v`
Expected: FAIL (module/`line_rates` undefined).

- [ ] **Step 3: Write minimal implementation**

Add to `breakout.py` (created in Task 0):

```python
from fantasy_baseball.utils.constants import safe_float

HITTER_COUNTING = ("hr", "r", "rbi", "sb")
def line_rates(line, player_type):
    if player_type == "hitter":
        pa = safe_float(line.get("pa", 0))
        rates = {k: (safe_float(line.get(k, 0)) / pa if pa > 0 else 0.0) for k in HITTER_COUNTING}
        rates["avg"] = safe_float(line.get("avg", 0))
        return rates
    ip = safe_float(line.get("ip", 0))
    rates = {"k": (safe_float(line.get("k", 0)) / ip if ip > 0 else 0.0),
             "w": (safe_float(line.get("w", 0)) / ip if ip > 0 else 0.0),
             "sv": (safe_float(line.get("sv", 0)) / ip if ip > 0 else 0.0),
             "era": safe_float(line.get("era", 0)), "whip": safe_float(line.get("whip", 0))}
    return rates
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_breakout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/breakout.py tests/test_analysis/test_breakout.py
git commit -m "feat(breakout): shared shapes + counting-line->rate extraction"
```

---

### Task 5: The w-mapping (reliability x confirmation), with seed defaults

**Files:**
- Modify: `src/fantasy_baseball/analysis/breakout.py`
- Test: `tests/test_analysis/test_breakout.py`

**Interfaces:**
- Consumes: `SkillLuckRow`.
- Produces:
  - `@dataclass(frozen=True) class WMapParams` with named seed defaults: `pa_stabilize: float = 300.0`, `ip_stabilize: float = 80.0`, `confirm_weight: float = 0.5`, plus per-stat stabilization overrides `stat_stabilize: dict[str, float]` (K%/barrel fast ~ 60-100 PA-equiv, BABIP slow ~ 800). `DEFAULT_WMAP = WMapParams()`.
  - `w_for_stat(stat: str, row: SkillLuckRow, player_type: str, params: WMapParams) -> float` in [0, 1].
- The believed fraction is `reliability * confirmation`, where `reliability = sample / (sample + stabilize[stat])` and `confirmation` in [0,1] is how well the matching underlying signal supports the surface (barrel/xSLG for hr; xBA vs BABIP for avg; K-BB and xwOBA for pitcher ratios; SB from PA-reliability only; SV conservative).

- [ ] **Step 1: Write the failing test**

```python
def _row(**kw):
    base = dict(mlbam=1, player_type="hitter", pa=600, ip=0.0, age=27.0, barrel_pct=None,
                xslg=None, slg=None, xba=None, ba=None, babip=None, xwoba=None, woba=None,
                k_pct=None, bb_pct=None)
    base.update(kw); return breakout.SkillLuckRow(**base)

def test_barrel_backed_hr_has_higher_w_than_unbacked():
    backed = _row(barrel_pct=0.16, xslg=0.560, slg=0.560)      # xSLG confirms the power
    lucky = _row(barrel_pct=0.06, xslg=0.410, slg=0.560)       # slg >> xslg -> luck
    p = breakout.DEFAULT_WMAP
    assert breakout.w_for_stat("hr", backed, "hitter", p) > breakout.w_for_stat("hr", lucky, "hitter", p)

def test_low_sample_shrinks_w():
    big = _row(pa=600, barrel_pct=0.16, xslg=0.560, slg=0.560)
    tiny = _row(pa=60, barrel_pct=0.16, xslg=0.560, slg=0.560)
    p = breakout.DEFAULT_WMAP
    assert breakout.w_for_stat("hr", tiny, "hitter", p) < breakout.w_for_stat("hr", big, "hitter", p)

def test_avg_mirage_low_w_when_xba_flat_babip_high():
    mirage = _row(pa=600, ba=0.320, xba=0.255, babip=0.380)
    real = _row(pa=600, ba=0.320, xba=0.315, babip=0.300)
    p = breakout.DEFAULT_WMAP
    assert breakout.w_for_stat("avg", mirage, "hitter", p) < breakout.w_for_stat("avg", real, "hitter", p)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_breakout.py -k "w_" -v`
Expected: FAIL (`w_for_stat`/`DEFAULT_WMAP` undefined).

- [ ] **Step 3: Write minimal implementation**

```python
from dataclasses import dataclass, field

@dataclass(frozen=True)
class WMapParams:
    pa_stabilize: float = 300.0
    ip_stabilize: float = 80.0
    confirm_weight: float = 0.5
    stat_stabilize: dict = field(default_factory=lambda: {
        "hr": 120.0, "r": 300.0, "rbi": 300.0, "sb": 200.0, "avg": 800.0,
        "k": 60.0, "w": 120.0, "sv": 120.0, "era": 120.0, "whip": 120.0,
    })

DEFAULT_WMAP = WMapParams()

def _reliability(sample: float, stabilize: float) -> float:
    return sample / (sample + stabilize) if sample > 0 else 0.0

def _confirm_gap(actual: float | None, expected: float | None, scale: float) -> float:
    # 1.0 when expected matches actual; decays as |actual-expected| grows relative to `scale`.
    if actual is None or expected is None:
        return 0.5  # no signal -> neutral
    return max(0.0, 1.0 - abs(actual - expected) / scale)

def w_for_stat(stat, row, player_type, params):
    sample = row.pa if player_type == "hitter" else row.ip
    reliability = _reliability(sample, params.stat_stabilize.get(stat, params.pa_stabilize))
    if stat == "hr":
        confirm = _confirm_gap(row.slg, row.xslg, 0.150)       # slg vs xslg
    elif stat == "avg":
        confirm = _confirm_gap(row.ba, row.xba, 0.060)         # ba vs xba (BABIP luck shows here)
    elif stat in ("r", "rbi"):
        confirm = _confirm_gap(row.woba, row.xwoba, 0.040)     # context stats track overall quality
    elif stat == "sb":
        confirm = 1.0                                           # role/speed sticky; reliability governs
    else:  # pitcher ratios + sv/w
        confirm = _confirm_gap(row.woba, row.xwoba, 0.040) if row.xwoba is not None else 0.5
    # blend: never let confirmation alone drive w to 0 when sample is huge, nor vice versa
    cw = params.confirm_weight
    return reliability * ((1.0 - cw) + cw * confirm)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_breakout.py -k "w_" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/breakout.py tests/test_analysis/test_breakout.py
git commit -m "feat(breakout): reliability x confirmation w-mapping with seed defaults"
```

---

### Task 6: adjust_line -> skill-adjusted counting line + label + reason

**Files:**
- Modify: `src/fantasy_baseball/analysis/breakout.py`
- Test: `tests/test_analysis/test_breakout.py`

**Interfaces:**
- Consumes: `line_rates`, `w_for_stat`, `SkillLuckRow`, `BreakoutResult`.
- Produces: `adjust_line(surface_line, projection_line, row, player_type, *, params=DEFAULT_WMAP, deviation_threshold=0.12) -> BreakoutResult`. Rates are shrunk `adjusted = proj + w*(surface - proj)`, re-multiplied by the surface line's PT (held) to a counting line; `avg`/`era`/`whip` are carried as adjusted rates directly. Label from the aggregate signed, w-weighted deviation vs `deviation_threshold`; confidence `"low"` when sample below the stabilization sample or xStats absent.

- [ ] **Step 1: Write the failing test**

```python
def test_adjust_line_orders_skill_above_luck_at_equal_surface():
    proj = {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80, "sb": 10, "avg": 0.260}
    surface = {"pa": 600, "ab": 540, "hr": 40, "r": 100, "rbi": 110, "sb": 10, "avg": 0.300}
    backed = breakout.SkillLuckRow(mlbam=1, player_type="hitter", pa=600, ip=0.0, age=26.0,
        barrel_pct=0.16, xslg=0.580, slg=0.580, xba=0.298, ba=0.300, babip=0.300,
        xwoba=0.380, woba=0.382, k_pct=0.20, bb_pct=0.10)
    lucky = breakout.SkillLuckRow(mlbam=2, player_type="hitter", pa=600, ip=0.0, age=26.0,
        barrel_pct=0.06, xslg=0.410, slg=0.580, xba=0.255, ba=0.300, babip=0.380,
        xwoba=0.315, woba=0.382, k_pct=0.24, bb_pct=0.06)
    rb = breakout.adjust_line(surface, proj, backed, "hitter")
    rl = breakout.adjust_line(surface, proj, lucky, "hitter")
    # same surface, but the barrel-backed hitter keeps more of the HR jump
    assert rb.adjusted_line["hr"] > rl.adjusted_line["hr"]
    assert rb.label == "real breakout"
    assert rl.label in ("lucky mirage", "stable")
    assert "hr" in rb.reason.lower() or "barrel" in rb.reason.lower()

def test_adjust_line_low_confidence_small_sample():
    proj = {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80, "sb": 10, "avg": 0.260}
    surface = {"pa": 80, "ab": 70, "hr": 8, "r": 15, "rbi": 16, "sb": 1, "avg": 0.330}
    row = breakout.SkillLuckRow(mlbam=3, player_type="hitter", pa=80, ip=0.0, age=24.0,
        barrel_pct=0.14, xslg=0.520, slg=0.560, xba=0.290, ba=0.330, babip=0.360,
        xwoba=0.360, woba=0.370, k_pct=0.22, bb_pct=0.08)
    r = breakout.adjust_line(surface, proj, row, "hitter")
    assert r.confidence == "low"
    # small sample -> adjusted HR rate pulled toward the projection rate, not the hot surface
    assert r.adjusted_line["hr"] < surface["hr"]

def test_adjust_line_real_decline_labeled():
    proj = {"pa": 600, "ab": 540, "hr": 35, "r": 100, "rbi": 105, "sb": 8, "avg": 0.290}
    surface = {"pa": 600, "ab": 540, "hr": 15, "r": 65, "rbi": 60, "sb": 4, "avg": 0.235}
    # underlying confirms the drop is real: xSLG/xBA/xwOBA all down with the surface
    row = breakout.SkillLuckRow(mlbam=4, player_type="hitter", pa=600, ip=0.0, age=34.0,
        barrel_pct=0.05, xslg=0.360, slg=0.360, xba=0.238, ba=0.235, babip=0.270,
        xwoba=0.300, woba=0.302, k_pct=0.27, bb_pct=0.06)
    r = breakout.adjust_line(surface, proj, row, "hitter")
    assert r.label == "real decline"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_breakout.py -k adjust_line -v`
Expected: FAIL (`adjust_line` undefined).

- [ ] **Step 3: Write minimal implementation**

```python
_RATE_ONLY = {"avg", "era", "whip"}

def adjust_line(surface_line, projection_line, row, player_type, *,
                params=DEFAULT_WMAP, deviation_threshold=0.12):
    s_rates = line_rates(surface_line, player_type)
    p_rates = line_rates(projection_line, player_type)
    pt = safe_float(surface_line.get("pa" if player_type == "hitter" else "ip", 0))
    adjusted = dict(surface_line)  # carry non-scored fields (positions, ab, ip, etc.)
    w_by_stat: dict[str, float] = {}
    believed = 0.0   # w-weighted signed deviation -> drives the label
    surface = 0.0    # raw (unweighted) signed deviation -> drives the deviator flag
    for stat, s_rate in s_rates.items():
        p_rate = p_rates.get(stat, s_rate)
        w = w_for_stat(stat, row, player_type, params)
        w_by_stat[stat] = w
        adj_rate = p_rate + w * (s_rate - p_rate)
        # luck-direction aware for era/whip (lower = better)
        direction = -1.0 if stat in ("era", "whip") else 1.0
        denom = abs(p_rate) if abs(p_rate) > 1e-9 else 1.0
        term = direction * (s_rate - p_rate) / denom
        believed += w * term
        surface += term
        if stat in _RATE_ONLY:
            adjusted[stat] = adj_rate
        else:
            adjusted[stat] = adj_rate * pt
    label = _label(believed, surface, deviation_threshold)
    reason = _reason(s_rates, p_rates, w_by_stat, player_type)
    sample = row.pa if player_type == "hitter" else row.ip
    stab = params.stat_stabilize.get("hr" if player_type == "hitter" else "k", params.pa_stabilize)
    confidence = "low" if sample < stab or row.xwoba is None else "full"
    return BreakoutResult(adjusted, label, reason, w_by_stat, confidence, surface, believed)

def _label(believed, surface, thr):
    # believed (w-weighted) confirms real movement; surface (raw) with a quieted
    # believed signal is luck.
    if believed >= thr:
        return "real breakout"
    if believed <= -thr:
        return "real decline"
    if surface >= thr:
        return "lucky mirage"   # surface jumped, w regressed it away
    if surface <= -thr:
        return "slump"          # surface cratered, underlying says it recovers
    return "stable"

def _reason(s_rates, p_rates, w_by_stat, player_type):
    # name the largest-magnitude believed mover
    best = max(s_rates, key=lambda k: abs((s_rates[k] - p_rates.get(k, s_rates[k])) * w_by_stat.get(k, 0)))
    delta = s_rates[best] - p_rates.get(best, s_rates[best])
    dirn = "up" if delta > 0 else "down"
    return f"{best} {dirn}, w={w_by_stat.get(best, 0):.2f}"
```

> `_label` distinguishes mirage/slump from stable by comparing the raw `surface`
> deviation against the `believed` (w-weighted) one: a big surface jump that `w`
> regressed away is a `lucky mirage`. The ordering test accepts
> `("lucky mirage", "stable")` for `rl`, so it passes either way; tighten it to
> exactly `"lucky mirage"` once you have confirmed the seed thresholds on real data.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_breakout.py -v`
Expected: PASS (all breakout tests).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/breakout.py tests/test_analysis/test_breakout.py
git commit -m "feat(breakout): adjust_line -> skill-adjusted line + label + reason"
```

---

## Phase 3: Report

### Task 7: Breakout report (surface vs skill-adjusted keeper value)

**Files:**
- Create: `scripts/run_breakout_report.py`
- Modify: `src/fantasy_baseball/analysis/breakout.py` (add the pure row-builder)
- Test: `tests/test_analysis/test_breakout.py`

**Interfaces:**
- Consumes: `adjust_line` (Task 6); `keeper_value.keeper_value` (unchanged); `scripts/keeper_value.py` helpers (`build_board_from_frames` result, `zips_index`, `_zips_by_year`, `load_current_full_season_lines`, `overlay_current_anchors`) imported as `import keeper_value as kv_script` (the `keeper_trades.py` pattern).
- Produces: `breakout_rows(board, scale, indices, skill_luck, projections, *, base_year, horizon, discount) -> list[dict]` -- pure over already-loaded inputs. For each board row it computes `surface_value` (via `keeper_value` with the row's anchor) and `adjusted_value` (via `keeper_value` with `adjust_line(...).adjusted_line`), plus `delta`, `label`, `reason`, `confidence`. The script wraps it with I/O and writes `breakout_report.csv` / `breakout_report.md`.

- [ ] **Step 1: Write the failing test** (pure row-builder, fake keeper_value inputs)

```python
def test_breakout_rows_surface_equals_kv_and_adjusted_regresses_luck(monkeypatch):
    import pandas as pd
    from fantasy_baseball.analysis import breakout, keeper_value
    # one lucky hitter: hot surface, flat xStats -> adjusted value below surface value
    board = pd.DataFrame([{
        "player_id": "Lucky Guy::hitter", "name": "Lucky Guy", "player_type": "hitter",
        "positions": ["OF"], "fg_id": "20123",
        "pa": 600, "ab": 540, "hr": 40, "r": 100, "rbi": 110, "sb": 10, "avg": 0.320,
    }])
    projections = {"20123::hitter": {"pa": 600, "ab": 540, "hr": 20, "r": 80, "rbi": 80,
                                     "sb": 10, "avg": 0.260}}
    skill_luck = {20123: breakout.SkillLuckRow(mlbam=665742, player_type="hitter", pa=600, ip=0.0,
        age=27.0, barrel_pct=0.06, xslg=0.410, slg=0.560, xba=0.255, ba=0.320, babip=0.385,
        xwoba=0.315, woba=0.380, k_pct=0.24, bb_pct=0.05)}
    # stub keeper_value to a monotonic function of HR so the test is deterministic
    def fake_kv(pid, name, anchor, pos, ptype, zby, scale, **kw):
        return keeper_value.KeeperValueResult(pid, name, {2026: anchor["hr"]}, float(anchor["hr"]),
                                              [], None)
    monkeypatch.setattr(breakout, "_kv", fake_kv, raising=False)
    rows = breakout.breakout_rows(board, scale=None, indices={}, skill_luck=skill_luck,
                                  projections=projections, base_year=2026, horizon=3, discount=0.8)
    row = rows[0]
    assert row["surface_value"] == 40.0            # surface anchor untouched
    assert row["adjusted_value"] < row["surface_value"]  # luck regressed out
    assert row["delta"] == row["adjusted_value"] - row["surface_value"]
    assert row["label"] in breakout.LABELS
    # spec-required deviator flag + underlying numbers
    assert row["deviator"] is True                 # HR 40 vs proj 20 is a big surface move
    assert abs(row["woba_xwoba_gap"] - (0.380 - 0.315)) < 1e-9
    assert row["babip"] == 0.385 and row["barrel_pct"] == 0.06
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_breakout.py::test_breakout_rows_surface_equals_kv_and_adjusted_regresses_luck -v`
Expected: FAIL (`breakout_rows` undefined).

- [ ] **Step 3: Write minimal implementation**

```python
# in breakout.py -- indirection so tests can stub the keeper_value call
from fantasy_baseball.analysis import keeper_value as _kv_mod

def _kv(pid, name, anchor, positions, ptype, zips_by_year, scale, **kw):
    return _kv_mod.keeper_value(pid, name, anchor, positions, ptype, zips_by_year, scale, **kw)

DEVIATION_THRESHOLD = 0.12  # shared by adjust_line's label and the report's deviator flag

def breakout_rows(board, scale, indices, skill_luck, projections, *, base_year, horizon, discount,
                  out_year_regression=_kv_mod.DEFAULT_OUT_YEAR_REGRESSION):
    # out_year_regression MUST match scripts/keeper_value.py:build_results so the surface
    # value equals today's --anchor current number and surface/adjusted differ ONLY in the anchor.
    rows = []
    for _, r in board.iterrows():
        row = r.to_dict()
        ptype = str(row["player_type"])
        fg = row.get("fg_id")
        fgid = int(fg) if fg is not None and str(fg).isdigit() else None
        positions = list(row["positions"])
        zby = _zips_for(row, indices, base_year, horizon)  # helper mirrors kv_script._zips_by_year
        surface = _kv(row["player_id"], row["name"], row, positions, ptype, zby, scale,
                      base_year=base_year, horizon=horizon, discount=discount,
                      out_year_regression=out_year_regression).total
        sl = skill_luck.get(fgid) if fgid is not None else None
        proj = projections.get(f"{fg}::{ptype}") if fg is not None else None
        if sl is not None and proj is not None:
            res = adjust_line(row, proj, sl, ptype, deviation_threshold=DEVIATION_THRESHOLD)
            adjusted = _kv(row["player_id"], row["name"], res.adjusted_line, positions, ptype, zby,
                           scale, base_year=base_year, horizon=horizon, discount=discount,
                           out_year_regression=out_year_regression).total
            gap = (sl.woba - sl.xwoba) if sl.woba is not None and sl.xwoba is not None else None
            under = {"woba_xwoba_gap": gap, "babip": sl.babip, "barrel_pct": sl.barrel_pct,
                     "k_pct": sl.k_pct, "bb_pct": sl.bb_pct}
            rows.append({"name": row["name"], "player_type": ptype, "surface_value": surface,
                         "adjusted_value": adjusted, "delta": adjusted - surface,
                         "label": res.label, "reason": res.reason, "confidence": res.confidence,
                         "deviator": abs(res.surface_deviation) >= DEVIATION_THRESHOLD, **under})
        else:
            rows.append({"name": row["name"], "player_type": ptype, "surface_value": surface,
                         "adjusted_value": surface, "delta": 0.0, "label": "stable",
                         "reason": "no skill/luck data", "confidence": "low", "deviator": False,
                         "woba_xwoba_gap": None, "babip": None, "barrel_pct": None,
                         "k_pct": None, "bb_pct": None})
    rows.sort(key=lambda d: d["adjusted_value"], reverse=True)
    return rows
```

`DEVIATION_THRESHOLD = 0.12` is a module constant in `breakout.py` (equal to the default `adjust_line` carries), so the report's `deviator` flag and the classifier's label share one threshold.

`_zips_for` mirrors `scripts/keeper_value.py:_zips_by_year` (look up each year's ZiPS index by fg_id then name; miss -> None). With the empty `indices={}` the test passes, it returns `{}`, which `keeper_value` treats as all-out-year-missing (year-0 anchor still values via the anchor line):

```python
def _zips_for(row, indices, base_year, horizon):
    from fantasy_baseball.sgp.rankings import lookup_rank
    fg = row.get("fg_id")
    fgid = str(fg) if fg is not None and str(fg).strip() else None
    ptype = str(row["player_type"])
    return {yr: (lookup_rank(idx, fgid, row["name"], ptype) or None)
            for yr, idx in indices.items()}
```

The `run_breakout_report.py` script loads board/scale/indices/anchors via `kv_script` helpers and `build_hitter_skill_luck`/`build_pitcher_skill_luck`, calls `breakout_rows`, and writes CSV + markdown (reuse the `render`-style table in `scripts/keeper_value.py`) with a "provisional -- pending backtest" banner. The script is not unit-tested (I/O + network); its logic lives in the tested `breakout_rows`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_breakout.py -v`
Expected: PASS.

- [ ] **Step 5: Manual smoke (documented, not CI)**

Run: `python scripts/run_breakout_report.py --limit 25` (requires synced `cache:full_season_projections` + a first `data/skill_luck/` fetch). Expected: a ranked table where known 2026 mirages carry a negative `delta` and barrel-backed breakouts a near-zero `delta`. Numbers are labeled **provisional** until Phase 4.

- [ ] **Step 6: Commit**

```bash
git add scripts/run_breakout_report.py src/fantasy_baseball/analysis/breakout.py tests/test_analysis/test_breakout.py
git commit -m "feat(breakout): report ranking surface vs skill-adjusted keeper value"
```

---

## Phase 4: Backtest

### Task 8: Marcel-style reconstructed prior (pure)

**Files:**
- Create: `src/fantasy_baseball/analysis/breakout_backtest.py`
- Test: `tests/test_analysis/test_breakout_backtest.py`

**Interfaces:**
- Consumes: nothing (pure).
- Produces: `marcel_prior(history: list[tuple[int, dict[str, float]]], league_mean: dict[str, float], age: float | None) -> dict[str, float]` -- weighted (5/4/3 by recency) prior-year rate lines, regressed toward `league_mean` with a fixed regression sample, plus a simple linear age adjustment (peak ~27). Returns a projected rate line usable as `projection_line` for the backtest estimators.

- [ ] **Step 1: Write the failing test**

```python
from fantasy_baseball.analysis import breakout_backtest as bb

def test_marcel_weights_recent_years_more():
    league = {"hr": 0.03}
    # year 2023 hot (0.06), 2021 cold (0.01); recency weighting -> closer to hot
    hist = [(2023, {"hr": 0.06}), (2022, {"hr": 0.04}), (2021, {"hr": 0.01})]
    p = bb.marcel_prior(hist, league, age=27.0)
    assert 0.03 < p["hr"] < 0.06

def test_marcel_regresses_thin_history_toward_league():
    league = {"hr": 0.03}
    p = bb.marcel_prior([(2023, {"hr": 0.09})], league, age=27.0)  # one loud year
    assert p["hr"] < 0.09  # pulled toward league mean
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_breakout_backtest.py -v`
Expected: FAIL (`marcel_prior` undefined).

- [ ] **Step 3: Write minimal implementation**

```python
# src/fantasy_baseball/analysis/breakout_backtest.py
from __future__ import annotations

_RECENCY = {0: 5.0, 1: 4.0, 2: 3.0}   # most-recent .. 3rd
_REGRESS_W = 4.0                        # league-mean pseudo-weight

def marcel_prior(history, league_mean, age):
    history = sorted(history, key=lambda t: t[0], reverse=True)[:3]
    stats = set().union(*[set(d) for _, d in history]) if history else set(league_mean)
    prior = {}
    for s in stats:
        num = _REGRESS_W * league_mean.get(s, 0.0)
        den = _REGRESS_W
        for i, (_, line) in enumerate(history):
            wt = _RECENCY.get(i, 0.0)
            if line.get(s) is not None:
                num += wt * line[s]
                den += wt
        val = num / den if den > 0 else league_mean.get(s, 0.0)
        if age is not None:
            val *= 1.0 - 0.003 * (age - 27.0)   # mild peak-27 age curve
        prior[s] = val
    return prior
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_breakout_backtest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/breakout_backtest.py tests/test_analysis/test_breakout_backtest.py
git commit -m "feat(breakout-backtest): Marcel-style reconstructed prior"
```

---

### Task 9: Fixed-yardstick scoring (rate MAE + held-PT SGP) for one estimator

**Files:**
- Modify: `src/fantasy_baseball/analysis/breakout_backtest.py`
- Test: `tests/test_analysis/test_breakout_backtest.py`

**Interfaces:**
- Consumes: `marcel_prior`; `keeper_value._line_sgp` is NOT reused (needs ScaleInputs); instead a lightweight `sgp_on_ruler(rates, weights) -> float` weights rates by fixed roto importance.
- Produces:
  - `rate_mae(pred_rates: dict, actual_rates: dict) -> float`.
  - `sgp_on_ruler(rates: dict, weights: dict) -> float` -- one fixed-yardstick scalar (sum of `weights[s]*rates[s]`, era/whip negated).
  - `DEFAULT_RULER: dict[str, float]` -- fixed roto weights (documented, from SGP denominators order-of-magnitude).

- [ ] **Step 1: Write the failing test**

```python
def test_rate_mae_and_ruler():
    pred = {"hr": 0.05, "avg": 0.270}
    actual = {"hr": 0.04, "avg": 0.300}
    assert abs(bb.rate_mae(pred, actual) - (0.01 + 0.030) / 2) < 1e-9
    # ruler: higher HR rate -> higher score
    w = {"hr": 100.0, "avg": 10.0}
    assert bb.sgp_on_ruler({"hr": 0.05, "avg": 0.27}, w) > bb.sgp_on_ruler({"hr": 0.02, "avg": 0.27}, w)

def test_ruler_penalizes_era():
    w = {"era": -1.0}  # negative weight (DEFAULT_RULER convention): lower ERA -> higher score
    assert bb.sgp_on_ruler({"era": 3.0}, w) > bb.sgp_on_ruler({"era": 5.0}, w)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_breakout_backtest.py -k "mae or ruler" -v`
Expected: FAIL.

- [ ] **Step 3: Write minimal implementation**

```python
DEFAULT_RULER = {"hr": 100.0, "r": 60.0, "rbi": 60.0, "sb": 120.0, "avg": 1500.0,
                 "k": 40.0, "w": 300.0, "sv": 250.0, "era": -200.0, "whip": -400.0}

def rate_mae(pred_rates, actual_rates):
    keys = set(pred_rates) & set(actual_rates)
    if not keys:
        return 0.0
    return sum(abs(pred_rates[k] - actual_rates[k]) for k in keys) / len(keys)

def sgp_on_ruler(rates, weights):
    return sum(weights.get(s, 0.0) * v for s, v in rates.items())
```

(Convention: era/whip carry **negative** weights in `DEFAULT_RULER`, so `sgp_on_ruler` stays a plain dot product and lower ERA/WHIP yields a higher score. The test above uses `era: -1.0` to match.)

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_breakout_backtest.py -k "mae or ruler" -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/breakout_backtest.py tests/test_analysis/test_breakout_backtest.py
git commit -m "feat(breakout-backtest): fixed-yardstick rate-MAE and ruler SGP"
```

---

### Task 10: Backtest orchestration -- three estimators, held-out split, bootstrap CI

**Files:**
- Create: `scripts/backtest_breakout.py`
- Modify: `src/fantasy_baseball/analysis/breakout_backtest.py` (pure `run_backtest` over pre-loaded corpus)
- Test: `tests/test_analysis/test_breakout_backtest.py`

**Scope:** the backtest is **hitters-only in v1** (pitcher expected-stats coverage on Savant is thinner and starts later). This is stated in the `run_backtest` docstring and printed in the script summary -- not a silent cap. Pitcher backtest is a named follow-up.

**Interfaces:**
- Consumes: `marcel_prior`, `rate_mae`, `sgp_on_ruler`, `adjust_line`, `line_rates`, `WMapParams`/`DEFAULT_WMAP`.
- Produces:
  - `tune_wmap(corpus, fit_years, *, ruler=DEFAULT_RULER) -> WMapParams` -- small grid search over `confirm_weight` and the `hr`/`avg` stabilizers, selecting the params that maximize the skill-adjusted Spearman **on `fit_years` only** (never touches `report_years`).
  - `run_backtest(corpus, *, fit_years, report_years, params=None, ruler=DEFAULT_RULER) -> dict` where `corpus[year]` maps `fg_id -> (surface_line, skill_luck_row, actual_next_rates, history, zips_line_or_None)`. When `params is None` it calls `tune_wmap(corpus, fit_years)` first, then evaluates the fixed tuned params on the **candidate population** of `report_years` (players whose raw surface-vs-prior deviation clears `CANDIDATE_DEVIATION`). Returns Spearman of THREE estimators (`surface`, `skill_adjusted`, `pure_zips`) vs realized next-year ruler-SGP, a bootstrap CI on `(skill_adjusted - surface)` and on `(skill_adjusted - pure_zips)` (the latter over the ZiPS-covered subset only), per-estimator rate-MAE, and a `label_lift` (retention across believed-deviation terciles). The verdict clears only when BOTH CIs exclude zero.
  - The script builds `corpus` from cached `data/skill_luck/` frames across 2015-2024 and writes `data/stats/breakout_backtest_results.csv`.

- [ ] **Step 1: Write the failing test** (synthetic 2-year corpus)

```python
def _mk_corpus():
    from fantasy_baseball.analysis import breakout
    surface = {"pa": 600, "ab": 540, "hr": 40, "r": 100, "rbi": 110, "sb": 10, "avg": 0.320}
    lucky = breakout.SkillLuckRow(mlbam=1, player_type="hitter", pa=600, ip=0.0, age=27.0,
        barrel_pct=0.06, xslg=0.41, slg=0.56, xba=0.255, ba=0.320, babip=0.385,
        xwoba=0.315, woba=0.380, k_pct=0.24, bb_pct=0.05)
    real = breakout.SkillLuckRow(mlbam=2, player_type="hitter", pa=600, ip=0.0, age=26.0,
        barrel_pct=0.16, xslg=0.58, slg=0.58, xba=0.298, ba=0.300, babip=0.300,
        xwoba=0.382, woba=0.380, k_pct=0.20, bb_pct=0.10)
    # next-year: lucky regresses to ~proj HR rate, real sustains
    actual_lucky = {"hr": 0.033, "avg": 0.262}
    actual_real = {"hr": 0.062, "avg": 0.298}
    hist = [(2022, {"hr": 0.033, "avg": 0.262})]
    zips_lucky = {"pa": 600, "hr": 21, "avg": 0.265}   # ZiPS already regressed the mirage
    zips_real = {"pa": 600, "hr": 37, "avg": 0.296}
    return {2023: {
        10: (surface, lucky, actual_lucky, hist, zips_lucky),
        20: (surface, real, actual_real, hist, zips_real),
    }}

def test_tune_wmap_returns_params_without_touching_report_years():
    from fantasy_baseball.analysis import breakout_backtest as bb
    from fantasy_baseball.analysis.breakout import WMapParams
    p = bb.tune_wmap(_mk_corpus(), fit_years=[2023])
    assert isinstance(p, WMapParams)

def test_run_backtest_three_estimators_and_ci():
    from fantasy_baseball.analysis import breakout_backtest as bb
    out = bb.run_backtest(_mk_corpus(), fit_years=[2023], report_years=[2023])
    # all three estimators present; skill-adjusted at least ties surface at ranking
    assert set(out["spearman"]) == {"surface", "skill_adjusted", "pure_zips"}
    assert out["spearman"]["skill_adjusted"] >= out["spearman"]["surface"]
    assert "ci_skill_vs_surface" in out and "ci_skill_vs_zips" in out
    assert "label_lift" in out and out["verdict"] in ("clears gate", "not good enough")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_analysis/test_breakout_backtest.py::test_run_backtest_skill_beats_surface_on_synthetic -v`
Expected: FAIL (`run_backtest` undefined).

- [ ] **Step 3: Write minimal implementation**

```python
from statistics import fmean

from fantasy_baseball.analysis.breakout import DEFAULT_WMAP, WMapParams, adjust_line, line_rates

CANDIDATE_DEVIATION = 0.15   # raw surface-vs-prior deviation to count as a breakout/decline candidate
_TUNE_GRID = {"confirm_weight": [0.3, 0.5, 0.7], "hr": [80.0, 120.0, 200.0], "avg": [600.0, 800.0, 1200.0]}

def _league_mean(year_data):
    rows = [line_rates(s, "hitter") for s, *_ in year_data.values()]
    keys = set().union(*[set(r) for r in rows]) if rows else set()
    return {k: fmean([r.get(k, 0.0) for r in rows]) for k in keys}

def _spearman(xs, ys):
    if len(xs) < 2:
        return 0.0
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for pos, i in enumerate(order):
            rk[i] = float(pos)
        return rk
    rx, ry = ranks(xs), ranks(ys)
    mx, my = fmean(rx), fmean(ry)
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx) ** 0.5
    vy = sum((b - my) ** 2 for b in ry) ** 0.5
    return cov / (vx * vy) if vx > 0 and vy > 0 else 0.0

def _params(confirm_weight, hr_stab, avg_stab):
    base = dict(DEFAULT_WMAP.stat_stabilize)
    base.update({"hr": hr_stab, "avg": avg_stab})
    return WMapParams(confirm_weight=confirm_weight, stat_stabilize=base)

def _records(corpus, years, params, ruler):
    """Per-candidate scored records for `years` under `params` (hitters only)."""
    recs = []
    for year in years:
        year_data = corpus[year]
        lg = _league_mean(year_data)
        for surface, sl, actual_next, hist, zips_line in year_data.values():
            proj_line = {**surface, **_rates_to_line(marcel_prior(hist, lg, sl.age), surface)}
            res = adjust_line(surface, proj_line, sl, "hitter", params=params,
                              deviation_threshold=CANDIDATE_DEVIATION)
            if abs(res.surface_deviation) < CANDIDATE_DEVIATION:
                continue  # not a breakout/decline candidate this year
            recs.append({
                "surface": sgp_on_ruler(line_rates(surface, "hitter"), ruler),
                "skill": sgp_on_ruler(line_rates(res.adjusted_line, "hitter"), ruler),
                "zips": (sgp_on_ruler(line_rates(zips_line, "hitter"), ruler)
                         if zips_line is not None else None),
                "actual": sgp_on_ruler(actual_next, ruler),
                "believed": res.believed_deviation,
                "surface_rates": line_rates(surface, "hitter"),
                "adjusted_rates": line_rates(res.adjusted_line, "hitter"),
                "prior_rates": line_rates(proj_line, "hitter"),
                "actual_rates": actual_next,
            })
    return recs

def tune_wmap(corpus, fit_years, *, ruler=DEFAULT_RULER):
    """Grid-search w-params on fit_years ONLY; never reads report_years."""
    best, best_rho = DEFAULT_WMAP, -2.0
    for cw in _TUNE_GRID["confirm_weight"]:
        for hr in _TUNE_GRID["hr"]:
            for avg in _TUNE_GRID["avg"]:
                p = _params(cw, hr, avg)
                recs = _records(corpus, fit_years, p, ruler)
                rho = _spearman([r["skill"] for r in recs], [r["actual"] for r in recs])
                if rho > best_rho:
                    best, best_rho = p, rho
    return best

def _retention(rec):
    holds = []
    for s, sr in rec["surface_rates"].items():
        pr = rec["prior_rates"].get(s, sr)
        ar = rec["actual_rates"].get(s)
        if ar is None or abs(sr - pr) < 1e-9:
            continue
        holds.append((ar - pr) / (sr - pr))   # 1.0 = fully held, 0.0 = fully regressed
    return fmean(holds) if holds else 0.0

def _label_lift(recs):
    if len(recs) < 3:
        return 0.0
    ordered = sorted(recs, key=lambda r: r["believed"])
    k = len(ordered) // 3
    return fmean([_retention(r) for r in ordered[-k:]]) - fmean([_retention(r) for r in ordered[:k]])

def run_backtest(corpus, *, fit_years, report_years, params=None, ruler=DEFAULT_RULER):
    """Hitters-only v1. Tunes w on fit_years (unless params given), evaluates the
    fixed params on the candidate population of report_years across three estimators.
    Pitcher backtest is a named follow-up (thinner Savant pitcher xStats coverage)."""
    if params is None:
        params = tune_wmap(corpus, fit_years, ruler=ruler)
    recs = _records(corpus, report_years, params, ruler)
    actual = [r["actual"] for r in recs]
    zrecs = [r for r in recs if r["zips"] is not None]
    spearman = {
        "surface": _spearman([r["surface"] for r in recs], actual),
        "skill_adjusted": _spearman([r["skill"] for r in recs], actual),
        "pure_zips": _spearman([r["zips"] for r in zrecs], [r["actual"] for r in zrecs]),
    }
    ci_vs_surface = _bootstrap_diff([r["skill"] for r in recs], [r["surface"] for r in recs], actual)
    ci_vs_zips = (_bootstrap_diff([r["skill"] for r in zrecs], [r["zips"] for r in zrecs],
                                  [r["actual"] for r in zrecs]) if len(zrecs) >= 2 else (0.0, 0.0))
    clears = ci_vs_surface[0] > 0 and ci_vs_zips[0] > 0
    return {
        "spearman": spearman,
        "ci_skill_vs_surface": ci_vs_surface,
        "ci_skill_vs_zips": ci_vs_zips,
        "rate_mae": {
            "surface": fmean([rate_mae(r["surface_rates"], r["actual_rates"]) for r in recs]) if recs else 0.0,
            "skill_adjusted": fmean([rate_mae(r["adjusted_rates"], r["actual_rates"]) for r in recs]) if recs else 0.0,
        },
        "label_lift": _label_lift(recs),
        "verdict": "clears gate" if clears else "not good enough",
        "n": len(recs),
    }
```

Local helpers in the same file (deterministic -- seeded RNG, no wall-clock entropy):

```python
import random

def _rates_to_line(rate_line, pt_source):
    # rebuild a counting line at the PT-source's playing time so adjust_line sees
    # a projection line in counting shape. Rate-only stats pass through.
    pt = float(pt_source.get("pa", 0.0))
    line = dict(pt_source)
    for s, v in rate_line.items():
        line[s] = v if s in ("avg", "era", "whip") else v * pt
    return line

def _bootstrap_diff(a, b, actual, *, iters=2000, seed=0):
    # 95% CI on spearman(a, actual) - spearman(b, actual) via seeded resampling.
    rng = random.Random(seed)
    n = len(actual)
    if n < 2:
        return 0.0, 0.0
    diffs = []
    for _ in range(iters):
        idx = [rng.randrange(n) for _ in range(n)]
        da = _spearman([a[i] for i in idx], [actual[i] for i in idx])
        db = _spearman([b[i] for i in idx], [actual[i] for i in idx])
        diffs.append(da - db)
    diffs.sort()
    return diffs[int(0.025 * iters)], diffs[int(0.975 * iters)]
```

The script iterates cached years and assembles `corpus`: for each year it joins the
cached `data/skill_luck/` frames (surface = that year's actual line, `SkillLuckRow` =
that year's underlying, `actual_next` = the following year's actual rates, `history` =
prior years' actual rate lines for the Marcel prior) and attaches `zips_line` from the
archived ZiPS export (via `scripts/keeper_value.py:load_zips_year`) for the 2022-2024
report years, `None` elsewhere. It then calls
`run_backtest(fit_years=range(2015, 2023), report_years=[2023, 2024])`, writes the
per-estimator results to `data/stats/breakout_backtest_results.csv`, and prints the
summary including `verdict` (`clears gate` = both `ci_skill_vs_surface[0] > 0` and
`ci_skill_vs_zips[0] > 0`; else `not good enough -> automation stays deferred`), a note
that v1 is **hitters-only**, and the ZiPS-covered subset size.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_analysis/test_breakout_backtest.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/backtest_breakout.py src/fantasy_baseball/analysis/breakout_backtest.py tests/test_analysis/test_breakout_backtest.py
git commit -m "feat(breakout-backtest): year-over-year 3-estimator backtest with held-out bootstrap CI"
```

---

## Final verification (run before declaring the plan complete)

- [ ] `pytest -v` (at minimum `tests/test_data/test_skill_luck.py` and `tests/test_analysis/test_breakout*.py`; run the full suite to catch regressions).
- [ ] `ruff check .` -- zero violations.
- [ ] `ruff format --check .` -- no drift.
- [ ] `vulture` -- no NEW dead-code findings (the pitcher siblings and script entry points may need a `# noqa`/whitelist entry if flagged as unused; script `main()` bodies are entry points).
- [ ] `mypy` -- clean for `src/fantasy_baseball/analysis/breakout.py` and `breakout_backtest.py` (covered by `[tool.mypy].files` via `analysis/`). Add precise types; `SkillLuckRow`/`BreakoutResult` are already typed dataclasses.
- [ ] Report the commands and their output in the final message. Do NOT claim checks pass without showing them.

## Notes for the executor

- The classifier's seed `w`-mapping and label thresholds are **provisional** until Task 10's backtest runs on real cached data; the report must print a "provisional -- pending backtest" banner (Phase 3).
- Live data fetches (Tasks 1-3 defaults) hit FanGraphs/Savant and are NOT exercised in tests. First real fetch may trip the `_rename_strict` guard if pybaseball's column spellings drifted -- that is the guard doing its job; print `raw.columns.tolist()` and update the rename map, do not silence it.
- Pitcher parity functions are folded into their hitter task (same seam). If a reviewer would reject the hitter path without the pitcher path, split them; otherwise keep together.
- The spec mentions name-normalization tie-breaks and a name-collision join test. This plan's cross-source join is **purely id-based** (FanGraphs `IDfg` <-> MLBAM via the Chadwick register; board joined on its own `fg_id`), so there is no name key in the join path and the collision test is moot -- a board row lacking a numeric `fg_id` simply gets no adjustment (falls back to the surface value) rather than risking a wrong name match. If a future change reintroduces a name-keyed fallback, add the deliberate-collision test then.
- The backtest (Task 10) is **hitters-only in v1** by design (thinner/later Savant pitcher expected-stats coverage); the classifier and report handle both. This is stated in the `run_backtest` docstring and the script summary, not silently capped.
