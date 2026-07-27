# Keepers Module Reorg (Raw Ingest Only) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-07-27-keepers-module-reorg-design.md`

**Goal:** Stand up a fresh `src/fantasy_baseball/keepers/` module holding ONLY the raw MLB Stats API + Baseball Savant network pulls (fully raw frames, no derivation), and delete the distrusted breakout/mirage/HR-confirmation analysis + backtests + reports.

**Architecture:** Split-by-source package: `cache.py` (fetch-on-miss plumbing), `mlb_stats.py` (statsapi season line via `requests`), `savant.py` (pybaseball expected-stats/barrels + a direct xHR CSV). Each fetcher returns the upstream response unmodified; #266 rebuilds all derivation. `keeper_value.py`/`keeper_trades.py` are left untouched. Build the new module first (green), then delete the old code, then verify all gates.

**Tech Stack:** Python 3.11, pandas (UNTYPED here -- no pandas-stubs), `requests` (typed via `types-requests`), `pybaseball` (untyped), pytest, ruff, mypy, vulture.

## Global Constraints

- **Fully raw, no calculation.** Fetchers return the upstream frame with NO column selection, rename, arithmetic, unit conversion, join, or shape-building. (Owner: "all calculation should not carry over"; "I don't trust any decisions made including choosing which columns to drop.")
- **ASCII only** in all source, strings, and log/format text (Windows cp1252 stdout). Use `--`, `->`, straight quotes -- never Unicode dashes/quotes/arrows.
- **mypy under `warn_return_any = true` + untyped pandas/pybaseball:** any function returning the result of a `pd.*` or `pybaseball` call MUST bind it to a typed local first (`result: pd.DataFrame = <call>(...); return result`) -- a bare `return <call>(...)` returns `Any` and fails the gate.
- **No numeric-falsy defaults** (`x or default`) -- irrelevant here (no numeric defaults), but keep the rule in mind.
- **Gates:** `pytest`, `ruff check .`, `ruff format --check .` per touched task; `mypy` for any file under `[tool.mypy] files` (we add `keepers/` to it); `vulture` at the FINAL task only (the new fetchers are legitimately callerless until #266, so intermediate per-task vulture runs would false-positive -- vulture is the end-of-effort gate).
- **Player IDs / keys:** raw frames keep the upstream id columns (`player.id` / `player_id`) as-is; do not re-key.

---

### Task 1: `cache.py` plumbing + package scaffolding + config

**Files:**
- Create: `src/fantasy_baseball/keepers/__init__.py`
- Create: `src/fantasy_baseball/keepers/cache.py`
- Create: `tests/test_keepers/__init__.py` (empty; sibling test dirs use one)
- Create: `tests/test_keepers/test_cache.py`
- Modify: `pyproject.toml` (`[tool.mypy] files` add `keepers/`; `[tool.vulture] ignore_names` add the 5 fetcher names)

**Interfaces:**
- Produces: `fetch_or_cache(path: Path, fetcher: Callable[[], pd.DataFrame], *, tolerate_empty: bool = False) -> pd.DataFrame` (used by Tasks 2-3). Private `_read_cached(path: Path) -> pd.DataFrame | None`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_keepers/__init__.py` (empty file) and `tests/test_keepers/test_cache.py`:

```python
from pathlib import Path

import pandas as pd
import pytest

from fantasy_baseball.keepers.cache import fetch_or_cache


def test_miss_fetches_and_writes(tmp_path: Path):
    path = tmp_path / "x.csv"
    df = pd.DataFrame({"a": [1, 2]})
    out = fetch_or_cache(path, lambda: df)
    pd.testing.assert_frame_equal(out, df)
    assert path.exists()


def test_hit_reads_cache_without_fetching(tmp_path: Path):
    path = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(path, index=False)

    def boom() -> pd.DataFrame:
        raise AssertionError("fetcher must not be called on a cache hit")

    out = fetch_or_cache(path, boom)
    assert list(out["a"]) == [1]


def test_empty_pull_raises_and_writes_nothing(tmp_path: Path):
    path = tmp_path / "x.csv"
    with pytest.raises(RuntimeError):
        fetch_or_cache(path, lambda: pd.DataFrame())
    assert not path.exists()


def test_tolerate_empty_returns_without_writing(tmp_path: Path):
    path = tmp_path / "x.csv"
    empty = pd.DataFrame()
    out = fetch_or_cache(path, lambda: empty, tolerate_empty=True)
    assert out.empty
    assert not path.exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_keepers/test_cache.py -v`
Expected: FAIL / collection error -- `ModuleNotFoundError: fantasy_baseball.keepers.cache`.

- [ ] **Step 3: Write minimal implementation**

Create `src/fantasy_baseball/keepers/__init__.py`:

```python
"""Keepers module: raw MLB Stats API + Baseball Savant data pulls.

Fetchers return the upstream response fully raw -- no derivation, rename, or join.
Downstream keeper-value logic (#266) is built on top of these; nothing here computes.
"""

from __future__ import annotations

from fantasy_baseball.keepers.cache import fetch_or_cache

__all__ = ["fetch_or_cache"]
```

Create `src/fantasy_baseball/keepers/cache.py`:

```python
"""Fetch-on-miss CSV cache plumbing shared by the keepers data pulls.

Never overwrites a good cache with an empty/failed pull. Not calculation --
pure I/O plumbing, preserved verbatim from the old data/skill_luck.py.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pandas as pd


def _read_cached(path: Path) -> pd.DataFrame | None:
    if path.exists():
        df: pd.DataFrame = pd.read_csv(path)
        if not df.empty:
            return df
    return None


def fetch_or_cache(
    path: Path, fetcher: Callable[[], pd.DataFrame], *, tolerate_empty: bool = False
) -> pd.DataFrame:
    cached = _read_cached(path)
    if cached is not None:
        return cached
    df = fetcher()
    if df is None or df.empty:
        if tolerate_empty and df is not None:
            return df  # expected-empty (e.g. pre-2016 xHR); do not cache emptiness
        raise RuntimeError(
            f"fetch for {path.name} returned empty; refusing to overwrite/write cache"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_keepers/test_cache.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Wire config (mypy + vulture)**

In `pyproject.toml`, add to the `[tool.mypy] files` list (alphabetically near the other `src/fantasy_baseball/` entries):

```toml
    "src/fantasy_baseball/keepers/",
```

In `pyproject.toml` `[tool.vulture]`, extend `ignore_names` with the five public fetchers introduced in Tasks 2-3 (harmless now; names need not exist yet). Do NOT add a `fetch_*` glob:

```toml
ignore_names = [
    "setUp",
    "tearDown",
    "test_*",
    "fetch_mlb_season",
    "fetch_batter_expected",
    "fetch_batter_barrels",
    "fetch_pitcher_expected",
    "fetch_savant_hr",
]
```

- [ ] **Step 6: Verify gates for touched files**

Run: `ruff check src/fantasy_baseball/keepers/ tests/test_keepers/ && ruff format --check src/fantasy_baseball/keepers/ tests/test_keepers/ && mypy src/fantasy_baseball/keepers/cache.py`
Expected: all clean (no errors).

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/keepers/__init__.py src/fantasy_baseball/keepers/cache.py \
        tests/test_keepers/__init__.py tests/test_keepers/test_cache.py pyproject.toml
git commit -m "feat(keepers): cache plumbing + package scaffold (#265)"
```

---

### Task 2: `mlb_stats.py` -- MLB Stats API season line (fully raw)

**Files:**
- Create: `src/fantasy_baseball/keepers/mlb_stats.py`
- Create: `tests/test_keepers/test_mlb_stats.py`
- Modify: `src/fantasy_baseball/keepers/__init__.py` (export `fetch_mlb_season`)

**Interfaces:**
- Consumes: `fetch_or_cache` from Task 1.
- Produces: `fetch_mlb_season(cache_dir: Path, year: int, group: str, *, fetcher: Callable[[], pd.DataFrame] | None = None) -> pd.DataFrame` where `group` in `{"hitting","pitching"}`. Private `_fetch_mlb_season(group: str, year: int, *, get: Callable[..., Any] | None = None) -> pd.DataFrame`, module constant `_MLB_PAGE`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_keepers/test_mlb_stats.py`:

```python
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.keepers import mlb_stats
from fantasy_baseball.keepers.mlb_stats import _fetch_mlb_season, fetch_mlb_season


class _FakeResp:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict[str, Any]:
        return self._payload


def _page(splits: list[dict[str, Any]]) -> dict[str, Any]:
    return {"stats": [{"splits": splits}]}


def test_fetch_mlb_season_passthrough_and_caches(tmp_path: Path):
    raw = pd.DataFrame(
        {"player.id": [1], "player.fullName": ["A"], "stat.homeRuns": [10]}
    )
    out = fetch_mlb_season(tmp_path, 2024, "hitting", fetcher=lambda: raw)
    pd.testing.assert_frame_equal(out, raw)
    assert (tmp_path / "mlb_hitting_2024.csv").exists()


def test_fetch_mlb_season_paginates_and_keeps_all_columns(monkeypatch):
    monkeypatch.setattr(mlb_stats, "_MLB_PAGE", 2)
    pages = [
        _page(
            [
                {"player": {"id": 1, "fullName": "A"}, "stat": {"homeRuns": 10}, "team": {"name": "X"}},
                {"player": {"id": 2, "fullName": "B"}, "stat": {"homeRuns": 5}, "team": {"name": "Y"}},
            ]
        ),
        _page(
            [
                {"player": {"id": 3, "fullName": "C"}, "stat": {"homeRuns": 1}, "team": {"name": "Z"}},
            ]
        ),
    ]
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, *, params: dict[str, Any], timeout: int) -> _FakeResp:
        calls.append(params)
        return _FakeResp(pages[params["offset"] // 2])

    df = _fetch_mlb_season("hitting", 2024, get=fake_get)

    assert len(df) == 3
    assert len(calls) == 2  # stopped after the short (< _MLB_PAGE) page
    # nothing dropped -- name and team survive (the old {mlbam, **stat} dropped them)
    assert "player.fullName" in df.columns
    assert "team.name" in df.columns
    assert "stat.homeRuns" in df.columns
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_keepers/test_mlb_stats.py -v`
Expected: FAIL -- `ModuleNotFoundError: fantasy_baseball.keepers.mlb_stats`.

- [ ] **Step 3: Write minimal implementation**

Create `src/fantasy_baseball/keepers/mlb_stats.py`:

```python
"""Raw MLB Stats API season-stats leaderboard pull, keyed by MLBAM.

Returns the fully raw response: every split is json_normalized once across all
pages -- no column dropped, no rate derived. #266 selects/derives what it needs.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from fantasy_baseball.keepers.cache import fetch_or_cache

_MLB_STATS_URL = "https://statsapi.mlb.com/api/v1/stats"
_MLB_PAGE = 1000


def _fetch_mlb_season(
    group: str, year: int, *, get: Callable[..., Any] | None = None
) -> pd.DataFrame:
    """Paginated season leaderboard for `group` ('hitting'|'pitching'). Accumulates
    the raw splits across every page, then json_normalizes ONCE (consistent columns).
    `get` defaults to `requests.get` (local import keeps the module network-free at
    import time); tests inject a fake."""
    if get is None:
        import requests

        get = requests.get
    splits: list[dict[str, Any]] = []
    offset = 0
    while True:
        params: dict[str, str | int] = {
            "stats": "season",
            "group": group,
            "season": year,
            "sportId": 1,
            "playerPool": "all",
            "limit": _MLB_PAGE,
            "offset": offset,
        }
        resp = get(_MLB_STATS_URL, params=params, timeout=60)
        resp.raise_for_status()
        stats = resp.json().get("stats", [])
        page_splits = stats[0]["splits"] if stats else []
        if not page_splits:
            break
        splits.extend(page_splits)
        if len(page_splits) < _MLB_PAGE:
            break
        offset += _MLB_PAGE
    result: pd.DataFrame = pd.json_normalize(splits)
    return result


def fetch_mlb_season(
    cache_dir: Path,
    year: int,
    group: str,
    *,
    fetcher: Callable[[], pd.DataFrame] | None = None,
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"mlb_{group}_{year}.csv",
        fetcher or (lambda: _fetch_mlb_season(group, year)),
    )
```

Update `src/fantasy_baseball/keepers/__init__.py`:

```python
"""Keepers module: raw MLB Stats API + Baseball Savant data pulls.

Fetchers return the upstream response fully raw -- no derivation, rename, or join.
Downstream keeper-value logic (#266) is built on top of these; nothing here computes.
"""

from __future__ import annotations

from fantasy_baseball.keepers.cache import fetch_or_cache
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season

__all__ = ["fetch_or_cache", "fetch_mlb_season"]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_keepers/test_mlb_stats.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Verify gates**

Run: `ruff check src/fantasy_baseball/keepers/ tests/test_keepers/ && ruff format --check src/fantasy_baseball/keepers/ tests/test_keepers/ && mypy src/fantasy_baseball/keepers/mlb_stats.py`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/keepers/mlb_stats.py src/fantasy_baseball/keepers/__init__.py \
        tests/test_keepers/test_mlb_stats.py
git commit -m "feat(keepers): raw MLB Stats API season pull (#265)"
```

---

### Task 3: `savant.py` -- Baseball Savant pulls (fully raw)

**Files:**
- Create: `src/fantasy_baseball/keepers/savant.py`
- Create: `tests/test_keepers/test_savant.py`
- Modify: `src/fantasy_baseball/keepers/__init__.py` (export the four savant fetchers; finalize `__all__`)

**Interfaces:**
- Consumes: `fetch_or_cache` from Task 1.
- Produces: `fetch_batter_expected`, `fetch_batter_barrels`, `fetch_pitcher_expected`, `fetch_savant_hr`, each `(cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None) -> pd.DataFrame`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_keepers/test_savant.py`:

```python
from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.savant import (
    fetch_batter_barrels,
    fetch_batter_expected,
    fetch_pitcher_expected,
    fetch_savant_hr,
)


def test_batter_expected_raw_passthrough_no_rename(tmp_path: Path):
    raw = pd.DataFrame({"player_id": [1], "est_woba": [0.350], "woba": [0.330]})
    out = fetch_batter_expected(tmp_path, 2024, fetcher=lambda: raw)
    pd.testing.assert_frame_equal(out, raw)  # est_woba NOT renamed to xwoba
    assert (tmp_path / "savant_batter_expected_2024.csv").exists()


def test_batter_barrels_no_unit_conversion(tmp_path: Path):
    raw = pd.DataFrame({"player_id": [1], "brl_percent": [12.0], "brl_pa": [4.9]})
    out = fetch_batter_barrels(tmp_path, 2024, fetcher=lambda: raw)
    assert out["brl_percent"].iloc[0] == 12.0  # NOT divided by 100
    assert out["brl_pa"].iloc[0] == 4.9


def test_pitcher_expected_raw_passthrough(tmp_path: Path):
    raw = pd.DataFrame({"player_id": [1], "est_woba": [0.300]})
    out = fetch_pitcher_expected(tmp_path, 2024, fetcher=lambda: raw)
    pd.testing.assert_frame_equal(out, raw)


def test_savant_hr_tolerates_empty_pre_2016(tmp_path: Path):
    out = fetch_savant_hr(tmp_path, 2015, fetcher=lambda: pd.DataFrame())
    assert out.empty
    assert not (tmp_path / "savant_hr_2015.csv").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_keepers/test_savant.py -v`
Expected: FAIL -- `ModuleNotFoundError: fantasy_baseball.keepers.savant`.

- [ ] **Step 3: Write minimal implementation**

Create `src/fantasy_baseball/keepers/savant.py`:

```python
"""Raw Baseball Savant pulls (expected stats + barrels via pybaseball; park-adjusted
xHR via a direct leaderboard CSV). Returned fully raw -- no rename, no percent->share
conversion, no merge. pybaseball is imported locally (heavy) so the module stays
import-safe.
"""

from __future__ import annotations

import io
import urllib.request
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from fantasy_baseball.keepers.cache import fetch_or_cache

_SAVANT_HR_URL = (
    "https://baseballsavant.mlb.com/leaderboard/home-runs?type=batter&year={year}&min=1&csv=true"
)


def _savant_batter_expected(year: int) -> pd.DataFrame:
    from pybaseball import statcast_batter_expected_stats

    result: pd.DataFrame = statcast_batter_expected_stats(year, minPA=1)
    return result


def _savant_batter_barrels(year: int) -> pd.DataFrame:
    from pybaseball import statcast_batter_exitvelo_barrels

    result: pd.DataFrame = statcast_batter_exitvelo_barrels(year, minBBE=1)
    return result


def _savant_pitcher_expected(year: int) -> pd.DataFrame:
    from pybaseball import statcast_pitcher_expected_stats

    result: pd.DataFrame = statcast_pitcher_expected_stats(year, minPA=1)
    return result


def _savant_hr(year: int) -> pd.DataFrame:
    """Park-adjusted xHR leaderboard CSV (no pybaseball wrapper). Browser UA +
    utf-8-sig BOM. Pre-2016 returns a header-only body (empty frame)."""
    req = urllib.request.Request(
        _SAVANT_HR_URL.format(year=year),
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        body = resp.read().decode("utf-8-sig", "replace")
    result: pd.DataFrame = pd.read_csv(io.StringIO(body))
    return result


def fetch_batter_expected(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_batter_expected_{year}.csv",
        fetcher or (lambda: _savant_batter_expected(year)),
    )


def fetch_batter_barrels(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_batter_barrels_{year}.csv",
        fetcher or (lambda: _savant_batter_barrels(year)),
    )


def fetch_pitcher_expected(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_pitcher_expected_{year}.csv",
        fetcher or (lambda: _savant_pitcher_expected(year)),
    )


def fetch_savant_hr(
    cache_dir: Path, year: int, *, fetcher: Callable[[], pd.DataFrame] | None = None
) -> pd.DataFrame:
    return fetch_or_cache(
        cache_dir / f"savant_hr_{year}.csv",
        fetcher or (lambda: _savant_hr(year)),
        tolerate_empty=True,
    )
```

Finalize `src/fantasy_baseball/keepers/__init__.py`:

```python
"""Keepers module: raw MLB Stats API + Baseball Savant data pulls.

Fetchers return the upstream response fully raw -- no derivation, rename, or join.
Downstream keeper-value logic (#266) is built on top of these; nothing here computes.
"""

from __future__ import annotations

from fantasy_baseball.keepers.cache import fetch_or_cache
from fantasy_baseball.keepers.mlb_stats import fetch_mlb_season
from fantasy_baseball.keepers.savant import (
    fetch_batter_barrels,
    fetch_batter_expected,
    fetch_pitcher_expected,
    fetch_savant_hr,
)

__all__ = [
    "fetch_or_cache",
    "fetch_mlb_season",
    "fetch_batter_expected",
    "fetch_batter_barrels",
    "fetch_pitcher_expected",
    "fetch_savant_hr",
]
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_keepers/test_savant.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Verify gates**

Run: `ruff check src/fantasy_baseball/keepers/ tests/test_keepers/ && ruff format --check src/fantasy_baseball/keepers/ tests/test_keepers/ && mypy src/fantasy_baseball/keepers/`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/keepers/savant.py src/fantasy_baseball/keepers/__init__.py \
        tests/test_keepers/test_savant.py
git commit -m "feat(keepers): raw Baseball Savant pulls (#265)"
```

---

### Task 4: Delete the distrusted analysis, backtests, reports, and old ingest

**Files (all deletions):**
- Delete src: `data/skill_luck.py`, `analysis/breakout.py`, `analysis/hr_confirm.py`, `analysis/breakout_backtest.py`
- Delete scripts: `run_breakout_report.py`, `backtest_breakout.py`, `backtest_coefficient.py`, `backtest_hr_confirm.py`, `backtest_hr_level.py`
- Delete tests: `test_analysis/test_breakout.py`, `test_analysis/test_breakout_backtest.py`, `test_analysis/test_hr_confirm.py`, `test_scripts/test_backtest_breakout.py`, `test_scripts/test_backtest_hr_confirm.py`, `test_scripts/test_backtest_hr_level.py`, `test_data/test_skill_luck.py`
- Delete docs: three specs + three plans (breakout diagnostic, barrel-anchored HR, HR-confirmation)
- Delete artifacts (untracked): `data/stats/{breakout_backtest,hr_confirm_backtest,hr_level_backtest}_results.csv`, and the stale `data/skill_luck/` cache dir if present

**Interfaces:** none produced. This task removes code; nothing outside the delete set imports it (grep-verified in the spec).

- [ ] **Step 1: Pre-check -- confirm no surviving importer**

Run:
```bash
grep -rlnE "analysis\.(breakout|hr_confirm|breakout_backtest)|data\.skill_luck|import (breakout|hr_confirm|skill_luck)" src/ scripts/ tests/ \
  | grep -vE "test_breakout|test_hr_confirm|test_skill_luck|test_backtest_(breakout|hr_confirm|hr_level)|/breakout|/hr_confirm|/skill_luck|breakout_backtest"
```
Expected: no output (every remaining reference is itself a file being deleted). If any OTHER file appears, STOP and reconcile before deleting.

- [ ] **Step 2: Delete tracked files**

```bash
git rm src/fantasy_baseball/data/skill_luck.py \
       src/fantasy_baseball/analysis/breakout.py \
       src/fantasy_baseball/analysis/hr_confirm.py \
       src/fantasy_baseball/analysis/breakout_backtest.py \
       scripts/run_breakout_report.py \
       scripts/backtest_breakout.py \
       scripts/backtest_coefficient.py \
       scripts/backtest_hr_confirm.py \
       scripts/backtest_hr_level.py \
       tests/test_analysis/test_breakout.py \
       tests/test_analysis/test_breakout_backtest.py \
       tests/test_analysis/test_hr_confirm.py \
       tests/test_scripts/test_backtest_breakout.py \
       tests/test_scripts/test_backtest_hr_confirm.py \
       tests/test_scripts/test_backtest_hr_level.py \
       tests/test_data/test_skill_luck.py \
       docs/superpowers/specs/2026-07-24-keeper-breakout-diagnostic-design.md \
       docs/superpowers/specs/2026-07-26-barrel-anchored-hr-design.md \
       docs/superpowers/specs/2026-07-26-hr-confirmation-backtest-design.md \
       docs/superpowers/plans/2026-07-24-keeper-breakout-diagnostic.md \
       docs/superpowers/plans/2026-07-26-barrel-anchored-hr.md \
       docs/superpowers/plans/2026-07-26-hr-confirmation-backtest.md
```

- [ ] **Step 3: Delete untracked artifacts**

```bash
rm -f data/stats/breakout_backtest_results.csv \
      data/stats/hr_confirm_backtest_results.csv \
      data/stats/hr_level_backtest_results.csv
rm -rf data/skill_luck
```

- [ ] **Step 4: Verify the suite is green (deletions removed their own tests)**

Run: `pytest -q`
Expected: PASS (no collection errors; the keeper_value / keeper_trades tests still pass, unaffected).

- [ ] **Step 5: Verify ruff on the whole tree**

Run: `ruff check . && ruff format --check .`
Expected: clean.

- [ ] **Step 6: Commit**

```bash
git commit -m "chore(keepers): delete distrusted breakout/HR analysis + backtests (#265)"
```

---

### Task 5: Full end-of-effort verification gate

**Files:** none (verification only; fix + amend if a gate fails).

- [ ] **Step 1: Full test suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 2: Lint + format**

Run: `ruff check . && ruff format --check .`
Expected: clean (run `ruff format .` and re-commit if drift).

- [ ] **Step 3: mypy**

Run: `mypy`
Expected: clean. (The new `keepers/` package is in `[tool.mypy] files`; deleted `analysis/` modules no longer checked.)

- [ ] **Step 4: vulture -- confirm the new fetchers are NOT reported and no new dead code**

Run: `vulture`
Expected: no NEW findings. The five public fetchers are suppressed via `[tool.vulture] ignore_names` + `__all__`; the private `_fetch_*`/`_savant_*`/`_read_cached` helpers are used by their public wrappers. If a fetcher IS reported, confirm its name is in `ignore_names` and present in `__all__`. Pre-existing unrelated findings are acceptable -- note them.

- [ ] **Step 5: Confirm acceptance criteria (spec)**

Run:
```bash
test -f src/fantasy_baseball/keepers/mlb_stats.py && test -f src/fantasy_baseball/keepers/savant.py && echo "keepers module: OK"
test ! -f src/fantasy_baseball/analysis/breakout.py && test ! -f src/fantasy_baseball/data/skill_luck.py && echo "old analysis removed: OK"
```
Expected: both "OK" lines print.

- [ ] **Step 6: Commit any fixes**

If Steps 1-4 required edits, commit them:
```bash
git commit -am "chore(keepers): end-of-effort gate fixes (#265)"
```
Otherwise no commit needed -- the branch is ready.
