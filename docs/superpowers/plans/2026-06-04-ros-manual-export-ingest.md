# ROS Manual-Export Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A guided local CLI that ingests the human's FanGraphs one-click member exports (5 systems x hitters/pitchers), stages them into today's snapshot dir, and pushes the blended result to prod Upstash — with no automated extraction from FanGraphs.

**Architecture:** Pure, testable logic (find newest export, validate type, stage, guided loop with injected I/O) lives in a library module `src/fantasy_baseball/data/ros_export_ingest.py`. The thin CLI plus the prod push (RENDER-flip -> `blend_and_cache_ros` -> verify, the proven 2026-06-04 restore tail) lives in `scripts/ingest_ros_export.py`. No new dependencies.

**Tech Stack:** Python 3.12, pandas (existing), pytest. Reuses `data/fangraphs.py` (`parse_hitting_csv`/`parse_pitching_csv`), `data/ros_pipeline.py` (`blend_and_cache_ros`), `data/kv_store.py` (RENDER-flip pattern from `scripts/refresh_remote.py`), `config`.

---

## File structure

- Create: `src/fantasy_baseball/data/ros_export_ingest.py` — pure ingest logic (steps, find/validate/stage, guided loop, result + push-gate).
- Create: `tests/test_data/test_ros_export_ingest.py` — unit tests for the above.
- Create: `scripts/ingest_ros_export.py` — CLI wiring + RENDER-flip prod push + verify.
- Modify: `pyproject.toml` — add the new data module to `[tool.mypy].files`.

Reused fixtures: `tests/fixtures/steamer_hitters.csv`, `tests/fixtures/steamer_pitchers.csv` (already used by `tests/test_data/test_ros_pipeline.py`; they parse via `parse_hitting_csv`/`parse_pitching_csv`).

---

### Task 1: Export-step ordering + newest-CSV finder

**Files:**
- Create: `src/fantasy_baseball/data/ros_export_ingest.py`
- Test: `tests/test_data/test_ros_export_ingest.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_data/test_ros_export_ingest.py
import os

from fantasy_baseball.data.ros_export_ingest import export_steps, find_newest_csv


def test_export_steps_orders_each_system_hitters_then_pitchers():
    steps = export_steps(["steamer", "zips"])
    assert steps == [
        ("steamer", "hitters"),
        ("steamer", "pitchers"),
        ("zips", "hitters"),
        ("zips", "pitchers"),
    ]


def test_find_newest_csv_returns_none_when_no_file_newer_than_since(tmp_path):
    old = tmp_path / "old.csv"
    old.write_text("x\n")
    os.utime(old, (1000.0, 1000.0))  # mtime = 1000
    assert find_newest_csv(tmp_path, since_ts=2000.0) is None


def test_find_newest_csv_picks_most_recent_at_or_after_since(tmp_path):
    a = tmp_path / "a.csv"
    b = tmp_path / "b.csv"
    a.write_text("x\n")
    b.write_text("y\n")
    os.utime(a, (3000.0, 3000.0))
    os.utime(b, (3001.0, 3001.0))
    # a non-csv newer file must be ignored
    other = tmp_path / "note.txt"
    other.write_text("z\n")
    os.utime(other, (9999.0, 9999.0))
    assert find_newest_csv(tmp_path, since_ts=2999.0) == b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_ros_export_ingest.py -q`
Expected: FAIL with `ImportError: cannot import name 'export_steps'`.

- [ ] **Step 3: Write the minimal implementation**

```python
# src/fantasy_baseball/data/ros_export_ingest.py
"""Ingest FanGraphs one-click member exports into a dated ROS snapshot dir.

FanGraphs prohibits automated extraction (scraping/API/web query); the supported
path is a human's one-click member export. This module only handles the CSV
files the user has already exported -- it never contacts FanGraphs. The guided
loop walks the user through the 5 systems x {hitters, pitchers}, stages each
freshly-downloaded + type-validated CSV under our naming convention, and reports
which systems are complete so the caller can blend + push to prod.
"""

from __future__ import annotations

from pathlib import Path

PLAYER_TYPES: tuple[str, str] = ("hitters", "pitchers")


def export_steps(systems: list[str]) -> list[tuple[str, str]]:
    """(system, player_type) pairs in prompt order: each system hitters then pitchers."""
    return [(system, ptype) for system in systems for ptype in PLAYER_TYPES]


def find_newest_csv(source_dir: Path, since_ts: float) -> Path | None:
    """Newest ``*.csv`` in ``source_dir`` with mtime >= ``since_ts``; ``None`` if none.

    ``since_ts`` is captured just before prompting the user, so this picks the file
    they just exported rather than a stale prior download.
    """
    candidates = [p for p in Path(source_dir).glob("*.csv") if p.stat().st_mtime >= since_ts]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_ros_export_ingest.py -q`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/ros_export_ingest.py tests/test_data/test_ros_export_ingest.py
git commit -m "feat(ros-ingest): export-step ordering + newest-CSV finder"
```

---

### Task 2: Type validation + stage helper

**Files:**
- Modify: `src/fantasy_baseball/data/ros_export_ingest.py`
- Test: `tests/test_data/test_ros_export_ingest.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_data/test_ros_export_ingest.py
import shutil
from pathlib import Path

from fantasy_baseball.data.ros_export_ingest import stage_export, validate_export_type

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def _copy_fixture(name: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(FIXTURES_DIR / name, dest)
    return dest


def test_validate_export_type_accepts_matching_type(tmp_path):
    h = _copy_fixture("steamer_hitters.csv", tmp_path / "h.csv")
    p = _copy_fixture("steamer_pitchers.csv", tmp_path / "p.csv")
    assert validate_export_type(h, "hitters") is True
    assert validate_export_type(p, "pitchers") is True


def test_validate_export_type_rejects_wrong_type(tmp_path):
    h = _copy_fixture("steamer_hitters.csv", tmp_path / "h.csv")
    # a hitters export lacks pitcher required cols (ip/w/sv/...) -> not a pitchers file
    assert validate_export_type(h, "pitchers") is False


def test_stage_export_stages_newest_valid_file(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    _copy_fixture("steamer_hitters.csv", source / "FanGraphs Leaderboard.csv")
    import os

    os.utime(source / "FanGraphs Leaderboard.csv", (5000.0, 5000.0))
    dest_dir = tmp_path / "snap"
    staged = stage_export(source, 4000.0, "steamer", "hitters", dest_dir)
    assert staged == dest_dir / "steamer-hitters.csv"
    assert staged.exists()


def test_stage_export_returns_none_for_wrong_type(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    f = _copy_fixture("steamer_hitters.csv", source / "x.csv")
    import os

    os.utime(f, (5000.0, 5000.0))
    dest_dir = tmp_path / "snap"
    # asking for pitchers but the only new file is a hitters export -> None, nothing staged
    assert stage_export(source, 4000.0, "steamer", "pitchers", dest_dir) is None
    assert not (dest_dir / "steamer-pitchers.csv").exists()


def test_stage_export_returns_none_when_no_new_file(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    f = _copy_fixture("steamer_hitters.csv", source / "old.csv")
    import os

    os.utime(f, (1000.0, 1000.0))
    assert stage_export(source, 4000.0, "steamer", "hitters", tmp_path / "snap") is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_ros_export_ingest.py -q`
Expected: FAIL with `ImportError: cannot import name 'stage_export'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `src/fantasy_baseball/data/ros_export_ingest.py` (after `find_newest_csv`):

```python
import shutil

from fantasy_baseball.data.fangraphs import parse_hitting_csv, parse_pitching_csv


def validate_export_type(path: Path, player_type: str) -> bool:
    """True if ``path`` parses as a FanGraphs export of ``player_type``.

    Reuses the production parsers, which raise ``ValueError`` when the required
    hitter/pitcher columns are absent -- so a "wrong page" export (e.g. a hitters
    CSV offered as pitchers) is rejected.
    """
    try:
        if player_type == "hitters":
            parse_hitting_csv(path)
        else:
            parse_pitching_csv(path)
    except Exception:
        return False
    return True


def stage_export(
    source_dir: Path, since_ts: float, system: str, player_type: str, dest_dir: Path
) -> Path | None:
    """Stage the newest valid export for ``(system, player_type)`` into ``dest_dir``.

    Returns the staged path ``dest_dir/{system}-{player_type}.csv``, or ``None``
    when no ``*.csv`` newer than ``since_ts`` exists or the newest one fails type
    validation (caller re-prompts; nothing is staged on ``None``).
    """
    src = find_newest_csv(Path(source_dir), since_ts)
    if src is None or not validate_export_type(src, player_type):
        return None
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{system}-{player_type}.csv"
    shutil.copy(src, dest)
    return dest
```

Move the `import shutil` and the `from fantasy_baseball.data.fangraphs import ...` to the module's top import block (next to `from pathlib import Path`) so imports are not mid-file.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_ros_export_ingest.py -q`
Expected: PASS (8 passed).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/ros_export_ingest.py tests/test_data/test_ros_export_ingest.py
git commit -m "feat(ros-ingest): type validation + stage-newest-export helper"
```

---

### Task 3: Guided loop + push-gate result

**Files:**
- Modify: `src/fantasy_baseball/data/ros_export_ingest.py`
- Test: `tests/test_data/test_ros_export_ingest.py`

- [ ] **Step 1: Write the failing tests**

```python
# add to tests/test_data/test_ros_export_ingest.py
from fantasy_baseball.data.ros_export_ingest import IngestResult, run_guided_ingest


def _seed(source: Path, name: str, fixture: str, ts: float) -> None:
    import os

    _copy_fixture(fixture, source / name)
    os.utime(source / name, (ts, ts))


def test_run_guided_ingest_stages_all_steps_and_reports_complete(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    dest = tmp_path / "snap"
    systems = ["steamer"]
    # now_fn yields an increasing clock; prompt_fn drops a matching file then "presses Enter".
    clock = {"t": 100.0}

    def now_fn():
        clock["t"] += 10.0
        return clock["t"]

    pending = iter([("steamer-h.csv", "steamer_hitters.csv"), ("steamer-p.csv", "steamer_pitchers.csv")])

    def prompt_fn(_msg):
        name, fixture = next(pending)
        _seed(source, name, fixture, clock["t"] + 1.0)  # file newer than the just-captured since_ts
        return ""  # Enter

    outputs: list[str] = []
    result = run_guided_ingest(
        systems, source, dest, prompt_fn=prompt_fn, output_fn=outputs.append, now_fn=now_fn
    )
    assert result.aborted is False
    assert result.complete_systems(systems) == ["steamer"]
    assert (dest / "steamer-hitters.csv").exists()
    assert (dest / "steamer-pitchers.csv").exists()


def test_run_guided_ingest_skip_excludes_system(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    result = run_guided_ingest(
        ["steamer"],
        source,
        tmp_path / "snap",
        prompt_fn=lambda _m: "s",
        output_fn=lambda _m: None,
        now_fn=lambda: 1.0,
    )
    assert result.complete_systems(["steamer"]) == []
    assert "steamer" in result.skipped_systems


def test_run_guided_ingest_abort_stops_immediately(tmp_path):
    result = run_guided_ingest(
        ["steamer", "zips"],
        tmp_path,
        tmp_path / "snap",
        prompt_fn=lambda _m: "q",
        output_fn=lambda _m: None,
        now_fn=lambda: 1.0,
    )
    assert result.aborted is True
    assert result.staged == {}


def test_run_guided_ingest_reprompts_on_missing_file(tmp_path):
    source = tmp_path / "dl"
    source.mkdir()
    dest = tmp_path / "snap"
    calls = {"n": 0}

    def prompt_fn(_msg):
        calls["n"] += 1
        if calls["n"] == 1:
            return ""  # hitters: no file yet -> stage_export None -> re-prompt
        if calls["n"] == 2:
            _seed(source, "h.csv", "steamer_hitters.csv", 1000.0)
            return ""  # hitters: now stages
        return "q"  # pitchers step: abort to end the run (avoids an infinite re-prompt)

    result = run_guided_ingest(
        ["steamer"],
        source,
        dest,
        prompt_fn=prompt_fn,
        output_fn=lambda _m: None,
        now_fn=lambda: 1.0,  # since_ts=1.0 so the ts=1000 file qualifies
    )
    assert calls["n"] >= 3  # 1 miss + 1 stage (hitters) + 1 abort (pitchers)
    assert ("steamer", "hitters") in result.staged
    assert result.aborted is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_data/test_ros_export_ingest.py -q`
Expected: FAIL with `ImportError: cannot import name 'IngestResult'`.

- [ ] **Step 3: Write the minimal implementation**

Add to `src/fantasy_baseball/data/ros_export_ingest.py`:

```python
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class IngestResult:
    """Outcome of a guided ingest run."""

    staged: dict[tuple[str, str], Path] = field(default_factory=dict)
    skipped_systems: set[str] = field(default_factory=set)
    aborted: bool = False

    def complete_systems(self, systems: list[str]) -> list[str]:
        """Systems with BOTH hitters and pitchers staged (push-eligible)."""
        return [
            s
            for s in systems
            if (s, "hitters") in self.staged and (s, "pitchers") in self.staged
        ]


def run_guided_ingest(
    systems: list[str],
    source_dir: Path,
    dest_dir: Path,
    *,
    prompt_fn: Callable[[str], str],
    output_fn: Callable[[str], None],
    now_fn: Callable[[], float],
) -> IngestResult:
    """Walk the user through exporting each (system, player_type) and stage each.

    ``prompt_fn(message) -> response`` returns the user's reply: ``""`` (Enter =
    "I exported it"), ``"s"`` (skip this system), or ``"q"`` (abort). ``output_fn``
    prints progress. ``now_fn`` supplies the timestamp captured just before each
    prompt so :func:`stage_export` can pick the just-exported file. I/O is injected
    so the loop is unit-testable without real stdin/clock.
    """
    result = IngestResult()
    for system, ptype in export_steps(systems):
        if system in result.skipped_systems:
            continue
        while True:
            since = now_fn()
            resp = prompt_fn(
                f"Export {system} {ptype} from FanGraphs, then press Enter "
                f"(s=skip system, q=abort): "
            ).strip().lower()
            if resp == "q":
                result.aborted = True
                return result
            if resp == "s":
                result.skipped_systems.add(system)
                output_fn(f"  skipped {system}")
                break
            staged = stage_export(source_dir, since, system, ptype, dest_dir)
            if staged is None:
                output_fn(
                    f"  no new valid {ptype} export found in {source_dir} -- "
                    f"export it and press Enter again"
                )
                continue
            result.staged[(system, ptype)] = staged
            output_fn(f"  staged {staged.name}")
            break
    return result
```

Move `from collections.abc import Callable` and `from dataclasses import dataclass, field` to the top import block.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_data/test_ros_export_ingest.py -q`
Expected: PASS (12 passed).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/ros_export_ingest.py tests/test_data/test_ros_export_ingest.py
git commit -m "feat(ros-ingest): guided ingest loop + push-gate result"
```

---

### Task 4: CLI script with prod push

**Files:**
- Create: `scripts/ingest_ros_export.py`
- (Manual verification — the prod push is not unit-tested; it reuses the tested `blend_and_cache_ros` and the proven RENDER-flip pattern.)

- [ ] **Step 1: Write the script**

```python
# scripts/ingest_ros_export.py
#!/usr/bin/env python3
"""Guided ingest of FanGraphs one-click member exports -> prod Upstash ROS.

FanGraphs supports only one-click member exports (no scraping/API/web query).
This walks you through exporting the 5 systems x {hitters, pitchers} in your
browser, stages each freshly-downloaded CSV into today's snapshot dir, then
blends and pushes to prod (the same RENDER-flip tail as the manual restore).

Usage:
    python scripts/ingest_ros_export.py                 # ~/Downloads, push to prod
    python scripts/ingest_ros_export.py --source D:/dl  # custom download dir
    python scripts/ingest_ros_export.py --no-push       # stage only (dry run)
"""

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _push_to_prod(season_year: int) -> None:
    """Blend today's staged snapshot and write it to prod Upstash, then verify.

    Mirrors scripts/refresh_remote.py: flip RENDER on BEFORE importing the
    pipeline so get_kv() resolves to Upstash.
    """
    import json

    os.environ["RENDER"] = "true"

    from fantasy_baseball.config import load_config
    from fantasy_baseball.data import kv_store
    from fantasy_baseball.data.kv_store import build_explicit_upstash_kv, get_kv
    from fantasy_baseball.data.redis_store import get_latest_roster_names
    from fantasy_baseball.data.ros_pipeline import blend_and_cache_ros

    kv_store._reset_singleton()
    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    projections_dir = PROJECT_ROOT / "data" / "projections"
    roster_names = get_latest_roster_names(get_kv())

    print("Blending staged snapshot -> prod Upstash...")
    ros_h, ros_p = blend_and_cache_ros(
        projections_dir,
        config.projection_systems,
        config.projection_weights,
        roster_names,
        season_year,
        progress_cb=lambda m: print(f"  {m}") if not m.startswith("QUALITY") else None,
    )
    print(f"Persisted {len(ros_h)} ROS hitters + {len(ros_p)} ROS pitchers to prod")

    remote = build_explicit_upstash_kv()
    for key in ("cache:ros_projections", "cache:full_season_projections"):
        obj = json.loads(remote.get(key))
        meta = obj.get("_meta", {})
        data = obj.get("_data", obj)
        print(
            f"{key}: snapshot={meta.get('_ros_snapshot_date')} "
            f"hitters={len(data.get('hitters', []))} pitchers={len(data.get('pitchers', []))}"
        )


def main() -> int:
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.ros_export_ingest import run_guided_ingest
    from fantasy_baseball.utils.time_utils import local_today

    parser = argparse.ArgumentParser(description="Ingest FanGraphs one-click exports to prod ROS.")
    parser.add_argument("--source", default=str(Path.home() / "Downloads"), help="download dir")
    parser.add_argument("--season", type=int, default=None, help="season year (default: config)")
    parser.add_argument("--no-push", action="store_true", help="stage only; do not push to prod")
    args = parser.parse_args()

    config = load_config(PROJECT_ROOT / "config" / "league.yaml")
    season = args.season or config.season_year
    dest_dir = (
        PROJECT_ROOT / "data" / "projections" / str(season) / "rest_of_season" / local_today().isoformat()
    )
    source_dir = Path(args.source)

    print(f"Exports source: {source_dir}")
    print(f"Staging into:   {dest_dir}\n")
    result = run_guided_ingest(
        config.projection_systems,
        source_dir,
        dest_dir,
        prompt_fn=input,
        output_fn=print,
        now_fn=time.time,
    )

    if result.aborted:
        print("\nAborted -- nothing pushed; last-good prod ROS unchanged.")
        return 1
    complete = result.complete_systems(config.projection_systems)
    if not complete:
        print("\nNo complete systems staged -- not pushing; last-good prod ROS unchanged.")
        return 1
    print(f"\nComplete systems: {', '.join(complete)}")
    if result.skipped_systems:
        print(f"Skipped: {', '.join(sorted(result.skipped_systems))}")
    if args.no_push:
        print("--no-push set; staged only, prod unchanged.")
        return 0

    _push_to_prod(season)
    print("\nDone. (Run scripts/refresh_remote.py to propagate into dashboard standings.)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Verify it parses and the help works**

Run: `python scripts/ingest_ros_export.py --help`
Expected: argparse usage text prints, exit 0 (no prompts, no prod contact).

- [ ] **Step 3: Dry-run staging smoke test (no prod)**

Manually copy a real or fixture `steamer_hitters.csv` into `~/Downloads` as any `.csv`, then:

Run: `python scripts/ingest_ros_export.py --no-push`
At the `steamer hitters` prompt, press Enter; confirm it prints `staged steamer-hitters.csv` (or a "no new valid export" retry if nothing fresh). Press `q` to abort once satisfied.
Expected: staging works against `~/Downloads`; `--no-push` exits without touching prod.

- [ ] **Step 4: Commit**

```bash
git add scripts/ingest_ros_export.py
git commit -m "feat(ros-ingest): guided CLI + RENDER-flip prod push"
```

---

### Task 5: mypy coverage + full verification

**Files:**
- Modify: `pyproject.toml` (`[tool.mypy].files`)

- [ ] **Step 1: Add the new module to mypy coverage**

In `pyproject.toml`, under `[tool.mypy]` `files = [...]`, add the line (keeping alphabetical grouping near the other `data/` entries):

```toml
    "src/fantasy_baseball/data/ros_export_ingest.py",
```

- [ ] **Step 2: Run mypy**

Run: `python -m mypy`
Expected: no NEW errors in `ros_export_ingest.py` (a pre-existing error in `streaks/data/statcast.py` is unrelated and acceptable).

- [ ] **Step 3: Run the full verification gate**

Run:
```bash
python -m pytest tests/test_data/test_ros_export_ingest.py -q
python -m ruff check src/fantasy_baseball/data/ros_export_ingest.py scripts/ingest_ros_export.py tests/test_data/test_ros_export_ingest.py
python -m ruff format --check src/fantasy_baseball/data/ros_export_ingest.py scripts/ingest_ros_export.py tests/test_data/test_ros_export_ingest.py
python -m vulture src/fantasy_baseball/data/ros_export_ingest.py
```
Expected: tests pass; ruff clean; format clean; vulture reports nothing new.

- [ ] **Step 4: Run the relevant suite subset**

Run: `python -m pytest tests/test_data/ -q`
Expected: PASS (existing data tests + the new ingest tests).

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml
git commit -m "chore(ros-ingest): add ros_export_ingest to mypy coverage"
```

---

## Self-review notes

- **Spec coverage:** guided 10-step flow (Task 3 + 4), newest+type-validated staging (Tasks 1-2), RENDER-flip blend+push+verify (Task 4), success gate `>=1 complete system` (Task 3 `complete_systems` + Task 4 `main`), `--source`/`--no-push` flags (Task 4), no new deps (none added), unit tests for stage + loop + gate (Tasks 1-3). All spec sections map to a task.
- **No automated FanGraphs access:** the library never imports a network client; it only reads local files. The script's only network is the prod-Upstash push.
- **Types consistent:** `stage_export` returns `Path | None` everywhere; `run_guided_ingest` returns `IngestResult`; `complete_systems(list[str]) -> list[str]` used identically in Task 3 tests and Task 4 `main`.
- **Manual-only step:** the prod push (Task 4) is verified manually, not in CI, because it writes real prod Upstash and depends on a live export; its building blocks (`blend_and_cache_ros`, RENDER-flip) are already covered by existing tests and the 2026-06-04 restore.
