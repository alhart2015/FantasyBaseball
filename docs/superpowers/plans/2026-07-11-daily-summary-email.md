# Daily Summary Email Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An automated morning email summarizing overnight team state (last night's results, hot/cold streaks, standings movement, lineup recommendations, injuries, upcoming probables), sent via Resend on a Render cron.

**Architecture:** A new `src/fantasy_baseball/summary/` package with three concerns -- `assemble` (KV + live Yahoo -> typed `DailySummary`), `render` (`DailySummary` -> HTML/text), `send` (Resend). A thin `scripts/send_daily_summary.py` orchestrates them behind an up-front cache-freshness gate. The job reads the Upstash KV that the morning refresh already populated; it is decoupled from the refresh pipeline.

**Tech Stack:** Python 3.11, `resend` SDK (new dep), existing `upstash-redis` KV, `yahoo-fantasy-api` for live injuries, `pandas` for crosswalk CSV reads.

**Spec:** `docs/superpowers/specs/2026-07-11-daily-summary-email-design.md`

## Global Constraints

- **Python `>=3.11`.** Copy verbatim from `pyproject.toml`.
- **ASCII-only** in all source, log messages, and any string that may hit `print()` (Windows cp1252). The HTML email body is UTF-8; the entry-point script must `sys.stdout.reconfigure(encoding="utf-8", errors="replace")` before printing player names.
- **Player IDs are `name::player_type`**; never key on bare names. The crosswalk is keyed by `(normalized_name, player_type)`.
- **No `x or default` for numeric fields** (`0`/`0.0` are falsy). Use `d["k"] if d.get("k") is not None else default`.
- **Read Upstash, not local SQLite, for live state.** The orchestrator sets `os.environ["RENDER"] = "true"` before the first KV import (mirrors `scripts/refresh_remote.py:33`).
- **mypy is enforced** on covered packages. Every new file under `src/fantasy_baseball/summary/` must be added to `[tool.mypy].files` in `pyproject.toml` and pass `mypy`.
- **End-of-effort verification:** `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, `mypy` all clean.
- **ruff:** line-length 100, double-quote format, selects `F,E,W,I,UP,B,ARG,SIM,RUF`.

---

### Task 1: Foundation -- dependency, cache key, config block, mypy coverage

**Files:**
- Modify: `pyproject.toml` (add `resend` dep; add summary module to `[tool.mypy].files`)
- Modify: `src/fantasy_baseball/data/cache_keys.py` (add `STANDINGS_SNAPSHOT`)
- Modify: `src/fantasy_baseball/config.py` (add `summary` block to `LeagueConfig` + `load_config`)
- Modify: `config/league.yaml` and `config/league.yaml.example` (add `summary:` block)
- Test: `tests/test_config.py` (extend), `tests/test_data/test_cache_keys.py` (create if absent)

**Interfaces:**
- Produces: `CacheKey.STANDINGS_SNAPSHOT` (stringifies to `"standings_snapshot"`); `LeagueConfig.summary: dict` (keys: `recipients: list[str]`, `from_address: str`).

- [ ] **Step 1: Write the failing test for the new cache key**

Add to `tests/test_data/test_cache_keys.py` (create the file if it does not exist):

```python
from fantasy_baseball.data.cache_keys import CacheKey, redis_key


def test_standings_snapshot_key():
    assert CacheKey.STANDINGS_SNAPSHOT == "standings_snapshot"
    assert redis_key(CacheKey.STANDINGS_SNAPSHOT) == "cache:standings_snapshot"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_data/test_cache_keys.py::test_standings_snapshot_key -v`
Expected: FAIL with `AttributeError: STANDINGS_SNAPSHOT`

- [ ] **Step 3: Add the enum member**

In `src/fantasy_baseball/data/cache_keys.py`, add inside the `CacheKey` class body (after `DRAFT_VALUE`):

```python
    STANDINGS_SNAPSHOT = "standings_snapshot"
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_data/test_cache_keys.py::test_standings_snapshot_key -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for the config summary block**

Add to `tests/test_config.py`:

```python
def test_load_config_reads_summary_block(tmp_path):
    from fantasy_baseball.config import load_config

    cfg_text = """
league:
  id: 12345
  num_teams: 10
  team_name: "My Team"
summary:
  recipients:
    - "me@example.com"
  from_address: "digest@example.com"
"""
    p = tmp_path / "league.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    cfg = load_config(p)
    assert cfg.summary["recipients"] == ["me@example.com"]
    assert cfg.summary["from_address"] == "digest@example.com"


def test_load_config_summary_defaults_empty(tmp_path):
    from fantasy_baseball.config import load_config

    p = tmp_path / "league.yaml"
    p.write_text("league:\n  id: 1\n  num_teams: 10\n  team_name: T\n", encoding="utf-8")
    cfg = load_config(p)
    assert cfg.summary == {}
```

- [ ] **Step 6: Run it to verify it fails**

Run: `pytest tests/test_config.py::test_load_config_reads_summary_block -v`
Expected: FAIL with `TypeError` (unexpected `summary`) or `AttributeError`.

- [ ] **Step 7: Add the `summary` field to `LeagueConfig` and `load_config`**

In `src/fantasy_baseball/config.py`, add to the `LeagueConfig` dataclass (with the other `field(default_factory=...)` members):

```python
    summary: dict = field(default_factory=dict)
```

In `load_config`, after the existing block reads, add:

```python
    summary = raw.get("summary", {})
```

and pass `summary=summary` into the `LeagueConfig(...)` constructor call.

- [ ] **Step 8: Run both config tests to verify they pass**

Run: `pytest tests/test_config.py -k summary -v`
Expected: PASS (both)

- [ ] **Step 9: Add the config blocks to the real config files**

In `config/league.yaml.example` (and `config/league.yaml`), add a top-level block:

```yaml
summary:
  recipients:
    - "you@example.com"
  from_address: "digest@yourdomain.com"
```

- [ ] **Step 10: Add the dependency and mypy coverage**

In `pyproject.toml`, append to `[project].dependencies`:

```toml
    "resend>=2.0",
```

and add these entries to `[tool.mypy].files` (comma-separated, matching existing style):

```
src/fantasy_baseball/summary/models.py,
src/fantasy_baseball/summary/crosswalk.py,
src/fantasy_baseball/summary/builders.py,
src/fantasy_baseball/summary/assemble.py,
src/fantasy_baseball/summary/render.py,
src/fantasy_baseball/summary/send.py,
```

- [ ] **Step 11: Install the new dependency**

Run: `pip install -e ".[dev]"`
Expected: installs `resend` with no error.

- [ ] **Step 12: Commit**

```bash
git add pyproject.toml src/fantasy_baseball/data/cache_keys.py src/fantasy_baseball/config.py config/league.yaml.example config/league.yaml tests/test_config.py tests/test_data/test_cache_keys.py
git commit -m "feat(summary): foundation - resend dep, STANDINGS_SNAPSHOT key, summary config (#200)"
```

---

### Task 2: Data model (`summary/models.py`)

**Files:**
- Create: `src/fantasy_baseball/summary/__init__.py` (empty)
- Create: `src/fantasy_baseball/summary/models.py`
- Test: `tests/test_summary/test_models.py`

**Interfaces:**
- Produces: `DailySummary` and the per-section dataclasses below. All frozen. These types are consumed by every later task.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary/__init__.py` (empty) and `tests/test_summary/test_models.py`:

```python
from datetime import date

from fantasy_baseball.summary.models import (
    DailySummary,
    InjuryItem,
    LineupMove,
    PlayerLine,
    ProbableMatchup,
    StandingsDelta,
    StreakItem,
    TeamDelta,
)


def test_daily_summary_is_constructible_and_frozen():
    summary = DailySummary(
        as_of=date(2026, 7, 10),
        last_night=[PlayerLine(name="Aaron Judge", group="hitting", stats={"h": 2, "hr": 1})],
        unmatched=["Nobody Matched"],
        streaks=[StreakItem(name="Judge", category="hr", label="hot", probability=0.71)],
        standings_delta=StandingsDelta(
            is_first_run=False,
            user_team_name="My Team",
            teams=[TeamDelta(
                name="My Team", rank_prev=3, rank_now=2,
                points_prev=52.0, points_now=54.5,
                category_points_delta={"HR": 1.0, "SB": -1.0},
            )],
            rate_cat_caveat=True,
        ),
        lineup_moves=[LineupMove(player="X", action="start", from_slot="BN", to_slot="OF", roto_delta=0.3)],
        injuries=[InjuryItem(name="Y", status="IL15", note="hamstring")],
        probables=[ProbableMatchup(pitcher="Z", starts=2, days="Mon, Sat", opponents="@ BAL, vs TOR", quality="Great")],
        section_errors=["build_streaks"],
    )
    assert summary.as_of == date(2026, 7, 10)
    assert summary.last_night[0].stats["hr"] == 1
    try:
        summary.as_of = date(2026, 1, 1)  # type: ignore[misc]
        raise AssertionError("expected frozen dataclass")
    except AttributeError:
        pass
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_summary/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: fantasy_baseball.summary.models`

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/summary/__init__.py` (empty file). Create `src/fantasy_baseball/summary/models.py`:

```python
"""Typed payload for the daily summary email.

One frozen dataclass per section so ``assemble`` and ``render`` never pass raw
dicts around. A section with no data is an empty list / sentinel; ``render``
omits it (except the first-run standings baseline, which renders a message).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class PlayerLine:
    """One rostered player's box-score line from last night."""

    name: str
    group: str  # "hitting" or "pitching"
    stats: dict[str, float]  # hitting: h/hr/r/rbi/sb/ab; pitching: ip/k/er/bb/w/sv/h_allowed


@dataclass(frozen=True)
class StreakItem:
    name: str
    category: str  # e.g. "hr", "sb", "avg"
    label: str  # "hot" or "cold"
    probability: float


@dataclass(frozen=True)
class TeamDelta:
    name: str
    rank_prev: int
    rank_now: int
    points_prev: float
    points_now: float
    category_points_delta: dict[str, float]


@dataclass(frozen=True)
class StandingsDelta:
    is_first_run: bool
    user_team_name: str
    teams: list[TeamDelta] = field(default_factory=list)
    rate_cat_caveat: bool = False


@dataclass(frozen=True)
class LineupMove:
    player: str
    action: str  # "start", "sit", "swap"
    from_slot: str
    to_slot: str
    roto_delta: float


@dataclass(frozen=True)
class InjuryItem:
    name: str
    status: str  # IL15 / IL60 / DTD / ...
    note: str


@dataclass(frozen=True)
class ProbableMatchup:
    pitcher: str
    starts: int
    days: str
    opponents: str
    quality: str  # "Great" / "Fair" / "Tough"


@dataclass(frozen=True)
class DailySummary:
    as_of: date
    last_night: list[PlayerLine]
    unmatched: list[str]
    streaks: list[StreakItem]
    standings_delta: StandingsDelta
    lineup_moves: list[LineupMove]
    injuries: list[InjuryItem]
    probables: list[ProbableMatchup]
    section_errors: list[str]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_summary/test_models.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/summary/__init__.py src/fantasy_baseball/summary/models.py tests/test_summary/
git commit -m "feat(summary): DailySummary data model (#200)"
```

---

### Task 3: Type-keyed name->MLBAM crosswalk (`summary/crosswalk.py`)

**Files:**
- Create: `src/fantasy_baseball/summary/crosswalk.py`
- Test: `tests/test_summary/test_crosswalk.py`

**Interfaces:**
- Consumes: `normalize_name` (`utils/name_utils.py`).
- Produces:
  - `build_typed_name_to_mlbam(projections_root: Path, *, season: int) -> dict[tuple[str, str], int]` -- key `(normalized_name, player_type)`, `player_type` in `{"hitter", "pitcher"}`.
  - `player_group(positions: list[str]) -> list[str]` -- returns the game-log groups to read: `["hitting"]`, `["pitching"]`, or both for a two-way player.

**Context:** The existing `build_name_to_mlbam_map` (`streaks/reports/sunday.py:130`) and `discover_projection_files` (`streaks/data/projections.py:50`) are deliberately hitter-only and key by bare name (first-write-wins), which risks a "Will Smith" hitter/pitcher collision returning the wrong id. This task builds a type-namespaced map from BOTH `*-hitters.csv` and `*-pitchers.csv` (both carry `Name` + `MLBAMID`). Do not import the streaks module (it pulls in `duckdb`).

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary/test_crosswalk.py`:

```python
from pathlib import Path

import pandas as pd

from fantasy_baseball.summary.crosswalk import build_typed_name_to_mlbam, player_group


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_crosswalk_is_type_namespaced_and_avoids_collisions(tmp_path):
    season_dir = tmp_path / "2026"
    _write_csv(season_dir / "steamer-hitters.csv", [{"Name": "Will Smith", "MLBAMID": 111}])
    _write_csv(season_dir / "steamer-pitchers.csv", [{"Name": "Will Smith", "MLBAMID": 222}])

    xmap = build_typed_name_to_mlbam(tmp_path, season=2026)

    assert xmap[("will smith", "hitter")] == 111
    assert xmap[("will smith", "pitcher")] == 222


def test_crosswalk_skips_rows_missing_mlbamid(tmp_path):
    season_dir = tmp_path / "2026"
    _write_csv(season_dir / "atc-hitters.csv", [{"Name": "No Id", "MLBAMID": ""}])
    xmap = build_typed_name_to_mlbam(tmp_path, season=2026)
    assert ("no id", "hitter") not in xmap


def test_player_group_classification():
    assert player_group(["1B", "OF"]) == ["hitting"]
    assert player_group(["SP"]) == ["pitching"]
    assert player_group(["RP", "P"]) == ["pitching"]
    assert sorted(player_group(["DH", "SP"])) == ["hitting", "pitching"]  # two-way
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_summary/test_crosswalk.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/summary/crosswalk.py`:

```python
"""Name -> MLBAM crosswalk, namespaced by player type.

Unlike ``streaks.build_name_to_mlbam_map`` (hitter-only, bare-name keyed), this
reads both hitter and pitcher projection CSVs and keys by
``(normalized_name, player_type)`` so a same-name hitter and pitcher resolve to
their own MLBAM ids -- a bare-name map would first-write-win and return the
wrong player's box-score line. Deliberately imports no streaks code (that module
pulls in duckdb, which the Render process cannot load).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from fantasy_baseball.utils.name_utils import normalize_name

_PITCHER_POSITIONS = {"SP", "RP", "P"}


def _read_name_id(path: Path) -> list[tuple[str, int]]:
    """Return (normalized_name, mlbam_id) pairs from one projection CSV."""
    df = pd.read_csv(path, encoding="utf-8-sig", usecols=["Name", "MLBAMID"])
    out: list[tuple[str, int]] = []
    for name, raw_id in zip(df["Name"], df["MLBAMID"], strict=True):
        if pd.isna(raw_id) or str(raw_id).strip() == "":
            continue
        try:
            mlbam = int(float(raw_id))
        except (ValueError, TypeError):
            continue
        out.append((normalize_name(str(name)), mlbam))
    return out


def build_typed_name_to_mlbam(
    projections_root: Path, *, season: int
) -> dict[tuple[str, str], int]:
    """Build a ``{(normalized_name, player_type): mlbam_id}`` map.

    ``player_type`` is ``"hitter"`` (from ``*-hitters.csv``) or ``"pitcher"``
    (from ``*-pitchers.csv``). First-write-wins within each type namespace.
    """
    season_dir = projections_root / str(season)
    result: dict[tuple[str, str], int] = {}
    for path in sorted(season_dir.glob("*.csv")):
        name = path.name
        if "hitters" in name and "pitchers" not in name:
            player_type = "hitter"
        elif "pitchers" in name:
            player_type = "pitcher"
        else:
            continue
        for norm_name, mlbam in _read_name_id(path):
            result.setdefault((norm_name, player_type), mlbam)
    return result


def player_group(positions: list[str]) -> list[str]:
    """Map Yahoo eligible positions to the game-log groups to read.

    A pitcher-eligible player reads ``"pitching"``; a hitter-eligible player
    reads ``"hitting"``; a two-way player (both) reads both.
    """
    groups: list[str] = []
    if any(p in _PITCHER_POSITIONS for p in positions):
        groups.append("pitching")
    if any(p not in _PITCHER_POSITIONS for p in positions):
        groups.append("hitting")
    return groups or ["hitting"]
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_summary/test_crosswalk.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/summary/crosswalk.py tests/test_summary/test_crosswalk.py
git commit -m "feat(summary): type-keyed name->MLBAM crosswalk (#200)"
```

---

### Task 4: `build_last_night` (`summary/builders.py`)

**Files:**
- Create: `src/fantasy_baseball/summary/builders.py`
- Test: `tests/test_summary/test_builders_last_night.py`

**Interfaces:**
- Consumes: `PlayerLine` (Task 2); `build_typed_name_to_mlbam`, `player_group` (Task 3); `get_player_game_log(client, season, mlbam_id, group)` (`data/redis_store.py`).
- Produces: `build_last_night(roster, xmap, client, season, yesterday) -> tuple[list[PlayerLine], list[str]]` -- returns `(lines, unmatched_names)`. `roster` is a `list[dict]` each with `name: str` and `positions: list[str]`.

**Context:** Game logs are stored at `game_logs:{season}:{mlbam_id}:{group}` and read via `get_player_game_log`, which returns `{"name": str, "games": [row, ...]}` or `None` on miss. Each row has a `date` field (`"YYYY-MM-DD"`). Hitter rows: `pa/ab/h/hr/r/rbi/sb`. Pitcher rows: `ip/k/er/bb/w/sv/h_allowed`. Filter to `yesterday.isoformat()`. A player whose `(normalized_name, type)` is not in `xmap` goes into `unmatched`, not silently dropped. A player matched but with no game row for yesterday is simply omitted (did not play).

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary/test_builders_last_night.py`:

```python
from datetime import date

from fantasy_baseball.summary.builders import build_last_night


class FakeKV:
    def __init__(self, store):
        self._store = store

    def get(self, key):
        return self._store.get(key)


def test_build_last_night_matches_and_filters_to_yesterday():
    import json

    yesterday = date(2026, 7, 10)
    store = {
        "game_logs:2026:111:hitting": json.dumps({
            "name": "Aaron Judge",
            "games": [
                {"date": "2026-07-10", "pa": 4, "ab": 4, "h": 2, "hr": 1, "r": 2, "rbi": 3, "sb": 0},
                {"date": "2026-07-09", "pa": 4, "ab": 4, "h": 0, "hr": 0, "r": 0, "rbi": 0, "sb": 0},
            ],
        }),
    }
    xmap = {("aaron judge", "hitter"): 111}
    roster = [{"name": "Aaron Judge", "positions": ["OF"]}]

    lines, unmatched = build_last_night(roster, xmap, FakeKV(store), 2026, yesterday)

    assert unmatched == []
    assert len(lines) == 1
    assert lines[0].name == "Aaron Judge"
    assert lines[0].group == "hitting"
    assert lines[0].stats["hr"] == 1
    assert lines[0].stats["h"] == 2


def test_build_last_night_records_unmatched():
    yesterday = date(2026, 7, 10)
    xmap: dict = {}
    roster = [{"name": "Ghost Player", "positions": ["2B"]}]
    lines, unmatched = build_last_night(roster, xmap, FakeKV({}), 2026, yesterday)
    assert lines == []
    assert unmatched == ["Ghost Player"]


def test_build_last_night_omits_players_who_did_not_play():
    import json

    yesterday = date(2026, 7, 10)
    store = {
        "game_logs:2026:222:hitting": json.dumps({
            "name": "Benched Guy",
            "games": [{"date": "2026-07-08", "pa": 3, "ab": 3, "h": 1, "hr": 0, "r": 0, "rbi": 0, "sb": 0}],
        }),
    }
    xmap = {("benched guy", "hitter"): 222}
    roster = [{"name": "Benched Guy", "positions": ["1B"]}]
    lines, unmatched = build_last_night(roster, xmap, FakeKV(store), 2026, yesterday)
    assert lines == []
    assert unmatched == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_summary/test_builders_last_night.py -v`
Expected: FAIL with `ModuleNotFoundError` / `ImportError: build_last_night`

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/summary/builders.py` with the module header and the first builder:

```python
"""Section builders for the daily summary email.

Each builder is a pure function returning a typed section model (or an empty
list). Builders never raise for "no data" -- that is an empty section. They read
KV payloads the morning refresh produced, plus (for last night) per-player game
logs. No builder imports the streaks/dashboard module (it pulls in duckdb).
"""

from __future__ import annotations

from datetime import date
from typing import Any

from fantasy_baseball.data.redis_store import get_player_game_log
from fantasy_baseball.summary.crosswalk import player_group
from fantasy_baseball.summary.models import PlayerLine

_HITTER_FIELDS = ("pa", "ab", "h", "hr", "r", "rbi", "sb")
_PITCHER_FIELDS = ("ip", "k", "er", "bb", "w", "sv", "h_allowed")


def build_last_night(
    roster: list[dict[str, Any]],
    xmap: dict[tuple[str, str], int],
    client: Any,
    season: int,
    yesterday: date,
) -> tuple[list[PlayerLine], list[str]]:
    """Box-score lines for rostered players who played on ``yesterday``.

    Returns ``(lines, unmatched_names)``. A player whose name+type is not in the
    crosswalk goes into ``unmatched``; a matched player with no game row for
    ``yesterday`` is omitted (did not play).
    """
    lines: list[PlayerLine] = []
    unmatched: list[str] = []
    target = yesterday.isoformat()

    for entry in roster:
        name = entry.get("name", "")
        positions = entry.get("positions", []) or []
        groups = player_group(positions)

        # A two-way player resolves under whichever type namespace matches; the
        # same person-level MLBAM id serves both game-log groups.
        norm = _normalize(name)
        mlbam: int | None = None
        for group in groups:
            key = (norm, "pitcher" if group == "pitching" else "hitter")
            if key in xmap:
                mlbam = xmap[key]
                break
        if mlbam is None:
            unmatched.append(name)
            continue

        for group in groups:
            log = get_player_game_log(client, season, str(mlbam), group)
            if not log:
                continue
            for row in log.get("games", []):
                if row.get("date") != target:
                    continue
                fields = _HITTER_FIELDS if group == "hitting" else _PITCHER_FIELDS
                stats = {f: _num(row.get(f)) for f in fields}
                lines.append(PlayerLine(name=name, group=group, stats=stats))

    return lines, unmatched


def _normalize(name: str) -> str:
    from fantasy_baseball.utils.name_utils import normalize_name

    return normalize_name(name)


def _num(value: Any) -> float:
    return float(value) if value is not None else 0.0
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_summary/test_builders_last_night.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/summary/builders.py tests/test_summary/test_builders_last_night.py
git commit -m "feat(summary): build_last_night box-score builder (#200)"
```

---

### Task 5: `build_streaks` (duckdb-free dict parse)

**Files:**
- Modify: `src/fantasy_baseball/summary/builders.py`
- Test: `tests/test_summary/test_builders_streaks.py`

**Interfaces:**
- Consumes: `StreakItem` (Task 2); the serialized `STREAK_SCORES` dict.
- Produces: `build_streaks(streak_payload: dict | None) -> list[StreakItem]`.

**Context:** `STREAK_SCORES` is `serialize_report(report)` output (`streaks/dashboard.py`), a dict with `roster_rows: [ {name, positions, player_id, composite, scores: {category: {label, probability, ...}}, ...}, ... ]`. `label` is `"hot"` / `"cold"`. It is hitters-only and single-window -- do NOT claim 7d/14d or pitcher streaks. Parse the dict directly; do not import `deserialize_report` (duckdb). Emit only categories whose `label` is `"hot"` or `"cold"`. A `None` payload (KV miss) yields `[]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary/test_builders_streaks.py`:

```python
from fantasy_baseball.summary.builders import build_streaks


def test_build_streaks_extracts_hot_cold_from_roster_rows():
    payload = {
        "roster_rows": [
            {
                "name": "Aaron Judge",
                "scores": {
                    "hr": {"label": "hot", "probability": 0.71},
                    "avg": {"label": "cold", "probability": 0.64},
                    "sb": {"label": "neutral", "probability": 0.10},
                },
            },
        ],
        "fa_rows": [],
    }
    items = build_streaks(payload)
    labels = {(i.category, i.label) for i in items}
    assert ("hr", "hot") in labels
    assert ("avg", "cold") in labels
    assert all(i.label in ("hot", "cold") for i in items)  # neutral dropped
    judge_hr = next(i for i in items if i.category == "hr")
    assert judge_hr.name == "Aaron Judge"
    assert judge_hr.probability == 0.71


def test_build_streaks_handles_missing_payload():
    assert build_streaks(None) == []
    assert build_streaks({}) == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_summary/test_builders_streaks.py -v`
Expected: FAIL with `ImportError: build_streaks`

- [ ] **Step 3: Write the implementation**

Append to `src/fantasy_baseball/summary/builders.py`:

```python
def build_streaks(streak_payload: dict[str, Any] | None) -> list["StreakItem"]:
    """Hot/cold hitter streaks from the serialized STREAK_SCORES report.

    Reads the serialized dict directly (no duckdb import). Hitters-only,
    single-window -- matches the underlying report; emits one item per category
    labelled "hot" or "cold".
    """
    from fantasy_baseball.summary.models import StreakItem

    if not streak_payload:
        return []
    items: list[StreakItem] = []
    for row in streak_payload.get("roster_rows", []):
        name = row.get("name", "")
        for category, score in (row.get("scores") or {}).items():
            label = score.get("label")
            if label not in ("hot", "cold"):
                continue
            prob = score.get("probability")
            items.append(
                StreakItem(
                    name=name,
                    category=str(category),
                    label=str(label),
                    probability=float(prob) if prob is not None else 0.0,
                )
            )
    return items
```

Add `from fantasy_baseball.summary.models import PlayerLine, StreakItem` to the top imports (replace the existing `PlayerLine` import line) so the return annotation resolves without the local import; keep the local import removed to satisfy ruff. (If ruff flags the string annotation, change `-> list["StreakItem"]` to `-> list[StreakItem]`.)

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_summary/test_builders_streaks.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/summary/builders.py tests/test_summary/test_builders_streaks.py
git commit -m "feat(summary): build_streaks duckdb-free parse (#200)"
```

---

### Task 6: `build_lineup_moves`, `build_injuries`, `build_probables`

**Files:**
- Modify: `src/fantasy_baseball/summary/builders.py`
- Test: `tests/test_summary/test_builders_misc.py`

**Interfaces:**
- Produces:
  - `build_lineup_moves(optimal_payload: dict | None) -> list[LineupMove]`
  - `build_injuries(injury_rows: list[dict]) -> list[InjuryItem]`
  - `build_probables(probable_rows: list[dict] | None) -> list[ProbableMatchup]`

**Context:**
- `LINEUP_OPTIMAL` payload has `["moves"] = {"swaps": [...], "unpaired_starts": [...], "unpaired_benches": [...]}`. A swap is `{"start": {"player","from","to","roto_delta"}, "bench": {"player","from","to"}}`. `unpaired_starts` rows are `{"player","from","to","roto_delta"}`; `unpaired_benches` are `{"player","from","to"}`.
- `fetch_injuries` rows are `{"name","status","status_full","injury_note",...}`.
- `PROBABLE_STARTERS` rows are `{"pitcher","starts","days","opponents","matchup_quality","matchups":[...]}`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary/test_builders_misc.py`:

```python
from fantasy_baseball.summary.builders import build_injuries, build_lineup_moves, build_probables


def test_build_lineup_moves_flattens_swaps_and_unpaired():
    payload = {
        "moves": {
            "swaps": [
                {"start": {"player": "A", "from": "BN", "to": "OF", "roto_delta": 0.4},
                 "bench": {"player": "B", "from": "OF", "to": "BN"}},
            ],
            "unpaired_starts": [{"player": "C", "from": "BN", "to": "UTIL", "roto_delta": 0.2}],
            "unpaired_benches": [{"player": "D", "from": "2B", "to": "BN"}],
        }
    }
    moves = build_lineup_moves(payload)
    actions = {(m.player, m.action) for m in moves}
    assert ("A", "start") in actions
    assert ("B", "sit") in actions
    assert ("C", "start") in actions
    assert ("D", "sit") in actions
    a = next(m for m in moves if m.player == "A")
    assert a.to_slot == "OF" and a.roto_delta == 0.4


def test_build_lineup_moves_handles_missing():
    assert build_lineup_moves(None) == []
    assert build_lineup_moves({}) == []


def test_build_injuries_maps_rows():
    rows = [
        {"name": "Hurt Guy", "status": "IL15", "status_full": "15-Day IL", "injury_note": "hamstring strain"},
        {"name": "No Note", "status": "DTD", "status_full": "Day-To-Day", "injury_note": ""},
    ]
    items = build_injuries(rows)
    assert items[0].name == "Hurt Guy"
    assert items[0].status == "IL15"
    assert items[0].note == "hamstring strain"
    assert items[1].note == ""


def test_build_probables_maps_and_handles_absent():
    assert build_probables(None) == []
    rows = [{"pitcher": "Ace", "starts": 2, "days": "Mon, Sat",
             "opponents": "@ BAL, vs TOR", "matchup_quality": "Great", "matchups": []}]
    items = build_probables(rows)
    assert items[0].pitcher == "Ace"
    assert items[0].starts == 2
    assert items[0].quality == "Great"
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_summary/test_builders_misc.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

Append to `src/fantasy_baseball/summary/builders.py`:

```python
def build_lineup_moves(optimal_payload: dict[str, Any] | None) -> list["LineupMove"]:
    """Flatten LINEUP_OPTIMAL["moves"] into start/sit LineupMove rows."""
    from fantasy_baseball.summary.models import LineupMove

    if not optimal_payload:
        return []
    moves = optimal_payload.get("moves") or {}
    out: list[LineupMove] = []

    def _start(row: dict[str, Any]) -> LineupMove:
        rd = row.get("roto_delta")
        return LineupMove(
            player=row.get("player", ""),
            action="start",
            from_slot=row.get("from", ""),
            to_slot=row.get("to", ""),
            roto_delta=float(rd) if rd is not None else 0.0,
        )

    def _sit(row: dict[str, Any]) -> LineupMove:
        return LineupMove(
            player=row.get("player", ""),
            action="sit",
            from_slot=row.get("from", ""),
            to_slot=row.get("to", ""),
            roto_delta=0.0,
        )

    for swap in moves.get("swaps", []):
        if swap.get("start"):
            out.append(_start(swap["start"]))
        if swap.get("bench"):
            out.append(_sit(swap["bench"]))
    for row in moves.get("unpaired_starts", []):
        out.append(_start(row))
    for row in moves.get("unpaired_benches", []):
        out.append(_sit(row))
    return out


def build_injuries(injury_rows: list[dict[str, Any]]) -> list["InjuryItem"]:
    """Map fetch_injuries rows to InjuryItem (injury_note carries the news)."""
    from fantasy_baseball.summary.models import InjuryItem

    return [
        InjuryItem(
            name=row.get("name", ""),
            status=row.get("status", ""),
            note=row.get("injury_note", "") or "",
        )
        for row in injury_rows
    ]


def build_probables(probable_rows: list[dict[str, Any]] | None) -> list["ProbableMatchup"]:
    """Map PROBABLE_STARTERS rollup rows to ProbableMatchup. Absent -> []."""
    from fantasy_baseball.summary.models import ProbableMatchup

    if not probable_rows:
        return []
    out: list[ProbableMatchup] = []
    for row in probable_rows:
        starts = row.get("starts")
        out.append(
            ProbableMatchup(
                pitcher=row.get("pitcher", ""),
                starts=int(starts) if starts is not None else 0,
                days=row.get("days", ""),
                opponents=row.get("opponents", ""),
                quality=row.get("matchup_quality", ""),
            )
        )
    return out
```

Consolidate the `from fantasy_baseball.summary.models import ...` imports at the top of the module (models: `InjuryItem, LineupMove, PlayerLine, ProbableMatchup, StreakItem`) and drop the per-function local imports to satisfy ruff `I`/`SIM`; keep string annotations or switch to direct types consistently.

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_summary/test_builders_misc.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/summary/builders.py tests/test_summary/test_builders_misc.py
git commit -m "feat(summary): lineup-moves, injuries, probables builders (#200)"
```

---

### Task 7: `build_standings_delta` + snapshot helpers

**Files:**
- Modify: `src/fantasy_baseball/summary/builders.py`
- Test: `tests/test_summary/test_builders_standings.py`

**Interfaces:**
- Consumes: `StandingsDelta`, `TeamDelta` (Task 2); `Standings.from_json` (`models/standings.py`); `score_roto` (`scoring.py`).
- Produces: `build_standings_delta(current_raw, snapshot_payload, user_team_name) -> StandingsDelta`. `current_raw` is `read_cache(STANDINGS)` output; `snapshot_payload` is `read_cache(STANDINGS_SNAPSHOT)` output (`{"last_refresh","standings"}`) or `None`.

**Context:** `Standings.to_json()` stores raw category totals + rank + `yahoo_points_for`, NOT per-category place points. Reconstruct both current and prior via `Standings.from_json`, run `score_roto(cast("Any", standings))` (no `team_sds`) to get `{team_name: CategoryPoints}` where `.values` is `{Category: float}`. Diff rank (from entries), total points (`CategoryPoints.total`), and per-category `.values`. Freshness is enforced up-front by the orchestrator, NOT here. First run (`snapshot_payload is None`) -> `is_first_run=True`, empty teams. Set `rate_cat_caveat=True` (AVG/ERA/WHIP recompute can differ from Yahoo by +/-0.5 per tie). `Category` is an enum; use `.value` (or `str(cat)`) for the delta dict keys.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary/test_builders_standings.py`:

```python
from fantasy_baseball.summary.builders import build_standings_delta


def _standings_json(effective_date, teams):
    return {"effective_date": effective_date, "teams": teams}


def _team(name, rank, stats, yahoo_points=None):
    return {"name": name, "team_key": f"k.{name}", "rank": rank,
            "yahoo_points_for": yahoo_points, "stats": stats, "extras": {}}


# Two teams, two counting categories, so rank-based roto points are unambiguous.
_STATS_A = {"HR": 100.0, "SB": 50.0}
_STATS_B = {"HR": 80.0, "SB": 60.0}


def test_first_run_yields_baseline():
    current = _standings_json("2026-07-14", [_team("My Team", 1, _STATS_A), _team("Rival", 2, _STATS_B)])
    delta = build_standings_delta(current, None, "My Team")
    assert delta.is_first_run is True
    assert delta.teams == []


def test_delta_computes_rank_and_category_movement():
    prior = _standings_json("2026-07-14", [_team("My Team", 2, _STATS_B), _team("Rival", 1, _STATS_A)])
    current = _standings_json("2026-07-14", [_team("My Team", 1, _STATS_A), _team("Rival", 2, _STATS_B)])
    snapshot = {"last_refresh": "2026-07-10 08:00", "standings": prior}

    delta = build_standings_delta(current, snapshot, "My Team")

    assert delta.is_first_run is False
    assert delta.user_team_name == "My Team"
    mine = next(t for t in delta.teams if t.name == "My Team")
    assert mine.rank_prev == 2
    assert mine.rank_now == 1
    # My Team went from trailing both cats to leading both -> +2 total roto points.
    assert mine.points_now - mine.points_prev == 2.0
    assert delta.rate_cat_caveat is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_summary/test_builders_standings.py -v`
Expected: FAIL with `ImportError`

- [ ] **Step 3: Write the implementation**

Append to `src/fantasy_baseball/summary/builders.py` (add `from typing import cast` and the standings/scoring imports at the top of the module):

```python
def build_standings_delta(
    current_raw: dict[str, Any] | None,
    snapshot_payload: dict[str, Any] | None,
    user_team_name: str,
) -> "StandingsDelta":
    """Overnight roto movement vs. the prior snapshot.

    Reconstructs both standings and re-scores per-category roto points (the
    stored payload holds raw totals, not place points). Freshness is enforced
    up-front by the orchestrator; this function assumes current is fresh.
    """
    from fantasy_baseball.models.standings import Standings
    from fantasy_baseball.scoring import score_roto
    from fantasy_baseball.summary.models import StandingsDelta, TeamDelta

    if current_raw is None or snapshot_payload is None:
        return StandingsDelta(is_first_run=True, user_team_name=user_team_name)

    current = Standings.from_json(current_raw)
    prior = Standings.from_json(snapshot_payload["standings"])

    cur_roto = score_roto(cast("Any", current))
    prev_roto = score_roto(cast("Any", prior))
    cur_rank = {e.team_name: e.rank for e in current.entries}
    prev_rank = {e.team_name: e.rank for e in prior.entries}

    teams: list[TeamDelta] = []
    for name, cur_points in cur_roto.items():
        prev_points = prev_roto.get(name)
        if prev_points is None:
            continue
        cat_delta = {
            str(getattr(cat, "value", cat)): cur_points.values[cat] - prev_points.values.get(cat, 0.0)
            for cat in cur_points.values
        }
        teams.append(
            TeamDelta(
                name=name,
                rank_prev=prev_rank.get(name, cur_rank.get(name, 0)),
                rank_now=cur_rank.get(name, 0),
                points_prev=prev_points.total,
                points_now=cur_points.total,
                category_points_delta=cat_delta,
            )
        )

    return StandingsDelta(
        is_first_run=False,
        user_team_name=user_team_name,
        teams=teams,
        rate_cat_caveat=True,
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_summary/test_builders_standings.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/summary/builders.py tests/test_summary/test_builders_standings.py
git commit -m "feat(summary): build_standings_delta with roto recompute (#200)"
```

---

### Task 8: Assembly + Yahoo wiring (`summary/assemble.py`)

**Files:**
- Create: `src/fantasy_baseball/summary/assemble.py`
- Test: `tests/test_summary/test_assemble.py`

**Interfaces:**
- Consumes: all builders (Tasks 4-7); `read_cache`, `read_cache_dict`, `read_cache_list`, `read_meta` (`web/season_data.py`); `get_kv` (`data/kv_store.py`); Yahoo helpers (`auth/yahoo_auth.py`, `lineup/yahoo_roster.py`); `LeagueConfig` (`config.py`); `local_today` (`utils/time_utils.py`).
- Produces:
  - `refresh_is_fresh(meta: dict, today: date) -> bool` -- freshness gate predicate.
  - `build_daily_summary(config, projections_root, *, today=None, league=None, team_key=None) -> DailySummary` -- assembles all sections, wrapping each builder so one failure => empty section + a `section_errors` entry.

**Context:** `read_meta()` returns `{}` on miss; `meta["last_refresh"]` is `"%Y-%m-%d %H:%M"` local. The freshness gate parses its date and compares to `local_today()`. `build_daily_summary` reads KV sections, builds the crosswalk from `projections_root`, reads the roster via `fetch_roster(league, team_key)`, and fetches live injuries via `fetch_injuries(league, team_key)`. Each builder call is wrapped in try/except: on exception, log, append the builder name to `section_errors`, and use an empty section.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary/test_assemble.py`:

```python
from datetime import date

from fantasy_baseball.summary.assemble import refresh_is_fresh


def test_refresh_is_fresh_true_when_last_refresh_is_today():
    meta = {"last_refresh": "2026-07-11 08:05"}
    assert refresh_is_fresh(meta, date(2026, 7, 11)) is True


def test_refresh_is_fresh_false_when_stale():
    meta = {"last_refresh": "2026-07-10 08:05"}
    assert refresh_is_fresh(meta, date(2026, 7, 11)) is False


def test_refresh_is_fresh_false_when_meta_empty_or_malformed():
    assert refresh_is_fresh({}, date(2026, 7, 11)) is False
    assert refresh_is_fresh({"last_refresh": "garbage"}, date(2026, 7, 11)) is False
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_summary/test_assemble.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the freshness predicate + assembly**

Create `src/fantasy_baseball/summary/assemble.py`:

```python
"""Assemble a DailySummary from the refreshed KV plus a live Yahoo injury fetch.

Each section builder is wrapped so one failure degrades to an empty section (and
a section_errors entry) rather than aborting the email. The send/skip decision
is a separate up-front check on META freshness (refresh_is_fresh), applied by the
orchestrator before this runs.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

from fantasy_baseball.config import LeagueConfig
from fantasy_baseball.data.cache_keys import CacheKey
from fantasy_baseball.data.kv_store import get_kv
from fantasy_baseball.summary.builders import (
    build_injuries,
    build_last_night,
    build_lineup_moves,
    build_probables,
    build_standings_delta,
    build_streaks,
)
from fantasy_baseball.summary.crosswalk import build_typed_name_to_mlbam
from fantasy_baseball.summary.models import DailySummary, StandingsDelta
from fantasy_baseball.utils.time_utils import local_today
from fantasy_baseball.web.season_data import (
    read_cache,
    read_cache_dict,
    read_cache_list,
    read_meta,
)

logger = logging.getLogger(__name__)


def refresh_is_fresh(meta: dict[str, Any], today: date) -> bool:
    """True iff META.last_refresh is from ``today`` (local calendar date)."""
    raw = meta.get("last_refresh")
    if not raw:
        return False
    try:
        parsed = datetime.strptime(str(raw), "%Y-%m-%d %H:%M").date()
    except ValueError:
        return False
    return parsed == today


def build_daily_summary(
    config: LeagueConfig,
    projections_root: Path,
    *,
    today: date | None = None,
    league: Any,
    team_key: str,
) -> DailySummary:
    """Assemble every section. One failing builder => empty section + a note."""
    today = today or local_today()
    yesterday = today - timedelta(days=1)
    season = config.season_year
    client = get_kv()
    section_errors: list[str] = []

    def _guard(name: str, fn: Any, fallback: Any) -> Any:
        try:
            return fn()
        except Exception:  # noqa: BLE001 - degrade one section, keep the email
            logger.exception("summary builder %s failed", name)
            section_errors.append(name)
            return fallback

    from fantasy_baseball.lineup.yahoo_roster import fetch_injuries, fetch_roster

    xmap = _guard("crosswalk", lambda: build_typed_name_to_mlbam(projections_root, season=season), {})
    roster = _guard("roster", lambda: fetch_roster(league, team_key), [])

    last_night, unmatched = _guard(
        "build_last_night",
        lambda: build_last_night(roster, xmap, client, season, yesterday),
        ([], []),
    )
    streaks = _guard("build_streaks", lambda: build_streaks(read_cache_dict(CacheKey.STREAK_SCORES)), [])
    lineup_moves = _guard(
        "build_lineup_moves", lambda: build_lineup_moves(read_cache_dict(CacheKey.LINEUP_OPTIMAL)), []
    )
    injuries = _guard("build_injuries", lambda: build_injuries(fetch_injuries(league, team_key)), [])
    probables = _guard(
        "build_probables", lambda: build_probables(read_cache_list(CacheKey.PROBABLE_STARTERS)), []
    )
    standings_delta = _guard(
        "build_standings_delta",
        lambda: build_standings_delta(
            read_cache(CacheKey.STANDINGS),
            read_cache(CacheKey.STANDINGS_SNAPSHOT),
            config.team_name,
        ),
        StandingsDelta(is_first_run=True, user_team_name=config.team_name),
    )

    return DailySummary(
        as_of=yesterday,
        last_night=last_night,
        unmatched=unmatched,
        streaks=streaks,
        standings_delta=standings_delta,
        lineup_moves=lineup_moves,
        injuries=injuries,
        probables=probables,
        section_errors=section_errors,
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_summary/test_assemble.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/summary/assemble.py tests/test_summary/test_assemble.py
git commit -m "feat(summary): assemble + freshness gate + Yahoo wiring (#200)"
```

---

### Task 9: Render (`summary/render.py`)

**Files:**
- Create: `src/fantasy_baseball/summary/render.py`
- Test: `tests/test_summary/test_render.py`

**Interfaces:**
- Consumes: `DailySummary` and section models (Task 2).
- Produces: `render_html(summary: DailySummary) -> str`, `render_text(summary: DailySummary) -> str`, `subject_line(summary: DailySummary) -> str`.

**Context:** HTML body is UTF-8 (player names may be non-ASCII). Omit empty sections; render the first-run standings baseline as an explicit "baseline established" line; render `section_errors` as a "could not build: X" note; render `unmatched` as "N players unmatched." Keep HTML self-contained (inline styles), no external assets.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary/test_render.py`:

```python
from datetime import date

from fantasy_baseball.summary.models import (
    DailySummary,
    InjuryItem,
    PlayerLine,
    StandingsDelta,
    StreakItem,
)
from fantasy_baseball.summary.render import render_html, render_text, subject_line


def _summary(**overrides):
    base = dict(
        as_of=date(2026, 7, 10),
        last_night=[PlayerLine(name="Aaron Judge", group="hitting",
                               stats={"h": 2, "hr": 1, "r": 2, "rbi": 3, "sb": 0, "ab": 4, "pa": 4})],
        unmatched=[],
        streaks=[StreakItem(name="Judge", category="hr", label="hot", probability=0.71)],
        standings_delta=StandingsDelta(is_first_run=False, user_team_name="My Team"),
        lineup_moves=[],
        injuries=[InjuryItem(name="Hurt Guy", status="IL15", note="hamstring")],
        probables=[],
        section_errors=[],
    )
    base.update(overrides)
    return DailySummary(**base)


def test_render_html_includes_populated_sections():
    html = render_html(_summary())
    assert "Aaron Judge" in html
    assert "Hurt Guy" in html
    assert "hamstring" in html
    assert "<html" in html.lower()


def test_render_omits_empty_and_notes_errors_and_firstrun():
    html = render_html(_summary(
        last_night=[], injuries=[], streaks=[],
        standings_delta=StandingsDelta(is_first_run=True, user_team_name="My Team"),
        section_errors=["build_streaks"],
        unmatched=["Ghost"],
    ))
    assert "baseline established" in html.lower()
    assert "build_streaks" in html
    assert "unmatched" in html.lower()


def test_render_text_is_plain_and_nonempty():
    text = render_text(_summary())
    assert "Aaron Judge" in text
    assert "<" not in text  # no HTML tags in the text part


def test_subject_line_mentions_date():
    assert "2026-07-10" in subject_line(_summary())
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_summary/test_render.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/summary/render.py`. Render each section only if non-empty; build an HTML string and a parallel plain-text string. (Full implementation -- keep it straightforward; below is the required structure with complete code.)

```python
"""Render a DailySummary into an HTML email body and a plain-text fallback."""

from __future__ import annotations

from fantasy_baseball.summary.models import DailySummary


def subject_line(summary: DailySummary) -> str:
    return f"Fantasy daily summary - {summary.as_of.isoformat()}"


def _hitter_line(stats: dict[str, float]) -> str:
    return (
        f"{int(stats.get('h', 0))}-{int(stats.get('ab', 0))}, "
        f"{int(stats.get('hr', 0))} HR, {int(stats.get('r', 0))} R, "
        f"{int(stats.get('rbi', 0))} RBI, {int(stats.get('sb', 0))} SB"
    )


def _pitcher_line(stats: dict[str, float]) -> str:
    return (
        f"{stats.get('ip', 0)} IP, {int(stats.get('k', 0))} K, "
        f"{int(stats.get('er', 0))} ER, {int(stats.get('w', 0))} W, {int(stats.get('sv', 0))} SV"
    )


def _sections(summary: DailySummary) -> list[tuple[str, list[str]]]:
    """Return (heading, lines) for each NON-empty section, in email order."""
    out: list[tuple[str, list[str]]] = []

    if summary.last_night:
        lines = [
            f"{p.name}: {_hitter_line(p.stats) if p.group == 'hitting' else _pitcher_line(p.stats)}"
            for p in summary.last_night
        ]
        out.append(("Last night", lines))

    if summary.streaks:
        out.append((
            "Hot / cold (hitters)",
            [f"{s.name} - {s.category} {s.label} ({s.probability:.2f})" for s in summary.streaks],
        ))

    sd = summary.standings_delta
    if sd.is_first_run:
        out.append(("Standings", ["Baseline established - deltas start next run."]))
    elif sd.teams:
        mine = next((t for t in sd.teams if t.name == sd.user_team_name), None)
        if mine is not None:
            change = mine.points_now - mine.points_prev
            out.append((
                "Standings",
                [
                    f"{mine.name}: rank {mine.rank_prev} -> {mine.rank_now}, "
                    f"roto {mine.points_prev:.1f} -> {mine.points_now:.1f} ({change:+.1f})",
                    "(AVG/ERA/WHIP movement approximate - averaged-rank recompute.)",
                ],
            ))

    if summary.lineup_moves:
        out.append((
            "Lineup moves",
            [f"{m.action.upper()} {m.player}: {m.from_slot} -> {m.to_slot}" for m in summary.lineup_moves],
        ))

    if summary.injuries:
        out.append((
            "Injuries",
            [f"{i.name} ({i.status}): {i.note}".rstrip(": ") for i in summary.injuries],
        ))

    if summary.probables:
        out.append((
            "Probable starts",
            [f"{p.pitcher}: {p.starts} start(s) - {p.opponents} [{p.quality}]" for p in summary.probables],
        ))

    notes: list[str] = []
    if summary.unmatched:
        notes.append(f"{len(summary.unmatched)} roster player(s) unmatched: {', '.join(summary.unmatched)}")
    if summary.section_errors:
        notes.append(f"Could not build: {', '.join(summary.section_errors)}")
    if notes:
        out.append(("Notes", notes))

    return out


def render_text(summary: DailySummary) -> str:
    parts = [f"Fantasy daily summary - {summary.as_of.isoformat()}", ""]
    for heading, lines in _sections(summary):
        parts.append(heading)
        parts.extend(f"  {line}" for line in lines)
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def render_html(summary: DailySummary) -> str:
    import html as html_lib

    blocks: list[str] = []
    for heading, lines in _sections(summary):
        items = "".join(f"<li>{html_lib.escape(line)}</li>" for line in lines)
        blocks.append(
            f'<h2 style="font-size:16px;margin:16px 0 4px">{html_lib.escape(heading)}</h2>'
            f'<ul style="margin:0;padding-left:20px">{items}</ul>'
        )
    body = "".join(blocks)
    title = html_lib.escape(subject_line(summary))
    return (
        '<html><head><meta charset="utf-8"></head>'
        '<body style="font-family:Arial,Helvetica,sans-serif;color:#111">'
        f"<h1 style=\"font-size:20px\">{title}</h1>{body}</body></html>"
    )
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_summary/test_render.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/summary/render.py tests/test_summary/test_render.py
git commit -m "feat(summary): HTML + text rendering (#200)"
```

---

### Task 10: Send (`summary/send.py`)

**Files:**
- Create: `src/fantasy_baseball/summary/send.py`
- Test: `tests/test_summary/test_send.py`

**Interfaces:**
- Produces: `send_email(*, api_key, from_address, recipients, subject, html, text) -> str` -- returns the Resend message id; raises on failure.

**Context:** Uses the `resend` SDK: set `resend.api_key`, call `resend.Emails.send({...})`. Tests monkeypatch `resend.Emails.send` -- never hit the network.

- [ ] **Step 1: Write the failing test**

Create `tests/test_summary/test_send.py`:

```python
import fantasy_baseball.summary.send as send_mod
from fantasy_baseball.summary.send import send_email


def test_send_email_builds_payload_and_returns_id(monkeypatch):
    captured = {}

    def fake_send(payload):
        captured.update(payload)
        return {"id": "msg_123"}

    monkeypatch.setattr(send_mod.resend.Emails, "send", staticmethod(fake_send))

    msg_id = send_email(
        api_key="key_test",
        from_address="digest@x.com",
        recipients=["me@x.com"],
        subject="Subj",
        html="<html></html>",
        text="Subj",
    )
    assert msg_id == "msg_123"
    assert captured["from"] == "digest@x.com"
    assert captured["to"] == ["me@x.com"]
    assert captured["subject"] == "Subj"
    assert captured["html"] == "<html></html>"


def test_send_email_raises_when_no_id(monkeypatch):
    monkeypatch.setattr(send_mod.resend.Emails, "send", staticmethod(lambda payload: {}))
    try:
        send_email(api_key="k", from_address="a@x.com", recipients=["b@x.com"],
                   subject="s", html="h", text="t")
        raise AssertionError("expected RuntimeError")
    except RuntimeError:
        pass
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_summary/test_send.py -v`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

Create `src/fantasy_baseball/summary/send.py`:

```python
"""Send the rendered summary via the Resend transactional email API."""

from __future__ import annotations

import resend


def send_email(
    *,
    api_key: str,
    from_address: str,
    recipients: list[str],
    subject: str,
    html: str,
    text: str,
) -> str:
    """Send one email to ``recipients``. Returns the Resend message id.

    Raises RuntimeError if Resend returns no id (treated as a send failure).
    """
    resend.api_key = api_key
    result = resend.Emails.send(
        {
            "from": from_address,
            "to": recipients,
            "subject": subject,
            "html": html,
            "text": text,
        }
    )
    msg_id = result.get("id") if isinstance(result, dict) else None
    if not msg_id:
        raise RuntimeError(f"Resend send failed; response={result!r}")
    return str(msg_id)
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_summary/test_send.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/summary/send.py tests/test_summary/test_send.py
git commit -m "feat(summary): Resend send wrapper (#200)"
```

---

### Task 11: Orchestrator script (`scripts/send_daily_summary.py`)

**Files:**
- Create: `scripts/send_daily_summary.py`
- Test: `tests/test_scripts/test_send_daily_summary.py`

**Interfaces:**
- Consumes: everything above.
- Produces: `run_summary(config, projections_root, *, api_key, league, team_key, today=None) -> int` -- the testable core: applies the freshness gate, assembles, renders, sends, and writes the post-send snapshot. Returns a process exit code (0 = sent, non-zero = skipped/failed). Plus a `main()` that wires config/auth/env and calls `run_summary`.

**Context:** `main()` mirrors `scripts/refresh_remote.py`: set `os.environ["RENDER"] = "true"` BEFORE importing KV-touching modules, inject `src/` into `sys.path`, `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`. The snapshot is written to `STANDINGS_SNAPSHOT` ONLY after a successful send: `{"last_refresh": meta["last_refresh"], "standings": read_cache(STANDINGS)}`. Freshness gate: if `read_meta()` is empty OR `not refresh_is_fresh(meta, today)`, log + return non-zero WITHOUT sending.

- [ ] **Step 1: Write the failing test**

Create `tests/test_scripts/test_send_daily_summary.py`:

```python
from datetime import date

import pytest


@pytest.fixture
def patched(monkeypatch):
    import scripts.send_daily_summary as mod

    sent = {}
    monkeypatch.setattr(mod, "read_meta", lambda: {"last_refresh": "2026-07-11 08:00"})
    monkeypatch.setattr(mod, "send_email", lambda **kw: sent.update(kw) or "msg_1")
    written = {}
    monkeypatch.setattr(mod, "_write_snapshot", lambda meta: written.update({"done": True}))
    return mod, sent, written


def test_run_summary_stale_refresh_skips_send(monkeypatch, patched):
    mod, sent, written = patched
    monkeypatch.setattr(mod, "read_meta", lambda: {"last_refresh": "2026-07-09 08:00"})
    rc = mod.run_summary(_cfg(), _root(), api_key="k", league=object(), team_key="t",
                         today=date(2026, 7, 11))
    assert rc != 0
    assert sent == {}  # never sent


def test_run_summary_missing_meta_skips_send(monkeypatch, patched):
    mod, sent, written = patched
    monkeypatch.setattr(mod, "read_meta", lambda: {})
    rc = mod.run_summary(_cfg(), _root(), api_key="k", league=object(), team_key="t",
                         today=date(2026, 7, 11))
    assert rc != 0
    assert sent == {}


def test_run_summary_fresh_sends_and_writes_snapshot(monkeypatch, patched):
    mod, sent, written = patched
    from fantasy_baseball.summary.models import DailySummary, StandingsDelta

    monkeypatch.setattr(mod, "build_daily_summary", lambda *a, **k: DailySummary(
        as_of=date(2026, 7, 10), last_night=[], unmatched=[], streaks=[],
        standings_delta=StandingsDelta(is_first_run=True, user_team_name="T"),
        lineup_moves=[], injuries=[], probables=[], section_errors=[]))
    rc = mod.run_summary(_cfg(), _root(), api_key="k", league=object(), team_key="t",
                         today=date(2026, 7, 11))
    assert rc == 0
    assert sent["subject"]
    assert written.get("done") is True


def _cfg():
    from fantasy_baseball.config import LeagueConfig
    c = LeagueConfig.__new__(LeagueConfig)
    c.team_name = "T"
    c.season_year = 2026
    c.summary = {"recipients": ["me@x.com"], "from_address": "d@x.com"}
    return c


def _root():
    from pathlib import Path
    return Path(".")
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_scripts/test_send_daily_summary.py -v`
Expected: FAIL with `ModuleNotFoundError: scripts.send_daily_summary`

- [ ] **Step 3: Write the implementation**

Create `scripts/send_daily_summary.py`:

```python
"""Send the daily summary email. Run as a Render cron after the morning refresh.

Freshness gate: only sends if META.last_refresh is from today (else exits
non-zero so the cron surfaces "the refresh didn't run"). Writes the standings
snapshot only after a successful send.
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import date
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

# Read Upstash, not local SQLite (must precede KV-touching imports).
os.environ["RENDER"] = "true"

from fantasy_baseball.config import LeagueConfig, load_config  # noqa: E402
from fantasy_baseball.data.cache_keys import CacheKey  # noqa: E402
from fantasy_baseball.summary.assemble import build_daily_summary, refresh_is_fresh  # noqa: E402
from fantasy_baseball.summary.render import render_html, render_text, subject_line  # noqa: E402
from fantasy_baseball.summary.send import send_email  # noqa: E402
from fantasy_baseball.utils.time_utils import local_today  # noqa: E402
from fantasy_baseball.web.season_data import read_cache, read_meta, write_cache  # noqa: E402

logger = logging.getLogger(__name__)


def _write_snapshot(meta: dict) -> None:
    """Persist the post-send standings snapshot for tomorrow's delta baseline."""
    standings = read_cache(CacheKey.STANDINGS)
    if standings is None:
        logger.warning("no STANDINGS to snapshot; skipping snapshot write")
        return
    write_cache(
        CacheKey.STANDINGS_SNAPSHOT,
        {"last_refresh": meta.get("last_refresh"), "standings": standings},
    )


def run_summary(
    config: LeagueConfig,
    projections_root: Path,
    *,
    api_key: str,
    league: object,
    team_key: str,
    today: date | None = None,
) -> int:
    """Freshness-gate, assemble, render, send, snapshot. Returns an exit code."""
    today = today or local_today()

    meta = read_meta()
    if not refresh_is_fresh(meta, today):
        logger.error(
            "refresh not fresh (last_refresh=%r, today=%s); skipping send",
            meta.get("last_refresh"), today,
        )
        return 1

    summary = build_daily_summary(
        config, projections_root, today=today, league=league, team_key=team_key
    )
    recipients = config.summary.get("recipients") or []
    from_address = config.summary.get("from_address") or ""
    if not recipients or not from_address:
        logger.error("summary.recipients / summary.from_address not configured")
        return 2

    send_email(
        api_key=api_key,
        from_address=from_address,
        recipients=recipients,
        subject=subject_line(summary),
        html=render_html(summary),
        text=render_text(summary),
    )
    _write_snapshot(meta)
    logger.info("daily summary sent to %s", recipients)
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]

    from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
    from fantasy_baseball.lineup.yahoo_roster import fetch_teams, find_user_team_key

    config = load_config(_PROJECT_ROOT / "config" / "league.yaml")
    api_key = os.environ.get("RESEND_API_KEY", "")
    if not api_key:
        logger.error("RESEND_API_KEY not set")
        return 2

    session = get_yahoo_session()
    league = get_league(session, config.league_id, config.game_code)
    teams = fetch_teams(league)
    team_key = find_user_team_key(teams, config.team_name)

    projections_root = _PROJECT_ROOT / "data" / "projections"
    return run_summary(config, projections_root, api_key=api_key, league=league, team_key=team_key)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run it to verify it passes**

Run: `pytest tests/test_scripts/test_send_daily_summary.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/send_daily_summary.py tests/test_scripts/test_send_daily_summary.py
git commit -m "feat(summary): orchestrator script with freshness gate + snapshot (#200)"
```

---

### Task 12: Full-suite verification + Render cron documentation

**Files:**
- Modify: `docs/superpowers/specs/2026-07-11-daily-summary-email-design.md` (append an "Operations runbook" note) OR create `docs/daily-summary-runbook.md`
- No new source.

- [ ] **Step 1: Run the whole suite**

Run: `pytest -n auto`
Expected: all pass. Fix any failure in the code (not the tests).

- [ ] **Step 2: Lint + format + dead code + types**

Run each and fix every finding:
```bash
ruff check .
ruff format --check .
vulture
mypy
```
Expected: zero violations. (If `mypy` flags the `resend` SDK as untyped, add `[[tool.mypy.overrides]] module = "resend.*"` with `ignore_missing_imports = true` in `pyproject.toml`.)

- [ ] **Step 3: Write the runbook**

Create `docs/daily-summary-runbook.md` documenting:
- The Render cron: command `python scripts/send_daily_summary.py`, schedule a UTC time shortly after the morning refresh cron (state the exact refresh cron time and pick +15 min).
- Required env vars on the cron: `RESEND_API_KEY`, `YAHOO_OAUTH_JSON`, `UPSTASH_REDIS_REST_URL`, `UPSTASH_REDIS_REST_TOKEN` (and `RENDER=true`, which the script also sets).
- Resend setup: verify the `from_address` domain in Resend; put recipients in `config/league.yaml` `summary.recipients`.
- Failure semantics: non-zero exit if the refresh didn't run (freshness gate) or Resend errors -- the cron surfaces it.
- Manual local test: `RENDER=true RESEND_API_KEY=... python scripts/send_daily_summary.py` (reads Upstash; sends a real email).

- [ ] **Step 4: Commit**

```bash
git add docs/daily-summary-runbook.md pyproject.toml
git commit -m "docs(summary): operations runbook + final verification (#200)"
```

---

## Self-Review

**Spec coverage:**
- Delivery via Resend -> Task 10 + Task 1 dep. [x]
- Module structure (models/assemble/render/send + script) -> Tasks 2, 8, 9, 10, 11. [x]
- `DailySummary` frozen dataclass, per-section models -> Task 2. [x]
- Last-night with type-keyed crosswalk + group derivation + unmatched -> Tasks 3, 4. [x]
- Streaks (duckdb-free, hitters-only, single window) -> Task 5. [x]
- Standings delta (recompute via score_roto, rate-cat caveat, first-run) -> Task 7. [x]
- Lineup moves / injuries (injury_note) / probables (required=False absent) -> Task 6. [x]
- Standings snapshot key + write-after-send -> Tasks 1, 11. [x]
- Whole-email freshness gate on META.last_refresh -> Tasks 8, 11. [x]
- Config `summary` block + secrets -> Tasks 1, 12. [x]
- Per-builder error isolation (section_errors) -> Task 8. [x]
- Render omits empty, notes errors/unmatched/first-run -> Task 9. [x]
- Scheduling / Render cron / env vars -> Task 12. [x]
- ASCII/UTF-8, RENDER=true before imports, no `x or default` -> Global Constraints + Task 11. [x]

**Placeholder scan:** No "TBD"/"similar to Task N"/bare "add error handling" -- every code step has complete code. [x]

**Type consistency:** `build_last_night` returns `(list[PlayerLine], list[str])` (Task 4) consumed as such in Task 8; `build_standings_delta(current_raw, snapshot_payload, user_team_name)` signature matches Task 7 <-> Task 8; `send_email(**kwargs)` keyword signature matches Task 10 <-> Task 11; `refresh_is_fresh(meta, today)` matches Task 8 <-> Task 11; `DailySummary` field names consistent across Tasks 2, 8, 9, 11. [x]
