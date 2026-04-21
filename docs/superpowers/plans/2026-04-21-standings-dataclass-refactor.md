# Standings Dataclass Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Collapse the standings web pipeline (Yahoo fetch → cache → Flask routes → templates → trade/evaluate) and the Redis persistence format onto one typed representation (`Standings` / `ProjectedStandings`), with strict `Category`-enum access everywhere and no bare-string category keys anywhere in Python.

**Architecture:** Five phases, each commit ≤5 files. Phase 1 introduces new types alongside the old ones and flips `Category` from `StrEnum` to `Enum`. Phases 2–4 migrate construction, consumers, and the Flask/Jinja boundary. Phase 5 runs a one-shot Redis rewrite, deletes the legacy shape parser, and removes `StandingsSnapshot`/`build_projected_standings`/dict-compat entirely.

**Tech Stack:** Python 3.11 (dataclasses, `Enum`, typing), Flask + Jinja2, Upstash Redis (via `redis_store`), pytest, mypy, ruff, vulture.

**Branch:** `standings-dataclass-refactor` (already created, contains the design spec commit).

**Spec:** `docs/superpowers/specs/2026-04-21-standings-dataclass-design.md`

---

## File Structure

### New files

- `scripts/migrate_standings_history.py` — one-shot Redis rewrite (Phase 5).

### Heavily modified

- `src/fantasy_baseball/utils/constants.py` — `Category(StrEnum)` → `Category(Enum)`.
- `src/fantasy_baseball/models/standings.py` — new `Standings`, `ProjectedStandings`, `CategoryPoints`; rewritten `CategoryStats`; `StandingsSnapshot` deleted in Phase 5.
- `src/fantasy_baseball/models/league.py` — `from_redis()` delegates to `Standings.from_json`; `League.standings` type changes to `list[Standings]`.
- `src/fantasy_baseball/data/redis_store.py` — `write_standings_snapshot`, `get_standings_day`, `get_latest_standings`, `get_standings_history` move to typed objects.
- `src/fantasy_baseball/lineup/yahoo_roster.py` — `fetch_standings()` returns `Standings`.
- `src/fantasy_baseball/scoring.py` — `score_roto()` returns `dict[str, CategoryPoints]`; `build_projected_standings` deleted in Phase 5.
- `src/fantasy_baseball/trades/evaluate.py` — `compute_roto_points_by_cat`, `compute_roto_points`, `compute_trade_impact`, `search_trades_away`, `search_trades_for` take typed objects.
- `src/fantasy_baseball/lineup/leverage.py` — rename `StandingsSnapshot` → `Standings`.
- `src/fantasy_baseball/web/season_data.py` — `_standings_to_snapshot` deleted; `format_standings_for_display`, `_compute_color_intensity`, `compute_comparison_standings`, `get_teams_list` take typed objects.
- `src/fantasy_baseball/web/season_routes.py` — `/standings` and related routes pass `Standings` to templates; context gains `all_categories`.
- `src/fantasy_baseball/web/refresh_pipeline.py` — passthrough sites carry typed objects.
- `src/fantasy_baseball/web/templates/season/standings.html` — loops use `Category` enum.

### Tests modified

- `tests/test_models/test_standings.py` — dict-compat tests rewritten to expect `TypeError` on string keys.
- `tests/test_data/test_redis_store_standings.py` — fixtures updated to canonical shape; add legacy-shape round-trip rejection.
- `tests/test_models/test_league.py` — `from_redis` fixtures migrated.
- `tests/test_trades/test_evaluate.py` (and neighbors) — signature updates.
- `tests/test_web/test_season_data.py` (and neighbors) — signature updates.
- New: `tests/test_scripts/test_migrate_standings_history.py`.

---

## Phase 0 — Cleanup Commit

Per CLAUDE.md "Step 0 Rule": remove dead code before the structural work so later phases don't propagate it.

### Task 0.1: Scope-limited cleanup

**Files:**
- Modify (only if findings): files under `src/fantasy_baseball/models/`, `src/fantasy_baseball/data/redis_store.py`, `src/fantasy_baseball/lineup/yahoo_roster.py`, `src/fantasy_baseball/lineup/leverage.py`, `src/fantasy_baseball/trades/evaluate.py`, `src/fantasy_baseball/web/season_data.py`, `src/fantasy_baseball/web/season_routes.py`, `src/fantasy_baseball/web/refresh_pipeline.py`, `src/fantasy_baseball/scoring.py`.

- [ ] **Step 1: Run ruff unused-import / sorting check over the affected files**

Run:
```bash
ruff check --select F,I src/fantasy_baseball/models/ src/fantasy_baseball/data/redis_store.py src/fantasy_baseball/lineup/yahoo_roster.py src/fantasy_baseball/lineup/leverage.py src/fantasy_baseball/trades/evaluate.py src/fantasy_baseball/web/season_data.py src/fantasy_baseball/web/season_routes.py src/fantasy_baseball/web/refresh_pipeline.py src/fantasy_baseball/scoring.py
```

Expected: either "All checks passed!" or a small list of findings.

- [ ] **Step 2: Apply fixes**

Run:
```bash
ruff check --select F,I --fix src/fantasy_baseball/
```

Review the diff; reject anything unrelated to F/I lints.

- [ ] **Step 3: Run vulture on the same scope**

Run:
```bash
vulture src/fantasy_baseball/models/ src/fantasy_baseball/data/redis_store.py src/fantasy_baseball/lineup/yahoo_roster.py src/fantasy_baseball/lineup/leverage.py src/fantasy_baseball/trades/evaluate.py src/fantasy_baseball/web/season_data.py src/fantasy_baseball/web/season_routes.py src/fantasy_baseball/web/refresh_pipeline.py src/fantasy_baseball/scoring.py
```

Delete anything the tool reports that is genuinely unused (read the finding; `vulture` can false-positive on Flask routes and `__init__` params — leave those). Do **not** delete code reachable from tests.

- [ ] **Step 4: Verify nothing broke**

Run:
```bash
pytest -v
```

Expected: all pre-existing tests pass. If any test now fails, your delete was wrong — restore it.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "chore(pre-refactor): remove dead imports and unused code in standings-touching modules

Per CLAUDE.md Step 0 rule — clean slate before the dataclass refactor.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

If there were no findings, skip the commit and proceed to Phase 1.

---

## Phase 1 — Typed Scaffolding

Introduce the new types alongside the old `StandingsSnapshot`, flip `Category` to `Enum`, and fix any fallout from the enum change. At the end of Phase 1, both old and new types exist; no consumers have migrated yet.

### Task 1.1: Flip `Category` from `StrEnum` to `Enum`

**Files:**
- Modify: `src/fantasy_baseball/utils/constants.py`

- [ ] **Step 1: Edit the enum definition**

In `src/fantasy_baseball/utils/constants.py`, change:

```python
from enum import StrEnum


class Category(StrEnum):
    """Roto scoring category.

    ``StrEnum`` members *are* strings, so existing code that compares
    category values to bare strings (``cat == "HR"``, ``cat in {"ERA",
    "WHIP"}``, dict lookups keyed on ``"R"``) continues to work
    unchanged. New code should prefer the enum members for type safety.
    """

    R = "R"
    HR = "HR"
    ...
```

to:

```python
from enum import Enum


class Category(Enum):
    """Roto scoring category.

    Plain ``Enum`` (not ``StrEnum``): members are not strings. Compare
    against ``Category`` members directly (``cat == Category.HR``), and
    use ``.value`` at I/O boundaries when you need the uppercase
    string form for JSON/Redis.
    """

    R = "R"
    HR = "HR"
    ...
```

Leave `ALL_CATEGORIES`, `HITTING_CATEGORIES`, `PITCHING_CATEGORIES`, `RATE_STATS`, `INVERSE_STATS`, and `DEFAULT_SGP_DENOMINATORS` unchanged — their types and values don't shift.

- [ ] **Step 2: Run the full test suite and enumerate fallout**

Run:
```bash
pytest -v 2>&1 | tail -80
```

Expected: some tests fail because `cat == "HR"` or `cat in {"ERA", "WHIP"}` or `stats["R"]` (when `"R"` is compared to a `Category`) silently returned `True` under `StrEnum` and now returns `False`. Write the list of failing tests down in your scratchpad — you'll fix them in Step 3.

- [ ] **Step 3: Grep for every string-vs-Category comparison and typed-dict lookup**

Run:
```bash
grep -rn 'cat == "' src/ tests/ scripts/
grep -rn '"R"\|"HR"\|"RBI"\|"SB"\|"AVG"\|"W"\|"K"\|"SV"\|"ERA"\|"WHIP"' src/fantasy_baseball/ | grep -v "test_\|yahoo_stat_id"
```

For each hit that genuinely compares a `Category` value against a bare string literal or uses a bare string as a key in a dict that's also indexed by `Category`, replace the string with the enum member. Examples:

- `cat == "HR"` → `cat == Category.HR`
- `cat in {"ERA", "WHIP"}` → `cat in INVERSE_STATS`
- `stats[cat]` when `cat: Category` — if the stats object is a plain dict keyed by strings, switch the dict to `dict[Category, float]` at the call site (fix each individually — most production dicts are already Category-keyed via `ALL_CATEGORIES` iteration).

Do **not** change I/O boundaries (Redis JSON, Yahoo stat_id_map, cache JSON) — those correctly use strings. Only fix in-memory Python code.

- [ ] **Step 4: Re-run tests until green**

Run:
```bash
pytest -v
```

Iterate until all pre-existing tests pass. If a test has expectations that were only true under `StrEnum` (e.g., `assert Category.HR == "HR"`), update it to `assert Category.HR.value == "HR"` — that's the correct enum semantics.

- [ ] **Step 5: Lint**

Run:
```bash
ruff check . && ruff format --check .
```

Expected: both pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(category): Category is a plain Enum, not StrEnum

Strings no longer compare equal to Category members. Every site that
used \`cat == \"HR\"\` or keyed a dict with bare string literals is now
fixed to use Category enum members. I/O boundaries (Redis payloads,
Yahoo stat_id_map, cache JSON) still use strings via .value — only
in-memory Python code shifts.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 1.2: Rewrite `CategoryStats` with typed `__getitem__`

**Files:**
- Modify: `src/fantasy_baseball/models/standings.py`
- Test: `tests/test_models/test_standings.py`

- [ ] **Step 1: Rewrite the failing-test cases in `tests/test_models/test_standings.py`**

Replace the `test_getitem_compat_uppercase`, `test_get_with_default`, `test_getitem_unknown_raises`, and `test_items_yields_all_categories` tests with:

```python
class TestCategoryStatsTypedAccess:
    def test_getitem_accepts_category_enum(self):
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.utils.constants import Category

        stats = CategoryStats(r=100, hr=40, era=3.5)
        assert stats[Category.R] == 100
        assert stats[Category.HR] == 40
        assert stats[Category.ERA] == pytest.approx(3.5)

    def test_getitem_rejects_bare_string(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats(r=100)
        with pytest.raises(TypeError, match="Category enum"):
            _ = stats["R"]

    def test_getitem_rejects_other_types(self):
        from fantasy_baseball.models.standings import CategoryStats

        stats = CategoryStats()
        with pytest.raises(TypeError, match="Category enum"):
            _ = stats[0]

    def test_items_yields_category_enums(self):
        from fantasy_baseball.models.standings import CategoryStats
        from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category

        stats = CategoryStats(r=100, hr=40, rbi=120, sb=15, avg=0.280,
                              w=50, k=700, sv=20, era=3.9, whip=1.20)
        items = list(stats.items())
        assert [k for k, _ in items] == ALL_CATEGORIES
        as_map = dict(items)
        assert as_map[Category.R] == 100
        assert as_map[Category.HR] == 40
        assert as_map[Category.WHIP] == pytest.approx(1.20)
```

Leave `test_default_values`, `test_construction_with_values`, `test_from_dict`, and `test_from_dict_missing_keys_default` untouched — those still pass after the refactor.

- [ ] **Step 2: Run the new tests, watch them fail**

Run:
```bash
pytest tests/test_models/test_standings.py::TestCategoryStatsTypedAccess -v
```

Expected: tests fail — the current `__getitem__` accepts strings and returns floats; it doesn't raise `TypeError`.

- [ ] **Step 3: Rewrite `CategoryStats` in `src/fantasy_baseball/models/standings.py`**

Replace the entire `CategoryStats` block plus `CATEGORY_ORDER` and `_KEY_TO_FIELD` with:

```python
"""League-level roto category statistics.

Three snapshot-layer dataclasses used by League:
:class:`CategoryStats` (the ten roto totals), :class:`StandingsEntry`
(one team's stats + rank), and :class:`StandingsSnapshot` (all teams
at an effective_date).

``CategoryStats`` is keyed exclusively by :class:`Category` enum. Bare
string access raises ``TypeError``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, Iterator, Mapping

from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category


# Private: single source of truth for Category <-> attribute mapping.
_CAT_TO_FIELD: dict[Category, str] = {
    Category.R:    "r",
    Category.HR:   "hr",
    Category.RBI:  "rbi",
    Category.SB:   "sb",
    Category.AVG:  "avg",
    Category.W:    "w",
    Category.K:    "k",
    Category.SV:   "sv",
    Category.ERA:  "era",
    Category.WHIP: "whip",
}


@dataclass
class CategoryStats:
    r:    float = 0.0
    hr:   float = 0.0
    rbi:  float = 0.0
    sb:   float = 0.0
    avg:  float = 0.0
    w:    float = 0.0
    k:    float = 0.0
    sv:   float = 0.0
    era:  float = 99.0
    whip: float = 99.0

    def __getitem__(self, cat: Category) -> float:
        if not isinstance(cat, Category):
            raise TypeError(
                f"CategoryStats indexing requires a Category enum, got "
                f"{type(cat).__name__}"
            )
        return float(getattr(self, _CAT_TO_FIELD[cat]))

    def items(self) -> Iterator[tuple[Category, float]]:
        for cat in ALL_CATEGORIES:
            yield cat, float(getattr(self, _CAT_TO_FIELD[cat]))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CategoryStats":
        """Build from an UPPERCASE-string-keyed dict (I/O boundary only).

        Missing keys fall back to dataclass defaults (0 for counting
        stats, 99 for ERA/WHIP).
        """
        kwargs: dict[str, Any] = {}
        for cat in ALL_CATEGORIES:
            if cat.value in d:
                kwargs[_CAT_TO_FIELD[cat]] = float(d[cat.value])
        return cls(**kwargs)

    def to_dict(self) -> dict[str, float]:
        """Produce an UPPERCASE-string-keyed dict (I/O boundary only)."""
        return {
            cat.value: float(getattr(self, _CAT_TO_FIELD[cat]))
            for cat in ALL_CATEGORIES
        }
```

The existing `StandingsEntry` and `StandingsSnapshot` classes stay as they are for now — they move in Task 1.3.

- [ ] **Step 4: Run the targeted tests to verify they pass**

Run:
```bash
pytest tests/test_models/test_standings.py -v
```

Expected: every test passes. Any other test that used `stats["R"]` or `stats.get("R")` dict-compat now fails — write those down.

- [ ] **Step 5: Fix any remaining call sites that broke**

Run:
```bash
pytest -v 2>&1 | tail -120
```

For each failure: the fix is either
- change `stats["R"]` → `stats[Category.R]`, or
- (if the test asserts a deleted feature like `stats.get(...)`) rewrite the test to exercise the typed `__getitem__` instead.

Don't skip or xfail tests — fix the code or delete the test if it's asserting behaviour we explicitly deleted (e.g., `test_get_with_default`).

- [ ] **Step 6: Full verification**

Run:
```bash
pytest -v
ruff check .
ruff format --check .
mypy
```

Expected: all green.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(standings): typed CategoryStats.__getitem__(Category)

Drop string-keyed dict-compat (get/keys/__iter__/CATEGORY_ORDER).
__getitem__ now raises TypeError on anything that isn't a Category
enum. items() yields (Category, float) pairs. from_dict/to_dict stay
at the I/O boundary using .value for JSON keys.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 1.3: Add new standings types alongside the old

**Files:**
- Modify: `src/fantasy_baseball/models/standings.py`
- Test: `tests/test_models/test_standings.py`

- [ ] **Step 1: Write failing tests for the new types**

Append to `tests/test_models/test_standings.py`:

```python
class TestStandingsJSON:
    def _canonical_payload(self):
        return {
            "effective_date": "2026-04-15",
            "teams": [
                {
                    "name": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "yahoo_points_for": 78.5,
                    "stats": {
                        "R": 45, "HR": 12, "RBI": 40, "SB": 8, "AVG": 0.268,
                        "W": 3, "K": 85, "SV": 4, "ERA": 3.21, "WHIP": 1.14,
                    },
                },
            ],
        }

    def test_from_json_canonical_round_trip(self):
        from fantasy_baseball.models.standings import Standings
        payload = self._canonical_payload()
        s = Standings.from_json(payload)
        assert s.effective_date == date(2026, 4, 15)
        assert len(s.entries) == 1
        e = s.entries[0]
        assert e.team_name == "Alpha"
        assert e.team_key == "431.l.1.t.1"
        assert e.rank == 1
        assert e.yahoo_points_for == 78.5
        assert e.stats.r == 45
        assert e.stats.whip == pytest.approx(1.14)
        assert s.to_json() == payload

    def test_from_json_rejects_legacy_shape(self):
        from fantasy_baseball.models.standings import Standings
        legacy = {
            "teams": [
                {
                    "team": "Alpha",
                    "team_key": "431.l.1.t.1",
                    "rank": 1,
                    "r": 45, "hr": 12, "rbi": 40, "sb": 8, "avg": 0.268,
                    "w": 3, "k": 85, "sv": 4, "era": 3.21, "whip": 1.14,
                },
            ],
        }
        with pytest.raises(ValueError, match="legacy|unknown|name"):
            Standings.from_json(legacy)


class TestProjectedStandingsJSON:
    def test_round_trip(self):
        from fantasy_baseball.models.standings import (
            CategoryStats, ProjectedStandings, ProjectedStandingsEntry,
        )
        ps = ProjectedStandings(
            effective_date=date(2026, 4, 15),
            entries=[
                ProjectedStandingsEntry(
                    team_name="Alpha",
                    stats=CategoryStats(r=600, hr=250, era=3.8, whip=1.18),
                ),
            ],
        )
        round_tripped = ProjectedStandings.from_json(ps.to_json())
        assert round_tripped == ps


class TestCategoryPoints:
    def test_getitem_and_total(self):
        from fantasy_baseball.models.standings import CategoryPoints
        from fantasy_baseball.utils.constants import Category

        cp = CategoryPoints(
            values={Category.R: 7.0, Category.HR: 4.5},
            total=11.5,
        )
        assert cp[Category.R] == 7.0
        assert cp[Category.HR] == 4.5
        assert cp.total == 11.5

    def test_getitem_rejects_string(self):
        from fantasy_baseball.models.standings import CategoryPoints
        cp = CategoryPoints(values={}, total=0.0)
        with pytest.raises(TypeError, match="Category"):
            _ = cp["R"]
```

- [ ] **Step 2: Run the new tests, watch them fail**

Run:
```bash
pytest tests/test_models/test_standings.py::TestStandingsJSON tests/test_models/test_standings.py::TestProjectedStandingsJSON tests/test_models/test_standings.py::TestCategoryPoints -v
```

Expected: `ImportError` / `AttributeError` — the new classes don't exist yet.

- [ ] **Step 3: Add the new types to `src/fantasy_baseball/models/standings.py`**

After the existing `CategoryStats` block, **before** `StandingsEntry`, add:

```python
@dataclass
class CategoryPoints:
    """Per-category roto points plus total, for one team.

    Replaces the ``{"R_pts": ..., "HR_pts": ..., "total": ...}`` dict
    returned by the old ``score_roto``. ``values`` is the per-category
    map; ``total`` is the sum of ``values`` by default, but ``score_roto``
    may override it with ``yahoo_points_for`` when the display layer
    needs an exact match with Yahoo's official standings page.
    """

    values: dict[Category, float]
    total: float

    def __getitem__(self, cat: Category) -> float:
        if not isinstance(cat, Category):
            raise TypeError(
                f"CategoryPoints indexing requires a Category enum, got "
                f"{type(cat).__name__}"
            )
        return self.values[cat]
```

After the existing `StandingsEntry` / `StandingsSnapshot` definitions, append:

```python
@dataclass
class Standings:
    """All teams' live standings at a single effective_date.

    ``entries`` carry real Yahoo ``team_key`` and ``rank`` (non-optional)
    plus ``yahoo_points_for`` when Yahoo has scored the week.
    """

    effective_date: date
    entries: list[StandingsEntry]

    def by_team(self) -> dict[str, StandingsEntry]:
        out: dict[str, StandingsEntry] = {}
        for entry in self.entries:
            if entry.team_name in out:
                raise ValueError(f"duplicate team in standings: {entry.team_name!r}")
            out[entry.team_name] = entry
        return out

    def sorted_by_rank(self) -> list[StandingsEntry]:
        return sorted(self.entries, key=lambda e: e.rank)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "Standings":
        """Canonical shape only: {'effective_date', 'teams': [{'name', ...}]}.

        Raises ``ValueError`` on legacy shapes ('team' instead of
        'name', lowercase stat keys, missing wrapper date).
        """
        if not isinstance(d, Mapping) or "teams" not in d or "effective_date" not in d:
            raise ValueError(
                f"Standings.from_json: not a canonical payload — "
                f"missing 'effective_date' or 'teams' wrapper (got keys: {sorted(d) if isinstance(d, Mapping) else type(d).__name__})"
            )
        eff = date.fromisoformat(d["effective_date"])
        entries: list[StandingsEntry] = []
        for row in d["teams"]:
            if "team" in row and "name" not in row:
                raise ValueError(
                    "Standings.from_json: legacy row shape detected ('team' field "
                    "instead of 'name') — run scripts/migrate_standings_history.py"
                )
            if "stats" not in row:
                raise ValueError(
                    f"Standings.from_json: row missing 'stats' wrapper "
                    f"(likely legacy flat-lowercase shape): {row.get('name')!r}"
                )
            entries.append(StandingsEntry(
                team_name=row["name"],
                team_key=row["team_key"],
                rank=int(row["rank"]),
                stats=CategoryStats.from_dict(row["stats"]),
                yahoo_points_for=row.get("yahoo_points_for"),
            ))
        return cls(effective_date=eff, entries=entries)

    def to_json(self) -> dict[str, Any]:
        return {
            "effective_date": self.effective_date.isoformat(),
            "teams": [
                {
                    "name": e.team_name,
                    "team_key": e.team_key,
                    "rank": e.rank,
                    "yahoo_points_for": e.yahoo_points_for,
                    "stats": e.stats.to_dict(),
                }
                for e in self.entries
            ],
        }


@dataclass
class ProjectedStandingsEntry:
    team_name: str
    stats: CategoryStats


@dataclass
class ProjectedStandings:
    effective_date: date
    entries: list[ProjectedStandingsEntry]

    def by_team(self) -> dict[str, ProjectedStandingsEntry]:
        out: dict[str, ProjectedStandingsEntry] = {}
        for entry in self.entries:
            if entry.team_name in out:
                raise ValueError(f"duplicate team in projected standings: {entry.team_name!r}")
            out[entry.team_name] = entry
        return out

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> "ProjectedStandings":
        if not isinstance(d, Mapping) or "teams" not in d or "effective_date" not in d:
            raise ValueError(
                "ProjectedStandings.from_json: missing 'effective_date' or 'teams'"
            )
        return cls(
            effective_date=date.fromisoformat(d["effective_date"]),
            entries=[
                ProjectedStandingsEntry(
                    team_name=row["name"],
                    stats=CategoryStats.from_dict(row["stats"]),
                )
                for row in d["teams"]
            ],
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "effective_date": self.effective_date.isoformat(),
            "teams": [
                {"name": e.team_name, "stats": e.stats.to_dict()}
                for e in self.entries
            ],
        }

    @classmethod
    def from_rosters(
        cls,
        team_rosters: Mapping[str, Any],
        effective_date: date,
    ) -> "ProjectedStandings":
        """Build from {team_name: roster_list} using project_team_stats."""
        from fantasy_baseball.scoring import project_team_stats

        return cls(
            effective_date=effective_date,
            entries=[
                ProjectedStandingsEntry(
                    team_name=tname,
                    stats=project_team_stats(roster, displacement=True),
                )
                for tname, roster in team_rosters.items()
            ],
        )
```

Leave `StandingsSnapshot` in place for now — it gets deleted in Phase 5.

- [ ] **Step 4: Run tests, verify they pass**

Run:
```bash
pytest tests/test_models/test_standings.py -v
```

Expected: every test passes.

- [ ] **Step 5: Full verification**

Run:
```bash
pytest -v
ruff check .
ruff format --check .
mypy
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(standings): add Standings, ProjectedStandings, CategoryPoints

New typed classes alongside the existing StandingsSnapshot. Standings
has non-optional team_key/rank and optional yahoo_points_for.
ProjectedStandings has no team_key/rank at all. Both get from_json /
to_json on the canonical payload shape; legacy shapes raise
ValueError. Old StandingsSnapshot stays in place for now.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase 2 — Construction Sites

Migrate every producer so downstream callers start receiving typed objects. Each producer keeps backward-compatibility for consumers that still want `list[dict]` only if unavoidable — prefer migrating consumers in Phase 3 instead.

### Task 2.1: `fetch_standings()` returns `Standings`

**Files:**
- Modify: `src/fantasy_baseball/lineup/yahoo_roster.py`
- Test: `tests/test_lineup/test_yahoo_roster.py` (or wherever `parse_standings_raw` tests live — grep to find them)

- [ ] **Step 1: Locate the existing test fixture for `parse_standings_raw`**

Run:
```bash
grep -rn "parse_standings_raw\|fetch_standings" tests/
```

If there's no dedicated test, skip to Step 2.

- [ ] **Step 2: Write a new test asserting `parse_standings_raw` returns `Standings`**

In the test file (create `tests/test_lineup/test_yahoo_roster.py` if needed), add:

```python
from datetime import date
import pytest

from fantasy_baseball.lineup.yahoo_roster import parse_standings_raw, YAHOO_STAT_ID_MAP
from fantasy_baseball.models.standings import Standings


def test_parse_standings_raw_returns_standings():
    raw = {
        "fantasy_content": {
            "league": [
                {},
                {
                    "standings": [{
                        "teams": {
                            "0": {"team": [
                                [
                                    {"team_key": "431.l.1.t.1"},
                                    {"name": "Alpha"},
                                ],
                                {
                                    "team_standings": {"rank": "1", "points_for": "42.0"},
                                    "team_stats": {
                                        "stats": [
                                            {"stat": {"stat_id": "7",  "value": "45"}},
                                            {"stat": {"stat_id": "12", "value": "12"}},
                                        ]
                                    },
                                },
                            ]},
                            "count": 1,
                        },
                    }],
                },
            ],
        },
    }
    result = parse_standings_raw(raw, YAHOO_STAT_ID_MAP, effective_date=date(2026, 4, 15))
    assert isinstance(result, Standings)
    assert result.effective_date == date(2026, 4, 15)
    assert len(result.entries) == 1
    e = result.entries[0]
    assert e.team_name == "Alpha"
    assert e.team_key == "431.l.1.t.1"
    assert e.rank == 1
    assert e.yahoo_points_for == 42.0
    assert e.stats.r == 45
    assert e.stats.hr == 12
```

- [ ] **Step 3: Run the test, watch it fail**

Run:
```bash
pytest tests/test_lineup/test_yahoo_roster.py -v
```

Expected: fails because `parse_standings_raw` returns `list[dict]` and doesn't accept `effective_date`.

- [ ] **Step 4: Update `parse_standings_raw` and `fetch_standings`**

In `src/fantasy_baseball/lineup/yahoo_roster.py`:

- Change `parse_standings_raw` signature to:

```python
def parse_standings_raw(
    raw: dict,
    stat_id_map: dict[str, str],
    *,
    effective_date: date,
) -> Standings:
```

- Replace the final `return [t.to_dict() for t in teams]` with:

```python
    from fantasy_baseball.models.standings import CategoryStats, Standings, StandingsEntry

    entries = [
        StandingsEntry(
            team_name=t.name,
            team_key=t.team_key,
            rank=t.rank,
            stats=CategoryStats.from_dict(t.stats),
            yahoo_points_for=t.points_for,
        )
        for t in teams
    ]
    return Standings(effective_date=effective_date, entries=entries)
```

- Update `fetch_standings` to:

```python
def fetch_standings(league, effective_date: date) -> Standings:
    """Fetch league standings with cumulative roto stats."""
    raw = league.yhandler.get_standings_raw(league.league_id)
    return parse_standings_raw(raw, YAHOO_STAT_ID_MAP, effective_date=effective_date)
```

- Add the needed imports at the top of the file: `from datetime import date` and `from fantasy_baseball.models.standings import Standings`.

- [ ] **Step 5: Fix all call sites of `fetch_standings`**

Run:
```bash
grep -rn "fetch_standings(" src/ scripts/ tests/
```

For every caller, pass `effective_date` (usually `local_today()` for live pulls; for Tuesday-locked snapshots use the existing lineup-lock date). The callers that currently do `standings_list = fetch_standings(league)` and pass `standings_list` onward will keep working *structurally* only after the Phase 3 consumer migrations — expect test regressions; they get fixed in Phase 3. For now, keep this commit scoped to the producer.

If a caller *within Phase 2's scope* needs something immediately, use `.to_json()["teams"]` as a temporary bridge back to list-of-dicts — but prefer moving the caller in its own Task in Phase 3 instead.

- [ ] **Step 6: Run targeted tests**

Run:
```bash
pytest tests/test_lineup/test_yahoo_roster.py -v
```

Expected: pass.

- [ ] **Step 7: Full verification (expect some failures from Phase 3 scope)**

Run:
```bash
pytest -v 2>&1 | tail -30
ruff check .
mypy src/fantasy_baseball/lineup/yahoo_roster.py
```

Note the test failures — they should all trace back to callers that haven't been migrated yet (season_routes, refresh_pipeline, etc.). If any failure is unrelated, stop and fix it. Record the expected-to-fail list before proceeding.

- [ ] **Step 8: Commit**

```bash
git add -A
git commit -m "refactor(yahoo_roster): fetch_standings/parse_standings_raw return Standings

Both now take an effective_date and emit a typed Standings object
(not list[dict]). Consumers migrate in Phase 3; for now some
downstream tests fail — tracked for Phase 3.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 2.2: `ProjectedStandings.from_rosters()` replaces `build_projected_standings`

**Files:**
- Modify: `src/fantasy_baseball/scoring.py`
- Test: `tests/test_scoring/test_scoring.py` (grep to find the existing test)

- [ ] **Step 1: Find the existing test**

Run:
```bash
grep -rn "build_projected_standings" tests/
```

- [ ] **Step 2: Write (or update) a test asserting the wrapper returns the same ProjectedStandings**

In the test file:

```python
def test_build_projected_standings_returns_projected_standings():
    from datetime import date
    from fantasy_baseball.scoring import build_projected_standings
    from fantasy_baseball.models.standings import ProjectedStandings

    team_rosters = {"Alpha": [], "Beta": []}  # empty rosters => zero stats
    result = build_projected_standings(team_rosters, effective_date=date(2026, 4, 15))
    assert isinstance(result, ProjectedStandings)
    assert {e.team_name for e in result.entries} == {"Alpha", "Beta"}
```

- [ ] **Step 3: Run, watch it fail**

- [ ] **Step 4: Rewrite `build_projected_standings` as a thin wrapper**

In `src/fantasy_baseball/scoring.py`, replace:

```python
def build_projected_standings(team_rosters: dict[str, list]) -> list[dict]:
    ...
    return [...]
```

with:

```python
def build_projected_standings(
    team_rosters: dict[str, list],
    *,
    effective_date: date,
) -> ProjectedStandings:
    """Thin wrapper around ProjectedStandings.from_rosters (kept for the
    duration of the migration; deleted in Phase 5)."""
    from fantasy_baseball.models.standings import ProjectedStandings
    return ProjectedStandings.from_rosters(team_rosters, effective_date=effective_date)
```

Add `from datetime import date` at the top if not present.

- [ ] **Step 5: Run targeted tests**

Run:
```bash
pytest tests/test_scoring/ -v
```

Expected: the new test passes; other scoring tests may fail where they assumed `list[dict]` (Phase 3 fixes those).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(scoring): build_projected_standings returns ProjectedStandings

Thin wrapper delegating to ProjectedStandings.from_rosters. Old
list[dict] return is gone. Consumer migrations in Phase 3 update
the call sites that fed this into the projections cache.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 2.3: Redis read/write uses `Standings.to_json` / `from_json`

**Files:**
- Modify: `src/fantasy_baseball/data/redis_store.py`
- Test: `tests/test_data/test_redis_store_standings.py`

- [ ] **Step 1: Rewrite the test fixtures in canonical shape**

In `tests/test_data/test_redis_store_standings.py`, replace the top of the file with:

```python
"""Tests for standings_history helpers."""
from datetime import date

import pytest

from fantasy_baseball.data import redis_store
from fantasy_baseball.models.standings import (
    CategoryStats,
    Standings,
    StandingsEntry,
)


def _standings(eff: date, teams: list[tuple[str, str, int, dict, float | None]]) -> Standings:
    return Standings(
        effective_date=eff,
        entries=[
            StandingsEntry(
                team_name=name,
                team_key=team_key,
                rank=rank,
                stats=CategoryStats.from_dict(stats),
                yahoo_points_for=pf,
            )
            for name, team_key, rank, stats, pf in teams
        ],
    )


STANDINGS_DAY_1 = _standings(
    date(2026, 4, 15),
    [
        ("Alpha", "431.l.1.t.1", 1, {
            "R": 45, "HR": 12, "RBI": 40, "SB": 8, "AVG": 0.268,
            "W": 3, "K": 85, "SV": 4, "ERA": 3.21, "WHIP": 1.14,
        }, 78.5),
        ("Beta", "431.l.1.t.2", 2, {
            "R": 38, "HR": 9, "RBI": 32, "SB": 6, "AVG": 0.255,
            "W": 2, "K": 72, "SV": 3, "ERA": 3.85, "WHIP": 1.22,
        }, 60.0),
    ],
)

STANDINGS_DAY_2 = _standings(
    date(2026, 4, 22),
    [
        ("Alpha", "431.l.1.t.1", 1, {
            "R": 60, "HR": 16, "RBI": 55, "SB": 10, "AVG": 0.272,
            "W": 5, "K": 110, "SV": 5, "ERA": 3.05, "WHIP": 1.10,
        }, 82.0),
    ],
)
```

Rewrite each `test_*` function to call `redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_1)` (no explicit date arg — the date comes from `Standings.effective_date`) and assert the returned object is a `Standings` equal to the input. Full shape:

```python
def test_write_and_read_single_day(fake_redis):
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_1)
    loaded = redis_store.get_standings_day(fake_redis, "2026-04-15")
    assert loaded == STANDINGS_DAY_1


def test_write_standings_snapshot_overwrites_same_date(fake_redis):
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_1)
    same_date_new_content = _standings(date(2026, 4, 15), [("Alpha", "431.l.1.t.1", 1, {"R": 99}, 99.0)])
    redis_store.write_standings_snapshot(fake_redis, same_date_new_content)
    loaded = redis_store.get_standings_day(fake_redis, "2026-04-15")
    assert loaded == same_date_new_content


def test_get_latest_standings_picks_max_date(fake_redis):
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_1)
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_2)
    latest = redis_store.get_latest_standings(fake_redis)
    assert latest == STANDINGS_DAY_2


def test_get_standings_history_returns_all_dates(fake_redis):
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_1)
    redis_store.write_standings_snapshot(fake_redis, STANDINGS_DAY_2)
    history = redis_store.get_standings_history(fake_redis)
    assert set(history.keys()) == {"2026-04-15", "2026-04-22"}
    assert history["2026-04-22"] == STANDINGS_DAY_2


def test_get_standings_history_empty(fake_redis):
    assert redis_store.get_standings_history(fake_redis) == {}


def test_write_standings_snapshot_none_client_noop():
    redis_store.write_standings_snapshot(None, STANDINGS_DAY_1)


def test_get_latest_standings_none_client_returns_none():
    assert redis_store.get_latest_standings(None) is None


def test_get_standings_day_none_client_returns_none():
    assert redis_store.get_standings_day(None, "2026-04-15") is None


def test_get_standings_history_none_client_returns_empty():
    assert redis_store.get_standings_history(None) == {}


def test_get_standings_day_ignores_corrupt_json(fake_redis):
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", "not json {{{")
    assert redis_store.get_standings_day(fake_redis, "2026-04-15") is None


def test_get_standings_history_raises_on_legacy_shape(fake_redis):
    import json
    legacy = {"teams": [{"team": "Alpha", "r": 10}]}
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", json.dumps(legacy))
    with pytest.raises(ValueError):
        redis_store.get_standings_day(fake_redis, "2026-04-15")
```

- [ ] **Step 2: Run tests, watch them fail**

Run:
```bash
pytest tests/test_data/test_redis_store_standings.py -v
```

Expected: fails — current functions take/return dicts, not `Standings`.

- [ ] **Step 3: Rewrite `redis_store.py` standings functions**

Replace the four functions in `src/fantasy_baseball/data/redis_store.py` (around line 336):

```python
STANDINGS_HISTORY_KEY = "standings_history"


def write_standings_snapshot(client, standings: Standings) -> None:
    """Write a Standings snapshot keyed by its effective_date. Idempotent overwrite.

    Canonical shape on disk: ``standings.to_json()`` — see spec. No-op
    when ``client`` is None.
    """
    if client is None:
        return
    client.hset(
        STANDINGS_HISTORY_KEY,
        standings.effective_date.isoformat(),
        json.dumps(standings.to_json()),
    )


def get_standings_day(client, snapshot_date: str) -> Standings | None:
    """Return the Standings for one snapshot date, or None if missing/corrupt.

    Raises ValueError if the stored payload is legacy-shape (see
    ``Standings.from_json``); run scripts/migrate_standings_history.py
    to rewrite.
    """
    if client is None:
        return None
    raw = client.hget(STANDINGS_HISTORY_KEY, snapshot_date)
    if raw is None:
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    return Standings.from_json(data)


def get_latest_standings(client) -> Standings | None:
    """Return the Standings for the maximum snapshot_date in the hash."""
    if client is None:
        return None
    dates = client.hkeys(STANDINGS_HISTORY_KEY)
    if not dates:
        return None
    return get_standings_day(client, max(dates))


def get_standings_history(client) -> dict[str, Standings]:
    """Return the entire history as {snapshot_date: Standings}.

    Corrupt JSON entries are silently skipped (matches previous behavior).
    Legacy-shape entries raise ValueError — by design; migration script
    rewrites them.
    """
    if client is None:
        return {}
    raw_map = client.hgetall(STANDINGS_HISTORY_KEY)
    if not raw_map:
        return {}
    out: dict[str, Standings] = {}
    for d, raw in raw_map.items():
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            out[d] = Standings.from_json(data)
    return out
```

Add the import at the top of the file: `from fantasy_baseball.models.standings import Standings`.

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_data/test_redis_store_standings.py -v
```

Expected: all pass.

- [ ] **Step 5: Full verification**

Run:
```bash
pytest -v 2>&1 | tail -40
ruff check .
mypy src/fantasy_baseball/data/redis_store.py
```

Expected: redis_store tests and any test that only touches these functions in isolation pass. Downstream tests that relied on dict returns may fail — they get fixed in Phase 3.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(redis_store): standings helpers use typed Standings

write_standings_snapshot / get_standings_day / get_latest_standings /
get_standings_history now take/return Standings objects. Canonical
shape on disk (UPPERCASE stat keys, 'name' not 'team',
effective_date wrapper). Legacy-shape reads raise ValueError.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 2.4: `League.from_redis()` delegates to `Standings.from_json`

**Files:**
- Modify: `src/fantasy_baseball/models/league.py`
- Test: `tests/test_models/test_league.py`

- [ ] **Step 1: Grep existing test fixtures for the Redis round-trip**

Run:
```bash
grep -rn "from_redis\|standings_history" tests/test_models/
```

- [ ] **Step 2: Update the test fixture(s) to the canonical shape**

In `tests/test_models/test_league.py`, any fixture that looks like `{"teams": [{"team": ...}]}` must become `{"effective_date": "YYYY-MM-DD", "teams": [{"name": ...}]}` with UPPERCASE stat keys. Use `Standings.from_json` / `to_json` for symmetry if convenient, or hard-code the canonical payload inline.

If tests write to `fake_redis` via `redis_store.write_standings_snapshot`, no change needed — the Task 2.3 edits already adapt that call.

- [ ] **Step 3: Rewrite `League.from_redis` to call `Standings.from_json`**

In `src/fantasy_baseball/models/league.py`, replace the `snapshots_by_date` / `team_key_by_name` construction block (lines ~113-142) with:

```python
        snapshots_by_date: dict[str, Standings] = {}
        team_key_by_name: dict[str, str] = {}
        for snap_date in sorted(all_standings.keys()):
            if not snap_date.startswith(prefix):
                continue
            payload = all_standings[snap_date]
            if isinstance(payload, Standings):
                standings = payload
            else:
                standings = Standings.from_json(payload)
            snapshots_by_date[snap_date] = standings
            for entry in standings.entries:
                if entry.team_key:
                    team_key_by_name[entry.team_name] = entry.team_key
```

Update `_assemble` to accept `dict[str, Standings]` and produce `list[Standings]`:

```python
    @classmethod
    def _assemble(
        cls,
        season_year: int,
        by_team_snap: dict[str, dict[str, list[RosterEntry]]],
        snapshots_by_date: dict[str, Standings],
        team_key_by_name: dict[str, str],
    ) -> League:
        standings_list = [
            snapshots_by_date[k] for k in sorted(snapshots_by_date)
        ]
        ...
        return cls(
            season_year=season_year,
            teams=teams,
            standings=standings_list,
        )
```

Change the dataclass field: `standings: list[Standings] = field(default_factory=list)`. Update the import: `from fantasy_baseball.models.standings import Standings, StandingsEntry, CategoryStats` (drop `StandingsSnapshot`).

Update `latest_standings()` / `standings_as_of()` return types from `StandingsSnapshot` → `Standings`.

**Note:** `get_standings_history` now returns `dict[str, Standings]`, not `dict[str, dict]`. The `isinstance(payload, Standings)` fallback handles direct use; the canonical path is `Standings.from_json(payload)` but that won't be reached anymore once Task 2.3 is committed — keep the `Standings.from_json` branch anyway for safety against future signature changes. (Alternative: just call `standings = payload` with a type-assert; pick one consistent with what `get_standings_history` returns after Task 2.3.)

- [ ] **Step 4: Run targeted tests**

Run:
```bash
pytest tests/test_models/test_league.py -v
```

Expected: pass.

- [ ] **Step 5: Full verification**

Run:
```bash
pytest -v
ruff check .
mypy src/fantasy_baseball/models/league.py
```

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(league): from_redis() delegates to Standings.from_json

League.standings is now list[Standings]. Bespoke row-by-row
CategoryStats construction is deleted; parsing lives entirely on
Standings.from_json.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase 3 — Consumers

Migrate every reader. After this phase, the only remaining `list[dict]` producer is the stub wrapper `build_projected_standings` (kept for Phase 5 deletion); all other standings flow through typed objects.

### Task 3.1: `score_roto()` takes `TeamStatsTable`, returns `dict[str, CategoryPoints]`

**Files:**
- Modify: `src/fantasy_baseball/scoring.py`
- Test: `tests/test_scoring/test_scoring.py`

- [ ] **Step 1: Update the test to assert the new return type**

Find existing `score_roto` tests via:
```bash
grep -rn "score_roto" tests/
```

For each test, update assertions from `roto["Alpha"]["R_pts"]` to `roto["Alpha"][Category.R]` and `roto["Alpha"]["total"]` to `roto["Alpha"].total`. Keep the input-construction paths the same — we'll make `score_roto` accept the typed protocol in Step 3.

Example updated test snippet:

```python
from fantasy_baseball.models.standings import (
    CategoryStats, Standings, StandingsEntry,
)
from fantasy_baseball.scoring import score_roto
from fantasy_baseball.utils.constants import Category


def test_score_roto_returns_category_points():
    s = Standings(
        effective_date=date(2026, 4, 15),
        entries=[
            StandingsEntry("Alpha", "k1", 1, CategoryStats(r=120, hr=40)),
            StandingsEntry("Beta",  "k2", 2, CategoryStats(r=100, hr=35)),
        ],
    )
    roto = score_roto(s)
    assert roto["Alpha"][Category.R] == 2.0   # winner of 2-team league
    assert roto["Beta"][Category.R]  == 1.0
    assert roto["Alpha"].total > roto["Beta"].total
```

- [ ] **Step 2: Run, watch it fail**

- [ ] **Step 3: Rewrite `score_roto`**

In `src/fantasy_baseball/scoring.py`, add a `Protocol` import at the top:

```python
from typing import Protocol, Sequence

from fantasy_baseball.models.standings import CategoryPoints, CategoryStats
```

Define the protocols before `score_roto`:

```python
class TeamStatsRow(Protocol):
    team_name: str
    stats: CategoryStats


class TeamStatsTable(Protocol):
    entries: Sequence[TeamStatsRow]
```

Rewrite `score_roto`:

```python
def score_roto(
    standings: TeamStatsTable,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
) -> dict[str, CategoryPoints]:
    """Assign expected-value roto points per team per category.

    Accepts any object with a ``.entries`` sequence of ``(team_name,
    stats)``-shaped rows — concretely ``Standings`` and
    ``ProjectedStandings``.
    """
    teams = [e.team_name for e in standings.entries]
    stats_by_team = {e.team_name: e.stats for e in standings.entries}

    per_team_cat: dict[str, dict[Category, float]] = {t: {} for t in teams}

    for cat in ALL_CATS:
        higher_is_better = cat not in INVERSE_CATS
        for me in teams:
            mu_me = stats_by_team[me][cat]
            sd_me = team_sds.get(me, {}).get(cat, 0.0) if team_sds else 0.0
            pts = 1.0
            for other in teams:
                if other == me:
                    continue
                mu_o = stats_by_team[other][cat]
                sd_o = team_sds.get(other, {}).get(cat, 0.0) if team_sds else 0.0
                pts += _prob_beats(mu_me, mu_o, sd_me, sd_o, higher_is_better=higher_is_better)
            per_team_cat[me][cat] = pts

    return {
        t: CategoryPoints(
            values=per_team_cat[t],
            total=sum(per_team_cat[t].values()),
        )
        for t in teams
    }
```

Note: `INVERSE_CATS` and `ALL_CATS` should already be imported — verify.

- [ ] **Step 4: Run the targeted test**

Run:
```bash
pytest tests/test_scoring/ -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "refactor(scoring): score_roto returns dict[str, CategoryPoints]

Accepts Standings / ProjectedStandings via the TeamStatsTable protocol.
Returns CategoryPoints (Category-enum-keyed values + total) instead of
{'R_pts': ..., 'HR_pts': ..., 'total': ...}. Callers migrate in
subsequent tasks.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 3.2: Migrate `trades/evaluate.py`

**Files:**
- Modify: `src/fantasy_baseball/trades/evaluate.py`
- Test: `tests/test_trades/test_evaluate.py`

- [ ] **Step 1: Update tests to feed typed objects and assert typed results**

For each test in `tests/test_trades/test_evaluate.py` that builds `standings = [{"name": ..., "stats": {...}}]`, switch to building a `Standings` (for actual) or `ProjectedStandings` (for projected) via the `_standings()` helper pattern from Task 2.3's test file. Assertions on the return shape: nothing changes (still `float` deltas), but *inputs* are now typed.

- [ ] **Step 2: Run, watch failures**

- [ ] **Step 3: Update signatures**

In `src/fantasy_baseball/trades/evaluate.py`:

```python
def compute_roto_points_by_cat(
    standings: TeamStatsTable,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
) -> dict[str, dict[Category, float]]:
    """Return per-category roto points for each team."""
    # _STAT_DEFAULTS is no longer needed — CategoryStats has defaults (0, 99)
    roto = score_roto(standings, team_sds=team_sds)
    return {name: dict(cp.values) for name, cp in roto.items()}


def compute_roto_points(standings: TeamStatsTable) -> dict[str, float]:
    by_cat = compute_roto_points_by_cat(standings)
    return {name: sum(cat_pts.values()) for name, cat_pts in by_cat.items()}
```

Import the protocol:

```python
from fantasy_baseball.models.standings import (
    CategoryStats, ProjectedStandings, ProjectedStandingsEntry,
    Standings, StandingsEntry,
)
from fantasy_baseball.scoring import TeamStatsTable, score_roto
```

For `compute_trade_impact`, replace the `list[dict]` traversal with entry-by-entry construction. Signature:

```python
def compute_trade_impact(
    standings: Standings,
    hart_name: str,
    opp_name: str,
    hart_loses_ros: dict[str, Any],
    hart_gains_ros: dict[str, Any],
    opp_loses_ros: dict[str, Any],
    opp_gains_ros: dict[str, Any],
    projected_standings: ProjectedStandings | None = None,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
) -> dict[str, Any]:
```

Body:

```python
    baseline: TeamStatsTable = projected_standings if projected_standings is not None else standings
    baseline_by_cat = compute_roto_points_by_cat(baseline, team_sds=team_sds)

    post_trade_entries: list[ProjectedStandingsEntry] = []
    for entry in baseline.entries:
        if entry.team_name == hart_name:
            new_stats_dict = apply_swap_delta(entry.stats.to_dict(), hart_loses_ros, hart_gains_ros)
            post_trade_entries.append(ProjectedStandingsEntry(
                team_name=entry.team_name,
                stats=CategoryStats.from_dict(new_stats_dict),
            ))
        elif entry.team_name == opp_name:
            new_stats_dict = apply_swap_delta(entry.stats.to_dict(), opp_loses_ros, opp_gains_ros)
            post_trade_entries.append(ProjectedStandingsEntry(
                team_name=entry.team_name,
                stats=CategoryStats.from_dict(new_stats_dict),
            ))
        else:
            post_trade_entries.append(ProjectedStandingsEntry(
                team_name=entry.team_name, stats=entry.stats,
            ))

    post_trade = ProjectedStandings(
        effective_date=baseline.effective_date,
        entries=post_trade_entries,
    )
    post_trade_by_cat = compute_roto_points_by_cat(post_trade, team_sds=team_sds)

    hart_base = sum(baseline_by_cat[hart_name].values())
    hart_proj = sum(post_trade_by_cat[hart_name].values())
    opp_base = sum(baseline_by_cat[opp_name].values())
    opp_proj = sum(post_trade_by_cat[opp_name].values())

    hart_cat_deltas = {
        cat: post_trade_by_cat[hart_name][cat] - baseline_by_cat[hart_name][cat]
        for cat in ALL_CATS
    }
    opp_cat_deltas = {
        cat: post_trade_by_cat[opp_name][cat] - baseline_by_cat[opp_name][cat]
        for cat in ALL_CATS
    }

    return {
        "hart_delta": hart_proj - hart_base,
        "opp_delta": opp_proj - opp_base,
        "hart_cat_deltas": hart_cat_deltas,
        "opp_cat_deltas": opp_cat_deltas,
    }
```

`apply_swap_delta` still operates on a flat `dict[str, float]` (that's an internal math helper — leave it). The bridge is `entry.stats.to_dict()` / `CategoryStats.from_dict(new_stats_dict)`.

Update `search_trades_away` and `search_trades_for` signatures to accept `Standings` and `ProjectedStandings | None`. The body only passes them through to `compute_trade_impact`, so minimal changes.

- [ ] **Step 4: Delete the `COUNTING_CATS` string list if no longer referenced, replacing with `HITTING_CATEGORIES + PITCHING_CATEGORIES - RATE_STATS` if needed**

Check: `COUNTING_CATS = ["R", "HR", "RBI", "SB", "W", "K", "SV"]` — this is still used by `apply_swap_delta`. Keep it as-is; it operates on the flat dict, which uses uppercase strings at that I/O boundary.

- [ ] **Step 5: Run tests**

Run:
```bash
pytest tests/test_trades/ -v
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(trades): evaluate.py takes Standings/ProjectedStandings

compute_roto_points_by_cat/compute_roto_points accept the
TeamStatsTable protocol. compute_trade_impact takes typed standings +
optional projected_standings; deltas keyed by Category enum.
apply_swap_delta's flat-dict interface is preserved (rate-stat math
needs ab/ip alongside the roto cats).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 3.3: `leverage.py` rename

**Files:**
- Modify: `src/fantasy_baseball/lineup/leverage.py`
- Test: `tests/test_lineup/test_leverage.py`

- [ ] **Step 1: Rename `StandingsSnapshot` → `Standings` in signatures and bodies**

Run:
```bash
grep -n "StandingsSnapshot" src/fantasy_baseball/lineup/leverage.py
```

For each occurrence in `leverage.py`, rewrite to `Standings`. The import at the top becomes:

```python
from fantasy_baseball.models.standings import Standings
```

- [ ] **Step 2: Update tests**

Run:
```bash
grep -n "StandingsSnapshot" tests/test_lineup/test_leverage.py
```

Replace with `Standings` in imports and fixture construction. `StandingsSnapshot(effective_date=..., entries=...)` → `Standings(effective_date=..., entries=...)`. The field names match, so nothing else changes.

- [ ] **Step 3: Run tests**

Run:
```bash
pytest tests/test_lineup/test_leverage.py -v
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(leverage): take Standings instead of StandingsSnapshot

Pure rename. Field layout is identical.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 3.4: `season_data.py` consumers

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`
- Test: `tests/test_web/test_season_data.py` (grep to find)

- [ ] **Step 1: Delete `_standings_to_snapshot`**

Remove the function at lines 61-82. Every caller must supply a `Standings` directly.

- [ ] **Step 2: Rewrite `format_standings_for_display`**

Change its signature from `standings: StandingsSnapshot` to `standings: Standings`. Everywhere inside that reads `roto[name]["R_pts"]` / `roto[name]["total"]`, rewrite to `roto[name][Category.R]` / `roto[name].total`.

The line `roto_pts = dict(roto[name])` — the new `CategoryPoints` is a dataclass, not a dict. Replace that block with:

```python
        roto_cat_pts = roto[name]                    # CategoryPoints
        total = roto_cat_pts.total
        if has_yahoo_totals:
            # Preserve score_roto total alongside Yahoo override for diagnostics.
            total_override = entry.yahoo_points_for
            team_total = float(total_override)
        else:
            team_total = float(total)

        team_totals[name] = team_total
        teams.append({
            "name": name,
            "team_key": entry.team_key,
            "stats": entry.stats,
            "roto_points": {                          # kept as dict for template compat
                cat: roto_cat_pts[cat] for cat in ALL_CATEGORIES
            },
            "roto_total": team_total,
            "score_roto_total": float(total),
            "is_user": name == user_team_name,
            "sds": team_sds.get(name, {}) if team_sds else {},
        })
```

Update the final sort lines to use `-t["roto_total"]` instead of `-t["roto_points"]["total"]`. The template (Phase 4) is the only other consumer of these keys and will match.

Also: the current code's `has_yahoo_totals` branch writes `roto_pts["score_roto_total"]` and `roto_pts["total"]` — the replacement above moves those onto the top-level dict (cleaner, and the Jinja template change in Phase 4 follows).

- [ ] **Step 3: Rewrite `_compute_color_intensity`**

Change its signature to accept `Standings`:

```python
def _compute_color_intensity(
    standings: Standings,
    team_totals: dict[str, float],
) -> dict[str, dict[Category, float]]:
```

Return type is `dict[team_name, {Category: float}]` now (was `{"R": ..., "HR": ..., ...}`). Inside: iterate `for cat in ALL_CATEGORIES` and use `entry.stats[cat]`. Callers already pass this through to teams dicts; ensure the dict keys are Category enums, not strings.

- [ ] **Step 4: Rewrite `compute_comparison_standings` and `get_teams_list`**

`compute_comparison_standings(projected_standings: ProjectedStandings, ...)` — drop the `list[dict]` traversal, iterate `projected_standings.entries` directly, build its output using typed entries.

`get_teams_list(standings: Standings, user_team_name: str)` — iterate `standings.entries`; map `t["name"]`/`t.get("team_key")`/`t.get("rank")` onto `entry.team_name`/`entry.team_key`/`entry.rank`.

- [ ] **Step 5: Update tests**

Find tests:
```bash
grep -rn "format_standings_for_display\|_compute_color_intensity\|compute_comparison_standings\|get_teams_list" tests/
```

For each test, migrate the fixture from `list[dict]` to `Standings` / `ProjectedStandings`. Update assertions: `result["teams"][0]["roto_points"]["R_pts"]` becomes `result["teams"][0]["roto_points"][Category.R]`, etc.

- [ ] **Step 6: Run tests**

Run:
```bash
pytest tests/test_web/ -v
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "refactor(season_data): consumers take typed Standings

format_standings_for_display, _compute_color_intensity,
compute_comparison_standings, get_teams_list now accept Standings /
ProjectedStandings. _standings_to_snapshot is deleted. Per-team
roto_points dict is keyed by Category enum. Internal total
lives at 'roto_total'.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase 4 — Flask + Jinja

### Task 4.1: Migrate Flask routes

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Test: `tests/test_web/test_season_routes.py` (grep to find)

- [ ] **Step 1: Identify every route that reads cached standings**

Run:
```bash
grep -n "CacheKey.STANDINGS\|CacheKey.ROS_PROJECTIONS\|projected_standings" src/fantasy_baseball/web/season_routes.py
```

- [ ] **Step 2: Update `/standings` and related routes**

Every `standings = read_cache(CacheKey.STANDINGS)` or similar now yields either a `Standings` object (once Task 4.2 wires the cache path through `Standings.to_json`) or raw dict JSON. For transitional robustness during this phase, wrap reads with:

```python
from fantasy_baseball.models.standings import Standings, ProjectedStandings

raw = read_cache(CacheKey.STANDINGS)
standings: Standings | None = Standings.from_json(raw) if raw else None
```

Once Task 4.3 (refresh_pipeline) lands, `read_cache` returns dict JSON that `from_json` parses correctly.

Pass `all_categories` into the template context:

```python
from fantasy_baseball.utils.constants import ALL_CATEGORIES

return render_template(
    "season/standings.html",
    standings=format_standings_for_display(standings, user_team_name, team_sds=...),
    all_categories=ALL_CATEGORIES,
    ...
)
```

Any route that passes raw `list[dict]` standings to `compute_trade_impact`, `search_trades_away`, `search_trades_for`, `get_teams_list`, or `compute_comparison_standings` now needs to pass typed objects. Look for these call sites and rewrite.

- [ ] **Step 3: Run route tests**

Run:
```bash
pytest tests/test_web/ -v
```

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(season_routes): pass Standings + all_categories to templates

Routes parse cached JSON into Standings/ProjectedStandings at the
boundary. Template context gains all_categories=ALL_CATEGORIES for
Category-enum-based loops.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 4.2: Migrate Jinja template

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/standings.html`

- [ ] **Step 1: Read the current template**

Open `src/fantasy_baseball/web/templates/season/standings.html`. Find every `{% for cat in ... %}` and every `team.stats[cat]` / `team.roto_points[cat ~ '_pts']` reference.

- [ ] **Step 2: Rewrite category loops**

Replace `{% for cat in categories %}` (where `categories` is a list of strings) with `{% for cat in all_categories %}` (list of `Category` enums). Every

- `team.stats[cat]` → `team.stats[cat]` (works — `cat` is a `Category` enum)
- `team.roto_points[cat ~ '_pts']` → `team.roto_points[cat]` (Category-enum-keyed dict from Task 3.4)
- `team.color_intensity.get(cat)` → `team.color_intensity.get(cat)` (unchanged — dict keyed by Category)
- `team.roto_points['total']` → `team.roto_total` (moved to top-level key in Task 3.4)
- Any header row that renders the category name: `{{ cat }}` renders as `Category.R`. Switch to `{{ cat.value }}` (renders `R`).
- Any `{% if cat == "HR" %}` → `{% if cat == Category.HR %}` — but first, verify Jinja can see `Category`; if not, pass it into context. Simpler: switch to `{% if cat == categories.HR %}` or compare by `cat.value`: `{% if cat.value == "HR" %}`.

- [ ] **Step 3: Manual smoke test — run the season dashboard locally**

Run (from repo root):
```bash
python scripts/run_season_dashboard.py
```

Open the `/standings` URL it prints. Verify:
- All ten category columns render values.
- Yahoo-points column reconciles for teams that have `yahoo_points_for`.
- Color intensity highlighting still applies.
- User-team highlight row still renders.

- [ ] **Step 4: Commit**

```bash
git add -A
git commit -m "refactor(templates/standings): iterate Category enum instead of strings

Template context now receives all_categories=ALL_CATEGORIES.
team.stats[cat] and team.roto_points[cat] use Category-enum keys.
Headers render cat.value for the uppercase string form.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 4.3: Migrate `refresh_pipeline.py`

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py`

- [ ] **Step 1: Update `_fetch_standings_and_roster`**

The function currently calls `fetch_standings(league)` and expects a `list[dict]`. Update to pass the effective_date and consume the resulting `Standings`:

```python
standings = fetch_standings(league, effective_date=effective_date)
...
write_cache(CacheKey.STANDINGS, standings.to_json())
```

Fill-in-defaults logic (the `_STAT_DEFAULTS` / `_fill_stat_defaults` step): `CategoryStats` now defaults missing stats at construction time (0 for counting, 99 for ERA/WHIP), so this step is unnecessary. Delete it.

- [ ] **Step 2: Update `_build_projected_standings` (on the `RefreshRun` class)**

It already calls `build_projected_standings(team_rosters)` which after Task 2.2 returns `ProjectedStandings`. Pass through:

```python
projected = build_projected_standings(team_rosters, effective_date=effective_date)
self.projected_standings = projected
write_cache(CacheKey.ROS_PROJECTIONS, projected.to_json())  # adjust key name as appropriate
```

- [ ] **Step 3: Update every passthrough site**

Every place the `RefreshRun` stores standings or passes them through (grep `self.projected_standings` / `self.standings`): assert they're typed and downstream consumers expect typed objects. Most of this should already be in place from Phase 3.

- [ ] **Step 4: Manual smoke test**

Run:
```bash
python scripts/run_season_dashboard.py
```

Verify full refresh completes end-to-end. Check the generated JSON files under `data/cache/` are in canonical shape (`name`, UPPERCASE, `effective_date`).

- [ ] **Step 5: Run all tests**

Run:
```bash
pytest -v
ruff check .
ruff format --check .
mypy
```

Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(refresh_pipeline): consume typed Standings end-to-end

fetch_standings → Standings; build_projected_standings →
ProjectedStandings. Cache writes go through .to_json(). Stat-default
fill is dropped — CategoryStats handles missing counting stats at
construction time (defaults 0 / 99).

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

---

## Phase 5 — Deletion + Redis Rewrite

### Task 5.1: Write `scripts/migrate_standings_history.py`

**Files:**
- Create: `scripts/migrate_standings_history.py`
- Create: `tests/test_scripts/test_migrate_standings_history.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_scripts/test_migrate_standings_history.py`:

```python
"""Tests for scripts/migrate_standings_history.py."""
import json
import sys
from pathlib import Path

import pytest

# Ensure scripts/ is on sys.path
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from migrate_standings_history import (  # type: ignore[import-not-found]
    _from_legacy_json,
    rewrite_hash,
)
from fantasy_baseball.data import redis_store
from fantasy_baseball.models.standings import Standings


LEGACY_DAY = {
    "teams": [
        {
            "team": "Alpha", "team_key": "431.l.1.t.1", "rank": 1,
            "r": 45, "hr": 12, "rbi": 40, "sb": 8, "avg": 0.268,
            "w": 3, "k": 85, "sv": 4, "era": 3.21, "whip": 1.14,
        },
    ],
}


def test_from_legacy_json_parses_legacy_shape():
    s = _from_legacy_json(LEGACY_DAY, snapshot_date="2026-04-15")
    assert isinstance(s, Standings)
    assert s.entries[0].team_name == "Alpha"
    assert s.entries[0].stats.r == 45
    assert s.entries[0].stats.whip == pytest.approx(1.14)
    assert s.entries[0].yahoo_points_for is None


def test_rewrite_hash_converts_legacy_entries(fake_redis):
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", json.dumps(LEGACY_DAY))
    stats = rewrite_hash(fake_redis)
    assert stats["rewritten"] == 1
    assert stats["skipped"] == 0
    # After rewrite, standard reader works
    reloaded = redis_store.get_standings_day(fake_redis, "2026-04-15")
    assert reloaded is not None
    assert reloaded.entries[0].team_name == "Alpha"


def test_rewrite_hash_is_idempotent(fake_redis):
    fake_redis.hset(redis_store.STANDINGS_HISTORY_KEY, "2026-04-15", json.dumps(LEGACY_DAY))
    rewrite_hash(fake_redis)
    stats = rewrite_hash(fake_redis)
    assert stats["rewritten"] == 0
    assert stats["skipped"] == 1
```

- [ ] **Step 2: Run it — it fails (script doesn't exist)**

- [ ] **Step 3: Write `scripts/migrate_standings_history.py`**

```python
"""One-shot migration: rewrite standings_history Redis hash into canonical shape.

Legacy entries have ``{"team", "r", "hr", ...}`` (no 'name', lowercase,
no effective_date wrapper). Canonical shape is what ``Standings.to_json``
emits. Idempotent: entries already in canonical shape are skipped.

Run once after the refactor is merged:

    python scripts/migrate_standings_history.py

Requires Upstash creds via the usual kv_store env vars.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

from fantasy_baseball.data import redis_store  # noqa: E402
from fantasy_baseball.data.kv_store import get_kv  # noqa: E402
from fantasy_baseball.models.standings import (  # noqa: E402
    CategoryStats, Standings, StandingsEntry,
)


def _from_legacy_json(payload: dict, *, snapshot_date: str) -> Standings:
    """Parse the legacy ``{"teams": [{"team", lowercase_stat_keys}]}`` shape."""
    if "teams" not in payload:
        raise ValueError(f"{snapshot_date}: payload missing 'teams' wrapper")
    rows = payload["teams"]
    entries: list[StandingsEntry] = []
    for row in rows:
        if "team" not in row:
            raise ValueError(f"{snapshot_date}: row missing 'team' field — not legacy shape either")
        stats = CategoryStats(
            r=float(row.get("r") or 0.0),
            hr=float(row.get("hr") or 0.0),
            rbi=float(row.get("rbi") or 0.0),
            sb=float(row.get("sb") or 0.0),
            avg=float(row.get("avg") or 0.0),
            w=float(row.get("w") or 0.0),
            k=float(row.get("k") or 0.0),
            sv=float(row.get("sv") or 0.0),
            era=float(row["era"]) if row.get("era") is not None else 99.0,
            whip=float(row["whip"]) if row.get("whip") is not None else 99.0,
        )
        entries.append(StandingsEntry(
            team_name=row["team"],
            team_key=row.get("team_key") or "",
            rank=int(row.get("rank") or 0),
            stats=stats,
            yahoo_points_for=None,  # legacy payloads never carried this
        ))
    return Standings(effective_date=date.fromisoformat(snapshot_date), entries=entries)


def rewrite_hash(client) -> dict[str, int]:
    """Walk the standings_history hash and rewrite legacy entries.

    Returns a stats dict: {"rewritten": N, "skipped": N, "errors": N}.
    """
    raw_map = client.hgetall(redis_store.STANDINGS_HISTORY_KEY)
    stats = {"rewritten": 0, "skipped": 0, "errors": 0}

    for snap_date, raw in raw_map.items():
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print(f"[{snap_date}] corrupt JSON — skipping")
            stats["errors"] += 1
            continue

        if not isinstance(payload, dict):
            print(f"[{snap_date}] not a dict payload — skipping")
            stats["errors"] += 1
            continue

        # Try canonical first
        try:
            Standings.from_json(payload)
            print(f"[{snap_date}] already canonical — skip")
            stats["skipped"] += 1
            continue
        except ValueError:
            pass

        # Fall through: treat as legacy
        try:
            s = _from_legacy_json(payload, snapshot_date=snap_date)
        except (ValueError, KeyError, TypeError) as e:
            print(f"[{snap_date}] legacy parse failed: {e} — SKIPPING (fix manually)")
            stats["errors"] += 1
            continue

        client.hset(
            redis_store.STANDINGS_HISTORY_KEY,
            snap_date,
            json.dumps(s.to_json()),
        )
        print(f"[{snap_date}] rewritten")
        stats["rewritten"] += 1

    return stats


def main() -> int:
    client = get_kv()
    if client is None:
        print("ERROR: no KV client available — is Upstash configured?")
        return 1

    stats = rewrite_hash(client)
    print()
    print(f"Rewritten: {stats['rewritten']}")
    print(f"Skipped (already canonical): {stats['skipped']}")
    print(f"Errors: {stats['errors']}")
    return 0 if stats["errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests**

Run:
```bash
pytest tests/test_scripts/test_migrate_standings_history.py -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add -A
git commit -m "feat(scripts): migrate_standings_history.py one-shot rewrite

Walks standings_history hash, rewrites legacy-shape entries into
canonical Standings.to_json(). Idempotent — canonical entries skip.
The legacy parser lives inside the script, not on Standings.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 5.2: Run the migration against live Redis

**Files:** none (operational step)

- [ ] **Step 1: Confirm with the user before running**

Ask the user to confirm: "Ready to run the Redis migration against live Upstash. This rewrites every legacy-shape entry in the standings_history hash. Proceed?" Wait for explicit confirmation.

- [ ] **Step 2: Run the migration**

```bash
python scripts/migrate_standings_history.py
```

Capture the stdout. Expected output: a line per snapshot date, then a summary (`Rewritten: N / Skipped: M / Errors: 0`). Errors stop the rollout — investigate before proceeding.

- [ ] **Step 3: Verify by re-running**

```bash
python scripts/migrate_standings_history.py
```

Expected: `Rewritten: 0 / Skipped: N / Errors: 0` — idempotence check.

- [ ] **Step 4: Verify `get_standings_history` returns clean data**

Start a Python REPL:
```bash
python -c "from fantasy_baseball.data.kv_store import get_kv; from fantasy_baseball.data.redis_store import get_standings_history; c = get_kv(); h = get_standings_history(c); print(f'{len(h)} snapshots loaded cleanly')"
```

Expected: a non-zero snapshot count with no exceptions.

### Task 5.3: Delete legacy code paths

**Files:**
- Modify: `src/fantasy_baseball/models/standings.py` (delete `StandingsSnapshot`)
- Modify: `src/fantasy_baseball/scoring.py` (delete `build_projected_standings`)
- Modify: `src/fantasy_baseball/models/league.py` (drop `StandingsSnapshot` import, no alias)
- Modify: any remaining `StandingsSnapshot` references

- [ ] **Step 1: Delete `StandingsSnapshot` from `models/standings.py`**

Remove the entire `StandingsSnapshot` class. Any test or module that still imports it must be updated to `Standings`.

- [ ] **Step 2: Delete `build_projected_standings` from `scoring.py`**

Remove the function. Call sites should already use `ProjectedStandings.from_rosters` directly; if anything still calls `build_projected_standings`, fix it.

- [ ] **Step 3: Grep for any remaining references**

Run:
```bash
grep -rn "StandingsSnapshot\|build_projected_standings\|_standings_to_snapshot" src/ tests/ scripts/
```

Expected: zero matches. Fix anything that remains.

- [ ] **Step 4: Grep for remaining `list[dict]` standings annotations**

Run:
```bash
grep -rn "standings: list\[dict\]\|projected_standings: list\[dict\]" src/
```

Expected: zero matches. Fix anything that remains.

- [ ] **Step 5: Full verification**

Run:
```bash
pytest -v
ruff check .
ruff format --check .
mypy
vulture src/ scripts/
```

Expected: all green. Vulture findings should be at most the same as pre-refactor (ignore pre-existing; fix anything new from this refactor).

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(standings): delete StandingsSnapshot and build_projected_standings

Standings is now the sole live-standings type; ProjectedStandings is
the sole projected type. All call sites migrated. No aliases.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>"
```

### Task 5.4: End-to-end smoke test and PR

**Files:** none (operational)

- [ ] **Step 1: Run the full refresh pipeline locally**

```bash
python scripts/run_season_dashboard.py
```

Verify:
- Refresh completes without exceptions.
- `/standings` renders correctly.
- All UI elements (rank column, category columns, roto_points, color intensity, user highlight, yahoo_points_for reconciliation) look right.

- [ ] **Step 2: Verify no new vulture findings unrelated to this refactor**

Run:
```bash
vulture src/ scripts/ 2>&1 | tee /tmp/vulture.txt
```

Compare against baseline. New findings from this refactor must be zero.

- [ ] **Step 3: Final paste of check outputs into PR description**

Run and paste summaries:
```bash
pytest -v 2>&1 | tail -5
ruff check . 2>&1 | tail -5
ruff format --check . 2>&1 | tail -5
mypy 2>&1 | tail -5
```

All must show zero failures.

- [ ] **Step 4: Ask the user before opening the PR**

"All phases complete. Full test suite green, mypy clean, refresh pipeline exercised end-to-end. Want me to open a PR from `standings-dataclass-refactor` → `main`?"

Wait for confirmation before running `gh pr create`.

---

## Self-Review Notes

- Every spec section maps to at least one task: construction (2.1, 2.2), Redis I/O (2.3, 5.1–5.2), consumers (3.1–3.4), Flask (4.1), Jinja (4.2), refresh pipeline (4.3), deletion (5.3).
- No placeholders remain — each code block is complete.
- Types and method names are consistent across tasks (`Standings.from_json` / `to_json`, `CategoryPoints.values` / `.total`, `ProjectedStandings.from_rosters`).
- Phase sizes respect CLAUDE.md's 5-file-per-phase rule; some tasks within a phase touch only one file, which is fine.
- Testing coverage: canonical round-trip, legacy-shape rejection, migration idempotence, typed `__getitem__` rejection of bare strings, end-to-end refresh smoke test.
- Memory-aligned: branch-first (CLAUDE.md), refresh-before-merge smoke test (task 4.3 + 5.4), no bare-name player keys (N/A — this refactor is about categories, not players).
