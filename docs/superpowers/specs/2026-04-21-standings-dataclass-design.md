# Standings Dataclass Refactor

**Date:** 2026-04-21
**Branch:** (TBD — create during implementation)

## Problem

Standings data flows through the codebase as dict-of-dicts with three incompatible shapes:

- **Yahoo/cache shape** — `[{"name", "team_key", "rank", "stats": {UPPERCASE keys}}, ...]`, sometimes with `"points_for"`.
- **Redis persisted shape** — `{"teams": [{"team" (not "name"), "team_key", "rank", flat lowercase keys}, ...]}`, no `points_for`.
- **Draft sim shape** — isolated, incompatible nested `{"categories": {cat: {"value", "points"}}}`. Out of scope here.

Functions that consume standings take untyped `list[dict]` and index by bare string (`t["stats"]["R"]`). A typed `StandingsSnapshot` hierarchy already exists in `src/fantasy_baseball/models/standings.py` (`CategoryStats`, `StandingsEntry`, `StandingsSnapshot`), but it runs in parallel with the dict pipeline — only `League.from_redis()` and a handful of display helpers convert to it. `CategoryStats` carries a dict-compat surface (`__getitem__`, `get`, `items`, `keys`, `__iter__`) that was introduced as a migration shim and never removed.

Drift points that bite:

- UPPERCASE vs lowercase stat keys between cache and Redis; silent corruption if either side adds a field.
- `"name"` vs `"team"` in different sinks for the same semantic field.
- `build_projected_standings()` returns `list[dict]` with sentinel `team_key=""` / `rank=0`, so projected and actual aren't statically distinguishable.
- `Category` is a `StrEnum`, so `cat == "HR"` and `stats["R"]` silently succeed — the type system doesn't help catch bare-string leaks.
- Jinja templates index `team.stats[cat]` via the dict-compat layer, blocking removal.

Goal: collapse the web pipeline (Yahoo fetch → cache → Flask routes → templates → trade/evaluate) and the Redis persistence format onto one typed representation, with strict `Category`-enum access everywhere and no bare-string category keys in Python code.

## Scope

**In scope (A + B):**

- Web/season pipeline standings: construction, cache I/O, consumers, Flask routes, Jinja template.
- Redis persistence format: `write_standings_snapshot`, `get_standings_day`, `get_latest_standings`, `get_standings_history`, `League.from_redis`.
- `Category` enum: flip from `StrEnum` to plain `Enum`.

**Out of scope:**

- `scripts/simulate_draft.py`'s nested standings shape. It's isolated from the main pipeline; leave alone.
- Backfilling legacy Redis history entries. New writes use the canonical shape; `from_json` accepts both shapes transparently so reads continue to work. A rewrite pass can happen later if desired.

## Design

### Core dataclasses

```python
# src/fantasy_baseball/utils/constants.py
from enum import Enum

class Category(Enum):
    R = "R"; HR = "HR"; RBI = "RBI"; SB = "SB"; AVG = "AVG"
    W = "W"; K = "K"; SV = "SV"; ERA = "ERA"; WHIP = "WHIP"

ALL_CATEGORIES: list[Category] = [Category.R, Category.HR, ...]  # (unchanged)
HITTING_CATEGORIES, PITCHING_CATEGORIES, RATE_STATS, INVERSE_STATS  # (unchanged)
```

```python
# src/fantasy_baseball/models/standings.py
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
                f"CategoryStats indexing requires a Category enum, got {type(cat).__name__}"
            )
        return getattr(self, _CAT_TO_FIELD[cat])

    def items(self) -> Iterator[tuple[Category, float]]:
        for cat in ALL_CATEGORIES:
            yield cat, getattr(self, _CAT_TO_FIELD[cat])

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "CategoryStats":
        """I/O boundary: accept UPPERCASE-string-keyed dicts from Yahoo/JSON."""
        kwargs = {
            _CAT_TO_FIELD[cat]: float(d[cat.value])
            for cat in ALL_CATEGORIES
            if cat.value in d
        }
        return cls(**kwargs)

    def to_dict(self) -> dict[str, float]:
        """I/O boundary: produce UPPERCASE-keyed dict for JSON/Redis."""
        return {cat.value: getattr(self, _CAT_TO_FIELD[cat]) for cat in ALL_CATEGORIES}

# Private: the only place Category<->field-name mapping lives.
_CAT_TO_FIELD: dict[Category, str] = {
    Category.R: "r", Category.HR: "hr", Category.RBI: "rbi", Category.SB: "sb",
    Category.AVG: "avg", Category.W: "w", Category.K: "k", Category.SV: "sv",
    Category.ERA: "era", Category.WHIP: "whip",
}
```

`CategoryStats` loses all of the old dict-compat: no `get`, no `keys`, no `__iter__`, no string-keyed `__getitem__`. The only category-indexed access is by `Category` enum. `CATEGORY_ORDER` in this file is deleted; `ALL_CATEGORIES` is the single source of truth.

```python
# Actual standings (from Yahoo)
@dataclass
class StandingsEntry:
    team_name: str
    team_key: str                 # non-optional — Yahoo always supplies
    rank: int                     # non-optional — Yahoo always supplies
    stats: CategoryStats
    yahoo_points_for: float | None = None

@dataclass
class Standings:
    effective_date: date
    entries: list[StandingsEntry]

    def by_team(self) -> dict[str, StandingsEntry]: ...
    def sorted_by_rank(self) -> list[StandingsEntry]: ...

    @classmethod
    def from_yahoo(cls, raw: list[dict], effective_date: date) -> "Standings": ...
    @classmethod
    def from_json(cls, d: dict) -> "Standings":
        """Accept canonical + legacy shapes. Canonical: {'effective_date', 'teams': [{'name', 'team_key', 'rank', 'yahoo_points_for', 'stats': {UPPERCASE}}]}. Legacy: 'team' key, lowercase stats, no wrapper date."""
    def to_json(self) -> dict: ...

# Projected standings
@dataclass
class ProjectedStandingsEntry:
    team_name: str
    stats: CategoryStats

@dataclass
class ProjectedStandings:
    effective_date: date
    entries: list[ProjectedStandingsEntry]

    def by_team(self) -> dict[str, ProjectedStandingsEntry]: ...

    @classmethod
    def from_rosters(cls, team_rosters: dict[str, list], effective_date: date) -> "ProjectedStandings":
        """Replaces build_projected_standings(). Uses project_team_stats(displacement=True)."""
    def to_json(self) -> dict: ...
```

`ProjectedStandings` has no `team_key`/`rank` fields at all — not "optional, sentinel-valued," just absent. That's the payoff for Option 2 typing: consumers that need Yahoo-assigned rank can only accept `Standings`, statically.

### Roto points

`score_roto()` returns `dict[str, CategoryPoints]` where:

```python
@dataclass
class CategoryPoints:
    values: dict[Category, float]  # per-category points
    total: float

    def __getitem__(self, cat: Category) -> float: ...
```

Roto points are not attached to `StandingsEntry`. They are a derived view whose computation requires knowing every team's stats simultaneously — bolting them onto the entry creates an awkward "sometimes populated, sometimes not" field. Consumers that want both pass a `Standings` and a `dict[str, CategoryPoints]` as separate arguments; that matches what they already do today.

### Shared protocol for category-indexed consumers

A minimal protocol for the "I just need per-team category totals" callers (`score_roto`, `compute_roto_points`, `compute_comparison_standings`):

```python
class TeamStatsRow(Protocol):
    team_name: str
    stats: CategoryStats

class TeamStatsTable(Protocol):
    entries: Sequence[TeamStatsRow]

def score_roto(
    standings: TeamStatsTable,
    *,
    team_sds: Mapping[str, Mapping[Category, float]] | None = None,
) -> dict[str, CategoryPoints]: ...
```

`StandingsEntry` and `ProjectedStandingsEntry` both structurally match `TeamStatsRow`; `Standings` and `ProjectedStandings` both match `TeamStatsTable`. Consumers that need Yahoo rank accept `Standings` concretely instead of the protocol.

### Redis / cache I/O boundary

- `redis_store.write_standings_snapshot(standings: Standings, key: str)` → calls `standings.to_json()`. Writes canonical shape (`"name"`, UPPERCASE stat keys, includes `yahoo_points_for`).
- `redis_store.get_standings_day(key: str, effective_date: date) -> Standings | None` → calls `Standings.from_json()`. Accepts both canonical and legacy shapes.
- `get_standings_history()` yields `list[Standings]`, ordered by effective_date.
- `League.from_redis()` stops doing its own parsing — it calls `Standings.from_json()`.
- `season_data.read_cache(CacheKey.STANDINGS)` / `write_cache(...)` use `Standings.to_json`/`from_json`. Cache JSON and Redis values are byte-identical.

`ProjectedStandings` gets the same `to_json`/`from_json` treatment; the projections cache writes `ProjectedStandings.to_json()`.

### Flask + Jinja

Routes pass `Standings` (and `ProjectedStandings`, `dict[str, CategoryPoints]`) to templates directly. The template context gains `all_categories=ALL_CATEGORIES`, and category loops use enum values:

```jinja
{% for cat in all_categories %}
  <td>{{ team.stats[cat] | round(2) }}</td>
  <td>{{ team.roto_points[cat] | round(1) }}</td>
{% endfor %}
```

`team.stats[cat]` dispatches to `CategoryStats.__getitem__(Category)`. No string-keyed access anywhere in the template. Any `{% if cat == "HR" %}`-style comparisons must be rewritten to enum identity — part of the template migration.

`format_standings_for_display()` keeps existing semantics (color intensity, SDs, user-team highlighting) but operates on `Standings` / `ProjectedStandings` throughout, not `list[dict]`.

### What gets deleted

- `CATEGORY_ORDER` in `models/standings.py`.
- Every dict-compat method on `CategoryStats` except the typed `__getitem__` and typed `items`.
- `StandingsSnapshot` — renamed to `Standings`; no alias left behind.
- `build_projected_standings()` — replaced by `ProjectedStandings.from_rosters()`.
- `_standings_to_snapshot()` in `season_data.py` — obsolete once cache stores/loads `Standings` directly.
- Bespoke dict → `CategoryStats` conversion logic in `League.from_redis()` and `season_data.py`.
- `StrEnum` inheritance on `Category`; any `cat == "HR"` comparisons that fall out of that change.

## Phased implementation

Each phase is its own commit, touches ≤5 files, and ends with `pytest -v && ruff check . && ruff format --check . && mypy` green. Vulture findings from prior phases are acknowledged but not blocking.

**Phase 0 — cleanup.** `ruff check --select F,I` and `vulture` over the affected modules. Remove stray imports, commented code, debug prints. Commit separately per CLAUDE.md "Step 0" rule.

**Phase 1 — typed scaffolding.** New types, `Category` flipped to plain `Enum`, `CategoryPoints` introduced.
- Edit `src/fantasy_baseball/models/standings.py`: add `Standings`, `ProjectedStandings`, `StandingsEntry` (non-optional rank/team_key), `ProjectedStandingsEntry`, `CategoryPoints`. Rewrite `CategoryStats` with typed `__getitem__`. Delete `CATEGORY_ORDER`.
- Edit `src/fantasy_baseball/utils/constants.py`: `Category(Enum)` instead of `Category(StrEnum)`. Update docstring.
- Fix any `cat == "X"` / `stats["X"]` sites that blow up. Grep methodically (Rule 10): direct comparisons, `in {...}` sets, dict keys in tests, JSON serialization call sites.
- Tests: unit tests for new types' `from_json`/`to_json` round-trip, including the legacy Redis shape acceptance.

**Phase 2 — construction sites.** Migrate every producer to return the new types.
- `src/fantasy_baseball/lineup/yahoo_roster.py`: `fetch_standings()` returns `Standings`.
- `src/fantasy_baseball/scoring.py`: replace `build_projected_standings` with a thin wrapper that calls `ProjectedStandings.from_rosters()` during the migration, then delete in Phase 5.
- `src/fantasy_baseball/data/redis_store.py`: read/write via `Standings.to_json`/`from_json`.
- `src/fantasy_baseball/data/db.py`: `load_standings` / `append_standings_snapshot` use typed objects (SQLite surface is offline-only per the feedback memory, so its row format is unchanged — only the Python boundary is).

**Phase 3 — consumers.** Migrate every reader to accept typed objects.
- `src/fantasy_baseball/scoring.py::score_roto`: accepts `HasTeamStats` protocol, returns `dict[str, CategoryPoints]`.
- `src/fantasy_baseball/trades/evaluate.py`: `compute_roto_points_by_cat`, `compute_roto_points`, `compute_trade_impact` accept `Standings` / `ProjectedStandings`.
- `src/fantasy_baseball/lineup/leverage.py`: already takes `StandingsSnapshot`; rename to `Standings`.
- `src/fantasy_baseball/web/season_data.py`: `_compute_color_intensity`, `format_standings_for_display`, `compute_comparison_standings` all typed.
- `src/fantasy_baseball/models/league.py`: `League.from_redis()` delegates to `Standings.from_json()`.

**Phase 4 — Flask + Jinja.**
- `src/fantasy_baseball/web/season_routes.py`: `/standings` and related routes pass `Standings` to templates; add `all_categories=ALL_CATEGORIES` to context.
- `src/fantasy_baseball/web/templates/season/standings.html`: iterate `Category` enum, no string-keyed access.
- `src/fantasy_baseball/web/refresh_pipeline.py`: `_fetch_standings_and_roster`, `_build_projected_standings`, and all the passthrough sites carry typed objects.

**Phase 5 — deletion.**
- Remove `StandingsSnapshot` (no alias).
- Remove `build_projected_standings()`.
- Remove `_standings_to_snapshot()`.
- Remove dict-compat `get`/`keys`/`__iter__`.
- Grep for any remaining `list[dict]` standings annotations and tighten them.
- `vulture` must report no new dead-code findings from this refactor.

**Verification per phase (CLAUDE.md Rule 4):**
```
pytest -v
ruff check .
ruff format --check .
mypy           # required — models/, scoring.py, trades/evaluate.py, lineup/, sgp/ all in mypy scope
vulture        # no NEW findings from this change
```

Output pasted into the PR description. Refresh pipeline exercised end-to-end locally via `python scripts/run_season_dashboard.py` before merge, per the `feedback_run_refresh_before_merge` memory.

## Testing

- **Round-trip:** `Standings.from_json(standings.to_json()) == standings` for representative fixtures. Same for `ProjectedStandings`.
- **Legacy shape acceptance:** `Standings.from_json(<legacy Redis shape>)` produces the same object as `Standings.from_json(<canonical shape>)` for equivalent data. Fixture stolen from `tests/test_data/test_redis_store_standings.py`.
- **Typed access:** `stats["R"]` raises `TypeError`; `stats[Category.R]` returns the float.
- **Enum comparisons:** `Category.HR == "HR"` is `False` after the `Enum` flip. Existing tests that assume `StrEnum` semantics get fixed, not deleted.
- **Score_roto:** refactored `score_roto` produces identical `CategoryPoints` values to the current dict output for a fixed fixture, up to float precision.
- **Display:** rendered standings page byte-identical (or structurally identical) for a known standings snapshot before vs after refactor.
- **Trade impact:** `compute_trade_impact` produces identical deltas before/after on a fixed fixture.

## Open items

None at time of writing. If the legacy Redis shape turns out to carry fields we didn't plan for (e.g., older schemas with fewer categories), `Standings.from_json` surfaces a validation error instead of silently zero-filling — handle case-by-case.
