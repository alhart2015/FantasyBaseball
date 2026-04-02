# Player Dataclass Refactor

**Date:** 2026-04-01
**Branch:** `player-data-class-refactor`

## Problem

Player data flows through the codebase as untyped dicts and pd.Series with no contract on what fields exist. A "player" is sometimes a dict from Yahoo parsing, sometimes a pd.Series from projection DataFrames, sometimes a JSON-deserialized cache entry. Functions silently get `None` from `.get()` calls when fields are missing. Same-name collisions happen because there's no reliable unique identifier. There's no way to know at a glance what data a player object carries at any point in the pipeline.

## Design

### Core dataclasses

```python
@dataclass
class HitterStats:
    pa: float = 0; ab: float = 0; h: float = 0
    r: float = 0; hr: float = 0; rbi: float = 0; sb: float = 0
    avg: float = 0  # computed from h/ab when not provided
    sgp: float | None = None  # unweighted total SGP, computed on demand

@dataclass
class PitcherStats:
    ip: float = 0; w: float = 0; k: float = 0; sv: float = 0
    er: float = 0; bb: float = 0; h_allowed: float = 0
    era: float = 0; whip: float = 0  # computed from components when not provided
    sgp: float | None = None

@dataclass
class RankInfo:
    ros: int | None = None
    preseason: int | None = None
    current: int | None = None

@dataclass
class Player:
    name: str
    player_type: str  # "hitter" | "pitcher"
    positions: list[str] = field(default_factory=list)
    team: str = ""
    fg_id: str | None = None
    mlbam_id: int | None = None
    yahoo_id: str | None = None

    # Projection stat bags (None = not available)
    ros: HitterStats | PitcherStats | None = None
    preseason: HitterStats | PitcherStats | None = None
    current: HitterStats | PitcherStats | None = None  # from game logs

    # Calculated fields
    wsgp: float = 0.0
    rank: RankInfo = field(default_factory=RankInfo)

    # Display/context fields
    selected_position: str = ""  # current Yahoo roster slot
    status: str = ""  # IL, DTD, etc.
    pace: dict | None = None  # from compute_player_pace (display only)
```

### Conversion methods

Each dataclass gets:
- `from_dict(d)` — construct from a dict (for cache deserialization, Yahoo parsing)
- `from_series(s)` — construct from a pd.Series (for projection DataFrame rows)
- `to_dict()` — serialize for JSON cache
- `to_series()` — convert to pd.Series for backward compatibility with SGP functions

`Player.from_projection_match(roster_entry, projection_row)` — combines Yahoo roster data with a matched projection row into a Player.

### Where the dataclass lives

New file: `src/fantasy_baseball/models/player.py`

### SGP integration

`HitterStats.compute_sgp()` and `PitcherStats.compute_sgp()` call `calculate_player_sgp` internally and cache the result on `self.sgp`. This replaces the pattern of `calculate_player_sgp(pd.Series(player_dict))` scattered across the codebase.

`Player.compute_wsgp(leverage)` calls `calculate_weighted_sgp` on the ROS stat bag and stores on `self.wsgp`.

## Phased implementation

### Phase 1: Define types + conversion layer
- Create `models/player.py` with all dataclasses
- Implement all `from_dict`, `from_series`, `to_dict`, `to_series` methods
- Implement `compute_sgp()` and `compute_wsgp()` methods
- Full unit test coverage
- No consumers yet — pure library code

### Phase 2: Adopt in season dashboard refresh
- `season_data.py` constructs `Player` objects after roster matching (Step 6)
- Preseason, ROS, and current stat bags populated from respective data sources
- SGP, wSGP, ranks computed via Player methods
- `write_cache("roster", [p.to_dict() for p in players])` for serialization
- Templates still read dicts from cache (no template changes yet)

### Phase 3: Adopt in route handlers + templates
- Route handlers deserialize cache back into `Player` objects where needed
- Player search API constructs Player objects from DB queries
- Templates updated to access typed fields (mostly transparent since `to_dict()` preserves keys)

### Phase 4: Adopt in waivers, trades, buy-low
- `scan_waivers` accepts/returns Player objects instead of pd.Series/dicts
- `find_trades` uses Player objects
- `find_buy_low_candidates` uses Player objects
- SGP functions (`calculate_player_sgp`, `calculate_weighted_sgp`) gain Player overloads or the callers use `player.to_series()`

### Phase 5: Draft pipeline (future, not this branch)
- `board.py`, `recommender.py`, `strategy.py` use Player
- Separate branch/session

## What this does NOT include

- Changing the database schema (projections stay as flat tables)
- Changing the DataFrame-based projection blending (pandas stays for vectorized operations)
- Solving the fg_id cross-reference problem (that's the TODO.md refactor)
- Changing the cache file format (dicts stay, just produced by `to_dict()`)
