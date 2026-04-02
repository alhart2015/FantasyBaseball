# Player Dataclass Phase 3: Adopt in Route Handlers

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deserialize cached roster data into Player objects in the lineup route handler, so `format_lineup_for_display` works with typed Player objects instead of raw dicts.

**Architecture:** `format_lineup_for_display` accepts cached roster dicts (as today) but internally constructs Player objects. Display entries are built from Player attributes instead of dict `.get()` calls. The template doesn't change — it still reads dict entries from the display function's output.

**Tech Stack:** Python dataclasses (from Phase 1), existing route/template code

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/web/season_data.py` | Modify | `format_lineup_for_display` uses Player objects |

---

### Task 1: Refactor format_lineup_for_display to use Player objects

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`

The function reads cached roster dicts and builds display entries. Refactor to construct Player objects from the cache, then build display entries from the Player's typed fields.

- [ ] **Step 1: Rewrite format_lineup_for_display**

Find `format_lineup_for_display` in `src/fantasy_baseball/web/season_data.py` (around line 490). Read the current code first.

The current code builds entry dicts with manual `.get()` calls:

```python
        entry = {
            "name": p["name"],
            "positions": p.get("positions", []),
            "selected_position": pos,
            "player_id": p.get("player_id", ""),
            "status": p.get("status", ""),
            "wsgp": p.get("wsgp", 0),
            "games": p.get("games_this_week", 0),
            "is_bench": pos in ("BN", "IL", "DL"),
            "is_il": "IL" in p.get("status", "") or pos == "IL",
            "stats": p.get("stats", {}),
            "ros": p.get("ros"),
            "rank": p.get("rank", {}),
            "preseason": p.get("preseason"),
        }
```

Replace with Player construction:

```python
def format_lineup_for_display(
    roster: list[dict], optimal: dict | None
) -> dict:
    """Format roster + optimizer output for the lineup template."""
    from fantasy_baseball.models.player import Player

    hitters = []
    pitchers = []

    for p in roster:
        player = Player.from_dict(p)
        pos = player.selected_position or "BN"
        is_pitcher = pos in PITCHER_POSITIONS or (
            pos == "BN" and set(player.positions).issubset(PITCHER_POSITIONS | {"BN"})
        )

        entry = {
            "name": player.name,
            "positions": player.positions,
            "selected_position": pos,
            "player_id": player.yahoo_id or "",
            "status": player.status,
            "wsgp": player.wsgp,
            "games": p.get("games_this_week", 0),
            "is_bench": pos in ("BN", "IL", "DL"),
            "is_il": "IL" in player.status or pos == "IL",
            "stats": player.pace or {},
            "rank": player.rank.to_dict(),
            "preseason": player.preseason.to_dict() if player.preseason else None,
        }
        # Flatten ROS stats for template tooltip (h[ros_key] access pattern)
        if player.ros is not None:
            entry.update(player.ros.to_dict())

        if is_pitcher:
            pitchers.append(entry)
        else:
            hitters.append(entry)

    slot_rank = {s: i for i, s in enumerate(HITTER_SLOTS_ORDER)}
    hitters.sort(key=lambda h: (slot_rank.get(h["selected_position"].upper(), 99), -h["wsgp"]))
    pitchers.sort(key=lambda p: (p["is_bench"], -p["wsgp"]))

    moves = optimal.get("moves", []) if optimal else []

    return {
        "hitters": hitters,
        "pitchers": pitchers,
        "is_optimal": len(moves) == 0,
        "moves": moves,
    }
```

Key changes:
- `Player.from_dict(p)` constructs typed object from the cached roster dict
- Fields accessed via typed attributes (`player.name`, `player.wsgp`, `player.rank.to_dict()`)
- `player.preseason.to_dict()` returns the full preseason stat bag (template accesses by key name, so extra keys are harmless)
- `entry.update(player.ros.to_dict())` flattens ROS stats to top level for the `h[ros_key]` template pattern
- `games_this_week` stays as `p.get(...)` since it's not a Player field (it's a transient display value)

- [ ] **Step 2: Run full test suite**

Run: `pytest tests/ -q`
Expected: All tests pass. The lineup template test (`test_lineup_page_renders`) verifies the page renders without error.

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "refactor: format_lineup_for_display uses Player objects from cache"
```
