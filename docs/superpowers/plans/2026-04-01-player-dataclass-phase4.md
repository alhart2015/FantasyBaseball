# Player Dataclass Phase 4: Eliminate Dict Intermediate

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the `roster_with_proj` dict intermediate from the refresh pipeline. Steps 6c through 11b work directly with `roster_players: list[Player]`, converting to Series/dict only at external function call boundaries.

**Architecture:** Remove the Phase 2 dict serialization loop. Steps 6c (pace), 6d (rankings) operate on Player attributes directly. Steps 7-11b convert to `player.to_series()` or `player.to_dict()` only when calling external library functions. Cache serialization happens once at the end.

**Tech Stack:** Python dataclasses (from Phase 1), existing pipeline code

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/web/season_data.py` | Modify | Replace all `roster_with_proj` dict usage with `roster_players` list[Player] |
| `src/fantasy_baseball/models/player.py` | Modify | Add `to_flat_dict()` for cache serialization with flat ROS stats |

---

### Task 1: Add to_flat_dict() to Player

**Files:**
- Modify: `src/fantasy_baseball/models/player.py`
- Modify: `tests/test_models/test_player.py`

The Phase 2 dict roundtrip does `d = player.to_dict()` then `d.update(player.ros.to_dict())` to flatten ROS stats. Encapsulate this in a proper method.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_models/test_player.py`:

```python
class TestToFlatDict:
    def test_flat_dict_has_ros_stats_at_top_level(self):
        from fantasy_baseball.models.player import Player, HitterStats
        p = Player(
            name="Aaron Judge", player_type="hitter",
            ros=HitterStats(pa=600, ab=500, h=145, r=95, hr=38, rbi=92, sb=7, avg=0.290),
        )
        d = p.to_flat_dict()
        assert d["hr"] == 38
        assert d["r"] == 95
        assert d["name"] == "Aaron Judge"

    def test_flat_dict_also_has_nested_ros(self):
        from fantasy_baseball.models.player import Player, HitterStats
        p = Player(
            name="Aaron Judge", player_type="hitter",
            ros=HitterStats(pa=600, ab=500, h=145, r=95, hr=38, rbi=92, sb=7, avg=0.290),
        )
        d = p.to_flat_dict()
        assert d["ros"]["hr"] == 38

    def test_flat_dict_no_ros_still_works(self):
        from fantasy_baseball.models.player import Player
        p = Player(name="Unknown", player_type="hitter")
        d = p.to_flat_dict()
        assert d["name"] == "Unknown"
        assert "hr" not in d
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_models/test_player.py::TestToFlatDict -v`
Expected: FAIL — `to_flat_dict` not found

- [ ] **Step 3: Implement to_flat_dict**

Add to `Player` class in `src/fantasy_baseball/models/player.py`:

```python
    def to_flat_dict(self) -> dict[str, Any]:
        """Serialize with ROS stats flattened to top level for legacy consumers.

        Produces both flat keys (r, hr, rbi...) AND nested ros dict.
        Used for cache serialization and backward compatibility with
        functions that expect flat stat keys.
        """
        d = self.to_dict()
        if self.ros is not None:
            d.update(self.ros.to_dict())
        return d
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_models/test_player.py -v`
Expected: All pass

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/models/player.py tests/test_models/test_player.py
git commit -m "feat: add Player.to_flat_dict() for cache-compatible serialization"
```

---

### Task 2: Replace Steps 6c, 6d, and cache write with roster_players

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`

Remove the `roster_with_proj` dict serialization loop. Steps 6c (pace) and 6d (rankings) work directly with `roster_players`.

- [ ] **Step 1: Remove the roster_with_proj serialization loop**

Find the block after `_progress(f"Matched {len(roster_players)} players to projections")` that creates `roster_with_proj`:

```python
        # Serialize to dicts for downstream pipeline steps (optimizer, waivers, trades)
        roster_with_proj = []
        for player in roster_players:
            d = player.to_dict()
            # Flatten ROS stats to top level for backward compatibility
            if player.ros is not None:
                d.update(player.ros.to_dict())
            roster_with_proj.append(d)
```

**Delete this entire block.**

- [ ] **Step 2: Update Step 6c (pace) to use roster_players**

Replace the Step 6c pace loop. Current code iterates `roster_with_proj` dicts:

```python
        for entry in roster_with_proj:
            norm = normalize_name(entry["name"])
            if "player_type" in entry:
                ptype = entry["player_type"]
            else:
                ptype = "pitcher" if set(entry.get("positions", [])) & PITCHER_POSITIONS else "hitter"
            if ptype == "hitter":
                actuals = hitter_logs.get(norm, {})
            else:
                actuals = pitcher_logs.get(norm, {})
            proj_keys = HITTER_PROJ_KEYS if ptype == "hitter" else PITCHER_PROJ_KEYS
            pre = preseason_lookup.get(norm, {})
            projected = {k: pre.get(k, 0) for k in proj_keys}
            entry["stats"] = compute_player_pace(actuals, projected, ptype)
```

Replace with:

```python
        for player in roster_players:
            norm = normalize_name(player.name)
            if player.player_type == "hitter":
                actuals = hitter_logs.get(norm, {})
            else:
                actuals = pitcher_logs.get(norm, {})
            proj_keys = HITTER_PROJ_KEYS if player.player_type == "hitter" else PITCHER_PROJ_KEYS
            pre = preseason_lookup.get(norm, {})
            projected = {k: pre.get(k, 0) for k in proj_keys}
            player.pace = compute_player_pace(actuals, projected, player.player_type)
```

- [ ] **Step 3: Update Step 6d (rankings) to use roster_players**

Replace the rankings attachment loop. Current code:

```python
        for entry in roster_with_proj:
            key = rank_key(entry["name"], entry.get("player_type", "hitter"))
            entry["rank"] = rankings_lookup.get(key, {})
```

Replace with:

```python
        from fantasy_baseball.models.player import RankInfo
        for player in roster_players:
            key = rank_key(player.name, player.player_type)
            rank_data = rankings_lookup.get(key, {})
            player.rank = RankInfo.from_dict(rank_data) if isinstance(rank_data, dict) else RankInfo()
```

- [ ] **Step 4: Update cache write to use roster_players**

Replace:

```python
        write_cache("roster", roster_with_proj, cache_dir)
```

With:

```python
        write_cache("roster", [p.to_flat_dict() for p in roster_players], cache_dir)
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/ -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "refactor: Steps 6c, 6d, cache write use roster_players directly"
```

---

### Task 3: Replace Steps 7-8 (optimizer, moves) with roster_players

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`

- [ ] **Step 1: Update Step 7 (optimizer) to use roster_players**

Replace:

```python
        hitter_players = []
        pitcher_players = []
        for p in roster_with_proj:
            positions = p.get("positions", [])
            if set(positions) & PITCHER_POSITIONS:
                pitcher_players.append(pd.Series(p))
            else:
                hitter_players.append(pd.Series(p))
```

With:

```python
        hitter_players = []
        pitcher_players = []
        for player in roster_players:
            if set(player.positions) & PITCHER_POSITIONS:
                pitcher_players.append(player.to_series())
            else:
                hitter_players.append(player.to_series())
```

- [ ] **Step 2: Update Step 8 (moves) to use roster_players**

Replace:

```python
        moves = []
        for slot, player_name in optimal_hitters.items():
            for p in roster_with_proj:
                if p["name"] == player_name:
                    current_slot = p.get("selected_position", "BN")
                    base_slot = slot.split("_")[0]
                    if current_slot.upper() != base_slot.upper():
                        moves.append({
                            "action": "START",
                            "player": player_name,
                            "slot": base_slot,
                            "reason": f"wSGP: {p.get('wsgp', 0):.1f}",
                        })
                    break
```

With:

```python
        moves = []
        for slot, player_name in optimal_hitters.items():
            for player in roster_players:
                if player.name == player_name:
                    current_slot = player.selected_position or "BN"
                    base_slot = slot.split("_")[0]
                    if current_slot.upper() != base_slot.upper():
                        moves.append({
                            "action": "START",
                            "player": player_name,
                            "slot": base_slot,
                            "reason": f"wSGP: {player.wsgp:.1f}",
                        })
                    break
```

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "refactor: Steps 7-8 (optimizer, moves) use roster_players directly"
```

---

### Task 4: Replace Steps 9-11b (starters, waivers, trades, buy-low) with roster_players

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`

- [ ] **Step 1: Update Step 9 (probable starters)**

Replace:

```python
        pitcher_roster_for_schedule = [
            p for p in roster_with_proj
            if set(p.get("positions", [])) & PITCHER_POSITIONS
        ]
```

With:

```python
        pitcher_roster_for_schedule = [
            p.to_flat_dict() for p in roster_players
            if set(p.positions) & PITCHER_POSITIONS
        ]
```

(`get_probable_starters` accesses dict fields like `p["name"]` and `p.get("team")`, so we pass dicts.)

- [ ] **Step 2: Update Step 10 (waivers)**

Replace:

```python
        roster_series = [pd.Series(p) for p in roster_with_proj]
```

With:

```python
        roster_series = [p.to_series() for p in roster_players]
```

- [ ] **Step 3: Update Step 11 (trades)**

Replace:

```python
        hart_roster_for_trades = [
            p for p in roster_with_proj
            if p.get("player_type") in ("hitter", "pitcher")
        ]
```

With:

```python
        hart_roster_for_trades = [
            p.to_flat_dict() for p in roster_players
            if p.player_type in ("hitter", "pitcher")
        ]
```

(`find_trades` accesses dict fields like `p["name"]`, `p.get("positions")`, `p["player_type"]` and passes to `_player_ros_stats` and `calculate_weighted_sgp`, so we pass flat dicts.)

- [ ] **Step 4: Remove any remaining references to roster_with_proj**

Search the function for any remaining `roster_with_proj` references. There should be none after all the above changes. If any remain (e.g., in Monte Carlo steps), replace with `roster_players` or `[p.to_flat_dict() for p in roster_players]` as appropriate.

Check specifically the Monte Carlo section (Step 13/13b) which may reference `roster_with_proj`.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "refactor: Steps 9-11b use roster_players, eliminate roster_with_proj"
```
