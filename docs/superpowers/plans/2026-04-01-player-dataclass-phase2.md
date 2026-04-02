# Player Dataclass Phase 2: Adopt in Season Dashboard Refresh

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Use Player dataclass for roster enrichment in the season dashboard refresh pipeline, replacing inline dict manipulation with typed construction and methods.

**Architecture:** After `match_roster_to_projections` builds the matched dict entries, construct Player objects to populate preseason stat bags, compute wSGP, and attach ranks. Write results back to the entry dicts for downstream compatibility (optimizer, waivers, trades still consume `list[dict]`). Cache serialization preserves the existing flat format so templates work unchanged.

**Tech Stack:** Python dataclasses (from Phase 1), existing pipeline code

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/web/season_data.py` | Modify | Use Player objects in Step 6 enrichment loop |
| `tests/test_models/test_player.py` | Modify | Add roundtrip test for cache format compatibility |

---

### Task 1: Add cache-compatible serialization test

**Files:**
- Modify: `tests/test_models/test_player.py`

This test verifies that `Player.to_dict()` produces a format that the rest of the pipeline can consume — specifically that `format_lineup_for_display` can read the output.

- [ ] **Step 1: Write the test**

Add to `tests/test_models/test_player.py`:

```python
class TestCacheCompatibility:
    def test_to_dict_preserves_flat_stat_keys_via_ros(self):
        """Verify to_dict includes nested ros dict that format_lineup_for_display can read."""
        from fantasy_baseball.models.player import Player, HitterStats, RankInfo
        p = Player(
            name="Aaron Judge",
            player_type="hitter",
            positions=["OF"],
            team="NYY",
            fg_id="15640",
            yahoo_id="12345",
            selected_position="OF",
            ros=HitterStats(pa=600, ab=500, h=145, r=95, hr=38, rbi=92, sb=7, avg=0.290),
            preseason=HitterStats(pa=650, ab=550, h=160, r=110, hr=45, rbi=120, sb=5, avg=0.291),
            wsgp=12.5,
            rank=RankInfo(ros=2, preseason=1, current=3),
            pace={"R": {"actual": 15, "expected": 14, "z_score": 0.5}},
        )
        d = p.to_dict()
        # Core identity
        assert d["name"] == "Aaron Judge"
        assert d["player_type"] == "hitter"
        assert d["player_id"] == "12345"
        # ROS stats in nested dict
        assert d["ros"]["hr"] == 38
        # Preseason in nested dict
        assert d["preseason"]["hr"] == 45
        # wSGP
        assert d["wsgp"] == 12.5
        # Rank
        assert d["rank"]["ros"] == 2
        # Pace stored as "stats"
        assert d["stats"]["R"]["actual"] == 15

    def test_player_from_dict_roundtrip_with_all_fields(self):
        """Full roundtrip: construct Player, serialize, reconstruct, compare."""
        from fantasy_baseball.models.player import Player, HitterStats, RankInfo
        original = Player(
            name="Aaron Judge",
            player_type="hitter",
            positions=["OF"],
            team="NYY",
            fg_id="15640",
            mlbam_id=592450,
            yahoo_id="12345",
            selected_position="OF",
            status="",
            ros=HitterStats(pa=600, ab=500, h=145, r=95, hr=38, rbi=92, sb=7, avg=0.290),
            preseason=HitterStats(pa=650, ab=550, h=160, r=110, hr=45, rbi=120, sb=5, avg=0.291),
            wsgp=12.5,
            rank=RankInfo(ros=2, preseason=1, current=3),
        )
        d = original.to_dict()
        restored = Player.from_dict(d)
        assert restored.name == original.name
        assert restored.player_type == original.player_type
        assert restored.ros.hr == original.ros.hr
        assert restored.preseason.hr == original.preseason.hr
        assert restored.wsgp == original.wsgp
        assert restored.rank.ros == original.rank.ros
```

- [ ] **Step 2: Run test**

Run: `pytest tests/test_models/test_player.py::TestCacheCompatibility -v`
Expected: PASS (these test existing functionality)

- [ ] **Step 3: Commit**

```bash
git add tests/test_models/test_player.py
git commit -m "test: add cache format compatibility tests for Player serialization"
```

---

### Task 2: Use Player objects in Step 6 roster enrichment

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`

This is the core Phase 2 change. Replace the inline dict enrichment with Player construction and methods.

- [ ] **Step 1: Rewrite Step 6 to use Player objects**

In `src/fantasy_baseball/web/season_data.py`, find Step 6 (~lines 813-844). The current code:

```python
        # --- Step 6: Match roster players to projections, compute wSGP ---
        _progress("Matching roster to projections...")

        # Match preseason projections for tooltip comparison (main stats are ROS)
        preseason_matched = match_roster_to_projections(
            roster_raw, preseason_hitters, preseason_pitchers,
        )
        preseason_lookup = {normalize_name(p["name"]): p for p in preseason_matched}

        # Build lookup of matched players, add wSGP
        matched_names = set()
        roster_with_proj = []
        for entry in matched:
            entry["wsgp"] = calculate_weighted_sgp(pd.Series(entry), leverage)
            norm = normalize_name(entry["name"])
            matched_names.add(norm)
            # Attach preseason projection stats for tooltip comparison
            pre_entry = preseason_lookup.get(norm)
            if pre_entry:
                entry["preseason"] = {
                    k: pre_entry.get(k, 0)
                    for k in (["r", "hr", "rbi", "sb", "avg"] if entry.get("player_type") == "hitter"
                              else ["w", "k", "sv", "era", "whip"])
                }
            roster_with_proj.append(entry)
        # Include unmatched players with wsgp=0
        for player in roster_raw:
            if normalize_name(player["name"]) not in matched_names:
                entry = dict(player)
                entry["wsgp"] = 0.0
                roster_with_proj.append(entry)
```

Replace with:

```python
        # --- Step 6: Match roster players to projections, compute wSGP ---
        _progress("Matching roster to projections...")
        from fantasy_baseball.models.player import Player, HitterStats, PitcherStats

        # Match preseason projections for tooltip comparison
        preseason_matched = match_roster_to_projections(
            roster_raw, preseason_hitters, preseason_pitchers,
        )
        preseason_lookup = {normalize_name(p["name"]): p for p in preseason_matched}

        # Build Player objects from matched entries
        matched_names = set()
        roster_players: list[Player] = []
        for entry in matched:
            norm = normalize_name(entry["name"])
            matched_names.add(norm)

            player = Player.from_dict(entry)

            # Attach preseason stat bag
            pre_entry = preseason_lookup.get(norm)
            if pre_entry:
                if player.player_type == "hitter":
                    player.preseason = HitterStats.from_dict(pre_entry)
                else:
                    player.preseason = PitcherStats.from_dict(pre_entry)

            # Compute wSGP via Player method
            player.compute_wsgp(leverage)

            roster_players.append(player)

        # Include unmatched players
        for raw_player in roster_raw:
            if normalize_name(raw_player["name"]) not in matched_names:
                player = Player.from_dict({
                    **raw_player,
                    "player_type": "pitcher" if set(raw_player.get("positions", [])) & PITCHER_POSITIONS else "hitter",
                })
                roster_players.append(player)

        _progress(f"Matched {len(roster_players)} players to projections")
```

- [ ] **Step 2: Convert roster_players to roster_with_proj dicts for downstream**

Right after the Player construction loop, add conversion back to dicts for backward compatibility with the optimizer, waivers, and trades steps:

```python
        # Serialize to dicts for downstream pipeline steps (optimizer, waivers, trades)
        # These will be migrated to use Player objects in Phase 4.
        roster_with_proj = []
        for player in roster_players:
            d = player.to_dict()
            # Flatten ROS stats to top level for backward compatibility
            if player.ros is not None:
                d.update(player.ros.to_dict())
            roster_with_proj.append(d)
```

This ensures `roster_with_proj` has flat stat keys at the top level (r, hr, rbi, etc.) for the optimizer, waivers, and trades to consume as before, plus the nested `ros` and `preseason` dicts.

- [ ] **Step 3: Update pace computation to use preseason_lookup (no change needed)**

The pace computation in Step 6c already uses `preseason_lookup.get(norm, {})`. This doesn't change — it still reads from the dict-based preseason_lookup. The pace result is written to `entry["stats"]`, which is now `d["stats"]` in `roster_with_proj`. No change needed here.

BUT: we need to update the pace loop to also set `player.pace` on the Player objects so they stay in sync:

After the pace loop (Step 6c), add:

```python
        # Sync pace back to Player objects
        for player, entry in zip(roster_players, roster_with_proj):
            player.pace = entry.get("stats")
```

Wait — this won't work cleanly because `roster_with_proj` includes unmatched players that may not align 1-to-1 with entries. Let me simplify: keep the pace loop writing to `roster_with_proj` dicts (as it does now). The Player objects already served their purpose (wsgp computation, preseason attachment). We don't need to sync pace back.

- [ ] **Step 4: Update rankings attachment to use Player objects**

In Step 6d, the rankings are attached to `roster_with_proj` entries. Update to use `roster_players` for the rank key construction (since players have `player_type`), then write back to the dict:

The existing code:
```python
        for entry in roster_with_proj:
            key = rank_key(entry["name"], entry.get("player_type", "hitter"))
            entry["rank"] = rankings_lookup.get(key, {})
```

This already works correctly with the dict entries since they have `player_type`. No change needed.

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "feat: use Player dataclass for roster enrichment in refresh pipeline"
```

---

### Task 3: Use Player objects in player search API

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`

The player search API currently builds result dicts manually. Use Player objects instead.

- [ ] **Step 1: Refactor result building to use Player**

In `src/fantasy_baseball/web/season_routes.py`, find the `api_player_search` endpoint. The result building loop currently constructs dicts manually. Replace with Player construction:

Find the loop that starts with `for ros in ros_rows:` and builds `results`. Replace the result dict construction with:

```python
            from fantasy_baseball.models.player import Player, HitterStats, PitcherStats, RankInfo

            # (move this import to the top of the function, before the loop)
```

Then in the loop, replace the manual dict construction with:

```python
                # Build Player object
                stats_cls = HitterStats if ptype == "hitter" else PitcherStats
                player = Player(
                    name=name,
                    player_type=ptype,
                    team=ros_dict.get("team", ""),
                    positions=positions,
                    ros=stats_cls.from_dict(ros_dict),
                    preseason=stats_cls.from_dict(pre) if pre else None,
                    wsgp=round(wsgp, 2),
                    rank=RankInfo.from_dict(rank),
                    pace=pace,
                )

                # Serialize — include ownership (not a Player field)
                result = player.to_dict()
                result["ownership"] = ownership
                results.append(result)
```

Remove the manual `ros_stats`, `pre_stats` dict construction and the final `results.append({...})` block — the Player handles it.

- [ ] **Step 2: Run tests**

Run: `pytest tests/test_web/test_player_search.py -v`
Expected: All pass

The test `test_search_returns_matching_players` checks `data[0]["ros"]["hr"] == 38` — this should still work since `Player.to_dict()` nests ROS under `"ros"`.

Actually, check: does the test expect nested or flat? Let me verify:

The test asserts:
```python
assert data[0]["ros"]["hr"] == 38
assert data[0]["preseason"]["hr"] == 45
```

This expects nested format — which is what `Player.to_dict()` produces. Good.

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -q`
Expected: All pass

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py
git commit -m "feat: use Player dataclass in player search API"
```
