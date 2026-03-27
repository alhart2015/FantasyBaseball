# Smarter Waiver Pickups

**Date:** 2026-03-27
**Status:** Approved

## Problem

The current waiver scanner has two issues:

1. **No positions displayed** — the recommendation card doesn't show positions for the add or drop player, making it hard to understand cross-position swaps.
2. **1-for-1 same-type only** — it only considers dropping the weakest player of the same type (hitter/pitcher). Can't pick up a 3B and drop an OF even if your bench SS can fill the OF slot. The resulting roster might not be valid, or might be suboptimal.

## Solution

Replace the swap logic in `scan_waivers()` with a whole-roster re-optimization approach: for each candidate FA × drop combination, build the hypothetical post-swap roster, run the Hungarian optimizer to find the best lineup, and compare total team wSGP before vs after. Show positions on the card, and an expandable before/after optimal lineup comparison.

## Algorithm

### Baseline

Before evaluating any swaps, compute the baseline:
1. Separate roster into hitters and pitchers (exclude IL-designated players from active consideration)
2. Run `optimize_hitter_lineup()` on hitters → baseline hitter lineup (returns `dict[str, str]` of slot → player name)
3. Run `optimize_pitcher_lineup()` on pitchers → baseline pitcher starters list
4. Sum wSGP of only **assigned players** (those who appear in the optimizer output) → `baseline_wsgp`. Players who don't get assigned to a slot contribute 0, even if their individual wSGP is positive.

### Pruning

Pre-filter free agents to reduce search space:
- Compute wSGP for every FA
- Find the 3rd-lowest wSGP among **active-slot roster players** (exclude IL/bench-only players from the floor calculation — they inflate the floor but don't contribute to the optimal lineup)
- Skip any FA with wSGP below `wsgp_floor` (they can't improve the roster)

### Swap evaluation

For each surviving FA, for each roster player (any type, not just same-type):

1. **Feasibility check**:
   - **Same-type swap (hitter→hitter):** Run `can_cover_slots()` on the hypothetical hitter roster after removing the dropped player and adding the FA. This catches position holes (e.g., dropping the only C-eligible player).
   - **Cross-type swap (drop hitter, add pitcher):** Run `can_cover_slots()` on the hypothetical hitter roster (minus the dropped player) to verify the remaining hitters can still fill all hitter slots. Also verify `len(new_pitchers) >= P_slots`.
   - **Cross-type swap (drop pitcher, add hitter):** Verify `len(new_pitchers) >= P_slots` (pitchers are interchangeable — just check count). Run `can_cover_slots()` on the new hitter roster (plus the FA) to verify all hitter slots can be filled.
   - **Same-type swap (pitcher→pitcher):** Verify `len(new_pitchers) >= P_slots`. No position matching needed since all P slots are interchangeable.

2. **Build hypothetical roster**: Remove dropped player, add FA.

3. **Re-optimize**: Run `optimize_hitter_lineup()` and `optimize_pitcher_lineup()` on the new roster. Both are needed because even a hitter swap can change who's on the bench.

4. **Compute team wSGP**: Sum wSGP of only **assigned players** in the optimizer output → `new_wsgp`. Iterate the `lineup` dict values (hitters) and pitcher starters list, look up each player's wSGP. Do NOT sum all roster players — only those the optimizer actually placed in a slot.

5. **Compare**: `gain = new_wsgp - baseline_wsgp`. If gain > 0, record the recommendation.

6. **Store lineups**: Save both the baseline and new optimal lineup assignments for the expanded card display.

### Deduplication

For each FA, keep only the best drop candidate (highest gain). Multiple FAs can recommend dropping the same roster player — that's fine, the user will only act on one.

### Result

Return top `max_results` (default 10) recommendations sorted by gain descending.

## Recommendation data structure

```python
{
    "add": "Jose Ramirez",
    "add_positions": ["3B", "SS"],
    "drop": "Josh Lowe",
    "drop_positions": ["OF"],
    "sgp_gain": 0.85,          # total team wSGP improvement
    "categories": {"R": 0.2, "HR": 0.1, ...},  # per-category delta
    "lineup_before": [         # baseline optimal lineup
        {"name": "Player A", "slot": "C", "wsgp": 1.2},
        {"name": "Josh Lowe", "slot": "OF", "wsgp": 0.5, "is_dropped": True},
        ...
    ],
    "lineup_after": [          # new optimal lineup
        {"name": "Player A", "slot": "C", "wsgp": 1.2},
        {"name": "Jose Ramirez", "slot": "SS", "wsgp": 1.8, "is_added": True},
        {"name": "Bobby Witt Jr.", "slot": "OF", "wsgp": 2.4, "moved_from": "SS"},
        ...
    ],
}
```

`lineup_before` and `lineup_after` include both hitters and pitchers, sorted by slot order. Each entry has `name`, `slot`, `wsgp`. Special flags: `is_dropped` (red in before), `is_added` (green in after), `moved_from` (arrow in after showing old slot).

## Display

### Collapsed card (same structure as current, plus positions)

```
ADD Jose Ramirez (3B, SS) / DROP Josh Lowe (OF)     +0.85 wSGP
[per-category deltas]
```

### Expanded card (click to toggle)

Two side-by-side lineup tables: "Before" and "After"

**Before lineup:**
- All starting players in their optimal slots
- Dropped player's row highlighted red
- Bench players shown dimmed

**After lineup:**
- All starting players in their new optimal slots
- Added player's row highlighted green
- Players who changed slots show "← was SS" indicator
- Bench players shown dimmed

## Architecture

### Changes to `waivers.py`

**`scan_waivers()`** — rewrite the Phase 2 (swap) logic:
- Compute baseline optimal lineups and total wSGP
- Pre-filter FAs by wSGP floor
- For each FA × roster player combination:
  - Quick feasibility check
  - Build hypothetical roster
  - Re-optimize
  - Compare total wSGP
  - Track best drop per FA
- Include `add_positions`, `drop_positions`, `lineup_before`, `lineup_after` in results

**`evaluate_pickup()`** — still used for per-category SGP deltas. Called after identifying the best swap to populate the `categories` field. **Important:** the `sgp_gain` value from `evaluate_pickup()` is the individual player delta (add_wsgp - drop_wsgp), NOT the team-level gain. The recommendation's `sgp_gain` field must come from the optimizer comparison (`new_wsgp - baseline_wsgp`). Discard `evaluate_pickup()`'s `sgp_gain` — only use its `categories` dict.

**New helper: `_compute_team_wsgp()`** — given a roster, leverage, and roster_slots, runs both optimizers and returns (total_wsgp, hitter_lineup_dict, pitcher_starters_list). This is the inner loop function that must be fast. Pre-compute `denoms` once via `get_sgp_denominators()` and pass to `calculate_weighted_sgp(player, leverage, denoms=denoms)` to avoid per-call dict copies.

**New helper: `_build_lineup_summary()`** — given optimizer output (hitter `dict[str, str]` and pitcher `list[dict]`), the player roster, and a wSGP lookup dict, builds the lineup list for the expanded card. Must handle:
- Stripping `_N` suffixes from optimizer slot keys (e.g., `"OF_2"` → `"OF"` for display)
- Looking up wSGP by player name from a pre-computed dict
- Identifying bench players as roster members not in any optimizer assignment
- Computing `moved_from` by diffing before and after lineups by player name (O(n) pass)
- Excluding IL-designated players from the lineup display

### Phase 1 (pure adds) stays the same

The empty-slot filling logic is unchanged — it doesn't need re-optimization since it's adding to open slots.

### Changes to `waivers_trades.html`

- Add positions after player names in waiver cards: `(3B, SS)` and `(OF)`
- Make waiver cards expandable (like buy-low cards)
- Expanded view shows two-column before/after lineup comparison
- CSS for red/green highlights and move arrows

### Changes to `season_routes.py`

None needed — the route already passes `waivers` to the template.

## Performance

The inner loop is: for each FA (~150 after pruning) × each roster player (~25) = ~3,750 combinations. Each combination runs:
- `can_cover_slots()` — bipartite matching on 14-slot matrix, microseconds
- `optimize_hitter_lineup()` — Hungarian on 14×14 matrix, microseconds
- `optimize_pitcher_lineup()` — simple sort of 9 pitchers, microseconds
- `calculate_weighted_sgp()` — arithmetic, microseconds

Total: ~3,750 iterations × ~1ms each = ~4 seconds. Acceptable within the refresh pipeline.

Pre-computing wSGP for all players once (not per-iteration) is important — `calculate_weighted_sgp` calls `get_sgp_denominators()` internally. Cache the denominators and pass them through.

## Edge cases

- **Dropping the only eligible player at a position**: `can_cover_slots()` catches this — the swap is skipped.
- **Cross-type swaps** (drop pitcher, add hitter or vice versa): Allowed. The hitter/pitcher split changes. Verify both sides still have enough players for their slots.
- **FA with no projection match**: Already filtered out by `fetch_and_match_free_agents()` — unmatched FAs aren't in the list.
- **Position sourcing**: `add_positions` comes from `fa["positions"]` (set by `fetch_and_match_free_agents`). `drop_positions` comes from the roster player's `positions` field (set by `match_roster_to_projections`). For unmatched roster players (wSGP = 0), fall back to the raw Yahoo roster's position data.
- **Dropping a player on IL**: Valid — IL players don't contribute wSGP to the optimal lineup. Dropping them frees a roster spot.
- **All FAs below wSGP floor**: No swap recommendations. Pure adds still work if there are open slots.

## What doesn't change

- `detect_open_slots()` — unchanged
- `fetch_and_match_free_agents()` — unchanged
- `evaluate_pickup()` — still used for category deltas
- `can_cover_slots()` — still used for feasibility
- `optimize_hitter_lineup()` and `optimize_pitcher_lineup()` — unchanged, just called more
- Phase 1 (pure adds to empty slots) — unchanged
- Refresh pipeline Step 10 call site — same function, richer output
