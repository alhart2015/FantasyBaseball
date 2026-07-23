# Keeper value: current-season anchor (in-season breakout fix)

## Problem

The keeper-value metric (`scripts/keeper_value.py` + `src/fantasy_baseball/analysis/keeper_value.py`,
shipped in PR #254) values every player off a **preseason** projection and ignores
~5 months of in-season results. A mid-season breakout (e.g. James Wood, the best
fantasy hitter in 2026) does not move his keeper value, so he never reaches the
top 30.

Root cause, confirmed by tracing the code and the data:

- The 2026 **anchor line** comes from `get_blended_projections(conn)` -> the
  `blended_projections` table, which `load_blended_projections` builds from
  `data/projections/2026/*.csv` -- all dated 2026-03-24 (the preseason 5-system
  blend: ATC/Steamer/ZiPS/THE BAT X/oopsy).
- Out-year values are `anchor x (ZiPS_year / ZiPS_2026)` per stat
  (`_scale_line`, `keeper_value.py:82`). ZiPS supplies only the aging *ratio*;
  the **level** is the anchor. So every year's value derives from the preseason
  anchor.
- The ZiPS out-year CSVs are themselves preseason: today's
  `data/projections/2027/zips-hitters.csv` differs from the co-located
  `zips-hitters-2027-proj-from-2026-03-25.csv` on only ~21 of ~1900 hitters, and
  Wood's 2027 line is byte-identical. ZiPS out-year ("ZiPS-3YR") projections are
  **start-of-season by design** and only refresh via a manual midseason full-fat
  run (2026: published June 24-25) that our downloaded file did not pick up.

Because the out-year *level* is `anchor x ratio`, the **anchor is the single
lever** for all three years. Fixing it fixes 2026, 2027, and 2028 together.

## Goal

Anchor the keeper valuation on a **current-season talent line** instead of the
preseason blend, so in-season breakouts (and busts) flow through to keeper value.

### Non-goals (explicit follow-ups, out of scope here)

- **Refreshing the ZiPS out-year *ratios*** to the midseason full-fat run. A
  stale aging ratio between two preseason years (Wood 26 -> 28 HR) is a fine
  aging prior; the damage was the stale *level*. Tracked separately.
- **A DARKO-style true-talent trajectory / comp-based career-arc model** that
  replaces the borrowed ZiPS aging shape and explicitly detects breakouts vs.
  declines. This is the larger vision; tracked as its own issue.
- Any change to the discounting, VAR pipeline, out-year loader, or report.

## Design

### The one change: the anchor source

Today `build_results` (`scripts/keeper_value.py`) feeds the preseason
`get_blended_projections(conn)` frames into `build_board_from_frames(...)`, and
each board row's line becomes the `anchor_line` passed to `keeper_value(...)`.

We add an **anchor mode** and, in the default `current` mode, replace each
player's anchor with a **current-talent full-season line** where one exists,
falling back to the preseason line otherwise. Everything downstream -- the
ZiPS-ratio out-year scaling, `_value_of_line`/VAR, discounting, the transparency
columns, and the report -- is unchanged. The `keeper_trades.py` generator, which
consumes `build_results`, inherits the fix automatically.

### Current-talent line: reuse the existing full-season blob

The current-talent line is already computed and maintained -- **no new YTD/ROS
math**:

- `data/ros_pipeline.py::derive_full_season` adds season-to-date actuals to the
  ROS-remaining blend -> full-season totals, persisted as
  `CacheKey.FULL_SEASON_PROJECTIONS` (`cache:full_season_projections`). This is a
  multi-system blend + YTD -- i.e. the current-season **consensus** full-season
  line, which preserves the design's "consensus level, borrowed aging shape"
  principle (now current instead of preseason).
- `analysis/draft_value.py::load_full_season_lines()` reads that blob and returns
  per-player lines keyed by `(mlbam_id, player_type)` (authoritative,
  namesake-immune) and by `name::player_type` (volume-tiebroken fallback).

### Assembling the anchor board (`current` mode)

1. Load the preseason blended hitter/pitcher frames (as today) -- the **coverage
   base** so every keeper-eligible player is still ranked.
2. Load current-talent lines from `cache:full_season_projections`, read **fresh
   from Upstash** (local SQLite may be stale -- see CLAUDE.md; mirror the
   `keeper_trades.py` / `refresh_remote.py` explicit-Upstash path rather than the
   default `get_kv()` local read).
3. **Per-player overlay:** for each preseason board row, if a current-talent line
   exists for that player, replace the stat line with it; else keep preseason and
   flag it (`anchor_preseason_fallback`, surfaced in the report like `fallback_A`).
   Join by `mlbam_id` where the board row carries it (the ZiPS CSVs include
   `MLBAMID`), else by normalized `name::player_type` -- the same dual-key scheme
   `load_full_season_lines` uses; namesake collisions resolve by the existing
   volume/VAR tiebreak conventions.
4. Feed the overlaid frames to `build_board_from_frames(...)`. The board, its
   `ScaleInputs` (denominators from `config` are league constants and unchanged;
   `team_ab`/`team_ip` and replacement levels re-derive from the current board),
   `positions`, and anchors are then all current-talent-consistent.

### Mode toggle

- `--anchor current|preseason`, default **`current`**.
- `preseason` reproduces today's behavior exactly (skip steps 2-3), for
  side-by-side comparison and offseason use.
- If `current` is requested but `cache:full_season_projections` is missing/empty
  (unsynced), fail loud with a clear message to run a refresh -- do NOT silently
  serve preseason under a `current` label (mislabeled data drives wrong keeper
  decisions).

## Data flow (current mode)

```
preseason blended_projections (SQLite)  ---- coverage base ---.
                                                              v
cache:full_season_projections (Upstash, YTD+ROS blend) -> per-player overlay
   via load_full_season_lines()            (current where present, else preseason)
                                                              |
                                                              v
                                   build_board_from_frames(overlaid frames)
                                                              |
                                          board + ScaleInputs (current-talent)
                                                              |
   ZiPS 2026/2027/2028 preseason ratios (unchanged) ---> keeper_value(...) per row
                                                              |
                          per-year VAR -> discounted_total -> ranked report
```

## Edge cases

- **Player in preseason board, no current line** (not yet debuted, injured all
  year, sub-min-PT): keep preseason anchor, flag `anchor_preseason_fallback`.
  The existing per-out-year `approach_a`/`fallback_A` path is unaffected.
- **Player in current blob but not the preseason board** (a call-up absent from
  the preseason blend): out of scope for the MVP -- the preseason board is the
  coverage base. Note the count of such skips so the gap is visible, don't hide
  it. (Adding them is a follow-up.)
- **Namesakes** (two "Mason Miller" pitchers): join by `mlbam_id` first; the
  `name::player_type` fallback keeps the higher-volume record, as
  `load_full_season_lines` already does.
- **Stale/missing Upstash blob:** fail loud in `current` mode (see Mode toggle).

## Testing

- `_value_of_line` / ratio math is untouched -- existing `test_keeper_value.py`
  guards it.
- New unit tests around the anchor overlay (pure, no I/O): given a preseason
  frame + a current-talent lines dict, assert (a) a player with a current line
  gets the current stats, (b) a player without one keeps preseason and is
  flagged, (c) the mlbam join beats the name fallback on a namesake, (d)
  `preseason` mode is a no-op overlay.
- A regression fixture pinning the motivating case: a preseason-modest / current-
  breakout hitter ranks materially higher under `current` than `preseason`.

## Reuse summary

| Need | Reuse |
|------|-------|
| Current full-season (YTD+ROS) lines | `cache:full_season_projections` via `load_full_season_lines()` (`draft_value.py:597`); derivation is `derive_full_season` (`ros_pipeline.py:140`) |
| Fresh Upstash read | explicit-Upstash path per `keeper_trades.py` / `refresh_remote.py` |
| Board + scale + anchors | `build_board_from_frames` (`draft/board.py:55`) -- unchanged, fed different frames |
| Out-year ratios, VAR, discounting, report | `analysis/keeper_value.py` + `scripts/keeper_value.py` -- unchanged |

## Follow-ups (file as issues)

1. Refresh the ZiPS out-year ratios to the midseason full-fat run (re-download +
   spot-check a known breakout's 2027 vs. preseason to confirm the pages
   refreshed).
2. DARKO-style true-talent trajectory / comp-based career-arc model with explicit
   breakout & decline detection (the long-term vision; supersedes the borrowed
   ZiPS aging shape).
3. Include current-blob players absent from the preseason board (call-ups).
