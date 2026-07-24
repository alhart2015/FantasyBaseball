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
3. **Per-player overlay:** for each preseason board row, use its current-talent
   line when one **exists and clears the min-PT floor** (below); otherwise keep
   preseason and flag it (`anchor_preseason_fallback`, surfaced in the report like
   `fallback_A`).
   **Join key:** normalized `name::player_type`. The board is keyed by `fg_id`
   (`blended_projections` PK is `(year, fg_id)`) and the current-talent lines are
   keyed by `mlbam_id`, so `name::player_type` is the only shared key -- which is
   the repo's documented cross-source-join convention (CLAUDE.md: "Name
   normalization ... is used for keeper matching and cross-source joins ... tie-break
   by VAR"). Use `load_full_season_lines()`'s `by_name` output, which already keeps
   the higher-volume record on a namesake collision.
   **Current-vs-preseason gate (min-PT floor):** use the current line only when its
   projected full-season volume clears `DEFAULT_MIN_AB` (100 AB, hitters) /
   `DEFAULT_MIN_IP` (20 IP, pitchers) -- the same floor `_below_min_pt` already
   applies to out-year ratios. A player who has barely played / not debuted has a
   present-but-tiny current line; the floor routes them to the preseason anchor
   rather than a noise-dominated one.
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

### Out-year regression (post-spot-check refinement)

The current anchor is right for 2026, but the out-years are `anchor x aging-ratio`,
so a partial-season breakout is *aged* forward, never *regressed* -- all three years
sit at the hot-2026 level. Live spot-check symptoms: a BABIP-driven career year (Otto
Lopez) out-ranked an elite talent in a down year (Soto), and one breakout (Wood)
dominated the three-year board.

Fix: `out_year_regression` (fraction in [0,1], `DEFAULT_OUT_YEAR_REGRESSION = 0.6`,
`--out-year-regression` CLI flag). For each out-year scored field, blend the
anchor-scaled value toward ZiPS's own out-year projection:
`line[f] = (1-lam)*scaled[f] + lam*zips_y[f]`. `lam=0` = pure anchor x ratio (the
over-indexing original); `lam=1` = pure ZiPS out-year (ignores 2026 for the future);
`0.6` = mostly-ZiPS. 2026 (the base year) is untouched.

Why lean on ZiPS's out-year: a ROS-signal diagnostic showed the projection systems
already sort skill from luck within the current year (Wood's gains are ISO/BB%-driven
and retained; Otto's are BABIP-driven), and ZiPS's *multi-year* out-year model regresses
the luck out entirely. So leaning the out-years on ZiPS inherits that skill-vs-luck
discrimination for free -- no custom breakout detector needed for v1 (that is the
DARKO trajectory follow-up). Empirically at `0.6`: Otto #18->#44, Soto #29->#10, Wood
stays #1 but at a sane level. The default flows through `build_results`, so the
keeper-trade generator uses the same regressed values.

### PT-heal for injury-shortened anchors (post-spot-check refinement)

The current anchor is projected-final *counting totals* (YTD+ROS), so a mid-season
injury shrinks the totals even though the rates (talent) are intact -- e.g. Aaron
Judge (half a season lost, 624->307 AB) fell to #136 at value 2.5 (would be 16.3
healthy). Fix: `heal_below` (`DEFAULT_PT_HEAL_BELOW = 0.65`, `--pt-heal` CLI flag)
in `overlay_current_anchors` -- **up only**. When a player's current PT (ab/ip) is
below `heal_below x` their preseason PT, scale the *counting* stats up to the healthy
PT (`factor = min(DEFAULT_PT_HEAL_CAP=2.0, preseason_PT/current_PT)`); the PT field
becomes `current_PT x factor`. Rates (avg/era/whip) carry the skill signal and are
never scaled, so a genuine decline is not healed -- only lost *time* is. The cap
prevents extrapolating a tiny, noisy sample to a full season; sub-floor players still
fall back to preseason. Empirically: Judge #136->#22; normal players move <=0.1.
`0` disables. Flows through `build_results` (trade generator inherits it).

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

- **No current line, or below the min-PT floor** (not yet debuted, barely played,
  season-long injury): keep preseason anchor, flag `anchor_preseason_fallback`.
  The existing per-out-year `approach_a`/`fallback_A` path is unaffected.
- **Mid-season injury -- handled by PT-heal (see above).** A player who played a
  meaningful chunk (clears the min-PT floor) then was hurt has a *depressed*
  projected-final line (YTD + near-zero ROS). The `heal_below` PT-heal scales his
  counting stats up to a healthy PT (rates held), so lost *time* no longer negates
  keeper talent; a full-season decline (normal PT, weak rates) is untouched.
- **Player in current blob but not the preseason board** (a call-up absent from
  the preseason blend): out of scope for the MVP -- the preseason board is the
  coverage base. Note the count of such skips so the gap is visible, don't hide
  it. (Adding them is a follow-up.)
- **Namesakes** (two "Mason Miller" pitchers): the `name::player_type` join keeps
  the higher-volume record, as `load_full_season_lines`' `by_name` output already
  does. (No mlbam join is available -- the board is fg_id-keyed.)
- **Stale/missing Upstash blob:** fail loud in `current` mode (see Mode toggle).

## Testing

- `_value_of_line` / ratio math is untouched -- existing `test_keeper_value.py`
  guards it.
- New unit tests around the anchor overlay (pure, no I/O): given a preseason
  frame + a current-talent lines dict, assert (a) a player with a current line
  above the min-PT floor gets the current stats, (b) a player with no current
  line keeps preseason and is flagged, (c) a player whose current line is BELOW
  the min-PT floor keeps preseason and is flagged, (d) a namesake collision
  resolves to the higher-volume record via the `name::player_type` join, (e)
  `preseason` mode is a no-op overlay.
- **Regression fixture (concrete):** a preseason-modest / current-breakout hitter
  has a strictly higher discounted keeper total AND a better (lower) rank under
  `current` than under `preseason` for the same discount/horizon.
- **Characterization:** `preseason` mode reproduces the pre-change `build_results`
  output exactly (same `KeeperValueResult` list), proving the default-off path is
  behavior-preserving.
- **Fail-loud:** `current` mode with a missing/empty `full_season_projections`
  blob raises the specified "run a refresh" error rather than returning a
  preseason-labelled result.

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
