# Standings page shows 75.5 / 74.5 when Yahoo shows a 74.5 tie — root cause and fix

**Date:** 2026-04-18
**Branch:** `debug-standings-mismatch`
**Reporter:** Alden

## Symptom

Our season dashboard (`/standings`) shows:
- Spacemen — 75.5 pts (rank 1)
- Hart of the Order — 74.5 pts (rank 2)

Yahoo's official standings page shows the two teams **tied at 74.5**.

## Investigation

`scripts/debug_standings_mismatch.py` pulls the raw Yahoo standings,
extracts both Yahoo's `team_standings.points_for` (the roto total Yahoo
displays) and the per-category stat totals, then runs our
`fantasy_baseball.scoring.score_roto` over those stats and prints the
diff.

Result on 2026-04-18:

| Team | Yahoo pts | Our total | Diff (ours − Yahoo) |
| --- | ---: | ---: | ---: |
| Spacemen | **74.5** | **75.50** | **+1.00** |
| Hart of the Order | 74.5 | 74.50 | 0.00 |
| Hello Peanuts! | 69.5 | 69.00 | −0.50 |
| Jon's Underdogs | 56.5 | 56.00 | −0.50 |
| Tortured Baseball Department | 56.5 | 56.50 | 0.00 |
| Springfield Isotopes | 55 | 55.00 | 0.00 |
| Boston Estrellas | 45 | 45.00 | 0.00 |
| SkeleThor | 44.5 | 44.50 | 0.00 |
| Work in Progress | 39 | 39.50 | +0.50 |
| Send in the Cavalli | 35 | 34.50 | −0.50 |

The diffs sum to zero, all in units of ±0.5 — the signature of
tie-breaking disagreement.

## Root cause

Yahoo's standings API exposes per-category stat totals only at display
precision:

- `AVG` → 3 decimals (`".241"`)
- `ERA` / `WHIP` → 2 decimals (`"3.60"`, `"1.03"`)
- counting stats are integers

Yahoo's *scoring engine*, however, runs on full-precision internal
stats. So when two teams display with the same rounded rate
(e.g. WHIP `1.03` for Spacemen and Isotopes, AVG `.241` for Spacemen
and Cavalli), Yahoo's engine still picks a winner — the teams'
underlying `(BB + H_allowed) / IP` and `H / AB` values diverge
somewhere below the rounding.

Our `score_roto` operates on those **rounded** values. When it sees
equal rounded stats, it treats them as genuine ties and awards the
averaged rank (9.5/9.5 instead of 10/9). Each display-level tie
shifts our total by ±0.5 relative to Yahoo.

Spacemen has two such ties:

- WHIP `1.03` vs Isotopes `1.03` → Yahoo breaks against Spacemen → our +0.5
- AVG `.241` vs Cavalli `.241` → Yahoo breaks against Spacemen → our +0.5

Net: +1.0 — exactly what the diff table shows. The other ±0.5 diffs
come from the other display-level ties in the league
(Underdogs/Isotopes tied SB at 24; Peanuts/Isotopes tied AVG at .238;
WIP/Underdogs tied AVG at .253).

This is **not** an error in the category stats we cache (those match
Yahoo's displayed values exactly). It is purely a scoring-engine
precision issue. Yahoo knows the real tie-break; we can't recover it
without fetching full-precision component stats.

## Fix

Yahoo's standings API already exposes the answer: each team's
`team_standings.points_for` field carries Yahoo's authoritative roto
total, computed against the full-precision internal stats.

1. **Parse `points_for`** in `yahoo_roster.parse_standings_raw` and
   include it in the cached standings dicts.
2. **Carry it through the typed layer** —
   `StandingsEntry.yahoo_points_for` (optional `float | None`).
3. **Prefer it in `format_standings_for_display`** — when every entry
   has `yahoo_points_for` set (live Yahoo standings path), the
   displayed total and rank come straight from Yahoo. When any entry
   lacks it (projected standings, built from our own ROS projections
   and scored by `score_roto`), we fall through to the existing
   behavior.

Per-category points remain `score_roto`'s output, so the per-cat
numbers may not exactly sum to the Yahoo-authoritative total — the
difference is ±0.5 per tie and nets to zero across the league. That
tradeoff is preferable to showing a wrong headline total. A fully
consistent fix would require fetching raw component stats (H, AB, BB,
H_allowed, ER, IP) and recomputing rate stats ourselves, which is
outside the scope of this PR.

## Verification

`scripts/verify_standings_fix.py` runs the full live path
(`fetch_standings` → `_standings_to_snapshot` →
`format_standings_for_display`) and asserts displayed total ==
Yahoo's points_for for every team. On 2026-04-18:

```
Rank  Team                            DisplayedTotal  YahooPointsFor    CatSum
1     Spacemen                                 74.50            74.5     75.50
1     Hart of the Order                        74.50            74.5     74.50
3     Hello Peanuts!                           69.50            69.5     69.00
...
0 total mismatches
```

- `DisplayedTotal` — what the `/standings` page will render
- `YahooPointsFor` — Yahoo's authoritative total (ground truth)
- `CatSum` — score_roto per-category sum (differs by ±0.5 on teams
  involved in display ties; shown as `score_roto_total` in the
  `roto_points` dict for diagnostics)

## Tests

- `tests/test_lineup/test_yahoo_roster.py::TestParseStandings::test_extracts_points_for`
- `tests/test_lineup/test_yahoo_roster.py::TestParseStandings::test_points_for_absent_is_none`
- `tests/test_web/test_season_data.py::test_format_standings_prefers_yahoo_points_for`
- `tests/test_web/test_season_data.py::test_format_standings_falls_back_without_points_for`

Full suite: 1123 tests passing.

## Residual risk / known limitations

- **Per-category sum ≠ headline total** for teams involved in
  display-level ties. Worst-case skew is ±(number of ties) × 0.5. The
  category cells still show the correct relative standing (score_roto
  averaged ranks); only the sum-consistency is lost. Could add a
  tooltip on the Total cell in a follow-up ("Yahoo's authoritative
  total; may differ from the category sum by ±0.5 per display tie").
- **Cache-compat:** old cached `standings.json` from before this PR
  won't have `points_for`. The code treats missing `points_for` as
  "not available" and falls back to `score_roto`. First refresh after
  deploy re-populates the cache.
- **Projected standings unchanged.** The Preseason and Current
  Projected tabs use `score_roto` over full-precision ROS projections,
  where display-rounding ties don't arise. No behavior change there.
