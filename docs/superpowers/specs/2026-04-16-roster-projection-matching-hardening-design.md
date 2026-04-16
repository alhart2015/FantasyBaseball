# Hardening roster ↔ projection matching (in-season refresh)

**Status:** design accepted, ready for implementation plan
**Date:** 2026-04-16
**TODO entry:** "Harden data ingest: position and name matching"

## Problem

Three known classes of edge cases have caused matching bugs in the past:

1. **Accent encoding** — Julio Rodriguez (`Julio Rodríguez` vs `Julio Rodriguez`, NFC vs NFD).
2. **Cross-type same-name collision** — Mason Miller exists as both an MLB hitter and an MLB pitcher.
3. **Dual-eligibility split** — Shohei Ohtani comes through Yahoo as two roster entries (`Shohei Ohtani` for the bat, `Shohei Ohtani (Pitcher)` for the arm).

These bugs are largely latent today, but the matcher has zero observability: when a roster player fails to match, when a name resolves ambiguously, or when the position hint fails to disambiguate a cross-type collision, the matcher silently picks one or drops the player. The next time one of these edge cases breaks, the user only finds out by noticing a player missing from the dashboard.

## Scope

**In:** in-season refresh path. Specifically:

- `src/fantasy_baseball/data/projections.py::match_roster_to_projections`
- Its thin adapter `hydrate_roster_entries` in the same file

Callers pick up new behavior transparently:

- `src/fantasy_baseball/web/refresh_pipeline.py::_match_roster_to_projections`
- `src/fantasy_baseball/web/refresh_pipeline.py::_hydrate_rosters` (user + every opponent)
- `src/fantasy_baseball/web/season_data.py::load_projections_for_date`

**Out:** draft pipeline (`build_draft_board`, `apply_keepers`, `_attach_positions`, `fetch_missing_keepers`). The draft pipeline will be overhauled before next preseason, so hardening it now is wasted work.

## Non-goals

- No change to `normalize_name` (no Jr./Sr. stripping, no punctuation handling — speculative).
- No new structured "match result" type — keep `match_roster_to_projections` returning `list[Player]`.
- No UI banner for unmatched players — logs only.
- No change to keeper removal, position collision resolution, or any draft-side logic.

## Behavior changes

`match_roster_to_projections` keeps its current signature and return type. Add **one** optional kwarg used only for log clarity:

```python
def match_roster_to_projections(
    roster: list[dict],
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
    *,
    context: str = "",
) -> list[Player]:
```

`hydrate_roster_entries` accepts the same kwarg and forwards it.

Inside the matching loop, a module-level `logger = logging.getLogger(__name__)` emits a `WARNING` in three cases:

| Trigger | Condition | Log message |
|---|---|---|
| **unmatched** | `proj is None` after all three branches | `[<context>] no projection match for <name!r> (positions=<positions!r>)` |
| **ambiguous** | `len(matches) > 1` in either positional branch (hitter or pitcher) | `[<context>] ambiguous <hitter\|pitcher> match for <name!r> — <n> candidates, picked first` |
| **fallback** | matched only via the third "any" branch | `[<context>] <name!r> matched via fallback branch — positions=<positions!r> did not disambiguate` |

When `context == ""`, drop the leading `[<context>] ` to avoid empty brackets in logs.

Each warning fires per-player per-call. The matcher runs ~12 times per refresh (user + 11 opponents); a single chronic mismatch will produce ~12 lines per refresh. That's the right cost — the user will see it and act.

Caller updates (one line each) to pass a useful context string:

- `_hydrate_rosters` user roster → `context="user"`
- `_hydrate_rosters` opponent loop → `context=f"opp:{team.name}"`
- `_match_roster_to_projections` preseason call → `context="preseason"`
- `season_data.load_projections_for_date` two calls → `context="preseason"` and `context="ros"`

Total added code: ~15 lines in `projections.py` plus 5 one-line caller edits.

## Tests

New file: `tests/test_data/test_projections_matching_edge_cases.py`. No fixtures, no Yahoo/Redis mocking — each test builds tiny in-memory DataFrames with the columns the matcher reads (`name`, `_name_norm`, plus the minimal stat columns required to construct `HitterStats` / `PitcherStats`).

### Edge-case regression tests

**`TestJulioRodriguezAccentEncoding`** — projection row has the accented form; verify all three roster-name forms match the same row:

- NFC precomposed: `"Julio Rodríguez"`
- NFD decomposed: `"Julio Rodri\u0301guez"`
- ASCII plain: `"Julio Rodriguez"`

Plus the mirror case: roster has accents, projection is ASCII.

**`TestMasonMillerCrossTypeCollision`** — projections contain both a hitter and a pitcher row sharing `_name_norm == "mason miller"`:

- Roster `{"name": "Mason Miller", "positions": ["3B"]}` → returns hitter projection, no warning logged.
- Roster `{"name": "Mason Miller", "positions": ["SP"]}` → returns pitcher projection, no warning logged.
- Roster `{"name": "Mason Miller", "positions": []}` → returns the **hitter** projection (the fallback loop checks `hitters_proj` first), and emits a `fallback` WARNING.

**`TestShoheiOhtaniDualEntry`** — roster contains two entries:

- `{"name": "Shohei Ohtani", "positions": ["Util"], "player_id": "100"}`
- `{"name": "Shohei Ohtani (Pitcher)", "positions": ["SP"], "player_id": "200"}`

Projections contain a hitter row and a pitcher row, both with `_name_norm == "shohei ohtani"`. Assert two `Player` objects returned, distinct `player_type`, distinct `yahoo_id`, no warnings.

### Observability tests

**`TestMatchObservability`** — uses `caplog` to capture log records:

- Roster contains a player not in either projection DataFrame → exactly one WARNING with `"no projection match"`.
- Two hitter rows share a `_name_norm` and both match a hitter roster entry → exactly one WARNING with `"ambiguous"`, first row returned.
- Roster entry has positions that don't qualify as hitter or pitcher (e.g. `[]`) but name matches a hitter projection → exactly one WARNING with `"fallback"`, hitter projection returned.
- `context` kwarg appears in the warning text in `[brackets]`; when `context=""`, brackets are absent.

## Open questions

None. Decisions made above:

- Per-player WARNINGs (not per-call summary) — visibility is the point.
- No structured return type — preserves existing API.
- No `normalize_name` changes — speculative without evidence.

## Out-of-scope follow-ups

If a warning fires repeatedly in production, the right next step is a focused fix for that specific player or class. Possible candidates that came up during design but aren't worth doing speculatively:

- Internal whitespace collapse in `normalize_name`.
- Suffix stripping for `Jr.`, `Sr.`, `II`, `III`.
- Punctuation handling (periods in `J.D. Martinez`, apostrophes in `D'Arnaud`).
- A dashboard banner that surfaces unmatched-player counts to the user instead of just logs.

These wait for evidence.
