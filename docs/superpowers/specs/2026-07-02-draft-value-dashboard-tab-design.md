# Draft Value Dashboard Tab -- Design Spec

**Date:** 2026-07-02
**Status:** Approved (brainstorming), pending spec-review
**Branch:** `feature/draft-value-dashboard-tab`

## Problem

The draft-value metric (realized VAR vs draft-slot par expectation, per-player +
per-team, with a skill/luck split) shipped as a CLI + markdown report only
(`scripts/draft_value.py` -> `data/analysis/draft_value_report.md`). The
`run_draft_value()` library in `src/fantasy_baseball/analysis/draft_value.py` is
done and hardened (PR #153). The report is currently invisible from the season
dashboard -- you have to run a script and open a markdown file. Bake it into the
product as a **new tab on the existing `/transactions` page** so the draft grade
lives next to the in-season transaction (deltaRoto) analysis it complements.

## Goals

- Surface the draft-value team leaderboard + per-player breakdown in the web
  dashboard, on the `/transactions` page as a second tab.
- Keep the page a fast cache read (no multi-second recompute on page load):
  compute during the refresh pipeline and cache the result, exactly like the
  transaction analyzer.
- Do not change the existing deltaRoto transaction analysis or its refresh step.

## Non-Goals

- No change to the draft-value math (`run_draft_value()` is frozen; we only
  serialize its output).
- No new top-level nav item -- the tab lives inside the existing `/transactions`
  page.
- The CLI (`scripts/draft_value.py`) and markdown report stay as-is (dev tool).
- No YTD-vs-projected horizon toggle in v1; projected is the headline, YTD is a
  per-player column.

## Architecture

Four moving parts, mirroring the existing `TRANSACTION_ANALYZER` path end to end.

### 1. Cache key

Add a member to `src/fantasy_baseball/data/cache_keys.py`:

```python
DRAFT_VALUE = "draft_value"
```

### 2. Serialization builder (pure, testable)

New function in `src/fantasy_baseball/analysis/draft_value.py`:

```python
def build_draft_value_cache(
    players: list[PlayerValue], teams: list[TeamRollup]
) -> dict[str, Any]:
    ...
```

It transforms the `run_draft_value()` output into a JSON-safe, template-ready
dict. This keeps all ordering / null-handling logic in Python (unit-testable),
out of the Jinja template.

**Output shape:**

```json
{
  "horizon": "proj",
  "teams": [
    {
      "team": "My Team",
      "avg_value": 4.2,
      "sum_value": 58.1,
      "credited_count": 14,
      "players": [
        {
          "name": "Juan Soto",
          "player_type": "hitter",
          "kind": "keeper",
          "slot": null,
          "preseason_var": 38.1,
          "est_var_proj": 44.3,
          "value_proj": 6.2,
          "value_ytd": 3.1,
          "skill": 2.1,
          "luck": 4.1
        }
      ]
    }
  ]
}
```

**Rules:**

- Teams sorted by `avg_value` **descending**, with `NaN` avg sunk to the bottom
  (same treatment as the markdown report's leaderboard sort:
  `-inf` for `NaN`).
- Players within a team sorted by `value_proj` **descending**, with
  `None`/`NaN` `value_proj` sunk to the bottom (mirrors the report's per-player
  sort key).
- **`NaN` and `inf` are converted to JSON `null`** on every float field. An
  unmatched keeper produces `NaN` `value_proj` (keeper_par is `NaN`); raw `NaN`
  in a payload breaks strict JSON serialization and renders as the string "nan"
  in Jinja. A single `_finite(x)` helper (`return x if x is not None and
  math.isfinite(x) else None`) is applied to every float field.
- `player_type` is serialized as a plain string (it may be a `str`/`StrEnum`;
  coerce with `str(...)`).
- Every drafted pick + keeper is included (off-board fliers too -- they have
  `preseason_var: null`, `skill: null`, `luck: null` but a real `value_proj`).

### 3. Refresh pipeline step

New method `_compute_draft_value(self)` in
`src/fantasy_baseball/web/refresh_pipeline.py`, called in the run sequence
**immediately after `self._analyze_transactions()`** (line ~520) and before
`self._compute_streaks()` -- both feed the `/transactions` page.

Posture mirrors `_compute_streaks` / other non-load-bearing panels:

```python
# --- Step 15c: Draft-value grade for the /transactions Draft Grade tab ---
def _compute_draft_value(self) -> None:
    self._progress("Computing draft value grade...")
    try:
        from fantasy_baseball.analysis.draft_value import (
            build_draft_value_cache,
            run_draft_value,
        )
        players, teams = run_draft_value()
        payload = build_draft_value_cache(players, teams)
        write_cache(CacheKey.DRAFT_VALUE, payload, required=False)
        self._progress(f"Draft value cached: {len(teams)} teams")
    except Exception:
        log.exception("Draft-value computation failed; cache unchanged")
        self._progress("Draft value computation failed (continuing)")
```

Rationale:

- `run_draft_value()` can raise (`RuntimeError` on a reconstruction-gate
  failure, `ValueError` on missing keepers). It must **never abort the refresh**
  -- draft grade is a cosmetic panel. Catch broadly, log, continue.
- `write_cache(..., required=False)` so a transient KV write blip on this key
  doesn't fail the whole run.
- **Runs on Render too** (unlike streaks): every input `run_draft_value()` needs
  is either git-tracked (`data/projections/2026/*.csv`,
  `data/player_positions.json`, `data/draft_state*.json`,
  `config/draft_order.json`, `config/league.yaml`) or in Upstash
  (`FULL_SEASON_PROJECTIONS`, game-log totals). No duckdb/`[dev]` dependency. So
  there is **no `is_remote()` early-return** -- it computes wherever the refresh
  runs.

### 4. Route + template

**Route** (`season_routes.py`, `transactions()` at line ~1890): additionally read
the draft-value cache and pass it to the template.

```python
draft_cache = read_cache_dict(CacheKey.DRAFT_VALUE) or {}
return render_template(
    "season/transactions.html",
    ...,
    draft_data=draft_cache.get("teams", []),
)
```

`user_team` is already passed (used to highlight the user's row).

**Template** (`templates/season/transactions.html`): restructure into two tab
panels under a `.tab-strip`.

- **Tab strip** with two buttons: "Transactions" (active by default) and
  "Draft Grade". Follows the `trends.html` `.tab-strip` idiom (button with a
  `data-*` target, `.active` class, JS toggles panel `display`).
- **Panel 1 "Transactions":** the existing deltaRoto table, unchanged.
- **Panel 2 "Draft Grade":** expandable team leaderboard.
  - Outer table columns: `Team | avg | sum | picks`. Rows sorted as the cache
    provides (avg desc). User's team highlighted (`.user-team`) like the
    deltaRoto table.
  - Each team row is click-to-expand (reuse the page's existing
    `toggleTxnDetail` expand-row idiom -- a hidden sibling `<tr>` toggled
    `.open`).
  - Detail row: inner table of that team's players, columns
    `Player | kind | slot | preVAR | estVAR | value | skill | luck`. `value` is
    `value_proj`. `null` numeric fields render as an em-dash placeholder ("-");
    positive/negative values get `value-positive` / `value-negative` classes
    like the deltaRoto table.
  - Empty state: when `draft_data` is empty, show "No draft data. Click Refresh
    Data." (mirrors the existing transactions empty state).

**Shared CSS cleanup:** the `.tab-strip` rules currently live inline in
`trends.html`. Promote them to `season.css` (single definition) and drop the
inline copy from `trends.html`, so both pages share one source. This is a small,
in-scope dedup (we are introducing the page's second consumer of the pattern).

**JS:** a small `switchTab(button)` function toggles the two panels' `display`
and the buttons' `.active` state. The existing `toggleTxnDetail(row)` is reused
for the Draft Grade expand rows (it toggles `row.nextElementSibling`).

## Data flow

```
refresh pipeline
  _analyze_transactions()  -> cache:transaction_analyzer  (unchanged)
  _compute_draft_value()   -> run_draft_value() -> build_draft_value_cache()
                           -> cache:draft_value
GET /transactions
  read cache:transaction_analyzer -> txn_data   (Panel 1)
  read cache:draft_value          -> draft_data (Panel 2)
  render transactions.html with both
```

## Error handling

- Refresh step: broad try/except, log + progress message, never re-raise;
  `required=False` write. A failure leaves the previous `cache:draft_value`
  untouched (stale but valid) rather than blanking it.
- Route: `read_cache_dict(...) or {}` then `.get("teams", [])` -> empty list ->
  template empty state. A missing/corrupt cache degrades to the placeholder, not
  a 500.
- Serialization: `NaN`/`inf`/`None` -> `null`; the template treats `null` as the
  "-" placeholder.

## Testing

- **Unit -- `build_draft_value_cache`** (`tests/test_analysis/test_draft_value.py`):
  - Team sort by `avg_value` desc; `NaN`-avg team sinks to the bottom.
  - Player sort by `value_proj` desc within a team; `None`/`NaN` sinks.
  - `NaN`/`inf` on any float field -> `null` (assert JSON round-trips via
    `json.dumps` with `allow_nan=False` succeeding).
  - Off-board flier: `preseason_var`/`skill`/`luck` are `null`, `value_proj` is
    a real number.
  - `player_type` is a plain `str` in the output.
- **Refresh pipeline** (`tests/test_web/test_refresh_pipeline.py`): extend the
  existing fixture-driven test to assert `cache:draft_value` is written with the
  expected top-level shape (`{"teams": [...], "horizon": "proj"}`). Mirror how
  the test already asserts `TRANSACTION_ANALYZER`.
- **Route/render** (`tests/test_web/`): `GET /transactions` returns 200 and the
  response contains both the tab strip and a Draft Grade team row when the
  draft-value cache is populated; empty cache renders the placeholder.

## Verification gates (from CLAUDE.md)

- `pytest -v` (at minimum the touched dirs: `tests/test_analysis`,
  `tests/test_web`) -- all green.
- `ruff check .` and `ruff format --check .` -- clean.
- `vulture` -- no new dead code.
- `mypy` -- required for any touched file listed under `[tool.mypy].files`;
  `analysis/draft_value.py` and the web modules are likely covered -- check the
  list and run it on touched files.
- Exercise the new refresh step locally via `run_season_dashboard.py`
  (`--no-sync` to avoid clobbering local while verifying not-yet-deployed code)
  before opening the PR, and confirm the tab renders.

## Files touched

| File | Change |
|---|---|
| `src/fantasy_baseball/data/cache_keys.py` | + `DRAFT_VALUE` member |
| `src/fantasy_baseball/analysis/draft_value.py` | + `build_draft_value_cache()` |
| `src/fantasy_baseball/web/refresh_pipeline.py` | + `_compute_draft_value()` step + call |
| `src/fantasy_baseball/web/season_routes.py` | read + pass `draft_data` |
| `src/fantasy_baseball/web/templates/season/transactions.html` | tab strip + Draft Grade panel + JS |
| `src/fantasy_baseball/web/static/season.css` | promote `.tab-strip` rules |
| `src/fantasy_baseball/web/templates/season/trends.html` | drop inline `.tab-strip` CSS |
| `tests/test_analysis/test_draft_value.py` | + `build_draft_value_cache` tests |
| `tests/test_web/test_refresh_pipeline.py` | assert `DRAFT_VALUE` written |
| `tests/test_web/...` | route renders both tabs |
```