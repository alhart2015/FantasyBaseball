# Draft Value Dashboard Tab -- Design Spec

**Date:** 2026-07-02
**Status:** Spec-review converged (4 iterations; clean above Low)
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
dict. This keeps all ordering / null-handling / grouping logic in Python
(unit-testable), out of the Jinja template.

**The join.** `run_draft_value()` returns `players` and `teams` as two *separate
flat lists* (`TeamRollup` carries no players). `build_draft_value_cache` **groups
`players` by `.team`** and nests each team's players under its `TeamRollup`. A
player is attached to the team whose `TeamRollup.team` equals `PlayerValue.team`.

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
          "display_name": "Juan Soto",
          "player_type": "hitter",
          "kind": "keeper",
          "slot": null,
          "preseason_var": 38.1,
          "est_var_proj": 44.3,
          "value_proj": 12.3,
          "value_ytd": 3.1,
          "skill": 6.1,
          "luck": 6.2
        }
      ]
    }
  ]
}
```

(Example numbers are self-consistent: `luck = est_var_proj - preseason_var`
= 44.3 - 38.1 = 6.2; `value_proj = skill + luck` = 6.1 + 6.2 = 12.3.)

**Field mapping.** Each output player dict is a projection of `PlayerValue`:
`name`, `display_name` (= `name`, plus a ` (H)`/` (P)` suffix for two-way rows;
see the `player_type` rule below), `player_type` (via `str(...)`), **`kind` =
`baseline_kind` verbatim** (`"keeper"`/`"drafted"`, renamed for brevity), `slot`,
`preseason_var`, `est_var_proj`, `value_proj`, `value_ytd`, `skill`, `luck`.
`est_var_ytd` and `est_var_proj` both exist on `PlayerValue`; only `est_var_proj`
is serialized (`est_var_ytd` is never displayed).

**Rules:**

- Build the team list from `teams` (the authoritative `TeamRollup` list); group
  `players` by `.team` and nest each team's players under its rollup. A team with
  no players emits `players: []`; a `PlayerValue` whose `.team` matches no
  `TeamRollup` is dropped (cannot occur from `run_draft_value`, which builds both
  from the same grouping -- this is a defensive contract for the pure function).
- Teams sorted by `avg_value` **descending**, with `NaN` avg sunk to the bottom
  (same treatment as the markdown report's leaderboard sort: `-inf` for `NaN`).
  Use a **stable** sort so equal-`avg_value` teams keep `run_draft_value`'s input
  order (alphabetical); a unit test may assert this.
- Players **within each team** sorted by `value_proj` **descending**, with
  `None`/`NaN` `value_proj` sunk to the bottom. This reuses the report's
  per-player sort *key* (`-inf` for `None`/`NaN`, descending) but applies it
  per-team, not over one global list as the CLI does.
- **`NaN` and `inf` are converted to JSON `null`** on every float field. The KV
  layer's `json.dumps` defaults to `allow_nan=True`, so raw `NaN` would
  *round-trip* through the cache -- but it then reaches Jinja as a float that
  renders as the literal string `"nan"`. The conversion exists to keep the
  template clean (every non-finite value becomes the `null` -> `—` placeholder),
  not because the write would fail. A single `_finite(x)` helper (`return x if x
  is not None and math.isfinite(x) else None`) is applied to every float field,
  so the payload also survives a strict `json.dumps(..., allow_nan=False)`
  round-trip (asserted in the unit tests). The helper defensively handles `None`
  even though (see next bullet) `run_draft_value` does not emit `None` floats.
- **What non-finite values actually occur in `run_draft_value` output** (so tests
  and the "picks" column are documented correctly): `run_draft_value` passes
  `missing_line_est=0.0` for *every* pick (`draft_value.py:924`), so a player who
  never played is scored at replacement and gets a **finite** `value_proj`
  (`0.0 - par`). An **off-board flier** therefore has a **finite** `value_proj`
  but `preseason_var`/`skill`/`luck` = `None` (it was never on the board). The
  **only** ungradeable `value_proj` is a keeper whose par is `NaN` -- and
  `keeper_par` is `NaN` **only when the *entire* keeper cohort fails to match the
  board** (`keeper_vars` empty -> `float("nan")`, `draft_value.py:406`; the code
  comment at `:717` says "no keeper matched the board"). In that all-or-nothing
  case *every* keeper's `value_proj` is `NaN`. A **single** unmatched keeper (with
  any other keeper matched) does NOT cause `NaN`: `keeper_par` is the finite mean
  of the matched keepers, so that lone keeper is scored against it and **is
  credited** with a finite `value_proj`. `value_proj` is never literally `None` in
  this pipeline. In the real 2026 data all keepers match, so in practice
  `value_proj` is finite for everyone. The `_finite`/None-sink handling is
  defensive for the pure function, exercised in unit tests with synthetic
  `NaN`/`None` rows -- not a routinely-hit path.
- `player_type` is serialized (as a plain string) **and drives a display marker
  the builder computes in Python** (not the template). A two-way player (e.g.
  Shohei Ohtani) appears as two rows under one name -- hitter and pitcher, and
  those two rows may even land on **different teams** (`_assign_pick_types`
  credits the keeper-hitter to the keeper's team and the drafted-pitcher pick to
  its own team). So `build_draft_value_cache` detects, **within each team's
  player list**, any `name` that appears under more than one `player_type` and
  emits a `display_name` field with a ` (H)`/` (P)` suffix for exactly those
  rows; all other rows get `display_name == name`. The template renders
  `display_name` verbatim -- no duplicate-scan in Jinja (consistent with the
  "logic in Python" rule above). Per-team scope means a solo row never gets a
  spurious suffix even if the same name appears (as the other type) on a
  different team. Framing note: the suffix is a **same-table collision guard**
  (two rows for one name in one team's detail table), not a general two-way
  indicator -- in the real 2026 data Ohtani's kept bat and drafted arm land on
  different teams, so the suffix typically never renders in production and is
  exercised mainly by the unit test. `player_type` itself stays in the payload
  for tests/debugging.
- **`credited_count` is the graded-pick count, not the row count.** It comes
  straight from `TeamRollup.credited_count`, which counts only players with a
  *finite* `value_proj` (`roll_up_team` drops `None`/`NaN`). The nested
  `players[]` list may additionally include ungradeable rows, so
  **`credited_count` may be less than `len(players)`**. In this pipeline that
  gap arises only in the degenerate all-keepers-unmatched case (every keeper row
  `NaN`; see the previous bullet) -- in the real 2026 data all keepers match and
  the two are equal. The pure function must still handle the inequality
  (unit-tested with a synthetic `NaN`-`value_proj` row). The template labels the
  outer column "picks" = graded picks.

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
  `config/draft_order.json`, `config/league.yaml`) or in the KV store. No
  duckdb/`[dev]` dependency. So there is **no `is_remote()` early-return** -- it
  computes wherever the refresh runs.

**Input freshness / provenance (important -- the step order matters only for game
logs):**

- **Game-log totals** (`get_game_log_totals`, the YTD side): written earlier in
  the *same* refresh by `_fetch_game_logs` (step ~4, well before the insertion
  point). So the to-date side of the grade is fresh on every refresh -- this is
  the one input for which placing the step after `_analyze_transactions` matters.
- **`FULL_SEASON_PROJECTIONS`** (the projected side): **NOT written by this
  pipeline.** The main refresh only ever *reads* that key
  (`refresh_pipeline.py:699`) and otherwise re-derives full-season projections
  in-memory; the cached blob is written by the separate `_run_rest_of_season_fetch`
  cron job (`ros_pipeline.py:273`). So `_compute_draft_value` consumes whatever
  vintage that job last cached -- exactly the same blob the CLI report reads.
  Placing the step earlier or later in *this* pipeline does not change its
  freshness. Acceptable: the projected side is a preseason-anchored estimate and
  the CLI has always read this same key.
- **Cold-KV degraded state:** if the ROS-fetch job has never run,
  `load_full_season_lines()` returns `({}, {})`; every player then falls to
  `missing_line_est=0.0` and the tab shows an all-negative `-par` grade. This is
  a degraded-but-valid render (identical to the CLI in that state), not a crash.
  On Render the ROS job runs regularly, so this is a fresh-environment edge only.

**Performance note (accepted):** `reproduce_draft_day_board()` re-blends the
projection systems and rebuilds the full board on every refresh, even though the
draft-day board is season-invariant (only the YTD/projected *estimates* drift).
This is the same recompute-every-refresh posture the transaction analyzer already
uses; the cost is a few seconds inside an already-heavy refresh. Not optimized in
v1 (YAGNI); a future pass could memoize the board/par curves if refresh latency
becomes a concern.

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

**CRITICAL restructure constraint -- hoist shared scaffolding out of the
data-emptiness conditional.** Today the entire deltaRoto table *and* the
`<script>` defining `toggleTxnDetail` live inside the page-level
`{% if not txn_data %}...{% else %}...{% endif %}` (only rendered when
`txn_data` is non-empty). If we merely wrap that, the common
**post-draft / pre-first-transaction state** (`txn_data` empty, `draft_data`
populated -- exactly when the draft grade is most wanted) would hide the tab
strip and leave both JS functions undefined, so the Draft Grade tab wouldn't
render or its expand rows would throw. The restructure MUST:

- Hoist the **tab strip** and **both JS functions** (`toggleTxnDetail` and the
  new `switchTab`) to the top level of `{% block content %}`, outside every
  `{% if %}` on `txn_data`/`draft_data`.
- Give **each panel its own independent empty state**: Panel 1 shows its "No
  transaction data..." placeholder when `txn_data` is empty; Panel 2 shows a
  "No draft data. Click Refresh Data." placeholder when `draft_data` is empty.
  The two conditionals are independent -- neither gates the other.
- **Replace the static `<h2>Transaction Analyzer</h2>`** (`transactions.html:13`)
  with the tab strip -- the tabs now label each panel, so a fixed "Transaction
  Analyzer" heading would mislabel the Draft Grade tab. (Keep the surrounding
  `.page-transactions` / `.page-header` wrappers.)

Layout:

- **Tab strip** with two buttons: "Transactions" (active by default) and
  "Draft Grade". It reuses only the **`.tab-strip` CSS look** from
  `trends.html`; it does NOT reuse `season_trends.js` (that drives Chart.js
  dataset swaps on a single canvas, not panel show/hide) -- and that script must
  NOT be added to this page. The panel toggle is the new `switchTab` function
  below.
- **Panel 1 "Transactions":** the existing deltaRoto table, moved verbatim into
  its panel `<div>` (its own `{% if not txn_data %}` empty state inside).
- **Panel 2 "Draft Grade":** expandable team leaderboard.
  - Outer table columns: `Team | avg | sum | picks` (`picks` = `credited_count`,
    the graded-pick count -- may be fewer than the expanded detail rows; see §2).
    Rows sorted as the cache provides (avg desc). User's team highlighted
    (`.user-team`) like the deltaRoto table.
  - Each team **summary** row uses `onclick="toggleTxnDetail(this)"`; its
    **detail** sibling `<tr>` **must carry `class="txn-detail"`** (and the inner
    table `class="txn-inner-table"`). `toggleTxnDetail` gates on
    `nextElementSibling.classList.contains('txn-detail')` (`transactions.html:100`)
    and the `.txn-detail { display:none }` / `.txn-detail.open { display:table-row }`
    rules drive show/hide -- a differently-named class would make the click a
    silent no-op and leave the row always visible. Reuse the existing classes;
    do not invent `draft-detail`.
  - The detail `<tr>` must use `colspan="4"` to match the 4-column outer table
    (the existing deltaRoto detail row uses `colspan="5"` for its 5-column table
    -- do not copy that number).
  - Detail row: inner table of that team's players, columns
    `Player | kind | slot | preVAR | estVAR | value | valueYTD | skill | luck`.
    - **`Player`** renders the builder-provided `display_name` (which already
      carries the ` (H)`/` (P)` suffix for two-way rows within a team; see §2).
      The template does not compute the suffix itself.
    - **`slot`** is `PlayerValue.slot`, which is the **draft-order ordinal among
      on-board drafted picks** (1-based, skipping keepers and off-board fliers) --
      NOT the overall draft-pick number. This matches the CLI report's "slot"
      column (`scripts/draft_value.py:61`); keepers and fliers show `—`. Keep the
      header "slot" for CLI parity, but do not reinterpret it as pick #1..200.
    - **`value`** = `value_proj`; **`valueYTD`** = `value_ytd`.
    - `null` numeric fields render as the placeholder character `—` (U+2014,
      matching the existing table at `transactions.html:46`; this is an HTML
      template, so U+2014 is fine -- the CLAUDE.md ASCII rule targets
      `print()`/cp1252 stdout, not UTF-8 templates, which already use `—`/`Δ`).
      Positive/negative values get `value-positive` / `value-negative` classes
      like the deltaRoto table.
  - Empty state: when `draft_data` is empty, show "No draft data. Click Refresh
    Data." (mirrors the existing transactions empty state).

**Shared CSS cleanup:** the `.tab-strip` rules currently live inline in
`trends.html`. Promote them to `season.css` (single, **byte-faithful** copy --
keep the `var(--amber)`/`var(--bg)` active-tab colors exactly; do NOT rescope
them under `.page-transactions`, or trends regresses) and drop the inline copy
from `trends.html`. `season.css` is loaded globally via `base.html`, and
`season_trends.js` keys on `data-*` attributes and `.active` (both preserved),
so moving only the CSS is safe. Small, in-scope dedup (this page is the pattern's
second consumer).

**JS:** a small `switchTab(button)` function toggles the two panels' `display`
and the buttons' `.active` state. The existing `toggleTxnDetail(row)` is reused
for the Draft Grade expand rows (it toggles `row.nextElementSibling`). Both must
live at top level (see the hoist constraint above), not inside a `txn_data`
conditional.

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
  `—` (U+2014) placeholder.
- **Monitoring gap (accepted, matches the streaks/SPoE posture):** the broad
  `except` + `required=False` means a *persistently* broken draft-value
  computation is invisible outside logs -- the tab just shows the empty state or
  a stale cache, and a caught exception writes nothing. This is the deliberate
  non-load-bearing-panel trade-off. Because of it, the refresh test must NOT rely
  on the real computation to prove the wiring (a swallowed raise is
  indistinguishable from "computed empty"); it patches `run_draft_value` to a
  known return so a missing/malformed cache is a genuine wiring failure (see
  Testing).

## Testing

- **Unit -- `build_draft_value_cache`** (`tests/test_analysis/test_draft_value.py`):
  Construct `PlayerValue`/`TeamRollup` instances directly (no KV, no board) and
  assert:
  - Players are grouped under the correct team (join by `.team`).
  - Team sort by `avg_value` desc; `NaN`-avg team sinks to the bottom.
  - Player sort by `value_proj` desc within a team; `None`/`NaN` sinks.
  - `NaN`/`inf` (and defensively `None`) on any float field -> `null` (assert the
    payload survives `json.dumps(payload, allow_nan=False)` -- no non-finite
    floats leak).
  - Off-board flier (`preseason_var=None`): `preseason_var`/`skill`/`luck` are
    `null`, `value_proj` is a real (finite) number, and the row is still present
    in `players[]`.
  - `est_var_ytd` is absent from each player dict; `value_ytd` is present;
    `kind` == the source `baseline_kind`.
  - Two-way disambiguation: a team with the same `name` under both `hitter` and
    `pitcher` gets `display_name` with ` (H)`/` (P)` suffixes on those two rows,
    while an identically-named solo row on a *different* team keeps
    `display_name == name` (per-team scope, no spurious suffix).
  - `credited_count` passes through from `TeamRollup` unchanged and can be less
    than `len(players)` -- construct a team whose `TeamRollup.credited_count` is
    below its player count and include a synthetic **`value_proj=NaN`** row (the
    unmatched-keeper case; NOT an off-board flier, which is finite). Assert that
    NaN row serializes to `value_proj: null` and still appears in `players[]`.
  - `player_type` is a plain `str` in the output.
  - **Typing (mypy):** `analysis/` is under `[tool.mypy].files`. Annotate
    `_finite` as `float | None -> float | None`; write the sort keys so
    `math.isnan`/`math.isfinite` never receive `None` (short-circuit
    `v is None or ...`). `build_draft_value_cache -> dict[str, Any]` returns a
    concrete dict (not `Any`), so `warn_return_any` is satisfied.
- **Refresh pipeline** (`tests/test_web/`): a **dedicated** test (new file or an
  added test alongside `test_refresh_pipeline.py`), NOT a bare presence check in
  the shared `test_all_expected_cache_files_written`. Rationale (spec-review
  finding): `run_draft_value()` reads its inputs by **absolute repo-root path**
  (`config/league.yaml`, `data/draft_state.json`, `data/projections/2026/*.csv`,
  ...), which `configured_test_env` cannot isolate, and its
  `from fantasy_baseball.config import load_config` binding is **not** reached by
  the fixture's `patch("fantasy_baseball.config.load_config", ...)`. Left
  unpatched it would run the full real-data computation (slow) and return the
  *real* league's team names (not `Team 01..12`), and any on-disk draft-file
  drift would raise -> get swallowed -> silently write nothing. The fixture
  already stubs `_compute_streaks` for the same class of reason.
  - So: **patch `fantasy_baseball.analysis.draft_value.run_draft_value`** (the
    name imported inside `_compute_draft_value`) to return canned
    `PlayerValue`/`TeamRollup` lists, run `_compute_draft_value` (or the whole
    refresh with this patch added to the fixture), and assert `cache:draft_value`
    is written with shape `{"horizon": "proj", "teams": [...]}` and that the
    canned team/player round-trips (grouping + serialization wired correctly).
  - Add the same `run_draft_value` patch to `patched_refresh_environment` so the
    existing full-refresh test does not invoke the heavy real computation; only
    then may `DRAFT_VALUE` be added to the `test_all_expected_cache_files_written`
    key list.
- **Route/render** (`tests/test_web/`): `GET /transactions` returns 200 and the
  response contains the tab strip and a Draft Grade team row when the
  draft-value cache is populated; with an empty draft cache the Draft Grade panel
  renders its "No draft data" placeholder; **and with an empty `txn_data` but a
  populated `draft_data`** (the post-draft/pre-transaction state) the tab strip
  and Draft Grade rows still render and the transactions panel shows its own
  placeholder -- guarding the hoist-out-of-conditional restructure.

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
| `tests/test_analysis/test_draft_value.py` | + `build_draft_value_cache` unit tests |
| `tests/test_web/_refresh_fixture.py` | patch `run_draft_value` -> canned dataclasses |
| `tests/test_web/test_refresh_pipeline.py` | dedicated `_compute_draft_value` cache-write + shape test; add `DRAFT_VALUE` to expected keys |
| `tests/test_web/...` | route renders both tabs incl. empty-`txn_data` state |
```