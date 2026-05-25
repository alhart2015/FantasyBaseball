# Category Bars Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Category Bars" tab to the season dashboard's `/standings` page that plots all 10 teams for a selected roto category as a horizontal dot-plot with +/-1 SD error bars, sorted best-on-top, toggleable between preseason and current projections.

**Architecture:** Pure presentation over already-cached data. A new backend formatter reshapes the two existing standings display dicts (`preseason_data`, `current_projected_data`) -- each team already carries `stats` (central value) and `sds` (+/-1 SD) -- into a chart-ready JSON structure. The `/standings` route embeds that JSON in the page (no new API endpoint). A new client-side Chart.js module renders the dot-plot and re-renders on category/projection change.

**Tech Stack:** Python 3 / Flask / Jinja2 (backend + template), Chart.js 4.4.4 + the `chartjs-chart-error-bars` plugin (frontend), pytest (tests).

---

## Background facts (verified against the codebase)

- **Display-dict shape** (output of `format_standings_for_display`, the input to the new formatter): `{"teams": [ {"name": str, "team_key": str, "is_user": bool, "stats": CategoryStats, "sds": {Category: float}, "roto_points": {...}, "roto_total": float, ...}, ... ]}`.
- `CategoryStats.__getitem__(cat: Category) -> float` (must pass a `Category` enum, not a string).
- `sds` is a plain `dict[Category, float]`; a category may be absent (e.g. no `team_sds` cached) -> default to `0.0` via `.get(cat, 0.0)`.
- Constants in `fantasy_baseball.utils.constants`: `ALL_CATEGORIES` (ordered list of 10 `Category`), `INVERSE_STATS = {Category.ERA, Category.WHIP}`. `season_data.py` already imports `INVERSE_STATS` under the alias `INVERSE_CATS` and already imports `ALL_CATEGORIES` and `Category`. **No new imports are needed in `season_data.py`.**
- `Category.value` is the uppercase short name (`"R"`, `"HR"`, ... `"WHIP"`) -- the JSON-safe key, matching the trends payload convention.
- Best-on-top sort: counting/AVG categories descending (highest best); `ERA`/`WHIP` ascending (lowest best). So `reverse = cat not in INVERSE_CATS`.
- The `/standings` route (`season_routes.py`) builds `preseason_data` and `current_projected_data` (both `None` when projections are absent) and passes them to `render_template("season/standings.html", ...)` around line 421-434.
- `standings.html` top tabs are pills with `data-view` + `onclick="toggleTopView(this)"`; each view is a `<div id="view-...">`; `toggleTopView` (around line 316) shows/hides them. The existing `breakdown-data` embedded-JSON pattern is `<script type="application/json" id="breakdown-data">{{ ... | tojson }}</script>`.
- Trends loads Chart.js via `<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>` and its module via `<script src="{{ url_for('static', filename='season_trends.js') }}"></script>`. Standings does NOT currently load Chart.js.
- ASCII-only rule (CLAUDE.md): in `.py` and `.js` source, emit `+/-` as the JS escape `'±'` (as the existing `formatStatCellsWithBounds` does); never a raw non-ASCII byte.

---

## File structure

- **Modify** `src/fantasy_baseball/web/season_data.py` -- add `format_category_bars_for_display` + private `_category_bars_one_flavor`. (Task 1)
- **Modify** `src/fantasy_baseball/web/season_routes.py` -- import the formatter, call it, pass `category_bars` to the template. (Task 2)
- **Modify** `src/fantasy_baseball/web/templates/season/standings.html` -- new pill, new view div (controls + canvas + embedded JSON), `toggleTopView` update, script includes. (Task 3)
- **Create** `src/fantasy_baseball/web/static/season_category_bars.js` -- the Chart.js render module. (Task 4)
- **Test** `tests/test_web/test_season_data.py` (Task 1), `tests/test_web/test_season_routes.py` (Tasks 2-3).

---

## Task 1: Backend formatter `format_category_bars_for_display`

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py` (add two functions; no new imports)
- Test: `tests/test_web/test_season_data.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_web/test_season_data.py`. (`CategoryStats`, `Category`, and `format_standings_for_display` are already imported at the top of that file.) Two import edits first: change the constants import line `from fantasy_baseball.utils.constants import Category` to `from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category`, and add `format_category_bars_for_display` to the existing `from fantasy_baseball.web.season_data import (...)` block (see Step 3).

```python
def _bars_display_dict():
    """Two-team display dict in the shape format_standings_for_display emits.

    Hart leads R (320 > 300) but trails ERA (3.20 vs 3.10 -> rival lower=better).
    """
    return {
        "teams": [
            {
                "name": "Hart of the Order",
                "team_key": "key_0",
                "is_user": True,
                "stats": CategoryStats(r=320, era=3.20),
                "sds": {Category.R: 25.0, Category.ERA: 0.18},
            },
            {
                "name": "SkeleThor",
                "team_key": "key_1",
                "is_user": False,
                "stats": CategoryStats(r=300, era=3.10),
                "sds": {Category.R: 30.0, Category.ERA: 0.15},
            },
        ]
    }


def test_category_bars_has_both_flavors_and_all_categories():
    data = _bars_display_dict()
    out = format_category_bars_for_display(data, data)
    assert set(out.keys()) == {"preseason", "current"}
    for flavor in ("preseason", "current"):
        assert set(out[flavor].keys()) == {c.value for c in ALL_CATEGORIES}


def test_category_bars_normal_category_sorts_best_on_top():
    out = format_category_bars_for_display(_bars_display_dict(), _bars_display_dict())
    runs = out["current"]["R"]
    # Higher runs is better -> Hart (320) on top.
    assert [r["team"] for r in runs] == ["Hart of the Order", "SkeleThor"]
    assert runs[0]["value"] == 320
    assert runs[0]["sd"] == 25.0
    assert runs[0]["is_user"] is True


def test_category_bars_inverse_category_sorts_lowest_on_top():
    out = format_category_bars_for_display(_bars_display_dict(), _bars_display_dict())
    era = out["current"]["ERA"]
    # Lower ERA is better -> SkeleThor (3.10) on top.
    assert [r["team"] for r in era] == ["SkeleThor", "Hart of the Order"]
    assert era[0]["value"] == 3.10


def test_category_bars_missing_sd_defaults_to_zero():
    data = {
        "teams": [
            {
                "name": "No SD Team",
                "team_key": "k",
                "is_user": False,
                "stats": CategoryStats(hr=40),
                "sds": {},  # no team_sds cached
            }
        ]
    }
    out = format_category_bars_for_display(data, data)
    assert out["current"]["HR"][0]["sd"] == 0.0


def test_category_bars_handles_missing_flavor():
    """Pre-refresh: a flavor's display dict may be None."""
    out = format_category_bars_for_display(None, _bars_display_dict())
    assert out["preseason"] == {}
    assert out["current"]["R"][0]["team"] == "Hart of the Order"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web/test_season_data.py -k category_bars -v`
Expected: FAIL with `ImportError: cannot import name 'format_category_bars_for_display'` (collection error).

- [ ] **Step 3: Implement the formatter**

Add to `src/fantasy_baseball/web/season_data.py` (place it immediately after `format_standings_for_display`, before `get_teams_list`). Uses the already-imported `ALL_CATEGORIES`, `Category`, and `INVERSE_CATS`.

```python
def _category_bars_one_flavor(data: dict | None) -> dict[str, list[dict]]:
    """Reshape one standings display dict into per-category ranked rows.

    Each category maps to a list of ``{team, value, sd, is_user}`` sorted
    best-on-top: counting/AVG categories descending, ERA/WHIP ascending
    (lower is better). ``sd`` defaults to 0.0 when the category is absent
    from a team's ``sds`` (e.g. no ``team_sds`` were cached), which renders
    as a bare dot with no whiskers.
    """
    if not data or not data.get("teams"):
        return {}
    out: dict[str, list[dict]] = {}
    for cat in ALL_CATEGORIES:
        rows = [
            {
                "team": team["name"],
                "value": team["stats"][cat],
                "sd": (team.get("sds") or {}).get(cat, 0.0),
                "is_user": team["is_user"],
            }
            for team in data["teams"]
        ]
        # reverse=True for "higher is better"; ERA/WHIP (INVERSE_CATS) sort
        # ascending so the lowest (best) team lands on top. Python's sort is
        # stable, so ties keep the input order.
        rows.sort(key=lambda r: r["value"], reverse=cat not in INVERSE_CATS)
        out[cat.value] = rows
    return out


def format_category_bars_for_display(
    preseason_data: dict | None,
    current_projected_data: dict | None,
) -> dict[str, dict[str, list[dict]]]:
    """Build the Category Bars chart payload from the two standings display dicts.

    Returns ``{"preseason": {CAT: [rows...]}, "current": {CAT: [rows...]}}``
    where each row is ``{team, value, sd, is_user}`` sorted best-on-top.
    Categories use uppercase short names (R, HR, ... WHIP). A missing flavor
    (``None``, pre-refresh) yields an empty ``{}`` for that flavor.
    """
    return {
        "preseason": _category_bars_one_flavor(preseason_data),
        "current": _category_bars_one_flavor(current_projected_data),
    }
```

Then add `format_category_bars_for_display` to the test file's import block (it is defined in `season_data`):

```python
from fantasy_baseball.web.season_data import (
    CacheKey,
    format_category_bars_for_display,
    format_lineup_for_display,
    format_monte_carlo_for_display,
    format_standings_for_display,
    read_cache,
    read_meta,
    write_cache,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web/test_season_data.py -k category_bars -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py
git commit -m "feat(standings): add format_category_bars_for_display"
```

---

## Task 2: Wire the formatter into the `/standings` route

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py` (import + call + template kwarg)
- Test: `tests/test_web/test_season_routes.py`

- [ ] **Step 1: Write the failing route test**

Append to `tests/test_web/test_season_routes.py`. This seeds `CacheKey.PROJECTIONS` with both flavors + both SD sets (the shape `refresh_pipeline` writes) and asserts the embedded JSON is present and non-empty. Check the top of the file for existing imports of `season_data`, `CacheKey`, `create_app`; reuse the `client` fixture. `kv_isolation` (per-test SQLite KV) is defined in this file -- include it as a parameter so `write_cache` does not touch shared state.

```python
def test_standings_embeds_category_bars_data(client, kv_isolation):
    """When projections are cached, /standings embeds non-empty category-bars JSON."""
    proj = {
        "preseason_standings": {
            "effective_date": "2026-04-01",
            "entries": [
                {"team_name": "Hart of the Order",
                 "stats": {"R": 320, "HR": 90, "RBI": 290, "SB": 50, "AVG": 0.270,
                           "W": 35, "K": 600, "SV": 25, "ERA": 3.50, "WHIP": 1.18}},
                {"team_name": "SkeleThor",
                 "stats": {"R": 300, "HR": 85, "RBI": 295, "SB": 40, "AVG": 0.265,
                           "W": 38, "K": 580, "SV": 30, "ERA": 3.40, "WHIP": 1.15}},
            ],
        },
        "preseason_team_sds": {
            "Hart of the Order": {"R": 25.0, "ERA": 0.18},
            "SkeleThor": {"R": 30.0, "ERA": 0.15},
        },
        "fraction_remaining": 0.8,
    }
    proj["projected_standings"] = proj["preseason_standings"]
    proj["team_sds"] = proj["preseason_team_sds"]

    standings = {
        "effective_date": "2026-04-01",
        "entries": [
            {"team_name": "Hart of the Order", "team_key": "k0", "rank": 1,
             "stats": {"R": 320, "HR": 90, "RBI": 290, "SB": 50, "AVG": 0.270,
                       "W": 35, "K": 600, "SV": 25, "ERA": 3.50, "WHIP": 1.18}},
            {"team_name": "SkeleThor", "team_key": "k1", "rank": 2,
             "stats": {"R": 300, "HR": 85, "RBI": 295, "SB": 40, "AVG": 0.265,
                       "W": 38, "K": 580, "SV": 30, "ERA": 3.40, "WHIP": 1.15}},
        ],
    }
    season_data.write_cache(CacheKey.STANDINGS, standings)
    season_data.write_cache(CacheKey.PROJECTIONS, proj)
    season_data.write_cache(CacheKey.META, {"last_refresh": "8:32 AM", "week": "3"})

    with patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cfg.return_value = type(
            "Cfg", (), {"team_name": "Hart of the Order"}
        )()
        resp = client.get("/standings")

    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'id="category-bars-data"' in body
    # The embedded JSON must carry both flavors and real team rows.
    assert '"preseason"' in body
    assert "Hart of the Order" in body
```

If the existing tests construct `_load_config`'s return differently (e.g. a real config object or a different patch target), mirror that pattern instead -- read `test_standings_renders_table_with_data` (around line 156) and copy its config-mocking approach so this test stays consistent.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py::test_standings_embeds_category_bars_data -v`
Expected: FAIL -- `'id="category-bars-data"' in body` is False (the template does not yet embed it). Note: this test depends on Task 3's template change too; it stays red until Task 3 lands. That is intentional -- it is the cross-cutting guardrail for the route+template wiring. Implement the route change now (Step 3); the test goes green after Task 3.

- [ ] **Step 3: Implement the route change**

In `src/fantasy_baseball/web/season_routes.py`:

1. Add `format_category_bars_for_display` to the existing `from fantasy_baseball.web.season_data import (...)` block (alphabetically near `format_standings_for_display`).

2. In the `standings()` handler, immediately before the `return render_template("season/standings.html", ...)` call, add:

```python
        category_bars = format_category_bars_for_display(
            preseason_data, current_projected_data
        )
```

3. Add the kwarg to the `render_template` call (after `rest_of_season_mc=rest_of_season_mc_data,`):

```python
            category_bars=category_bars,
```

`preseason_data` and `current_projected_data` are already in scope here (they are initialized to `None` near the top of the handler and set inside the `if raw_projected:` block), so `format_category_bars_for_display` receives `None` gracefully pre-refresh.

- [ ] **Step 4: Run test (expect still red until Task 3)**

Run: `pytest tests/test_web/test_season_routes.py::test_standings_embeds_category_bars_data -v`
Expected: still FAIL on the `id="category-bars-data"` assertion (template not updated yet). The route now passes `category_bars`, but nothing renders it. Proceed to Task 3.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py tests/test_web/test_season_routes.py
git commit -m "feat(standings): pass category_bars payload to standings template"
```

---

## Task 3: Template -- new tab, controls, canvas, embedded JSON, script includes

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/standings.html`
- Test: `tests/test_web/test_season_routes.py` (the Task 2 test goes green here)

- [ ] **Step 1: Add the top-level pill**

In the `#standings-top-toggle` group (currently three buttons: Current / Projected / Monte Carlo), add a fourth button after the Monte Carlo button:

```html
    <button class="pill" data-view="categorybars" onclick="toggleTopView(this)">Category Bars</button>
```

- [ ] **Step 2: Add the view container with controls, canvas, and embedded JSON**

Insert this block immediately AFTER the closing `</div>` of `#view-montecarlo` and BEFORE the `<dialog id="breakdown-modal">` element. The buttons mirror the existing `pill` / `pill-group` pattern.

```html
{# -- Category Bars (ranked dot-plot with +/-1 SD error bars) -- #}
<div id="view-categorybars" style="display: none;">
    <div class="pill-group" id="catbars-proj-toggle">
        <button class="pill active" data-cbproj="current" onclick="catBarsSetProjection(this)">Current</button>
        <button class="pill" data-cbproj="preseason" onclick="catBarsSetProjection(this)">Preseason</button>
    </div>

    <div class="pill-group" id="catbars-cat-toggle">
        {% for cat in all_categories %}
        <button class="pill {% if loop.first %}active{% endif %}"
                data-cbcat="{{ cat.value }}" onclick="catBarsSetCategory(this)">{{ cat.value }}</button>
        {% endfor %}
    </div>

    <p id="catbars-hint" class="placeholder-text" style="display: none;">Lower is better.</p>
    <div class="catbars-wrapper"><canvas id="category-bars-canvas"></canvas></div>
    <p id="catbars-empty" class="placeholder-text" style="display: none;">
        No projections available. Click "Refresh Data" first.
    </p>

    <script type="application/json" id="category-bars-data">{{ category_bars | tojson }}</script>
</div>
```

- [ ] **Step 3: Teach `toggleTopView` about the new view**

In the `toggleTopView` function, add a line alongside the existing three `style.display` assignments:

```javascript
    document.getElementById('view-categorybars').style.display = v === 'categorybars' ? '' : 'none';
```

Then, at the end of `toggleTopView`, trigger an initial/refresh render when the tab becomes active (the canvas has zero size while hidden, so Chart.js must (re)size when shown):

```javascript
    if (v === 'categorybars' && window.renderCategoryBars) window.renderCategoryBars();
```

- [ ] **Step 4: Add a minimal layout style and the script includes**

Just before the closing `</style>` tag in the existing `<style>` block, add:

```css
.catbars-wrapper { position: relative; height: 460px; margin-top: 0.75rem; }
```

Then, immediately after the closing `</style>` tag and before the `{% endif %}` (so they load only when standings data is present), add the Chart.js core, the error-bars plugin, and the module:

```html
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.4/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-chart-error-bars@4.4.1/build/index.umd.min.js"></script>
<script src="{{ url_for('static', filename='season_category_bars.js') }}"></script>
```

- [ ] **Step 5: Run the route test (now expect green)**

Run: `pytest tests/test_web/test_season_routes.py::test_standings_embeds_category_bars_data -v`
Expected: PASS (`id="category-bars-data"` now present, JSON carries `"preseason"` and the team names).

- [ ] **Step 6: Guard against template regressions**

Run: `pytest tests/test_web/test_season_routes.py -k standings -v`
Expected: all standings route tests PASS (the new tab must not break `test_standings_page_renders` or `test_standings_renders_table_with_data`).

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/standings.html
git commit -m "feat(standings): add Category Bars tab markup, controls, embedded data"
```

---

## Task 4: Chart.js render module `season_category_bars.js`

**Files:**
- Create: `src/fantasy_baseball/web/static/season_category_bars.js`

No JS unit-test harness exists in this repo; this task is verified by the route test (script tag present, Task 3) plus the manual dashboard smoke check in Task 5. Keep the module dependency-light and defensive (no data -> show the empty message, do not throw).

- [ ] **Step 1: Write the module**

Create `src/fantasy_baseball/web/static/season_category_bars.js` with exactly this content. It uses the `scatterWithErrorBars` chart type from the error-bars plugin (auto-registered by its UMD build) with a category y-axis so each team is a dot with horizontal whiskers. ASCII only: `+/-` is emitted as `'±'`.

```javascript
/* Category Bars: ranked dot-plot of all teams for one roto category, with
 * +/-1 SD error bars. Reads the JSON embedded by standings.html, renders into
 * #category-bars-canvas, and re-renders on category/projection change.
 * Data shape: { preseason: {CAT: [{team, value, sd, is_user}, ...]}, current: {...} }
 */
(function () {
  "use strict";

  var RATE_CATS = { AVG: 3, ERA: 2, WHIP: 2 };
  var INVERSE_CATS = { ERA: true, WHIP: true };
  var USER_COLOR = "#e15759";
  var OTHER_COLOR = "#4e79a7";

  var state = { projection: "current", category: "R" };
  var chart = null;
  var payload = null;

  function loadPayload() {
    var node = document.getElementById("category-bars-data");
    if (!node) return null;
    try {
      return JSON.parse(node.textContent);
    } catch (e) {
      return null;
    }
  }

  function fmt(value, cat) {
    if (RATE_CATS[cat] != null) return value.toFixed(RATE_CATS[cat]);
    return String(Math.round(value));
  }

  function rowsFor() {
    if (!payload) return [];
    var flavor = payload[state.projection];
    if (!flavor) return [];
    return flavor[state.category] || [];
  }

  function render() {
    if (payload == null) payload = loadPayload();
    var rows = rowsFor();
    var canvas = document.getElementById("category-bars-canvas");
    var empty = document.getElementById("catbars-empty");
    if (!canvas) return;

    if (!rows.length) {
      if (chart) { chart.destroy(); chart = null; }
      canvas.style.display = "none";
      if (empty) empty.style.display = "";
      return;
    }
    canvas.style.display = "";
    if (empty) empty.style.display = "none";

    // rows arrive sorted best-on-top; the category y-axis lists top->bottom in
    // the order of `labels`, so use the rows as-is.
    var labels = rows.map(function (r) { return r.team; });
    var points = rows.map(function (r) {
      return { x: r.value, y: r.team, xMin: r.value - r.sd, xMax: r.value + r.sd };
    });
    var colors = rows.map(function (r) { return r.is_user ? USER_COLOR : OTHER_COLOR; });
    var cat = state.category;

    var hint = document.getElementById("catbars-hint");
    if (hint) hint.style.display = INVERSE_CATS[cat] ? "" : "none";

    var config = {
      type: "scatterWithErrorBars",
      data: {
        labels: labels,
        datasets: [{
          label: cat,
          data: points,
          backgroundColor: colors,
          borderColor: colors,
          pointRadius: 6,
          pointHoverRadius: 8,
          errorBarColor: "#888",
          errorBarWhiskerColor: "#888",
          errorBarLineWidth: 1.5,
          errorBarWhiskerSize: 8
        }]
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: function (ctx) {
                var r = rows[ctx.dataIndex];
                return r.team + ": " + fmt(r.value, cat) +
                  " ± " + fmt(r.sd, cat);
              }
            }
          }
        },
        scales: {
          x: { title: { display: true, text: cat } },
          y: { type: "category", labels: labels }
        }
      }
    };

    if (chart) chart.destroy();
    chart = new Chart(canvas.getContext("2d"), config);
  }

  // Exposed so toggleTopView can (re)render when the tab is shown (the canvas
  // has zero size while its view is display:none).
  window.renderCategoryBars = render;

  window.catBarsSetProjection = function (el) {
    document.querySelectorAll("#catbars-proj-toggle .pill").forEach(function (p) {
      p.classList.remove("active");
    });
    el.classList.add("active");
    state.projection = el.dataset.cbproj;
    render();
  };

  window.catBarsSetCategory = function (el) {
    document.querySelectorAll("#catbars-cat-toggle .pill").forEach(function (p) {
      p.classList.remove("active");
    });
    el.classList.add("active");
    state.category = el.dataset.cbcat;
    render();
  };
})();
```

- [ ] **Step 2: Verify the route still serves the page and references the module**

Run: `pytest tests/test_web/test_season_routes.py -k standings -v`
Expected: PASS (sanity -- the new static file does not affect server-side rendering, but confirms nothing regressed).

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/static/season_category_bars.js
git commit -m "feat(standings): add Category Bars Chart.js render module"
```

---

## Task 5: Full verification + manual smoke check

**Files:** none (verification only; commit any fixes the checks force).

- [ ] **Step 1: Run the touched test areas**

Run: `pytest tests/test_web/test_season_data.py tests/test_web/test_season_routes.py -v`
Expected: all PASS, including the 5 `category_bars` data tests and the route embed test.

- [ ] **Step 2: Lint + format + dead-code + types**

```bash
ruff check src/fantasy_baseball/web/season_data.py src/fantasy_baseball/web/season_routes.py tests/test_web/test_season_data.py tests/test_web/test_season_routes.py
ruff format --check src/fantasy_baseball/web/season_data.py src/fantasy_baseball/web/season_routes.py
vulture src/fantasy_baseball/web/season_data.py
```

Expected: zero violations; no NEW vulture findings. `_category_bars_one_flavor` is referenced by `format_category_bars_for_display`, and the latter by `season_routes`, so neither should flag as dead. The three `window.*` JS functions are referenced from `onclick`/`toggleTopView` in the template (vulture does not scan JS).

If any file you touched is listed under `[tool.mypy].files` in `pyproject.toml`, also run:

```bash
mypy src/fantasy_baseball/web/season_data.py src/fantasy_baseball/web/season_routes.py
```

Expected: no new errors. (Check the current `[tool.mypy].files` list first; if these files are not covered, state that mypy was skipped for them.)

- [ ] **Step 3: Manual dashboard smoke check**

Run: `python scripts/run_season_dashboard.py`
Then in a browser at `http://localhost:5001/standings` (after a data refresh, or with cached projections present):
1. Click the "Category Bars" tab -- the dot-plot renders for Runs (R) by default.
2. Confirm your team ("Hart of the Order") dot is red; others are blue.
3. Click through several categories: counting cats sort highest-on-top; ERA and WHIP sort lowest-on-top and show the "Lower is better." hint.
4. Toggle Current / Preseason -- dots and whiskers update; preseason whiskers are wider (full-season SD) than current.
5. Open the browser dev console -- confirm no errors (especially that `chart.umd.min.js` and `chartjs-chart-error-bars` loaded; a 404 on the plugin CDN means the pinned version needs adjusting). If the plugin URL 404s, find the correct published version on jsdelivr and update the `<script src>` in `standings.html` (Task 3, Step 4), then re-check.

- [ ] **Step 4: Final commit (only if Step 2/3 forced fixes)**

```bash
git add -A
git commit -m "fix(standings): address Category Bars verification findings"
```

---

## Self-Review notes

- **Spec coverage:** concept/tab (Tasks 3-4), no-new-math data reuse (Task 1 reads cached display dicts), formatter with inverse-cat sort + SD default (Task 1), route embed no-new-endpoint (Task 2), tab/controls/Chart.js+plugin (Tasks 3-4), best-on-top + user highlight + lower-is-better hint (Tasks 3-4), empty-SD bare dot + pre-refresh empty state (Tasks 1, 3, 4), tests (Tasks 1-2), verification incl. manual (Task 5). All spec sections map to a task.
- **Type consistency:** `format_category_bars_for_display(preseason_data, current_projected_data)` signature is identical in Task 1 (definition), Task 2 (call), and the spec. Row keys `{team, value, sd, is_user}` and the JSON shape `{preseason|current: {CAT: [rows]}}` are identical across the formatter (Task 1), the route test (Task 2), and the JS consumer (Task 4: `r.team`, `r.value`, `r.sd`, `r.is_user`; `payload[state.projection][state.category]`).
- **Known execution risk (flagged, not a placeholder):** the exact `chartjs-chart-error-bars` CDN version and the `scatterWithErrorBars` + category-y-axis combination are verified by Task 5 Step 3. If scatter+category misbehaves, the spec's documented fallback is `barWithErrorBars` with `indexAxis: "y"` (native horizontal error bars on a category axis) -- swap the `type` and drop the `y` from each data point, keeping `xMin`/`xMax`. This is the only part not provable by an automated test in this repo (no JS harness).
```
