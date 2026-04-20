# Standings Stat-Distance Coloring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the five-bucket rank-based standings coloring with a continuous stat-distance gradient so clustered leaders (100/99/99/99 HR) all render bright green.

**Architecture:** Replace the `_compute_category_ranks` helper in `src/fantasy_baseball/web/season_data.py` with `_compute_color_intensity`, which emits a signed float in `[-1, 1]` per category plus the Total column. The template renders two mutually-exclusive CSS custom properties (`--pos` or `--neg`) on each cell; CSS uses attribute-present selectors and `color-mix` / `calc` to interpolate background and text color against a shared palette. The Monte Carlo view is unchanged.

**Tech Stack:** Python / Jinja / CSS (no new deps).

**Spec:** `docs/superpowers/specs/2026-04-20-standings-stat-distance-coloring-design.md`

**Branch:** `feat/standings-stat-distance-coloring` (already checked out; spec is already committed).

---

## Background for implementer

The relevant files today:

- **`src/fantasy_baseball/web/season_data.py`** — `format_standings_for_display()` (around lines 204–285) is the **only** function that assembles the per-category color metadata. It is called three times from `src/fantasy_baseball/web/season_routes.py`:
  1. Current standings (live Yahoo data).
  2. Projected / Preseason ERoto (`preseason_standings`).
  3. Projected / Current ERoto (`projected_standings` — ROS).
  All three flow through `format_standings_for_display`, so a single change here updates every tab. No separate "projected" builder exists.
- **`_compute_category_ranks`** (line 751) is the rank helper consumed at line 240. It is used **only** by `format_standings_for_display`. After this plan, it is dead and must be deleted.
- **`src/fantasy_baseball/web/templates/season/standings.html`** — two places render per-category cells: the Current table (lines 44–52) and the `eroto_table(data, cell_class)` macro (lines 67–98). The macro is reused for both Projected sub-tabs (preseason and current ROS). Total-column cells are at lines 52 and 93.
- **`src/fantasy_baseball/web/static/season.css`** — the five `td.rank-*` rules are at lines 194–198.
- **`tests/test_web/test_season_data.py`** — two existing tests reference `color_classes` today and must move to `color_intensity`: `test_format_standings_color_codes_all_teams` (line 72) and `test_format_standings_tied_teams_same_color` (line 138).

Repo conventions worth knowing for this plan:

- Roto categories live in `fantasy_baseball.utils.constants.ALL_CATEGORIES` (10 items). Rate stats where lower is better come from `INVERSE_STATS` in the same module, imported as `INVERSE_CATS` at `season_data.py:37`. Reuse that import — do not inline a literal set.
- `CategoryStats` objects support indexing by category string (`entry.stats[cat]`), as used in the existing `_compute_category_ranks`.
- `score_roto` returns a `{team_name: {cat_pts, ..., total}}` dict; the user-visible `total` may be overridden by Yahoo's `points_for` (line 249). The Total-column intensity must use the final, post-override total.
- Template style: existing inline styles in `standings.html` use simple `style="..."` strings. Jinja number formatting — use `|round(3)` or similar so emitted numbers are stable.
- `--text-primary` and similar CSS variables are defined in `season.css` (used throughout, e.g., `var(--text-secondary)` at `standings.html:43`).

---

## File Structure

Files touched by this plan:

- **`src/fantasy_baseball/web/season_data.py`** — add `_compute_color_intensity`; delete `_compute_category_ranks`; update `format_standings_for_display` to emit `color_intensity` instead of `color_classes`; update docstring.
- **`src/fantasy_baseball/web/templates/season/standings.html`** — swap `color_classes` rendering for `color_intensity` rendering (Current table + `eroto_table` macro); add coloring to both Total-column cells.
- **`src/fantasy_baseball/web/static/season.css`** — delete the five `td.rank-*` rules; add continuous gradient rules keyed on `--pos` / `--neg` attribute presence.
- **`tests/test_web/test_season_data.py`** — update two existing tests (`test_format_standings_color_codes_all_teams`, `test_format_standings_tied_teams_same_color`); add four new tests covering leader/trailer, ERA/WHIP flip, all-tied case, clustered case, and Total column.

No new files.

---

## Task 1: Tests for `color_intensity`

**Files:**
- Modify: `tests/test_web/test_season_data.py` (replaces the two tests that reference `color_classes`; adds new tests)

Write the failing tests first. All six assertions below live in the same file.

- [ ] **Step 1: Replace `test_format_standings_color_codes_all_teams`**

Open `tests/test_web/test_season_data.py`. Delete the existing function at line 72 and replace with the block below. The sample fixture at lines 50–61 provides three teams; SB values are Hart=50, SkeleThor=40, Cavalli=55, so `(50-40)/(55-40) = 0.667` → intensity `2*0.667 - 1 ≈ 0.333` for Hart.

```python
def test_format_standings_color_intensity_per_team():
    data = format_standings_for_display(_standings_to_snapshot(_sample_standings()), "Hart of the Order")
    hart = next(t for t in data["teams"] if t["name"] == "Hart of the Order")
    skel = next(t for t in data["teams"] if t["name"] == "SkeleThor")
    cav = next(t for t in data["teams"] if t["name"] == "Send in the Cavalli")

    assert hart["is_user"] is True
    assert "color_intensity" in hart
    # SB: Hart=50, SkeleThor=40, Cavalli=55 → Cavalli leads, SkeleThor trails.
    assert cav["color_intensity"]["SB"] == pytest.approx(1.0)
    assert skel["color_intensity"]["SB"] == pytest.approx(-1.0)
    # Hart sits in between: (50-40)/(55-40) = 0.667, intensity = 2*0.667-1 ≈ 0.333
    assert hart["color_intensity"]["SB"] == pytest.approx(0.333, abs=0.01)
    # Total column gets an intensity too.
    assert "total" in hart["color_intensity"]
```

- [ ] **Step 2: Replace `test_format_standings_tied_teams_same_color`**

Delete the existing function at line 138 and replace with this. When all teams are tied in a category, the category key must be absent from `color_intensity`.

```python
def test_format_standings_tied_category_has_no_intensity():
    """When every team is tied in a category (max == min), the key is absent."""
    standings = _standings_to_snapshot([
        {"name": "Team A", "team_key": "a", "rank": 1,
         "stats": {"R": 100, "HR": 30, "RBI": 90, "SB": 20, "AVG": 0.260,
                   "W": 10, "K": 200, "SV": 10, "ERA": 3.50, "WHIP": 1.20}},
        {"name": "Team B", "team_key": "b", "rank": 2,
         "stats": {"R": 100, "HR": 25, "RBI": 90, "SB": 15, "AVG": 0.255,
                   "W": 8, "K": 180, "SV": 8, "ERA": 3.80, "WHIP": 1.25}},
    ])
    data = format_standings_for_display(standings, "Team A")
    a = next(t for t in data["teams"] if t["name"] == "Team A")
    b = next(t for t in data["teams"] if t["name"] == "Team B")
    # R and RBI are tied across all teams — the key is absent for everyone.
    assert "R" not in a["color_intensity"]
    assert "R" not in b["color_intensity"]
    assert "RBI" not in a["color_intensity"]
    assert "RBI" not in b["color_intensity"]
    # Non-tied categories still populated.
    assert "HR" in a["color_intensity"]
```

- [ ] **Step 3: Add `test_format_standings_era_whip_inverted`**

Append this test to the end of the file's standings-format test group (right after the new `test_format_standings_tied_category_has_no_intensity`).

```python
def test_format_standings_era_whip_inverted():
    """Lower ERA/WHIP → higher intensity (+1.0 at the min, not the max)."""
    standings = _standings_to_snapshot([
        {"name": "LowEra", "team_key": "a", "rank": 1,
         "stats": {"R": 100, "HR": 30, "RBI": 90, "SB": 20, "AVG": 0.260,
                   "W": 10, "K": 200, "SV": 10, "ERA": 2.50, "WHIP": 1.00}},
        {"name": "HighEra", "team_key": "b", "rank": 2,
         "stats": {"R": 90, "HR": 25, "RBI": 80, "SB": 15, "AVG": 0.255,
                   "W": 8, "K": 180, "SV": 8, "ERA": 5.00, "WHIP": 1.50}},
    ])
    data = format_standings_for_display(standings, "LowEra")
    low = next(t for t in data["teams"] if t["name"] == "LowEra")
    high = next(t for t in data["teams"] if t["name"] == "HighEra")
    # Lowest ERA / WHIP should read as the leader (+1.0).
    assert low["color_intensity"]["ERA"] == pytest.approx(1.0)
    assert low["color_intensity"]["WHIP"] == pytest.approx(1.0)
    assert high["color_intensity"]["ERA"] == pytest.approx(-1.0)
    assert high["color_intensity"]["WHIP"] == pytest.approx(-1.0)
```

- [ ] **Step 4: Add `test_format_standings_clustered_leaders_share_intensity`**

Append this test after `test_format_standings_era_whip_inverted`. It pins the central scenario from the spec.

```python
def test_format_standings_clustered_leaders_share_intensity():
    """Three teams at 99 HR (one behind 100) should get the same intensity."""
    hr_values = [100, 99, 99, 99, 70, 65, 55, 50, 45, 40]
    teams = []
    for i, hr in enumerate(hr_values):
        teams.append({
            "name": f"Team{i}", "team_key": f"k{i}", "rank": i + 1,
            "stats": {"R": 100, "HR": hr, "RBI": 90, "SB": 20, "AVG": 0.260,
                      "W": 10, "K": 200, "SV": 10, "ERA": 3.50, "WHIP": 1.20},
        })
    data = format_standings_for_display(_standings_to_snapshot(teams), "Team0")
    by_name = {t["name"]: t for t in data["teams"]}
    # Leader at 100 → +1.0. Trailer at 40 → -1.0.
    assert by_name["Team0"]["color_intensity"]["HR"] == pytest.approx(1.0)
    assert by_name["Team9"]["color_intensity"]["HR"] == pytest.approx(-1.0)
    # The three 99s share the same intensity: (99-40)/(100-40)=0.9833, intensity=0.9667.
    expected = pytest.approx(0.9667, abs=0.001)
    assert by_name["Team1"]["color_intensity"]["HR"] == expected
    assert by_name["Team2"]["color_intensity"]["HR"] == expected
    assert by_name["Team3"]["color_intensity"]["HR"] == expected
```

- [ ] **Step 5: Add `test_format_standings_total_column_intensity`**

Append after the clustered-leaders test.

```python
def test_format_standings_total_column_intensity():
    """Total column intensity tracks distance from top/bottom total roto points."""
    data = format_standings_for_display(_standings_to_snapshot(_sample_standings()), "Hart of the Order")
    teams = data["teams"]
    # Team with highest total → +1.0; lowest → -1.0.
    top = max(teams, key=lambda t: t["roto_points"]["total"])
    bot = min(teams, key=lambda t: t["roto_points"]["total"])
    assert top["color_intensity"]["total"] == pytest.approx(1.0)
    assert bot["color_intensity"]["total"] == pytest.approx(-1.0)
```

- [ ] **Step 6: Run the new tests — expect failure**

Run: `pytest tests/test_web/test_season_data.py -v -k "color_intensity or tied_category or era_whip_inverted or clustered_leaders or total_column"`

Expected: 5 of the 5 new tests fail with `AssertionError` or `KeyError: 'color_intensity'`. This confirms the old code still emits `color_classes` and no `color_intensity`.

- [ ] **Step 7: Commit the failing tests together with the implementation in Task 2**

Do not commit yet — tests will be red. They'll turn green in Task 2 and be committed there.

---

## Task 2: `_compute_color_intensity` helper + updated formatter

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py` (add helper; delete `_compute_category_ranks`; update `format_standings_for_display`)

- [ ] **Step 1: Add the intensity helper**

Open `src/fantasy_baseball/web/season_data.py`. Insert the new helper **replacing** the existing `_compute_category_ranks` at line 751. The helper signature takes the standings snapshot and a precomputed map of team name → total roto points (so the Total column can be handled in the same pass).

```python
def _compute_color_intensity(
    standings: StandingsSnapshot,
    team_totals: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Per-team, per-category signed intensity in [-1, 1].

    For each category, intensity = 2 * ((value - min) / (max - min)) - 1,
    with ERA / WHIP (``INVERSE_CATS``) flipped so the lowest value is +1.0.
    Categories where every team is tied (``max == min``) are omitted —
    callers render those cells neutral.

    The ``"total"`` key is populated from ``team_totals`` using the same
    formula (leader of total roto points = +1.0, no inversion).
    """
    out: dict[str, dict[str, float]] = {e.team_name: {} for e in standings.entries}

    for cat in ALL_CATEGORIES:
        vals = {e.team_name: float(e.stats[cat]) for e in standings.entries}
        lo, hi = min(vals.values()), max(vals.values())
        if hi - lo < 1e-12:
            continue  # tied category — omit the key for every team
        span = hi - lo
        for name, v in vals.items():
            t = (v - lo) / span
            if cat in INVERSE_CATS:
                t = 1.0 - t
            out[name][cat] = 2.0 * t - 1.0

    # Total column
    lo_t, hi_t = min(team_totals.values()), max(team_totals.values())
    if hi_t - lo_t >= 1e-12:
        span = hi_t - lo_t
        for name, v in team_totals.items():
            t = (v - lo_t) / span
            out[name]["total"] = 2.0 * t - 1.0

    return out
```

- [ ] **Step 2: Update `format_standings_for_display` to call the new helper**

Still in `season_data.py`. Replace the block at lines 240–273 (from `cat_ranks = _compute_category_ranks(standings)` through the existing `teams.append({...})` loop) with the block below. The key changes:

1. Remove the `cat_ranks` call and the rank-to-class lookup.
2. Build `team_totals` from `roto_pts["total"]` (which already reflects the Yahoo `points_for` override when present — see the line right above).
3. After the loop, call `_compute_color_intensity` and attach `color_intensity` per team.

```python
    teams = []
    team_totals: dict[str, float] = {}
    for entry in standings.entries:
        name = entry.team_name
        roto_pts = dict(roto[name])

        if has_yahoo_totals:
            roto_pts["score_roto_total"] = roto_pts["total"]
            roto_pts["total"] = entry.yahoo_points_for

        team_totals[name] = float(roto_pts["total"])
        teams.append({
            "name": name,
            "team_key": entry.team_key,
            "stats": entry.stats,
            "roto_points": roto_pts,
            "is_user": name == user_team_name,
            "sds": team_sds.get(name, {}) if team_sds else {},
        })

    intensity = _compute_color_intensity(standings, team_totals)
    for team in teams:
        team["color_intensity"] = intensity[team["name"]]
```

- [ ] **Step 3: Update the docstring of `format_standings_for_display`**

Replace the `Returns:` line at line 218:

```
    Returns:
        {"teams": [...]} where each team has roto_points, is_user flag,
        color_intensity, and rank. color_intensity is a dict of
        {category: float in [-1, 1]} plus a "total" key; categories
        where all teams tie are absent.
```

- [ ] **Step 4: Delete the now-dead `_compute_category_ranks`**

At what was line 751, delete the entire `_compute_category_ranks` function (including its docstring). It has no remaining callers (grep confirms: the only call site was the line just removed).

- [ ] **Step 5: Run the new tests — expect pass**

Run: `pytest tests/test_web/test_season_data.py -v`

Expected: all tests pass, including the five new ones plus the two replaced ones.

- [ ] **Step 6: Run the full test suite for safety**

Run: `pytest -v`

Expected: all pass. (If `vulture` is in the test suite and newly flags `_compute_category_ranks` as dead, that's fine — we deleted it.)

- [ ] **Step 7: Run ruff + format**

Run: `ruff check src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py`
Expected: zero violations.

Run: `ruff format --check src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py`
Expected: no formatting drift. (If drift, run `ruff format src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py`.)

- [ ] **Step 8: Commit the Python changes**

```bash
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py
git commit -m "$(cat <<'EOF'
refactor(web): replace standings rank-buckets with color_intensity

Compute per-category signed intensity in [-1, 1] from the min/max stat
spread across teams, plus a "total" key for the Total column. ERA and
WHIP are inverted so lowest = +1.0. Categories where all teams tie are
omitted. Deletes the now-unused _compute_category_ranks helper.

Template rendering of the new shape comes in the next commit.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Template — emit `--pos` / `--neg` on standings cells

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/standings.html`

The template emits exactly one of `--pos` or `--neg` per cell (whichever is non-zero). Absent intensity (tied category or no entry) → no style attribute, cell renders neutral.

- [ ] **Step 1: Update the Current-view per-category cell (lines 44–51)**

Replace:

```html
                {% for cat in categories %}
                <td class="{{ team.color_classes[cat] }} stat-cell"
                    data-points="{{ team.roto_points[cat ~ '_pts'] | round(1) }}"
                    data-stat="{{ team.stats[cat] }}"
                    data-cat="{{ cat }}">
                    {{ team.roto_points[cat ~ '_pts'] | round(1) }}
                </td>
                {% endfor %}
```

With:

```html
                {% for cat in categories %}
                {% set i = team.color_intensity.get(cat) %}
                <td class="stat-cell"
                    {% if i is not none and i > 0 %}style="--pos: {{ i | round(3) }}"
                    {% elif i is not none and i < 0 %}style="--neg: {{ (-i) | round(3) }}"{% endif %}
                    data-points="{{ team.roto_points[cat ~ '_pts'] | round(1) }}"
                    data-stat="{{ team.stats[cat] }}"
                    data-cat="{{ cat }}">
                    {{ team.roto_points[cat ~ '_pts'] | round(1) }}
                </td>
                {% endfor %}
```

- [ ] **Step 2: Update the Current-view Total cell (line 52)**

Replace:

```html
                <td class="total-col">{{ team.roto_points.total | round(1) }}</td>
```

With:

```html
                {% set i = team.color_intensity.get("total") %}
                <td class="total-col"
                    {% if i is not none and i > 0 %}style="--pos: {{ i | round(3) }}"
                    {% elif i is not none and i < 0 %}style="--neg: {{ (-i) | round(3) }}"{% endif %}>
                    {{ team.roto_points.total | round(1) }}
                </td>
```

- [ ] **Step 3: Update the `eroto_table` macro per-category cell (lines 84–92)**

Replace:

```html
                {% for cat in categories %}
                <td class="{{ team.color_classes[cat] }} {{ cell_class }}"
                    data-points="{{ team.roto_points[cat ~ '_pts'] | round(1) }}"
                    data-stat="{{ team.stats[cat] }}"
                    data-sd="{{ team.sds.get(cat, 0) }}"
                    data-cat="{{ cat }}">
                    {{ team.roto_points[cat ~ '_pts'] | round(1) }}
                </td>
                {% endfor %}
```

With:

```html
                {% for cat in categories %}
                {% set i = team.color_intensity.get(cat) %}
                <td class="stat-cell {{ cell_class }}"
                    {% if i is not none and i > 0 %}style="--pos: {{ i | round(3) }}"
                    {% elif i is not none and i < 0 %}style="--neg: {{ (-i) | round(3) }}"{% endif %}
                    data-points="{{ team.roto_points[cat ~ '_pts'] | round(1) }}"
                    data-stat="{{ team.stats[cat] }}"
                    data-sd="{{ team.sds.get(cat, 0) }}"
                    data-cat="{{ cat }}">
                    {{ team.roto_points[cat ~ '_pts'] | round(1) }}
                </td>
                {% endfor %}
```

(`stat-cell` is added as a base class so the same CSS selectors work. The existing JS in the template selects by `.proj-preseason-cell` / `.proj-current-cell` — that class is still attached via `{{ cell_class }}`, so toggle behavior is unchanged.)

- [ ] **Step 4: Update the `eroto_table` macro Total cell (line 93)**

Replace:

```html
                <td class="total-col">{{ team.roto_points.total | round(1) }}</td>
```

With:

```html
                {% set i = team.color_intensity.get("total") %}
                <td class="total-col"
                    {% if i is not none and i > 0 %}style="--pos: {{ i | round(3) }}"
                    {% elif i is not none and i < 0 %}style="--neg: {{ (-i) | round(3) }}"{% endif %}>
                    {{ team.roto_points.total | round(1) }}
                </td>
```

- [ ] **Step 5: Grep for any remaining `color_classes` references**

Run: `grep -rn "color_classes" src/ tests/`
Expected: no hits. If any remain, update them consistently. (There should be none — only `season_data.py` and the two template files wrote or read this field; `season_data.py` was updated in Task 2.)

---

## Task 4: CSS — replace rank buckets with continuous gradient

**Files:**
- Modify: `src/fantasy_baseball/web/static/season.css`

- [ ] **Step 1: Delete the five `rank-*` rules**

In `src/fantasy_baseball/web/static/season.css`, delete lines 193–198 (the comment and the five `td.rank-*` rules):

```css
/* Rank-based background colors for standings cells */
td.rank-top    { background: rgba(34, 197, 94, 0.25); color: #4ade80; font-weight: 600; }
td.rank-high   { background: rgba(34, 197, 94, 0.10); color: #86efac; }
td.rank-mid    { }
td.rank-low    { background: rgba(239, 68, 68, 0.10); color: #fca5a5; }
td.rank-bottom { background: rgba(239, 68, 68, 0.25); color: #f87171; font-weight: 600; }
```

- [ ] **Step 2: Add the gradient rules in their place**

Insert the following block at the same position (just before the existing `.total-col { color: var(--accent); font-weight: bold; }` rule):

```css
/* Stat-distance coloring for standings cells.
   Template sets exactly one of --pos / --neg in [0, 1] per cell;
   cells with neither render neutral. Peak alpha 0.25 matches the
   prior rank-top / rank-bottom look. */
td[style*="--pos"] {
    background: rgba(34, 197, 94, calc(var(--pos) * 0.25));
    color: color-mix(in srgb, var(--text-primary), #4ade80 calc(var(--pos) * 70%));
}
td[style*="--neg"] {
    background: rgba(239, 68, 68, calc(var(--neg) * 0.25));
    color: color-mix(in srgb, var(--text-primary), #f87171 calc(var(--neg) * 70%));
}
```

Notes:
- Attribute-present selectors (`[style*="--pos"]`, `[style*="--neg"]`) trigger the rule whenever that custom property name appears in the inline `style`. Because the template emits **exactly one** of the two variables (never both), there is no ambiguity.
- Specificity (`0,1,1`) beats the existing `.total-col { color: var(--accent); ... }` (`0,1,0`), so the Total column picks up the gradient color when intensity is set and falls back to accent when not.
- `color-mix` is supported in all modern evergreen browsers (Chrome 111+, Firefox 113+, Safari 16.2+). The local dashboard and Render deploy run in those.

- [ ] **Step 3: Run `ruff check` + `ruff format --check`**

Run: `ruff check .`
Expected: zero violations.

Run: `ruff format --check .`
Expected: no formatting drift.

---

## Task 5: Manual browser verification

**Files:** none (runtime check)

- [ ] **Step 1: Start the dev dashboard**

Run (new terminal): `python scripts/run_season_dashboard.py --port 5001`

Expected: dashboard starts; Yahoo auth may prompt if token is stale — follow the CLI flow once.

- [ ] **Step 2: Load `/standings` and walk through every view**

In a browser, open `http://localhost:5001/standings` and visually confirm:

1. **Current → Roto Points** — per-category cells show the red/neutral/green gradient based on stat distance, not rank position. The Total column is gradient-colored too. Categories where every team is tied (early-season SV, for instance) render neutral for all teams.
2. **Current → Stat Totals toggle** — category coloring persists when the toggle swaps the displayed value from roto points to raw stats.
3. **Projected → Preseason → both toggles** — same visual treatment, same gradient direction, same Total coloring.
4. **Projected → Current ERoto → both toggles** — same.
5. **Monte Carlo (any sub-tab)** — unchanged (no per-category cells; table shape different).
6. ERA and WHIP: the team with the **lowest** ERA / WHIP should read bright green, not bright red.

- [ ] **Step 3: Sanity-check clustered categories**

Open the browser inspector and look at any category with tightly-clustered leaders (in the current live data this will vary; HR is often a good one mid-season). Confirm adjacent teams with near-identical values render near-identical colors, not stepped buckets.

- [ ] **Step 4: Stop the dev server**

Ctrl-C in the dashboard terminal.

---

## Task 6: Final verification and commit

**Files:** none (verification) / commit

- [ ] **Step 1: Run forced verification checklist from CLAUDE.md**

In order:

```bash
pytest -v
ruff check .
ruff format --check .
vulture
```

Expected:
- `pytest -v` — all green.
- `ruff check .` — zero violations.
- `ruff format --check .` — no drift.
- `vulture` — no **new** findings introduced by this change. Pre-existing findings unrelated to standings / season_data are acceptable (note them in the final summary if any appear). `_compute_category_ranks` should be gone, not newly flagged.

`mypy` is **not required** for this change — the modified files (`season_data.py`, `standings.html`, `season.css`, test file) are not under `[tool.mypy].files` in `pyproject.toml`.

- [ ] **Step 2: Commit the template + CSS changes**

```bash
git add src/fantasy_baseball/web/templates/season/standings.html src/fantasy_baseball/web/static/season.css
git commit -m "$(cat <<'EOF'
feat(web): render standings color gradient by stat distance

Each standings cell receives --pos or --neg (in [0, 1]) based on how
close the team is to the category leader / trailer. CSS interpolates
background alpha and text color against a shared palette. Applied to
the Current view, both Projected sub-tabs, and the Total column;
Monte Carlo view is unchanged.

Co-Authored-By: Claude Opus 4.7 <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 3: Summary**

Paste the final `git log --oneline -3` output into the user-facing summary, confirming:

- spec commit (already present on branch)
- Python / test refactor commit
- template + CSS commit

Do **not** merge to main — per user's standing instruction, never merge without asking.

---

## Spec coverage check

- Continuous red-to-green gradient based on stat distance → `_compute_color_intensity` + CSS in Tasks 2 & 4.
- Applied to Current, Projected Preseason, Projected Current ROS → all three go through `format_standings_for_display` (Task 2) and both template paths (Task 3).
- Total column colored → team_totals branch in helper (Task 2) + both Total cells in template (Task 3).
- ERA / WHIP flipped → `INVERSE_CATS` branch in helper (Task 2) + test (Task 1 Step 3).
- Tied-category handling (key absent) → `if hi - lo < 1e-12: continue` in helper (Task 2) + test (Task 1 Step 2).
- Monte Carlo unchanged → no edits to `mc_*` template blocks.
- Clustered-leaders behavior → test (Task 1 Step 4) locks in `0.9667` for three tied 99s.
- Verification against existing palette weight → CSS uses 0.25 peak alpha matching prior `rank-top` / `rank-bottom` (Task 4 Step 2).
