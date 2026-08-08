# Trajectory Comp Charts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Source spec:** `docs/superpowers/specs/2026-08-07-trajectory-comp-charts-design.md`
**Issue:** [#346](https://github.com/alhart2015/FantasyBaseball/issues/346)

**Goal:** Draw the paced current season on the trajectory career line, give every comp its own small career chart with a match-age marker, and strip reading-instruction prose from the three trajectory pages.

**Architecture:** Three independent slices. (A) `PlayerView` grows a `paced` point read from the board's existing `now` field plus a suppression rule for when the base season is already realized, and the templates lose five prose clauses. (B) The chart blob grows comp `id`s and a top-level deduped `careers` map written by the push script. (C) The player template renders a grid of per-comp canvases, drawn by `trajectory_chart.js` with a ten-line inline Chart.js plugin for the vertical match line. Each slice degrades cleanly against a blob written before it.

**Tech Stack:** Python 3.11+, pandas, Flask + Jinja2, Chart.js (CDN, loaded via `window.ensureChartJs` in `season_trends.js`), pytest.

## Global Constraints

- **ASCII-only** in every source file, log message, format string, and template. This box is Windows; stdout is cp1252. Use `-` not en/em dash, `->` not arrow, straight quotes only. (`CLAUDE.md`)
- **Player identity is `(mlbam_id, pool)`**, serialized by `chart_key(mlbam_id, pool)` -> `"12345:hitter"`. Never key chart data on a bare id or a name.
- **Never hand-type an MLBAM id in a query.** Resolve it in code first. (`CLAUDE.md`)
- **Do not loosen a failing test to make it pass.** Task 1 edits four test assertions; its justification is written into the task and the commit message. No other task may edit an existing assertion.
- **Numeric defaults must not use `x or default`** -- `0.0` is falsy. Use `d.get(k, default)` or an explicit `is not None` check.
- **End-of-effort verification** (Task 13): `pytest -v`, `ruff check .`, `ruff format --check .`, `vulture`, and `mypy` if any touched file is under `[tool.mypy].files`.
- **Do NOT run `scripts/push_trajectory_board.py`** at any point in this plan. It writes to the same prod Upstash the live Render build reads. See Task 14's deploy gate.

---

## File Structure

**Modified:**
- `scripts/push_trajectory_board.py` -- `player_comps` emits `id`; the payload builder emits `base_season_partial` and a deduped `careers` map
- `src/fantasy_baseball/trajectory/sweep.py` -- `to_chart_payload` carries `careers`
- `src/fantasy_baseball/trajectory/comp_paths.py` -- one docstring paragraph
- `src/fantasy_baseball/web/trajectory_view.py` -- `PlayerView.paced`, `.paced_label`, `.comp_careers`; `_chart_extras` returns a triple; `_board_meta` passes `base_season_partial`
- `src/fantasy_baseball/web/templates/season/trajectory_player.html` -- prose out, paced table row in, comp grid in
- `src/fantasy_baseball/web/templates/season/trajectory.html` -- prose out
- `src/fantasy_baseball/web/templates/season/trajectory_teams.html` -- prose out
- `src/fantasy_baseball/web/static/trajectory_chart.js` -- paced point styling; the card grid renderer and the match-line plugin
- `src/fantasy_baseball/web/static/season.css` -- `.comp-grid`, `.comp-card`

**Test files modified:**
- `tests/test_web/test_trajectory_view.py`, `tests/test_web/test_season_routes.py`, `tests/test_scripts/test_push_trajectory_board.py`, `tests/test_trajectory/test_sweep.py`

**Created:**
- `tests/test_trajectory/test_no_ros_dependency.py`

---

# Phase A: the paced current season, and the prose

## Task 1: Make the chart fixture production-faithful

The module fixture in `tests/test_web/test_trajectory_view.py` stores a career history of ages `[25, 26, 27]` for players who are age 27 in the base season. **A real blob can never look like that.** The push script builds history from `complete = live[~live["partial_season"]]`, which excludes the base season by construction, so the subject's own base-season age is never in `history`.

This matters now because Task 3 suppresses the paced point exactly when the base-season age is already in `history`. Left as-is, the fixture puts every one of ~20 existing tests on the suppressed branch and the production path -- paced point drawn -- would be exercised only by new tests.

**This is a deliberate edit to existing test assertions.** Justification: the fixture asserts a state the pipeline cannot produce, and four assertions are pass-throughs of that fixture literal. No assertion's *meaning* changes; the ages shift down one year so the fixture describes a blob the writer could actually have written.

**Files:**
- Test: `tests/test_web/test_trajectory_view.py:1007,1050,1255,1342-1345`

**Interfaces:**
- Consumes: nothing
- Produces: a `_chart()` fixture whose history is `[[24, 14.0], [25, 16.0], [26, 20.0]]` for age-27 players, which every later task's tests rely on

- [ ] **Step 1: Shift the fixture history down one year**

In `_chart()` at line 1007:

```python
                "history": [[24, 14.0], [25, 16.0], [26, 20.0]],
```

- [ ] **Step 2: Update the four assertions that read those literals**

Line 1050, in `test_a_player_is_found_by_name_with_his_career_and_projection`:

```python
    assert [pt[0] for pt in view.history] == [24, 25, 26], "career ascends by age"
```

Line 1255, in `test_the_chart_lookup_keeps_the_pool_so_a_two_way_player_keeps_his_own_career`:

```python
    assert pitcher.history == [[24, 14.0], [25, 16.0], [26, 20.0]], "and not the other way"
```

Lines 1342 and 1345, in `test_history_is_sorted_by_age_even_if_the_payload_is_not`:

```python
    scrambled["players"][key] = {"history": [[26, 20.0], [24, 14.0], [25, 16.0]], "comps": []}
```

and the assertion below it:

```python
    ages = [pt[0] for pt in view.history]
    assert ages == [24, 25, 26], "the view sorts by age; the payload's order is not trusted"
```

(Keep whatever the existing assertion message says if it differs -- only the numbers change.)

- [ ] **Step 3: Run the full trajectory-view suite**

Run: `pytest tests/test_web/test_trajectory_view.py -v`
Expected: PASS, all tests. If anything else fails, it is reading a history literal this step missed -- fix that literal the same way; do not weaken the assertion.

- [ ] **Step 4: Commit**

```bash
git add tests/test_web/test_trajectory_view.py
git commit -m "test(trajectory): the chart fixture held a history the writer cannot produce

The subject's base-season age was in his own career history. push_trajectory_board
builds history from complete = live[~live[partial_season]], so the base season is
excluded by construction and that age can never appear. Four assertions were
pass-throughs of the fixture literal; the ages shift down a year and nothing else
changes. Prerequisite for the paced-point suppression rule (#346)."
```

---

## Task 2: The push script records whether the base season is in progress

`paced_label` must not call a finished season a "pace". The push knows the answer exactly.

**Files:**
- Modify: `scripts/push_trajectory_board.py` (the `to_payload(...)` call, currently around line 294)
- Modify: `src/fantasy_baseball/web/trajectory_view.py` (`_board_meta`, around line 209)
- Test: `tests/test_scripts/test_push_trajectory_board.py`

**Interfaces:**
- Consumes: `calendar` (the era-normalized hitter panel including partials) and `season`, both already local in the payload builder
- Produces: `payload["base_season_partial"]: bool`, surfaced as `Board.meta["base_season_partial"]` / `PlayerView.meta["base_season_partial"]`. **Absent means `True`** -- every currently-deployed blob is mid-season.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_scripts/test_push_trajectory_board.py`:

```python
def test_the_payload_records_whether_the_base_season_is_still_running() -> None:
    """`paced_label` must not call a finished season a pace, and only the push can
    tell: `partial_season` is a panel column, and the board carries no other signal
    that survives to the reader."""
    from fantasy_baseball.web.trajectory_view import _board_meta

    assert _board_meta({"base_season_partial": False})["base_season_partial"] is False
    assert _board_meta({"base_season_partial": True})["base_season_partial"] is True
    # An old blob predates the field. Mid-season is what every deployed blob means,
    # and it is also the safe default: it labels the point a pace, which is what the
    # suppression rule in `build_player_view` has already decided is drawable.
    assert _board_meta({})["base_season_partial"] is True
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_scripts/test_push_trajectory_board.py::test_the_payload_records_whether_the_base_season_is_still_running -v`
Expected: FAIL with `KeyError: 'base_season_partial'`

- [ ] **Step 3: Add the field to `_board_meta`**

In `src/fantasy_baseball/web/trajectory_view.py`, inside `_board_meta`'s returned dict, add:

```python
        # Whether the base season was still running when this board was built, which is
        # what stops `paced_label` calling a finished year a "pace". Defaults TRUE:
        # every blob written before this field existed was written mid-season, and the
        # suppression rule in `build_player_view` -- not this flag -- decides whether
        # the point is drawn at all. `dict.get` with a default rather than `or`, per
        # CLAUDE.md: `False or True` is `True` and would invert a real answer.
        "base_season_partial": bool(payload.get("base_season_partial", True)),
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_scripts/test_push_trajectory_board.py::test_the_payload_records_whether_the_base_season_is_still_running -v`
Expected: PASS

- [ ] **Step 5: Write the flag on the push side**

In `scripts/push_trajectory_board.py`, in the `to_payload(...)` call, beside `season_elapsed=round(elapsed, 4),` add:

```python
        # Whether `season` was still in progress when this panel was built. Read off
        # the panel rather than the calendar date: `_live_seasons` in build_pt_panel.py
        # flags a season partial iff `year >= today.year`, so a panel rebuilt in
        # January un-flags the season that just ended -- and the reader must follow the
        # panel it was actually built from, not today's date.
        base_season_partial=bool(
            calendar.loc[calendar["season"] == season, "partial_season"].any()
        ),
```

- [ ] **Step 6: Run the push-script suite**

Run: `pytest tests/test_scripts/test_push_trajectory_board.py -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add scripts/push_trajectory_board.py src/fantasy_baseball/web/trajectory_view.py tests/test_scripts/test_push_trajectory_board.py
git commit -m "feat(trajectory): record whether the base season was still running (#346)"
```

---

## Task 3: `PlayerView.paced` and `paced_label`

**Files:**
- Modify: `src/fantasy_baseball/web/trajectory_view.py` (`PlayerView` dataclass; the `empty` construction; the `replace(...)` at the end of `build_player_view`)
- Test: `tests/test_web/test_trajectory_view.py`

**Interfaces:**
- Consumes: `sp.age`, `sp.now`, `floor = sp.offset(scale)`, `payload["base_season"]`, `_board_meta(payload)["base_season_partial"]` -- all already in scope in `build_player_view`
- Produces:
  - `PlayerView.paced: list[float] | None` -- `[age, value]`, floor-netted
  - `PlayerView.paced_label: str` -- `"2026 pace"` or `"2026"`; `""` when not found

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_web/test_trajectory_view.py`. Note `_chart_ending_at`, a local helper for the second test -- the module fixture (after Task 1) is the *drawn* case, so the *suppressed* case needs its own blob:

```python
def _chart_including_base_age(payload: dict) -> dict:
    """A chart blob whose history runs THROUGH the subject's base-season age.

    The panel produces this only after the season ends and the panel is rebuilt --
    `_live_seasons` un-flags the finished year, it enters `complete`, and it lands in
    `history` while `base_season` still names it. Hand-built here because the fixture
    (correctly) does not contain it.
    """
    return to_chart_payload(
        {
            (p["id"], p["pool"]): {
                "history": [[25, 16.0], [26, 20.0], [27, 21.0]],
                "comps": [],
            }
            for p in payload["players"]
        },
        generated_at=str(payload["generated_at"]),
    )


def test_the_paced_season_is_the_boards_now_netted_against_the_same_floor(
    payload: dict,
) -> None:
    """The gap at the base season was the most useful point on the chart. `now` is
    already in the board payload -- this only draws it."""
    row = next(p for p in payload["players"] if p["name"] == "Big Bat")
    var = _view(payload, player="Big Bat", scale="var")
    sgp = _view(payload, player="Big Bat", scale="sgp")

    assert var.paced == [row["age"], pytest.approx(row["now"] - row["floor"])]
    assert sgp.paced == [row["age"], pytest.approx(row["now"])]


def test_the_paced_season_is_ordered_after_every_realized_one(payload: dict) -> None:
    """The chart concatenates history + paced into one line with no re-sort, so the
    paced age must be strictly the largest."""
    view = _view(payload, player="Big Bat")
    assert view.paced is not None
    assert view.history, "the fixture stores a career"
    assert max(pt[0] for pt in view.history) < view.paced[0]


def test_a_base_season_already_realized_gets_no_paced_point(payload: dict) -> None:
    """The offseason case. A panel rebuilt after the season ends un-flags it, so it
    enters `history` -- and appending `now` beside it would draw two points at one age,
    one of them labelled a pace, on a finished year."""
    view = build_player_view(
        payload, player="Big Bat", chart=_chart_including_base_age(payload)
    )
    assert view.age == 27, "the fixture's players are 27, which this blob's history covers"
    assert view.paced is None
    assert [pt[0] for pt in view.history] == [25, 26, 27], "history is left alone"


def test_the_paced_point_survives_a_chart_blob_that_does_not(payload: dict) -> None:
    """`now` comes from the BOARD, so it is never stale against the projection beside
    it. A refused chart blob costs the career line and the comps, not the anchor."""
    stale = _chart(payload)
    stale["generated_at"] = "some other build"
    view = build_player_view(payload, player="Big Bat", chart=stale)

    assert view.history == [] and view.comps == []
    assert view.chart_vintage_mismatch
    assert view.paced is not None, "board data, not chart data"


def test_an_unfound_player_has_no_paced_point(payload: dict) -> None:
    view = _view(payload, player="Nobody At All")
    assert view.paced is None
    assert view.paced_label == ""


@pytest.mark.parametrize(
    ("stored", "expected"),
    [(True, "2026 pace"), (False, "2026"), (None, "2026 pace")],
)
def test_the_paced_label_follows_the_boards_own_partial_flag(
    payload: dict, stored: object, expected: str
) -> None:
    """Built server-side like `axis_label`: the chart and the table read one string and
    cannot disagree about whether the season is finished. `None` is an old blob, which
    was written mid-season."""
    blob = {**payload}
    if stored is not None:
        blob["base_season_partial"] = stored
    blob["generated_at"] = f"hand-{next(_HAND_SEQ)}"  # do not share the parse cache
    view = build_player_view(blob, player="Big Bat", chart=None)
    assert view.paced_label == expected
```

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_web/test_trajectory_view.py -k "paced" -v`
Expected: FAIL with `AttributeError: 'PlayerView' object has no attribute 'paced'`

- [ ] **Step 3: Add the two fields to `PlayerView`**

In `src/fantasy_baseball/web/trajectory_view.py`, in the `PlayerView` dataclass, after the `history` field:

```python
    #: The PACED base season as [age, value], floor-netted exactly like `history`.
    #:
    #: NOT part of `history`, which means "realized complete seasons" -- several
    #: template branches key on `not board.history` to report a missing or mismatched
    #: chart blob, and folding a board-sourced point into a chart-blob-sourced list
    #: would make that check stop meaning what it says. It is drawn as the same line.
    #:
    #: The VALUE is straight-line prorated realized stats, never a projection blend:
    #: `board_inputs` -> `_paced` -> `prorate_partial` divides realized SGP by the
    #: elapsed fraction. No ROS projection reaches this model (#346), which
    #: tests/test_trajectory/test_no_ros_dependency.py keeps true.
    #:
    #: `None` when the player was not found, or when `age` is ALREADY a realized row in
    #: `history` -- see `build_player_view`.
    paced: list[float] | None = None
    #: What to call the paced point, finished server-side. Same rule as `axis_label`:
    #: ship the string, not the ingredients, so the chart and the table cannot disagree.
    paced_label: str = ""
```

Because these carry defaults, they must come after every other defaulted field or before none -- place them immediately before `meta`, which is the first defaulted field, and give them defaults as shown so field ordering stays legal.

- [ ] **Step 4: Populate them in `build_player_view`**

In `build_player_view`, after `floor = sp.offset(scale)` and after `extras, mismatch = _chart_extras(...)`, compute the history first so the collision check can see it. Replace the `history=sorted(...)` argument to `replace(...)` with a local built above the call:

```python
    history = sorted(
        ([int(a), float(v) - floor] for a, v in extras.get("history", [])),
        key=lambda pt: pt[0],
    )
    # THE SUPPRESSION RULE. `_live_seasons` (build_pt_panel.py) flags a season partial
    # iff `year >= today.year`, so a panel rebuilt in January un-flags the season that
    # just ended: it enters `complete`, lands in `history`, and `base_season` still
    # names it. Appending `now` beside it draws two points at one age, one of them
    # labelled a pace, on a finished year.
    #
    # Decided from `history` rather than from the stored `base_season_partial` flag
    # because this works on EVERY blob, including ones written before that flag
    # existed. The flag labels the point; this decides whether there is one.
    realized_ages = {pt[0] for pt in history}
    paced = None if sp.age in realized_ages else [sp.age, float(sp.now) - floor]
```

and pass `history=history, paced=paced,` plus:

```python
        paced_label=(
            f"{base} pace" if _board_meta(payload)["base_season_partial"] else str(base)
        ),
```

`base` is already bound at the top of the function. `paced_label` is set on the RESOLVED view only -- `empty` keeps its `""` default, which is what the not-found test asserts.

- [ ] **Step 5: Run the tests**

Run: `pytest tests/test_web/test_trajectory_view.py -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/trajectory_view.py tests/test_web/test_trajectory_view.py
git commit -m "feat(trajectory): draw the paced base season, suppressed once it is realized (#346)"
```

---

## Task 4: Render the paced point on the chart and in the table

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/trajectory_player.html`
- Modify: `src/fantasy_baseball/web/static/trajectory_chart.js`
- Test: `tests/test_web/test_season_routes.py`

**Interfaces:**
- Consumes: `PlayerView.paced`, `PlayerView.paced_label` (Task 3)
- Produces: `#trajectory-chart-data` gains `paced` and `paced_label` keys; the numbers table gains a `pace` row

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web/test_season_routes.py`, beside the other trajectory page tests:

```python
def test_the_player_page_ships_the_paced_point_to_the_chart(client):
    """The gap at the base season is the most useful point on the chart. It has to
    reach the JS island, not just the view model."""
    payload = _trajectory_payload()
    with _patched_cache(payload):
        html = client.get("/trajectory?view=player&player=Big Bat").data.decode()

    assert '"paced"' in html, "the island carries the point"
    assert '"paced_label"' in html, "and the finished label beside it"
    assert "pace</td>" in html, "the numbers table marks the row"
```

Match the existing file's fixture helpers -- reuse whatever the neighbouring trajectory route tests use to install a payload (`_trajectory_payload()` exists at line 2264); if the patch helper has a different name, use that name rather than inventing `_patched_cache`.

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py -k paced -v`
Expected: FAIL -- `'"paced"' in html` is False.

- [ ] **Step 3: Put the two fields on the island**

In `trajectory_player.html`, extend the JSON island:

```jinja
<script type="application/json" id="trajectory-chart-data">{{ {
  "name": board.name, "axis_label": board.axis_label, "history": board.history,
  "paced": board.paced, "paced_label": board.paced_label,
  "projection": board.projection, "comps": board.comps} | tojson }}</script>
```

- [ ] **Step 4: Add the table row**

In the same file, in the numbers table's `<tbody>`, between the history loop and the projection loop:

```jinja
    {% if board.paced %}
    <tr><td>{{ board.paced[0] }}</td><td>{{ '%.1f'|format(board.paced[1]) }}</td>
      <td class="muted">pace</td></tr>
    {% endif %}
```

- [ ] **Step 5: Draw it as the career line's last point**

In `trajectory_chart.js`, replace the career dataset (the last entry of `datasets`, labelled `data.name`) with:

```js
    (() => {
      // ONE line, history + the paced base season. The paced point is styled per
      // point rather than split into its own dataset: a second dataset would be a
      // second legend entry and a visible seam at the join, and the whole point of
      // this change is that there is no gap there.
      const career = data.history.map(([age, v]) => ({ x: age, y: v }));
      if (data.paced) career.push({ x: data.paced[0], y: data.paced[1] });
      const last = career.length - 1;
      const paced = (filled, hollow) =>
        career.map((_, i) => (data.paced && i === last ? hollow : filled));
      return {
        label: data.name,
        data: career,
        borderColor: "#4e79a7",
        borderWidth: 2.5,
        // An OPEN marker on the paced point: it is a full-season pace off a partial
        // year, not a finished season, and it must not read as one.
        pointRadius: paced(2, 5),
        pointBackgroundColor: paced("#4e79a7", "transparent"),
        pointBorderColor: "#4e79a7",
        pointBorderWidth: paced(1, 2),
        order: 0,
      };
    })(),
```

- [ ] **Step 6: Label it in the tooltip**

In the same file's `options.plugins.tooltip`, extend the existing object (keep the `_p10` filter):

```js
          tooltip: {
            filter: (item) => item.dataset.label !== "_p10",
            callbacks: {
              label: (item) => {
                const base = `${item.dataset.label}: ${item.formattedValue}`;
                const isPaced =
                  data.paced &&
                  item.dataset.label === data.name &&
                  item.dataIndex === item.dataset.data.length - 1;
                return isPaced ? `${base} (${data.paced_label})` : base;
              },
            },
          },
```

- [ ] **Step 7: Run the tests**

Run: `pytest tests/test_web/test_season_routes.py -k trajectory -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/trajectory_player.html src/fantasy_baseball/web/static/trajectory_chart.js tests/test_web/test_season_routes.py
git commit -m "feat(trajectory): the career line runs into the paced base season (#346)"
```

---

## Task 5: Remove the interpretation prose

Five clause-level deletions across three templates. **Delete only the clauses named. Every surrounding paragraph survives** -- see the spec's Requirement 3 table.

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/trajectory_player.html`
- Modify: `src/fantasy_baseball/web/templates/season/trajectory.html`
- Modify: `src/fantasy_baseball/web/templates/season/trajectory_teams.html`
- Test: `tests/test_web/test_season_routes.py`

**Interfaces:**
- Consumes: nothing
- Produces: nothing (rendering-only)

- [ ] **Step 1: Write the failing test**

```python
def test_the_trajectory_pages_do_not_explain_how_to_read_themselves(client):
    """Reading instructions are out; disclosure about the DATA stays. The two are
    easy to conflate, so both halves are asserted here."""
    payload = _trajectory_payload()
    with _patched_cache(payload):
        player = client.get("/trajectory?view=player&player=Big Bat").data.decode()
        board = client.get("/trajectory").data.decode()
        teams = client.get("/trajectory?view=teams").data.decode()

    assert "Solid is what happened" not in player
    assert "on purpose" not in player
    assert "not evidence for the forecast" not in player
    assert "value above the position-aware waiver floor" not in board
    assert "Ranks are LEAGUE ranks" not in teams

    # The kept half. These are facts about the build, not instructions.
    assert "Built" in board or "panel" in board, "the vintage disclosure survives"
    assert "age 27" in player, "the player's own header line survives"
    assert "Of everyone you could hold" in board
    assert "strongest team first" in teams
```

Adjust the two "kept half" substrings to whatever `ctl.vintage_note` actually renders in this fixture -- read the macro in `_trajectory_controls.html` and assert a literal it emits.

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py -k "explain" -v`
Expected: FAIL on the first `not in` assertion.

- [ ] **Step 3: `trajectory_player.html` -- the header paragraph**

Cut only the middle sentence. The result:

```jinja
<p class="muted">
  {{ board.name }} -- age {{ board.age }}, {{ board.slot }}.
  {% if board.extrapolated %}<strong>(!)</strong> This fit was evaluated outside its own
  support -- read the band, not the point estimate.{% endif %}
</p>
```

- [ ] **Step 4: `trajectory_player.html` -- the vintage paragraph**

Remove the whole `{% if board.history %}...{% endif %}` block (it is now false as well as unwanted -- Task 4 draws that season). The result:

```jinja
<p class="muted trajectory-vintage">
  {{ ctl.vintage_note(board) }}
</p>
```

- [ ] **Step 5: `trajectory_player.html` -- the comps paragraph**

Delete the entire `<p class="muted">` that begins "These are the closest realized paths" and ends "...the axis compares shapes on one line, not raw SGP.{% endif %}", including its closing `</p>`. **Keep** the `<h3>Closest realized paths ...</h3>` line above it and the `<table class="data-table">` below it.

- [ ] **Step 6: `trajectory.html` -- the VAR/SGP branch**

Cut the whole conditional, keeping the first sentence:

```jinja
<p class="muted">
  Of everyone you could hold, who is worth the most over {{ board.span }}.
</p>
```

- [ ] **Step 7: `trajectory_teams.html` -- the league-ranks sentence**

```jinja
<p class="muted">
  The best {{ board.per_team }} on every roster over {{ board.span }}, strongest team first.
</p>
```

- [ ] **Step 8: Run the tests**

Run: `pytest tests/test_web/test_season_routes.py -k trajectory -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/ tests/test_web/test_season_routes.py
git commit -m "refactor(trajectory): drop the how-to-read-this prose from all three pages (#346)"
```

---

## Task 6: Pin the no-ROS boundary

The trajectory model's current-season anchor is prorated realized stats. Nothing enforces that today, and "just blend in ROS for the current year" is a one-import change that would silently redefine what the chart's anchor means.

**Files:**
- Create: `tests/test_trajectory/test_no_ros_dependency.py`

**Interfaces:**
- Consumes: nothing
- Produces: nothing

- [ ] **Step 1: Write the test (it should pass immediately -- it is a guard, not a task)**

```python
"""The trajectory model reads REALIZED stats, prorated. Never a projection blend.

`board_inputs` -> `_paced` -> `prorate_partial` divides realized SGP by the elapsed
fraction, and that paced figure is what every fit is anchored on. Blending an ROS
projection into it would change what the chart's anchor MEANS -- from "on this pace"
to "on this pace, adjusted by somebody's forecast" -- while every number on the page
still rendered and every other test still passed. #346.

Surfaces are enumerated by name rather than matched on a "ROS" substring: a concept
this test cannot check is a test that asserts nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Import paths and symbols that would pull a projection blend into the trajectory fit.
FORBIDDEN = (
    "fantasy_baseball.data.ros_pipeline",
    "fantasy_baseball.data.ros_export_ingest",
    "fantasy_baseball.data.projections",
    "ROS_PROJECTIONS",
)


def _scoped_sources() -> list[Path]:
    files = sorted((PROJECT_ROOT / "src" / "fantasy_baseball" / "trajectory").rglob("*.py"))
    files.append(PROJECT_ROOT / "scripts" / "push_trajectory_board.py")
    return files


def test_the_scope_is_not_empty() -> None:
    """A glob that silently matched nothing would make every assertion below vacuous."""
    files = _scoped_sources()
    assert len(files) > 5
    assert all(path.is_file() for path in files)


@pytest.mark.parametrize("path", _scoped_sources(), ids=lambda p: p.name)
def test_the_trajectory_model_reads_no_ros_projection(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    hits = [name for name in FORBIDDEN if name in source]
    assert not hits, (
        f"{path.name} references {hits}. The trajectory model is anchored on PRORATED "
        "REALIZED stats (prorate_partial), and blending a projection into that changes "
        "what every fit on the keeper board means. If this is deliberate, the change "
        "belongs in the spec first -- see #346."
    )
```

- [ ] **Step 2: Run it**

Run: `pytest tests/test_trajectory/test_no_ros_dependency.py -v`
Expected: PASS. If it FAILS, stop -- the spec's finding was wrong and the whole premise needs re-checking before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_trajectory/test_no_ros_dependency.py
git commit -m "test(trajectory): pin the model to prorated actuals, never an ROS blend (#346)"
```

---

# Phase B: comp ids and the careers map

## Task 7: Comps carry their MLBAM id

**Files:**
- Modify: `scripts/push_trajectory_board.py` (`player_comps`, around line 94)
- Test: `tests/test_scripts/test_push_trajectory_board.py`

**Interfaces:**
- Consumes: `CompPath.mlbam_id` from `closest_paths`
- Produces: each stored comp dict gains `"id": int` -- the join key Task 8 writes against and Task 9 reads

- [ ] **Step 1: Write the failing test**

```python
def test_a_stored_comp_carries_the_id_its_career_is_keyed_on() -> None:
    """A comp used to be a display NAME and four numbers. Joining a chart to a name is
    the defect class #284 exists for -- two players share one normalized name and one
    of them gets the other's career drawn under his own."""
    prepared, player, horizons, names = _comp_fixture()  # existing helper in this file
    comps = player_comps(prepared, player, horizons, names)

    assert comps, "the fixture player has comps"
    assert all(isinstance(c["id"], int) for c in comps)
    assert all(c["id"] > 0 for c in comps)
```

Use whatever fixture the neighbouring `player_comps` tests already build; if none exists, build `prepared` with `prepare()` over `synthetic_panel()` the way `tests/test_trajectory/test_comp_paths.py` does.

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_scripts/test_push_trajectory_board.py -k "carries_the_id" -v`
Expected: FAIL with `KeyError: 'id'`

- [ ] **Step 3: Emit the id**

In `player_comps`'s returned dict, as the first key:

```python
        {
            # THE JOIN KEY for the stored career map, not decoration. `chart_key(id,
            # pool)` is how a card finds the arc it draws; a display name cannot do it
            # (#284) and two players share one normalized name often enough to matter.
            "id": c.mlbam_id,
            "name": names.get(c.mlbam_id, str(c.mlbam_id)),
            "season": c.season,
            "rmse": round(c.rmse, 3),
            "path": [round(v, 3) for v in c.path],
        }
```

- [ ] **Step 4: Run the tests**

Run: `pytest tests/test_scripts/test_push_trajectory_board.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add scripts/push_trajectory_board.py tests/test_scripts/test_push_trajectory_board.py
git commit -m "feat(trajectory): stored comps carry their mlbam id (#346)"
```

---

## Task 8: The deduped `careers` map

**Files:**
- Modify: `src/fantasy_baseball/trajectory/sweep.py` (`to_chart_payload`, around line 388)
- Modify: `scripts/push_trajectory_board.py` (the pool loop and the final `to_chart_payload` call)
- Test: `tests/test_trajectory/test_sweep.py`, `tests/test_scripts/test_push_trajectory_board.py`

**Interfaces:**
- Consumes: `player_comps(...)[i]["id"]` (Task 7); `by_id`, already built in the pool loop
- Produces: `chart_payload["careers"]: dict[str, list[list[float]]]` keyed by `chart_key(mlbam_id, pool)`, values ascending by age

- [ ] **Step 1: Write the failing serializer test**

In `tests/test_trajectory/test_sweep.py`:

```python
def test_the_chart_payload_carries_comp_careers_beside_the_players() -> None:
    """Comp careers are deduped at the TOP level, not nested under each comp: comps
    repeat heavily across the board's ~1,169 players, and nesting writes the same arc
    thousands of times. Keyed like `players`, because a two-way comp has one career
    per pool."""
    payload = to_chart_payload(
        {(1, "hitter"): {"history": [[24, 5.0]], "comps": []}},
        careers={chart_key(9, "hitter"): [[21, 3.0], [22, 7.5]]},
        generated_at="2026-08-07T12:00:00",
    )

    assert payload["careers"] == {"9:hitter": [[21, 3.0], [22, 7.5]]}
    assert payload["players"]["1:hitter"]["history"] == [[24, 5.0]]


def test_a_chart_payload_written_without_careers_still_has_the_key() -> None:
    """The reader treats a missing key and an empty map the same way, but a writer
    that omits it entirely makes every blob's shape depend on its vintage."""
    payload = to_chart_payload({}, generated_at="2026-08-07T12:00:00")
    assert payload["careers"] == {}
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_trajectory/test_sweep.py -k careers -v`
Expected: FAIL with `TypeError: to_chart_payload() got an unexpected keyword argument 'careers'`

- [ ] **Step 3: Carry `careers` through the serializer**

In `sweep.py`, change the signature and the returned dict:

```python
def to_chart_payload(
    extras: dict[tuple[int, str], dict],
    *,
    generated_at: str,
    careers: dict[str, list] | None = None,
) -> dict:
```

and in the returned dict, after `"players"`:

```python
        # COMP CAREERS, deduped across the whole board rather than nested under each
        # comp. The same arc is a comp for many players -- nesting writes it once per
        # (player, comp) pair, roughly 5x the entries for the same information. Keyed
        # by `chart_key` like `players`, so a two-way comp keeps one career per pool.
        "careers": dict(careers or {}),
```

`careers or {}` is safe here -- the value is a dict, not a number, and an empty dict and `None` mean the same thing to every reader. Add a line to the docstring saying so.

- [ ] **Step 4: Run the serializer test**

Run: `pytest tests/test_trajectory/test_sweep.py -k careers -v`
Expected: PASS

- [ ] **Step 5: Write the failing push-side test**

In `tests/test_scripts/test_push_trajectory_board.py`:

```python
def test_the_push_stores_one_career_per_comp_and_no_more() -> None:
    """Every id a stored comp names must resolve to an arc, or its card draws nothing;
    every id BEYOND that set is dead weight in a blob the player page fetches."""
    _, chart, _ = build_payloads(panel_dir=FIXTURE_PANEL_DIR, max_horizon=3)

    referenced = {
        f"{c['id']}:{key.split(':')[1]}"
        for key, block in chart["players"].items()
        for c in block["comps"]
    }
    assert referenced, "the fixture produces comps"
    assert set(chart["careers"]) == referenced

    for arc in chart["careers"].values():
        ages = [pt[0] for pt in arc]
        assert ages == sorted(ages), "an arc is drawn left to right without a re-sort"
        assert len(set(ages)) == len(ages), "one point per age; split seasons collapsed"
```

Use this file's existing helper for invoking the payload builder against a fixture panel; if it invokes the builder differently, follow that call and keep the assertions.

- [ ] **Step 6: Run it to verify it fails**

Run: `pytest tests/test_scripts/test_push_trajectory_board.py -k "one_career_per_comp" -v`
Expected: FAIL with `KeyError: 'careers'`

- [ ] **Step 7: Build the map in the push script**

Declare it beside `extras`, **outside** the pool loop:

```python
    # Deduped comp careers, keyed by `chart_key(id, pool)` -- see `to_chart_payload`.
    # Outside the pool loop like `extras`, because the key already carries the pool and
    # a two-way comp gets one entry per pool under distinct keys.
    careers: dict[str, list] = {}
```

Inside the pool loop, accumulate ids during the existing `for player in produced` loop by adding one line after `comps = player_comps(...)`:

```python
            if comps:
                wanted.update(c["id"] for c in comps)
```

with `wanted: set[int] = set()` declared just above that loop (per pool -- `by_id` is per pool, so an id must be resolved against its own pool's frame). Then, after the `for player in produced` loop closes:

```python
        # THE SAME `by_id` the subject's own history comes from, so a comp's arc and the
        # subject overlay drawn on top of it are on one scale by construction. An id
        # absent from it writes nothing: comps are drawn from the same `prepared` this
        # frame built, so absence is a defect, and an empty card is the honest rendering
        # of one rather than a fabricated arc.
        for comp_id in sorted(wanted):
            seasons = by_id.get(comp_id)
            if seasons is None:
                continue
            careers[chart_key(comp_id, kind)] = [
                [int(a), round(float(s), 4)]
                for a, s in sorted(
                    zip(seasons["age"], seasons["sgp"], strict=True), key=lambda pt: pt[0]
                )
            ]
```

Import `chart_key` alongside the existing `sweep` imports:

```python
    from fantasy_baseball.trajectory.sweep import (
        chart_key,
        sweep_pool,
        to_chart_payload,
        to_payload,
    )
```

Finally, pass it through at the return:

```python
    return payload, to_chart_payload(extras, careers=careers, generated_at=generated_at), len(swept)
```

- [ ] **Step 8: Run both suites**

Run: `pytest tests/test_scripts/test_push_trajectory_board.py tests/test_trajectory/test_sweep.py -v`
Expected: PASS

- [ ] **Step 9: Commit**

```bash
git add scripts/push_trajectory_board.py src/fantasy_baseball/trajectory/sweep.py tests/test_scripts/test_push_trajectory_board.py tests/test_trajectory/test_sweep.py
git commit -m "feat(trajectory): store a deduped comp-career map in the chart blob (#346)"
```

---

## Task 9: `PlayerView.comp_careers`

**Files:**
- Modify: `src/fantasy_baseball/web/trajectory_view.py` (`_chart_extras`, `PlayerView`, `build_player_view`)
- Test: `tests/test_web/test_trajectory_view.py`

**Interfaces:**
- Consumes: `chart["careers"]` (Task 8), `comps[i]["id"]` (Task 7)
- Produces: `PlayerView.comp_careers: list[dict]`, parallel to `comps`, entries `{name, season, rmse, match_age, career}`

- [ ] **Step 1: Write the failing tests**

Extend the module's `_chart()` helper to store ids and careers. Change its `comps` entries to include `"id": 100 + i` and add a `careers=` argument to the `to_chart_payload` call:

```python
        careers={
            chart_key(100 + i, pool): [[20 + j, float(i + j)] for j in range(6)]
            for i in range(1, 8)
            for pool in ("hitter", "pitcher")
        },
```

Then add:

```python
def test_every_comp_carries_the_whole_career_behind_it(payload: dict) -> None:
    """A comp drawn as five forward points says nothing about the comp. The card needs
    his whole arc."""
    view = _view(payload, player="Big Bat", n=3)

    assert len(view.comp_careers) == len(view.comps) == 3
    assert [c["name"] for c in view.comp_careers] == [c["name"] for c in view.comps]
    assert all(c["career"] for c in view.comp_careers), "every fixture comp has an arc"
    for entry in view.comp_careers:
        ages = [pt[0] for pt in entry["career"]]
        assert ages == sorted(ages)


def test_the_match_line_sits_at_the_subjects_own_age(payload: dict) -> None:
    """`closest_paths` selects on `prepared.age == float(age)` -- an EXACT match -- so
    every card marks the same age. If that ever becomes a tolerance window this test is
    what stops the line being drawn somewhere it does not belong."""
    view = _view(payload, player="Big Bat")
    assert view.comp_careers
    assert all(c["match_age"] == view.age for c in view.comp_careers)


def test_a_comp_career_is_netted_against_the_QUERY_players_floor(payload: dict) -> None:
    """Same rule the comp PATHS already follow: the card asks what this arc would be
    worth in the subject's slot, so per-comp floors would put non-comparable lines on
    one axis."""
    row = next(p for p in payload["players"] if p["name"] == "Big Bat")
    var = _view(payload, player="Big Bat", scale="var")
    sgp = _view(payload, player="Big Bat", scale="sgp")

    assert var.comp_careers[0]["career"][0][1] == pytest.approx(
        sgp.comp_careers[0]["career"][0][1] - row["floor"]
    )


def test_a_blob_with_no_careers_yields_empty_arcs_rather_than_raising(payload: dict) -> None:
    """Every deployed blob predates this feature. The page must render."""
    blob = _chart(payload)
    del blob["careers"]
    view = build_player_view(payload, player="Big Bat", chart=blob)

    assert view.comps, "the comps themselves still render"
    assert len(view.comp_careers) == len(view.comps)
    assert all(c["career"] == [] for c in view.comp_careers)


def test_a_careers_map_of_the_wrong_shape_is_refused_not_raised(payload: dict) -> None:
    """Refused rather than raised, for the reason #332 stands as: refusing a page over
    auxiliary data it could largely render is how /trajectory goes down."""
    blob = _chart(payload)
    blob["careers"] = [["not", "a", "mapping"]]
    view = build_player_view(payload, player="Big Bat", chart=blob)

    assert all(c["career"] == [] for c in view.comp_careers)


def test_a_comp_stored_without_an_id_gets_an_empty_arc(payload: dict) -> None:
    """An older push wrote comps with no `id`. That is a missing join key, not a crash."""
    blob = _chart(payload)
    for block in blob["players"].values():
        for comp in block["comps"]:
            comp.pop("id", None)
    view = build_player_view(payload, player="Big Bat", chart=blob)

    assert all(c["career"] == [] for c in view.comp_careers)


def test_a_two_way_comps_two_careers_do_not_collide(payload: dict) -> None:
    """One id, two pools, two different arcs. Keying on the bare id hands the hitter
    card the pitching career -- the same collapse `chart_key` exists to stop."""
    blob = _chart(payload)
    blob["careers"]["101:hitter"] = [[22, 1.0], [23, 2.0]]
    blob["careers"]["101:pitcher"] = [[22, 90.0], [23, 91.0]]

    hitter = build_player_view(payload, player="Big Bat", chart=blob)
    pitcher = build_player_view(payload, player="Big Arm", chart=blob)

    assert hitter.comp_careers[0]["career"] == [[22, 1.0 - 4.0], [23, 2.0 - 4.0]]
    assert pitcher.comp_careers[0]["career"][0][1] > 50
```

Adjust `"Big Arm"` to whatever the fixture's first pitcher is named, and the `- 4.0` to that hitter's `floor` -- read them off the `payload` fixture rather than hardcoding if the values differ.

- [ ] **Step 2: Run them to verify they fail**

Run: `pytest tests/test_web/test_trajectory_view.py -k "comp_career or match_line or two_way_comp" -v`
Expected: FAIL with `AttributeError: 'PlayerView' object has no attribute 'comp_careers'`

- [ ] **Step 3: Return `careers` from `_chart_extras`**

Change the signature and every `return` in `_chart_extras`:

```python
def _chart_extras(
    payload: dict, chart: Any, mlbam_id: int, pool: str
) -> tuple[dict, dict, bool]:
```

Returns become, in order: `({}, {}, False)` for `chart is None`; `({}, {}, True)` for each refusal; and at the end:

```python
    # A blob written before comp careers existed has no `careers` key, and one written
    # by a future/foreign writer may have something that is not a mapping. Both are
    # "no arcs", never an exception: the comps table and the whole rest of the page
    # render without them (#332).
    careers = chart.get("careers")
    return (
        players.get(chart_key(mlbam_id, pool), {}),
        careers if isinstance(careers, Mapping) else {},
        False,
    )
```

Update the docstring's "Returns `({}, True)`" sentence to describe the triple.

- [ ] **Step 4: Add the field to `PlayerView`**

After the `comps` field:

```python
    #: One entry per RENDERED comp, parallel to `comps` and in the same order:
    #: {name, season, rmse, match_age, career}. `career` is his whole arc, floor-netted
    #: like every other series, and is empty when the blob carries no entry for him.
    #:
    #: `name`/`season`/`rmse` are re-carried from `comps` rather than looked up by index
    #: on the other side: one list, one card, no zip -- a card indexing into a second
    #: array to title itself is one off-by-one from putting one man's name over
    #: another's arc.
    comp_careers: list[dict] = field(default_factory=list)
```

- [ ] **Step 5: Populate it in `build_player_view`**

Change the unpack to `extras, careers, mismatch = _chart_extras(...)`, then build the list from the SAME sliced comps the view already renders. Bind them once above the `replace(...)` call so the two cannot drift:

```python
    shown = extras.get("comps", [])[:want]
```

Use `shown` in the existing `comps=[...]` comprehension, and add:

```python
        comp_careers=[
            {
                "name": c["name"],
                "season": c["season"],
                "rmse": c["rmse"],
                # THE SUBJECT'S age, and correct for every card: `closest_paths` selects
                # on `prepared.age == float(age)`, an exact match, so a comp's age in his
                # match season IS this one. Stored per card rather than read off the view
                # so the card is self-contained, and pinned by a test in case the matcher
                # ever gains a tolerance window.
                "match_age": sp.age,
                # `c.get("id")` -- an older push wrote comps with no id at all, which is
                # a missing join key rather than an error. `chart_key` is never called
                # with None: the lookup short-circuits first.
                "career": [
                    [int(a), float(v) - floor]
                    for a, v in careers.get(chart_key(c["id"], sp.pool), [])
                ]
                if c.get("id") is not None
                else [],
            }
            for c in shown
        ],
```

Import `chart_key` from `fantasy_baseball.trajectory.sweep` -- it is already imported in this module's import block.

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_web/test_trajectory_view.py -v`
Expected: PASS, including every pre-existing test.

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/trajectory_view.py tests/test_web/test_trajectory_view.py
git commit -m "feat(trajectory): the view reads each comp's whole career (#346)"
```

---

# Phase C: the card grid

## Task 10: The grid markup and styling

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/trajectory_player.html`
- Modify: `src/fantasy_baseball/web/static/season.css` (append after the `.trajectory-search` block, around line 3923)
- Test: `tests/test_web/test_season_routes.py`

**Interfaces:**
- Consumes: `PlayerView.comp_careers` (Task 9)
- Produces: one `<canvas id="comp-chart-{{ loop.index0 }}">` per comp with an arc; `#trajectory-chart-data` gains `comp_careers`

- [ ] **Step 1: Write the failing test**

```python
def test_the_player_page_gives_every_comp_its_own_canvas(client):
    """A comp stacked on the subject's chart shows a forward path and nothing else.
    Each one gets its own card so his whole arc is readable."""
    payload = _trajectory_payload()
    with _patched_cache(payload):
        html = client.get("/trajectory?view=player&player=Big Bat&n=3").data.decode()

    assert html.count('id="comp-chart-') == 3
    assert '"comp_careers"' in html, "the island carries the arcs"
```

This test needs `_trajectory_payload()`'s chart blob to carry ids and careers. If that helper builds no chart blob at all, extend it the same way Task 9 extended `_chart()`; if the route test installs a chart key separately, extend that.

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py -k own_canvas -v`
Expected: FAIL -- count is 0.

- [ ] **Step 3: Put `comp_careers` on the island**

```jinja
<script type="application/json" id="trajectory-chart-data">{{ {
  "name": board.name, "axis_label": board.axis_label, "history": board.history,
  "paced": board.paced, "paced_label": board.paced_label,
  "projection": board.projection, "comps": board.comps,
  "comp_careers": board.comp_careers} | tojson }}</script>
```

- [ ] **Step 4: Add the grid, directly below the comps table**

Inside the existing `{% if board.comps %}` branch, after `</table>`:

```jinja
{#- ONE CARD PER COMP, because a comp stacked on the projection shows a forward path
    and nothing about the player it belongs to. The whole arc, with the match age
    marked, is what makes "Bogaerts at 25" mean something.

    The grid is skipped entirely when NO comp has a stored arc -- an old blob, or a
    push that wrote comps before careers existed. Ten identical notices in a ten-cell
    grid is noise about a single cause. -#}
{% if board.comp_careers | selectattr('career') | first is defined %}
<div class="comp-grid">
  {% for c in board.comp_careers %}
  <div class="comp-card">
    <h4>{{ c.name }} <span class="muted">{{ c.season }} -- RMSE {{ '%.2f'|format(c.rmse) }}</span></h4>
    {% if c.career %}
    <div class="comp-card-chart"><canvas id="comp-chart-{{ loop.index0 }}"></canvas></div>
    {% else %}
    <p class="muted">No stored career -- re-run <code>push_trajectory_board.py</code>.</p>
    {% endif %}
  </div>
  {% endfor %}
</div>
{% else %}
<p class="muted">
  No stored comp careers -- this board predates them. A re-push of
  <code>push_trajectory_board.py</code> will fill them in.
</p>
{% endif %}
```

- [ ] **Step 5: Style it**

Append to `season.css` after the `.trajectory-search button:hover` rule:

```css
/* Per-comp career cards (#346). Small multiples: each comp's whole arc with the
   match age marked, rather than a forward path stacked on the subject's chart.
   Theme vars only, so the grid follows light/dark like every other surface. */
.comp-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
  gap: 12px;
  margin: 12px 0 18px;
}
.comp-card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 10px 12px;
}
.comp-card h4 {
  margin: 0 0 6px;
  font-size: 13px;
  color: var(--ink);
}
.comp-card h4 .muted { font-size: 11px; font-weight: normal; }
/* Fixed height for the same reason .chart-wrapper has one: Chart.js's default
   maintainAspectRatio ignores the box and draws roughly twice its height. */
.comp-card-chart { position: relative; height: 180px; }
.comp-card-chart canvas { display: block; width: 100%; height: 100%; }
```

- [ ] **Step 6: Run the tests**

Run: `pytest tests/test_web/test_season_routes.py -k trajectory -v`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/trajectory_player.html src/fantasy_baseball/web/static/season.css tests/test_web/test_season_routes.py
git commit -m "feat(trajectory): a card per comp on the player page (#346)"
```

---

## Task 11: Draw the cards

**Files:**
- Modify: `src/fantasy_baseball/web/static/trajectory_chart.js`

**Interfaces:**
- Consumes: `data.comp_careers`, `data.history`, `data.paced`, `data.projection`, `data.axis_label` from the island; the canvases Task 10 emits
- Produces: nothing (leaf)

No new test: this is canvas rendering, which the Python suite cannot observe. Task 10 pins that the canvases and the data reach the page; Task 12 confirms the drawing by eye.

- [ ] **Step 1: Factor out the subject overlay**

Near the top of the IIFE in `trajectory_chart.js`, after `const at = ...`, add:

```js
  // The subject's own arc, as ONE series: realized seasons, the paced base season,
  // then the projected means. Built once and reused by every comp card -- the card
  // asks "how does this comp's shape compare to his", and that needs him on it.
  const subject = [
    ...data.history.map(([age, v]) => ({ x: age, y: v })),
    ...(data.paced ? [{ x: data.paced[0], y: data.paced[1] }] : []),
    ...at(data.projection, "mean"),
  ];
```

- [ ] **Step 2: Add the match-line plugin**

Below `subject`:

```js
  // A dashed vertical rule at the age where the comp matched the subject. Ten lines
  // of canvas rather than the chartjs-plugin-annotation CDN bundle: a second external
  // script is a second thing that can fail to load, and `ensureChartJs` would have to
  // grow a dependency-ordering concept to serve it.
  const matchLine = {
    id: "matchLine",
    afterDatasetsDraw(chart, _args, opts) {
      if (typeof opts.age !== "number") return;
      const x = chart.scales.x.getPixelForValue(opts.age);
      const { top, bottom, left, right } = chart.chartArea;
      if (x < left || x > right) return;
      const ctx = chart.ctx;
      ctx.save();
      ctx.setLineDash([4, 4]);
      ctx.strokeStyle = "rgba(120,120,120,0.8)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
      ctx.restore();
    },
  };
```

- [ ] **Step 3: Draw one chart per card**

Inside the existing `window.ensureChartJs().then(() => {...})` callback, after the main `new Chart(...)` call:

```js
    (data.comp_careers || []).forEach((comp, i) => {
      const el = document.getElementById(`comp-chart-${i}`);
      // Absent when this comp had no stored arc -- the template rendered a note in
      // place of the canvas. Skip it rather than treating it as a failure.
      if (!el || !comp.career || !comp.career.length) return;
      new Chart(el.getContext("2d"), {
        type: "line",
        data: {
          datasets: [
            {
              // The SUBJECT, faint and underneath. He is the reason this card is on
              // the page; without him the arc is just a stranger's career.
              label: data.name,
              data: subject,
              borderColor: "rgba(78,121,167,0.35)",
              borderWidth: 1,
              borderDash: [3, 3],
              pointRadius: 0,
              order: 1,
            },
            {
              label: comp.name,
              data: comp.career.map(([age, v]) => ({ x: age, y: v })),
              borderColor: "#59a14f",
              borderWidth: 2,
              pointRadius: 1.5,
              order: 0,
            },
          ],
        },
        plugins: [matchLine],
        options: {
          responsive: true,
          maintainAspectRatio: false,
          parsing: false,
          // X-DOMAIN: whatever Chart.js autoscales over BOTH series -- the comp's
          // career and the subject's drawn ages. Clipping to the comp's career alone
          // would cut off the subject's projection whenever the comp retired young,
          // which is the half of the card being compared. No domain is shared across
          // cards: forcing one would squeeze a 20-season career into a 6-season card
          // and shrink the region around the match line, which is the part being read.
          scales: {
            x: { type: "linear", title: { display: true, text: "age" } },
            y: { title: { display: false, text: data.axis_label } },
          },
          plugins: {
            legend: { display: false },
            matchLine: { age: comp.match_age },
          },
        },
      });
    });
```

- [ ] **Step 4: Verify no syntax error**

Run: `node --check src/fantasy_baseball/web/static/trajectory_chart.js`
Expected: no output (exit 0). If `node` is unavailable, skip -- Task 12 catches it in the browser console.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/static/trajectory_chart.js
git commit -m "feat(trajectory): draw each comp's career with the match age marked (#346)"
```

---

## Task 12: See it in the browser

**Files:** none

- [ ] **Step 1: Start the dashboard against the local cache**

Run: `python scripts/run_season_dashboard.py --no-sync`

`--no-sync` is required: a sync clobbers not-yet-deployed local state with prod's.

- [ ] **Step 2: Open the player view**

Open `http://localhost:5000/trajectory?view=player&player=CJ%20Abrams` (adjust the port to what the script prints, and the name to anyone on the cached board).

- [ ] **Step 3: Check each of these, and fix what fails**

- [ ] The career line runs into the base season with an open marker at its end, no gap before the dashed projection
- [ ] Hovering that point shows the `pace` label
- [ ] No "Solid is what happened", "on purpose", or "not evidence for the forecast" text anywhere on the page
- [ ] A grid of comp cards below the comps table, one per comp at the current `n`
- [ ] Each card shows a full career arc, longer than five points, with a dashed vertical line
- [ ] The vertical line sits at the same age on every card, and that age is the subject's
- [ ] The subject's faint dashed arc is visible under each comp's
- [ ] `?n=10` renders ten cards; `?n=1` renders one
- [ ] The browser console has no errors
- [ ] `/trajectory` and `/trajectory?view=teams` still render, minus their explainer sentences

**The cards will show "No stored comp careers" against the currently-cached blob** -- it was written before Task 8. That is the correct degradation, and it is what Task 13's checks and Task 14's deploy gate exist for. To see real arcs, rebuild the blob into a THROWAWAY local file rather than pushing:

Run: `python scripts/push_trajectory_board.py --dry-run` if that flag exists; otherwise read the script's `main()` for a local-output flag. **Do not run the plain command** -- it writes to prod Upstash.

- [ ] **Step 4: Commit any fixes**

```bash
git commit -am "fix(trajectory): <what the browser showed>"
```

---

## Task 13: End-of-effort verification

**Files:** whatever the checks flag

- [ ] **Step 1: Full suite**

Run: `pytest -n auto -q`
Expected: all pass. Re-run any failure without `-n auto` to read it cleanly.

- [ ] **Step 2: Lint**

Run: `ruff check .`
Expected: zero violations.

- [ ] **Step 3: Format**

Run: `ruff format --check .`
Expected: no drift. Run `ruff format .` and re-check if it reports any.

- [ ] **Step 4: Dead code**

Run: `vulture`
Expected: no NEW findings from these changes. Note any pre-existing ones in the final report rather than fixing them.

- [ ] **Step 5: Types, if in scope**

Read `[tool.mypy].files` in `pyproject.toml`. If `trajectory_view.py`, `sweep.py`, or `comp_paths.py` is listed, run `mypy` and fix what it reports for those files.

- [ ] **Step 6: Amend `comp_paths.py`'s docstring**

The module docstring says a consumer drawing comp paths "without the p10-p90 band beside them is making the forecast look more certain than it is." The card grid is such a consumer, deliberately. Amend that paragraph so the code does not read as violating a standing instruction (spec, Requirement 3):

```python
THE RESULT IS SELECTED ON THE OUTCOME. These are the paths that happened to land closest
out of ~1,200. That makes them a fair illustration of what this shape looked like when it
played out, and it makes them NOT evidence for the prediction. Any surface presenting
them AS the forecast's range must draw the p10-p90 band beside them, or it makes the
forecast look more certain than it is -- which is why the main chart on /trajectory
carries the band and the comps are thin and faint on it. The per-comp career cards below
it (#346) are a different question -- what this player's whole arc looked like, and where
the match sits in it -- and carry no band because they make no claim about the spread.
```

- [ ] **Step 7: Commit and report**

```bash
git add -A
git commit -m "docs(trajectory): comp_paths names which surfaces owe the band (#346)"
```

Report every command run and what it returned. Never claim the checks pass without showing them.

---

## Task 14: Ship it, in this order

**Files:** none

- [ ] **Step 1: Open the PR**

```bash
git push -u origin feat/346-comp-career-charts
gh pr create --fill --base main
```

- [ ] **Step 2: Merge after review**

- [ ] **Step 3: Wait for Render to finish deploying that commit**

- [ ] **Step 4: Confirm prod renders against the OLD blob**

Load `/trajectory?view=player&player=<anyone>` on prod. Expected: the page renders, the paced point is there, and the comp grid shows "No stored comp careers". That is the reader tolerating a pre-`careers` blob, which is the whole reason it was built that way.

- [ ] **Step 5: Only now, push the new blob**

```bash
python scripts/push_trajectory_board.py
```

**A local push writes to the same prod Upstash the live Render build reads.** Running this before step 4 passes is what broke `/trajectory` on 2026-08-06. Nothing earlier in this plan may run it.

- [ ] **Step 6: Confirm prod renders against the NEW blob**

Reload the same URL. Expected: real comp career arcs in the grid.

- [ ] **Step 7: Close the issue**

```bash
gh issue close 346 --comment "Shipped in <PR link>."
```

---

## Self-Review

**Spec coverage:**

| Spec requirement | Task |
|---|---|
| R1 `paced` field, floor-netted | 3 |
| R1 suppression rule (offseason) | 3 |
| R1 `base_season_partial` + `paced_label` | 2, 3 |
| R1 open-marker rendering, unbroken line | 4 |
| R1 numbers-table `pace` row | 4 |
| R1 island gains `paced`/`paced_label`/`comp_careers` | 4, 10 |
| R2 boundary test, four named surfaces | 6 |
| R2 finding referenced from `paced`'s docstring | 3 (Step 3) |
| R3 five clause deletions, paragraphs preserved | 5 |
| R3 `comp_paths.py` docstring amendment | 13 (Step 6) |
| R4 comp `id` | 7 |
| R4 deduped `careers` map, per-pool keys | 8 |
| R4 `comp_careers` + `match_age` + floor-netting | 9 |
| R4 grid, empty-career note, all-empty collapse | 10 |
| R4 cards, subject overlay, match line, x-domain | 11 |
| Failure-mode table (12 rows) | 3, 9, 10 tests |
| Deploy gate | 14 |
| End-of-effort checks | 13 |

No spec requirement is unassigned.

**Placeholder scan:** clean. Every code step carries the code. Four steps say "adjust to what the existing helper is named" -- those are instructions to read one named neighbouring function, not deferred decisions, and each names the function to read.

**Type consistency:** `chart_key(mlbam_id, pool) -> str` used identically in Tasks 8, 9, and their tests. `to_chart_payload(extras, *, generated_at, careers=None)` -- the keyword is `careers` in Task 8's definition, its two tests, and the push call. `PlayerView.paced` is `list[float] | None` in Task 3 and is read as `board.paced[0]`/`[1]` in Task 4 and `data.paced[0]`/`[1]` in Tasks 4 and 11. `comp_careers` entries carry `{name, season, rmse, match_age, career}` in Task 9 and every one of those keys is read in Tasks 10 and 11 (`c.name`, `c.season`, `c.rmse`, `comp.match_age`, `comp.career`). `_chart_extras` returns a 3-tuple in Task 9 and is unpacked as one at its single call site.
