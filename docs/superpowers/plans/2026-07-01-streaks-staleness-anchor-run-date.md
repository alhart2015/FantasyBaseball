# Streaks Staleness Anchor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the /streaks page and lineup chip from showing a frozen hot/cold label for a player who hasn't played recently (e.g. an IL player), by anchoring the "current streak" on the run date.

**Architecture:** Add a staleness anchor in `inference.score_player_windows`: a player whose most-recent scoreable window ends more than `STALE_TOLERANCE_DAYS` (=4) before the run date is forced to all-neutral category scores (kept, not skipped). The report layer (`reports/sunday.py`) computes a player-level `days_since_last_game` from the same `window_end` + run date and carries it through the serialized payload so the markdown/terminal reports, the /streaks web template, and the lineup chip can annotate "inactive - N days". Forcing `label="neutral"` collapses `_composite` to 0, which makes every downstream tone go neutral automatically.

**Tech Stack:** Python 3.11, DuckDB, pandas, Flask + Jinja2, pytest.

**Spec:** `docs/superpowers/specs/2026-06-30-streaks-staleness-anchor-run-date-design.md`

## Global Constraints

- **ASCII-only** in all new code, strings, and rendered markers. Use a hyphen `-`, not an en/em dash. (Pre-existing non-ASCII glyphs in `sunday.py`/`indicator.py` are out of scope — do not touch them.)
- **Tolerance constant:** `STALE_TOLERANCE_DAYS = 4`, defined once in `streaks/inference.py` and imported by `reports/sunday.py`. Boundary is strict `>` (exactly 4 days stale is still current).
- **`indicator.py` must stay duckdb-free** — it is a pure dict-payload consumer (Render never imports duckdb). Only read the new field from the row dict.
- **Run tests with:** `python -m pytest <path> -v` from repo root `C:/Users/alden/FantasyBaseball`.
- **Verification gates (run before declaring done):** `python -m pytest tests/test_streaks/ tests/test_web/ -q`, `ruff check .`, `ruff format --check .`, `python -m mypy src/fantasy_baseball/streaks/` (the streaks package is under mypy coverage).

---

### Task 1: Staleness anchor in `score_player_windows`

**Files:**
- Modify: `src/fantasy_baseball/streaks/inference.py` (add constant near `REPORT_CATEGORIES` ~line 63; edit `score_player_windows` ~lines 698-762)
- Test: `tests/test_streaks/test_inference.py`

**Interfaces:**
- Produces: `STALE_TOLERANCE_DAYS: int = 4` (module constant). `score_player_windows(..., stale_after_days: int | None = STALE_TOLERANCE_DAYS)` — new keyword-only param; `None` disables staleness forcing. When a player's `(window_end_on_or_before - window_end).days > stale_after_days`, every `PlayerCategoryScore` for that player is returned with `label="neutral"`, `probability=None`, `drivers=()`, and `window_end` preserved.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_streaks/test_inference.py` (it already imports `date`, `timedelta`, `score_player_windows`, `load_models_from_fits`, `REPORT_CATEGORIES`, and the `seeded_fitted_conn` fixture).

`_stale_target` deliberately selects a player whose **latest** 14d window carries a *non-neutral* label in the partitions `score_player_windows` reads. This makes the forcing test a valid TDD red-first: before the change, scoring that player at a far run_date yields his non-neutral label (so `all label=="neutral"` FAILS); after the change he is forced neutral (PASS). The `assert row is not None` fails loudly if the fixture ever stops producing a non-neutral latest window, rather than letting the test go vacuously green.

```python
def _stale_target(conn):
    """(player_id, latest window_end) for a player whose latest 14d window has a
    non-neutral label in the scored partitions -- so forcing him neutral is observable."""
    row = conn.execute(
        """
        WITH latest AS (
            SELECT player_id, MAX(window_end) AS we
            FROM hitter_windows WHERE window_days = 14 GROUP BY player_id
        )
        SELECT latest.player_id, latest.we
        FROM latest
        JOIN hitter_streak_labels l
          ON l.player_id = latest.player_id
         AND l.window_end = latest.we
         AND l.window_days = 14
        WHERE l.label != 'neutral'
          AND ((l.category IN ('r', 'rbi', 'avg') AND l.cold_method = 'empirical')
               OR (l.category IN ('hr', 'sb') AND l.cold_method = 'poisson_p20'))
        LIMIT 1
        """
    ).fetchone()
    assert row is not None, "fixture has no player with a non-neutral latest window"
    pid, we = int(row[0]), row[1]
    return pid, we if isinstance(we, date) else date.fromisoformat(str(we))


def test_score_player_windows_forces_neutral_when_window_is_stale(seeded_fitted_conn) -> None:
    conn = seeded_fitted_conn
    models = load_models_from_fits(conn)
    pid, end = _stale_target(conn)
    kw = dict(
        models=models,
        player_ids=[pid],
        window_end_on_or_before=end + timedelta(days=10),  # 10 > 4 -> stale
        window_days=14,
        scoring_season=2024,
    )

    # Baseline with staleness disabled: the stale window still scores its DB
    # labels, so at least one category is non-neutral (guaranteed by _stale_target).
    raw, _ = score_player_windows(conn, stale_after_days=None, **kw)
    assert any(s.label != "neutral" for s in raw), "target should have a live streak"

    # Default (staleness on) forces every category neutral for the same player.
    forced, skips = score_player_windows(conn, **kw)
    assert not skips  # still has a window; not a no_window skip
    assert len(forced) == len(REPORT_CATEGORIES)
    for s in forced:
        assert s.label == "neutral"
        assert s.probability is None
        assert s.drivers == ()
        assert s.window_end == end  # window_end preserved for the day-count


def test_score_player_windows_stale_after_days_none_disables_forcing(seeded_fitted_conn) -> None:
    conn = seeded_fitted_conn
    models = load_models_from_fits(conn)
    pid, end = _stale_target(conn)

    # Active baseline: run_date 1 day after window end -> days_since=1, not stale.
    active, _ = score_player_windows(
        conn, models=models, player_ids=[pid],
        window_end_on_or_before=end + timedelta(days=1),
        window_days=14, scoring_season=2024,
    )
    # Far run_date but staleness disabled -> same labels as the active baseline
    # (the same window is selected; nothing is forced).
    disabled, _ = score_player_windows(
        conn, models=models, player_ids=[pid],
        window_end_on_or_before=end + timedelta(days=10),
        window_days=14, scoring_season=2024, stale_after_days=None,
    )
    active_labels = {s.category: s.label for s in active}
    disabled_labels = {s.category: s.label for s in disabled}
    assert disabled_labels == active_labels
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_streaks/test_inference.py::test_score_player_windows_forces_neutral_when_window_is_stale tests/test_streaks/test_inference.py::test_score_player_windows_stale_after_days_none_disables_forcing -v`
Expected: FAIL — both tests call `score_player_windows(..., stale_after_days=...)`, which is an unknown keyword before Step 4, so both raise `TypeError: score_player_windows() got an unexpected keyword argument 'stale_after_days'`. (After Step 4 the forcing behavior is what turns the first test green.)

- [ ] **Step 3: Add the constant**

In `src/fantasy_baseball/streaks/inference.py`, just after the `REPORT_CATEGORIES` definition (~line 63), add:

```python
# A player whose most-recent scoreable window ends more than this many days
# before the run date has no *current* streak: his hot/cold label is frozen at
# his last game (see the staleness spec). Such a player is forced neutral in
# score_player_windows. Absorbs Statcast's 1-2 day publication lag plus normal
# off-days; catches any multi-week IL stint. Strict >: exactly 4 days is current.
STALE_TOLERANCE_DAYS: int = 4
```

- [ ] **Step 4: Add the param and forcing logic**

In `score_player_windows`, add the keyword-only parameter (after `scoring_season`):

```python
def score_player_windows(
    conn: duckdb.DuckDBPyConnection,
    *,
    models: dict[tuple[StreakCategory, StreakDirection], FittedModel],
    player_ids: Iterable[int],
    window_end_on_or_before: date,
    window_days: int = 14,
    scoring_season: int,
    stale_after_days: int | None = STALE_TOLERANCE_DAYS,
) -> tuple[list[PlayerCategoryScore], list[ScoreSkip]]:
```

Then change the per-player loop body (currently lines 740-760) to force neutral when stale:

```python
    for player_id in unique_ids:
        window = windows.get(player_id)
        if window is None:
            skips.append(ScoreSkip(player_id=player_id, reason="no_window"))
            continue
        window_end = pd.Timestamp(window["window_end"]).date()
        is_stale = (
            stale_after_days is not None
            and (window_end_on_or_before - window_end).days > stale_after_days
        )
        peripherals_null = any(pd.isna(window[c]) for c in _PERIPHERAL_COLS)

        for category in REPORT_CATEGORIES:
            if is_stale:
                # No current streak: the window is frozen at the player's last
                # game. Emit a neutral score (kept, not skipped) so the report
                # grid stays uniform and the day-count stays recoverable.
                scores.append(
                    PlayerCategoryScore(
                        player_id=player_id,
                        category=category,
                        label="neutral",
                        probability=None,
                        drivers=(),
                        window_end=window_end,
                    )
                )
                continue
            label = labels.get((player_id, category), "neutral")
            score = _score_one(
                player_id=player_id,
                category=category,
                label=label,
                window=window,
                window_end=window_end,
                peripherals_null=peripherals_null,
                models=models,
                season_rate=rates.get((player_id, category)),
            )
            scores.append(score)
```

Also update the `score_player_windows` docstring to mention `stale_after_days` (one sentence): "Players whose most-recent window ends more than `stale_after_days` before `window_end_on_or_before` are forced to neutral scores (pass `None` to disable)."

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_streaks/test_inference.py -v`
Expected: PASS (all tests in the file, including the two new ones and the pre-existing `test_score_player_windows_*` ones — those anchor `window_end_on_or_before = scoring_end + 1 day` so `days_since=1` and are unaffected).

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/streaks/inference.py tests/test_streaks/test_inference.py
git commit -m "feat(streaks): force neutral scores for stale windows (run-date anchor)"
```

---

### Task 2: `ReportRow.days_since_last_game` + `_row_from_scores(today)` + `build_report` threading

**Files:**
- Modify: `src/fantasy_baseball/streaks/reports/sunday.py` (`ReportRow` ~lines 67-90; `_row_from_scores` ~lines 218-233; `build_report` ~lines 309-314 and 327-332; import line)
- Test: `tests/test_streaks/test_sunday_report.py`

**Interfaces:**
- Consumes: `STALE_TOLERANCE_DAYS` from `streaks.inference`.
- Produces: `ReportRow.days_since_last_game: int | None` (default `None`; last field). `_row_from_scores(*, name, positions, player_id, scores, today: date)` — new required `today` param; sets `days_since_last_game = (today - window_end).days` when that exceeds `STALE_TOLERANCE_DAYS`, else `None`.

- [ ] **Step 1: Write the failing test**

Imports first. `test_sunday_report.py` already imports `date`, `Driver`,
`PlayerCategoryScore`, `Report`, `ReportRow`, `render_markdown`, `render_terminal`.
Do NOT re-import those (ruff `F811` redefinition fails `ruff check`). Make only
these two edits to the existing import blocks:
- change `from datetime import date` to `from datetime import date, timedelta`;
- add `_row_from_scores` to the existing
  `from fantasy_baseball.streaks.reports.sunday import (...)` block.

Then add these test helpers + tests:

```python
def _neutral_scores(window_end: date) -> list[PlayerCategoryScore]:
    return [
        PlayerCategoryScore(
            player_id=1, category=cat, label="neutral",
            probability=None, drivers=(), window_end=window_end,
        )
        for cat in ("hr", "r", "rbi", "sb", "avg")
    ]


def test_row_from_scores_marks_inactive_when_window_is_stale() -> None:
    window_end = date(2026, 6, 1)
    today = window_end + timedelta(days=10)  # 10 > 4
    row = _row_from_scores(
        name="Oneil Cruz", positions=("SS",), player_id=1,
        scores=_neutral_scores(window_end), today=today,
    )
    assert row.days_since_last_game == 10


def test_row_from_scores_active_when_window_is_recent() -> None:
    window_end = date(2026, 6, 1)
    today = window_end + timedelta(days=2)  # 2 <= 4
    row = _row_from_scores(
        name="Active Guy", positions=("OF",), player_id=1,
        scores=_neutral_scores(window_end), today=today,
    )
    assert row.days_since_last_game is None
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_streaks/test_sunday_report.py::test_row_from_scores_marks_inactive_when_window_is_stale tests/test_streaks/test_sunday_report.py::test_row_from_scores_active_when_window_is_recent -v`
Expected: FAIL — `_row_from_scores() got an unexpected keyword argument 'today'`.

- [ ] **Step 3: Add the field and the import**

In `src/fantasy_baseball/streaks/reports/sunday.py`, add `STALE_TOLERANCE_DAYS`
to the existing `from fantasy_baseball.streaks.inference import (...)` block.

Then add ONE field to `ReportRow`. Do **not** rewrite the class — it has two
`@property` methods (`sort_key`/`fa_sort_key`, ~lines 82-90) that are load-bearing
in `build_report`'s sort; leave them and the existing docstring intact. Insert
this single line immediately after `max_probability: float` (the last data field,
~line 80) and BEFORE the first `@property` (it is the only defaulted field, so it
must be last among the fields):

```python
    days_since_last_game: int | None = None
```

Also append one sentence to the `ReportRow` docstring: `` ``days_since_last_game``
is set (an int) only when the player's most-recent window is stale (>
STALE_TOLERANCE_DAYS before the run date) -- i.e. he is inactive; ``None`` means
active/recent.``

- [ ] **Step 4: Add `today` to `_row_from_scores`**

```python
def _row_from_scores(
    *,
    name: str,
    positions: tuple[str, ...],
    player_id: int,
    scores: list[PlayerCategoryScore],
    today: date,
) -> ReportRow:
    by_cat = {s.category: s for s in scores}
    # window_end is identical across a player's category scores (stamped once
    # per player in score_player_windows), so any score carries it.
    window_end = scores[0].window_end if scores else None
    days_since_last_game: int | None = None
    if window_end is not None:
        days = (today - window_end).days
        if days > STALE_TOLERANCE_DAYS:
            days_since_last_game = days
    return ReportRow(
        name=name,
        positions=positions,
        player_id=player_id,
        composite=_composite(scores),
        scores=by_cat,
        max_probability=_max_probability(scores),
        days_since_last_game=days_since_last_game,
    )
```

- [ ] **Step 5: Thread `today` in `build_report`**

In `build_report`, both `_row_from_scores(...)` calls (the roster loop ~line 309 and the FA loop ~line 327) must pass `today=today`:

```python
        roster_rows.append(
            _row_from_scores(
                name=hitter.name,
                positions=tuple(hitter.positions),
                player_id=mlbam,
                scores=scores,
                today=today,
            )
        )
```

and likewise:

```python
        row = _row_from_scores(
            name=hitter.name,
            positions=tuple(hitter.positions),
            player_id=mlbam,
            scores=scores,
            today=today,
        )
```

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_streaks/test_sunday_report.py -v`
Expected: PASS. In particular `test_build_report_end_to_end_against_seeded_db` still passes: it anchors `today = latest_end` (global `MAX(window_end)`), inactive players are kept (not dropped) and inactive FAs collapse to composite 0 and drop; its assertions (`len(report.roster_rows) == 2`, `len(fa_rows) <= len(fas)`) are robust. If a player intended to be active flips to neutral, the fixture windows are more than 4 days apart — do not weaken the assertion; instead verify the fixture seeds windows within 4 days of `latest_end`.

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/streaks/reports/sunday.py tests/test_streaks/test_sunday_report.py
git commit -m "feat(streaks): ReportRow.days_since_last_game + _row_from_scores(today)"
```

---

### Task 3: Serialize / deserialize `days_since_last_game` in the payload

**Files:**
- Modify: `src/fantasy_baseball/streaks/dashboard.py` (`_serialize_row` ~lines 72-80; `_deserialize_row` ~lines 83-94; module docstring ~lines 14-17)
- Test: `tests/test_streaks/test_dashboard.py`

**Interfaces:**
- Consumes: `ReportRow.days_since_last_game` (Task 2).
- Produces: the serialized row dict carries `"days_since_last_game"` (int or null); `_deserialize_row` reads it with `.get(...)` (tolerant of pre-existing cached payloads that lack the key).

- [ ] **Step 1: Write the failing test**

Add to `tests/test_streaks/test_dashboard.py` (imports `date`, `serialize_report`, `deserialize_report`, `Report`, `ReportRow`, `PlayerCategoryScore`, `Driver` are already present):

```python
def _example_report_inactive() -> Report:
    """A report whose single roster row is inactive (days_since_last_game set)."""
    score = PlayerCategoryScore(
        player_id=592518, category="hr", label="neutral",
        probability=None, drivers=(), window_end=date(2026, 5, 1),
    )
    row = ReportRow(
        name="Injured Guy", positions=("1B",), player_id=592518,
        composite=0, scores={"hr": score}, max_probability=0.0,
        days_since_last_game=30,  # MUST be set explicitly; it is a stored field
    )
    return Report(
        report_date=date(2026, 5, 31), window_end=date(2026, 5, 1),
        team_name="Hart of the Order", league_id=5652,
        season_set_train="2023-2025", roster_rows=(row,), fa_rows=(),
        driver_lines=(), skipped=(),
    )


def test_serialize_report_round_trips_inactive_row() -> None:
    original = _example_report_inactive()
    payload = serialize_report(original)
    assert payload["roster_rows"][0]["days_since_last_game"] == 30
    rebuilt = deserialize_report(payload)
    assert rebuilt == original
    assert rebuilt.roster_rows[0].days_since_last_game == 30
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_streaks/test_dashboard.py::test_serialize_report_round_trips_inactive_row -v`
Expected: FAIL — `KeyError: 'days_since_last_game'` on the payload assertion (serialize does not emit it yet), or an equality mismatch (`rebuilt.days_since_last_game` is `None` because `_deserialize_row` drops it).

- [ ] **Step 3: Emit the field in `_serialize_row`**

```python
def _serialize_row(r: ReportRow) -> dict[str, Any]:
    return {
        "name": r.name,
        "positions": list(r.positions),
        "player_id": r.player_id,
        "composite": r.composite,
        "max_probability": r.max_probability,
        "scores": {cat: _serialize_score(score) for cat, score in r.scores.items()},
        "days_since_last_game": r.days_since_last_game,
    }
```

- [ ] **Step 4: Read the field in `_deserialize_row` (tolerant)**

```python
def _deserialize_row(p: dict[str, Any]) -> ReportRow:
    return ReportRow(
        name=str(p["name"]),
        positions=tuple(p["positions"]),
        player_id=int(p["player_id"]),
        composite=int(p["composite"]),
        max_probability=float(p["max_probability"]),
        scores={
            cast(StreakCategory, cat): _deserialize_score(score)
            for cat, score in p["scores"].items()
        },
        days_since_last_game=p.get("days_since_last_game"),
    )
```

- [ ] **Step 5: Update the module docstring**

In `dashboard.py`, the module docstring (~lines 14-17) claims the schema mirrors the dataclass fields 1:1 with round-trip equality. Add a sentence noting the tolerant read: "``days_since_last_game`` is read with ``.get()`` so payloads cached before that field existed deserialize to ``None`` (the field default) rather than raising."

- [ ] **Step 6: Run the tests to verify they pass**

Run: `python -m pytest tests/test_streaks/test_dashboard.py -v`
Expected: PASS (the new test and the pre-existing `test_serialize_report_round_trips`).

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/streaks/dashboard.py tests/test_streaks/test_dashboard.py
git commit -m "feat(streaks): serialize days_since_last_game in the streak payload"
```

---

### Task 4: Inactive marker in the markdown + terminal renderers

**Files:**
- Modify: `src/fantasy_baseball/streaks/reports/sunday.py` (add `_name_cell` helper; use it in `_roster_table_markdown` ~line 416 and `render_terminal` ~line 596)
- Test: `tests/test_streaks/test_sunday_report.py`

**Interfaces:**
- Consumes: `ReportRow.days_since_last_game`.
- Produces: `_name_cell(row: ReportRow) -> str` — the player-name cell text, with `" (inactive - N days)"` appended when inactive.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_streaks/test_sunday_report.py`. `render_markdown`,
`render_terminal`, and `Report` are already imported (do NOT re-import — `F811`).
This test reuses `_neutral_scores` and `_row_from_scores` from Task 2 (same file):

```python
def _report_with_inactive_roster_row() -> Report:
    row = _row_from_scores(
        name="Oneil Cruz", positions=("SS",), player_id=1,
        scores=_neutral_scores(date(2026, 6, 1)),
        today=date(2026, 6, 1) + timedelta(days=30),
    )
    return Report(
        report_date=date(2026, 7, 1), window_end=date(2026, 6, 1),
        team_name="Hart of the Order", league_id=5652,
        season_set_train="2023-2025", roster_rows=(row,), fa_rows=(),
        driver_lines=(), skipped=(),
    )


def test_render_markdown_shows_inactive_marker() -> None:
    md = render_markdown(_report_with_inactive_roster_row())
    assert "Oneil Cruz (inactive - 30 days)" in md


def test_render_terminal_shows_inactive_marker() -> None:
    txt = render_terminal(_report_with_inactive_roster_row(), no_color=True)
    assert "Oneil Cruz (inactive - 30 days)" in txt
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m pytest tests/test_streaks/test_sunday_report.py::test_render_markdown_shows_inactive_marker tests/test_streaks/test_sunday_report.py::test_render_terminal_shows_inactive_marker -v`
Expected: FAIL — the marker string is not present (renderers emit only `row.name`).

- [ ] **Step 3: Add the `_name_cell` helper**

In `sunday.py`, near `_format_positions` (~line 398):

```python
def _name_cell(row: ReportRow) -> str:
    """Player-name cell, with an inactive marker appended when the player's
    most-recent window is stale (see ReportRow.days_since_last_game). Appended
    into the name cell rather than added as a column so the fixed header /
    separator / column-width layout of both renderers is untouched. ASCII only.
    """
    if row.days_since_last_game is not None:
        return f"{row.name} (inactive - {row.days_since_last_game} days)"
    return row.name
```

- [ ] **Step 4: Use it in both renderers**

In `_roster_table_markdown`, change `cells[0]` from `row.name` to `_name_cell(row)`:

```python
        cells = [
            _name_cell(row),
            _format_positions(row.positions),
            _signed(row.composite),
        ]
```

In `render_terminal`, change `row[0]` from `r.name` to `_name_cell(r)`:

```python
            row = [
                _name_cell(r),
                _format_positions(r.positions),
                _signed(r.composite),
            ]
```

(The FA table `_fa_table_markdown` needs no change: inactive FAs collapse to composite 0 and are dropped from `fa_rows` in `build_report`, so no FA row is ever inactive.)

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest tests/test_streaks/test_sunday_report.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/streaks/reports/sunday.py tests/test_streaks/test_sunday_report.py
git commit -m "feat(streaks): inactive marker in markdown + terminal reports"
```

---

### Task 5: Inactive tooltip on the lineup chip

**Files:**
- Modify: `src/fantasy_baseball/streaks/indicator.py` (`build_indicator` neutral `composite == 0` branch ~lines 88-93)
- Test: `tests/test_streaks/test_dashboard.py` (indicator tests live here)

**Interfaces:**
- Consumes: the serialized row's `days_since_last_game` (Task 3).
- Produces: for an inactive row (`composite == 0` and `days_since_last_game` set), the neutral `Indicator.tooltip` reads `"Inactive - N days"`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_streaks/test_dashboard.py` (uses `serialize_report`, `build_indicator`; reuse `_example_report_inactive` from Task 3):

```python
def test_build_indicator_inactive_row_shows_days_tooltip() -> None:
    payload = serialize_report(_example_report_inactive())
    ind = build_indicator("Injured Guy", payload)
    assert ind is not None
    assert ind.tone == "neutral"
    assert ind.tooltip == "Inactive - 30 days"
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest tests/test_streaks/test_dashboard.py::test_build_indicator_inactive_row_shows_days_tooltip -v`
Expected: FAIL — tooltip is the generic `"composite=0 (no active streaks)"`.

- [ ] **Step 3: Enhance the neutral branch**

In `indicator.py`, replace the `composite == 0` branch (currently returns the generic tooltip):

```python
    else:
        days = row.get("days_since_last_game")
        if days is not None:
            tooltip = f"Inactive - {days} days"
        else:
            tooltip = "composite=0 (no active streaks)"
        return Indicator(tone="neutral", label="—", tooltip=tooltip)
```

(Written as an if/else block, not a one-line ternary, so it stays under the
100-char line limit and passes `ruff format --check`. Leave the `label="—"`
as-is; it is pre-existing non-ASCII and out of scope. Only the new
`"Inactive - N days"` string is added, and it is ASCII.)

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest tests/test_streaks/test_dashboard.py -v`
Expected: PASS (new test + the existing indicator tests, whose active-player fixtures have no `days_since_last_game`).

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/streaks/indicator.py tests/test_streaks/test_dashboard.py
git commit -m "feat(streaks): inactive tooltip on the lineup chip"
```

---

### Task 6: Inactive marker on the /streaks web page + snapshot

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/_streaks_row.html` (name cell, line 6)
- Test/regen: `tests/test_web/test_streaks_snapshot.py`, `tests/test_web/snapshots/streaks.html`

**Interfaces:**
- Consumes: the serialized row's `days_since_last_game` (Task 3), reachable in the Jinja row context via `{% include %}` from `streaks.html`.

- [ ] **Step 1: Add the guarded marker to the name cell**

In `_streaks_row.html`, replace line 6 (`<td>{{ row.name }}</td>`) with:

```html
  <td>{{ row.name }}{% if row.days_since_last_game %} <span class="streak-inactive">(inactive - {{ row.days_since_last_game }} days)</span>{% endif %}</td>
```

(The guard `{% if row.days_since_last_game %}` is falsy for `null`/absent — so active rows render exactly as before, and payloads cached before this field deserialize fine.)

- [ ] **Step 2: Confirm the snapshot is unchanged (active fixture)**

Run: `python -m pytest tests/test_web/test_streaks_snapshot.py -v`
Expected: PASS. The snapshot's canonical row is a hand-built `ReportRow(...)`
(`_seed_canonical_report`) that never sets `days_since_last_game`, so it is `None`
by default (the only thing that computes it, `_row_from_scores`, is not on this
path). `None` -> the `{% if %}` guard is falsy -> the marker renders nothing ->
the HTML does not drift.

If instead it FAILS with drift (should not happen for the active fixture, but if template whitespace shifts), regenerate and eyeball:

```bash
rm tests/test_web/snapshots/streaks.html
python -m pytest tests/test_web/test_streaks_snapshot.py -v   # recreates snapshot, SKIPs
git diff --stat   # confirm only the snapshot changed
python -m pytest tests/test_web/test_streaks_snapshot.py -v   # now PASSES
```

Only accept a regenerated snapshot whose diff shows no unexpected structural change.

- [ ] **Step 3: Add a stale-row template test**

Add to `tests/test_web/test_streaks_snapshot.py` a focused test that an inactive row renders the marker (independent of the golden snapshot):

```python
def test_streaks_page_shows_inactive_marker(client, kv_isolation) -> None:
    score = PlayerCategoryScore(
        player_id=2, category="hr", label="neutral",
        probability=None, drivers=(), window_end=date(2026, 5, 1),
    )
    row = ReportRow(
        name="Injured Guy", positions=("1B",), player_id=2,
        composite=0, scores={"hr": score}, max_probability=0.0,
        days_since_last_game=30,
    )
    rpt = Report(
        report_date=date(2026, 5, 31), window_end=date(2026, 5, 1),
        team_name="Hart of the Order", league_id=5652,
        season_set_train="2023-2025", roster_rows=(row,), fa_rows=(),
        driver_lines=(), skipped=(),
    )
    write_cache(CacheKey.STREAK_SCORES, serialize_report(rpt))
    resp = client.get("/streaks")
    assert resp.status_code == 200
    assert "(inactive - 30 days)" in resp.data.decode()
```

- [ ] **Step 4: Run the web tests to verify they pass**

Run: `python -m pytest tests/test_web/test_streaks_snapshot.py tests/test_web/test_streaks_route.py tests/test_web/test_season_routes.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/_streaks_row.html tests/test_web/test_streaks_snapshot.py tests/test_web/snapshots/streaks.html
git commit -m "feat(streaks): inactive marker on the /streaks web page"
```

---

### Task 7: Full verification sweep

**Files:** none (verification only).

- [ ] **Step 1: Run the streaks + web suites**

Run: `python -m pytest tests/test_streaks/ tests/test_web/ -q`
Expected: all pass.

- [ ] **Step 2: Lint + format + types**

Run:
```bash
ruff check .
ruff format --check .
python -m mypy src/fantasy_baseball/streaks/
```
Expected: `ruff check` clean; `ruff format --check` reports no drift (run `ruff format .` if it does, then re-commit); `mypy` reports no issues in the streaks package. Fix any failure before proceeding.

- [ ] **Step 3: Commit any formatting fixes**

```bash
git add -A
git commit -m "chore(streaks): formatting/lint after staleness anchor"   # only if there were fixes
```

---

## Notes for the implementer

- The staleness DECISION is window-based (`window_end`), not literal-last-game-based, by design (see spec "Staleness metric"). For a player out for weeks, `window_end` == last game, so the count reads naturally; for a just-returned player with < 5 PA over the last two weeks, he is correctly shown inactive (no scoreable 14-day streak). Do not "fix" this by sourcing the last-game date from `hitter_games` — that is explicitly out of scope.
- Every `ReportRow(...)` construction in the test suite uses keyword args, so the trailing defaulted field is compile-safe. Do not reorder `ReportRow` fields.
- Do not touch `compute_windows`' `pa >= 5` filter or the `[first_played, last_played]` calendar — the fix is entirely at the scoring/report layer.
