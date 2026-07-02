# Draft Value Dashboard Tab Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Surface the existing draft-value metric as a second "Draft Grade" tab on the `/transactions` page, fed by a cached refresh step.

**Architecture:** Mirror the `TRANSACTION_ANALYZER` path end to end: a pure serializer (`build_draft_value_cache`) turns `run_draft_value()`'s dataclass output into a JSON-safe blob; a new refresh step caches it under `CacheKey.DRAFT_VALUE`; the `/transactions` route reads that cache and renders an expandable team-leaderboard panel behind a tab strip.

**Tech Stack:** Python 3.11, Flask + Jinja2, pytest + fakeredis, the project KV store (`kv_store.get_kv()`), `ruff`/`mypy`/`vulture`.

**Spec:** `docs/superpowers/specs/2026-07-02-draft-value-dashboard-tab-design.md`

## Global Constraints

- **ASCII-only in Python source, log messages, and any string that hits `print()`.** Jinja templates are UTF-8 and may use `—` (U+2014) / `Δ` — the ASCII rule targets `print()`/cp1252 stdout, not templates.
- **Never key on bare names for data joins** (use `name::player_type`); the per-team two-way `display_name` suffix keys on `name` only for *cosmetic* per-team labeling, which is acceptable.
- **Do not use `x or default` for numeric defaults** — use `x if x is not None else default`. Especially in sort keys.
- **`build_draft_value_cache` must survive `json.dumps(payload, allow_nan=False)`** — every float field passes through `_finite()` (`NaN`/`inf`/`None` -> `None`).
- **mypy** covers `src/fantasy_baseball/analysis/` (`[tool.mypy].files` in `pyproject.toml`). New code in `analysis/draft_value.py` must type-check.
- **Frequent commits**, one per task. Verification gates (`pytest`, `ruff check`, `ruff format --check`, `vulture`, `mypy` for analysis/) before declaring a task done.
- **Never modify a failing test to make it pass** — fix the code.

---

## File Structure

| File | Responsibility |
|---|---|
| `src/fantasy_baseball/analysis/draft_value.py` | + `_finite()` helper, + `build_draft_value_cache()` pure serializer |
| `src/fantasy_baseball/data/cache_keys.py` | + `DRAFT_VALUE` enum member |
| `src/fantasy_baseball/web/refresh_pipeline.py` | + `_compute_draft_value()` step + call in `run()` |
| `src/fantasy_baseball/web/season_routes.py` | `/transactions` reads `DRAFT_VALUE`, passes `draft_data` |
| `src/fantasy_baseball/web/static/season.css` | promoted `.tab-strip` rules |
| `src/fantasy_baseball/web/templates/season/trends.html` | drop inline `.tab-strip` CSS |
| `src/fantasy_baseball/web/templates/season/transactions.html` | tab strip + Draft Grade panel + hoisted JS |
| `tests/test_analysis/test_draft_value.py` | + `build_draft_value_cache` unit tests |
| `tests/test_web/_refresh_fixture.py` | patch `run_draft_value` -> canned dataclasses |
| `tests/test_web/test_refresh_pipeline.py` | dedicated `draft_value` cache-write test + expected key |
| `tests/test_web/test_season_routes.py` | `/transactions` render tests (both tabs, empty states) |

---

## Task 1: Pure serializer `build_draft_value_cache`

**Files:**
- Modify: `src/fantasy_baseball/analysis/draft_value.py` (add `_finite` + `build_draft_value_cache` near the bottom, after `run_draft_value`)
- Test: `tests/test_analysis/test_draft_value.py`

**Interfaces:**
- Consumes: `PlayerValue` (fields: `team, name, player_type, slot, baseline_kind, preseason_var, est_var_proj, est_var_ytd, value_proj, value_ytd, skill, luck`) and `TeamRollup` (`team, sum_value, avg_value, credited_count`), both already defined in this module.
- Produces: `build_draft_value_cache(players: list[PlayerValue], teams: list[TeamRollup]) -> dict[str, Any]` returning `{"horizon": "proj", "teams": [{team, avg_value, sum_value, credited_count, players: [{name, display_name, player_type, kind, slot, preseason_var, est_var_proj, value_proj, value_ytd, skill, luck}]}]}`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_analysis/test_draft_value.py` (top-level). The file already imports `json as _json` and `draft_value as dv` (lines 1, 6) — reuse `_json`; **do not add new imports** (an unused `import math`/duplicate `import json` would fail the Task 1 Step 5 ruff gate with F401):

```python
def _pv(team, name, value_proj, **kw):
    """Construct a PlayerValue with sensible finite defaults; override via kw."""
    defaults = dict(
        player_type="hitter",
        slot=None,
        baseline_kind="drafted",
        preseason_var=10.0,
        est_var_proj=12.0,
        est_var_ytd=6.0,
        value_ytd=2.0,
        skill=1.0,
        luck=1.0,
    )
    defaults.update(kw)
    return dv.PlayerValue(
        team=team,
        name=name,
        player_type=defaults["player_type"],
        slot=defaults["slot"],
        baseline_kind=defaults["baseline_kind"],
        preseason_var=defaults["preseason_var"],
        est_var_proj=defaults["est_var_proj"],
        est_var_ytd=defaults["est_var_ytd"],
        value_proj=value_proj,
        value_ytd=defaults["value_ytd"],
        skill=defaults["skill"],
        luck=defaults["luck"],
    )


def test_build_cache_groups_and_sorts_teams_and_players():
    players = [
        _pv("Bravo", "B1", 1.0),
        _pv("Bravo", "B2", 5.0),
        _pv("Alpha", "A1", 3.0),
    ]
    teams = [
        dv.TeamRollup("Alpha", 3.0, 3.0, 1),
        dv.TeamRollup("Bravo", 6.0, 3.0, 2),
    ]
    out = dv.build_draft_value_cache(players, teams)
    assert out["horizon"] == "proj"
    names = [t["team"] for t in out["teams"]]
    # Equal avg_value (3.0, 3.0) -> stable order preserves input (Alpha, Bravo)
    assert names == ["Alpha", "Bravo"]
    bravo = next(t for t in out["teams"] if t["team"] == "Bravo")
    # players sorted by value_proj desc within team
    assert [p["name"] for p in bravo["players"]] == ["B2", "B1"]
    assert bravo["credited_count"] == 2


def test_build_cache_nan_avg_team_sinks():
    players = [_pv("Good", "G", 4.0), _pv("Empty", "E", float("nan"))]
    teams = [
        dv.TeamRollup("Empty", 0.0, float("nan"), 0),
        dv.TeamRollup("Good", 4.0, 4.0, 1),
    ]
    out = dv.build_draft_value_cache(players, teams)
    assert out["teams"][0]["team"] == "Good"
    assert out["teams"][-1]["team"] == "Empty"
    assert out["teams"][-1]["avg_value"] is None  # NaN -> null


def test_build_cache_nonfinite_to_null_and_strict_json():
    players = [_pv("T", "P", float("nan"), skill=float("inf"), luck=float("-inf"))]
    teams = [dv.TeamRollup("T", 0.0, 0.0, 0)]
    out = dv.build_draft_value_cache(players, teams)
    p = out["teams"][0]["players"][0]
    assert p["value_proj"] is None
    assert p["skill"] is None
    assert p["luck"] is None
    # No non-finite float leaks -> strict JSON succeeds.
    _json.dumps(out, allow_nan=False)


def test_build_cache_off_board_flier_nulls_but_finite_value():
    players = [_pv("T", "Flier", 0.0, preseason_var=None, skill=None, luck=None)]
    teams = [dv.TeamRollup("T", 0.0, 0.0, 1)]
    out = dv.build_draft_value_cache(players, teams)
    p = out["teams"][0]["players"][0]
    assert p["preseason_var"] is None
    assert p["skill"] is None
    assert p["luck"] is None
    assert p["value_proj"] == 0.0  # finite, still present


def test_build_cache_field_mapping():
    players = [_pv("T", "P", 3.0, baseline_kind="keeper")]
    teams = [dv.TeamRollup("T", 3.0, 3.0, 1)]
    p = dv.build_draft_value_cache(players, teams)["teams"][0]["players"][0]
    assert "est_var_ytd" not in p  # dropped
    assert p["value_ytd"] == 2.0  # kept
    assert p["kind"] == "keeper"  # baseline_kind -> kind
    assert isinstance(p["player_type"], str)


def test_build_cache_credited_count_may_be_below_player_count():
    # Two rows, one ungradeable (NaN value_proj); rollup credits only 1.
    players = [_pv("T", "Good", 3.0), _pv("T", "NaNrow", float("nan"))]
    teams = [dv.TeamRollup("T", 3.0, 3.0, 1)]
    team = dv.build_draft_value_cache(players, teams)["teams"][0]
    assert team["credited_count"] == 1
    assert len(team["players"]) == 2
    nan_row = next(p for p in team["players"] if p["name"] == "NaNrow")
    assert nan_row["value_proj"] is None


def test_build_cache_two_way_display_name_per_team():
    # Same name, both types, SAME team -> suffixed; identical name solo on
    # ANOTHER team -> no suffix (per-team scope).
    players = [
        _pv("T1", "Shohei Ohtani", 5.0, player_type="hitter"),
        _pv("T1", "Shohei Ohtani", 4.0, player_type="pitcher"),
        _pv("T2", "Shohei Ohtani", 3.0, player_type="hitter"),
    ]
    teams = [
        dv.TeamRollup("T1", 9.0, 4.5, 2),
        dv.TeamRollup("T2", 3.0, 3.0, 1),
    ]
    out = dv.build_draft_value_cache(players, teams)
    t1 = next(t for t in out["teams"] if t["team"] == "T1")
    t2 = next(t for t in out["teams"] if t["team"] == "T2")
    disp_t1 = sorted(p["display_name"] for p in t1["players"])
    assert disp_t1 == ["Shohei Ohtani (H)", "Shohei Ohtani (P)"]
    assert t2["players"][0]["display_name"] == "Shohei Ohtani"  # solo -> no suffix
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_analysis/test_draft_value.py -k build_cache -v`
Expected: FAIL with `AttributeError: module 'fantasy_baseball.analysis.draft_value' has no attribute 'build_draft_value_cache'`.

- [ ] **Step 3: Implement `_finite` + `build_draft_value_cache`**

`math` is already imported at the top of `draft_value.py`. Add at the end of the module (after `run_draft_value`):

```python
def _finite(x: float | None) -> float | None:
    """Map None/NaN/inf -> None so the payload survives strict JSON + Jinja."""
    return x if x is not None and math.isfinite(x) else None


def build_draft_value_cache(
    players: list[PlayerValue], teams: list[TeamRollup]
) -> dict[str, Any]:
    """Serialize run_draft_value() output into a JSON-safe, template-ready dict.

    Groups ``players`` by ``.team`` under each ``TeamRollup`` (teams sorted by
    ``avg_value`` desc with NaN sunk; players sorted by ``value_proj`` desc with
    None/NaN sunk). Every float field passes through ``_finite`` so no non-finite
    value reaches strict JSON or Jinja. Within each team, a ``name`` appearing
    under more than one ``player_type`` gets a ` (H)`/` (P)` ``display_name``
    suffix (two-way disambiguation); all other rows get ``display_name == name``.
    """
    by_team: dict[str, list[PlayerValue]] = {}
    for p in players:
        by_team.setdefault(p.team, []).append(p)

    out_teams: list[dict[str, Any]] = []
    for tr in sorted(
        teams,
        key=lambda t: -math.inf if math.isnan(t.avg_value) else t.avg_value,
        reverse=True,
    ):
        roster = by_team.get(tr.team, [])
        types_by_name: dict[str, set[str]] = {}
        for p in roster:
            types_by_name.setdefault(p.name, set()).add(str(p.player_type))
        out_players: list[dict[str, Any]] = []
        for p in sorted(
            roster,
            key=lambda p: (
                -math.inf
                if p.value_proj is None or math.isnan(p.value_proj)
                else p.value_proj
            ),
            reverse=True,
        ):
            suffix = ""
            if len(types_by_name.get(p.name, ())) > 1:
                suffix = " (P)" if str(p.player_type) == "pitcher" else " (H)"
            out_players.append(
                {
                    "name": p.name,
                    "display_name": f"{p.name}{suffix}",
                    "player_type": str(p.player_type),
                    "kind": p.baseline_kind,
                    "slot": p.slot,
                    "preseason_var": _finite(p.preseason_var),
                    "est_var_proj": _finite(p.est_var_proj),
                    "value_proj": _finite(p.value_proj),
                    "value_ytd": _finite(p.value_ytd),
                    "skill": _finite(p.skill),
                    "luck": _finite(p.luck),
                }
            )
        out_teams.append(
            {
                "team": tr.team,
                "avg_value": _finite(tr.avg_value),
                "sum_value": _finite(tr.sum_value),
                "credited_count": tr.credited_count,
                "players": out_players,
            }
        )
    return {"horizon": "proj", "teams": out_teams}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `pytest tests/test_analysis/test_draft_value.py -k build_cache -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Run the gates**

Run: `ruff check src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py && ruff format --check src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py && mypy src/fantasy_baseball/analysis/draft_value.py`
Expected: no violations; mypy clean. (If mypy flags the sort-key lambdas, confirm the `p.value_proj is None or math.isnan(...)` short-circuit is intact — it narrows `p.value_proj` to `float` in the `else`.)

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/analysis/draft_value.py tests/test_analysis/test_draft_value.py
git commit -m "feat(draft-value): add build_draft_value_cache serializer"
```

---

## Task 2: Refresh step + `DRAFT_VALUE` cache key

**Files:**
- Modify: `src/fantasy_baseball/data/cache_keys.py` (add `DRAFT_VALUE`)
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py` (add `_compute_draft_value` method + call it in `run()`)
- Modify: `tests/test_web/_refresh_fixture.py` (patch `run_draft_value` to canned dataclasses)
- Test: `tests/test_web/test_refresh_pipeline.py` (dedicated write test + add key to shape list)

**Interfaces:**
- Consumes: `build_draft_value_cache` (Task 1), `run_draft_value` (existing), `write_cache`, `CacheKey`, `log` (all module-scope in `refresh_pipeline.py`).
- Produces: writes `cache:draft_value` = `{"horizon": "proj", "teams": [...]}` during `run_full_refresh()`.

- [ ] **Step 1: Add the `DRAFT_VALUE` cache key**

In `src/fantasy_baseball/data/cache_keys.py`, add after `STASH = "stash"` (line ~36):

```python
    DRAFT_VALUE = "draft_value"
```

- [ ] **Step 2: Write the failing refresh test + fixture patch**

In `tests/test_web/_refresh_fixture.py`, add a canned factory near the other canned data (module level):

```python
def _canned_draft_value():
    """Canned (players, teams) so the refresh test never runs the heavy
    real-data draft_value computation (which reads real repo files by
    absolute path and would return the real league's teams)."""
    from fantasy_baseball.analysis.draft_value import PlayerValue, TeamRollup

    players = [
        PlayerValue(
            team=USER_TEAM_NAME,
            name="Canned Keeper",
            player_type="hitter",
            slot=None,
            baseline_kind="keeper",
            preseason_var=10.0,
            est_var_proj=12.0,
            est_var_ytd=6.0,
            value_proj=2.0,
            value_ytd=1.0,
            skill=1.0,
            luck=1.0,
        )
    ]
    teams = [TeamRollup(USER_TEAM_NAME, 2.0, 2.0, 1)]
    return players, teams
```

Then add this patch to the `patches` list in `patched_refresh_environment` (alongside the `_compute_streaks` stub, near line ~543):

```python
        # draft-value reads real repo files by absolute path and its
        # `from fantasy_baseball.config import load_config` binding is not
        # reached by this fixture's config patch, so left unpatched it would run
        # the full real-data computation. Patch the function _compute_draft_value
        # imports (function-body import resolves the module attr at call time).
        patch(
            "fantasy_baseball.analysis.draft_value.run_draft_value",
            side_effect=_canned_draft_value,
        ),
```

In `tests/test_web/test_refresh_pipeline.py`, add `CacheKey.DRAFT_VALUE` to the `expected_keys` list in `test_all_expected_cache_files_written` (after `CacheKey.TRANSACTION_ANALYZER`), and add a dedicated test to `class TestRefreshShape`:

```python
    def test_draft_value_cache_written(self, configured_test_env, fake_redis):
        with patched_refresh_environment(fake_redis):
            refresh_pipeline.run_full_refresh()
        data = _read(fake_redis, "draft_value")
        assert isinstance(data, dict)
        assert data.get("horizon") == "proj"
        teams = data.get("teams")
        assert isinstance(teams, list) and teams
        team = teams[0]
        assert {"team", "avg_value", "sum_value", "credited_count", "players"} <= team.keys()
        assert team["players"], "team should carry its canned player row"
        assert team["players"][0]["display_name"] == "Canned Keeper"
```

- [ ] **Step 3: Run the tests to verify they fail**

Run: `pytest tests/test_web/test_refresh_pipeline.py -k "draft_value or all_expected" -v`
Expected: FAIL — `test_draft_value_cache_written` errors (no `_compute_draft_value` / no cache written) and `test_all_expected_cache_files_written` fails on `Missing cache key: draft_value`.

- [ ] **Step 4: Add the `_compute_draft_value` step**

In `src/fantasy_baseball/web/refresh_pipeline.py`, add the call in the step sequence in `RefreshRun._run_pipeline_steps()` (line ~498) immediately after `self._analyze_transactions()` (line ~520):

```python
        self._analyze_transactions()
        self._compute_draft_value()
        self._compute_streaks()
```

Add the method after `_analyze_transactions` (before `_compute_streaks`, near line ~1591):

```python
    # --- Step 15c: Draft-value grade for the /transactions Draft Grade tab ---
    def _compute_draft_value(self) -> None:
        """Grade the draft (keepers + drafted picks) and cache it for the tab.

        Non-load-bearing: run_draft_value() can raise (reconstruction gate /
        missing keepers), so catch broadly, log, and continue -- a cosmetic
        panel must never abort a refresh whose load-bearing steps succeeded.
        required=False leaves any prior cache untouched on failure. Runs on
        Render too (all inputs are git-tracked or in the KV store; no duckdb).
        """
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

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_web/test_refresh_pipeline.py -k "draft_value or all_expected" -v`
Expected: PASS.

- [ ] **Step 6: Run the gates**

Run: `ruff check src/fantasy_baseball/data/cache_keys.py src/fantasy_baseball/web/refresh_pipeline.py tests/test_web/ && ruff format --check src/fantasy_baseball/data/cache_keys.py src/fantasy_baseball/web/refresh_pipeline.py tests/test_web/ && mypy src/fantasy_baseball/web/refresh_pipeline.py`
Expected: clean. (`refresh_pipeline.py` IS under `[tool.mypy].files` — run mypy on it; the new method returns `None` and uses typed `write_cache`/`CacheKey`, so it type-checks.)

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/data/cache_keys.py src/fantasy_baseball/web/refresh_pipeline.py tests/test_web/_refresh_fixture.py tests/test_web/test_refresh_pipeline.py
git commit -m "feat(draft-value): cache draft grade in the refresh pipeline"
```

---

## Task 3: Promote `.tab-strip` CSS to `season.css`

**Files:**
- Modify: `src/fantasy_baseball/web/static/season.css` (add `.tab-strip` rules)
- Modify: `src/fantasy_baseball/web/templates/season/trends.html` (delete the inline `.tab-strip` rules)
- Test: `tests/test_web/test_trends_route.py` (assert `/trends` still renders — existing `test_trends_page_renders` covers this)

**Interfaces:**
- Produces: a global `.tab-strip` style set (loaded via `base.html`) that both `trends.html` and `transactions.html` (Task 4) consume.

- [ ] **Step 1: Append the promoted rules to `season.css`**

At the end of `src/fantasy_baseball/web/static/season.css`, add (byte-faithful copy of the `trends.html` block — keep `var(--amber)`/`var(--bg)`; do NOT rescope under `.page-transactions`):

```css
/* Tab strip — shared by /trends and /transactions. Promoted from trends.html. */
.tab-strip {
  display: flex;
  gap: 4px;
  margin-bottom: 8px;
  flex-wrap: wrap;
}
.tab-strip button {
  background: var(--surface);
  color: var(--ink-soft);
  border: 1px solid var(--line);
  padding: 5px 12px;
  font-family: 'Outfit', sans-serif;
  font-size: 12.5px;
  font-weight: 500;
  cursor: pointer;
  border-radius: 4px;
  transition: all .12s;
}
.tab-strip button:hover {
  background: var(--surface-soft);
  color: var(--ink);
}
.tab-strip button.active {
  background: var(--amber);
  color: var(--bg);
  border-color: var(--amber);
  font-weight: 600;
}
```

- [ ] **Step 2: Delete the inline rules from `trends.html`**

In `src/fantasy_baseball/web/templates/season/trends.html`, remove exactly the `.tab-strip`, `.tab-strip button`, `.tab-strip button:hover`, and `.tab-strip button.active` rule blocks from the `<style>` block (currently lines ~17-44). Leave `.trends-section`, `.chart-wrapper`, and all other rules untouched.

- [ ] **Step 3: Verify `/trends` still renders and the CSS moved**

Run: `pytest tests/test_web/test_trends_route.py::test_trends_page_renders -v`
Expected: PASS.

Run: `grep -c "\.tab-strip" src/fantasy_baseball/web/static/season.css` (expect >= 1) and `grep -c "\.tab-strip" src/fantasy_baseball/web/templates/season/trends.html` (expect 0). Use the Grep tool, not shell grep.
Expected: `season.css` has the rules; `trends.html` no longer does.

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/static/season.css src/fantasy_baseball/web/templates/season/trends.html
git commit -m "refactor(web): promote .tab-strip CSS to season.css"
```

---

## Task 4: Route wiring + `transactions.html` tab restructure

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py` (`transactions()` reads `DRAFT_VALUE`, passes `draft_data`)
- Modify: `src/fantasy_baseball/web/templates/season/transactions.html` (tab strip, two panels, hoisted JS)
- Test: `tests/test_web/test_season_routes.py` (render tests)

**Interfaces:**
- Consumes: `cache:draft_value` shape from Task 1/2; the `.tab-strip` CSS from Task 3; existing `toggleTxnDetail` idiom + `.user-team`/`value-positive`/`value-negative`/`placeholder-text` classes.
- Produces: `/transactions` renders two tab panels; the Draft Grade panel is an expandable team leaderboard.

- [ ] **Step 1: Write the failing render tests**

Add to `tests/test_web/test_season_routes.py` (uses the existing `client` fixture and `patch`/`CacheKey` already imported at top):

```python
def _txn_draft_cache(txn_teams, draft_teams):
    """side_effect for read_cache_dict keyed by CacheKey.value."""
    values = {
        CacheKey.TRANSACTION_ANALYZER.value: {"teams": txn_teams} if txn_teams is not None else None,
        CacheKey.DRAFT_VALUE.value: {"horizon": "proj", "teams": draft_teams} if draft_teams is not None else None,
    }

    def _fake(key, *_a, **_k):
        v = values.get(key.value)
        return v if isinstance(v, dict) else None

    return _fake


_DRAFT_TEAM = {
    "team": "Hart of the Order",
    "avg_value": 4.2,
    "sum_value": 58.1,
    "credited_count": 14,
    "players": [
        {
            "name": "Juan Soto",
            "display_name": "Juan Soto",
            "player_type": "hitter",
            "kind": "keeper",
            "slot": None,
            "preseason_var": 38.1,
            "est_var_proj": 44.3,
            "value_proj": 12.3,
            "value_ytd": 3.1,
            "skill": 6.1,
            "luck": 6.2,
        }
    ],
}


def test_transactions_renders_both_tabs(client):
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        side_effect=_txn_draft_cache([], [_DRAFT_TEAM]),
    ):
        resp = client.get("/transactions")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "tab-strip" in body
    assert "Draft Grade" in body
    assert "Juan Soto" in body


def test_transactions_draft_empty_placeholder(client):
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        side_effect=_txn_draft_cache([], None),
    ):
        resp = client.get("/transactions")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "tab-strip" in body  # tab strip present even with empty draft data
    assert "No draft data" in body


def test_transactions_empty_txn_but_populated_draft(client):
    # Post-draft / pre-first-transaction: txn empty, draft populated.
    # Guards the hoist-out-of-conditional restructure.
    with patch(
        "fantasy_baseball.web.season_routes.read_cache_dict",
        side_effect=_txn_draft_cache([], [_DRAFT_TEAM]),
    ):
        resp = client.get("/transactions")
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert "tab-strip" in body
    assert "switchTab" in body  # tab JS hoisted, present regardless of txn_data
    assert "toggleTxnDetail" in body  # expand JS hoisted too
    assert "Juan Soto" in body  # draft rows render
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `pytest tests/test_web/test_season_routes.py -k "transactions_renders_both or draft_empty or empty_txn_but" -v`
Expected: FAIL (`Draft Grade`/`tab-strip`/`Juan Soto` not in body — template not yet restructured; route not passing `draft_data`).

- [ ] **Step 3: Wire the route**

In `src/fantasy_baseball/web/season_routes.py`, update `transactions()` (line ~1890):

```python
    @app.route("/transactions")
    def transactions():
        meta = read_meta()
        txn_cache = read_cache_dict(CacheKey.TRANSACTION_ANALYZER) or {}
        draft_cache = read_cache_dict(CacheKey.DRAFT_VALUE) or {}
        config = _load_config()
        return render_template(
            "season/transactions.html",
            meta=meta,
            active_page="transactions",
            txn_data=txn_cache.get("teams", []),
            draft_data=draft_cache.get("teams", []),
            user_team=config.team_name,
        )
```

- [ ] **Step 4: Restructure the template**

Replace the entire contents of `src/fantasy_baseball/web/templates/season/transactions.html` with:

```html
{% extends "season/base.html" %}
{% block content %}
<style>
/* Layout-only rules; theming lives in season.css. */
.txn-summary { cursor: pointer; }
.txn-detail { display: none; }
.txn-detail.open { display: table-row; }
.txn-inner-table { width: 100%; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
</style>

<div class="page-transactions">
<div class="page-header">
    <nav class="tab-strip" id="txn-tab-strip">
        <button type="button" class="active" data-panel="panel-transactions" onclick="switchTab(this)">Transactions</button>
        <button type="button" data-panel="panel-draft" onclick="switchTab(this)">Draft Grade</button>
    </nav>
</div>

<div id="panel-transactions" class="tab-panel active">
{% if not txn_data %}
<p class="placeholder-text">No transaction data. Click "Refresh Data" to fetch from Yahoo.</p>
{% else %}
<table class="data-table">
    <thead>
        <tr>
            <th>Team</th>
            <th>Moves</th>
            <th>Net ΔRoto</th>
            <th>Best Move</th>
            <th>Worst Move</th>
        </tr>
    </thead>
    <tbody>
    {% for team in txn_data %}
        {% set best = team.transactions | sort(attribute='delta_roto', reverse=true) | first %}
        {% set worst = team.transactions | sort(attribute='delta_roto') | first %}
        <tr class="txn-summary {% if team.team == user_team %}user-team{% endif %}" onclick="toggleTxnDetail(this)">
            <td class="team-name">{{ team.team }}</td>
            <td>{{ team.transaction_count }}</td>
            <td class="{{ 'value-positive' if team.net_value > 0 else 'value-negative' if team.net_value < 0 else '' }}">
                {{ '%+.1f'|format(team.net_value) }}
            </td>
            <td>
                {% if best and best.delta_roto > 0 %}
                <span class="value-positive">{{ '%+.1f'|format(best.delta_roto) }}</span>
                <span style="color:var(--text-secondary);font-size:11px;">
                    {% if best.add_name %}+{{ best.add_name }}{% endif %}
                </span>
                {% else %}—{% endif %}
            </td>
            <td>
                {% if worst and worst.delta_roto < 0 %}
                <span class="value-negative">{{ '%+.1f'|format(worst.delta_roto) }}</span>
                <span style="color:var(--text-secondary);font-size:11px;">
                    {% if worst.drop_name %}-{{ worst.drop_name }}{% endif %}
                </span>
                {% else %}—{% endif %}
            </td>
        </tr>
        <tr class="txn-detail">
            <td colspan="5">
                <table class="txn-inner-table">
                    <thead>
                        <tr style="color:var(--text-secondary);font-size:11px;">
                            <th style="text-align:left">Date</th>
                            <th style="text-align:left">Added</th>
                            <th style="text-align:left">Dropped</th>
                            <th style="text-align:right">ΔRoto</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for txn in team.transactions %}
                        <tr>
                            <td>{{ txn.date or '—' }}{% if txn.paired %} <span class="badge-paired">Paired</span>{% endif %}</td>
                            <td>
                                {% if txn.add_name %}
                                {{ txn.add_name }}
                                <span style="color:var(--text-secondary);font-size:11px;">({{ txn.add_positions | join(', ') }})</span>
                                {% else %}—{% endif %}
                            </td>
                            <td>
                                {% if txn.drop_name %}
                                {{ txn.drop_name }}
                                <span style="color:var(--text-secondary);font-size:11px;">({{ txn.drop_positions | join(', ') }})</span>
                                {% else %}—{% endif %}
                            </td>
                            <td style="text-align:right" class="{{ 'value-positive' if txn.delta_roto > 0 else 'value-negative' if txn.delta_roto < 0 else '' }}">
                                {{ '%+.1f'|format(txn.delta_roto) }}
                            </td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </td>
        </tr>
    {% endfor %}
    </tbody>
</table>
{% endif %}
</div>

<div id="panel-draft" class="tab-panel">
{% if not draft_data %}
<p class="placeholder-text">No draft data. Click "Refresh Data" to compute the draft grade.</p>
{% else %}
<table class="data-table">
    <thead>
        <tr>
            <th>Team</th>
            <th>avg</th>
            <th>sum</th>
            <th>picks</th>
        </tr>
    </thead>
    <tbody>
    {% for team in draft_data %}
        <tr class="txn-summary {% if team.team == user_team %}user-team{% endif %}" onclick="toggleTxnDetail(this)">
            <td class="team-name">{{ team.team }}</td>
            <td class="{{ 'value-positive' if team.avg_value is not none and team.avg_value > 0 else 'value-negative' if team.avg_value is not none and team.avg_value < 0 else '' }}">
                {{ '%+.1f'|format(team.avg_value) if team.avg_value is not none else '—' }}
            </td>
            <td>{{ '%+.1f'|format(team.sum_value) if team.sum_value is not none else '—' }}</td>
            <td>{{ team.credited_count }}</td>
        </tr>
        <tr class="txn-detail">
            <td colspan="4">
                <table class="txn-inner-table">
                    <thead>
                        <tr style="color:var(--text-secondary);font-size:11px;">
                            <th style="text-align:left">Player</th>
                            <th style="text-align:left">kind</th>
                            <th style="text-align:right">slot</th>
                            <th style="text-align:right">preVAR</th>
                            <th style="text-align:right">estVAR</th>
                            <th style="text-align:right">value</th>
                            <th style="text-align:right">valueYTD</th>
                            <th style="text-align:right">skill</th>
                            <th style="text-align:right">luck</th>
                        </tr>
                    </thead>
                    <tbody>
                    {% for p in team.players %}
                        <tr>
                            <td>{{ p.display_name }}</td>
                            <td>{{ p.kind }}</td>
                            <td style="text-align:right">{{ p.slot if p.slot is not none else '—' }}</td>
                            <td style="text-align:right">{{ '%.1f'|format(p.preseason_var) if p.preseason_var is not none else '—' }}</td>
                            <td style="text-align:right">{{ '%.1f'|format(p.est_var_proj) if p.est_var_proj is not none else '—' }}</td>
                            <td style="text-align:right" class="{{ 'value-positive' if p.value_proj is not none and p.value_proj > 0 else 'value-negative' if p.value_proj is not none and p.value_proj < 0 else '' }}">
                                {{ '%+.1f'|format(p.value_proj) if p.value_proj is not none else '—' }}
                            </td>
                            <td style="text-align:right">{{ '%+.1f'|format(p.value_ytd) if p.value_ytd is not none else '—' }}</td>
                            <td style="text-align:right">{{ '%+.1f'|format(p.skill) if p.skill is not none else '—' }}</td>
                            <td style="text-align:right">{{ '%+.1f'|format(p.luck) if p.luck is not none else '—' }}</td>
                        </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </td>
        </tr>
    {% endfor %}
    </tbody>
</table>
{% endif %}
</div>

<script>
function switchTab(btn) {
    document.querySelectorAll('#txn-tab-strip button').forEach(function (b) { b.classList.remove('active'); });
    btn.classList.add('active');
    var target = btn.dataset.panel;
    document.querySelectorAll('.tab-panel').forEach(function (panel) {
        panel.classList.toggle('active', panel.id === target);
    });
}
function toggleTxnDetail(row) {
    var detail = row.nextElementSibling;
    if (detail && detail.classList.contains('txn-detail')) {
        detail.classList.toggle('open');
    }
}
</script>

</div>
{% endblock %}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `pytest tests/test_web/test_season_routes.py -k "transactions_renders_both or draft_empty or empty_txn_but" -v`
Expected: PASS (3 tests).

- [ ] **Step 6: Run the gates**

Run: `ruff check src/fantasy_baseball/web/season_routes.py tests/test_web/test_season_routes.py && ruff format --check src/fantasy_baseball/web/season_routes.py tests/test_web/test_season_routes.py && mypy src/fantasy_baseball/web/season_routes.py`
Expected: clean. (`season_routes.py` IS under `[tool.mypy].files`; the added `read_cache_dict(...) or {}` + `.get("teams", [])` are already-typed calls, so it type-checks.)

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py src/fantasy_baseball/web/templates/season/transactions.html tests/test_web/test_season_routes.py
git commit -m "feat(draft-value): render Draft Grade tab on /transactions"
```

---

## Task 5: Full verification + local refresh smoke

**Files:** none (verification only).

- [ ] **Step 1: Full test suite**

Run: `pytest tests/test_analysis/test_draft_value.py tests/test_web/ -v` (parallel: `pytest -n auto tests/test_analysis tests/test_web`).
Expected: all green.

- [ ] **Step 2: Repo-wide lint/format/dead-code**

Run: `ruff check . && ruff format --check . && vulture` and `mypy` (per `[tool.mypy].files`).
Expected: zero violations; no NEW vulture findings (`build_draft_value_cache`/`_finite` are referenced by the refresh step and tests, so not dead). Note any pre-existing unrelated findings.

- [ ] **Step 3: Local refresh smoke (exercise the new step against real data)**

Run: `python scripts/run_season_dashboard.py --no-sync` (the `--no-sync` avoids clobbering local SQLite while verifying not-yet-deployed code), trigger a refresh, and confirm the log shows "Computing draft value grade..." / "Draft value cached: N teams" without an error, then load `/transactions` and click the Draft Grade tab. If auth/live data is unavailable, at minimum run `python scripts/draft_value.py` to confirm `run_draft_value()` still succeeds against the current KV, then rely on the dedicated refresh test for the wiring.
Expected: draft grade cached; the tab renders the leaderboard with expandable rows.

- [ ] **Step 4: Commit any fixups**

```bash
git add -A
git commit -m "chore(draft-value): verification fixups"  # only if needed
```

---

## Self-Review Notes

- **Spec coverage:** cache key (Task 2) · serializer with grouping/sort/NaN/two-way (Task 1) · refresh step + Render-safe + fixture patch + dedicated test (Task 2) · route reads `DRAFT_VALUE` (Task 4) · template hoist + per-panel empty states + `txn-detail` class + `colspan=4` + `display_name`/`slot`/YTD columns (Task 4) · `.tab-strip` promotion, byte-faithful, trends unbroken (Task 3) · unit/refresh/route tests incl. empty-`txn_data` state (Tasks 1/2/4) · local refresh smoke (Task 5).
- **Placeholder glyphs:** the template uses the literal `—` (U+2014) placeholder and `Δ` (in the ΔRoto headers), matching the existing `transactions.html` verbatim (repo template style is literal glyphs, not HTML entities). The CLAUDE.md ASCII rule targets `print()`/cp1252 stdout, not UTF-8 templates, so this is in-policy. Panel 1's ΔRoto table is moved unchanged.
- **Type consistency:** `build_draft_value_cache(players, teams)` signature and output keys are identical across Task 1 (definition), Task 2 (refresh call), and Task 4 (template/render tests).
```