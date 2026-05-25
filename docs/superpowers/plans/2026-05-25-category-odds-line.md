# Category Odds Line Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a line of three user-team stats between the Category Bars category selector and the chart -- `1st: N%` (P finish 1st in this category), `top 3: N%` (P finish top-3), `wins: x/9` (opponents whose band the user's band clearly clears) -- computed analytically from the same per-team means + SDs the chart already draws.

**Architecture:** A new pure module `category_odds.py` computes the odds via Gauss-Hermite integration over the user's normal distribution against opponents' normals (Poisson-binomial for the top-3 count). The Category Bars formatter in `season_data.py` calls it per category and embeds the result; the payload's per-category entry changes from a bare row list to `{rows, odds}`. The JS reads `.rows` for the chart and `.odds` for a new stats line.

**Tech Stack:** Python (math.erf + numpy Gauss-Hermite nodes), Flask/Jinja2, vanilla JS + Chart.js, pytest.

**Branch:** `feat/category-bars-refinements` (stacks on PR #100, per the user's choice).

---

## Background facts (verified against the branch)

- `scoring._prob_beats(mu_a, mu_b, sd_a, sd_b, *, higher_is_better)` (scoring.py:114) is the existing pairwise Gaussian win-prob; it uses `0.5 * (1 + erf(z / sqrt(2)))` for the normal CDF and guards `combined == 0`. We follow the same erf convention but need a one-sided tail `P(opp > x)` for a fixed scalar `x`, which `_prob_beats` does not provide -- so we write a small local helper.
- `delta_roto.py` already integrates over a normal with Gauss-Hermite (probabilists', weight `exp(-x^2/2)`, normalized by `sqrt(2*pi)`): `E[g(X)] = (1/sqrt(2*pi)) * sum_k w_k * g(mu + sd * node_k)`. We use the same scheme but generate nodes with `numpy.polynomial.hermite_e.hermegauss(N)` (numpy is a declared dep) so there are no hardcoded node constants.
- `numpy` and `scipy` are declared dependencies (`pyproject.toml`). We use numpy only at module load to build the quadrature nodes; the per-call math is pure-Python `math.erf`.
- Constants (`fantasy_baseball.utils.constants`): `ALL_CATEGORIES` (10 `Category`), `INVERSE_STATS` (ERA, WHIP), `Category.value` is the uppercase short name. `season_data.py` already imports `INVERSE_STATS` as `INVERSE_CATS`, plus `ALL_CATEGORIES` and `Category`.
- Current Category Bars payload (this branch): `_category_bars_one_flavor` returns `{CAT: [ {team, value, sd, is_user}, ... sorted best-on-top ]}`; `format_category_bars_for_display` wraps two flavors. The route embeds it as `<script id="category-bars-data">`. The JS `rowsFor()` reads `payload[projection][category]` as a list.
- Existing tests that will need updating for the payload-shape change: in `tests/test_web/test_season_data.py` -- `test_category_bars_normal_category_sorts_best_on_top`, `test_category_bars_inverse_category_sorts_lowest_on_top`, `test_category_bars_missing_sd_defaults_to_zero`, `test_category_bars_handles_missing_flavor` (these index `out[flavor]["R"]` as a list); in `tests/test_web/test_season_routes.py` -- `test_standings_embeds_category_bars_data` (indexes `bars["current"]["R"]` as a list). `test_category_bars_has_both_flavors_and_all_categories` only checks keys (unaffected). `test_standings_category_bars_empty_without_projections` asserts `bars == {"preseason": {}, "current": {}}` (unaffected -- empty flavors stay `{}`).
- ASCII-only source (CLAUDE.md). Don't use `x or <numeric-default>`.

## File structure

- **Create** `src/fantasy_baseball/category_odds.py` -- the pure odds math (Task 1).
- **Create** `tests/test_category_odds.py` -- unit tests (Task 1).
- **Modify** `src/fantasy_baseball/web/season_data.py` -- formatter attaches odds, payload shape -> `{rows, odds}` (Task 2).
- **Modify** `tests/test_web/test_season_data.py` + `tests/test_web/test_season_routes.py` -- update for new shape, assert odds (Task 2).
- **Modify** `src/fantasy_baseball/web/static/season_category_bars.js` + `src/fantasy_baseball/web/templates/season/standings.html` -- consume `.rows`, render the odds line (Task 3).

---

## Task 1: `category_odds.py` -- analytic per-category finish odds

**Files:**
- Create: `src/fantasy_baseball/category_odds.py`
- Test: `tests/test_category_odds.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_category_odds.py`:

```python
import pytest

from fantasy_baseball.category_odds import CategoryOdds, category_finish_odds


def test_symmetric_teams_give_uniform_odds():
    """10 identical teams -> each equally likely in any rank slot."""
    means = [100.0] * 10
    sds = [20.0] * 10
    odds = category_finish_odds(means, sds, 0, higher_is_better=True)
    assert odds.first_pct == pytest.approx(10.0, abs=0.5)
    assert odds.top3_pct == pytest.approx(30.0, abs=0.5)
    assert odds.clear_wins == 0
    assert odds.opponents == 9


def test_dominant_team_first_and_top3_certain():
    means = [200.0] + [100.0] * 9
    sds = [1.0] * 10
    odds = category_finish_odds(means, sds, 0, higher_is_better=True)
    assert odds.first_pct == pytest.approx(100.0, abs=0.01)
    assert odds.top3_pct == pytest.approx(100.0, abs=0.01)
    assert odds.clear_wins == 9


def test_worst_team_first_near_zero():
    means = [50.0] + [100.0] * 9
    sds = [1.0] * 10
    odds = category_finish_odds(means, sds, 0, higher_is_better=True)
    assert odds.first_pct == pytest.approx(0.0, abs=0.01)
    assert odds.clear_wins == 0


def test_inverse_category_lowest_is_best():
    """ERA-style: user has the lowest (best) ERA -> wins."""
    means = [3.00] + [4.00] * 9
    sds = [0.05] * 10
    odds = category_finish_odds(means, sds, 0, higher_is_better=False)
    assert odds.first_pct == pytest.approx(100.0, abs=0.01)
    assert odds.clear_wins == 9

    # Highest (worst) ERA -> never first.
    means_bad = [5.00] + [3.00] * 9
    odds_bad = category_finish_odds(means_bad, sds, 0, higher_is_better=False)
    assert odds_bad.first_pct == pytest.approx(0.0, abs=0.01)


def test_zero_sd_is_deterministic_rank():
    """No uncertainty -> first is 100% iff strictly best; top3 by exact rank."""
    means = [100.0, 90.0, 80.0, 70.0, 60.0, 50.0, 40.0, 30.0, 20.0, 10.0]
    sds = [0.0] * 10
    best = category_finish_odds(means, sds, 0, higher_is_better=True)
    assert best.first_pct == pytest.approx(100.0, abs=0.01)
    assert best.top3_pct == pytest.approx(100.0, abs=0.01)
    assert best.clear_wins == 9

    # 4th-best team (index 3, value 70): two strictly better than the two
    # above... actually 3 teams are above it -> not top3.
    fourth = category_finish_odds(means, sds, 3, higher_is_better=True)
    assert fourth.first_pct == pytest.approx(0.0, abs=0.01)
    assert fourth.top3_pct == pytest.approx(0.0, abs=0.01)
    assert fourth.clear_wins == 6  # clears the 6 teams strictly below it


def test_clear_wins_respects_band_overlap():
    """wins counts only opponents whose +1SD is below the user's -1SD."""
    # User 100 +/-5 -> lower bound 95. Opp A 80 +/-5 -> upper 85 (cleared).
    # Opp B 92 +/-5 -> upper 97 (overlaps, not cleared).
    means = [100.0, 80.0, 92.0]
    sds = [5.0, 5.0, 5.0]
    odds = category_finish_odds(means, sds, 0, higher_is_better=True)
    assert odds.clear_wins == 1
    assert odds.opponents == 2


def test_returns_categoryodds_dataclass():
    odds = category_finish_odds([1.0, 2.0], [0.0, 0.0], 0, higher_is_better=True)
    assert isinstance(odds, CategoryOdds)
    assert isinstance(odds.first_pct, float)
    assert isinstance(odds.clear_wins, int)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_category_odds.py -v`
Expected: FAIL (ModuleNotFoundError: no module named fantasy_baseball.category_odds).

- [ ] **Step 3: Implement the module**

Create `src/fantasy_baseball/category_odds.py`:

```python
"""Analytic per-category finish odds for the Category Bars view.

Given each team's projected mean and SD in one roto category, computes the
user team's probability of finishing 1st and top-3 in that category, plus a
count of opponents the user's +/-1 SD band clearly clears. Pure functions,
no I/O.

The probabilities integrate the user's projected normal distribution against
the opponents' normals via Gauss-Hermite quadrature, treating opponents as
independent given the user's draw -- each fantasy team's total is built from
a disjoint set of players. This is the same Gaussian model the chart's bands
come from, so the odds agree with what is drawn. Higher-is-better and inverse
(ERA/WHIP) categories are unified by negating the means for inverse cats (SD
is unchanged under negation), so every formula below reads "bigger wins".
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import erf, pi, sqrt

import numpy as np

# Gauss-Hermite nodes/weights (probabilists', weight exp(-x^2/2)) for
# E[g(X)], X ~ Normal. 24 nodes: the integrands (a product of normal CDFs and
# a Poisson-binomial CDF) are smooth and bounded, so this is far more than
# enough and still costs microseconds. Materialized to plain float tuples so
# numpy is touched only at import time.
_GH_N = 24
_gh_nodes, _gh_weights = np.polynomial.hermite_e.hermegauss(_GH_N)
_GH_NODES: tuple[float, ...] = tuple(float(x) for x in _gh_nodes)
_GH_WEIGHTS: tuple[float, ...] = tuple(float(x) for x in _gh_weights)
_SQRT_2PI = sqrt(2.0 * pi)


@dataclass
class CategoryOdds:
    """User-team odds for one category. Percentages are 0-100, unrounded."""

    first_pct: float
    top3_pct: float
    clear_wins: int
    opponents: int


def _prob_opp_above(x: float, mu: float, sd: float) -> float:
    """P(an opponent ~ N(mu, sd) exceeds the fixed value x)."""
    if sd == 0.0:
        if mu > x:
            return 1.0
        if mu < x:
            return 0.0
        return 0.5
    z = (x - mu) / sd
    cdf_below = 0.5 * (1.0 + erf(z / sqrt(2.0)))  # P(opp < x)
    return 1.0 - cdf_below


def _poisson_binomial_le2(qs: Sequence[float]) -> tuple[float, float]:
    """Return (P(k=0), P(k<=2)) for independent Bernoulli(q) opponents.

    k is the number of opponents that beat the user. Exact O(n^2) DP over the
    full pmf -- robust when a q is exactly 0 or 1.
    """
    pmf = [1.0]
    for q in qs:
        nxt = [0.0] * (len(pmf) + 1)
        for count, p in enumerate(pmf):
            nxt[count] += p * (1.0 - q)
            nxt[count + 1] += p * q
        pmf = nxt
    p0 = pmf[0]
    p_le2 = sum(pmf[: 3])
    return p0, p_le2


def category_finish_odds(
    means: Sequence[float],
    sds: Sequence[float],
    user_index: int,
    *,
    higher_is_better: bool,
) -> CategoryOdds:
    """Analytic odds the user finishes 1st / top-3 in one roto category.

    ``means``/``sds`` are parallel per-team sequences; ``user_index`` selects
    the user. ``higher_is_better`` is False for ERA/WHIP. Percentages are
    0-100 (unrounded floats).
    """
    n = len(means)
    sign = 1.0 if higher_is_better else -1.0
    mu = [sign * m for m in means]
    mu_u = mu[user_index]
    sd_u = sds[user_index]
    opponents = [(mu[i], sds[i]) for i in range(n) if i != user_index]

    e_first = 0.0
    e_top3 = 0.0
    for node, weight in zip(_GH_NODES, _GH_WEIGHTS, strict=True):
        x = mu_u + sd_u * node
        qs = [_prob_opp_above(x, mu_o, sd_o) for (mu_o, sd_o) in opponents]
        p0, p_le2 = _poisson_binomial_le2(qs)
        w = weight / _SQRT_2PI
        e_first += w * p0
        e_top3 += w * p_le2

    lower_u = mu_u - sd_u
    clear_wins = sum(1 for (mu_o, sd_o) in opponents if lower_u > mu_o + sd_o)

    return CategoryOdds(
        first_pct=float(100.0 * e_first),
        top3_pct=float(100.0 * e_top3),
        clear_wins=clear_wins,
        opponents=len(opponents),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_category_odds.py -v`
Expected: 7 passed.

- [ ] **Step 5: Lint / type / dead-code**

```
python -m ruff check src/fantasy_baseball/category_odds.py tests/test_category_odds.py
python -m ruff format --check src/fantasy_baseball/category_odds.py tests/test_category_odds.py
python -m mypy src/fantasy_baseball/category_odds.py
python -m vulture src/fantasy_baseball/category_odds.py
```
Fix any violations your change introduced (`python -m ruff format <file>` fixes formatting). If mypy flags the numpy `hermegauss` return types, the `tuple(float(x) for x in ...)` materialization should already satisfy it; if not, add a precise annotation rather than an ignore. `vulture` may report `top3_pct`/`opponents` as unused at this point (they are consumed in Task 2) -- that is expected; note it and move on.

- [ ] **Step 6: Commit**

```
git add src/fantasy_baseball/category_odds.py tests/test_category_odds.py
git commit -m "feat(scoring): analytic per-category finish odds (1st / top-3 / clear wins)"
```

---

## Task 2: Wire odds into the Category Bars payload

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`
- Test: `tests/test_web/test_season_data.py`, `tests/test_web/test_season_routes.py`

This changes the per-category payload entry from a bare row list to `{"rows": [...], "odds": {...} | None}`, so the existing season_data + route tests that index the entry as a list must be updated in the same commit to stay green.

- [ ] **Step 1: Update the season_data unit tests for the new shape + odds**

In `tests/test_web/test_season_data.py`, update the affected tests and add an odds test. Replace the bodies of these existing tests so they read `["rows"]`:

```python
def test_category_bars_normal_category_sorts_best_on_top():
    out = format_category_bars_for_display(_bars_display_dict(), _bars_display_dict())
    runs = out["current"]["R"]["rows"]
    assert [r["team"] for r in runs] == ["Hart of the Order", "SkeleThor"]
    assert runs[0]["value"] == 320
    assert runs[0]["sd"] == 25.0
    assert runs[0]["is_user"] is True


def test_category_bars_inverse_category_sorts_lowest_on_top():
    out = format_category_bars_for_display(_bars_display_dict(), _bars_display_dict())
    era = out["current"]["ERA"]["rows"]
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
    assert out["current"]["HR"]["rows"][0]["sd"] == 0.0


def test_category_bars_handles_missing_flavor():
    """Pre-refresh: a flavor's display dict may be None."""
    out = format_category_bars_for_display(None, _bars_display_dict())
    assert out["preseason"] == {}
    assert out["current"]["R"]["rows"][0]["team"] == "Hart of the Order"


def test_category_bars_attaches_user_odds():
    """Each category entry carries the user team's odds; whole-number pcts."""
    out = format_category_bars_for_display(_bars_display_dict(), _bars_display_dict())
    odds = out["current"]["R"]["odds"]
    assert set(odds.keys()) == {"first_pct", "top3_pct", "wins", "opponents"}
    assert isinstance(odds["first_pct"], int)
    assert isinstance(odds["top3_pct"], int)
    assert odds["opponents"] == 1  # _bars_display_dict has 2 teams -> 1 opponent
    assert 0 <= odds["first_pct"] <= 100


def test_category_bars_odds_none_without_user_row():
    """No is_user team -> odds is None (line hides client-side)."""
    data = {
        "teams": [
            {"name": "A", "team_key": "a", "is_user": False,
             "stats": CategoryStats(r=100), "sds": {}},
            {"name": "B", "team_key": "b", "is_user": False,
             "stats": CategoryStats(r=90), "sds": {}},
        ]
    }
    out = format_category_bars_for_display(data, data)
    assert out["current"]["R"]["odds"] is None
```

(`test_category_bars_has_both_flavors_and_all_categories` is unchanged -- it only checks the category-key set, which is unaffected.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_web/test_season_data.py -k category_bars -v`
Expected: FAIL (entries are still bare lists -> `out["current"]["R"]["rows"]` raises TypeError / KeyError; `odds` tests fail).

- [ ] **Step 3: Implement the formatter change**

In `src/fantasy_baseball/web/season_data.py`:

1. Add the import (near the other `fantasy_baseball.*` imports at the top):
```python
from fantasy_baseball.category_odds import category_finish_odds
```

2. Replace `_category_bars_one_flavor` and update `format_category_bars_for_display`'s return annotation/docstring. New `_category_bars_one_flavor` plus a new private `_category_odds` helper:

```python
def _category_odds(rows: list[dict], cat: Category) -> dict | None:
    """User-team finish odds for one category's rows, or None if no user row.

    Percentages are rounded to whole numbers (display spec). ``rows`` may be
    in any order; ``category_finish_odds`` works over the team set.
    """
    user_index = next((i for i, r in enumerate(rows) if r["is_user"]), None)
    if user_index is None:
        return None
    odds = category_finish_odds(
        [r["value"] for r in rows],
        [r["sd"] for r in rows],
        user_index,
        higher_is_better=cat not in INVERSE_CATS,
    )
    return {
        "first_pct": round(odds.first_pct),
        "top3_pct": round(odds.top3_pct),
        "wins": odds.clear_wins,
        "opponents": odds.opponents,
    }


def _category_bars_one_flavor(data: dict | None) -> dict[str, dict]:
    """Reshape one standings display dict into per-category ranked rows + odds.

    Each category maps to ``{"rows": [...], "odds": {...} | None}``. ``rows``
    are ``{team, value, sd, is_user}`` sorted best-on-top: counting/AVG
    descending, ERA/WHIP ascending (lower is better). ``sd`` defaults to 0.0
    when a category is absent from a team's ``sds``. ``odds`` carries the user
    team's whole-number 1st/top-3 percentages, clear-win count, and opponent
    count (None when there is no user team in the data).
    """
    if not data or not data.get("teams"):
        return {}
    out: dict[str, dict] = {}
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
        out[cat.value] = {"rows": rows, "odds": _category_odds(rows, cat)}
    return out
```

3. Update `format_category_bars_for_display`'s signature annotation + docstring:

```python
def format_category_bars_for_display(
    preseason_data: dict | None,
    current_projected_data: dict | None,
) -> dict[str, dict[str, dict]]:
    """Build the Category Bars chart payload from the two standings display dicts.

    Returns ``{"preseason": {CAT: {"rows": [...], "odds": {...}}}, "current": {...}}``
    where each row is ``{team, value, sd, is_user}`` sorted best-on-top and
    ``odds`` is the user team's per-category finish odds (see
    ``_category_bars_one_flavor``). A missing flavor (``None``, pre-refresh)
    yields an empty ``{}`` for that flavor.
    """
    return {
        "preseason": _category_bars_one_flavor(preseason_data),
        "current": _category_bars_one_flavor(current_projected_data),
    }
```

- [ ] **Step 4: Run the season_data tests to verify they pass**

Run: `python -m pytest tests/test_web/test_season_data.py -k category_bars -v`
Expected: all pass (the 4 updated + 2 new + the unchanged keys test).

- [ ] **Step 5: Update the route test for the new shape**

In `tests/test_web/test_season_routes.py`, in `test_standings_embeds_category_bars_data`, replace the final assertions block (the part after `bars = json.loads(match.group(1))`) with:

```python
    # Each category entry is now {"rows": [...], "odds": {...}}.
    assert bars["current"]["R"]["rows"], "current/R rows should be non-empty"
    assert bars["preseason"]["R"]["rows"], "preseason/R rows should be non-empty"

    # Best-on-top for a normal category: Hart (320 R) ranks first, inside the
    # category-bars JSON specifically (not just elsewhere on the page).
    top_runs = bars["current"]["R"]["rows"][0]
    assert top_runs["team"] == "Hart of the Order"
    assert top_runs["value"] == 320.0
    assert top_runs["sd"] == 25.0
    assert top_runs["is_user"] is True

    # Best-on-top for an inverse category: lowest ERA ranks first.
    assert bars["current"]["ERA"]["rows"][0]["team"] == "SkeleThor"

    # The user team's per-category odds ride along.
    odds = bars["current"]["R"]["odds"]
    assert set(odds.keys()) == {"first_pct", "top3_pct", "wins", "opponents"}
    assert odds["opponents"] == 1  # 2 teams seeded -> 1 opponent
```

(`test_standings_category_bars_empty_without_projections` is unchanged -- empty flavors stay `{"preseason": {}, "current": {}}`.)

- [ ] **Step 6: Run the route tests + full season_data**

Run: `python -m pytest tests/test_web/test_season_routes.py -k standings tests/test_web/test_season_data.py -q`
Expected: all pass.

- [ ] **Step 7: Lint / type**

```
python -m ruff check src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py tests/test_web/test_season_routes.py
python -m ruff format --check src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py tests/test_web/test_season_routes.py
python -m mypy src/fantasy_baseball/web/season_data.py
```
Expected: clean. (`vulture` on category_odds.py now shows no unused fields, since `top3_pct`/`opponents` are consumed here.)

- [ ] **Step 8: Commit**

```
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py tests/test_web/test_season_routes.py
git commit -m "feat(standings): attach per-category user odds to Category Bars payload"
```

---

## Task 3: Render the odds line (JS + template)

**Files:**
- Modify: `src/fantasy_baseball/web/static/season_category_bars.js`
- Modify: `src/fantasy_baseball/web/templates/season/standings.html`

No JS test harness exists; verified by `node --check` + the route render test + the manual browser check in Task 4.

- [ ] **Step 1: Update the JS to read the new shape and populate the odds line**

In `src/fantasy_baseball/web/static/season_category_bars.js`:

1. Replace the `rowsFor` function with entry/rows/odds accessors:

```javascript
  function entryFor() {
    if (!payload) return null;
    var flavor = payload[state.projection];
    if (!flavor) return null;
    return flavor[state.category] || null;
  }

  function rowsFor() {
    var entry = entryFor();
    return entry && entry.rows ? entry.rows : [];
  }

  function oddsFor() {
    var entry = entryFor();
    return entry ? entry.odds : null;
  }

  function setText(id, text) {
    var el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function updateOdds() {
    var box = document.getElementById("catbars-odds");
    if (!box) return;
    var odds = oddsFor();
    if (!odds) {
      box.style.display = "none";
      return;
    }
    box.style.display = "";
    setText("catbars-first", odds.first_pct + "%");
    setText("catbars-top3", odds.top3_pct + "%");
    setText("catbars-wins", odds.wins + "/" + odds.opponents);
  }
```

2. In `render()`, call `updateOdds()` immediately after the payload is (lazily) loaded, so the line updates for both the chart and empty paths. Change the top of `render()` from:

```javascript
  function render() {
    if (payload == null) payload = loadPayload();
    var rows = rowsFor();
```

to:

```javascript
  function render() {
    if (payload == null) payload = loadPayload();
    updateOdds();
    var rows = rowsFor();
```

- [ ] **Step 2: Add the odds line + styling to the template**

In `src/fantasy_baseball/web/templates/season/standings.html`, insert the odds line between the category toggle (`#catbars-cat-toggle`) and the chart wrapper. Change:

```html
    <div class="pill-group" id="catbars-cat-toggle">
        {% for cat in all_categories %}
        <button class="pill {% if loop.first %}active{% endif %}"
                data-cbcat="{{ cat.value }}" onclick="catBarsSetCategory(this)">{{ cat.value }}</button>
        {% endfor %}
    </div>

    <div class="catbars-wrapper"><canvas id="category-bars-canvas"></canvas></div>
```

to:

```html
    <div class="pill-group" id="catbars-cat-toggle">
        {% for cat in all_categories %}
        <button class="pill {% if loop.first %}active{% endif %}"
                data-cbcat="{{ cat.value }}" onclick="catBarsSetCategory(this)">{{ cat.value }}</button>
        {% endfor %}
    </div>

    <div id="catbars-odds" class="catbars-odds" style="display: none;">
        <span><strong>1st:</strong> <span id="catbars-first">--</span></span>
        <span><strong>top 3:</strong> <span id="catbars-top3">--</span></span>
        <span><strong>wins:</strong> <span id="catbars-wins">--</span></span>
    </div>

    <div class="catbars-wrapper"><canvas id="category-bars-canvas"></canvas></div>
```

Then add a CSS rule next to the existing `#catbars-cat-toggle` rule in the inline `<style>` block (right after the `#catbars-cat-toggle .pill:first-child` line):

```css
.catbars-odds {
    display: flex;
    gap: 1.5rem;
    margin: 0.25rem 0 0;
    font-size: 0.95rem;
    color: var(--ink);
}
.catbars-odds strong { color: var(--ink-soft); font-weight: 600; }
```

- [ ] **Step 3: Verify JS syntax + ASCII + page still renders**

```
node --check src/fantasy_baseball/web/static/season_category_bars.js
rg -n "[^\x00-\x7f]" src/fantasy_baseball/web/static/season_category_bars.js   # expect no NEW matches in added lines
python -m pytest tests/test_web/test_season_routes.py -k standings -q
```
Expected: `node --check` prints nothing (OK); no non-ASCII in the JS; standings route tests pass.

- [ ] **Step 4: Commit**

```
git add src/fantasy_baseball/web/static/season_category_bars.js src/fantasy_baseball/web/templates/season/standings.html
git commit -m "feat(standings): show 1st/top-3/wins odds line above the Category Bars chart"
```

---

## Task 4: Full verification + manual smoke

**Files:** none (verification; commit any fixes the checks force).

- [ ] **Step 1: Run the touched test areas**

Run: `python -m pytest tests/test_category_odds.py tests/test_web/test_season_data.py tests/test_web/test_season_routes.py -q`
Expected: all pass.

- [ ] **Step 2: Lint / format / dead-code / types (feature files)**

```
python -m ruff check src/fantasy_baseball/category_odds.py src/fantasy_baseball/web/season_data.py tests/test_category_odds.py tests/test_web/test_season_data.py tests/test_web/test_season_routes.py
python -m ruff format --check src/fantasy_baseball/category_odds.py src/fantasy_baseball/web/season_data.py tests/test_category_odds.py tests/test_web/test_season_data.py tests/test_web/test_season_routes.py
python -m vulture src/fantasy_baseball/category_odds.py src/fantasy_baseball/web/season_data.py
python -m mypy src/fantasy_baseball/category_odds.py src/fantasy_baseball/web/season_data.py
```
Expected: zero violations; no new vulture findings; mypy success. If `[tool.mypy].files` is an explicit enumerated list that includes `season_data.py`, add `src/fantasy_baseball/category_odds.py` to it so the new module stays type-checked, then re-run mypy.

- [ ] **Step 3: Manual dashboard smoke**

Run: `python scripts/run_season_dashboard.py`, open `/standings` -> Category Bars (with cached projections present). Confirm:
1. A line `1st: N%   top 3: N%   wins: x/9` sits between the category buttons and the chart.
2. The numbers change when you switch category and when you toggle Current/Preseason.
3. The percentages are whole numbers; `wins` shows `x/9` in a 10-team league.
4. Sanity: for a category where your dot is clearly highest with a tight band, `1st` is high and `wins` is large; for a tightly-bunched category, `1st` is small.
5. Browser console shows no errors.

- [ ] **Step 4: Final commit (only if Steps 1-2 forced fixes)**

```
git add -A
git commit -m "fix(standings): address Category Odds verification findings"
```

---

## Self-Review notes

- **Spec coverage:** the math (P first/top3 via Gauss-Hermite, Poisson-binomial, sign-flip for inverse, edge cases) -> Task 1; payload `{rows, odds}` + analytic computation reuse -> Task 2; display line between selector and chart, whole-number percents, hide-when-null -> Task 3; consistency-with-bands is inherent (same means/SDs); testing + verification -> Tasks 1-4. The spec's "place in scoring.py" is deliberately refined to a dedicated `category_odds.py` module (keeps numpy out of scoring's hot import path); behavior is identical and the function/signature match the spec's `category_finish_odds` / `CategoryOdds`.
- **Type/name consistency:** `category_finish_odds(means, sds, user_index, *, higher_is_better) -> CategoryOdds(first_pct, top3_pct, clear_wins, opponents)` is identical across Task 1 (def + tests), Task 2 (call), and the spec. Payload keys `first_pct/top3_pct/wins/opponents` match across Task 2 (producer), the route test, and Task 3 (`odds.first_pct`, `odds.top3_pct`, `odds.wins`, `odds.opponents`). The `{rows, odds}` entry shape is consistent across Task 2 (producer + tests) and Task 3 (`entry.rows`, `entry.odds`).
- **Placeholder scan:** none -- every code step has complete code.
- **Known un-automatable item:** the rendered odds line (DOM text + visibility) is verified manually in Task 4; everything server-side and the payload are covered by tests.
```
