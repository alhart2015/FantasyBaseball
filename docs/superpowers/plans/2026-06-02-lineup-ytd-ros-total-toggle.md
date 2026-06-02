# Lineup YTD / ROS / Total Toggle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a three-way YTD / ROS / Total toggle to the `/lineup` page that re-bases each player's stat-cell values, SGP, rank badge, and row sort, while leaving the pace tooltip / z-score / +/- deviation corner unchanged.

**Architecture:** Display-only, server-side re-render (Approach B). All three bases come from data already on each cached `Player`: ROS = `rest_of_season`, Total = `full_season_projection`, YTD = `full_season_projection - rest_of_season`. The only new computation is a full-season leaguewide rank in the refresh pipeline. A small partial endpoint re-renders the two `<tbody>` fragments per basis; a vanilla-JS toggle swaps them in.

**Tech Stack:** Python 3, Flask + Jinja2, pandas, pytest. Windows dev box (ASCII-only in Python source/log strings; templates render UTF-8 HTML and may keep existing non-ASCII glyphs).

**Spec:** `docs/superpowers/specs/2026-06-02-lineup-ytd-ros-total-toggle-design.md`

---

## File structure

- `src/fantasy_baseball/models/player.py` — `RankInfo` gains a `total` field.
- `src/fantasy_baseball/sgp/rankings.py` — `build_rankings_lookup` gains a `total` input.
- `src/fantasy_baseball/web/refresh_pipeline.py` — `_compute_rankings` computes + threads the total ranking.
- `src/fantasy_baseball/web/season_data.py` — YTD derivation helper + `basis` param in `format_lineup_for_display`.
- `src/fantasy_baseball/web/templates/season/macros.html` — `rank_badge` renders a basis rank + a Total tooltip row.
- `src/fantasy_baseball/web/templates/season/_lineup_hitters_tbody.html` / `_lineup_pitchers_tbody.html` — cell main value reads `display_stats`; rank badge passes `rank_display`.
- `src/fantasy_baseball/web/templates/season/lineup.html` — toggle UI + JS handler.
- `src/fantasy_baseball/web/season_routes.py` — `/lineup?basis=` + new `/lineup/tbodies` partial endpoint.
- Tests: `tests/test_sgp/test_rankings.py`, `tests/test_models/test_player.py` (or existing player test), `tests/test_web/test_season_data.py`, `tests/test_web/test_season_routes.py`.

---

## Task 1: `RankInfo.total` + `build_rankings_lookup` total input

**Files:**
- Modify: `src/fantasy_baseball/models/player.py:117-136`
- Modify: `src/fantasy_baseball/sgp/rankings.py:120-140`
- Test: `tests/test_sgp/test_rankings.py`, `tests/test_models/test_player.py`

- [ ] **Step 1: Write the failing test for `RankInfo.total`**

In `tests/test_models/test_player.py` (create if absent, with `from fantasy_baseball.models.player import RankInfo` at top):

```python
def test_rankinfo_total_roundtrips():
    r = RankInfo.from_dict({"rest_of_season": 1, "preseason": 2, "current": 3, "total": 4})
    assert r.total == 4
    assert r.to_dict()["total"] == 4


def test_rankinfo_total_defaults_none():
    r = RankInfo.from_dict({"rest_of_season": 1})
    assert r.total is None
    assert r.to_dict()["total"] is None
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_models/test_player.py -k rankinfo_total -v`
Expected: FAIL — `RankInfo` has no `total` (TypeError on unexpected key or AttributeError).

- [ ] **Step 3: Add the `total` field to `RankInfo`**

In `player.py`, edit the `RankInfo` dataclass:

```python
@dataclass
class RankInfo:
    rest_of_season: int | None = None
    preseason: int | None = None
    current: int | None = None
    total: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RankInfo:
        return cls(
            rest_of_season=d.get("rest_of_season"),
            preseason=d.get("preseason"),
            current=d.get("current"),
            total=d.get("total"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rest_of_season": self.rest_of_season,
            "preseason": self.preseason,
            "current": self.current,
            "total": self.total,
        }
```

- [ ] **Step 4: Run the RankInfo tests to verify they pass**

Run: `pytest tests/test_models/test_player.py -k rankinfo_total -v`
Expected: PASS

- [ ] **Step 5: Write the failing test for `build_rankings_lookup` total**

In `tests/test_sgp/test_rankings.py`, add:

```python
def test_build_rankings_lookup_includes_total():
    result = build_rankings_lookup(
        {"a::hitter": 1}, {"a::hitter": 2}, {"a::hitter": 3}, {"a::hitter": 4}
    )
    assert result["a::hitter"] == {
        "rest_of_season": 1,
        "preseason": 2,
        "current": 3,
        "total": 4,
    }


def test_build_rankings_lookup_total_defaults_none():
    result = build_rankings_lookup({"a::hitter": 1}, {}, {})
    assert result["a::hitter"]["total"] is None
```

- [ ] **Step 6: Run it to verify it fails**

Run: `pytest tests/test_sgp/test_rankings.py -k build_rankings_lookup_includes_total -v`
Expected: FAIL — `build_rankings_lookup()` takes 3 positional args.

- [ ] **Step 7: Add the optional `total` param to `build_rankings_lookup`**

In `rankings.py`, replace the function:

```python
def build_rankings_lookup(
    ros: dict[str, Any],
    preseason: dict[str, Any],
    current: dict[str, Any],
    total: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Four-way merge of player ranking dicts keyed by ``name::player_type``.

    The output maps each player key to a dict with four keys
    (``rest_of_season``, ``preseason``, ``current``, ``total``); missing
    entries are ``None``. ``total`` is optional so legacy 3-arg callers keep
    working (they get ``total=None`` for every player).
    """
    total = total or {}
    all_keys = set(ros) | set(preseason) | set(current) | set(total)
    return {
        key: {
            "rest_of_season": ros.get(key),
            "preseason": preseason.get(key),
            "current": current.get(key),
            "total": total.get(key),
        }
        for key in all_keys
    }
```

- [ ] **Step 8: Update the existing exact-shape assertions in `test_rankings.py`**

The contract is intentionally extended, so these existing assertions must now
include `"total"`. This is a deliberate change (not a silent loosening).
Update each of these tests to add `"total": None` to the expected dict:

- `test_player_in_all_three`
- `test_player_only_in_ros_has_none_for_others`
- `test_player_only_in_preseason_has_none_for_others`
- `test_player_only_in_current_has_none_for_others`
- `test_union_includes_keys_from_all_three` (add `"total": None` to each expected entry)

For example, `test_player_in_all_three` becomes:

```python
def test_player_in_all_three(self):
    ros = {"juan soto::hitter": 1}
    pre = {"juan soto::hitter": 2}
    cur = {"juan soto::hitter": 3}
    result = build_rankings_lookup(ros, pre, cur)
    assert result["juan soto::hitter"] == {
        "rest_of_season": 1,
        "preseason": 2,
        "current": 3,
        "total": None,
    }
```

Apply the analogous `"total": None` addition to the other four tests (read each
and add the key to its expected dict).

- [ ] **Step 9: Run the full rankings + player test files**

Run: `pytest tests/test_sgp/test_rankings.py tests/test_models/test_player.py -v`
Expected: PASS (all)

- [ ] **Step 10: Commit**

```bash
git add src/fantasy_baseball/models/player.py src/fantasy_baseball/sgp/rankings.py tests/test_sgp/test_rankings.py tests/test_models/test_player.py
git commit -m "feat(rankings): add total (full-season) rank to RankInfo + build_rankings_lookup"
```

---

## Task 2: Compute the full-season ranking in the refresh pipeline

**Files:**
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py:1044-1073`
- Test: `tests/test_web/test_refresh_pipeline.py` (regression run only — see Step 3)

- [ ] **Step 1: Add the total ranking computation and thread it through**

In `refresh_pipeline.py::_compute_rankings`, after the `current_ranks = ...`
line and before `build_rankings_lookup(...)`, compute the total ranking from
the full-season pools (already populated on `self`), then pass it in:

```python
        rest_of_season_ranks = compute_sgp_rankings(self.hitters_proj, self.pitchers_proj)
        preseason_ranks = compute_sgp_rankings(self.preseason_hitters, self.preseason_pitchers)
        current_ranks = compute_rankings_from_game_logs(self.hitter_logs, self.pitcher_logs)

        # Full-season (YTD + ROS) leaguewide ranking. The full-season pools are
        # already derived earlier in the pipeline; if they are absent (e.g. no
        # ROS projections), fall back to an empty ranking so rank.total is None.
        if self.full_hitters_proj is not None and self.full_pitchers_proj is not None:
            total_ranks = compute_sgp_rankings(self.full_hitters_proj, self.full_pitchers_proj)
        else:
            total_ranks = {}

        from fantasy_baseball.sgp.rankings import build_rankings_lookup

        self.rankings_lookup = build_rankings_lookup(
            rest_of_season_ranks,
            preseason_ranks,
            current_ranks,
            total_ranks,
        )
```

Also update the progress line to include the total count:

```python
        self._progress(
            f"Ranked {len(rest_of_season_ranks)} ROS, {len(preseason_ranks)} preseason, "
            f"{len(current_ranks)} current, {len(total_ranks)} total"
        )
```

- [ ] **Step 2: Verify `lookup_rank` already returns `total`**

No change needed: `lookup_rank` returns the per-player dict from
`rankings_lookup`, which now carries `total`; `RankInfo.from_dict` (Task 1)
reads it. Confirm by reading `refresh_pipeline.py:1081-1085` — it calls
`lookup_rank(...)` then `RankInfo.from_dict(rank_data)`.

- [ ] **Step 3: Run the refresh pipeline test suite (regression)**

Run: `pytest tests/test_web/test_refresh_pipeline.py -v`
Expected: PASS — the wiring change is additive; existing tests must stay green.
(The new `total` path is unit-covered by Task 1; this step guards regressions.)

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/refresh_pipeline.py
git commit -m "feat(refresh): compute full-season (total) leaguewide rank"
```

---

## Task 3: YTD derivation + `basis` param in `format_lineup_for_display`

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:855-940`
- Test: `tests/test_web/test_season_data.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_web/test_season_data.py`, add (imports at top:
`from fantasy_baseball.web.season_data import format_lineup_for_display`):

```python
def _hitter_entry():
    # Cached-roster flat dict shape consumed by Player.from_dict. ROS is the
    # remaining-season projection; full = ROS + YTD actuals; rank carries all
    # four bases.
    return {
        "name": "Test Hitter",
        "player_type": "hitter",
        "positions": ["OF", "UTIL"],
        "selected_position": "OF",
        "player_id": "1001",
        "status": "",
        "rest_of_season": {"pa": 300, "ab": 270, "h": 75, "r": 40, "hr": 12, "rbi": 38, "sb": 8},
        "full_season_projection": {"pa": 500, "ab": 450, "h": 130, "r": 70, "hr": 22, "rbi": 66, "sb": 14},
        "rank": {"rest_of_season": 25, "preseason": 30, "current": 18, "total": 21},
        "pace": {
            "HR": {"actual": 10, "expected": 9, "z_score": 0.5, "color_class": "stat-up",
                   "rest_of_season_deviation_sgp": 1, "projection": 20},
        },
    }


def test_basis_rebases_cells_sgp_and_rank():
    roster = [_hitter_entry()]
    ros = format_lineup_for_display(roster, None, basis="ros")["hitters"][0]
    ytd = format_lineup_for_display(roster, None, basis="ytd")["hitters"][0]
    tot = format_lineup_for_display(roster, None, basis="total")["hitters"][0]

    # Cell values follow the basis (HR: ros=12, ytd=full-ros=22-12=10, total=22)
    assert ros["display_stats"]["HR"] == 12
    assert ytd["display_stats"]["HR"] == 10
    assert tot["display_stats"]["HR"] == 22

    # Rank badge follows the basis
    assert ros["rank_display"] == 25
    assert ytd["rank_display"] == 18
    assert tot["rank_display"] == 21

    # SGP differs across bases and ROS sgp is positive
    assert ros["sgp"] != ytd["sgp"]
    assert ros["sgp"] is not None

    # Pace payload is IDENTICAL across bases (tooltip/z-score/deviation unchanged)
    assert ros["pace"] == ytd["pace"] == tot["pace"]


def test_basis_unknown_falls_back_to_ros():
    roster = [_hitter_entry()]
    weird = format_lineup_for_display(roster, None, basis="bogus")["hitters"][0]
    ros = format_lineup_for_display(roster, None, basis="ros")["hitters"][0]
    assert weird["display_stats"] == ros["display_stats"]
    assert weird["rank_display"] == ros["rank_display"]


def test_ytd_zero_for_unplayed_player():
    entry = _hitter_entry()
    # full == ros  -> no YTD production
    entry["full_season_projection"] = dict(entry["rest_of_season"])
    ytd = format_lineup_for_display([entry], None, basis="ytd")["hitters"][0]
    # No PA so cells render as None (template shows "--")
    assert ytd["display_stats"]["HR"] is None
    assert ytd["sgp"] == 0.0
```

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_web/test_season_data.py -k basis -v`
Expected: FAIL — `format_lineup_for_display()` has no `basis` param / no
`display_stats` / `rank_display` keys.

- [ ] **Step 3: Add the YTD derivation helper + display-map helper**

In `season_data.py`, above `format_lineup_for_display`, add module-level
helpers (import `HitterStats, PitcherStats, PlayerType` at top of the function
or module as the file already imports `Player`):

```python
def _derive_ytd_stats(full, ros, player_type):
    """YTD actuals = full_season - rest_of_season, per counting component,
    clamped at >= 0. Rate stats (avg/era/whip) are recomputed from the
    subtracted components by the stats from_dict constructors. Returns a
    HitterStats/PitcherStats, or None when either input is missing."""
    from fantasy_baseball.models.player import HitterStats, PitcherStats, PlayerType

    if full is None or ros is None:
        return None
    if player_type == PlayerType.HITTER:
        cols = ["pa", "ab", "h", "r", "hr", "rbi", "sb"]
        d = {k: max(0.0, getattr(full, k) - getattr(ros, k)) for k in cols}
        return HitterStats.from_dict(d)
    cols = ["ip", "w", "k", "sv", "er", "bb", "h_allowed"]
    d = {k: max(0.0, getattr(full, k) - getattr(ros, k)) for k in cols}
    return PitcherStats.from_dict(d)


def _display_map(stats, player_type, basis):
    """Per-category display values for the chosen basis, keyed by the same
    uppercase category names the tbody templates loop over. For the YTD basis,
    a player with zero volume (PA/IP) yields all-None values so the template
    renders '--' (matching today's no-games appearance)."""
    from fantasy_baseball.models.player import PlayerType

    if stats is None:
        return {}
    if player_type == PlayerType.HITTER:
        m = {"PA": stats.pa, "R": stats.r, "HR": stats.hr, "RBI": stats.rbi,
             "SB": stats.sb, "AVG": stats.avg}
        volume = stats.pa
    else:
        m = {"IP": stats.ip, "W": stats.w, "K": stats.k, "SV": stats.sv,
             "ERA": stats.era, "WHIP": stats.whip}
        volume = stats.ip
    if basis == "ytd" and volume == 0:
        return {k: None for k in m}
    return m
```

- [ ] **Step 4: Add the `basis` param and per-basis selection in `format_lineup_for_display`**

Change the signature:

```python
def format_lineup_for_display(roster: list[dict], optimal: dict | None, basis: str = "ros") -> dict:
```

Just after `entry["sgp"] = ros_sgp` (the line near 916 that re-pins ros_sgp
after the flatten), insert the per-basis selection. Replace that single line
with:

```python
        # --- Per-basis selection (display-only): ROS / YTD / Total ---
        from fantasy_baseball.models.player import PlayerType

        if basis not in ("ros", "ytd", "total"):
            basis = "ros"

        ros_stats = player.rest_of_season
        full_stats = player.full_season_projection or player.rest_of_season
        ytd_stats = _derive_ytd_stats(player.full_season_projection, player.rest_of_season, player.player_type)

        sgp_total = full_stats.compute_sgp() if full_stats is not None else None
        sgp_ytd = ytd_stats.compute_sgp() if ytd_stats is not None else 0.0

        basis_choice = {
            "ros": (ros_stats, ros_sgp, player.rank.rest_of_season),
            "ytd": (ytd_stats, sgp_ytd, player.rank.current),
            "total": (full_stats, sgp_total, player.rank.total),
        }
        sel_stats, sel_sgp, sel_rank = basis_choice[basis]

        entry["sgp"] = sel_sgp
        entry["rank_display"] = sel_rank
        entry["display_stats"] = _display_map(sel_stats, player.player_type, basis)
```

Note: `ros_sgp` is computed earlier in the function (the existing
`ros_sgp = player.rest_of_season.sgp if ... else ... compute_sgp()` block);
keep that block. The sort lines at the end already sort by `entry["sgp"]`
(now the basis SGP), so no sort change is needed.

- [ ] **Step 5: Run the basis tests to verify they pass**

Run: `pytest tests/test_web/test_season_data.py -k basis -v`
Expected: PASS

- [ ] **Step 6: Update any existing `format_lineup_for_display` tests that asserted YTD-actual cell values under the default**

The default ROS view's stat cells now show ROS-remaining projections (decision
5 in the spec). Read `tests/test_web/test_season_data.py` for existing
assertions that the default lineup cell shows the YTD actual; update them to
assert `display_stats` on the intended basis. This is an intended behavior
change — update deliberately, do not delete coverage. Run the whole file:

Run: `pytest tests/test_web/test_season_data.py -v`
Expected: PASS (after deliberate updates)

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py
git commit -m "feat(lineup): per-basis SGP/rank/cell selection in format_lineup_for_display"
```

---

## Task 4: Templates — rank badge, tbody cells, toggle UI

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/macros.html:21-31`
- Modify: `src/fantasy_baseball/web/templates/season/_lineup_hitters_tbody.html:49,58-61`
- Modify: `src/fantasy_baseball/web/templates/season/_lineup_pitchers_tbody.html:43,49-52`
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html`

This task has no unit test (Jinja markup); it is verified end-to-end by the
route test in Task 5 and by manual rendering. Commit at the end.

- [ ] **Step 1: Update `rank_badge` to accept a basis rank + show a Total tooltip row**

Replace the `rank_badge` macro in `macros.html` with:

```jinja
{%- macro rank_badge(rank_obj, display_rank=none) -%}
{%- set shown = display_rank if display_rank is not none else (rank_obj.rest_of_season if rank_obj else none) -%}
{%- if shown -%}
<span class="rank-badge">#{{ shown }}
    <span class="rank-tooltip">
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">ROS</span><span>{{ "#" ~ rank_obj.rest_of_season if rank_obj.rest_of_season else "—" }}</span></div>
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">Preseason</span><span>{{ "#" ~ rank_obj.preseason if rank_obj.preseason else "—" }}</span></div>
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">Current</span><span>{{ "#" ~ rank_obj.current if rank_obj.current else "—" }}</span></div>
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">Total</span><span>{{ "#" ~ rank_obj.total if rank_obj.total else "—" }}</span></div>
    </span>
</span>
{%- endif -%}
{%- endmacro -%}
```

(Opponent partials call `rank_badge(p.rank)` with no second arg, so they fall
back to `rest_of_season` exactly as today.)

- [ ] **Step 2: Update the hitter tbody — rank badge call + cell main value**

In `_lineup_hitters_tbody.html`:

Change the name cell (line 49) from `{{ rank_badge(p.rank) }}` to:

```jinja
<td style="text-align: left; font-weight: 500;">{{ p.name }} {{ rank_badge(p.rank, p.get('rank_display')) }}</td>
```

Replace the cell main-value block (currently lines 59-61, the
`{% if st and st.get('actual') ... %}` ... `{% else %}—{% endif %}` that
precedes the tooltip `{% if st and st.get('z_score') ... %}`) with:

```jinja
            {% set dv = p.display_stats.get(cat) if p.display_stats else none %}
            {% if dv is not none %}
                {% if cat == 'AVG' %}{{ "%.3f"|format(dv) }}{% else %}{{ dv | round | int }}{% endif %}
            {% else %}—{% endif %}
```

Leave the `<td class=...>` line (deviation classes from `st`) and the tooltip
block (`{% if st and st.get('z_score') ... %}`) unchanged.

- [ ] **Step 3: Update the pitcher tbody — rank badge call + cell main value**

In `_lineup_pitchers_tbody.html`:

Change the name cell (line 43) to:

```jinja
<td style="text-align: left; font-weight: 500;">{{ p.name }} {{ rank_badge(p.rank, p.get('rank_display')) }}</td>
```

Replace the cell main-value block (currently lines 50-52) with:

```jinja
            {% set dv = p.display_stats.get(cat) if p.display_stats else none %}
            {% if dv is not none %}
                {% if cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(dv) }}{% elif cat == 'IP' %}{{ dv | format_ip }}{% else %}{{ dv | round | int }}{% endif %}
            {% else %}—{% endif %}
```

Leave the deviation `<td class=...>` line and the tooltip block unchanged.

- [ ] **Step 4: Add the toggle UI + styles to `lineup.html`**

Add inside the `.page-header` div (after the week-label block, before its
closing `</div>` at line 37) a basis toggle:

```jinja
    <div class="basis-toggle" role="group" aria-label="Stat basis" style="margin-left: 12px;">
        <button type="button" data-basis="ytd" class="{% if basis == 'ytd' %}active{% endif %}" onclick="onBasisChange('ytd')">YTD</button>
        <button type="button" data-basis="ros" class="{% if basis == 'ros' %}active{% endif %}" onclick="onBasisChange('ros')">ROS</button>
        <button type="button" data-basis="total" class="{% if basis == 'total' %}active{% endif %}" onclick="onBasisChange('total')">Total</button>
    </div>
```

Add to the page `<style>` block (lines 6-15):

```css
.basis-toggle { display: inline-flex; border: 1px solid var(--panel-border); border-radius: 6px; overflow: hidden; }
.basis-toggle button { background: var(--panel-bg); color: var(--text-secondary); border: none; padding: 6px 12px; font-size: 13px; cursor: pointer; }
.basis-toggle button + button { border-left: 1px solid var(--panel-border); }
.basis-toggle button.active { background: var(--accent, #2d6cdf); color: #fff; }
```

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/macros.html src/fantasy_baseball/web/templates/season/_lineup_hitters_tbody.html src/fantasy_baseball/web/templates/season/_lineup_pitchers_tbody.html src/fantasy_baseball/web/templates/season/lineup.html
git commit -m "feat(lineup): basis toggle UI + basis-aware rank badge and stat cells"
```

---

## Task 5: Routes — `/lineup?basis=` + `/lineup/tbodies` partial endpoint

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py:589-638`
- Test: `tests/test_web/test_season_routes.py`

- [ ] **Step 1: Write the failing test**

In `tests/test_web/test_season_routes.py`, follow the existing app-fixture
pattern in that file (find how it builds a Flask test client and seeds the
cache). Add:

```python
def test_lineup_accepts_basis_param(client):
    resp = client.get("/lineup?basis=ytd")
    assert resp.status_code == 200


def test_lineup_tbodies_returns_html_for_basis(client):
    resp = client.get("/lineup/tbodies?basis=total")
    assert resp.status_code in (200, 404)  # 404 only if no roster seeded
    if resp.status_code == 200:
        data = resp.get_json()
        assert data["basis"] == "total"
        assert "hitters_html" in data
        assert "pitchers_html" in data


def test_lineup_tbodies_unknown_basis_falls_back(client):
    resp = client.get("/lineup/tbodies?basis=bogus")
    if resp.status_code == 200:
        assert resp.get_json()["basis"] == "ros"
```

(If the existing tests seed a roster into the cache fixture, reuse that so the
200 branch is exercised. Match the file's existing client fixture name.)

- [ ] **Step 2: Run it to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py -k "basis or tbodies" -v`
Expected: FAIL — `/lineup/tbodies` route does not exist (404 for the param
test is acceptable; the tbodies route returning JSON is what fails).

- [ ] **Step 3: Add the `basis` param to `/lineup` and the new partial endpoint**

In `season_routes.py`, in the `lineup()` view, read and validate the basis and
pass it through:

```python
    @app.route("/lineup")
    def lineup():
        from fantasy_baseball.streaks.indicator import build_indicator

        basis = request.args.get("basis", "ros")
        if basis not in ("ros", "ytd", "total"):
            basis = "ros"

        meta = read_meta()
        roster_raw = read_cache_list(CacheKey.ROSTER)
        optimal_raw = read_cache_dict(CacheKey.LINEUP_OPTIMAL)
        starters_raw = read_cache_list(CacheKey.PROBABLE_STARTERS)
        pending_moves_raw = read_cache_list(CacheKey.PENDING_MOVES) or []
        streak_payload = read_cache_dict(CacheKey.STREAK_SCORES)

        lineup_data = None
        if roster_raw:
            from fantasy_baseball.web.season_data import format_lineup_for_display

            lineup_data = format_lineup_for_display(roster_raw, optimal_raw, basis=basis)
            for hitter in lineup_data["hitters"]:
                hitter["streak_indicator"] = build_indicator(hitter["name"], streak_payload)
```

Add `basis=basis` to the `render_template("season/lineup.html", ...)` kwargs.

Then add the partial endpoint immediately after the `lineup()` view:

```python
    @app.route("/lineup/tbodies")
    def lineup_tbodies():
        from fantasy_baseball.streaks.indicator import build_indicator
        from fantasy_baseball.web.season_data import format_lineup_for_display

        basis = request.args.get("basis", "ros")
        if basis not in ("ros", "ytd", "total"):
            basis = "ros"

        roster_raw = read_cache_list(CacheKey.ROSTER)
        if not roster_raw:
            return jsonify({"error": "No roster data. Run a refresh first."}), 404
        optimal_raw = read_cache_dict(CacheKey.LINEUP_OPTIMAL)
        streak_payload = read_cache_dict(CacheKey.STREAK_SCORES)

        lineup_data = format_lineup_for_display(roster_raw, optimal_raw, basis=basis)
        for hitter in lineup_data["hitters"]:
            hitter["streak_indicator"] = build_indicator(hitter["name"], streak_payload)

        hitters_html = render_template(
            "season/_lineup_hitters_tbody.html",
            players=lineup_data["hitters"],
            totals=lineup_data["hitter_totals"],
        )
        pitchers_html = render_template(
            "season/_lineup_pitchers_tbody.html",
            players=lineup_data["pitchers"],
            totals=lineup_data["pitcher_totals"],
        )
        return jsonify(
            {"basis": basis, "hitters_html": hitters_html, "pitchers_html": pitchers_html}
        )
```

- [ ] **Step 4: Run the route tests to verify they pass**

Run: `pytest tests/test_web/test_season_routes.py -k "basis or tbodies" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py tests/test_web/test_season_routes.py
git commit -m "feat(lineup): basis query param + /lineup/tbodies partial endpoint"
```

---

## Task 6: Frontend JS — toggle handler + opponent interaction

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html` (the `<script>` block, lines 269-380)

No unit test (browser JS). Verified manually + by Task 5's endpoint test.

- [ ] **Step 1: Add the `onBasisChange` handler**

Inside the `<script>` block (after `runOptimize`), add:

```javascript
function onBasisChange(basis) {
    document.querySelectorAll('.basis-toggle button').forEach(function(b) {
        b.classList.toggle('active', b.dataset.basis === basis);
    });
    var url = new URL(window.location);
    url.searchParams.set('basis', basis);
    history.replaceState(null, '', url);

    fetch('/lineup/tbodies?basis=' + encodeURIComponent(basis))
        .then(function(r) { return r.json(); })
        .then(function(data) {
            var h = document.getElementById('hitters-tbody');
            var p = document.getElementById('pitchers-tbody');
            if (h) h.innerHTML = data.hitters_html;
            if (p) p.innerHTML = data.pitchers_html;
            bindTooltips();
        })
        .catch(function(err) { console.error('basis change failed', err); });
}
```

- [ ] **Step 2: Hide the toggle when viewing an opponent**

The basis toggle is a user-roster feature (opponents stay ROS-only). In
`renderOpponentLineup`, after the existing `if (movesBanner) ...` line, add:

```javascript
    var basisToggle = document.querySelector('.basis-toggle');
    if (basisToggle) basisToggle.style.display = 'none';
```

(Returning to your own team reloads `/lineup`, which re-renders the toggle.)

- [ ] **Step 3: Manual verification**

Start the dashboard and click through YTD / ROS / Total:

Run: `python scripts/run_season_dashboard.py`
Then load `http://localhost:<port>/lineup` and verify:
- ROS shows ROS-remaining cell values; YTD shows actuals; Total shows full-season.
- SGP column and rank badge change with the toggle; bench re-sorts.
- The +/- deviation corner and the per-cell hover tooltip (Actual/Expected/Z)
  are identical in all three views.
- Selecting an opponent hides the toggle; reselecting your team restores it.
- Reloading `/lineup?basis=ytd` keeps YTD selected.

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/lineup.html
git commit -m "feat(lineup): wire basis toggle to partial endpoint, hide for opponents"
```

---

## Task 7: Full verification (END-OF-EFFORT CHECKLIST)

**Files:** none (verification only)

- [ ] **Step 1: Run the full test suite**

Run: `pytest -n auto -q`
Expected: all pass. If any pre-existing-default cell tests fail, confirm they
are the deliberate ROS-cell behavior change (Task 3 Step 6) and update them;
do not loosen unrelated assertions.

- [ ] **Step 2: Lint**

Run: `ruff check .`
Expected: zero violations. Fix any introduced (e.g. unused imports — prefer the
function-local imports shown above, or hoist to module top if the file already
imports from those modules).

- [ ] **Step 3: Format check**

Run: `ruff format --check .`
Expected: no drift. If it reports changes, run `ruff format .` and re-commit.

- [ ] **Step 4: Dead-code check**

Run: `vulture`
Expected: no NEW findings from `_derive_ytd_stats`, `_display_map`,
`lineup_tbodies`, or `onBasisChange`. Pre-existing findings unrelated to this
change are acceptable — call them out.

- [ ] **Step 5: Type check (if covered)**

Check `pyproject.toml` `[tool.mypy].files`. If `season_data.py`,
`refresh_pipeline.py`, `player.py`, `rankings.py`, or `season_routes.py` are
listed, run:

Run: `mypy`
Expected: no new errors. (`_derive_ytd_stats` returns
`HitterStats | PitcherStats | None`; annotate if mypy requests it.)

- [ ] **Step 6: Final commit (if any fixups)**

```bash
git add -A
git commit -m "chore(lineup): lint/format/type fixups for basis toggle"
```

---

## Out of scope (documented)

- **Team Totals row** stays on its current pace/YTD basis (it is a separate
  aggregate with actual-vs-expected semantics). Only player rows re-base.
- **Opponent lineups** stay ROS-only; the toggle is hidden for them.
- **Optimizer / delta-Roto / moves** stay ROS-based and forward-looking.
- **Unmatched call-ups** (no game-log mlbam_id match) derive YTD = 0 even if
  they have played; their `rank.current` may still rank them. Known limitation.
