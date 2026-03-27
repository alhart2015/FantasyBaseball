# Buy-Low Detector Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add buy-low candidate detection to the waivers/trades page, showing underperforming players on opponent rosters and the waiver wire, with collapsible sections throughout the page.

**Architecture:** New `analysis/buy_low.py` computes average z-scores using existing `compute_player_pace()`, filters to > 1 SD below. Refresh pipeline reuses game log lookup dicts to compute candidates for opponents + free agents. Template adds two new collapsible sections with expandable player cards.

**Tech Stack:** Python, existing `pace.py`, Jinja2/Flask, vanilla JS for collapse/expand

**Spec:** `docs/superpowers/specs/2026-03-27-buy-low-detector-design.md`

---

## File Structure

- **Create:** `src/fantasy_baseball/analysis/buy_low.py` — candidate detection logic
- **Create:** `tests/test_analysis/test_buy_low.py` — tests for buy_low module
- **Modify:** `src/fantasy_baseball/web/season_data.py:62-72` — add `buy_low` to CACHE_FILES
- **Modify:** `src/fantasy_baseball/web/season_data.py:~655` — add Step 11b to refresh pipeline
- **Modify:** `src/fantasy_baseball/web/season_routes.py:119-131` — pass buy_low to template
- **Modify:** `src/fantasy_baseball/web/templates/season/waivers_trades.html` — collapsible sections + buy-low cards

---

### Task 1: Add `buy_low` to CACHE_FILES + route

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:62-72`
- Modify: `src/fantasy_baseball/web/season_routes.py:119-131`

- [ ] **Step 1: Add `buy_low` key to CACHE_FILES dict**

In `src/fantasy_baseball/web/season_data.py`, add to the `CACHE_FILES` dict (after `"meta"` on line 71):

```python
    "buy_low": "buy_low.json",
```

- [ ] **Step 2: Pass buy_low data to template in route**

In `src/fantasy_baseball/web/season_routes.py`, modify the `waivers_trades()` function (line 119-131) to read and pass buy_low cache:

```python
    @app.route("/waivers-trades")
    def waivers_trades():
        meta = read_meta()
        waivers_raw = read_cache("waivers")
        trades_raw = read_cache("trades")
        buy_low_raw = read_cache("buy_low") or {}
        return render_template(
            "season/waivers_trades.html",
            meta=meta,
            active_page="waivers_trades",
            waivers=waivers_raw or [],
            trades=trades_raw or [],
            buy_low_targets=buy_low_raw.get("trade_targets", []),
            buy_low_free_agents=buy_low_raw.get("free_agents", []),
            categories=ALL_CATEGORIES,
        )
```

- [ ] **Step 3: Verify template still renders**

Run: `python -c "from fantasy_baseball.web.season_app import create_app; app = create_app(); c = app.test_client(); r = c.get('/waivers-trades'); print(r.status_code)"`
Expected: 200

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py src/fantasy_baseball/web/season_routes.py
git commit -m "feat: add buy_low to CACHE_FILES and route"
```

---

### Task 2: Create `buy_low.py` — core detection logic

**Files:**
- Create: `src/fantasy_baseball/analysis/buy_low.py`
- Create: `tests/test_analysis/test_buy_low.py`

- [ ] **Step 1: Write failing test — hitter below pace qualifies**

Create `tests/test_analysis/test_buy_low.py`:

```python
from fantasy_baseball.analysis.buy_low import find_buy_low_candidates


def test_hitter_below_pace_qualifies():
    """A hitter > 1 SD below projection pace across categories is a buy-low candidate."""
    players = [{
        "name": "Struggling Hitter",
        "positions": ["OF"],
        "player_type": "hitter",
        "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
        "h": 150, "ab": 540, "avg": 0.278,
    }]
    # 10% PA consumed, but well below expected in all counting stats
    game_logs = {
        "struggling hitter": {
            "pa": 60, "ab": 54, "h": 10, "r": 4, "hr": 0, "rbi": 3, "sb": 0,
        },
    }
    leverage = {"R": 1.0, "HR": 1.0, "RBI": 1.0, "SB": 1.0, "AVG": 1.0,
                "W": 1.0, "K": 1.0, "SV": 1.0, "ERA": 1.0, "WHIP": 1.0}

    result = find_buy_low_candidates(players, game_logs, leverage, owner="Opponent A")
    assert len(result) == 1
    assert result[0]["name"] == "Struggling Hitter"
    assert result[0]["owner"] == "Opponent A"
    assert result[0]["avg_z"] < -1.0
    assert "stats" in result[0]
```

- [ ] **Step 2: Write failing test — hitter on pace does NOT qualify**

```python
def test_hitter_on_pace_excluded():
    """A hitter near projection pace should not be a buy-low candidate."""
    players = [{
        "name": "Normal Hitter",
        "positions": ["1B"],
        "player_type": "hitter",
        "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
        "h": 150, "ab": 540, "avg": 0.278,
    }]
    game_logs = {
        "normal hitter": {
            "pa": 60, "ab": 54, "h": 15, "r": 9, "hr": 3, "rbi": 9, "sb": 1,
        },
    }
    leverage = {"R": 1.0, "HR": 1.0, "RBI": 1.0, "SB": 1.0, "AVG": 1.0,
                "W": 1.0, "K": 1.0, "SV": 1.0, "ERA": 1.0, "WHIP": 1.0}

    result = find_buy_low_candidates(players, game_logs, leverage)
    assert len(result) == 0
```

- [ ] **Step 3: Write failing test — player with no game logs excluded**

```python
def test_no_game_logs_excluded():
    """A player with no game logs (below sample threshold) is excluded."""
    players = [{
        "name": "No Games Player",
        "positions": ["SS"],
        "player_type": "hitter",
        "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
        "h": 150, "ab": 540, "avg": 0.278,
    }]
    game_logs = {}  # no game logs at all
    leverage = {"R": 1.0, "HR": 1.0, "RBI": 1.0, "SB": 1.0, "AVG": 1.0,
                "W": 1.0, "K": 1.0, "SV": 1.0, "ERA": 1.0, "WHIP": 1.0}

    result = find_buy_low_candidates(players, game_logs, leverage)
    assert len(result) == 0
```

- [ ] **Step 4: Write failing test — sorted by most underperforming**

```python
def test_sorted_most_underperforming_first():
    """Results are sorted by avg_z ascending (most negative first)."""
    players = [
        {"name": "Somewhat Bad", "positions": ["OF"], "player_type": "hitter",
         "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
         "h": 150, "ab": 540, "avg": 0.278},
        {"name": "Very Bad", "positions": ["1B"], "player_type": "hitter",
         "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
         "h": 150, "ab": 540, "avg": 0.278},
    ]
    game_logs = {
        "somewhat bad": {"pa": 60, "ab": 54, "h": 10, "r": 4, "hr": 1, "rbi": 4, "sb": 0},
        "very bad": {"pa": 60, "ab": 54, "h": 5, "r": 2, "hr": 0, "rbi": 1, "sb": 0},
    }
    leverage = {"R": 1.0, "HR": 1.0, "RBI": 1.0, "SB": 1.0, "AVG": 1.0,
                "W": 1.0, "K": 1.0, "SV": 1.0, "ERA": 1.0, "WHIP": 1.0}

    result = find_buy_low_candidates(players, game_logs, leverage)
    assert len(result) >= 2
    assert result[0]["name"] == "Very Bad"
    assert result[0]["avg_z"] < result[1]["avg_z"]
```

- [ ] **Step 5: Write failing test — pitcher buy-low**

```python
def test_pitcher_below_pace_qualifies():
    """A pitcher with bad ERA and low K qualifies."""
    players = [{
        "name": "Bad Pitcher",
        "positions": ["SP"],
        "player_type": "pitcher",
        "ip": 180, "w": 12, "k": 190, "sv": 0,
        "er": 60, "bb": 50, "h_allowed": 150,
        "era": 3.00, "whip": 1.11,
    }]
    # 18 IP, terrible ERA, low Ks
    game_logs = {
        "bad pitcher": {"ip": 18.0, "k": 10, "w": 0, "sv": 0, "er": 14, "bb": 12, "h_allowed": 22},
    }
    leverage = {"R": 1.0, "HR": 1.0, "RBI": 1.0, "SB": 1.0, "AVG": 1.0,
                "W": 1.0, "K": 1.0, "SV": 1.0, "ERA": 1.0, "WHIP": 1.0}

    result = find_buy_low_candidates(players, game_logs, leverage)
    assert len(result) == 1
    assert result[0]["name"] == "Bad Pitcher"
    assert result[0]["avg_z"] < -1.0
```

- [ ] **Step 6: Write failing test — z-score exclusion for below-threshold stats**

```python
def test_below_threshold_stats_excluded_from_average():
    """Stats below sample threshold (z=0, neutral) are excluded from average, not diluted."""
    players = [{
        "name": "Small Sample Hitter",
        "positions": ["OF"],
        "player_type": "hitter",
        "pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10,
        "h": 150, "ab": 540, "avg": 0.278,
    }]
    # 15 PA — counting stats colored (>= 10 PA), AVG neutral (< 30 PA)
    # Very bad in counting stats
    game_logs = {
        "small sample hitter": {"pa": 15, "ab": 13, "h": 3, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
    }
    leverage = {"R": 1.0, "HR": 1.0, "RBI": 1.0, "SB": 1.0, "AVG": 1.0,
                "W": 1.0, "K": 1.0, "SV": 1.0, "ERA": 1.0, "WHIP": 1.0}

    result = find_buy_low_candidates(players, game_logs, leverage)
    # AVG should be excluded from average (below 30 PA threshold)
    # Only R, HR, RBI, SB contribute — all very negative
    assert len(result) == 1
    # Verify AVG was excluded: if AVG (z=0) were included, avg_z would be diluted toward 0
    # With only 4 counting stats averaging, it should be more negative
    assert result[0]["avg_z"] < -1.0
```

- [ ] **Step 7: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_buy_low.py -v`
Expected: FAIL — module not found

- [ ] **Step 8: Implement `find_buy_low_candidates`**

Create `src/fantasy_baseball/analysis/buy_low.py`:

```python
"""Buy-low candidate detection — players underperforming projections."""

import pandas as pd

from fantasy_baseball.analysis.pace import compute_player_pace
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.utils.constants import HITTING_CATEGORIES, PITCHING_CATEGORIES
from fantasy_baseball.utils.name_utils import normalize_name


def find_buy_low_candidates(
    players: list[dict],
    game_log_lookup: dict,
    leverage: dict,
    owner: str = "Free Agent",
) -> list[dict]:
    """Find players underperforming projections by > 1 SD.

    Args:
        players: Roster entries with projection stats (dict with lowercase stat keys).
        game_log_lookup: {normalized_name: {stat: value}} from bulk game log query.
        leverage: Per-category leverage weights for wSGP computation.
        owner: Team name or "Free Agent" for display.

    Returns:
        List of candidate dicts sorted by avg_z ascending (most negative first).
    """
    candidates = []

    for player in players:
        name = player.get("name", "")
        ptype = player.get("player_type", "")
        if ptype not in ("hitter", "pitcher"):
            continue

        norm = normalize_name(name)
        actuals = game_log_lookup.get(norm, {})

        # Build projection dict from player entry
        if ptype == "hitter":
            proj_keys = ["pa", "r", "hr", "rbi", "sb", "h", "ab", "avg"]
            categories = HITTING_CATEGORIES
        else:
            proj_keys = ["ip", "w", "k", "sv", "er", "bb", "h_allowed", "era", "whip"]
            categories = PITCHING_CATEGORIES

        projected = {k: player.get(k, 0) or 0 for k in proj_keys}
        pace = compute_player_pace(actuals, projected, ptype)

        # Average z-scores, excluding stats where z=0 and color=neutral
        # (below sample threshold or no projection — not informative)
        z_scores = []
        for cat in categories:
            st = pace.get(cat, {})
            z = st.get("z_score", 0.0)
            color = st.get("color_class", "stat-neutral")
            if z == 0.0 and color == "stat-neutral":
                continue  # skip non-informative stats
            z_scores.append(z)

        if not z_scores:
            continue  # no stats with enough sample

        avg_z = round(sum(z_scores) / len(z_scores), 2)

        if avg_z >= -1.0:
            continue  # not underperforming enough

        # Compute wSGP using projection stats and user's leverage
        try:
            wsgp = round(calculate_weighted_sgp(pd.Series(player), leverage), 2)
        except Exception:
            wsgp = 0.0

        candidates.append({
            "name": name,
            "positions": player.get("positions", []),
            "owner": owner,
            "player_type": ptype,
            "avg_z": avg_z,
            "stats": pace,
            "wsgp": wsgp,
        })

    candidates.sort(key=lambda c: c["avg_z"])
    return candidates
```

- [ ] **Step 9: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_buy_low.py -v`
Expected: all PASS

- [ ] **Step 10: Commit**

```bash
git add src/fantasy_baseball/analysis/buy_low.py tests/test_analysis/test_buy_low.py
git commit -m "feat: add buy-low candidate detection module"
```

---

### Task 3: Add buy-low computation to refresh pipeline

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:~655`

- [ ] **Step 1: Insert Step 11b into `run_full_refresh()`**

In `season_data.py`, find `write_cache("trades", trade_proposals, cache_dir)` (around line 655). Insert the following AFTER that line and BEFORE Step 12 (`_set_refresh_progress("Projecting standings...")`):

Add import near the other lazy imports at the top of `run_full_refresh()`:

```python
from fantasy_baseball.analysis.buy_low import find_buy_low_candidates
```

Insert the new step:

```python
        # --- Step 11b: Compute buy-low candidates ---
        _set_refresh_progress("Finding buy-low candidates...")
        buy_low_trade_targets = []
        for tname, opp_roster in opp_rosters.items():
            candidates = find_buy_low_candidates(
                opp_roster,
                hitter_logs if opp_roster and opp_roster[0].get("player_type") == "hitter" else pitcher_logs,
                leverage,
                owner=tname,
            )
            buy_low_trade_targets.extend(candidates)
```

Wait — opponent rosters contain both hitters and pitchers. The game_log_lookup needs both. Let me fix: merge the two dicts into one lookup for buy_low:

```python
        # --- Step 11b: Compute buy-low candidates ---
        _set_refresh_progress("Finding buy-low candidates...")
        all_game_logs = {**hitter_logs, **pitcher_logs}

        buy_low_trade_targets = []
        for tname, opp_roster in opp_rosters.items():
            candidates = find_buy_low_candidates(
                opp_roster, all_game_logs, leverage, owner=tname,
            )
            buy_low_trade_targets.extend(candidates)
        buy_low_trade_targets.sort(key=lambda c: c["avg_z"])

        buy_low_free_agents = find_buy_low_candidates(
            [s.to_dict() if hasattr(s, 'to_dict') else dict(s) for s in fa_players],
            all_game_logs, leverage, owner="Free Agent",
        )

        write_cache("buy_low", {
            "trade_targets": buy_low_trade_targets,
            "free_agents": buy_low_free_agents,
        }, cache_dir)
```

Note: `fa_players` may be `list[pd.Series]` — convert with `.to_dict()`. `hitter_logs` and `pitcher_logs` are still in scope from Step 6b.

- [ ] **Step 2: Verify existing tests pass**

Run: `python -m pytest tests/test_web/ -v`
Expected: all PASS

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "feat: compute buy-low candidates in refresh pipeline"
```

---

### Task 4: Make all sections collapsible

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`

- [ ] **Step 1: Add CSS for collapsible sections**

Add to the template (before the content block, or in a `<style>` block at the top):

```css
<style>
.section-toggle {
    display: flex;
    align-items: center;
    justify-content: space-between;
    cursor: pointer;
    user-select: none;
    font-size: 15px;
    font-weight: 600;
    color: var(--text);
    margin: 28px 0 10px;
    padding: 0;
}
.section-toggle:first-of-type { margin-top: 20px; }
.section-toggle .chevron {
    font-size: 12px;
    color: var(--text-secondary);
    transition: transform 0.2s;
}
.section-toggle .chevron.collapsed { transform: rotate(-90deg); }
.section-body { overflow: hidden; }
.section-body.collapsed { display: none; }
</style>
```

- [ ] **Step 2: Wrap Waiver Wire section in collapsible container**

Replace the existing `<h3>Waiver Wire</h3>` (line 13) with:

```html
<div class="section-toggle" onclick="toggleSection('waivers')">
    <span>Waiver Wire</span>
    <span class="chevron" id="chevron-waivers">&#9660;</span>
</div>
<div class="section-body" id="section-waivers">
```

Add closing `</div>` after the waiver `{% endif %}` (line 59).

- [ ] **Step 3: Wrap Trade Recommendations in collapsible container**

Replace the existing `<h3>Trade Recommendations</h3>` (line 62) with:

```html
<div class="section-toggle" onclick="toggleSection('trades')">
    <span>Trade Recommendations</span>
    <span class="chevron" id="chevron-trades">&#9660;</span>
</div>
<div class="section-body" id="section-trades">
```

Add closing `</div>` after the trades `{% endif %}` (line 139).

- [ ] **Step 4: Add toggleSection JS**

Add to the `<script>` block:

```javascript
function toggleSection(id) {
    var body = document.getElementById('section-' + id);
    var chevron = document.getElementById('chevron-' + id);
    if (!body) return;
    body.classList.toggle('collapsed');
    if (chevron) chevron.classList.toggle('collapsed');
}
```

- [ ] **Step 5: Verify page renders and sections collapse**

Run: `python -c "from fantasy_baseball.web.season_app import create_app; app = create_app(); c = app.test_client(); r = c.get('/waivers-trades'); print(r.status_code)"`
Expected: 200

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat: make waiver and trade sections collapsible"
```

---

### Task 5: Add buy-low sections to template

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`

- [ ] **Step 1: Add CSS for buy-low cards**

Add to the existing `<style>` block:

```css
.buy-low-card {
    background: var(--panel-bg); border: 1px solid var(--panel-border);
    border-radius: 6px; padding: 10px 14px; margin-bottom: 8px; cursor: pointer;
}
.buy-low-card:hover { border-color: var(--accent); }
.buy-low-header {
    display: flex; align-items: center; justify-content: space-between;
}
.buy-low-header .player-info { display: flex; align-items: baseline; gap: 8px; }
.buy-low-header .player-name { font-weight: 600; font-size: 14px; }
.buy-low-header .player-pos { color: var(--text-secondary); font-size: 12px; }
.buy-low-header .player-owner { color: var(--text-secondary); font-size: 11px; }
.buy-low-badge {
    font-weight: 600; font-size: 12px; padding: 2px 8px; border-radius: 3px;
}
.buy-low-detail { display: none; margin-top: 10px; }
.buy-low-detail.open { display: block; }
.buy-low-detail table { width: 100%; font-size: 12px; border-collapse: collapse; }
.buy-low-detail th {
    text-align: left; font-size: 11px; color: var(--text-secondary);
    font-weight: 500; text-transform: uppercase; padding: 4px 6px;
    border-bottom: 1px solid var(--panel-border);
}
.buy-low-detail td { padding: 4px 6px; }
.show-more-btn {
    display: block; width: 100%; text-align: center; padding: 8px;
    background: var(--panel-bg); border: 1px solid var(--panel-border);
    border-radius: 6px; cursor: pointer; color: var(--accent);
    font-size: 12px; margin-top: 8px;
}
.show-more-btn:hover { border-color: var(--accent); }
```

- [ ] **Step 2: Add Buy-Low Trade Targets section**

After the closing `</div>` of the trades section body, add:

```html
{# === Buy-Low Trade Targets === #}
<div class="section-toggle" onclick="toggleSection('buy-low-targets')">
    <span>Buy-Low Trade Targets</span>
    <span class="chevron" id="chevron-buy-low-targets">&#9660;</span>
</div>
<div class="section-body" id="section-buy-low-targets">
{% if not buy_low_targets %}
<p class="placeholder-text">No buy-low candidates yet — game log data needed.</p>
{% else %}
{% for bl in buy_low_targets %}
<div class="buy-low-card {% if loop.index > 5 %}buy-low-overflow buy-low-overflow-targets{% endif %}"
     {% if loop.index > 5 %}style="display: none;"{% endif %}
     onclick="toggleBuyLowDetail(this)">
    <div class="buy-low-header">
        <div class="player-info">
            <span class="player-name">{{ bl.name }}</span>
            <span class="player-pos">{{ bl.positions | join(", ") }}</span>
            <span class="player-owner">{{ bl.owner }}</span>
        </div>
        <span class="buy-low-badge {{ 'stat-cold-2' if bl.avg_z < -1.5 else 'stat-cold-1' }}">
            {{ "%.2f"|format(bl.avg_z) }} SD
        </span>
    </div>
    <div class="buy-low-detail">
        <table>
            <thead>
                <tr><th>Cat</th><th>Actual</th><th>Expected</th><th>Z-score</th></tr>
            </thead>
            <tbody>
            {% set cats = ["R", "HR", "RBI", "SB", "AVG"] if bl.player_type == "hitter" else ["W", "K", "SV", "ERA", "WHIP"] %}
            {% for cat in cats %}
            {% set st = bl.stats.get(cat, {}) %}
            <tr>
                <td>{{ cat }}</td>
                <td>{% if cat in ['AVG'] %}{{ "%.3f"|format(st.get('actual', 0)) }}{% elif cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(st.get('actual', 0)) }}{% else %}{{ st.get('actual', 0) }}{% endif %}</td>
                <td>{% if cat in ['AVG'] %}{{ "%.3f"|format(st.get('expected', 0)) }}{% elif cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(st.get('expected', 0)) }}{% else %}{{ st.get('expected', 0) }}{% endif %}</td>
                <td class="{{ st.get('color_class', 'stat-neutral') }}">{{ "%.2f"|format(st.get('z_score', 0)) }}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        <div style="margin-top: 6px; font-size: 12px; color: var(--text-secondary);">
            wSGP: <strong>{{ "%.2f"|format(bl.wsgp) }}</strong>
        </div>
    </div>
</div>
{% endfor %}
{% if buy_low_targets | length > 5 %}
<button class="show-more-btn" onclick="toggleShowMore(event, 'targets', {{ buy_low_targets | length }})">
    Show More ({{ buy_low_targets | length }} total)
</button>
{% endif %}
{% endif %}
</div>
```

- [ ] **Step 3: Add Buy-Low Free Agents section**

Same pattern, using `buy_low_free_agents` and `buy-low-overflow-fa` class:

```html
{# === Buy-Low Free Agents === #}
<div class="section-toggle" onclick="toggleSection('buy-low-fa')">
    <span>Buy-Low Free Agents</span>
    <span class="chevron" id="chevron-buy-low-fa">&#9660;</span>
</div>
<div class="section-body" id="section-buy-low-fa">
{% if not buy_low_free_agents %}
<p class="placeholder-text">No buy-low free agents found — all tracking near projections.</p>
{% else %}
{% for bl in buy_low_free_agents %}
<div class="buy-low-card {% if loop.index > 5 %}buy-low-overflow buy-low-overflow-fa{% endif %}"
     {% if loop.index > 5 %}style="display: none;"{% endif %}
     onclick="toggleBuyLowDetail(this)">
    <div class="buy-low-header">
        <div class="player-info">
            <span class="player-name">{{ bl.name }}</span>
            <span class="player-pos">{{ bl.positions | join(", ") }}</span>
            <span class="player-owner" style="color: var(--accent);">Free Agent</span>
        </div>
        <span class="buy-low-badge {{ 'stat-cold-2' if bl.avg_z < -1.5 else 'stat-cold-1' }}">
            {{ "%.2f"|format(bl.avg_z) }} SD
        </span>
    </div>
    <div class="buy-low-detail">
        <table>
            <thead>
                <tr><th>Cat</th><th>Actual</th><th>Expected</th><th>Z-score</th></tr>
            </thead>
            <tbody>
            {% set cats = ["R", "HR", "RBI", "SB", "AVG"] if bl.player_type == "hitter" else ["W", "K", "SV", "ERA", "WHIP"] %}
            {% for cat in cats %}
            {% set st = bl.stats.get(cat, {}) %}
            <tr>
                <td>{{ cat }}</td>
                <td>{% if cat == 'AVG' %}{{ "%.3f"|format(st.get('actual', 0)) }}{% elif cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(st.get('actual', 0)) }}{% else %}{{ st.get('actual', 0) }}{% endif %}</td>
                <td>{% if cat == 'AVG' %}{{ "%.3f"|format(st.get('expected', 0)) }}{% elif cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(st.get('expected', 0)) }}{% else %}{{ st.get('expected', 0) }}{% endif %}</td>
                <td class="{{ st.get('color_class', 'stat-neutral') }}">{{ "%.2f"|format(st.get('z_score', 0)) }}</td>
            </tr>
            {% endfor %}
            </tbody>
        </table>
        <div style="margin-top: 6px; font-size: 12px; color: var(--text-secondary);">
            wSGP: <strong>{{ "%.2f"|format(bl.wsgp) }}</strong>
        </div>
    </div>
</div>
{% endfor %}
{% if buy_low_free_agents | length > 5 %}
<button class="show-more-btn" onclick="toggleShowMore(event, 'fa', {{ buy_low_free_agents | length }})">
    Show More ({{ buy_low_free_agents | length }} total)
</button>
{% endif %}
{% endif %}
</div>
```

- [ ] **Step 4: Add JS for buy-low card expand and show-more toggle**

Add to the `<script>` block:

```javascript
function toggleBuyLowDetail(card) {
    var detail = card.querySelector('.buy-low-detail');
    if (detail) detail.classList.toggle('open');
}

function toggleShowMore(ev, group, total) {
    var cards = document.querySelectorAll('.buy-low-overflow-' + group);
    var btn = ev.currentTarget;
    var visible = cards[0] && cards[0].style.display !== 'none';
    for (var i = 0; i < cards.length; i++) {
        cards[i].style.display = visible ? 'none' : 'block';
    }
    btn.textContent = visible ? 'Show More (' + total + ' total)' : 'Show Less';
}
```

- [ ] **Step 5: Verify page renders**

Run: `python -c "from fantasy_baseball.web.season_app import create_app; app = create_app(); c = app.test_client(); r = c.get('/waivers-trades'); print(r.status_code)"`
Expected: 200

- [ ] **Step 6: Run all web tests**

Run: `python -m pytest tests/test_web/ -v`
Expected: all PASS

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat: add buy-low sections with expandable cards and show-more"
```

---

### Task 6: Run full test suite and verify

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest -v`
Expected: all tests pass, no regressions

- [ ] **Step 2: Verify template renders all sections**

Run: `python -c "from fantasy_baseball.web.season_app import create_app; app = create_app(); c = app.test_client(); r = c.get('/waivers-trades'); print('buy-low-targets' in r.text, 'buy-low-fa' in r.text)"`
Expected: `True True`

- [ ] **Step 3: Final commit if any cleanup needed**

If any adjustments were made, commit them.
