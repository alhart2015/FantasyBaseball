# ROS Deviation Indicators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stat significance checkmark with SGP-based indicators showing how much each player's updated outlook (actual + ROS) has shifted from preseason.

**Architecture:** Add a `ros_deviation_sgp` float to each stat in the pace dict, computed as `(ros - preseason) / sgp_denominator` with sign-flip for inverse stats. Remove all significance/stabilization-threshold code. Render `+` or `×` symbols in the stat cell corner via CSS `::after`, count driven by truncated SGP deviation, capped at 3.

**Tech Stack:** Python (pace.py, player.py, constants.py, season_data.py), Jinja2 (lineup.html), CSS (season.css)

---

### Task 1: Add `ros_deviation_sgp` to pace computation

**Files:**
- Modify: `src/fantasy_baseball/analysis/pace.py:59-205`
- Test: `tests/test_analysis/test_pace.py`

- [ ] **Step 1: Write failing test for hitter ROS deviation**

Add to `tests/test_analysis/test_pace.py`:

```python
def test_hitter_ros_deviation_sgp():
    """ROS deviation = (ros - preseason) / sgp_denom, positive = good."""
    preseason = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    ros = {"r": 100, "hr": 33, "rbi": 85, "sb": 12, "avg": 0.290}
    sgp = {"R": 20, "HR": 9, "RBI": 20, "SB": 8, "AVG": 0.005}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, preseason, "hitter", ros_stats=ros, sgp_denoms=sgp)

    # R: (100 - 90) / 20 = 0.5
    assert result["R"]["ros_deviation_sgp"] == pytest.approx(0.5, abs=0.01)
    # HR: (33 - 30) / 9 = 0.33
    assert result["HR"]["ros_deviation_sgp"] == pytest.approx(0.333, abs=0.01)
    # RBI: (85 - 90) / 20 = -0.25
    assert result["RBI"]["ros_deviation_sgp"] == pytest.approx(-0.25, abs=0.01)
    # SB: (12 - 10) / 8 = 0.25
    assert result["SB"]["ros_deviation_sgp"] == pytest.approx(0.25, abs=0.01)
    # AVG: (0.290 - 0.278) / 0.005 = 2.4
    assert result["AVG"]["ros_deviation_sgp"] == pytest.approx(2.4, abs=0.01)
```

Add `import pytest` at the top of the test file if not already present.

- [ ] **Step 2: Write failing test for pitcher ROS deviation with inverse stats**

Add to `tests/test_analysis/test_pace.py`:

```python
def test_pitcher_ros_deviation_sgp():
    """Pitcher ROS deviation: ERA/WHIP are inverse (lower = positive deviation)."""
    preseason = {"ip": 180, "w": 12, "k": 190, "sv": 0, "er": 60, "bb": 50, "h_allowed": 150,
                 "era": 3.00, "whip": 1.11}
    ros = {"w": 14, "k": 200, "sv": 0, "era": 2.70, "whip": 1.05}
    sgp = {"W": 3, "K": 30, "SV": 7, "ERA": 0.15, "WHIP": 0.015}
    actual = {"ip": 18.0, "k": 22, "w": 1, "sv": 0, "er": 5, "bb": 5, "h_allowed": 16}
    result = compute_player_pace(actual, preseason, "pitcher", ros_stats=ros, sgp_denoms=sgp)

    # W: (14 - 12) / 3 = 0.667
    assert result["W"]["ros_deviation_sgp"] == pytest.approx(0.667, abs=0.01)
    # K: (200 - 190) / 30 = 0.333
    assert result["K"]["ros_deviation_sgp"] == pytest.approx(0.333, abs=0.01)
    # ERA: (2.70 - 3.00) / 0.15 = -2.0, flip sign -> +2.0 (lower ERA = good)
    assert result["ERA"]["ros_deviation_sgp"] == pytest.approx(2.0, abs=0.01)
    # WHIP: (1.05 - 1.11) / 0.015 = -4.0, flip sign -> +4.0
    assert result["WHIP"]["ros_deviation_sgp"] == pytest.approx(4.0, abs=0.01)
```

- [ ] **Step 3: Write failing test for missing ROS stats**

Add to `tests/test_analysis/test_pace.py`:

```python
def test_ros_deviation_zero_when_no_ros():
    """When ros_stats or sgp_denoms are None, ros_deviation_sgp should be 0."""
    preseason = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, preseason, "hitter")
    assert result["HR"]["ros_deviation_sgp"] == 0.0
    assert result["AVG"]["ros_deviation_sgp"] == 0.0
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `pytest tests/test_analysis/test_pace.py::test_hitter_ros_deviation_sgp tests/test_analysis/test_pace.py::test_pitcher_ros_deviation_sgp tests/test_analysis/test_pace.py::test_ros_deviation_zero_when_no_ros -v`
Expected: FAIL — `ros_deviation_sgp` key not in result dicts.

- [ ] **Step 5: Implement `ros_deviation_sgp` in `compute_player_pace`**

In `src/fantasy_baseball/analysis/pace.py`, update the function signature and add deviation computation:

Change the signature at line 59:

```python
def compute_player_pace(
    actual_stats: dict,
    projected_stats: dict,
    player_type: str,
    ros_stats: dict | None = None,
    sgp_denoms: dict | None = None,
) -> dict:
```

Add a helper inside the function body (after the existing local variable setup, before the counting stats loop):

```python
    def _ros_deviation(cat: str) -> float:
        """Compute SGP deviation: (ros - preseason) / denom, positive = good."""
        if not ros_stats or not sgp_denoms:
            return 0.0
        ros_key = cat.lower()
        pre_key = cat.lower()
        ros_val = ros_stats.get(ros_key)
        pre_val = projected_stats.get(pre_key)
        denom = sgp_denoms.get(cat)
        if ros_val is None or pre_val is None or not denom:
            return 0.0
        dev = (ros_val - pre_val) / denom
        if cat in INVERSE_STATS:
            dev = -dev
        return round(dev, 2)
```

Then add `"ros_deviation_sgp": _ros_deviation(display_key)` to each counting stat result dict (line ~123-131), and `"ros_deviation_sgp": _ros_deviation("AVG")` to the AVG dict (line ~149), and `"ros_deviation_sgp": _ros_deviation("ERA")` / `"ros_deviation_sgp": _ros_deviation("WHIP")` to the pitcher rate stat dicts (lines ~176, ~195).

Also add `"ros_deviation_sgp": _ros_deviation("SV")` for the SV counting stat — it's in the pitcher counting loop already.

- [ ] **Step 6: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_pace.py -v`
Expected: All tests pass, including existing ones (the new params default to `None`).

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/analysis/pace.py tests/test_analysis/test_pace.py
git commit -m "feat(pace): add ros_deviation_sgp to pace output"
```

---

### Task 2: Remove significance feature

**Files:**
- Modify: `src/fantasy_baseball/models/player.py:61-74, 125-139`
- Modify: `src/fantasy_baseball/utils/constants.py:196-207`
- Modify: `src/fantasy_baseball/analysis/pace.py`
- Modify: `src/fantasy_baseball/web/season_routes.py:710-723`
- Delete tests: `tests/test_models/test_player.py:312-385`
- Delete tests: `tests/test_analysis/test_pace.py` (significance-specific tests)

- [ ] **Step 1: Remove `is_significant` and `significant_dict` from `HitterStats`**

In `src/fantasy_baseball/models/player.py`, delete lines 61-74 (the `is_significant` method and `significant_dict` method on `HitterStats`).

- [ ] **Step 2: Remove `is_significant` and `significant_dict` from `PitcherStats`**

In `src/fantasy_baseball/models/player.py`, delete lines 125-139 (the `is_significant` method and `significant_dict` method on `PitcherStats`).

- [ ] **Step 3: Remove `STABILIZATION_THRESHOLDS` from constants**

In `src/fantasy_baseball/utils/constants.py`, delete lines 196-207 (the comment block and `STABILIZATION_THRESHOLDS` dict).

- [ ] **Step 4: Remove `significant` and `below_threshold` from pace output**

In `src/fantasy_baseball/analysis/pace.py`:

1. Remove the `stats_cls` / `actual_obj` construction (lines 93-94):
   ```python
   # DELETE these two lines:
   stats_cls = HitterStats if player_type == PlayerType.HITTER else PitcherStats
   actual_obj = stats_cls.from_dict(actual_stats)
   ```

2. Remove the import of `HitterStats, PitcherStats` from inside the function (line 75):
   ```python
   # DELETE this line:
   from fantasy_baseball.models.player import HitterStats, PitcherStats
   ```

3. Remove `"significant"` and `"below_threshold"` keys from every result dict:
   - Counting stats dict (line ~129-130): delete `"significant": ...,` and `"below_threshold": ...,`
   - AVG dict (line ~155-156): delete `"significant": ...,` and `"below_threshold": ...,`
   - ERA dict (line ~182-183): delete `"significant": ...,` and `"below_threshold": ...,`
   - WHIP dict (line ~201-202): delete `"significant": ...,` and `"below_threshold": ...,`

- [ ] **Step 5: Remove `significant_dict` calls from `season_routes.py`**

In `src/fantasy_baseball/web/season_routes.py`, remove the significance computation at lines 710-711 and 717-723. Specifically:

At line 710-711, delete:
```python
                    actual_obj = HitterStats(pa=actual_pa.get(norm, 0))
                    result["significant"] = actual_obj.significant_dict()
```

At lines 717-723, delete:
```python
                    logs = actual_pitcher_logs.get(norm, {})
                    actual_obj = PitcherStats(
                        ip=logs.get("ip", 0),
                        bb=logs.get("bb", 0),
                        h_allowed=logs.get("h_allowed", 0),
                    )
                    result["significant"] = actual_obj.significant_dict()
```

Check if `HitterStats` and `PitcherStats` imports are still needed in that file for other uses. If not, remove the imports too.

- [ ] **Step 6: Delete significance tests from `test_player.py`**

In `tests/test_models/test_player.py`, delete the entire `TestHitterSignificance` class (lines 312-340) and `TestPitcherSignificance` class (lines 343-385).

- [ ] **Step 7: Delete significance tests from `test_pace.py`**

In `tests/test_analysis/test_pace.py`, delete `test_significance_flags_in_pace_output` (lines 185-198) and `test_significance_pitcher` (lines 201-214).

- [ ] **Step 8: Run all tests**

Run: `pytest tests/ -v`
Expected: All tests pass. No references to `is_significant`, `significant_dict`, `STABILIZATION_THRESHOLDS`, or `significant` key in pace output remain.

- [ ] **Step 9: Commit**

```bash
git add -A
git commit -m "refactor: remove stat significance/stabilization threshold feature"
```

---

### Task 3: Wire ROS stats and SGP denoms into pace call sites

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:1083-1096` (full refresh)
- Modify: `src/fantasy_baseball/web/season_data.py:452-460` (opponent lineup)

- [ ] **Step 1: Pass `ros_stats` and `sgp_denoms` in the full refresh call site**

In `src/fantasy_baseball/web/season_data.py`, at line ~1083-1096, the pace loop currently does:

```python
        for player in roster_players:
            norm = normalize_name(player.name)
            if player.player_type == PlayerType.HITTER:
                actuals = hitter_logs.get(norm, {})
            else:
                actuals = pitcher_logs.get(norm, {})
            proj_keys = HITTER_PROJ_KEYS if player.player_type == PlayerType.HITTER else PITCHER_PROJ_KEYS
            pre_player = preseason_lookup.get(norm)
            if pre_player and pre_player.ros:
                projected = {k: getattr(pre_player.ros, k, 0) for k in proj_keys}
            else:
                projected = {k: 0 for k in proj_keys}
            player.pace = compute_player_pace(actuals, projected, player.player_type)
```

Change to:

```python
        sgp_denoms = config.sgp_denominators
        for player in roster_players:
            norm = normalize_name(player.name)
            if player.player_type == PlayerType.HITTER:
                actuals = hitter_logs.get(norm, {})
                ros_keys = ["r", "hr", "rbi", "sb", "avg"]
            else:
                actuals = pitcher_logs.get(norm, {})
                ros_keys = ["w", "k", "sv", "era", "whip"]
            proj_keys = HITTER_PROJ_KEYS if player.player_type == PlayerType.HITTER else PITCHER_PROJ_KEYS
            pre_player = preseason_lookup.get(norm)
            if pre_player and pre_player.ros:
                projected = {k: getattr(pre_player.ros, k, 0) for k in proj_keys}
            else:
                projected = {k: 0 for k in proj_keys}
            ros_dict = {k: getattr(player.ros, k, 0) for k in ros_keys} if player.ros else None
            player.pace = compute_player_pace(
                actuals, projected, player.player_type,
                ros_stats=ros_dict, sgp_denoms=sgp_denoms,
            )
```

- [ ] **Step 2: Pass `ros_stats` and `sgp_denoms` in the opponent lineup call site**

In `src/fantasy_baseball/web/season_data.py`, at line ~452-460, the `build_opponent_lineup` function builds pace differently — it uses `player.ros` as `projected_stats` (not preseason). The opponent view doesn't have preseason data, so ROS deviation doesn't apply. Simply pass the defaults (no `ros_stats`, no `sgp_denoms`) — the existing call is already fine since the new params default to `None`. No change needed here.

- [ ] **Step 3: Run tests**

Run: `pytest tests/ -v`
Expected: All pass.

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "feat(season_data): pass ROS stats and SGP denoms to pace computation"
```

---

### Task 4: Update lineup template and CSS

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html:157, 166, 224, 233`
- Modify: `src/fantasy_baseball/web/templates/season/players.html:321`
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html:238, 294`
- Modify: `src/fantasy_baseball/web/static/season.css:386-397`

- [ ] **Step 1: Replace `.stat-significant` CSS with `.stat-ros-*` styles**

In `src/fantasy_baseball/web/static/season.css`, replace lines 386-397:

```css
/* Significance checkmark — shown when sample size exceeds stabilization threshold */
.stat-significant { position: relative; }
.stat-significant::after {
    content: "\2713";
    position: absolute;
    top: 1px;
    right: 2px;
    font-size: 8px;
    color: #22c55e;
    opacity: 0.7;
    line-height: 1;
}
```

With:

```css
/* ROS deviation indicators — shows how much actual+ROS has shifted from preseason in SGP units */
.stat-ros-up, .stat-ros-down { position: relative; }
.stat-ros-up::after, .stat-ros-down::after {
    position: absolute;
    top: 1px;
    right: 2px;
    font-size: 8px;
    opacity: 0.7;
    line-height: 1;
}
.stat-ros-up::after { color: #22c55e; }
.stat-ros-down::after { color: #ef4444; }
.stat-ros-up.stat-ros-1::after { content: "+"; }
.stat-ros-up.stat-ros-2::after { content: "++"; }
.stat-ros-up.stat-ros-3::after { content: "+++"; }
.stat-ros-down.stat-ros-1::after { content: "\00d7"; }
.stat-ros-down.stat-ros-2::after { content: "\00d7\00d7"; }
.stat-ros-down.stat-ros-3::after { content: "\00d7\00d7\00d7"; }
```

- [ ] **Step 2: Update hitter stat cells in `lineup.html`**

In `src/fantasy_baseball/web/templates/season/lineup.html`, replace line 157:

```html
            <td class="stat-cell {{ st.get('color_class', 'stat-neutral') }}{% if cat == 'PA' %} pa-col{% endif %}{% if st.get('significant') %} stat-significant{% endif %}">
```

With:

```html
            {% set dev = st.get('ros_deviation_sgp', 0)|int %}
            {% set dev_abs = dev if dev > 0 else -dev %}
            {% set dev_clamped = [dev_abs, 3]|min %}
            <td class="stat-cell {{ st.get('color_class', 'stat-neutral') }}{% if cat == 'PA' %} pa-col{% endif %}{% if dev > 0 %} stat-ros-up stat-ros-{{ dev_clamped }}{% elif dev < 0 %} stat-ros-down stat-ros-{{ dev_clamped }}{% endif %}">
```

- [ ] **Step 3: Remove `below_threshold` from hitter tooltip**

In `src/fantasy_baseball/web/templates/season/lineup.html`, replace line 166:

```html
                    <div class="tooltip-row"><span class="tooltip-label">Z-score</span><span class="tooltip-val">{% if st.get('below_threshold') %}<span style="color: var(--text-secondary); font-style: italic;">below threshold</span>{% else %}{{ "%+.1f"|format(st.z_score) }}{% endif %}</span></div>
```

With:

```html
                    <div class="tooltip-row"><span class="tooltip-label">Z-score</span><span class="tooltip-val">{{ "%+.1f"|format(st.z_score) }}</span></div>
```

- [ ] **Step 4: Update pitcher stat cells in `lineup.html`**

In `src/fantasy_baseball/web/templates/season/lineup.html`, replace line 224:

```html
            <td class="stat-cell {{ st.get('color_class', 'stat-neutral') }}{% if cat == 'IP' %} pa-col{% endif %}{% if st.get('significant') %} stat-significant{% endif %}">
```

With:

```html
            {% set dev = st.get('ros_deviation_sgp', 0)|int %}
            {% set dev_abs = dev if dev > 0 else -dev %}
            {% set dev_clamped = [dev_abs, 3]|min %}
            <td class="stat-cell {{ st.get('color_class', 'stat-neutral') }}{% if cat == 'IP' %} pa-col{% endif %}{% if dev > 0 %} stat-ros-up stat-ros-{{ dev_clamped }}{% elif dev < 0 %} stat-ros-down stat-ros-{{ dev_clamped }}{% endif %}">
```

- [ ] **Step 5: Remove `below_threshold` from pitcher tooltip**

In `src/fantasy_baseball/web/templates/season/lineup.html`, replace line 233:

```html
                    <div class="tooltip-row"><span class="tooltip-label">Z-score</span><span class="tooltip-val">{% if st.get('below_threshold') %}<span style="color: var(--text-secondary); font-style: italic;">below threshold</span>{% else %}{{ "%+.1f"|format(st.z_score) }}{% endif %}</span></div>
```

With:

```html
                    <div class="tooltip-row"><span class="tooltip-label">Z-score</span><span class="tooltip-val">{{ "%+.1f"|format(st.z_score) }}</span></div>
```

- [ ] **Step 6: Remove `stat-significant` from `players.html`**

In `src/fantasy_baseball/web/templates/season/players.html`, at line 321, replace:

```javascript
            const sig = p.significant && p.significant[c] ? 'stat-significant' : '';
```

With:

```javascript
            const sig = '';
```

- [ ] **Step 7: Remove `stat-significant` from `waivers_trades.html`**

In `src/fantasy_baseball/web/templates/season/waivers_trades.html`, at line 238, replace:

```html
                <td{% if st.get('significant') %} class="stat-significant"{% endif %}>
```

With:

```html
                <td>
```

At line 294, make the same replacement:

```html
                <td{% if st.get('significant') %} class="stat-significant"{% endif %}>
```

With:

```html
                <td>
```

- [ ] **Step 8: Run tests**

Run: `pytest tests/ -v`
Expected: All pass.

- [ ] **Step 9: Test locally in browser**

Run: `python -m fantasy_baseball.web.season_app`

Open the lineup page. Verify:
- No green checkmarks appear
- Players with ROS deviations >= 1 SGP show green `+` symbols in the top-right corner
- Players with ROS deviations <= -1 SGP show red `×` symbols
- Tooltip z-scores display without "below threshold" text
- Players page and waivers/trades page no longer show checkmarks

- [ ] **Step 10: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/lineup.html src/fantasy_baseball/web/templates/season/players.html src/fantasy_baseball/web/templates/season/waivers_trades.html src/fantasy_baseball/web/static/season.css
git commit -m "feat(lineup): replace significance checkmarks with ROS deviation indicators"
```
