# Lineup Stats Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add color-coded season-to-date stats to the lineup page, showing player performance relative to pre-season projection pace with tooltips.

**Architecture:** New `analysis/pace.py` computes z-scores from game log actuals vs scaled projections. Refresh pipeline bulk-loads game logs, computes pace per player, attaches to roster cache. Template adds stat columns with CSS color classes and hover tooltips.

**Tech Stack:** Python, SQLite, Jinja2/Flask, existing `STAT_VARIANCE` constants

**Spec:** `docs/superpowers/specs/2026-03-27-lineup-stats-design.md`

---

## File Structure

- **Create:** `src/fantasy_baseball/analysis/pace.py` — z-score/pace computation (pure function, no DB)
- **Create:** `tests/test_analysis/test_pace.py` — tests for pace module
- **Modify:** `src/fantasy_baseball/web/season_data.py:440-459` — add game log fetch + pace computation to refresh pipeline
- **Modify:** `src/fantasy_baseball/web/season_data.py:254-293` — pass stats through in `format_lineup_for_display()`
- **Modify:** `src/fantasy_baseball/web/templates/season/lineup.html` — stat columns, color classes, tooltips

---

### Task 1: Create `pace.py` — hitter counting stat z-scores

**Files:**
- Create: `src/fantasy_baseball/analysis/pace.py`
- Create: `tests/test_analysis/test_pace.py`

- [ ] **Step 1: Write failing test for hitter counting stats on pace**

Create `tests/test_analysis/test_pace.py`:

```python
from fantasy_baseball.analysis.pace import compute_player_pace


def test_hitter_counting_on_pace():
    """A hitter exactly on pace for all counting stats gets neutral colors."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    # 10% of PA consumed -> expect 10% of counting stats
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["color_class"] == "stat-neutral"
    assert result["R"]["color_class"] == "stat-neutral"
    assert abs(result["HR"]["z_score"]) < 0.5
    assert result["HR"]["actual"] == 3
    assert result["HR"]["expected"] == 3.0
    assert result["HR"]["projection"] == 30
```

- [ ] **Step 2: Write failing test for hitter counting stats above pace**

```python
def test_hitter_counting_above_pace():
    """A hitter well above pace on HR gets hot color."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    # 10% of PA consumed but 2x the expected HR
    actual = {"pa": 60, "r": 9, "hr": 6, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    # ratio = 6/3 = 2.0, deviation = 1.0, z = 1.0/0.343 = 2.92 -> stat-hot-2
    assert result["HR"]["color_class"] == "stat-hot-2"
    assert result["HR"]["z_score"] > 1.0
```

- [ ] **Step 3: Write failing test for hitter counting stats below pace**

```python
def test_hitter_counting_below_pace():
    """A hitter well below pace on SB gets cold color."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    # 10% of PA consumed but 0 SB (expected 1)
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 0, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    # ratio = 0/1 = 0, deviation = -1.0, z = -1.0/0.715 = -1.40 -> stat-cold-2
    assert result["SB"]["color_class"] == "stat-cold-2"
    assert result["SB"]["z_score"] < -1.0
```

- [ ] **Step 4: Write failing test for expected == 0 edge case**

```python
def test_expected_zero_shows_neutral():
    """When projected stat is 0, show neutral regardless of actual."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 0, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 2, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    # SB projected 0, actual 2 -> cannot compute ratio, show neutral
    assert result["SB"]["color_class"] == "stat-neutral"
    assert result["SB"]["z_score"] == 0.0
```

- [ ] **Step 5: Write failing test for PA column always neutral**

```python
def test_pa_always_neutral():
    """PA is sample-size context, never color-coded."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["PA"]["color_class"] == "stat-neutral"
    assert result["PA"]["actual"] == 60
    assert "z_score" not in result["PA"]
```

- [ ] **Step 6: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_pace.py -v`
Expected: FAIL — module not found

- [ ] **Step 7: Implement `compute_player_pace` for hitter counting stats**

Create `src/fantasy_baseball/analysis/pace.py`:

```python
"""Compute player performance vs projection pace with z-score color coding."""

from fantasy_baseball.utils.constants import INVERSE_STATS, STAT_VARIANCE

# Roto categories by player type
HITTER_COUNTING = ["r", "hr", "rbi", "sb"]
PITCHER_COUNTING = ["w", "k", "sv"]

# Rate stat -> component stat for variance lookup
RATE_COMPONENT = {"avg": "h", "era": "er", "whip": "h_allowed"}

# Color class thresholds
def _z_to_color(z: float) -> str:
    if z > 1.0:
        return "stat-hot-2"
    if z > 0.5:
        return "stat-hot-1"
    if z < -1.0:
        return "stat-cold-2"
    if z < -0.5:
        return "stat-cold-1"
    return "stat-neutral"


def compute_player_pace(
    actual_stats: dict,
    projected_stats: dict,
    player_type: str,
) -> dict:
    """Compute z-scores and color classes for each roto stat.

    Args:
        actual_stats: Season-to-date from game_logs (lowercase keys).
        projected_stats: Full-season from blended_projections (lowercase keys).
        player_type: "hitter" or "pitcher".

    Returns:
        Dict with UPPERCASE display keys, each containing:
        {"actual", "expected", "z_score", "color_class", "projection"}
    """
    result = {}

    if player_type == "hitter":
        opp_key = "pa"
        counting = HITTER_COUNTING
        rate_stats = {"avg": ("h", "ab")}
    else:
        opp_key = "ip"
        counting = PITCHER_COUNTING
        rate_stats = {
            "era": ("er",),
            "whip": ("bb", "h_allowed"),
        }

    actual_opp = actual_stats.get(opp_key, 0) or 0
    proj_opp = projected_stats.get(opp_key, 0) or 0

    # Opportunity column (PA or IP) — always neutral
    result[opp_key.upper()] = {
        "actual": actual_opp if player_type == "hitter" else actual_stats.get("ip", 0),
        "color_class": "stat-neutral",
    }

    # Counting stats
    for stat in counting:
        actual = actual_stats.get(stat, 0) or 0
        proj = projected_stats.get(stat, 0) or 0

        if proj_opp > 0 and proj > 0:
            expected = proj * (actual_opp / proj_opp)
        else:
            expected = 0.0

        if expected > 0:
            ratio = actual / expected
            variance = STAT_VARIANCE.get(stat, 0.0)
            z = (ratio - 1.0) / variance if variance > 0 else 0.0
        else:
            z = 0.0

        display_key = stat.upper()
        result[display_key] = {
            "actual": actual,
            "expected": round(expected, 1),
            "z_score": round(z, 2),
            "color_class": _z_to_color(z),
            "projection": proj,
        }

    return result
```

- [ ] **Step 8: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_pace.py -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add src/fantasy_baseball/analysis/pace.py tests/test_analysis/test_pace.py
git commit -m "feat: add pace.py with hitter counting stat z-scores"
```

---

### Task 2: Add rate stats and pitcher support to `pace.py`

**Files:**
- Modify: `src/fantasy_baseball/analysis/pace.py`
- Modify: `tests/test_analysis/test_pace.py`

- [ ] **Step 1: Write failing test for hitter AVG z-score**

```python
def test_hitter_avg_above_projection():
    """Hitter batting well above projected AVG gets hot color."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 120, "r": 18, "hr": 6, "rbi": 18, "sb": 2, "h": 42, "ab": 108}
    # actual AVG = 42/108 = .389, proj .278, dev = +0.111
    # z = 0.111 / (0.103 * 0.278) = 0.111 / 0.0286 = 3.88 -> stat-hot-2
    result = compute_player_pace(actual, projected, "hitter")
    assert "AVG" in result
    assert result["AVG"]["color_class"] == "stat-hot-2"
    assert result["AVG"]["actual"] == 0.389  # rounded to 3 places
```

- [ ] **Step 2: Write failing test for AVG below minimum sample**

```python
def test_hitter_avg_neutral_below_min_sample():
    """AVG with < 30 PA should be neutral regardless of value."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 20, "r": 3, "hr": 1, "rbi": 3, "sb": 0, "h": 10, "ab": 18}
    # actual AVG = .556, way above proj, but only 20 PA -> neutral
    result = compute_player_pace(actual, projected, "hitter")
    assert result["AVG"]["color_class"] == "stat-neutral"
```

- [ ] **Step 3: Write failing test for counting stats below minimum sample**

```python
def test_counting_neutral_below_min_sample():
    """With < 10 PA, counting stats should be neutral too."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {"pa": 5, "r": 3, "hr": 2, "rbi": 3, "sb": 0, "h": 3, "ab": 5}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["color_class"] == "stat-neutral"
    assert result["AVG"]["color_class"] == "stat-neutral"
```

- [ ] **Step 4: Write failing test for pitcher counting and rate stats**

```python
def test_pitcher_counting_and_rates():
    """Pitcher with good K rate and bad ERA."""
    projected = {"ip": 180, "w": 12, "k": 190, "sv": 0, "er": 60, "bb": 50, "h_allowed": 150,
                 "era": 3.00, "whip": 1.11}
    # 10% IP consumed, Ks on pace, ERA high
    actual = {"ip": 18.0, "k": 22, "w": 1, "sv": 0, "er": 10, "bb": 5, "h_allowed": 16}
    result = compute_player_pace(actual, projected, "pitcher")

    assert "IP" in result
    assert result["IP"]["color_class"] == "stat-neutral"

    # K: expected = 190 * (18/180) = 19, actual 22, ratio 1.16, z = 0.16/0.139 = 1.13 -> hot-2
    assert result["K"]["color_class"] == "stat-hot-2"

    # ERA: actual = 10*9/18 = 5.00, proj 3.00, dev = +2.0
    # z = 2.0 / (0.252 * 3.00) = 2.0 / 0.756 = 2.65
    # ERA is inverse -> negate -> -2.65 -> stat-cold-2
    assert result["ERA"]["color_class"] == "stat-cold-2"
    assert result["ERA"]["z_score"] < -1.0

    # WHIP: actual = (5+16)/18 = 1.167, proj 1.11, dev = +0.057
    # z = 0.057 / (0.143 * 1.11) = 0.057 / 0.159 = 0.36
    # WHIP is inverse -> negate -> -0.36 -> neutral
    assert result["WHIP"]["color_class"] == "stat-neutral"
```

- [ ] **Step 5: Write failing test for pitcher ERA below minimum IP**

```python
def test_pitcher_era_neutral_below_min_ip():
    """ERA with < 10 IP should be neutral, but counting stats with >= 5 IP should be colored."""
    projected = {"ip": 180, "w": 12, "k": 190, "sv": 0, "er": 60, "bb": 50, "h_allowed": 150,
                 "era": 3.00, "whip": 1.11}
    # 5 IP, massively above pace on K (expected ~5.3, actual 15)
    actual = {"ip": 5.0, "k": 15, "w": 0, "sv": 0, "er": 0, "bb": 1, "h_allowed": 3}
    result = compute_player_pace(actual, projected, "pitcher")
    assert result["ERA"]["color_class"] == "stat-neutral"
    assert result["WHIP"]["color_class"] == "stat-neutral"
    # K: ratio = 15/5.3 = 2.83, z = 1.83/0.139 = 13.2 -> stat-hot-2
    assert result["K"]["color_class"] == "stat-hot-2"
```

- [ ] **Step 5b: Write failing test for intermediate color classes**

```python
def test_intermediate_color_classes():
    """z-scores between 0.5 and 1.0 should produce stat-hot-1 / stat-cold-1."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    # 10% PA, HR slightly above pace: expected 3.0, actual 4
    # ratio = 4/3 = 1.33, deviation = 0.33, z = 0.33/0.343 = 0.97 -> stat-hot-1
    actual = {"pa": 60, "r": 9, "hr": 4, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["color_class"] == "stat-hot-1"
    assert 0.5 < result["HR"]["z_score"] < 1.0
```

- [ ] **Step 5c: Write failing test for 10-29 PA middle tier**

```python
def test_middle_sample_counting_colored_rates_neutral():
    """With 10-29 PA: counting stats colored, AVG neutral."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    # 20 PA, HR way above pace
    actual = {"pa": 20, "r": 3, "hr": 5, "rbi": 3, "sb": 0, "h": 10, "ab": 18}
    result = compute_player_pace(actual, projected, "hitter")
    # Counting stats should be colored (>= 10 PA threshold)
    assert result["HR"]["color_class"] != "stat-neutral"
    # Rate stat should be neutral (< 30 PA threshold)
    assert result["AVG"]["color_class"] == "stat-neutral"
```

- [ ] **Step 6: Run tests to verify new ones fail**

Run: `python -m pytest tests/test_analysis/test_pace.py -v`
Expected: new tests FAIL

- [ ] **Step 7: Implement rate stats, pitcher support, and sample size thresholds**

Update `compute_player_pace` in `src/fantasy_baseball/analysis/pace.py` to add after counting stats loop:

```python
    # Sample size thresholds
    if player_type == "hitter":
        min_for_counting = 10   # PA
        min_for_rates = 30      # PA
    else:
        min_for_counting = 5    # IP
        min_for_rates = 10      # IP

    # If below counting threshold, force all counting stats to neutral
    if actual_opp < min_for_counting:
        for stat in counting:
            key = stat.upper()
            if key in result:
                result[key]["color_class"] = "stat-neutral"
                result[key]["z_score"] = 0.0

    # Rate stats
    if player_type == "hitter":
        actual_ab = actual_stats.get("ab", 0) or 0
        actual_h = actual_stats.get("h", 0) or 0
        actual_avg = round(actual_h / actual_ab, 3) if actual_ab > 0 else 0.0
        proj_avg = projected_stats.get("avg", 0) or 0

        if actual_opp >= min_for_rates and proj_avg > 0:
            dev = actual_avg - proj_avg
            variance = STAT_VARIANCE.get("h", 0.0)
            z = dev / (variance * proj_avg) if variance > 0 else 0.0
        else:
            z = 0.0

        result["AVG"] = {
            "actual": actual_avg,
            "expected": proj_avg,
            "z_score": round(z, 2),
            "color_class": _z_to_color(z) if actual_opp >= min_for_rates else "stat-neutral",
            "projection": proj_avg,
        }
    else:
        actual_ip = actual_stats.get("ip", 0) or 0
        actual_er = actual_stats.get("er", 0) or 0
        actual_bb = actual_stats.get("bb", 0) or 0
        actual_ha = actual_stats.get("h_allowed", 0) or 0

        # ERA
        actual_era = round((actual_er * 9 / actual_ip), 2) if actual_ip > 0 else 0.0
        proj_era = projected_stats.get("era", 0) or 0
        if actual_opp >= min_for_rates and proj_era > 0:
            dev = actual_era - proj_era
            variance = STAT_VARIANCE.get("er", 0.0)
            z = dev / (variance * proj_era) if variance > 0 else 0.0
            z = -z  # inverse stat: lower is better
        else:
            z = 0.0
        result["ERA"] = {
            "actual": actual_era,
            "expected": proj_era,
            "z_score": round(z, 2),
            "color_class": _z_to_color(z) if actual_opp >= min_for_rates else "stat-neutral",
            "projection": proj_era,
        }

        # WHIP
        actual_whip = round((actual_bb + actual_ha) / actual_ip, 2) if actual_ip > 0 else 0.0
        proj_whip = projected_stats.get("whip", 0) or 0
        if actual_opp >= min_for_rates and proj_whip > 0:
            dev = actual_whip - proj_whip
            variance = STAT_VARIANCE.get("h_allowed", 0.0)
            z = dev / (variance * proj_whip) if variance > 0 else 0.0
            z = -z  # inverse: lower WHIP is better
        else:
            z = 0.0
        result["WHIP"] = {
            "actual": actual_whip,
            "expected": proj_whip,
            "z_score": round(z, 2),
            "color_class": _z_to_color(z) if actual_opp >= min_for_rates else "stat-neutral",
            "projection": proj_whip,
        }

    return result
```

- [ ] **Step 8: Run all tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_pace.py -v`
Expected: all PASS

- [ ] **Step 9: Commit**

```bash
git add src/fantasy_baseball/analysis/pace.py tests/test_analysis/test_pace.py
git commit -m "feat: add rate stats, pitcher support, sample size thresholds to pace.py"
```

---

### Task 3: Add no-game-logs and no-projection edge cases

**Files:**
- Modify: `src/fantasy_baseball/analysis/pace.py`
- Modify: `tests/test_analysis/test_pace.py`

- [ ] **Step 1: Write failing test for player with no game logs**

```python
def test_no_game_logs_shows_dashes():
    """Player with no actual stats gets None actuals (template renders as dashes)."""
    projected = {"pa": 600, "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "avg": 0.278}
    actual = {}  # no game logs at all
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["actual"] == 0
    assert result["HR"]["color_class"] == "stat-neutral"
    assert result["PA"]["actual"] == 0
```

- [ ] **Step 2: Write failing test for player with no projection**

```python
def test_no_projection_shows_actuals_neutral():
    """Player not matched to projections — show actuals, all neutral."""
    projected = {}  # unmatched
    actual = {"pa": 60, "r": 9, "hr": 3, "rbi": 9, "sb": 1, "h": 15, "ab": 54}
    result = compute_player_pace(actual, projected, "hitter")
    assert result["HR"]["actual"] == 3
    assert result["HR"]["color_class"] == "stat-neutral"
    assert result["HR"].get("z_score", 0) == 0.0
```

- [ ] **Step 3: Run tests, verify new ones handle edge cases correctly**

Run: `python -m pytest tests/test_analysis/test_pace.py -v`

If existing implementation already handles these (empty dicts result in 0 PA -> below min sample -> neutral), tests will pass. If not, add guards at the top of `compute_player_pace`:

```python
    # Handle missing projections
    if not projected_stats:
        # Return actuals with neutral coloring, no z-scores
        ...
```

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/analysis/pace.py tests/test_analysis/test_pace.py
git commit -m "feat: handle no-game-logs and no-projection edge cases in pace.py"
```

---

### Task 4: Add game log bulk query to refresh pipeline

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:440-459`

- [ ] **Step 1: Write failing test for game log loading in refresh**

Add to `tests/test_web/test_season_data.py`:

```python
def test_roster_cache_includes_stats(tmp_path, monkeypatch):
    """After refresh, roster entries should include a 'stats' dict."""
    # This is an integration-style test — we'll mock the data sources
    # and verify the pipeline attaches stats to roster entries.
    from fantasy_baseball.web.season_data import format_lineup_for_display

    roster = [
        {"name": "Juan Soto", "positions": ["OF"], "selected_position": "OF",
         "player_id": "1", "status": "", "wsgp": 3.0, "player_type": "hitter",
         "r": 90, "hr": 30, "rbi": 90, "sb": 10, "h": 150, "ab": 540, "pa": 600, "avg": 0.278,
         "stats": {
             "PA": {"actual": 102, "color_class": "stat-neutral"},
             "R": {"actual": 19, "expected": 15.3, "z_score": 1.2, "color_class": "stat-hot-2", "projection": 90},
             "HR": {"actual": 9, "expected": 5.1, "z_score": 1.6, "color_class": "stat-hot-2", "projection": 30},
             "RBI": {"actual": 18, "expected": 15.3, "z_score": 0.3, "color_class": "stat-neutral", "projection": 90},
             "SB": {"actual": 2, "expected": 1.7, "z_score": 0.2, "color_class": "stat-neutral", "projection": 10},
             "AVG": {"actual": 0.298, "expected": 0.278, "z_score": 0.7, "color_class": "stat-hot-1", "projection": 0.278},
         }},
    ]
    result = format_lineup_for_display(roster, {"moves": []})
    assert "stats" in result["hitters"][0]
    assert result["hitters"][0]["stats"]["HR"]["actual"] == 9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_web/test_season_data.py::test_roster_cache_includes_stats -v`
Expected: FAIL — `stats` key not passed through

- [ ] **Step 3: Update `format_lineup_for_display()` to pass through stats**

In `src/fantasy_baseball/web/season_data.py`, modify the `entry` dict construction in `format_lineup_for_display()` (line 266-276). Add `"stats"` to the entry:

```python
        entry = {
            "name": p["name"],
            "positions": p.get("positions", []),
            "selected_position": pos,
            "player_id": p.get("player_id", ""),
            "status": p.get("status", ""),
            "wsgp": p.get("wsgp", 0),
            "games": p.get("games_this_week", 0),
            "is_bench": pos in ("BN", "IL", "DL"),
            "is_il": "IL" in p.get("status", "") or pos == "IL",
            "stats": p.get("stats", {}),
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_web/test_season_data.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py
git commit -m "feat: pass stats dict through format_lineup_for_display"
```

---

### Task 5: Add pace computation to refresh pipeline

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:440-459`

- [ ] **Step 1: Add game log bulk query and pace computation to `run_full_refresh()`**

In `src/fantasy_baseball/web/season_data.py`, insert a new step between the current Step 6 (match roster to projections, line 440-457) and `write_cache("roster", ...)` (line 459).

Add these imports near the top of the function (or at the module level with other lazy imports):

```python
from fantasy_baseball.analysis.pace import compute_player_pace
```

Note: `get_db_connection` is already imported at line 386 of `run_full_refresh()` (aliased from `get_connection`). Reuse it — do NOT add a duplicate import.

Insert before `write_cache("roster", roster_with_proj, cache_dir)`:

```python
        # --- Step 6b: Compute season-to-date pace vs projections ---
        _set_refresh_progress("Computing player pace...")
        pace_conn = get_db_connection()
        try:
            season_year = config.season_year

            # Bulk-load hitter season totals
            hitter_logs = {}
            rows = pace_conn.execute(
                "SELECT name, SUM(pa) as pa, SUM(ab) as ab, SUM(h) as h, "
                "SUM(r) as r, SUM(hr) as hr, SUM(rbi) as rbi, SUM(sb) as sb "
                "FROM game_logs WHERE season = ? AND player_type = 'hitter' "
                "GROUP BY name", (season_year,)
            ).fetchall()
            for row in rows:
                norm = normalize_name(row["name"])
                hitter_logs[norm] = {
                    "pa": row["pa"] or 0, "ab": row["ab"] or 0, "h": row["h"] or 0,
                    "r": row["r"] or 0, "hr": row["hr"] or 0, "rbi": row["rbi"] or 0, "sb": row["sb"] or 0,
                }

            # Bulk-load pitcher season totals
            pitcher_logs = {}
            rows = pace_conn.execute(
                "SELECT name, SUM(ip) as ip, SUM(k) as k, SUM(w) as w, SUM(sv) as sv, "
                "SUM(er) as er, SUM(bb) as bb, SUM(h_allowed) as h_allowed "
                "FROM game_logs WHERE season = ? AND player_type = 'pitcher' "
                "GROUP BY name", (season_year,)
            ).fetchall()
            for row in rows:
                norm = normalize_name(row["name"])
                pitcher_logs[norm] = {
                    "ip": row["ip"] or 0, "k": row["k"] or 0, "w": row["w"] or 0, "sv": row["sv"] or 0,
                    "er": row["er"] or 0, "bb": row["bb"] or 0, "h_allowed": row["h_allowed"] or 0,
                }

            # Attach pace data to each roster player
            for entry in roster_with_proj:
                norm = normalize_name(entry["name"])
                ptype = entry.get("player_type", "hitter")
                if ptype == "hitter":
                    actuals = hitter_logs.get(norm, {})
                else:
                    actuals = pitcher_logs.get(norm, {})
                projected = {k: entry.get(k, 0) for k in
                             (["pa", "r", "hr", "rbi", "sb", "h", "ab", "avg"]
                              if ptype == "hitter"
                              else ["ip", "w", "k", "sv", "er", "bb", "h_allowed", "era", "whip"])}
                entry["stats"] = compute_player_pace(actuals, projected, ptype)
        finally:
            pace_conn.close()
```

Note: `config.season_year` is available from the loaded config. Check that this field exists — look at `config/league.yaml` for `season_year`. If it doesn't exist, use `datetime.now().year` as fallback.

- [ ] **Step 2: Verify existing tests still pass**

Run: `python -m pytest tests/test_web/ -v`
Expected: all PASS (the refresh pipeline tests may not exercise this path, but format tests should still work)

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "feat: compute player pace from game logs in refresh pipeline"
```

---

### Task 6: Update lineup template with stat columns and color coding

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html`

- [ ] **Step 1: Add CSS for color classes and tooltips**

In `lineup.html`, add to the existing `<style>` block (after line 16):

```css
/* Stat color coding */
.stat-hot-2 { color: #22c55e; font-weight: 600; background: rgba(34, 197, 94, 0.15); }
.stat-hot-1 { color: #86efac; background: rgba(134, 239, 172, 0.08); }
.stat-neutral { }
.stat-cold-1 { color: #fca5a5; background: rgba(252, 165, 165, 0.08); }
.stat-cold-2 { color: #ef4444; font-weight: 600; background: rgba(239, 68, 68, 0.15); }

.stat-cell { position: relative; cursor: default; }
.stat-cell .tooltip {
    display: none; position: absolute; bottom: 100%; left: 50%; transform: translateX(-50%);
    background: var(--panel-bg, #1a1a2e); border: 1px solid var(--panel-border, #3a3a5e);
    border-radius: 6px; padding: 10px 14px; font-size: 11px; white-space: nowrap;
    z-index: 100; box-shadow: 0 4px 12px rgba(0,0,0,0.5); min-width: 180px; text-align: left;
}
.stat-cell:hover .tooltip { display: block; }
.tooltip-row { display: flex; justify-content: space-between; padding: 2px 0; }
.tooltip-label { color: var(--text-secondary, #8888aa); }
.tooltip-val { font-weight: 500; }
.tooltip-divider { border-top: 1px solid var(--panel-border, #2a2a3e); margin: 4px 0; }
.pa-col { color: var(--text-secondary) !important; font-size: 12px; }
```

- [ ] **Step 2: Replace hitter table with stat columns**

Replace the hitter `<table>` (lines 58-87) with:

```html
<table class="data-table">
    <thead>
        <tr>
            <th>Slot</th>
            <th>Player</th>
            <th>Elig</th>
            <th>PA</th>
            <th>R</th>
            <th>HR</th>
            <th>RBI</th>
            <th>SB</th>
            <th>AVG</th>
            <th>wSGP</th>
            <th></th>
        </tr>
    </thead>
    <tbody>
    {% for h in lineup.hitters %}
        {% set s = h.stats or {} %}
        <tr class="{% if h.is_bench %}bench-row{% endif %}{% if h.is_il and not h.is_bench %} il-active-row{% endif %}">
            <td>{{ h.selected_position }}</td>
            <td style="text-align: left; font-weight: 500;">{{ h.name }}</td>
            <td style="text-align: left; color: var(--text-secondary); font-size: 11px;">
                {{ h.positions | join(", ") }}
            </td>
            {% for cat in ["PA", "R", "HR", "RBI", "SB", "AVG"] %}
            {% set st = s.get(cat, {}) %}
            <td class="stat-cell {{ st.get('color_class', 'stat-neutral') }} {% if cat == 'PA' %}pa-col{% endif %}">
                {% if st and st.get('actual') is not none and (st.get('actual') != 0 or cat == 'PA' or s.get('PA', {}).get('actual', 0) > 0) %}
                    {% if cat == 'AVG' %}{{ "%.3f"|format(st.actual) }}{% else %}{{ st.actual }}{% endif %}
                {% else %}—{% endif %}
                {% if st and st.get('z_score') is defined and cat != 'PA' %}
                <div class="tooltip">
                    <div style="font-weight: 600; margin-bottom: 6px;">{{ h.name }} — {{ cat }}</div>
                    <div class="tooltip-row"><span class="tooltip-label">Actual</span><span class="tooltip-val">{% if cat == 'AVG' %}{{ "%.3f"|format(st.actual) }}{% else %}{{ st.actual }}{% endif %}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">Expected pace</span><span class="tooltip-val">{% if cat == 'AVG' %}{{ "%.3f"|format(st.expected) }}{% else %}{{ st.expected }}{% endif %}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">Z-score</span><span class="tooltip-val">{{ "%+.1f"|format(st.z_score) }}</span></div>
                    <div class="tooltip-divider"></div>
                    <div class="tooltip-row"><span class="tooltip-label">Pre-season proj</span><span class="tooltip-val">{% if cat == 'AVG' %}{{ "%.3f"|format(st.projection) }}{% else %}{{ st.projection }} {{ cat }}{% endif %}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">ROS proj</span><span class="tooltip-val" style="color: var(--text-secondary); font-style: italic;">Coming soon</span></div>
                </div>
                {% endif %}
            </td>
            {% endfor %}
            <td>{{ "%.2f" | format(h.wsgp) if h.wsgp else "—" }}</td>
            <td>
                {% if h.status %}
                <span class="badge badge-il">{{ h.status }}</span>
                {% endif %}
            </td>
        </tr>
    {% endfor %}
    </tbody>
</table>
```

- [ ] **Step 3: Replace pitcher table with stat columns**

Replace the pitcher `<table>` (lines 94-119) with the same pattern but using pitcher stats (IP, W, K, SV, ERA, WHIP) and no Elig column.

```html
<table class="data-table">
    <thead>
        <tr>
            <th>Slot</th>
            <th>Player</th>
            <th>IP</th>
            <th>W</th>
            <th>K</th>
            <th>SV</th>
            <th>ERA</th>
            <th>WHIP</th>
            <th>wSGP</th>
            <th></th>
        </tr>
    </thead>
    <tbody>
    {% for p in lineup.pitchers %}
        {% set s = p.stats or {} %}
        <tr class="{% if p.is_bench %}bench-row{% endif %}{% if p.is_il and not p.is_bench %} il-active-row{% endif %}">
            <td>{{ p.selected_position }}</td>
            <td style="text-align: left; font-weight: 500;">{{ p.name }}</td>
            {% for cat in ["IP", "W", "K", "SV", "ERA", "WHIP"] %}
            {% set st = s.get(cat, {}) %}
            <td class="stat-cell {{ st.get('color_class', 'stat-neutral') }} {% if cat == 'IP' %}pa-col{% endif %}">
                {% if st and st.get('actual') is not none and (st.get('actual') != 0 or cat == 'IP' or s.get('IP', {}).get('actual', 0) > 0) %}
                    {% if cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(st.actual) }}{% else %}{{ st.actual }}{% endif %}
                {% else %}—{% endif %}
                {% if st and st.get('z_score') is defined and cat != 'IP' %}
                <div class="tooltip">
                    <div style="font-weight: 600; margin-bottom: 6px;">{{ p.name }} — {{ cat }}</div>
                    <div class="tooltip-row"><span class="tooltip-label">Actual</span><span class="tooltip-val">{% if cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(st.actual) }}{% else %}{{ st.actual }}{% endif %}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">Expected pace</span><span class="tooltip-val">{% if cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(st.expected) }}{% else %}{{ st.expected }}{% endif %}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">Z-score</span><span class="tooltip-val">{{ "%+.1f"|format(st.z_score) }}</span></div>
                    <div class="tooltip-divider"></div>
                    <div class="tooltip-row"><span class="tooltip-label">Pre-season proj</span><span class="tooltip-val">{% if cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(st.projection) }}{% else %}{{ st.projection }} {{ cat }}{% endif %}</span></div>
                    <div class="tooltip-row"><span class="tooltip-label">ROS proj</span><span class="tooltip-val" style="color: var(--text-secondary); font-style: italic;">Coming soon</span></div>
                </div>
                {% endif %}
            </td>
            {% endfor %}
            <td>{{ "%.2f" | format(p.wsgp) if p.wsgp else "—" }}</td>
            <td>
                {% if p.status %}
                <span class="badge badge-il">{{ p.status }}</span>
                {% endif %}
            </td>
        </tr>
    {% endfor %}
    </tbody>
</table>
```

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/lineup.html
git commit -m "feat: add stat columns with color coding and tooltips to lineup page"
```

---

### Task 7: Run full test suite and verify

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest -v`
Expected: all tests pass, no regressions

- [ ] **Step 2: Verify pace module imports cleanly**

Run: `python -c "from fantasy_baseball.analysis.pace import compute_player_pace; print('OK')"`
Expected: prints OK

- [ ] **Step 3: Spot-check the template renders without errors**

Run: `python -c "from fantasy_baseball.web.season_app import create_app; app = create_app(); c = app.test_client(); r = c.get('/lineup'); print(r.status_code)"`
Expected: 200 (the page renders, even with empty/missing cache data)

- [ ] **Step 4: Final commit if any cleanup needed**

If any adjustments were made, commit them.
