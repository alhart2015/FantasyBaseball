# Overall Pace Marker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Color each player's Slot column based on the average of their per-category z-scores, giving an at-a-glance overall pace indicator.

**Architecture:** New `compute_overall_pace()` function in `pace.py` averages z-scores from the existing per-category pace dict and returns a color class. Wired through `season_data.py` into the template's Slot `<td>`.

**Tech Stack:** Python, Jinja2, existing CSS classes

---

### Task 1: Add `compute_overall_pace()` to pace.py

**Files:**
- Create: `tests/test_analysis/test_overall_pace.py`
- Modify: `src/fantasy_baseball/analysis/pace.py:187` (append new function)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_analysis/test_overall_pace.py`:

```python
from fantasy_baseball.analysis.pace import compute_overall_pace


def test_overall_pace_all_hot():
    """All categories above pace -> overall hot."""
    pace = {
        "R": {"z_score": 2.5},
        "HR": {"z_score": 2.0},
        "RBI": {"z_score": 1.5},
        "SB": {"z_score": 1.8},
        "AVG": {"z_score": 2.2},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 2.0
    assert result["color_class"] == "stat-hot-2"


def test_overall_pace_all_cold():
    """All categories below pace -> overall cold."""
    pace = {
        "R": {"z_score": -1.5},
        "HR": {"z_score": -2.0},
        "RBI": {"z_score": -1.8},
        "SB": {"z_score": -1.2},
        "AVG": {"z_score": -1.5},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == -1.6
    assert result["color_class"] == "stat-cold-1"


def test_overall_pace_mixed_signals():
    """Mixed hot/cold categories -> net result near neutral."""
    pace = {
        "R": {"z_score": 1.5},
        "HR": {"z_score": -1.2},
        "RBI": {"z_score": -0.8},
        "SB": {"z_score": 0.3},
        "AVG": {"z_score": 0.2},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 0.0  # (1.5 - 1.2 - 0.8 + 0.3 + 0.2) / 5 = 0.0
    assert result["color_class"] == "stat-neutral"


def test_overall_pace_skips_pa_ip():
    """PA and IP entries (no z_score) are excluded from the average."""
    pace = {
        "PA": {"actual": 60, "color_class": "stat-neutral"},
        "R": {"z_score": 2.0},
        "HR": {"z_score": 2.0},
        "RBI": {"z_score": 2.0},
        "SB": {"z_score": 2.0},
        "AVG": {"z_score": 2.0},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 2.0
    assert result["color_class"] == "stat-hot-2"


def test_overall_pace_skips_none_z_scores():
    """Categories with z_score=None are excluded."""
    pace = {
        "R": {"z_score": 1.5},
        "HR": {"z_score": None},
        "RBI": {"z_score": 1.5},
        "SB": {"z_score": 1.5},
        "AVG": {"z_score": 1.5},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 1.5
    assert result["color_class"] == "stat-hot-1"


def test_overall_pace_empty_dict():
    """Empty pace dict -> neutral with None avg_z."""
    result = compute_overall_pace({})
    assert result["avg_z"] is None
    assert result["color_class"] == "stat-neutral"


def test_overall_pace_none_input():
    """None pace input -> neutral with None avg_z."""
    result = compute_overall_pace(None)
    assert result["avg_z"] is None
    assert result["color_class"] == "stat-neutral"


def test_overall_pace_pitcher():
    """Pitcher categories work the same way."""
    pace = {
        "IP": {"actual": 18, "color_class": "stat-neutral"},
        "W": {"z_score": 0.5},
        "K": {"z_score": 1.3},
        "SV": {"z_score": 0.0},
        "ERA": {"z_score": -2.5},
        "WHIP": {"z_score": -0.3},
    }
    result = compute_overall_pace(pace)
    # avg = (0.5 + 1.3 + 0.0 - 2.5 - 0.3) / 5 = -1.0 / 5 = -0.2
    assert result["avg_z"] == -0.2
    assert result["color_class"] == "stat-neutral"


def test_overall_pace_light_hot_threshold():
    """Average z exactly at 1.0 boundary -> stat-hot-1."""
    pace = {
        "R": {"z_score": 1.0},
        "HR": {"z_score": 1.0},
        "RBI": {"z_score": 1.0},
        "SB": {"z_score": 1.0},
        "AVG": {"z_score": 1.0},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 1.0
    # _z_to_color: z > Z_LIGHT (1.0) is false for exactly 1.0, so neutral
    # But z_to_color uses > not >=, so 1.0 exactly is neutral
    assert result["color_class"] == "stat-neutral"


def test_overall_pace_just_above_threshold():
    """Average z just above 1.0 -> stat-hot-1."""
    pace = {
        "R": {"z_score": 1.1},
        "HR": {"z_score": 1.1},
        "RBI": {"z_score": 1.1},
        "SB": {"z_score": 1.1},
        "AVG": {"z_score": 1.1},
    }
    result = compute_overall_pace(pace)
    assert result["avg_z"] == 1.1
    assert result["color_class"] == "stat-hot-1"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis/test_overall_pace.py -v`
Expected: FAIL — `ImportError: cannot import name 'compute_overall_pace'`

- [ ] **Step 3: Implement `compute_overall_pace`**

Append to `src/fantasy_baseball/analysis/pace.py` after line 187 (end of `compute_player_pace`):

```python


def compute_overall_pace(pace: dict | None) -> dict:
    """Average per-category z-scores into an overall pace summary.

    Args:
        pace: Dict from compute_player_pace() with UPPERCASE keys.
              Each value may contain a 'z_score' float.

    Returns:
        {"avg_z": float | None, "color_class": str}
    """
    if not pace:
        return {"avg_z": None, "color_class": "stat-neutral"}

    z_scores = [
        entry["z_score"]
        for entry in pace.values()
        if isinstance(entry, dict) and entry.get("z_score") is not None
    ]

    if not z_scores:
        return {"avg_z": None, "color_class": "stat-neutral"}

    avg_z = round(sum(z_scores) / len(z_scores), 1)
    return {"avg_z": avg_z, "color_class": _z_to_color(avg_z)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis/test_overall_pace.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_analysis/test_overall_pace.py src/fantasy_baseball/analysis/pace.py
git commit -m "feat(pace): add compute_overall_pace for slot column coloring"
```

---

### Task 2: Wire overall pace into lineup data assembly

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:513-526` (inside `format_lineup_for_display`)

- [ ] **Step 1: Add import and compute overall_pace in the player loop**

In `src/fantasy_baseball/web/season_data.py`, add the import near the top of `format_lineup_for_display` (after the existing `from fantasy_baseball.models.player import Player` on line 501):

```python
    from fantasy_baseball.analysis.pace import compute_overall_pace
```

Then after line 523 (`"stats": player.pace or {},`), add:

```python
            "overall_pace": compute_overall_pace(player.pace),
```

The entry dict block (lines 513-526) becomes:

```python
        entry = {
            "name": player.name,
            "positions": player.positions,
            "selected_position": pos,
            "player_id": player.yahoo_id or "",
            "status": player.status,
            "wsgp": player.wsgp,
            "games": p.get("games_this_week", 0),
            "is_bench": pos in ("BN", "IL", "DL"),
            "is_il": "IL" in player.status or pos == "IL",
            "stats": player.pace or {},
            "overall_pace": compute_overall_pace(player.pace),
            "rank": player.rank.to_dict(),
            "preseason": player.preseason.to_dict() if player.preseason else None,
        }
```

- [ ] **Step 2: Verify no import errors**

Run: `python -c "from fantasy_baseball.web.season_data import format_lineup_for_display; print('OK')"`
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "feat(lineup): wire overall_pace into lineup display data"
```

---

### Task 3: Apply overall pace color to Slot column in template

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html:106` (hitter Slot td)
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html:168` (pitcher Slot td)

- [ ] **Step 1: Update hitter Slot `<td>` (line 106)**

Change:
```html
            <td>{{ h.selected_position }}</td>
```

To:
```html
            <td class="{{ h.overall_pace.color_class if h.overall_pace else 'stat-neutral' }}"
                {% if h.overall_pace and h.overall_pace.avg_z is not none %}title="Overall pace: {{ '%+.1f'|format(h.overall_pace.avg_z) }} z"{% endif %}>
                {{ h.selected_position }}</td>
```

- [ ] **Step 2: Update pitcher Slot `<td>` (line 168)**

Change:
```html
            <td>{{ p.selected_position }}</td>
```

To:
```html
            <td class="{{ p.overall_pace.color_class if p.overall_pace else 'stat-neutral' }}"
                {% if p.overall_pace and p.overall_pace.avg_z is not none %}title="Overall pace: {{ '%+.1f'|format(p.overall_pace.avg_z) }} z"{% endif %}>
                {{ p.selected_position }}</td>
```

- [ ] **Step 3: Update opponent JS `buildTableHtml` to pass through overall_pace (line 338)**

In the `buildTableHtml` function, update the Slot `<td>` to include the color class. Change line 338:

```javascript
        html += '<td>' + (p.selected_position || 'BN') + '</td>';
```

To:

```javascript
        var opClass = (p.overall_pace && p.overall_pace.color_class) ? p.overall_pace.color_class : '';
        var opTitle = (p.overall_pace && p.overall_pace.avg_z !== null) ? ' title="Overall pace: ' + (p.overall_pace.avg_z > 0 ? '+' : '') + p.overall_pace.avg_z.toFixed(1) + ' z"' : '';
        html += '<td class="' + opClass + '"' + opTitle + '>' + (p.selected_position || 'BN') + '</td>';
```

- [ ] **Step 4: Verify template renders without errors**

Run: `python -c "from fantasy_baseball.web.app import create_app; app = create_app(); print('App created OK')"`
Expected: `App created OK`

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/lineup.html
git commit -m "feat(lineup): color Slot column by overall pace z-score"
```

---

### Task 4: Wire overall_pace into opponent lineup builder

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:356,408-409,418-419` (inside `build_opponent_lineup`)

- [ ] **Step 1: Add import**

In `build_opponent_lineup` (line 356), the existing import block already has:
```python
    from fantasy_baseball.analysis.pace import compute_player_pace
```

Change to:
```python
    from fantasy_baseball.analysis.pace import compute_player_pace, compute_overall_pace
```

- [ ] **Step 2: Add overall_pace for matched players**

After line 408 (`entry["stats"] = compute_player_pace(actuals, projected, ptype)`), add:

```python
        entry["overall_pace"] = compute_overall_pace(entry["stats"])
```

- [ ] **Step 3: Add overall_pace for unmatched players**

After line 418 (`entry["stats"] = {}`), add:

```python
            entry["overall_pace"] = compute_overall_pace(entry["stats"])
```

- [ ] **Step 4: Verify no import errors**

Run: `python -c "from fantasy_baseball.web.season_data import build_opponent_lineup; print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "feat(lineup): include overall_pace in opponent lineup data"
```

---

### Task 5: Run full test suite

- [ ] **Step 1: Run all tests**

Run: `pytest -v`
Expected: All tests pass, no regressions.

- [ ] **Step 2: Final commit if any cleanup needed**

If any fixes were needed, commit them.
