# Stat Significance Indicators Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show a green checkmark on individual player stat cells that have accumulated enough season-to-date sample size to be empirically significant, based on sabermetric stabilization research.

**Architecture:** Define stabilization thresholds in `constants.py`. Add `is_significant(cat)` methods to `HitterStats`/`PitcherStats` dataclasses. Inject `significant` flags into `compute_player_pace` output (the choke point for all actual-stat rendering). Add a `.stat-significant` CSS class with a green checkmark pseudo-element. Update three templates: lineup, waivers/trades, and the player browse table.

**Tech Stack:** Python dataclasses, Jinja2 templates, vanilla CSS/JS

---

### Task 1: Add stabilization thresholds to constants.py

**Files:**
- Modify: `src/fantasy_baseball/utils/constants.py`
- Test: `tests/test_models/test_player.py`

- [ ] **Step 1: Add thresholds**

Add at the end of `src/fantasy_baseball/utils/constants.py`, before `DEFAULT_SGP_DENOMINATORS`:

```python
# Empirical stat stabilization thresholds (Carleton / FanGraphs research).
# The sample size at which a stat reaches ~50% reliability (r ≈ 0.70).
# Stats not listed here (R, RBI, SB, W, SV, AVG) are treated as always
# significant — either they lack a canonical threshold or it exceeds a
# full season of data.
STABILIZATION_THRESHOLDS: dict[str, tuple[int, str]] = {
    # category: (threshold, unit)
    "HR": (170, "pa"),     # HR rate stabilizes at ~170 PA
    "K": (70, "bf"),       # K rate stabilizes at ~70 BF
    "ERA": (630, "bf"),    # Compound stat, ~630 BF estimated
    "WHIP": (570, "bf"),   # Midpoint of component range (540-670 BF)
}
```

- [ ] **Step 2: Commit**

```bash
git add src/fantasy_baseball/utils/constants.py
git commit -m "feat: add empirical stat stabilization thresholds"
```

---

### Task 2: Add is_significant() to HitterStats and PitcherStats

**Files:**
- Modify: `src/fantasy_baseball/models/player.py`
- Test: `tests/test_models/test_player.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_models/test_player.py` (or append if it exists). Check first:

```bash
ls tests/test_models/test_player.py 2>/dev/null && echo "exists" || echo "create"
```

Add these tests:

```python
import pytest
from fantasy_baseball.models.player import HitterStats, PitcherStats


class TestHitterSignificance:
    def test_hr_below_threshold(self):
        stats = HitterStats(pa=169, r=20, hr=8, rbi=25, sb=3, h=45, ab=150, avg=0.300)
        assert stats.is_significant("HR") is False

    def test_hr_at_threshold(self):
        stats = HitterStats(pa=170, r=20, hr=8, rbi=25, sb=3, h=45, ab=150, avg=0.300)
        assert stats.is_significant("HR") is True

    def test_hr_above_threshold(self):
        stats = HitterStats(pa=300, r=40, hr=15, rbi=50, sb=5, h=80, ab=270, avg=0.296)
        assert stats.is_significant("HR") is True

    def test_counting_stats_always_significant(self):
        stats = HitterStats(pa=1)  # minimal sample
        assert stats.is_significant("R") is True
        assert stats.is_significant("RBI") is True
        assert stats.is_significant("SB") is True
        assert stats.is_significant("AVG") is True

    def test_significant_dict(self):
        stats = HitterStats(pa=100)
        d = stats.significant_dict()
        assert d == {"R": True, "HR": False, "RBI": True, "SB": True, "AVG": True}


class TestPitcherSignificance:
    def test_k_below_threshold(self):
        # BF = ip*3 + h_allowed + bb = 20*3 + 5 + 3 = 68
        stats = PitcherStats(ip=20, k=25, w=2, sv=0, er=8, bb=3, h_allowed=5)
        assert stats.is_significant("K") is False

    def test_k_at_threshold(self):
        # BF = ip*3 + h_allowed + bb = 20*3 + 7 + 3 = 70
        stats = PitcherStats(ip=20, k=25, w=2, sv=0, er=8, bb=3, h_allowed=7)
        assert stats.is_significant("K") is True

    def test_era_below_threshold(self):
        # BF = 60*3 + 50 + 20 = 250 < 630
        stats = PitcherStats(ip=60, k=55, w=5, sv=0, er=25, bb=20, h_allowed=50)
        assert stats.is_significant("ERA") is False

    def test_era_above_threshold(self):
        # BF = 110*3 + 100 + 35 = 465... still below. Need more.
        # BF = 180*3 + 160 + 55 = 755 > 630
        stats = PitcherStats(ip=180, k=180, w=12, sv=0, er=60, bb=55, h_allowed=160)
        assert stats.is_significant("ERA") is True

    def test_whip_threshold(self):
        # BF = 160*3 + 140 + 50 = 670 > 570
        stats = PitcherStats(ip=160, k=150, w=10, sv=0, er=55, bb=50, h_allowed=140)
        assert stats.is_significant("WHIP") is True

    def test_counting_stats_always_significant(self):
        stats = PitcherStats(ip=1)  # minimal sample
        assert stats.is_significant("W") is True
        assert stats.is_significant("SV") is True

    def test_significant_dict(self):
        # BF = 20*3 + 5 + 3 = 68 (below K=70, ERA=630, WHIP=570)
        stats = PitcherStats(ip=20, k=25, w=2, sv=0, er=8, bb=3, h_allowed=5)
        d = stats.significant_dict()
        assert d == {"W": True, "K": False, "SV": True, "ERA": False, "WHIP": False}
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_models/test_player.py -v
```

Expected: FAIL — `is_significant` and `significant_dict` not defined.

- [ ] **Step 3: Implement is_significant and significant_dict on HitterStats**

Add to `HitterStats` in `src/fantasy_baseball/models/player.py`, after the `compute_sgp` method:

```python
    def is_significant(self, cat: str) -> bool:
        """Check if this stat has enough sample to be empirically significant."""
        from fantasy_baseball.utils.constants import STABILIZATION_THRESHOLDS
        entry = STABILIZATION_THRESHOLDS.get(cat)
        if entry is None:
            return True  # No threshold — always significant
        threshold, unit = entry
        if unit == "pa":
            return self.pa >= threshold
        return True  # Hitters don't use BF-based thresholds

    def significant_dict(self) -> dict[str, bool]:
        """Return significance for all 5 hitting roto categories."""
        return {cat: self.is_significant(cat) for cat in ["R", "HR", "RBI", "SB", "AVG"]}
```

- [ ] **Step 4: Implement is_significant and significant_dict on PitcherStats**

Add to `PitcherStats` in `src/fantasy_baseball/models/player.py`, after the `compute_sgp` method:

```python
    def is_significant(self, cat: str) -> bool:
        """Check if this stat has enough sample to be empirically significant."""
        from fantasy_baseball.utils.constants import STABILIZATION_THRESHOLDS
        entry = STABILIZATION_THRESHOLDS.get(cat)
        if entry is None:
            return True  # No threshold — always significant
        threshold, unit = entry
        if unit == "bf":
            bf = self.ip * 3 + self.h_allowed + self.bb
            return bf >= threshold
        return True  # Pitchers don't use PA-based thresholds

    def significant_dict(self) -> dict[str, bool]:
        """Return significance for all 5 pitching roto categories."""
        return {cat: self.is_significant(cat) for cat in ["W", "K", "SV", "ERA", "WHIP"]}
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_models/test_player.py -v
```

Expected: All PASS.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/models/player.py tests/test_models/test_player.py
git commit -m "feat: add is_significant() to HitterStats and PitcherStats"
```

---

### Task 3: Add significance to compute_player_pace output

**Files:**
- Modify: `src/fantasy_baseball/analysis/pace.py`
- Test: `tests/test_analysis/test_pace.py`

The `compute_player_pace` function is the choke point — every page that displays actual player stats calls it. Adding `"significant"` to each stat entry here propagates to lineup, waivers, and buy-low pages automatically.

- [ ] **Step 1: Write failing test**

Add to `tests/test_analysis/test_pace.py`:

```python
def test_significance_flags_in_pace_output():
    """Pace output includes 'significant' key per stat based on stabilization thresholds."""
    from fantasy_baseball.analysis.pace import compute_player_pace

    # Hitter with 100 PA — below HR threshold (170) but above for counting stats
    actual = {"pa": 100, "ab": 90, "h": 25, "r": 12, "hr": 5, "rbi": 15, "sb": 2}
    projected = {"pa": 600, "ab": 540, "h": 150, "r": 80, "hr": 25, "rbi": 85, "sb": 10, "avg": 0.278}
    result = compute_player_pace(actual, projected, "hitter")

    assert result["R"]["significant"] is True
    assert result["HR"]["significant"] is False  # 100 PA < 170
    assert result["RBI"]["significant"] is True
    assert result["SB"]["significant"] is True
    assert result["AVG"]["significant"] is True


def test_significance_pitcher():
    """Pitcher significance uses BF = ip*3 + h_allowed + bb."""
    from fantasy_baseball.analysis.pace import compute_player_pace

    # BF = 25*3 + 20 + 8 = 103 — above K (70) but below ERA (630) and WHIP (570)
    actual = {"ip": 25, "k": 30, "w": 2, "sv": 0, "er": 10, "bb": 8, "h_allowed": 20}
    projected = {"ip": 180, "w": 12, "k": 180, "sv": 0, "er": 60, "bb": 50, "h_allowed": 160, "era": 3.00, "whip": 1.17}
    result = compute_player_pace(actual, projected, "pitcher")

    assert result["K"]["significant"] is True   # 103 >= 70
    assert result["ERA"]["significant"] is False  # 103 < 630
    assert result["WHIP"]["significant"] is False  # 103 < 570
    assert result["W"]["significant"] is True
    assert result["SV"]["significant"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
pytest tests/test_analysis/test_pace.py::test_significance_flags_in_pace_output -v
pytest tests/test_analysis/test_pace.py::test_significance_pitcher -v
```

Expected: FAIL — KeyError on `result["R"]["significant"]`.

- [ ] **Step 3: Implement**

In `src/fantasy_baseball/analysis/pace.py`, add the import at the top:

```python
from fantasy_baseball.utils.constants import INVERSE_STATS, STAT_VARIANCE, STABILIZATION_THRESHOLDS
```

Then add a helper function after the existing `_z_to_color` function:

```python
def _is_significant(cat: str, player_type: str, actual_opp: float, actual_stats: dict) -> bool:
    """Check if a stat has enough sample to be empirically significant."""
    entry = STABILIZATION_THRESHOLDS.get(cat)
    if entry is None:
        return True
    threshold, unit = entry
    if unit == "pa":
        return actual_opp >= threshold  # actual_opp is PA for hitters
    if unit == "bf":
        if player_type == "pitcher":
            bf = actual_stats.get("ip", 0) * 3 + actual_stats.get("h_allowed", 0) + actual_stats.get("bb", 0)
            return bf >= threshold
    return True
```

Then in the `compute_player_pace` function, add `"significant"` to each stat entry. In the counting stats loop (after the `result[display_key] = {` block), add the key:

Find this block (around line 101-107):

```python
        result[display_key] = {
            "actual": actual,
            "expected": round(expected, 1),
            "z_score": round(z, 2),
            "color_class": _z_to_color(z) if abs(actual - expected) >= COUNTING_MIN_ABS_DIFF else "stat-neutral",
            "projection": round(proj),
        }
```

Change to:

```python
        result[display_key] = {
            "actual": actual,
            "expected": round(expected, 1),
            "z_score": round(z, 2),
            "color_class": _z_to_color(z) if abs(actual - expected) >= COUNTING_MIN_ABS_DIFF else "stat-neutral",
            "projection": round(proj),
            "significant": _is_significant(display_key, player_type, actual_opp, actual_stats),
        }
```

Do the same for every rate stat entry in the function. There are three rate stat blocks:

1. **AVG** (hitter, around line 125-131) — add `"significant": _is_significant("AVG", player_type, actual_opp, actual_stats),`
2. **ERA** (pitcher, around line 144-151) — add `"significant": _is_significant("ERA", player_type, actual_opp, actual_stats),`
3. **WHIP** (pitcher, around line 161-167) — add `"significant": _is_significant("WHIP", player_type, actual_opp, actual_stats),`

- [ ] **Step 4: Run tests to verify they pass**

```bash
pytest tests/test_analysis/test_pace.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/pace.py tests/test_analysis/test_pace.py
git commit -m "feat: add significance flags to compute_player_pace output"
```

---

### Task 4: Add CSS for green checkmark

**Files:**
- Modify: `src/fantasy_baseball/web/static/season.css`

- [ ] **Step 1: Add the `.stat-significant` style**

Add after the stat color classes (after line 384 in `season.css`):

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

- [ ] **Step 2: Commit**

```bash
git add src/fantasy_baseball/web/static/season.css
git commit -m "feat: add .stat-significant CSS with green checkmark"
```

---

### Task 5: Update lineup.html to show significance checkmarks

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html`

The lineup template renders stat cells via Jinja. Each stat entry `st` now has a `significant` key from `compute_player_pace`.

- [ ] **Step 1: Update hitter stat cells (around line 113)**

Find:
```jinja2
<td class="stat-cell {{ st.get('color_class', 'stat-neutral') }}{% if cat == 'PA' %} pa-col{% endif %}">
```

Change to:
```jinja2
<td class="stat-cell {{ st.get('color_class', 'stat-neutral') }}{% if cat == 'PA' %} pa-col{% endif %}{% if st.get('significant') %} stat-significant{% endif %}">
```

- [ ] **Step 2: Update pitcher stat cells (around line 172)**

Find the equivalent `<td class="stat-cell ...` line in the pitcher section and apply the same change:

```jinja2
<td class="stat-cell {{ st.get('color_class', 'stat-neutral') }}{% if cat == 'IP' %} pa-col{% endif %}{% if st.get('significant') %} stat-significant{% endif %}">
```

- [ ] **Step 3: Run existing tests**

```bash
pytest tests/test_web/ -v
```

Expected: All PASS (template changes don't break existing tests).

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/lineup.html
git commit -m "feat: show significance checkmarks on lineup stat cells"
```

---

### Task 6: Update waivers_trades.html buy-low cards to show significance checkmarks

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`

Buy-low stat cells currently have no class. Add `stat-significant` when the stat is significant.

- [ ] **Step 1: Update buy-low target actual cells (around line 294)**

Find (the `<td>` for "Actual" column in the buy-low targets table):
```jinja2
<td>{% if cat in ['AVG'] %}{{ "%.3f"|format(st.get('actual', 0)) }}{% elif cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(st.get('actual', 0)) }}{% else %}{{ st.get('actual', 0) }}{% endif %}</td>
```

Change to:
```jinja2
<td{% if st.get('significant') %} class="stat-significant"{% endif %}>{% if cat in ['AVG'] %}{{ "%.3f"|format(st.get('actual', 0)) }}{% elif cat in ['ERA', 'WHIP'] %}{{ "%.2f"|format(st.get('actual', 0)) }}{% else %}{{ st.get('actual', 0) }}{% endif %}</td>
```

- [ ] **Step 2: Update buy-low FA actual cells (around line 345-354)**

Apply the same pattern to the free agent buy-low table's actual column.

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_web/ -v
```

Expected: All PASS.

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat: show significance checkmarks on buy-low stat cells"
```

---

### Task 7: Add significance to player browse table

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Modify: `src/fantasy_baseball/web/templates/season/players.html`

The browse endpoint shows projected stats but significance is based on actual season-to-date PA/BF from game logs. We need to query game log totals and compute significance per player.

- [ ] **Step 1: Add game log query to browse endpoint**

In `src/fantasy_baseball/web/season_routes.py`, in the `api_player_browse` function, add a game log aggregation query after the `leverage = _get_leverage()` line:

```python
            # Actual PA/BF from game logs for significance indicators
            actual_pa: dict[str, float] = {}
            for r in conn.execute(
                "SELECT name, SUM(pa) as pa FROM game_logs "
                "WHERE season = ? AND player_type = 'hitter' GROUP BY name",
                (season,),
            ).fetchall():
                actual_pa[normalize_name(r["name"])] = r["pa"] or 0

            actual_pitcher_bf: dict[str, float] = {}
            for r in conn.execute(
                "SELECT name, SUM(ip) as ip, SUM(bb) as bb, SUM(h_allowed) as h_allowed "
                "FROM game_logs WHERE season = ? AND player_type = 'pitcher' GROUP BY name",
                (season,),
            ).fetchall():
                ip = r["ip"] or 0
                actual_pitcher_bf[normalize_name(r["name"])] = ip * 3 + (r["h_allowed"] or 0) + (r["bb"] or 0)
```

- [ ] **Step 2: Add significance dict to each player result**

In the same function, after building the `result` dict (after the `if ptype == "hitter": result.update(...)` block), add:

```python
                if ptype == "hitter":
                    pa = actual_pa.get(norm, 0)
                    actual_stats_obj = HitterStats(pa=pa)
                    result["significant"] = actual_stats_obj.significant_dict()
                else:
                    bf = actual_pitcher_bf.get(norm, 0)
                    # Reverse-engineer ip/bb/h_allowed from BF for the stats object
                    # We only need the total BF, so distribute arbitrarily
                    actual_stats_obj = PitcherStats(ip=bf / 3, bb=0, h_allowed=0)
                    result["significant"] = actual_stats_obj.significant_dict()
```

Wait — that reverse-engineering is hacky. Better: just store the actual ip/bb/h_allowed from the query and build a proper PitcherStats:

Replace the `actual_pitcher_bf` query with storing components:

```python
            actual_pitcher_logs: dict[str, dict] = {}
            for r in conn.execute(
                "SELECT name, SUM(ip) as ip, SUM(bb) as bb, SUM(h_allowed) as h_allowed "
                "FROM game_logs WHERE season = ? AND player_type = 'pitcher' GROUP BY name",
                (season,),
            ).fetchall():
                actual_pitcher_logs[normalize_name(r["name"])] = {
                    "ip": r["ip"] or 0, "bb": r["bb"] or 0, "h_allowed": r["h_allowed"] or 0,
                }
```

Then build significance:

```python
                if ptype == "hitter":
                    actual_obj = HitterStats(pa=actual_pa.get(norm, 0))
                    result["significant"] = actual_obj.significant_dict()
                else:
                    logs = actual_pitcher_logs.get(norm, {})
                    actual_obj = PitcherStats(
                        ip=logs.get("ip", 0),
                        bb=logs.get("bb", 0),
                        h_allowed=logs.get("h_allowed", 0),
                    )
                    result["significant"] = actual_obj.significant_dict()
```

- [ ] **Step 3: Update players.html to render checkmarks**

In `src/fantasy_baseball/web/templates/season/players.html`, find the stat cell rendering (around line 233):

```javascript
cats.forEach(c => { cells += '<td>' + fmt(p[c], c) + '</td>'; });
```

Change to:

```javascript
cats.forEach(c => {
    const sig = p.significant && p.significant[c] ? ' stat-significant' : '';
    cells += '<td class="' + sig.trim() + '">' + fmt(p[c], c) + '</td>';
});
```

- [ ] **Step 4: Run all tests**

```bash
pytest tests/test_web/ -v
pytest tests/test_models/test_player.py -v
pytest tests/test_analysis/test_pace.py -v
```

Expected: All PASS.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py src/fantasy_baseball/web/templates/season/players.html
git commit -m "feat: show significance checkmarks on player browse table"
```
