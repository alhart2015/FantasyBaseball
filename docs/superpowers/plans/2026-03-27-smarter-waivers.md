# Smarter Waiver Pickups Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace 1-for-1 same-type waiver swaps with whole-roster re-optimization that considers cross-position moves, shows positions on cards, and displays before/after optimal lineups.

**Architecture:** New `_compute_team_wsgp()` helper runs both optimizers and returns total assigned wSGP + lineup assignments. `scan_waivers()` Phase 2 rewritten to try all FA × roster player combinations, re-optimize each, and keep the best. Template adds positions to cards and expandable before/after lineup comparison.

**Tech Stack:** Python, existing Hungarian optimizer, Jinja2/Flask, vanilla JS

**Spec:** `docs/superpowers/specs/2026-03-27-smarter-waivers-design.md`

---

## File Structure

- **Modify:** `src/fantasy_baseball/lineup/waivers.py` — rewrite Phase 2 swap logic, add helpers
- **Modify:** `tests/test_lineup/test_waivers.py` — update/add tests for new swap logic
- **Modify:** `src/fantasy_baseball/web/templates/season/waivers_trades.html` — positions on cards, expandable before/after lineup

---

### Task 1: Add `_compute_team_wsgp()` helper

**Files:**
- Modify: `src/fantasy_baseball/lineup/waivers.py`
- Modify: `tests/test_lineup/test_waivers.py`

This is the core inner-loop function. Given a roster, it runs both optimizers and returns the total wSGP of only assigned (starting) players.

- [ ] **Step 1: Write failing test for team wSGP computation**

Add to `tests/test_lineup/test_waivers.py`:

```python
from fantasy_baseball.lineup.waivers import _compute_team_wsgp

ROSTER_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1, "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2}


class TestComputeTeamWsgp:
    def test_returns_total_and_lineups(self):
        """_compute_team_wsgp returns total wSGP of assigned starters plus lineup dicts."""
        roster = [
            _make_player("Hitter A", "hitter", r=80, hr=25, rbi=75, sb=10, avg=.270, ab=500, h=135,
                         positions=["1B"]),
            _make_player("Hitter B", "hitter", r=60, hr=15, rbi=55, sb=5, avg=.260, ab=450, h=117,
                         positions=["OF"]),
            _make_player("Pitcher A", "pitcher", w=12, k=180, sv=0, ip=180, er=60, bb=50, h_allowed=150,
                         era=3.00, whip=1.11, positions=["SP"]),
        ]
        result = _compute_team_wsgp(roster, EQUAL_LEVERAGE, ROSTER_SLOTS)
        assert "total_wsgp" in result
        assert "hitter_lineup" in result
        assert "pitcher_starters" in result
        assert result["total_wsgp"] > 0
        # Only assigned players count — with 1 hitter per slot and 14 hitter slots,
        # only 2 hitters can be assigned
        assert isinstance(result["hitter_lineup"], dict)
        assert isinstance(result["pitcher_starters"], list)

    def test_unassigned_players_dont_count(self):
        """Players who can't be assigned to any slot contribute 0 to total."""
        # Two catchers but only 1 C slot, 0 UTIL for this test
        roster = [
            _make_player("Catcher 1", "hitter", r=50, hr=15, rbi=50, sb=2, avg=.260, ab=400, h=104,
                         positions=["C"]),
            _make_player("Catcher 2", "hitter", r=40, hr=10, rbi=35, sb=1, avg=.240, ab=350, h=84,
                         positions=["C"]),  # Only C eligible, can't fill other slots
        ]
        slots = {"C": 1, "P": 0, "BN": 1, "IL": 0}
        result = _compute_team_wsgp(roster, EQUAL_LEVERAGE, slots)
        # Only 1 catcher assigned, the other benched
        assert len(result["hitter_lineup"]) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lineup/test_waivers.py::TestComputeTeamWsgp -v`
Expected: FAIL — `_compute_team_wsgp` not found

- [ ] **Step 3: Implement `_compute_team_wsgp`**

Add to `src/fantasy_baseball/lineup/waivers.py`:

```python
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup


def _compute_team_wsgp(
    roster: list[pd.Series],
    leverage: dict[str, float],
    roster_slots: dict[str, int],
    denoms: dict[str, float] | None = None,
) -> dict:
    """Run both optimizers and return total wSGP of assigned starters.

    Returns:
        {"total_wsgp": float, "hitter_lineup": dict, "pitcher_starters": list,
         "player_wsgp": dict}  # name -> wsgp lookup
    """
    if denoms is None:
        denoms = get_sgp_denominators()

    hitters = [p for p in roster if p.get("player_type") != "pitcher"]
    pitchers = [p for p in roster if p.get("player_type") == "pitcher"]

    # Pre-compute wSGP for all players
    player_wsgp = {}
    for p in roster:
        player_wsgp[p["name"]] = calculate_weighted_sgp(p, leverage, denoms=denoms)

    # Optimize hitters
    hitter_lineup = optimize_hitter_lineup(hitters, leverage, roster_slots)

    # Optimize pitchers
    p_slots = roster_slots.get("P", 9)
    pitcher_starters, _ = optimize_pitcher_lineup(pitchers, leverage, slots=p_slots)

    # Sum wSGP of assigned players only
    total = 0.0
    for name in hitter_lineup.values():
        total += player_wsgp.get(name, 0.0)
    for ps in pitcher_starters:
        total += player_wsgp.get(ps["name"], 0.0)

    return {
        "total_wsgp": total,
        "hitter_lineup": hitter_lineup,
        "pitcher_starters": pitcher_starters,
        "player_wsgp": player_wsgp,
    }
```

Add the import at the top of `waivers.py`:

```python
from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_lineup/test_waivers.py::TestComputeTeamWsgp -v`
Expected: PASS

- [ ] **Step 5: Run all existing waiver tests to verify no regressions**

Run: `python -m pytest tests/test_lineup/test_waivers.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/lineup/waivers.py tests/test_lineup/test_waivers.py
git commit -m "feat: add _compute_team_wsgp helper for whole-roster evaluation"
```

---

### Task 2: Add `_build_lineup_summary()` helper

**Files:**
- Modify: `src/fantasy_baseball/lineup/waivers.py`
- Modify: `tests/test_lineup/test_waivers.py`

This builds the before/after lineup lists for the expanded card.

- [ ] **Step 1: Write failing test for lineup summary**

```python
from fantasy_baseball.lineup.waivers import _build_lineup_summary


class TestBuildLineupSummary:
    def test_basic_lineup(self):
        """Builds lineup list from optimizer output."""
        hitter_lineup = {"C": "Player A", "1B": "Player B", "OF": "Player C"}
        pitcher_starters = [{"name": "Pitcher X", "wsgp": 2.0}]
        player_wsgp = {"Player A": 1.5, "Player B": 2.0, "Player C": 1.0, "Pitcher X": 2.0}
        all_players = ["Player A", "Player B", "Player C", "Player D", "Pitcher X", "Pitcher Y"]

        result = _build_lineup_summary(hitter_lineup, pitcher_starters, player_wsgp, all_players)
        assert len(result) > 0
        # Assigned players have slots
        player_a = next(e for e in result if e["name"] == "Player A")
        assert player_a["slot"] == "C"
        assert player_a["wsgp"] == 1.5

    def test_strips_slot_suffixes(self):
        """OF_2, UTIL_3 etc. are stripped to base slot name for display."""
        hitter_lineup = {"OF": "Player A", "OF_2": "Player B"}
        pitcher_starters = []
        player_wsgp = {"Player A": 1.0, "Player B": 0.8}

        result = _build_lineup_summary(hitter_lineup, pitcher_starters, player_wsgp, ["Player A", "Player B"])
        slots = [e["slot"] for e in result if e["name"] in ("Player A", "Player B")]
        assert all(s == "OF" for s in slots)

    def test_bench_players_flagged(self):
        """Players not in any optimizer output are marked as bench."""
        hitter_lineup = {"C": "Starter"}
        pitcher_starters = []
        player_wsgp = {"Starter": 2.0, "Benched": 0.5}

        result = _build_lineup_summary(hitter_lineup, pitcher_starters, player_wsgp, ["Starter", "Benched"])
        benched = next(e for e in result if e["name"] == "Benched")
        assert benched["slot"] == "BN"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_lineup/test_waivers.py::TestBuildLineupSummary -v`
Expected: FAIL

- [ ] **Step 3: Implement `_build_lineup_summary`**

Add to `waivers.py`:

```python
def _build_lineup_summary(
    hitter_lineup: dict[str, str],
    pitcher_starters: list[dict],
    player_wsgp: dict[str, float],
    all_player_names: list[str],
) -> list[dict]:
    """Build a lineup summary list for display.

    Returns list of {"name", "slot", "wsgp"} dicts.
    Hitter slots have _N suffixes stripped. Unassigned players get slot="BN".
    """
    summary = []
    assigned_names = set()

    # Hitters from optimizer
    for slot_key, name in hitter_lineup.items():
        display_slot = slot_key.split("_")[0]  # "OF_2" -> "OF"
        summary.append({
            "name": name,
            "slot": display_slot,
            "wsgp": round(player_wsgp.get(name, 0.0), 2),
        })
        assigned_names.add(name)

    # Pitcher starters
    for ps in pitcher_starters:
        name = ps["name"]
        summary.append({
            "name": name,
            "slot": "P",
            "wsgp": round(ps.get("wsgp", player_wsgp.get(name, 0.0)), 2),
        })
        assigned_names.add(name)

    # Bench: everyone not assigned
    for name in all_player_names:
        if name not in assigned_names:
            summary.append({
                "name": name,
                "slot": "BN",
                "wsgp": round(player_wsgp.get(name, 0.0), 2),
            })

    return summary
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest tests/test_lineup/test_waivers.py::TestBuildLineupSummary -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/waivers.py tests/test_lineup/test_waivers.py
git commit -m "feat: add _build_lineup_summary helper for before/after display"
```

---

### Task 3: Rewrite `scan_waivers()` Phase 2 with re-optimization

**Files:**
- Modify: `src/fantasy_baseball/lineup/waivers.py`
- Modify: `tests/test_lineup/test_waivers.py`

This is the core algorithm change. Replace the same-type-only swap logic with whole-roster re-optimization.

- [ ] **Step 1: Write failing test for cross-position swap**

```python
class TestScanWaiversReoptimize:
    def test_cross_position_swap_recommended(self):
        """Can pick up a 3B and drop an OF if roster reshuffles to cover."""
        roster = [
            _make_player("Good SS", "hitter", r=80, hr=20, rbi=70, sb=15, avg=.280, ab=500, h=140,
                         positions=["SS", "3B", "OF"]),
            _make_player("Weak OF", "hitter", r=40, hr=5, rbi=25, sb=2, avg=.230, ab=300, h=69,
                         positions=["OF"]),
            _make_player("Decent 1B", "hitter", r=60, hr=15, rbi=55, sb=5, avg=.260, ab=450, h=117,
                         positions=["1B"]),
        ]
        free_agents = [
            _make_player("Great 3B", "hitter", r=90, hr=30, rbi=85, sb=10, avg=.275, ab=540, h=148,
                         positions=["3B"]),
        ]
        # 3 hitter slots + roster of 3 (minus 1 drop + 1 add = still 3)
        slots = {"SS": 1, "1B": 1, "OF": 1, "P": 0, "BN": 0, "IL": 0}

        result = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                              roster_slots=slots, max_results=10)
        # Should recommend: ADD Great 3B / DROP Weak OF
        # Good SS has OF eligibility so can cover the OF slot after reshuffle
        assert len(result) >= 1
        assert result[0]["add"] == "Great 3B"
        assert result[0]["drop"] == "Weak OF"
        assert result[0]["sgp_gain"] > 0
        assert "add_positions" in result[0]
        assert "drop_positions" in result[0]

    def test_position_infeasible_swap_skipped(self):
        """Can't drop the only C if no one else can play C."""
        roster = [
            _make_player("Only C", "hitter", r=50, hr=12, rbi=45, sb=3, avg=.250, ab=400, h=100,
                         positions=["C"]),
            _make_player("OF Guy", "hitter", r=60, hr=15, rbi=55, sb=5, avg=.260, ab=450, h=117,
                         positions=["OF"]),
        ]
        free_agents = [
            _make_player("Great OF", "hitter", r=90, hr=30, rbi=85, sb=15, avg=.280, ab=540, h=151,
                         positions=["OF"]),
        ]
        # 2 slots for 2 players — dropping OF Guy + adding Great OF is feasible
        slots = {"C": 1, "OF": 1, "P": 0, "BN": 0, "IL": 0}

        result = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                              roster_slots=slots, max_results=10)
        # Should recommend: ADD Great OF / DROP OF Guy (same position, valid)
        # Should NOT recommend dropping "Only C" — no one can fill C
        assert len(result) >= 1
        assert result[0]["add"] == "Great OF"
        assert result[0]["drop"] == "OF Guy"
        for r in result:
            assert r["drop"] != "Only C"

    def test_includes_lineup_before_after(self):
        """Recommendations include before/after lineup data for expanded card."""
        roster = [
            _make_player("Starter", "hitter", r=60, hr=15, rbi=55, sb=5, avg=.260, ab=450, h=117,
                         positions=["OF"]),
        ]
        free_agents = [
            _make_player("Better", "hitter", r=80, hr=25, rbi=75, sb=10, avg=.270, ab=500, h=135,
                         positions=["OF"]),
        ]
        slots = {"OF": 1, "UTIL": 1, "P": 0, "BN": 0, "IL": 0}

        result = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                              roster_slots=slots, max_results=10)
        assert len(result) >= 1
        assert "lineup_before" in result[0]
        assert "lineup_after" in result[0]
        # Before has the dropped player, after has the added player
        before_names = [e["name"] for e in result[0]["lineup_before"]]
        after_names = [e["name"] for e in result[0]["lineup_after"]]
        assert "Starter" in before_names
        assert "Better" in after_names

    def test_best_drop_per_fa(self):
        """For each FA, only the best drop candidate is kept."""
        roster = [
            _make_player("OK OF", "hitter", r=60, hr=15, rbi=55, sb=5, avg=.260, ab=450, h=117,
                         positions=["OF"]),
            _make_player("Bad OF", "hitter", r=30, hr=5, rbi=20, sb=1, avg=.220, ab=300, h=66,
                         positions=["OF"]),
        ]
        free_agents = [
            _make_player("Good OF", "hitter", r=80, hr=25, rbi=75, sb=10, avg=.270, ab=500, h=135,
                         positions=["OF"]),
        ]
        slots = {"OF": 2, "UTIL": 1, "P": 0, "BN": 0, "IL": 0}

        result = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                              roster_slots=slots, max_results=10)
        # Only one recommendation for "Good OF", dropping the worse player
        good_of_recs = [r for r in result if r["add"] == "Good OF"]
        assert len(good_of_recs) == 1
        assert good_of_recs[0]["drop"] == "Bad OF"

    def test_wsgp_floor_prunes_bad_fas(self):
        """FAs below the wSGP floor are skipped."""
        roster = [
            _make_player("Decent", "hitter", r=70, hr=20, rbi=65, sb=8, avg=.265, ab=480, h=127,
                         positions=["1B"]),
        ]
        free_agents = [
            _make_player("Terrible", "hitter", r=10, hr=1, rbi=5, sb=0, avg=.180, ab=100, h=18,
                         positions=["1B"]),
        ]
        slots = {"1B": 1, "UTIL": 1, "P": 0, "BN": 0, "IL": 0}

        result = scan_waivers(roster, free_agents, EQUAL_LEVERAGE,
                              roster_slots=slots, max_results=10)
        # Terrible FA should not produce a recommendation
        assert len(result) == 0
```

- [ ] **Step 2: Run tests to verify new ones fail (existing should still pass)**

Run: `python -m pytest tests/test_lineup/test_waivers.py -v`
Expected: new tests FAIL, existing tests still PASS (for now — some existing tests may need updating once we rewrite scan_waivers)

- [ ] **Step 3: Rewrite `scan_waivers()` Phase 2**

Replace the Phase 2 swap logic (lines 242-275 of `waivers.py`) with the re-optimization approach. Keep Phase 1 (pure adds) unchanged.

The new Phase 2:

```python
    # Phase 2: Re-optimization swaps
    if not roster or not roster_slots:
        recommendations.sort(key=lambda x: x["sgp_gain"], reverse=True)
        return recommendations[:max_results]

    denoms = get_sgp_denominators()

    # Compute baseline optimal lineup
    baseline = _compute_team_wsgp(roster, leverage, roster_slots, denoms=denoms)
    baseline_wsgp = baseline["total_wsgp"]
    baseline_summary = _build_lineup_summary(
        baseline["hitter_lineup"], baseline["pitcher_starters"],
        baseline["player_wsgp"], [p["name"] for p in roster],
    )

    # Pre-compute wSGP for all FAs
    fa_wsgp = {}
    for fa in free_agents:
        if fa["name"] not in recommended_adds:
            fa_wsgp[fa["name"]] = calculate_weighted_sgp(fa, leverage, denoms=denoms)

    # Compute wSGP floor: 3rd-lowest wSGP among active-slot players
    active_wsgps = sorted([
        baseline["player_wsgp"].get(name, 0.0)
        for name in list(baseline["hitter_lineup"].values())
        + [ps["name"] for ps in baseline["pitcher_starters"]]
    ])
    wsgp_floor = active_wsgps[2] if len(active_wsgps) > 2 else 0.0

    p_slots = roster_slots.get("P", 9)

    for fa in free_agents:
        if fa["name"] in recommended_adds:
            continue
        if fa_wsgp.get(fa["name"], 0.0) < wsgp_floor:
            continue  # below floor, can't improve roster

        fa_type = fa.get("player_type", "hitter")
        best_for_fa = None

        for drop_player in roster:
            drop_name = drop_player["name"]
            drop_type = drop_player.get("player_type", "hitter")

            # Build hypothetical roster
            new_roster = [p for p in roster if p["name"] != drop_name] + [fa]

            # Feasibility check
            new_hitters = [p for p in new_roster if p.get("player_type") != "pitcher"]
            new_pitchers = [p for p in new_roster if p.get("player_type") == "pitcher"]

            # Hitter feasibility (always check if hitter count changed or positions changed)
            if drop_type == "hitter" or fa_type == "hitter":
                hitter_positions = [list(p.get("positions", [])) for p in new_hitters]
                if not can_cover_slots(hitter_positions, roster_slots):
                    continue

            # Pitcher feasibility
            if drop_type == "pitcher" or fa_type == "pitcher":
                if len(new_pitchers) < p_slots:
                    continue

            # Re-optimize
            new_result = _compute_team_wsgp(new_roster, leverage, roster_slots, denoms=denoms)
            gain = round(new_result["total_wsgp"] - baseline_wsgp, 2)

            if gain > 0 and (best_for_fa is None or gain > best_for_fa["sgp_gain"]):
                # Build after summary with diff annotations
                after_summary = _build_lineup_summary(
                    new_result["hitter_lineup"], new_result["pitcher_starters"],
                    new_result["player_wsgp"], [p["name"] for p in new_roster],
                )

                # Annotate before/after
                before_annotated = []
                for entry in baseline_summary:
                    e = dict(entry)
                    if e["name"] == drop_name:
                        e["is_dropped"] = True
                    before_annotated.append(e)

                after_annotated = []
                before_slots = {e["name"]: e["slot"] for e in baseline_summary}
                for entry in after_summary:
                    e = dict(entry)
                    if e["name"] == fa["name"]:
                        e["is_added"] = True
                    elif e["name"] in before_slots and before_slots[e["name"]] != e["slot"]:
                        e["moved_from"] = before_slots[e["name"]]
                    after_annotated.append(e)

                # Get per-category deltas
                cat_result = evaluate_pickup(fa, drop_player, leverage)

                best_for_fa = {
                    "add": fa["name"],
                    "add_positions": list(fa.get("positions", [])),
                    "drop": drop_name,
                    "drop_positions": list(drop_player.get("positions", [])),
                    "sgp_gain": gain,
                    "categories": cat_result["categories"],
                    "lineup_before": before_annotated,
                    "lineup_after": after_annotated,
                }

        if best_for_fa:
            recommendations.append(best_for_fa)

    recommendations.sort(key=lambda x: x["sgp_gain"], reverse=True)
    return recommendations[:max_results]
```

- [ ] **Step 4: Update existing `TestScanWaivers` tests (REQUIRED)**

The existing tests call `scan_waivers()` without `roster_slots`. The new Phase 2 returns early when `roster_slots` is None, so all existing swap tests will break. You MUST:

1. Add `roster_slots=` to every existing `scan_waivers()` call in `TestScanWaivers` that expects swap recommendations. Use a slots dict that matches the test's roster positions.
2. The output format now includes `add_positions`, `drop_positions`, `lineup_before`, `lineup_after`. Existing assertions on `sgp_gain` may need adjustment since it's now team-level gain (optimizer comparison), not individual player delta.
3. The helper `_make_player` and `EQUAL_LEVERAGE` are already defined at module scope — new test classes can use them directly.

Read each existing test, understand what it asserts, and update the call + assertions to match the new behavior.

- [ ] **Step 5: Run all waiver tests**

Run: `python -m pytest tests/test_lineup/test_waivers.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/lineup/waivers.py tests/test_lineup/test_waivers.py
git commit -m "feat: rewrite scan_waivers Phase 2 with whole-roster re-optimization"
```

---

### Task 4: Add positions and expandable lineup to waiver template

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`

- [ ] **Step 1: Add CSS for expandable waiver cards and lineup display**

Add to the existing `<style>` block:

```css
.waiver-detail { display: none; margin-top: 10px; }
.waiver-detail.open { display: block; }
.lineup-compare { display: flex; gap: 16px; flex-wrap: wrap; }
.lineup-col { flex: 1; min-width: 200px; }
.lineup-col h4 { font-size: 12px; color: var(--text-secondary); text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.lineup-table { width: 100%; font-size: 12px; border-collapse: collapse; }
.lineup-table th { text-align: left; font-size: 11px; color: var(--text-secondary); padding: 4px 6px; border-bottom: 1px solid var(--panel-border); }
.lineup-table td { padding: 4px 6px; }
.lineup-dropped { background: rgba(239, 68, 68, 0.1); color: #fca5a5; text-decoration: line-through; }
.lineup-added { background: rgba(34, 197, 94, 0.1); color: #86efac; font-weight: 600; }
.lineup-moved { color: var(--accent); }
.lineup-bench td { opacity: 0.5; }
```

- [ ] **Step 2: Update waiver card markup**

Replace the waiver card content in the template. The card should:
- Show positions after player names
- Be clickable to expand
- Show before/after lineup when expanded

Replace the waiver `{% for w in waivers %}` block with:

```html
{% for w in waivers %}
<div class="card" onclick="toggleWaiverDetail(this)">
    <div class="card-header">
        <div>
            <span style="font-weight: 600; font-size: 14px;">{{ w.add }}</span>
            {% if w.add_positions %}
            <span style="color: var(--text-secondary); font-size: 12px; margin-left: 4px;">({{ w.add_positions | join(", ") }})</span>
            {% endif %}
        </div>
        <div style="color: var(--success); font-weight: bold; font-size: 14px;">
            +{{ "%.2f"|format(w.sgp_gain) }} wSGP
        </div>
    </div>

    <div style="font-size: 12px; color: var(--text-secondary); margin-top: 6px;">
        <span style="margin-right: 4px;">ADD</span>
        <strong style="color: var(--success);">{{ w.add }}</strong>
        {% if w.add_positions %}<span style="font-size: 11px;"> ({{ w.add_positions | join(", ") }})</span>{% endif %}
        <span style="margin: 0 6px; color: var(--text-secondary);">/</span>
        <span style="margin-right: 4px;">DROP</span>
        <strong style="color: var(--danger);">{{ w.drop }}</strong>
        {% if w.drop_positions %}<span style="font-size: 11px;"> ({{ w.drop_positions | join(", ") }})</span>{% endif %}
    </div>

    {% if w.categories %}
    <div class="cat-impact">
        {% for cat, delta in w.categories.items() %}
        {% if delta > 0 %}
        <span class="cat-gain">{{ cat }} +{{ "%.2f"|format(delta) }}</span>
        {% elif delta < 0 %}
        <span class="cat-loss">{{ cat }} {{ "%.2f"|format(delta) }}</span>
        {% endif %}
        {% endfor %}
    </div>
    {% endif %}

    {% if w.lineup_before and w.lineup_after %}
    <div class="waiver-detail">
        <div class="lineup-compare">
            <div class="lineup-col">
                <h4>Before</h4>
                <table class="lineup-table">
                    <thead><tr><th>Slot</th><th>Player</th><th>wSGP</th></tr></thead>
                    <tbody>
                    {% for e in w.lineup_before %}
                    <tr class="{% if e.get('is_dropped') %}lineup-dropped{% endif %}{% if e.slot == 'BN' %} lineup-bench{% endif %}">
                        <td>{{ e.slot }}</td>
                        <td>{{ e.name }}</td>
                        <td>{{ "%.2f"|format(e.wsgp) }}</td>
                    </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
            <div class="lineup-col">
                <h4>After</h4>
                <table class="lineup-table">
                    <thead><tr><th>Slot</th><th>Player</th><th>wSGP</th></tr></thead>
                    <tbody>
                    {% for e in w.lineup_after %}
                    <tr class="{% if e.get('is_added') %}lineup-added{% elif e.get('moved_from') %}lineup-moved{% endif %}{% if e.slot == 'BN' %} lineup-bench{% endif %}">
                        <td>{{ e.slot }}</td>
                        <td>
                            {{ e.name }}
                            {% if e.get('moved_from') %}
                            <span style="font-size: 10px; color: var(--text-secondary);">&larr; was {{ e.moved_from }}</span>
                            {% endif %}
                        </td>
                        <td>{{ "%.2f"|format(e.wsgp) }}</td>
                    </tr>
                    {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    {% endif %}
</div>
{% endfor %}
```

- [ ] **Step 3: Add JS for waiver card expand**

Add to the `<script>` block:

```javascript
function toggleWaiverDetail(card) {
    var detail = card.querySelector('.waiver-detail');
    if (detail) detail.classList.toggle('open');
}
```

- [ ] **Step 4: Verify page renders**

Run: `python -c "from fantasy_baseball.web.season_app import create_app; app = create_app(); c = app.test_client(); r = c.get('/waivers-trades'); print(r.status_code)"`
Expected: 200

- [ ] **Step 5: Run web tests**

Run: `python -m pytest tests/test_web/test_season_routes.py -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/waivers_trades.html
git commit -m "feat: add positions and expandable before/after lineup to waiver cards"
```

---

### Task 5: Run full test suite and verify

- [ ] **Step 1: Run the complete test suite**

Run: `python -m pytest -v`
Expected: all tests pass, no regressions

- [ ] **Step 2: Verify waiver card structure**

Run: `python -c "from fantasy_baseball.web.season_app import create_app; app = create_app(); c = app.test_client(); r = c.get('/waivers-trades'); print('waiver-detail' in r.text, 'lineup-compare' in r.text)"`
Expected: `True True` (or `True False` if no cached data — that's fine, the template structure is there)

- [ ] **Step 3: Final commit if any cleanup needed**
