# Player Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a player comparison section to the Players page — select 2 players (one yours, one not), see side-by-side ROS stats and projected standings impact of swapping them.

**Architecture:** Checkboxes in the existing browse table feed a client-side stat comparison (from already-loaded data) plus a server-side standings computation (new `/api/players/compare` endpoint that uses `project_team_stats` + `score_roto` to show before/after roto standings).

**Tech Stack:** Jinja2 template (inline JS), Flask route, existing `scoring.py` functions.

---

### Task 0: Create branch

**Files:** None

- [ ] **Step 1: Create and switch to the feature branch**

```bash
git checkout -b player-comparison
```

- [ ] **Step 2: Commit the spec and plan**

```bash
git add docs/superpowers/specs/2026-04-07-player-comparison-design.md docs/superpowers/plans/2026-04-07-player-comparison.md
git commit -m "docs: add player comparison spec and plan"
```

---

### Task 1: Backend — `compute_comparison_standings` function

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py` (add function after `compute_trade_standings_impact` at line ~610)
- Test: `tests/test_web/test_season_data.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web/test_season_data.py`:

```python
class TestComputeComparisonStandings:
    def test_swap_changes_user_team_stats(self):
        """Swapping a hitter should change the user's projected stats and roto points."""
        from fantasy_baseball.web.season_data import compute_comparison_standings

        # Two-team league for simplicity
        projected_standings = [
            {"name": "My Team", "team_key": "", "rank": 0, "stats": {
                "R": 700, "HR": 200, "RBI": 700, "SB": 100, "AVG": 0.260,
                "W": 80, "K": 1200, "SV": 50, "ERA": 3.50, "WHIP": 1.20,
            }},
            {"name": "Other Team", "team_key": "", "rank": 0, "stats": {
                "R": 680, "HR": 190, "RBI": 680, "SB": 110, "AVG": 0.255,
                "W": 85, "K": 1100, "SV": 40, "ERA": 3.80, "WHIP": 1.25,
            }},
        ]

        # Roster: flat dicts with component stats (as stored in roster cache)
        roster = [
            {"name": "Willy Adames", "player_type": "hitter",
             "r": 80, "hr": 25, "rbi": 81, "sb": 11, "h": 133, "ab": 567,
             "avg": 0.235},
            {"name": "Other Hitter", "player_type": "hitter",
             "r": 90, "hr": 30, "rbi": 95, "sb": 5, "h": 150, "ab": 550,
             "avg": 0.273},
            {"name": "My Pitcher", "player_type": "pitcher",
             "w": 12, "k": 180, "sv": 0, "ip": 180, "er": 60, "bb": 50,
             "h_allowed": 150, "era": 3.00, "whip": 1.11},
        ]

        # Player to add (also a flat dict)
        other_player = {
            "name": "Ezequiel Tovar", "player_type": "hitter",
            "r": 73, "hr": 20, "rbi": 74, "sb": 8, "h": 135, "ab": 513,
            "avg": 0.263,
        }

        result = compute_comparison_standings(
            roster_player_name="Willy Adames",
            other_player=other_player,
            user_roster=roster,
            projected_standings=projected_standings,
            user_team_name="My Team",
        )

        # Before and after should both have roto scores
        assert "before" in result
        assert "after" in result
        assert "categories" in result

        before_total = result["before"]["roto"]["My Team"]["total"]
        after_total = result["after"]["roto"]["My Team"]["total"]

        # Stats should differ — the swap changes the user's team composition
        assert result["before"]["stats"]["My Team"] != result["after"]["stats"]["My Team"]

        # Other team should be unchanged
        assert result["before"]["stats"]["Other Team"] == result["after"]["stats"]["Other Team"]

    def test_swap_not_found_returns_error(self):
        """If roster_player_name doesn't match anyone in user_roster, return error."""
        from fantasy_baseball.web.season_data import compute_comparison_standings

        result = compute_comparison_standings(
            roster_player_name="Nobody",
            other_player={"name": "X", "player_type": "hitter",
                          "r": 0, "hr": 0, "rbi": 0, "sb": 0, "h": 0, "ab": 0},
            user_roster=[{"name": "A", "player_type": "hitter",
                          "r": 50, "hr": 10, "rbi": 40, "sb": 5, "h": 80, "ab": 300}],
            projected_standings=[{"name": "My Team", "team_key": "", "rank": 0,
                                  "stats": {"R": 700, "HR": 200, "RBI": 700, "SB": 100,
                                            "AVG": 0.260, "W": 80, "K": 1200, "SV": 50,
                                            "ERA": 3.50, "WHIP": 1.20}}],
            user_team_name="My Team",
        )
        assert "error" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_data.py::TestComputeComparisonStandings -v`
Expected: FAIL with `ImportError` (function doesn't exist yet)

- [ ] **Step 3: Write the implementation**

Add to `src/fantasy_baseball/web/season_data.py` after `compute_trade_standings_impact` (around line 610):

```python
def compute_comparison_standings(
    roster_player_name: str,
    other_player: dict,
    user_roster: list[dict],
    projected_standings: list[dict],
    user_team_name: str,
) -> dict:
    """Compute before/after roto standings for a player swap.

    Replaces ``roster_player_name`` on the user's roster with
    ``other_player`` and re-projects team stats.  Rate stats (AVG, ERA,
    WHIP) are recomputed from components, not added/subtracted.

    Returns dict with before/after stats and roto, or {"error": ...}.
    """
    from fantasy_baseball.scoring import project_team_stats, score_roto

    # Find the roster player to remove
    drop_idx = None
    for i, p in enumerate(user_roster):
        if p["name"] == roster_player_name:
            drop_idx = i
            break

    if drop_idx is None:
        return {"error": f"Player '{roster_player_name}' not found on roster"}

    # Before: project from current roster
    all_stats_before = {t["name"]: dict(t["stats"]) for t in projected_standings}

    # After: rebuild user roster with the swap, re-project
    roster_after = [p for i, p in enumerate(user_roster) if i != drop_idx]
    roster_after.append(other_player)
    all_stats_after = {t["name"]: dict(t["stats"]) for t in projected_standings}
    all_stats_after[user_team_name] = project_team_stats(roster_after)

    roto_before = score_roto(all_stats_before)
    roto_after = score_roto(all_stats_after)

    return {
        "before": {
            "stats": all_stats_before,
            "roto": roto_before,
        },
        "after": {
            "stats": all_stats_after,
            "roto": roto_after,
        },
        "categories": ALL_CATEGORIES,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web/test_season_data.py::TestComputeComparisonStandings -v`
Expected: Both tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py tests/test_web/test_season_data.py
git commit -m "feat: add compute_comparison_standings for player swap impact"
```

---

### Task 2: Backend — `/api/players/compare` route

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py` (add route after `api_player_browse` at line ~610)

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web/test_season_routes.py`:

```python
def test_compare_missing_params(client):
    """Missing required params should return 400."""
    resp = client.get("/api/players/compare")
    assert resp.status_code == 400

    resp2 = client.get("/api/players/compare?roster_player=X")
    assert resp2.status_code == 400
```

The computation logic is already unit-tested in Task 1 (`TestComputeComparisonStandings`). This test verifies the route exists and validates params.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_routes.py::test_compare_missing_params -v`
Expected: FAIL with 404 (route doesn't exist yet)

- [ ] **Step 3: Write the route**

Add to `src/fantasy_baseball/web/season_routes.py` after the `api_player_browse` function (around line 610):

```python
    @app.route("/api/players/compare")
    def api_player_compare():
        """Return projected standings before/after swapping a roster player."""
        roster_player = request.args.get("roster_player")
        other_name = request.args.get("other_player")
        other_type = request.args.get("other_type")

        if not roster_player or not other_name or not other_type:
            return jsonify({"error": "roster_player, other_player, and other_type are required"}), 400

        roster_cache = read_cache("roster")
        if not roster_cache:
            return jsonify({"error": "No roster data available"}), 404

        proj_cache = read_cache("projections") or {}
        projected_standings = proj_cache.get("projected_standings")
        if not projected_standings:
            return jsonify({"error": "No projected standings available"}), 404

        # Build the other player dict from query params
        def _float(key, default=0.0):
            try:
                return float(request.args.get(key, default))
            except (TypeError, ValueError):
                return default

        other_player = {
            "name": other_name,
            "player_type": other_type,
            "r": _float("other_r"), "hr": _float("other_hr"),
            "rbi": _float("other_rbi"), "sb": _float("other_sb"),
            "h": _float("other_h"), "ab": _float("other_ab"),
            "w": _float("other_w"), "k": _float("other_k"),
            "sv": _float("other_sv"), "ip": _float("other_ip"),
            "er": _float("other_er"), "bb": _float("other_bb"),
            "h_allowed": _float("other_ha"),
        }

        config = _load_config()

        from fantasy_baseball.web.season_data import compute_comparison_standings
        result = compute_comparison_standings(
            roster_player_name=roster_player,
            other_player=other_player,
            user_roster=roster_cache,
            projected_standings=projected_standings,
            user_team_name=config.team_name,
        )

        if "error" in result:
            return jsonify(result), 404

        return jsonify(result)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web/test_season_routes.py::test_compare_missing_params -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py tests/test_web/test_season_routes.py
git commit -m "feat: add /api/players/compare endpoint for standings impact"
```

---

### Task 3: Frontend — add checkboxes and comparison panel to players.html

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/players.html`

This is the largest task — all client-side. No new test file since this is template/JS work tested manually.

- [ ] **Step 1: Add the comparison panel HTML**

Below the `<div id="empty-msg">` line (line 95) and before `<script>`, add:

```html
<div id="comparison-panel" style="display:none; margin:20px 0; padding:20px; background:var(--panel-bg); border:1px solid var(--panel-border); border-radius:8px;">
    <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:16px;">
        <h3 style="margin:0; font-size:16px; color:var(--accent);">Player Comparison</h3>
        <button id="clear-compare" style="background:none; border:1px solid var(--panel-border); color:var(--text-secondary); padding:4px 10px; border-radius:4px; cursor:pointer; font-size:12px;">✕ Clear</button>
    </div>
    <div id="compare-validation" style="color:var(--text-secondary); font-size:13px;"></div>
    <table class="data-table" id="compare-stats-table" style="display:none; margin-bottom:20px;">
        <thead><tr id="compare-stats-head"></tr></thead>
        <tbody id="compare-stats-body"></tbody>
    </table>
    <div id="standings-section" style="display:none;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:12px;">
            <h4 style="margin:0; font-size:14px;">Projected Standings Impact</h4>
            <div>
                <button class="standings-toggle active" data-mode="roto">Roto Points</button>
                <button class="standings-toggle" data-mode="stats">Stat Totals</button>
            </div>
        </div>
        <div id="standings-subtitle" style="font-size:12px; color:var(--text-secondary); margin-bottom:12px;"></div>
        <div style="display:flex; gap:16px; flex-wrap:wrap;">
            <div style="flex:1; min-width:300px;">
                <div style="font-size:12px; color:var(--text-secondary); margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;">Current Team</div>
                <table class="data-table" id="standings-before">
                    <thead><tr id="standings-before-head"></tr></thead>
                    <tbody id="standings-before-body"></tbody>
                </table>
            </div>
            <div style="flex:1; min-width:300px;">
                <div id="standings-after-label" style="font-size:12px; color:var(--text-secondary); margin-bottom:6px; text-transform:uppercase; letter-spacing:1px;"></div>
                <table class="data-table" id="standings-after">
                    <thead><tr id="standings-after-head"></tr></thead>
                    <tbody id="standings-after-body"></tbody>
                </table>
            </div>
        </div>
    </div>
</div>
```

- [ ] **Step 2: Add CSS for the toggle buttons and comparison styling**

In the `<style>` block at the top of the template, add:

```css
.standings-toggle {
    background: none;
    border: 1px solid var(--panel-border);
    color: var(--text-secondary);
    padding: 4px 10px;
    font-size: 12px;
    cursor: pointer;
    border-radius: 4px;
}
.standings-toggle.active {
    background: var(--accent);
    color: #fff;
    border-color: var(--accent);
}
#compare-stats-table td.better { font-weight: 600; }
#compare-stats-table td.diff-positive { color: var(--success); }
#compare-stats-table td.diff-negative { color: var(--danger); }
```

- [ ] **Step 3: Add checkbox column to the render function**

In the `<script>` block, add selection state at the top (after existing variable declarations around line 103-104):

```javascript
let selectedPlayers = []; // array of player objects, max 2
```

Modify the `cols` array in `render()` to prepend a checkbox column:

```javascript
    const cols = [
        {key: '_select', label: ''},
        {key: 'rank', label: 'Rank'},
        // ... rest unchanged
    ];
```

Modify the header rendering so the checkbox column header is not sortable:

```javascript
    headEl.innerHTML = cols.map(c => {
        if (c.key === '_select') return '<th style="width:30px;"></th>';
        const arrow = sortCol === c.key ? (sortAsc ? ' \u25B2' : ' \u25BC') : '';
        return '<th data-col="' + c.key + '">' + c.label +
               '<span class="sort-arrow">' + arrow + '</span></th>';
    }).join('');
```

Modify the row rendering to prepend a checkbox cell:

```javascript
    bodyEl.innerHTML = filtered.map(p => {
        const isSelected = selectedPlayers.some(s => s.name === p.name);
        let cells = '<td style="text-align:center"><input type="checkbox" class="compare-cb" data-name="' +
            p.name.replace(/"/g, '&quot;') + '"' +
            (isSelected ? ' checked' : '') +
            ' style="accent-color:var(--accent);cursor:pointer"></td>';
        cells += '<td>' + rankText + '</td>' +
        // ... rest unchanged
    }).join('');
```

After the `bodyEl.innerHTML = ...` block, add event listeners:

```javascript
    document.querySelectorAll('.compare-cb').forEach(cb => {
        cb.addEventListener('change', () => {
            const name = cb.dataset.name;
            if (cb.checked) {
                const player = filtered.find(p => p.name === name);
                if (player && selectedPlayers.length >= 2) {
                    selectedPlayers.shift(); // remove oldest
                }
                if (player && !selectedPlayers.some(s => s.name === name)) {
                    selectedPlayers.push(player);
                }
            } else {
                selectedPlayers = selectedPlayers.filter(s => s.name !== name);
            }
            render(); // re-render to update checkboxes
            updateComparison();
        });
    });
```

- [ ] **Step 4: Add the `updateComparison` function**

Add after the `render()` function:

```javascript
const INVERSE_CATS = new Set(['ERA', 'WHIP']);
const compPanel = document.getElementById('comparison-panel');
const compValidation = document.getElementById('compare-validation');
const compStatsTable = document.getElementById('compare-stats-table');
const compStatsHead = document.getElementById('compare-stats-head');
const compStatsBody = document.getElementById('compare-stats-body');
const standingsSection = document.getElementById('standings-section');
let standingsData = null;
let standingsMode = 'roto';

document.getElementById('clear-compare').addEventListener('click', () => {
    selectedPlayers = [];
    standingsData = null;
    compPanel.style.display = 'none';
    render();
});

document.querySelectorAll('.standings-toggle').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.standings-toggle').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        standingsMode = btn.dataset.mode;
        if (standingsData) renderStandings();
    });
});

function updateComparison() {
    if (selectedPlayers.length < 2) {
        compPanel.style.display = selectedPlayers.length > 0 ? '' : 'none';
        compStatsTable.style.display = 'none';
        standingsSection.style.display = 'none';
        compValidation.textContent = selectedPlayers.length === 1
            ? 'Select one more player to compare.'
            : '';
        return;
    }

    const [a, b] = selectedPlayers;
    const rosterPlayer = a.owner === 'roster' ? a : (b.owner === 'roster' ? b : null);
    const otherPlayer = a.owner === 'roster' ? b : (b.owner === 'roster' ? a : null);

    compPanel.style.display = '';

    if (!rosterPlayer || !otherPlayer || otherPlayer.owner === 'roster') {
        compStatsTable.style.display = 'none';
        standingsSection.style.display = 'none';
        compValidation.textContent = 'Select one player from your roster and one who is not on your roster.';
        return;
    }

    compValidation.textContent = '';
    renderStatComparison(rosterPlayer, otherPlayer);
    fetchStandings(rosterPlayer, otherPlayer);
}

function renderStatComparison(roster, other) {
    const isHitter = roster.player_type === 'hitter';
    const cats = isHitter ? HITTING_CATS : PITCHING_CATS;
    const rows = [
        {label: 'ROS Rank', key: 'rank', inverse: true},
        ...cats.map(c => ({label: c, key: c, inverse: INVERSE_CATS.has(c)})),
        {label: 'Total SGP', key: 'sgp', inverse: false},
        {label: 'wSGP', key: 'wsgp', inverse: false},
    ];

    const rLabel = roster.name + ' (Yours)';
    const oLabel = other.name + (other.owner ? '' : ' (FA)');

    compStatsHead.innerHTML =
        '<th style="text-align:left">Stat</th>' +
        '<th style="text-align:right;color:var(--success)">' + rLabel + '</th>' +
        '<th style="text-align:right;color:#2196f3">' + oLabel + '</th>' +
        '<th style="text-align:right">Diff</th>';

    compStatsBody.innerHTML = rows.map(r => {
        const rv = roster[r.key];
        const ov = other[r.key];
        if (rv == null && ov == null) return '';

        const rFmt = fmt(rv, r.key);
        const oFmt = fmt(ov, r.key);
        const diff = (ov != null && rv != null) ? ov - rv : null;
        let diffFmt = '\u2014';
        let diffClass = '';
        if (diff != null) {
            const isRate = RATE_CATS.has(r.key);
            const better = r.inverse ? diff < 0 : diff > 0;
            const worse = r.inverse ? diff > 0 : diff < 0;
            diffClass = better ? 'diff-positive' : (worse ? 'diff-negative' : '');
            const sign = diff > 0 ? '+' : '';
            diffFmt = isRate ? sign + diff.toFixed(r.key === 'AVG' ? 3 : 2)
                     : r.key === 'sgp' || r.key === 'wsgp' ? sign + diff.toFixed(1)
                     : sign + Math.round(diff);
        }

        const rBetter = diff != null && (r.inverse ? diff > 0 : diff < 0);
        const oBetter = diff != null && (r.inverse ? diff < 0 : diff > 0);

        return '<tr>' +
            '<td style="color:var(--text-secondary)">' + r.label + '</td>' +
            '<td style="text-align:right" class="' + (rBetter ? 'better' : '') + '">' + rFmt + '</td>' +
            '<td style="text-align:right" class="' + (oBetter ? 'better' : '') + '">' + oFmt + '</td>' +
            '<td style="text-align:right" class="' + diffClass + '">' + diffFmt + '</td>' +
            '</tr>';
    }).join('');

    compStatsTable.style.display = '';
}
```

- [ ] **Step 5: Add the standings fetch and render functions**

Continue in the `<script>` block:

```javascript
function fetchStandings(rosterPlayer, otherPlayer) {
    standingsSection.style.display = 'none';
    const isHitter = otherPlayer.player_type === 'hitter';
    const params = new URLSearchParams({
        roster_player: rosterPlayer.name,
        other_player: otherPlayer.name,
        other_type: otherPlayer.player_type,
        other_r: otherPlayer.R || 0,
        other_hr: otherPlayer.HR || 0,
        other_rbi: otherPlayer.RBI || 0,
        other_sb: otherPlayer.SB || 0,
        other_h: otherPlayer.h || 0,
        other_ab: otherPlayer.ab || 0,
        other_w: otherPlayer.W || 0,
        other_k: otherPlayer.K || 0,
        other_sv: otherPlayer.SV || 0,
        other_ip: otherPlayer.ip || 0,
        other_er: otherPlayer.er || 0,
        other_bb: otherPlayer.bb || 0,
        other_ha: otherPlayer.h_allowed || 0,
    });
    fetch('/api/players/compare?' + params)
        .then(r => r.json())
        .then(data => {
            if (data.error) return;
            standingsData = data;
            document.getElementById('standings-subtitle').textContent =
                'Replacing ' + rosterPlayer.name + ' with ' + otherPlayer.name;
            document.getElementById('standings-after-label').textContent =
                'With ' + otherPlayer.name;
            renderStandings();
            standingsSection.style.display = '';
        });
}

function renderStandings() {
    if (!standingsData) return;
    const cats = standingsData.categories;
    const mode = standingsMode;

    const headHtml = '<th style="text-align:left">Team</th><th style="text-align:right">Pts</th>' +
        cats.map(c => '<th style="text-align:right">' + c + '</th>').join('');

    document.getElementById('standings-before-head').innerHTML = headHtml;
    document.getElementById('standings-after-head').innerHTML = headHtml;

    const teamNames = Object.keys(standingsData.before.roto);
    const sorted = teamNames.slice().sort((a, b) =>
        standingsData.before.roto[b].total - standingsData.before.roto[a].total
    );

    function renderBody(container, view, diffView) {
        container.innerHTML = sorted.map(team => {
            const roto = view.roto[team];
            const stats = view.stats[team];
            const isUser = roto.total !== undefined &&
                team === sorted.find(t => standingsData.before.roto[t] !== standingsData.after.roto[t] &&
                    t === team) || false;
            // Detect user team by checking if stats changed
            const beforeStats = standingsData.before.stats[team];
            const afterStats = standingsData.after.stats[team];
            const changed = JSON.stringify(beforeStats) !== JSON.stringify(afterStats);
            const rowStyle = changed ? 'color:var(--accent); font-weight:600' : '';

            const total = roto.total.toFixed(0);
            let totalDiff = '';
            if (diffView) {
                const d = roto.total - diffView.roto[team].total;
                if (Math.abs(d) > 0.01) {
                    const cls = d > 0 ? 'diff-positive' : 'diff-negative';
                    totalDiff = ' <span class="' + cls + '" style="font-size:11px">' +
                        (d > 0 ? '+' : '') + d.toFixed(0) + '</span>';
                }
            }

            const cells = cats.map(c => {
                const val = mode === 'roto' ? roto[c + '_pts'] : stats[c];
                const fmtVal = mode === 'roto' ? val.toFixed(0)
                    : (c === 'AVG' ? val.toFixed(3) : ['ERA','WHIP'].includes(c) ? val.toFixed(2) : val.toFixed(0));
                let delta = '';
                if (diffView) {
                    const dVal = mode === 'roto'
                        ? roto[c + '_pts'] - diffView.roto[team][c + '_pts']
                        : stats[c] - diffView.stats[team][c];
                    if (Math.abs(dVal) > 0.01) {
                        const cls = (INVERSE_CATS.has(c) ? dVal < 0 : dVal > 0) ? 'diff-positive' : 'diff-negative';
                        const fmtD = mode === 'roto' ? (dVal > 0 ? '+' : '') + dVal.toFixed(0)
                            : (c === 'AVG' ? (dVal > 0 ? '+' : '') + dVal.toFixed(3)
                            : ['ERA','WHIP'].includes(c) ? (dVal > 0 ? '+' : '') + dVal.toFixed(2)
                            : (dVal > 0 ? '+' : '') + dVal.toFixed(0));
                        delta = ' <span class="' + cls + '" style="font-size:11px">' + fmtD + '</span>';
                    }
                }
                return '<td style="text-align:right">' + fmtVal + delta + '</td>';
            }).join('');

            return '<tr style="' + rowStyle + '"><td style="text-align:left">' + team + '</td>' +
                '<td style="text-align:right;font-weight:600">' + total + totalDiff + '</td>' +
                cells + '</tr>';
        }).join('');
    }

    renderBody(
        document.getElementById('standings-before-body'),
        standingsData.before, null
    );
    renderBody(
        document.getElementById('standings-after-body'),
        standingsData.after, standingsData.before
    );
}
```

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/players.html
git commit -m "feat: add player comparison UI with stat table and standings"
```

---

### Task 4: Enrich browse API with component stats for standings computation

The `/api/players/browse` endpoint currently returns display stats (R, HR, etc.) but not the component stats (h, ab, ip, er, bb, h_allowed) needed by `project_team_stats`. The compare endpoint receives the other player's stats as query params from the client, so the browse API must include component stats.

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py` (inside `api_player_browse`, around line 590-604)

- [ ] **Step 1: Add component stats to the browse API response**

In `api_player_browse`, after the existing `result.update(...)` blocks for hitters and pitchers (around lines 590-604), add the component stats:

```python
                if ptype == PlayerType.HITTER:
                    result.update({"R": ros.r, "HR": ros.hr, "RBI": ros.rbi,
                                   "SB": ros.sb, "AVG": ros.avg,
                                   "h": ros.h, "ab": ros.ab})
                    actual_obj = HitterStats(pa=actual_pa.get(norm, 0))
                    result["significant"] = actual_obj.significant_dict()
                else:
                    result.update({"W": ros.w, "K": ros.k, "SV": ros.sv,
                                   "ERA": ros.era, "WHIP": ros.whip,
                                   "ip": ros.ip, "er": ros.er,
                                   "bb": ros.bb, "h_allowed": ros.h_allowed})
                    logs = actual_pitcher_logs.get(norm, {})
                    actual_obj = PitcherStats(
                        ip=logs.get("ip", 0),
                        bb=logs.get("bb", 0),
                        h_allowed=logs.get("h_allowed", 0),
                    )
                    result["significant"] = actual_obj.significant_dict()
```

This replaces the existing two blocks — the only change is adding `"h": ros.h, "ab": ros.ab` for hitters and `"ip": ros.ip, "er": ros.er, "bb": ros.bb, "h_allowed": ros.h_allowed` for pitchers.

- [ ] **Step 2: Run existing tests to make sure nothing breaks**

Run: `pytest tests/test_web/ -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py
git commit -m "feat: include component stats in browse API for comparison"
```

---

### Task 5: Manual testing and polish

**Files:** Potentially tweak `src/fantasy_baseball/web/templates/season/players.html`

- [ ] **Step 1: Run the full test suite**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 2: Start the dashboard locally and test the comparison flow**

```bash
python scripts/run_season_dashboard.py
```

Open the Players page. Test:
1. Select a position filter (e.g., SS)
2. Check a player on your roster (e.g., Willy Adames — green "Yours")
3. Check a free agent (e.g., Ezequiel Tovar — blue "FA")
4. Verify: stat comparison table appears with correct diffs and coloring
5. Verify: projected standings load with before/after tables
6. Toggle between "Roto Points" and "Stat Totals" — both should render
7. Click "Clear" — comparison panel should disappear
8. Select two FA players — should show validation message
9. Select two of your own players — should show validation message

- [ ] **Step 3: Fix any issues found during manual testing**

- [ ] **Step 4: Commit any fixes**

```bash
git add -u
git commit -m "fix: polish player comparison UI after manual testing"
```
