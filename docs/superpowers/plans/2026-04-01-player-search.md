# Player Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Players" tab to the dashboard with a search box that queries the full ROS projection pool and shows each player's ROS projections, preseason comparison, pace vs preseason, wSGP, and ownership status.

**Architecture:** New Flask route `/players` renders the page. API endpoint `/api/players/search?q=name` queries SQLite, enriches with preseason stats, game logs, leverage/wSGP, and ownership. Frontend uses debounced fetch to render player cards.

**Tech Stack:** Python/Flask, SQLite, Jinja2, vanilla JS

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/web/season_routes.py` | Modify | Add `/players` route + `/api/players/search` endpoint |
| `src/fantasy_baseball/web/templates/season/players.html` | Create | Search page template |
| `src/fantasy_baseball/web/templates/season/base.html` | Modify | Add "Players" nav link |
| `tests/test_web/test_player_search.py` | Create | Tests for search API + page route |

---

### Task 1: Add nav link and empty page route

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/base.html:23-26`
- Create: `src/fantasy_baseball/web/templates/season/players.html`
- Modify: `src/fantasy_baseball/web/season_routes.py`

- [ ] **Step 1: Add nav link to base.html**

In `src/fantasy_baseball/web/templates/season/base.html`, after the "Waivers & Trades" link (line 25) and before the "SQL" link (line 27), add:

```html
            <a href="{{ url_for('player_search') }}"
               class="nav-link {% if active_page == 'players' %}active{% endif %}">
                Players
            </a>
```

- [ ] **Step 2: Create the players template**

Create `src/fantasy_baseball/web/templates/season/players.html`:

```html
{% extends "season/base.html" %}
{% block title %}Players — Season Dashboard{% endblock %}
{% block content %}

<style>
.search-container {
    margin: 20px 0;
}
.search-input {
    width: 100%;
    max-width: 400px;
    padding: 10px 14px;
    font-size: 14px;
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    color: var(--text);
    outline: none;
}
.search-input:focus {
    border-color: var(--accent);
}
.search-input::placeholder {
    color: var(--text-secondary);
}
.search-hint {
    color: var(--text-secondary);
    font-size: 12px;
    margin-top: 6px;
}
.search-status {
    color: var(--text-secondary);
    font-size: 13px;
    margin: 12px 0;
}
.player-card {
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 6px;
    padding: 14px 18px;
    margin-bottom: 10px;
}
.player-card-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 10px;
}
.player-card-header .player-info {
    display: flex;
    align-items: baseline;
    gap: 8px;
}
.player-card-header .player-name {
    font-weight: 600;
    font-size: 15px;
}
.player-card-header .player-meta {
    color: var(--text-secondary);
    font-size: 12px;
}
.ownership-badge {
    font-size: 11px;
    padding: 2px 8px;
    border-radius: 3px;
    font-weight: 600;
}
.ownership-badge.roster { background: rgba(76, 175, 80, 0.15); color: #4caf50; }
.ownership-badge.opponent { background: rgba(255, 152, 0, 0.15); color: #ff9800; }
.ownership-badge.fa { background: rgba(33, 150, 243, 0.15); color: #2196f3; }
.wsgp-val {
    font-weight: 600;
    font-size: 14px;
    color: var(--accent);
    margin-left: 12px;
}
.stat-table {
    width: 100%;
    font-size: 13px;
    border-collapse: collapse;
}
.stat-table th {
    text-align: left;
    color: var(--text-secondary);
    font-weight: 500;
    font-size: 11px;
    text-transform: uppercase;
    padding: 4px 8px;
    border-bottom: 1px solid var(--panel-border);
}
.stat-table td {
    padding: 4px 8px;
}
.stat-table tr:not(:last-child) td {
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
#results-container { margin-top: 16px; }
</style>

<div class="search-container">
    <input type="text" id="player-search" class="search-input"
           placeholder="Search players by name..." autocomplete="off">
    <div class="search-hint">Type at least 2 characters to search</div>
</div>
<div id="search-status" class="search-status"></div>
<div id="results-container"></div>

<script>
const input = document.getElementById('player-search');
const status = document.getElementById('search-status');
const results = document.getElementById('results-container');
let debounceTimer = null;

input.addEventListener('input', function() {
    clearTimeout(debounceTimer);
    const q = this.value.trim();
    if (q.length < 2) {
        status.textContent = '';
        results.innerHTML = '';
        return;
    }
    status.textContent = 'Searching...';
    debounceTimer = setTimeout(() => doSearch(q), 300);
});

function doSearch(q) {
    fetch('/api/players/search?q=' + encodeURIComponent(q))
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                status.textContent = data.error;
                results.innerHTML = '';
                return;
            }
            status.textContent = data.length + ' result' + (data.length !== 1 ? 's' : '');
            results.innerHTML = data.map(renderCard).join('');
        })
        .catch(() => {
            status.textContent = 'Search failed';
            results.innerHTML = '';
        });
}

function fmt(val, cat) {
    if (val === null || val === undefined) return '—';
    if (cat === 'AVG') return val.toFixed(3);
    if (cat === 'ERA' || cat === 'WHIP') return val.toFixed(2);
    return Math.round(val);
}

function zBadge(z) {
    if (z === null || z === undefined) return '';
    const cls = z > 1 ? 'stat-hot-1' : z > 2 ? 'stat-hot-2' : z < -1 ? 'stat-cold-1' : z < -2 ? 'stat-cold-2' : 'stat-neutral';
    return '<span class="' + cls + '">' + (z >= 0 ? '+' : '') + z.toFixed(1) + '</span>';
}

function renderCard(p) {
    const cats = p.player_type === 'hitter'
        ? ['R', 'HR', 'RBI', 'SB', 'AVG']
        : ['W', 'K', 'SV', 'ERA', 'WHIP'];

    const ownerClass = p.ownership === 'Your roster' ? 'roster'
        : p.ownership === 'Free Agent' ? 'fa' : 'opponent';

    let rows = cats.map(cat => {
        const key = cat.toLowerCase();
        const ros = p.ros ? p.ros[key] : null;
        const pre = p.preseason ? p.preseason[key] : null;
        const pace = p.pace && p.pace[cat] ? p.pace[cat] : null;
        return '<tr>' +
            '<td style="font-weight:600">' + cat + '</td>' +
            '<td>' + fmt(ros, cat) + '</td>' +
            '<td>' + fmt(pre, cat) + '</td>' +
            '<td>' + (pace ? fmt(pace.actual, cat) : '—') + '</td>' +
            '<td>' + (pace ? fmt(pace.expected, cat) : '—') + '</td>' +
            '<td>' + (pace ? zBadge(pace.z_score) : '—') + '</td>' +
            '</tr>';
    }).join('');

    const positions = p.positions ? p.positions.join(', ') : '';
    const team = p.team || '';
    const wsgp = p.wsgp !== null ? p.wsgp.toFixed(2) : '—';

    return '<div class="player-card">' +
        '<div class="player-card-header">' +
            '<div class="player-info">' +
                '<span class="player-name">' + p.name + '</span>' +
                '<span class="player-meta">' + positions + ' — ' + team + '</span>' +
                '<span class="wsgp-val">' + wsgp + ' wSGP</span>' +
            '</div>' +
            '<span class="ownership-badge ' + ownerClass + '">' + p.ownership + '</span>' +
        '</div>' +
        '<table class="stat-table">' +
            '<thead><tr><th>Cat</th><th>ROS</th><th>Preseason</th><th>Actual</th><th>Pace</th><th>Z</th></tr></thead>' +
            '<tbody>' + rows + '</tbody>' +
        '</table>' +
    '</div>';
}
</script>

{% endblock %}
```

- [ ] **Step 3: Add route to season_routes.py**

In `src/fantasy_baseball/web/season_routes.py`, inside the `create_season_app` function, add before the login route:

```python
    @app.route("/players")
    def player_search():
        meta = read_meta()
        return render_template(
            "season/players.html",
            meta=meta,
            active_page="players",
        )
```

- [ ] **Step 4: Write test for page route**

Create `tests/test_web/test_player_search.py`:

```python
import pytest
from fantasy_baseball.web.season_routes import create_season_app


@pytest.fixture
def client():
    app = create_season_app()
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test"
    with app.test_client() as c:
        yield c


def test_players_page_renders(client):
    resp = client.get("/players")
    assert resp.status_code == 200
    assert b"Search players by name" in resp.data
```

- [ ] **Step 5: Run test**

Run: `pytest tests/test_web/test_player_search.py::test_players_page_renders -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/base.html src/fantasy_baseball/web/templates/season/players.html src/fantasy_baseball/web/season_routes.py tests/test_web/test_player_search.py
git commit -m "feat: add Players nav tab with search page skeleton"
```

---

### Task 2: Implement search API endpoint

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Test: `tests/test_web/test_player_search.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_web/test_player_search.py`:

```python
import json
import sqlite3
from unittest.mock import patch


def _seed_test_db(conn):
    """Insert test projection data into an in-memory SQLite database."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS blended_projections (
            year INTEGER, fg_id TEXT, name TEXT, team TEXT, player_type TEXT,
            pa REAL, ab REAL, h REAL, r REAL, hr REAL, rbi REAL, sb REAL, avg REAL,
            w REAL, k REAL, sv REAL, ip REAL, er REAL, bb REAL, h_allowed REAL,
            era REAL, whip REAL, adp REAL,
            PRIMARY KEY (year, fg_id)
        );
        CREATE TABLE IF NOT EXISTS ros_blended_projections (
            year INTEGER, snapshot_date TEXT, fg_id TEXT, name TEXT, team TEXT,
            player_type TEXT,
            pa REAL, ab REAL, h REAL, r REAL, hr REAL, rbi REAL, sb REAL, avg REAL,
            w REAL, k REAL, sv REAL, ip REAL, er REAL, bb REAL, h_allowed REAL,
            era REAL, whip REAL, adp REAL,
            PRIMARY KEY (year, snapshot_date, fg_id)
        );
        CREATE TABLE IF NOT EXISTS game_logs (
            season INTEGER, mlbam_id INTEGER, name TEXT, team TEXT,
            player_type TEXT, date TEXT,
            pa INTEGER, ab INTEGER, h INTEGER, r INTEGER, hr INTEGER,
            rbi INTEGER, sb INTEGER,
            ip REAL, k INTEGER, er INTEGER, bb INTEGER, h_allowed INTEGER,
            w INTEGER, sv INTEGER, gs INTEGER,
            PRIMARY KEY (season, mlbam_id, date)
        );
    """)
    conn.execute(
        "INSERT INTO ros_blended_projections VALUES "
        "(2026, '2026-04-01', '15640', 'Aaron Judge', 'NYY', 'hitter', "
        "600, 500, 145, 95, 38, 92, 7, 0.290, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 5.0)"
    )
    conn.execute(
        "INSERT INTO blended_projections VALUES "
        "(2026, '15640', 'Aaron Judge', 'NYY', 'hitter', "
        "650, 550, 160, 110, 45, 120, 5, 0.291, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 5.0)"
    )
    conn.execute(
        "INSERT INTO ros_blended_projections VALUES "
        "(2026, '2026-04-01', '28027', 'Gerrit Cole', 'NYY', 'pitcher', "
        "NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, 14, 200, 0, 190, 60, 40, 140, 2.84, 0.95, 20.0)"
    )
    conn.commit()


def test_search_returns_matching_players(client):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed_test_db(conn)

    with patch("fantasy_baseball.web.season_routes._get_search_db") as mock_db:
        mock_db.return_value = conn
        resp = client.get("/api/players/search?q=judge")

    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert len(data) == 1
    assert data[0]["name"] == "Aaron Judge"
    assert data[0]["player_type"] == "hitter"
    assert data[0]["ros"]["hr"] == 38
    assert data[0]["preseason"]["hr"] == 45


def test_search_requires_min_2_chars(client):
    resp = client.get("/api/players/search?q=j")
    assert resp.status_code == 200
    data = json.loads(resp.data)
    assert data == []


def test_search_no_results(client):
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    _seed_test_db(conn)

    with patch("fantasy_baseball.web.season_routes._get_search_db") as mock_db:
        mock_db.return_value = conn
        resp = client.get("/api/players/search?q=nonexistent")

    data = json.loads(resp.data)
    assert data == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web/test_player_search.py -v`
Expected: 1 pass (page route), 3 fail (API not implemented)

- [ ] **Step 3: Implement the search API**

In `src/fantasy_baseball/web/season_routes.py`, add a helper function near the top (after `_load_config`):

```python
def _get_search_db():
    """Get a SQLite connection for player search queries."""
    from fantasy_baseball.data.db import get_connection
    return get_connection()
```

Then add the API route inside `create_season_app`, after the `player_search` route:

```python
    @app.route("/api/players/search")
    def api_player_search():
        from datetime import date
        from fantasy_baseball.utils.name_utils import normalize_name
        from fantasy_baseball.lineup.leverage import calculate_leverage
        from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
        from fantasy_baseball.analysis.pace import compute_player_pace
        from fantasy_baseball.utils.constants import HITTER_PROJ_KEYS, PITCHER_PROJ_KEYS
        import pandas as pd

        query = request.args.get("q", "").strip()
        if len(query) < 2:
            return jsonify([])

        conn = _get_search_db()
        try:
            season = date.today().year

            # Find latest ROS snapshot
            row = conn.execute(
                "SELECT MAX(snapshot_date) as d FROM ros_blended_projections WHERE year = ?",
                (season,),
            ).fetchone()
            snapshot = row["d"] if row and row["d"] else None
            if not snapshot:
                return jsonify([])

            # Search ROS projections by name (case-insensitive LIKE)
            like_pattern = f"%{query}%"
            ros_rows = conn.execute(
                "SELECT * FROM ros_blended_projections "
                "WHERE year = ? AND snapshot_date = ? AND name LIKE ? "
                "ORDER BY CASE WHEN adp IS NOT NULL THEN adp ELSE 9999 END ASC "
                "LIMIT 25",
                (season, snapshot, like_pattern),
            ).fetchall()

            if not ros_rows:
                return jsonify([])

            # Load preseason projections for comparison
            preseason_map = {}
            for r in conn.execute(
                "SELECT * FROM blended_projections WHERE year = ? AND name LIKE ?",
                (season, like_pattern),
            ).fetchall():
                preseason_map[r["fg_id"]] = dict(r)

            # Load game log totals for pace
            hitter_logs = {}
            for r in conn.execute(
                "SELECT name, SUM(pa) as pa, SUM(ab) as ab, SUM(h) as h, "
                "SUM(r) as r, SUM(hr) as hr, SUM(rbi) as rbi, SUM(sb) as sb "
                "FROM game_logs WHERE season = ? AND player_type = 'hitter' "
                "GROUP BY name", (season,),
            ).fetchall():
                hitter_logs[normalize_name(r["name"])] = dict(r)

            pitcher_logs = {}
            for r in conn.execute(
                "SELECT name, SUM(ip) as ip, SUM(k) as k, SUM(w) as w, SUM(sv) as sv, "
                "SUM(er) as er, SUM(bb) as bb, SUM(h_allowed) as h_allowed "
                "FROM game_logs WHERE season = ? AND player_type = 'pitcher' "
                "GROUP BY name", (season,),
            ).fetchall():
                pitcher_logs[normalize_name(r["name"])] = dict(r)

            # Leverage for wSGP
            standings = read_cache("standings") or []
            config = _load_config()
            projected_standings_cache = read_cache("projections") or {}
            projected_standings = projected_standings_cache.get("projected_standings")
            leverage = calculate_leverage(
                standings, config.team_name,
                projected_standings=projected_standings,
            ) if standings else {c: 1.0 / 10 for c in ALL_CATEGORIES}

            # Ownership lookup
            roster_cache = read_cache("roster") or []
            roster_names = {normalize_name(p["name"]): "Your roster" for p in roster_cache}
            # Check opponent rosters from weekly_rosters table
            opp_rows = conn.execute(
                "SELECT player_name, team FROM weekly_rosters "
                "WHERE snapshot_date = (SELECT MAX(snapshot_date) FROM weekly_rosters) "
                "AND team != ?",
                (config.team_name,),
            ).fetchall()
            for r in opp_rows:
                norm = normalize_name(r["player_name"])
                if norm not in roster_names:
                    roster_names[norm] = r["team"]

            # Build results
            results = []
            for ros in ros_rows:
                ros_dict = dict(ros)
                name = ros_dict["name"]
                norm = normalize_name(name)
                ptype = ros_dict["player_type"]
                fg_id = ros_dict.get("fg_id")

                # ROS stats
                if ptype == "hitter":
                    ros_stats = {k: ros_dict.get(k) for k in ["r", "hr", "rbi", "sb", "avg"]}
                else:
                    ros_stats = {k: ros_dict.get(k) for k in ["w", "k", "sv", "era", "whip"]}

                # Preseason stats
                pre = preseason_map.get(fg_id, {})
                if ptype == "hitter":
                    pre_stats = {k: pre.get(k) for k in ["r", "hr", "rbi", "sb", "avg"]}
                else:
                    pre_stats = {k: pre.get(k) for k in ["w", "k", "sv", "era", "whip"]}

                # wSGP
                wsgp = calculate_weighted_sgp(pd.Series(ros_dict), leverage)

                # Pace
                pace = None
                logs = hitter_logs if ptype == "hitter" else pitcher_logs
                actuals = logs.get(norm)
                if actuals:
                    proj_keys = HITTER_PROJ_KEYS if ptype == "hitter" else PITCHER_PROJ_KEYS
                    projected = {k: pre.get(k, 0) or 0 for k in proj_keys}
                    if any(v > 0 for v in projected.values()):
                        pace = compute_player_pace(actuals, projected, ptype)

                # Ownership
                ownership = roster_names.get(norm, "Free Agent")

                # Positions from weekly_rosters or projection data
                positions = []
                pos_row = conn.execute(
                    "SELECT positions FROM weekly_rosters WHERE player_name = ? "
                    "ORDER BY snapshot_date DESC LIMIT 1",
                    (name,),
                ).fetchone()
                if pos_row and pos_row["positions"]:
                    positions = [p.strip() for p in pos_row["positions"].split(",")]

                results.append({
                    "name": name,
                    "team": ros_dict.get("team", ""),
                    "positions": positions,
                    "player_type": ptype,
                    "ownership": ownership,
                    "wsgp": round(wsgp, 2),
                    "ros": ros_stats,
                    "preseason": pre_stats,
                    "pace": pace,
                })

            return jsonify(results)
        finally:
            conn.close()
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_web/test_player_search.py -v`
Expected: All 4 tests pass

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py tests/test_web/test_player_search.py
git commit -m "feat: add /api/players/search endpoint with ROS, preseason, pace, wSGP"
```
