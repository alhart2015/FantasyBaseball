# Season Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an in-season Flask + htmx dashboard that shows standings, lineup optimization, and waiver/trade recommendations, loading instantly from cached data with on-demand refresh.

**Architecture:** New Flask app (`season_app.py`) alongside the existing draft dashboard. Three pages (Standings, Lineup, Waivers & Trades) served from Jinja2 templates with htmx for interactivity. A `season_data.py` module manages cached JSON files in `data/cache/` and orchestrates Yahoo API calls + computation on refresh. No new math — all computation reuses existing modules.

**Tech Stack:** Flask, htmx 2.0, Jinja2, vanilla JS, existing Python modules (scoring, simulation, lineup optimization, trades)

**Spec:** `docs/superpowers/specs/2026-03-26-season-dashboard-design.md`

---

## File Structure

```
src/fantasy_baseball/web/
├── season_app.py               # Flask app factory + __main__ launch
├── season_routes.py            # Route handlers for all 3 pages + API endpoints
├── season_data.py              # Cache read/write, refresh orchestration, data formatting
├── templates/season/
│   ├── base.html               # Sidebar + main content shell (extends nothing, standalone)
│   ├── standings.html          # Standings page (extends base.html)
│   ├── lineup.html             # Lineup page (extends base.html)
│   └── waivers_trades.html     # Waivers & Trades page (extends base.html)
├── static/
│   └── season.css              # All season dashboard styles (dark theme, tables, cards)

scripts/
└── run_season_dashboard.py     # Entry point script

tests/test_web/
├── test_season_data.py         # Cache layer + data formatting tests
└── test_season_routes.py       # Flask test client route tests

data/cache/                     # Cached computation results (gitignored)
```

**Key module responsibilities:**
- **`season_data.py`** — Pure data layer. Reads/writes cache files, formats standings/lineup/waiver/trade data for templates, orchestrates the refresh pipeline (Yahoo fetch → compute → cache). No Flask imports.
- **`season_routes.py`** — Thin route handlers. Reads data via `season_data`, passes to templates. Handles refresh API (background thread) and optimize API.
- **`season_app.py`** — App factory. Creates Flask app, registers routes, configures paths. Also serves as `__main__` entry point.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `src/fantasy_baseball/web/season_app.py`
- Create: `src/fantasy_baseball/web/season_routes.py`
- Create: `src/fantasy_baseball/web/season_data.py`
- Create: `src/fantasy_baseball/web/templates/season/base.html`
- Create: `src/fantasy_baseball/web/static/season.css`
- Create: `scripts/run_season_dashboard.py`
- Create: `tests/test_web/test_season_routes.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add `data/cache/` to `.gitignore`**

Append to `.gitignore`:
```
data/cache/
```

- [ ] **Step 2: Create `season_data.py` with cache path constants and read/write helpers**

```python
"""Cache management and data assembly for the season dashboard."""

import json
import os
import tempfile
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "cache"

CACHE_FILES = {
    "standings": "standings.json",
    "roster": "roster.json",
    "projections": "projections.json",
    "lineup_optimal": "lineup_optimal.json",
    "probable_starters": "probable_starters.json",
    "waivers": "waivers.json",
    "trades": "trades.json",
    "monte_carlo": "monte_carlo.json",
    "meta": "meta.json",
}


def read_cache(key: str, cache_dir: Path = CACHE_DIR) -> dict | list | None:
    """Read a cached JSON file. Returns None if missing or corrupt."""
    path = cache_dir / CACHE_FILES[key]
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        return None


def write_cache(key: str, data: dict | list, cache_dir: Path = CACHE_DIR) -> None:
    """Atomically write a cached JSON file (tmpfile + rename)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / CACHE_FILES[key]
    fd, tmp = tempfile.mkstemp(dir=cache_dir, suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        # On Windows, must remove target before rename
        if path.exists():
            path.unlink()
        Path(tmp).rename(path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def read_meta(cache_dir: Path = CACHE_DIR) -> dict:
    """Read cache metadata (last refresh time, week, etc.). Returns empty dict if missing."""
    return read_cache("meta", cache_dir) or {}
```

- [ ] **Step 3: Write test for cache read/write**

Create `tests/test_web/__init__.py` (empty) and `tests/test_web/test_season_data.py`:

```python
import json
from pathlib import Path

from fantasy_baseball.web.season_data import read_cache, write_cache, read_meta


def test_write_and_read_cache(tmp_path):
    data = {"teams": [{"name": "Hart of the Order", "total": 67}]}
    write_cache("standings", data, cache_dir=tmp_path)

    result = read_cache("standings", cache_dir=tmp_path)
    assert result == data


def test_read_cache_missing_file(tmp_path):
    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None


def test_read_cache_corrupt_json(tmp_path):
    path = tmp_path / "standings.json"
    path.write_text("not json", encoding="utf-8")
    result = read_cache("standings", cache_dir=tmp_path)
    assert result is None


def test_read_meta_missing(tmp_path):
    result = read_meta(cache_dir=tmp_path)
    assert result == {}


def test_write_cache_overwrites(tmp_path):
    write_cache("standings", {"v": 1}, cache_dir=tmp_path)
    write_cache("standings", {"v": 2}, cache_dir=tmp_path)
    result = read_cache("standings", cache_dir=tmp_path)
    assert result == {"v": 2}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web/test_season_data.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Create `season_app.py` — Flask app factory**

```python
"""Season dashboard Flask application."""

from pathlib import Path

from flask import Flask

from fantasy_baseball.web.season_routes import register_routes


def create_app() -> Flask:
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    register_routes(app)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(port=5001, debug=True)
```

- [ ] **Step 6: Create `season_routes.py` — route stubs**

```python
"""Route handlers for the season dashboard."""

from flask import Flask, redirect, render_template, url_for

from fantasy_baseball.web.season_data import read_cache, read_meta


def register_routes(app: Flask) -> None:

    @app.route("/")
    def index():
        return redirect(url_for("standings"))

    @app.route("/standings")
    def standings():
        meta = read_meta()
        return render_template("season/standings.html", meta=meta, active_page="standings")

    @app.route("/lineup")
    def lineup():
        meta = read_meta()
        return render_template("season/lineup.html", meta=meta, active_page="lineup")

    @app.route("/waivers-trades")
    def waivers_trades():
        meta = read_meta()
        return render_template(
            "season/waivers_trades.html", meta=meta, active_page="waivers_trades"
        )
```

- [ ] **Step 7: Create `base.html` — sidebar + content shell**

Create `src/fantasy_baseball/web/templates/season/base.html`:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Season Dashboard{% endblock %}</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='season.css') }}">
    <script src="https://unpkg.com/htmx.org@2.0.4"></script>
</head>
<body>
    <div class="layout">
        <nav class="sidebar">
            <div class="sidebar-header">Hart of the Order</div>
            <a href="{{ url_for('standings') }}"
               class="nav-link {% if active_page == 'standings' %}active{% endif %}">
                Standings
            </a>
            <a href="{{ url_for('lineup') }}"
               class="nav-link {% if active_page == 'lineup' %}active{% endif %}">
                Lineup
            </a>
            <a href="{{ url_for('waivers_trades') }}"
               class="nav-link {% if active_page == 'waivers_trades' %}active{% endif %}">
                Waivers &amp; Trades
            </a>
            <div class="sidebar-footer">
                <div class="refresh-info">
                    {% if meta and meta.get('last_refresh') %}
                        Last refresh: {{ meta['last_refresh'] }}
                    {% else %}
                        No data yet
                    {% endif %}
                </div>
                <button class="btn-refresh"
                        hx-post="/api/refresh"
                        hx-swap="none"
                        id="refresh-btn">
                    Refresh Data
                </button>
            </div>
        </nav>
        <main class="content">
            {% block content %}{% endblock %}
        </main>
    </div>
</body>
</html>
```

- [ ] **Step 8: Create stub page templates**

Create `src/fantasy_baseball/web/templates/season/standings.html`:
```html
{% extends "season/base.html" %}
{% block title %}Standings — Season Dashboard{% endblock %}
{% block content %}
<div class="page-header">
    <h2>Standings</h2>
</div>
<p class="placeholder-text">Standings data will appear here after a refresh.</p>
{% endblock %}
```

Create `src/fantasy_baseball/web/templates/season/lineup.html`:
```html
{% extends "season/base.html" %}
{% block title %}Lineup — Season Dashboard{% endblock %}
{% block content %}
<div class="page-header">
    <h2>Lineup</h2>
</div>
<p class="placeholder-text">Lineup data will appear here after a refresh.</p>
{% endblock %}
```

Create `src/fantasy_baseball/web/templates/season/waivers_trades.html`:
```html
{% extends "season/base.html" %}
{% block title %}Waivers &amp; Trades — Season Dashboard{% endblock %}
{% block content %}
<div class="page-header">
    <h2>Waivers &amp; Trades</h2>
</div>
<p class="placeholder-text">Waiver and trade data will appear here after a refresh.</p>
{% endblock %}
```

- [ ] **Step 9: Create `season.css` — base styles + sidebar + layout**

Create `src/fantasy_baseball/web/static/season.css`:

```css
/* === Season Dashboard Styles === */

:root {
    --bg: #0f0f1a;
    --panel-bg: #1a1a2e;
    --panel-border: #16213e;
    --card-bg: #1e1e1e;
    --accent: #64b5f6;
    --text: #eee;
    --text-secondary: #888;
    --success: #66bb6a;
    --warning: #f9a825;
    --danger: #ef5350;
    --row-highlight: rgba(100, 181, 246, 0.08);
}

* { margin: 0; padding: 0; box-sizing: border-box; }

body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
}

/* --- Layout --- */
.layout {
    display: flex;
    min-height: 100vh;
}

/* --- Sidebar --- */
.sidebar {
    width: 220px;
    background: var(--panel-bg);
    border-right: 1px solid var(--panel-border);
    display: flex;
    flex-direction: column;
    flex-shrink: 0;
    position: sticky;
    top: 0;
    height: 100vh;
}

.sidebar-header {
    padding: 16px 20px;
    font-size: 16px;
    font-weight: bold;
    color: var(--accent);
    border-bottom: 1px solid var(--panel-border);
}

.nav-link {
    display: block;
    padding: 12px 20px;
    color: var(--text);
    text-decoration: none;
    font-size: 14px;
    opacity: 0.7;
    transition: opacity 0.15s, background 0.15s;
}

.nav-link:hover {
    opacity: 1;
    background: rgba(255, 255, 255, 0.03);
}

.nav-link.active {
    opacity: 1;
    font-weight: bold;
    background: rgba(100, 181, 246, 0.08);
    border-left: 3px solid var(--accent);
    padding-left: 17px;
}

.sidebar-footer {
    margin-top: auto;
    padding: 16px 20px;
    border-top: 1px solid var(--panel-border);
}

.refresh-info {
    font-size: 12px;
    color: var(--text-secondary);
    margin-bottom: 10px;
}

.btn-refresh {
    width: 100%;
    padding: 8px 16px;
    background: var(--accent);
    color: var(--bg);
    border: none;
    border-radius: 4px;
    cursor: pointer;
    font-weight: bold;
    font-size: 13px;
    transition: opacity 0.15s;
}

.btn-refresh:hover { opacity: 0.9; }
.btn-refresh:disabled { opacity: 0.5; cursor: default; }

/* --- Content --- */
.content {
    flex: 1;
    padding: 24px 32px;
    overflow-y: auto;
}

.page-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
}

.page-header h2 {
    font-size: 20px;
    font-weight: 600;
    color: var(--text);
}

.placeholder-text {
    color: var(--text-secondary);
    font-size: 14px;
    font-style: italic;
}

/* --- Pill Toggles --- */
.pill-group {
    display: flex;
    gap: 6px;
    margin-bottom: 16px;
}

.pill {
    padding: 6px 14px;
    border-radius: 16px;
    font-size: 13px;
    border: 1px solid var(--panel-border);
    background: var(--card-bg);
    color: var(--text);
    cursor: pointer;
    opacity: 0.6;
    transition: all 0.15s;
}

.pill:hover { opacity: 0.8; }

.pill.active {
    opacity: 1;
    border-color: var(--accent);
    color: var(--accent);
    background: rgba(100, 181, 246, 0.1);
}

/* --- Tables --- */
.data-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    background: var(--card-bg);
    border-radius: 8px;
    overflow: hidden;
    border: 1px solid var(--panel-border);
}

.data-table th {
    text-align: center;
    padding: 8px 6px;
    color: var(--text-secondary);
    font-weight: 600;
    font-size: 12px;
    border-bottom: 2px solid var(--panel-border);
    background: var(--panel-bg);
}

.data-table th:first-child,
.data-table td:first-child { text-align: left; }

.data-table td {
    padding: 8px 6px;
    text-align: center;
    border-bottom: 1px solid rgba(255, 255, 255, 0.04);
}

.data-table tr:last-child td { border-bottom: none; }

.data-table .team-name { text-align: left; font-weight: 500; }

/* Team highlight and category coloring */
.data-table tr.user-team { background: var(--row-highlight); }
.data-table tr.user-team .team-name { color: var(--accent); font-weight: bold; }

td.cat-top { color: var(--success); font-weight: bold; }
td.cat-bottom { color: var(--danger); font-weight: bold; }

.total-col { color: var(--accent); font-weight: bold; }

/* --- Cards --- */
.card {
    background: var(--card-bg);
    border: 1px solid var(--panel-border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 12px;
}

.card-header {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
}

/* --- Badges --- */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 3px;
    font-size: 11px;
    font-weight: bold;
}

.badge-success { background: var(--success); color: #fff; }
.badge-warning { background: var(--warning); color: #1a1a1a; }
.badge-danger { background: var(--danger); color: #fff; }
.badge-info { background: rgba(100, 181, 246, 0.2); color: var(--accent); }
.badge-il { background: #c62828; color: #fff; }

/* --- Buttons --- */
.btn-optimize {
    padding: 8px 18px;
    border: none;
    border-radius: 4px;
    font-weight: bold;
    font-size: 13px;
    cursor: pointer;
}

.btn-optimize.available { background: var(--success); color: #fff; }
.btn-optimize.optimal { background: #555; color: #aaa; cursor: default; }

/* --- Suggested Moves Banner --- */
.moves-banner {
    background: rgba(100, 181, 246, 0.1);
    border: 1px solid rgba(100, 181, 246, 0.3);
    border-radius: 6px;
    padding: 14px;
    margin: 12px 0;
}

.moves-banner h4 { color: var(--accent); font-size: 13px; margin-bottom: 10px; }

.move-row {
    display: flex;
    align-items: center;
    gap: 8px;
    font-size: 12px;
    margin-bottom: 6px;
}

/* --- Trade Cards --- */
.trade-swap {
    display: flex;
    align-items: center;
    gap: 12px;
    font-size: 13px;
    margin-top: 10px;
}

.trade-side {
    flex: 1;
    text-align: center;
    padding: 10px;
    border-radius: 6px;
}

.trade-send { background: rgba(239, 83, 80, 0.08); }
.trade-receive { background: rgba(102, 187, 106, 0.08); }

.trade-arrow { font-size: 20px; color: var(--text-secondary); }

.trade-details {
    margin-top: 12px;
    border-top: 1px solid rgba(255, 255, 255, 0.06);
    padding-top: 12px;
    display: none;
}

.trade-details.open { display: block; }

.trade-pitch {
    margin-top: 10px;
    background: #252530;
    border-radius: 6px;
    padding: 12px;
    font-size: 12px;
    font-style: italic;
    color: #ccc;
    border-left: 3px solid var(--accent);
}

/* --- Expandable Rows --- */
.expandable { cursor: pointer; }
.expandable:hover { background: rgba(255, 255, 255, 0.02); }

.expand-content {
    display: none;
    padding: 8px 6px 8px 24px;
    font-size: 12px;
    color: var(--text-secondary);
    background: rgba(0, 0, 0, 0.15);
}

.expand-content.open { display: table-row; }

/* --- Category Impact --- */
.cat-impact {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    font-size: 11px;
    margin-top: 6px;
}

.cat-gain { color: var(--success); }
.cat-loss { color: var(--danger); }

/* --- Bench/IL rows --- */
.bench-row { opacity: 0.5; }
.il-active-row { background: rgba(239, 83, 80, 0.06); }

/* --- Scrollbar --- */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--panel-border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-secondary); }

/* --- Responsive (mobile: sidebar collapses to top bar) --- */
@media (max-width: 768px) {
    .layout { flex-direction: column; }

    .sidebar {
        width: 100%;
        height: auto;
        position: static;
        flex-direction: row;
        flex-wrap: wrap;
        align-items: center;
        padding: 8px 12px;
        gap: 4px;
    }

    .sidebar-header {
        padding: 4px 8px;
        border-bottom: none;
        font-size: 14px;
    }

    .nav-link {
        padding: 6px 12px;
        font-size: 13px;
    }

    .nav-link.active {
        border-left: none;
        border-bottom: 2px solid var(--accent);
        padding-left: 12px;
    }

    .sidebar-footer {
        margin-top: 0;
        padding: 4px 8px;
        border-top: none;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    .refresh-info { margin-bottom: 0; }

    .btn-refresh { width: auto; padding: 6px 12px; font-size: 12px; }

    .content { padding: 16px; }
}
```

- [ ] **Step 10: Create `run_season_dashboard.py` launch script**

```python
#!/usr/bin/env python3
"""Launch the season dashboard."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fantasy_baseball.web.season_app import create_app

if __name__ == "__main__":
    app = create_app()
    print("Season dashboard: http://localhost:5001")
    app.run(port=5001, debug=True)
```

- [ ] **Step 11: Write route tests with Flask test client**

Create `tests/test_web/test_season_routes.py`:

```python
import pytest

from fantasy_baseball.web.season_app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_index_redirects_to_standings(client):
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/standings" in resp.headers["Location"]


def test_standings_page_renders(client):
    resp = client.get("/standings")
    assert resp.status_code == 200
    assert b"Standings" in resp.data


def test_lineup_page_renders(client):
    resp = client.get("/lineup")
    assert resp.status_code == 200
    assert b"Lineup" in resp.data


def test_waivers_trades_page_renders(client):
    resp = client.get("/waivers-trades")
    assert resp.status_code == 200
    assert b"Waivers" in resp.data


def test_sidebar_nav_links_present(client):
    resp = client.get("/standings")
    html = resp.data.decode()
    assert 'href="/standings"' in html
    assert 'href="/lineup"' in html
    assert 'href="/waivers-trades"' in html


def test_active_page_highlighted(client):
    resp = client.get("/standings")
    html = resp.data.decode()
    # The standings link should have the active class
    assert 'class="nav-link active"' in html or "active" in html
```

- [ ] **Step 12: Run all tests**

Run: `pytest tests/test_web/ -v`
Expected: All tests PASS

- [ ] **Step 13: Commit scaffolding**

```bash
git add .gitignore src/fantasy_baseball/web/season_app.py \
    src/fantasy_baseball/web/season_routes.py \
    src/fantasy_baseball/web/season_data.py \
    src/fantasy_baseball/web/templates/season/base.html \
    src/fantasy_baseball/web/templates/season/standings.html \
    src/fantasy_baseball/web/templates/season/lineup.html \
    src/fantasy_baseball/web/templates/season/waivers_trades.html \
    src/fantasy_baseball/web/static/season.css \
    scripts/run_season_dashboard.py \
    tests/test_web/__init__.py \
    tests/test_web/test_season_data.py \
    tests/test_web/test_season_routes.py
git commit -m "feat: scaffold season dashboard with Flask app, sidebar layout, and cache layer"
```

---

## Task 2: Standings Page — Current View

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Modify: `src/fantasy_baseball/web/templates/season/standings.html`
- Modify: `tests/test_web/test_season_data.py`
- Modify: `tests/test_web/test_season_routes.py`

**Context:** The standings data comes from `yahoo_roster.fetch_standings()` which returns `[{"name": str, "team_key": str, "rank": int, "stats": {cat: float}}]`. The `scoring.score_roto()` function computes roto points from raw stats. Both are cached after a refresh. This task builds the Current Standings view with Roto Points / Stat Totals toggle and per-category color coding.

- [ ] **Step 1: Write test for `format_standings_for_display()`**

Add to `tests/test_web/test_season_data.py`:

```python
from fantasy_baseball.web.season_data import format_standings_for_display


def _sample_standings():
    """10-team standings data as returned by yahoo_roster.fetch_standings()."""
    teams = [
        ("Hart of the Order", {"R": 300, "HR": 90, "RBI": 290, "SB": 50, "AVG": 0.270,
                               "W": 35, "K": 600, "SV": 25, "ERA": 3.50, "WHIP": 1.18}),
        ("SkeleThor", {"R": 310, "HR": 85, "RBI": 295, "SB": 40, "AVG": 0.265,
                       "W": 38, "K": 580, "SV": 30, "ERA": 3.40, "WHIP": 1.15}),
        ("Send in the Cavalli", {"R": 280, "HR": 95, "RBI": 280, "SB": 55, "AVG": 0.260,
                                 "W": 30, "K": 620, "SV": 20, "ERA": 3.60, "WHIP": 1.22}),
    ]
    return [{"name": n, "team_key": f"key_{i}", "rank": i + 1, "stats": s}
            for i, (n, s) in enumerate(teams)]


def test_format_standings_has_roto_points():
    data = format_standings_for_display(_sample_standings(), "Hart of the Order")
    assert "teams" in data
    hart = next(t for t in data["teams"] if t["name"] == "Hart of the Order")
    assert "roto_points" in hart
    assert "total" in hart["roto_points"]


def test_format_standings_color_codes_user_team():
    data = format_standings_for_display(_sample_standings(), "Hart of the Order")
    hart = next(t for t in data["teams"] if t["name"] == "Hart of the Order")
    assert hart["is_user"] is True
    # color_classes is a dict {cat: "cat-top" | "cat-bottom" | ""}
    assert "color_classes" in hart
    # With 3 teams, ranks 1 = top, 3 = bottom
    # Hart has highest SB (50) → should be cat-top
    assert hart["color_classes"]["SB"] == "cat-top"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_data.py::test_format_standings_has_roto_points -v`
Expected: FAIL with ImportError (function doesn't exist yet)

- [ ] **Step 3: Implement `format_standings_for_display()` in `season_data.py`**

Add to `season_data.py`:

```python
from fantasy_baseball.scoring import score_roto

ALL_CATEGORIES = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
# Lower is better for these categories
INVERSE_CATS = {"ERA", "WHIP"}


def format_standings_for_display(
    standings: list[dict], user_team_name: str
) -> dict:
    """Format raw standings into display-ready data with roto points and color coding.

    Returns dict with:
      - teams: list of team dicts sorted by total roto points (desc), each containing:
          name, stats, roto_points, is_user, color_classes, rank
    """
    if not standings:
        return {"teams": []}

    # Compute roto points using existing scoring module
    all_stats = {t["name"]: t["stats"] for t in standings}
    roto = score_roto(all_stats)

    # Compute per-category ranks for color coding
    cat_ranks = _compute_category_ranks(standings)
    num_teams = len(standings)

    teams = []
    for t in standings:
        name = t["name"]
        is_user = name == user_team_name
        roto_pts = roto[name]

        # Color classes for user team only
        color_classes = {}
        if is_user:
            for cat in ALL_CATEGORIES:
                rank = cat_ranks[cat][name]
                if rank <= 3:
                    color_classes[cat] = "cat-top"
                elif rank > num_teams - 3:
                    color_classes[cat] = "cat-bottom"
                else:
                    color_classes[cat] = ""
        else:
            color_classes = {cat: "" for cat in ALL_CATEGORIES}

        teams.append({
            "name": name,
            "stats": t["stats"],
            "roto_points": roto_pts,
            "is_user": is_user,
            "color_classes": color_classes,
        })

    # Sort by total roto points descending
    teams.sort(key=lambda t: t["roto_points"]["total"], reverse=True)

    # Assign display rank
    for i, t in enumerate(teams):
        t["rank"] = i + 1

    return {"teams": teams}


def _compute_category_ranks(standings: list[dict]) -> dict[str, dict[str, int]]:
    """Compute per-category rank (1 = best) for each team."""
    ranks = {}
    for cat in ALL_CATEGORIES:
        reverse = cat not in INVERSE_CATS  # Higher is better for most cats
        sorted_teams = sorted(standings, key=lambda t: t["stats"][cat], reverse=reverse)
        ranks[cat] = {t["name"]: i + 1 for i, t in enumerate(sorted_teams)}
    return ranks
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web/test_season_data.py -v`
Expected: All tests PASS

- [ ] **Step 5: Update standings route to pass formatted data**

Modify `season_routes.py` — update the `standings()` route:

```python
    @app.route("/standings")
    def standings():
        meta = read_meta()
        raw_standings = read_cache("standings")
        standings_data = None
        if raw_standings:
            from fantasy_baseball.web.season_data import format_standings_for_display

            config = _load_config()
            standings_data = format_standings_for_display(
                raw_standings, config.team_name
            )
        return render_template(
            "season/standings.html",
            meta=meta,
            active_page="standings",
            standings=standings_data,
        )
```

Add a `_load_config()` helper at the top of `register_routes`:

```python
def register_routes(app: Flask) -> None:
    _config = None

    def _load_config():
        nonlocal _config
        if _config is None:
            from pathlib import Path as P

            from fantasy_baseball.config import load_config

            config_path = P(__file__).resolve().parents[3] / "config" / "league.yaml"
            _config = load_config(config_path)
        return _config
```

- [ ] **Step 6: Build `standings.html` template with Current view**

Replace `standings.html` with:

```html
{% extends "season/base.html" %}
{% block title %}Standings — Season Dashboard{% endblock %}
{% block content %}
<div class="page-header">
    <h2>Standings</h2>
    {% if meta and meta.get('week') %}
    <div class="week-label">Week {{ meta['week'] }}</div>
    {% endif %}
</div>

{% if not standings %}
<p class="placeholder-text">No standings data. Click "Refresh Data" to fetch from Yahoo.</p>
{% else %}

<!-- Top-level toggle: Current / Projected -->
<div class="pill-group" id="standings-top-toggle">
    <button class="pill active" data-view="current" onclick="toggleStandingsView(this)">Current</button>
    <button class="pill" data-view="projected" onclick="toggleStandingsView(this)">Projected</button>
</div>

<!-- Current standings sub-toggle -->
<div id="current-view">
    <div class="pill-group" id="current-sub-toggle">
        <button class="pill active" data-sub="points" onclick="toggleCurrentSub(this)">Roto Points</button>
        <button class="pill" data-sub="stats" onclick="toggleCurrentSub(this)">Stat Totals</button>
    </div>

    <table class="data-table" id="standings-table">
        <thead>
            <tr>
                <th>#</th>
                <th>Team</th>
                {% for cat in categories %}
                <th>{{ cat }}</th>
                {% endfor %}
                <th class="total-col">Total</th>
            </tr>
        </thead>
        <tbody>
        {% for team in standings.teams %}
            <tr class="{% if team.is_user %}user-team{% endif %}">
                <td>{{ team.rank }}</td>
                <td class="team-name">{{ team.name }}</td>
                {% for cat in categories %}
                <td class="{{ team.color_classes[cat] }} stat-cell"
                    data-points="{{ team.roto_points[cat ~ '_pts'] | round(1) }}"
                    data-stat="{{ team.stats[cat] }}">
                    {{ team.roto_points[cat ~ '_pts'] | round(1) }}
                </td>
                {% endfor %}
                <td class="total-col">{{ team.roto_points.total | round(1) }}</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
</div>

<!-- Projected standings (placeholder for Task 3) -->
<div id="projected-view" style="display: none;">
    <p class="placeholder-text">Projected standings coming soon.</p>
</div>

<script>
function toggleStandingsView(el) {
    document.querySelectorAll('#standings-top-toggle .pill').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    const view = el.dataset.view;
    document.getElementById('current-view').style.display = view === 'current' ? '' : 'none';
    document.getElementById('projected-view').style.display = view === 'projected' ? '' : 'none';
}

function toggleCurrentSub(el) {
    document.querySelectorAll('#current-sub-toggle .pill').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    const mode = el.dataset.sub;
    document.querySelectorAll('.stat-cell').forEach(cell => {
        if (mode === 'points') {
            cell.textContent = cell.dataset.points;
        } else {
            const raw = parseFloat(cell.dataset.stat);
            // Format rate stats with 3 decimal places, counting stats as integers
            cell.textContent = raw < 1 && raw > 0 ? raw.toFixed(3) :
                               raw < 10 && raw % 1 !== 0 ? raw.toFixed(2) : Math.round(raw);
        }
    });
}
</script>

{% endif %}
{% endblock %}
```

- [ ] **Step 7: Pass `categories` list to template**

In `season_routes.py`, update the standings route render call:

```python
        return render_template(
            "season/standings.html",
            meta=meta,
            active_page="standings",
            standings=standings_data,
            categories=["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"],
        )
```

- [ ] **Step 8: Write route test for standings with data**

Add to `tests/test_web/test_season_routes.py`:

```python
from unittest.mock import patch


def _mock_standings():
    teams = [
        ("Hart of the Order", {"R": 300, "HR": 90, "RBI": 290, "SB": 50, "AVG": 0.270,
                               "W": 35, "K": 600, "SV": 25, "ERA": 3.50, "WHIP": 1.18}),
        ("SkeleThor", {"R": 310, "HR": 85, "RBI": 295, "SB": 40, "AVG": 0.265,
                       "W": 38, "K": 580, "SV": 30, "ERA": 3.40, "WHIP": 1.15}),
    ]
    return [{"name": n, "team_key": f"key_{i}", "rank": i + 1, "stats": s}
            for i, (n, s) in enumerate(teams)]


def test_standings_renders_table_with_data(client):
    with patch("fantasy_baseball.web.season_routes.read_cache") as mock_cache, \
         patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
        mock_cache.side_effect = lambda k: _mock_standings() if k == "standings" else {}
        mock_cfg.return_value.team_name = "Hart of the Order"
        resp = client.get("/standings")
        assert resp.status_code == 200
        assert b"Hart of the Order" in resp.data
        assert b"user-team" in resp.data
```

- [ ] **Step 9: Run all tests**

Run: `pytest tests/test_web/ -v`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add -u && git add tests/
git commit -m "feat: standings page with current view, roto points/stat totals toggle, color coding"
```

---

## Task 3: Standings Page — Projected View

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`
- Modify: `src/fantasy_baseball/web/templates/season/standings.html`
- Modify: `tests/test_web/test_season_data.py`

**Context:** Projected standings come from three sources: static projections (cached as `projections.json`), Monte Carlo results (cached as `monte_carlo.json`), and MC + Roster Mgmt (also in `monte_carlo.json` with a separate key). The static view reuses `score_roto()`. The MC views show probability data. All are computed during refresh and cached.

- [ ] **Step 1: Write test for `format_projected_standings()`**

Add to `tests/test_web/test_season_data.py`:

```python
from fantasy_baseball.web.season_data import format_monte_carlo_for_display


def _sample_monte_carlo():
    return {
        "team_results": {
            "Hart of the Order": {
                "median_pts": 68.5, "p10": 58, "p90": 76,
                "first_pct": 18.3, "top3_pct": 52.1,
            },
            "SkeleThor": {
                "median_pts": 65.0, "p10": 55, "p90": 73,
                "first_pct": 14.7, "top3_pct": 41.8,
            },
        },
        "category_risk": {
            "R": {"median_pts": 7, "p10": 5, "p90": 9, "top3_pct": 62, "bot3_pct": 8},
            "SV": {"median_pts": 4, "p10": 2, "p90": 7, "top3_pct": 22, "bot3_pct": 38},
        },
    }


def test_format_monte_carlo_sorted_by_median():
    data = format_monte_carlo_for_display(
        _sample_monte_carlo(), "Hart of the Order"
    )
    assert data["teams"][0]["name"] == "Hart of the Order"
    assert data["teams"][0]["median_pts"] == 68.5
    assert data["teams"][0]["is_user"] is True


def test_format_monte_carlo_category_risk_colors():
    data = format_monte_carlo_for_display(
        _sample_monte_carlo(), "Hart of the Order"
    )
    risk = data["category_risk"]
    # SV has high bot3_pct (38%) → should be danger
    sv = next(r for r in risk if r["cat"] == "SV")
    assert sv["risk_class"] == "cat-bottom"
    # R has low bot3_pct (8%) → should be safe
    r_cat = next(r for r in risk if r["cat"] == "R")
    assert r_cat["risk_class"] == "cat-top"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_web/test_season_data.py::test_format_monte_carlo_sorted_by_median -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `format_monte_carlo_for_display()`**

Add to `season_data.py`:

```python
def format_monte_carlo_for_display(
    mc_data: dict, user_team_name: str
) -> dict:
    """Format Monte Carlo results for template display.

    Returns dict with:
      - teams: list sorted by median_pts desc, each with median_pts, p10, p90,
               first_pct, top3_pct, is_user
      - category_risk: list of dicts with cat, median_pts, p10, p90,
                       top3_pct, bot3_pct, risk_class
    """
    if not mc_data or "team_results" not in mc_data:
        return {"teams": [], "category_risk": []}

    teams = []
    for name, res in mc_data["team_results"].items():
        teams.append({
            "name": name,
            "median_pts": res["median_pts"],
            "p10": res["p10"],
            "p90": res["p90"],
            "first_pct": res["first_pct"],
            "top3_pct": res["top3_pct"],
            "is_user": name == user_team_name,
        })
    teams.sort(key=lambda t: t["median_pts"], reverse=True)

    # Category risk with color coding
    risk = []
    for cat, data in mc_data.get("category_risk", {}).items():
        # Green if top3_pct >= 50%, red if bot3_pct >= 30%
        if data["top3_pct"] >= 50:
            risk_class = "cat-top"
        elif data["bot3_pct"] >= 30:
            risk_class = "cat-bottom"
        else:
            risk_class = ""
        risk.append({
            "cat": cat,
            "median_pts": data["median_pts"],
            "p10": data["p10"],
            "p90": data["p90"],
            "top3_pct": data["top3_pct"],
            "bot3_pct": data["bot3_pct"],
            "risk_class": risk_class,
        })

    return {"teams": teams, "category_risk": risk}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web/test_season_data.py -v`
Expected: All tests PASS

- [ ] **Step 5: Update standings route for projected data**

In `season_routes.py`, update the `standings()` route to also load projected and MC data:

```python
    @app.route("/standings")
    def standings():
        meta = read_meta()
        raw_standings = read_cache("standings")
        config = _load_config()
        standings_data = None
        projected_data = None
        mc_data = None
        mc_mgmt_data = None

        if raw_standings:
            from fantasy_baseball.web.season_data import (
                format_standings_for_display,
                format_monte_carlo_for_display,
            )

            standings_data = format_standings_for_display(
                raw_standings, config.team_name
            )

            # Projected static uses cached projected standings
            raw_projected = read_cache("projections")
            if raw_projected and "projected_standings" in raw_projected:
                projected_data = format_standings_for_display(
                    raw_projected["projected_standings"], config.team_name
                )

            # Monte Carlo results
            raw_mc = read_cache("monte_carlo")
            if raw_mc:
                mc_data = format_monte_carlo_for_display(
                    raw_mc.get("base", raw_mc), config.team_name
                )
                if "with_management" in raw_mc:
                    mc_mgmt_data = format_monte_carlo_for_display(
                        raw_mc["with_management"], config.team_name
                    )

        return render_template(
            "season/standings.html",
            meta=meta,
            active_page="standings",
            standings=standings_data,
            projected=projected_data,
            mc=mc_data,
            mc_mgmt=mc_mgmt_data,
            categories=["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"],
        )
```

- [ ] **Step 6: Update `standings.html` with projected views**

Replace the projected-view placeholder in `standings.html`:

```html
<div id="projected-view" style="display: none;">
    <div class="pill-group" id="projected-sub-toggle">
        <button class="pill active" data-sub="static" onclick="toggleProjectedSub(this)">Static</button>
        <button class="pill" data-sub="mc" onclick="toggleProjectedSub(this)">Monte Carlo</button>
        <button class="pill" data-sub="mc-mgmt" onclick="toggleProjectedSub(this)">MC + Roster Mgmt</button>
    </div>

    <!-- Static projected standings (same format as current) -->
    <div id="proj-static">
        {% if projected %}
        <div class="pill-group" id="proj-static-sub-toggle">
            <button class="pill active" data-sub="points" onclick="toggleProjStaticSub(this)">Roto Points</button>
            <button class="pill" data-sub="stats" onclick="toggleProjStaticSub(this)">Stat Totals</button>
        </div>
        <table class="data-table">
            <thead>
                <tr>
                    <th>#</th><th>Team</th>
                    {% for cat in categories %}<th>{{ cat }}</th>{% endfor %}
                    <th class="total-col">Total</th>
                </tr>
            </thead>
            <tbody>
            {% for team in projected.teams %}
                <tr class="{% if team.is_user %}user-team{% endif %}">
                    <td>{{ team.rank }}</td>
                    <td class="team-name">{{ team.name }}</td>
                    {% for cat in categories %}
                    <td class="{{ team.color_classes[cat] }} proj-stat-cell"
                        data-points="{{ team.roto_points[cat ~ '_pts'] | round(1) }}"
                        data-stat="{{ team.stats[cat] }}">
                        {{ team.roto_points[cat ~ '_pts'] | round(1) }}
                    </td>
                    {% endfor %}
                    <td class="total-col">{{ team.roto_points.total | round(1) }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="placeholder-text">No projected standings data available.</p>
        {% endif %}
    </div>

    <!-- Monte Carlo -->
    <div id="proj-mc" style="display: none;">
        {% if mc %}
        <table class="data-table">
            <thead>
                <tr>
                    <th>Team</th><th>Median Pts</th><th>P10</th><th>P90</th>
                    <th>1st %</th><th>Top 3 %</th>
                </tr>
            </thead>
            <tbody>
            {% for team in mc.teams %}
                <tr class="{% if team.is_user %}user-team{% endif %}">
                    <td class="team-name">{{ team.name }}</td>
                    <td>{{ team.median_pts }}</td>
                    <td style="opacity: 0.6;">{{ team.p10 }}</td>
                    <td style="opacity: 0.6;">{{ team.p90 }}</td>
                    <td>{{ team.first_pct }}%</td>
                    <td>{{ team.top3_pct }}%</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>

        {% if mc.category_risk %}
        <div class="card" style="margin-top: 16px;">
            <h4 style="color: var(--warning); margin-bottom: 12px;">Category Risk — Your Team</h4>
            <table class="data-table">
                <thead>
                    <tr>
                        <th>Cat</th><th>Median Pts</th><th>P10</th><th>P90</th>
                        <th>Top 3 %</th><th>Bot 3 %</th>
                    </tr>
                </thead>
                <tbody>
                {% for r in mc.category_risk %}
                    <tr>
                        <td class="team-name" style="font-weight: bold;">{{ r.cat }}</td>
                        <td>{{ r.median_pts }}</td>
                        <td style="opacity: 0.6;">{{ r.p10 }}</td>
                        <td style="opacity: 0.6;">{{ r.p90 }}</td>
                        <td class="{{ r.risk_class }}">{{ r.top3_pct }}%</td>
                        <td class="{{ 'cat-bottom' if r.bot3_pct >= 30 else '' }}">{{ r.bot3_pct }}%</td>
                    </tr>
                {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}
        {% else %}
        <p class="placeholder-text">Run a refresh to generate Monte Carlo projections.</p>
        {% endif %}
    </div>

    <!-- MC + Roster Mgmt -->
    <div id="proj-mc-mgmt" style="display: none;">
        {% if mc_mgmt %}
        <table class="data-table">
            <thead>
                <tr>
                    <th>Team</th><th>Median Pts</th><th>P10</th><th>P90</th>
                    <th>1st %</th><th>Top 3 %</th>
                </tr>
            </thead>
            <tbody>
            {% for team in mc_mgmt.teams %}
                <tr class="{% if team.is_user %}user-team{% endif %}">
                    <td class="team-name">{{ team.name }}</td>
                    <td>{{ team.median_pts }}</td>
                    <td style="opacity: 0.6;">{{ team.p10 }}</td>
                    <td style="opacity: 0.6;">{{ team.p90 }}</td>
                    <td>{{ team.first_pct }}%</td>
                    <td>{{ team.top3_pct }}%</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
        {% else %}
        <p class="placeholder-text">Run a refresh to generate MC + Roster Mgmt projections.</p>
        {% endif %}
    </div>
</div>

<script>
/* ... existing toggleStandingsView and toggleCurrentSub ... */

function toggleProjectedSub(el) {
    document.querySelectorAll('#projected-sub-toggle .pill').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    const sub = el.dataset.sub;
    document.getElementById('proj-static').style.display = sub === 'static' ? '' : 'none';
    document.getElementById('proj-mc').style.display = sub === 'mc' ? '' : 'none';
    document.getElementById('proj-mc-mgmt').style.display = sub === 'mc-mgmt' ? '' : 'none';
}

function toggleProjStaticSub(el) {
    document.querySelectorAll('#proj-static-sub-toggle .pill').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    const mode = el.dataset.sub;
    document.querySelectorAll('.proj-stat-cell').forEach(cell => {
        if (mode === 'points') {
            cell.textContent = cell.dataset.points;
        } else {
            const raw = parseFloat(cell.dataset.stat);
            cell.textContent = raw < 1 && raw > 0 ? raw.toFixed(3) :
                               raw < 10 && raw % 1 !== 0 ? raw.toFixed(2) : Math.round(raw);
        }
    });
}
</script>
```

- [ ] **Step 7: Run all tests**

Run: `pytest tests/test_web/ -v`
Expected: All tests PASS

- [ ] **Step 8: Commit**

```bash
git add -u
git commit -m "feat: projected standings with static, Monte Carlo, and MC + roster mgmt views"
```

---

## Task 4: Lineup Page — Roster Display + Optimize

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html`
- Modify: `tests/test_web/test_season_data.py`

**Context:** Roster data comes from `yahoo_roster.fetch_roster()` cached as `roster.json`. The optimizer output is cached as `lineup_optimal.json` after running `optimize_hitter_lineup()` and `optimize_pitcher_lineup()`. Probable starters come from `probable_starters.json` which cross-references the user's pitchers with `weekly_schedule.json` matchup data.

- [ ] **Step 1: Write test for `format_lineup_for_display()`**

Add to `tests/test_web/test_season_data.py`:

```python
from fantasy_baseball.web.season_data import format_lineup_for_display


def _sample_roster():
    return [
        {"name": "Adley Rutschman", "positions": ["C"], "selected_position": "C",
         "player_id": "123", "status": ""},
        {"name": "Mike Trout", "positions": ["OF"], "selected_position": "OF",
         "player_id": "456", "status": "IL"},
        {"name": "Masataka Yoshida", "positions": ["OF", "UTIL"], "selected_position": "BN",
         "player_id": "789", "status": ""},
    ]


def _sample_optimal():
    return {
        "hitters": {"C": "Adley Rutschman", "OF": "Masataka Yoshida"},
        "pitchers": {},
        "moves": [
            {"action": "START", "player": "Masataka Yoshida", "slot": "OF", "reason": "wSGP: 1.9"},
            {"action": "BENCH", "player": "Mike Trout", "slot": "IL", "reason": "IL-eligible"},
        ],
    }


def test_format_lineup_separates_hitters_pitchers():
    data = format_lineup_for_display(_sample_roster(), _sample_optimal())
    assert "hitters" in data
    assert "pitchers" in data
    # Trout and Yoshida are hitters (OF); Rutschman is hitter (C)
    assert len(data["hitters"]) >= 2


def test_format_lineup_detects_suboptimal():
    data = format_lineup_for_display(_sample_roster(), _sample_optimal())
    assert data["is_optimal"] is False
    assert len(data["moves"]) == 2


def test_format_lineup_optimal_when_no_moves():
    optimal = {"hitters": {}, "pitchers": {}, "moves": []}
    data = format_lineup_for_display(_sample_roster(), optimal)
    assert data["is_optimal"] is True
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_web/test_season_data.py::test_format_lineup_separates_hitters_pitchers -v`
Expected: FAIL with ImportError

- [ ] **Step 3: Implement `format_lineup_for_display()`**

Add to `season_data.py`:

```python
PITCHER_POSITIONS = {"SP", "RP", "P"}
HITTER_SLOTS_ORDER = ["C", "1B", "2B", "3B", "SS", "IF", "OF", "OF", "OF", "OF",
                       "UTIL", "UTIL", "BN", "IL"]


def format_lineup_for_display(
    roster: list[dict], optimal: dict | None
) -> dict:
    """Format roster + optimizer output for the lineup template.

    Returns dict with:
      - hitters: list of player dicts sorted by slot order
      - pitchers: list of player dicts sorted by wSGP desc
      - is_optimal: bool
      - moves: list of suggested move dicts
    """
    hitters = []
    pitchers = []

    for p in roster:
        pos = p.get("selected_position", "BN")
        is_pitcher = pos in PITCHER_POSITIONS or (
            pos == "BN" and set(p.get("positions", [])).issubset(PITCHER_POSITIONS | {"BN"})
        )
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
        }
        if is_pitcher:
            pitchers.append(entry)
        else:
            hitters.append(entry)

    # Sort hitters by slot order (active first, then bench, then IL)
    slot_rank = {s: i for i, s in enumerate(HITTER_SLOTS_ORDER)}
    hitters.sort(key=lambda h: (slot_rank.get(h["selected_position"], 99), -h["wsgp"]))
    pitchers.sort(key=lambda p: (p["is_bench"], -p["wsgp"]))

    moves = optimal.get("moves", []) if optimal else []

    return {
        "hitters": hitters,
        "pitchers": pitchers,
        "is_optimal": len(moves) == 0,
        "moves": moves,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_web/test_season_data.py -v`
Expected: All tests PASS

- [ ] **Step 5: Update lineup route**

In `season_routes.py`, update the `lineup()` route:

```python
    @app.route("/lineup")
    def lineup():
        meta = read_meta()
        roster_raw = read_cache("roster")
        optimal_raw = read_cache("lineup_optimal")
        starters_raw = read_cache("probable_starters")

        lineup_data = None
        if roster_raw:
            from fantasy_baseball.web.season_data import format_lineup_for_display

            lineup_data = format_lineup_for_display(roster_raw, optimal_raw)

        return render_template(
            "season/lineup.html",
            meta=meta,
            active_page="lineup",
            lineup=lineup_data,
            starters=starters_raw,
        )
```

- [ ] **Step 6: Add optimize API endpoint**

Add to `season_routes.py` inside `register_routes()`:

```python
    @app.route("/api/optimize", methods=["POST"])
    def api_optimize():
        """Re-run the lineup optimizer and return suggested moves."""
        from fantasy_baseball.web.season_data import run_optimize

        try:
            result = run_optimize()
            return jsonify(result)
        except Exception as e:
            return jsonify({"error": str(e)}), 500
```

Add `jsonify` to the flask imports at the top of the file:

```python
from flask import Flask, jsonify, redirect, render_template, url_for
```

- [ ] **Step 7: Add `run_optimize()` stub to `season_data.py`**

```python
def run_optimize() -> dict:
    """Re-run lineup optimizer from cached data. Returns moves list.

    Full implementation will load cached roster + projections + standings,
    run optimize_hitter_lineup() and optimize_pitcher_lineup(), compare
    with current assignments, and return the diff as moves.
    """
    # Stub — full implementation in Task 6 (refresh pipeline)
    optimal = read_cache("lineup_optimal")
    if optimal:
        return {"moves": optimal.get("moves", []), "is_optimal": len(optimal.get("moves", [])) == 0}
    return {"moves": [], "is_optimal": True}
```

- [ ] **Step 8: Build `lineup.html` template**

Replace `lineup.html`:

```html
{% extends "season/base.html" %}
{% block title %}Lineup — Season Dashboard{% endblock %}
{% block content %}

<div class="page-header">
    <h2>Lineup</h2>
    {% if meta and meta.get('week') %}
    <div class="week-label">Week {{ meta['week'] }}</div>
    {% endif %}
</div>

{% if not lineup %}
<p class="placeholder-text">No roster data. Click "Refresh Data" to fetch from Yahoo.</p>
{% else %}

<!-- Hitters -->
<div class="section-header">
    <h3>Hitters</h3>
    {% if lineup.is_optimal %}
    <button class="btn-optimize optimal" disabled>Optimal ✓</button>
    {% else %}
    <button class="btn-optimize available"
            hx-post="/api/optimize" hx-target="#moves-banner" hx-swap="innerHTML">
        Optimize
    </button>
    {% endif %}
</div>

<div id="moves-banner">
{% if lineup.moves %}
<div class="moves-banner">
    <h4>Suggested Moves</h4>
    {% for move in lineup.moves %}
    <div class="move-row">
        {% if move.action == "START" %}
        <span class="badge badge-success">START</span>
        {% else %}
        <span class="badge badge-danger">{{ move.action }}</span>
        {% endif %}
        <strong>{{ move.player }}</strong>
        <span style="color: var(--text-secondary);">→ {{ move.slot }}</span>
        <span style="color: var(--text-secondary); margin-left: auto; font-size: 11px;">{{ move.reason }}</span>
    </div>
    {% endfor %}
    <div style="font-size: 11px; color: var(--text-secondary); margin-top: 8px; font-style: italic;">
        Make these moves manually in Yahoo. Refresh data after to confirm.
    </div>
</div>
{% endif %}
</div>

<table class="data-table">
    <thead>
        <tr>
            <th style="width: 50px;">Slot</th>
            <th>Player</th>
            <th>Elig</th>
            <th>Games</th>
            <th>wSGP</th>
            <th style="width: 60px;">Status</th>
        </tr>
    </thead>
    <tbody>
    {% for p in lineup.hitters %}
        <tr class="{% if p.is_bench %}bench-row{% endif %}{% if p.is_il and not p.is_bench %} il-active-row{% endif %}">
            <td style="color: var(--text-secondary);">{{ p.selected_position }}</td>
            <td style="font-weight: {% if not p.is_bench %}bold{% else %}normal{% endif %};">{{ p.name }}</td>
            <td style="font-size: 11px; color: var(--text-secondary);">{{ p.positions | join(', ') }}</td>
            <td>{{ p.games }}</td>
            <td {% if p.wsgp > 0 %}style="color: var(--success);"{% endif %}>
                {% if p.wsgp %}{{ "%.1f" | format(p.wsgp) }}{% else %}—{% endif %}
            </td>
            <td>
                {% if p.status %}
                <span class="badge badge-il">{{ p.status }}</span>
                {% endif %}
            </td>
        </tr>
    {% endfor %}
    </tbody>
</table>

<!-- Pitchers -->
<div class="section-header" style="margin-top: 24px;">
    <h3>Pitchers</h3>
</div>

<table class="data-table">
    <thead>
        <tr>
            <th style="width: 50px;">Slot</th>
            <th>Player</th>
            <th>Games</th>
            <th>wSGP</th>
            <th style="width: 60px;">Status</th>
        </tr>
    </thead>
    <tbody>
    {% for p in lineup.pitchers %}
        <tr class="{% if p.is_bench %}bench-row{% endif %}">
            <td style="color: var(--text-secondary);">{{ p.selected_position }}</td>
            <td style="font-weight: {% if not p.is_bench %}bold{% else %}normal{% endif %};">{{ p.name }}</td>
            <td>{{ p.games }}</td>
            <td {% if p.wsgp > 0 %}style="color: var(--success);"{% endif %}>
                {% if p.wsgp %}{{ "%.1f" | format(p.wsgp) }}{% else %}—{% endif %}
            </td>
            <td>
                {% if p.status %}
                <span class="badge badge-il">{{ p.status }}</span>
                {% endif %}
            </td>
        </tr>
    {% endfor %}
    </tbody>
</table>

<!-- Probable Starters -->
{% if starters %}
<div class="section-header" style="margin-top: 24px;">
    <h3>Probable Starters This Week</h3>
</div>

<table class="data-table">
    <thead>
        <tr>
            <th>Pitcher</th>
            <th>Day</th>
            <th>Opponent</th>
            <th>Matchup</th>
            <th>Starts</th>
        </tr>
    </thead>
    <tbody>
    {% for s in starters %}
        <tr class="expandable" onclick="toggleExpand(this)">
            <td style="font-weight: bold;">{{ s.name }}</td>
            <td>{{ s.days }}</td>
            <td>{{ s.opponents }}</td>
            <td>
                {% if s.matchup_quality == "Great" %}
                <span class="badge badge-success">Great</span>
                {% elif s.matchup_quality == "Tough" %}
                <span class="badge badge-danger">Tough</span>
                {% else %}
                <span class="badge badge-warning">Fair</span>
                {% endif %}
            </td>
            <td>
                {% if s.starts >= 2 %}
                <span class="badge badge-info">{{ s.starts }}-start</span>
                {% else %}
                {{ s.starts }}
                {% endif %}
            </td>
        </tr>
        <tr class="expand-content">
            <td colspan="5">
                {% for m in s.matchups %}
                <div style="margin-bottom: 6px;">
                    <strong>{{ m.date }}</strong> vs {{ m.opponent }}
                    — OPS: {{ m.opp_ops }} ({{ m.opp_ops_rank }})
                    · K%: {{ m.opp_k_pct }}% ({{ m.opp_k_rank }})
                </div>
                {% endfor %}
            </td>
        </tr>
    {% endfor %}
    </tbody>
</table>
{% endif %}

<script>
function toggleExpand(row) {
    const detail = row.nextElementSibling;
    if (detail && detail.classList.contains('expand-content')) {
        detail.classList.toggle('open');
    }
}
</script>

{% endif %}

<style>
.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 12px;
}
.section-header h3 { font-size: 16px; }
</style>
{% endblock %}
```

- [ ] **Step 9: Run all tests**

Run: `pytest tests/test_web/ -v`
Expected: All tests PASS

- [ ] **Step 10: Commit**

```bash
git add -u
git commit -m "feat: lineup page with roster tables, optimize button, and probable starters"
```

---

## Task 5: Waivers & Trades Page

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`
- Modify: `tests/test_web/test_season_routes.py`

**Context:** Waiver data is cached as `waivers.json` (list of add/drop dicts from `scan_waivers()`). Trade data is cached as `trades.json` (list of trade proposal dicts from `find_trades()` plus generated pitches). The trade standings API endpoint computes before/after roto standings for a specific trade.

- [ ] **Step 1: Update waivers-trades route**

In `season_routes.py`, update the `waivers_trades()` route:

```python
    @app.route("/waivers-trades")
    def waivers_trades():
        meta = read_meta()
        waivers_raw = read_cache("waivers")
        trades_raw = read_cache("trades")
        return render_template(
            "season/waivers_trades.html",
            meta=meta,
            active_page="waivers_trades",
            waivers=waivers_raw or [],
            trades=trades_raw or [],
            categories=["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"],
        )
```

- [ ] **Step 2: Add trade standings API endpoint**

Add to `season_routes.py`:

```python
    @app.route("/api/trade/<int:idx>/standings")
    def api_trade_standings(idx):
        """Return before/after roto standings for a specific trade."""
        trades_raw = read_cache("trades")
        if not trades_raw or idx >= len(trades_raw):
            return jsonify({"error": "Trade not found"}), 404

        trade = trades_raw[idx]
        standings_raw = read_cache("standings")
        if not standings_raw:
            return jsonify({"error": "No standings data"}), 404

        from fantasy_baseball.web.season_data import compute_trade_standings_impact

        config = _load_config()
        result = compute_trade_standings_impact(trade, standings_raw, config.team_name)
        return jsonify(result)
```

- [ ] **Step 3: Add `compute_trade_standings_impact()` to `season_data.py`**

```python
def compute_trade_standings_impact(
    trade: dict, standings: list[dict], user_team_name: str
) -> dict:
    """Compute before/after roto standings for a trade.

    Returns dict with:
      - before: {user_team: {cat: points}, opp_team: {cat: points}}
      - after: {user_team: {cat: points}, opp_team: {cat: points}}
      - before_stats: {user_team: {cat: stat}, opp_team: {cat: stat}}
      - after_stats: {user_team: {cat: stat}, opp_team: {cat: stat}}
    """
    opp_name = trade["opponent"]

    # Before: current standings roto points
    all_stats_before = {t["name"]: dict(t["stats"]) for t in standings}
    roto_before = score_roto(all_stats_before)

    # After: apply trade deltas to stats
    all_stats_after = {t["name"]: dict(t["stats"]) for t in standings}

    # Use hart_cat_deltas and opp_cat_deltas if available (stat-level deltas)
    # These represent the change in projected end-of-season stats
    if "hart_stats_after" in trade and "opp_stats_after" in trade:
        all_stats_after[user_team_name] = trade["hart_stats_after"]
        all_stats_after[opp_name] = trade["opp_stats_after"]
    else:
        # Fallback: use the cat_deltas from the trade as rough stat adjustments
        for cat in ALL_CATEGORIES:
            hart_delta = trade.get("hart_cat_deltas", {}).get(cat, 0)
            opp_delta = trade.get("opp_cat_deltas", {}).get(cat, 0)
            all_stats_after[user_team_name][cat] += hart_delta
            all_stats_after[opp_name][cat] += opp_delta

    roto_after = score_roto(all_stats_after)

    return {
        "before": {
            user_team_name: roto_before[user_team_name],
            opp_name: roto_before[opp_name],
        },
        "after": {
            user_team_name: roto_after[user_team_name],
            opp_name: roto_after[opp_name],
        },
        "before_stats": {
            user_team_name: all_stats_before[user_team_name],
            opp_name: all_stats_before[opp_name],
        },
        "after_stats": {
            user_team_name: all_stats_after[user_team_name],
            opp_name: all_stats_after[opp_name],
        },
        "categories": ALL_CATEGORIES,
    }
```

- [ ] **Step 4: Build `waivers_trades.html` template**

Replace `waivers_trades.html`:

```html
{% extends "season/base.html" %}
{% block title %}Waivers &amp; Trades — Season Dashboard{% endblock %}
{% block content %}

<div class="page-header">
    <h2>Waivers &amp; Trades</h2>
</div>

<!-- Waiver Wire -->
<h3 style="margin-bottom: 12px;">Waiver Wire</h3>

{% if not waivers %}
<p class="placeholder-text">No waiver data. Click "Refresh Data" to scan available players.</p>
{% else %}
<div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 14px;">
    Ranked by leverage-weighted SGP gain
</div>

{% for w in waivers %}
<div class="card">
    <div class="card-header">
        <div>
            <div style="font-weight: bold; font-size: 14px;">{{ w.add }}</div>
            <div style="font-size: 11px; color: var(--text-secondary); margin-top: 2px;">
                {{ w.add_positions | default('') }} — FA
            </div>
        </div>
        <div style="color: var(--success); font-weight: bold; font-size: 16px;">
            +{{ "%.1f" | format(w.sgp_gain) }} wSGP
        </div>
    </div>

    {% if w.projected_stats %}
    <div style="margin-top: 8px; font-size: 11px; color: var(--text-secondary);">
        Proj: {{ w.projected_stats }}
    </div>
    {% endif %}

    <div style="margin-top: 10px; border-top: 1px solid rgba(255,255,255,0.06); padding-top: 8px; font-size: 12px;">
        <span class="cat-gain">ADD</span> <strong>{{ w.add }}</strong>
        <span style="color: var(--text-secondary);">→</span>
        <span class="cat-loss">DROP</span> <strong>{{ w.drop }}</strong>
    </div>

    {% if w.categories %}
    <div style="margin-top: 8px; font-size: 11px; color: var(--text-secondary);">
        <div style="font-weight: bold; color: #aaa; margin-bottom: 4px;">Category Impact</div>
        <div class="cat-impact">
            {% for cat, delta in w.categories.items() %}
            {% if delta > 0 %}
            <span class="cat-gain">{{ cat }} +{{ delta }}</span>
            {% elif delta < 0 %}
            <span class="cat-loss">{{ cat }} {{ delta }}</span>
            {% endif %}
            {% endfor %}
        </div>
    </div>
    {% endif %}
</div>
{% endfor %}
{% endif %}

<!-- Trade Recommendations -->
<h3 style="margin-top: 32px; margin-bottom: 12px;">Trade Recommendations</h3>

{% if not trades %}
<p class="placeholder-text">No trade data. Click "Refresh Data" to find win-win trades.</p>
{% else %}
<div style="font-size: 13px; color: var(--text-secondary); margin-bottom: 14px;">
    Win-win 1-for-1 swaps, sorted by your wSGP gain
</div>

{% for trade in trades %}
<div class="card">
    <div class="card-header">
        <div style="font-size: 13px; color: var(--text-secondary);">
            Trade with <strong style="color: var(--text);">{{ trade.opponent }}</strong>
        </div>
        <div style="color: var(--success); font-weight: bold;">+{{ "%.1f" | format(trade.hart_wsgp_gain) }} wSGP</div>
    </div>

    <div class="trade-swap">
        <div class="trade-side trade-send">
            <div style="font-size: 11px; color: var(--danger); margin-bottom: 4px;">YOU SEND</div>
            <div style="font-weight: bold;">{{ trade.send }}</div>
            <div style="font-size: 11px; color: var(--text-secondary);">{{ trade.send_positions | join(', ') }}</div>
        </div>
        <div class="trade-arrow">⇄</div>
        <div class="trade-side trade-receive">
            <div style="font-size: 11px; color: var(--success); margin-bottom: 4px;">YOU GET</div>
            <div style="font-weight: bold;">{{ trade.receive }}</div>
            <div style="font-size: 11px; color: var(--text-secondary);">{{ trade.receive_positions | join(', ') }}</div>
        </div>
    </div>

    <div style="margin-top: 10px; font-size: 11px; color: var(--accent); cursor: pointer;"
         onclick="toggleTradeDetails(this, {{ loop.index0 }})">
        ▶ Show details &amp; pitch
    </div>

    <div class="trade-details" id="trade-details-{{ loop.index0 }}">
        <!-- Why this works -->
        <div style="font-weight: bold; color: #aaa; margin-bottom: 8px;">Why this works</div>
        <div style="display: flex; gap: 24px; font-size: 12px;">
            <div>
                <div style="color: var(--accent); margin-bottom: 4px;">You gain:</div>
                {% for cat, delta in trade.hart_cat_deltas.items() %}
                {% if delta > 0 %}
                <div class="cat-gain">{{ cat }} +{{ delta }}</div>
                {% elif delta < 0 %}
                <div class="cat-loss">{{ cat }} {{ delta }}</div>
                {% endif %}
                {% endfor %}
            </div>
            <div>
                <div style="color: var(--warning); margin-bottom: 4px;">They gain:</div>
                {% for cat, delta in trade.opp_cat_deltas.items() %}
                {% if delta > 0 %}
                <div class="cat-gain">{{ cat }} +{{ delta }}</div>
                {% elif delta < 0 %}
                <div class="cat-loss">{{ cat }} {{ delta }}</div>
                {% endif %}
                {% endfor %}
            </div>
        </div>

        <!-- Before/After Standings (loaded via AJAX) -->
        <div style="margin-top: 12px;">
            <button class="pill active" onclick="loadTradeStandings({{ loop.index0 }})">
                Load Before/After Standings
            </button>
            <div id="trade-standings-{{ loop.index0 }}" style="margin-top: 8px;"></div>
        </div>

        <!-- Pitch -->
        {% if trade.pitch %}
        <div class="trade-pitch">{{ trade.pitch }}</div>
        {% endif %}
    </div>
</div>
{% endfor %}
{% endif %}

<script>
function toggleTradeDetails(el, idx) {
    const details = document.getElementById('trade-details-' + idx);
    details.classList.toggle('open');
    el.textContent = details.classList.contains('open') ? '▼ Hide details' : '▶ Show details & pitch';
}

function loadTradeStandings(idx) {
    const container = document.getElementById('trade-standings-' + idx);
    container.innerHTML = '<span style="color: var(--text-secondary); font-size: 12px;">Loading...</span>';

    fetch('/api/trade/' + idx + '/standings')
        .then(r => r.json())
        .then(data => {
            if (data.error) {
                container.innerHTML = '<span class="cat-loss">' + data.error + '</span>';
                return;
            }
            container.innerHTML = renderTradeStandings(data);
        })
        .catch(err => {
            container.innerHTML = '<span class="cat-loss">Failed to load standings</span>';
        });
}

function renderTradeStandings(data) {
    const cats = data.categories;
    let html = '<div class="pill-group" style="margin-bottom: 8px;">';
    html += '<button class="pill active" onclick="toggleTradeStandingsView(this, \'points\')">Roto Points</button>';
    html += '<button class="pill" onclick="toggleTradeStandingsView(this, \'stats\')">Stat Totals</button>';
    html += '</div>';

    // Build table for each team (before → after)
    for (const team of Object.keys(data.before)) {
        const before = data.before[team];
        const after = data.after[team];
        const beforeStats = data.before_stats[team];
        const afterStats = data.after_stats[team];

        html += '<div style="margin-bottom: 12px;"><strong>' + esc(team) + '</strong>';
        html += '<table class="data-table" style="margin-top: 4px; font-size: 11px;">';
        html += '<thead><tr><th></th>';
        cats.forEach(c => html += '<th>' + c + '</th>');
        html += '<th class="total-col">Total</th></tr></thead><tbody>';

        html += '<tr><td style="color: var(--text-secondary);">Before</td>';
        cats.forEach(c => {
            const pts = (before[c + '_pts'] || 0).toFixed(1);
            const stat = beforeStats[c];
            html += '<td class="trade-standings-cell" data-points="' + pts + '" data-stat="' + stat + '">' + pts + '</td>';
        });
        html += '<td class="total-col">' + (before.total || 0).toFixed(1) + '</td></tr>';

        html += '<tr><td style="color: var(--text-secondary);">After</td>';
        cats.forEach(c => {
            const pts = (after[c + '_pts'] || 0).toFixed(1);
            const stat = afterStats[c];
            const diff = pts - (before[c + '_pts'] || 0);
            const cls = diff > 0 ? 'cat-gain' : diff < 0 ? 'cat-loss' : '';
            html += '<td class="trade-standings-cell ' + cls + '" data-points="' + pts + '" data-stat="' + stat + '">' + pts + '</td>';
        });
        const totalDiff = (after.total || 0) - (before.total || 0);
        const totalCls = totalDiff > 0 ? 'cat-gain' : totalDiff < 0 ? 'cat-loss' : '';
        html += '<td class="total-col ' + totalCls + '">' + (after.total || 0).toFixed(1) + '</td></tr>';

        html += '</tbody></table></div>';
    }
    return html;
}

function toggleTradeStandingsView(el, mode) {
    el.parentElement.querySelectorAll('.pill').forEach(p => p.classList.remove('active'));
    el.classList.add('active');
    // Find cells within the same trade detail container
    const container = el.closest('.trade-details');
    container.querySelectorAll('.trade-standings-cell').forEach(cell => {
        if (mode === 'points') {
            cell.textContent = cell.dataset.points;
        } else {
            const raw = parseFloat(cell.dataset.stat);
            cell.textContent = raw < 1 && raw > 0 ? raw.toFixed(3) :
                               raw < 10 && raw % 1 !== 0 ? raw.toFixed(2) : Math.round(raw);
        }
    });
}

function esc(s) {
    const d = document.createElement('div');
    d.textContent = s;
    return d.innerHTML;
}
</script>
{% endblock %}
```

- [ ] **Step 5: Write route test for trade standings endpoint**

Add to `tests/test_web/test_season_routes.py`:

```python
def test_trade_standings_returns_404_without_data(client):
    resp = client.get("/api/trade/0/standings")
    assert resp.status_code == 404
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/test_web/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add -u
git commit -m "feat: waivers & trades page with waiver cards, trade cards, expandable standings"
```

---

## Task 6: Data Refresh Pipeline

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Modify: `tests/test_web/test_season_data.py`
- Modify: `tests/test_web/test_season_routes.py`

**Context:** The refresh pipeline connects to Yahoo, fetches all data, runs computations (projections, leverage, optimization, simulation, waivers, trades), and writes results to cache files. This is the most complex task as it orchestrates many existing modules. The refresh runs in a background thread; the frontend polls `/api/refresh-status`.

- [ ] **Step 1: Add refresh state management to `season_data.py`**

```python
import threading
from datetime import datetime

_refresh_lock = threading.Lock()
_refresh_status = {"running": False, "progress": "", "error": None}


def get_refresh_status() -> dict:
    with _refresh_lock:
        return dict(_refresh_status)


def _set_refresh_progress(msg: str) -> None:
    with _refresh_lock:
        _refresh_status["progress"] = msg
```

- [ ] **Step 2: Implement `run_full_refresh()` in `season_data.py`**

```python
def run_full_refresh(cache_dir: Path = CACHE_DIR) -> None:
    """Run the full data refresh pipeline.

    Steps:
    1. Authenticate with Yahoo
    2. Fetch standings, roster, free agents
    3. Blend projections, match to roster
    4. Calculate leverage, run optimizer
    5. Build probable starters with matchup quality
    6. Scan waivers
    7. Find trades
    8. Run Monte Carlo simulation
    9. Write all cache files + meta
    """
    import sys
    from pathlib import Path as P

    import numpy as np
    import pandas as pd

    project_root = P(__file__).resolve().parents[3]
    if str(project_root / "src") not in sys.path:
        sys.path.insert(0, str(project_root / "src"))

    from fantasy_baseball.auth.yahoo_auth import get_league, get_yahoo_session
    from fantasy_baseball.config import load_config
    from fantasy_baseball.data.mlb_schedule import get_week_schedule
    from fantasy_baseball.data.projections import blend_projections
    from fantasy_baseball.lineup.leverage import calculate_leverage
    from fantasy_baseball.lineup.matchups import (
        calculate_matchup_factors,
        get_team_batting_stats,
    )
    from fantasy_baseball.lineup.optimizer import (
        optimize_hitter_lineup,
        optimize_pitcher_lineup,
    )
    from fantasy_baseball.lineup.waivers import (
        fetch_and_match_free_agents,
        scan_waivers,
    )
    from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
    from fantasy_baseball.lineup.yahoo_roster import (
        fetch_roster,
        fetch_standings,
        fetch_scoring_period,
    )
    from fantasy_baseball.scoring import project_team_stats, score_roto
    from fantasy_baseball.trades.evaluate import find_trades
    from fantasy_baseball.trades.pitch import generate_pitch

    config = load_config(project_root / "config" / "league.yaml")
    projections_dir = project_root / "data" / "projections"

    with _refresh_lock:
        _refresh_status["running"] = True
        _refresh_status["error"] = None

    try:
        # 1. Yahoo auth
        _set_refresh_progress("Connecting to Yahoo...")
        session = get_yahoo_session()
        league = get_league(session, config.league_id, config.game_code)
        team_key = None
        for t in league.teams():
            if t["name"] == config.team_name:
                team_key = t["team_key"]
                break
        if not team_key:
            raise ValueError(f"Team '{config.team_name}' not found in league")

        # 2. Fetch standings + roster
        _set_refresh_progress("Fetching standings...")
        standings = fetch_standings(league)
        write_cache("standings", standings, cache_dir)

        _set_refresh_progress("Fetching roster...")
        roster = fetch_roster(league, team_key)
        write_cache("roster", roster, cache_dir)

        scoring_start, scoring_end = fetch_scoring_period(league)

        # 3. Blend projections
        _set_refresh_progress("Blending projections...")
        season_year = config.season_year
        hitters_proj, pitchers_proj = blend_projections(
            projections_dir / str(season_year),
            config.projection_systems,
            config.projection_weights,
        )

        # 4. Calculate leverage
        _set_refresh_progress("Calculating leverage weights...")
        leverage = calculate_leverage(standings, config.team_name)

        # Match roster to projections for wSGP calculation
        from fantasy_baseball.utils.name_utils import normalize_name

        hitters_proj["_name_norm"] = hitters_proj["name"].apply(normalize_name)
        pitchers_proj["_name_norm"] = pitchers_proj["name"].apply(normalize_name)

        roster_enriched = []
        for p in roster:
            norm = normalize_name(p["name"])
            match = hitters_proj[hitters_proj["_name_norm"] == norm]
            if match.empty:
                match = pitchers_proj[pitchers_proj["_name_norm"] == norm]
            if not match.empty:
                row = match.iloc[0]
                wsgp = calculate_weighted_sgp(row, leverage)
                p["wsgp"] = round(wsgp, 1)
                # Copy stat columns for optimizer
                for col in row.index:
                    if col not in ("_name_norm",):
                        p[col] = row[col]
            roster_enriched.append(p)

        # 5. Run optimizer
        _set_refresh_progress("Optimizing lineup...")
        hitter_roster = [
            pd.Series(p) for p in roster_enriched
            if p.get("player_type") == "hitter"
        ]
        pitcher_roster = [
            pd.Series(p) for p in roster_enriched
            if p.get("player_type") == "pitcher"
        ]

        optimal_hitters = {}
        optimal_pitchers = {}
        moves = []

        if hitter_roster:
            optimal_hitters = optimize_hitter_lineup(hitter_roster, leverage)
            # Compare with current assignments to find moves
            for slot, player_name in optimal_hitters.items():
                current_holder = next(
                    (p for p in roster if p.get("selected_position") == slot), None
                )
                if current_holder and current_holder["name"] != player_name:
                    moves.append({
                        "action": "START",
                        "player": player_name,
                        "slot": slot,
                        "reason": f"wSGP: {next((p['wsgp'] for p in roster_enriched if p['name'] == player_name), '?')}",
                    })

        write_cache("lineup_optimal", {
            "hitters": optimal_hitters,
            "pitchers": optimal_pitchers,
            "moves": moves,
        }, cache_dir)

        # 6. Probable starters
        _set_refresh_progress("Loading schedule...")
        schedule_cache = project_root / "data" / "weekly_schedule.json"
        schedule = get_week_schedule(scoring_start, scoring_end, schedule_cache)
        batting_stats_path = project_root / "data" / "team_batting_stats.json"
        matchup_factors = {}
        if batting_stats_path.exists():
            team_stats = get_team_batting_stats(batting_stats_path)
            matchup_factors = calculate_matchup_factors(team_stats)

        starters_data = _build_probable_starters(
            pitcher_roster, schedule, matchup_factors, team_stats if batting_stats_path.exists() else {}
        )
        write_cache("probable_starters", starters_data, cache_dir)

        # 7. Waivers
        _set_refresh_progress("Scanning waivers...")
        fa_players, _ = fetch_and_match_free_agents(
            league, hitters_proj, pitchers_proj
        )
        roster_series = [pd.Series(p) for p in roster_enriched]
        waiver_results = scan_waivers(roster_series, fa_players, leverage)
        write_cache("waivers", waiver_results, cache_dir)

        # 8. Trades
        _set_refresh_progress("Finding trades...")
        opp_rosters = {}
        leverage_by_team = {config.team_name: leverage}
        for t in league.teams():
            if t["name"] != config.team_name:
                opp_key = t["team_key"]
                opp_roster = fetch_roster(league, opp_key)
                opp_rosters[t["name"]] = opp_roster
                opp_leverage = calculate_leverage(standings, t["name"])
                leverage_by_team[t["name"]] = opp_leverage

        trade_results = find_trades(
            config.team_name, roster_enriched, opp_rosters,
            standings, leverage_by_team, config.roster_slots,
        )
        # Add pitches
        for trade in trade_results:
            trade["pitch"] = generate_pitch(trade, standings, config.team_name)

        write_cache("trades", trade_results, cache_dir)

        # 9. Monte Carlo
        _set_refresh_progress("Running Monte Carlo (1000 iterations)...")
        mc_data = _run_monte_carlo(
            standings, config, opp_rosters, roster_enriched,
            hitters_proj, pitchers_proj, config.team_name,
        )
        write_cache("monte_carlo", mc_data, cache_dir)

        # 10. Write meta
        now = datetime.now()
        write_cache("meta", {
            "last_refresh": now.strftime("%-I:%M %p").lstrip("0") if not sys.platform.startswith("win") else now.strftime("%#I:%M %p"),
            "last_refresh_iso": now.isoformat(),
            "week": scoring_start,
            "scoring_period": {"start": scoring_start, "end": scoring_end},
        }, cache_dir)

        _set_refresh_progress("Done!")

    except Exception as e:
        with _refresh_lock:
            _refresh_status["error"] = str(e)
        raise
    finally:
        with _refresh_lock:
            _refresh_status["running"] = False


def _run_monte_carlo(
    standings: list[dict],
    config,
    opp_rosters: dict,
    user_roster: list[dict],
    hitters_proj,
    pitchers_proj,
    user_team_name: str,
    n_iterations: int = 1000,
) -> dict:
    """Run Monte Carlo simulation and return formatted results.

    Assembles all team rosters with projections, runs simulate_season()
    n_iterations times, and computes probability distributions.
    """
    import numpy as np

    from fantasy_baseball.scoring import score_roto
    from fantasy_baseball.simulation import simulate_season
    from fantasy_baseball.utils.name_utils import normalize_name

    # Build team_rosters dict: {team_name: [player_dicts]}
    # User roster is already enriched; opponent rosters need projection matching
    team_rosters = {user_team_name: user_roster}
    h_norm = {normalize_name(n): i for i, n in enumerate(hitters_proj["name"])}
    p_norm = {normalize_name(n): i for i, n in enumerate(pitchers_proj["name"])}

    for opp_name, opp_roster in opp_rosters.items():
        enriched = []
        for p in opp_roster:
            norm = normalize_name(p["name"])
            if norm in h_norm:
                row = hitters_proj.iloc[h_norm[norm]]
                for col in row.index:
                    if col != "_name_norm":
                        p[col] = row[col]
            elif norm in p_norm:
                row = pitchers_proj.iloc[p_norm[norm]]
                for col in row.index:
                    if col != "_name_norm":
                        p[col] = row[col]
            enriched.append(p)
        team_rosters[opp_name] = enriched

    h_slots = sum(v for k, v in config.roster_slots.items() if k not in ("P", "BN", "IL"))
    p_slots = config.roster_slots.get("P", 9)

    rng = np.random.default_rng(42)
    totals = {name: [] for name in team_rosters}
    cat_points = {cat: [] for cat in ALL_CATEGORIES}

    for _ in range(n_iterations):
        sim_stats, _ = simulate_season(team_rosters, rng, h_slots, p_slots)
        roto = score_roto(sim_stats)
        for name in team_rosters:
            totals[name].append(roto[name]["total"])
        # Track user's per-category points
        for cat in ALL_CATEGORIES:
            cat_points[cat].append(roto[user_team_name][f"{cat}_pts"])

    # Compute team-level results
    team_results = {}
    for name, pts_list in totals.items():
        arr = np.array(pts_list)
        team_results[name] = {
            "median_pts": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10)), 0),
            "p90": round(float(np.percentile(arr, 90)), 0),
            "first_pct": round(float(np.mean(arr == np.max(np.column_stack(
                [np.array(totals[n]) for n in totals]
            ), axis=1))) * 100, 1) if len(totals) > 1 else 100.0,
            "top3_pct": 0.0,  # Computed below
        }

    # Compute rank distributions
    all_totals = np.column_stack([np.array(totals[n]) for n in totals])
    team_names = list(totals.keys())
    for i_iter in range(n_iterations):
        ranks = np.argsort(-all_totals[i_iter]) + 1
        for j, name in enumerate(team_names):
            rank = int(ranks[j])
            if rank == 1:
                team_results[name]["first_pct"] = team_results[name].get("first_pct", 0)
            if rank <= 3:
                team_results[name]["top3_pct"] += 1
    for name in team_results:
        team_results[name]["first_pct"] = round(
            sum(1 for i in range(n_iterations)
                if np.argmax(all_totals[i]) == team_names.index(name))
            / n_iterations * 100, 1
        )
        team_results[name]["top3_pct"] = round(
            team_results[name]["top3_pct"] / n_iterations * 100, 1
        )

    # Category risk for user team
    num_teams = len(team_rosters)
    category_risk = {}
    for cat in ALL_CATEGORIES:
        arr = np.array(cat_points[cat])
        category_risk[cat] = {
            "median_pts": round(float(np.median(arr)), 1),
            "p10": round(float(np.percentile(arr, 10)), 1),
            "p90": round(float(np.percentile(arr, 90)), 1),
            "top3_pct": round(float(np.mean(arr >= num_teams - 2)) * 100, 1),
            "bot3_pct": round(float(np.mean(arr <= 3)) * 100, 1),
        }

    return {
        "base": {"team_results": team_results, "category_risk": category_risk},
        # TODO: "with_management" key for MC + Roster Mgmt variant
    }


def _build_probable_starters(
    pitcher_roster: list, schedule: dict, matchup_factors: dict, team_stats: dict
) -> list[dict]:
    """Build probable starters data for the template."""
    from fantasy_baseball.utils.name_utils import normalize_name

    probable = schedule.get("probable_pitchers", [])
    games_per_team = schedule.get("games_per_team", {})

    # Build lookup of pitcher names on roster
    roster_names = {normalize_name(p.get("name", "")): p for p in pitcher_roster}

    starters = {}
    for game in probable:
        for side in ("away_pitcher", "home_pitcher"):
            pitcher_name = game.get(side, "TBD")
            if pitcher_name == "TBD":
                continue
            norm = normalize_name(pitcher_name)
            if norm not in roster_names:
                continue

            opp_team = game["home_team"] if side == "away_pitcher" else game["away_team"]
            prefix = "@" if side == "away_pitcher" else "vs"
            matchup_str = f"{prefix} {opp_team}"

            # Matchup quality
            opp_stats = team_stats.get(opp_team, {})
            opp_ops = opp_stats.get("ops", 0.730)
            opp_k_pct = opp_stats.get("k_pct", 0.22)

            # Rank OPS across all teams
            all_ops = sorted([s.get("ops", 0.730) for s in team_stats.values()])
            ops_rank = len(all_ops) - sorted(all_ops).index(opp_ops) if opp_ops in all_ops else 15

            all_k = sorted([s.get("k_pct", 0.22) for s in team_stats.values()], reverse=True)
            k_rank = all_k.index(opp_k_pct) + 1 if opp_k_pct in all_k else 15

            if opp_ops < 0.700:
                quality = "Great"
            elif opp_ops > 0.750:
                quality = "Tough"
            else:
                quality = "Fair"

            matchup_detail = {
                "date": game.get("date", ""),
                "opponent": matchup_str,
                "opp_ops": f"{opp_ops:.3f}",
                "opp_ops_rank": f"{ops_rank}th",
                "opp_k_pct": f"{opp_k_pct * 100:.1f}" if opp_k_pct < 1 else f"{opp_k_pct:.1f}",
                "opp_k_rank": f"{k_rank}th",
                "quality": quality,
            }

            if pitcher_name not in starters:
                starters[pitcher_name] = {
                    "name": pitcher_name,
                    "matchups": [],
                    "days": [],
                    "opponents": [],
                }

            starters[pitcher_name]["matchups"].append(matchup_detail)
            starters[pitcher_name]["days"].append(game.get("date", ""))
            starters[pitcher_name]["opponents"].append(matchup_str)

    # Finalize
    result = []
    for name, data in starters.items():
        # Overall matchup quality = worst individual matchup
        qualities = [m["quality"] for m in data["matchups"]]
        if "Tough" in qualities:
            overall = "Tough"
        elif "Fair" in qualities:
            overall = "Fair"
        else:
            overall = "Great"

        result.append({
            "name": name,
            "days": ", ".join(data["days"]),
            "opponents": ", ".join(data["opponents"]),
            "matchup_quality": overall,
            "starts": len(data["matchups"]),
            "matchups": data["matchups"],
        })

    result.sort(key=lambda s: (-s["starts"], s["name"]))
    return result
```

- [ ] **Step 3: Add refresh API endpoints to `season_routes.py`**

```python
    import threading

    @app.route("/api/refresh", methods=["POST"])
    def api_refresh():
        """Trigger a full data refresh in the background."""
        from fantasy_baseball.web.season_data import get_refresh_status, run_full_refresh

        status = get_refresh_status()
        if status["running"]:
            return jsonify({"status": "already_running"})

        thread = threading.Thread(target=run_full_refresh, daemon=True)
        thread.start()
        return jsonify({"status": "started"})

    @app.route("/api/refresh-status")
    def api_refresh_status():
        from fantasy_baseball.web.season_data import get_refresh_status

        return jsonify(get_refresh_status())
```

- [ ] **Step 4: Update `base.html` refresh button to poll for status**

Replace the refresh button section in `base.html`:

```html
            <div class="sidebar-footer">
                <div class="refresh-info" id="refresh-info">
                    {% if meta and meta.get('last_refresh') %}
                        Last refresh: {{ meta['last_refresh'] }}
                    {% else %}
                        No data yet
                    {% endif %}
                </div>
                <button class="btn-refresh" id="refresh-btn"
                        onclick="startRefresh()">
                    Refresh Data
                </button>
                <div id="refresh-progress" style="display: none; font-size: 11px; color: var(--accent); margin-top: 6px;"></div>
            </div>
```

Add script to base.html before `</body>`:

```html
<script>
function startRefresh() {
    const btn = document.getElementById('refresh-btn');
    const progress = document.getElementById('refresh-progress');
    btn.disabled = true;
    btn.textContent = 'Refreshing...';
    progress.style.display = 'block';

    fetch('/api/refresh', {method: 'POST'})
        .then(r => r.json())
        .then(() => pollRefreshStatus());
}

function pollRefreshStatus() {
    const progress = document.getElementById('refresh-progress');
    const btn = document.getElementById('refresh-btn');

    fetch('/api/refresh-status')
        .then(r => r.json())
        .then(data => {
            progress.textContent = data.progress || 'Working...';
            if (data.running) {
                setTimeout(pollRefreshStatus, 1000);
            } else if (data.error) {
                progress.textContent = 'Error: ' + data.error;
                progress.style.color = 'var(--danger)';
                btn.disabled = false;
                btn.textContent = 'Refresh Data';
            } else {
                window.location.reload();
            }
        })
        .catch(() => setTimeout(pollRefreshStatus, 2000));
}
</script>
```

- [ ] **Step 5: Write refresh status test**

Add to `tests/test_web/test_season_routes.py`:

```python
def test_refresh_status_not_running(client):
    resp = client.get("/api/refresh-status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["running"] is False
```

- [ ] **Step 6: Run all tests**

Run: `pytest tests/test_web/ -v`
Expected: All tests PASS

- [ ] **Step 7: Commit**

```bash
git add -u
git commit -m "feat: data refresh pipeline with Yahoo integration, optimizer, waivers, trades"
```

---

## Task 7: Launch Script + Integration Test

**Files:**
- Modify: `scripts/run_season_dashboard.py` (already created in Task 1)
- Modify: `tests/test_web/test_season_routes.py`

- [ ] **Step 1: Write integration test — full page render with mock cache data**

Add to `tests/test_web/test_season_routes.py`:

```python
import json


def test_full_standings_page_with_cached_data(client, tmp_path):
    """Integration test: standings page renders correctly with all cached data present."""
    from fantasy_baseball.web import season_data

    old_cache_dir = season_data.CACHE_DIR
    season_data.CACHE_DIR = tmp_path

    try:
        standings = [
            {"name": "Hart of the Order", "team_key": "k1", "rank": 1,
             "stats": {"R": 300, "HR": 90, "RBI": 290, "SB": 50, "AVG": 0.270,
                       "W": 35, "K": 600, "SV": 25, "ERA": 3.50, "WHIP": 1.18}},
            {"name": "SkeleThor", "team_key": "k2", "rank": 2,
             "stats": {"R": 310, "HR": 85, "RBI": 295, "SB": 40, "AVG": 0.265,
                       "W": 38, "K": 580, "SV": 30, "ERA": 3.40, "WHIP": 1.15}},
        ]
        season_data.write_cache("standings", standings, tmp_path)
        season_data.write_cache("meta", {"last_refresh": "8:32 AM", "week": "3"}, tmp_path)

        with patch("fantasy_baseball.web.season_routes.read_cache") as mock_rc, \
             patch("fantasy_baseball.web.season_routes.read_meta") as mock_rm, \
             patch("fantasy_baseball.web.season_routes._load_config") as mock_cfg:
            mock_rc.side_effect = lambda k: season_data.read_cache(k, tmp_path)
            mock_rm.return_value = season_data.read_meta(tmp_path)
            mock_cfg.return_value.team_name = "Hart of the Order"

            resp = client.get("/standings")
            assert resp.status_code == 200
            html = resp.data.decode()
            assert "Hart of the Order" in html
            assert "SkeleThor" in html
            assert "8:32 AM" in html
    finally:
        season_data.CACHE_DIR = old_cache_dir
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/test_web/ -v`
Expected: All tests PASS

- [ ] **Step 3: Verify the dashboard starts**

Run: `python scripts/run_season_dashboard.py`
Expected: Flask dev server starts on port 5001, prints URL. Ctrl+C to stop.

- [ ] **Step 4: Final commit**

```bash
git add -u
git commit -m "feat: integration tests and verified dashboard launch"
```

---

## Summary

| Task | Description | Key Files |
|------|-------------|-----------|
| 1 | Project scaffolding | season_app.py, season_routes.py, season_data.py, base.html, season.css |
| 2 | Current standings | format_standings_for_display(), standings.html, color coding |
| 3 | Projected standings | format_monte_carlo_for_display(), MC/static/mgmt views |
| 4 | Lineup + optimize | format_lineup_for_display(), lineup.html, probable starters |
| 5 | Waivers & trades | waivers_trades.html, trade standings API, expandable cards |
| 6 | Refresh pipeline | run_full_refresh(), Yahoo integration, background thread |
| 7 | Integration test | Full render test, launch verification |
