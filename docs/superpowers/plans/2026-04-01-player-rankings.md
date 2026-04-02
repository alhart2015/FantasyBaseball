# Player Rankings Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ordinal SGP rankings (ROS, preseason, current) to every player-facing surface in the dashboard.

**Architecture:** New `compute_sgp_rankings()` function computes rankings from projection DataFrames and game log totals. Rankings are cached as `rankings.json` during refresh and attached to all player data structures by normalized name lookup. Templates show ROS rank badge with tooltip for all three ranks.

**Tech Stack:** Python, pandas, SQLite, Jinja2, vanilla JS/CSS

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/sgp/rankings.py` | Create | `compute_sgp_rankings()` — core ranking logic |
| `src/fantasy_baseball/web/season_data.py` | Modify | Compute & cache rankings during refresh, attach to player data |
| `src/fantasy_baseball/web/season_routes.py` | Modify | Attach ranks in player search API |
| `src/fantasy_baseball/web/templates/season/lineup.html` | Modify | Show rank badge + tooltip |
| `src/fantasy_baseball/web/templates/season/waivers_trades.html` | Modify | Show rank badge + tooltip |
| `src/fantasy_baseball/web/templates/season/players.html` | Modify | Show rank in search results |
| `src/fantasy_baseball/web/static/season.css` | Modify | Rank badge + tooltip styles |
| `tests/test_sgp/test_rankings.py` | Create | Tests for ranking computation |

---

### Task 1: Core ranking function

**Files:**
- Create: `src/fantasy_baseball/sgp/rankings.py`
- Create: `tests/test_sgp/test_rankings.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_sgp/test_rankings.py`:

```python
import pytest
import pandas as pd
from fantasy_baseball.sgp.rankings import compute_sgp_rankings


class TestComputeSgpRankings:
    def _make_hitters_df(self):
        return pd.DataFrame([
            {"name": "Aaron Judge", "player_type": "hitter", "r": 100, "hr": 40, "rbi": 100, "sb": 5, "h": 160, "ab": 550, "avg": 0.291, "pa": 650},
            {"name": "Juan Soto", "player_type": "hitter", "r": 110, "hr": 35, "rbi": 90, "sb": 10, "h": 155, "ab": 540, "avg": 0.287, "pa": 680},
            {"name": "Marcus Semien", "player_type": "hitter", "r": 80, "hr": 20, "rbi": 70, "sb": 12, "h": 140, "ab": 600, "avg": 0.233, "pa": 660},
        ])

    def _make_pitchers_df(self):
        return pd.DataFrame([
            {"name": "Gerrit Cole", "player_type": "pitcher", "w": 15, "k": 220, "sv": 0, "ip": 200, "era": 2.80, "whip": 0.95, "er": 62, "bb": 40, "h_allowed": 150},
            {"name": "Emmanuel Clase", "player_type": "pitcher", "w": 3, "k": 70, "sv": 40, "ip": 70, "era": 2.50, "whip": 0.90, "er": 19, "bb": 15, "h_allowed": 48},
        ])

    def test_returns_dict_keyed_by_normalized_name(self):
        from fantasy_baseball.utils.name_utils import normalize_name
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        assert normalize_name("Aaron Judge") in rankings
        assert normalize_name("Gerrit Cole") in rankings

    def test_hitters_ranked_separately_from_pitchers(self):
        from fantasy_baseball.utils.name_utils import normalize_name
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        # Best hitter should be rank 1, best pitcher should also be rank 1
        hitter_ranks = [rankings[normalize_name(n)] for n in ["Aaron Judge", "Juan Soto", "Marcus Semien"]]
        pitcher_ranks = [rankings[normalize_name(n)] for n in ["Gerrit Cole", "Emmanuel Clase"]]
        assert 1 in hitter_ranks
        assert 1 in pitcher_ranks

    def test_ranks_are_ordinal_1_based(self):
        from fantasy_baseball.utils.name_utils import normalize_name
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        hitter_ranks = sorted([rankings[normalize_name(n)] for n in ["Aaron Judge", "Juan Soto", "Marcus Semien"]])
        assert hitter_ranks == [1, 2, 3]

    def test_higher_sgp_gets_lower_rank_number(self):
        from fantasy_baseball.utils.name_utils import normalize_name
        rankings = compute_sgp_rankings(self._make_hitters_df(), self._make_pitchers_df())
        # Judge has more HR+RBI power than Semien, should rank higher (lower number)
        assert rankings[normalize_name("Aaron Judge")] < rankings[normalize_name("Marcus Semien")]

    def test_empty_dataframes_return_empty_dict(self):
        rankings = compute_sgp_rankings(pd.DataFrame(), pd.DataFrame())
        assert rankings == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sgp/test_rankings.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement compute_sgp_rankings**

Create `src/fantasy_baseball/sgp/rankings.py`:

```python
"""Compute ordinal SGP rankings across the full player pool."""

import pandas as pd
from fantasy_baseball.sgp.player_value import calculate_player_sgp
from fantasy_baseball.utils.name_utils import normalize_name


def compute_sgp_rankings(
    hitters: pd.DataFrame,
    pitchers: pd.DataFrame,
) -> dict[str, int]:
    """Rank all players by unweighted SGP within hitter/pitcher pools.

    Returns {normalized_name: rank} where rank is 1-based ordinal
    (1 = highest SGP in that pool).
    """
    rankings = {}

    for df in [hitters, pitchers]:
        if df.empty:
            continue

        sgp_list = []
        for _, row in df.iterrows():
            sgp = calculate_player_sgp(row)
            sgp_list.append((normalize_name(row["name"]), sgp))

        sgp_list.sort(key=lambda x: x[1], reverse=True)

        for rank, (norm_name, _sgp) in enumerate(sgp_list, start=1):
            rankings[norm_name] = rank

    return rankings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sgp/test_rankings.py -v`
Expected: All 5 pass

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/sgp/rankings.py tests/test_sgp/test_rankings.py
git commit -m "feat: add compute_sgp_rankings() for ordinal SGP ranking"
```

---

### Task 2: Compute rankings for game log actuals

**Files:**
- Modify: `src/fantasy_baseball/sgp/rankings.py`
- Modify: `tests/test_sgp/test_rankings.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_sgp/test_rankings.py`:

```python
class TestRankingsFromGameLogs:
    def test_ranks_from_game_log_totals(self):
        from fantasy_baseball.sgp.rankings import compute_rankings_from_game_logs

        hitter_logs = {
            "aaron judge": {"pa": 100, "ab": 80, "h": 25, "r": 15, "hr": 8, "rbi": 20, "sb": 1},
            "juan soto": {"pa": 110, "ab": 90, "h": 30, "r": 18, "hr": 6, "rbi": 15, "sb": 3},
        }
        pitcher_logs = {
            "gerrit cole": {"ip": 30, "k": 35, "w": 3, "sv": 0, "er": 8, "bb": 5, "h_allowed": 20},
        }
        rankings = compute_rankings_from_game_logs(hitter_logs, pitcher_logs)
        assert "aaron judge" in rankings
        assert "gerrit cole" in rankings
        assert rankings["aaron judge"] in (1, 2)
        assert rankings["gerrit cole"] == 1

    def test_empty_logs_return_empty_dict(self):
        from fantasy_baseball.sgp.rankings import compute_rankings_from_game_logs
        assert compute_rankings_from_game_logs({}, {}) == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_sgp/test_rankings.py::TestRankingsFromGameLogs -v`
Expected: FAIL — function not found

- [ ] **Step 3: Implement compute_rankings_from_game_logs**

Add to `src/fantasy_baseball/sgp/rankings.py`:

```python
def compute_rankings_from_game_logs(
    hitter_logs: dict[str, dict],
    pitcher_logs: dict[str, dict],
) -> dict[str, int]:
    """Rank players by SGP of actual accumulated stats from game logs.

    Args:
        hitter_logs: {normalized_name: {pa, ab, h, r, hr, rbi, sb}}
        pitcher_logs: {normalized_name: {ip, k, w, sv, er, bb, h_allowed}}

    Returns {normalized_name: rank} where rank is 1-based ordinal.
    """
    rankings = {}

    for logs, player_type in [(hitter_logs, "hitter"), (pitcher_logs, "pitcher")]:
        if not logs:
            continue

        sgp_list = []
        for norm_name, stats in logs.items():
            player_dict = dict(stats)
            player_dict["player_type"] = player_type
            # Compute rate stats from components for SGP calculation
            if player_type == "hitter":
                ab = player_dict.get("ab", 0) or 0
                h = player_dict.get("h", 0) or 0
                player_dict["avg"] = h / ab if ab > 0 else 0.0
            else:
                ip = player_dict.get("ip", 0) or 0
                if ip > 0:
                    er = player_dict.get("er", 0) or 0
                    bb = player_dict.get("bb", 0) or 0
                    ha = player_dict.get("h_allowed", 0) or 0
                    player_dict["era"] = er * 9.0 / ip
                    player_dict["whip"] = (bb + ha) / ip
                else:
                    player_dict["era"] = 0.0
                    player_dict["whip"] = 0.0

            sgp = calculate_player_sgp(pd.Series(player_dict))
            sgp_list.append((norm_name, sgp))

        sgp_list.sort(key=lambda x: x[1], reverse=True)

        for rank, (name, _sgp) in enumerate(sgp_list, start=1):
            rankings[name] = rank

    return rankings
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_sgp/test_rankings.py -v`
Expected: All 7 pass

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/sgp/rankings.py tests/test_sgp/test_rankings.py
git commit -m "feat: add compute_rankings_from_game_logs() for actual stats ranking"
```

---

### Task 3: Compute and cache rankings during refresh

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`

This task wires the ranking functions into the refresh pipeline. No new tests — the ranking functions are already tested. Verified by running the full suite.

- [ ] **Step 1: Add "rankings" to CACHE_FILES**

In `src/fantasy_baseball/web/season_data.py`, add to the `CACHE_FILES` dict:

```python
    "rankings": "rankings.json",
```

- [ ] **Step 2: Add ranking computation after Step 6c (pace)**

After the `write_cache("roster", ...)` call (after Step 6c), insert a new step:

```python
        # --- Step 6d: Compute SGP rankings ---
        _progress("Computing SGP rankings...")
        from fantasy_baseball.sgp.rankings import compute_sgp_rankings, compute_rankings_from_game_logs

        ros_ranks = compute_sgp_rankings(hitters_proj, pitchers_proj)
        preseason_ranks = compute_sgp_rankings(preseason_hitters, preseason_pitchers)
        current_ranks = compute_rankings_from_game_logs(hitter_logs, pitcher_logs)

        # Build combined lookup: {normalized_name: {ros, preseason, current}}
        all_names = set(ros_ranks) | set(preseason_ranks) | set(current_ranks)
        rankings_lookup = {}
        for norm in all_names:
            rankings_lookup[norm] = {
                "ros": ros_ranks.get(norm),
                "preseason": preseason_ranks.get(norm),
                "current": current_ranks.get(norm),
            }

        write_cache("rankings", rankings_lookup, cache_dir)
        _progress(f"Ranked {len(ros_ranks)} ROS, {len(preseason_ranks)} preseason, {len(current_ranks)} current")
```

Note: `hitters_proj` and `pitchers_proj` are the ROS projections (swapped in Step 4). `preseason_hitters` and `preseason_pitchers` are saved from before the swap. `hitter_logs` and `pitcher_logs` are from Step 6c.

- [ ] **Step 3: Attach ranks to roster_with_proj**

Right after the rankings computation, attach ranks to each player in `roster_with_proj`:

```python
        # Attach ranks to roster players
        for entry in roster_with_proj:
            norm = normalize_name(entry["name"])
            entry["rank"] = rankings_lookup.get(norm, {})
```

Note: `roster_with_proj` was already written to cache before this point. We need to move the `write_cache("roster", ...)` call to AFTER the ranks are attached. Move it from its current location to after this rank attachment code:

```python
        write_cache("roster", roster_with_proj, cache_dir)
```

- [ ] **Step 4: Attach ranks to waiver results**

After the `waiver_recs = scan_waivers(...)` call and before `write_cache("waivers", ...)`, add:

```python
        # Attach ranks to waiver recommendations
        for rec in waiver_recs:
            rec["add_rank"] = rankings_lookup.get(normalize_name(rec["add"]), {})
            rec["drop_rank"] = rankings_lookup.get(normalize_name(rec["drop"]), {})
```

- [ ] **Step 5: Attach ranks to trade proposals**

After the trade pitch generation loop and before `write_cache("trades", ...)`, add:

```python
        # Attach ranks to trade proposals
        for trade in trade_proposals:
            trade["send_rank"] = rankings_lookup.get(normalize_name(trade["send"]), {})
            trade["receive_rank"] = rankings_lookup.get(normalize_name(trade["receive"]), {})
```

- [ ] **Step 6: Attach ranks to buy-low candidates**

After `buy_low_trade_targets` and `buy_low_free_agents` are built and before `write_cache("buy_low", ...)`, add:

```python
        # Attach ranks to buy-low candidates
        for candidate in buy_low_trade_targets + buy_low_free_agents:
            candidate["rank"] = rankings_lookup.get(normalize_name(candidate["name"]), {})
```

- [ ] **Step 7: Run tests**

Run: `pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 8: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "feat: compute and cache SGP rankings, attach to all player data"
```

---

### Task 4: Add rank badge styles to CSS

**Files:**
- Modify: `src/fantasy_baseball/web/static/season.css`

- [ ] **Step 1: Add rank badge and tooltip styles**

Append to `src/fantasy_baseball/web/static/season.css`:

```css
/* SGP Rank Badge */
.rank-badge {
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    color: var(--accent);
    background: rgba(33, 150, 243, 0.1);
    padding: 1px 6px;
    border-radius: 3px;
    cursor: default;
    position: relative;
}
.rank-badge:hover .rank-tooltip {
    display: block;
}
.rank-tooltip {
    display: none;
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    background: var(--panel-bg);
    border: 1px solid var(--panel-border);
    border-radius: 4px;
    padding: 6px 10px;
    font-size: 11px;
    font-weight: 400;
    white-space: nowrap;
    z-index: 100;
    margin-bottom: 4px;
    color: var(--text);
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
}
.rank-tooltip::after {
    content: "";
    position: absolute;
    top: 100%;
    left: 50%;
    transform: translateX(-50%);
    border: 5px solid transparent;
    border-top-color: var(--panel-border);
}
.rank-tooltip-row {
    display: flex;
    justify-content: space-between;
    gap: 12px;
}
.rank-tooltip-label {
    color: var(--text-secondary);
}
```

- [ ] **Step 2: Commit**

```bash
git add src/fantasy_baseball/web/static/season.css
git commit -m "feat: add rank badge and tooltip CSS styles"
```

---

### Task 5: Display ranks in templates

**Files:**
- Modify: `src/fantasy_baseball/web/templates/season/lineup.html`
- Modify: `src/fantasy_baseball/web/templates/season/waivers_trades.html`
- Modify: `src/fantasy_baseball/web/templates/season/players.html`

This task adds rank badges across all surfaces. The badge HTML pattern is a Jinja2 macro-like snippet used consistently everywhere:

```html
<span class="rank-badge">#{{ rank.ros }}
    <span class="rank-tooltip">
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">ROS</span><span>#{{ rank.ros }}</span></div>
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">Preseason</span><span>{{ "#" ~ rank.preseason if rank.preseason else "—" }}</span></div>
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">Current</span><span>{{ "#" ~ rank.current if rank.current else "—" }}</span></div>
    </span>
</span>
```

- [ ] **Step 1: Add rank badge to lineup hitter rows**

In `src/fantasy_baseball/web/templates/season/lineup.html`, find where each hitter's name is displayed. It will be inside a table row rendering `h.name`. After the name span, add the rank badge. Read the file first to find the exact location.

The badge should show `h.rank.ros` with tooltip for all three. Guard with `{% if h.rank and h.rank.ros %}`:

```html
{% if h.rank and h.rank.ros %}
<span class="rank-badge">#{{ h.rank.ros }}
    <span class="rank-tooltip">
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">ROS</span><span>#{{ h.rank.ros }}</span></div>
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">Preseason</span><span>{{ "#" ~ h.rank.preseason if h.rank.preseason else "—" }}</span></div>
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">Current</span><span>{{ "#" ~ h.rank.current if h.rank.current else "—" }}</span></div>
    </span>
</span>
{% endif %}
```

Do the same for pitcher rows (using `p.rank`).

- [ ] **Step 2: Add rank badge to waiver recommendations**

In `src/fantasy_baseball/web/templates/season/waivers_trades.html`, find where `w.add` (the add player name) is displayed. After the name, add:

```html
{% if w.add_rank and w.add_rank.ros %}
<span class="rank-badge">#{{ w.add_rank.ros }}
    <span class="rank-tooltip">
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">ROS</span><span>#{{ w.add_rank.ros }}</span></div>
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">Preseason</span><span>{{ "#" ~ w.add_rank.preseason if w.add_rank.preseason else "—" }}</span></div>
        <div class="rank-tooltip-row"><span class="rank-tooltip-label">Current</span><span>{{ "#" ~ w.add_rank.current if w.add_rank.current else "—" }}</span></div>
    </span>
</span>
{% endif %}
```

Do the same for `w.drop` using `w.drop_rank`.

- [ ] **Step 3: Add rank badge to trade recommendations**

In the same template, find where trade `send` and `receive` player names are displayed. Add rank badges using `trade.send_rank` and `trade.receive_rank`, same pattern as above.

- [ ] **Step 4: Add rank badge to buy-low candidates**

In the same template, find where buy-low candidate names are displayed. Add rank badge using `candidate.rank` (or the variable name used in the template loop).

- [ ] **Step 5: Add rank to player search results**

In `src/fantasy_baseball/web/templates/season/players.html`, update the `renderCard` JavaScript function to show the rank. The rank data comes from the API response.

In `src/fantasy_baseball/web/season_routes.py`, in the `api_player_search` endpoint, after building each result dict, add the rank lookup:

```python
                # Rank
                rankings_cache = read_cache("rankings") or {}
                rank = rankings_cache.get(norm, {})
```

Add `"rank": rank` to the result dict.

Then in the JS `renderCard` function in `players.html`, add the rank badge after the wSGP value:

```javascript
const rankHtml = p.rank && p.rank.ros
    ? '<span class="rank-badge">#' + p.rank.ros +
      '<span class="rank-tooltip">' +
      '<div class="rank-tooltip-row"><span class="rank-tooltip-label">ROS</span><span>#' + p.rank.ros + '</span></div>' +
      '<div class="rank-tooltip-row"><span class="rank-tooltip-label">Preseason</span><span>' + (p.rank.preseason ? '#' + p.rank.preseason : '—') + '</span></div>' +
      '<div class="rank-tooltip-row"><span class="rank-tooltip-label">Current</span><span>' + (p.rank.current ? '#' + p.rank.current : '—') + '</span></div>' +
      '</span></span>'
    : '';
```

Insert `rankHtml` into the card header after the wSGP span.

- [ ] **Step 6: Run tests**

Run: `pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 7: Commit**

```bash
git add src/fantasy_baseball/web/templates/season/lineup.html src/fantasy_baseball/web/templates/season/waivers_trades.html src/fantasy_baseball/web/templates/season/players.html src/fantasy_baseball/web/season_routes.py
git commit -m "feat: display SGP rank badges across all player surfaces"
```
