# Recency Weighting Backtest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Determine whether blending recent actual stats with preseason projections improves next-week and rest-of-season prediction accuracy for fantasy-relevant stats using 2025 data.

**Architecture:** A game log fetcher caches per-player daily stats from the MLB API. A recency module contains five prediction models (preseason, season-to-date, fixed blend, reliability blend, exponential decay). A backtest runner evaluates all models at five monthly checkpoints against next-week and ROS targets, outputting a comparison table and CSV.

**Tech Stack:** `requests` (MLB Stats API), pandas, numpy, JSON caching.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/analysis/recency.py` | Create | Five prediction model functions + helpers |
| `src/fantasy_baseball/analysis/__init__.py` | Create | Package init |
| `src/fantasy_baseball/analysis/game_logs.py` | Create | Fetch + cache game logs from MLB API |
| `tests/test_analysis/test_recency.py` | Create | Unit tests for the five models |
| `tests/test_analysis/__init__.py` | Create | Package init |
| `scripts/backtest_recency.py` | Create | Runner script that wires everything together |

---

### Task 1: Game log fetcher with caching

**Files:**
- Create: `src/fantasy_baseball/analysis/__init__.py`
- Create: `src/fantasy_baseball/analysis/game_logs.py`
- Create: `tests/test_analysis/__init__.py`
- Create: `tests/test_analysis/test_game_logs.py`

- [ ] **Step 1: Write test for game log parsing**

```python
# tests/test_analysis/test_game_logs.py
from fantasy_baseball.analysis.game_logs import parse_hitter_game_log, parse_pitcher_game_log


def test_parse_hitter_game_log():
    raw_split = {
        "date": "2025-06-15",
        "stat": {
            "atBats": 4, "hits": 2, "homeRuns": 1, "runs": 1,
            "rbi": 2, "stolenBases": 0, "plateAppearances": 5,
        },
    }
    result = parse_hitter_game_log(raw_split)
    assert result["date"] == "2025-06-15"
    assert result["ab"] == 4
    assert result["h"] == 2
    assert result["hr"] == 1
    assert result["pa"] == 5


def test_parse_pitcher_game_log():
    raw_split = {
        "date": "2025-06-15",
        "stat": {
            "inningsPitched": "6.0", "strikeOuts": 8, "earnedRuns": 2,
            "baseOnBalls": 1, "hits": 4, "wins": 1, "losses": 0,
            "saves": 0, "gamesStarted": 1, "gamesPlayed": 1,
            "battersFaced": 23,
        },
    }
    result = parse_pitcher_game_log(raw_split)
    assert result["date"] == "2025-06-15"
    assert result["ip"] == 6.0
    assert result["k"] == 8
    assert result["er"] == 2
    assert result["gs"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_analysis/test_game_logs.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement game log parsing and fetching**

```python
# src/fantasy_baseball/analysis/__init__.py
# (empty)
```

```python
# src/fantasy_baseball/analysis/game_logs.py
"""Fetch and cache player game logs from the MLB Stats API."""
import json
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

MLB_API_BASE = "https://statsapi.mlb.com/api/v1"


def parse_hitter_game_log(split: dict) -> dict:
    """Parse a single hitter game log entry from the MLB API."""
    stat = split["stat"]
    return {
        "date": split["date"],
        "pa": int(stat.get("plateAppearances", 0)),
        "ab": int(stat.get("atBats", 0)),
        "h": int(stat.get("hits", 0)),
        "hr": int(stat.get("homeRuns", 0)),
        "r": int(stat.get("runs", 0)),
        "rbi": int(stat.get("rbi", 0)),
        "sb": int(stat.get("stolenBases", 0)),
    }


def parse_pitcher_game_log(split: dict) -> dict:
    """Parse a single pitcher game log entry from the MLB API."""
    stat = split["stat"]
    ip_str = str(stat.get("inningsPitched", "0"))
    # MLB API returns IP as "6.1" meaning 6 and 1/3
    if "." in ip_str:
        whole, frac = ip_str.split(".")
        ip = int(whole) + int(frac) / 3.0
    else:
        ip = float(ip_str)
    return {
        "date": split["date"],
        "ip": round(ip, 4),
        "k": int(stat.get("strikeOuts", 0)),
        "er": int(stat.get("earnedRuns", 0)),
        "bb": int(stat.get("baseOnBalls", 0)),
        "h_allowed": int(stat.get("hits", 0)),
        "w": int(stat.get("wins", 0)),
        "sv": int(stat.get("saves", 0)),
        "gs": int(stat.get("gamesStarted", 0)),
        "g": int(stat.get("gamesPlayed", 0)),
    }


def fetch_player_game_log(
    mlbam_id: int, season: int, group: str = "hitting",
) -> list[dict]:
    """Fetch game log from MLB Stats API for one player.

    Args:
        mlbam_id: The player's MLBAM ID (e.g. 592450 for Aaron Judge).
        season: The season year (e.g. 2025).
        group: 'hitting' or 'pitching'.

    Returns:
        List of parsed game log dicts, sorted by date.
    """
    url = f"{MLB_API_BASE}/people/{mlbam_id}/stats"
    params = {"stats": "gameLog", "group": group, "season": season}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    splits = data.get("stats", [{}])[0].get("splits", [])
    parser = parse_hitter_game_log if group == "hitting" else parse_pitcher_game_log
    return [parser(s) for s in splits]


def fetch_all_game_logs(
    players: list[dict], season: int = 2025, cache_path: Path | None = None,
) -> dict[int, dict]:
    """Fetch game logs for a list of players, with JSON caching.

    Args:
        players: List of dicts with keys 'mlbam_id', 'name', 'type' ('hitter'/'pitcher').
        season: Season year.
        cache_path: Path to JSON cache file. If exists and complete, returns cached data.

    Returns:
        Dict keyed by mlbam_id: {'name': str, 'type': str, 'games': [game_log_dicts]}.
    """
    # Try cache first
    if cache_path and cache_path.exists():
        with open(cache_path) as f:
            cached = json.load(f)
        # Check if cache covers all requested players
        cached_ids = {int(k) for k in cached.keys()}
        requested_ids = {p["mlbam_id"] for p in players}
        if requested_ids.issubset(cached_ids):
            logger.info("Using cached game logs (%d players)", len(cached))
            return {int(k): v for k, v in cached.items()}

    results = {}
    for i, player in enumerate(players):
        mid = player["mlbam_id"]
        name = player["name"]
        ptype = player["type"]
        group = "hitting" if ptype == "hitter" else "pitching"
        try:
            games = fetch_player_game_log(mid, season, group)
            results[mid] = {"name": name, "type": ptype, "games": games}
        except Exception:
            logger.warning("Failed to fetch game log for %s (ID %s)", name, mid)
            results[mid] = {"name": name, "type": ptype, "games": []}

        if (i + 1) % 25 == 0:
            print(f"  Fetched {i + 1}/{len(players)} game logs...")

    # Save cache
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump({str(k): v for k, v in results.items()}, f)

    return results
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_game_logs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/ tests/test_analysis/
git commit -m "feat(analysis): game log fetcher with MLB API and JSON caching"
```

---

### Task 2: Five prediction models

**Files:**
- Create: `src/fantasy_baseball/analysis/recency.py`
- Create: `tests/test_analysis/test_recency.py`

- [ ] **Step 1: Write tests for all five models (hitter case)**

```python
# tests/test_analysis/test_recency.py
import pytest
from fantasy_baseball.analysis.recency import (
    predict_preseason,
    predict_season_to_date,
    predict_fixed_blend,
    predict_reliability_blend,
    predict_exponential_decay,
)

# Preseason projection: per-PA rates
PROJ_HITTER = {
    "hr_per_pa": 0.040, "r_per_pa": 0.120, "rbi_per_pa": 0.110,
    "sb_per_pa": 0.015, "avg": 0.270,
}

# 30 fake daily game logs (higher performance than projection)
HOT_GAMES = [
    {"date": f"2025-06-{d:02d}", "pa": 4, "ab": 4, "h": 2, "hr": 1,
     "r": 1, "rbi": 1, "sb": 0}
    for d in range(1, 31)
]
# Totals: 120 PA, 60 H, 30 HR, 30 R, 30 RBI, 0 SB -> .500 AVG, 0.25 HR/PA


def test_preseason_ignores_actuals():
    result = predict_preseason(PROJ_HITTER, HOT_GAMES, "2025-07-01")
    assert result["hr_per_pa"] == pytest.approx(0.040)
    assert result["avg"] == pytest.approx(0.270)


def test_season_to_date_ignores_projection():
    result = predict_season_to_date(PROJ_HITTER, HOT_GAMES, "2025-07-01")
    assert result["hr_per_pa"] == pytest.approx(30 / 120, abs=0.001)
    assert result["avg"] == pytest.approx(60 / 120, abs=0.001)


def test_fixed_blend_between_proj_and_actual():
    result = predict_fixed_blend(PROJ_HITTER, HOT_GAMES, "2025-07-01")
    # 30% actual (0.25 HR/PA) + 70% proj (0.04) = 0.103
    assert result["hr_per_pa"] == pytest.approx(0.103, abs=0.005)


def test_reliability_blend_weights_by_sample_size():
    result = predict_reliability_blend(PROJ_HITTER, HOT_GAMES, "2025-07-01")
    # 120 PA, HR reliability constant = 200 PA
    # actual_weight = 120 / (120 + 200) = 0.375
    # blended = 0.375 * 0.25 + 0.625 * 0.04 = 0.119
    assert result["hr_per_pa"] == pytest.approx(0.119, abs=0.01)


def test_exponential_decay_weights_recent_more():
    # Make first 15 days cold, last 15 days hot
    mixed = []
    for d in range(1, 16):
        mixed.append({"date": f"2025-06-{d:02d}", "pa": 4, "ab": 4,
                       "h": 0, "hr": 0, "r": 0, "rbi": 0, "sb": 0})
    for d in range(16, 31):
        mixed.append({"date": f"2025-06-{d:02d}", "pa": 4, "ab": 4,
                       "h": 3, "hr": 1, "r": 1, "rbi": 2, "sb": 1})
    result_decay = predict_exponential_decay(PROJ_HITTER, mixed, "2025-07-01")
    result_flat = predict_season_to_date(PROJ_HITTER, mixed, "2025-07-01")
    # Decay should produce a higher HR rate than flat STD because
    # it weights the hot recent period more heavily
    assert result_decay["hr_per_pa"] > result_flat["hr_per_pa"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_analysis/test_recency.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement the five models**

```python
# src/fantasy_baseball/analysis/recency.py
"""Five prediction models for the recency weighting backtest."""
import math
from datetime import datetime, timedelta

# Reliability constants: how many PA/IP of projection a stat is "worth"
HITTER_RELIABILITY = {
    "hr_per_pa": 200, "r_per_pa": 300, "rbi_per_pa": 300,
    "sb_per_pa": 300, "avg": 400,
}
PITCHER_RELIABILITY = {
    "k_per_ip": 50, "era": 120, "whip": 80,
    "w_per_gs": 200, "sv_per_g": 200,
}

DECAY_HALF_LIFE_DAYS = 7
FIXED_BLEND_ACTUAL_WEIGHT = 0.30
FIXED_BLEND_WINDOW_DAYS = 30


def _aggregate_hitter_games(games: list[dict]) -> dict:
    """Aggregate game logs into totals and per-PA rates."""
    pa = sum(g["pa"] for g in games)
    ab = sum(g["ab"] for g in games)
    if pa == 0:
        return {"pa": 0, "ab": 0, "hr_per_pa": 0, "r_per_pa": 0,
                "rbi_per_pa": 0, "sb_per_pa": 0, "avg": 0}
    return {
        "pa": pa, "ab": ab,
        "hr_per_pa": sum(g["hr"] for g in games) / pa,
        "r_per_pa": sum(g["r"] for g in games) / pa,
        "rbi_per_pa": sum(g["rbi"] for g in games) / pa,
        "sb_per_pa": sum(g["sb"] for g in games) / pa,
        "avg": sum(g["h"] for g in games) / ab if ab > 0 else 0,
    }


def _aggregate_pitcher_games(games: list[dict]) -> dict:
    """Aggregate game logs into totals and per-IP rates."""
    ip = sum(g["ip"] for g in games)
    gs = sum(g.get("gs", 0) for g in games)
    g_count = sum(g.get("g", 1) for g in games)
    if ip == 0:
        return {"ip": 0, "gs": gs, "g": g_count, "k_per_ip": 0,
                "era": 0, "whip": 0, "w_per_gs": 0, "sv_per_g": 0}
    k = sum(g["k"] for g in games)
    er = sum(g["er"] for g in games)
    bb = sum(g["bb"] for g in games)
    h_a = sum(g["h_allowed"] for g in games)
    w = sum(g["w"] for g in games)
    sv = sum(g["sv"] for g in games)
    return {
        "ip": ip, "gs": gs, "g": g_count,
        "k_per_ip": k / ip,
        "era": er * 9 / ip,
        "whip": (bb + h_a) / ip,
        "w_per_gs": w / gs if gs > 0 else 0,
        "sv_per_g": sv / g_count if g_count > 0 else 0,
    }


def _filter_games_before(games: list[dict], cutoff: str) -> list[dict]:
    """Return games with date < cutoff."""
    return [g for g in games if g["date"] < cutoff]


def _filter_games_window(games: list[dict], cutoff: str, window_days: int) -> list[dict]:
    """Return games in the window_days before cutoff."""
    cutoff_dt = datetime.strptime(cutoff, "%Y-%m-%d")
    start_dt = cutoff_dt - timedelta(days=window_days)
    start_str = start_dt.strftime("%Y-%m-%d")
    return [g for g in games if start_str <= g["date"] < cutoff]


def _detect_player_type(projection: dict) -> str:
    """Detect whether a projection dict is for a hitter or pitcher."""
    if "hr_per_pa" in projection:
        return "hitter"
    return "pitcher"


def _aggregate(games: list[dict], player_type: str) -> dict:
    if player_type == "hitter":
        return _aggregate_hitter_games(games)
    return _aggregate_pitcher_games(games)


def _stat_keys(player_type: str) -> list[str]:
    if player_type == "hitter":
        return ["hr_per_pa", "r_per_pa", "rbi_per_pa", "sb_per_pa", "avg"]
    return ["k_per_ip", "era", "whip", "w_per_gs", "sv_per_g"]


def _reliability_constants(player_type: str) -> dict:
    return HITTER_RELIABILITY if player_type == "hitter" else PITCHER_RELIABILITY


def _sample_size_key(player_type: str) -> str:
    return "pa" if player_type == "hitter" else "ip"


# ---- Model 1: Preseason only ----

def predict_preseason(
    projection: dict, games: list[dict], cutoff: str,
) -> dict:
    """Return preseason projection rates unchanged."""
    keys = _stat_keys(_detect_player_type(projection))
    return {k: projection[k] for k in keys}


# ---- Model 2: Season-to-date ----

def predict_season_to_date(
    projection: dict, games: list[dict], cutoff: str,
) -> dict:
    """Return rates from all games before the cutoff date."""
    ptype = _detect_player_type(projection)
    before = _filter_games_before(games, cutoff)
    agg = _aggregate(before, ptype)
    keys = _stat_keys(ptype)
    return {k: agg.get(k, 0) for k in keys}


# ---- Model 3: Fixed blend (30% last-30-day, 70% projection) ----

def predict_fixed_blend(
    projection: dict, games: list[dict], cutoff: str,
    actual_weight: float = FIXED_BLEND_ACTUAL_WEIGHT,
    window_days: int = FIXED_BLEND_WINDOW_DAYS,
) -> dict:
    """Blend last-N-day actuals with preseason projection at fixed ratio."""
    ptype = _detect_player_type(projection)
    window = _filter_games_window(games, cutoff, window_days)
    agg = _aggregate(window, ptype)
    keys = _stat_keys(ptype)
    proj_weight = 1.0 - actual_weight
    result = {}
    for k in keys:
        actual_val = agg.get(k, 0)
        proj_val = projection.get(k, 0)
        # If no games in window, fall back to projection
        ss = agg.get(_sample_size_key(ptype), 0)
        if ss == 0:
            result[k] = proj_val
        else:
            result[k] = actual_weight * actual_val + proj_weight * proj_val
    return result


# ---- Model 4: Reliability blend (weight scales with sample size) ----

def predict_reliability_blend(
    projection: dict, games: list[dict], cutoff: str,
) -> dict:
    """Blend where actual weight = actual_PA / (actual_PA + reliability_constant)."""
    ptype = _detect_player_type(projection)
    before = _filter_games_before(games, cutoff)
    agg = _aggregate(before, ptype)
    keys = _stat_keys(ptype)
    reliability = _reliability_constants(ptype)
    ss = agg.get(_sample_size_key(ptype), 0)
    result = {}
    for k in keys:
        rc = reliability.get(k, 300)
        actual_w = ss / (ss + rc) if (ss + rc) > 0 else 0
        proj_w = 1.0 - actual_w
        result[k] = actual_w * agg.get(k, 0) + proj_w * projection.get(k, 0)
    return result


# ---- Model 5: Exponential decay ----

def predict_exponential_decay(
    projection: dict, games: list[dict], cutoff: str,
    half_life_days: float = DECAY_HALF_LIFE_DAYS,
) -> dict:
    """Weight each game by exponential decay from cutoff, blend with projection."""
    ptype = _detect_player_type(projection)
    before = _filter_games_before(games, cutoff)
    if not before:
        return predict_preseason(projection, games, cutoff)

    cutoff_dt = datetime.strptime(cutoff, "%Y-%m-%d")
    decay_rate = math.log(2) / half_life_days
    keys = _stat_keys(ptype)

    if ptype == "hitter":
        raw_keys = {"hr_per_pa": "hr", "r_per_pa": "r", "rbi_per_pa": "rbi",
                    "sb_per_pa": "sb", "avg": "h"}
        denom_key = {"hr_per_pa": "pa", "r_per_pa": "pa", "rbi_per_pa": "pa",
                     "sb_per_pa": "pa", "avg": "ab"}
    else:
        raw_keys = {"k_per_ip": "k", "era": "er", "whip": "bb",
                    "w_per_gs": "w", "sv_per_g": "sv"}
        denom_key = {"k_per_ip": "ip", "era": "ip", "whip": "ip",
                     "w_per_gs": "gs", "sv_per_g": "g"}

    weighted_nums = {k: 0.0 for k in keys}
    weighted_denoms = {k: 0.0 for k in keys}
    total_weight = 0.0

    for game in before:
        game_dt = datetime.strptime(game["date"], "%Y-%m-%d")
        days_ago = (cutoff_dt - game_dt).days
        weight = math.exp(-decay_rate * days_ago)
        total_weight += weight

        for k in keys:
            rk = raw_keys[k]
            dk = denom_key[k]
            if k == "whip":
                # WHIP = (BB + H) / IP
                num = (game.get("bb", 0) + game.get("h_allowed", 0)) * weight
            elif k == "era":
                num = game.get("er", 0) * weight
            else:
                num = game.get(rk, 0) * weight
            den = game.get(dk, 0) * weight
            weighted_nums[k] += num
            weighted_denoms[k] += den

    # Compute weighted rates
    actual_rates = {}
    for k in keys:
        den = weighted_denoms[k]
        if den > 0:
            if k == "era":
                actual_rates[k] = weighted_nums[k] * 9 / den
            else:
                actual_rates[k] = weighted_nums[k] / den
        else:
            actual_rates[k] = projection.get(k, 0)

    # Blend with projection using reliability constants
    reliability = _reliability_constants(ptype)
    ss_key = _sample_size_key(ptype)
    # Use effective sample size (weighted) as a rough proxy
    total_ss = sum(g.get(ss_key, 0) for g in before)
    result = {}
    for k in keys:
        rc = reliability.get(k, 300)
        actual_w = total_ss / (total_ss + rc) if (total_ss + rc) > 0 else 0
        result[k] = actual_w * actual_rates[k] + (1 - actual_w) * projection.get(k, 0)

    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_analysis/test_recency.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/analysis/recency.py tests/test_analysis/test_recency.py
git commit -m "feat(analysis): five prediction models for recency backtest"
```

---

### Task 3: Backtest runner — data loading and player matching

**Files:**
- Create: `scripts/backtest_recency.py` (partial — data loading only)

- [ ] **Step 1: Create script with data loading**

```python
# scripts/backtest_recency.py
"""Recency weighting backtest: compare prediction models using 2025 game logs."""
import csv
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from backtest_2025 import DRAFT_2025
from fantasy_baseball.analysis.game_logs import fetch_all_game_logs
from fantasy_baseball.analysis.recency import (
    predict_preseason,
    predict_season_to_date,
    predict_fixed_blend,
    predict_reliability_blend,
    predict_exponential_decay,
)

GAME_LOG_CACHE = PROJECT_ROOT / "data" / "stats" / "game_logs_2025.json"
RESULTS_CSV = PROJECT_ROOT / "data" / "stats" / "recency_backtest_results.csv"
PROJ_DIR = PROJECT_ROOT / "data" / "projections"

CHECKPOINTS = ["2025-05-01", "2025-06-01", "2025-07-01", "2025-08-01", "2025-09-01"]
NEXT_WEEK_DAYS = 7

MODELS = {
    "preseason": predict_preseason,
    "season_to_date": predict_season_to_date,
    "fixed_blend": predict_fixed_blend,
    "reliability_blend": predict_reliability_blend,
    "exponential_decay": predict_exponential_decay,
}

HITTER_STATS = ["hr_per_pa", "r_per_pa", "rbi_per_pa", "sb_per_pa", "avg"]
PITCHER_STATS = ["k_per_ip", "era", "whip", "w_per_gs", "sv_per_g"]


def normalize_name(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    return "".join(c for c in nfkd if not unicodedata.combining(c)).lower().strip()


def load_preseason_projections() -> dict[int, dict]:
    """Load blended Steamer+ZiPS 2025 projections, keyed by MLBAMID.

    Returns per-PA/IP rate projections for each player.
    """
    sh = pd.read_csv(PROJ_DIR / "steamer-hitters-2025.csv")
    zh = pd.read_csv(PROJ_DIR / "zips-hitters-2025.csv")
    sp = pd.read_csv(PROJ_DIR / "steamer-pitchers-2025.csv")
    zp = pd.read_csv(PROJ_DIR / "zips-pitchers-2025.csv")

    projections = {}

    # Hitters: blend and compute per-PA rates
    for df in [sh, zh]:
        df = df.dropna(subset=["MLBAMID"])
        for _, row in df.iterrows():
            mid = int(row["MLBAMID"])
            pa = float(row.get("PA", 0) or 0)
            ab = float(row.get("AB", 0) or 0)
            if pa < 50:
                continue
            entry = projections.get(mid, {})
            if "type" not in entry:
                entry = {
                    "type": "hitter", "name": row["Name"], "pa": 0,
                    "hr": 0, "r": 0, "rbi": 0, "sb": 0, "h": 0, "ab": 0,
                    "_count": 0,
                }
            entry["pa"] += pa
            entry["hr"] += float(row.get("HR", 0) or 0)
            entry["r"] += float(row.get("R", 0) or 0)
            entry["rbi"] += float(row.get("RBI", 0) or 0)
            entry["sb"] += float(row.get("SB", 0) or 0)
            entry["h"] += float(row.get("H", 0) or 0)
            entry["ab"] += ab
            entry["_count"] += 1
            projections[mid] = entry

    # Average across systems
    for mid, entry in projections.items():
        if entry.get("type") != "hitter":
            continue
        n = entry["_count"]
        if n > 0:
            pa = entry["pa"] / n
            ab = entry["ab"] / n
            entry["hr_per_pa"] = (entry["hr"] / n) / pa if pa > 0 else 0
            entry["r_per_pa"] = (entry["r"] / n) / pa if pa > 0 else 0
            entry["rbi_per_pa"] = (entry["rbi"] / n) / pa if pa > 0 else 0
            entry["sb_per_pa"] = (entry["sb"] / n) / pa if pa > 0 else 0
            entry["avg"] = (entry["h"] / n) / ab if ab > 0 else 0

    # Pitchers
    for df in [sp, zp]:
        df = df.dropna(subset=["MLBAMID"])
        for _, row in df.iterrows():
            mid = int(row["MLBAMID"])
            ip = float(row.get("IP", 0) or 0)
            if ip < 10:
                continue
            if mid in projections and projections[mid].get("type") == "hitter":
                continue  # Don't overwrite hitter with pitcher
            entry = projections.get(mid, {})
            if "type" not in entry or entry.get("type") != "pitcher":
                entry = {
                    "type": "pitcher", "name": row["Name"], "ip": 0,
                    "k": 0, "er": 0, "bb": 0, "h_allowed": 0,
                    "w": 0, "sv": 0, "gs": 0, "g": 0, "_count": 0,
                }
            entry["ip"] += ip
            entry["k"] += float(row.get("SO", 0) or 0)
            entry["er"] += float(row.get("ER", 0) or 0)
            entry["bb"] += float(row.get("BB", 0) or 0)
            entry["h_allowed"] += float(row.get("H", 0) or 0)
            entry["w"] += float(row.get("W", 0) or 0)
            entry["sv"] += float(row.get("SV", 0) or 0)
            gs = float(row.get("GS", 0) or 0)
            g = float(row.get("G", 0) or 0)
            entry["gs"] += gs
            entry["g"] += g
            entry["_count"] += 1
            projections[mid] = entry

    for mid, entry in projections.items():
        if entry.get("type") != "pitcher":
            continue
        n = entry["_count"]
        if n > 0:
            ip = entry["ip"] / n
            gs = entry["gs"] / n
            g = entry["g"] / n
            entry["k_per_ip"] = (entry["k"] / n) / ip if ip > 0 else 0
            entry["era"] = (entry["er"] / n) * 9 / ip if ip > 0 else 0
            entry["whip"] = ((entry["bb"] / n) + (entry["h_allowed"] / n)) / ip if ip > 0 else 0
            entry["w_per_gs"] = (entry["w"] / n) / gs if gs > 0 else 0
            entry["sv_per_g"] = (entry["sv"] / n) / g if g > 0 else 0

    return projections


def match_draft_to_projections(projections: dict[int, dict]) -> list[dict]:
    """Match DRAFT_2025 player names to MLBAMID via projections.

    Returns list of {mlbam_id, name, type} for game log fetching.
    """
    # Build name -> mlbamid lookup
    name_to_id = {}
    for mid, entry in projections.items():
        name_norm = normalize_name(entry["name"])
        name_to_id[name_norm] = mid

    players = []
    seen = set()
    for _, player_name, _ in DRAFT_2025:
        name_norm = normalize_name(player_name)
        # Try exact match, then without suffixes
        mid = name_to_id.get(name_norm)
        if mid is None:
            clean = name_norm.replace(" jr.", "").replace(" jr", "").replace(" ii", "").strip()
            for nk, nid in name_to_id.items():
                nclean = nk.replace(" jr.", "").replace(" jr", "").replace(" ii", "").strip()
                if clean == nclean and len(clean) > 4:
                    mid = nid
                    break
        if mid and mid not in seen:
            seen.add(mid)
            entry = projections[mid]
            players.append({
                "mlbam_id": mid,
                "name": player_name,
                "type": entry["type"],
            })

    return players
```

- [ ] **Step 2: Verify it loads and matches by running a quick test**

Run: `python -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); exec(open('scripts/backtest_recency.py').read()); projs = load_preseason_projections(); players = match_draft_to_projections(projs); print(f'Matched {len(players)}/230 drafted players')"`

Expected: Matches ~200+ players

- [ ] **Step 3: Commit**

```bash
git add scripts/backtest_recency.py
git commit -m "feat(backtest): recency backtest runner — data loading and player matching"
```

---

### Task 4: Backtest evaluation loop

**Files:**
- Modify: `scripts/backtest_recency.py` (add main function with evaluation loop)

- [ ] **Step 1: Add evaluation and summary functions**

Append to `scripts/backtest_recency.py`:

```python
def compute_actual_rates(games: list[dict], start: str, end: str, player_type: str) -> dict | None:
    """Compute actual per-PA/IP rates for games in [start, end).

    Returns None if insufficient data (hitters: <10 PA, pitchers: <3 IP).
    """
    window = [g for g in games if start <= g["date"] < end]
    if player_type == "hitter":
        from fantasy_baseball.analysis.recency import _aggregate_hitter_games
        agg = _aggregate_hitter_games(window)
        if agg["pa"] < 10:
            return None
        return {k: agg[k] for k in HITTER_STATS}
    else:
        from fantasy_baseball.analysis.recency import _aggregate_pitcher_games
        agg = _aggregate_pitcher_games(window)
        if agg["ip"] < 3:
            return None
        return {k: agg[k] for k in PITCHER_STATS}


def compute_actual_ros(games: list[dict], start: str, player_type: str) -> dict | None:
    """Compute actual rates from start through end of season.

    Returns None if insufficient data (hitters: <50 PA, pitchers: <20 IP).
    """
    remaining = [g for g in games if g["date"] >= start]
    if player_type == "hitter":
        from fantasy_baseball.analysis.recency import _aggregate_hitter_games
        agg = _aggregate_hitter_games(remaining)
        if agg["pa"] < 50:
            return None
        return {k: agg[k] for k in HITTER_STATS}
    else:
        from fantasy_baseball.analysis.recency import _aggregate_pitcher_games
        agg = _aggregate_pitcher_games(remaining)
        if agg["ip"] < 20:
            return None
        return {k: agg[k] for k in PITCHER_STATS}


def next_week_end(checkpoint: str) -> str:
    """Return the date 7 days after the checkpoint."""
    from datetime import datetime, timedelta
    dt = datetime.strptime(checkpoint, "%Y-%m-%d")
    return (dt + timedelta(days=NEXT_WEEK_DAYS)).strftime("%Y-%m-%d")


def main():
    print("=" * 80)
    print("RECENCY WEIGHTING BACKTEST — 2025 Season")
    print("=" * 80)

    # Load projections
    print("\nLoading preseason projections...")
    projections = load_preseason_projections()
    print(f"  {sum(1 for v in projections.values() if v['type'] == 'hitter')} hitters, "
          f"{sum(1 for v in projections.values() if v['type'] == 'pitcher')} pitchers")

    # Match draft players
    players = match_draft_to_projections(projections)
    print(f"  Matched {len(players)}/{len(set((p, t) for _, p, t in DRAFT_2025))} unique drafted players")

    # Fetch game logs
    print("\nFetching game logs (cached after first run)...")
    game_logs = fetch_all_game_logs(players, season=2025, cache_path=GAME_LOG_CACHE)
    total_games = sum(len(v["games"]) for v in game_logs.values())
    print(f"  {total_games} total game log entries across {len(game_logs)} players")

    # Run evaluation
    all_results = []

    for checkpoint in CHECKPOINTS:
        week_end = next_week_end(checkpoint)
        print(f"\n{'='*60}")
        print(f"CHECKPOINT: {checkpoint}")
        print(f"{'='*60}")

        for model_name, model_fn in MODELS.items():
            for target_name, get_actual_fn, target_label in [
                ("next_week", lambda g, pt: compute_actual_rates(g, checkpoint, week_end, pt), "Next Week"),
                ("ros", lambda g, pt: compute_actual_ros(g, checkpoint, pt), "Rest of Season"),
            ]:
                errors_by_stat = {}
                n_players = 0

                for player in players:
                    mid = player["mlbam_id"]
                    ptype = player["type"]
                    log_entry = game_logs.get(mid, {})
                    games = log_entry.get("games", [])
                    if not games:
                        continue

                    proj = projections.get(mid)
                    if proj is None:
                        continue

                    stats = HITTER_STATS if ptype == "hitter" else PITCHER_STATS

                    # Get actual performance for this target
                    actual = get_actual_fn(games, ptype)
                    if actual is None:
                        continue

                    # Get model prediction
                    proj_rates = {k: proj[k] for k in stats if k in proj}
                    prediction = model_fn(proj_rates, games, checkpoint)

                    n_players += 1
                    for stat in stats:
                        pred_val = prediction.get(stat, 0)
                        actual_val = actual.get(stat, 0)
                        err = abs(pred_val - actual_val)
                        errors_by_stat.setdefault(stat, []).append(err)

                for stat, errs in errors_by_stat.items():
                    mae = np.mean(errs) if errs else 0
                    all_results.append({
                        "checkpoint": checkpoint,
                        "model": model_name,
                        "target": target_name,
                        "stat": stat,
                        "mae": round(mae, 6),
                        "n_players": len(errs),
                    })

    # Save CSV
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["checkpoint", "model", "target", "stat", "mae", "n_players"])
        writer.writeheader()
        writer.writerows(all_results)
    print(f"\nDetailed results saved to {RESULTS_CSV}")

    # ============================
    # SUMMARY
    # ============================
    print(f"\n{'='*80}")
    print("SUMMARY: Mean Absolute Error by Model and Target")
    print(f"{'='*80}")

    for target in ["next_week", "ros"]:
        target_label = "NEXT WEEK" if target == "next_week" else "REST OF SEASON"
        print(f"\n--- {target_label} ---")

        # Average MAE across all stats and checkpoints per model
        model_avg = {}
        for model_name in MODELS:
            subset = [r for r in all_results if r["model"] == model_name and r["target"] == target]
            if subset:
                model_avg[model_name] = np.mean([r["mae"] for r in subset])

        sorted_models = sorted(model_avg.items(), key=lambda x: x[1])
        baseline = model_avg.get("preseason", 0)
        print(f"\n{'Model':<22} {'Avg MAE':>10} {'vs Preseason':>14}")
        print("-" * 50)
        for model_name, avg_mae in sorted_models:
            diff = avg_mae - baseline
            diff_str = f"{diff:+.6f}" if model_name != "preseason" else "baseline"
            print(f"{model_name:<22} {avg_mae:>10.6f} {diff_str:>14}")

    # Per-stat breakdown for next_week
    print(f"\n{'='*80}")
    print("NEXT-WEEK MAE BY STAT (averaged across checkpoints)")
    print(f"{'='*80}")

    all_stats = HITTER_STATS + PITCHER_STATS
    for stat in all_stats:
        stat_results = [r for r in all_results if r["stat"] == stat and r["target"] == "next_week"]
        if not stat_results:
            continue
        print(f"\n  {stat}:")
        model_maes = {}
        for model_name in MODELS:
            subset = [r for r in stat_results if r["model"] == model_name]
            if subset:
                model_maes[model_name] = np.mean([r["mae"] for r in subset])
        baseline = model_maes.get("preseason", 0)
        for model_name, mae in sorted(model_maes.items(), key=lambda x: x[1]):
            pct = ((mae - baseline) / baseline * 100) if baseline > 0 else 0
            marker = " ***" if mae < baseline and model_name != "preseason" else ""
            print(f"    {model_name:<22} {mae:.6f}  ({pct:+.1f}%){marker}")

    # Per-checkpoint trend
    print(f"\n{'='*80}")
    print("NEXT-WEEK ACCURACY BY CHECKPOINT (does recency help more late-season?)")
    print(f"{'='*80}")
    for checkpoint in CHECKPOINTS:
        print(f"\n  {checkpoint}:")
        for model_name in MODELS:
            subset = [r for r in all_results
                      if r["checkpoint"] == checkpoint and r["model"] == model_name
                      and r["target"] == "next_week"]
            if subset:
                mae = np.mean([r["mae"] for r in subset])
                print(f"    {model_name:<22} {mae:.6f}")

    print(f"\n{'='*80}")
    print("CONCLUSION")
    print(f"{'='*80}")

    # Find best next-week model
    nw_avgs = {}
    for model_name in MODELS:
        subset = [r for r in all_results if r["model"] == model_name and r["target"] == "next_week"]
        if subset:
            nw_avgs[model_name] = np.mean([r["mae"] for r in subset])
    best_nw = min(nw_avgs, key=nw_avgs.get) if nw_avgs else "?"
    baseline_nw = nw_avgs.get("preseason", 0)
    best_nw_mae = nw_avgs.get(best_nw, 0)
    improvement = ((baseline_nw - best_nw_mae) / baseline_nw * 100) if baseline_nw > 0 else 0

    # Find best ROS model
    ros_avgs = {}
    for model_name in MODELS:
        subset = [r for r in all_results if r["model"] == model_name and r["target"] == "ros"]
        if subset:
            ros_avgs[model_name] = np.mean([r["mae"] for r in subset])
    best_ros = min(ros_avgs, key=ros_avgs.get) if ros_avgs else "?"

    print(f"\n  Best next-week predictor: {best_nw} ({improvement:+.1f}% vs preseason)")
    print(f"  Best ROS predictor: {best_ros}")
    if best_nw != "preseason":
        print(f"\n  RECOMMENDATION: Use {best_nw} for start/sit decisions")
    else:
        print(f"\n  FINDING: Preseason projections are hard to beat — recency blending adds noise")
    if best_ros != best_nw:
        print(f"  Use {best_ros} for trade/waiver evaluation")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the full backtest**

Run: `python scripts/backtest_recency.py`

This will take a few minutes on first run (fetching 200+ game logs from the MLB API). Subsequent runs use the cache.

- [ ] **Step 3: Commit**

```bash
git add scripts/backtest_recency.py
git commit -m "feat(backtest): complete recency backtest runner with evaluation and summary"
```
