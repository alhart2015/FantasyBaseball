# Roster Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a roster audit that evaluates every roster slot against the best available FA using recency-blended data for both sides.

**Architecture:** New `blending.py` module extracts recency-blend logic for `Player` objects. New `roster_audit.py` computes per-slot audit entries reusing `_compute_team_wsgp` from `waivers.py`. Both are integrated into the `season_data.py` refresh pipeline, with a new `/roster-audit` web page.

**Tech Stack:** Python, Flask, Jinja2, SQLite, existing `predict_reliability_blend` from `analysis/recency.py`

---

### Task 1: Load per-game logs from SQLite

**Files:**
- Create: `src/fantasy_baseball/lineup/blending.py`
- Test: `tests/test_lineup/test_blending.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lineup/test_blending.py

import sqlite3
from fantasy_baseball.lineup.blending import load_game_logs_by_name


def _create_game_logs_table(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS game_logs (
            season INTEGER NOT NULL,
            mlbam_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            team TEXT,
            player_type TEXT NOT NULL,
            date TEXT NOT NULL,
            pa INTEGER, ab INTEGER, h INTEGER, r INTEGER, hr INTEGER,
            rbi INTEGER, sb INTEGER,
            ip REAL, k INTEGER, er INTEGER, bb INTEGER, h_allowed INTEGER,
            w INTEGER, sv INTEGER, gs INTEGER,
            PRIMARY KEY (season, mlbam_id, date)
        )
    """)


class TestLoadGameLogsByName:
    def test_groups_hitter_games_by_normalized_name(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_game_logs_table(conn)
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, player_type, date, "
            "pa, ab, h, r, hr, rbi, sb) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, 1, "Juan Soto", "hitter", "2026-04-01", 5, 4, 2, 1, 1, 2, 0),
        )
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, player_type, date, "
            "pa, ab, h, r, hr, rbi, sb) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, 1, "Juan Soto", "hitter", "2026-04-02", 4, 3, 1, 0, 0, 0, 1),
        )
        conn.commit()

        result = load_game_logs_by_name(conn, 2026)
        assert "juan soto" in result
        assert len(result["juan soto"]) == 2
        assert result["juan soto"][0]["date"] == "2026-04-01"
        assert result["juan soto"][0]["pa"] == 5
        assert result["juan soto"][0]["hr"] == 1

    def test_groups_pitcher_games_with_g_field(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_game_logs_table(conn)
        conn.execute(
            "INSERT INTO game_logs (season, mlbam_id, name, player_type, date, "
            "ip, k, er, bb, h_allowed, w, sv, gs) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (2026, 2, "Bryan Abreu", "pitcher", "2026-04-01",
             1.0, 1, 2, 1, 2, 0, 0, 0),
        )
        conn.commit()

        result = load_game_logs_by_name(conn, 2026)
        assert "bryan abreu" in result
        games = result["bryan abreu"]
        assert len(games) == 1
        assert games[0]["g"] == 1  # synthesized from row
        assert games[0]["gs"] == 0
        assert games[0]["ip"] == 1.0

    def test_empty_table_returns_empty_dict(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_game_logs_table(conn)

        result = load_game_logs_by_name(conn, 2026)
        assert result == {}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lineup/test_blending.py -v`
Expected: FAIL — `ImportError: cannot import name 'load_game_logs_by_name'`

- [ ] **Step 3: Write the implementation**

```python
# src/fantasy_baseball/lineup/blending.py

"""Recency blending for Player objects.

Bridges the gap between the recency model (which works on rate dicts and
game-log lists) and the Player/Stats dataclasses used by the lineup and
waiver modules.
"""

from __future__ import annotations

from collections import defaultdict

from fantasy_baseball.utils.name_utils import normalize_name

# Hitter game log columns to extract from SQLite
_HITTER_LOG_COLS = ("date", "pa", "ab", "h", "r", "hr", "rbi", "sb")

# Pitcher game log columns to extract from SQLite (g synthesized as 1 per row)
_PITCHER_LOG_COLS = ("date", "ip", "k", "er", "bb", "h_allowed", "w", "sv", "gs")


def load_game_logs_by_name(
    conn,
    season: int,
) -> dict[str, list[dict]]:
    """Load per-game log entries from SQLite, keyed by normalized name.

    Returns {normalized_name: [game_dicts]} where each game dict has the
    fields expected by predict_reliability_blend.  Pitcher dicts include a
    synthesized ``g = 1`` field (each row is one game appearance).
    """
    logs: dict[str, list[dict]] = defaultdict(list)

    # Hitters
    rows = conn.execute(
        "SELECT name, date, pa, ab, h, r, hr, rbi, sb "
        "FROM game_logs WHERE season = ? AND player_type = 'hitter' "
        "ORDER BY date",
        (season,),
    ).fetchall()
    for row in rows:
        norm = normalize_name(row["name"])
        logs[norm].append({col: row[col] for col in _HITTER_LOG_COLS})

    # Pitchers
    rows = conn.execute(
        "SELECT name, date, ip, k, er, bb, h_allowed, w, sv, gs "
        "FROM game_logs WHERE season = ? AND player_type = 'pitcher' "
        "ORDER BY date",
        (season,),
    ).fetchall()
    for row in rows:
        norm = normalize_name(row["name"])
        entry = {col: row[col] for col in _PITCHER_LOG_COLS}
        entry["g"] = 1  # each row is one game appearance
        logs[norm].append(entry)

    return dict(logs)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lineup/test_blending.py -v`
Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/blending.py tests/test_lineup/test_blending.py
git commit -m "feat: add load_game_logs_by_name for per-game log retrieval"
```

---

### Task 2: Blend a Player with game logs

**Files:**
- Modify: `src/fantasy_baseball/lineup/blending.py`
- Test: `tests/test_lineup/test_blending.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_lineup/test_blending.py`:

```python
from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats
from fantasy_baseball.lineup.blending import blend_player_with_game_logs


def _hitter(name, **kwargs):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=["OF"],
        ros=HitterStats(
            pa=kwargs.get("pa", 600), ab=kwargs.get("ab", 540),
            h=kwargs.get("h", 150), r=kwargs.get("r", 80),
            hr=kwargs.get("hr", 25), rbi=kwargs.get("rbi", 80),
            sb=kwargs.get("sb", 10), avg=kwargs.get("avg", 0.278),
        ),
    )


def _pitcher(name, **kwargs):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=["RP"],
        ros=PitcherStats(
            ip=kwargs.get("ip", 65.0), w=kwargs.get("w", 4.0),
            k=kwargs.get("k", 70.0), sv=kwargs.get("sv", 20.0),
            er=kwargs.get("er", 20.0), bb=kwargs.get("bb", 22.0),
            h_allowed=kwargs.get("h_allowed", 50.0),
            era=kwargs.get("era", 2.77), whip=kwargs.get("whip", 1.11),
        ),
    )


class TestBlendPlayerWithGameLogs:
    def test_returns_unchanged_player_when_no_logs(self):
        player = _hitter("Test Hitter", hr=25)
        result = blend_player_with_game_logs(player, [], "2026-04-06")
        assert result.ros.hr == 25
        assert result is not player  # should be a copy

    def test_blends_hitter_stats_toward_actuals(self):
        # Projection: .278 AVG, 25 HR
        player = _hitter("Slugger", pa=600, ab=540, h=150, hr=25, avg=0.278)
        # Actuals: 50 PA, much worse — .100 AVG, 0 HR
        logs = [
            {"date": "2026-04-01", "pa": 5, "ab": 5, "h": 0, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
            {"date": "2026-04-02", "pa": 5, "ab": 5, "h": 1, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
            {"date": "2026-04-03", "pa": 5, "ab": 4, "h": 0, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
            {"date": "2026-04-04", "pa": 5, "ab": 5, "h": 0, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
            {"date": "2026-04-05", "pa": 5, "ab": 4, "h": 0, "r": 0, "hr": 0, "rbi": 0, "sb": 0},
        ] * 2  # 50 PA total
        result = blend_player_with_game_logs(player, logs, "2026-04-06")
        # With 50 PA and reliability of 200, actual weight = 50/250 = 20%
        # HR projection should decrease but still be near projection
        assert result.ros.hr < player.ros.hr
        assert result.ros.hr > 0  # not fully actual (which is 0)
        # AVG should be pulled down
        assert result.ros.avg < player.ros.avg

    def test_blends_pitcher_era_toward_actuals(self):
        # Projection: 2.77 ERA
        player = _pitcher("Reliever", ip=65, era=2.77, er=20)
        # Actuals: 5 IP, terrible ERA (9.00)
        logs = [
            {"date": "2026-04-01", "ip": 1.0, "k": 1, "er": 1, "bb": 1,
             "h_allowed": 2, "w": 0, "sv": 0, "gs": 0, "g": 1},
            {"date": "2026-04-02", "ip": 1.0, "k": 0, "er": 1, "bb": 0,
             "h_allowed": 1, "w": 0, "sv": 0, "gs": 0, "g": 1},
            {"date": "2026-04-03", "ip": 1.0, "k": 1, "er": 1, "bb": 1,
             "h_allowed": 2, "w": 0, "sv": 0, "gs": 0, "g": 1},
            {"date": "2026-04-04", "ip": 1.0, "k": 2, "er": 1, "bb": 0,
             "h_allowed": 1, "w": 0, "sv": 0, "gs": 0, "g": 1},
            {"date": "2026-04-05", "ip": 1.0, "k": 1, "er": 1, "bb": 1,
             "h_allowed": 2, "w": 0, "sv": 0, "gs": 0, "g": 1},
        ]
        result = blend_player_with_game_logs(player, logs, "2026-04-06")
        # ERA should be pulled up from 2.77 toward 9.00 but stay closer to projection
        # With 5 IP and reliability of 120: actual_weight = 5/125 = 4%
        assert result.ros.era > player.ros.era
        assert result.ros.era < 9.0

    def test_preserves_player_metadata(self):
        player = _pitcher("Test Guy", ip=65, era=3.00, sv=20)
        player.team = "HOU"
        player.wsgp = 1.5
        result = blend_player_with_game_logs(player, [], "2026-04-06")
        assert result.name == "Test Guy"
        assert result.team == "HOU"
        assert result.positions == ["RP"]
        assert result.player_type == PlayerType.PITCHER

    def test_handles_player_with_no_ros(self):
        player = Player(name="Nobody", player_type=PlayerType.HITTER, positions=["OF"])
        result = blend_player_with_game_logs(player, [], "2026-04-06")
        assert result.ros is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lineup/test_blending.py::TestBlendPlayerWithGameLogs -v`
Expected: FAIL — `ImportError: cannot import name 'blend_player_with_game_logs'`

- [ ] **Step 3: Write the implementation**

Add to `src/fantasy_baseball/lineup/blending.py`:

```python
import copy

from fantasy_baseball.analysis.recency import predict_reliability_blend
from fantasy_baseball.models.player import (
    HitterStats, PitcherStats, Player, PlayerType,
)
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip


def blend_player_with_game_logs(
    player: Player,
    game_logs: list[dict],
    cutoff: str,
) -> Player:
    """Apply reliability-weighted recency blend to a Player's ROS stats.

    Converts the Player's ROS stats to per-PA/IP projection rates,
    runs predict_reliability_blend against game log entries, and
    returns a new Player with updated ROS stats.

    If game_logs is empty or player has no ROS stats, returns a copy unchanged.
    """
    result = copy.copy(player)

    if player.ros is None or not game_logs:
        return result

    if player.player_type == PlayerType.HITTER:
        result.ros = _blend_hitter(player.ros, game_logs, cutoff)
    else:
        result.ros = _blend_pitcher(player.ros, game_logs, cutoff)

    return result


def _blend_hitter(ros: HitterStats, game_logs: list[dict], cutoff: str) -> HitterStats:
    pa, ab = ros.pa, ros.ab
    if pa <= 0 or ab <= 0:
        return copy.copy(ros)

    proj_rates = {
        "hr_per_pa": ros.hr / pa,
        "r_per_pa": ros.r / pa,
        "rbi_per_pa": ros.rbi / pa,
        "sb_per_pa": ros.sb / pa,
        "avg": ros.avg,
    }
    rates = predict_reliability_blend(proj_rates, game_logs, cutoff)

    return HitterStats(
        pa=pa,
        ab=ab,
        h=rates["avg"] * ab,
        r=rates["r_per_pa"] * pa,
        hr=rates["hr_per_pa"] * pa,
        rbi=rates["rbi_per_pa"] * pa,
        sb=rates["sb_per_pa"] * pa,
        avg=rates["avg"],
    )


def _blend_pitcher(ros: PitcherStats, game_logs: list[dict], cutoff: str) -> PitcherStats:
    ip = ros.ip
    if ip <= 0:
        return copy.copy(ros)

    # Estimate G and GS from projection shape (not stored in PitcherStats).
    # Relievers: SV > 0 and W < 5 → ~1 IP/game.  Starters: ~6 IP/start.
    if ros.sv > 0 and ros.w < 5:
        est_g = max(ip, 1)
        est_gs = 0.0
    else:
        est_gs = max(ip / 6, 1)
        est_g = est_gs

    proj_rates = {
        "k_per_ip": ros.k / ip,
        "era": ros.era,
        "whip": ros.whip,
        "w_per_gs": ros.w / est_gs if est_gs > 0 else 0,
        "sv_per_g": ros.sv / est_g if est_g > 0 else 0,
    }
    rates = predict_reliability_blend(proj_rates, game_logs, cutoff)

    # Convert blended rates back to counting stats
    blended_k = rates["k_per_ip"] * ip
    blended_era = rates["era"]
    blended_whip = rates["whip"]
    blended_er = blended_era * ip / 9
    # Approximate BB/H split from WHIP (60/40 H/BB typical split)
    blended_bb = blended_whip * ip * 0.4
    blended_h_allowed = blended_whip * ip * 0.6
    blended_w = rates["w_per_gs"] * est_gs if est_gs > 0 else ros.w
    blended_sv = rates["sv_per_g"] * est_g if est_g > 0 else ros.sv

    return PitcherStats(
        ip=ip,
        w=blended_w,
        k=blended_k,
        sv=blended_sv,
        er=blended_er,
        bb=blended_bb,
        h_allowed=blended_h_allowed,
        era=blended_era,
        whip=blended_whip,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_blending.py -v`
Expected: 8 PASSED

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/lineup/blending.py tests/test_lineup/test_blending.py
git commit -m "feat: add blend_player_with_game_logs for Player-level recency blending"
```

---

### Task 3: Roster audit core logic

**Files:**
- Create: `src/fantasy_baseball/lineup/roster_audit.py`
- Test: `tests/test_lineup/test_roster_audit.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_lineup/test_roster_audit.py

from fantasy_baseball.models.player import Player, PlayerType, HitterStats, PitcherStats
from fantasy_baseball.lineup.roster_audit import audit_roster


EQUAL_LEVERAGE = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}

ROSTER_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "OF": 3, "UTIL": 1, "P": 3, "BN": 2, "IL": 0}


def _hitter(name, positions, **stats):
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=positions,
        ros=HitterStats(
            pa=int(stats.get("ab", 500) * 1.15),
            ab=stats.get("ab", 500), h=stats.get("h", 130),
            r=stats.get("r", 70), hr=stats.get("hr", 20),
            rbi=stats.get("rbi", 70), sb=stats.get("sb", 5),
            avg=stats.get("avg", 0.260),
        ),
    )


def _pitcher(name, positions, **stats):
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=positions,
        ros=PitcherStats(
            ip=stats.get("ip", 60.0), w=stats.get("w", 3.0),
            k=stats.get("k", 60.0), sv=stats.get("sv", 0.0),
            er=stats.get("er", 20.0), bb=stats.get("bb", 20.0),
            h_allowed=stats.get("h_allowed", 50.0),
            era=stats.get("era", 3.00), whip=stats.get("whip", 1.17),
        ),
    )


class TestAuditRoster:
    def test_identifies_upgrade_available(self):
        roster = [
            _hitter("Weak OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
            _pitcher("Decent SP", ["SP"], ip=180, w=12, k=180, era=3.50, whip=1.20,
                     er=70, bb=40, h_allowed=176),
            _pitcher("Decent SP2", ["SP"], ip=170, w=10, k=160, era=3.60, whip=1.22,
                     er=68, bb=40, h_allowed=167),
            _pitcher("Decent RP", ["RP"], ip=60, w=3, k=60, era=3.00, whip=1.17,
                     sv=20, er=20, bb=20, h_allowed=50),
        ]
        free_agents = [
            _hitter("Better OF", ["OF"], r=80, hr=28, rbi=85, sb=12, avg=0.280, ab=550, h=154),
        ]
        results = audit_roster(roster, free_agents, EQUAL_LEVERAGE, ROSTER_SLOTS)

        # Should have an entry for every roster player
        assert len(results) == len(roster)

        # The weak OF should have an upgrade identified
        weak_entry = next(e for e in results if e["player"] == "Weak OF")
        assert weak_entry["best_fa"] == "Better OF"
        assert weak_entry["gap"] > 0

    def test_shows_no_better_option(self):
        roster = [
            _hitter("Star OF", ["OF"], r=100, hr=40, rbi=110, sb=20, avg=0.300, ab=550, h=165),
        ]
        free_agents = [
            _hitter("Scrub", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
        ]
        results = audit_roster(roster, free_agents, EQUAL_LEVERAGE,
                               {"OF": 1, "P": 0, "BN": 0, "IL": 0})
        assert len(results) == 1
        assert results[0]["best_fa"] is None
        assert results[0]["gap"] == 0.0

    def test_sorted_by_gap_descending(self):
        roster = [
            _hitter("OK 1B", ["1B"], r=60, hr=15, rbi=55, sb=3, avg=0.255, ab=480, h=122),
            _hitter("Bad OF", ["OF"], r=30, hr=5, rbi=20, sb=1, avg=0.220, ab=300, h=66),
        ]
        free_agents = [
            _hitter("Good 1B", ["1B"], r=75, hr=22, rbi=70, sb=5, avg=0.270, ab=520, h=140),
            _hitter("Great OF", ["OF"], r=90, hr=30, rbi=85, sb=10, avg=0.285, ab=550, h=157),
        ]
        results = audit_roster(roster, free_agents, EQUAL_LEVERAGE,
                               {"1B": 1, "OF": 1, "P": 0, "BN": 0, "IL": 0})
        gaps = [e["gap"] for e in results]
        assert gaps == sorted(gaps, reverse=True)

    def test_empty_free_agents_all_no_upgrade(self):
        roster = [
            _hitter("Solo", ["OF"], r=70, hr=20, rbi=65, sb=8, avg=0.270, ab=500, h=135),
        ]
        results = audit_roster(roster, [], EQUAL_LEVERAGE,
                               {"OF": 1, "P": 0, "BN": 0, "IL": 0})
        assert len(results) == 1
        assert results[0]["best_fa"] is None
        assert results[0]["gap"] == 0.0

    def test_cross_type_swap_pitcher_slot(self):
        """A starter could replace a weak reliever if it produces more team wSGP."""
        roster = [
            _hitter("Hitter", ["OF"], r=80, hr=25, rbi=80, sb=10, avg=0.275, ab=540, h=149),
            _pitcher("Bad RP", ["RP"], ip=30, w=1, k=20, sv=2, era=5.50, whip=1.60,
                     er=18, bb=15, h_allowed=33),
            _pitcher("OK SP", ["SP"], ip=150, w=9, k=140, era=3.80, whip=1.25,
                     er=63, bb=40, h_allowed=148),
        ]
        free_agents = [
            _pitcher("Good SP", ["SP"], ip=180, w=12, k=180, era=3.20, whip=1.10,
                     er=64, bb=30, h_allowed=168),
        ]
        results = audit_roster(roster, free_agents, EQUAL_LEVERAGE,
                               {"OF": 1, "P": 2, "BN": 1, "IL": 0})
        # The bad RP should have the Good SP as best_fa (cross-type upgrade)
        bad_rp_entry = next(e for e in results if e["player"] == "Bad RP")
        assert bad_rp_entry["best_fa"] == "Good SP"
        assert bad_rp_entry["gap"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_lineup/test_roster_audit.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'fantasy_baseball.lineup.roster_audit'`

- [ ] **Step 3: Write the implementation**

```python
# src/fantasy_baseball/lineup/roster_audit.py

"""Roster audit — evaluate every roster slot against the best available FA."""

from __future__ import annotations

from fantasy_baseball.lineup.waivers import (
    _compute_team_wsgp,
    _build_lineup_summary,
    evaluate_pickup,
)
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.positions import can_cover_slots


def audit_roster(
    roster: list[Player],
    free_agents: list[Player],
    leverage: dict[str, float],
    roster_slots: dict[str, int],
) -> list[dict]:
    """Evaluate every roster slot against the best available FA.

    For each roster player, finds the FA that produces the largest team
    wSGP gain when swapped in.  Returns an entry for every roster player,
    sorted by gap descending (biggest problems first).  Entries with no
    upgrade available have gap=0.0 and best_fa=None.
    """
    if not roster:
        return []

    denoms = get_sgp_denominators()

    # Baseline optimal lineup
    baseline = _compute_team_wsgp(roster, leverage, roster_slots, denoms=denoms)
    baseline_wsgp = baseline["total_wsgp"]
    baseline_summary = _build_lineup_summary(
        baseline["hitter_lineup"], baseline["pitcher_starters"],
        baseline["player_wsgp"], [p.name for p in roster],
    )

    # Map player name → assigned slot from baseline
    slot_lookup = {e["name"]: e["slot"] for e in baseline_summary}

    # Pre-compute FA wSGP
    fa_wsgp: dict[str, float] = {}
    for fa in free_agents:
        fa_wsgp[fa.name] = calculate_weighted_sgp(fa.ros, leverage, denoms=denoms)

    p_slots = roster_slots.get("P", 9)

    entries: list[dict] = []
    for player in roster:
        entry = {
            "player": player.name,
            "player_type": player.player_type.value,
            "positions": list(player.positions),
            "slot": slot_lookup.get(player.name, "BN"),
            "player_wsgp": round(baseline["player_wsgp"].get(player.name, 0.0), 2),
            "best_fa": None,
            "best_fa_type": None,
            "best_fa_positions": None,
            "best_fa_wsgp": None,
            "gap": 0.0,
            "categories": {},
        }

        best_gain = 0.0
        best_fa_player = None
        best_new_result = None

        # Pre-build wSGP dict without this player for swap simulation
        base_wsgp = {k: v for k, v in baseline["player_wsgp"].items()
                     if k != player.name}

        for fa in free_agents:
            # Quick skip: FA not better than this player individually
            if fa_wsgp.get(fa.name, 0) <= entry["player_wsgp"]:
                continue

            new_roster = [p for p in roster if p.name != player.name] + [fa]
            new_hitters = [p for p in new_roster if p.player_type != PlayerType.PITCHER]
            new_pitchers = [p for p in new_roster if p.player_type == PlayerType.PITCHER]

            # Position feasibility
            if player.player_type == PlayerType.HITTER or fa.player_type == PlayerType.HITTER:
                hitter_positions = [list(p.positions) for p in new_hitters]
                if not can_cover_slots(hitter_positions, roster_slots):
                    continue
            if player.player_type == PlayerType.PITCHER or fa.player_type == PlayerType.PITCHER:
                if len(new_pitchers) < p_slots:
                    continue

            swap_wsgp = dict(base_wsgp)
            swap_wsgp[fa.name] = fa_wsgp[fa.name]

            new_result = _compute_team_wsgp(
                new_roster, leverage, roster_slots,
                denoms=denoms, player_wsgp=swap_wsgp,
            )
            gain = round(new_result["total_wsgp"] - baseline_wsgp, 2)

            if gain > best_gain:
                best_gain = gain
                best_fa_player = fa
                best_new_result = new_result

        if best_fa_player and best_new_result:
            cat_result = evaluate_pickup(best_fa_player, player, leverage)
            entry["best_fa"] = best_fa_player.name
            entry["best_fa_type"] = best_fa_player.player_type.value
            entry["best_fa_positions"] = list(best_fa_player.positions)
            entry["best_fa_wsgp"] = round(fa_wsgp.get(best_fa_player.name, 0.0), 2)
            entry["gap"] = best_gain
            entry["categories"] = cat_result["categories"]

        entries.append(entry)

    entries.sort(key=lambda e: e["gap"], reverse=True)
    return entries
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_lineup/test_roster_audit.py -v`
Expected: 5 PASSED

- [ ] **Step 5: Run all lineup tests to check for regressions**

Run: `pytest tests/test_lineup/ -v`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/lineup/roster_audit.py tests/test_lineup/test_roster_audit.py
git commit -m "feat: add roster audit module — per-slot evaluation against FA pool"
```

---

### Task 4: Integrate blending + audit into refresh pipeline

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py`

This task modifies the `run_full_refresh` function to:
1. Load per-game logs after game log fetch
2. Blend roster players with game logs
3. Blend FA players with game logs after FA fetch
4. Run roster audit and cache results

- [ ] **Step 1: Add imports and load game logs after fetch**

In `src/fantasy_baseball/web/season_data.py`, find the lazy import block near line 656 (inside `run_full_refresh`). Add `blend_player_with_game_logs` and `load_game_logs_by_name` to the imports:

```python
        from fantasy_baseball.lineup.blending import (
            blend_player_with_game_logs,
            load_game_logs_by_name,
        )
        from fantasy_baseball.lineup.roster_audit import audit_roster
```

Then after step 6b (the `fetch_and_load_game_logs` call around line 866-871) and after step 6c (pace computation), add the game log loading for blending:

```python
        # --- Step 6d: Load per-game logs for recency blending ---
        _progress("Loading game logs for recency blending...")
        gl_conn = get_db_connection()
        try:
            game_logs_by_name = load_game_logs_by_name(gl_conn, config.season_year)
        finally:
            gl_conn.close()

        today = dt.now().strftime("%Y-%m-%d")
        blended_count = 0
        for i, player in enumerate(roster_players):
            logs = game_logs_by_name.get(normalize_name(player.name), [])
            if logs:
                roster_players[i] = blend_player_with_game_logs(player, logs, today)
                roster_players[i].compute_wsgp(leverage)
                blended_count += 1
        if blended_count:
            _progress(f"Blended {blended_count} roster players with game logs")
```

- [ ] **Step 2: Blend FAs and run audit after FA fetch**

Find the waiver scan section (around line 989-1012). After `fetch_and_match_free_agents` returns `fa_players` and before `scan_waivers` is called, add FA blending and the audit:

```python
        # Blend FA players with game logs
        fa_blended = 0
        for i, fa in enumerate(fa_players):
            logs = game_logs_by_name.get(normalize_name(fa.name), [])
            if logs:
                fa_players[i] = blend_player_with_game_logs(fa, logs, today)
                fa_blended += 1
        if fa_blended:
            _progress(f"Blended {fa_blended} free agents with game logs")

        # --- Roster Audit ---
        _progress("Running roster audit...")
        audit_results = audit_roster(
            roster_players, fa_players, leverage, config.roster_slots,
        )
        write_cache("roster_audit", audit_results, cache_dir)
        upgrades = sum(1 for e in audit_results if e["gap"] > 0)
        _progress(f"Roster audit complete: {upgrades} upgrade(s) found")
```

- [ ] **Step 3: Run the full test suite to check for regressions**

Run: `pytest tests/ -v --timeout=30`
Expected: All existing tests pass (the new integration code won't run in unit tests since it's inside `run_full_refresh` which requires Yahoo auth)

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "feat: integrate recency blending + roster audit into refresh pipeline"
```

---

### Task 5: Web page — route, template, and nav link

**Files:**
- Modify: `src/fantasy_baseball/web/season_routes.py`
- Create: `src/fantasy_baseball/web/templates/season/roster_audit.html`
- Modify: `src/fantasy_baseball/web/templates/season/base.html`

- [ ] **Step 1: Add route to season_routes.py**

In `src/fantasy_baseball/web/season_routes.py`, find the `waivers_trades` route (around line 333). Add the roster audit route before it:

```python
    @app.route("/roster-audit")
    def roster_audit():
        meta = read_meta()
        audit_raw = read_cache("roster_audit")
        return render_template(
            "season/roster_audit.html",
            meta=meta,
            active_page="roster_audit",
            audit=audit_raw or [],
            categories=ALL_CATEGORIES,
        )
```

- [ ] **Step 2: Create the template**

```html
{# src/fantasy_baseball/web/templates/season/roster_audit.html #}
{% extends "season/base.html" %}
{% block title %}Roster Audit — Season Dashboard{% endblock %}
{% block content %}

<style>
.audit-table { width: 100%; border-collapse: collapse; font-size: 13px; }
.audit-table th {
    text-align: left; font-size: 11px; color: var(--text-secondary);
    font-weight: 500; text-transform: uppercase; padding: 6px 8px;
    border-bottom: 1px solid var(--panel-border);
}
.audit-table td { padding: 8px; border-bottom: 1px solid var(--panel-border); }
.audit-row { cursor: pointer; }
.audit-row:hover { background: rgba(255, 255, 255, 0.03); }
.audit-row.has-upgrade { background: rgba(251, 191, 36, 0.05); }
.audit-row.has-upgrade:hover { background: rgba(251, 191, 36, 0.10); }
.gap-badge {
    font-weight: 600; font-size: 12px; padding: 2px 8px;
    border-radius: 3px; display: inline-block;
}
.gap-high { background: rgba(239, 68, 68, 0.15); color: #fca5a5; }
.gap-medium { background: rgba(251, 191, 36, 0.15); color: #fcd34d; }
.gap-low { background: rgba(34, 197, 94, 0.15); color: #86efac; }
.gap-none { color: var(--text-secondary); font-size: 12px; }
.audit-detail { display: none; }
.audit-detail.open { display: table-row; }
.audit-detail td { padding: 12px 8px; background: rgba(0,0,0,0.2); }
.cat-impact { display: flex; gap: 6px; flex-wrap: wrap; margin-top: 6px; }
.cat-gain { color: var(--success); font-size: 11px; font-weight: 600; }
.cat-loss { color: var(--danger); font-size: 11px; font-weight: 600; }
.player-pos { color: var(--text-secondary); font-size: 11px; margin-left: 4px; }
.player-type-tag {
    font-size: 10px; padding: 1px 5px; border-radius: 3px;
    background: rgba(255,255,255,0.08); color: var(--text-secondary);
    margin-left: 4px;
}
</style>

<div class="page-header">
    <h2>Roster Audit</h2>
    {% if meta and meta.get('week') %}
    <div class="week-label">Week {{ meta['week'] }}</div>
    {% endif %}
</div>

{% if not audit %}
<p class="placeholder-text">No roster audit data. Click "Refresh Data" to generate the audit.</p>
{% else %}

{% set upgrade_count = audit | selectattr('gap', '>', 0) | list | length %}
{% if upgrade_count > 0 %}
<p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 16px;">
    {{ upgrade_count }} upgrade{{ 's' if upgrade_count != 1 }} available on the waiver wire.
    Sorted by impact — biggest improvements first.
</p>
{% else %}
<p style="font-size: 13px; color: var(--text-secondary); margin-bottom: 16px;">
    No upgrades found — your roster is the best available at every slot.
</p>
{% endif %}

<table class="audit-table">
    <thead>
        <tr>
            <th>Slot</th>
            <th>Your Player</th>
            <th>wSGP</th>
            <th>Best Available</th>
            <th>FA wSGP</th>
            <th>Gap</th>
        </tr>
    </thead>
    <tbody>
    {% for entry in audit %}
    <tr class="audit-row {% if entry.gap > 0 %}has-upgrade{% endif %}"
        onclick="toggleAuditDetail('audit-detail-{{ loop.index }}')">
        <td>{{ entry.slot }}</td>
        <td>
            {{ entry.player }}
            <span class="player-pos">({{ entry.positions | join(", ") }})</span>
        </td>
        <td>{{ "%.2f"|format(entry.player_wsgp) }}</td>
        {% if entry.best_fa %}
        <td>
            {{ entry.best_fa }}
            <span class="player-pos">({{ entry.best_fa_positions | join(", ") }})</span>
            {% if entry.best_fa_type != entry.player_type %}
            <span class="player-type-tag">{{ entry.best_fa_type }}</span>
            {% endif %}
        </td>
        <td>{{ "%.2f"|format(entry.best_fa_wsgp) }}</td>
        <td>
            {% if entry.gap >= 0.5 %}
            <span class="gap-badge gap-high">+{{ "%.2f"|format(entry.gap) }}</span>
            {% elif entry.gap >= 0.2 %}
            <span class="gap-badge gap-medium">+{{ "%.2f"|format(entry.gap) }}</span>
            {% else %}
            <span class="gap-badge gap-low">+{{ "%.2f"|format(entry.gap) }}</span>
            {% endif %}
        </td>
        {% else %}
        <td colspan="2" style="color: var(--text-secondary); font-style: italic;">—</td>
        <td><span class="gap-none">No better option</span></td>
        {% endif %}
    </tr>
    {% if entry.best_fa and entry.categories %}
    <tr class="audit-detail" id="audit-detail-{{ loop.index }}">
        <td colspan="6">
            <div style="font-size: 12px; font-weight: 600; margin-bottom: 4px;">Category impact of swapping {{ entry.player }} &rarr; {{ entry.best_fa }}</div>
            <div class="cat-impact">
                {% for cat, delta in entry.categories.items() %}
                {% if delta > 0.01 %}
                <span class="cat-gain">{{ cat }} +{{ "%.2f"|format(delta) }}</span>
                {% elif delta < -0.01 %}
                <span class="cat-loss">{{ cat }} {{ "%.2f"|format(delta) }}</span>
                {% endif %}
                {% endfor %}
            </div>
        </td>
    </tr>
    {% endif %}
    {% endfor %}
    </tbody>
</table>
{% endif %}

<script>
function toggleAuditDetail(id) {
    const row = document.getElementById(id);
    if (row) row.classList.toggle('open');
}
</script>

{% endblock %}
```

- [ ] **Step 3: Add nav link to base.html**

In `src/fantasy_baseball/web/templates/season/base.html`, find the "Waivers & Trades" nav link (around line 23). Add the Roster Audit link between "Lineup" and "Waivers & Trades":

```html
            <a href="{{ url_for('roster_audit') }}"
               class="nav-link {% if active_page == 'roster_audit' %}active{% endif %}">
                Roster Audit
            </a>
```

- [ ] **Step 4: Verify the template renders without errors**

Run: `python -c "from fantasy_baseball.web.season_app import create_app; app = create_app(); app.test_client().get('/roster-audit')"`
Expected: 200 OK (empty audit, placeholder text shown)

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/web/season_routes.py src/fantasy_baseball/web/templates/season/roster_audit.html src/fantasy_baseball/web/templates/season/base.html
git commit -m "feat: add /roster-audit page with per-slot evaluation table"
```

---

### Task 6: Final integration test and cleanup

**Files:**
- Test: `tests/test_lineup/test_blending.py` (verify round-trip)
- Test: `tests/test_lineup/test_roster_audit.py` (verify with blended data)

- [ ] **Step 1: Add round-trip integration test**

Append to `tests/test_lineup/test_blending.py`:

```python
class TestBlendingRoundTrip:
    """Verify blend → wSGP calculation works end-to-end."""

    def test_blended_pitcher_produces_valid_wsgp(self):
        from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
        leverage = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}

        player = _pitcher("Test RP", ip=65, w=4, k=70, sv=20, era=2.77, whip=1.11,
                          er=20, bb=22, h_allowed=50)
        logs = [
            {"date": f"2026-04-0{d}", "ip": 1.0, "k": 1, "er": 1, "bb": 1,
             "h_allowed": 2, "w": 0, "sv": 0, "gs": 0, "g": 1}
            for d in range(1, 6)
        ]
        blended = blend_player_with_game_logs(player, logs, "2026-04-06")
        wsgp = calculate_weighted_sgp(blended.ros, leverage)
        original_wsgp = calculate_weighted_sgp(player.ros, leverage)

        # Blended ERA is worse, so wSGP should decrease
        assert wsgp < original_wsgp
        assert wsgp > 0  # still a positive contributor

    def test_blended_hitter_produces_valid_wsgp(self):
        from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
        leverage = {cat: 0.1 for cat in ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]}

        player = _hitter("Test OF", pa=600, ab=540, h=150, hr=25, r=80, rbi=80, sb=10, avg=0.278)
        # Hot start: lots of HRs
        logs = [
            {"date": f"2026-04-0{d}", "pa": 5, "ab": 4, "h": 2, "r": 1, "hr": 1, "rbi": 2, "sb": 0}
            for d in range(1, 6)
        ]
        blended = blend_player_with_game_logs(player, logs, "2026-04-06")
        wsgp = calculate_weighted_sgp(blended.ros, leverage)
        original_wsgp = calculate_weighted_sgp(player.ros, leverage)

        # Hot start should increase wSGP slightly
        assert wsgp > original_wsgp
```

- [ ] **Step 2: Run all tests**

Run: `pytest tests/ -v`
Expected: All pass

- [ ] **Step 3: Commit**

```bash
git add tests/test_lineup/test_blending.py
git commit -m "test: add blending round-trip integration tests"
```
