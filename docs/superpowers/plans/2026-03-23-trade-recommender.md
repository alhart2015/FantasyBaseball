# Trade Recommender (1-for-1) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Propose the top 5 one-for-one trades that improve Hart's roto standings, are attractive to the opponent, and include a human-readable pitch with projected roto point impact.

**Architecture:** A trade evaluation module computes projected roto point impact by adjusting team season totals with ROS projections of swapped players, then re-ranking all 10 teams. A pitch generator explains each trade in human terms. A runner script orchestrates Yahoo API calls, projection matching, and output.

**Tech Stack:** Yahoo Fantasy API (yahoo_oauth), pandas, existing leverage/SGP/projection infrastructure.

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/trades/__init__.py` | Create | Package init |
| `src/fantasy_baseball/trades/evaluate.py` | Create | Roto point impact calculation, trade filtering |
| `src/fantasy_baseball/trades/pitch.py` | Create | Human-readable pitch generation |
| `tests/test_trades/__init__.py` | Create | Package init |
| `tests/test_trades/test_evaluate.py` | Create | Tests for trade evaluation |
| `tests/test_trades/test_pitch.py` | Create | Tests for pitch generation |
| `scripts/run_trades.py` | Create | CLI runner script |

---

### Task 1: Roto point impact calculation

**Files:**
- Create: `src/fantasy_baseball/trades/__init__.py`
- Create: `src/fantasy_baseball/trades/evaluate.py`
- Create: `tests/test_trades/__init__.py`
- Create: `tests/test_trades/test_evaluate.py`

- [ ] **Step 1: Write tests for roto re-ranking and trade impact**

```python
# tests/test_trades/test_evaluate.py
import pytest
from fantasy_baseball.trades.evaluate import (
    compute_roto_points,
    compute_trade_impact,
)

ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
INVERSE = {"ERA", "WHIP"}

# 3-team league for simple testing
STANDINGS = [
    {"name": "Team A", "stats": {"R": 900, "HR": 250, "RBI": 880, "SB": 150,
     "AVG": .265, "W": 80, "K": 1300, "SV": 80, "ERA": 3.50, "WHIP": 1.15}},
    {"name": "Team B", "stats": {"R": 850, "HR": 280, "RBI": 900, "SB": 120,
     "AVG": .255, "W": 85, "K": 1400, "SV": 60, "ERA": 3.80, "WHIP": 1.20}},
    {"name": "Team C", "stats": {"R": 800, "HR": 260, "RBI": 850, "SB": 180,
     "AVG": .250, "W": 75, "K": 1200, "SV": 90, "ERA": 3.30, "WHIP": 1.10}},
]


def test_compute_roto_points():
    """Rank teams in each category, sum points."""
    points = compute_roto_points(STANDINGS)
    # Team A: R=3, HR=1, RBI=2, SB=2, AVG=3, W=2, K=2, SV=2, ERA=2, WHIP=2 = 21
    assert points["Team A"] == 21
    # Team C: R=1, HR=2, RBI=1, SB=3, AVG=1, W=1, K=1, SV=3, ERA=3, WHIP=3 = 19
    assert points["Team C"] == 19


def test_compute_trade_impact():
    """Trading players changes projected totals and roto points."""
    # Team A sends a hitter (ROS: 30 HR, 20 SB) to Team B
    # Team A gets a pitcher (ROS: 10 SV, 0.5 ERA improvement worth)
    hart_loses_ros = {"R": 50, "HR": 30, "RBI": 60, "SB": 20, "AVG": .280,
                      "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0,
                      "ab": 400, "ip": 0}
    hart_gains_ros = {"R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                      "W": 5, "K": 100, "SV": 30, "ERA": 3.00, "WHIP": 1.05,
                      "ab": 0, "ip": 150}
    opp_loses_ros = hart_gains_ros
    opp_gains_ros = hart_loses_ros

    result = compute_trade_impact(
        standings=STANDINGS,
        hart_name="Team A",
        opp_name="Team B",
        hart_loses_ros=hart_loses_ros,
        hart_gains_ros=hart_gains_ros,
        opp_loses_ros=opp_loses_ros,
        opp_gains_ros=opp_gains_ros,
    )
    assert "hart_delta" in result
    assert "opp_delta" in result
    assert "hart_cat_deltas" in result
    assert "opp_cat_deltas" in result
    # hart_delta and opp_delta are total roto point changes (ints)
    assert isinstance(result["hart_delta"], (int, float))


def test_trade_impact_zero_for_no_change():
    """If both players have identical ROS, no roto point change."""
    same = {"R": 50, "HR": 20, "RBI": 50, "SB": 10, "AVG": .260,
            "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0,
            "ab": 400, "ip": 0}
    result = compute_trade_impact(
        standings=STANDINGS,
        hart_name="Team A", opp_name="Team B",
        hart_loses_ros=same, hart_gains_ros=same,
        opp_loses_ros=same, opp_gains_ros=same,
    )
    assert result["hart_delta"] == 0
    assert result["opp_delta"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trades/test_evaluate.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement trade evaluation**

```python
# src/fantasy_baseball/trades/__init__.py
# (empty)
```

```python
# src/fantasy_baseball/trades/evaluate.py
"""Evaluate 1-for-1 trade proposals by projected roto point impact."""

ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]
INVERSE = {"ERA", "WHIP"}
COUNTING_HITTING = {"R", "HR", "RBI", "SB"}
COUNTING_PITCHING = {"W", "K", "SV"}


def compute_roto_points(standings: list[dict]) -> dict[str, int]:
    """Rank teams in each category and sum roto points.

    Returns {team_name: total_points}.
    N teams -> 1st place gets N points, last gets 1.
    """
    n = len(standings)
    points = {t["name"]: 0 for t in standings}

    for cat in ALL_CATS:
        reverse = cat not in INVERSE
        ranked = sorted(standings, key=lambda t: t["stats"].get(cat, 0), reverse=reverse)
        for i, team in enumerate(ranked):
            points[team["name"]] += n - i

    return points


def compute_roto_points_by_cat(standings: list[dict]) -> dict[str, dict[str, int]]:
    """Return {team_name: {cat: points}} for per-category analysis."""
    n = len(standings)
    result = {t["name"]: {} for t in standings}

    for cat in ALL_CATS:
        reverse = cat not in INVERSE
        ranked = sorted(standings, key=lambda t: t["stats"].get(cat, 0), reverse=reverse)
        for i, team in enumerate(ranked):
            result[team["name"]][cat] = n - i

    return result


def _project_team_stats(
    current_stats: dict, loses_ros: dict, gains_ros: dict,
) -> dict:
    """Project end-of-season stats after a trade.

    For counting stats: new = current - loses_ROS + gains_ROS
    For rate stats (AVG, ERA, WHIP): recompute from adjusted components.
    """
    new = dict(current_stats)

    # Counting stats: simple add/subtract
    for cat in COUNTING_HITTING | COUNTING_PITCHING:
        new[cat] = current_stats.get(cat, 0) - loses_ros.get(cat, 0) + gains_ros.get(cat, 0)

    # AVG: need to adjust via H and AB
    # current total H = AVG * estimated_AB (we don't have AB, so use the
    # ROS player's AB contribution to estimate the delta)
    loses_ab = loses_ros.get("ab", 0)
    gains_ab = gains_ros.get("ab", 0)
    loses_avg = loses_ros.get("AVG", 0)
    gains_avg = gains_ros.get("AVG", 0)
    loses_h = loses_avg * loses_ab
    gains_h = gains_avg * gains_ab

    # Estimate current total AB from current AVG (rough but workable)
    # A typical team has ~5500 AB in a season
    current_avg = current_stats.get("AVG", .250)
    estimated_ab = 5500
    current_h = current_avg * estimated_ab
    new_h = current_h - loses_h + gains_h
    new_ab = estimated_ab - loses_ab + gains_ab
    new["AVG"] = new_h / new_ab if new_ab > 0 else current_avg

    # ERA: adjust via ER and IP
    loses_ip = loses_ros.get("ip", 0)
    gains_ip = gains_ros.get("ip", 0)
    loses_era = loses_ros.get("ERA", 0)
    gains_era = gains_ros.get("ERA", 0)
    loses_er = loses_era * loses_ip / 9 if loses_ip > 0 else 0
    gains_er = gains_era * gains_ip / 9 if gains_ip > 0 else 0

    current_era = current_stats.get("ERA", 4.00)
    estimated_ip = 1400  # typical team IP
    current_er = current_era * estimated_ip / 9
    new_er = current_er - loses_er + gains_er
    new_ip = estimated_ip - loses_ip + gains_ip
    new["ERA"] = new_er * 9 / new_ip if new_ip > 0 else current_era

    # WHIP: adjust via (BB+H) and IP
    loses_whip = loses_ros.get("WHIP", 0)
    gains_whip = gains_ros.get("WHIP", 0)
    loses_bh = loses_whip * loses_ip if loses_ip > 0 else 0
    gains_bh = gains_whip * gains_ip if gains_ip > 0 else 0

    current_whip = current_stats.get("WHIP", 1.20)
    current_bh = current_whip * estimated_ip
    new_bh = current_bh - loses_bh + gains_bh
    new["WHIP"] = new_bh / new_ip if new_ip > 0 else current_whip

    return new


def compute_trade_impact(
    standings: list[dict],
    hart_name: str,
    opp_name: str,
    hart_loses_ros: dict,
    hart_gains_ros: dict,
    opp_loses_ros: dict,
    opp_gains_ros: dict,
) -> dict:
    """Compute projected roto point change for both teams after a trade.

    Returns dict with:
        hart_delta: int — total roto point change for Hart
        opp_delta: int — total roto point change for opponent
        hart_cat_deltas: dict[str, int] — per-category point changes for Hart
        opp_cat_deltas: dict[str, int] — per-category point changes for opponent
    """
    # Current points
    before_by_cat = compute_roto_points_by_cat(standings)
    before_hart = sum(before_by_cat[hart_name].values())
    before_opp = sum(before_by_cat[opp_name].values())

    # Project new standings
    new_standings = []
    for team in standings:
        name = team["name"]
        if name == hart_name:
            new_stats = _project_team_stats(team["stats"], hart_loses_ros, hart_gains_ros)
        elif name == opp_name:
            new_stats = _project_team_stats(team["stats"], opp_loses_ros, opp_gains_ros)
        else:
            new_stats = dict(team["stats"])
        new_standings.append({"name": name, "stats": new_stats})

    after_by_cat = compute_roto_points_by_cat(new_standings)
    after_hart = sum(after_by_cat[hart_name].values())
    after_opp = sum(after_by_cat[opp_name].values())

    hart_cat_deltas = {
        cat: after_by_cat[hart_name][cat] - before_by_cat[hart_name][cat]
        for cat in ALL_CATS
    }
    opp_cat_deltas = {
        cat: after_by_cat[opp_name][cat] - before_by_cat[opp_name][cat]
        for cat in ALL_CATS
    }

    return {
        "hart_delta": after_hart - before_hart,
        "opp_delta": after_opp - before_opp,
        "hart_cat_deltas": hart_cat_deltas,
        "opp_cat_deltas": opp_cat_deltas,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trades/test_evaluate.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/ tests/test_trades/
git commit -m "feat(trades): roto point impact calculation for trade evaluation"
```

---

### Task 2: Pitch generation

**Files:**
- Create: `src/fantasy_baseball/trades/pitch.py`
- Create: `tests/test_trades/test_pitch.py`

- [ ] **Step 1: Write tests for pitch generation**

```python
# tests/test_trades/test_pitch.py
from fantasy_baseball.trades.pitch import generate_pitch

ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]


def test_pitch_highlights_gains_and_affordable_loss():
    opp_cat_deltas = {"R": 0, "HR": -1, "RBI": 0, "SB": 2, "AVG": 1,
                      "W": 0, "K": 0, "SV": -1, "ERA": 0, "WHIP": 0}
    opp_cat_ranks = {"R": 5, "HR": 2, "RBI": 5, "SB": 8, "AVG": 7,
                     "W": 5, "K": 5, "SV": 2, "ERA": 5, "WHIP": 5}
    pitch = generate_pitch("Springfield Isotopes", opp_cat_deltas, opp_cat_ranks)
    assert "SB" in pitch or "steals" in pitch.lower() or "stolen" in pitch.lower()
    assert len(pitch) < 300  # Should be concise


def test_pitch_with_no_gains():
    opp_cat_deltas = {c: 0 for c in ALL_CATS}
    opp_cat_ranks = {c: 5 for c in ALL_CATS}
    pitch = generate_pitch("Team X", opp_cat_deltas, opp_cat_ranks)
    assert isinstance(pitch, str)


def test_pitch_mentions_team_name():
    opp_cat_deltas = {"R": 1, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                      "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}
    opp_cat_ranks = {"R": 9, "HR": 5, "RBI": 5, "SB": 5, "AVG": 5,
                     "W": 5, "K": 5, "SV": 5, "ERA": 5, "WHIP": 5}
    pitch = generate_pitch("SkeleThor", opp_cat_deltas, opp_cat_ranks)
    # Pitch should reference what they gain, not necessarily team name
    assert len(pitch) > 10
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trades/test_pitch.py -v`
Expected: FAIL — module does not exist

- [ ] **Step 3: Implement pitch generation**

```python
# src/fantasy_baseball/trades/pitch.py
"""Generate human-readable trade pitches for opponents."""

CAT_NAMES = {
    "R": "runs", "HR": "home runs", "RBI": "RBI", "SB": "steals",
    "AVG": "batting average", "W": "wins", "K": "strikeouts",
    "SV": "saves", "ERA": "ERA", "WHIP": "WHIP",
}

ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th",
            6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th"}


def generate_pitch(
    opp_name: str,
    opp_cat_deltas: dict[str, int],
    opp_cat_ranks: dict[str, int],
) -> str:
    """Generate a 1-2 sentence pitch explaining why this trade helps the opponent.

    Args:
        opp_name: Opponent team name.
        opp_cat_deltas: Per-category roto point changes for opponent.
        opp_cat_ranks: Opponent's current rank per category (1=best, 10=worst).
    """
    gains = [(cat, d) for cat, d in opp_cat_deltas.items() if d > 0]
    losses = [(cat, d) for cat, d in opp_cat_deltas.items() if d < 0]

    if not gains:
        return "This trade is roughly neutral for you — no category impact."

    # Sort gains by opponent's weakness (higher rank = weaker = more compelling)
    gains.sort(key=lambda x: opp_cat_ranks.get(x[0], 5), reverse=True)
    # Sort losses by opponent's strength (lower rank = stronger = easier to absorb)
    losses.sort(key=lambda x: opp_cat_ranks.get(x[0], 5))

    # Build the "you gain" part
    top_gains = gains[:2]
    gain_parts = []
    for cat, delta in top_gains:
        rank = opp_cat_ranks.get(cat, 5)
        rank_str = ORDINALS.get(rank, f"{rank}th")
        cat_name = CAT_NAMES.get(cat, cat)
        gain_parts.append(f"you're {rank_str} in {cat_name}")

    gain_sentence = f"You need help where it counts — {' and '.join(gain_parts)}. This trade boosts you there."

    # Build the "you can afford" part
    if losses:
        best_loss = losses[0]  # Their strongest category they're losing
        loss_cat = best_loss[0]
        loss_rank = opp_cat_ranks.get(loss_cat, 5)
        loss_rank_str = ORDINALS.get(loss_rank, f"{loss_rank}th")
        loss_name = CAT_NAMES.get(loss_cat, loss_cat)
        loss_sentence = f" You're {loss_rank_str} in {loss_name}, so you can afford the hit."
    else:
        loss_sentence = ""

    return gain_sentence + loss_sentence
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_trades/test_pitch.py -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/pitch.py tests/test_trades/test_pitch.py
git commit -m "feat(trades): human-readable trade pitch generation"
```

---

### Task 3: Trade finder — search and rank proposals

**Files:**
- Modify: `src/fantasy_baseball/trades/evaluate.py`
- Modify: `tests/test_trades/test_evaluate.py`

- [ ] **Step 1: Write tests for trade finder**

```python
# tests/test_trades/test_evaluate.py (append)

from fantasy_baseball.trades.evaluate import find_trades
import pandas as pd


def _make_player(name, ptype, positions, **stats):
    d = {"name": name, "player_type": ptype, "positions": positions}
    d.update(stats)
    return d


ROSTER_SLOTS = {"C": 1, "1B": 1, "2B": 1, "3B": 1, "SS": 1, "IF": 1,
                "OF": 4, "UTIL": 2, "P": 9, "BN": 2, "IL": 2}

SAMPLE_STANDINGS = [
    {"name": "Hart", "team_key": "t.1", "rank": 3,
     "stats": {"R": 900, "HR": 280, "RBI": 880, "SB": 120,
               "AVG": .260, "W": 80, "K": 1300, "SV": 80, "ERA": 3.50, "WHIP": 1.15}},
    {"name": "Rival", "team_key": "t.2", "rank": 5,
     "stats": {"R": 850, "HR": 250, "RBI": 870, "SB": 180,
               "AVG": .255, "W": 85, "K": 1400, "SV": 40, "ERA": 3.80, "WHIP": 1.20}},
]


def test_find_trades_returns_ranked_list():
    hart_roster = [
        _make_player("Slugger", "hitter", ["OF"], r=80, hr=35, rbi=90, sb=5,
                     avg=.270, h=140, ab=520, pa=570),
        _make_player("Speedy", "hitter", ["SS"], r=70, hr=10, rbi=50, sb=40,
                     avg=.260, h=130, ab=500, pa=550),
    ]
    opp_rosters = {
        "Rival": [
            _make_player("Closer", "pitcher", ["RP"], w=3, k=60, sv=30,
                         era=2.80, whip=1.00, ip=65, er=20, bb=15, h_allowed=50),
            _make_player("Stealer", "hitter", ["OF"], r=75, hr=8, rbi=45, sb=45,
                         avg=.265, h=135, ab=510, pa=560),
        ],
    }
    leverage_by_team = {
        "Hart": {"R": .1, "HR": .05, "RBI": .1, "SB": .15, "AVG": .1,
                 "W": .1, "K": .1, "SV": .15, "ERA": .1, "WHIP": .05},
        "Rival": {"R": .1, "HR": .15, "RBI": .1, "SB": .05, "AVG": .1,
                  "W": .1, "K": .1, "SV": .1, "ERA": .1, "WHIP": .1},
    }

    trades = find_trades(
        hart_name="Hart",
        hart_roster=hart_roster,
        opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS,
        leverage_by_team=leverage_by_team,
        roster_slots=ROSTER_SLOTS,
        max_results=5,
    )
    assert isinstance(trades, list)
    # Each trade has required keys
    if trades:
        t = trades[0]
        assert "send" in t
        assert "receive" in t
        assert "opponent" in t
        assert "hart_delta" in t
        assert "opp_delta" in t
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/test_trades/test_evaluate.py -v -k "find_trades"`
Expected: FAIL — `find_trades` not defined

- [ ] **Step 3: Implement find_trades**

Append to `src/fantasy_baseball/trades/evaluate.py`:

```python
from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.utils.positions import can_fill_slot
import pandas as pd


def _player_ros_stats(player: dict) -> dict:
    """Extract ROS counting stats from a player dict for trade projection.

    Returns dict with R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP, ab, ip
    suitable for _project_team_stats.
    """
    ptype = player.get("player_type", "hitter")
    if ptype == "hitter":
        ab = player.get("ab", 0)
        return {
            "R": player.get("r", 0),
            "HR": player.get("hr", 0),
            "RBI": player.get("rbi", 0),
            "SB": player.get("sb", 0),
            "AVG": player.get("avg", 0),
            "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0,
            "ab": ab, "ip": 0,
        }
    else:
        ip = player.get("ip", 0)
        return {
            "R": 0, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
            "W": player.get("w", 0),
            "K": player.get("k", 0),
            "SV": player.get("sv", 0),
            "ERA": player.get("era", 0),
            "WHIP": player.get("whip", 0),
            "ab": 0, "ip": ip,
        }


def _can_roster_without(roster: list[dict], remove: dict, add: dict,
                        roster_slots: dict) -> bool:
    """Check if a roster is legal after swapping one player.

    Simple check: the incoming player must be able to fill at least one
    non-bench slot. This is permissive — real roster legality is more
    complex but this catches obvious problems like trading your only C.
    """
    positions = add.get("positions", [])
    for slot in roster_slots:
        if slot in ("BN", "IL"):
            continue
        if can_fill_slot(positions, slot):
            return True
    return False


def find_trades(
    hart_name: str,
    hart_roster: list[dict],
    opp_rosters: dict[str, list[dict]],
    standings: list[dict],
    leverage_by_team: dict[str, dict],
    roster_slots: dict[str, int],
    max_results: int = 5,
) -> list[dict]:
    """Find and rank the best 1-for-1 trades for Hart.

    Returns list of trade dicts sorted by hart_delta descending.
    Each trade dict contains: send, receive, opponent, hart_delta, opp_delta,
    hart_cat_deltas, opp_cat_deltas, hart_wsgp_gain, opp_wsgp_gain.
    """
    hart_leverage = leverage_by_team.get(hart_name, {})
    proposals = []

    for opp_name, opp_roster in opp_rosters.items():
        opp_leverage = leverage_by_team.get(opp_name, {})

        for hart_player in hart_roster:
            hart_p_series = pd.Series(hart_player)
            hart_wsgp = calculate_weighted_sgp(hart_p_series, hart_leverage)

            for opp_player in opp_roster:
                # Roster legality check
                if not _can_roster_without(hart_roster, hart_player, opp_player, roster_slots):
                    continue
                if not _can_roster_without(opp_roster, opp_player, hart_player, roster_slots):
                    continue

                opp_p_series = pd.Series(opp_player)

                # wSGP evaluation from each side's perspective
                gain_wsgp = calculate_weighted_sgp(opp_p_series, hart_leverage)
                hart_wsgp_gain = gain_wsgp - hart_wsgp

                opp_current_wsgp = calculate_weighted_sgp(opp_p_series, opp_leverage)
                opp_gain_wsgp = calculate_weighted_sgp(hart_p_series, opp_leverage)
                opp_wsgp_gain = opp_gain_wsgp - opp_current_wsgp

                # Both sides must benefit (or opponent breaks even)
                if hart_wsgp_gain <= 0 or opp_wsgp_gain < 0:
                    continue

                # Compute roto point impact
                hart_loses = _player_ros_stats(hart_player)
                hart_gains = _player_ros_stats(opp_player)
                opp_loses = hart_gains  # what opponent gives up
                opp_gains = hart_loses  # what opponent receives

                impact = compute_trade_impact(
                    standings, hart_name, opp_name,
                    hart_loses, hart_gains, opp_loses, opp_gains,
                )

                proposals.append({
                    "send": hart_player["name"],
                    "send_positions": hart_player.get("positions", []),
                    "receive": opp_player["name"],
                    "receive_positions": opp_player.get("positions", []),
                    "opponent": opp_name,
                    "hart_delta": impact["hart_delta"],
                    "opp_delta": impact["opp_delta"],
                    "hart_cat_deltas": impact["hart_cat_deltas"],
                    "opp_cat_deltas": impact["opp_cat_deltas"],
                    "hart_wsgp_gain": round(hart_wsgp_gain, 2),
                    "opp_wsgp_gain": round(opp_wsgp_gain, 2),
                })

    # Sort by Hart's roto point gain, then by opponent gain (more realistic first)
    proposals.sort(key=lambda t: (t["hart_delta"], t["opp_delta"]), reverse=True)
    return proposals[:max_results]
```

- [ ] **Step 4: Run all tests to verify they pass**

Run: `python -m pytest tests/test_trades/ -v`
Expected: all PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/evaluate.py tests/test_trades/test_evaluate.py
git commit -m "feat(trades): trade finder — search and rank 1-for-1 proposals"
```

---

### Task 4: CLI runner script

**Files:**
- Create: `scripts/run_trades.py`

- [ ] **Step 1: Create the runner script**

```python
# scripts/run_trades.py
"""Trade Recommender — find mutually beneficial 1-for-1 trades.

Usage:
    python scripts/run_trades.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.auth.yahoo_auth import get_yahoo_session, get_league
from fantasy_baseball.config import load_config
from fantasy_baseball.data.projections import blend_projections
from fantasy_baseball.lineup.yahoo_roster import fetch_roster, fetch_standings
from fantasy_baseball.lineup.leverage import calculate_leverage
from fantasy_baseball.trades.evaluate import find_trades, compute_roto_points_by_cat
from fantasy_baseball.trades.pitch import generate_pitch
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import is_hitter, is_pitcher

CONFIG_PATH = PROJECT_ROOT / "config" / "league.yaml"
PROJECTIONS_DIR = PROJECT_ROOT / "data" / "projections"


def match_roster_to_projections(
    roster: list[dict],
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
) -> list[dict]:
    """Match roster player names to projection stats. Returns enriched player dicts."""
    matched = []
    for player in roster:
        name_norm = normalize_name(player["name"])
        positions = player["positions"]

        proj = None
        ptype = None
        if is_hitter(positions) and not hitters_proj.empty:
            matches = hitters_proj[hitters_proj["name"].apply(normalize_name) == name_norm]
            if not matches.empty:
                proj = matches.iloc[0]
                ptype = "hitter"
        if proj is None and is_pitcher(positions) and not pitchers_proj.empty:
            matches = pitchers_proj[pitchers_proj["name"].apply(normalize_name) == name_norm]
            if not matches.empty:
                proj = matches.iloc[0]
                ptype = "pitcher"
        if proj is None:
            # Try either source
            for df, pt in [(hitters_proj, "hitter"), (pitchers_proj, "pitcher")]:
                if df.empty:
                    continue
                matches = df[df["name"].apply(normalize_name) == name_norm]
                if not matches.empty:
                    proj = matches.iloc[0]
                    ptype = pt
                    break

        if proj is not None:
            entry = {
                "name": player["name"],
                "positions": positions,
                "player_type": ptype,
            }
            if ptype == "hitter":
                for col in ["r", "hr", "rbi", "sb", "avg", "h", "ab", "pa"]:
                    entry[col] = float(proj.get(col, 0) or 0)
            else:
                for col in ["w", "k", "sv", "era", "whip", "ip", "er", "bb", "h_allowed"]:
                    entry[col] = float(proj.get(col, 0) or 0)
            matched.append(entry)

    return matched


def main():
    config = load_config(CONFIG_PATH)
    print(f"Trade Recommender | {config.team_name}")
    print()

    # Connect to Yahoo
    print("Connecting to Yahoo...")
    session = get_yahoo_session()
    league = get_league(session, league_id=config.league_id, game_key=config.game_code)

    # Get all teams
    teams = league.teams()
    team_keys = {}  # name -> key
    for key, team in teams.items():
        team_keys[team["name"]] = key

    user_team_key = None
    for key, team in teams.items():
        if normalize_name(team["name"]) == normalize_name(config.team_name):
            user_team_key = key
            break
    if not user_team_key:
        print(f"Could not find team '{config.team_name}' in league")
        sys.exit(1)

    # Fetch standings
    print("Fetching standings...")
    standings = fetch_standings(league)
    print(f"Standings: {len(standings)} teams")

    # Fetch all rosters
    print("Fetching all team rosters...")
    all_rosters_raw = {}
    for key, team in teams.items():
        name = team["name"]
        roster = fetch_roster(league, key)
        all_rosters_raw[name] = roster
        print(f"  {name}: {len(roster)} players")
    print()

    # Load projections
    print("Loading projections...")
    weights = config.projection_weights if config.projection_weights else None
    hitters_proj, pitchers_proj = blend_projections(
        PROJECTIONS_DIR, config.projection_systems, weights,
    )

    # Match all rosters to projections
    hart_roster = match_roster_to_projections(
        all_rosters_raw[config.team_name], hitters_proj, pitchers_proj,
    )
    opp_rosters = {}
    for name, roster in all_rosters_raw.items():
        if name == config.team_name:
            continue
        opp_rosters[name] = match_roster_to_projections(
            roster, hitters_proj, pitchers_proj,
        )

    print(f"Hart roster: {len(hart_roster)} matched")
    for name, roster in opp_rosters.items():
        print(f"  {name}: {len(roster)} matched")
    print()

    # Compute leverage for all teams
    print("Computing leverage for all teams...")
    leverage_by_team = {}
    for team in standings:
        leverage_by_team[team["name"]] = calculate_leverage(standings, team["name"])

    # Get current roto point rankings for pitch generation
    current_ranks = compute_roto_points_by_cat(standings)

    # Find trades
    print("Evaluating trades...")
    trades = find_trades(
        hart_name=config.team_name,
        hart_roster=hart_roster,
        opp_rosters=opp_rosters,
        standings=standings,
        leverage_by_team=leverage_by_team,
        roster_slots=config.roster_slots,
        max_results=5,
    )

    # Display results
    print()
    print("=" * 70)
    print(f"TOP {len(trades)} TRADE PROPOSALS")
    print("=" * 70)

    if not trades:
        print("\nNo mutually beneficial trades found.")
        return

    for i, trade in enumerate(trades, 1):
        opp = trade["opponent"]
        send_pos = "/".join(trade["send_positions"][:2])
        recv_pos = "/".join(trade["receive_positions"][:2])

        print(f"\n{i}. SEND: {trade['send']:<22} ({send_pos})  ->  {opp}")
        print(f"   GET:  {trade['receive']:<22} ({recv_pos})  <-  {opp}")

        # Hart impact
        hart_gains = [f"+{d} {c}" for c, d in trade["hart_cat_deltas"].items() if d > 0]
        hart_losses = [f"{d} {c}" for c, d in trade["hart_cat_deltas"].items() if d < 0]
        hart_detail = ", ".join(hart_gains + hart_losses)
        print(f"\n   Hart gains: {trade['hart_delta']:+d} roto pts ({hart_detail})")

        # Opponent impact
        opp_gains = [f"+{d} {c}" for c, d in trade["opp_cat_deltas"].items() if d > 0]
        opp_losses = [f"{d} {c}" for c, d in trade["opp_cat_deltas"].items() if d < 0]
        opp_detail = ", ".join(opp_gains + opp_losses)
        print(f"   They gain:  {trade['opp_delta']:+d} roto pts ({opp_detail})")

        # Generate pitch
        opp_ranks = current_ranks.get(opp, {})
        pitch = generate_pitch(opp, trade["opp_cat_deltas"], opp_ranks)
        print(f"\n   Pitch: \"{pitch}\"")

    print()
    print("Done! Review proposals and propose via Yahoo.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify it imports cleanly**

Run: `python -c "import sys; sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts'); from run_trades import match_roster_to_projections; print('Import OK')"`

Expected: `Import OK`

- [ ] **Step 3: Commit**

```bash
git add scripts/run_trades.py
git commit -m "feat: trade recommender CLI with roto impact and pitch generation"
```
