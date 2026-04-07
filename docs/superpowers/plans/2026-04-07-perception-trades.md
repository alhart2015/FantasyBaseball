# Perception-Based Trade Recommendations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the mutual-benefit trade recommender with a perception-based approach that proposes trades where the player sent is similarly ranked (looks fair to the opponent) but the wSGP gain for Hart is maximized.

**Architecture:** Modify `find_trades()` to filter by ROS ranking proximity (max +5 rank gap) instead of raw SGP gap and mutual benefit. Sort results by Hart's wSGP gain (descending), with rank generosity as tiebreaker. Rewrite `generate_pitch()` to focus on ranking/positional value. Pass rankings into `find_trades()` and move rank attachment inside the function.

**Tech Stack:** Python, pytest. No new dependencies.

---

### Task 1: Rewrite `generate_pitch()` with ranking-focused template

**Files:**
- Modify: `src/fantasy_baseball/trades/pitch.py` (full rewrite)
- Test: `tests/test_trades/test_pitch.py` (full rewrite)

- [ ] **Step 1: Write the failing tests**

Replace the contents of `tests/test_trades/test_pitch.py` with:

```python
from fantasy_baseball.trades.pitch import generate_pitch


def test_pitch_mentions_rankings():
    pitch = generate_pitch(
        send_rank=42,
        receive_rank=47,
        send_positions=["SS", "2B"],
        receive_positions=["OF"],
    )
    assert "#42" in pitch
    assert "#47" in pitch


def test_pitch_includes_positional_need_for_different_positions():
    pitch = generate_pitch(
        send_rank=30,
        receive_rank=33,
        send_positions=["SS"],
        receive_positions=["OF"],
    )
    assert "#30" in pitch
    assert "#33" in pitch
    # Should mention positional context when positions differ
    assert "position" in pitch.lower() or "need" in pitch.lower()


def test_pitch_no_positional_note_for_same_position():
    pitch = generate_pitch(
        send_rank=20,
        receive_rank=22,
        send_positions=["OF"],
        receive_positions=["OF"],
    )
    assert "#20" in pitch
    assert "#22" in pitch
    # Should NOT mention positional need when same position
    assert "position" not in pitch.lower() and "need" not in pitch.lower()


def test_pitch_when_sending_better_ranked_player():
    """When we send a better-ranked player, pitch should frame it as generous."""
    pitch = generate_pitch(
        send_rank=25,
        receive_rank=30,
        send_positions=["1B"],
        receive_positions=["3B"],
    )
    assert "#25" in pitch
    assert "#30" in pitch


def test_pitch_is_short():
    pitch = generate_pitch(
        send_rank=50,
        receive_rank=55,
        send_positions=["C"],
        receive_positions=["SP"],
    )
    assert len(pitch) < 200
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trades/test_pitch.py -v`
Expected: FAIL — `generate_pitch()` has the old signature

- [ ] **Step 3: Rewrite `generate_pitch()`**

Replace the contents of `src/fantasy_baseball/trades/pitch.py` with:

```python
"""Generate human-readable trade pitches for opponents."""


def generate_pitch(
    send_rank: int,
    receive_rank: int,
    send_positions: list[str],
    receive_positions: list[str],
) -> str:
    """Generate a 1-2 sentence pitch framing the trade as fair by ranking.

    Args:
        send_rank: ROS SGP rank of the player we're sending.
        receive_rank: ROS SGP rank of the player we'd receive.
        send_positions: Eligible positions of the player we're sending.
        receive_positions: Eligible positions of the player we'd receive.
    """
    rank_part = f"You're getting the #{send_rank} overall player for your #{receive_rank}"

    # Add positional justification when players don't share a position
    shared = set(send_positions) & set(receive_positions)
    if shared:
        return f"{rank_part} — straight swap."
    else:
        return f"{rank_part}. Fills a positional need for both of us."
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trades/test_pitch.py -v`
Expected: all 5 PASS

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/trades/pitch.py tests/test_trades/test_pitch.py
git commit -m "feat: rewrite trade pitch to focus on ranking value"
```

---

### Task 2: Add rankings parameter and ranking filter to `find_trades()`

**Files:**
- Modify: `src/fantasy_baseball/trades/evaluate.py:269-363`
- Test: `tests/test_trades/test_evaluate.py`

- [ ] **Step 1: Write the failing tests**

Add the following tests to the end of `tests/test_trades/test_evaluate.py`:

```python
from fantasy_baseball.sgp.rankings import rank_key

# Shared fixtures for perception-based trade tests
_EQUAL_LEVERAGE = {cat: 0.1 for cat in ALL_CATS}

def _make_hitter(name, positions, r=70, hr=20, rbi=65, sb=8, avg=.270, ab=500):
    h = int(avg * ab)
    return Player(name=name, player_type="hitter", positions=positions,
                  ros=HitterStats(pa=int(ab * 1.15), ab=ab, h=h,
                                  r=r, hr=hr, rbi=rbi, sb=sb, avg=avg))

def _make_pitcher(name, positions, ip=150, w=9, k=140, sv=0, era=3.80, whip=1.25):
    er = int(era * ip / 9)
    bb = int((whip * ip - ip * 0.8) / 1)  # rough estimate
    h_allowed = int(whip * ip - bb)
    return Player(name=name, player_type="pitcher", positions=positions,
                  ros=PitcherStats(ip=ip, w=w, k=k, sv=sv, era=era, whip=whip,
                                   er=er, bb=bb, h_allowed=h_allowed))


def test_rank_filter_accepts_within_threshold():
    """Trade where send_rank - receive_rank = 5 should be accepted."""
    hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
    opp_rosters = {"Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)]}
    rankings = {
        rank_key("Hart OF", "hitter"): 55,
        rank_key("Opp OF", "hitter"): 50,
    }
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    assert any(t["send"] == "Hart OF" and t["receive"] == "Opp OF" for t in trades)


def test_rank_filter_rejects_beyond_threshold():
    """Trade where send_rank - receive_rank = 6 should be rejected."""
    hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
    opp_rosters = {"Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)]}
    rankings = {
        rank_key("Hart OF", "hitter"): 56,
        rank_key("Opp OF", "hitter"): 50,
    }
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    assert not any(t["send"] == "Hart OF" and t["receive"] == "Opp OF" for t in trades)


def test_rank_filter_accepts_sending_better_ranked():
    """Sending a better-ranked player (negative gap) should always be accepted."""
    hart_roster = [_make_hitter("Hart Star", ["OF"], hr=30, sb=3)]
    opp_rosters = {"Rival": [_make_hitter("Opp Guy", ["OF"], hr=10, sb=30)]}
    rankings = {
        rank_key("Hart Star", "hitter"): 20,
        rank_key("Opp Guy", "hitter"): 50,
    }
    leverage = {"Hart": {"R": .1, "HR": .05, "RBI": .1, "SB": .2, "AVG": .1,
                         "W": .1, "K": .1, "SV": .1, "ERA": .05, "WHIP": .05},
                "Rival": _EQUAL_LEVERAGE}
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team=leverage,
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    assert any(t["send"] == "Hart Star" and t["receive"] == "Opp Guy" for t in trades)


def test_rejects_trade_with_no_wsgp_gain():
    """Trade must have positive hart_wsgp_gain even if ranking looks fair."""
    # Hart's player is better in all categories — swapping would lose wSGP
    hart_roster = [_make_hitter("Hart Star", ["OF"], r=100, hr=40, rbi=110, sb=20, avg=.300)]
    opp_rosters = {"Rival": [_make_hitter("Opp Scrub", ["OF"], r=40, hr=5, rbi=30, sb=2, avg=.220)]}
    rankings = {
        rank_key("Hart Star", "hitter"): 10,
        rank_key("Opp Scrub", "hitter"): 12,
    }
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    assert not any(t["send"] == "Hart Star" for t in trades)


def test_sort_by_wsgp_gain_descending():
    """Trades should be sorted by hart_wsgp_gain descending."""
    # Three opponents with players of increasing SB (Hart's high-leverage cat)
    hart_roster = [_make_hitter("Hart OF", ["OF"], hr=20, sb=3)]
    opp_rosters = {
        "Rival A": [_make_hitter("Opp A", ["OF"], hr=18, sb=10)],
        "Rival B": [_make_hitter("Opp B", ["OF"], hr=18, sb=25)],
        "Rival C": [_make_hitter("Opp C", ["OF"], hr=18, sb=18)],
    }
    rankings = {
        rank_key("Hart OF", "hitter"): 50,
        rank_key("Opp A", "hitter"): 48,
        rank_key("Opp B", "hitter"): 49,
        rank_key("Opp C", "hitter"): 47,
    }
    leverage = {"Hart": {"R": .05, "HR": .05, "RBI": .05, "SB": .3, "AVG": .05,
                         "W": .1, "K": .1, "SV": .1, "ERA": .1, "WHIP": .1},
                "Rival A": _EQUAL_LEVERAGE, "Rival B": _EQUAL_LEVERAGE,
                "Rival C": _EQUAL_LEVERAGE}
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team=leverage,
        roster_slots=ROSTER_SLOTS, rankings=rankings, max_results=10,
    )
    gains = [t["hart_wsgp_gain"] for t in trades]
    assert gains == sorted(gains, reverse=True)


def test_sort_tiebreaker_by_rank_generosity():
    """Trades with equal wSGP gain should prefer sending better-ranked player."""
    # Two opponents with identical players (same wSGP gain), different ranks
    hart_roster = [_make_hitter("Hart OF", ["OF"], hr=20, sb=5)]
    opp_rosters = {
        "Rival A": [_make_hitter("Opp A", ["OF"], hr=20, sb=5)],
        "Rival B": [_make_hitter("Opp B", ["OF"], hr=20, sb=5)],
    }
    rankings = {
        rank_key("Hart OF", "hitter"): 50,
        rank_key("Opp A", "hitter"): 52,  # gap = -2 (sending better)
        rank_key("Opp B", "hitter"): 48,  # gap = +2 (sending worse)
    }
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE,
                                                      "Rival A": _EQUAL_LEVERAGE,
                                                      "Rival B": _EQUAL_LEVERAGE},
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    if len(trades) >= 2:
        # Opp A trade has rank_gap -2, Opp B has +2 — Opp A should come first
        assert trades[0]["receive"] == "Opp A"


def test_roster_legality_still_enforced():
    """A swap that violates position coverage is rejected even if ranking is fair."""
    hart_roster = [_make_hitter("Hart C", ["C"])]
    opp_rosters = {"Rival": [_make_pitcher("Opp SP", ["SP"])]}
    rankings = {
        rank_key("Hart C", "hitter"): 50,
        rank_key("Opp SP", "pitcher"): 50,
    }
    # Roster requires C slot — can't replace catcher with a pitcher
    slots = {"C": 1, "P": 0, "BN": 0, "IL": 0}
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
        roster_slots=slots, rankings=rankings,
    )
    assert len(trades) == 0


def test_trades_include_rank_data():
    """Each trade result should include send_rank and receive_rank."""
    hart_roster = [_make_hitter("Hart OF", ["OF"], hr=15, sb=5)]
    opp_rosters = {"Rival": [_make_hitter("Opp OF", ["OF"], hr=25, sb=15)]}
    rankings = {
        rank_key("Hart OF", "hitter"): 55,
        rank_key("Opp OF", "hitter"): 50,
    }
    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team={"Hart": _EQUAL_LEVERAGE, "Rival": _EQUAL_LEVERAGE},
        roster_slots=ROSTER_SLOTS, rankings=rankings,
    )
    assert len(trades) > 0
    t = trades[0]
    assert t["send_rank"] == 55
    assert t["receive_rank"] == 50
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trades/test_evaluate.py::test_rank_filter_accepts_within_threshold -v`
Expected: FAIL — `find_trades()` does not accept `rankings` parameter

- [ ] **Step 3: Update `find_trades()` with ranking filter and new sort**

In `src/fantasy_baseball/trades/evaluate.py`, make these changes:

**a) Remove `EQUAL_LEVERAGE` and `MAX_SGP_GAP` constants (lines 25-32), replace with:**

```python
# Maximum ranking gap for perception-based filtering. A trade is accepted
# when send_rank - receive_rank <= MAX_RANK_GAP (the player we send can
# be up to this many spots worse-ranked than the player we receive).
MAX_RANK_GAP = 5
```

**b) Remove the `calculate_weighted_sgp` import of `EQUAL_LEVERAGE` usage — `calculate_weighted_sgp` is still needed for Hart's wSGP gain computation.**

**c) Replace `find_trades()` (lines 269-363) with:**

```python
def find_trades(
    hart_name: str,
    hart_roster: list[Player],
    opp_rosters: dict[str, list[Player]],
    standings: list[dict],
    leverage_by_team: dict[str, dict],
    roster_slots: dict[str, int],
    rankings: dict[str, int],
    max_results: int = 5,
    projected_standings: list[dict] | None = None,
) -> list[dict]:
    """Find and rank the best 1-for-1 trades for Hart.

    Uses a perception-based approach: filters to trades where the player
    sent is similarly ranked to the player received (looks fair to the
    opponent), then ranks by Hart's wSGP gain (biggest hidden value first).

    Args:
        hart_name: Hart's team name in standings.
        hart_roster: Hart's roster as Player objects.
        opp_rosters: {opponent_name: [Player]} for each opponent.
        standings: current league standings.
        leverage_by_team: {team_name: {cat: weight}} leverage weights.
        roster_slots: league roster slot configuration.
        rankings: {rank_key: int} unweighted SGP ROS rankings.
        max_results: maximum number of trade proposals to return.
        projected_standings: optional projected end-of-season standings.

    Returns list of trade dicts with: send, receive, opponent, hart_delta,
    opp_delta, hart_cat_deltas, opp_cat_deltas, hart_wsgp_gain,
    send_positions, receive_positions, send_rank, receive_rank.
    """
    hart_leverage = leverage_by_team.get(hart_name, {})
    proposals = []

    for opp_name, opp_roster in opp_rosters.items():
        for hart_player in hart_roster:
            send_rank = rankings.get(
                rank_key_from_positions(hart_player.name, hart_player.positions))
            if send_rank is None:
                continue

            hart_wsgp = calculate_weighted_sgp(hart_player.ros, hart_leverage)

            for opp_player in opp_roster:
                receive_rank = rankings.get(
                    rank_key_from_positions(opp_player.name, opp_player.positions))
                if receive_rank is None:
                    continue

                # Roster legality
                if not _can_roster_without(hart_roster, hart_player, opp_player, roster_slots):
                    continue
                if not _can_roster_without(opp_roster, opp_player, hart_player, roster_slots):
                    continue

                # Ranking proximity: looks fair to the opponent
                rank_gap = send_rank - receive_rank
                if rank_gap > MAX_RANK_GAP:
                    continue

                # wSGP gain for Hart
                gain_wsgp = calculate_weighted_sgp(opp_player.ros, hart_leverage)
                hart_wsgp_gain = gain_wsgp - hart_wsgp

                if hart_wsgp_gain <= 0:
                    continue

                # Roto point impact
                hart_ros = _player_ros_stats(hart_player)
                opp_ros = _player_ros_stats(opp_player)

                impact = compute_trade_impact(
                    standings, hart_name, opp_name,
                    hart_ros, opp_ros, opp_ros, hart_ros,
                    projected_standings=projected_standings,
                )

                proposals.append({
                    "send": hart_player.name,
                    "send_positions": hart_player.positions,
                    "receive": opp_player.name,
                    "receive_positions": opp_player.positions,
                    "opponent": opp_name,
                    "hart_delta": impact["hart_delta"],
                    "opp_delta": impact["opp_delta"],
                    "hart_cat_deltas": impact["hart_cat_deltas"],
                    "opp_cat_deltas": impact["opp_cat_deltas"],
                    "hart_wsgp_gain": round(hart_wsgp_gain, 2),
                    "send_rank": send_rank,
                    "receive_rank": receive_rank,
                })

    # Sort: biggest wSGP gain first, then most generous rank gap as tiebreaker
    proposals.sort(
        key=lambda t: (-t["hart_wsgp_gain"], t["send_rank"] - t["receive_rank"]),
    )
    return proposals[:max_results]
```

**d) Add import for `rank_key_from_positions` at the top of the file:**

```python
from fantasy_baseball.sgp.rankings import rank_key_from_positions
```

- [ ] **Step 4: Update the existing `test_find_trades_returns_ranked_list` test**

This test uses the old signature without `rankings`. Update it to pass rankings:

```python
def test_find_trades_returns_ranked_list():
    hart_roster = [
        Player(name="Slugger", player_type="hitter", positions=["OF"],
               ros=HitterStats(pa=570, ab=520, h=140, r=80, hr=35, rbi=90, sb=5, avg=.270)),
        Player(name="Speedy", player_type="hitter", positions=["SS"],
               ros=HitterStats(pa=550, ab=500, h=130, r=70, hr=10, rbi=50, sb=40, avg=.260)),
    ]
    opp_rosters = {
        "Rival": [
            Player(name="Closer", player_type="pitcher", positions=["RP"],
                   ros=PitcherStats(ip=65, w=3, k=60, sv=30, era=2.80, whip=1.00,
                                    er=20, bb=15, h_allowed=50)),
            Player(name="Stealer", player_type="hitter", positions=["OF"],
                   ros=HitterStats(pa=560, ab=510, h=135, r=75, hr=8, rbi=45, sb=45, avg=.265)),
        ],
    }
    leverage_by_team = {
        "Hart": {"R": .1, "HR": .05, "RBI": .1, "SB": .15, "AVG": .1,
                 "W": .1, "K": .1, "SV": .15, "ERA": .1, "WHIP": .05},
        "Rival": {"R": .1, "HR": .15, "RBI": .1, "SB": .05, "AVG": .1,
                  "W": .1, "K": .1, "SV": .1, "ERA": .1, "WHIP": .1},
    }
    rankings = {
        rank_key("Slugger", "hitter"): 30,
        rank_key("Speedy", "hitter"): 40,
        rank_key("Closer", "pitcher"): 35,
        rank_key("Stealer", "hitter"): 38,
    }

    trades = find_trades(
        hart_name="Hart", hart_roster=hart_roster, opp_rosters=opp_rosters,
        standings=SAMPLE_STANDINGS, leverage_by_team=leverage_by_team,
        roster_slots=ROSTER_SLOTS, rankings=rankings, max_results=5,
    )
    assert isinstance(trades, list)
    if trades:
        t = trades[0]
        assert "send" in t
        assert "receive" in t
        assert "opponent" in t
        assert "hart_delta" in t
        assert "hart_wsgp_gain" in t
        assert "send_rank" in t
        assert "receive_rank" in t
```

- [ ] **Step 5: Run all trade tests**

Run: `pytest tests/test_trades/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/trades/evaluate.py tests/test_trades/test_evaluate.py
git commit -m "feat: replace SGP gap filter with ranking proximity filter"
```

---

### Task 3: Update `season_data.py` call site

**Files:**
- Modify: `src/fantasy_baseball/web/season_data.py:1062-1091`

- [ ] **Step 1: Update `find_trades()` call to pass rankings**

In `src/fantasy_baseball/web/season_data.py`, update the `find_trades()` call (around line 1062) to pass the `rankings` parameter. Add `rankings=ros_ranks,` to the call — `ros_ranks` is the unweighted SGP rankings dict already computed earlier in the refresh pipeline (around line 910).

Replace the block from the `find_trades()` call through the rank attachment loop (lines 1062-1089) with:

```python
        trade_proposals = find_trades(
            hart_name=config.team_name,
            hart_roster=hart_roster_for_trades,
            opp_rosters=opp_rosters,
            standings=standings,
            leverage_by_team=leverage_by_team,
            roster_slots=config.roster_slots,
            rankings=ros_ranks,
            max_results=10,
            projected_standings=projected_standings,
        )

        # Attach trade pitches
        for trade in trade_proposals:
            trade["pitch"] = generate_pitch(
                send_rank=trade["send_rank"],
                receive_rank=trade["receive_rank"],
                send_positions=trade.get("send_positions", []),
                receive_positions=trade.get("receive_positions", []),
            )
```

This removes:
- The old `generate_pitch()` call with category deltas/ranks
- The post-hoc rank attachment loop (ranks are now set inside `find_trades()`)

- [ ] **Step 2: Remove unused imports if needed**

Check if `rank_key_from_positions` import in `season_data.py` is still used elsewhere (waiver recs, buy-low). If it's still used, keep it. If the only uses were for trade rank attachment, remove it.

- [ ] **Step 3: Run the full test suite**

Run: `pytest tests/ -v`
Expected: all PASS

- [ ] **Step 4: Commit**

```bash
git add src/fantasy_baseball/web/season_data.py
git commit -m "feat: wire perception-based trades into refresh pipeline"
```
