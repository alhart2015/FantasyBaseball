# Roster ↔ projection matching hardening — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add regression test coverage for three historically fragile players (Julio Rodriguez, Mason Miller, Shohei Ohtani) and add WARNING-level observability to `match_roster_to_projections` so future matching failures stop being silent.

**Architecture:** Single-function audit. `match_roster_to_projections` in `src/fantasy_baseball/data/projections.py` keeps its signature and return type. We add an optional `context: str = ""` kwarg used only for log clarity, a module logger, and three `logger.warning(...)` calls (unmatched, ambiguous, fallback). The thin adapter `hydrate_roster_entries` plumbs `context` through. Five callers in `refresh_pipeline.py` and `season_data.py` pass useful context strings ("user", "opp:Team Name", "preseason", "ros"). Draft pipeline is **out of scope** — slated for preseason overhaul.

**Tech Stack:** Python 3.x, pandas, pytest, Python `logging` module.

**Spec:** `docs/superpowers/specs/2026-04-16-roster-projection-matching-hardening-design.md`

**Branch:** `harden-roster-projection-matching` (already created)

---

## File Structure

**New file:**
- `tests/test_data/test_projections_matching_edge_cases.py` — all new regression and observability tests

**Modified files:**
- `src/fantasy_baseball/data/projections.py` — add module logger, `context` kwarg on `match_roster_to_projections` and `hydrate_roster_entries`, three WARNING log calls
- `src/fantasy_baseball/web/refresh_pipeline.py` — two callers pass `context`
- `src/fantasy_baseball/web/season_data.py` — two callers pass `context`

Total LOC added: ~25 in `projections.py`, ~5 in callers, ~250 in tests.

---

## Task 1: Test scaffold + Julio Rodriguez accent encoding regression

**Files:**
- Create: `tests/test_data/test_projections_matching_edge_cases.py`

**Why this first:** Establishes the test file shape (tiny in-memory DataFrames, no fixtures) and verifies `normalize_name` handles all three Unicode forms today. These tests should pass on first run — if they fail, that's a real bug we caught before observability obscures it.

- [ ] **Step 1: Create the test file with helper builders and the Julio Rodriguez test class**

Path: `tests/test_data/test_projections_matching_edge_cases.py`

```python
"""Regression and observability tests for match_roster_to_projections.

Covers three historically fragile players:
- Julio Rodriguez: accent encoding (NFC vs NFD vs ASCII)
- Mason Miller: cross-type same-name collision (hitter + pitcher)
- Shohei Ohtani: dual roster entries with "(Pitcher)" suffix

And the matcher's observability: WARNING logs on unmatched, ambiguous,
and fallback matches so future regressions surface immediately instead
of silently dropping or mis-matching players.
"""
import logging
import pandas as pd
import pytest

from fantasy_baseball.data.projections import match_roster_to_projections
from fantasy_baseball.models.player import HitterStats, PitcherStats, PlayerType


# --- Tiny in-memory DataFrame builders ---

def _hitters_df(rows):
    """Build a hitters projection DataFrame with the minimum columns the
    matcher and HitterStats.from_dict require. Each row in ``rows`` is a
    dict with at least ``name`` and ``_name_norm``; missing stat columns
    default to 0.
    """
    defaults = {
        "r": 0, "hr": 0, "rbi": 0, "sb": 0, "avg": 0.0,
        "ab": 0, "h": 0, "pa": 0, "player_type": "hitter",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _pitchers_df(rows):
    """Build a pitchers projection DataFrame with the minimum columns the
    matcher and PitcherStats.from_dict require.
    """
    defaults = {
        "w": 0, "k": 0, "sv": 0, "ip": 0, "er": 0, "bb": 0, "h_allowed": 0,
        "era": 0.0, "whip": 0.0, "player_type": "pitcher",
    }
    return pd.DataFrame([{**defaults, **r} for r in rows])


def _empty_hitters():
    return pd.DataFrame(columns=["name", "_name_norm", "player_type"])


def _empty_pitchers():
    return pd.DataFrame(columns=["name", "_name_norm", "player_type"])


# --- Julio Rodriguez: accent encoding ---

class TestJulioRodriguezAccentEncoding:
    """Verify normalize_name handles all three Unicode forms of 'í'.

    These tests exercise the matcher end-to-end with realistic encoding
    variants Yahoo and FanGraphs have been observed to send.
    """

    PROJECTION = {"name": "Julio Rodríguez", "_name_norm": "julio rodriguez", "hr": 32}

    def test_roster_nfc_precomposed_matches(self):
        roster = [{"name": "Julio Rodríguez", "positions": ["OF"]}]
        result = match_roster_to_projections(
            roster, _hitters_df([self.PROJECTION]), _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32

    def test_roster_nfd_decomposed_matches(self):
        # 'í' as 'i' + combining acute accent (U+0301)
        roster = [{"name": "Julio Rodri\u0301guez", "positions": ["OF"]}]
        result = match_roster_to_projections(
            roster, _hitters_df([self.PROJECTION]), _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32

    def test_roster_ascii_matches_accented_projection(self):
        roster = [{"name": "Julio Rodriguez", "positions": ["OF"]}]
        result = match_roster_to_projections(
            roster, _hitters_df([self.PROJECTION]), _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32

    def test_accented_roster_matches_ascii_projection(self):
        # Mirror: roster has accents, projection is plain ASCII
        roster = [{"name": "Julio Rodríguez", "positions": ["OF"]}]
        ascii_proj = {"name": "Julio Rodriguez", "_name_norm": "julio rodriguez", "hr": 32}
        result = match_roster_to_projections(
            roster, _hitters_df([ascii_proj]), _empty_pitchers(),
        )
        assert len(result) == 1
        assert result[0].rest_of_season.hr == 32
```

- [ ] **Step 2: Run the new tests — all four should pass**

Run: `pytest tests/test_data/test_projections_matching_edge_cases.py::TestJulioRodriguezAccentEncoding -v`
Expected: 4 passed (normalize_name already handles NFKD decomposition + combining-character stripping).

If any fail: **stop** and investigate. A failure here means `normalize_name` has a real encoding bug, which changes the scope of this work.

- [ ] **Step 3: Commit**

```bash
git add tests/test_data/test_projections_matching_edge_cases.py
git commit -m "test(projections): regression coverage for Julio Rodriguez accent encoding"
```

---

## Task 2: Mason Miller cross-type same-name collision regression

**Files:**
- Modify: `tests/test_data/test_projections_matching_edge_cases.py` (append class)

**Why:** Verify that when both a hitter and a pitcher named "Mason Miller" exist in projections, the matcher uses the roster's `positions` to disambiguate correctly. The third subtest (empty positions) documents the current fallback behavior — hitter wins because the fallback loop checks `hitters_proj` first.

- [ ] **Step 1: Append the Mason Miller test class**

Append to `tests/test_data/test_projections_matching_edge_cases.py`:

```python
# --- Mason Miller: cross-type same-name collision ---

class TestMasonMillerCrossTypeCollision:
    """Verify the matcher uses positions to pick the right Mason Miller
    when both hitter and pitcher entries exist in projections.
    """

    HITTER_PROJ = {
        "name": "Mason Miller", "_name_norm": "mason miller",
        "hr": 18, "ab": 480,
    }
    PITCHER_PROJ = {
        "name": "Mason Miller", "_name_norm": "mason miller",
        "k": 95, "sv": 28, "ip": 65,
    }

    def test_hitter_position_picks_hitter_projection(self):
        roster = [{"name": "Mason Miller", "positions": ["3B"]}]
        result = match_roster_to_projections(
            roster, _hitters_df([self.HITTER_PROJ]), _pitchers_df([self.PITCHER_PROJ]),
        )
        assert len(result) == 1
        assert result[0].player_type == PlayerType.HITTER
        assert isinstance(result[0].rest_of_season, HitterStats)
        assert result[0].rest_of_season.hr == 18

    def test_pitcher_position_picks_pitcher_projection(self):
        roster = [{"name": "Mason Miller", "positions": ["SP"]}]
        result = match_roster_to_projections(
            roster, _hitters_df([self.HITTER_PROJ]), _pitchers_df([self.PITCHER_PROJ]),
        )
        assert len(result) == 1
        assert result[0].player_type == PlayerType.PITCHER
        assert isinstance(result[0].rest_of_season, PitcherStats)
        assert result[0].rest_of_season.sv == 28

    def test_empty_positions_falls_back_to_hitter_first(self, caplog):
        """Empty positions: matcher falls through both branches and uses
        the 'any' fallback, which checks hitters first.
        """
        roster = [{"name": "Mason Miller", "positions": []}]
        with caplog.at_level(logging.WARNING):
            result = match_roster_to_projections(
                roster, _hitters_df([self.HITTER_PROJ]), _pitchers_df([self.PITCHER_PROJ]),
            )
        assert len(result) == 1
        assert result[0].player_type == PlayerType.HITTER
        # Fallback warning is asserted in TestMatchObservability — not here.
```

- [ ] **Step 2: Run the new tests — all three should pass**

Run: `pytest tests/test_data/test_projections_matching_edge_cases.py::TestMasonMillerCrossTypeCollision -v`
Expected: 3 passed.

If `test_empty_positions_falls_back_to_hitter_first` fails: that's a real behavior bug worth investigating before continuing.

- [ ] **Step 3: Commit**

```bash
git add tests/test_data/test_projections_matching_edge_cases.py
git commit -m "test(projections): regression coverage for Mason Miller hitter/pitcher collision"
```

---

## Task 3: Shohei Ohtani dual-entry roster regression

**Files:**
- Modify: `tests/test_data/test_projections_matching_edge_cases.py` (append class)

**Why:** Verify Yahoo's two-entry Ohtani roster (one with "(Pitcher)" suffix) produces two distinct `Player` objects with the right player_type and yahoo_id. This exercises the `" (Batter)"` / `" (Pitcher)"` suffix stripping at line 375 of `projections.py`.

- [ ] **Step 1: Append the Ohtani test class**

Append to `tests/test_data/test_projections_matching_edge_cases.py`:

```python
# --- Shohei Ohtani: dual-entry roster ---

class TestShoheiOhtaniDualEntry:
    """Yahoo returns Ohtani as two roster entries:
    - "Shohei Ohtani" with hitter positions
    - "Shohei Ohtani (Pitcher)" with pitcher positions

    Both must match correctly. The "(Pitcher)" suffix is stripped before
    name normalization so both find the right projection by name.
    """

    HITTER_PROJ = {
        "name": "Shohei Ohtani", "_name_norm": "shohei ohtani",
        "hr": 44, "r": 110,
    }
    PITCHER_PROJ = {
        "name": "Shohei Ohtani", "_name_norm": "shohei ohtani",
        "k": 180, "ip": 140, "w": 12,
    }

    def test_dual_roster_entries_produce_two_player_objects(self):
        roster = [
            {"name": "Shohei Ohtani", "positions": ["Util"],
             "selected_position": "Util", "player_id": "100", "status": ""},
            {"name": "Shohei Ohtani (Pitcher)", "positions": ["SP"],
             "selected_position": "SP", "player_id": "200", "status": ""},
        ]
        result = match_roster_to_projections(
            roster, _hitters_df([self.HITTER_PROJ]), _pitchers_df([self.PITCHER_PROJ]),
        )
        assert len(result) == 2

        by_yahoo_id = {p.yahoo_id: p for p in result}
        assert set(by_yahoo_id) == {"100", "200"}

        hitter = by_yahoo_id["100"]
        assert hitter.player_type == PlayerType.HITTER
        assert hitter.name == "Shohei Ohtani"
        assert hitter.rest_of_season.hr == 44

        pitcher = by_yahoo_id["200"]
        assert pitcher.player_type == PlayerType.PITCHER
        # Suffix stripped from the stored Player.name as well
        assert pitcher.name == "Shohei Ohtani"
        assert pitcher.rest_of_season.k == 180
```

- [ ] **Step 2: Run the new test — should pass**

Run: `pytest tests/test_data/test_projections_matching_edge_cases.py::TestShoheiOhtaniDualEntry -v`
Expected: 1 passed.

- [ ] **Step 3: Commit**

```bash
git add tests/test_data/test_projections_matching_edge_cases.py
git commit -m "test(projections): regression coverage for Ohtani dual-entry roster"
```

---

## Task 4: Add `context` kwarg + module logger + unmatched WARNING

**Files:**
- Modify: `src/fantasy_baseball/data/projections.py`
- Modify: `tests/test_data/test_projections_matching_edge_cases.py` (append observability class)

**Why:** First piece of new behavior. Set up the logger and `context` kwarg infrastructure, then add the simplest of the three warnings (unmatched). TDD: failing test first.

- [ ] **Step 1: Write the failing observability test**

Append to `tests/test_data/test_projections_matching_edge_cases.py`:

```python
# --- Observability ---

class TestMatchObservability:
    """Verify match_roster_to_projections emits WARNING logs for the three
    insidious cases (unmatched, ambiguous, fallback) so future matching
    regressions surface immediately instead of silently dropping or
    mis-matching players.
    """

    def test_unmatched_player_logs_warning(self, caplog):
        roster = [{"name": "Nobody Special", "positions": ["OF"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(
                roster, _empty_hitters(), _empty_pitchers(),
            )
        assert result == []
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "no projection match" in msg
        assert "Nobody Special" in msg
        assert "OF" in msg

    def test_unmatched_player_with_context_includes_context_in_log(self, caplog):
        roster = [{"name": "Nobody Special", "positions": ["OF"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            match_roster_to_projections(
                roster, _empty_hitters(), _empty_pitchers(), context="opp:Sharks",
            )
        msg = caplog.records[0].getMessage()
        assert "[opp:Sharks]" in msg

    def test_unmatched_player_without_context_omits_brackets(self, caplog):
        roster = [{"name": "Nobody Special", "positions": ["OF"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            match_roster_to_projections(
                roster, _empty_hitters(), _empty_pitchers(),
            )
        msg = caplog.records[0].getMessage()
        assert "[]" not in msg
        assert not msg.startswith("[")
```

- [ ] **Step 2: Run the test to confirm failure**

Run: `pytest tests/test_data/test_projections_matching_edge_cases.py::TestMatchObservability -v`
Expected: 3 failed (matcher doesn't accept `context`, doesn't log warnings).

- [ ] **Step 3: Add module logger and `context` kwarg + unmatched WARNING**

Edit `src/fantasy_baseball/data/projections.py`:

After the existing imports (around line 9), add the logger:

```python
import logging

logger = logging.getLogger(__name__)
```

Replace the `match_roster_to_projections` function signature and body. Current signature is on line 359. Change the signature to:

```python
def match_roster_to_projections(
    roster: list[dict],
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
    *,
    context: str = "",
) -> list[Player]:
    """Match roster players to blended projections by normalized name.

    Expects ``_name_norm`` column precomputed on both DataFrames
    (call ``df["_name_norm"] = df["name"].apply(normalize_name)`` first).

    Returns a list of :class:`Player` objects with ``.rest_of_season`` populated as
    :class:`HitterStats` or :class:`PitcherStats`. Unmatched players are
    omitted.

    Emits ``WARNING`` logs for three matching anomalies so silent failures
    surface in the refresh log:

    - Unmatched roster player (no projection found in either DataFrame)
    - Ambiguous match (multiple projection rows share a normalized name)
    - Fallback match (positions did not disambiguate hitter vs pitcher)

    The ``context`` kwarg is included in log messages as a ``[context]``
    prefix to identify which call site produced the warning (e.g.
    ``"user"``, ``"opp:Sharks"``, ``"preseason"``, ``"ros"``).
    """
    prefix = f"[{context}] " if context else ""
    matched: list[Player] = []
    for player in roster:
        name = player["name"].replace(" (Batter)", "").replace(" (Pitcher)", "")
        name_norm = normalize_name(name)
        positions = player.get("positions", [])

        proj = None
        ptype = None
        if is_hitter(positions) and not hitters_proj.empty:
            matches = hitters_proj[hitters_proj["_name_norm"] == name_norm]
            if not matches.empty:
                proj = matches.iloc[0]
                ptype = PlayerType.HITTER
        if proj is None and is_pitcher(positions) and not pitchers_proj.empty:
            matches = pitchers_proj[pitchers_proj["_name_norm"] == name_norm]
            if not matches.empty:
                proj = matches.iloc[0]
                ptype = PlayerType.PITCHER
        if proj is None:
            for df, pt in [(hitters_proj, PlayerType.HITTER), (pitchers_proj, PlayerType.PITCHER)]:
                if df.empty:
                    continue
                matches = df[df["_name_norm"] == name_norm]
                if not matches.empty:
                    proj = matches.iloc[0]
                    ptype = pt
                    break

        if proj is None:
            logger.warning(
                "%sno projection match for %r (positions=%r)",
                prefix, name, positions,
            )
            continue

        if ptype == PlayerType.HITTER:
            ros = HitterStats.from_dict(proj.to_dict())
        else:
            ros = PitcherStats.from_dict(proj.to_dict())

        # Parse positions and selected_position explicitly
        parsed_positions = [
            p if isinstance(p, Position) else Position.parse(p)
            for p in positions
        ]
        raw_slot = player.get("selected_position", "")
        if raw_slot is None or raw_slot == "":
            parsed_slot = None
        elif isinstance(raw_slot, Position):
            parsed_slot = raw_slot
        else:
            parsed_slot = Position.parse(raw_slot)

        p = Player(
            name=name,
            player_type=ptype,
            positions=parsed_positions,
            yahoo_id=player.get("player_id", ""),
            selected_position=parsed_slot,
            status=player.get("status", ""),
            rest_of_season=ros,
        )
        matched.append(p)

    return matched
```

Notes on the rewrite:
- The `if proj is not None:` block at line 401 is now flipped to `if proj is None: ... continue` so the warning sits at the top of the failure path. Functional behavior is identical: unmatched players are still omitted from the return list.
- The `context` kwarg is keyword-only (`*,`) so existing positional callers don't break.
- Log format uses `%`-style lazy formatting (Python logging convention) — the message string isn't built unless the WARNING level is enabled.

- [ ] **Step 4: Run the observability tests — should pass**

Run: `pytest tests/test_data/test_projections_matching_edge_cases.py::TestMatchObservability -v`
Expected: 3 passed.

- [ ] **Step 5: Run the full edge-case file and the existing matcher tests to confirm no regression**

Run: `pytest tests/test_data/test_projections_matching_edge_cases.py tests/test_data/test_projections.py::TestMatchRosterToProjections -v`
Expected: all pass (Mason Miller empty-positions test will now also emit a fallback warning, but doesn't assert against caplog so still passes).

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/data/projections.py tests/test_data/test_projections_matching_edge_cases.py
git commit -m "feat(projections): add context kwarg and unmatched WARNING to matcher"
```

---

## Task 5: Add ambiguous WARNING

**Files:**
- Modify: `src/fantasy_baseball/data/projections.py`
- Modify: `tests/test_data/test_projections_matching_edge_cases.py`

**Why:** Currently when `matches.iloc[0]` runs against a multi-row result, the matcher silently picks the first. This catches the case where two projections share a normalized name (e.g., a major-leaguer and a prospect with the same name).

- [ ] **Step 1: Add failing tests**

Append to the `TestMatchObservability` class in `tests/test_data/test_projections_matching_edge_cases.py`:

```python
    def test_ambiguous_hitter_match_logs_warning(self, caplog):
        # Two projection rows with the same normalized name and matching positions
        hitters = _hitters_df([
            {"name": "John Smith", "_name_norm": "john smith", "hr": 25},
            {"name": "John Smith", "_name_norm": "john smith", "hr": 12},
        ])
        roster = [{"name": "John Smith", "positions": ["OF"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(roster, hitters, _empty_pitchers())
        assert len(result) == 1
        # First row wins (matches.iloc[0])
        assert result[0].rest_of_season.hr == 25
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "ambiguous" in msg
        assert "hitter" in msg
        assert "John Smith" in msg
        assert "2 candidates" in msg

    def test_ambiguous_pitcher_match_logs_warning(self, caplog):
        pitchers = _pitchers_df([
            {"name": "Joe Pitcher", "_name_norm": "joe pitcher", "k": 200},
            {"name": "Joe Pitcher", "_name_norm": "joe pitcher", "k": 50},
        ])
        roster = [{"name": "Joe Pitcher", "positions": ["SP"]}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(roster, _empty_hitters(), pitchers)
        assert len(result) == 1
        assert result[0].rest_of_season.k == 200
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "ambiguous" in msg
        assert "pitcher" in msg
        assert "2 candidates" in msg
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_data/test_projections_matching_edge_cases.py::TestMatchObservability::test_ambiguous_hitter_match_logs_warning tests/test_data/test_projections_matching_edge_cases.py::TestMatchObservability::test_ambiguous_pitcher_match_logs_warning -v`
Expected: 2 failed.

- [ ] **Step 3: Add ambiguous warning to the matcher**

In `src/fantasy_baseball/data/projections.py`, modify the hitter branch and pitcher branch to log when `len(matches) > 1`. Replace these two stanzas inside `match_roster_to_projections`:

```python
        if is_hitter(positions) and not hitters_proj.empty:
            matches = hitters_proj[hitters_proj["_name_norm"] == name_norm]
            if not matches.empty:
                if len(matches) > 1:
                    logger.warning(
                        "%sambiguous hitter match for %r — %d candidates, picked first",
                        prefix, name, len(matches),
                    )
                proj = matches.iloc[0]
                ptype = PlayerType.HITTER
        if proj is None and is_pitcher(positions) and not pitchers_proj.empty:
            matches = pitchers_proj[pitchers_proj["_name_norm"] == name_norm]
            if not matches.empty:
                if len(matches) > 1:
                    logger.warning(
                        "%sambiguous pitcher match for %r — %d candidates, picked first",
                        prefix, name, len(matches),
                    )
                proj = matches.iloc[0]
                ptype = PlayerType.PITCHER
```

- [ ] **Step 4: Run the new tests + the rest of the file**

Run: `pytest tests/test_data/test_projections_matching_edge_cases.py -v`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/data/projections.py tests/test_data/test_projections_matching_edge_cases.py
git commit -m "feat(projections): warn on ambiguous projection matches"
```

---

## Task 6: Add fallback WARNING

**Files:**
- Modify: `src/fantasy_baseball/data/projections.py`
- Modify: `tests/test_data/test_projections_matching_edge_cases.py`

**Why:** The third "any" branch fires when `is_hitter(positions)` and `is_pitcher(positions)` both returned False — usually because positions is empty or contains only bench/IL slots. This is the silent path that mis-matched Mason-Miller-style players in the past.

- [ ] **Step 1: Add failing test**

Append to the `TestMatchObservability` class:

```python
    def test_fallback_match_logs_warning(self, caplog):
        # Position list doesn't qualify as hitter or pitcher (empty),
        # but name matches a hitter projection via fallback.
        hitters = _hitters_df([
            {"name": "Mystery Player", "_name_norm": "mystery player", "hr": 10},
        ])
        roster = [{"name": "Mystery Player", "positions": []}]
        with caplog.at_level(logging.WARNING, logger="fantasy_baseball.data.projections"):
            result = match_roster_to_projections(roster, hitters, _empty_pitchers())
        assert len(result) == 1
        assert result[0].player_type == PlayerType.HITTER
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        msg = warnings[0].getMessage()
        assert "fallback" in msg
        assert "Mystery Player" in msg
        assert "did not disambiguate" in msg
```

- [ ] **Step 2: Run to confirm failure**

Run: `pytest tests/test_data/test_projections_matching_edge_cases.py::TestMatchObservability::test_fallback_match_logs_warning -v`
Expected: 1 failed.

- [ ] **Step 3: Add fallback warning**

In `src/fantasy_baseball/data/projections.py`, modify the third "any" branch to track that it was hit and log after the loop. Replace the third branch:

```python
        if proj is None:
            for df, pt in [(hitters_proj, PlayerType.HITTER), (pitchers_proj, PlayerType.PITCHER)]:
                if df.empty:
                    continue
                matches = df[df["_name_norm"] == name_norm]
                if not matches.empty:
                    proj = matches.iloc[0]
                    ptype = pt
                    logger.warning(
                        "%s%r matched via fallback branch — positions=%r did not disambiguate",
                        prefix, name, positions,
                    )
                    break
```

- [ ] **Step 4: Run the full edge-case file**

Run: `pytest tests/test_data/test_projections_matching_edge_cases.py -v`
Expected: all pass. Note: `TestMasonMillerCrossTypeCollision::test_empty_positions_falls_back_to_hitter_first` will now produce a fallback warning, but it doesn't assert on log silence so still passes.

- [ ] **Step 5: Run the full test suite to catch any unintended regressions**

Run: `pytest -q`
Expected: all pass. (If any pre-existing test in `tests/test_web/` or `tests/test_data/` started failing because it now sees unexpected WARNINGs, that's a real signal worth investigating, not a test to silence.)

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/data/projections.py tests/test_data/test_projections_matching_edge_cases.py
git commit -m "feat(projections): warn on fallback-branch projection matches"
```

---

## Task 7: Plumb `context` through `hydrate_roster_entries` and call sites

**Files:**
- Modify: `src/fantasy_baseball/data/projections.py` (`hydrate_roster_entries`, lines 434–463)
- Modify: `src/fantasy_baseball/web/refresh_pipeline.py` (two call sites: lines 465, 476, 541)
- Modify: `src/fantasy_baseball/web/season_data.py` (two call sites: lines 329, 334)

**Why:** Without context strings, every warning shows the same nameless prefix and the user can't tell which roster (user vs opponent vs preseason vs ROS) the issue came from. This is the payoff for the kwarg infrastructure.

- [ ] **Step 1: Add `context` kwarg to `hydrate_roster_entries`**

In `src/fantasy_baseball/data/projections.py`, replace `hydrate_roster_entries`:

```python
def hydrate_roster_entries(
    roster: Roster,
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
    *,
    context: str = "",
) -> list[Player]:
    """Convert a :class:`Roster`'s entries into ``list[Player]`` with
    projection stats populated.

    Thin adapter around :func:`match_roster_to_projections`: converts
    each :class:`RosterEntry` into the dict shape the legacy matcher
    expects, then delegates so every edge case (name normalization,
    accent handling, "(Batter)"/"(Pitcher)" suffix stripping, position
    collisions) is preserved for free.

    Unmatched entries are omitted, matching
    :func:`match_roster_to_projections`'s contract. The ``context`` kwarg
    is forwarded for log clarity.
    """
    roster_dicts = [
        {
            "name": entry.name,
            "positions": [p.value for p in entry.positions],
            "selected_position": entry.selected_position.value,
            "status": entry.status,
            "player_id": entry.yahoo_id,
        }
        for entry in roster.entries
    ]
    return match_roster_to_projections(
        roster_dicts, hitters_proj, pitchers_proj, context=context,
    )
```

- [ ] **Step 2: Wire `context` into refresh_pipeline.py**

In `src/fantasy_baseball/web/refresh_pipeline.py`:

Find the user roster hydration (around line 465):

```python
        self.matched = hydrate_roster_entries(
            user_roster_model, self.hitters_proj, self.pitchers_proj,
        )
```

Replace with:

```python
        self.matched = hydrate_roster_entries(
            user_roster_model, self.hitters_proj, self.pitchers_proj,
            context="user",
        )
```

Find the opponent roster hydration (around line 476):

```python
            hydrated = hydrate_roster_entries(
                latest, self.hitters_proj, self.pitchers_proj,
            )
```

Replace with:

```python
            hydrated = hydrate_roster_entries(
                latest, self.hitters_proj, self.pitchers_proj,
                context=f"opp:{team.name}",
            )
```

Find the preseason match call (around line 541):

```python
        preseason_matched = match_roster_to_projections(
            self.roster_raw, self.preseason_hitters, self.preseason_pitchers,
        )
```

Replace with:

```python
        preseason_matched = match_roster_to_projections(
            self.roster_raw, self.preseason_hitters, self.preseason_pitchers,
            context="preseason",
        )
```

- [ ] **Step 3: Wire `context` into season_data.py**

In `src/fantasy_baseball/web/season_data.py`, both calls live inside `build_opponent_lineup` (which has access to the `opponent_name` parameter). Edit the two existing single-line calls.

Replace line 329:

```python
    matched = match_roster_to_projections(roster, hitters_proj, pitchers_proj)
```

with:

```python
    matched = match_roster_to_projections(
        roster, hitters_proj, pitchers_proj,
        context=f"opp-lineup:{opponent_name}",
    )
```

Replace line 334:

```python
        rest_of_season_matched = match_roster_to_projections(roster, rest_of_season_hitters, rest_of_season_pitchers)
```

with:

```python
        rest_of_season_matched = match_roster_to_projections(
            roster, rest_of_season_hitters, rest_of_season_pitchers,
            context=f"opp-lineup:{opponent_name}:ros",
        )
```

- [ ] **Step 4: Verify no callers were missed**

Run: `git grep -n "match_roster_to_projections\|hydrate_roster_entries" -- src/ scripts/`
Expected: every call site in `src/fantasy_baseball/web/` either passes `context=` or is a definition. Calls in `scripts/` (if any) are allowed to omit `context` — they get `context=""` which produces unprefixed log messages, the existing default behavior.

- [ ] **Step 5: Run the full test suite**

Run: `pytest -q`
Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/data/projections.py src/fantasy_baseball/web/refresh_pipeline.py src/fantasy_baseball/web/season_data.py
git commit -m "feat(refresh): pass match context to surface which roster produced warnings"
```

---

## Verification before finishing

- [ ] **Step 1: Run the full test suite one final time**

Run: `pytest -q`
Expected: all pass with no skips beyond pre-existing ones.

- [ ] **Step 2: Run the local refresh once and inspect the log**

Per `feedback_local_testing.md` and `feedback_run_refresh_before_merge.md` in the user's memory: exercise the refresh path locally to make sure the new WARNINGs don't spam unexpectedly.

Run: `python scripts/run_season_dashboard.py 2>&1 | grep -E "WARNING|projection"`
Expected: any WARNING lines should be informative (named players, identifiable contexts). If hundreds of warnings appear, that's a real signal — investigate the root cause before merging. If a small number appear, note them in the PR description.

- [ ] **Step 3: Surface findings**

If the refresh produced unexpected warnings, write up a brief summary (which players, which context) and decide whether to fix the underlying issue in this branch or open follow-up tasks. Do **not** silence warnings to clean up the log — that defeats the purpose of the work.

- [ ] **Step 4: Stop here for user review before any merge**

Per `feedback_no_merge_without_asking.md` and `feedback_always_use_branches.md`: the branch is `harden-roster-projection-matching`. Do not merge to main without explicit user approval. Push the branch and wait.
