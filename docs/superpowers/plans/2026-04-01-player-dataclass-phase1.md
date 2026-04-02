# Player Dataclass Phase 1: Define Types + Conversion Layer

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Define `Player`, `HitterStats`, `PitcherStats`, and `RankInfo` dataclasses with conversion methods for dicts, pd.Series, and SGP computation.

**Architecture:** New `models/player.py` module with pure dataclass definitions and conversion helpers. No consumers changed — this is foundation code only. Conversion methods bridge the gap between the current untyped world (dicts/Series) and the typed world, so adoption can happen incrementally in later phases.

**Tech Stack:** Python dataclasses, pandas (for Series conversion), existing SGP functions

---

### File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/fantasy_baseball/models/__init__.py` | Create | Package init, re-exports |
| `src/fantasy_baseball/models/player.py` | Create | All dataclass definitions + conversion methods |
| `tests/test_models/__init__.py` | Create | Test package init |
| `tests/test_models/test_player.py` | Create | Full test coverage |

---

### Task 1: HitterStats and PitcherStats dataclasses

**Files:**
- Create: `src/fantasy_baseball/models/__init__.py`
- Create: `src/fantasy_baseball/models/player.py`
- Create: `tests/test_models/__init__.py`
- Create: `tests/test_models/test_player.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_models/__init__.py` (empty file).

Create `tests/test_models/test_player.py`:

```python
import pytest
import pandas as pd


class TestHitterStats:
    def test_from_dict(self):
        from fantasy_baseball.models.player import HitterStats
        d = {"pa": 650, "ab": 550, "h": 160, "r": 100, "hr": 40, "rbi": 100, "sb": 5, "avg": 0.291}
        stats = HitterStats.from_dict(d)
        assert stats.pa == 650
        assert stats.hr == 40
        assert stats.avg == 0.291

    def test_from_dict_missing_keys_default_to_zero(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats.from_dict({"hr": 30})
        assert stats.hr == 30
        assert stats.pa == 0
        assert stats.avg == 0

    def test_from_series(self):
        from fantasy_baseball.models.player import HitterStats
        s = pd.Series({"pa": 650, "ab": 550, "h": 160, "r": 100, "hr": 40, "rbi": 100, "sb": 5, "avg": 0.291})
        stats = HitterStats.from_series(s)
        assert stats.hr == 40
        assert stats.avg == 0.291

    def test_to_dict(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291)
        d = stats.to_dict()
        assert d["hr"] == 40
        assert d["avg"] == 0.291
        assert "sgp" not in d  # None sgp excluded

    def test_to_dict_includes_sgp_when_set(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291, sgp=12.5)
        d = stats.to_dict()
        assert d["sgp"] == 12.5

    def test_to_series(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291)
        s = stats.to_series()
        assert s["hr"] == 40
        assert s["player_type"] == "hitter"

    def test_compute_avg_from_components(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats.from_dict({"h": 150, "ab": 500})
        assert stats.avg == pytest.approx(0.300)


class TestPitcherStats:
    def test_from_dict(self):
        from fantasy_baseball.models.player import PitcherStats
        d = {"ip": 200, "w": 15, "k": 220, "sv": 0, "er": 62, "bb": 40, "h_allowed": 150, "era": 2.79, "whip": 0.95}
        stats = PitcherStats.from_dict(d)
        assert stats.ip == 200
        assert stats.k == 220
        assert stats.era == 2.79

    def test_from_dict_computes_era_whip_from_components(self):
        from fantasy_baseball.models.player import PitcherStats
        stats = PitcherStats.from_dict({"ip": 180, "er": 60, "bb": 40, "h_allowed": 130})
        assert stats.era == pytest.approx(3.0)
        assert stats.whip == pytest.approx((40 + 130) / 180)

    def test_from_series(self):
        from fantasy_baseball.models.player import PitcherStats
        s = pd.Series({"ip": 200, "w": 15, "k": 220, "sv": 0, "er": 62, "bb": 40, "h_allowed": 150, "era": 2.79, "whip": 0.95})
        stats = PitcherStats.from_series(s)
        assert stats.k == 220

    def test_to_dict(self):
        from fantasy_baseball.models.player import PitcherStats
        stats = PitcherStats(ip=200, w=15, k=220, sv=0, er=62, bb=40, h_allowed=150, era=2.79, whip=0.95)
        d = stats.to_dict()
        assert d["k"] == 220
        assert d["era"] == 2.79

    def test_to_series(self):
        from fantasy_baseball.models.player import PitcherStats
        stats = PitcherStats(ip=200, w=15, k=220, sv=0, er=62, bb=40, h_allowed=150, era=2.79, whip=0.95)
        s = stats.to_series()
        assert s["player_type"] == "pitcher"
        assert s["k"] == 220
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_player.py -v`
Expected: FAIL — module not found

- [ ] **Step 3: Implement HitterStats and PitcherStats**

Create `src/fantasy_baseball/models/__init__.py`:

```python
from .player import HitterStats, PitcherStats, RankInfo, Player

__all__ = ["HitterStats", "PitcherStats", "RankInfo", "Player"]
```

Create `src/fantasy_baseball/models/player.py`:

```python
"""Strongly typed player data model.

Replaces the untyped dicts and pd.Series that currently flow through the
pipeline. Conversion methods bridge between the typed world and the
existing dict/Series interfaces for incremental adoption.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass
class HitterStats:
    pa: float = 0
    ab: float = 0
    h: float = 0
    r: float = 0
    hr: float = 0
    rbi: float = 0
    sb: float = 0
    avg: float = 0
    sgp: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HitterStats:
        stats = cls(
            pa=float(d.get("pa", 0) or 0),
            ab=float(d.get("ab", 0) or 0),
            h=float(d.get("h", 0) or 0),
            r=float(d.get("r", 0) or 0),
            hr=float(d.get("hr", 0) or 0),
            rbi=float(d.get("rbi", 0) or 0),
            sb=float(d.get("sb", 0) or 0),
            avg=float(d.get("avg", 0) or 0),
            sgp=d.get("sgp"),
        )
        # Compute avg from components if not provided
        if stats.avg == 0 and stats.ab > 0:
            stats.avg = stats.h / stats.ab
        return stats

    @classmethod
    def from_series(cls, s: pd.Series) -> HitterStats:
        return cls.from_dict(s.to_dict() if hasattr(s, "to_dict") else dict(s))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "pa": self.pa, "ab": self.ab, "h": self.h,
            "r": self.r, "hr": self.hr, "rbi": self.rbi, "sb": self.sb,
            "avg": self.avg,
        }
        if self.sgp is not None:
            d["sgp"] = self.sgp
        return d

    def to_series(self) -> pd.Series:
        d = self.to_dict()
        d["player_type"] = "hitter"
        return pd.Series(d)


@dataclass
class PitcherStats:
    ip: float = 0
    w: float = 0
    k: float = 0
    sv: float = 0
    er: float = 0
    bb: float = 0
    h_allowed: float = 0
    era: float = 0
    whip: float = 0
    sgp: float | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PitcherStats:
        stats = cls(
            ip=float(d.get("ip", 0) or 0),
            w=float(d.get("w", 0) or 0),
            k=float(d.get("k", 0) or 0),
            sv=float(d.get("sv", 0) or 0),
            er=float(d.get("er", 0) or 0),
            bb=float(d.get("bb", 0) or 0),
            h_allowed=float(d.get("h_allowed", 0) or 0),
            era=float(d.get("era", 0) or 0),
            whip=float(d.get("whip", 0) or 0),
            sgp=d.get("sgp"),
        )
        # Compute rate stats from components if not provided
        if stats.ip > 0:
            if stats.era == 0 and stats.er > 0:
                stats.era = stats.er * 9.0 / stats.ip
            if stats.whip == 0 and (stats.bb > 0 or stats.h_allowed > 0):
                stats.whip = (stats.bb + stats.h_allowed) / stats.ip
        return stats

    @classmethod
    def from_series(cls, s: pd.Series) -> PitcherStats:
        return cls.from_dict(s.to_dict() if hasattr(s, "to_dict") else dict(s))

    def to_dict(self) -> dict[str, Any]:
        d = {
            "ip": self.ip, "w": self.w, "k": self.k, "sv": self.sv,
            "er": self.er, "bb": self.bb, "h_allowed": self.h_allowed,
            "era": self.era, "whip": self.whip,
        }
        if self.sgp is not None:
            d["sgp"] = self.sgp
        return d

    def to_series(self) -> pd.Series:
        d = self.to_dict()
        d["player_type"] = "pitcher"
        return pd.Series(d)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_player.py -v`
Expected: All 12 pass

- [ ] **Step 5: Commit**

```bash
git add src/fantasy_baseball/models/__init__.py src/fantasy_baseball/models/player.py tests/test_models/__init__.py tests/test_models/test_player.py
git commit -m "feat: add HitterStats and PitcherStats dataclasses with conversions"
```

---

### Task 2: RankInfo and Player dataclasses

**Files:**
- Modify: `src/fantasy_baseball/models/player.py`
- Modify: `tests/test_models/test_player.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_models/test_player.py`:

```python
class TestRankInfo:
    def test_from_dict(self):
        from fantasy_baseball.models.player import RankInfo
        r = RankInfo.from_dict({"ros": 5, "preseason": 8, "current": 12})
        assert r.ros == 5
        assert r.preseason == 8
        assert r.current == 12

    def test_from_dict_missing_keys(self):
        from fantasy_baseball.models.player import RankInfo
        r = RankInfo.from_dict({"ros": 5})
        assert r.ros == 5
        assert r.preseason is None
        assert r.current is None

    def test_to_dict(self):
        from fantasy_baseball.models.player import RankInfo
        r = RankInfo(ros=5, preseason=8, current=12)
        assert r.to_dict() == {"ros": 5, "preseason": 8, "current": 12}

    def test_empty_rank(self):
        from fantasy_baseball.models.player import RankInfo
        r = RankInfo()
        assert r.ros is None


class TestPlayer:
    def test_from_dict_hitter(self):
        from fantasy_baseball.models.player import Player, HitterStats
        d = {
            "name": "Aaron Judge", "player_type": "hitter",
            "positions": ["OF", "DH"], "team": "NYY",
            "fg_id": "15640", "mlbam_id": 592450,
            "selected_position": "OF", "status": "",
            "wsgp": 12.5,
            "rank": {"ros": 2, "preseason": 1, "current": 3},
            "ros": {"pa": 600, "ab": 500, "h": 145, "r": 95, "hr": 38, "rbi": 92, "sb": 7, "avg": 0.290},
            "preseason": {"pa": 650, "ab": 550, "h": 160, "r": 110, "hr": 45, "rbi": 120, "sb": 5, "avg": 0.291},
        }
        p = Player.from_dict(d)
        assert p.name == "Aaron Judge"
        assert p.player_type == "hitter"
        assert p.fg_id == "15640"
        assert p.mlbam_id == 592450
        assert isinstance(p.ros, HitterStats)
        assert p.ros.hr == 38
        assert isinstance(p.preseason, HitterStats)
        assert p.preseason.hr == 45
        assert p.current is None
        assert p.wsgp == 12.5
        assert p.rank.ros == 2

    def test_from_dict_pitcher(self):
        from fantasy_baseball.models.player import Player, PitcherStats
        d = {
            "name": "Gerrit Cole", "player_type": "pitcher",
            "positions": ["P"], "team": "NYY",
            "ros": {"ip": 190, "w": 14, "k": 200, "sv": 0, "er": 60, "bb": 40, "h_allowed": 140, "era": 2.84, "whip": 0.95},
        }
        p = Player.from_dict(d)
        assert p.player_type == "pitcher"
        assert isinstance(p.ros, PitcherStats)
        assert p.ros.k == 200

    def test_to_dict_roundtrip(self):
        from fantasy_baseball.models.player import Player
        d = {
            "name": "Aaron Judge", "player_type": "hitter",
            "positions": ["OF"], "team": "NYY",
            "fg_id": "15640", "mlbam_id": 592450,
            "wsgp": 12.5,
            "rank": {"ros": 2, "preseason": 1, "current": 3},
            "ros": {"pa": 600, "ab": 500, "h": 145, "r": 95, "hr": 38, "rbi": 92, "sb": 7, "avg": 0.290},
        }
        p = Player.from_dict(d)
        result = p.to_dict()
        assert result["name"] == "Aaron Judge"
        assert result["ros"]["hr"] == 38
        assert result["rank"]["ros"] == 2
        assert result["wsgp"] == 12.5

    def test_from_dict_flat_stats_hitter(self):
        """Player.from_dict handles flat dicts where stats are top-level keys."""
        from fantasy_baseball.models.player import Player
        d = {
            "name": "Aaron Judge", "player_type": "hitter",
            "positions": ["OF"], "team": "NYY",
            "r": 95, "hr": 38, "rbi": 92, "sb": 7, "h": 145, "ab": 500, "pa": 600, "avg": 0.290,
        }
        p = Player.from_dict(d)
        assert p.ros is not None
        assert p.ros.hr == 38

    def test_from_dict_flat_stats_pitcher(self):
        """Player.from_dict handles flat dicts where stats are top-level keys."""
        from fantasy_baseball.models.player import Player
        d = {
            "name": "Gerrit Cole", "player_type": "pitcher",
            "positions": ["P"],
            "ip": 190, "w": 14, "k": 200, "sv": 0, "era": 2.84, "whip": 0.95,
        }
        p = Player.from_dict(d)
        assert p.ros is not None
        assert p.ros.k == 200

    def test_to_series(self):
        from fantasy_baseball.models.player import Player
        d = {
            "name": "Aaron Judge", "player_type": "hitter",
            "positions": ["OF"], "team": "NYY",
            "ros": {"pa": 600, "ab": 500, "h": 145, "r": 95, "hr": 38, "rbi": 92, "sb": 7, "avg": 0.290},
        }
        p = Player.from_dict(d)
        s = p.to_series()
        assert s["name"] == "Aaron Judge"
        assert s["player_type"] == "hitter"
        assert s["hr"] == 38
        assert s["positions"] == ["OF"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_player.py::TestRankInfo -v`
Expected: FAIL — RankInfo not found

- [ ] **Step 3: Implement RankInfo and Player**

Add to `src/fantasy_baseball/models/player.py`:

```python
@dataclass
class RankInfo:
    ros: int | None = None
    preseason: int | None = None
    current: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RankInfo:
        return cls(
            ros=d.get("ros"),
            preseason=d.get("preseason"),
            current=d.get("current"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {"ros": self.ros, "preseason": self.preseason, "current": self.current}


@dataclass
class Player:
    name: str
    player_type: str  # "hitter" | "pitcher"
    positions: list[str] = field(default_factory=list)
    team: str = ""
    fg_id: str | None = None
    mlbam_id: int | None = None
    yahoo_id: str | None = None

    ros: HitterStats | PitcherStats | None = None
    preseason: HitterStats | PitcherStats | None = None
    current: HitterStats | PitcherStats | None = None

    wsgp: float = 0.0
    rank: RankInfo = field(default_factory=RankInfo)

    selected_position: str = ""
    status: str = ""
    pace: dict | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Player:
        """Construct a Player from a dict.

        Handles two formats:
        1. Nested: {"ros": {"hr": 38, ...}, "preseason": {...}}
        2. Flat: {"hr": 38, "r": 95, ...} — stats at top level, treated as ROS
        """
        ptype = d.get("player_type", "hitter")
        stats_cls = HitterStats if ptype == "hitter" else PitcherStats

        # Parse stat bags
        ros = None
        if isinstance(d.get("ros"), dict):
            ros = stats_cls.from_dict(d["ros"])
        elif _has_stat_keys(d, ptype):
            ros = stats_cls.from_dict(d)

        preseason = None
        if isinstance(d.get("preseason"), dict):
            preseason = stats_cls.from_dict(d["preseason"])

        current = None
        if isinstance(d.get("current"), dict):
            current = stats_cls.from_dict(d["current"])

        rank = RankInfo()
        if isinstance(d.get("rank"), dict):
            rank = RankInfo.from_dict(d["rank"])

        return cls(
            name=d.get("name", ""),
            player_type=ptype,
            positions=d.get("positions", []),
            team=d.get("team", ""),
            fg_id=d.get("fg_id"),
            mlbam_id=int(d["mlbam_id"]) if d.get("mlbam_id") is not None else None,
            yahoo_id=d.get("yahoo_id") or d.get("player_id"),
            ros=ros,
            preseason=preseason,
            current=current,
            wsgp=float(d.get("wsgp", 0) or 0),
            rank=rank,
            selected_position=d.get("selected_position", ""),
            status=d.get("status", ""),
            pace=d.get("pace") or d.get("stats"),
        )

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "player_type": self.player_type,
            "positions": self.positions,
            "team": self.team,
            "wsgp": self.wsgp,
            "rank": self.rank.to_dict(),
            "selected_position": self.selected_position,
        }
        if self.fg_id is not None:
            d["fg_id"] = self.fg_id
        if self.mlbam_id is not None:
            d["mlbam_id"] = self.mlbam_id
        if self.yahoo_id is not None:
            d["player_id"] = self.yahoo_id
        if self.status:
            d["status"] = self.status
        if self.ros is not None:
            d["ros"] = self.ros.to_dict()
        if self.preseason is not None:
            d["preseason"] = self.preseason.to_dict()
        if self.current is not None:
            d["current"] = self.current.to_dict()
        if self.pace is not None:
            d["stats"] = self.pace
        return d

    def to_series(self) -> pd.Series:
        """Convert to pd.Series for backward compatibility with SGP functions.

        Flattens ROS stats to top-level keys (r, hr, rbi, etc.) since that's
        what calculate_player_sgp and calculate_weighted_sgp expect.
        """
        d: dict[str, Any] = {
            "name": self.name,
            "player_type": self.player_type,
            "positions": self.positions,
            "team": self.team,
        }
        if self.ros is not None:
            d.update(self.ros.to_dict())
        return pd.Series(d)


def _has_stat_keys(d: dict, player_type: str) -> bool:
    """Check if a dict has stat keys at the top level."""
    if player_type == "hitter":
        return "hr" in d or "r" in d or "rbi" in d
    return "ip" in d or "k" in d or "era" in d
```

Update `src/fantasy_baseball/models/__init__.py` to export RankInfo:

```python
from .player import HitterStats, PitcherStats, RankInfo, Player

__all__ = ["HitterStats", "PitcherStats", "RankInfo", "Player"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_player.py -v`
Expected: All 22 pass

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -q`
Expected: All tests pass

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/models/player.py src/fantasy_baseball/models/__init__.py tests/test_models/test_player.py
git commit -m "feat: add Player and RankInfo dataclasses with conversions"
```

---

### Task 3: SGP and wSGP computation methods

**Files:**
- Modify: `src/fantasy_baseball/models/player.py`
- Modify: `tests/test_models/test_player.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_models/test_player.py`:

```python
class TestSgpComputation:
    def test_hitter_stats_compute_sgp(self):
        from fantasy_baseball.models.player import HitterStats
        stats = HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291)
        sgp = stats.compute_sgp()
        assert sgp > 0
        assert stats.sgp == sgp  # cached on the instance

    def test_pitcher_stats_compute_sgp(self):
        from fantasy_baseball.models.player import PitcherStats
        stats = PitcherStats(ip=200, w=15, k=220, sv=0, er=62, bb=40, h_allowed=150, era=2.79, whip=0.95)
        sgp = stats.compute_sgp()
        assert sgp > 0
        assert stats.sgp == sgp

    def test_player_compute_wsgp(self):
        from fantasy_baseball.models.player import Player, HitterStats
        p = Player(
            name="Aaron Judge", player_type="hitter",
            ros=HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291),
        )
        leverage = {"R": 0.1, "HR": 0.1, "RBI": 0.1, "SB": 0.1, "AVG": 0.1,
                    "W": 0.1, "K": 0.1, "SV": 0.1, "ERA": 0.1, "WHIP": 0.1}
        wsgp = p.compute_wsgp(leverage)
        assert wsgp > 0
        assert p.wsgp == wsgp

    def test_player_compute_wsgp_no_ros_returns_zero(self):
        from fantasy_baseball.models.player import Player
        p = Player(name="Unknown", player_type="hitter")
        wsgp = p.compute_wsgp({"R": 0.1, "HR": 0.1, "RBI": 0.1, "SB": 0.1, "AVG": 0.1,
                               "W": 0.1, "K": 0.1, "SV": 0.1, "ERA": 0.1, "WHIP": 0.1})
        assert wsgp == 0.0

    def test_hitter_sgp_matches_calculate_player_sgp(self):
        """Verify our compute_sgp produces same result as the standalone function."""
        from fantasy_baseball.models.player import HitterStats
        from fantasy_baseball.sgp.player_value import calculate_player_sgp
        stats = HitterStats(pa=650, ab=550, h=160, r=100, hr=40, rbi=100, sb=5, avg=0.291)
        our_sgp = stats.compute_sgp()
        standalone_sgp = calculate_player_sgp(stats.to_series())
        assert our_sgp == pytest.approx(standalone_sgp)

    def test_pitcher_sgp_matches_calculate_player_sgp(self):
        from fantasy_baseball.models.player import PitcherStats
        from fantasy_baseball.sgp.player_value import calculate_player_sgp
        stats = PitcherStats(ip=200, w=15, k=220, sv=0, er=62, bb=40, h_allowed=150, era=2.79, whip=0.95)
        our_sgp = stats.compute_sgp()
        standalone_sgp = calculate_player_sgp(stats.to_series())
        assert our_sgp == pytest.approx(standalone_sgp)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_models/test_player.py::TestSgpComputation -v`
Expected: FAIL — `compute_sgp` not found

- [ ] **Step 3: Implement compute_sgp and compute_wsgp**

Add `compute_sgp()` method to `HitterStats`:

```python
    def compute_sgp(self) -> float:
        from fantasy_baseball.sgp.player_value import calculate_player_sgp
        self.sgp = calculate_player_sgp(self.to_series())
        return self.sgp
```

Add `compute_sgp()` method to `PitcherStats`:

```python
    def compute_sgp(self) -> float:
        from fantasy_baseball.sgp.player_value import calculate_player_sgp
        self.sgp = calculate_player_sgp(self.to_series())
        return self.sgp
```

Add `compute_wsgp()` method to `Player`:

```python
    def compute_wsgp(self, leverage: dict[str, float]) -> float:
        from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
        if self.ros is None:
            self.wsgp = 0.0
            return 0.0
        self.wsgp = calculate_weighted_sgp(self.to_series(), leverage)
        return self.wsgp
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_models/test_player.py -v`
Expected: All 28 pass

- [ ] **Step 5: Run full test suite**

Run: `pytest tests/ -q`
Expected: All pass

- [ ] **Step 6: Commit**

```bash
git add src/fantasy_baseball/models/player.py tests/test_models/test_player.py
git commit -m "feat: add compute_sgp() and compute_wsgp() methods to stat/player dataclasses"
```
