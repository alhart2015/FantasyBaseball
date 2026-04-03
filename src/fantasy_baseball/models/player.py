from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Optional


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
    sgp: Optional[float] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HitterStats":
        stat_fields = {f.name for f in fields(cls) if f.name != "sgp"}
        kwargs: dict[str, Any] = {k: float(d.get(k, 0) or 0) for k in stat_fields}
        kwargs["sgp"] = d.get("sgp", None)

        # Compute avg from h/ab if avg not provided or is zero
        if kwargs["avg"] == 0 and kwargs["ab"] > 0:
            kwargs["avg"] = kwargs["h"] / kwargs["ab"]

        return cls(**kwargs)



    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "sgp"}
        if self.sgp is not None:
            d["sgp"] = self.sgp
        return d

    def compute_sgp(self) -> float:
        from fantasy_baseball.sgp.player_value import calculate_player_sgp
        self.sgp = calculate_player_sgp(self)
        return self.sgp

    def is_significant(self, cat: str) -> bool:
        """Check if this stat has enough sample to be empirically significant."""
        from fantasy_baseball.utils.constants import STABILIZATION_THRESHOLDS
        entry = STABILIZATION_THRESHOLDS.get(cat)
        if entry is None:
            return True  # No threshold — always significant
        threshold, unit = entry
        if unit == "pa":
            return self.pa >= threshold
        return True  # Hitters don't use BF-based thresholds

    def significant_dict(self) -> dict[str, bool]:
        """Return significance for all 5 hitting roto categories."""
        return {cat: self.is_significant(cat) for cat in ["R", "HR", "RBI", "SB", "AVG"]}


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
    sgp: Optional[float] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PitcherStats":
        stat_fields = {f.name for f in fields(cls) if f.name != "sgp"}
        kwargs: dict[str, Any] = {k: float(d.get(k, 0) or 0) for k in stat_fields}
        kwargs["sgp"] = d.get("sgp", None)

        # Compute ERA and WHIP from components if not provided
        ip = kwargs["ip"]
        if ip > 0:
            if kwargs["era"] == 0:
                kwargs["era"] = kwargs["er"] * 9 / ip
            if kwargs["whip"] == 0:
                kwargs["whip"] = (kwargs["bb"] + kwargs["h_allowed"]) / ip

        return cls(**kwargs)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "sgp"}
        if self.sgp is not None:
            d["sgp"] = self.sgp
        return d

    def compute_sgp(self) -> float:
        from fantasy_baseball.sgp.player_value import calculate_player_sgp
        self.sgp = calculate_player_sgp(self)
        return self.sgp

    def is_significant(self, cat: str) -> bool:
        """Check if this stat has enough sample to be empirically significant."""
        from fantasy_baseball.utils.constants import STABILIZATION_THRESHOLDS
        entry = STABILIZATION_THRESHOLDS.get(cat)
        if entry is None:
            return True  # No threshold — always significant
        threshold, unit = entry
        if unit == "bf":
            bf = self.ip * 3 + self.h_allowed + self.bb
            return bf >= threshold
        return True  # Pitchers don't use PA-based thresholds

    def significant_dict(self) -> dict[str, bool]:
        """Return significance for all 5 pitching roto categories."""
        return {cat: self.is_significant(cat) for cat in ["W", "K", "SV", "ERA", "WHIP"]}


# ---------------------------------------------------------------------------
# RankInfo
# ---------------------------------------------------------------------------

@dataclass
class RankInfo:
    ros: Optional[int] = None
    preseason: Optional[int] = None
    current: Optional[int] = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RankInfo":
        return cls(ros=d.get("ros"), preseason=d.get("preseason"), current=d.get("current"))

    def to_dict(self) -> dict[str, Any]:
        return {"ros": self.ros, "preseason": self.preseason, "current": self.current}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _has_stat_keys(d: dict[str, Any], player_type: str) -> bool:
    """Return True if d contains top-level stat keys for the given player_type."""
    if player_type == "pitcher":
        return any(k in d for k in ("ip", "k", "era"))
    # hitter (default)
    return any(k in d for k in ("hr", "r", "rbi"))


def _make_stats(
    d: dict[str, Any], player_type: str
) -> "HitterStats | PitcherStats | None":
    """Build the appropriate stats object from a sub-dict."""
    if not d:
        return None
    if player_type == "pitcher":
        return PitcherStats.from_dict(d)
    return HitterStats.from_dict(d)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------

@dataclass
class Player:
    name: str
    player_type: str  # "hitter" | "pitcher"
    positions: list[str] = field(default_factory=list)
    team: str = ""
    fg_id: Optional[str] = None
    mlbam_id: Optional[int] = None
    yahoo_id: Optional[str] = None

    ros: "HitterStats | PitcherStats | None" = None
    preseason: "HitterStats | PitcherStats | None" = None
    current: "HitterStats | PitcherStats | None" = None

    wsgp: float = 0.0
    rank: RankInfo = field(default_factory=RankInfo)

    selected_position: str = ""
    status: str = ""
    pace: Optional[dict] = None

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Player":
        name = d.get("name", "")
        player_type = d.get("player_type", "hitter")

        # Stat bags: detect nested vs flat format
        if _has_stat_keys(d, player_type):
            # Flat format — all stats at top level, treat as ROS
            ros = _make_stats(d, player_type)
            preseason = None
            current = None
        else:
            ros_raw = d.get("ros")
            ros = _make_stats(ros_raw, player_type) if ros_raw is not None else None

            pre_raw = d.get("preseason")
            preseason = _make_stats(pre_raw, player_type) if pre_raw is not None else None

            cur_raw = d.get("current")
            current = _make_stats(cur_raw, player_type) if cur_raw is not None else None

        rank_raw = d.get("rank")
        rank = RankInfo.from_dict(rank_raw) if isinstance(rank_raw, dict) else RankInfo()

        return cls(
            name=name,
            player_type=player_type,
            positions=d.get("positions", []),
            team=d.get("team", ""),
            fg_id=d.get("fg_id"),
            mlbam_id=d.get("mlbam_id"),
            yahoo_id=d.get("player_id"),  # cache format uses "player_id"
            ros=ros,
            preseason=preseason,
            current=current,
            wsgp=d.get("wsgp", 0.0),
            rank=rank,
            selected_position=d.get("selected_position", ""),
            status=d.get("status", ""),
            pace=d.get("pace") or d.get("stats"),  # "pace" preferred, "stats" for legacy cache
        )

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "name": self.name,
            "player_type": self.player_type,
            "positions": self.positions,
            "team": self.team,
        }
        if self.fg_id is not None:
            d["fg_id"] = self.fg_id
        if self.mlbam_id is not None:
            d["mlbam_id"] = self.mlbam_id
        if self.yahoo_id is not None:
            d["player_id"] = self.yahoo_id
        if self.ros is not None:
            d["ros"] = self.ros.to_dict()
        if self.preseason is not None:
            d["preseason"] = self.preseason.to_dict()
        if self.current is not None:
            d["current"] = self.current.to_dict()
        d["wsgp"] = self.wsgp
        d["rank"] = self.rank.to_dict()
        if self.selected_position:
            d["selected_position"] = self.selected_position
        if self.status:
            d["status"] = self.status
        if self.pace is not None:
            d["pace"] = self.pace
            d["stats"] = self.pace  # legacy key for cache/template compatibility
        return d

    def to_flat_dict(self) -> dict[str, Any]:
        """Serialize with ROS stats flattened to top level for legacy consumers.

        Produces both flat keys (r, hr, rbi...) AND nested ros dict.
        Used for cache serialization and backward compatibility with
        functions that expect flat stat keys.
        """
        d = self.to_dict()
        if self.ros is not None:
            d.update(self.ros.to_dict())
        return d

    def compute_wsgp(self, leverage: dict[str, float]) -> float:
        from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
        if self.ros is None:
            self.wsgp = 0.0
            return 0.0
        self.wsgp = calculate_weighted_sgp(self.ros, leverage)
        return self.wsgp
