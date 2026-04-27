from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import StrEnum
from typing import Any

from fantasy_baseball.models.positions import IL_SLOTS, Position
from fantasy_baseball.utils.constants import IL_STATUSES
from fantasy_baseball.utils.rate_stats import calculate_avg, calculate_era, calculate_whip


class PlayerType(StrEnum):
    HITTER = "hitter"
    PITCHER = "pitcher"


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

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> HitterStats:
        stat_fields = {f.name for f in fields(cls) if f.name != "sgp"}
        kwargs: dict[str, Any] = {k: float(d.get(k, 0) or 0) for k in stat_fields}
        kwargs["sgp"] = d.get("sgp")

        # Compute avg from h/ab if avg not provided or is zero
        if kwargs["avg"] == 0:
            kwargs["avg"] = calculate_avg(kwargs["h"], kwargs["ab"])

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

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> PitcherStats:
        stat_fields = {f.name for f in fields(cls) if f.name != "sgp"}
        kwargs: dict[str, Any] = {k: float(d.get(k, 0) or 0) for k in stat_fields}
        kwargs["sgp"] = d.get("sgp")

        # Compute ERA and WHIP from components if not provided
        ip = kwargs["ip"]
        if ip > 0:
            if kwargs["era"] == 0:
                kwargs["era"] = calculate_era(kwargs["er"], ip)
            if kwargs["whip"] == 0:
                kwargs["whip"] = calculate_whip(kwargs["bb"], kwargs["h_allowed"], ip)

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


# ---------------------------------------------------------------------------
# RankInfo
# ---------------------------------------------------------------------------


@dataclass
class RankInfo:
    rest_of_season: int | None = None
    preseason: int | None = None
    current: int | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> RankInfo:
        return cls(
            rest_of_season=d.get("rest_of_season"),
            preseason=d.get("preseason"),
            current=d.get("current"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "rest_of_season": self.rest_of_season,
            "preseason": self.preseason,
            "current": self.current,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _has_stat_keys(d: dict[str, Any], player_type: PlayerType) -> bool:
    """Return True if d contains top-level stat keys for the given player_type."""
    if player_type == PlayerType.PITCHER:
        return any(k in d for k in ("ip", "k", "era"))
    return any(k in d for k in ("hr", "r", "rbi"))


def _make_stats(d: dict[str, Any], player_type: PlayerType) -> HitterStats | PitcherStats | None:
    """Build the appropriate stats object from a sub-dict."""
    if not d:
        return None
    if player_type == PlayerType.PITCHER:
        return PitcherStats.from_dict(d)
    return HitterStats.from_dict(d)


# ---------------------------------------------------------------------------
# Player
# ---------------------------------------------------------------------------


@dataclass
class Player:
    name: str
    player_type: PlayerType
    positions: list[Position] = field(default_factory=list)
    team: str = ""
    fg_id: str | None = None
    mlbam_id: int | None = None
    yahoo_id: str | None = None

    rest_of_season: HitterStats | PitcherStats | None = None
    full_season_projection: HitterStats | PitcherStats | None = None
    preseason: HitterStats | PitcherStats | None = None
    current: HitterStats | PitcherStats | None = None

    rank: RankInfo = field(default_factory=RankInfo)

    selected_position: Position | None = None
    status: str = ""
    pace: dict[str, Any] | None = None

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Player:
        name = d.get("name", "")
        player_type = PlayerType(d.get("player_type", "hitter"))

        # Stat bags: detect nested vs flat format
        if _has_stat_keys(d, player_type):
            # Flat format — all stats at top level, treat as ROS
            ros = _make_stats(d, player_type)
            preseason = None
            current = None
            full_season_projection = None
        else:
            ros_raw = d.get("rest_of_season")
            ros = _make_stats(ros_raw, player_type) if ros_raw is not None else None

            fs_raw = d.get("full_season_projection")
            full_season_projection = (
                _make_stats(fs_raw, player_type) if fs_raw is not None else None
            )

            pre_raw = d.get("preseason")
            preseason = _make_stats(pre_raw, player_type) if pre_raw is not None else None

            cur_raw = d.get("current")
            current = _make_stats(cur_raw, player_type) if cur_raw is not None else None

        rank_raw = d.get("rank")
        rank = RankInfo.from_dict(rank_raw) if isinstance(rank_raw, dict) else RankInfo()

        raw_positions = d.get("positions", [])
        parsed_positions = [
            p if isinstance(p, Position) else Position.parse(p) for p in raw_positions
        ]

        raw_slot = d.get("selected_position")
        parsed_slot: Position | None
        if raw_slot is None or raw_slot == "":
            parsed_slot = None
        elif isinstance(raw_slot, Position):
            parsed_slot = raw_slot
        else:
            parsed_slot = Position.parse(raw_slot)

        return cls(
            name=name,
            player_type=player_type,
            positions=parsed_positions,
            team=d.get("team", ""),
            fg_id=d.get("fg_id"),
            mlbam_id=d.get("mlbam_id"),
            yahoo_id=d.get("player_id"),  # cache format uses "player_id"
            rest_of_season=ros,
            full_season_projection=full_season_projection,
            preseason=preseason,
            current=current,
            rank=rank,
            selected_position=parsed_slot,
            status=d.get("status", ""),
            pace=d.get("pace"),
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
        if self.rest_of_season is not None:
            d["rest_of_season"] = self.rest_of_season.to_dict()
        if self.full_season_projection is not None:
            d["full_season_projection"] = self.full_season_projection.to_dict()
        if self.preseason is not None:
            d["preseason"] = self.preseason.to_dict()
        if self.current is not None:
            d["current"] = self.current.to_dict()
        d["rank"] = self.rank.to_dict()
        if self.selected_position:
            d["selected_position"] = self.selected_position
        if self.status:
            d["status"] = self.status
        if self.pace is not None:
            d["pace"] = self.pace
        return d

    def to_flat_dict(self) -> dict[str, Any]:
        """Serialize with ROS stats flattened to top level for legacy consumers.

        Produces both flat keys (r, hr, rbi...) AND nested ros dict.
        Used for cache serialization and backward compatibility with
        functions that expect flat stat keys.
        """
        d = self.to_dict()
        if self.rest_of_season is not None:
            d.update(self.rest_of_season.to_dict())
        return d

    def is_on_il(self) -> bool:
        """True if the player is on the IL by Yahoo status or selected slot.

        Yahoo roster data has three production shapes that all mean IL:
          - status='IL10' + slot='BN' (bench-slotted IL — status check catches it)
          - status='IL15' + slot='IL' (formally on IL — both checks catch it)
          - status=''     + slot='IL' (freshly slotted, status not yet propagated
            — only the slot check catches it)
        Either signal alone is enough.
        """
        if self.status in IL_STATUSES:
            return True
        slot = self.selected_position
        return slot is not None and slot in IL_SLOTS
