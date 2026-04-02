from __future__ import annotations

from dataclasses import dataclass, field, fields, asdict
from typing import Any, Optional

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
    sgp: Optional[float] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HitterStats":
        stat_fields = {f.name for f in fields(cls) if f.name != "sgp"}
        kwargs: dict[str, Any] = {k: d.get(k, 0) for k in stat_fields}
        kwargs["sgp"] = d.get("sgp", None)

        # Compute avg from h/ab if avg not provided or is zero
        if kwargs["avg"] == 0 and kwargs["ab"] > 0:
            kwargs["avg"] = kwargs["h"] / kwargs["ab"]

        return cls(**kwargs)

    @classmethod
    def from_series(cls, s: pd.Series) -> "HitterStats":
        return cls.from_dict(s.to_dict())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "sgp"}
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
    sgp: Optional[float] = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PitcherStats":
        stat_fields = {f.name for f in fields(cls) if f.name != "sgp"}
        kwargs: dict[str, Any] = {k: d.get(k, 0) for k in stat_fields}
        kwargs["sgp"] = d.get("sgp", None)

        # Compute ERA and WHIP from components if not provided
        ip = kwargs["ip"]
        if ip > 0:
            if kwargs["era"] == 0:
                kwargs["era"] = kwargs["er"] * 9 / ip
            if kwargs["whip"] == 0:
                kwargs["whip"] = (kwargs["bb"] + kwargs["h_allowed"]) / ip

        return cls(**kwargs)

    @classmethod
    def from_series(cls, s: pd.Series) -> "PitcherStats":
        return cls.from_dict(s.to_dict())

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        d = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "sgp"}
        if self.sgp is not None:
            d["sgp"] = self.sgp
        return d

    def to_series(self) -> pd.Series:
        d = self.to_dict()
        d["player_type"] = "pitcher"
        return pd.Series(d)
