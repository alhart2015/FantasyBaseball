"""Position enum and eligibility sets.

Canonical uppercase values match the `roster_slots` config keys. The
``parse`` classmethod normalizes Yahoo's inconsistent casing (e.g.,
``"Util"`` → :attr:`Position.UTIL`) at the loader boundary so downstream
code never has to worry about it.
"""

from __future__ import annotations

from enum import StrEnum


class Position(StrEnum):
    # Hitter-eligible + starter slots
    C           = "C"
    FIRST_BASE  = "1B"
    SECOND_BASE = "2B"
    THIRD_BASE  = "3B"
    SS          = "SS"
    IF          = "IF"        # infield flex
    OF          = "OF"
    DH          = "DH"
    UTIL        = "UTIL"
    # Pitcher-eligible + pitcher slots
    P           = "P"
    SP          = "SP"
    RP          = "RP"
    # Non-active slots
    BN          = "BN"
    IL          = "IL"
    IL_PLUS     = "IL+"
    DL          = "DL"         # legacy — kept for parsing historical snapshots
    DL_PLUS     = "DL+"

    @classmethod
    def parse(cls, s: str) -> "Position":
        """Parse a position string, normalizing casing.

        Yahoo returns ``"Util"`` in eligible_positions but ``"UTIL"`` in
        some other fields; config uses ``"UTIL"``. This method accepts
        any casing and returns the canonical enum member.

        Raises:
            ValueError: if ``s`` is empty or does not match any member.
        """
        if not s:
            raise ValueError(f"Unknown position: {s!r}")
        norm = s.strip().upper()
        try:
            return cls(norm)
        except ValueError:
            raise ValueError(f"Unknown position: {s!r}") from None

    @classmethod
    def parse_list(cls, s: str | None) -> list["Position"]:
        """Parse a comma-separated position string into a list.

        Empty string and ``None`` return ``[]``. Used by the DB loader
        which stores positions as a joined string in weekly_rosters.
        """
        if not s:
            return []
        return [cls.parse(tok) for tok in s.split(",") if tok.strip()]


HITTER_ELIGIBLE: frozenset[Position] = frozenset({
    Position.C, Position.FIRST_BASE, Position.SECOND_BASE,
    Position.THIRD_BASE, Position.SS, Position.IF, Position.OF,
    Position.DH, Position.UTIL,
})

PITCHER_ELIGIBLE: frozenset[Position] = frozenset({
    Position.P, Position.SP, Position.RP,
})

BENCH_SLOTS: frozenset[Position] = frozenset({
    Position.BN, Position.IL, Position.IL_PLUS,
    Position.DL, Position.DL_PLUS,
})

IL_SLOTS: frozenset[Position] = frozenset({
    Position.IL, Position.IL_PLUS, Position.DL, Position.DL_PLUS,
})
