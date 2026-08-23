"""Position enum and eligibility sets.

Canonical uppercase values match the `roster_slots` config keys. The
``parse`` classmethod normalizes Yahoo's inconsistent casing (e.g.,
``"Util"`` → :attr:`Position.UTIL`) at the loader boundary so downstream
code never has to worry about it.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - `player` imports THIS module, so the edge
    # only exists for the type checker. A runtime import here is a genuine cycle:
    # models.player line 8 imports models.positions.
    from .player import PlayerType

# Matches trailing digits used to disambiguate same-named slots in
# historical roster JSON files and legacy DB rows (e.g. "OF2", "BN3",
# "P5"). Leading digits in "1B"/"2B"/"3B" are preserved because the
# regex anchors on the end of the string.
_TRAILING_DIGITS = re.compile(r"\d+$")


class Position(StrEnum):
    # Hitter-eligible + starter slots
    C = "C"
    FIRST_BASE = "1B"
    SECOND_BASE = "2B"
    THIRD_BASE = "3B"
    SS = "SS"
    IF = "IF"  # infield flex
    OF = "OF"
    DH = "DH"
    UTIL = "UTIL"
    # Pitcher-eligible + pitcher slots
    P = "P"
    SP = "SP"
    RP = "RP"
    # Non-active slots
    BN = "BN"
    IL = "IL"
    IL_PLUS = "IL+"
    DL = "DL"  # legacy — kept for parsing historical snapshots
    DL_PLUS = "DL+"

    @classmethod
    def parse(cls, s: str) -> Position:
        """Parse a position string, normalizing casing and numbered slots.

        Yahoo returns ``"Util"`` in eligible_positions but ``"UTIL"`` in
        some other fields; config uses ``"UTIL"``. This method accepts
        any casing and returns the canonical enum member.

        Historical roster snapshots (from JSON files loaded via
        ``load_weekly_rosters``) use trailing digits to disambiguate
        multiple same-named slots — ``"OF2"``, ``"BN3"``, ``"P5"``.
        Those are collapsed to their base position so
        :class:`Position.OF` / ``BN`` / ``P`` round-trip cleanly. The
        leading digits in ``"1B"`` / ``"2B"`` / ``"3B"`` are preserved
        because only trailing digits are stripped.

        Raises:
            ValueError: if ``s`` is empty or does not match any member
                after normalization.
        """
        if not s:
            raise ValueError(f"Unknown position: {s!r}")
        norm = _TRAILING_DIGITS.sub("", s.strip().upper())
        if not norm:
            raise ValueError(f"Unknown position: {s!r}")
        try:
            return cls(norm)
        except ValueError:
            raise ValueError(f"Unknown position: {s!r}") from None

    @classmethod
    def parse_list(cls, s: str | None) -> list[Position]:
        """Parse a comma-separated position string into a list.

        Empty string and ``None`` return ``[]``. Used by the DB loader
        which stores positions as a joined string in weekly_rosters.
        """
        if not s:
            return []
        return [cls.parse(tok) for tok in s.split(",") if tok.strip()]


HITTER_ELIGIBLE: frozenset[Position] = frozenset(
    {
        Position.C,
        Position.FIRST_BASE,
        Position.SECOND_BASE,
        Position.THIRD_BASE,
        Position.SS,
        Position.IF,
        Position.OF,
        Position.DH,
        Position.UTIL,
    }
)

PITCHER_ELIGIBLE: frozenset[Position] = frozenset(
    {
        Position.P,
        Position.SP,
        Position.RP,
    }
)


class UnknownPositions(ValueError):
    """An eligibility string that does not parse into any known slot.

    Its own type so a caller can decide: `manual.transcripts` validates positions
    before it ever asks, so this is unreachable there; `data.rosters` joins roster
    rows to a board and NAMES the players it could not place, because guessing a
    half would drop them off the board with nothing on screen to say so.
    """


def player_type_for_positions(positions: str) -> PlayerType:
    """Hitter or pitcher, from the eligibility string Yahoo prints.

    The only type signal a roster row carries: roster blobs have no ``mlbam_id``
    and a hand transcription has no projection frame loaded at parse time. Any
    pitcher-eligible slot makes it a pitcher -- Yahoo prints "P" for every
    pitcher in this league, and a two-way player appears as TWO rows, "Util" and
    "P", which is exactly the split ``name::player_type`` exists to keep apart.

    Raises:
        UnknownPositions: the string parsed to nothing. Deliberately NOT defaulted
            to hitter: `player_type` is half the join key, so a guess does not
            degrade the answer, it silently attributes the row to a player who
            does not exist.
    """
    from .player import PlayerType

    try:
        parsed = Position.parse_list(positions or "")
    except ValueError as exc:
        raise UnknownPositions(f"cannot parse positions {positions!r}: {exc}") from exc
    if not parsed:
        raise UnknownPositions(f"positions {positions!r} parsed to no known slot")
    return PlayerType.PITCHER if any(p in PITCHER_ELIGIBLE for p in parsed) else PlayerType.HITTER


BENCH_SLOTS: frozenset[Position] = frozenset(
    {
        Position.BN,
        Position.IL,
        Position.IL_PLUS,
        Position.DL,
        Position.DL_PLUS,
    }
)

IL_SLOTS: frozenset[Position] = frozenset(
    {
        Position.IL,
        Position.IL_PLUS,
        Position.DL,
        Position.DL_PLUS,
    }
)
