"""Free agent pool — NOT part of League per the scoping decision.

A :class:`FreeAgentPool` is a point-in-time list of
:class:`RosterEntry` objects representing players who aren't on any
team's latest roster. It has its own loaders (``from_yahoo``,
``from_cache``) and is passed explicitly to any analysis function that
needs it (audit, waivers).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

from fantasy_baseball.models.positions import Position
from fantasy_baseball.models.roster import RosterEntry
from fantasy_baseball.utils.time_utils import local_today


@dataclass
class FreeAgentPool:
    effective_date: date
    entries: list[RosterEntry] = field(default_factory=list)

    def __iter__(self) -> Iterator[RosterEntry]:
        return iter(self.entries)

    def __len__(self) -> int:
        return len(self.entries)

    def names(self) -> set[str]:
        return {e.name for e in self.entries}

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_yahoo_entries(raw_entries: list[dict[str, Any]]) -> list[RosterEntry]:
        """Convert raw Yahoo FA dicts to RosterEntry.

        Unknown position tokens are dropped silently (FA data quality
        is lower than roster data; we'd rather lose a position chip
        than crash the refresh). Entries that end up with zero
        recognized positions are skipped entirely.
        """
        parsed: list[RosterEntry] = []
        for p in raw_entries:
            positions: list[Position] = []
            for tok in p.get("positions", []):
                try:
                    positions.append(Position.parse(tok))
                except ValueError:
                    continue
            if not positions:
                continue
            parsed.append(
                RosterEntry(
                    name=p.get("name", ""),
                    positions=positions,
                    # FAs have no assigned slot; BN is the closest meaning
                    selected_position=Position.BN,
                    status=p.get("status", ""),
                    yahoo_id=p.get("player_id", ""),
                )
            )
        return parsed

    @classmethod
    def from_yahoo(
        cls,
        yahoo_league,
        positions: list[Position] | None = None,
    ) -> FreeAgentPool:
        """Fetch the free-agent pool live from Yahoo.

        Args:
            yahoo_league: The ``yahoo_fantasy_api.League`` instance.
            positions: If given, restricts the fetch to these positions.
                Default fetches C, 1B, 2B, 3B, SS, OF, SP, RP — matches
                the list in ``lineup.waivers.fetch_and_match_free_agents``.
                UTIL is intentionally excluded: Yahoo's player-fetch API
                requires primary positions, and UTIL would duplicate
                hitters already returned by their primary slot fetches.
        """
        from fantasy_baseball.lineup.yahoo_roster import fetch_free_agents

        if positions is None:
            positions = [
                Position.C,
                Position.FIRST_BASE,
                Position.SECOND_BASE,
                Position.THIRD_BASE,
                Position.SS,
                Position.OF,
                Position.SP,
                Position.RP,
            ]

        raw: list[dict[str, Any]] = []
        seen_names: set[str] = set()
        for pos in positions:
            for player in fetch_free_agents(yahoo_league, pos.value):
                if player["name"] in seen_names:
                    continue
                seen_names.add(player["name"])
                raw.append(player)

        return cls(
            effective_date=local_today(),
            entries=cls._parse_yahoo_entries(raw),
        )
