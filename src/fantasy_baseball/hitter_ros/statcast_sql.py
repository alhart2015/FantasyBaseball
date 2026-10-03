"""SQL fragments over Statcast pitch rows, shared by the training table and the PA tokens.

One copy of the spray-angle formula and of what counts as a swing, whiff or contact, so
the window features (``table.py``) and the plate-appearance tokens (``sequence.py``)
can't drift apart.
"""

from __future__ import annotations

from collections.abc import Iterable

from fantasy_baseball.keepers.savant import (
    CONTACT_DESCRIPTIONS,
    SWING_DESCRIPTIONS,
    WHIFF_DESCRIPTIONS,
)


def sql_in(values: Iterable[str]) -> str:
    """``('a', 'b')`` for a SQL ``IN`` list."""
    return "(" + ", ".join(f"'{v}'" for v in sorted(values)) + ")"


# Spray angle in degrees from home plate; negative = toward left field. Savant's
# hit-coordinate origin (125.42, 198.27) is home plate. atan2, not atan of the ratio:
# a ball fielded behind home's y origin (hc_y > 198.27) would otherwise flip sides.
SPRAY_SQL = "degrees(atan2(hc_x - 125.42, 198.27 - hc_y))"

SWING_SQL = f"description IN {sql_in(SWING_DESCRIPTIONS)}"
WHIFF_SQL = f"description IN {sql_in(WHIFF_DESCRIPTIONS)}"
CONTACT_SQL = f"description IN {sql_in(CONTACT_DESCRIPTIONS)}"
