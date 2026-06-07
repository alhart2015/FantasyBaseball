from typing import Literal, overload

import pandas as pd

from fantasy_baseball.utils.constants import STARTER_IP_THRESHOLD


def _pitcher_floor_key(player: pd.Series, replacement_levels: dict[str, float]) -> str:
    """Pick the SP/RP empirical floor key for a pitcher by projected IP.

    Role is taken from ``IP >= STARTER_IP_THRESHOLD`` (matching
    ``scoring.py`` / ``playing_time.py``). The position token is deliberately
    ignored: the draft board carries no real SP/RP eligibility -- matched
    pitchers are stored as bare ``"P"`` and unmatched ones default to ``"SP"``
    (``board.py``), so a closer can be mislabeled ``"SP"``. Falls back to the
    unified ``"P"`` floor when role floors are absent (demand-based dict).
    """
    ip = player.get("ip", 0.0)
    ip = float(ip) if ip is not None else 0.0
    role = "SP" if ip >= STARTER_IP_THRESHOLD else "RP"
    return role if role in replacement_levels else "P"


@overload
def calculate_var(
    player: pd.Series,
    replacement_levels: dict[str, float],
    return_position: Literal[False] = False,
) -> float: ...


@overload
def calculate_var(
    player: pd.Series,
    replacement_levels: dict[str, float],
    return_position: Literal[True],
) -> tuple[float, str]: ...


def calculate_var(
    player: pd.Series,
    replacement_levels: dict[str, float],
    return_position: bool = False,
) -> float | tuple[float, str]:
    """Calculate Value Above Replacement for a player."""
    total_sgp = player["total_sgp"]
    positions = player["positions"]

    best_var = float("-inf")
    best_pos = None

    for pos in positions:
        if pos in ("P", "SP", "RP"):
            # Pitchers net against the role's empirical floor (SP vs RP), but
            # report "P" -- the slot they fill -- so recommender display that
            # groups by best_position is unaffected.
            floor_key = _pitcher_floor_key(player, replacement_levels)
            report_pos = "P"
        else:
            floor_key = pos
            report_pos = pos
        if floor_key in replacement_levels:
            var = total_sgp - replacement_levels[floor_key]
            if var > best_var:
                best_var = var
                best_pos = report_pos

    if best_pos is None:
        # Fall back to UTIL replacement level (e.g. DH-only players).
        util_repl = replacement_levels.get("UTIL", 0.0)
        best_var = total_sgp - util_repl
        best_pos = "UTIL"

    if return_position:
        return best_var, best_pos
    return best_var
