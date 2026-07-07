from typing import Literal, overload

import pandas as pd

from fantasy_baseball.utils.constants import role_from_ip


def _pitcher_floor_key(
    player: pd.Series,
    replacement_levels: dict[str, float],
    role_ip: float | None = None,
) -> str:
    """Pick the SP/RP empirical floor key for a pitcher by projected IP.

    Role comes from the shared ``role_from_ip`` classifier (NaN/None-safe).
    The position token is deliberately ignored: the draft board carries no
    real SP/RP eligibility -- matched pitchers are stored as bare ``"P"`` and
    unmatched ones default to ``"SP"`` (``board.py``), so a closer can be
    mislabeled ``"SP"``. Falls back to the unified ``"P"`` floor when role
    floors are absent (demand-based dict).

    ``role_ip`` overrides the IP used for role classification. SP/RP is a
    full-season ROLE, so a mid-season caller whose ``player["ip"]`` is a
    partial to-date total must pass the full-season-equivalent IP here to keep
    the actual and par sides on the same side of the role cutoff. ``None`` uses
    ``player["ip"]`` (the projected full-season case).
    """
    ip = player.get("ip", 0.0) if role_ip is None else role_ip
    role = role_from_ip(ip)
    return role if role in replacement_levels else "P"


@overload
def calculate_var(
    player: pd.Series,
    replacement_levels: dict[str, float],
    return_position: Literal[False] = False,
    *,
    role_ip: float | None = None,
) -> float: ...


@overload
def calculate_var(
    player: pd.Series,
    replacement_levels: dict[str, float],
    return_position: Literal[True],
    *,
    role_ip: float | None = None,
) -> tuple[float, str]: ...


def calculate_var(
    player: pd.Series,
    replacement_levels: dict[str, float],
    return_position: bool = False,
    *,
    role_ip: float | None = None,
) -> float | tuple[float, str]:
    """Calculate Value Above Replacement for a player.

    ``role_ip`` (keyword-only) overrides the IP used to route a pitcher to the
    SP vs RP empirical floor. Pass a full-season-equivalent IP when scoring a
    partial to-date pitcher line so the role does not flip at the mid-season IP
    cutoff; ``None`` routes by ``player["ip"]``.
    """
    total_sgp = player["total_sgp"]
    positions = player["positions"]

    best_var = float("-inf")
    best_pos = None

    for pos in positions:
        if pos in ("P", "SP", "RP"):
            # Pitchers net against the role's empirical floor (SP vs RP), but
            # report "P" -- the slot they fill -- so recommender display that
            # groups by best_position is unaffected.
            floor_key = _pitcher_floor_key(player, replacement_levels, role_ip=role_ip)
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
