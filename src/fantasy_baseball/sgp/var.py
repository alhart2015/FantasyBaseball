import pandas as pd
from fantasy_baseball.utils.positions import is_pitcher


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
        lookup_pos = "P" if pos in ("P", "SP", "RP") else pos
        if lookup_pos in replacement_levels:
            var = total_sgp - replacement_levels[lookup_pos]
            if var > best_var:
                best_var = var
                best_pos = lookup_pos

    if best_pos is None:
        best_var = total_sgp
        best_pos = positions[0] if positions else "UTIL"

    if return_position:
        return best_var, best_pos
    return best_var
