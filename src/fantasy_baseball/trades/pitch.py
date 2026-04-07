"""Generate human-readable trade pitches for opponents."""


def generate_pitch(
    send_rank: int,
    receive_rank: int,
    send_positions: list[str],
    receive_positions: list[str],
) -> str:
    """Generate a 1-2 sentence pitch framing the trade as fair by ranking.

    Args:
        send_rank: ROS SGP rank of the player we're sending.
        receive_rank: ROS SGP rank of the player we'd receive.
        send_positions: Eligible positions of the player we're sending.
        receive_positions: Eligible positions of the player we'd receive.
    """
    rank_part = f"You're getting the #{send_rank} overall player for your #{receive_rank}"

    # Add positional justification when players don't share a position
    shared = set(send_positions) & set(receive_positions)
    if shared:
        return f"{rank_part} — straight swap."
    else:
        return f"{rank_part}. Fills a positional need for both of us."
