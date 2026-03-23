"""Generate human-readable trade pitches for opponents."""

CAT_NAMES = {
    "R": "runs", "HR": "home runs", "RBI": "RBI", "SB": "steals",
    "AVG": "batting average", "W": "wins", "K": "strikeouts",
    "SV": "saves", "ERA": "ERA", "WHIP": "WHIP",
}

ORDINALS = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th", 5: "5th",
            6: "6th", 7: "7th", 8: "8th", 9: "9th", 10: "10th"}


def generate_pitch(
    opp_name: str,
    opp_cat_deltas: dict[str, int],
    opp_cat_ranks: dict[str, int],
) -> str:
    """Generate a 1-2 sentence pitch explaining why this trade helps the opponent.

    Args:
        opp_name: Opponent team name.
        opp_cat_deltas: Per-category roto point changes for opponent (e.g. {"SB": 2, "HR": -1}).
        opp_cat_ranks: Opponent's current rank per category (1=best, 10=worst in a 10-team league).
    """
    gains = [(cat, d) for cat, d in opp_cat_deltas.items() if d > 0]
    losses = [(cat, d) for cat, d in opp_cat_deltas.items() if d < 0]

    if not gains:
        return "This trade is roughly neutral for you — no category impact."

    # Sort gains by opponent's weakness (higher rank number = weaker = more compelling)
    gains.sort(key=lambda x: opp_cat_ranks.get(x[0], 5), reverse=True)
    # Sort losses by opponent's strength (lower rank number = stronger = easier to absorb)
    losses.sort(key=lambda x: opp_cat_ranks.get(x[0], 5))

    # Build the "you gain" part — highlight their 1-2 weakest categories that improve
    top_gains = gains[:2]
    gain_parts = []
    for cat, delta in top_gains:
        rank = opp_cat_ranks.get(cat, 5)
        rank_str = ORDINALS.get(rank, f"{rank}th")
        cat_name = CAT_NAMES.get(cat, cat)
        gain_parts.append(f"you're {rank_str} in {cat_name}")

    gain_sentence = f"You need help where it counts — {' and '.join(gain_parts)}. This trade boosts you there."

    # Build the "you can afford it" part — their strongest category that takes a hit
    if losses:
        best_loss = losses[0]
        loss_cat = best_loss[0]
        loss_rank = opp_cat_ranks.get(loss_cat, 5)
        loss_rank_str = ORDINALS.get(loss_rank, f"{loss_rank}th")
        loss_name = CAT_NAMES.get(loss_cat, loss_cat)
        loss_sentence = f" You're {loss_rank_str} in {loss_name}, so you can afford the hit."
    else:
        loss_sentence = ""

    return gain_sentence + loss_sentence
