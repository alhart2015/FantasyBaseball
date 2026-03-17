from difflib import SequenceMatcher


def find_player(
    query: str,
    player_names: list[str],
    threshold: float = 0.4,
    return_top_n: int | None = None,
) -> str | list[str] | None:
    """Find the best matching player name using fuzzy search."""
    query_lower = query.lower().strip()
    scored: list[tuple[float, str]] = []

    for name in player_names:
        name_lower = name.lower()
        full_score = SequenceMatcher(None, query_lower, name_lower).ratio()
        last_name = name_lower.split()[-1] if " " in name_lower else name_lower
        last_score = SequenceMatcher(None, query_lower, last_name).ratio()
        substring_bonus = 0.3 if query_lower in name_lower else 0.0
        best = max(full_score, last_score) + substring_bonus
        scored.append((best, name))

    scored.sort(key=lambda x: x[0], reverse=True)

    if return_top_n is not None:
        return [name for score, name in scored[:return_top_n] if score >= threshold]

    if scored and scored[0][0] >= threshold:
        return scored[0][1]
    return None
