from difflib import SequenceMatcher
from unicodedata import normalize

from fantasy_baseball.utils.name_utils import normalize_name


def split_team_and_player(
    raw_input: str,
    team_names: list[str],
    threshold: float = 0.6,
) -> tuple[str | None, str]:
    """Try to split input into a team-name prefix and a player-name remainder.

    Tries every possible split point (1 word, 2 words, ... N-1 words as the
    team candidate) and picks the best fuzzy match against *team_names*.

    Returns (matched_team_name, remaining_player_query).
    If no team prefix scores above *threshold*, returns (None, original_input).
    """
    words = raw_input.strip().split()
    if len(words) < 2:
        return None, raw_input

    # Normalize team names for comparison
    normed_teams = {_norm(t): t for t in team_names}

    best_score = 0.0
    best_team: str | None = None
    best_remainder = raw_input

    # Try each split: first 1 word as team, first 2 words, etc.
    for i in range(1, len(words)):
        candidate = " ".join(words[:i])
        remainder = " ".join(words[i:])
        candidate_norm = _norm(candidate)

        for normed, original in normed_teams.items():
            # Prefix match: check if candidate is a prefix of the team name
            if normed.startswith(candidate_norm):
                score = len(candidate_norm) / len(normed)
                # Boost exact prefix matches
                score = min(score + 0.3, 1.0)
            else:
                score = SequenceMatcher(None, candidate_norm, normed).ratio()

            if score > best_score:
                best_score = score
                best_team = original
                best_remainder = remainder

    if best_score >= threshold:
        return best_team, best_remainder
    return None, raw_input


def _norm(s: str) -> str:
    """Lowercase, strip punctuation/accents for comparison."""
    s = normalize("NFKD", s.lower())
    return "".join(c for c in s if c.isalnum() or c == " ").strip()


def find_player(
    query: str,
    player_names: list[str],
    threshold: float = 0.4,
    return_top_n: int | None = None,
) -> str | list[str] | None:
    """Find the best matching player name using fuzzy search."""
    query_norm = normalize_name(query)
    scored: list[tuple[float, str]] = []

    for name in player_names:
        name_norm = normalize_name(name)
        full_score = SequenceMatcher(None, query_norm, name_norm).ratio()
        last_name = name_norm.split()[-1] if " " in name_norm else name_norm
        last_score = SequenceMatcher(None, query_norm, last_name).ratio()
        substring_bonus = 0.3 if query_norm in name_norm else 0.0
        best = max(full_score, last_score) + substring_bonus
        scored.append((best, name))

    scored.sort(key=lambda x: x[0], reverse=True)

    if return_top_n is not None:
        return [name for score, name in scored[:return_top_n] if score >= threshold]

    if scored and scored[0][0] >= threshold:
        return scored[0][1]
    return None
