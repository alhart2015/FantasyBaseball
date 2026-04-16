"""Pure helpers extracted from run_full_refresh.

These pieces are refresh-specific orchestration glue — they don't
belong in a domain module like ``scoring`` or ``analysis.pace``
because they only exist to compose those domains' outputs into the
shape the cache files need.
"""
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import PITCHER_POSITIONS


def merge_matched_and_raw_roster(
    matched: list[Player],
    roster_raw: list[dict],
    preseason_lookup: dict[str, Player],
) -> list[Player]:
    """Combine projection-matched players with any unmatched raw entries.

    For every matched player, attaches ``player.preseason`` from the
    corresponding entry in ``preseason_lookup`` (keyed by normalized
    name) if one exists. Then appends a Player built from each raw
    roster entry that wasn't matched, inferring ``player_type`` from
    positions (any pitcher position → PITCHER, otherwise HITTER).

    Mutates each matched Player. Returns the combined list.
    """
    matched_names: set[str] = set()
    out: list[Player] = []
    for player in matched:
        norm = normalize_name(player.name)
        matched_names.add(norm)
        pre_entry = preseason_lookup.get(norm)
        if pre_entry and pre_entry.rest_of_season:
            player.preseason = pre_entry.rest_of_season
        out.append(player)

    for raw_player in roster_raw:
        if normalize_name(raw_player["name"]) not in matched_names:
            inferred_type = (
                PlayerType.PITCHER
                if set(raw_player.get("positions", [])) & PITCHER_POSITIONS
                else PlayerType.HITTER
            )
            out.append(Player.from_dict({**raw_player, "player_type": inferred_type}))

    return out
