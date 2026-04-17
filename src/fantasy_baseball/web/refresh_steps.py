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


def compute_lineup_moves(
    optimal_hitters: dict[str, str],
    roster_players: list[Player],
) -> list[dict]:
    """Compare optimizer output to current slots; emit START moves.

    Only emits a move when the player is crossing the bench/active
    boundary. Bench-like slots: BN, IL, DL. Slot keys may have suffixes
    like ``OF_1`` — only the prefix before ``_`` matters for comparison.
    """
    bench_slots = {"BN", "IL", "DL"}
    moves: list[dict] = []
    for slot, player_name in optimal_hitters.items():
        for player in roster_players:
            if player.name != player_name:
                continue
            current_slot = player.selected_position or "BN"
            base_slot = slot.split("_")[0]
            if current_slot != base_slot and (
                current_slot in bench_slots or base_slot in bench_slots
            ):
                if player.rest_of_season is not None:
                    sgp = (
                        player.rest_of_season.sgp
                        if player.rest_of_season.sgp is not None
                        else player.rest_of_season.compute_sgp()
                    )
                    reason = f"SGP: {sgp:.1f}"
                else:
                    reason = "Optimal slot"
                moves.append({
                    "action": "START",
                    "player": player_name,
                    "slot": base_slot,
                    "reason": reason,
                })
            break
    return moves


def build_positions_map(
    roster_players: list[Player],
    opp_rosters: dict[str, list[Player]],
    fa_players: list[Player],
) -> dict[str, list[str]]:
    """Build a normalized-name → positions-list map from three sources.

    Iteration order is roster → opponents → FAs, so a player appearing
    in multiple sources gets the FA positions if present, then opponent,
    then user roster. FAs with empty positions are skipped (Yahoo
    sometimes returns no position data for them).
    """
    out: dict[str, list[str]] = {}
    for p in roster_players:
        out[normalize_name(p.name)] = list(p.positions)
    for opp_roster in opp_rosters.values():
        for p in opp_roster:
            out[normalize_name(p.name)] = list(p.positions)
    for fa in fa_players:
        if fa.positions:
            out[normalize_name(fa.name)] = list(fa.positions)
    return out
