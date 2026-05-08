"""Pure helpers extracted from run_full_refresh.

These pieces are refresh-specific orchestration glue — they don't
belong in a domain module like ``scoring`` or ``analysis.pace``
because they only exist to compose those domains' outputs into the
shape the cache files need.
"""

from fantasy_baseball.lineup.optimizer import HitterAssignment, PitcherStarter
from fantasy_baseball.models.player import Player, PlayerType
from fantasy_baseball.models.positions import BENCH_SLOTS
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
    optimal_hitters: list[HitterAssignment],
    optimal_pitchers: list[PitcherStarter],
    pitcher_bench: list[Player],
    roster_players: list[Player],
) -> dict:
    """Compare optimizer output to current slots; emit paired swap rows.

    Returns a structured payload:

        {
            "swaps": [
                {
                    "start": {"player", "from", "to", "roto_delta"},
                    "bench": {"player", "from", "to"},
                },
                ...
            ],
            "unpaired_starts":  [{"player", "from", "to", "roto_delta"}, ...],
            "unpaired_benches": [{"player", "from", "to"}, ...],
        }

    Pairing is two-pass greedy. Both passes consume starts in descending
    ``roto_delta`` order and benches in descending current SGP order, so the
    documented "highest-impact start pairs with highest-loss bench" property
    holds regardless of caller input order.

    1. Exact base-slot match (a START with target slot ``S`` pairs with a
       BENCH whose vacated slot was ``S``). Handles the common hitter case.
    2. Type-compatible by ΔRoto / SGP order — remaining starts pair with
       remaining benches only within the same player type
       (hitter↔hitter, pitcher↔pitcher).

    Anything left after both passes is returned as ``unpaired_starts`` /
    ``unpaired_benches`` (rare: caused by IL transitions or asymmetric
    roster changes).
    """
    # --- Build START moves (bench → active boundary crossings) ---
    starts: list[dict] = []
    for assignment in optimal_hitters:
        player = assignment.player
        current = player.selected_position or "BN"
        target = assignment.slot.value
        if current in BENCH_SLOTS and target not in BENCH_SLOTS:
            starts.append(
                {
                    "player": player.name,
                    "from": current,
                    "to": target,
                    "roto_delta": assignment.roto_delta,
                    "_player_type": PlayerType.HITTER,
                }
            )
    for starter in optimal_pitchers:
        player = starter.player
        current = player.selected_position or "BN"
        if current in BENCH_SLOTS:
            starts.append(
                {
                    "player": player.name,
                    "from": current,
                    "to": "P",
                    "roto_delta": starter.roto_delta,
                    "_player_type": PlayerType.PITCHER,
                }
            )

    # --- Build BENCH moves (active → bench boundary crossings) ---
    benches: list[dict] = []
    optimal_hitter_names = {a.name for a in optimal_hitters}
    for player in roster_players:
        if player.player_type != PlayerType.HITTER:
            continue
        current = player.selected_position or "BN"
        if current in BENCH_SLOTS:
            continue
        if player.name in optimal_hitter_names:
            continue
        benches.append(
            {
                "player": player.name,
                "from": current,
                "to": "BN",
                "_player_type": PlayerType.HITTER,
                "_sgp": _player_sgp(player),
            }
        )
    for player in pitcher_bench:
        current = player.selected_position or "BN"
        if current in BENCH_SLOTS:
            continue
        benches.append(
            {
                "player": player.name,
                "from": current,
                "to": "BN",
                "_player_type": PlayerType.PITCHER,
                "_sgp": _player_sgp(player),
            }
        )

    return _pair_swaps(starts, benches)


def _player_sgp(player: Player) -> float:
    """Best-effort SGP read from a Player; 0.0 if no projection attached.

    Used as the bench-side ordering key in pass 2 of swap pairing — higher
    SGP means a bigger contribution lost when benched, so it should pair
    with the highest-impact START.
    """
    if player.rest_of_season is None:
        return 0.0
    if player.rest_of_season.sgp is not None:
        return float(player.rest_of_season.sgp)
    return float(player.rest_of_season.compute_sgp())


def _pair_swaps(starts: list[dict], benches: list[dict]) -> dict:
    """Two-pass greedy pairing of START and BENCH moves.

    See :func:`compute_lineup_moves` for the algorithm description. Mutates
    nothing in the inputs; returns a fresh dict with stripped private keys.
    """
    # Pre-sort once so BOTH passes see starts in descending ΔRoto order and
    # benches in descending SGP order. Pass 1's "first match wins" then
    # produces the documented behavior (highest-impact start pairs with the
    # highest-loss compatible bench within the slot constraint) regardless
    # of caller input order.
    starts = sorted(starts, key=lambda s: -s["roto_delta"])
    benches = sorted(benches, key=lambda b: -b["_sgp"])
    swaps: list[dict] = []

    # Pass 1: exact base-slot match. A START with target slot S pairs with
    # a BENCH whose vacated slot was S. Same-type only (slot equality
    # already implies same type, but the explicit check guards future
    # additions).
    for start in list(starts):
        for bench in benches:
            if start["to"] == bench["from"] and start["_player_type"] == bench["_player_type"]:
                swaps.append(_to_swap(start, bench))
                starts.remove(start)
                benches.remove(bench)
                break

    # Pass 2: type-compatible by descending ΔRoto / descending SGP. Pair
    # within HITTER first, then PITCHER, so cross-type pairings can't sneak
    # in via index drift. Inputs are already sorted globally, so per-type
    # filtering preserves order.
    for ptype in (PlayerType.HITTER, PlayerType.PITCHER):
        type_starts = [s for s in starts if s["_player_type"] == ptype]
        type_benches = [b for b in benches if b["_player_type"] == ptype]
        for start, bench in zip(type_starts, type_benches, strict=False):
            swaps.append(_to_swap(start, bench))
            starts.remove(start)
            benches.remove(bench)

    return {
        "swaps": swaps,
        "unpaired_starts": [_strip_private(s) for s in starts],
        "unpaired_benches": [_strip_private(b) for b in benches],
    }


def _to_swap(start: dict, bench: dict) -> dict:
    return {
        "start": _strip_private(start),
        "bench": {k: v for k, v in bench.items() if not k.startswith("_") and k != "_sgp"},
    }


def _strip_private(move: dict) -> dict:
    return {k: v for k, v in move.items() if not k.startswith("_")}


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
