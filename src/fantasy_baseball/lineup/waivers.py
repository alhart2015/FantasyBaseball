from typing import Callable

import pandas as pd

from fantasy_baseball.lineup.weighted_sgp import calculate_weighted_sgp
from fantasy_baseball.lineup.yahoo_roster import fetch_free_agents
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.sgp.player_value import (
    calculate_counting_sgp,
    calculate_hitting_rate_sgp,
    calculate_pitching_rate_sgp,
    DEFAULT_TEAM_AB,
    DEFAULT_TEAM_IP,
    REPLACEMENT_AVG,
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
)
from fantasy_baseball.utils.name_utils import normalize_name
from fantasy_baseball.utils.positions import can_cover_slots, is_pitcher


def detect_open_slots(
    yahoo_roster: list[dict],
    roster_slots: dict[str, int],
) -> tuple[int, int, int]:
    """Count empty active roster slots by type from Yahoo selected_position.

    Yahoo returns position names like "Util" (not "UTIL"), "SP"/"RP" (not "P"),
    so values are normalized to lowercase for matching.

    Returns:
        Tuple of (open_hitter_slots, open_pitcher_slots, open_bench_slots).
    """
    il_positions = {"il", "il+", "dl", "dl+"}
    bench_positions = {"bn"}

    filled_hitter = filled_pitcher = filled_bench = 0
    for p in yahoo_roster:
        slot = (p.get("selected_position") or "").lower()
        if slot in il_positions:
            pass
        elif slot in bench_positions:
            filled_bench += 1
        elif is_pitcher([slot.upper()]) or slot in ("sp", "rp", "p"):
            filled_pitcher += 1
        elif slot:
            filled_hitter += 1

    total_hitter_slots = sum(
        v for k, v in roster_slots.items()
        if k.lower() not in {"p", "bn", "il", "il+", "dl", "dl+"}
    )
    return (
        max(0, total_hitter_slots - filled_hitter),
        max(0, roster_slots.get("P", 0) - filled_pitcher),
        max(0, roster_slots.get("BN", 0) - filled_bench),
    )


def fetch_and_match_free_agents(
    league,
    hitters_proj: pd.DataFrame,
    pitchers_proj: pd.DataFrame,
    fa_per_position: int = 100,
    on_position_loaded: Callable[[str, int], None] | None = None,
) -> tuple[list[pd.Series], int]:
    """Fetch available players from Yahoo, match to projections.

    Fetches FA + waiver players across 8 positions, deduplicates by
    normalized name, and matches each to projections using position-aware
    search order (pitcher positions check pitchers_proj first).

    Projection DataFrames must have a ``_name_norm`` column precomputed
    via ``df["_name_norm"] = df["name"].apply(normalize_name)``.

    Args:
        league: Yahoo league object.
        hitters_proj: Blended hitter projections with _name_norm column.
        pitchers_proj: Blended pitcher projections with _name_norm column.
        fa_per_position: Number of players to fetch per position.
        on_position_loaded: Optional callback(position, count) for progress.

    Returns:
        Tuple of (matched_fa_players as list[pd.Series], total_fetched_count).
    """
    fa_players: list[pd.Series] = []
    fa_fetched = 0
    seen_names: set[str] = set()

    for pos in ["C", "1B", "2B", "3B", "SS", "OF", "SP", "RP"]:
        fas = fetch_free_agents(league, pos, count=fa_per_position)
        fa_fetched += len(fas)
        if on_position_loaded:
            on_position_loaded(pos, len(fas))

        if pos in ("SP", "RP"):
            search_order = [pitchers_proj, hitters_proj]
        else:
            search_order = [hitters_proj, pitchers_proj]

        for fa in fas:
            fa_name_norm = normalize_name(fa["name"])
            if fa_name_norm in seen_names:
                continue
            seen_names.add(fa_name_norm)

            proj_row = None
            for df in search_order:
                if df.empty:
                    continue
                matches = df[df["_name_norm"] == fa_name_norm]
                if not matches.empty:
                    proj_row = matches.iloc[0].copy()
                    break
            if proj_row is not None:
                proj_row["positions"] = fa["positions"]
                fa_players.append(proj_row)

    return fa_players, fa_fetched


def evaluate_pickup(
    add_player: pd.Series,
    drop_player: pd.Series,
    leverage: dict[str, float],
) -> dict:
    """Evaluate the SGP gain of adding one player and dropping another.

    Returns:
        Dict with add, drop, sgp_gain, and per-category breakdown.
    """
    add_wsgp = calculate_weighted_sgp(add_player, leverage)
    drop_wsgp = calculate_weighted_sgp(drop_player, leverage)

    denoms = get_sgp_denominators()
    categories = {}
    for stat, col in _get_stat_cols(add_player):
        add_val = _category_sgp(add_player, stat, col, denoms)
        drop_val = _category_sgp(drop_player, stat, col, denoms)
        weight = leverage.get(stat, 0)
        categories[stat] = (add_val - drop_val) * weight

    return {
        "add": add_player["name"],
        "drop": drop_player["name"],
        "sgp_gain": add_wsgp - drop_wsgp,
        "categories": categories,
    }


def scan_waivers(
    roster: list[pd.Series],
    free_agents: list[pd.Series],
    leverage: dict[str, float],
    max_results: int = 5,
    open_hitter_slots: int = 0,
    open_pitcher_slots: int = 0,
    open_bench_slots: int = 0,
    roster_slots: dict[str, int] | None = None,
) -> list[dict]:
    """Scan free agents and rank add/drop recommendations.

    For each free agent, finds the weakest roster player they could replace
    (same position type: hitter vs pitcher) and evaluates the swap.
    When open slots exist, also recommends pure adds (no drop required)
    matching the slot type (hitter slots filled by hitters, pitcher slots
    by pitchers, bench slots by either).

    When roster_slots is provided, hitter swaps are checked for position
    feasibility — a swap is skipped if the post-swap roster can't fill
    all required position slots.

    Returns only positive-gain recommendations, sorted best-first.

    Args:
        roster: List of player stat Series (must have 'positions' and 'player_type').
        free_agents: List of free agent stat Series.
        leverage: Category leverage weights.
        max_results: Maximum number of recommendations to return.
        open_hitter_slots: Empty hitter-only active slots.
        open_pitcher_slots: Empty pitcher-only active slots.
        open_bench_slots: Empty bench slots (either type).
        roster_slots: Config roster slots dict for position feasibility checks.

    Returns:
        List of evaluate_pickup result dicts, sorted by sgp_gain descending.
    """
    total_open = open_hitter_slots + open_pitcher_slots + open_bench_slots

    if not free_agents:
        return []
    if not roster and total_open <= 0:
        return []

    # Pre-compute wSGP for all roster players
    roster_scores = []
    for p in roster:
        wsgp = calculate_weighted_sgp(p, leverage)
        roster_scores.append({"player": p, "wsgp": wsgp})

    recommendations = []
    recommended_adds: set[str] = set()
    recommended_swaps: set[tuple[str, str]] = set()

    # Pure adds for empty slots — type-aware ranking
    if total_open > 0:
        fa_hitters = []
        fa_pitchers = []
        for fa in free_agents:
            wsgp = calculate_weighted_sgp(fa, leverage)
            if wsgp <= 0:
                continue
            if fa.get("player_type") == "pitcher":
                fa_pitchers.append((fa, wsgp))
            else:
                fa_hitters.append((fa, wsgp))
        fa_hitters.sort(key=lambda x: x[1], reverse=True)
        fa_pitchers.sort(key=lambda x: x[1], reverse=True)

        def _add_pure(pool, count, label):
            added = 0
            for fa, wsgp in pool:
                if added >= count or fa["name"] in recommended_adds:
                    continue
                recommendations.append({
                    "add": fa["name"],
                    "drop": f"(empty {label} slot)",
                    "sgp_gain": wsgp,
                    "categories": {},
                })
                recommended_adds.add(fa["name"])
                added += 1

        _add_pure(fa_hitters, open_hitter_slots, "hitter")
        _add_pure(fa_pitchers, open_pitcher_slots, "pitcher")
        # Bench slots: pick best remaining from either type
        remaining = [(fa, w) for fa, w in fa_hitters + fa_pitchers
                     if fa["name"] not in recommended_adds]
        remaining.sort(key=lambda x: x[1], reverse=True)
        _add_pure(remaining, open_bench_slots, "bench")

    # Drop/add swaps for remaining free agents
    for fa in free_agents:
        if fa["name"] in recommended_adds:
            continue
        fa_type = fa.get("player_type", "hitter")

        # Find worst roster player of same type, sorted weakest-first
        same_type = [rs for rs in roster_scores if rs["player"].get("player_type") == fa_type]
        if not same_type:
            continue
        same_type_sorted = sorted(same_type, key=lambda x: x["wsgp"])

        for candidate in same_type_sorted:
            pair_key = (fa["name"], candidate["player"]["name"])
            if pair_key in recommended_swaps:
                continue

            # Position feasibility check for hitter swaps
            if roster_slots and fa_type == "hitter":
                post_swap_positions = [
                    list(rs["player"].get("positions", []))
                    for rs in roster_scores
                    if rs["player"]["name"] != candidate["player"]["name"]
                    and rs["player"].get("player_type") == "hitter"
                ]
                post_swap_positions.append(list(fa.get("positions", [])))
                if not can_cover_slots(post_swap_positions, roster_slots):
                    continue  # this drop leaves a position hole — try next

            recommended_swaps.add(pair_key)
            result = evaluate_pickup(fa, candidate["player"], leverage)
            if result["sgp_gain"] > 0:
                recommendations.append(result)
            break  # found a valid drop candidate for this FA

    recommendations.sort(key=lambda x: x["sgp_gain"], reverse=True)
    return recommendations[:max_results]


def _get_stat_cols(player: pd.Series) -> list[tuple[str, str]]:
    """Get relevant stat/column pairs for a player's type."""
    if player.get("player_type") == "hitter":
        return [("R", "r"), ("HR", "hr"), ("RBI", "rbi"), ("SB", "sb"), ("AVG", "avg")]
    elif player.get("player_type") == "pitcher":
        return [("W", "w"), ("K", "k"), ("SV", "sv"), ("ERA", "era"), ("WHIP", "whip")]
    return []


def _category_sgp(player: pd.Series, stat: str, col: str, denoms: dict) -> float:
    """Calculate raw SGP for a single category."""
    if stat in ("AVG",):
        return calculate_hitting_rate_sgp(
            player_avg=player.get("avg", 0),
            player_ab=int(player.get("ab", 0)),
            replacement_avg=REPLACEMENT_AVG,
            sgp_denominator=denoms["AVG"],
            team_ab=DEFAULT_TEAM_AB,
        )
    elif stat in ("ERA",):
        ip = player.get("ip", 0)
        if ip > 0:
            return calculate_pitching_rate_sgp(
                player_rate=player.get("era", 0), player_ip=ip,
                replacement_rate=REPLACEMENT_ERA,
                sgp_denominator=denoms["ERA"],
                team_ip=DEFAULT_TEAM_IP, innings_divisor=9,
            )
        return 0.0
    elif stat in ("WHIP",):
        ip = player.get("ip", 0)
        if ip > 0:
            return calculate_pitching_rate_sgp(
                player_rate=player.get("whip", 0), player_ip=ip,
                replacement_rate=REPLACEMENT_WHIP,
                sgp_denominator=denoms["WHIP"],
                team_ip=DEFAULT_TEAM_IP, innings_divisor=1,
            )
        return 0.0
    else:
        return calculate_counting_sgp(player.get(col, 0), denoms[stat])
