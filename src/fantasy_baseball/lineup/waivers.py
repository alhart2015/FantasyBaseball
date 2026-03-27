from typing import Callable

import pandas as pd

from fantasy_baseball.lineup.optimizer import optimize_hitter_lineup, optimize_pitcher_lineup
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


def _compute_team_wsgp(
    roster: list[pd.Series],
    leverage: dict[str, float],
    roster_slots: dict[str, int],
    denoms: dict[str, float] | None = None,
) -> dict:
    """Run both optimizers and return total wSGP of assigned starters.

    Returns dict with:
        total_wsgp: float — sum of wSGP for players actually assigned to a slot
        hitter_lineup: dict[str, str] — slot -> player name from Hungarian optimizer
        pitcher_starters: list[dict] — pitcher starters from ranking
        player_wsgp: dict[str, float] — name -> wSGP lookup for all roster players
    """
    if denoms is None:
        denoms = get_sgp_denominators()

    hitters = [p for p in roster if p.get("player_type") != "pitcher"]
    pitchers = [p for p in roster if p.get("player_type") == "pitcher"]

    # Pre-compute wSGP for all players
    player_wsgp = {}
    for p in roster:
        player_wsgp[p["name"]] = calculate_weighted_sgp(p, leverage, denoms=denoms)

    # Optimize hitters (Hungarian algorithm)
    hitter_lineup = optimize_hitter_lineup(hitters, leverage, roster_slots)

    # Optimize pitchers (simple ranking)
    p_slots = roster_slots.get("P", 9)
    pitcher_starters, _ = optimize_pitcher_lineup(pitchers, leverage, slots=p_slots)

    # Sum wSGP of assigned players only
    total = 0.0
    for name in hitter_lineup.values():
        total += player_wsgp.get(name, 0.0)
    for ps in pitcher_starters:
        total += player_wsgp.get(ps["name"], 0.0)

    return {
        "total_wsgp": total,
        "hitter_lineup": hitter_lineup,
        "pitcher_starters": pitcher_starters,
        "player_wsgp": player_wsgp,
    }


def _build_lineup_summary(
    hitter_lineup: dict[str, str],
    pitcher_starters: list[dict],
    player_wsgp: dict[str, float],
    all_player_names: list[str],
) -> list[dict]:
    """Build a lineup summary list for display.

    Returns list of {"name", "slot", "wsgp"} dicts.
    Hitter slots have _N suffixes stripped. Unassigned players get slot="BN".
    """
    summary = []
    assigned_names = set()

    # Hitters from optimizer
    for slot_key, name in hitter_lineup.items():
        display_slot = slot_key.split("_")[0]  # "OF_2" -> "OF"
        summary.append({
            "name": name,
            "slot": display_slot,
            "wsgp": round(player_wsgp.get(name, 0.0), 2),
        })
        assigned_names.add(name)

    # Pitcher starters
    for ps in pitcher_starters:
        name = ps["name"]
        summary.append({
            "name": name,
            "slot": "P",
            "wsgp": round(player_wsgp.get(name, 0.0), 2),
        })
        assigned_names.add(name)

    # Bench: everyone not assigned
    for name in all_player_names:
        if name not in assigned_names:
            summary.append({
                "name": name,
                "slot": "BN",
                "wsgp": round(player_wsgp.get(name, 0.0), 2),
            })

    return summary


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

    # Phase 2: Re-optimization swaps
    if not roster or not roster_slots:
        recommendations.sort(key=lambda x: x["sgp_gain"], reverse=True)
        return recommendations[:max_results]

    denoms = get_sgp_denominators()

    # Compute baseline optimal lineup
    baseline = _compute_team_wsgp(roster, leverage, roster_slots, denoms=denoms)
    baseline_wsgp = baseline["total_wsgp"]
    baseline_summary = _build_lineup_summary(
        baseline["hitter_lineup"], baseline["pitcher_starters"],
        baseline["player_wsgp"], [p["name"] for p in roster],
    )

    # Pre-compute wSGP for all FAs
    fa_wsgp = {}
    for fa in free_agents:
        if fa["name"] not in recommended_adds:
            fa_wsgp[fa["name"]] = calculate_weighted_sgp(fa, leverage, denoms=denoms)

    # Compute wSGP floor: 3rd-lowest wSGP among active-slot players
    active_wsgps = sorted([
        baseline["player_wsgp"].get(name, 0.0)
        for name in list(baseline["hitter_lineup"].values())
        + [ps["name"] for ps in baseline["pitcher_starters"]]
    ])
    wsgp_floor = active_wsgps[2] if len(active_wsgps) > 2 else 0.0

    p_slots = roster_slots.get("P", 9)

    for fa in free_agents:
        if fa["name"] in recommended_adds:
            continue
        if fa_wsgp.get(fa["name"], 0.0) < wsgp_floor:
            continue

        fa_type = fa.get("player_type", "hitter")
        best_for_fa = None

        for drop_player in roster:
            drop_name = drop_player["name"]
            drop_type = drop_player.get("player_type", "hitter")

            # Build hypothetical roster
            new_roster = [p for p in roster if p["name"] != drop_name] + [fa]
            new_hitters = [p for p in new_roster if p.get("player_type") != "pitcher"]
            new_pitchers = [p for p in new_roster if p.get("player_type") == "pitcher"]

            # Feasibility checks
            if drop_type == "hitter" or fa_type == "hitter":
                hitter_positions = [list(p.get("positions", [])) for p in new_hitters]
                if not can_cover_slots(hitter_positions, roster_slots):
                    continue
            if drop_type == "pitcher" or fa_type == "pitcher":
                if len(new_pitchers) < p_slots:
                    continue

            # Re-optimize
            new_result = _compute_team_wsgp(new_roster, leverage, roster_slots, denoms=denoms)
            gain = round(new_result["total_wsgp"] - baseline_wsgp, 2)

            if gain > 0 and (best_for_fa is None or gain > best_for_fa["sgp_gain"]):
                after_summary = _build_lineup_summary(
                    new_result["hitter_lineup"], new_result["pitcher_starters"],
                    new_result["player_wsgp"], [p["name"] for p in new_roster],
                )

                # Annotate before lineup (mark dropped player)
                before_annotated = []
                for entry in baseline_summary:
                    e = dict(entry)
                    if e["name"] == drop_name:
                        e["is_dropped"] = True
                    before_annotated.append(e)

                # Annotate after lineup (mark added player and moved players)
                before_slots = {e["name"]: e["slot"] for e in baseline_summary}
                after_annotated = []
                for entry in after_summary:
                    e = dict(entry)
                    if e["name"] == fa["name"]:
                        e["is_added"] = True
                    elif e["name"] in before_slots and before_slots[e["name"]] != e["slot"]:
                        e["moved_from"] = before_slots[e["name"]]
                    after_annotated.append(e)

                # Get per-category deltas (from evaluate_pickup, discard its sgp_gain)
                cat_result = evaluate_pickup(fa, drop_player, leverage)

                best_for_fa = {
                    "add": fa["name"],
                    "add_positions": list(fa.get("positions", [])),
                    "drop": drop_name,
                    "drop_positions": list(drop_player.get("positions", [])),
                    "sgp_gain": gain,
                    "categories": cat_result["categories"],
                    "lineup_before": before_annotated,
                    "lineup_after": after_annotated,
                }

        if best_for_fa:
            recommendations.append(best_for_fa)

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
