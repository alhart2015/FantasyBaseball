"""Monte Carlo season simulation with injuries and stat variance.

Shared by scripts/monte_carlo.py (post-draft analysis) and
scripts/summary.py (in-season weekly projections).
"""

import numpy as np

from fantasy_baseball.utils.constants import (
    CLOSER_SV_THRESHOLD,
    HITTING_COUNTING,
    INJURY_PROB,
    INJURY_SEVERITY,
    PITCHING_COUNTING,
    REPLACEMENT_HITTER,
    REPLACEMENT_RP,
    REPLACEMENT_SP,
    STAT_VARIANCE,
)


def simulate_season(
    team_rosters: dict,
    rng: np.random.Generator,
    h_slots: int = 13,
    p_slots: int = 9,
) -> tuple[dict, dict]:
    """Run one simulated season with injuries and stat variance.

    For each team, applies random injuries (probability-based) and
    performance variance to every player, then selects the best
    h_slots hitters and p_slots pitchers as the active roster.
    Bench players don't contribute stats.

    Args:
        team_rosters: Dict of team_key -> list of player dicts/Series.
            Each player must have ``player_type`` ("hitter"/"pitcher")
            and the stat columns defined in constants (r, hr, rbi, etc.).
        rng: NumPy random generator for reproducibility.
        h_slots: Number of active hitter slots.
        p_slots: Number of active pitcher slots.

    Returns:
        Tuple of (team_stats, injuries):
        - team_stats: {team_key: {R, HR, RBI, SB, AVG, W, K, SV, ERA, WHIP}}
        - injuries: {team_key: [(player_name, frac_missed), ...]}
    """
    team_stats = {}
    injuries = {}

    for team_key, players in team_rosters.items():
        hitters = [p for p in players if p.get("player_type") == "hitter"]
        pitchers = [p for p in players if p.get("player_type") == "pitcher"]
        team_injuries = []

        adj_hitters = _apply_variance(
            hitters, "hitter", rng, team_injuries,
        )
        adj_pitchers = _apply_variance(
            pitchers, "pitcher", rng, team_injuries,
        )

        # Select active roster (bench excluded)
        adj_hitters.sort(
            key=lambda h: h["r"] + h["hr"] + h["rbi"] + h["sb"],
            reverse=True,
        )
        adj_pitchers.sort(
            key=lambda p: (
                p.get("sv", 0) >= CLOSER_SV_THRESHOLD,
                p["w"] + p["k"] + p.get("sv", 0),
            ),
            reverse=True,
        )
        active_h = adj_hitters[:h_slots]
        active_p = adj_pitchers[:p_slots]

        # Aggregate team stats
        total_ab = sum(h["ab"] for h in active_h)
        total_h = sum(h["h"] for h in active_h)
        total_ip = sum(p["ip"] for p in active_p)
        total_er = sum(p["er"] for p in active_p)
        total_bb = sum(p["bb"] for p in active_p)
        total_ha = sum(p["h_allowed"] for p in active_p)

        team_stats[team_key] = {
            "R": sum(h["r"] for h in active_h),
            "HR": sum(h["hr"] for h in active_h),
            "RBI": sum(h["rbi"] for h in active_h),
            "SB": sum(h["sb"] for h in active_h),
            "AVG": total_h / total_ab if total_ab > 0 else 0,
            "W": sum(p["w"] for p in active_p),
            "K": sum(p["k"] for p in active_p),
            "SV": sum(p.get("sv", 0) for p in active_p),
            "ERA": total_er * 9 / total_ip if total_ip > 0 else 99,
            "WHIP": (total_bb + total_ha) / total_ip if total_ip > 0 else 99,
        }
        injuries[team_key] = team_injuries

    return team_stats, injuries


def _apply_variance(
    players: list,
    player_type: str,
    rng: np.random.Generator,
    injuries_out: list,
) -> list[dict]:
    """Apply injury and performance variance to a list of players.

    Mutates injuries_out by appending (name, frac_missed) for injured players.
    """
    is_hitter = player_type == "hitter"
    counting_cols = HITTING_COUNTING if is_hitter else PITCHING_COUNTING
    injury_prob = INJURY_PROB[player_type]
    injury_lo, injury_hi = INJURY_SEVERITY[player_type]
    variance = STAT_VARIANCE[player_type]

    adjusted = []
    for p in players:
        frac_missed = 0.0
        if rng.random() < injury_prob:
            frac_missed = rng.uniform(injury_lo, injury_hi)
            injuries_out.append((p.get("name", "?"), frac_missed))

        scale = 1.0 - frac_missed
        perf = max(0, 1.0 + rng.normal(0, variance))

        if is_hitter:
            repl = REPLACEMENT_HITTER
        else:
            is_closer = p.get("sv", 0) >= CLOSER_SV_THRESHOLD
            repl = REPLACEMENT_RP if is_closer else REPLACEMENT_SP

        row = {}
        inv_perf = max(0, 2.0 - perf)
        for col in counting_cols:
            base = float(p.get(col, 0) or 0)
            repl_contrib = repl.get(col, 0) * frac_missed
            if col in ("ab", "ip"):
                row[col] = base * scale + repl_contrib
            elif not is_hitter and col in ("er", "bb", "h_allowed"):
                row[col] = base * inv_perf * scale + repl_contrib
            else:
                row[col] = base * perf * scale + repl_contrib

        row["name"] = p.get("name", "?")
        row["player_type"] = player_type
        adjusted.append(row)

    return adjusted
