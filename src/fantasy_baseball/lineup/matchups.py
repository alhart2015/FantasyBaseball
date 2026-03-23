"""Pitcher matchup quality adjustments based on opponent team batting stats."""

import pandas as pd


def normalize_team_batting_stats(raw_stats: list[dict]) -> dict[str, dict]:
    """Convert raw MLB API team batting data to {abbrev: {ops, k_pct}}.

    Args:
        raw_stats: List of dicts with keys: abbreviation, ops (str),
                   strikeouts (int), plate_appearances (int).
    Returns:
        Dict keyed by team abbreviation with float ops and k_pct values.
    """
    result = {}
    for team in raw_stats:
        abbrev = team["abbreviation"]
        pa = team["plate_appearances"]
        k_pct = team["strikeouts"] / pa if pa > 0 else 0.0
        result[abbrev] = {
            "ops": float(team["ops"]),
            "k_pct": k_pct,
        }
    return result


DEFAULT_DAMPENING = 0.5


def calculate_matchup_factors(
    team_stats: dict[str, dict],
    dampening: float = DEFAULT_DAMPENING,
) -> dict[str, dict]:
    """Compute matchup adjustment factors for each team relative to league average.

    For each team, produces:
      - era_whip_factor: multiplier for pitcher ERA/WHIP (>1 = harder matchup)
      - k_factor: multiplier for pitcher K (>1 = more Ks expected)

    Deviations from league average are dampened by the dampening parameter
    (0.5 = half the raw deviation applied).
    """
    if not team_stats:
        return {}

    ops_values = [t["ops"] for t in team_stats.values()]
    k_values = [t["k_pct"] for t in team_stats.values()]
    avg_ops = sum(ops_values) / len(ops_values)
    avg_k = sum(k_values) / len(k_values)

    factors = {}
    for abbrev, stats in team_stats.items():
        if avg_ops > 0:
            ops_dev = (stats["ops"] - avg_ops) / avg_ops
            era_whip_factor = 1.0 + dampening * ops_dev
        else:
            era_whip_factor = 1.0

        if avg_k > 0:
            k_dev = (stats["k_pct"] - avg_k) / avg_k
            k_factor = 1.0 + dampening * k_dev
        else:
            k_factor = 1.0

        factors[abbrev] = {
            "era_whip_factor": era_whip_factor,
            "k_factor": k_factor,
        }
    return factors
