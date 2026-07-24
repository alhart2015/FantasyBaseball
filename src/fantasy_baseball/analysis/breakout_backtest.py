from __future__ import annotations

_RECENCY = {0: 5.0, 1: 4.0, 2: 3.0}  # most-recent .. 3rd
_REGRESS_W = 4.0  # league-mean pseudo-weight

DEFAULT_RULER = {
    "hr": 100.0,
    "r": 60.0,
    "rbi": 60.0,
    "sb": 120.0,
    "avg": 1500.0,
    "k": 40.0,
    "w": 300.0,
    "sv": 250.0,
    "era": -200.0,
    "whip": -400.0,
}


def marcel_prior(
    history: list[tuple[int, dict[str, float]]], league_mean: dict[str, float], age: float | None
) -> dict[str, float]:
    history = sorted(history, key=lambda t: t[0], reverse=True)[:3]
    stats = set().union(*[set(d) for _, d in history]) if history else set(league_mean)
    prior = {}
    for s in stats:
        num = _REGRESS_W * league_mean.get(s, 0.0)
        den = _REGRESS_W
        for i, (_, line) in enumerate(history):
            wt = _RECENCY.get(i, 0.0)
            if line.get(s) is not None:
                num += wt * line[s]
                den += wt
        val = num / den if den > 0 else league_mean.get(s, 0.0)
        if age is not None:
            val *= 1.0 - 0.003 * (age - 27.0)  # mild peak-27 age curve
        prior[s] = val
    return prior


def rate_mae(pred_rates: dict[str, float], actual_rates: dict[str, float]) -> float:
    keys = set(pred_rates) & set(actual_rates)
    if not keys:
        return 0.0
    return sum(abs(pred_rates[k] - actual_rates[k]) for k in keys) / len(keys)


def sgp_on_ruler(rates: dict[str, float], weights: dict[str, float]) -> float:
    return sum(weights.get(s, 0.0) * v for s, v in rates.items())
