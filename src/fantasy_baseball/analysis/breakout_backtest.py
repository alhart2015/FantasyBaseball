from __future__ import annotations

_RECENCY = {0: 5.0, 1: 4.0, 2: 3.0}  # most-recent .. 3rd
_REGRESS_W = 4.0  # league-mean pseudo-weight


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
