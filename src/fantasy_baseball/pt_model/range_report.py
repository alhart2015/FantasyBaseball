"""The held-out report for ``ranges.backtest`` (#505), shared by the hitter and pitcher
backtest scripts: P10/P90 coverage, the volume middle check and the quantile loss of
the player-specific ranges against the one-size ones, with how sure (0-100)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from fantasy_baseball.pt_model.ranges import Spec, pinball

BOOTSTRAPS = 4000
MODELS = ("player-specific", "one-size")


def sure_better(
    loss_a: pd.Series, loss_b: pd.Series, players: pd.Series, rng: np.random.Generator
) -> float:
    """How sure (0-100) a's loss is lower than b's, resampling players."""
    gain = (loss_b - loss_a).groupby(players.to_numpy()).sum().to_numpy()
    draws = rng.integers(0, len(gain), (BOOTSTRAPS, len(gain)))
    return float((gain[draws].sum(axis=1) > 0).mean() * 100)


def middle(scored: pd.DataFrame, volume: str, bins: Sequence[float]) -> str:
    """Player-specific volume: share of real volume under the predicted median, by
    predicted-median bin."""
    p = scored[(scored["stat"] == volume) & (scored["model"] == MODELS[0])]
    t = p.groupby(pd.cut(p["q50"], list(bins)), observed=True).apply(
        lambda g: pd.Series(
            {
                "n": len(g),
                "predicted": g["q50"].median(),
                "real": g["actual"].median(),
                "below %": (g["actual"] < g["q50"]).mean() * 100,
            }
        ),
        include_groups=False,
    )
    return str(t.round(1).to_string())


def summarize(
    scored: pd.DataFrame,
    spec: Spec,
    rng: np.random.Generator,
    *,
    regular: float,
    median_bins: Sequence[float],
    min_scored: Mapping[str, float],
    by_season: Sequence[str],
) -> list[str]:
    """The report's lines. ``scored``: ``ranges.backtest`` frames with a ``model``
    column (``MODELS``). ``regular``: the ranked volume of an expected regular.
    ``min_scored``: a rate stat is scored where its trials reach this. ``by_season``:
    the stats whose coverage is shown by season."""
    keep = pd.Series(True, index=scored.index)
    for stat, least in min_scored.items():
        keep &= ~((scored["stat"] == stat) & (scored["trials"] < least))
    s = scored[keep].copy()
    s["below"], s["above"] = s["actual"] < s["q10"], s["actual"] > s["q90"]
    s["loss"] = pinball(s["actual"].to_numpy(), s["q10"].to_numpy(), 0.1) + pinball(
        s["actual"].to_numpy(), s["q90"].to_numpy(), 0.9
    )
    order = [spec.volume, *spec.stat_names]
    v = spec.volume.upper()

    def cover(df: pd.DataFrame) -> str:
        t = df.groupby(["stat", "model"])[["below", "above"]].mean().unstack("model").reindex(order)
        return str((t * 100).round(1).to_string())

    md = [
        "Share of seasons below P10 / above P90, % (target 10 / 10):",
        cover(s),
        "",
        f"Expected regulars (rank {v} {regular:g}+):",
        cover(s[s["ranked"] >= regular]),
        "",
        f"{', '.join(c.upper() for c in by_season)} by season:",
        (
            s[s["stat"].isin(list(by_season))]
            .groupby(["season", "model"])[["below", "above"]]
            .mean()
            .unstack("model")
            * 100
        )
        .round(1)
        .to_string(),
        "",
        f"{v} middle, player-specific: share of real {v} below the predicted median, % (50 is",
        "right), by predicted median:",
        middle(scored, spec.volume, median_bins),
        "",
        "Quantile loss (P10 + P90), player-specific vs one-size:",
    ]
    wide = s.pivot_table(
        index=["player_id", "season", "stat"], columns="model", values="loss"
    ).dropna()
    for stat in order:
        w = wide.xs(stat, level="stat")
        sure = sure_better(
            w[MODELS[0]], w[MODELS[1]], w.index.get_level_values("player_id").to_series(), rng
        )
        change = w[MODELS[0]].mean() / w[MODELS[1]].mean() - 1
        md.append(f"  {stat:4s} {change:+.1%}  (how sure better: {sure:.0f})")
    return md
