"""The first pitcher net (#465 Stage 4): a plain MLP from ``features.input_frame`` to the
five rest-of-season rates (``features.answers``), on the hitter net's machinery
(``hitter_ros.net``: the MLP, early stopping, loss weights).

Plain on purpose: weighted squared error on standardized rates, absolute rates (not
league-relative), no sequence encoder. The hitter net's other options are switched off
in :func:`plain_config`, and :func:`fit_season` refuses a config that turns one on (it
would be ignored here); each gets tried for pitchers on its own, against this.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.net import NetConfig, predict, train
from fantasy_baseball.pitcher_ros.stats import TARGETS
from fantasy_baseball.ros_training import holdout_and_scaler, weighted_moments

ID_COLS = ["player_id", "season", "week", "as_of"]

# The NetConfig settings the pitcher net uses; every other one must stay as
# plain_config() leaves it.
APPLIED = (
    "hidden",
    "dropout",
    "lr",
    "weight_decay",
    "batch_size",
    "max_epochs",
    "patience",
    "val_frac",
    "seed",
    "weighting",
    "micro_batch",
    "amp",
)


def plain_config(**overrides: Any) -> NetConfig:
    """The hitter net's config with its hitter-only options off: no sequence, plain
    squared error on absolute rates, one output per stat, no hitter feature files. The
    ``APPLIED`` settings (widths, dropout, learning rate, early stopping, pre/mid-season
    loss balance) keep the hitter defaults unless overridden."""
    plain = NetConfig(
        seq="none",
        relative_target="none",
        count_loss="none",
        avg_loss="mse",
        avg_pieces="none",
        sb_pieces="none",
        sb_inputs="full",
        horizons=False,
        probes="none",
        milb="none",
        parks="none",
        pedigree="none",
        steal_inputs=False,
    )
    return replace(plain, **overrides)


def applied_settings(config: NetConfig) -> dict[str, Any]:
    """The ``APPLIED`` settings of ``config``: what a run's config.json records."""
    return {k: v for k, v in asdict(config).items() if k in APPLIED}


def check_plain(config: NetConfig) -> None:
    """Refuse a config with a setting outside ``APPLIED`` changed from
    :func:`plain_config`'s: the pitcher net would ignore it, and a run would be
    credited to a setting it never had."""
    plain = asdict(plain_config())
    changed = sorted(k for k, v in asdict(config).items() if k not in APPLIED and plain[k] != v)
    if changed:
        raise ValueError(
            f"the pitcher net doesn't use {', '.join(changed)}; leave them as plain_config sets them"
        )


def check_week_zero(table: pd.DataFrame) -> None:
    """The pre/mid-season loss balance (``NetConfig.weighting`` "pre_mid") finds week 0
    as the rows with the whole season left; refuse a table where those differ."""
    if not ((table["week"] == 0) == (table["frac_season_left"] >= 1)).all():
        raise ValueError("week 0 is not exactly the rows with frac_season_left 1")


def fit_season(
    table: pd.DataFrame,
    x_all: pd.DataFrame,
    y_all: pd.DataFrame,
    w_all: pd.DataFrame,
    test_season: int,
    config: NetConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train on complete seasons before ``test_season``; predict every row of it.

    A random ``config.val_frac`` of the training pitchers is held out for early
    stopping. Targets are standardized by their loss-weighted mean and SD over the
    training rows; predictions are put back in rates (never below 0)."""
    check_plain(config)
    check_week_zero(table)
    train_rows = (
        table["season_complete"] & (table["season"] < test_season) & (w_all.sum(axis=1) > 0)
    )
    val_mask, scaler = holdout_and_scaler(table, x_all, train_rows, config.val_frac, config.seed)

    cols = list(TARGETS)
    y_train, w_train = y_all.loc[train_rows, cols], w_all.loc[train_rows, cols]
    mu, sd = weighted_moments(y_train, w_train, ~val_mask)
    y_std = np.column_stack([(y_train[c] - mu[c]) / sd[c] for c in cols])
    result = train(
        scaler.transform(x_all[train_rows]),
        y_std,
        w_train.to_numpy(dtype=np.float32),
        val_mask,
        config,
        season_time=table.loc[train_rows, "frac_season_left"].to_numpy(),
        target_stats=cols,
    )
    test_rows = table["season"] == test_season
    z = predict(result.model, scaler.transform(x_all[test_rows]))
    preds = pd.DataFrame(
        {c: np.clip(z[:, i] * sd[c] + mu[c], 0.0, None) for i, c in enumerate(cols)},
        index=table.index[test_rows],
    )
    preds = pd.concat([table.loc[test_rows, ID_COLS], preds], axis=1)
    info = {
        "test_season": test_season,
        "train_rows": int(train_rows.sum() - val_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "n_features": scaler.n_features,
        "best_epoch": result.best_epoch,
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
    }
    return preds, info


# Seeds a run trains and averages by default. Run 005-plain-avg (these seeds, before the
# minor-league inputs), 2022-2026: the average beat 5 of 5 single seeds on preseason W,
# ERA and WHIP, 4 of 5 on SV and 3 of 5 on K (single seeds 190.5-198.9 K MSE, the
# average 192.1), 4 of 5 on every mid-season stat, and was never really worse than one.
DEFAULT_SEEDS = (0, 1, 2, 3, 4)


def average_seeds(preds: list[pd.DataFrame]) -> pd.DataFrame:
    """The ``TARGETS`` rates averaged over seeds' predictions (each ``ID_COLS`` plus
    rates, as :func:`fit_season` returns them), row for row, sorted by ``ID_COLS``.
    Refuses unless every row appears exactly once per seed: seeds that predicted
    different rows would quietly average over fewer seeds there, and a repeated row
    would be merged into one."""
    if not preds:
        raise ValueError("no predictions to average")
    stacked = pd.concat(preds, ignore_index=True)
    if not (stacked.groupby(ID_COLS).size() == len(preds)).all():
        raise ValueError("the seeds predicted different rows, or one repeated a row")
    return stacked.groupby(ID_COLS, as_index=False)[list(TARGETS)].mean()
