"""The first pitcher net (#465 Stage 4): a plain MLP from ``features.input_frame`` to the
five rest-of-season rates (``features.target_frame``), on the hitter net's machinery
(``hitter_ros.net``: the MLP, early stopping, loss weights).

Plain on purpose: weighted squared error on standardized rates, absolute rates (not
league-relative), no sequence encoder. The hitter net's other options are switched off
in :func:`plain_config`; each one gets tried for pitchers on its own, against this.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.features import Standardizer
from fantasy_baseball.hitter_ros.net import NetConfig, predict, train
from fantasy_baseball.pitcher_ros.stats import TARGETS

ID_COLS = ["player_id", "season", "week", "as_of"]

def plain_config(**overrides: Any) -> NetConfig:
    """The hitter net's config with its hitter-only options off: no sequence, plain
    squared error on absolute rates, one output per stat, no hitter feature files. The
    rest (widths, dropout, learning rate, early stopping, pre/mid-season loss balance)
    keep the hitter defaults unless overridden."""
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
    train_rows = (
        table["season_complete"] & (table["season"] < test_season) & (w_all.sum(axis=1) > 0)
    )
    rng = np.random.default_rng(config.seed)
    players = table.loc[train_rows, "player_id"].unique()
    val_players = rng.choice(players, size=int(len(players) * config.val_frac), replace=False)
    val_mask = table.loc[train_rows, "player_id"].isin(val_players).to_numpy()
    scaler = Standardizer().fit(x_all[train_rows & ~table["player_id"].isin(val_players)])

    cols = list(TARGETS)
    y_train, w_train = y_all.loc[train_rows, cols], w_all.loc[train_rows, cols]
    fit = ~val_mask
    mu = {c: np.average(y_train[c][fit].fillna(0), weights=w_train[c][fit]) for c in cols}
    sd = {
        c: float(
            np.sqrt(np.average((y_train[c][fit].fillna(0) - mu[c]) ** 2, weights=w_train[c][fit]))
        )
        for c in cols
    }
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
