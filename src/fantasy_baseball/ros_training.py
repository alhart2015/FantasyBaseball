"""Training steps shared by the hitter and pitcher nets (``scripts/train_hitter_ros.py``,
``pitcher_ros/model.py``): which players are held out for early stopping, the input
scaling fit on the rest, and the targets' standardization.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.features import Standardizer


def holdout_players(
    table: pd.DataFrame, train_rows: pd.Series, val_frac: float, seed: int
) -> np.ndarray:
    """A random ``val_frac`` of the players in ``train_rows`` (whole players, so none
    is on both sides of an early-stopping split)."""
    rng = np.random.default_rng(seed)
    players = table.loc[train_rows, "player_id"].unique()
    held: np.ndarray = rng.choice(players, size=int(len(players) * val_frac), replace=False)
    return held


def holdout_and_scaler(
    table: pd.DataFrame,
    x_all: pd.DataFrame,
    train_rows: pd.Series,
    val_frac: float,
    seed: int,
) -> tuple[np.ndarray, Standardizer]:
    """:func:`holdout_players` as a mask over the ``train_rows`` rows, and the input
    scaler fit on the training rows of the other players."""
    val_players = holdout_players(table, train_rows, val_frac, seed)
    val_mask = table.loc[train_rows, "player_id"].isin(val_players).to_numpy()
    scaler = Standardizer().fit(x_all[train_rows & ~table["player_id"].isin(val_players)])
    return val_mask, scaler


def weighted_moments(
    y: pd.DataFrame, w: pd.DataFrame, rows: np.ndarray
) -> tuple[dict[str, float], dict[str, float]]:
    """Each target column's loss-weighted mean and SD over ``rows`` (NaN targets count
    as 0; they carry weight 0)."""
    mu = {c: np.average(y[c][rows].fillna(0), weights=w[c][rows]) for c in y.columns}
    sd = {
        c: np.sqrt(np.average((y[c][rows].fillna(0) - mu[c]) ** 2, weights=w[c][rows]))
        for c in y.columns
    }
    return mu, sd
