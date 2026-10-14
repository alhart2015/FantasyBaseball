"""The pitcher net (#465 Stage 4): an MLP from ``features.input_frame`` to the five
rest-of-season rates (``features.answers``), on the hitter net's machinery
(``hitter_ros.net``: the MLP, early stopping, loss weights).

:func:`plain_config` is the plain baseline: weighted squared error on standardized
rates, no sequence encoder, no feature files. The hitter net's other options are
switched off there, and
:func:`fit_season` refuses a config that turns one on (it would be ignored here); each
gets tried for pitchers on its own, against this. Applied so far (:func:`net_config`):
``relative_target``, learning each rate relative to the league
(``DEFAULT_RELATIVE_TARGET``); ``seq``, a transformer over recent appearances
(``DEFAULT_SEQ``); and ``parks``, park inputs (``DEFAULT_PARKS``).
"""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.net import PIECE_LOSS_WEIGHT, NetConfig, predict, train
from fantasy_baseball.hitter_ros.parks import PARK_FEATURES
from fantasy_baseball.hitter_ros.sequence import SequenceBatcher
from fantasy_baseball.pitcher_ros.features import (
    ERA_PIECES,
    league_answer_rates,
    league_reference,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS
from fantasy_baseball.ros_training import (
    holdout_and_scaler,
    usable_league_rates,
    weighted_moments,
)

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
    "relative_target",
    "seq",
    "seq_len",
    "seq_dim",
    "seq_layers",
    "micro_batch",
    "amp",
    "parks",
)


# The relative target the pitcher net uses (see fit_season). "answer", over 5 seeds x
# 2022-2026 (run 008-rel-answer vs 007-milb-avg; how sure better, 0-100): preseason K
# MSE 178.0 vs 183.8 (>99%), vets 189.7 vs 198.9 (>99%); mid-season K 61.97 vs 63.68
# (99%); preseason W (98%), SV on the fantasy-relevant (98%); ERA 1.329 vs 1.321
# preseason (4%) and 2.410 vs 2.403 mid-season (7%); the rest 36% or more. "known":
# preseason K 181.2 (90%), ERA 1.337 (<1%). Leaving ERA, or ERA and WHIP, as plain rates
# gave about the same K gain but traded ERA for WHIP, each version under 1% sure better
# on one of them somewhere. :func:`plain_config` keeps "none", the plain baseline;
# :func:`net_config` is the net as trained.
DEFAULT_RELATIVE_TARGET = "answer"

# The sequence encoder the pitcher net uses (fit_season; pitcher_ros/sequence.py): a
# transformer over each pitcher's last 150 appearances, their results against the
# league. 5 seeds x 2022-2026 (run 014-seq-lgctx vs 008-rel-answer; how sure better,
# 0-100), preseason: SV 18.28 vs 18.83 (98%), ERA 1.313 vs 1.329 (>99%), WHIP 0.0350
# vs 0.0354 (>99%), W (66%); K 180.2 vs 178.0 (2%) raw but 173.9 vs 173.9 (50%)
# level-free -- its K loss is the league level, mostly 2022 (+5.0% vs +3.9%).
# Mid-season: SV (93%), K 62.5 vs 62.0 (14%), the rest 55-68%.
DEFAULT_SEQ: dict[str, Any] = {
    "seq": "transformer",
    "seq_len": 150,
    "seq_dim": 64,
    "seq_layers": 1,
}


# The park inputs the pitcher net uses (pitcher_ros/parks.py): "none" (off), or the name
# of a data/pitcher_ros/parks_<name>.parquet file (scripts/build_pitcher_ros_parks.py).
# 5 seeds x 2022-2026 (run 016-parks vs 014-seq-lgctx; how sure better, 0-100),
# preseason: K 177.4 vs 180.2 (99%, better in 5/5 seeds), level-free ERA 1.241 vs 1.247
# (97%) and WHIP 0.0338 vs 0.0340 (95%); raw ERA/WHIP and W even. Mid-season K, W, ERA,
# WHIP 58-82%; SV 5.38 vs 5.24 (0%), but the single seeds overlap (5.29-5.42 vs
# 5.25-5.41), so mostly training noise. Top-25 starters' ERA unchanged (0.687 vs 0.688):
# the gap to FanGraphs there is not the park (#486).
DEFAULT_PARKS = "p3"

# Whether the pitcher net also predicts ERA's pieces (features.ERA_PIECES: BB, HBP, HR per
# batter faced) as extra outputs, ERA still its own (features.answers era_pieces). 5 seeds
# x 2022-2026 (run 017-era-pieces vs 016-parks; how sure better, 0-100), preseason on the
# fantasy-relevant: ERA MSE 1.051 vs 1.058 (99%), W 6.583 vs 6.562 (8%), the rest 47-76%;
# mid-season ERA 2.395 vs 2.405 (97%). In roto points (squared SGP miss per pitcher,
# preseason fantasy-relevant): ERA -1.0% (100%), W +0.3% (10%), the whole pitcher -0.07%
# (61%), and never worse in any view (52-78%). The pieces also showed where the top-25
# starters' ERA gap to FanGraphs is: home runs, then strikeouts (#486).
DEFAULT_ERA_PIECES = True


class LeagueRates(NamedTuple):
    """Each table row's league ``TARGETS`` (and ``features.ERA_PIECES``) rates, usable as
    divisors (0 -> NaN): ``reference``, the forecast known on the date
    (``features.league_reference``), and ``answer``, the league's actual rates over the
    answer window (``features.league_answer_rates``, a training denominator only)."""

    reference: pd.DataFrame
    answer: pd.DataFrame


def league_rates(table: pd.DataFrame) -> LeagueRates:
    """:class:`LeagueRates` for ``table``: built once per table, not per fit."""
    return LeagueRates(
        usable_league_rates(league_reference(table)),
        usable_league_rates(league_answer_rates(table)),
    )


def check_league_forecast(table: pd.DataFrame, league: LeagueRates, season: int) -> None:
    """Refuse ``season`` when a relative prediction of one of its rows couldn't be scaled
    back: no forecast (no earlier season in the table) or a league rate of 0."""
    forecast = league.reference.loc[table["season"] == season, list(TARGETS)]
    if not (forecast > 0).all().all():  # NaN compares False too
        raise ValueError(
            f"{season} has rows with no usable league forecast (no earlier season in the "
            "table, or a league rate of 0) to scale relative predictions back by"
        )


def net_config(**overrides: Any) -> NetConfig:
    """The pitcher net as trained: :func:`plain_config` with the applied defaults
    (``DEFAULT_RELATIVE_TARGET``, ``DEFAULT_SEQ``, ``DEFAULT_PARKS``), then ``overrides``."""
    return plain_config(
        **{
            "relative_target": DEFAULT_RELATIVE_TARGET,
            **DEFAULT_SEQ,
            "parks": DEFAULT_PARKS,
            **overrides,
        }
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
    league: LeagueRates | None = None,
    batcher: SequenceBatcher | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Train on complete seasons before ``test_season``; predict every row of it.

    A random ``config.val_frac`` of the training pitchers is held out for early
    stopping. Targets are standardized by their loss-weighted mean and SD over the
    training rows; predictions are put back in rates (never below 0).

    ``config.relative_target`` (as the hitter net's, #421 / #424): "none" learns the
    rates themselves. Otherwise each rate is learned divided by a league rate -- the
    league's actual rate over the answer window ("answer": how much better than the
    league, with no forecast of its level) or the forecast known on the date ("known",
    ``features.league_reference``) -- and the prediction is multiplied back by the
    forecast. Learned as rates, the net picks up the league's trend (K rose every season
    to 2020) and carries it on after it stops: before this it projected too many K in
    every test season (2022-2026), by 0.4 to 1.7 per 100 batters faced. ``league``:
    :func:`league_rates` of ``table``, if already built (a run fits many seasons and
    seeds on one table).

    ``config.seq`` (as the hitter net's, #414): "none", or an encoder ("gru",
    "transformer") over each pitcher's last ``seq_len`` appearances before the row's
    date, whose summary joins the inputs. ``batcher`` reads them: a
    ``hitter_ros.sequence.SequenceBatcher`` built on ``table``'s rows, in order, from
    ``pitcher_ros.sequence``'s tokens.

    ``y_all`` / ``w_all``: ``features.answers``, the ``TARGETS`` and, with its
    ``era_pieces``, ERA's pieces as extra outputs (predicted and returned beside them).

    ``config.parks``: "none", or the park file whose inputs (``pitcher_ros.parks``)
    ``x_all`` carries; refused when the two disagree, so a run is never credited to
    park inputs it didn't have (or the reverse)."""
    check_plain(config)
    check_week_zero(table)
    has_parks = set(PARK_FEATURES) <= set(x_all.columns)
    if (config.parks != "none") != has_parks:
        raise ValueError(
            f"parks {config.parks!r} needs the park inputs in x_all exactly when it is not 'none'"
        )
    if (config.seq != "none") != (batcher is not None):
        raise ValueError(f"seq {config.seq!r} needs a batcher exactly when it is not 'none'")
    if batcher is not None:
        # A longer batcher would index past the transformer's position table; a shorter
        # one would quietly read fewer appearances than the config (and config.json) says.
        if batcher.max_len != config.seq_len:
            raise ValueError(
                f"the batcher reads {batcher.max_len} appearances, config.seq_len {config.seq_len}"
            )
        if len(batcher.row_start) != len(table):
            raise ValueError("the batcher was built on a different table (row count differs)")
    positions = np.arange(len(table))
    cols = list(y_all.columns)
    if cols[: len(TARGETS)] != list(TARGETS) or not set(cols[len(TARGETS) :]) <= set(ERA_PIECES):
        raise ValueError(f"targets must be TARGETS then any of ERA_PIECES, got {cols}")
    # ERA's pieces (features.answers era_pieces) are extra outputs: each weighs a third
    # of a target, as the hitter net's AVG pieces do (#433).
    target_weights = np.array([1.0 if c in TARGETS else PIECE_LOSS_WEIGHT for c in cols])
    ref = None
    if config.relative_target != "none":
        if league is None:
            league = league_rates(table)
        check_league_forecast(table, league, test_season)
        ref = league.reference
        denominator = league.answer if config.relative_target == "answer" else ref
        y_all = y_all[cols] / denominator[cols]
        # train() expects weight 0 wherever the target is NaN.
        w_all = w_all[cols].where(y_all.notna(), 0.0)
    train_rows = (
        table["season_complete"] & (table["season"] < test_season) & (w_all.sum(axis=1) > 0)
    )
    val_mask, scaler = holdout_and_scaler(table, x_all, train_rows, config.val_frac, config.seed)

    y_train, w_train = y_all.loc[train_rows, cols], w_all.loc[train_rows, cols]
    unweighted = [c for c in cols if not (w_train.loc[~val_mask, c] > 0).any()]
    if unweighted:
        raise ValueError(
            f"no training row has an answer for {', '.join(unweighted)} (with a relative "
            "target: no league rate to divide by)"
        )
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
        target_weights=target_weights,
        rows=positions[train_rows.to_numpy()] if batcher else None,
        batcher=batcher,
    )
    test_rows = table["season"] == test_season
    z = predict(
        result.model,
        scaler.transform(x_all[test_rows]),
        rows=positions[test_rows.to_numpy()] if batcher else None,
        batcher=batcher,
        amp=config.amp,
    )
    preds = pd.DataFrame(
        {c: np.clip(z[:, i] * sd[c] + mu[c], 0.0, None) for i, c in enumerate(cols)},
        index=table.index[test_rows],
    )
    if ref is not None:
        preds = preds * ref.loc[test_rows, cols]
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
    """The rates (``TARGETS``, and ERA's pieces when predicted) averaged over seeds'
    predictions (each ``ID_COLS`` plus rates, as :func:`fit_season` returns them), row
    for row, sorted by ``ID_COLS``.
    Refuses unless every row appears exactly once per seed: seeds that predicted
    different rows would quietly average over fewer seeds there, and a repeated row
    would be merged into one."""
    if not preds:
        raise ValueError("no predictions to average")
    stacked = pd.concat(preds, ignore_index=True)
    rates = [c for c in stacked.columns if c not in ID_COLS]
    missing = [s for s in TARGETS if s not in rates]
    if missing:
        raise ValueError(f"the predictions have no {', '.join(missing)}")
    by_row = stacked.groupby(ID_COLS)
    if not (by_row.size() == len(preds)).all():
        raise ValueError("the seeds predicted different rows, or one repeated a row")
    # A rate one seed left blank (NaN) where another predicted it would be averaged
    # over fewer seeds there; blank in every seed (no league rate to scale by) is fine.
    if not by_row[rates].count().isin([0, len(preds)]).all().all():
        raise ValueError("a seed left a rate blank where another seed predicted it")
    return by_row[rates].mean().reset_index()
