"""Train the hitter ROS net and score it against FanGraphs, walk-forward (#404).

For each test season T, the net trains only on complete seasons before T (validation =
a random slice of those players), then predicts every row of T. Scores:

* Preseason (week 0) vs. each FanGraphs system in data/projections/T/, their blend, and
  two simple floors (league average, Marcel-style).
* Mid-season vs. each dated ROS snapshot in data/projections/T/rest_of_season/<date>/,
  using our row from the latest as-of date on or before the snapshot.

Each run is saved under data/hitter_ros/runs/<name>/: config.json (settings, plus each
test season's train/val loss per epoch), predictions.parquet, scored_preseason.parquet
and scored_snapshots.parquet (one row per player x system x stat), and summary.md (paste
into #404). Scoring is hitter_ros/backtest.py, shared with score_hitter_ros_run.py.
A name that already has a run is refused unless --overwrite.

Setup (once): pip install torch --index-url https://download.pytorch.org/whl/cu128
Usage:
    python scripts/build_hitter_ros_table.py      # if the table is stale
    python scripts/train_hitter_ros.py --name baseline-mlp
    python scripts/train_hitter_ros.py --name wider --hidden 256 128 --dropout 0.2
    python scripts/build_hitter_ros_pa_tokens.py   # once, for sequence runs
    python scripts/train_hitter_ros.py --name 003a-gru --seq gru
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros import backtest
from fantasy_baseball.hitter_ros.features import (
    ERA_MODES,
    ERA_TABLE_COLUMNS,
    STEAL_TABLE_COLUMNS,
    TARGETS,
    Standardizer,
    input_frame,
    league_answer_rates,
    league_reference,
    target_frame,
)
from fantasy_baseball.hitter_ros.net import (
    COUNT_LOSS_TARGETS,
    EVAL_BATCH,
    LOG_RATE_CLAMP,
    RELATIVE_TARGETS,
    NetConfig,
    device,
    predict,
    train,
)
from fantasy_baseball.hitter_ros.probes import (
    PROBE_FEATURES,
    check_probes,
    probe_inputs,
    probe_path,
)
from fantasy_baseball.hitter_ros.sequence import SequenceBatcher

TABLE = PROJECT_ROOT / "data" / "hitter_ros" / "table.parquet"
STORE = PROJECT_ROOT / "data" / "pitch_data"
TOKENS = PROJECT_ROOT / "data" / "hitter_ros" / "pa_tokens.parquet"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"

logger = logging.getLogger("train_hitter_ros")


def fit_season(
    table: pd.DataFrame,
    x_all: pd.DataFrame,
    y_all: pd.DataFrame,
    w_all: pd.DataFrame,
    test_season: int,
    config: NetConfig,
    batcher: SequenceBatcher | None = None,
    shuffle_test_order: bool = False,
    train_from: int | None = None,
    weeks: str = "all",
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train on complete seasons before ``test_season``; predict that season's rows.

    With a sequence model, ``batcher`` was built on ``table`` and is handed each row's
    table position to fetch its plate appearances. ``weeks`` ("all", "pre" = week 0,
    "mid" = week 1+) limits both the training and the predicted rows (``--split``).
    """
    positions = np.arange(len(table))
    in_weeks = WEEKS[weeks](table["week"])
    train_rows = (
        table["season_complete"]
        & (table["season"] < test_season)
        & (w_all.sum(axis=1) > 0)
        & in_weeks
    )
    if train_from is not None:  # learning-curve runs: train on fewer, more recent seasons
        train_rows &= table["season"] >= train_from
    rng = np.random.default_rng(config.seed)
    players = table.loc[train_rows, "player_id"].unique()
    val_players = rng.choice(players, size=int(len(players) * config.val_frac), replace=False)
    val_mask = table.loc[train_rows, "player_id"].isin(val_players).to_numpy()

    scaler = Standardizer().fit(x_all[train_rows & ~table["player_id"].isin(val_players)])
    # relative_target: learn each player's rates divided by a league rate -- the forecast
    # known on the date (#421) or the league's actual rate over the answer window (#424)
    # -- and multiply back by the forecast. The forecast's own error then shows up only in
    # raw MAE, never in the relative scores.
    ref = league_reference(table) if config.relative_target != "none" else None
    denominator = league_answer_rates(table) if config.relative_target == "answer" else ref
    y_fit = y_all / denominator if denominator is not None else y_all
    # A row whose answer can't be computed (no PA, or no league reference) must not
    # count: train() expects weight 0 wherever the target is NaN.
    w_fit = w_all.where(y_fit.notna(), 0.0)
    y_train, w_train = y_fit[train_rows], w_fit[train_rows]
    fit_rows = ~val_mask
    mu = {
        s: np.average(y_train[s][fit_rows].fillna(0), weights=w_train[s][fit_rows]) for s in TARGETS
    }
    sd = {
        s: np.sqrt(
            np.average((y_train[s][fit_rows].fillna(0) - mu[s]) ** 2, weights=w_train[s][fit_rows])
        )
        for s in TARGETS
    }
    # Poisson targets (count_loss) stay as rates: the net predicts their log.
    poisson = set(COUNT_LOSS_TARGETS[config.count_loss])
    y_std = np.column_stack(
        [y_train[s] if s in poisson else (y_train[s] - mu[s]) / sd[s] for s in TARGETS]
    )

    def inputs(rows: pd.Series) -> np.ndarray:
        x = scaler.transform(x_all[rows])
        if config.heads > 1:  # last column: the head index, 0 = preseason, 1 = mid-season
            mid = (table.loc[rows, "week"] > 0).to_numpy(np.float32)
            x = np.column_stack([x, mid])
        return x

    result = train(
        inputs(train_rows),
        y_std,
        w_train.to_numpy(dtype=np.float32),
        val_mask,
        config,
        rows=positions[train_rows.to_numpy()] if batcher else None,
        batcher=batcher,
        season_time=table.loc[train_rows, "frac_season_left"].to_numpy(),
    )
    test_rows = (table["season"] == test_season) & in_weeks
    z = predict(
        result.model,
        inputs(test_rows),
        rows=positions[test_rows.to_numpy()] if batcher else None,
        batcher=batcher,
        shuffle_order=shuffle_test_order,
        chunk=config.micro_batch or EVAL_BATCH,
        amp=config.amp,
    )
    preds = pd.DataFrame(
        {
            s: np.exp(np.clip(z[:, i], *LOG_RATE_CLAMP))
            if s in poisson
            else z[:, i] * sd[s] + mu[s]
            for i, s in enumerate(TARGETS)
        },
        index=table.index[test_rows],
    )
    if ref is not None:
        preds = preds * ref.loc[test_rows, list(TARGETS)]
    preds = pd.concat(
        [table.loc[test_rows, ["player_id", "season", "week", "as_of"]], preds], axis=1
    )
    info = {
        "test_season": test_season,
        "weeks": weeks,
        "train_rows": int(train_rows.sum() - val_mask.sum()),
        "val_rows": int(val_mask.sum()),
        "n_features": scaler.n_features,
        "best_epoch": result.best_epoch,
        "train_loss": result.train_loss,
        "val_loss": result.val_loss,
    }
    return preds, info


WEEKS = {
    "all": lambda week: week >= 0,
    "pre": lambda week: week == 0,
    "mid": lambda week: week >= 1,
}


def _train_from(season: int, n_seasons: int | None, first: int | None) -> int | None:
    """Earliest training season: the later of --first-train-season and the
    --train-seasons window; None (no limit) when neither is given."""
    limits = [
        x for x in (first, None if n_seasons is None else season - n_seasons) if x is not None
    ]
    return max(limits) if limits else None


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError(f"must be >= 1, got {value}")
    return value


def _non_negative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError(f"must be >= 0, got {value}")
    return value


def main() -> int:
    defaults = NetConfig()
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--name", default=f"run-{date.today().isoformat()}")
    parser.add_argument(
        "--test-seasons", type=int, nargs="+", default=[2022, 2023, 2024, 2025, 2026]
    )
    parser.add_argument("--hidden", type=int, nargs="+", default=defaults.hidden)
    parser.add_argument("--dropout", type=float, default=defaults.dropout)
    parser.add_argument("--lr", type=float, default=defaults.lr)
    parser.add_argument("--weight-decay", type=float, default=defaults.weight_decay)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--max-epochs", type=int, default=defaults.max_epochs)
    parser.add_argument("--patience", type=int, default=defaults.patience)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--era",
        choices=list(ERA_MODES),
        default=defaults.era,
        help="era inputs (#421): player-vs-league ratios; full adds league rates and rule flags",
    )
    parser.add_argument(
        "--relative-target",
        choices=list(RELATIVE_TARGETS),
        default=defaults.relative_target,
        help="predict rates relative to the league: 'known' = the forecast known on the "
        "date (last 3 seasons + this season so far, #421); 'answer' = the league's actual "
        "rate over the answer window (#424). Both multiply back by the forecast.",
    )
    parser.add_argument(
        "--probes",
        default=defaults.probes,
        help="add probe features (#417) from this pretraining run (default p004; none = off) "
        "(build them first with scripts/build_hitter_ros_probes.py)",
    )
    parser.add_argument(
        "--heads",
        type=int,
        choices=[1, 2],
        default=defaults.heads,
        help="2: shared body with a preseason head (week 0) and a mid-season head (#422)",
    )
    parser.add_argument(
        "--count-loss",
        choices=list(COUNT_LOSS_TARGETS),
        default=defaults.count_loss,
        help="Poisson loss on counts for SB only, or for R/HR/RBI/SB (#413)",
    )
    parser.add_argument(
        "--steal-inputs",
        action=argparse.BooleanOptionalAction,
        default=defaults.steal_inputs,
        help="steal opportunity, position and team green-light inputs (#413); on by default",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="two separate models: one trained on week-0 rows, one on week 1+ (#422)",
    )
    parser.add_argument(
        "--weighting",
        choices=["pa", "balanced", "pre_mid"],
        default=defaults.weighting,
        help="balanced: each fifth of the season gets equal total loss weight; "
        "pre_mid: week 0 and week 1+ get equal totals (#422 control)",
    )
    parser.add_argument(
        "--seq",
        choices=["none", "gru", "transformer"],
        default=defaults.seq,
        help="sequence encoder over recent plate appearances (#414)",
    )
    parser.add_argument("--seq-len", type=int, default=defaults.seq_len)
    parser.add_argument("--seq-dim", type=int, default=defaults.seq_dim)
    parser.add_argument("--seq-layers", type=int, default=defaults.seq_layers)
    parser.add_argument(
        "--micro-batch",
        type=_non_negative_int,
        default=defaults.micro_batch,
        help="rows per forward pass inside a batch (saves GPU memory; same update)",
    )
    parser.add_argument(
        "--amp", action="store_true", help="bfloat16 for the sequence model on the GPU"
    )
    parser.add_argument(
        "--shuffle-test-order",
        action="store_true",
        help="predict with each hitter's PAs in random order (does the net use order?)",
    )
    parser.add_argument("--note", default="", help="what this run changes and why")
    parser.add_argument(
        "--first-train-season",
        type=int,
        help="train only on seasons from this one on (e.g. 2015, the first Statcast year)",
    )
    parser.add_argument(
        "--train-seasons",
        type=_positive_int,
        help="train only on the N seasons right before each test season (learning curves)",
    )
    parser.add_argument("--overwrite", action="store_true", help="replace a run with this name")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    try:
        config = NetConfig(
            hidden=args.hidden,
            dropout=args.dropout,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            max_epochs=args.max_epochs,
            patience=args.patience,
            seed=args.seed,
            seq=args.seq,
            seq_len=args.seq_len,
            seq_dim=args.seq_dim,
            seq_layers=args.seq_layers,
            micro_batch=args.micro_batch,
            amp=args.amp,
            weighting=args.weighting,
            era=args.era,
            relative_target=args.relative_target,
            probes=args.probes,
            heads=args.heads,
            split=args.split,
            steal_inputs=args.steal_inputs,
            count_loss=args.count_loss,
        )
    except ValueError as err:
        parser.error(str(err))
    if args.shuffle_test_order and args.seq == "none":
        parser.error("--shuffle-test-order needs a sequence model (--seq gru/transformer)")
    out = RUNS / args.name
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        parser.error(f"{out} already has a run; pick another --name or pass --overwrite")
    if config.seq != "none" and not TOKENS.exists():
        parser.error(f"{TOKENS} is missing; run scripts/build_hitter_ros_pa_tokens.py")

    table = pd.read_parquet(TABLE)
    if (config.era != "none" or config.relative_target != "none") and not set(
        ERA_TABLE_COLUMNS
    ) <= set(table.columns):
        parser.error(f"{TABLE} predates the era columns; run scripts/build_hitter_ros_table.py")
    # pre_mid finds week 0 as the rows with the whole season left; --heads by week > 0.
    # They must be the same rows, or pre_mid would quietly stop balancing.
    if not ((table["week"] == 0) == (table["frac_season_left"] >= 1)).all():
        parser.error(f"{TABLE}: week 0 is not exactly the rows with frac_season_left 1")
    if config.steal_inputs and not set(STEAL_TABLE_COLUMNS) <= set(table.columns):
        parser.error(f"{TABLE} predates the steal columns; run scripts/build_hitter_ros_table.py")
    x_all = input_frame(table, era=config.era, steal=config.steal_inputs)
    if config.probes != "none":
        probes_file = probe_path(TABLE.parent, config.probes)
        if not probes_file.exists():
            parser.error(f"{probes_file} is missing; run scripts/build_hitter_ros_probes.py")
        probes = pd.read_parquet(probes_file)
        problem = check_probes(table, probes)
        if problem:
            parser.error(f"{probes_file}: {problem}; rebuild it for this table")
        x_all = pd.concat([x_all, probe_inputs(table, probes)], axis=1)
        logger.info("added %d probe features from %s", len(PROBE_FEATURES), probes_file.name)
    y_all, w_all = target_frame(table)
    batcher = None
    if config.seq != "none":
        batcher = SequenceBatcher(pd.read_parquet(TOKENS), table, config.seq_len, device())
    # Only now, with every input loaded, replace an old run of the same name.
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    all_preds, infos = [], []
    for season in args.test_seasons:
        logger.info("test season %s: training on complete seasons before it", season)
        # --split: a preseason-only model and a mid-season-only model (#422).
        for weeks in ("pre", "mid") if config.split else ("all",):
            preds, info = fit_season(
                table,
                x_all,
                y_all,
                w_all,
                season,
                config,
                batcher,
                args.shuffle_test_order,
                train_from=_train_from(season, args.train_seasons, args.first_train_season),
                weeks=weeks,
            )
            all_preds.append(preds)
            infos.append(info)
    predictions = pd.concat(all_preds)
    predictions.to_parquet(out / "predictions.parquet")
    meta = {
        "note": args.note,
        "config": config.to_dict(),
        "shuffle_test_order": args.shuffle_test_order,
        "train_seasons": args.train_seasons,
        "first_train_season": args.first_train_season,
        "seasons": infos,
    }
    (out / "config.json").write_text(json.dumps(meta, indent=2))
    pre, snap = backtest.score_predictions(table, predictions, PROJECTIONS, STORE)
    backtest.write_scores(out, pre, snap)
    md = [
        f"### Run `{args.name}`",
        "",
        args.note,
        "",
        f"Config: `{json.dumps(config.to_dict())}`",
        "",
        *backtest.summarize(pre, snap),
        *backtest.league_forecast_lines(table, args.test_seasons),
    ]
    # A --split run has two models per season: label them so the epochs aren't ambiguous.
    epochs = ", ".join(
        f"{i['test_season']}{'' if i['weeks'] == 'all' else ' ' + str(i['weeks'])}: "
        f"{i['best_epoch']}"
        for i in infos
    )
    md += ["", f"Best epoch per test season: {epochs}. Inputs: {infos[0]['n_features']}."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
