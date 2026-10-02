"""Train the hitter ROS net and score it against FanGraphs, walk-forward (#404).

For each test season T, the net trains only on complete seasons before T (validation =
a random slice of those players), then predicts every row of T. Scores:

* Preseason (week 0) vs. each FanGraphs system in data/projections/T/ and their blend.
* Mid-season vs. each dated ROS snapshot in data/projections/T/rest_of_season/<date>/,
  using our row from the latest as-of date on or before the snapshot.

Each run is saved under data/hitter_ros/runs/<name>/: config.json, scores.csv,
summary.md (paste into #404), predictions.parquet, loss curves.

Setup (once): pip install torch --index-url https://download.pytorch.org/whl/cu128
Usage:
    python scripts/build_hitter_ros_table.py      # if the table is stale
    python scripts/train_hitter_ros.py --name baseline-mlp
    python scripts/train_hitter_ros.py --name wider --hidden 512 512 256 --dropout 0.2
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fantasy_baseball.hitter_ros.evaluate import blend, load_systems, rates_from_counts, score
from fantasy_baseball.hitter_ros.features import (
    TARGETS,
    Standardizer,
    input_frame,
    target_frame,
)
from fantasy_baseball.hitter_ros.net import NetConfig, predict, train
from fantasy_baseball.pitch_data.store import connect

TABLE = PROJECT_ROOT / "data" / "hitter_ros" / "table.parquet"
STORE = PROJECT_ROOT / "data" / "pitch_data"
PROJECTIONS = PROJECT_ROOT / "data" / "projections"
RUNS = PROJECT_ROOT / "data" / "hitter_ros" / "runs"
PRESEASON_MIN_PA = 300
SNAPSHOT_MIN_PA = 100
OURS = "ours"

logger = logging.getLogger("train_hitter_ros")


def fit_season(
    table: pd.DataFrame, x_all: pd.DataFrame, test_season: int, config: NetConfig
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Train on complete seasons before ``test_season``; predict that season's rows."""
    y_all, w_all = target_frame(table)
    train_rows = (
        table["season_complete"] & (table["season"] < test_season) & (w_all.sum(axis=1) > 0)
    )
    rng = np.random.default_rng(config.seed)
    players = table.loc[train_rows, "player_id"].unique()
    val_players = rng.choice(players, size=int(len(players) * config.val_frac), replace=False)
    val_mask = table.loc[train_rows, "player_id"].isin(val_players).to_numpy()

    scaler = Standardizer().fit(x_all[train_rows & ~table["player_id"].isin(val_players)])
    y_train, w_train = y_all[train_rows], w_all[train_rows]
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
    y_std = np.column_stack([(y_train[s] - mu[s]) / sd[s] for s in TARGETS])

    result = train(
        scaler.transform(x_all[train_rows]),
        y_std,
        w_train.to_numpy(dtype=np.float32),
        val_mask,
        config,
    )
    test_rows = table["season"] == test_season
    z = predict(result.model, scaler.transform(x_all[test_rows]))
    preds = pd.DataFrame(
        {s: z[:, i] * sd[s] + mu[s] for i, s in enumerate(TARGETS)},
        index=table.index[test_rows],
    )
    preds = pd.concat(
        [table.loc[test_rows, ["player_id", "season", "week", "as_of"]], preds], axis=1
    )
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


def preseason_scores(table: pd.DataFrame, preds: pd.DataFrame, season: int) -> pd.DataFrame | None:
    systems = load_systems(PROJECTIONS / str(season))
    if not systems:
        return None
    week0 = table[(table["season"] == season) & (table["week"] == 0)].set_index("player_id")
    actual = rates_from_counts(week0.rename(columns=lambda c: c.removeprefix("ros_")))
    actual["pa"] = week0["ros_pa"]
    ours = preds[preds["week"] == 0].set_index("player_id")[list(TARGETS)]
    projections = {OURS: ours, **systems}
    if len(systems) > 1:
        projections["fg_blend"] = blend(systems)
    return score(projections, actual, PRESEASON_MIN_PA)


def snapshot_scores(preds: pd.DataFrame, season: int) -> pd.DataFrame | None:
    root = PROJECTIONS / str(season) / "rest_of_season"
    if not root.is_dir():
        return None
    conn = connect(STORE)
    tables = []
    for snap_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        snap = date.fromisoformat(snap_dir.name)
        systems = load_systems(snap_dir)
        if not systems:
            continue
        actual_counts = (
            conn.execute(
                """
            SELECT player_id, sum(pa) AS pa, sum(ab) AS ab, sum(h) AS h, sum(r) AS r,
                   sum(hr) AS hr, sum(rbi) AS rbi, sum(sb) AS sb
            FROM lineups
            WHERE year(CAST(game_date AS DATE)) = ? AND CAST(game_date AS DATE) >= ?
            GROUP BY player_id
            """,
                [season, snap],
            )
            .df()
            .set_index("player_id")
        )
        actual = rates_from_counts(actual_counts)
        actual["pa"] = actual_counts["pa"]
        if (actual["pa"] >= SNAPSHOT_MIN_PA).sum() == 0:
            continue
        ours_rows = preds[preds["as_of"].dt.date <= snap]
        ours = (ours_rows.sort_values("as_of").groupby("player_id").tail(1).set_index("player_id"))[
            list(TARGETS)
        ]
        projections = {OURS: ours, **systems}
        if len(systems) > 1:
            projections["fg_blend"] = blend(systems)
        t = score(projections, actual, SNAPSHOT_MIN_PA)
        t.insert(0, "snapshot", snap.isoformat())
        tables.append(t)
    conn.close()
    return pd.concat(tables) if tables else None


def to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| system | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in df.iterrows():
        cells = [
            f"{v:.2f}" if isinstance(v, float) and c != "n" else str(int(v)) if c == "n" else str(v)
            for c, v in row.items()
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


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
    parser.add_argument("--note", default="", help="what this run changes and why")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    config = NetConfig(
        hidden=args.hidden,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
        patience=args.patience,
        seed=args.seed,
    )
    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=True)

    table = pd.read_parquet(TABLE)
    x_all = input_frame(table)

    all_preds, infos, pre_tables, snap_tables = [], [], [], []
    for season in args.test_seasons:
        logger.info("test season %s: training on complete seasons before it", season)
        preds, info = fit_season(table, x_all, season, config)
        all_preds.append(preds)
        infos.append(info)
        pre = preseason_scores(table, preds, season)
        if pre is not None:
            pre.insert(0, "season", season)
            pre_tables.append(pre)
        snap = snapshot_scores(preds, season)
        if snap is not None:
            snap.insert(0, "season", season)
            snap_tables.append(snap)

    pd.concat(all_preds).to_parquet(out / "predictions.parquet")
    (out / "config.json").write_text(
        json.dumps({"note": args.note, "config": config.to_dict(), "seasons": infos}, indent=2)
    )
    md = [
        f"### Run `{args.name}`",
        "",
        args.note,
        "",
        f"Config: `{json.dumps(config.to_dict())}`",
        "",
    ]
    md.append(
        f"Raw error, lower is better: R/HR/RBI/SB = MAE per 600 PA, AVG = MAE in points. "
        f"Preseason: players with >= {PRESEASON_MIN_PA} actual PA. "
        f"Mid-season: >= {SNAPSHOT_MIN_PA} PA after the snapshot."
    )
    if pre_tables:
        pre_all = pd.concat(pre_tables)
        pre_all.to_csv(out / "scores_preseason.csv")
        md += ["", "#### Preseason"]
        for season, g in pre_all.groupby("season"):
            md += ["", f"**{season}**", "", to_markdown(g.drop(columns="season"))]
        mean = pre_all.drop(columns=["season"]).groupby(level=0).mean()
        md += ["", "**Mean over seasons** (systems present every season only)", ""]
        present = pre_all.groupby(level=0)["season"].nunique() == len(pre_tables)
        md.append(to_markdown(mean[present.reindex(mean.index).fillna(False)]))
    if snap_tables:
        snap_all = pd.concat(snap_tables)
        snap_all.to_csv(out / "scores_snapshots.csv")
        md += ["", "#### Mid-season (ROS snapshots)"]
        for (_season, snapshot), g in snap_all.groupby(["season", "snapshot"]):
            md += ["", f"**{snapshot}**", "", to_markdown(g.drop(columns=["season", "snapshot"]))]
    epochs = ", ".join(f"{i['test_season']}: {i['best_epoch']}" for i in infos)
    md += ["", f"Best epoch per test season: {epochs}. Inputs: {infos[0]['n_features']}."]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
