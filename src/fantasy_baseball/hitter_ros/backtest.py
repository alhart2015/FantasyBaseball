"""Score a run's predictions against FanGraphs and the simple baselines (#404, #410).

Given the net's predictions for some test seasons, this builds the long "scored" frames
(one row per player x system x stat, tagged with ``season`` and, mid-season,
``snapshot``) for:

* Preseason (week 0) vs. each FanGraphs system in ``data/projections/T/``, their blend,
  and the ``league_avg`` / ``marcel`` baselines.
* Mid-season vs. each dated ROS snapshot in ``data/projections/T/rest_of_season/<date>/``,
  using each projection's row from the latest as-of date on or before the snapshot;
  actuals are the games on or after the snapshot date.

and renders a markdown summary with MAE tables, a paired bootstrap of ours vs. the
FanGraphs blend, and how spread out each system's projections are.

Every scored row is also tagged vet or rookie (#433): a vet had at least
``VET_MIN_CAREER_PA`` MLB plate appearances when the projection was made (before the
season, or by the snapshot mid-season). Projecting the two is a different problem (a vet
has MLB history; a rookie needs minor-league stats and pedigree we don't have yet), so
the summary scores each group on its own, with pairs formed only inside a group.

And every row is tagged ``relevant`` (#442): the player was among the
``RELEVANT_TOP`` hitters by fantasy value (SGP in this league's categories) either in
the FanGraphs projections being compared against (the preseason files, or that
snapshot's rest-of-season files) or in what actually happened over the same span. Those
are the hitters a team would roster, or wish it had; a score over everyone is mostly
about bench bats nobody drafts. Picking by FanGraphs' value alone would choose the set
on the compared system's own output: on 2026 mid-season that showed an R gap that
vanished when the set was picked on actual value, so both count. A season or snapshot
without FanGraphs files has no tag (NA). The summary scores the relevant hitters on
their own.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball.hitter_ros.baselines import baseline_predictions
from fantasy_baseball.hitter_ros.evaluate import (
    SCALE,
    SURE_BAR,
    blend,
    fantasy_value,
    load_systems,
    mae_table,
    order_scores,
    order_table,
    paired_bootstrap,
    pairwise_bootstrap,
    scored_players,
    spread,
)
from fantasy_baseball.hitter_ros.features import (
    COUNTS,
    ERA_TABLE_COLUMNS,
    TARGETS,
    league_answer_rates,
    league_reference,
    rates_from_counts,
)
from fantasy_baseball.pitch_data.store import connect
from fantasy_baseball.sgp.denominators import get_sgp_denominators
from fantasy_baseball.utils.constants import Category

PRESEASON_MIN_PA = 300
SNAPSHOT_MIN_PA = 100
OURS = "ours"
BLEND = "fg_blend"
# At least this many MLB PA when projected = vet; fewer = rookie (#433).
VET_MIN_CAREER_PA = 300
# A short career count is only trusted with this many earlier seasons in the box-score
# store (it starts in 2008, so from 2012 on); before that a vet can look like a rookie.
MIN_HISTORY_SEASONS = 4
GROUPS = ("vet", "rookie")  # scored separately; "unknown" rows are in neither
# The hitters that matter in a fantasy league (#442): this many by FanGraphs' value.
RELEVANT_TOP = 200
# Young hitters among them, shown on their own: under this many MLB PA when projected.
YOUNG_MAX_CAREER_PA = 700


def vet_or_unknown_rows(table: pd.DataFrame, vet_min_pa: float) -> np.ndarray:
    """Per ``table`` row: at least ``vet_min_pa`` MLB PA when projected (``car_pa +
    std_pa``), or a career count that can't be trusted (fewer than
    ``MIN_HISTORY_SEASONS`` earlier seasons in the store). The rows the minor-league and
    pedigree build options blank."""
    mlb_pa = table["car_pa"].astype(float) + table["std_pa"].astype(float)
    short_history = table["car_seasons_in_store"] < MIN_HISTORY_SEASONS
    out: np.ndarray = ((mlb_pa >= vet_min_pa) | short_history).to_numpy()
    return out


def _with_fangraphs(
    ours: dict[str, pd.DataFrame], systems: dict[str, pd.DataFrame]
) -> dict[str, pd.DataFrame]:
    projections = {**ours, **systems}
    if len(systems) > 1:
        projections[BLEND] = blend(systems)
    return projections


def league_denominators(config_path: Path) -> dict[Category, float]:
    """The league's SGP denominators (``league.yaml``'s overrides on the defaults); the
    defaults alone when there is no league file."""
    if not config_path.exists():
        return get_sgp_denominators()
    from fantasy_baseball.config import load_config

    return get_sgp_denominators(load_config(config_path).sgp_overrides)


def relevant_players(
    systems: dict[str, pd.DataFrame],
    actual: pd.DataFrame,
    denoms: dict[Category, float],
    top: int = RELEVANT_TOP,
) -> pd.Index:
    """The ``top`` hitters by fantasy value averaged over the FanGraphs systems that
    project each one, together with the ``top`` by actual value (``actual``: rates plus
    ``pa`` and ``ab`` over the scored span)."""
    values = pd.concat([fantasy_value(p, denoms) for p in systems.values()], axis=1)
    projected = values.mean(axis=1).nlargest(top).index
    return projected.union(fantasy_value(actual, denoms).nlargest(top).index)


def _tag_relevant(
    scored: pd.DataFrame,
    systems: dict[str, pd.DataFrame],
    actual: pd.DataFrame,
    denoms: dict[Category, float] | None,
) -> pd.DataFrame:
    if not systems:
        return scored.assign(relevant=pd.array([pd.NA] * len(scored), dtype="boolean"))
    denoms = denoms if denoms is not None else get_sgp_denominators()
    top = relevant_players(systems, actual, denoms, RELEVANT_TOP)
    return scored.assign(relevant=pd.array(scored["player_id"].isin(top), dtype="boolean"))


def preseason(
    table: pd.DataFrame,
    candidates: dict[str, pd.DataFrame],
    season: int,
    projections_dir: Path,
    denoms: dict[Category, float] | None = None,
) -> pd.DataFrame:
    """Score week-0 rows of ``season``. ``candidates``: our predictions and baselines.

    Seasons without FanGraphs files (before 2022) are still scored, on ours and the
    baselines only, so our own variants can be compared over many more seasons.
    ``denoms``: SGP denominators for the ``relevant`` tag (None: the code defaults).
    """
    systems = load_systems(projections_dir / str(season), preseason=True)
    week0 = table[(table["season"] == season) & (table["week"] == 0)].set_index("player_id")
    actual = rates_from_counts(week0.rename(columns=lambda c: c.removeprefix("ros_")))
    actual["pa"] = week0["ros_pa"]
    actual["ab"] = week0["ros_ab"]
    ours = {
        name: p[p["week"] == 0].set_index("player_id")[list(TARGETS)]
        for name, p in candidates.items()
    }
    scored = scored_players(_with_fangraphs(ours, systems), actual, PRESEASON_MIN_PA)
    return _tag_relevant(scored, systems, actual, denoms).assign(season=season)


def _season_games(store: Path, season: int) -> pd.DataFrame:
    conn = connect(store)
    try:
        games = conn.execute(
            f"""
            SELECT player_id, CAST(game_date AS DATE) AS game_date, {", ".join(COUNTS)}
            FROM lineups WHERE year(CAST(game_date AS DATE)) = ?
            """,
            [season],
        ).df()
    finally:
        conn.close()
    games["game_date"] = pd.to_datetime(games["game_date"])
    return games


def snapshots(
    candidates: dict[str, pd.DataFrame],
    season: int,
    projections_dir: Path,
    store: Path,
    denoms: dict[Category, float] | None = None,
) -> pd.DataFrame | None:
    """Score every dated ROS snapshot of ``season``; ``denoms`` as for :func:`preseason`."""
    root = projections_dir / str(season) / "rest_of_season"
    if not root.is_dir():
        return None
    games = _season_games(store, season)
    by_date = {name: p.sort_values("as_of") for name, p in candidates.items()}
    parts = []
    for snap_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        snap = pd.Timestamp(date.fromisoformat(snap_dir.name))
        systems = load_systems(snap_dir)
        if not systems:
            continue
        counts = games[games["game_date"] >= snap].groupby("player_id")[list(COUNTS)].sum()
        actual = rates_from_counts(counts)
        actual["pa"] = counts["pa"]
        actual["ab"] = counts["ab"]
        if (actual["pa"] >= SNAPSHOT_MIN_PA).sum() == 0:
            continue
        ours = {}
        for name, p in by_date.items():
            known = p[p["as_of"] <= snap]
            ours[name] = known.groupby("player_id").tail(1).set_index("player_id")[list(TARGETS)]
        scored = scored_players(_with_fangraphs(ours, systems), actual, SNAPSHOT_MIN_PA)
        scored = _tag_relevant(scored, systems, actual, denoms)
        parts.append(scored.assign(season=season, snapshot=snap.date().isoformat()))
    return pd.concat(parts, ignore_index=True) if parts else None


def tag_experience(scored: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    """``scored`` plus ``career_pa`` (MLB PA when the projection was made) and ``group``:
    "vet" (at least ``VET_MIN_CAREER_PA``), "rookie", or "unknown".

    Career PA is ``car_pa + std_pa`` on the table row the projection came from: the
    week-0 row preseason, the latest row on or before the snapshot mid-season (so a
    call-up who has since piled up MLB PA turns into a vet). ``car_pa`` counts every
    earlier season in the box-score store, which starts in 2008; with fewer than
    ``MIN_HISTORY_SEASONS`` of them a short count may just be missing history, so such a
    player is "unknown" rather than a rookie. A scored player with no table row to read
    is an error, not a silent rookie.
    """
    cols = ["season", "player_id", "as_of", "car_pa", "std_pa", "car_seasons_in_store"]
    keys = ["season", "player_id"]
    # One key dtype on both sides: the table stores season as int32, merge_asof refuses a mix.
    rows = table.loc[:, [*cols, "week"]].astype({k: "int64" for k in keys})
    base = scored.drop(columns=["career_pa", "group"], errors="ignore").astype(
        {k: "int64" for k in keys}
    )
    if "snapshot" in base.columns:
        wanted = base[[*keys, "snapshot"]].drop_duplicates()
        wanted = wanted.assign(when=pd.to_datetime(wanted["snapshot"]).astype("datetime64[ns]"))
        found = pd.merge_asof(
            wanted.sort_values("when"),
            rows[cols].assign(as_of=rows["as_of"].astype("datetime64[ns]")).sort_values("as_of"),
            left_on="when",
            right_on="as_of",
            by=keys,
            direction="backward",
        )
        keys = [*keys, "snapshot"]
    else:
        found = rows.loc[rows["week"] == 0, cols]
    found = found.assign(career_pa=found["car_pa"] + found["std_pa"])
    tagged = base.merge(
        found[[*keys, "career_pa", "car_seasons_in_store"]],
        on=keys,
        how="left",
        validate="many_to_one",
    )
    missing = tagged["career_pa"].isna()
    if missing.any():
        examples = tagged.loc[missing, keys].drop_duplicates().head()
        raise ValueError(
            f"{missing.sum()} scored rows have no table row to take career PA from, "
            f"e.g. {examples.to_dict('records')}"
        )
    vet = tagged["career_pa"] >= VET_MIN_CAREER_PA
    known = tagged["car_seasons_in_store"] >= MIN_HISTORY_SEASONS
    group = pd.Series("unknown", index=tagged.index)
    group[vet] = "vet"
    group[~vet & known] = "rookie"
    return tagged.drop(columns="car_seasons_in_store").assign(group=group)


def unplayed_seasons(table: pd.DataFrame, seasons: Iterable[int]) -> list[int]:
    """The ``seasons`` with rows in ``table`` but no game played yet (every answer
    empty): next season's preseason rows, with nothing to score them against."""
    rows = table.loc[table["season"].isin(list(seasons))]
    ros_pa = rows.groupby("season")["ros_pa"].sum()
    return sorted(int(season) for season, pa in ros_pa.items() if pa == 0)


def score_predictions(
    table: pd.DataFrame,
    preds: pd.DataFrame,
    projections_dir: Path,
    store: Path,
    denoms: dict[Category, float] | None = None,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """(preseason, snapshots) scored frames for every season in ``preds``, each row
    tagged vet or rookie (``tag_experience``) and ``relevant`` or not (``denoms``: the
    league's SGP denominators; None = the code defaults)."""
    pre_parts, snap_parts = [], []
    for season in sorted(int(s) for s in preds["season"].unique()):
        candidates = {OURS: preds[preds["season"] == season], **baseline_predictions(table, season)}
        pre_parts.append(preseason(table, candidates, season, projections_dir, denoms))
        snap = snapshots(candidates, season, projections_dir, store, denoms)
        if snap is not None:
            snap_parts.append(snap)
    return (
        tag_experience(pd.concat(pre_parts, ignore_index=True), table) if pre_parts else None,
        tag_experience(pd.concat(snap_parts, ignore_index=True), table) if snap_parts else None,
    )


def write_scores(run_dir: Path, pre: pd.DataFrame | None, snap: pd.DataFrame | None) -> None:
    """Write the scored frames; a kind with nothing scored now has its old file removed."""
    for frame, name in ((pre, "scored_preseason.parquet"), (snap, "scored_snapshots.parquet")):
        path = run_dir / name
        if frame is not None:
            frame.to_parquet(path)
        else:
            path.unlink(missing_ok=True)


def systems_in_every_season(scored: pd.DataFrame, unit: str = "season") -> pd.DataFrame:
    """The rows of systems scored in every season (or snapshot, with ``unit``), so their
    means are over the same units and compare down a column."""
    n_units = scored[unit].nunique()
    counts = scored.groupby("system")[unit].nunique()
    return scored[scored["system"].isin(counts.index[counts == n_units])]


def mean_over_seasons(
    scored: pd.DataFrame, value: str = "abs_err", unit: str = "season"
) -> pd.DataFrame:
    """Per-system MAE averaged over seasons (each season counts once), systems x stats.
    ``value``: ``abs_err`` (raw MAE) or ``lf_err`` (level-free MAE). ``unit="snapshot"``
    averages over snapshots instead."""
    per_unit = scored.groupby([unit, "system", "stat"])[value].mean()
    table = per_unit.groupby(level=["system", "stat"]).mean().unstack("stat")
    order = [s for s in dict.fromkeys(scored["system"]) if s in table.index]
    return table.loc[order, list(TARGETS)]


def to_markdown(df: pd.DataFrame, digits: int = 2) -> str:
    cols = list(df.columns)
    lines = ["| | " + " | ".join(cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, row in df.iterrows():
        cells = [
            str(int(v)) if c == "n" else f"{v:.{digits}f}" if isinstance(v, float) else str(v)
            for c, v in row.items()
        ]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def sure_text(value: float) -> str:
    """``evaluate.sure`` in words: "96% sure better (real)", "90% sure worse", "no lean"."""
    if pd.isna(value):
        return "n/a"
    if value == 0:
        return "no lean"
    real = " (real)" if abs(value) >= SURE_BAR else ""
    return f"{abs(value):.0%} sure {'better' if value > 0 else 'worse'}{real}"


def _interval_line(b: pd.DataFrame | None, better: str) -> str:
    """One line of ours-minus-blend intervals per stat, each with how sure ours is better
    or worse; ``better`` names the good sign."""
    if b is None:
        return ""
    cells = [
        f"{s} {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}] {sure_text(r['sure'])}"
        for s, r in b.iterrows()
    ]
    return f"ours - fg_blend ({better} = ours better), 95% interval, how sure: " + "; ".join(cells)


def _unit_block(g: pd.DataFrame) -> list[str]:
    """One season's or snapshot's tables: gap-weighted pairwise accuracy (the main
    score), plain pairwise accuracy, raw MAE and level-free MAE, each vs. the blend."""
    per_unit = order_scores(g)
    blend = BLEND in set(g["system"])

    def pairwise_line(weighted: bool) -> str:
        b = pairwise_bootstrap(g, OURS, BLEND, weighted=weighted) if blend else None
        return _interval_line(b, "positive")

    def mae_line(value: str) -> str:
        return _interval_line(
            paired_bootstrap(g, OURS, BLEND, value=value) if blend else None, "negative"
        )

    return [
        "",
        "Gap-weighted pairwise accuracy (%) -- main score:",
        "",
        to_markdown(order_table(g, "pairwise_w", per_unit), digits=1),
        "",
        pairwise_line(weighted=True),
        "",
        "Pairwise accuracy (%):",
        "",
        to_markdown(order_table(g, "pairwise", per_unit), digits=1),
        "",
        pairwise_line(weighted=False),
        "",
        "Raw MAE:",
        "",
        to_markdown(mae_table(g)),
        "",
        mae_line("abs_err"),
        "",
        "Level-free MAE:",
        "",
        to_markdown(mae_table(g, "lf_err")),
        "",
        mae_line("lf_err"),
    ]


def _mean_blocks(frame: pd.DataFrame, unit: str) -> list[str]:
    """Every score averaged over the seasons or snapshots (``unit``) in ``frame``."""
    over = f"{unit}s"
    per_unit = order_scores(frame)
    # Players redrawn once across every season or snapshot (pairwise_bootstrap), so
    # overlapping snapshots don't count as independent evidence.
    luck = (
        [_interval_line(pairwise_bootstrap(frame, OURS, BLEND), "positive"), ""]
        if {OURS, BLEND} <= set(frame["system"])
        else []
    )
    return [
        "",
        f"Gap-weighted pairwise accuracy (%) -- main score, mean over {over}:",
        "",
        to_markdown(order_table(frame, "pairwise_w", per_unit), digits=1),
        "",
        *luck,
        f"Pairwise accuracy (%), mean over {over}:",
        "",
        to_markdown(order_table(frame, "pairwise", per_unit), digits=1),
        "",
        f"Raw MAE, mean over {over}:",
        "",
        to_markdown(mean_over_seasons(frame, unit=unit)),
        "",
        f"Level-free MAE, mean over {over}:",
        "",
        to_markdown(mean_over_seasons(frame, "lf_err", unit)),
        "",
        f"Rank correlation (Spearman), mean over {over}:",
        "",
        to_markdown(order_table(frame, "spearman", per_unit), digits=3),
    ]


def _group_blocks(frame: pd.DataFrame, unit: str) -> list[str]:
    """Main score and raw MAE for vets and rookies separately, averaged over the
    seasons or snapshots (``unit``) in ``frame``. Pairs form only inside a group, so
    each group's pairwise score asks "did we order these players right among
    themselves". Empty for a frame scored before the tag existed."""
    if "group" not in frame.columns:
        return []
    md = [
        "",
        f"**Vets vs rookies** (vet = {VET_MIN_CAREER_PA}+ MLB PA when projected: before the "
        f"season, or by the snapshot; pairs only inside a group; mean over {unit}s)",
    ]
    unknown = frame.loc[frame["group"] == "unknown", [unit, "player_id"]].drop_duplicates()
    if len(unknown):
        md += [
            "",
            f"{len(unknown)} player-{unit}s are in neither group: under "
            f"{VET_MIN_CAREER_PA} PA with under {MIN_HISTORY_SEASONS} earlier seasons in the "
            "store, so a vet could look like a rookie.",
        ]
    for group in GROUPS:
        g = frame[frame["group"] == group]
        if g.empty:
            continue
        n = g.drop_duplicates([unit, "player_id"]).groupby(unit).size().mean()
        md += [
            "",
            f"{group.capitalize()}s, {n:.0f} players per {unit} -- gap-weighted pairwise (%):",
            "",
            to_markdown(order_table(g, "pairwise_w"), digits=1),
            "",
            f"{group.capitalize()}s -- raw MAE:",
            "",
            to_markdown(mean_over_seasons(g, unit=unit)),
        ]
    return md


def _relevant_blocks(frame: pd.DataFrame, unit: str) -> list[str]:
    """Every score over only the fantasy-relevant hitters (``RELEVANT_TOP`` by
    FanGraphs' projected or by actual value), averaged over the seasons or snapshots (``unit``) that have the
    tag, plus their young hitters (under ``YOUNG_MAX_CAREER_PA`` MLB PA) on their own.
    Pairs form only inside the set. Empty for a frame scored before the tag existed."""
    if "relevant" not in frame.columns:
        return []
    rel = frame[frame["relevant"].fillna(False).astype(bool)]
    if rel.empty:
        return []
    rel = systems_in_every_season(rel, unit)
    n = rel.drop_duplicates([unit, "player_id"]).groupby(unit).size().mean()
    md = [
        "",
        f"**Fantasy-relevant hitters** (top {RELEVANT_TOP} by FanGraphs' projected value "
        f"or by actual value, in this league's categories; {n:.0f} of them scored per "
        f"{unit}; pairs only among "
        f"them; mean over the {rel[unit].nunique()} {unit}s with FanGraphs files)",
        *_mean_blocks(rel, unit),
    ]
    if "career_pa" in rel.columns:
        young = rel[rel["career_pa"] < YOUNG_MAX_CAREER_PA]
        if not young.empty:
            n = young.drop_duplicates([unit, "player_id"]).groupby(unit).size().mean()
            md += [
                "",
                f"Of them, under {YOUNG_MAX_CAREER_PA} MLB PA when projected, {n:.0f} per "
                f"{unit} -- gap-weighted pairwise (%):",
                "",
                to_markdown(order_table(young, "pairwise_w"), digits=1),
            ]
    return md


def league_forecast_lines(table: pd.DataFrame, seasons: list[int]) -> list[str]:
    """Markdown: the preseason league-rate forecast (``league_reference``: the last
    three seasons, last season for SB; the multiplier that turns a relative projection
    back into rates) vs. the league's actual rates that season, so its error shows on
    its own (#424). Empty for a table built before the league columns (#421): the
    forecast can't be computed there."""
    if not set(ERA_TABLE_COLUMNS) <= set(table.columns):
        return []
    week0 = table[table["season"].isin(seasons) & (table["week"] == 0)]
    # Both are the same for every week-0 row of a season.
    forecast = league_reference(week0).groupby(week0["season"]).first()
    actual = league_answer_rates(week0).groupby(week0["season"]).first()
    err = ((forecast - actual) * pd.Series(SCALE)[list(TARGETS)]).dropna(how="all")
    if err.empty:
        return []
    err.loc["mean abs"] = err.abs().mean()
    return [
        "",
        "#### League-level forecast (last 3 seasons; SB last season) minus actual, preseason",
        "",
        "Same units as MAE; positive = forecast too high. Relative scores ignore this; "
        "raw MAE pays it for every player.",
        "",
        to_markdown(err),
    ]


def summarize(pre: pd.DataFrame | None, snap: pd.DataFrame | None) -> list[str]:
    """Markdown lines: per-season and per-snapshot tables (gap-weighted and plain
    pairwise accuracy, raw and level-free MAE, each with a bootstrap vs. the blend),
    means over seasons and snapshots, and spread."""
    md = [
        "**Main score: gap-weighted pairwise accuracy** -- % of player pairs ordered as "
        "they turned out, each pair counted by how far apart they really finished; higher "
        "is better (50 = coin flip). Plain pairwise counts every pair the same. "
        "Raw error, lower is better: R/HR/RBI/SB = MAE per 600 PA, AVG = MAE in points. "
        "Level-free MAE: the same after scaling each projection so its PA-weighted mean "
        "matches the actuals' (a league-wide miss costs nothing). "
        f"Preseason: players with >= {PRESEASON_MIN_PA} actual PA. "
        f"Mid-season: >= {SNAPSHOT_MIN_PA} PA after the snapshot. "
        "`league_avg` and `marcel` are simple floors (see hitter_ros/baselines.py). "
        "Intervals resample players; each line says which sign means ours is better; "
        'an interval crossing 0 = can\'t tell apart. "N% sure better/worse" means what '
        "it says: how sure ours really differs, and which way; 95%+ is marked real (the "
        "interval clears 0). It counts only the luck of which hitters were scored, not seed or "
        "season swings. Over several seasons or snapshots each hitter is redrawn once "
        "for all of them."
    ]
    if pre is not None:
        md += ["", "#### Preseason"]
        for season, g in pre.groupby("season"):
            md += ["", f"**{season}**", *_unit_block(g)]
        pooled = systems_in_every_season(pre)
        md += ["", "**Mean over seasons** (systems present every season)"]
        md += _mean_blocks(pooled, "season")
        md += _group_blocks(pooled, "season")
        # From ``pre``, not ``pooled``: pooled drops the FanGraphs systems whenever an
        # older season without their files is scored, and the relevant rows are
        # exactly the seasons that have them.
        md += _relevant_blocks(pre, "season")
        fg_seasons = pre.loc[pre["system"] == BLEND, "season"].unique()
        if 0 < len(fg_seasons) < pre["season"].nunique():
            # Older seasons have no FanGraphs files; keep the comparison with them visible.
            fg = systems_in_every_season(pre[pre["season"].isin(fg_seasons)])
            years = ", ".join(str(s) for s in sorted(fg_seasons))
            md += ["", f"**Mean over the seasons with FanGraphs files** ({years})"]
            md += _mean_blocks(fg, "season")
            md += _group_blocks(fg, "season")
        md += [
            "",
            "**Spread of projections** (SD across scored player-seasons; '(actual)' = outcomes)",
            "",
        ]
        md.append(to_markdown(spread(pooled)))
    if snap is not None:
        md += ["", "#### Mid-season (ROS snapshots)"]
        for snapshot, g in snap.groupby("snapshot"):
            md += ["", f"**{snapshot}**", *_unit_block(g)]
        md += ["", "**Mean over snapshots** (systems present in every snapshot)"]
        every = systems_in_every_season(snap, "snapshot")
        md += _mean_blocks(every, "snapshot")
        md += _group_blocks(every, "snapshot")
        md += _relevant_blocks(snap, "snapshot")
    return md
