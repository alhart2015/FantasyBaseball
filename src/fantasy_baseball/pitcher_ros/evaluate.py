"""Score pitcher projections against what happened (#465).

The **main score is MSE**, as for hitters (#433): squared error pooled over every
scored pitcher-unit (a pitcher in one season, or after one snapshot), lower is better.
Each stat is scored as a rate on the volume the pitcher actually had, so a projection
isn't charged for missing on playing time (that is scored on its own):

* K, W, SV: on season totals over the batters he actually faced,
  ``(bf x (projected rate - actual rate))^2`` -- "how many K off, squared".
* ERA, WHIP: ``(projected - actual)^2`` in the stat's own units, weighted by the
  innings he actually pitched, the way hitters' AVG is weighted by AB. So a 10-inning
  reliever counts a tenth as much as a 100-inning one.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from fantasy_baseball import projection_eval
from fantasy_baseball.pitcher_ros.stats import PER_BF, TARGETS
from fantasy_baseball.projection_eval import unit_key
from fantasy_baseball.sgp.player_value import (
    REPLACEMENT_ERA,
    REPLACEMENT_WHIP,
    calculate_counting_sgp,
    calculate_pitching_rate_sgp,
)
from fantasy_baseball.utils.constants import DEFAULT_TEAM_IP, Category

logger = logging.getLogger(__name__)

# A full-season (preseason) projection has its top starters near 180+ IP. A file whose
# biggest projection is under this is a rest-of-season export saved in the wrong place.
PRESEASON_MIN_TOP_IP = 150


def load_fangraphs_pitchers(path: Path) -> pd.DataFrame:
    """One FanGraphs pitcher CSV -> ``TARGETS`` rates plus projected ``bf`` (TBF) and
    ``ip``, indexed by MLBAM id. ERA and WHIP are the file's own."""
    raw = pd.read_csv(path, encoding="utf-8-sig")
    raw = raw.assign(MLBAMID=pd.to_numeric(raw["MLBAMID"], errors="coerce"))
    raw = raw[raw["MLBAMID"].notna()]
    raw = raw.assign(MLBAMID=raw["MLBAMID"].astype(int))
    dupes = raw["MLBAMID"].duplicated(keep="first")
    if dupes.any():
        logger.warning("%s: %d duplicate MLBAM ids, keeping the first", path.name, dupes.sum())
        raw = raw[~dupes]
    bf = raw["TBF"].astype(float).where(raw["TBF"] > 0)
    out = pd.DataFrame(
        {
            "k": raw["SO"] / bf,
            "w": raw["W"] / bf,
            "sv": raw["SV"] / bf,
            "era": raw["ERA"].astype(float),
            "whip": raw["WHIP"].astype(float),
            "bf": raw["TBF"].astype(float),
            "ip": raw["IP"].astype(float),
        }
    )
    out.index = pd.Index(raw["MLBAMID"], name="player_id")
    return out


def load_systems(directory: Path, *, preseason: bool = False) -> dict[str, pd.DataFrame]:
    """Every ``<system>-pitchers*.csv`` in ``directory`` (not its subfolders), by system.

    With ``preseason``, refuse a file that looks like a rest-of-season export (no
    projection reaches ``PRESEASON_MIN_TOP_IP``): scoring one against the full season
    would hand FanGraphs part of it in hindsight. A file with no rows (a failed
    download) or with a stat blank for everyone (some ZiPS rest-of-season exports
    have no SV) is left out, with a warning: every system is scored on the same
    pitchers and stats. Two files for one system (``zips-pitchers.csv`` next to
    ``zips-pitchers-old.csv``) is an error: which one to score is not a guess to make.
    """
    systems = {}
    for path in sorted(directory.glob("*-pitchers*.csv")):
        proj = load_fangraphs_pitchers(path)
        blank = [s for s in TARGETS if proj[s].isna().all()]
        if blank:
            logger.warning("%s: no %s projections in the file, left out", path, "/".join(blank))
            continue
        top_ip = proj["ip"].max()
        if preseason and top_ip < PRESEASON_MIN_TOP_IP:
            raise ValueError(
                f"{path}: top projected IP is {top_ip:.0f}, so this looks like a "
                "rest-of-season export, not a preseason projection"
            )
        name = path.name.split("-pitchers")[0]
        if name in systems:
            raise ValueError(
                f"{directory}: more than one {name}-pitchers*.csv; keep one per system"
            )
        systems[name] = proj
    return systems


def blend(systems: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Plain average of the systems' rates over the pitchers they all cover."""
    return projection_eval.blend(systems, TARGETS)


def fantasy_value(projection: pd.DataFrame, denoms: dict[Category, float]) -> pd.Series:
    """Roto value (SGP) per pitcher: K, W and SV from rate x ``bf``; ERA and WHIP as
    runs and baserunners saved over a replacement pitcher across ``ip`` innings
    (``sgp.player_value``). ``projection``: ``TARGETS`` plus ``bf`` and ``ip``."""
    bf = projection["bf"].astype(float)
    ip = projection["ip"].astype(float)
    value = (
        calculate_counting_sgp(projection["k"] * bf, denoms[Category.K])
        + calculate_counting_sgp(projection["w"] * bf, denoms[Category.W])
        + calculate_counting_sgp(projection["sv"] * bf, denoms[Category.SV])
    )
    # The sgp helpers are plain arithmetic, so they work elementwise on Series too.
    rate = calculate_pitching_rate_sgp(
        player_rate=projection["era"],
        player_ip=ip,
        replacement_rate=REPLACEMENT_ERA,
        sgp_denominator=denoms[Category.ERA],
        team_ip=DEFAULT_TEAM_IP,
        innings_divisor=9,
    ) + calculate_pitching_rate_sgp(
        player_rate=projection["whip"],
        player_ip=ip,
        replacement_rate=REPLACEMENT_WHIP,
        sgp_denominator=denoms[Category.WHIP],
        team_ip=DEFAULT_TEAM_IP,
        innings_divisor=1,
    )
    return pd.Series(value + pd.Series(rate).fillna(0.0), index=projection.index, name="value")


def scored_players(
    projections: dict[str, pd.DataFrame], actual: pd.DataFrame, min_bf: float
) -> pd.DataFrame:
    """One row per (pitcher, system, stat): projected and actual rate, and the actual
    ``bf`` and ``ip`` they're scored on. ``actual``: ``TARGETS`` plus ``bf`` and ``ip``
    over the scored span. Only pitchers with ``bf >= min_bf`` that every projection
    covers are kept, so every system is scored on the same pitchers."""
    players = actual.index[actual["bf"] >= min_bf]
    players = players.intersection(
        projection_eval.common_index([*projections.values(), actual], TARGETS)
    )
    volume = {v: actual.loc[players, v].to_numpy(dtype=float) for v in ("bf", "ip")}
    parts = [
        pd.DataFrame(
            {
                "player_id": players,
                "system": name,
                "stat": s,
                "projected": proj.loc[players, s].to_numpy(dtype=float),
                "actual": actual.loc[players, s].to_numpy(dtype=float),
                **volume,
            }
        )
        for name, proj in projections.items()
        for s in TARGETS
    ]
    return pd.concat(parts, ignore_index=True)


def squared_error(scored: pd.DataFrame, keep: list[str]) -> pd.DataFrame:
    """``scored[keep]`` plus, per row, MSE's numerator ``sq`` and weight ``w`` (module
    doc): K/W/SV ``(bf x err)^2`` with weight 1, ERA/WHIP ``ip x err^2`` with weight
    ``ip``. MSE over any set of rows = sum(sq) / sum(w)."""
    err = (scored["projected"] - scored["actual"]).to_numpy(dtype=float)
    bf = scored["bf"].to_numpy(dtype=float)
    ip = scored["ip"].to_numpy(dtype=float)
    per_bf = scored["stat"].isin(PER_BF).to_numpy()
    return (
        scored[keep]
        .reset_index(drop=True)
        .assign(
            sq=np.where(per_bf, (bf * err) ** 2, ip * err**2),
            w=np.where(per_bf, 1.0, ip),
        )
    )


def mse_table(scored: pd.DataFrame) -> pd.DataFrame:
    """Systems (in first-seen order) x stats, plus ``n`` scored pitcher-units: MSE pooled
    over every scored pitcher-unit."""
    parts = squared_error(scored, ["system", "stat"])
    sums = parts.groupby(["system", "stat"])[["sq", "w"]].sum()
    order = list(dict.fromkeys(scored["system"]))
    table = (sums["sq"] / sums["w"]).unstack("stat").reindex(index=order, columns=list(TARGETS))
    units = scored.drop_duplicates([*unit_key(scored), "system"])
    table["n"] = units.groupby("system").size()
    return table


def mse_bootstrap(
    scored: pd.DataFrame, a: str, b: str, *, n_boot: int = 2000, seed: int = 0
) -> pd.DataFrame:
    """MSE of ``a`` minus ``b`` per stat (negative = ``a`` better), with ``a``'s and
    ``b``'s own MSE, a 95% interval and ``sure`` (how sure ``a``'s MSE is lower, 0-1:
    ``projection_eval.sure``), from resampling pitchers. Paired: both are scored on each draw. Each
    draw redraws a pitcher once for every season or snapshot he is in, so overlapping
    snapshots aren't counted as independent evidence (``projection_eval``)."""
    key = unit_key(scored)
    parts = squared_error(scored[scored["system"].isin([a, b])], [*key, "system", "stat"])
    return projection_eval.mse_bootstrap(parts, key, a, b, TARGETS, n_boot=n_boot, seed=seed)
