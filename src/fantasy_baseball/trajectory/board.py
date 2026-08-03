"""Every player's query line, for scoring a whole board rather than one name (#311).

`scripts/player_trajectory.py` answers "what about THIS player": it resolves a name,
pace-adjusts his in-progress season, reads his prior year, works out which slot he nets
against, and scores him. A board needs the same five things for everyone at once, and
the per-player helpers cannot simply be looped -- each one scans the full panel, so a
600-player sweep would be 600 passes over 16,000 rows before a single fit runs.

So the same rules are restated here as whole-column operations. They are RULES, not
conveniences, and getting one wrong is silent:

  * a split season is collapsed, or a traded player enters as two half-seasons
  * an in-progress season is PACE-ADJUSTED, or two-thirds of a year is compared against
    full ones
  * a prior season the panel cannot see is 0 only when he was genuinely out of the
    league -- never when the panel simply starts too late
  * a pitcher's SP/RP role comes from a SETTLED season, not an in-progress fragment
  * an unknown slot falls back to UTIL, the HIGHEST floor, so a missing lookup can only
    understate a player

A two-way player appears ONCE PER POOL, as this league scores him: his bat nets against
a hitter's floor and his arm against a pitcher's. Build one board per pool and
concatenate; nothing here tries to merge them.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

from ..utils.name_utils import normalize_name
from .panel import prorate_partial, season_elapsed_fraction
from .value import ROLE_MIN_GAMES, best_floor, resolve_slots


def people(cache_dir: Path) -> pd.DataFrame:
    """Every MLBAM people cache in `cache_dir`, unioned, for id -> name lookup.

    UNIONED rather than ranked, because no single-file rule is stable here. This
    directory is shared with the keeper pipeline, whose `--end` defaults to the current
    year, so any 2027 keeper rebuild drops `mlb_people_all_2010_2027.csv` beside the
    trajectory build's `..._2000_2026.csv`. Ranking on (end, -start) then prefers the
    2027 file and silently loses the 2000-2009 players -- 2060 of them, 1923 present in
    the trajectory panel. A plain string sort is worse still ("2010" > "2000").

    A union has no such failure mode: ids are stable and a name never disagrees between
    caches, so more files can only mean better coverage.
    """
    caches = sorted(cache_dir.glob("mlb_people_all_*.csv"))
    if not caches:
        raise FileNotFoundError(
            f"no people cache in {cache_dir}; run scripts/build_pt_panel.py first"
        )
    frame = pd.concat(
        [pd.read_csv(path, usecols=["id", "fullName"]) for path in caches],
        ignore_index=True,
    ).drop_duplicates(subset="id", keep="last")
    frame["norm"] = frame["fullName"].map(normalize_name)
    return frame


def player_names(cache_dir: Path) -> pd.Series:
    """mlbam_id -> full name, so a board is readable rather than a column of ids."""
    return people(cache_dir).set_index("id")["fullName"]


@lru_cache(maxsize=4)
def season_slots(cache_dir: Path, season: int) -> dict[int, frozenset[str]]:
    """MLBAM id -> slots with 10+ games that season, the league's own eligibility rule.

    READS a cache; never populates one. `fetch_mlb_season` falls through to a live
    paginated download on a miss and writes into the KEEPER pipeline's directory -- a
    standalone tool quietly filling another pipeline's cache, and a network dependency on
    a command that otherwise runs offline. The fielding pull covers only the keeper range
    while this panel spans 2000, so a miss is routine rather than exceptional: it
    degrades to UTIL, the HIGHEST floor, which understates rather than invents.

    Even a present cache is a point-in-time snapshot for the live season, so eligibility
    only grows after it was taken. Conservative in both directions.
    """
    from ..keepers.appearances import season_eligibility

    path = cache_dir / f"mlb_fielding_{season}.csv"
    # NON-EMPTY, not merely present. A header-only file passes `exists()` and then makes
    # `fetch_or_cache` treat it as a miss, reintroducing the cross-pipeline write and the
    # live download this guard exists to stop, invisibly.
    if not path.exists() or path.stat().st_size == 0:
        print(
            f"  NOTE: no cached {season} fielding data, so eligibility is unknown and "
            "every hitter nets against the UTIL floor."
        )
        return {}
    try:
        # Read directly. The shared fetch helper deliberately downloads when a cached
        # frame has headers but no rows, which would break the offline contract. The
        # PARSE is inside the guard too: `season_eligibility` raises KeyError on a
        # schema-shifted file, which is what half of "corrupt" looks like once the CSV
        # itself still parses.
        fielding = pd.read_csv(path)
        if fielding.empty:
            print(f"  NOTE: {season} fielding cache unusable; netting against UTIL.")
            return {}
        eligibility = season_eligibility(fielding)
    except Exception:  # a corrupt cache degrades rather than crashing
        print(f"  NOTE: {season} fielding cache unusable; netting against UTIL.")
        return {}
    return {pid: frozenset(slots) for pid, slots in eligibility.items()}


@dataclass(frozen=True)
class BoardRow:
    """One player's query line: everything a trajectory needs, before it is fitted."""

    mlbam_id: int
    name: str
    pool: str
    age: int
    #: This season's SGP, paced to a full year if the season is still in progress.
    sgp: float
    #: Last season's SGP. A real 0 means he was out of the league, which for a young
    #: player is the normal case rather than missing data.
    prior_sgp: float
    slot: str
    floor: float


#: Columns the board needs that `collapse_split_seasons` does not carry, and how a split
#: season combines them. It aggregates `sgp` and `age` ONLY, so calling it first and then
#: reading `partial_season` raises KeyError, and reading `starts`/`games` off the
#: uncollapsed frame double-counts a traded pitcher.
#:
#: `pa` and `games` are here because `season_elapsed_fraction` dates the season off the
#: busiest player's games, and it REFUSES a frame without `pa` -- so dropping either one
#: turns a hitter panel into something it rejects as a pitcher panel. Collapsing them
#: also matters on its own: a traded everyday player has two rows of ~56 games, and the
#: max over the uncollapsed frame reads the season as 35% elapsed rather than 70%.
_SPLIT_RULES = {
    "partial_season": "any",
    "starts": "sum",
    "games": "sum",
    "pa": "sum",
}


def _collapse(panel: pd.DataFrame) -> pd.DataFrame:
    """One row per (player, season), carrying the columns the board reads.

    NOT `collapse_split_seasons`, which aggregates `sgp` and `age` and drops everything
    else -- fine for the estimators, which need nothing else, and lossy here. The `sgp`
    rule is kept identical to it deliberately (a plain sum) and a test asserts the two
    agree, so this cannot drift into a second, different definition of the same rule.

    The other rules follow from what a mid-season trade means: half a season plus half a
    season is one season, so appearances ADD; and if either half is in progress then the
    combined season is too.
    """
    aggregations: dict[str, tuple[str, str]] = {"age": ("age", "first"), "sgp": ("sgp", "sum")}
    for column, how in _SPLIT_RULES.items():
        if column in panel.columns:
            aggregations[column] = (column, how)
    return panel.groupby(["mlbam_id", "season"], as_index=False).agg(**aggregations)


def _paced(current: pd.DataFrame, calendar: pd.DataFrame, season: int) -> pd.Series:
    """Current-season SGP, prorated where the season is still running.

    `calendar` must be the HITTER panel even when pricing pitchers -- pitcher `games`
    counts appearances rather than team games, so dating the season off it would put the
    league somewhere it is not. Same rule `_resolve_player` follows.
    """
    fraction = season_elapsed_fraction(calendar, season)
    return current.apply(
        lambda r: (
            prorate_partial(float(r["sgp"]), fraction)
            if bool(r["partial_season"])
            else float(r["sgp"])
        ),
        axis=1,
    ).astype(float)


def _pitcher_slots(live: pd.DataFrame, season: int) -> dict[int, set[str]]:
    """SP or RP per pitcher, decided from a SETTLED season.

    A starter back from the IL with two September relief outings is not a reliever, but
    `starts / games` on that fragment says he is -- and the pace adjustment applied to
    his SGP was never applied to this. So the role comes from his most recent season
    clearing `ROLE_MIN_GAMES`, falling back to the latest if he has never cleared it.
    """
    rows = live[live["season"] <= season].sort_values(["mlbam_id", "season"]).copy()
    for column in ("starts", "games"):
        if column not in rows:
            rows[column] = 0.0
    # NaN is TRUTHY, so a null would survive `or 0.0` and poison the comparison; and a
    # missing column would route every starter to the reliever floor. Fill explicitly.
    rows = rows.assign(
        starts=rows["starts"].fillna(0.0).astype(float),
        games=rows["games"].fillna(0.0).astype(float),
    )
    settled = rows[rows["games"] >= ROLE_MIN_GAMES]
    # Last settled season per player, else his latest of any size.
    chosen = pd.concat([rows.groupby("mlbam_id").tail(1), settled.groupby("mlbam_id").tail(1)])
    chosen = chosen.groupby("mlbam_id").tail(1)
    return {
        int(r.mlbam_id): resolve_slots(None, "pitcher", starts=r.starts, games=r.games)
        for r in chosen.itertuples(index=False)
    }


def board_inputs(
    live: pd.DataFrame,
    *,
    kind: str,
    names: pd.Series,
    replacement_levels: dict[str, float],
    eligibility: dict[int, frozenset[str]] | None = None,
    calendar: pd.DataFrame | None = None,
    season: int | None = None,
) -> list[BoardRow]:
    """A query line for every player with a season in `season` (default: the latest).

    `live` is a scored, era-normalized panel INCLUDING partial seasons -- the query needs
    the in-progress year even though the comp pool must not have it.

    `eligibility` maps mlbam id to the slots he has 10+ games at, the league's own rule.
    Pitchers never consult it: `resolve_slots` decides SP/RP from starts. Pass None and
    every hitter falls back to UTIL, the highest floor, which understates rather than
    invents.
    """
    live = _collapse(live)
    if season is None:
        season = int(live["season"].max())

    current = live[live["season"] == season].copy()
    if current.empty:
        return []
    needs_pacing = "partial_season" in current.columns and bool(current["partial_season"].any())
    if calendar is None:
        if kind != "hitter" and needs_pacing:
            # Refuse at the boundary rather than let `season_elapsed_fraction` reject the
            # pitcher panel three frames down. In the pitcher panel `games` counts
            # appearances rather than team games, so dating the season off it comes out
            # near HALF the truth and roughly doubles every pitcher's projected pace.
            raise ValueError(
                "pacing a pitcher board needs the HITTER panel as `calendar`: elapsed "
                "season is a league fact, and pitcher `games` counts appearances rather "
                "than team games played"
            )
        calendar = live
    else:
        calendar = _collapse(calendar)
    current["paced"] = (
        _paced(current, calendar, season) if needs_pacing else current["sgp"].astype(float)
    )

    # The prior year, looked up as a whole column. A player absent from it was out of
    # the league, which is a real 0 -- the same convention the forward path uses.
    prior = live[live["season"] == season - 1].set_index("mlbam_id")["sgp"].astype(float).to_dict()
    slots_by_id = (
        _pitcher_slots(live, season)
        if kind == "pitcher"
        else {
            pid: resolve_slots(set(eligibility.get(pid, frozenset())), kind)
            for pid in current["mlbam_id"].astype(int)
        }
        if eligibility
        else {}
    )

    rows = []
    for r in current.itertuples(index=False):
        pid = int(r.mlbam_id)
        slot, floor = best_floor(slots_by_id.get(pid, set()), replacement_levels)
        rows.append(
            BoardRow(
                mlbam_id=pid,
                name=str(names.get(pid, f"mlbam {pid}")),
                pool=kind,
                age=int(r.age),
                sgp=float(r.paced),
                prior_sgp=float(prior.get(pid, 0.0)),
                slot=slot,
                floor=floor,
            )
        )
    return rows
