"""Historical per-player-per-season hitter panel for the playing-time model (#291).

Pure and I/O-free: takes already-fetched raw MLB Stats API frames, returns the
training panel. Fetching/caching is `keepers.mlb_stats`; the CLI is
`scripts/build_pt_panel.py`.

Three representation decisions drive everything here, because each is a way the
panel could silently teach the model something false.

**Absence is NaN, never 0.** The season leaderboard returns a row only for players
who recorded a plate appearance. A player missing from 2015 was injured, in AAA, in
another organization, or retired -- and `pa = 0` would read downstream as an
observed season of zero playing time, which is a different claim than "we have no
observation". `observed` marks which rows are real; every stat column is NaN on the
rest. Resolving *why* a player was absent needs the MiLB pull or the transactions
feed, both deliberately out of scope for #291.

**Rows span first-observed to last-observed season, not the whole window.** Interior
gaps are the signal (a lost season between two played ones); trailing absence is
just a career ending, and emitting NaN rows through the panel's last year for every
player who retired in 2011 would bury the real gaps under an order of magnitude more
noise. The cost is that a player who missed ALL of the final season gets no row for
it -- fine for training, but prediction-time feature building must not assume the
panel carries a row for every active player. See `build_hitter_panel`.

**Age is derived, not read.** The API's `stat.age` is the player's age on June 30 of
that season (verified: 1000/1000 exact for 2015). Unobserved rows have no stat block
to read an age from, so age is computed from the birth date on the same June 30
convention for every row -- which keeps the column consistent rather than mixing a
read value with a derived one. `test_age_uses_the_june_30_convention` pins the rule.

Two structural breaks in the window are carried as columns rather than silently
absorbed, because each would otherwise read as a playing-time signal:

* **`is_pitcher`.** The hitting leaderboard includes pitchers who took a plate
  appearance -- 537 of 2010's 1157 rows, at a median of 1 PA. They vanish with the
  universal DH (2020, then permanently from 2022), so a raw row count halves between
  2021 and 2022 with nothing having happened to real hitters. Non-pitcher counts are
  flat at 577-687 across the whole window. The rows are kept, not dropped, because
  dropping on primary position would also drop a two-way player's hitting season;
  filtering is the model's call.
* **`scheduled_games`.** 2020 was a 60-game season. Every PA that year is ~37% of a
  normal one, which a model without this column reads as a league-wide injury.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping

import numpy as np
import pandas as pd

from fantasy_baseball.keepers.actuals import (
    HITTER_RATES,
    coerce_numeric,
    index_by_mlbam,
    normalize_hitting,
)

# The panel's column order, which is also its schema. Consumers select by name, but
# a stable order keeps the on-disk CSV diffable across rebuilds.
HITTER_PANEL_COLUMNS: tuple[str, ...] = (
    "mlbam_id",
    "season",
    "observed",
    "partial_season",
    "scheduled_games",
    "pa",
    "games",
    "age",
    "primary_position",
    "is_pitcher",
    "birth_date",
    "debut_date",
    "seasons_since_debut",
    "history_truncated",
    *HITTER_RATES,
)

# Age reference date. MLB's convention, and what `stat.age` reports.
_AGE_MONTH = 6
_AGE_DAY = 30

# Scheduled games per team, where the season was not the standard 162.
_FULL_SEASON_GAMES = 162
_SHORTENED_SEASONS = {2020: 60}  # COVID


def _age_on_june_30(birth: pd.Series, season: pd.Series) -> pd.Series:
    """Age in completed years as of June 30 of `season`.

    Vectorized `season - birth.year - (birthday not yet reached)`. Done on integer
    parts rather than a timedelta because a day-count division by 365.25 rounds the
    wrong way for players born within a day or two of the reference date.
    """
    birth_dt = pd.to_datetime(birth, errors="coerce")
    not_yet = (birth_dt.dt.month > _AGE_MONTH) | (
        (birth_dt.dt.month == _AGE_MONTH) & (birth_dt.dt.day > _AGE_DAY)
    )
    age = season.to_numpy() - birth_dt.dt.year - not_yet.astype("float64")
    return pd.Series(np.where(birth_dt.isna(), np.nan, age), index=birth.index)


def _observed_season(raw: pd.DataFrame, year: int) -> pd.DataFrame:
    """One raw season frame -> observed rows: pa, games, and the rate line."""
    rates = normalize_hitting(raw)
    games = index_by_mlbam(raw, "player.id")["stat.gamesPlayed"].map(coerce_numeric)
    duplicated = rates.index[rates.index.duplicated()]
    if len(duplicated) > 0:
        # The leaderboard is one row per player-season (multi-team seasons arrive
        # pre-aggregated). A duplicate means that stopped being true, and silently
        # keeping both would double a player's weight in every fit downstream.
        raise ValueError(f"{year}: duplicate mlbam ids in season frame: {sorted(set(duplicated))}")
    out = rates.join(games.rename("games"))
    out.insert(0, "season", year)
    return out.reset_index()


def _career_span_index(observed: pd.DataFrame) -> pd.MultiIndex:
    """Every (player, season) from each player's first to last OBSERVED season.

    Not the full panel window -- see the module docstring on trailing absence.
    """
    bounds = observed.groupby("mlbam_id")["season"].agg(["min", "max"])
    pairs = [
        (pid, season)
        for pid, (lo, hi) in bounds.iterrows()
        for season in range(int(lo), int(hi) + 1)
    ]
    return pd.MultiIndex.from_tuples(pairs, names=["mlbam_id", "season"])


def _people_frame(people: pd.DataFrame) -> pd.DataFrame:
    """Raw `people` response -> the covariates the panel carries, by mlbam id."""
    if people.empty:
        return pd.DataFrame(
            columns=["primary_position", "birth_date", "debut_date"],
            index=pd.Index([], name="mlbam_id", dtype="int64"),
        )
    indexed = index_by_mlbam(people, "id")
    # `primaryPosition.abbreviation` is absent entirely if no player in the response
    # carried one; reindex rather than subscript so that is NaN, not a KeyError.
    cols = {
        "primary_position": "primaryPosition.abbreviation",
        "birth_date": "birthDate",
        "debut_date": "mlbDebutDate",
    }
    out = pd.DataFrame(index=indexed.index)
    for name, source in cols.items():
        out[name] = indexed[source] if source in indexed.columns else np.nan
    return out.loc[~out.index.duplicated()]


def build_hitter_panel(
    seasons: Mapping[int, pd.DataFrame],
    people: pd.DataFrame,
    *,
    partial_seasons: Collection[int] = (),
) -> pd.DataFrame:
    """Assemble the hitter playing-time panel.

    Args:
        seasons: season year -> raw MLB Stats API hitting frame for that year, as
            `keepers.mlb_stats.fetch_mlb_season(..., "hitting")` returns it.
        people: raw `keepers.mlb_stats.fetch_mlb_people` frame covering every player
            id appearing in `seasons`. Players it omits keep NaN covariates.
        partial_seasons: years that were still in progress when fetched. Flagged, not
            dropped -- a live season is a legitimate feature source but must never be
            trained on as a completed one.

    Returns:
        One row per (mlbam_id, season) over each player's observed career span,
        ordered by id then season, with `HITTER_PANEL_COLUMNS`. Unobserved seasons
        carry NaN for every stat column and `observed = False`.
    """
    if not seasons:
        raise ValueError("no season frames supplied; cannot build a panel")
    unknown_partial = sorted(set(partial_seasons) - set(seasons))
    if unknown_partial:
        # Otherwise a typo'd year silently flags nothing, and a live season gets
        # trained on as complete.
        raise ValueError(f"partial_seasons not present in seasons: {unknown_partial}")

    observed = pd.concat(
        [_observed_season(raw, year) for year, raw in sorted(seasons.items())],
        ignore_index=True,
    )
    full = (
        observed.set_index(["mlbam_id", "season"])
        .reindex(_career_span_index(observed))
        .sort_index()
    )
    panel = full.reset_index()
    # `pa` is NaN exactly on the reindexed-in rows, which is what defines `observed`.
    panel["observed"] = panel["pa"].notna()
    panel["partial_season"] = panel["season"].isin(set(partial_seasons))
    panel["scheduled_games"] = (
        panel["season"].map(_SHORTENED_SEASONS).fillna(_FULL_SEASON_GAMES).astype(int)
    )

    covariates = _people_frame(people)
    panel = panel.join(covariates, on="mlbam_id")
    panel["age"] = _age_on_june_30(panel["birth_date"], panel["season"])
    # NaN position (player absent from `people`) stays NaN rather than collapsing to
    # False, which would assert "not a pitcher" about a player we know nothing about.
    panel["is_pitcher"] = panel["primary_position"].map(
        lambda p: np.nan if pd.isna(p) else p == "P"
    )

    debut_year = pd.to_datetime(panel["debut_date"], errors="coerce").dt.year
    panel["seasons_since_debut"] = panel["season"] - debut_year
    # A career that began before the window has real prior seasons the panel cannot
    # see, so its earliest rows have missing-not-absent lags. Modeling must be able
    # to tell those apart from a true rookie season.
    panel["history_truncated"] = debut_year < min(seasons)

    return panel.loc[:, list(HITTER_PANEL_COLUMNS)]
