"""Player-specific outcome ranges for a season (#505): P10/P50/P90 of each roto stat,
from a player's projected rates and a playing-time range of his own. One engine for
every player type; a :class:`Spec` (``range_specs``) says what his playing time is (PA,
IP), what each stat is counted over and which inputs his PT range reads.

Pure and I/O-free: rows in, quantiles out. ``range_inputs`` builds the rows from the
files; ``scripts/backtest_*_ranges.py`` score the method on held-out seasons and
``scripts/*_ranges.py`` make next season's ranges.

**Where the player-specific part lives.** How far a player beats or misses his rate
projection is not player-specific: the size of his miss one season is uncorrelated
with the next (hitters 2022-2026, |corr| <= 0.08 on every stat; pitchers K, W, ERA,
WHIP |corr| <= 0.07, saves +0.26 from the role, so pitchers' luck is drawn within the
role). Playing time is: a player who lost time last season, or is old, has a lower
floor. So the range is a playing-time model of his own, times rate luck drawn from
everyone of his type.

**Projected playing time is a rank.** A PT source's raw volume isn't comparable across
vintages: Steamer's 2025-26 hitter PA split was flatter (38-42 hitters at 600+ PA vs
69-89 in 2022-24) with the same ~185k league total. So a row carries ``proj_rank``, his
rank in the source's projected volume among its top ``Spec.top_n``, and
:func:`with_rank` maps the rank to the volume that rank actually got in the training
seasons (``rank_<volume>``). The PT target is ``ratio`` = actual / ranked volume.

**PT range** (:class:`QuantilePT`): linear quantile regression on ``Spec.features``,
one fit per level of ``GRID``, of his volume on a saturating scale, ``-log(1 - V / C)``
with ``C`` = ``Spec.headroom`` x the most volume in the training rows. Quantiles carry
through a monotone transform, so each level's volume is ``C (1 - exp(-y))``: it nears
``C`` with ever smaller gains and never passes it. Real PA tops out around 650-680
however good a hitter's situation (rest days); fit straight on PA / ``rank_pa``, a
young, durable, top-ranked hitter was given a median of 700-750 PA, and held out the
real PA came in under it 92-94% of the time (2023-2026, 13 with Steamer PT, 33 with last
season's). :class:`BandPT` is the one-size baseline (quantiles by ranked-volume band)
that the player-specific one has to beat.

**Luck** (:class:`LuckPool`): whole past player-seasons, each a pair -- ``u``, where his
volume landed in his own predicted PT range (0-1, cross-fitted by season so it is the
held-out view), and ``z``, his rate miss per stat (:class:`Stat`). Drawing the pair
together keeps the links between stats and with playing time (hitters who play more
also hit better). Every past season keeps its ``u``, 0 included: those are the injury
tail, and leaving them out raised the hitters' floors (13% of seasons under P10, not
10%). A season under ``Spec.min_rate_volume`` has no usable rate miss and borrows one
from a random season that has.

**One draw** for a player: volume = his PT quantile at ``u`` x ranked volume (capped);
each stat's rate = projected + his miss rescaled to the drawn volume; a counting total
= rate x its trials, a rate stat = the rate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

# Levels the PT range is fit at; draws interpolate between them (flat past the ends).
GRID = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
HISTORY_SEASONS = 3
# QuantileRegressor's L1 penalty, on standardized inputs.
ALPHA = 0.001
# A rate stat's range needs this many draws with enough trials (``Stat.min_trials``);
# with fewer, it is his projected rate alone.
MIN_RATE_DRAWS = 20


@dataclass(frozen=True)
class Stat:
    """One roto stat. ``trials``: what it is counted over -- the spec's volume or one of
    its ``Spec.trials``. ``binomial``: a rate per trial whose luck scales as a coin
    flip (SD sqrt(p(1-p)/n)); else a ratio whose miss shrinks as 1/sqrt(trials) (ERA,
    WHIP). ``counting``: the season total is rate x trials (HR); else the rate is the
    total (AVG). ``numerator``: the actual columns (``a_<name>``) summed over ``trials``,
    times ``scale`` (9 for ERA). A rate stat's draw needs ``min_trials`` trials."""

    name: str
    trials: str
    binomial: bool
    counting: bool
    numerator: tuple[str, ...]
    scale: float = 1.0
    min_trials: float = 0.0

    def __post_init__(self) -> None:
        # A binomial rate is a per-trial probability (drawn in 0-1, its luck a coin
        # flip at the projected rate), so a scale would put the actual rate and the
        # projection in different units.
        if self.binomial and self.scale != 1.0:
            raise ValueError(f"{self.name}: a binomial stat can't be scaled")


@dataclass(frozen=True)
class Spec:
    """A player type for the engine. ``volume``: his playing time (``pa``, ``ip``);
    ``trials``: other counts derived from it, each with its per-volume column on the
    rows (AB from PA via ``ab_pa``). ``features``: the PT range's inputs (the first is
    ``rank_<volume>``); ``never_blank``: those with no missing-value flag.
    ``history_columns``: :func:`history`'s per-season columns, the first marking that
    he played. ``top_n``: players ranked per season. ``bands``: :class:`BandPT`'s
    ranked-volume bands. ``headroom``: :class:`QuantilePT`'s ceiling over the most
    volume. ``min_rate_volume``: a past season's volume for its rate miss to count.
    ``player_specific``: whether its ranges use :class:`QuantilePT` (else the one-size
    :class:`BandPT`, where his history added nothing held out); :func:`pt_model`.
    ``role``: the job within the type (pitchers: ``starter``, ``reliever``).
    ``luck_by`` / ``luck_bands``: draw a player's luck only from past seasons in his band
    of that projected rate, the bands cut at those quantiles of the past seasons' own
    (relievers by saves: a setup man's lucky saves replayed on a closer gave ceilings
    above any real season)."""

    name: str
    volume: str
    trials: Mapping[str, str]
    stats: tuple[Stat, ...]
    features: tuple[str, ...]
    never_blank: tuple[str, ...]
    history_columns: tuple[str, ...]
    top_n: int
    bands: tuple[float, ...]
    headroom: float
    min_rate_volume: float
    player_specific: bool
    role: str | None = None
    luck_by: str | None = None
    luck_bands: tuple[float, ...] = ()

    @property
    def rank_col(self) -> str:
        return f"rank_{self.volume}"

    @property
    def stat_names(self) -> tuple[str, ...]:
        return tuple(s.name for s in self.stats)

    @property
    def actuals(self) -> tuple[str, ...]:
        cols = [self.volume, *self.trials]
        for s in self.stats:
            cols += [c for c in s.numerator if c not in cols]
        return tuple(f"a_{c}" for c in cols)


def top_ranked(proj: pd.Series, top_n: int) -> pd.Series:
    """Rank (1 = most projected volume) of the ``top_n`` players in ``proj``; ties go
    to the lower id, so a rerun ranks the same."""
    # Only a positive projection ranks: 0 projected volume has no per-volume rates.
    order = proj[proj > 0].sort_index().sort_values(ascending=False, kind="stable")
    top = order.head(top_n)
    return pd.Series(np.arange(1, len(top) + 1), index=top.index, name="proj_rank")


def volume_by_rank(actual: Mapping[int, np.ndarray], seasons: Iterable[int], n: int) -> np.ndarray:
    """The volume the n-th most-used player got, averaged over ``seasons``: each
    season's volume sorted high to low (padded with 0 past its last player), first
    ``n``."""
    sorted_volume = []
    for season in seasons:
        v = np.sort(np.asarray(actual[season], dtype=float))[::-1][:n]
        sorted_volume.append(np.pad(v, (0, n - len(v))))
    if not sorted_volume:
        raise ValueError("no seasons to map ranks from")
    return np.mean(sorted_volume, axis=0)


def with_rank(rows: pd.DataFrame, by_rank: np.ndarray, spec: Spec) -> pd.DataFrame:
    """``rows`` with the ranked volume (what his ``proj_rank`` got, ``by_rank``) and,
    where the actual volume is known, ``ratio`` = actual / ranked volume."""
    rank = rows["proj_rank"].to_numpy(dtype=int)
    if rank.min() < 1 or rank.max() > len(by_rank):
        raise ValueError(f"proj_rank must be 1-{len(by_rank)}")
    ranked = by_rank[rank - 1]
    if (ranked <= 0).any():
        raise ValueError(
            f"a rank mapped to 0 {spec.volume}; map from seasons with that many players"
        )
    out = rows.assign(**{spec.rank_col: ranked})
    if f"a_{spec.volume}" in rows:
        out["ratio"] = rows[f"a_{spec.volume}"].to_numpy(dtype=float) / ranked
    return out


def history(
    ids: pd.Index,
    season: int,
    lines: Mapping[int, pd.DataFrame],
    debut_year: pd.Series,
    columns: Sequence[str] = ("g", "pa"),
) -> pd.DataFrame:
    """``columns`` (per scheduled game) in each of the ``HISTORY_SEASONS`` before
    ``season`` (``<col>1`` the season before), per id. ``lines[y]``: those columns, by
    id, for the players who played in ``y``. Not in a season's lines: 0 if he had
    debuted by then (``debut_year``), else NaN -- "not in the majors yet" is not "played
    0"."""
    debut = debut_year.reindex(ids)
    out = {}
    for k in range(1, HISTORY_SEASONS + 1):
        year = season - k
        if year not in lines:
            raise ValueError(f"no season lines for {year}")
        line = lines[year].reindex(ids)
        known = line[columns[0]].notna() | (debut <= year)
        for c in columns:
            out[f"{c}{k}"] = line[c].fillna(0.0).where(known)
    return pd.DataFrame(out, index=ids)


class PTModel(Protocol):
    def fit(self, rows: pd.DataFrame) -> PTModel: ...

    def predict(self, rows: pd.DataFrame) -> np.ndarray: ...


def _ordered(q: np.ndarray) -> np.ndarray:
    """Quantiles along ``GRID`` made non-decreasing and >= 0."""
    return np.sort(np.clip(q, 0.0, None), axis=1)


class QuantilePT:
    """The player-specific PT range: per level of ``GRID``, a linear quantile regression
    of his volume on the saturating scale (module docstring) on ``spec.features``,
    returned as ``ratio`` quantiles (volume / ranked volume). A missing input is filled
    with the training median and flagged (its own input), standardized on the training
    rows."""

    def __init__(self, spec: Spec, alpha: float = ALPHA, headroom: float | None = None) -> None:
        headroom = spec.headroom if headroom is None else headroom
        if headroom <= 1:
            raise ValueError("headroom must be above 1 (the ceiling sits above every volume)")
        self.spec = spec
        self.alpha = alpha
        self.headroom = headroom

    def _scale(self, volume: np.ndarray) -> np.ndarray:
        return np.asarray(-np.log1p(-volume / self.ceiling))

    def _unscale(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(self.ceiling * -np.expm1(-np.clip(y, 0.0, None)))

    def _raw(self, rows: pd.DataFrame) -> pd.DataFrame:
        """The inputs, blanks filled with the training median, plus the blank flags."""
        features = list(self.spec.features)
        x = rows[features].astype(float)
        cols = {c: x[c].fillna(self.medians[c]) for c in features}
        cols.update({f"{c}_na": x[c].isna().astype(float) for c in self.flagged})
        return pd.DataFrame(cols, index=rows.index)

    def _design(self, rows: pd.DataFrame) -> pd.DataFrame:
        return (self._raw(rows) - self.mu) / self.sd

    def fit(self, rows: pd.DataFrame) -> QuantilePT:
        self.medians = rows[list(self.spec.features)].astype(float).median()
        # Flag every input that can be missing, not just those missing here, so a
        # prediction row with a blank the training rows never had still has its flag.
        self.flagged = [c for c in self.spec.features if c not in self.spec.never_blank]
        raw = self._raw(rows)
        self.mu, self.sd = raw.mean(), raw.std().replace(0.0, 1.0).fillna(1.0)
        design = (raw - self.mu) / self.sd
        volume = rows[f"a_{self.spec.volume}"].to_numpy(dtype=float)
        self.ceiling = self.headroom * float(volume.max())
        y = self._scale(volume)
        self.models = [
            QuantileRegressor(quantile=float(t), alpha=self.alpha, solver="highs").fit(design, y)
            for t in GRID
        ]
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        design = self._design(rows)
        volume = self._unscale(np.column_stack([m.predict(design) for m in self.models]))
        return _ordered(volume / rows[self.spec.rank_col].to_numpy(dtype=float)[:, None])


class BandPT:
    """The one-size baseline: the training rows' ``ratio`` quantiles within his
    ranked-volume band (``spec.bands``), the same for every player in it."""

    def __init__(self, spec: Spec) -> None:
        self.spec = spec

    def _band(self, rows: pd.DataFrame) -> np.ndarray:
        ranked = rows[self.spec.rank_col].to_numpy(dtype=float)
        return np.asarray(np.digitize(ranked, self.spec.bands[1:-1]))

    def fit(self, rows: pd.DataFrame) -> BandPT:
        band = self._band(rows)
        ratio = rows["ratio"].to_numpy(dtype=float)
        overall = np.quantile(ratio, GRID)
        self.table = np.array(
            [
                np.quantile(ratio[band == b], GRID) if (band == b).any() else overall
                for b in range(len(self.spec.bands) - 1)
            ]
        )
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return _ordered(self.table[self._band(rows)])


def pt_model(spec: Spec) -> PTModel:
    """The PT range ``spec``'s ranges use (``Spec.player_specific``)."""
    return QuantilePT(spec) if spec.player_specific else BandPT(spec)


def coinflip_sd(rate: np.ndarray | float, n: np.ndarray) -> np.ndarray:
    """Binomial SD of a rate over ``n`` trials (at least 1)."""
    p = np.clip(rate, 1e-4, 0.999)
    return np.asarray(np.sqrt(p * (1 - p) / np.maximum(n, 1.0)))


def _trials(rows: pd.DataFrame, stat: Stat) -> np.ndarray:
    return np.asarray(rows[f"a_{stat.trials}"].to_numpy(dtype=float))


def actual_rates(rows: pd.DataFrame, spec: Spec) -> dict[str, np.ndarray]:
    """Each stat's actual rate (NaN with no trials)."""
    out = {}
    for s in spec.stats:
        numerator = sum(rows[f"a_{c}"].to_numpy(dtype=float) for c in s.numerator)
        with np.errstate(divide="ignore", invalid="ignore"):
            out[s.name] = s.scale * np.asarray(numerator) / _trials(rows, s)
    return out


def actual_totals(rows: pd.DataFrame, spec: Spec) -> dict[str, np.ndarray]:
    """Season totals: a counting stat's numerator, a rate stat's rate (NaN with no
    trials)."""
    rates = actual_rates(rows, spec)
    out = {}
    for s in spec.stats:
        if s.counting:
            out[s.name] = s.scale * np.asarray(
                sum(rows[f"a_{c}"].to_numpy(dtype=float) for c in s.numerator)
            )
        else:
            out[s.name] = np.where(_trials(rows, s) > 0, rates[s.name], np.nan)
    return out


def rate_miss(
    stat: Stat, actual: np.ndarray, projected: np.ndarray, trials: np.ndarray
) -> np.ndarray:
    """A season's miss in luck units: coin flips for a binomial stat, x sqrt(trials) for
    a ratio stat."""
    with np.errstate(divide="ignore", invalid="ignore"):
        if stat.binomial:
            return np.asarray((actual - projected) / coinflip_sd(projected, trials))
        return np.asarray((actual - projected) * np.sqrt(trials))


def drawn_rate(stat: Stat, projected: float, miss: np.ndarray, trials: np.ndarray) -> np.ndarray:
    """His rate on each draw: the projection plus a past miss rescaled to the drawn
    trials (a binomial rate stays in 0-1)."""
    if stat.binomial:
        return np.asarray(np.clip(projected + miss * coinflip_sd(projected, trials), 0.0, 1.0))
    return np.asarray(np.clip(projected + miss / np.sqrt(np.maximum(trials, 1e-9)), 0.0, None))


@dataclass(frozen=True)
class LuckPool:
    """Past player-seasons as (``u``, ``z``) pairs; see the module docstring."""

    u: np.ndarray
    z: Mapping[str, np.ndarray]
    # Each past season's projected ``Spec.luck_by`` rate (None without one).
    by: np.ndarray | None = None

    @classmethod
    def from_rows(
        cls, rows: pd.DataFrame, q: np.ndarray, rng: np.random.Generator, spec: Spec
    ) -> LuckPool:
        """``rows``: past seasons with ``ratio``, the actuals and the projected rates
        (``p_<stat>``); ``q``: each row's held-out PT quantiles on ``GRID``."""
        ratio = rows["ratio"].to_numpy(dtype=float)
        u = np.array(
            [np.interp(r, qr, GRID, left=0.0, right=1.0) for r, qr in zip(ratio, q, strict=True)]
        )
        # Several levels can sit at 0; spread a 0-volume season evenly below where 0 lands.
        zero = ratio == 0
        u[zero] = u[zero] * rng.uniform(size=int(zero.sum()))
        usable = rows[f"a_{spec.volume}"].to_numpy(dtype=float) >= spec.min_rate_volume
        if not usable.any():
            raise ValueError(
                f"no past season has {spec.min_rate_volume:g}+ {spec.volume} to read a rate miss from"
            )
        donor = np.arange(len(rows))
        donor[~usable] = rng.choice(np.flatnonzero(usable), int((~usable).sum()))
        act = actual_rates(rows, spec)
        z = {}
        for s in spec.stats:
            p = rows[f"p_{s.name}"].to_numpy(dtype=float)
            z[s.name] = rate_miss(s, act[s.name], p, _trials(rows, s))[donor]
        by = None if spec.luck_by is None else rows[f"p_{spec.luck_by}"].to_numpy(dtype=float)
        return cls(u=u, z=z, by=by)

    def bands(self, spec: Spec, projected: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """(each projected value's band, the pool rows in each band) for
        ``spec.luck_by``; one band of everyone without it."""
        everyone = np.arange(len(self.u))
        if spec.luck_by is None or self.by is None:
            return np.zeros(len(projected), dtype=int), [everyone]
        edges = np.quantile(self.by, spec.luck_bands)
        members = [np.flatnonzero(np.digitize(self.by, edges) == b) for b in range(len(edges) + 1)]
        # A band with no past seasons (tied cut points) draws from everyone.
        members = [m if len(m) else everyone for m in members]
        return np.digitize(projected, edges), members


def simulate(
    rows: pd.DataFrame,
    q: np.ndarray,
    pool: LuckPool,
    levels: Sequence[float],
    rng: np.random.Generator,
    spec: Spec,
    n_draws: int = 2000,
    cap: float = np.inf,
    stat_caps: Mapping[str, float] | None = None,
) -> dict[str, np.ndarray]:
    """Each row's season totals at ``levels`` (rows x levels), by stat, plus the volume.
    ``rows``: the ranked volume, ``spec.trials``' per-volume columns and the projected
    rates (``p_<stat>``); ``q``: the rows' PT quantiles on ``GRID``. ``cap``: no draw
    plays more (QuantilePT's saturating ceiling sits at ``headroom`` x the most volume,
    and BandPT's ratio quantiles have no ceiling, so either can run past what anyone
    has had). ``stat_caps``: no counting total beats these (the best season in the
    training rows: a closer's saves are bounded by his chances, which aren't drawn)."""
    stat_caps = {} if stat_caps is None else stat_caps
    n_rows = len(rows)
    out = {c: np.empty((n_rows, len(levels))) for c in (*spec.stat_names, spec.volume)}
    ranked = rows[spec.rank_col].to_numpy(dtype=float)
    per_volume = {t: rows[col].to_numpy(dtype=float) for t, col in spec.trials.items()}
    proj = {s.name: rows[f"p_{s.name}"].to_numpy(dtype=float) for s in spec.stats}
    band, members = pool.bands(spec, proj[spec.luck_by] if spec.luck_by else np.zeros(n_rows))
    for i in range(n_rows):
        near = members[band[i]]
        j = near[rng.integers(0, len(near), n_draws)]
        volume = np.minimum(np.interp(pool.u[j], GRID, q[i]) * ranked[i], cap)
        trials = {spec.volume: volume, **{t: volume * per_volume[t][i] for t in spec.trials}}
        out[spec.volume][i] = np.quantile(volume, levels)
        for s in spec.stats:
            n = trials[s.trials]
            rate = drawn_rate(s, proj[s.name][i], pool.z[s.name][j], n)
            if s.counting:
                # The rate carries the scale (actual_rates), so the total is rate x n.
                total = np.minimum(rate * n, stat_caps.get(s.name, np.inf))
            else:
                enough = n >= s.min_trials
                total = (
                    rate[enough] if enough.sum() >= MIN_RATE_DRAWS else np.array([proj[s.name][i]])
                )
            out[s.name][i] = np.quantile(total, levels)
    return out


@dataclass(frozen=True)
class Ranges:
    """A fitted method: the PT model on every training season, the rank map, the luck
    pool and the volume cap (the most anyone had in a training season).
    :meth:`predict` gives new rows their ranges."""

    spec: Spec
    model: PTModel
    by_rank: np.ndarray
    pool: LuckPool
    cap: float
    stat_caps: Mapping[str, float]

    def predict(
        self,
        rows: pd.DataFrame,
        levels: Sequence[float],
        rng: np.random.Generator,
        n_draws: int = 2000,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """(rows with the ranked volume, their totals at ``levels`` by stat and volume)."""
        check_rows(rows, self.spec, actuals=False)
        ranked = with_rank(rows, self.by_rank, self.spec)
        q = self.model.predict(ranked)
        return ranked, simulate(
            ranked, q, self.pool, levels, rng, self.spec, n_draws, self.cap, self.stat_caps
        )


def check_rows(rows: pd.DataFrame, spec: Spec, *, actuals: bool) -> None:
    need = [
        "season",
        "proj_rank",
        *spec.trials.values(),
        *(f for f in spec.features if f != spec.rank_col),
        *(f"p_{s}" for s in spec.stat_names),
    ]
    if actuals:
        need += list(spec.actuals)
    missing = [c for c in need if c not in rows]
    if missing:
        raise ValueError(f"rows lack {missing}")
    if rows[[f"p_{s}" for s in spec.stat_names]].isna().any().any():
        raise ValueError("every row needs projected rates")
    for col in spec.trials.values():
        v = rows[col].to_numpy(dtype=float)
        bad = ~(np.isfinite(v) & (v > 0))
        if bad.any():
            raise ValueError(f"{int(bad.sum())} rows have no usable {col} (missing, 0 or infinite)")


def fit(
    rows: pd.DataFrame,
    actual_volume: Mapping[int, np.ndarray],
    make_model: Callable[[Spec], PTModel],
    rng: np.random.Generator,
    spec: Spec,
) -> Ranges:
    """Fit on ``rows`` (past seasons, with actuals): the PT model on every season, and
    the luck pool's ``u`` cross-fitted by season (each season's PT quantiles from a fit
    without it). ``actual_volume``: every player's volume by season, for the rank map."""
    check_rows(rows, spec, actuals=True)
    seasons = sorted(rows["season"].unique())
    if len(seasons) < 2:
        raise ValueError("fit needs 2+ seasons (the luck pool is cross-fitted by season)")
    # Every rank the spec ranks, not just the most seen here: a new season's last-ranked
    # player may have had no rates (so no row) in every training season.
    n = max(spec.top_n, int(rows["proj_rank"].max()))
    by_rank = volume_by_rank(actual_volume, seasons, n)
    model = make_model(spec).fit(with_rank(rows, by_rank, spec))
    q = np.empty((len(rows), len(GRID)))
    ratio = np.empty(len(rows))
    for season in seasons:
        others = [s for s in seasons if s != season]
        inner = volume_by_rank(actual_volume, others, n)
        held = (rows["season"] == season).to_numpy()
        fitted = make_model(spec).fit(with_rank(rows[rows["season"].isin(others)], inner, spec))
        ranked = with_rank(rows[held], inner, spec)
        q[held] = fitted.predict(ranked)
        ratio[held] = ranked["ratio"].to_numpy()
    pool = LuckPool.from_rows(rows.assign(ratio=ratio), q, rng, spec)
    # The most anyone had: in the rank map's seasons, or in these rows (a projected
    # reliever who moved to the rotation is in the reliever rows, not the reliever IP).
    cap = max(
        max(float(np.max(actual_volume[season])) for season in seasons),
        float(rows[f"a_{spec.volume}"].max()),
    )
    totals = actual_totals(rows, spec)
    stat_caps = {st.name: float(np.nanmax(totals[st.name])) for st in spec.stats if st.counting}
    return Ranges(spec=spec, model=model, by_rank=by_rank, pool=pool, cap=cap, stat_caps=stat_caps)


def backtest(
    rows: pd.DataFrame,
    actual_volume: Mapping[int, np.ndarray],
    make_model: Callable[[Spec], PTModel],
    rng: np.random.Generator,
    spec: Spec,
    levels: Sequence[float] = (0.1, 0.9),
    n_draws: int = 2000,
) -> pd.DataFrame:
    """Leave one season out: fit on the others, predict it. One row per (player,
    season, stat) -- the volume included -- with the actual and the predicted totals at
    ``levels`` (columns ``q<level x 100>``), the ranked volume (``ranked``) and each
    stat's actual trials (``trials``)."""
    parts = []
    for season in sorted(rows["season"].unique()):
        test = rows[rows["season"] == season]
        ranges = fit(rows[rows["season"] != season], actual_volume, make_model, rng, spec)
        ranked, sims = ranges.predict(test, levels, rng, n_draws)
        actual = actual_totals(test, spec)
        actual[spec.volume] = test[f"a_{spec.volume}"].to_numpy(dtype=float)
        trials = {s.name: s.trials for s in spec.stats} | {spec.volume: spec.volume}
        for c in (*spec.stat_names, spec.volume):
            frame = pd.DataFrame(
                {
                    "player_id": test.index.to_numpy(),
                    "season": season,
                    "stat": c,
                    "ranked": ranked[spec.rank_col].to_numpy(),
                    "actual": actual[c],
                    "trials": test[f"a_{trials[c]}"].to_numpy(dtype=float),
                }
            )
            for k, level in enumerate(levels):
                frame[f"q{round(level * 100)}"] = sims[c][:, k]
            parts.append(frame)
    return pd.concat(parts, ignore_index=True)


def pinball(actual: np.ndarray, predicted: np.ndarray, level: float) -> np.ndarray:
    """Quantile (pinball) loss of a prediction at ``level``."""
    miss = actual - predicted
    return np.asarray(np.maximum(level * miss, (level - 1) * miss))
