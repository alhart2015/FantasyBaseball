"""Player-specific outcome ranges for a hitter's season (#505): P10/P50/P90 of R, HR,
RBI, SB and AVG, from his projected rates and a playing-time range of his own.

Pure and I/O-free: rows in, quantiles out. ``range_inputs`` builds the rows from the
files; ``scripts/backtest_hitter_ranges.py`` scores the method on held-out seasons
and ``scripts/hitter_ranges.py`` makes next season's ranges.

**Where the player-specific part lives.** How far a hitter beats or misses his rate
projection is not player-specific: the size of his miss one season is uncorrelated
with the next (2022-2026, |corr| <= 0.08 on every stat). Playing time is: a hitter
who lost time last season, or is old, has a lower floor. So the range is a playing-
time model of his own, times rate luck drawn from everyone.

**Projected playing time is a rank.** The PT source's raw PA isn't comparable across
vintages: Steamer's 2025-26 split was flatter (38-42 hitters at 600+ PA vs 69-89 in
2022-24) with the same ~185k league total. So a row carries ``proj_rank``, his rank in
the source's projected PA among its top ``TOP_N``, and :func:`with_rank_pa` maps the
rank to the PA that rank actually got in the training seasons (``rank_pa``). The PT
target is ``ratio`` = actual PA / ``rank_pa``.

**PT range** (:class:`QuantilePT`): linear quantile regression on ``FEATURES``, one
fit per level of ``GRID``, of his PA on a saturating scale, ``-log(1 - PA / C)`` with
``C`` = ``HEADROOM`` x the most PA in the training rows. Quantiles carry through a
monotone transform, so each level's PA is ``C (1 - exp(-y))``: it nears ``C`` with
ever smaller gains and never passes it. Real PA tops out around 650-680 however good a
hitter's situation (rest days); fit straight on PA / ``rank_pa``, a young, durable,
top-ranked hitter was given a median of 700-750 PA, and held out the real PA came in
under it 86-100% of the time (2023-2026, 14 with Steamer PT, 33 with last season's).
:class:`BandPT` is the one-size baseline (quantiles by ``rank_pa`` band) that the
player-specific one has to beat.

**Luck** (:class:`LuckPool`): whole past player-seasons, each a pair -- ``u``, where his
PA landed in his own predicted PT range (0-1, cross-fitted by season so it is the
held-out view), and ``z``, his rate miss per stat in coin-flip units at his actual PA
(AB for AVG). Drawing the pair together keeps the links between stats and with playing
time (hitters who play more also hit better). Every past season keeps its ``u``, 0 PA
included: those are the injury tail, and leaving them out raised the floors (13% of
seasons under P10, not 10%). A season under ``MIN_RATE_PA`` PA has no usable rate miss
and borrows one from a random season that has.

**One draw** for a player: PA = his PT quantile at ``u`` x ``rank_pa``; rate = projected
+ ``z`` x coin-flip SD at his rate and drawn PA; counting total = rate x PA, AVG = rate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor

STATS = ("r", "hr", "rbi", "sb", "avg")
COUNTING = ("r", "hr", "rbi", "sb")
# Levels the PT range is fit at; draws interpolate between them (flat past the ends).
GRID = np.array([0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98])
# Hitters ranked per season by the PT source: Steamer's 150+ PA projections were about
# 390 a season in 2022-2026, the population the method was validated on.
TOP_N = 390
HISTORY_SEASONS = 3
FEATURES = (
    "rank_pa",
    "age",
    "catcher",
    "dh",
    *(f"{c}{k}" for k in range(1, HISTORY_SEASONS + 1) for c in ("g", "pa")),
)
# A past season needs this many PA for its rate miss to join the luck pool.
MIN_RATE_PA = 50
# An AVG draw needs this many AB; with fewer than MIN_AVG_DRAWS such draws, his AVG
# range is his projected AVG alone.
MIN_AVG_AB = 30
MIN_AVG_DRAWS = 20
# QuantileRegressor's L1 penalty, on standardized inputs.
ALPHA = 0.001
# The saturating PT scale's ceiling, over the most PA in the training rows. Held out
# 2023-2026 (PA pinball loss, P10 / P50 / P90; predictions of a 700+ median and the real
# PA's share under them): straight 28.6 / 59.6 / 23.0, 14 at 86% (Steamer PT) and 28.8 /
# 64.4 / 25.4, 33 at 100% (last season's PA); 2.0: 28.6 / 59.6 / 21.7, 2 and 28.9 / 63.4
# / 23.7, 7. 1.02-1.5 squeezed the 620-660 PA medians low (37-46% under); 3.0 scored as
# 2.0 but let 700+ medians back (4 and 13, 85-100% under).
HEADROOM = 2.0
ACTUALS = ("a_pa", "a_ab", "a_h", "a_r", "a_hr", "a_rbi", "a_sb")


def top_ranked(proj_pa: pd.Series, top_n: int = TOP_N) -> pd.Series:
    """Rank (1 = most projected PA) of the ``top_n`` hitters in ``proj_pa``; ties go to
    the lower id, so a rerun ranks the same."""
    order = proj_pa.dropna().sort_index().sort_values(ascending=False, kind="stable")
    top = order.head(top_n)
    return pd.Series(np.arange(1, len(top) + 1), index=top.index, name="proj_rank")


def pa_by_rank(actual_pa: Mapping[int, np.ndarray], seasons: Iterable[int], n: int) -> np.ndarray:
    """The PA the n-th most-used hitter got, averaged over ``seasons``: each season's PA
    sorted high to low (padded with 0 past its last hitter), first ``n``."""
    sorted_pa = []
    for season in seasons:
        pa = np.sort(np.asarray(actual_pa[season], dtype=float))[::-1][:n]
        sorted_pa.append(np.pad(pa, (0, n - len(pa))))
    if not sorted_pa:
        raise ValueError("no seasons to map ranks from")
    return np.mean(sorted_pa, axis=0)


def with_rank_pa(rows: pd.DataFrame, by_rank: np.ndarray) -> pd.DataFrame:
    """``rows`` with ``rank_pa`` (the PA his ``proj_rank`` got, ``by_rank``) and, where
    the actual PA is known (``a_pa``), ``ratio`` = actual PA / ``rank_pa``."""
    rank = rows["proj_rank"].to_numpy(dtype=int)
    if rank.min() < 1 or rank.max() > len(by_rank):
        raise ValueError(f"proj_rank must be 1-{len(by_rank)}")
    rank_pa = by_rank[rank - 1]
    if (rank_pa <= 0).any():
        raise ValueError("a rank mapped to 0 PA; map from seasons with that many hitters")
    out = rows.assign(rank_pa=rank_pa)
    if "a_pa" in rows:
        out["ratio"] = rows["a_pa"].to_numpy(dtype=float) / rank_pa
    return out


def history(
    ids: pd.Index, season: int, lines: Mapping[int, pd.DataFrame], debut_year: pd.Series
) -> pd.DataFrame:
    """Games share and PA per scheduled game in each of the ``HISTORY_SEASONS`` before
    ``season`` (``g1``/``pa1`` the season before), per id. ``lines[y]``: ``g`` and
    ``pa`` per scheduled game, by id, for the hitters who batted in ``y``. Not in a
    season's lines: 0 if he had debuted by then (``debut_year``), else NaN -- "not in
    the majors yet" is not "played 0"."""
    debut = debut_year.reindex(ids)
    out = {}
    for k in range(1, HISTORY_SEASONS + 1):
        year = season - k
        if year not in lines:
            raise ValueError(f"no season lines for {year}")
        line = lines[year].reindex(ids)
        known = line["g"].notna() | (debut <= year)
        for c in ("g", "pa"):
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
    of his PA on the saturating scale (module docstring) on ``FEATURES``, returned as
    ``ratio`` quantiles (PA / ``rank_pa``). A missing input is filled with the training
    median and flagged (its own input), standardized on the training rows."""

    def __init__(self, alpha: float = ALPHA, headroom: float = HEADROOM) -> None:
        if headroom <= 1:
            raise ValueError("headroom must be above 1 (the ceiling sits above every PA)")
        self.alpha = alpha
        self.headroom = headroom

    def _scale(self, pa: np.ndarray) -> np.ndarray:
        return np.asarray(-np.log1p(-pa / self.ceiling))

    def _unscale(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(self.ceiling * -np.expm1(-np.clip(y, 0.0, None)))

    def _design(self, rows: pd.DataFrame) -> pd.DataFrame:
        x = rows[list(FEATURES)].astype(float)
        cols = {c: x[c].fillna(self.medians[c]) for c in FEATURES}
        cols.update({f"{c}_na": x[c].isna().astype(float) for c in self.flagged})
        return (pd.DataFrame(cols, index=rows.index) - self.mu) / self.sd

    def fit(self, rows: pd.DataFrame) -> QuantilePT:
        x = rows[list(FEATURES)].astype(float)
        self.medians = x.median()
        # Flag every input that can be missing, not just those missing here, so a
        # prediction row with a blank the training rows never had still has its flag.
        self.flagged = [c for c in FEATURES if c not in ("rank_pa", "catcher", "dh")]
        raw = pd.DataFrame(
            {
                **{c: x[c].fillna(self.medians[c]) for c in FEATURES},
                **{f"{c}_na": x[c].isna().astype(float) for c in self.flagged},
            },
            index=rows.index,
        )
        self.mu, self.sd = raw.mean(), raw.std().replace(0.0, 1.0).fillna(1.0)
        design = (raw - self.mu) / self.sd
        pa = rows["a_pa"].to_numpy(dtype=float)
        self.ceiling = self.headroom * float(pa.max())
        y = self._scale(pa)
        self.models = [
            QuantileRegressor(quantile=float(t), alpha=self.alpha, solver="highs").fit(design, y)
            for t in GRID
        ]
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        design = self._design(rows)
        pa = self._unscale(np.column_stack([m.predict(design) for m in self.models]))
        return _ordered(pa / rows["rank_pa"].to_numpy(dtype=float)[:, None])


# rank_pa bands for the one-size baseline.
BANDS = (0.0, 300.0, 450.0, 550.0, np.inf)


class BandPT:
    """The one-size baseline: the training rows' ``ratio`` quantiles within his
    ``rank_pa`` band (``BANDS``), the same for every hitter in it."""

    def _band(self, rows: pd.DataFrame) -> np.ndarray:
        return np.asarray(np.digitize(rows["rank_pa"].to_numpy(dtype=float), BANDS[1:-1]))

    def fit(self, rows: pd.DataFrame) -> BandPT:
        band = self._band(rows)
        ratio = rows["ratio"].to_numpy(dtype=float)
        overall = np.quantile(ratio, GRID)
        self.table = np.array(
            [
                np.quantile(ratio[band == b], GRID) if (band == b).any() else overall
                for b in range(len(BANDS) - 1)
            ]
        )
        return self

    def predict(self, rows: pd.DataFrame) -> np.ndarray:
        return _ordered(self.table[self._band(rows)])


def coinflip_sd(rate: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Binomial SD of a rate over ``n`` trials (at least 1)."""
    p = np.clip(rate, 1e-4, 0.999)
    return np.asarray(np.sqrt(p * (1 - p) / np.maximum(n, 1.0)))


def actual_rates(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    pa = rows["a_pa"].to_numpy(dtype=float)
    ab = rows["a_ab"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out = {c: rows[f"a_{c}"].to_numpy(dtype=float) / pa for c in COUNTING}
        out["avg"] = rows["a_h"].to_numpy(dtype=float) / ab
    return out


def actual_totals(rows: pd.DataFrame) -> dict[str, np.ndarray]:
    """Season totals; AVG is NaN without an at-bat."""
    out = {c: rows[f"a_{c}"].to_numpy(dtype=float) for c in COUNTING}
    ab = rows["a_ab"].to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["avg"] = np.where(ab > 0, rows["a_h"].to_numpy(dtype=float) / ab, np.nan)
    return out


@dataclass(frozen=True)
class LuckPool:
    """Past player-seasons as (``u``, ``z``) pairs; see the module docstring."""

    u: np.ndarray
    z: Mapping[str, np.ndarray]

    @classmethod
    def from_rows(cls, rows: pd.DataFrame, q: np.ndarray, rng: np.random.Generator) -> LuckPool:
        """``rows``: past seasons with ``ratio``, the actuals and the projected rates
        (``p_<stat>``); ``q``: each row's held-out PT quantiles on ``GRID``."""
        ratio = rows["ratio"].to_numpy(dtype=float)
        u = np.array(
            [np.interp(r, qr, GRID, left=0.0, right=1.0) for r, qr in zip(ratio, q, strict=True)]
        )
        # Several levels can sit at 0 PA; spread a 0-PA season evenly below where 0 lands.
        zero = ratio == 0
        u[zero] = u[zero] * rng.uniform(size=int(zero.sum()))
        usable = rows["a_pa"].to_numpy(dtype=float) >= MIN_RATE_PA
        if not usable.any():
            raise ValueError(f"no past season has {MIN_RATE_PA}+ PA to read a rate miss from")
        donor = np.arange(len(rows))
        donor[~usable] = rng.choice(np.flatnonzero(usable), int((~usable).sum()))
        act = actual_rates(rows)
        z = {}
        for c in STATS:
            n = rows["a_ab" if c == "avg" else "a_pa"].to_numpy(dtype=float)
            p = rows[f"p_{c}"].to_numpy(dtype=float)
            with np.errstate(divide="ignore", invalid="ignore"):
                z[c] = ((act[c] - p) / coinflip_sd(p, n))[donor]
        return cls(u=u, z=z)


def simulate(
    rows: pd.DataFrame,
    q: np.ndarray,
    pool: LuckPool,
    levels: Sequence[float],
    rng: np.random.Generator,
    n_draws: int = 2000,
    pa_cap: float = np.inf,
) -> dict[str, np.ndarray]:
    """Each row's season totals at ``levels`` (rows x levels), by stat, plus ``pa``.
    ``rows``: ``rank_pa``, ``ab_pa`` and the projected rates (``p_<stat>``); ``q``: the
    rows' PT quantiles on ``GRID``. ``pa_cap``: no draw plays more (the linear PT range
    can run past what any hitter has had, e.g. 818 PA for a mid-ranked regular)."""
    n_rows = len(rows)
    out = {c: np.empty((n_rows, len(levels))) for c in (*STATS, "pa")}
    rank_pa = rows["rank_pa"].to_numpy(dtype=float)
    ab_pa = rows["ab_pa"].to_numpy(dtype=float)
    proj = {c: rows[f"p_{c}"].to_numpy(dtype=float) for c in STATS}
    for i in range(n_rows):
        j = rng.integers(0, len(pool.u), n_draws)
        pa = np.minimum(np.interp(pool.u[j], GRID, q[i]) * rank_pa[i], pa_cap)
        ab = pa * ab_pa[i]
        out["pa"][i] = np.quantile(pa, levels)
        for c in STATS:
            n = ab if c == "avg" else pa
            rate = np.clip(proj[c][i] + pool.z[c][j] * coinflip_sd(proj[c][i], n), 0.0, 1.0)
            if c == "avg":
                enough = ab >= MIN_AVG_AB
                total = rate[enough] if enough.sum() >= MIN_AVG_DRAWS else np.array([proj[c][i]])
            else:
                total = rate * pa
            out[c][i] = np.quantile(total, levels)
    return out


@dataclass(frozen=True)
class Ranges:
    """A fitted method: the PT model on every training season, the rank map, the luck
    pool and the PA cap (the most any hitter had in a training season).
    :meth:`predict` gives new rows their ranges."""

    model: PTModel
    by_rank: np.ndarray
    pool: LuckPool
    pa_cap: float

    def predict(
        self,
        rows: pd.DataFrame,
        levels: Sequence[float],
        rng: np.random.Generator,
        n_draws: int = 2000,
    ) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
        """(rows with ``rank_pa``, their totals at ``levels`` by stat and ``pa``)."""
        ranked = with_rank_pa(rows, self.by_rank)
        q = self.model.predict(ranked)
        return ranked, simulate(ranked, q, self.pool, levels, rng, n_draws, self.pa_cap)


def check_rows(rows: pd.DataFrame, *, actuals: bool) -> None:
    need = [
        "season",
        "proj_rank",
        "ab_pa",
        *(f for f in FEATURES if f != "rank_pa"),
        *(f"p_{c}" for c in STATS),
    ]
    if actuals:
        need += list(ACTUALS)
    missing = [c for c in need if c not in rows]
    if missing:
        raise ValueError(f"rows lack {missing}")
    if rows[[f"p_{c}" for c in STATS]].isna().any().any():
        raise ValueError("every row needs projected rates")


def fit(
    rows: pd.DataFrame,
    actual_pa: Mapping[int, np.ndarray],
    make_model: Callable[[], PTModel],
    rng: np.random.Generator,
) -> Ranges:
    """Fit on ``rows`` (past seasons, with actuals): the PT model on every season, and
    the luck pool's ``u`` cross-fitted by season (each season's PT quantiles from a fit
    without it). ``actual_pa``: every hitter's PA by season, for the rank map."""
    check_rows(rows, actuals=True)
    seasons = sorted(rows["season"].unique())
    if len(seasons) < 2:
        raise ValueError("fit needs 2+ seasons (the luck pool is cross-fitted by season)")
    n = int(rows["proj_rank"].max())
    by_rank = pa_by_rank(actual_pa, seasons, n)
    model = make_model().fit(with_rank_pa(rows, by_rank))
    q = np.empty((len(rows), len(GRID)))
    ratio = np.empty(len(rows))
    for season in seasons:
        others = [s for s in seasons if s != season]
        inner = pa_by_rank(actual_pa, others, n)
        held = (rows["season"] == season).to_numpy()
        fitted = make_model().fit(with_rank_pa(rows[rows["season"].isin(others)], inner))
        ranked = with_rank_pa(rows[held], inner)
        q[held] = fitted.predict(ranked)
        ratio[held] = ranked["ratio"].to_numpy()
    pool = LuckPool.from_rows(rows.assign(ratio=ratio), q, rng)
    pa_cap = max(float(np.max(actual_pa[season])) for season in seasons)
    return Ranges(model=model, by_rank=by_rank, pool=pool, pa_cap=pa_cap)


def backtest(
    rows: pd.DataFrame,
    actual_pa: Mapping[int, np.ndarray],
    make_model: Callable[[], PTModel],
    rng: np.random.Generator,
    levels: Sequence[float] = (0.1, 0.9),
    n_draws: int = 2000,
) -> pd.DataFrame:
    """Leave one season out: fit on the others, predict it. One row per (player,
    season, stat) -- ``pa`` included -- with the actual and the predicted totals at
    ``levels`` (columns ``q<level x 100>``)."""
    parts = []
    for season in sorted(rows["season"].unique()):
        test = rows[rows["season"] == season]
        ranges = fit(rows[rows["season"] != season], actual_pa, make_model, rng)
        ranked, sims = ranges.predict(test, levels, rng, n_draws)
        actual = actual_totals(test)
        actual["pa"] = test["a_pa"].to_numpy(dtype=float)
        for c in (*STATS, "pa"):
            frame = pd.DataFrame(
                {
                    "player_id": test.index.to_numpy(),
                    "season": season,
                    "stat": c,
                    "rank_pa": ranked["rank_pa"].to_numpy(),
                    "actual": actual[c],
                    "ab": test["a_ab"].to_numpy(dtype=float),
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
