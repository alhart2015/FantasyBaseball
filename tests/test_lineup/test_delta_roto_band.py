"""Tests for the analytic deltaRoto confidence band.

The band is closed-form (no Monte Carlo): ``mean`` reuses the EV
deltaRoto so it is identical to the point estimate, and ``sd``
propagates the swapped players' per-category stat variance through each
category's Gaussian roto-points curve. These tests pin the mean-equals-EV
contract, the honest-signal property (noisier categories -> wider band),
determinism, and the crosses-zero verdict mapping.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest

from fantasy_baseball.lineup.delta_roto import (
    DeltaRotoBand,
    compute_delta_roto,
    compute_delta_roto_band,
    compute_one_for_one_band,
)
from fantasy_baseball.models.player import HitterStats, PitcherStats, Player, PlayerType
from fantasy_baseball.models.standings import (
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
)
from fantasy_baseball.scoring import build_team_sds, project_team_stats

FRACTION_REMAINING = 0.6


def _hitter(name: str, **ros: float) -> Player:
    base = dict(pa=600, ab=540, h=150, r=85, hr=25, rbi=85, sb=10)
    base.update(ros)
    stats = HitterStats(**base)
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=["OF"],
        rest_of_season=stats,
        full_season_projection=stats,
    )


def _pitcher(name: str, **ros: float) -> Player:
    base = dict(ip=180, w=12, k=190, sv=0, er=68, bb=48, h_allowed=150)
    base.update(ros)
    stats = PitcherStats(**base)
    return Player(
        name=name,
        player_type=PlayerType.PITCHER,
        positions=["P"],
        rest_of_season=stats,
        full_season_projection=stats,
    )


def _field() -> dict[str, CategoryStats]:
    field: dict[str, CategoryStats] = {}
    for i in range(8):
        field[f"Team{i}"] = CategoryStats(
            r=800 + i * 15,
            hr=220 + i * 5,
            rbi=780 + i * 12,
            sb=110 + i * 4,
            avg=0.255 + i * 0.002,
            w=85,
            k=1300,
            sv=70,
            era=3.8,
            whip=1.20,
        )
    return field


@dataclass
class _Swap:
    """A before/after roster swap with the kwargs both APIs need."""

    band_kwargs: dict[str, Any]
    point_kwargs: dict[str, Any]


def _contested_field() -> dict[str, CategoryStats]:
    """Field where BOTH R and SB sit mid-pack relative to the user.

    The default ``_field`` leaves R uncontested (the user's R total far
    exceeds the field), which pins the R-swap mean at ~0. Centering the
    field's R and SB near the user's totals keeps both categories on the
    steep part of the win-probability curve, so an equal-mean R-vs-SB
    comparison is meaningful for the honest-signal test.
    """
    field: dict[str, CategoryStats] = {}
    for i in range(8):
        field[f"Team{i}"] = CategoryStats(
            r=1040 + i * 20,
            hr=300 + i * 5,
            rbi=1100 + i * 12,
            sb=115 + i * 5,
            avg=0.255 + i * 0.002,
            w=85,
            k=1300,
            sv=70,
            era=3.8,
            whip=1.20,
        )
    return field


def _build_swap(
    before: list[Player],
    drop_name: str,
    add_player: Player,
    field: dict[str, CategoryStats] | None = None,
) -> _Swap:
    """Construct a 1-for-1 swap, wiring the analytic band + EV point kwargs.

    The field is fixed; the user's ("Me") baseline row is the summed
    before-roster projection so ``compute_delta_roto`` and the band see
    the same standings.
    """
    after = [p for p in before if p.name != drop_name] + [add_player]
    if field is None:
        field = _field()
    me_stats = project_team_stats(before)
    entries = [ProjectedStandingsEntry(team_name="Me", stats=me_stats)]
    entries += [ProjectedStandingsEntry(team_name=t, stats=cs) for t, cs in field.items()]
    projected = ProjectedStandings(effective_date=date(2026, 4, 1), entries=entries)

    rosters = {"Me": before, **{t: [] for t in field}}
    team_sds = build_team_sds(rosters, sd_scale=FRACTION_REMAINING**0.5)

    band_kwargs = dict(
        before_players=before,
        after_players=after,
        field_stats=field,
        team_name="Me",
        fraction_remaining=FRACTION_REMAINING,
        projected_standings=projected,
        team_sds=team_sds,
    )
    point_kwargs = dict(
        drop_name=drop_name,
        add_player=add_player,
        user_roster=before,
        projected_standings=projected,
        team_name="Me",
        team_sds=team_sds,
    )
    return _Swap(band_kwargs=band_kwargs, point_kwargs=point_kwargs)


@pytest.fixture
def sample_swap() -> _Swap:
    before = [_hitter(f"H{i}") for i in range(13)]
    add_player = _hitter("BigBat", hr=45, r=105, rbi=110, sb=18)
    return _build_swap(before, "H12", add_player)


@pytest.fixture
def identity_swap() -> _Swap:
    before = [_hitter(f"H{i}") for i in range(13)]
    # Drop H12 and add a fresh copy of H12 -> identical before/after stats.
    add_player = _hitter("H12")
    after = [p for p in before if p.name != "H12"] + [add_player]
    field = _field()
    me_stats = project_team_stats(before)
    entries = [ProjectedStandingsEntry(team_name="Me", stats=me_stats)]
    entries += [ProjectedStandingsEntry(team_name=t, stats=cs) for t, cs in field.items()]
    projected = ProjectedStandings(effective_date=date(2026, 4, 1), entries=entries)
    rosters = {"Me": before, **{t: [] for t in field}}
    team_sds = build_team_sds(rosters, sd_scale=FRACTION_REMAINING**0.5)
    band_kwargs = dict(
        before_players=before,
        after_players=after,
        field_stats=field,
        team_name="Me",
        fraction_remaining=FRACTION_REMAINING,
        projected_standings=projected,
        team_sds=team_sds,
    )
    point_kwargs = dict(
        drop_name="H12",
        add_player=add_player,
        user_roster=before,
        projected_standings=projected,
        team_name="Me",
        team_sds=team_sds,
    )
    return _Swap(band_kwargs=band_kwargs, point_kwargs=point_kwargs)


def test_band_mean_equals_ev_delta(sample_swap: _Swap) -> None:
    band = compute_delta_roto_band(**sample_swap.band_kwargs)
    point = compute_delta_roto(**sample_swap.point_kwargs)
    assert band.mean == pytest.approx(point.total, abs=1e-9)


def test_band_sd_positive_for_real_swap(sample_swap: _Swap) -> None:
    band = compute_delta_roto_band(**sample_swap.band_kwargs)
    assert band.sd > 0
    assert 0.0 <= band.p_positive <= 1.0


def test_identity_swap_zero_band(identity_swap: _Swap) -> None:
    band = compute_delta_roto_band(**identity_swap.band_kwargs)
    assert band.mean == pytest.approx(0.0, abs=1e-6)
    assert band.sd == pytest.approx(0.0, abs=1e-6)


def test_determinism(sample_swap: _Swap) -> None:
    a = compute_delta_roto_band(**sample_swap.band_kwargs)
    b = compute_delta_roto_band(**sample_swap.band_kwargs)
    assert (a.mean, a.sd, a.p_positive) == (b.mean, b.sd, b.p_positive)


def test_one_for_one_band_mean_matches_ev(sample_swap: _Swap) -> None:
    """compute_one_for_one_band's mean equals the 1-for-1 EV delta."""
    bk = sample_swap.band_kwargs
    pk = sample_swap.point_kwargs
    band = compute_one_for_one_band(
        drop_name=pk["drop_name"],
        add_player=pk["add_player"],
        active_players=pk["user_roster"],
        field_stats=bk["field_stats"],
        team_name=bk["team_name"],
        fraction_remaining=bk["fraction_remaining"],
        projected_standings=bk["projected_standings"],
        team_sds=bk["team_sds"],
    )
    point = compute_delta_roto(**pk)
    assert isinstance(band, DeltaRotoBand)
    assert band.mean == pytest.approx(point.total, abs=1e-9)


def test_noisy_category_swap_has_wider_band() -> None:
    """An SB-driven swap (CV 0.715) yields a wider sd than an R-driven swap
    (CV 0.156) of approximately equal mean.

    This is the only player-independent thing the band legitimately
    encodes: noisier categories -> wider band.
    """
    before = [_hitter(f"H{i}") for i in range(13)]
    field = _contested_field()

    # R-driven: add a player who improves R only (+35). SB-driven: improves
    # SB only (+10). The deltas are tuned (against this contested field) so
    # the mean roto gains land within 0.25 of each other.
    r_add = _hitter("RGuy", r=120, hr=25, rbi=85, sb=10)
    sb_add = _hitter("SBGuy", r=85, hr=25, rbi=85, sb=20)

    r_swap = _build_swap(before, "H12", r_add, field=field)
    sb_swap = _build_swap(before, "H12", sb_add, field=field)

    r_band = compute_delta_roto_band(**r_swap.band_kwargs)
    sb_band = compute_delta_roto_band(**sb_swap.band_kwargs)

    assert sb_band.mean == pytest.approx(r_band.mean, abs=0.25)
    assert sb_band.sd > r_band.sd


def test_band_class_real_when_band_clears_zero() -> None:
    """A large positive swap whose mean - sd > 0 reads 'real'."""
    before = [_hitter(f"H{i}") for i in range(13)]
    add_player = _hitter("Monster", r=140, hr=50, rbi=140, sb=30)
    swap = _build_swap(before, "H12", add_player)
    band = compute_delta_roto_band(**swap.band_kwargs)
    assert band.mean - band.sd > 0
    assert band.to_dict()["verdict"] == "real"


def test_band_class_downgrade_when_band_below_zero() -> None:
    """A clearly negative swap whose mean + sd < 0 reads 'downgrade'."""
    before = [_hitter(f"H{i}") for i in range(13)]
    # Replace a strong starter with a weak player -> negative delta.
    before = [_hitter("Star", r=120, hr=40, rbi=120, sb=25), *before[1:]]
    add_player = _hitter("Scrub", r=20, hr=2, rbi=18, sb=1)
    swap = _build_swap(before, "Star", add_player)
    band = compute_delta_roto_band(**swap.band_kwargs)
    assert band.mean + band.sd < 0
    assert band.to_dict()["verdict"] == "downgrade"


def test_band_class_coin_flip_when_band_straddles_zero() -> None:
    """A small swap whose band straddles zero reads 'coin-flip'."""
    before = [_hitter(f"H{i}") for i in range(13)]
    # Add a player nearly identical to the dropped one, nudged in noisy SB.
    add_player = _hitter("Marginal", sb=16)
    swap = _build_swap(before, "H12", add_player)
    band = compute_delta_roto_band(**swap.band_kwargs)
    assert band.mean - band.sd <= 0 <= band.mean + band.sd
    assert band.to_dict()["verdict"] == "coin-flip"


def test_to_dict_includes_verdict(sample_swap: _Swap) -> None:
    band = compute_delta_roto_band(**sample_swap.band_kwargs)
    d = band.to_dict()
    assert set(d) == {"mean", "sd", "p_positive", "verdict"}
    assert d["verdict"] in {"real", "coin-flip", "downgrade"}
