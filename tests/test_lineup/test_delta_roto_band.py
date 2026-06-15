"""Tests for the analytic deltaRoto confidence band.

The band is closed-form (no Monte Carlo): ``mean`` reuses the EV
deltaRoto so it is identical to the point estimate, and ``sd``
propagates the swapped players' per-category stat variance through each
category's Gaussian roto-points curve. These tests pin the mean-equals-EV
contract, the honest-signal property (noisier categories -> wider band),
determinism, and the P(helps) verdict mapping.

Verdict rule (user-requested, replacing the old +/-1 SD crosses-zero rule):
  p_positive > 0.75 -> "real"
  p_positive < 0.25 -> "downgrade"
  else              -> "coin-flip"
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pytest

from fantasy_baseball.lineup.delta_roto import (
    DeltaRotoBand,
    _swap_category_variance,
    _swap_sets,
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
from fantasy_baseball.scoring import build_team_sds, project_team_stats, score_roto_dict
from fantasy_baseball.trades.evaluate import aggregate_player_stats, apply_swap_delta
from fantasy_baseball.utils.constants import Category

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
    # from_dict computes ERA/WHIP from the er/bb/h_allowed/ip components; the
    # bare PitcherStats(**base) constructor would leave them at 0.
    stats = PitcherStats.from_dict(base)
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


def test_band_class_real_when_p_positive_high() -> None:
    """A large positive swap with p_positive > 0.75 reads 'real'.

    Verdict is now keyed on P(helps) >= 75% (user-requested change,
    replacing the old +/-1 SD crosses-zero rule).
    """
    before = [_hitter(f"H{i}") for i in range(13)]
    add_player = _hitter("Monster", r=140, hr=50, rbi=140, sb=30)
    swap = _build_swap(before, "H12", add_player)
    band = compute_delta_roto_band(**swap.band_kwargs)
    assert band.p_positive > 0.75
    assert band.to_dict()["verdict"] == "real"


def test_band_class_downgrade_when_p_positive_low() -> None:
    """A clearly negative swap with p_positive < 0.25 reads 'downgrade'.

    Verdict is now keyed on P(helps) <= 25% (user-requested change,
    replacing the old +/-1 SD crosses-zero rule).
    """
    before = [_hitter(f"H{i}") for i in range(13)]
    # Replace a strong starter with a weak player -> negative delta.
    before = [_hitter("Star", r=120, hr=40, rbi=120, sb=25), *before[1:]]
    add_player = _hitter("Scrub", r=20, hr=2, rbi=18, sb=1)
    swap = _build_swap(before, "Star", add_player)
    band = compute_delta_roto_band(**swap.band_kwargs)
    assert band.p_positive < 0.25
    assert band.to_dict()["verdict"] == "downgrade"


def test_band_class_coin_flip_when_p_positive_near_50() -> None:
    """A marginal swap with p_positive near 0.5 reads 'coin-flip'.

    Verdict is now keyed on P(helps) -- 25% to 75% is coin-flip
    (user-requested change, replacing the old +/-1 SD crosses-zero rule).
    """
    before = [_hitter(f"H{i}") for i in range(13)]
    # Add a player nearly identical to the dropped one (sb=10), nudged in noisy
    # SB. +2 SB lands p_positive ~0.62 under the NegBin dispersion -- a real but
    # marginal gain, comfortably inside the 25-75% coin-flip band.
    add_player = _hitter("Marginal", sb=12)
    swap = _build_swap(before, "H12", add_player)
    band = compute_delta_roto_band(**swap.band_kwargs)
    assert 0.25 <= band.p_positive <= 0.75
    assert band.to_dict()["verdict"] == "coin-flip"


def test_to_dict_includes_verdict(sample_swap: _Swap) -> None:
    band = compute_delta_roto_band(**sample_swap.band_kwargs)
    d = band.to_dict()
    assert set(d) == {"mean", "sd", "p_positive", "verdict"}
    assert d["verdict"] in {"real", "coin-flip", "downgrade"}


def test_two_for_one_band_mean_matches_multi_swap_ev() -> None:
    """A 2-out / 1-in swap: band.mean equals the multi-swap EV delta.

    Locks the general before/after path (the one ``multi_trade`` and
    ``compute_delta_roto`` share) for N-for-M swaps. The expected mean is
    hand-rolled the same way ``multi_trade`` derives it: aggregate the OUT
    players' ROS, aggregate the IN player's ROS, apply the swap delta to
    the user's projected row, then take the ``score_roto_dict`` total
    before/after diff with the same ``team_sds``. Also asserts sd > 0
    (real per-category stat uncertainty enters the band).
    """
    before = [_hitter(f"H{i}") for i in range(13)]
    # Two players leave, one stronger player enters.
    out_a = before[11]
    out_b = before[12]
    add_player = _hitter("BigBat", hr=45, r=105, rbi=110, sb=18)
    after = [p for p in before if p.name not in {out_a.name, out_b.name}] + [add_player]

    field = _field()
    me_stats = project_team_stats(before)
    entries = [ProjectedStandingsEntry(team_name="Me", stats=me_stats)]
    entries += [ProjectedStandingsEntry(team_name=t, stats=cs) for t, cs in field.items()]
    projected = ProjectedStandings(effective_date=date(2026, 4, 1), entries=entries)
    rosters = {"Me": before, **{t: [] for t in field}}
    team_sds = build_team_sds(rosters, sd_scale=FRACTION_REMAINING**0.5)

    band = compute_delta_roto_band(
        before_players=before,
        after_players=after,
        field_stats=field,
        team_name="Me",
        fraction_remaining=FRACTION_REMAINING,
        projected_standings=projected,
        team_sds=team_sds,
    )

    # Hand-rolled multi-swap EV: aggregate OUT and IN ROS, apply the swap
    # delta, diff the score_roto_dict totals. Mirrors multi_trade's mean.
    all_before = {e.team_name: e.stats.to_dict() for e in projected.entries}
    loses_ros = aggregate_player_stats([out_a, out_b])
    gains_ros = aggregate_player_stats([add_player])
    user_after = apply_swap_delta(all_before["Me"], loses_ros, gains_ros)
    all_after = dict(all_before)
    all_after["Me"] = user_after
    roto_before = score_roto_dict(all_before, team_sds=team_sds)
    roto_after = score_roto_dict(all_after, team_sds=team_sds)
    expected_mean = roto_after["Me"]["total"] - roto_before["Me"]["total"]

    assert band.mean == pytest.approx(expected_mean, abs=1e-9)
    assert band.sd > 0


def _pitcher_field() -> dict[str, CategoryStats]:
    """Field with ERA/WHIP centered near a pitcher-heavy user roster.

    ``_field`` pins ERA at 3.8 / WHIP at 1.20, which the staff below
    (ERA ~3.4) clears comfortably, parking the rate categories off the
    steep part of the curve. Raising the field ERA/WHIP to straddle the
    user's keeps the ERA/WHIP win-probabilities responsive so a rate-edge
    swap moves both the mean and the band.
    """
    field: dict[str, CategoryStats] = {}
    for i in range(8):
        field[f"Team{i}"] = CategoryStats(
            r=800 + i * 15,
            hr=220 + i * 5,
            rbi=780 + i * 12,
            sb=110 + i * 4,
            avg=0.255 + i * 0.002,
            w=70 + i * 2,
            k=1100 + i * 20,
            sv=70,
            era=3.30 + i * 0.04,
            whip=1.15 + i * 0.01,
        )
    return field


def test_rate_category_pitcher_swap_band_responds() -> None:
    """A pitcher swap whose edge is in ERA (a rate category) moves the band.

    Exercises the ERA/WHIP branch of ``_swap_category_variance``: dropping
    a high-ERA arm for an equal-volume low-ERA arm improves the team ERA
    (an inverse stat, so lower is better -> positive mean) and shifts the
    team's rate variance, so sd > 0. Holds IP roughly constant so the edge
    lands in the rate, not in counting K/W.
    """
    # Six-arm staff: five solid, one weak ERA arm to be dropped.
    before = [_pitcher(f"P{i}", ip=170, w=11, k=180, er=60, bb=45, h_allowed=145) for i in range(5)]
    weak = _pitcher("WeakArm", ip=170, w=11, k=180, er=100, bb=45, h_allowed=145)
    before = [*before, weak]
    # Equal-volume replacement with a much better ERA (fewer ER, same IP).
    strong = _pitcher("StrongArm", ip=170, w=11, k=180, er=45, bb=45, h_allowed=145)

    field = _pitcher_field()
    me_stats = project_team_stats(before)
    entries = [ProjectedStandingsEntry(team_name="Me", stats=me_stats)]
    entries += [ProjectedStandingsEntry(team_name=t, stats=cs) for t, cs in field.items()]
    projected = ProjectedStandings(effective_date=date(2026, 4, 1), entries=entries)
    rosters = {"Me": before, **{t: [] for t in field}}
    team_sds = build_team_sds(rosters, sd_scale=FRACTION_REMAINING**0.5)

    after = [p for p in before if p.name != "WeakArm"] + [strong]

    # Pin the ERA branch directly: the rate-variance term must be positive
    # in its own right, so the test would fail if the ERA/WHIP branch of
    # _swap_category_variance regressed to 0 (the counting K/W noise alone
    # would otherwise keep band.sd > 0 and mask the regression).
    in_players, out_players = _swap_sets(before, after)
    era_sigma2 = _swap_category_variance(
        Category.ERA, in_players, out_players, before, after, FRACTION_REMAINING
    )
    assert era_sigma2 > 0

    band = compute_delta_roto_band(
        before_players=before,
        after_players=after,
        field_stats=field,
        team_name="Me",
        fraction_remaining=FRACTION_REMAINING,
        projected_standings=projected,
        team_sds=team_sds,
    )

    # Improving ERA (inverse stat) is a gain -> positive mean; the rate
    # variance shifts -> the band has positive width.
    assert band.mean > 0
    assert band.sd > 0
