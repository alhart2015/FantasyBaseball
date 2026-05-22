from datetime import date

from fantasy_baseball.lineup.delta_roto import (
    DeltaRotoBand,
    compute_delta_roto,
    compute_delta_roto_band,
)
from fantasy_baseball.models.player import HitterStats, Player, PlayerType
from fantasy_baseball.models.standings import (
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
)


def _hitter(name, **ros):
    base = dict(pa=600, ab=540, h=150, r=85, hr=25, rbi=85, sb=10)
    base.update(ros)
    return Player(
        name=name,
        player_type=PlayerType.HITTER,
        positions=["OF"],
        rest_of_season=HitterStats(**base),
        full_season_projection=HitterStats(**base),
    )


def _field():
    field = {}
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


def test_band_returns_mean_sd_ppos():
    before = [_hitter(f"H{i}") for i in range(13)]
    after = [*before[:-1], _hitter("BigBat", hr=45, r=105, rbi=110)]
    band = compute_delta_roto_band(
        before, after, _field(), "Me", fraction_remaining=0.6, n_draws=300, seed=1
    )
    assert isinstance(band, DeltaRotoBand)
    assert band.sd > 0
    assert 0.0 <= band.p_positive <= 1.0


def test_band_is_deterministic_for_fixed_seed():
    before = [_hitter(f"H{i}") for i in range(13)]
    after = [*before[:-1], _hitter("BigBat", hr=45)]
    a = compute_delta_roto_band(
        before, after, _field(), "Me", fraction_remaining=0.6, n_draws=200, seed=7
    )
    b = compute_delta_roto_band(
        before, after, _field(), "Me", fraction_remaining=0.6, n_draws=200, seed=7
    )
    assert a.mean == b.mean and a.sd == b.sd and a.p_positive == b.p_positive


def test_identity_swap_has_near_zero_mean():
    before = [_hitter(f"H{i}") for i in range(13)]
    after = list(before)
    band = compute_delta_roto_band(
        before, after, _field(), "Me", fraction_remaining=0.6, n_draws=300, seed=3
    )
    assert abs(band.mean) < 0.05


def test_mean_tracks_point_estimate():
    """The MC band's mean must approximately track the EV point estimate.

    Build a ProjectedStandings whose "Me" entry equals the summed
    before-roster projection and whose other 8 entries equal the fixed
    field; run the existing point-estimate compute_delta_roto for the
    same one-for-one swap and assert the band mean lands within 0.5 roto
    points of it. A larger gap means the sampling seam is wrong (e.g.
    sampling the wrong projection field or building a mismatched
    baseline/field).
    """
    from fantasy_baseball.scoring import project_team_stats

    before = [_hitter(f"H{i}") for i in range(13)]
    drop_name = "H12"
    add_player = _hitter("BigBat", hr=45, r=105, rbi=110)
    after = [p for p in before if p.name != drop_name] + [add_player]

    field = _field()
    me_stats = project_team_stats(before)
    entries = [ProjectedStandingsEntry(team_name="Me", stats=me_stats)]
    entries += [ProjectedStandingsEntry(team_name=t, stats=cs) for t, cs in field.items()]
    projected = ProjectedStandings(effective_date=date(2026, 4, 1), entries=entries)

    point = compute_delta_roto(
        drop_name,
        add_player,
        before,
        projected,
        "Me",
        team_sds=None,
    )

    band = compute_delta_roto_band(
        before, after, field, "Me", fraction_remaining=0.6, n_draws=600, seed=11
    )

    assert abs(band.mean - point.total) < 0.5, (
        f"band.mean={band.mean:.4f} vs point.total={point.total:.4f}"
    )


def test_to_dict_includes_verdict():
    before = [_hitter(f"H{i}") for i in range(13)]
    after = [*before[:-1], _hitter("BigBat", hr=45, r=105, rbi=110)]
    band = compute_delta_roto_band(
        before, after, _field(), "Me", fraction_remaining=0.6, n_draws=200, seed=5
    )
    d = band.to_dict()
    assert set(d) == {"mean", "sd", "p_positive", "verdict"}
    assert d["verdict"] in {"real", "coin-flip", "downgrade"}
