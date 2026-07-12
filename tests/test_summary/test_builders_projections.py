from fantasy_baseball.summary.builders import build_projection_delta

_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]


def _stats(vals):
    return dict(zip(_CATS, vals, strict=True))


def _sds():
    row = dict(zip(_CATS, [30, 12, 28, 10, 0.006, 6, 60, 7, 0.15, 0.03], strict=True))
    return {"My Team": row, "Rival": dict(row)}


def _projections():
    return {
        "projected_standings": {
            "effective_date": "2026-10-01",
            "teams": [
                {
                    "name": "My Team",
                    "stats": _stats([820, 255, 790, 120, 0.265, 95, 1450, 48, 3.70, 1.15]),
                    "total_ab": 5400,
                    "total_ip": 1400,
                },
                {
                    "name": "Rival",
                    "stats": _stats([780, 240, 760, 100, 0.258, 88, 1380, 40, 3.95, 1.22]),
                    "total_ab": 5400,
                    "total_ip": 1400,
                },
            ],
        },
        "team_sds": _sds(),
    }


def _mc(first_pct):
    return {
        "rest_of_season": {
            "team_results": {
                "My Team": {
                    "first_pct": first_pct,
                    "top3_pct": 55.0,
                    "median_pts": 78.0,
                    "p10": 70,
                    "p90": 86,
                }
            }
        }
    }


def _expected_total():
    from fantasy_baseball.models.standings import ProjectedStandings
    from fantasy_baseball.scoring import score_roto, team_sds_from_json

    p = _projections()
    roto = score_roto(
        ProjectedStandings.from_json(p["projected_standings"]),
        team_sds=team_sds_from_json(p["team_sds"]),
    )
    return roto["My Team"]


def test_eroto_all_ten_categories_and_total_match_score_roto():
    pd = build_projection_delta(_projections(), _projections(), _mc(18.4), _mc(17.2), "My Team")
    assert {e.category for e in pd.eroto} == set(_CATS)
    assert len(pd.eroto) == 10
    exp = _expected_total()
    assert abs(pd.eroto_total_now - float(exp.total)) < 1e-9
    # per-category matches score_roto's values
    exp_vals = {str(getattr(c, "value", c)): float(v) for c, v in exp.values.items()}
    for e in pd.eroto:
        assert abs(e.now - exp_vals[e.category]) < 1e-9


def test_overnight_deltas_and_champ():
    # Same projections current vs prior -> zero eRoto delta; champ from MC.
    pd = build_projection_delta(_projections(), _projections(), _mc(18.4), _mc(17.2), "My Team")
    assert pd.is_first_run is False
    assert all(e.prev == e.now for e in pd.eroto)
    assert pd.champ_pct_now == 18.4
    assert pd.champ_pct_prev == 17.2


def test_first_run_has_no_prior():
    pd = build_projection_delta(_projections(), None, _mc(18.4), None, "My Team")
    assert pd.is_first_run is True
    assert all(e.prev is None for e in pd.eroto)
    assert pd.champ_pct_now == 18.4
    assert pd.champ_pct_prev is None
    assert pd.has_content is True


def test_mc_absent_still_shows_eroto():
    pd = build_projection_delta(_projections(), None, None, None, "My Team")
    assert pd.champ_pct_now is None
    assert len(pd.eroto) == 10
    assert pd.has_content is True


def test_all_absent_is_empty():
    pd = build_projection_delta(None, None, None, None, "My Team")
    assert pd.eroto == []
    assert pd.champ_pct_now is None
    assert pd.has_content is False


def test_team_absent_from_projections_and_mc():
    pd = build_projection_delta(_projections(), None, _mc(18.4), None, "Nonexistent")
    assert pd.eroto == []  # team not in projected standings
    assert pd.champ_pct_now is None  # team not in MC team_results
    assert pd.has_content is False
