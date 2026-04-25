import pytest


def _two_team_projected_standings():
    """Minimal ProjectedStandings stub for delta math tests.

    CategoryStats only holds the ten roto categories — no volume stats
    like ab/ip/h/er/bb. TeamA already leads every category except HR,
    where it sits just behind TeamB. That way a 40-HR candidate trips
    the HR rank flip (biggest category delta) without changing ranks
    elsewhere — even though the raw R gain is larger in counting units.
    """
    from fantasy_baseball.models.standings import ProjectedStandings

    rows = [
        {
            "name": "TeamA",
            "stats": {
                "R": 800,
                "HR": 190,
                "RBI": 800,
                "SB": 120,
                "AVG": 0.270,
                "W": 80,
                "K": 1400,
                "SV": 60,
                "ERA": 3.60,
                "WHIP": 1.15,
            },
        },
        {
            "name": "TeamB",
            "stats": {
                "R": 600,
                "HR": 200,
                "RBI": 600,
                "SB": 80,
                "AVG": 0.260,
                "W": 60,
                "K": 1200,
                "SV": 40,
                "ERA": 3.80,
                "WHIP": 1.20,
            },
        },
    ]
    return ProjectedStandings.from_json({"effective_date": "2026-04-01", "teams": rows})


def test_immediate_eroto_delta_positive_for_category_gain():
    from fantasy_baseball.draft.eroto_recs import immediate_delta
    from fantasy_baseball.models.player import HitterStats, Player, PlayerType

    candidate = Player(
        name="Bomber",
        player_type=PlayerType.HITTER,
        positions=["OF"],
        rest_of_season=HitterStats(r=100, hr=40, rbi=110, sb=5, avg=0.280, ab=580, h=163),
    )
    standings = _two_team_projected_standings()
    replacement = Player(
        name="Replacement OF",
        player_type=PlayerType.HITTER,
        positions=["OF"],
        rest_of_season=HitterStats(r=55, hr=10, rbi=45, sb=2, avg=0.240, ab=450, h=108),
    )

    delta = immediate_delta(
        candidate=candidate,
        replacement=replacement,
        team_name="TeamA",
        projected_standings=standings,
        team_sds=None,
    )
    assert delta.total > 0
    assert max(delta.per_category, key=lambda c: delta.per_category[c]) == "HR"


def test_rank_candidates_sorts_by_immediate_delta():
    from fantasy_baseball.draft.eroto_recs import rank_candidates
    from fantasy_baseball.models.player import HitterStats, Player, PlayerType

    bomber = Player(
        name="Bomber",
        player_type=PlayerType.HITTER,
        positions=["OF"],
        rest_of_season=HitterStats(r=100, hr=40, rbi=110, sb=5, avg=0.280, ab=580, h=163),
    )
    slap = Player(
        name="Slap Hitter",
        player_type=PlayerType.HITTER,
        positions=["OF"],
        rest_of_season=HitterStats(r=85, hr=8, rbi=55, sb=10, avg=0.305, ab=570, h=174),
    )
    standings = _two_team_projected_standings()
    replacements = {
        "OF": Player(
            name="Replacement OF",
            player_type=PlayerType.HITTER,
            positions=["OF"],
            rest_of_season=HitterStats(r=55, hr=10, rbi=45, sb=2, avg=0.240, ab=450, h=108),
        ),
    }

    rows = rank_candidates(
        candidates=[slap, bomber],
        replacements=replacements,
        team_name="TeamA",
        projected_standings=standings,
        team_sds=None,
    )
    names = [r.name for r in rows]
    assert names[0] == "Bomber"
    assert rows[0].immediate_delta > rows[1].immediate_delta


def test_value_of_picking_now_positive_when_player_is_scarce():
    """If only one bomber remains in the pool and opponents are HR-hungry,
    the value-of-picking-now must be positive — waiting costs us him."""
    from fantasy_baseball.draft.adp import ADPTable
    from fantasy_baseball.draft.eroto_recs import rank_candidates
    from fantasy_baseball.models.player import HitterStats, Player, PlayerType

    bomber = Player(
        name="Bomber",
        player_type=PlayerType.HITTER,
        positions=["OF"],
        yahoo_id="bomber::hitter",
        rest_of_season=HitterStats(r=100, hr=40, rbi=110, sb=5, avg=0.280, ab=580, h=163),
    )
    slap1 = Player(
        name="Slap One",
        player_type=PlayerType.HITTER,
        positions=["OF"],
        yahoo_id="slap1::hitter",
        rest_of_season=HitterStats(r=85, hr=8, rbi=55, sb=10, avg=0.305, ab=570, h=174),
    )
    slap2 = Player(
        name="Slap Two",
        player_type=PlayerType.HITTER,
        positions=["OF"],
        yahoo_id="slap2::hitter",
        rest_of_season=HitterStats(r=82, hr=7, rbi=52, sb=9, avg=0.300, ab=560, h=168),
    )
    standings = _two_team_projected_standings()
    replacements = {
        "OF": Player(
            name="Replacement OF",
            player_type=PlayerType.HITTER,
            positions=["OF"],
            rest_of_season=HitterStats(r=55, hr=10, rbi=45, sb=2, avg=0.240, ab=450, h=108),
        )
    }
    adp_table = ADPTable(
        adp={
            "bomber::hitter": 15.0,
            "slap1::hitter": 140.0,
            "slap2::hitter": 155.0,
        }
    )

    rows = rank_candidates(
        candidates=[bomber, slap1, slap2],
        replacements=replacements,
        team_name="TeamA",
        projected_standings=standings,
        team_sds=None,
        picks_until_next_turn=3,
        adp_table=adp_table,
    )
    bomber_row = next(r for r in rows if r.name == "Bomber")
    slap_row = next(r for r in rows if r.name == "Slap One")

    assert bomber_row.value_of_picking_now > 0
    assert abs(slap_row.value_of_picking_now) < bomber_row.value_of_picking_now


def test_vopn_sort_differs_from_immediate_delta_sort_when_some_survive():
    """VOPN must produce a different sort order than immediate_delta when
    some candidates survive opponent picks. A high-immediate-delta
    surviving player should rank LOWER in VOPN sort than a sniped player
    with somewhat lower immediate_delta — that's the whole point.
    """
    from fantasy_baseball.draft.adp import ADPTable
    from fantasy_baseball.draft.eroto_recs import rank_candidates
    from fantasy_baseball.models.player import HitterStats, Player, PlayerType

    # Three candidates: stud (highest delta, low ADP — sniped),
    # mid (mid delta, late ADP — survives), late (low delta, late ADP).
    stud = Player(
        name="Stud",
        player_type=PlayerType.HITTER,
        positions=["OF"],
        yahoo_id="stud::hitter",
        rest_of_season=HitterStats(r=110, hr=40, rbi=110, sb=10, avg=0.290, ab=580, h=168),
    )
    mid = Player(
        name="Mid",
        player_type=PlayerType.HITTER,
        positions=["OF"],
        yahoo_id="mid::hitter",
        rest_of_season=HitterStats(r=85, hr=22, rbi=70, sb=15, avg=0.270, ab=560, h=151),
    )
    late = Player(
        name="Late",
        player_type=PlayerType.HITTER,
        positions=["OF"],
        yahoo_id="late::hitter",
        rest_of_season=HitterStats(r=70, hr=10, rbi=50, sb=8, avg=0.260, ab=520, h=135),
    )
    standings = _two_team_projected_standings()
    replacements = {
        "OF": Player(
            name="Replacement OF",
            player_type=PlayerType.HITTER,
            positions=["OF"],
            rest_of_season=HitterStats(r=55, hr=10, rbi=45, sb=2, avg=0.240, ab=450, h=108),
        )
    }
    # Stud goes early; mid and late are late-round.
    adp = ADPTable(adp={"stud::hitter": 5.0, "mid::hitter": 80.0, "late::hitter": 110.0})

    rows = rank_candidates(
        candidates=[stud, mid, late],
        replacements=replacements,
        team_name="TeamA",
        projected_standings=standings,
        team_sds=None,
        picks_until_next_turn=1,  # exactly one snipe → only stud goes
        adp_table=adp,
    )

    vopns = {r.name: r.value_of_picking_now for r in rows}
    deltas = {r.name: r.immediate_delta for r in rows}

    # Surviving players have no regret — I can wait and grab them later.
    assert vopns["Mid"] == 0.0
    assert vopns["Late"] == 0.0

    # Core invariant break: under the OLD bug, vopn[X] - vopn[Y] equaled
    # delta[X] - delta[Y] for ALL pairs (single constant K subtracted).
    # The fix breaks that for at least one pair so VOPN sort actually
    # differs from immediate_delta sort.
    names = ["Stud", "Mid", "Late"]
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1 :]]
    differing = [
        (a, b) for a, b in pairs if (vopns[a] - vopns[b]) != pytest.approx(deltas[a] - deltas[b])
    ]
    assert differing, "VOPN sort would equal immediate_delta sort if every pair matched"
