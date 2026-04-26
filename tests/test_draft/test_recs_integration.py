"""Unit tests for recs_integration.monte_carlo_roto_totals + compute_standings_cache."""

from __future__ import annotations

from datetime import date

from fantasy_baseball.draft.recs_integration import (
    compute_standings_cache,
    monte_carlo_roto_totals,
)
from fantasy_baseball.models.standings import (
    CategoryStats,
    ProjectedStandings,
    ProjectedStandingsEntry,
)
from fantasy_baseball.utils.constants import ALL_CATEGORIES, Category


def _three_team_standings(hr_spread: tuple[int, int, int]) -> ProjectedStandings:
    """Build a minimal 3-team ProjectedStandings with the given HR spread.

    All other categories are identical across teams so ranks are driven
    by HR alone — easy to reason about for the SD assertions.
    """
    entries = []
    for team_name, hr in zip(("A", "B", "C"), hr_spread, strict=True):
        entries.append(
            ProjectedStandingsEntry(
                team_name=team_name,
                stats=CategoryStats(
                    r=600,
                    hr=hr,
                    rbi=600,
                    sb=80,
                    avg=0.260,
                    w=60,
                    k=1200,
                    sv=40,
                    era=3.80,
                    whip=1.20,
                ),
            )
        )
    return ProjectedStandings(effective_date=date(2026, 1, 1), entries=entries)


def _uniform_sds(sd_per_cat: float) -> dict[str, dict[Category, float]]:
    """Every team has the same SD for every category."""
    return {t: {cat: sd_per_cat for cat in ALL_CATEGORIES} for t in ("A", "B", "C")}


def test_mc_sd_larger_for_close_race_than_blowout():
    """When HR totals are close, projection noise frequently flips the
    HR rank — per-team totals swing more, so SDs should be noticeably
    larger than in a blowout where ranks are stable."""
    # Close race: A=180, B=182, C=184 — HR rank easily flips under noise.
    close = monte_carlo_roto_totals(
        _three_team_standings((180, 182, 184)),
        _uniform_sds(sd_per_cat=10.0),
        n_iters=500,
        seed=42,
    )
    # Blowout: A=100, B=200, C=300 — HR rank is locked in under the same noise.
    blowout = monte_carlo_roto_totals(
        _three_team_standings((100, 200, 300)),
        _uniform_sds(sd_per_cat=10.0),
        n_iters=500,
        seed=42,
    )

    close_sd_avg = sum(s for _, s in close.values()) / 3
    blowout_sd_avg = sum(s for _, s in blowout.values()) / 3
    assert close_sd_avg > blowout_sd_avg
    # Sanity: SDs should be roto-points-scale, not raw-stat-scale. In a
    # 3-team league the per-team total roto range is 10..30, so SD > 10
    # would be suspicious. The old quadrature-of-raw-SDs bug produced
    # values above the scale max.
    assert close_sd_avg < 5.0


def test_mc_sd_zero_when_team_sds_zero():
    """Zero projection SD → deterministic ranks → zero total SD."""
    result = monte_carlo_roto_totals(
        _three_team_standings((100, 200, 300)),
        _uniform_sds(sd_per_cat=0.0),
        n_iters=200,
        seed=0,
    )
    for _, sd in result.values():
        assert sd == 0.0


def test_mc_mean_approximates_expected_rank_sum():
    """With zero noise, each team's total equals rank-per-cat summed
    across all categories. With ``hr`` the only varying category and
    everything else tied, rank-averaging on ties puts every team at
    ``2.0`` per tied cat; HR gives 1, 2, 3. Total = 2.0*(N-1) + HR_rank."""
    n_cats = len(ALL_CATEGORIES)
    expected_tied_points = 2.0 * (n_cats - 1)  # every non-HR cat is a 3-way tie

    result = monte_carlo_roto_totals(
        _three_team_standings((100, 200, 300)),
        _uniform_sds(sd_per_cat=0.0),
        n_iters=100,
        seed=0,
    )
    # A has worst HR → rank 1, C has best → rank 3.
    assert result["A"][0] == expected_tied_points + 1.0
    assert result["B"][0] == expected_tied_points + 2.0
    assert result["C"][0] == expected_tied_points + 3.0


def test_compute_standings_cache_returns_typed_rows():
    """Cache returns ``TeamStandingsRow`` instances with float total/total_sd
    and a ``Category``-keyed per-category dict."""
    from fantasy_baseball.draft.recs_integration import TeamStandingsRow

    standings = _three_team_standings((150, 200, 250))
    team_sds = _uniform_sds(sd_per_cat=5.0)
    cache = compute_standings_cache(standings, team_sds, mc_iters=100, mc_seed=7)

    assert set(cache.keys()) == {"A", "B", "C"}
    for row in cache.values():
        assert isinstance(row, TeamStandingsRow)
        assert isinstance(row.total, float)
        assert isinstance(row.total_sd, float)
        assert row.total_sd >= 0.0
        # Categories keyed on the Category enum, valued as float roto points.
        assert set(row.categories.keys()) == set(ALL_CATEGORIES)
        for ev in row.categories.values():
            assert isinstance(ev, float)


def test_standings_cache_round_trips_through_json():
    """``serialize_standings_cache`` and ``deserialize_standings_cache``
    must be inverses so the cache survives a write_state/read_state cycle."""
    from fantasy_baseball.draft.recs_integration import (
        deserialize_standings_cache,
        serialize_standings_cache,
    )

    standings = _three_team_standings((150, 200, 250))
    team_sds = _uniform_sds(sd_per_cat=5.0)
    original = compute_standings_cache(standings, team_sds, mc_iters=100, mc_seed=7)
    json_shape = serialize_standings_cache(original)
    parsed = deserialize_standings_cache(json_shape)

    assert parsed == original


def test_deserialize_drops_malformed_entries():
    """Legacy / malformed entries are skipped, not crashed on."""
    from fantasy_baseball.draft.recs_integration import deserialize_standings_cache

    cache = deserialize_standings_cache(
        {
            "GoodTeam": {"total": 55.0, "total_sd": 4.2, "categories": {"R": 6.0}},
            "LegacyTeam": {"R": {"point_estimate": 6.0, "sd": 25.0}},  # old schema
            "EmptyTeam": {},
        }
    )
    assert set(cache.keys()) == {"GoodTeam"}
    assert cache["GoodTeam"].total == 55.0
    assert cache["GoodTeam"].total_sd == 4.2


def test_compute_rec_inputs_uses_marginal_replacements(tmp_path):
    """compute_rec_inputs must surface true replacement-level players,
    not top-tier MLB regulars. Regression: the old
    build_replacements_by_position used `demand = capacity * num_teams`
    per slot, which produced replacements like 'top-21 hitter for UTIL'
    instead of 'marginal hitter for UTIL'."""
    import json

    from fantasy_baseball.draft.recs_integration import compute_rec_inputs
    from fantasy_baseball.draft.state import StateKey

    # Build a 30-hitter pool. With 1 OF starter and 1 UTIL starter
    # in a 1-team league: positional_hitter_starters=1, total UTIL
    # demand = 1+1 = 2. The UTIL replacement is the 3rd-best hitter
    # overall (index 2).
    rows = [
        {
            "name": f"H{i}",
            "player_id": f"{i}::hitter",
            "player_type": "hitter",
            "positions": ["OF", "Util"],
            "total_sgp": 30.0 - i,
            "var": 30.0 - i,
            "ab": 500,
            "h": 130,
            "r": 80,
            "hr": 20,
            "rbi": 75,
            "sb": 5,
            "avg": 0.260,
        }
        for i in range(30)
    ]
    board_path = tmp_path / "board.json"
    board_path.write_text(json.dumps(rows))

    state = {
        StateKey.KEEPERS: [],
        StateKey.PICKS: [],
    }
    league_yaml = {
        "draft": {"teams": {1: "Solo"}},
        "roster_slots": {"OF": 1, "UTIL": 1, "BN": 0},
    }

    inputs = compute_rec_inputs(state, board_path, league_yaml)

    # OF replacement: 2nd-best OF (index 1) → "H1"
    assert inputs.replacements["OF"].name == "H1"
    # UTIL replacement: 3rd-best hitter overall (index 2) → "H2"
    assert inputs.replacements["UTIL"].name == "H2"


def test_build_team_rosters_pads_pitcher_slots_with_pitchers(tmp_path):
    """build_team_rosters must fill empty pitcher slots with a pitcher
    replacement, not the generic hitter replacement.

    Regression: the old code used _generic_replacement (highest-SGP
    across ALL positions, always a hitter) for every empty slot.
    A team with empty P slots ended up with hitter clones counted as
    pitchers — giving the team phantom R/HR/RBI/SB/AVG from pitcher
    slots and zero W/K/SV/ERA/WHIP. This made recommender's swap math
    (which assumed the pitcher slot held a P replacement) disagree with
    the standings path (which saw a hitter clone in the P slot).
    """
    import json

    from fantasy_baseball.draft.recs_integration import build_team_rosters
    from fantasy_baseball.draft.state import StateKey
    from fantasy_baseball.models.player import PlayerType

    # Build a minimal board: some hitters and pitchers with known stats.
    rows = []
    for i in range(10):
        rows.append(
            {
                "name": f"H{i}",
                "player_id": f"h{i}::hitter",
                "player_type": "hitter",
                "positions": ["OF", "Util"],
                "total_sgp": 10.0 - i,
                "var": 10.0 - i,
                "ab": 500,
                "h": 130,
                "r": 80,
                "hr": 20,
                "rbi": 75,
                "sb": 5,
                "avg": 0.260,
            }
        )
    for i in range(10):
        rows.append(
            {
                "name": f"P{i}",
                "player_id": f"p{i}::pitcher",
                "player_type": "pitcher",
                "positions": ["P"],
                "total_sgp": 10.0 - i,
                "var": 10.0 - i,
                "ip": 180,
                "er": 70,
                "bb": 50,
                "h_allowed": 160,
                "w": 12,
                "k": 170,
                "sv": 0,
                "era": 3.50,
                "whip": 1.18,
            }
        )

    # Build a fake board_by_id with one player (H0 on a team).
    board_path = tmp_path / "board.json"
    board_path.write_text(json.dumps(rows))
    from fantasy_baseball.draft.recs_integration import rows_to_players

    players = rows_to_players(rows)
    board_by_id = {p.yahoo_id: p for p in players if p.yahoo_id}

    # Create type-specific replacements as Player objects.
    hitter_rep = next(p for p in players if p.player_type == PlayerType.HITTER and p.name == "H9")
    pitcher_rep = next(p for p in players if p.player_type == PlayerType.PITCHER and p.name == "P9")
    replacements = {"OF": hitter_rep, "UTIL": hitter_rep, "P": pitcher_rep}

    # One team with one hitter drafted; all pitcher slots empty.
    state = {
        StateKey.KEEPERS: [],
        StateKey.PICKS: [
            {"player_id": "h0::hitter", "player_name": "H0", "team": "Solo", "position": "OF"}
        ],
    }
    roster_slots = {"OF": 2, "UTIL": 1, "P": 3}  # 3 hitter slots + 3 pitcher slots

    roster = build_team_rosters(state, board_by_id, ["Solo"], roster_slots, replacements)["Solo"]

    # Count actual picks by type in the padded roster.
    hitters_on_roster = [p for p in roster if p.player_type == PlayerType.HITTER]
    pitchers_on_roster = [p for p in roster if p.player_type == PlayerType.PITCHER]

    # Should have exactly 3 hitters (1 real + 2 padding) and 3 pitchers (all padding).
    assert len(hitters_on_roster) == 3, (
        f"Expected 3 hitter slots filled, got {len(hitters_on_roster)}: "
        f"{[p.name for p in hitters_on_roster]}"
    )
    assert len(pitchers_on_roster) == 3, (
        f"Expected 3 pitcher slots filled (all with pitcher replacement), "
        f"got {len(pitchers_on_roster)}: {[p.name for p in pitchers_on_roster]}. "
        f"If this is 0, the bug (all slots padded with hitter) is present."
    )


def test_recommended_pick_increases_projected_standings_total(tmp_path):
    """A recommended pick must INCREASE the team's projected total roto
    points (relative to the recommender's starting baseline).

    Regression: build_team_rosters used a single _generic_replacement
    (always a hitter) for ALL empty slots, but the recommender's swap
    math used position-specific replacements. Picking a recommended
    pitcher would drop a hitter clone from the projected roster, often
    making the team's total points go DOWN even though the recommender
    promised they'd go UP.

    Setup: Solo has one hitter already drafted. Rival has six strong
    hitters, giving them an edge in all hitting cats. P1 (a solid SP)
    gives Solo pitching value — the recommender computes delta vs the
    P replacement (P replacement ≈ P18, a mediocre SP). The two paths
    must agree that picking a pitcher with positive immediate_delta
    actually increases the projected total.
    """
    import json

    from fantasy_baseball.draft.eroto_recs import rank_candidates
    from fantasy_baseball.draft.recs_integration import compute_rec_inputs
    from fantasy_baseball.draft.state import StateKey
    from fantasy_baseball.scoring import score_roto

    rows = []
    for i in range(30):
        rows.append(
            {
                "name": f"H{i}",
                "player_id": f"h{i}::hitter",
                "player_type": "hitter",
                "positions": ["OF", "Util"],
                "total_sgp": 30.0 - i,
                "var": 30.0 - i,
                "ab": 500,
                "h": 130,
                "r": 80,
                "hr": 20,
                "rbi": 75,
                "sb": 5,
                "avg": 0.260,
            }
        )
    # All pitchers are SPs — ordered by quality. P0 is the best (ace),
    # P18 is the replacement level. Any P with idx < 18 should have
    # positive immediate_delta (better than the replacement).
    for i in range(30):
        rows.append(
            {
                "name": f"P{i}",
                "player_id": f"p{i}::pitcher",
                "player_type": "pitcher",
                "positions": ["P"],
                "total_sgp": 30.0 - i,
                "var": 30.0 - i,
                "ip": max(50, 200 - i * 5),
                "er": 50 + i * 3,
                "bb": 40 + i * 2,
                "h_allowed": 150 + i * 3,
                "w": max(3, 18 - i // 2),
                "k": max(40, 250 - i * 8),
                "sv": 0,
                "era": 2.25 + i * 0.12,
                "whip": 1.00 + i * 0.02,
            }
        )
    board_path = tmp_path / "board.json"
    board_path.write_text(json.dumps(rows))

    # Rival has 6 strong hitters (dominates hitting cats).
    # Solo has 1 hitter + empty P slots. Bug: those P slots get padded
    # with hitter clones instead of pitcher replacements.
    rival_keepers = [
        {"player_id": f"h{i}::hitter", "player_name": f"H{i}", "team": "Rival", "position": "OF"}
        for i in range(6)
    ]
    solo_keepers = [
        {"player_id": "h6::hitter", "player_name": "H6", "team": "Solo", "position": "OF"}
    ]

    state_before = {StateKey.KEEPERS: rival_keepers + solo_keepers, StateKey.PICKS: []}
    league_yaml = {
        "draft": {"teams": {1: "Solo", 2: "Rival"}},
        "roster_slots": {"OF": 4, "UTIL": 2, "P": 9, "BN": 0},
    }

    inputs_before = compute_rec_inputs(state_before, board_path, league_yaml)
    recs = rank_candidates(
        candidates=inputs_before.candidates,
        replacements=inputs_before.replacements,
        team_name="Solo",
        projected_standings=inputs_before.projected_standings,
        team_sds=inputs_before.team_sds,
        picks_until_next_turn=0,
        adp_table=inputs_before.adp_table,
    )

    # Find any pitcher with positive immediate_delta — these are better
    # than the P replacement and the recommender says to pick them.
    positive_pitcher_recs = [r for r in recs if r.immediate_delta > 0 and "P" in r.positions]
    assert positive_pitcher_recs, (
        "No pitcher with positive immediate_delta found — test fixture is invalid. "
        "At least P0 (best SP) should beat the P replacement."
    )
    top_pitcher = positive_pitcher_recs[0]

    # Baseline standings total for Solo.
    roto_before = score_roto(inputs_before.projected_standings, team_sds=inputs_before.team_sds)
    total_before = roto_before["Solo"].total

    # Pick that pitcher and recompute.
    state_after = {
        StateKey.KEEPERS: rival_keepers + solo_keepers,
        StateKey.PICKS: [
            {
                "player_id": top_pitcher.player_id,
                "player_name": top_pitcher.name,
                "team": "Solo",
                "position": top_pitcher.positions[0],
            }
        ],
    }
    inputs_after = compute_rec_inputs(state_after, board_path, league_yaml)
    roto_after = score_roto(inputs_after.projected_standings, team_sds=inputs_after.team_sds)
    total_after = roto_after["Solo"].total

    # The actual standings total must not DROP after picking a pitcher
    # that the recommender said was positive. If it drops, the bench-
    # padding mismatch bug caused the standings path to lose a bench
    # hitter (instead of a pitcher placeholder), making the trade look
    # worse than the recommender promised.
    assert total_after >= total_before, (
        f"Recommended pick {top_pitcher.name} (delta={top_pitcher.immediate_delta:+.3f}) caused "
        f"projected total to DROP from {total_before:.3f} to {total_after:.3f} — "
        f"recs/standings disagree about what gets displaced."
    )
