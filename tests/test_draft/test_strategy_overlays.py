from fantasy_baseball.draft.recommend import RankedPick
from fantasy_baseball.draft.strategy import (
    BALANCED_MAX_SKEW,
    CLOSER_DEADLINE_ROUND,
    FOUR_CLOSERS_DEADLINES,
    NO_PUNT_CAP3_TARGET,
    NO_PUNT_STAGGER_DEADLINES,
    NO_PUNT_STAGGER_TARGET,
    OVERLAYS,
    THREE_CLOSERS_DEADLINES,
    THREE_CLOSERS_TARGET,
    TWO_CLOSERS_DEADLINES,
)
from fantasy_baseball.models.player import PlayerType
from fantasy_baseball.models.positions import Position


def _pick(name, score, pos=Position.OF):
    return RankedPick(
        player_id=name,
        name=name,
        positions=[pos],
        player_type=PlayerType.HITTER,
        score=score,
        metrics={"immediate_delta": score},
    )


def _closer(name, score, sv):
    return RankedPick(
        player_id=name,
        name=name,
        positions=[Position.RP],
        player_type=PlayerType.PITCHER,
        score=score,
        metrics={"immediate_delta": score},
        per_category={"SV": sv},
    )


def test_default_overlay_defers_to_slot_gate():
    # default applies NO constraint -- it returns None so recommend()'s
    # select_from_ranked makes the position-aware greedy pick (verdict winner).
    ranked = [_pick("A", 5.0), _pick("B", 3.0)]
    assert OVERLAYS["default"](ranked, roster_state=None, config=None) is None


# ---------------------------------------------------------------------------
# nonzero_sv overlay
# ---------------------------------------------------------------------------


def test_nonzero_sv_skips_zero_save_relievers_for_closer_slot():
    # Middle reliever scores higher but has SV=0 -- overlay must skip it.
    ranked = [_closer("MiddleReliever", 9.0, 0.0), _closer("Closer", 4.0, 30.0)]
    chosen = OVERLAYS["nonzero_sv"](
        ranked,
        roster_state=None,
        config=None,
        current_round=CLOSER_DEADLINE_ROUND,
        closer_count=0,
    )
    assert chosen is not None
    assert chosen.name == "Closer"


def test_nonzero_sv_defers_before_deadline():
    # Before the deadline, even with no closer, should defer (return None).
    ranked = [_closer("MiddleReliever", 9.0, 0.0), _closer("Closer", 4.0, 30.0)]
    result = OVERLAYS["nonzero_sv"](
        ranked,
        roster_state=None,
        config=None,
        current_round=CLOSER_DEADLINE_ROUND - 1,
        closer_count=0,
    )
    assert result is None


def test_nonzero_sv_defers_when_closer_already_drafted():
    # Already have a closer -- no forcing needed regardless of round.
    ranked = [_closer("AnotherCloser", 5.0, 25.0)]
    result = OVERLAYS["nonzero_sv"](
        ranked,
        roster_state=None,
        config=None,
        current_round=CLOSER_DEADLINE_ROUND + 5,
        closer_count=1,
    )
    assert result is None


def test_nonzero_sv_sv_zero_treated_as_zero_not_falsy():
    # SV=0.0 must NOT be treated as missing -- the MR with SV=0.0 must be skipped.
    ranked = [_closer("MR", 10.0, 0.0)]
    result = OVERLAYS["nonzero_sv"](
        ranked,
        roster_state=None,
        config=None,
        current_round=CLOSER_DEADLINE_ROUND,
        closer_count=0,
    )
    # No real closer available -- overlay returns None (can't force what's not there).
    assert result is None


# ---------------------------------------------------------------------------
# two_closers overlay
# ---------------------------------------------------------------------------


def test_two_closers_forces_first_closer_at_first_deadline():
    deadline = TWO_CLOSERS_DEADLINES[0]  # 8
    ranked = [_closer("C1", 5.0, 35.0), _closer("C2", 3.0, 28.0)]
    chosen = OVERLAYS["two_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=0,
    )
    assert chosen is not None and chosen.name == "C1"


def test_two_closers_defers_before_first_deadline():
    deadline = TWO_CLOSERS_DEADLINES[0]  # 8
    ranked = [_closer("C1", 5.0, 35.0)]
    result = OVERLAYS["two_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline - 1,
        closer_count=0,
    )
    assert result is None


def test_two_closers_forces_second_closer_at_second_deadline():
    deadline = TWO_CLOSERS_DEADLINES[1]  # 14
    ranked = [_closer("C2", 4.0, 22.0)]
    chosen = OVERLAYS["two_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=1,
    )
    assert chosen is not None and chosen.name == "C2"


def test_two_closers_defers_when_target_met():
    # Already have 2 closers -- no forcing at any round.
    ranked = [_closer("C3", 6.0, 30.0)]
    result = OVERLAYS["two_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=20,
        closer_count=2,
    )
    assert result is None


# ---------------------------------------------------------------------------
# three_closers overlay
# ---------------------------------------------------------------------------


def test_three_closers_forces_first_closer_at_first_deadline():
    deadline = THREE_CLOSERS_DEADLINES[0]  # 5
    ranked = [_closer("C1", 7.0, 40.0), _closer("MR", 9.0, 0.0)]
    chosen = OVERLAYS["three_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=0,
    )
    # MR has higher score but SV=0 -- must pick C1
    assert chosen is not None and chosen.name == "C1"


def test_three_closers_defers_before_deadline():
    deadline = THREE_CLOSERS_DEADLINES[0]  # 5
    ranked = [_closer("C1", 7.0, 40.0)]
    result = OVERLAYS["three_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline - 1,
        closer_count=0,
    )
    assert result is None


def test_three_closers_defers_when_target_met():
    ranked = [_closer("C4", 6.0, 30.0)]
    result = OVERLAYS["three_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=20,
        closer_count=3,
    )
    assert result is None


# ---------------------------------------------------------------------------
# four_closers overlay
# ---------------------------------------------------------------------------


def test_four_closers_forces_first_closer_at_first_deadline():
    deadline = FOUR_CLOSERS_DEADLINES[0]  # 5
    ranked = [_closer("C1", 6.0, 35.0)]
    chosen = OVERLAYS["four_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=0,
    )
    assert chosen is not None and chosen.name == "C1"


def test_four_closers_forces_fourth_closer_at_last_deadline():
    deadline = FOUR_CLOSERS_DEADLINES[3]  # 16
    ranked = [_closer("C4", 3.0, 21.0)]
    chosen = OVERLAYS["four_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=3,
    )
    assert chosen is not None and chosen.name == "C4"


def test_four_closers_defers_when_target_met():
    ranked = [_closer("C5", 8.0, 45.0)]
    result = OVERLAYS["four_closers"](
        ranked,
        roster_state=None,
        config=None,
        current_round=20,
        closer_count=4,
    )
    assert result is None


# ---------------------------------------------------------------------------
# no_punt overlay -- documented FALLBACK
# ---------------------------------------------------------------------------


def test_no_punt_overlay_always_defers():
    # Documented FALLBACK: missing team H/AB totals and team_rosters.
    # The overlay must always return None regardless of round or closer count.
    ranked = [_pick("A", 5.0), _closer("C", 3.0, 30.0)]
    assert OVERLAYS["no_punt"](ranked, roster_state=None, config=None) is None
    assert (
        OVERLAYS["no_punt"](
            ranked,
            roster_state=None,
            config=None,
            current_round=15,
            closer_count=0,
        )
        is None
    )


# ---------------------------------------------------------------------------
# no_punt_opp overlay -- documented FALLBACK
# ---------------------------------------------------------------------------


def test_no_punt_opp_overlay_always_defers():
    # Documented FALLBACK: missing team_rosters for opponent-relative SV check
    # and team H/AB for AVG floor filtering.
    ranked = [_pick("A", 5.0), _closer("C", 3.0, 30.0)]
    assert OVERLAYS["no_punt_opp"](ranked, roster_state=None, config=None) is None
    assert (
        OVERLAYS["no_punt_opp"](
            ranked,
            roster_state=None,
            config=None,
            current_round=20,
            closer_count=0,
        )
        is None
    )


# ---------------------------------------------------------------------------
# no_punt_stagger overlay -- PARTIAL PORT (stagger deadlines; AVG floor deferred)
# ---------------------------------------------------------------------------


def test_no_punt_stagger_forces_first_closer_at_first_deadline():
    # Defining behavior: at deadline[0] (round 13) with 0 closers, must force a closer.
    deadline = NO_PUNT_STAGGER_DEADLINES[0]  # 13
    ranked = [_closer("C1", 5.0, 35.0), _closer("MR", 9.0, 0.0)]
    chosen = OVERLAYS["no_punt_stagger"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=0,
    )
    # MR has higher score but SV=0 -- must pick C1
    assert chosen is not None
    assert chosen.name == "C1"


def test_no_punt_stagger_forces_second_closer_at_second_deadline():
    deadline = NO_PUNT_STAGGER_DEADLINES[1]  # 17
    ranked = [_closer("C2", 4.0, 22.0)]
    chosen = OVERLAYS["no_punt_stagger"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=1,
    )
    assert chosen is not None and chosen.name == "C2"


def test_no_punt_stagger_defers_before_first_deadline():
    deadline = NO_PUNT_STAGGER_DEADLINES[0]  # 13
    ranked = [_closer("C1", 5.0, 35.0)]
    result = OVERLAYS["no_punt_stagger"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline - 1,
        closer_count=0,
    )
    assert result is None


def test_no_punt_stagger_defers_when_target_met():
    ranked = [_closer("C4", 6.0, 30.0)]
    result = OVERLAYS["no_punt_stagger"](
        ranked,
        roster_state=None,
        config=None,
        current_round=25,
        closer_count=NO_PUNT_STAGGER_TARGET,
    )
    assert result is None


# ---------------------------------------------------------------------------
# no_punt_cap3 overlay -- PARTIAL PORT (stagger + cap; AVG floor deferred)
# ---------------------------------------------------------------------------


def test_no_punt_cap3_forces_first_closer_at_first_deadline():
    # Same staggered deadlines as no_punt_stagger.
    deadline = NO_PUNT_STAGGER_DEADLINES[0]  # 13
    ranked = [_closer("C1", 5.0, 35.0)]
    chosen = OVERLAYS["no_punt_cap3"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=0,
    )
    assert chosen is not None and chosen.name == "C1"


def test_no_punt_cap3_defers_when_cap_reached():
    # Hard cap: at NO_PUNT_CAP3_TARGET closers, defer regardless of round.
    ranked = [_closer("C4", 8.0, 40.0)]
    result = OVERLAYS["no_punt_cap3"](
        ranked,
        roster_state=None,
        config=None,
        current_round=20,
        closer_count=NO_PUNT_CAP3_TARGET,
    )
    assert result is None


def test_no_punt_cap3_defers_before_deadline():
    deadline = NO_PUNT_STAGGER_DEADLINES[0]  # 13
    ranked = [_closer("C1", 5.0, 35.0)]
    result = OVERLAYS["no_punt_cap3"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline - 1,
        closer_count=0,
    )
    assert result is None


# ---------------------------------------------------------------------------
# avg_hedge overlay -- documented FALLBACK
# ---------------------------------------------------------------------------


def test_avg_hedge_overlay_always_defers():
    # Documented FALLBACK: needs team accumulated H/AB (balance.get_avg_components()).
    ranked = [_pick("A", 5.0), _pick("B", 3.0)]
    assert OVERLAYS["avg_hedge"](ranked, roster_state=None, config=None) is None


# ---------------------------------------------------------------------------
# avg_anchor overlay -- documented FALLBACK
# ---------------------------------------------------------------------------


def test_avg_anchor_overlay_always_defers():
    # Documented FALLBACK: needs candidate's absolute AVG (board['avg']),
    # not available via per_category (which carries marginal roto deltas).
    ranked = [_pick("HighAVG", 5.0), _pick("LowAVG", 3.0)]
    assert OVERLAYS["avg_anchor"](ranked, roster_state=None, config=None) is None
    assert (
        OVERLAYS["avg_anchor"](
            ranked,
            roster_state=None,
            config=None,
            hitter_count=0,
        )
        is None
    )


# ---------------------------------------------------------------------------
# closers_avg overlay -- COMPOSED (closer gate ported; AVG anchor deferred)
# ---------------------------------------------------------------------------


def test_closers_avg_forces_closer_at_deadline():
    # Defining behavior: closer deadline fires at THREE_CLOSERS_DEADLINES[0] (round 5).
    deadline = THREE_CLOSERS_DEADLINES[0]  # 5
    ranked = [_closer("MR", 9.0, 0.0), _closer("C1", 5.0, 35.0)]
    chosen = OVERLAYS["closers_avg"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline,
        closer_count=0,
    )
    # MR has higher score but SV=0; must pick C1
    assert chosen is not None
    assert chosen.name == "C1"


def test_closers_avg_defers_before_deadline():
    # Before closer deadline and target not met -- defer.
    deadline = THREE_CLOSERS_DEADLINES[0]  # 5
    ranked = [_closer("C1", 5.0, 35.0)]
    result = OVERLAYS["closers_avg"](
        ranked,
        roster_state=None,
        config=None,
        current_round=deadline - 1,
        closer_count=0,
    )
    assert result is None


def test_closers_avg_defers_when_target_met():
    # Once THREE_CLOSERS_TARGET closers are drafted, no closer forcing.
    ranked = [_closer("C4", 6.0, 30.0)]
    result = OVERLAYS["closers_avg"](
        ranked,
        roster_state=None,
        config=None,
        current_round=20,
        closer_count=THREE_CLOSERS_TARGET,
    )
    assert result is None


def test_closers_avg_defers_between_deadlines_no_avg_anchor():
    # Between deadlines with target not yet met but no deadline firing -- defer.
    # (AVG anchor is omitted; this confirms we don't spuriously pick based on missing signal.)
    ranked = [_pick("Hitter", 8.0), _closer("C1", 5.0, 35.0)]
    result = OVERLAYS["closers_avg"](
        ranked,
        roster_state=None,
        config=None,
        current_round=THREE_CLOSERS_DEADLINES[0] - 1,
        closer_count=0,
    )
    assert result is None


# ---------------------------------------------------------------------------
# balanced overlay -- PORTED
# ---------------------------------------------------------------------------


def _pitcher(name, score):
    return RankedPick(
        player_id=name,
        name=name,
        positions=[Position.SP],
        player_type=PlayerType.PITCHER,
        score=score,
        metrics={"immediate_delta": score},
    )


def test_balanced_forces_hitter_when_pitchers_dominate():
    # n_pitchers - n_hitters > BALANCED_MAX_SKEW (2) -> force a hitter.
    ranked = [_pitcher("SP1", 9.0), _pitcher("SP2", 7.0), _pick("H1", 5.0)]
    chosen = OVERLAYS["balanced"](
        ranked,
        roster_state=None,
        config=None,
        n_hitters=1,
        n_pitchers=1 + BALANCED_MAX_SKEW + 1,
    )
    assert chosen is not None
    assert chosen.name == "H1"
    assert chosen.player_type == PlayerType.HITTER


def test_balanced_forces_pitcher_when_hitters_dominate():
    # n_hitters - n_pitchers > BALANCED_MAX_SKEW (2) -> force a pitcher.
    ranked = [_pick("H1", 9.0), _pick("H2", 7.0), _pitcher("SP1", 5.0)]
    chosen = OVERLAYS["balanced"](
        ranked,
        roster_state=None,
        config=None,
        n_hitters=1 + BALANCED_MAX_SKEW + 1,
        n_pitchers=1,
    )
    assert chosen is not None
    assert chosen.name == "SP1"
    assert chosen.player_type == PlayerType.PITCHER


def test_balanced_defers_when_no_skew():
    # Exactly at balance limit -- no skew beyond BALANCED_MAX_SKEW, so defer.
    ranked = [_pick("H1", 9.0), _pitcher("SP1", 5.0)]
    result = OVERLAYS["balanced"](
        ranked,
        roster_state=None,
        config=None,
        n_hitters=5,
        n_pitchers=5,
    )
    assert result is None


def test_balanced_defers_when_counts_equal():
    ranked = [_pick("H1", 9.0), _pitcher("SP1", 5.0)]
    result = OVERLAYS["balanced"](
        ranked,
        roster_state=None,
        config=None,
        n_hitters=0,
        n_pitchers=0,
    )
    assert result is None


def test_balanced_picks_highest_score_of_forced_type():
    # When forcing a hitter, must pick the highest-score hitter, not just any hitter.
    ranked = [
        _pitcher("SP1", 10.0),
        _pick("H_Low", 4.0),
        _pick("H_High", 6.0),
    ]
    # Re-order so H_High is first hitter in the ranked list for clarity.
    ranked = [
        _pitcher("SP1", 10.0),
        _pick("H_High", 6.0),
        _pick("H_Low", 4.0),
    ]
    chosen = OVERLAYS["balanced"](
        ranked,
        roster_state=None,
        config=None,
        n_hitters=0,
        n_pitchers=BALANCED_MAX_SKEW + 1,
    )
    # The overlay iterates ranked in order and picks the FIRST matching type.
    # H_High appears before H_Low in ranked, so it must be chosen.
    assert chosen is not None
    assert chosen.name == "H_High"


# ---------------------------------------------------------------------------
# anti_fragile overlay -- documented FALLBACK
# ---------------------------------------------------------------------------


def test_anti_fragile_overlay_always_defers():
    # Documented FALLBACK: needs candidate's absolute IP projection to apply
    # the high-IP penalty (ANTI_FRAGILE_IP_THRESHOLD=170, discount=25%/30IP).
    # IP is not a roto category and is absent from per_category.
    ranked = [_pitcher("Ace", 9.0), _pitcher("Workhorse", 7.0)]
    assert OVERLAYS["anti_fragile"](ranked, roster_state=None, config=None) is None
