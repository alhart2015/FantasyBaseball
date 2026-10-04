"""#417: probe features read off the pretrained pitch model."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fantasy_baseball.hitter_ros.net import device  # noqa: E402
from fantasy_baseball.hitter_ros.pitch_tokens import (  # noqa: E402
    CONTEXT_FEATURES,
    OUTCOMES,
    PITCH_GROUPS,
)
from fantasy_baseball.hitter_ros.pretrain import PitchStore, PretrainModel  # noqa: E402
from fantasy_baseball.hitter_ros.probes import (  # noqa: E402
    PITCH_SHAPES,
    PROBE_FEATURES,
    PROBES,
    _bats_left,
    _probs_to_features,
    check_probes,
    probe_contexts,
    probe_features,
    probe_inputs,
    reference_cohort,
    side_prefix,
    standardize,
)
from tests.test_hitter_ros.test_pretrain import _small, _tokens  # noqa: E402

WHIFF = OUTCOMES.index("whiff")
BALL = OUTCOMES.index("ball")


def _store(tokens):
    return PitchStore(tokens, device(), torch.float32)


def _model(window=8):
    torch.manual_seed(0)
    return PretrainModel(_small(window=window)).to(device()).eval()


def _features(tokens, as_of, window=8):
    return probe_features(
        _model(window),
        _store(tokens),
        np.array([1]),
        pd.Series([pd.Timestamp(as_of)]),
        window=window,
        amp=False,
    ).iloc[0]


def _history(n, outcome=BALL):
    return {1: [(2025, d, outcome if d % 2 else WHIFF) for d in range(n)]}


def test_probes_ignore_pitches_on_or_after_the_date():
    base = _tokens(_history(20))
    as_of = "2025-04-11"  # pitches on days 0-9 are before it
    future = base.copy()
    later = future["game_date"] >= pd.Timestamp(as_of)
    future.loc[later, "speed"] = 3.0
    future.loc[later, [f"out_{o}" for o in OUTCOMES]] = 0.0
    future.loc[later, "out_bip_barrel"] = 1.0
    pd.testing.assert_series_equal(_features(base, as_of), _features(future, as_of))
    # Sanity: changing a pitch before the date does change them.
    past = base.copy()
    past.loc[past["game_date"] == pd.Timestamp("2025-04-10"), "speed"] = 3.0
    assert not np.allclose(_features(base, as_of), _features(past, as_of))


def test_history_is_the_last_window_minus_one_pitches():
    base = _tokens(_history(20))
    as_of = "2025-04-21"
    old = base.copy()
    # window 8 reads the last 7 pitches (days 13-19); days before 13 must not matter.
    old.loc[old["game_date"] < pd.Timestamp("2025-04-14"), "speed"] = 3.0
    pd.testing.assert_series_equal(_features(base, as_of), _features(old, as_of))


def test_a_row_without_history_is_blank():
    feats = _features(_tokens(_history(5)), "2025-04-01")  # nothing before day 0
    assert feats.isna().all()


def test_every_probe_has_a_pitch_shape_and_a_full_context():
    groups = {p.group for p in PROBES}
    assert groups <= set(PITCH_GROUPS)
    assert all((g, hand) in PITCH_SHAPES for g in groups for hand in (0, 1))
    ctx = probe_contexts(1.0)
    col = {c: i for i, c in enumerate(CONTEXT_FEATURES)}
    assert ctx.shape == (len(PROBES), len(CONTEXT_FEATURES))
    assert (ctx[:, col["bats_left"]] == 1.0).all()
    # Exactly one pitch-type flag per probe; LHP flag matches the probe.
    type_cols = [col[f"pt_{g}"] for g in (*PITCH_GROUPS, "other")]
    assert (ctx[:, type_cols].sum(axis=1) == 1).all()
    assert list(ctx[:, col["vs_lhp"]]) == [float(p.lhp) for p in PROBES]


def test_a_switch_hitter_bats_from_the_side_he_used_against_each_hand():
    tokens = _tokens({1: [(2025, d, BALL) for d in range(6)]})
    tokens["vs_lhp"] = [0, 0, 0, 1, 1, 1]
    tokens["bats_left"] = [1, 1, 1, 0, 0, 0]  # left vs RHP, right vs LHP
    store = _store(tokens)
    sides = _bats_left(side_prefix(store), np.array([0]), np.array([6]))
    assert sides[0][0] == 1.0 and sides[1][0] == 0.0
    # Never faced a lefty: fall back on his overall side.
    only_rhp = _bats_left(side_prefix(store), np.array([0]), np.array([3]))
    assert only_rhp[1][0] == 1.0


def test_features_from_known_probabilities():
    probs = np.zeros((1, len(PROBES), len(OUTCOMES)))
    probs[..., OUTCOMES.index("whiff")] = 0.25
    probs[..., OUTCOMES.index("bip_barrel")] = 0.25
    probs[..., OUTCOMES.index("bip_weak")] = 0.25
    probs[..., BALL] = 0.25
    f = _probs_to_features(probs).iloc[0]
    assert list(f.index) == list(PROBE_FEATURES)
    assert f["probe_zone_swing_fb"] == pytest.approx(0.75)
    assert f["probe_zone_whiff_brk"] == pytest.approx(1 / 3)
    assert f["probe_zone_barrel_off"] == pytest.approx(0.5)
    assert f["probe_platoon_zone_whiff"] == pytest.approx(0.0)


def test_probe_inputs_align_to_the_table_and_leave_gaps_blank():
    table = pd.DataFrame(
        {"player_id": [1, 2, 1], "season": [2025, 2025, 2025], "week": [0, 0, 1]},
        index=[10, 11, 12],
    )
    probes = pd.DataFrame({"player_id": [1], "season": [2025], "week": [1]})
    for c in PROBE_FEATURES:
        probes[c] = 0.5
    x = probe_inputs(table, probes)
    assert list(x.index) == [10, 11, 12]
    assert x.loc[12].eq(0.5).all() and x.loc[[10, 11]].isna().all().all()


def _table_and_probes():
    table = pd.DataFrame(
        {
            "player_id": [1, 2],
            "season": [2025, 2025],
            "week": [0, 0],
            "as_of": pd.to_datetime(["2025-03-27", "2025-03-27"]),
        }
    )
    probes = table.copy()
    for c in PROBE_FEATURES:
        probes[c] = 0.0
    return table, probes


def test_check_probes_catches_gaps_duplicates_and_moved_dates():
    table, probes = _table_and_probes()
    assert check_probes(table, probes) is None
    assert "lacks 1" in check_probes(table, probes.iloc[:1])
    assert "duplicate" in check_probes(table, pd.concat([probes, probes.iloc[:1]]))
    moved = probes.assign(as_of=pd.Timestamp("2025-03-28"))
    assert "different as-of" in check_probes(table, moved)


def test_first_contact_season():
    from fantasy_baseball.hitter_ros.probes import CONTACT_FEATURES, first_contact_season

    tokens = pd.DataFrame(
        {"season": [2014, 2015, 2016], "out_bip_solid": [0, 1, 0], "out_bip_barrel": [0, 0, 1]}
    )
    assert first_contact_season(tokens) == 2015
    assert "probe_zone_hard_fb" in CONTACT_FEATURES
    assert "probe_zone_whiff_fb" not in CONTACT_FEATURES


def test_standardize_uses_the_reference_group_only():
    reference = pd.DataFrame({"a": [1.0, 3.0], "b": [2.0, 2.0]})
    feats = pd.DataFrame({"a": [5.0], "b": [7.0]})
    out = standardize(feats, reference)
    assert out.loc[0, "a"] == pytest.approx((5 - 2) / np.std([1, 3], ddof=1))
    assert np.isnan(out.loc[0, "b"])  # no spread in the reference: blank, not infinite


def test_reference_cohort_is_last_seasons_table_hitters():
    table = pd.DataFrame({"player_id": [1, 1, 2, 3], "season": [2024, 2024, 2025, 2024]})
    assert sorted(reference_cohort(table, 2025)) == [1, 3]
