"""#492: pitcher probe features read off the pitcher-side pretrained pitch model."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fantasy_baseball.hitter_ros.net import device  # noqa: E402
from fantasy_baseball.hitter_ros.pitch_tokens import CONTEXT_FEATURES, OUTCOMES  # noqa: E402
from fantasy_baseball.hitter_ros.pretrain import PitchStore, PretrainModel  # noqa: E402
from fantasy_baseball.hitter_ros.probes import PITCH_SHAPES  # noqa: E402
from fantasy_baseball.pitcher_ros.probes import (  # noqa: E402
    PROBE_FEATURES,
    PROBES,
    _probs_to_features,
    probe_contexts,
    probe_features,
    reference_cohort,
)
from tests.test_hitter_ros.test_pretrain import _small, _tokens  # noqa: E402

WHIFF = OUTCOMES.index("whiff")
BALL = OUTCOMES.index("ball")
COL = {c: i for i, c in enumerate(CONTEXT_FEATURES)}


def _features(tokens, as_of, window=8, player=1):
    torch.manual_seed(0)
    model = PretrainModel(_small(window=window)).to(device()).eval()
    return probe_features(
        model,
        PitchStore(tokens, device(), torch.float32),
        np.array([player]),
        pd.Series([pd.Timestamp(as_of)]),
        window=window,
        amp=False,
    ).iloc[0]


def _history(n, player=1):
    return {player: [(2025, d, BALL if d % 2 else WHIFF) for d in range(n)]}


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


def test_a_pitcher_without_history_is_blank():
    feats = _features(_tokens(_history(5)), "2025-04-01")  # his first pitch is that day
    assert feats.isna().all()


def test_his_hand_picks_the_probe_shapes():
    righty = _tokens(_history(10))
    lefty = righty.assign(vs_lhp=1.0)
    as_of = "2025-04-20"
    assert not np.allclose(_features(righty, as_of), _features(lefty, as_of))
    for lhp in (0, 1):
        ctx = probe_contexts(lhp)
        assert (ctx[:, COL["vs_lhp"]] == lhp).all()
        for k, p in enumerate(PROBES):
            assert ctx[k, COL["speed"]] == PITCH_SHAPES[(p.group, lhp)][0]
            assert ctx[k, COL["bats_left"]] == p.bats_left
            assert ctx[k, COL[f"pt_{p.group}"]] == 1


def test_every_probe_is_thrown_to_both_sides():
    lefties = {(p.group, p.location, p.count) for p in PROBES if p.bats_left}
    righties = {(p.group, p.location, p.count) for p in PROBES if not p.bats_left}
    assert lefties == righties and len(PROBES) == 2 * len(lefties)


def test_platoon_gap_is_opposite_side_batters_minus_same_side():
    probs = np.full((2, len(PROBES), len(OUTCOMES)), 0.1, dtype=np.float32)
    whiff_heavy = probs[0].copy()
    whiff_heavy[:, WHIFF] = 0.5
    for k, p in enumerate(PROBES):
        if p.bats_left:  # lefty batters whiff more, for both pitchers
            probs[:, k] = whiff_heavy[k]
    feats = _probs_to_features(probs, np.array([0.0, 1.0]))
    assert list(feats.columns) == list(PROBE_FEATURES)
    gap = feats["probe_platoon_zone_whiff"]
    # A righty gets more whiffs from opposite-side (left) batters; a lefty fewer.
    assert gap[0] > 0 and gap[1] == pytest.approx(-gap[0])


def test_reference_cohort_is_last_seasons_table_pitchers():
    table = pd.DataFrame({"player_id": [1, 2, 2, 3], "season": [2024, 2024, 2024, 2025]})
    assert sorted(reference_cohort(table, 2025)) == [1, 2]
