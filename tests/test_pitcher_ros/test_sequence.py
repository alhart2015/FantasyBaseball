import numpy as np
import pandas as pd
import pytest
import torch

from fantasy_baseball.hitter_ros.net import device
from fantasy_baseball.hitter_ros.sequence import BATCH_FEATURES, SequenceBatcher
from fantasy_baseball.pitcher_ros.features import answers, input_frame
from fantasy_baseball.pitcher_ros.model import fit_season, plain_config
from fantasy_baseball.pitcher_ros.sequence import (
    TOKEN_FEATURES,
    TOKEN_ORDER,
    appearance_tokens,
    build_appearance_tokens,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS
from fantasy_baseball.pitcher_ros.table import build_table

from .test_table import STARTER, _seasons, _write


def _box(player, day, pk, **stats):
    row = dict.fromkeys(
        (
            "gs",
            "gf",
            "w",
            "l",
            "sv",
            "hld",
            "bs",
            "outs",
            "bf",
            "k",
            "bb",
            "hbp",
            "h",
            "hr",
            "er",
            "pitches",
            "strikes",
        ),
        0,
    )
    row.update(player_id=player, game_date=pd.Timestamp(day), game_pk=pk, **stats)
    return row


def _tracked(player, pk, **sums):
    row = dict.fromkeys(
        (
            "n",
            "fb_velo",
            "fb_velo_max",
            "fb_spin",
            "whiffs",
            "called",
            "in_zone",
            "out_zone",
            "chases",
            "fb",
            "fc",
            "brk",
            "off",
        ),
        0,
    )
    row.update(player_id=player, game_pk=pk, **sums)
    return row


def test_tokens_read_each_outing_against_the_league_known_before_it():
    box = pd.DataFrame(
        [
            # 2024: the league (one pitcher) struck out 20 of 100, 10 ER in 75 outs.
            _box(2, "2024-09-01", 1, bf=100, k=20, outs=75, er=10, pitches=400),
            # Out of order on purpose: tokens come back oldest first.
            _box(1, "2025-04-07", 12, bf=4, k=2, outs=3, pitches=15, strikes=10, sv=1, gf=1),
            _box(1, "2025-04-01", 11, gs=1, bf=24, k=6, outs=18, er=2, pitches=90, strikes=60,
                 w=1),
            # Same day as pitcher 1's start: not yet known when he pitched.
            _box(2, "2025-04-01", 13, bf=10, k=1, outs=6, pitches=40),
        ]
    )  # fmt: skip
    tracked = pd.DataFrame(
        [
            _tracked(2, 1, n=400, whiffs=40, called=60),
            _tracked(1, 11, n=90, fb_velo=96.0, fb_velo_max=98.0, fb_spin=2400.0, whiffs=12,
                     called=15, in_zone=45, out_zone=40, chases=12, fb=54, brk=27, off=9),
        ]
    )  # fmt: skip
    t = appearance_tokens(box, tracked).set_index("game_pk")
    start, relief = t.loc[11], t.loc[12]
    assert (start.started, start.w, start.first_seen, start.rest) == (1, 1, 1, 0)
    assert start.bf == pytest.approx(24 / 30) and start.strike_share == pytest.approx(60 / 90)
    # Against 2024's league only: 6 K where 0.2 x 24 = 4.8 were expected.
    assert start.k_vs_lg == pytest.approx((6 - 0.2 * 24) / 10)
    assert start.er_vs_lg == pytest.approx((2 - 18 * 10 / 75) / 7)
    assert start.whiff_vs_lg == pytest.approx(12 / 90 - 40 / 400)
    assert start.csw_vs_lg == pytest.approx(27 / 90 - 100 / 400)
    assert start.fb_velo == pytest.approx((96 - 93) / 5) and start.spin_tracked == 1
    assert start.chase_rate == pytest.approx(12 / 40) and start.fb_share == pytest.approx(0.6)
    # A week later the league also has April 1's games: (20 + 6 + 1) / (100 + 24 + 10).
    assert relief.k_vs_lg == pytest.approx((2 - 4 * 27 / 134) / 10)
    assert relief.rest == pytest.approx(np.log1p(6) / 5)
    assert (relief.started, relief.finished, relief.sv, relief.first_seen) == (0, 1, 1, 0)
    # Untracked: no tracking features, not a guess.
    assert relief.tracked == 0 and relief.fb_velo == 0 and relief.whiff_vs_lg == 0
    # The store's first day has no league known before it.
    assert t.loc[1].k_vs_lg == 0 and t.loc[1].whiff_vs_lg == 0
    assert set(TOKEN_FEATURES) <= set(t.columns)


def test_a_lob_from_a_position_player_is_clipped():
    box = pd.DataFrame([_box(9, "2025-04-01", 11, bf=6, pitches=20)])
    tracked = pd.DataFrame([_tracked(9, 11, n=20, fb_velo=45.0, fb_velo_max=50.0)])
    t = appearance_tokens(box, tracked)
    assert t.fb_velo.iloc[0] == -4 and t.fb_velo_max.iloc[0] == -4


@pytest.fixture(scope="module")
def store(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    _write(root, _seasons())
    return root


def test_each_row_reads_the_outings_before_its_date(store):
    table = build_table(store)
    tokens = build_appearance_tokens(store)
    starter = tokens[tokens.player_id == STARTER]
    assert len(starter) == 10 + 21 and starter.tracked.all()
    b = SequenceBatcher(tokens, table, 40, torch.device("cpu"), TOKEN_FEATURES, TOKEN_ORDER)
    assert b.n_features == len(TOKEN_FEATURES) + len(BATCH_FEATURES)
    rows = table.index[(table.player_id == STARTER) & (table.season == 2025)].to_numpy()
    weeks = table.loc[rows, "week"].to_numpy()
    # Week w of 2025: all 10 of 2024's starts, and the 7 w daily ones before its date.
    assert b.lengths(torch.as_tensor(rows)).tolist() == [10 + 7 * w for w in weeks]


def test_the_sequence_net_trains_and_predicts_every_row(store):
    table = build_table(store)
    x, (y, w) = input_frame(table), answers(table)
    tokens = build_appearance_tokens(store)
    batcher = SequenceBatcher(tokens, table, 16, device(), TOKEN_FEATURES, TOKEN_ORDER)
    config = plain_config(
        hidden=[8],
        max_epochs=2,
        patience=1,
        val_frac=0.34,
        seq="transformer",
        seq_len=16,
        seq_dim=8,
    )
    preds, _ = fit_season(table, x, y, w, 2025, config, batcher=batcher)
    assert len(preds) == (table.season == 2025).sum()
    assert preds[list(TARGETS)].notna().all().all()
    with pytest.raises(ValueError, match="needs a batcher"):
        fit_season(table, x, y, w, 2025, config)
    with pytest.raises(ValueError, match="needs a batcher"):
        fit_season(table, x, y, w, 2025, plain_config(hidden=[8]), batcher=batcher)
