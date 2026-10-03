from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fantasy_baseball.hitter_ros.net import NetConfig, device, predict, train  # noqa: E402
from fantasy_baseball.hitter_ros.sequence import (  # noqa: E402
    N_TOKEN_FEATURES,
    STORED_FEATURES,
    SequenceBatcher,
    build_pa_tokens,
)
from tests.test_hitter_ros.test_table import _pitch  # noqa: E402

CPU = torch.device("cpu")


def _tokens(player_days):
    """Token frame: {player: [day offsets from 2025-04-01]}; feature 'ev' = day offset/100."""
    rows = []
    for player, days in player_days.items():
        for i, d in enumerate(days):
            row = {f: 0.0 for f in STORED_FEATURES}
            row.update(
                player_id=player,
                game_date=pd.Timestamp("2025-04-01") + pd.Timedelta(days=d),
                season=2025,
                game_pk=d,
                at_bat_number=i,
                ev=d / 100,
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _rows(*pairs):
    return pd.DataFrame(
        [
            {
                "player_id": p,
                "as_of": pd.Timestamp("2025-04-01") + pd.Timedelta(days=d),
                "season": 2025,
            }
            for p, d in pairs
        ]
    )


def test_batch_takes_only_pas_before_the_date_newest_last():
    tokens = _tokens({1: [0, 1, 2, 5, 9], 2: [3]})
    rows = _rows((1, 5), (2, 0), (7, 4))  # player 7 has no PAs at all
    b = SequenceBatcher(tokens, rows, max_len=2, device=CPU)
    x, n = b.batch(torch.arange(3))
    assert n.tolist() == [2, 0, 0]
    ev = STORED_FEATURES.index("ev")
    # Player 1 before day 5: days 0, 1, 2 -> last two, oldest first; day 5 is excluded.
    assert x[0, :, ev].tolist() == pytest.approx([0.01, 0.02])
    assert x.shape == (3, 2, N_TOKEN_FEATURES)
    assert torch.all(x[1:] == 0)
    days_ago = x[0, :, len(STORED_FEATURES)]
    assert days_ago.tolist() == pytest.approx([np.log1p(4) / 5, np.log1p(3) / 5])


def test_shuffle_keeps_the_same_pas():
    tokens = _tokens({1: list(range(30))})
    b = SequenceBatcher(tokens, _rows((1, 25)), max_len=10, device=CPU)
    x, _ = b.batch(torch.arange(1))
    xs, _ = b.batch(torch.arange(1), shuffle_order=True)
    ev = STORED_FEATURES.index("ev")
    assert sorted(xs[0, :, ev].tolist()) == pytest.approx(sorted(x[0, :, ev].tolist()))


def test_sequences_never_see_the_as_of_date_or_later():
    """Change every PA on or after the date; the batch for that date must not move."""
    early = _tokens({1: [0, 1, 2, 3, 4, 5, 6]})
    late = early.copy()
    cut = pd.Timestamp("2025-04-01") + pd.Timedelta(days=4)
    late.loc[late["game_date"] >= cut, "ev"] = 99.0
    rows = _rows((1, 4))
    a, _ = SequenceBatcher(early, rows, 10, CPU).batch(torch.arange(1))
    b, _ = SequenceBatcher(late, rows, 10, CPU).batch(torch.arange(1))
    assert torch.equal(a, b)


def test_build_pa_tokens_from_a_store(tmp_path):
    d = date(2025, 4, 1)
    pitches = [
        _pitch(d, 1, game_pk=10, at_bat_number=1, description="swinging_strike", type="S"),
        _pitch(
            d,
            1,
            game_pk=10,
            at_bat_number=1,
            description="hit_into_play",
            type="X",
            events="home_run",
            launch_speed=105.0,
            launch_angle=28.0,
            hc_x=60.0,
            hc_y=100.0,
            woba_value=2.0,
            outs_when_up=1,
            on_1b=999,
        ),
        _pitch(d + timedelta(days=1), 1, game_pk=11, at_bat_number=1, events="walk"),
    ]
    for p in pitches:
        p.setdefault("events", None)
        p.setdefault("woba_value", 0.7 if p["events"] == "walk" else None)
        p.setdefault("outs_when_up", 0)
        p.setdefault("on_1b", None)
        p.setdefault("on_2b", None)
        p.setdefault("on_3b", None)
        p.setdefault("game_pk", 0)
        p.setdefault("at_bat_number", 0)
    lineups = [
        {"game_pk": 10, "game_date": d.isoformat(), "player_id": 1, "pa": 1},
    ]
    _write_raw(tmp_path, pitches, lineups)
    tokens = build_pa_tokens(tmp_path)
    assert len(tokens) == 2
    hr = tokens.iloc[0]
    assert hr.res_hr == 1 and hr.contact == 1 and hr.ev == pytest.approx(1.05)
    assert hr.n_pitches == pytest.approx(2 / 6) and hr.whiffs == pytest.approx(1 / 3)
    assert hr.on_1b == 1 and hr.outs == pytest.approx(0.5) and hr.spray < 0
    assert tokens.iloc[1].res_bb == 1 and tokens.iloc[1].has_ev == 0


def _write_raw(root, pitches, lineups):
    path = root / "pitches" / "season=2025" / "chunk.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(pitches).to_parquet(path, index=False)
    path = root / "lineups" / "2025.parquet"
    path.parent.mkdir(parents=True)
    pd.DataFrame(lineups).to_parquet(path, index=False)


@pytest.mark.parametrize("kind", ["gru", "transformer"])
def test_hybrid_net_learns_from_the_sequence(kind):
    """The answer depends only on the PAs: a hybrid must beat a plain MLP on it."""
    rng = np.random.default_rng(0)
    n_players = 300
    player_days = {
        p: sorted(rng.choice(60, size=20, replace=False).tolist()) for p in range(n_players)
    }
    tokens = _tokens(player_days)
    # Each player's PAs carry a hidden skill in 'woba_value'.
    skill = rng.normal(size=n_players)
    tokens["woba_value"] = skill[tokens["player_id"]].astype(np.float32)
    rows = _rows(*((p, 61) for p in range(n_players)))
    batcher = SequenceBatcher(tokens, rows, max_len=20, device=device())
    x = rng.normal(size=(n_players, 3)).astype(np.float32)  # static inputs: pure noise
    y = skill[:, None].astype(np.float32)
    w = np.ones_like(y)
    val = np.arange(n_players) < 60
    positions = np.arange(n_players)
    base = dict(hidden=[16], dropout=0.0, lr=3e-3, batch_size=64, max_epochs=40, patience=8)

    seq = train(
        x,
        y,
        w,
        val,
        NetConfig(**base, seq=kind, seq_len=20, seq_dim=16, seq_layers=1),
        rows=positions,
        batcher=batcher,
    )
    plain = train(x, y, w, val, NetConfig(**base))
    assert min(seq.val_loss) < 0.5 * min(plain.val_loss)
    out = predict(seq.model, x, rows=positions, batcher=batcher)
    assert out.shape == (n_players, 1)
