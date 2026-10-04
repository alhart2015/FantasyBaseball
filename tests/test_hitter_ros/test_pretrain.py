import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

from fantasy_baseball.hitter_ros.net import device  # noqa: E402
from fantasy_baseball.hitter_ros.pitch_tokens import OUTCOMES, TOKEN_FEATURES  # noqa: E402
from fantasy_baseball.hitter_ros.pretrain import (  # noqa: E402
    N_TOKEN,
    PitchStore,
    PretrainConfig,
    PretrainModel,
    count_baseline_ce,
    next_pitch_loss,
    pretrain,
)

WHIFF = OUTCOMES.index("whiff")
BALL = OUTCOMES.index("ball")


def _tokens(spec):
    """spec: {player: [(season, day_offset, outcome), ...]} -> token frame."""
    rows = []
    for player, pitches in spec.items():
        for i, (season, day, outcome) in enumerate(pitches):
            row = {f: 0.0 for f in TOKEN_FEATURES}
            row.update(
                player_id=player,
                game_date=pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=day),
                season=season,
                game_pk=season * 1000 + day,
                at_bat_number=i,
                pitch_number=1,
                outcome=outcome,
            )
            row[f"out_{OUTCOMES[outcome]}"] = 1.0
            rows.append(row)
    return pd.DataFrame(rows)


def _small(**kw):
    base = dict(dim=16, layers=1, heads=2, window=8, batch_size=8, dropout=0.0, amp=False)
    base.update(kw)
    return PretrainConfig(**base)


def test_model_is_causal():
    """Changing pitches after t must not change the prediction made at t."""
    torch.manual_seed(0)
    model = PretrainModel(_small()).eval()
    x = torch.randn(1, 8, N_TOKEN)
    changed = x.clone()
    changed[:, 5:] = torch.randn(1, 3, N_TOKEN)
    with torch.no_grad():
        a, b = model(x), model(changed)
    # Logit t predicts pitch t+1 from pitches <= t and pitch t+1's context; t <= 3
    # only uses pitches 0..4, which are unchanged.
    torch.testing.assert_close(a[:, :4], b[:, :4])
    assert not torch.allclose(a[:, 5:], b[:, 5:])


def test_loss_skips_padding():
    logits = torch.zeros(1, 3, len(OUTCOMES))
    y = torch.tensor([[0, WHIFF, -1, -1]])
    ce, n = next_pitch_loss(logits, y)
    assert n.item() == 1
    assert ce.item() == pytest.approx(np.log(len(OUTCOMES)))


def test_windows_stop_before_the_cutoff_season():
    tokens = _tokens(
        {1: [(2023, d, BALL) for d in range(10)] + [(2024, d, WHIFF) for d in range(10)]}
    )
    store = PitchStore(tokens, torch.device("cpu"))
    starts, lengths = store.windows(np.array([1]), before_season=2024, length=4)
    assert len(starts) and (starts + lengths).max() <= 10  # nothing reaches a 2024 pitch
    assert 0 in starts and 6 in starts  # first and newest pitches covered
    assert set(lengths) == {4}


def test_short_career_window_stops_at_the_cutoff():
    tokens = _tokens({1: [(2023, d, BALL) for d in range(3)] + [(2024, 0, WHIFF)]})
    store = PitchStore(tokens, torch.device("cpu"))
    starts, lengths = store.windows(np.array([1, 99]), before_season=2024, length=8)
    assert starts.tolist() == [0] and lengths.tolist() == [3]  # unknown hitter 99 skipped


def test_history_is_strictly_before_the_date():
    tokens = _tokens({1: [(2024, d, BALL) for d in range(10)], 2: [(2024, 5, WHIFF)]})
    store = PitchStore(tokens, torch.device("cpu"))
    as_of = np.array([pd.Timestamp("2024-04-06")] * 3, dtype="datetime64[D]").astype(np.int64)
    start, n = store.history(np.array([1, 2, 3]), as_of, length=3)
    # Player 1: days 0-4 are before Apr 6 -> last three = days 2, 3, 4. Player 2's only
    # pitch is on the date; player 3 is unknown.
    assert n.tolist() == [3, 0, 0] and start[0] == 2


def test_pitch_store_keeps_float32_when_asked():
    tokens = _tokens({1: [(2024, 0, BALL)]})
    assert PitchStore(tokens, torch.device("cpu"), torch.float32).feats.dtype == torch.float32


def test_config_rejects_no_validation_hitters():
    with pytest.raises(ValueError, match="val_frac"):
        PretrainConfig(val_frac=0.0)


def test_count_baseline():
    tokens = pd.DataFrame({"balls": [0.0] * 4, "strikes": [0.0] * 4, "outcome": [0, 0, 1, 1]})
    assert count_baseline_ce(tokens) == pytest.approx(np.log(2))


def test_pretraining_learns_a_hitter_pattern_the_count_cannot_see():
    """Half the hitters always whiff, half always take a ball. History reveals which."""
    rng = np.random.default_rng(0)
    spec = {
        p: [(2023, d, WHIFF if p % 2 else BALL) for d in range(int(rng.integers(20, 40)))]
        for p in range(80)
    }
    tokens = _tokens(spec)
    store = PitchStore(tokens, device())
    config = _small(max_epochs=15, patience=5, lr=3e-3, warmup_steps=5, val_frac=0.2)
    result = pretrain(store, before_season=2024, config=config)
    assert result.baseline_ce == pytest.approx(np.log(2), abs=0.05)
    assert min(result.val_ce) < 0.5 * result.baseline_ce


def test_unknown_descriptions_fall_back_on_statcast_type(caplog):
    from fantasy_baseball.hitter_ros.pitch_tokens import outcome_index

    desc = pd.Series(["ball", "hit_into_play", "new_thing", "hit_into_play"])
    kind = pd.Series(["B", "S", "B", "X"])
    lsa = pd.Series([None, None, None, 6.0])
    idx = outcome_index(desc, lsa, kind)
    assert [OUTCOMES[i] for i in idx] == ["ball", "called_strike", "ball", "bip_barrel"]
    assert "unrecognized descriptions" in caplog.text


def test_season_ce_scores_each_season_pitch_once_with_history():
    from fantasy_baseball.hitter_ros.pretrain import season_ce

    spec = {
        1: [(2023, d, BALL) for d in range(30)] + [(2024, d, WHIFF) for d in range(25)],
        2: [(2024, d, BALL) for d in range(5)],  # rookie: first pitch has no history
        3: [(2023, d, BALL) for d in range(5)],  # no 2024 pitches
    }
    store = PitchStore(_tokens(spec), torch.device("cpu"), torch.float32)
    config = _small()
    torch.manual_seed(0)
    _, n = season_ce(PretrainModel(config), store, 2024, config)
    assert n == 24 + 4  # each hitter's first 2024 pitch is skipped
