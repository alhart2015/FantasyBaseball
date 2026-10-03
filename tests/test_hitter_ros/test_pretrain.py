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
    embed_rows,
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
    starts = store.windows(np.array([1]), before_season=2024, length=4)
    assert len(starts) and starts.max() + 4 <= 10  # nothing reaches a 2024 pitch
    assert 0 in starts and 6 in starts  # first and newest pitches covered


def test_embeddings_ignore_pitches_on_or_after_the_date():
    spec = {1: [(2024, d, BALL) for d in range(10)], 2: [(2024, 5, WHIFF)]}
    early = _tokens(spec)
    late = early.copy()
    late.loc[late["game_date"] >= pd.Timestamp("2024-04-06"), "outcome"] = WHIFF
    late.loc[late["game_date"] >= pd.Timestamp("2024-04-06"), "speed"] = 5.0
    rows = pd.DataFrame(
        {
            "player_id": [1, 2, 3],
            "as_of": pd.to_datetime(["2024-04-06", "2024-04-06", "2024-04-06"]),
        }
    )
    config = _small()
    torch.manual_seed(0)
    encoder = PretrainModel(config).encoder
    a = embed_rows(encoder, PitchStore(early, torch.device("cpu")), rows, config)
    b = embed_rows(encoder, PitchStore(late, torch.device("cpu")), rows, config)
    np.testing.assert_allclose(a, b)
    # Player 2's only pitch is on the date, and player 3 has none: no history.
    assert np.all(a[1] == 0) and np.all(a[2] == 0)
    assert np.any(a[0] != 0)
    assert a[0, -1] == pytest.approx(np.log1p(1) / 5)  # newest pitch 1 day before


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
