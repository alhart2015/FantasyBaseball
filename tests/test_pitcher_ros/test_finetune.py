"""#492: the pretrained pitch encoder joined to the pitcher net and fine-tuned."""

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from fantasy_baseball.hitter_ros.net import device  # noqa: E402
from fantasy_baseball.hitter_ros.pitch_tokens import OUTCOMES  # noqa: E402
from fantasy_baseball.hitter_ros.pretrain import (  # noqa: E402
    PitchStore,
    PretrainModel,
    player_half,
)
from fantasy_baseball.hitter_ros.sequence import SequenceBatcher  # noqa: E402
from fantasy_baseball.pitcher_ros.features import answers, input_frame  # noqa: E402
from fantasy_baseball.pitcher_ros.finetune import (  # noqa: E402
    FinetuneBatcher,
    FinetuneConfig,
    PretrainRun,
    check_cross,
    check_length,
    cross_encoders,
    model_maker,
)
from fantasy_baseball.pitcher_ros.model import fit_season, plain_config  # noqa: E402
from fantasy_baseball.pitcher_ros.sequence import (  # noqa: E402
    TOKEN_FEATURES,
    TOKEN_ORDER,
    build_appearance_tokens,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS  # noqa: E402
from fantasy_baseball.pitcher_ros.table import build_table  # noqa: E402
from tests.test_hitter_ros.test_pretrain import _small, _tokens  # noqa: E402

from .test_table import RELIEVER, STARTER, _seasons, _write  # noqa: E402

WHIFF, BALL = OUTCOMES.index("whiff"), OUTCOMES.index("ball")
LENGTH = 7  # window 8, less the last position


def _pitches():
    """Two pitches a day for the starter and one for the reliever, on the store's days."""
    spec = {}
    for player, per_day in ((STARTER, 2), (RELIEVER, 1)):
        spec[player] = [
            (season, day, WHIFF if (day + k) % 2 else BALL)
            for season, days in ((2024, 10), (2025, 21))
            for day in range(days)
            for k in range(per_day)
        ]
    return _tokens(spec)


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("store")
    _write(root, _seasons())
    table = build_table(root)
    apps = SequenceBatcher(
        build_appearance_tokens(root), table, 16, device(), TOKEN_FEATURES, TOKEN_ORDER
    )
    runs = tmp_path_factory.mktemp("pretrain")
    config = _small(window=LENGTH + 1)
    (runs / "q" / "2025").mkdir(parents=True)
    torch.manual_seed(0)
    torch.save(PretrainModel(config).state_dict(), runs / "q" / "2025" / "model.pt")
    meta = {"config": config.to_dict(), "seasons": [2025], "tokens": {}}
    (runs / "q" / "run.json").write_text(json.dumps(meta))
    return table, apps, PretrainRun(runs, "q")


def _net(run, lr_scale=0.1):
    config = plain_config(hidden=[8], seq="transformer", seq_len=16, seq_dim=8)
    make = model_maker(run, 2025, FinetuneConfig("q", LENGTH, lr_scale, 4), device())
    return make(5, len(TARGETS), config, len(TOKEN_FEATURES) + 2).to(device()).eval()


def _row(table, player, week):
    hit = table.index[(table.player_id == player) & (table.season == 2025) & (table.week == week)]
    return torch.as_tensor(hit.to_numpy(), device=device())


def test_rows_read_only_pitches_before_their_date(setup):
    table, apps, run = setup
    tokens = _pitches()
    row = _row(table, STARTER, 1)
    as_of = table.loc[int(row[0]), "as_of"]
    before = int(((tokens.player_id == STARTER) & (tokens.game_date < as_of)).sum())
    batcher = FinetuneBatcher(apps, PitchStore(tokens, device()), table, LENGTH)
    (_, _, n, _), _ = batcher.batch(row)
    assert int(n[0]) == min(before, LENGTH)
    net, x = _net(run), torch.zeros(1, 5, device=device())

    def out(frame):
        b = FinetuneBatcher(apps, PitchStore(frame, device()), table, LENGTH)
        seq, lengths = b.batch(row)
        with torch.no_grad():
            return net(x, seq, lengths)

    later = tokens.copy()
    on_or_after = later.game_date >= as_of
    later.loc[on_or_after, "speed"] = 3.0
    later.loc[on_or_after, "out_bip_barrel"] = 1.0
    torch.testing.assert_close(out(tokens), out(later))
    earlier = tokens.copy()
    earlier.loc[(earlier.game_date < as_of) & (earlier.player_id == STARTER), "speed"] = 3.0
    assert not torch.allclose(out(tokens), out(earlier))


def test_a_row_with_no_pitches_reads_a_zero_state(setup):
    table, apps, run = setup
    tokens = _pitches()
    tokens = tokens[tokens.player_id != RELIEVER]
    batcher = FinetuneBatcher(apps, PitchStore(tokens, device()), table, LENGTH)
    seq, lengths = batcher.batch(_row(table, RELIEVER, 0))
    assert int(seq[2][0]) == 0
    net, x = _net(run), torch.zeros(1, 5, device=device())
    with torch.no_grad():
        a = net(x, seq, lengths)
        for p in net.pitches.parameters():
            p.add_(1.0)
        b = net(x, seq, lengths)
    torch.testing.assert_close(a, b)


def test_the_encoder_learns_at_its_share_of_the_rate(setup):
    _, _, run = setup
    net = _net(run, lr_scale=0.1)
    rest, encoder = net.optimizer_groups(1e-3)
    assert rest["lr"] == 1e-3 and encoder["lr"] == pytest.approx(1e-4)
    ids = [id(p) for g in (rest, encoder) for p in g["params"]]
    assert sorted(ids) == sorted(id(p) for p in net.parameters())  # each once
    assert {id(p) for p in encoder["params"]} == {id(p) for p in net.pitches.parameters()}
    # Starts from the pretrained weights of the season's model.
    saved = torch.load(run.dir / "2025" / "model.pt", map_location="cpu")
    torch.testing.assert_close(
        net.pitches[0].inp.weight.cpu(), saved["encoder.inp.weight"].to(torch.float32)
    )


def test_the_finetuned_net_trains_and_predicts_every_test_row(setup):
    table, apps, run = setup
    x, (y, w) = input_frame(table), answers(table)
    batcher = FinetuneBatcher(apps, PitchStore(_pitches(), device()), table, LENGTH)
    config = plain_config(
        hidden=[8],
        max_epochs=2,
        patience=1,
        val_frac=0.34,
        seq="transformer",
        seq_len=16,
        seq_dim=8,
        micro_batch=4,
    )
    make = model_maker(run, 2025, FinetuneConfig("q", LENGTH, 0.1, 4), device())
    preds, _ = fit_season(table, x, y, w, 2025, config, batcher=batcher, make_model=make)
    assert len(preds) == (table.season == 2025).sum()
    assert preds[list(TARGETS)].notna().all().all()
    assert np.isfinite(preds[list(TARGETS)].to_numpy()).all()


def test_history_must_fit_the_pretrained_positions(setup):
    _, _, run = setup
    check_length(FinetuneConfig("q", LENGTH), run)
    with pytest.raises(ValueError, match="pretraining window"):
        check_length(FinetuneConfig("q", LENGTH + 1), run)
    with pytest.raises(ValueError, match="no model for 2024"):
        run.encoder(2024, device())


def test_settings_are_checked():
    with pytest.raises(ValueError, match="lr_scale"):
        FinetuneConfig("q", lr_scale=2.0)
    with pytest.raises(ValueError, match="length"):
        FinetuneConfig("q", length=0)


@pytest.fixture(scope="module")
def cross_runs(tmp_path_factory):
    """Two runs pretrained on opposite halves of the pitchers (different weights)."""
    root = tmp_path_factory.mktemp("cross")
    config = _small(window=LENGTH + 1)
    runs = []
    for half, seed in ((0, 1), (1, 2)):
        (root / f"h{half}" / "2025").mkdir(parents=True)
        torch.manual_seed(seed)
        torch.save(PretrainModel(config).state_dict(), root / f"h{half}" / "2025" / "model.pt")
        meta = {"config": config.to_dict(), "seasons": [2025], "tokens": {}, "half": half}
        (root / f"h{half}" / "run.json").write_text(json.dumps(meta))
        runs.append(PretrainRun(root, f"h{half}"))
    return runs


def _cross_net(h0, h1):
    config = plain_config(hidden=[8], seq="transformer", seq_len=16, seq_dim=8)
    finetune = FinetuneConfig("h0", LENGTH, 0.1, 4, cross="h1")
    make = model_maker(h0, 2025, finetune, device(), h1)
    return make(5, len(TARGETS), config, len(TOKEN_FEATURES) + 2).to(device()).eval()


def test_each_pitcher_is_read_by_the_encoder_that_never_saw_him(setup, cross_runs):
    table, apps, _ = setup
    h0, h1 = cross_runs
    which = cross_encoders(table["player_id"], h0)
    # h0 read half 0's pitches: those pitchers go to encoder 1 (h1, half 1 only).
    np.testing.assert_array_equal(which, player_half(table["player_id"]) == 0)
    assert set(which) == {0, 1}  # the fixture has pitchers in both halves
    batcher = FinetuneBatcher(apps, PitchStore(_pitches(), device()), table, LENGTH, which)
    rows = torch.cat([_row(table, STARTER, 1), _row(table, RELIEVER, 1)])
    seq, lengths = batcher.batch(rows)
    expected = [int(which[int(r)]) for r in rows]
    assert seq[3].tolist() == expected and sorted(expected) == [0, 1]
    assert (seq[2] > 0).all()  # both have pitches to read
    net, x = _cross_net(h0, h1), torch.zeros(2, 5, device=device())
    with torch.no_grad():
        before = net(x, seq, lengths)
        for p in net.pitches[1].parameters():
            p.add_(0.5)
        after = net(x, seq, lengths)
    one = torch.as_tensor(expected, device=device()) == 1
    torch.testing.assert_close(before[~one], after[~one])  # encoder 0's row: untouched
    assert not torch.allclose(before[one], after[one])


def test_cross_runs_must_be_opposite_halves_of_one_token_file(setup, cross_runs):
    _, _, whole = setup
    h0, h1 = cross_runs
    check_cross(h0, h1)
    with pytest.raises(ValueError, match="opposite halves"):
        check_cross(h0, h0)
    with pytest.raises(ValueError, match="opposite halves"):
        check_cross(whole, h1)
    with pytest.raises(ValueError, match="no half to cross-fit"):
        cross_encoders(np.array([STARTER]), whole)


def test_the_cross_fitted_net_trains_every_pitcher_and_predicts(setup, cross_runs):
    table, apps, _ = setup
    h0, h1 = cross_runs
    x, (y, w) = input_frame(table), answers(table)
    which = cross_encoders(table["player_id"], h0)
    batcher = FinetuneBatcher(apps, PitchStore(_pitches(), device()), table, LENGTH, which)
    config = plain_config(
        hidden=[8],
        max_epochs=2,
        patience=1,
        val_frac=0.34,
        seq="transformer",
        seq_len=16,
        seq_dim=8,
        micro_batch=4,
    )
    make = model_maker(h0, 2025, FinetuneConfig("h0", LENGTH, 0.1, 4, cross="h1"), device(), h1)
    preds, _ = fit_season(table, x, y, w, 2025, config, batcher=batcher, make_model=make)
    assert len(preds) == (table.season == 2025).sum()
    assert np.isfinite(preds[list(TARGETS)].to_numpy()).all()
