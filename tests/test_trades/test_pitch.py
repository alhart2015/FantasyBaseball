from fantasy_baseball.trades.pitch import generate_pitch


def test_pitch_mentions_rankings():
    pitch = generate_pitch(
        send_rank=42,
        receive_rank=47,
        send_positions=["SS", "2B"],
        receive_positions=["OF"],
    )
    assert "#42" in pitch
    assert "#47" in pitch


def test_pitch_includes_positional_need_for_different_positions():
    pitch = generate_pitch(
        send_rank=30,
        receive_rank=33,
        send_positions=["SS"],
        receive_positions=["OF"],
    )
    assert "#30" in pitch
    assert "#33" in pitch
    # Should mention positional context when positions differ
    assert "position" in pitch.lower() or "need" in pitch.lower()


def test_pitch_no_positional_note_for_same_position():
    pitch = generate_pitch(
        send_rank=20,
        receive_rank=22,
        send_positions=["OF"],
        receive_positions=["OF"],
    )
    assert "#20" in pitch
    assert "#22" in pitch
    # Should NOT mention positional need when same position
    assert "position" not in pitch.lower() and "need" not in pitch.lower()


def test_pitch_when_sending_better_ranked_player():
    """When we send a better-ranked player, pitch should frame it as generous."""
    pitch = generate_pitch(
        send_rank=25,
        receive_rank=30,
        send_positions=["1B"],
        receive_positions=["3B"],
    )
    assert "#25" in pitch
    assert "#30" in pitch


def test_pitch_is_short():
    pitch = generate_pitch(
        send_rank=50,
        receive_rank=55,
        send_positions=["C"],
        receive_positions=["SP"],
    )
    assert len(pitch) < 200
