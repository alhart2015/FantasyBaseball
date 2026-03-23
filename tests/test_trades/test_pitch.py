from fantasy_baseball.trades.pitch import generate_pitch

ALL_CATS = ["R", "HR", "RBI", "SB", "AVG", "W", "K", "SV", "ERA", "WHIP"]


def test_pitch_highlights_gains_and_affordable_loss():
    opp_cat_deltas = {"R": 0, "HR": -1, "RBI": 0, "SB": 2, "AVG": 1,
                      "W": 0, "K": 0, "SV": -1, "ERA": 0, "WHIP": 0}
    opp_cat_ranks = {"R": 5, "HR": 2, "RBI": 5, "SB": 8, "AVG": 7,
                     "W": 5, "K": 5, "SV": 2, "ERA": 5, "WHIP": 5}
    pitch = generate_pitch("Springfield Isotopes", opp_cat_deltas, opp_cat_ranks)
    # Should mention their weak categories (SB=8th, AVG=7th)
    assert "SB" in pitch or "steals" in pitch.lower() or "stolen" in pitch.lower()
    assert len(pitch) < 300


def test_pitch_with_no_gains():
    opp_cat_deltas = {c: 0 for c in ALL_CATS}
    opp_cat_ranks = {c: 5 for c in ALL_CATS}
    pitch = generate_pitch("Team X", opp_cat_deltas, opp_cat_ranks)
    assert "neutral" in pitch.lower()


def test_pitch_no_losses():
    opp_cat_deltas = {"R": 1, "HR": 0, "RBI": 0, "SB": 0, "AVG": 0,
                      "W": 0, "K": 0, "SV": 0, "ERA": 0, "WHIP": 0}
    opp_cat_ranks = {"R": 9, "HR": 5, "RBI": 5, "SB": 5, "AVG": 5,
                     "W": 5, "K": 5, "SV": 5, "ERA": 5, "WHIP": 5}
    pitch = generate_pitch("SkeleThor", opp_cat_deltas, opp_cat_ranks)
    # Should mention their weak R (9th) and not have a "you can afford" part
    assert "afford" not in pitch.lower()
    assert len(pitch) > 10
