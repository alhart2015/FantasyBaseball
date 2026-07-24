"""Tests for breakout.py skill/luck classifier."""
from fantasy_baseball.analysis import breakout


def test_shapes_exist():
    """Verify SkillLuckRow and LABELS are defined and importable."""
    row = breakout.SkillLuckRow(
        mlbam=1, player_type="hitter", pa=600, ip=0.0, age=27.0,
        barrel_pct=None, xslg=None, slg=None, xba=None, ba=None, babip=None,
        xwoba=None, woba=None, k_pct=None, bb_pct=None)
    assert row.player_type == "hitter"
    assert "real breakout" in breakout.LABELS
