import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros import backtest
from fantasy_baseball.pitcher_ros.backtest import BLEND, VET_MIN_CAREER_BF, score_span
from fantasy_baseball.sgp.denominators import get_sgp_denominators

from .conftest import game, league_games

HEADER = "Name,W,SV,IP,TBF,ER,H,BB,SO,ERA,WHIP,MLBAMID"


def _games():
    # The store starts in 2016 here, so 2025 has 9 earlier seasons: rookies are known.
    return pd.DataFrame(
        [
            game(900, "2016-06-01", bf=10, outs=7),
            *league_games(),
            # 1: a vet (1000 BF before 2025). 2: a rookie (first games in 2025).
            game(1, "2024-06-01", bf=1000, outs=700, k=300, er=80, h=180, bb=60),
            game(1, "2025-04-15", bf=300, outs=210, k=90, w=3, er=20, h=50, bb=20),
            game(1, "2025-07-15", bf=300, outs=210, k=60, w=2, er=40, h=70, bb=30),
            game(2, "2025-04-20", bf=150, outs=100, k=30, w=1, er=20, h=40, bb=20),
            game(2, "2025-07-20", bf=150, outs=100, k=30, w=1, er=20, h=40, bb=20),
        ]
    )


def _systems():
    idx = pd.Index([1, 2], name="player_id")
    base = {"w": [0.01, 0.01], "sv": [0.0, 0.0], "era": [3.5, 4.5], "whip": [1.1, 1.3]}
    vol = {"bf": [700.0, 300.0], "ip": [170.0, 70.0]}
    return {
        "steamer": pd.DataFrame({"k": [0.28, 0.2], **base, **vol}, index=idx),
        "zips": pd.DataFrame({"k": [0.30, 0.2], **base, **vol}, index=idx),
    }


def test_preseason_span_scores_baselines_systems_and_blend_with_tags():
    scored = score_span(
        _games(), 2025, pd.Timestamp(2025, 1, 1), _systems(), 100, get_sgp_denominators()
    )
    assert scored is not None
    assert set(scored["system"]) == {"marcel", "league_avg", "steamer", "zips", BLEND}
    blend_k = scored[(scored["system"] == BLEND) & (scored["stat"] == "k")]
    assert blend_k.set_index("player_id").loc[1, "projected"] == pytest.approx(0.29)
    by_player = scored.drop_duplicates("player_id").set_index("player_id")
    assert by_player.loc[1, "group"] == "vet" and by_player.loc[1, "career_bf"] == 1000
    assert by_player.loc[2, "group"] == "rookie" and by_player.loc[2, "career_bf"] == 0
    assert by_player["relevant"].all()  # two pitchers: both in any top 150
    # Whole-season actuals: player 1 faced 600 batters in 2025.
    assert by_player.loc[1, "bf"] == 600


def test_mid_season_span_counts_games_before_as_history():
    snap = pd.Timestamp("2025-07-01")
    scored = score_span(_games(), 2025, snap, _systems(), 100, get_sgp_denominators())
    assert scored is not None
    by_player = scored.drop_duplicates("player_id").set_index("player_id")
    # Actuals are only the games from the snapshot on.
    assert by_player.loc[1, "bf"] == 300
    # The rookie's April games count toward his career by July, but not enough.
    assert by_player.loc[2, "career_bf"] == 150 < VET_MIN_CAREER_BF


def test_span_with_no_games_yet_is_none():
    out = score_span(
        _games(), 2026, pd.Timestamp(2026, 1, 1), _systems(), 100, get_sgp_denominators()
    )
    assert out is None


def test_short_store_history_makes_rookies_unknown():
    games = _games()
    games = games[games["season"] >= 2022]  # 2025 has only 3 earlier seasons in the store
    scored = score_span(
        games, 2025, pd.Timestamp(2025, 1, 1), _systems(), 100, get_sgp_denominators()
    )
    assert scored is not None
    groups = scored.drop_duplicates("player_id").set_index("player_id")["group"]
    assert groups.to_dict() == {1: "vet", 2: "unknown"}


def test_snapshots_score_each_dated_folder(tmp_path):
    folder = tmp_path / "2025" / "rest_of_season" / "2025-07-01"
    folder.mkdir(parents=True)
    rows = ["A,4,0,80,330,30,70,20,90,3.4,1.12,1", "B,2,0,40,170,20,40,15,30,4.5,1.4,2"]
    (folder / "steamer-pitchers.csv").write_text("\n".join([HEADER, *rows]) + "\n")
    out = backtest.snapshots(_games(), 2025, tmp_path)
    assert out is not None
    assert set(out["snapshot"]) == {"2025-07-01"} and set(out["season"]) == {2025}
    # One system: no blend.
    assert set(out["system"]) == {"marcel", "league_avg", "steamer"}
    md = "\n".join(backtest.summarize(None, out))
    assert "2025-07-01" in md and "All snapshots" in md
