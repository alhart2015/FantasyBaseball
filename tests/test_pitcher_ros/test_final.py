import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.blend_fit import (
    GRID,
    best_weights,
    blended_scored,
    held_out_weights,
    mse_at,
    wide,
)
from fantasy_baseball.pitcher_ros.final import (
    BOARD,
    SCORING,
    SYSTEMS,
    WEIGHTS,
    blend_rates,
    board_rates,
    fg_average,
    final_rates,
)
from fantasy_baseball.pitcher_ros.stats import TARGETS


def _rates(ids, **cols):
    return pd.DataFrame(
        {s: cols.get(s, 0.0) for s in TARGETS}, index=pd.Index(ids, name="player_id")
    )


def test_the_weights_in_use_cover_every_stat_on_the_grid():
    for span in WEIGHTS.values():
        assert set(span) == set(TARGETS)
        assert all(w in GRID for w in span.values())


def test_blend_rates_mixes_each_stat_by_its_own_weight():
    ours = _rates([1, 2], k=0.30, w=0.02, sv=0.04, era=3.0, whip=1.1)
    fg = _rates([1, 3], k=0.20, w=0.04, sv=0.00, era=5.0, whip=1.3)
    weights = {"k": 0.5, "w": 1.0, "sv": 0.25, "era": 0.0, "whip": 0.75}
    out = blend_rates(ours, fg, weights)
    one = out.loc[1]
    assert one.k == pytest.approx(0.25) and one.w == pytest.approx(0.02)
    assert one.sv == pytest.approx(0.01) and one.era == pytest.approx(5.0)
    assert one.whip == pytest.approx(0.75 * 1.1 + 0.25 * 1.3)
    # A pitcher only one side projects gets that side's rates, whatever the weight.
    assert out.loc[2].era == pytest.approx(3.0) and out.loc[3].w == pytest.approx(0.04)
    assert list(out.index) == [1, 2, 3]


def test_blend_rates_refuses_a_missing_or_impossible_weight():
    ours, fg = _rates([1]), _rates([1])
    with pytest.raises(ValueError, match="no weight"):
        blend_rates(ours, fg, {"k": 0.5})
    with pytest.raises(ValueError, match="0-1"):
        blend_rates(ours, fg, {**WEIGHTS["preseason"], "sv": 1.2})


def _scored(true_w: dict[str, float], seasons=(2024, 2025), n=40, seed=0):
    """Scored rows where the actual is exactly ``true_w[stat] x ours + (1 - w) x fg``,
    per season (``true_w`` may be keyed by (season, stat))."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for stat in TARGETS:
            w = true_w.get((season, stat), true_w.get(stat))
            ours = rng.uniform(0.01, 0.3, n) if stat in ("k", "w", "sv") else rng.uniform(2, 6, n)
            fg = ours * rng.uniform(0.5, 1.5, n)
            actual = w * ours + (1 - w) * fg
            for i in range(n):
                base = {
                    "player_id": i,
                    "season": season,
                    "stat": stat,
                    "actual": actual[i],
                    "bf": 300.0,
                    "ip": 70.0,
                    "relevant": i % 2 == 0,
                }
                rows.append({**base, "system": "ours", "projected": ours[i]})
                rows.append({**base, "system": "fg_blend", "projected": fg[i]})
                rows.append({**base, "system": "marcel", "projected": fg[i]})
    return pd.DataFrame(rows)  # fmt: skip


def test_best_weights_finds_the_mix_the_actuals_were_made_from():
    true = {"k": 0.3, "w": 0.7, "sv": 0.0, "era": 1.0, "whip": 0.55}
    got = best_weights(_scored(true), SCORING)
    assert got.to_dict() == pytest.approx(true)


def test_held_out_weights_never_see_the_group_they_score():
    true = {**{(2024, s): 0.2 for s in TARGETS}, **{(2025, s): 0.8 for s in TARGETS}}
    scored = _scored(true)
    held = held_out_weights(scored, scored["season"], SCORING)
    # 2024's weights come from 2025's rows alone, and the other way round.
    assert held.loc[2024].to_dict() == pytest.approx(dict.fromkeys(TARGETS, 0.8))
    assert held.loc[2025].to_dict() == pytest.approx(dict.fromkeys(TARGETS, 0.2))
    with pytest.raises(ValueError, match="two or more"):
        held_out_weights(scored, pd.Series(2024, index=scored.index), SCORING)
    with pytest.raises(ValueError, match="aligned"):
        held_out_weights(scored, scored["season"].iloc[1:], SCORING)


def test_blended_scored_adds_the_blend_as_its_own_system():
    scored = _scored(dict.fromkeys(TARGETS, 0.5))
    weights = pd.Series(dict.fromkeys(TARGETS, 0.25)).rename_axis("stat")
    out = blended_scored(scored, weights)
    final = out[out.system == "final"].set_index(["player_id", "season", "stat"])
    pair = scored.pivot_table(
        index=["player_id", "season", "stat"], columns="system", values="projected"
    )
    expected = 0.25 * pair["ours"] + 0.75 * pair["fg_blend"]
    assert np.allclose(final["projected"], expected.reindex(final.index))
    assert len(final) == (scored.system == "ours").sum()
    assert (final["relevant"] == scored[scored.system == "ours"]
            .set_index(["player_id", "season", "stat"])["relevant"].reindex(final.index)).all()  # fmt: skip
    with pytest.raises(ValueError, match="no weight"):
        blended_scored(scored, weights.drop("sv"))
    with pytest.raises(ValueError, match="no \\['fg_blend'\\]"):
        blended_scored(scored[scored.system != "fg_blend"], weights)


def test_best_weights_matches_scoring_every_grid_point():
    # Noisy actuals, so the best weight falls between grid points: the shortcut (score
    # only the two grid points around the parabola's low point) must agree with a full
    # search of the grid.
    scored = _scored(dict.fromkeys(TARGETS, 0.37), n=60, seed=3)
    rng = np.random.default_rng(4)
    scored["actual"] = scored["actual"] * rng.uniform(0.6, 1.4, len(scored))
    scored["actual"] = scored.groupby(["player_id", "season", "stat"])["actual"].transform("first")
    rows_wide = wide(scored)
    for stat, rows in rows_wide.groupby("stat"):
        scores = {w: mse_at(rows, w, SCORING) for w in GRID}
        assert best_weights(scored, SCORING)[stat] == min(GRID, key=lambda w: (scores[w], -w))


def test_best_weights_takes_the_ends_and_ties_to_ours():
    # Actuals beyond ours (weight > 1) clip to 1; ours = FanGraphs ties every weight -> 1.
    scored = _scored(dict.fromkeys(TARGETS, 1.6))
    assert set(best_weights(scored, SCORING)) == {1.0}
    same = scored.copy()
    fg = same["system"] == "fg_blend"
    same.loc[fg, "projected"] = same.loc[same["system"] == "ours", "projected"].to_numpy()
    assert set(best_weights(same, SCORING)) == {1.0}


def test_a_unit_fangraphs_lacks_is_refused_not_dropped():
    scored = _scored(dict.fromkeys(TARGETS, 0.5))
    gone = scored[~((scored.system == "fg_blend") & (scored.player_id == 3))]
    with pytest.raises(ValueError, match="no projection from"):
        best_weights(gone, SCORING)
    blank = scored.copy()
    blank.loc[(blank.system == "fg_blend") & (blank.player_id == 3), "projected"] = np.nan
    with pytest.raises(ValueError, match="no projection from"):
        best_weights(blank, SCORING)


def test_held_out_weights_refuses_blank_groups_and_keeps_any_name():
    scored = _scored(dict.fromkeys(TARGETS, 0.5))
    by = scored["season"].astype(float)
    by.iloc[0] = np.nan
    with pytest.raises(ValueError, match="blank"):
        held_out_weights(scored, by, SCORING)
    named_zero = scored["season"].rename(0)
    assert held_out_weights(scored, named_zero, SCORING).index.names == [0, "stat"]
    assert held_out_weights(scored, scored["season"].rename(None), SCORING).index.names == [
        "group",
        "stat",
    ]


def test_fg_average_averages_whichever_systems_project_him():
    steamer = _rates([1, 2], k=0.20, sv=0.04)
    zips = _rates([1, 3], k=0.30, sv=0.00)
    avg = fg_average({"steamer": steamer, "zips": zips})
    assert list(avg.index) == [1, 2, 3]
    assert avg.loc[1].k == pytest.approx(0.25) and avg.loc[1].sv == pytest.approx(0.02)
    # One system only: that system's rates, not dropped.
    assert avg.loc[2].sv == pytest.approx(0.04) and avg.loc[3].k == pytest.approx(0.30)
    with pytest.raises(ValueError, match="no FanGraphs"):
        fg_average({})


def test_final_rates_needs_the_systems_the_weights_were_fitted_with():
    ours = _rates([1], k=0.30)
    systems = {s: _rates([1], k=0.20) for s in SYSTEMS["preseason"]}
    out = final_rates(ours, systems, "preseason")
    w = WEIGHTS["preseason"]["k"]
    assert out.loc[1].k == pytest.approx(w * 0.30 + (1 - w) * 0.20)
    with pytest.raises(ValueError, match="fitted with"):
        final_rates(ours, {**systems, "atc": _rates([1])}, "preseason")
    with pytest.raises(ValueError, match="span"):
        final_rates(ours, systems, "playoffs")
    with pytest.raises(ValueError, match="span"):  # its FanGraphs side is the board's blend
        final_rates(ours, systems, BOARD)


def test_board_rates_mixes_ours_with_the_board_blend_at_the_board_weights():
    ours = _rates([1, 2], k=0.30, era=3.0)
    out = board_rates(ours, _rates([1, 3], k=0.20, era=5.0))
    w = WEIGHTS[BOARD]
    assert out.loc[1].k == pytest.approx(w["k"] * 0.30 + (1 - w["k"]) * 0.20)
    assert out.loc[1].era == pytest.approx(w["era"] * 3.0 + (1 - w["era"]) * 5.0)
    assert out.loc[2].k == pytest.approx(0.30) and out.loc[3].era == pytest.approx(5.0)


def _script():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    import fit_pitcher_blend

    return fit_pitcher_blend


def test_the_script_scores_one_season_in_sample_and_says_so():
    md = "\n".join(
        _script().span_lines(_scored(dict.fromkeys(TARGETS, 0.5), seasons=(2026,)), "midseason")
    )
    assert "Not validated" in md and "without" not in md
    # The scored frame has no FanGraphs systems by name, so it warns about SYSTEMS.
    assert "Warning" in md


def test_the_script_holds_out_each_season_when_there_are_several():
    md = "\n".join(_script().span_lines(_scored(dict.fromkeys(TARGETS, 0.5)), "preseason"))
    assert "without 2024" in md and "without 2025" in md and "Not validated" not in md
