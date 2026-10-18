import numpy as np
import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros.final import (
    GRID,
    WEIGHTS,
    best_weights,
    blend_rates,
    blended_scored,
    held_out_weights,
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
    got = best_weights(_scored(true))
    assert got.to_dict() == pytest.approx(true)


def test_held_out_weights_never_see_the_group_they_score():
    true = {**{(2024, s): 0.2 for s in TARGETS}, **{(2025, s): 0.8 for s in TARGETS}}
    scored = _scored(true)
    held = held_out_weights(scored, scored["season"])
    # 2024's weights come from 2025's rows alone, and the other way round.
    assert held.loc[2024].to_dict() == pytest.approx(dict.fromkeys(TARGETS, 0.8))
    assert held.loc[2025].to_dict() == pytest.approx(dict.fromkeys(TARGETS, 0.2))
    with pytest.raises(ValueError, match="two or more"):
        held_out_weights(scored, pd.Series(2024, index=scored.index))
    with pytest.raises(ValueError, match="aligned"):
        held_out_weights(scored, scored["season"].iloc[1:])


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
