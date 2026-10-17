import pandas as pd
import pytest

from fantasy_baseball.pitcher_ros.horizons import horizon_summary, score_horizons
from fantasy_baseball.pitcher_ros.model import HORIZON_WEIGHTS
from fantasy_baseball.pitcher_ros.stats import TARGETS, rates_from_counts
from fantasy_baseball.pitcher_ros.table import HORIZON_DAYS

KEYS = ["player_id", "season", "week"]
COUNTS = ("bf", "outs", "k", "w", "sv", "er", "h", "bb")


def _frames():
    """Three pitchers x weeks 0-1 of 2025; pitcher 3 faced nobody in either window of
    week 1. The head is exactly right; the rest-of-season rate is off."""
    rows = []
    for pid, scale in ((1, 1.0), (2, 2.0), (3, 0.5)):
        for week in (0, 1):
            row = {"player_id": pid, "season": 2025, "week": week}
            none = pid == 3 and week == 1
            for n in HORIZON_DAYS:
                k = n / 7
                line = dict(bf=20 * k, outs=15 * k, k=5 * k * scale, w=1, sv=0, er=2 * k, h=4 * k)
                line["bb"] = 2 * k
                for c in COUNTS:
                    row[f"ros_d{n}_{c}"] = 0.0 if none else float(line[c])
            rows.append(row)
    table = pd.DataFrame(rows)
    preds = table[KEYS].copy()
    for n in HORIZON_DAYS:
        counts = table[[f"ros_d{n}_{c}" for c in COUNTS]].set_axis(list(COUNTS), axis=1)
        actual = rates_from_counts(counts)[list(TARGETS)].fillna(0.0)
        for s in TARGETS:
            preds[f"d{n}_{s}"] = actual[s]
    for s in TARGETS:
        preds[s] = preds[f"d{HORIZON_DAYS[0]}_{s}"] * 1.5 + 0.01
    return table, preds


def test_short_horizon_outputs_are_scored_against_the_next_days():
    table, preds = _frames()
    scored = score_horizons(table, preds)
    assert set(scored["horizon"]) == {f"d{n}" for n in HORIZON_DAYS}
    assert set(scored["system"]) == {"head", "ros_rate"}
    # Pitcher 3 faced nobody in week 1's windows: not scored there.
    week1 = scored[scored["snapshot"] == "2025-w01"]
    assert set(week1["player_id"]) == {1, 2}
    assert set(scored.loc[scored["snapshot"] == "2025-w00", "span"]) == {"preseason"}
    assert set(week1["span"]) == {"mid-season"}
    head = scored[scored["system"] == "head"]
    assert (head["projected"] - head["actual"]).abs().max() < 1e-9
    md = "\n".join(horizon_summary(scored))
    assert "Next 7 days, mid-season" in md and "Next 30 days, preseason" in md


def test_without_short_horizon_outputs_nothing_is_scored():
    table, preds = _frames()
    assert score_horizons(table, preds[[*KEYS, *TARGETS]]) is None
    assert horizon_summary(None) == []


def test_a_table_without_short_horizon_answers_is_refused():
    table, preds = _frames()
    with pytest.raises(ValueError, match="build_pitcher_ros_table"):
        score_horizons(table[KEYS], preds)


def test_every_horizon_the_table_answers_has_a_loss_weight():
    assert set(HORIZON_WEIGHTS) == {f"d{n}" for n in HORIZON_DAYS}
