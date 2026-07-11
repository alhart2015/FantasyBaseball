from fantasy_baseball.summary.builders import build_streaks


def test_build_streaks_extracts_hot_cold_from_roster_rows():
    payload = {
        "roster_rows": [
            {
                "name": "Aaron Judge",
                "scores": {
                    "hr": {"label": "hot", "probability": 0.71},
                    "avg": {"label": "cold", "probability": 0.64},
                    "sb": {"label": "neutral", "probability": 0.10},
                },
            },
        ],
        "fa_rows": [],
    }
    items = build_streaks(payload)
    labels = {(i.category, i.label) for i in items}
    assert ("hr", "hot") in labels
    assert ("avg", "cold") in labels
    assert all(i.label in ("hot", "cold") for i in items)  # neutral dropped
    judge_hr = next(i for i in items if i.category == "hr")
    assert judge_hr.name == "Aaron Judge"
    assert judge_hr.probability == 0.71


def test_build_streaks_handles_missing_payload():
    assert build_streaks(None) == []
    assert build_streaks({}) == []
