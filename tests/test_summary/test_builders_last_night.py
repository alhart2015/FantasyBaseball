from datetime import date

from fantasy_baseball.summary.builders import build_last_night


class FakeKV:
    def __init__(self, store):
        self._store = store

    def get(self, key):
        return self._store.get(key)


def test_build_last_night_matches_and_filters_to_yesterday():
    import json

    yesterday = date(2026, 7, 10)
    store = {
        "game_logs:2026:111:hitting": json.dumps(
            {
                "name": "Aaron Judge",
                "games": [
                    {
                        "date": "2026-07-10",
                        "pa": 4,
                        "ab": 4,
                        "h": 2,
                        "hr": 1,
                        "r": 2,
                        "rbi": 3,
                        "sb": 0,
                    },
                    {
                        "date": "2026-07-09",
                        "pa": 4,
                        "ab": 4,
                        "h": 0,
                        "hr": 0,
                        "r": 0,
                        "rbi": 0,
                        "sb": 0,
                    },
                ],
            }
        ),
    }
    xmap = {("aaron judge", "hitter"): 111}
    roster = [{"name": "Aaron Judge", "positions": ["OF"]}]

    lines, unmatched = build_last_night(roster, xmap, FakeKV(store), 2026, yesterday)

    assert unmatched == []
    assert len(lines) == 1
    assert lines[0].name == "Aaron Judge"
    assert lines[0].group == "hitting"
    assert lines[0].stats["hr"] == 1
    assert lines[0].stats["h"] == 2


def test_build_last_night_records_unmatched():
    yesterday = date(2026, 7, 10)
    xmap: dict = {}
    roster = [{"name": "Ghost Player", "positions": ["2B"]}]
    lines, unmatched = build_last_night(roster, xmap, FakeKV({}), 2026, yesterday)
    assert lines == []
    assert unmatched == ["Ghost Player"]


def test_build_last_night_omits_players_who_did_not_play():
    import json

    yesterday = date(2026, 7, 10)
    store = {
        "game_logs:2026:222:hitting": json.dumps(
            {
                "name": "Benched Guy",
                "games": [
                    {
                        "date": "2026-07-08",
                        "pa": 3,
                        "ab": 3,
                        "h": 1,
                        "hr": 0,
                        "r": 0,
                        "rbi": 0,
                        "sb": 0,
                    }
                ],
            }
        ),
    }
    xmap = {("benched guy", "hitter"): 222}
    roster = [{"name": "Benched Guy", "positions": ["1B"]}]
    lines, unmatched = build_last_night(roster, xmap, FakeKV(store), 2026, yesterday)
    assert lines == []
    assert unmatched == []
