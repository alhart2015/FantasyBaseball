from fantasy_baseball.summary.builders import build_injuries, build_lineup_moves, build_probables


def test_build_lineup_moves_flattens_swaps_and_unpaired():
    payload = {
        "moves": {
            "swaps": [
                {
                    "start": {"player": "A", "from": "BN", "to": "OF", "roto_delta": 0.4},
                    "bench": {"player": "B", "from": "OF", "to": "BN"},
                },
            ],
            "unpaired_starts": [{"player": "C", "from": "BN", "to": "UTIL", "roto_delta": 0.2}],
            "unpaired_benches": [{"player": "D", "from": "2B", "to": "BN"}],
        }
    }
    moves = build_lineup_moves(payload)
    actions = {(m.player, m.action) for m in moves}
    assert ("A", "start") in actions
    assert ("B", "sit") in actions
    assert ("C", "start") in actions
    assert ("D", "sit") in actions
    a = next(m for m in moves if m.player == "A")
    assert a.to_slot == "OF" and a.roto_delta == 0.4


def test_build_lineup_moves_handles_missing():
    assert build_lineup_moves(None) == []
    assert build_lineup_moves({}) == []


def test_build_injuries_maps_rows():
    rows = [
        {
            "name": "Hurt Guy",
            "status": "IL15",
            "status_full": "15-Day IL",
            "injury_note": "hamstring strain",
        },
        {"name": "No Note", "status": "DTD", "status_full": "Day-To-Day", "injury_note": ""},
    ]
    items = build_injuries(rows)
    assert items[0].name == "Hurt Guy"
    assert items[0].status == "IL15"
    assert items[0].note == "hamstring strain"
    assert items[1].note == ""


def test_build_probables_maps_and_handles_absent():
    assert build_probables(None) == []
    rows = [
        {
            "pitcher": "Ace",
            "starts": 2,
            "days": "Mon, Sat",
            "opponents": "@ BAL, vs TOR",
            "matchup_quality": "Great",
            "matchups": [],
        }
    ]
    items = build_probables(rows)
    assert items[0].pitcher == "Ace"
    assert items[0].starts == 2
    assert items[0].quality == "Great"
