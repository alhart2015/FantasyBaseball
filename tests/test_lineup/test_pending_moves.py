from fantasy_baseball.lineup.yahoo_roster import parse_pending_moves


def _make_transaction(txn_id, status, ttype, team_name, team_key,
                      adds=None, drops=None, timestamp="1712700000"):
    """Build a raw transaction dict matching yahoo_fantasy_api output."""
    txn = {
        "transaction_id": txn_id,
        "status": status,
        "type": ttype,
        "timestamp": timestamp,
    }
    players = {}
    idx = 0
    for p in (adds or []):
        players[str(idx)] = {
            "player": [[
                {"name": {"full": p["name"]}},
                {"player_id": p.get("player_id", "")},
                {"eligible_positions": [
                    {"position": pos} for pos in p.get("positions", [])
                ]},
            ]],
            "transaction_data": {
                "type": "add",
                "destination_team_name": team_name,
                "destination_team_key": team_key,
            },
        }
        idx += 1
    for p in (drops or []):
        players[str(idx)] = {
            "player": [[
                {"name": {"full": p["name"]}},
                {"player_id": p.get("player_id", "")},
                {"eligible_positions": [
                    {"position": pos} for pos in p.get("positions", [])
                ]},
            ]],
            "transaction_data": {
                "type": "drop",
                "source_team_name": team_name,
                "source_team_key": team_key,
            },
        }
        idx += 1
    txn["players"] = players
    return txn


class TestParsePendingMoves:
    def test_filters_to_pending_only(self):
        transactions = [
            _make_transaction("1", "successful", "add/drop", "Team A", "t.1",
                              adds=[{"name": "Otto Lopez"}],
                              drops=[{"name": "Marcus Semien"}]),
            _make_transaction("2", "pending", "add/drop", "Team A", "t.1",
                              adds=[{"name": "Ryan Walker"}],
                              drops=[{"name": "Bryan Woo"}]),
        ]
        result = parse_pending_moves(transactions)
        assert len(result) == 1
        assert result[0]["transaction_id"] == "2"

    def test_parses_add_drop(self):
        transactions = [
            _make_transaction("1", "pending", "add/drop", "Team A", "t.1",
                              adds=[{"name": "Otto Lopez", "player_id": "123",
                                     "positions": ["2B", "SS"]}],
                              drops=[{"name": "Marcus Semien", "player_id": "456",
                                      "positions": ["2B", "SS"]}]),
        ]
        result = parse_pending_moves(transactions)
        assert len(result) == 1
        move = result[0]
        assert move["team"] == "Team A"
        assert move["team_key"] == "t.1"
        assert move["type"] == "add/drop"
        assert len(move["adds"]) == 1
        assert move["adds"][0]["name"] == "Otto Lopez"
        assert move["adds"][0]["positions"] == ["2B", "SS"]
        assert len(move["drops"]) == 1
        assert move["drops"][0]["name"] == "Marcus Semien"

    def test_empty_transactions_returns_empty(self):
        assert parse_pending_moves([]) == []

    def test_all_successful_returns_empty(self):
        transactions = [
            _make_transaction("1", "successful", "add", "Team A", "t.1",
                              adds=[{"name": "Otto Lopez"}]),
        ]
        assert parse_pending_moves(transactions) == []

    def test_multiple_teams(self):
        transactions = [
            _make_transaction("1", "pending", "add/drop", "Team A", "t.1",
                              adds=[{"name": "Player A"}],
                              drops=[{"name": "Player B"}]),
            _make_transaction("2", "pending", "add/drop", "Team B", "t.2",
                              adds=[{"name": "Player C"}],
                              drops=[{"name": "Player D"}]),
        ]
        result = parse_pending_moves(transactions)
        assert len(result) == 2
        teams = {m["team"] for m in result}
        assert teams == {"Team A", "Team B"}
