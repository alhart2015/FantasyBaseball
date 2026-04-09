from fantasy_baseball.lineup.yahoo_roster import parse_all_transactions


def _make_raw_txn(txn_id, status, ttype, team_name, team_key,
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


class TestParseAllTransactions:
    def test_includes_successful_transactions(self):
        raw = [
            _make_raw_txn("1", "successful", "add/drop", "Team A", "t.1",
                          adds=[{"name": "Otto Lopez", "positions": ["2B"]}],
                          drops=[{"name": "Marcus Semien", "positions": ["SS"]}]),
        ]
        result = parse_all_transactions(raw)
        assert len(result) == 1
        assert result[0]["transaction_id"] == "1"

    def test_parses_add_drop_players(self):
        raw = [
            _make_raw_txn("1", "successful", "add/drop", "Team A", "t.1",
                          adds=[{"name": "Otto Lopez", "player_id": "123",
                                 "positions": ["2B", "SS"]}],
                          drops=[{"name": "Marcus Semien", "player_id": "456",
                                  "positions": ["2B"]}]),
        ]
        result = parse_all_transactions(raw)
        move = result[0]
        assert move["add_name"] == "Otto Lopez"
        assert move["add_player_id"] == "123"
        assert move["add_positions"] == "2B, SS"
        assert move["drop_name"] == "Marcus Semien"
        assert move["drop_player_id"] == "456"
        assert move["drop_positions"] == "2B"

    def test_add_only_transaction(self):
        raw = [
            _make_raw_txn("1", "successful", "add", "Team A", "t.1",
                          adds=[{"name": "Player X", "positions": ["OF"]}]),
        ]
        result = parse_all_transactions(raw)
        assert result[0]["add_name"] == "Player X"
        assert result[0]["drop_name"] is None

    def test_drop_only_transaction(self):
        raw = [
            _make_raw_txn("1", "successful", "drop", "Team A", "t.1",
                          drops=[{"name": "Player Y", "positions": ["SP"]}]),
        ]
        result = parse_all_transactions(raw)
        assert result[0]["drop_name"] == "Player Y"
        assert result[0]["add_name"] is None

    def test_excludes_pending(self):
        raw = [
            _make_raw_txn("1", "successful", "add/drop", "Team A", "t.1",
                          adds=[{"name": "A"}], drops=[{"name": "B"}]),
            _make_raw_txn("2", "pending", "add/drop", "Team A", "t.1",
                          adds=[{"name": "C"}], drops=[{"name": "D"}]),
        ]
        result = parse_all_transactions(raw)
        assert len(result) == 1
        assert result[0]["transaction_id"] == "1"

    def test_empty_input(self):
        assert parse_all_transactions([]) == []
