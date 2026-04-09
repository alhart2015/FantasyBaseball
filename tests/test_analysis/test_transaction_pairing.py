from fantasy_baseball.analysis.transactions import pair_standalone_moves


def _make_txn(txn_id, team, ttype, timestamp,
              add_name=None, add_positions=None,
              drop_name=None, drop_positions=None):
    return {
        "transaction_id": txn_id,
        "team": team,
        "type": ttype,
        "timestamp": timestamp,
        "add_name": add_name,
        "add_positions": add_positions,
        "drop_name": drop_name,
        "drop_positions": drop_positions,
        "paired_with": None,
    }


class TestPairStandaloneMoves:
    def test_pairs_drop_then_add_same_team_within_24h(self):
        txns = [
            _make_txn("1", "Team A", "drop", "1712700000",
                      drop_name="Player X", drop_positions="3B, Util"),
            _make_txn("2", "Team A", "add", "1712750000",
                      add_name="Player Y", add_positions="3B, SS"),
        ]
        pairs = pair_standalone_moves(txns)
        assert len(pairs) == 1
        assert pairs[0] == ("1", "2")

    def test_no_pairing_across_teams(self):
        txns = [
            _make_txn("1", "Team A", "drop", "1712700000",
                      drop_name="Player X", drop_positions="OF"),
            _make_txn("2", "Team B", "add", "1712750000",
                      add_name="Player Y", add_positions="OF"),
        ]
        pairs = pair_standalone_moves(txns)
        assert len(pairs) == 0

    def test_no_pairing_beyond_24h(self):
        txns = [
            _make_txn("1", "Team A", "drop", "1712700000",
                      drop_name="Player X", drop_positions="OF"),
            _make_txn("2", "Team A", "add", "1712900000",
                      add_name="Player Y", add_positions="OF"),
        ]
        pairs = pair_standalone_moves(txns)
        assert len(pairs) == 0

    def test_position_match_preferred(self):
        txns = [
            _make_txn("1", "Team A", "drop", "1712700000",
                      drop_name="Dropped 3B", drop_positions="3B, Util"),
            _make_txn("2", "Team A", "add", "1712750000",
                      add_name="Added OF", add_positions="OF, Util"),
            _make_txn("3", "Team A", "add", "1712750000",
                      add_name="Added 3B", add_positions="3B, Util"),
        ]
        pairs = pair_standalone_moves(txns)
        assert len(pairs) == 1
        assert pairs[0] == ("1", "3")

    def test_type_match_fallback(self):
        txns = [
            _make_txn("1", "Team A", "drop", "1712700000",
                      drop_name="Dropped SS", drop_positions="SS"),
            _make_txn("2", "Team A", "add", "1712750000",
                      add_name="Added OF", add_positions="OF"),
        ]
        pairs = pair_standalone_moves(txns)
        assert len(pairs) == 1
        assert pairs[0] == ("1", "2")

    def test_no_cross_type_pairing(self):
        txns = [
            _make_txn("1", "Team A", "drop", "1712700000",
                      drop_name="Dropped SS", drop_positions="SS"),
            _make_txn("2", "Team A", "add", "1712750000",
                      add_name="Added SP", add_positions="SP"),
        ]
        pairs = pair_standalone_moves(txns)
        assert len(pairs) == 0

    def test_already_paired_skipped(self):
        txns = [
            _make_txn("1", "Team A", "drop", "1712700000",
                      drop_name="Player X", drop_positions="OF"),
            _make_txn("2", "Team A", "add", "1712750000",
                      add_name="Player Y", add_positions="OF"),
        ]
        txns[0]["paired_with"] = "99"
        pairs = pair_standalone_moves(txns)
        assert len(pairs) == 0

    def test_add_drop_transactions_skipped(self):
        txns = [
            _make_txn("1", "Team A", "add/drop", "1712700000",
                      add_name="A", add_positions="OF",
                      drop_name="B", drop_positions="OF"),
        ]
        pairs = pair_standalone_moves(txns)
        assert len(pairs) == 0
