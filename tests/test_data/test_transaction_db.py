from fantasy_baseball.data.db import (
    create_tables,
    get_connection,
    insert_transactions,
    get_transaction_ids,
    get_all_transactions,
    update_transaction_pairing,
)


def _make_txn(txn_id, team="Team A", ttype="add/drop",
              add_name="Player X", drop_name="Player Y",
              add_wsgp=1.0, drop_wsgp=0.5, value=0.5,
              timestamp="1712700000"):
    return {
        "year": 2026,
        "transaction_id": txn_id,
        "timestamp": timestamp,
        "team": team,
        "team_key": "t.1",
        "type": ttype,
        "add_name": add_name,
        "add_player_id": "100",
        "add_positions": "OF, Util",
        "drop_name": drop_name,
        "drop_player_id": "200",
        "drop_positions": "OF, Util",
        "add_wsgp": add_wsgp,
        "drop_wsgp": drop_wsgp,
        "value": value,
        "paired_with": None,
    }


class TestTransactionDB:
    def test_insert_and_retrieve(self):
        conn = get_connection(":memory:")
        create_tables(conn)
        txns = [_make_txn("1"), _make_txn("2")]
        insert_transactions(conn, txns)
        result = get_all_transactions(conn, 2026)
        assert len(result) == 2

    def test_get_transaction_ids(self):
        conn = get_connection(":memory:")
        create_tables(conn)
        insert_transactions(conn, [_make_txn("1"), _make_txn("2")])
        ids = get_transaction_ids(conn, 2026)
        assert ids == {"1", "2"}

    def test_insert_is_idempotent(self):
        conn = get_connection(":memory:")
        create_tables(conn)
        insert_transactions(conn, [_make_txn("1")])
        insert_transactions(conn, [_make_txn("1")])  # duplicate
        assert len(get_all_transactions(conn, 2026)) == 1

    def test_update_pairing(self):
        conn = get_connection(":memory:")
        create_tables(conn)
        insert_transactions(conn, [_make_txn("1"), _make_txn("2")])
        update_transaction_pairing(conn, 2026, "1", "2")
        result = get_all_transactions(conn, 2026)
        t1 = next(r for r in result if r["transaction_id"] == "1")
        t2 = next(r for r in result if r["transaction_id"] == "2")
        assert t1["paired_with"] == "2"
        assert t2["paired_with"] == "1"

    def test_add_only_transaction(self):
        conn = get_connection(":memory:")
        create_tables(conn)
        txn = _make_txn("1", ttype="add", add_name="Player X",
                        drop_name=None, drop_wsgp=0, value=1.0)
        txn["drop_player_id"] = None
        txn["drop_positions"] = None
        insert_transactions(conn, [txn])
        result = get_all_transactions(conn, 2026)
        assert result[0]["drop_name"] is None
        conn.close()
