import pytest


@pytest.fixture
def app(tmp_path, monkeypatch):
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers: []\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    return a


@pytest.fixture
def client(app):
    with app.test_client() as c:
        yield c


def test_new_draft_seeds_state(client):
    r = client.post("/api/new-draft")
    assert r.status_code == 200
    body = r.get_json()
    assert body["on_the_clock"] == "Hart of the Order"
    assert body["picks"] == []


def test_pick_then_undo_round_trips(client):
    client.post("/api/new-draft")
    r = client.post(
        "/api/pick",
        json={
            "player_id": "P1::hitter",
            "player_name": "Player One",
            "position": "OF",
            "team": "Hart of the Order",
        },
    )
    assert r.status_code == 200
    assert r.get_json()["on_the_clock"] == "Opp"

    r = client.post("/api/undo")
    assert r.status_code == 200
    assert r.get_json()["on_the_clock"] == "Hart of the Order"


def test_pick_wrong_team_rejected(client):
    client.post("/api/new-draft")
    r = client.post(
        "/api/pick",
        json={
            "player_id": "P1::hitter",
            "player_name": "Player One",
            "position": "OF",
            "team": "Opp",
        },
    )
    assert r.status_code == 409


def test_reset_deletes_state(client, tmp_path):
    client.post("/api/new-draft")
    r = client.post("/api/reset", json={"confirm": "RESET"})
    assert r.status_code == 200
    assert not (tmp_path / "draft_state.json").exists()


def test_reset_without_confirm_rejected(client):
    client.post("/api/new-draft")
    r = client.post("/api/reset", json={})
    assert r.status_code == 400


def test_recs_endpoint_exists(client):
    client.post("/api/new-draft")
    r = client.get("/api/recs?team=Hart of the Order")
    # With no board file on disk, the real-data path can't run — the
    # endpoint signals 503 so the frontend can render a placeholder.
    assert r.status_code == 503


def test_pick_missing_fields_returns_400(client):
    client.post("/api/new-draft")
    r = client.post("/api/pick", json={"player_id": "P1::hitter"})  # missing most fields
    assert r.status_code == 400
    body = r.get_json()
    assert "missing" in body["error"].lower()


def test_on_the_clock_missing_team_returns_400(client):
    client.post("/api/new-draft")
    r = client.post("/api/on-the-clock", json={})
    assert r.status_code == 400


def test_recs_returns_ranked_rows(client, monkeypatch):
    """/api/recs returns a list of rec rows sorted by immediate_delta desc."""
    from fantasy_baseball.draft import eroto_recs
    from fantasy_baseball.draft.eroto_recs import RecRow

    def fake_rank(**_kwargs):
        return [
            RecRow("p1::hitter", "Player One", ["OF"], 3.2, 0.5, 1.1, {"HR": 1.2}),
            RecRow("p2::hitter", "Player Two", ["SS"], 2.1, 0.3, 0.4, {"SB": 0.9}),
        ]

    monkeypatch.setattr(eroto_recs, "rank_candidates", fake_rank)

    from fantasy_baseball.web import app as web_app

    monkeypatch.setattr(
        web_app, "_build_rec_inputs", lambda *_a, **_kw: (None, None, None, None, None)
    )
    monkeypatch.setattr(web_app, "_load_board_cached", lambda _app: None)
    monkeypatch.setattr(web_app, "_picks_until_next_turn", lambda state, team: 3)

    client.post("/api/new-draft")
    r = client.get("/api/recs?team=Hart of the Order")
    assert r.status_code == 200
    body = r.get_json()
    assert body[0]["name"] == "Player One"
    assert body[0]["immediate_delta"] == 3.2
    assert body[0]["per_category"]["HR"] == 1.2


def test_roster_endpoint_returns_empty_for_fresh_draft(client):
    client.post("/api/new-draft")
    r = client.get("/api/roster?team=Hart of the Order")
    assert r.status_code == 200
    body = r.get_json()
    # Empty league.yaml has no roster_slots -> no replacement padding.
    assert body == []


def test_roster_endpoint_reflects_a_pick(client):
    client.post("/api/new-draft")
    client.post(
        "/api/pick",
        json={
            "player_id": "P1::hitter",
            "player_name": "Player One",
            "position": "OF",
            "team": "Hart of the Order",
        },
    )
    r = client.get("/api/roster?team=Hart of the Order")
    assert r.status_code == 200
    body = r.get_json()
    assert any(row["name"] == "Player One" for row in body)


def test_roster_missing_team_returns_400(client):
    client.post("/api/new-draft")
    r = client.get("/api/roster")
    assert r.status_code == 400


def test_standings_endpoint_returns_empty_list_with_empty_cache(client):
    client.post("/api/new-draft")
    r = client.get("/api/standings")
    assert r.status_code == 200
    assert r.get_json() == []


def test_resolve_keeper_finds_pitcher_by_normalized_name(tmp_path, monkeypatch):
    """Real keeper resolver must honor player_type and best_position
    from the board — not hardcode hitter/OF."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web.app import create_app

    # Seed league.yaml with one pitcher keeper.
    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers:\n"
        "  - {name: Tarik Skubal, team: Hart of the Order}\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))

    # Seed a tiny board.
    board_path = tmp_path / "draft_state_board.json"
    write_board(
        [
            {
                "name": "Tarik Skubal",
                "name_normalized": "tarik skubal",
                "player_id": "12345::pitcher",
                "player_type": "pitcher",
                "positions": ["SP"],
                "best_position": "SP",
                "var": 8.5,
            },
        ],
        board_path,
    )

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True

    with a.test_client() as c:
        r = c.post("/api/new-draft")
        assert r.status_code == 200
        state = r.get_json()

    assert len(state["keepers"]) == 1
    keeper = state["keepers"][0]
    assert keeper["player_id"] == "12345::pitcher"  # real ID, not {name}::hitter
    assert keeper["position"] == "SP"  # real position, not "OF"
    assert keeper["player_name"] == "Tarik Skubal"


def test_resolve_keeper_tie_breaks_by_var(tmp_path, monkeypatch):
    """When two board rows share a normalized name, pick the one with
    higher VAR (real player vs namesake)."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers:\n"
        "  - {name: Jose Ramirez, team: Hart of the Order}\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))

    board_path = tmp_path / "draft_state_board.json"
    # Two Jose Ramirezes: the real one (high VAR) and a namesake (low VAR).
    write_board(
        [
            {
                "name": "Jose Ramirez",
                "name_normalized": "jose ramirez",
                "player_id": "111::hitter",
                "player_type": "hitter",
                "positions": ["3B"],
                "best_position": "3B",
                "var": 9.2,
            },
            {
                "name": "Jose Ramirez",
                "name_normalized": "jose ramirez",
                "player_id": "222::pitcher",
                "player_type": "pitcher",
                "positions": ["SP"],
                "best_position": "SP",
                "var": 0.5,
            },
        ],
        board_path,
    )

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True

    with a.test_client() as c:
        r = c.post("/api/new-draft")
        assert r.status_code == 200
        state = r.get_json()

    assert state["keepers"][0]["player_id"] == "111::hitter"


def test_resolve_keeper_missing_player_returns_400(tmp_path, monkeypatch):
    """Keeper not on board → UnresolvedKeeperError → HTTP 400."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web.app import create_app

    league_path = tmp_path / "league.yaml"
    league_path.write_text(
        "league:\n  team_name: Hart of the Order\n"
        "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
        "keepers:\n"
        "  - {name: Nobody Important, team: Hart of the Order}\n"
    )
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))
    write_board([], tmp_path / "draft_state_board.json")

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True

    with a.test_client() as c:
        r = c.post("/api/new-draft")
        assert r.status_code == 400
        assert "Nobody Important" in r.get_json()["error"]


# Shared 4-player board fixture for integration tests that need real
# stats: 2 hitters, 2 pitchers, enough to exercise project_team_stats.
_INTEGRATION_BOARD_ROWS = [
    {
        "name": "Slugger",
        "name_normalized": "slugger",
        "player_id": "1::hitter",
        "player_type": "hitter",
        "positions": ["OF"],
        "best_position": "OF",
        "var": 9.0,
        "r": 100.0,
        "hr": 40.0,
        "rbi": 110.0,
        "sb": 5.0,
        "avg": 0.280,
        "h": 160.0,
        "ab": 580.0,
    },
    {
        "name": "Slapper",
        "name_normalized": "slapper",
        "player_id": "2::hitter",
        "player_type": "hitter",
        "positions": ["OF"],
        "best_position": "OF",
        "var": 5.0,
        "r": 85.0,
        "hr": 10.0,
        "rbi": 55.0,
        "sb": 12.0,
        "avg": 0.300,
        "h": 170.0,
        "ab": 570.0,
    },
    {
        "name": "Ace",
        "name_normalized": "ace",
        "player_id": "3::pitcher",
        "player_type": "pitcher",
        "positions": ["SP"],
        "best_position": "SP",
        "var": 8.0,
        "w": 15.0,
        "k": 220.0,
        "sv": 0.0,
        "era": 3.10,
        "whip": 1.05,
        "ip": 195.0,
        "h_allowed": 160.0,
        "er": 67.0,
        "bb": 50.0,
    },
    {
        "name": "Closer",
        "name_normalized": "closer",
        "player_id": "4::pitcher",
        "player_type": "pitcher",
        "positions": ["RP"],
        "best_position": "RP",
        "var": 6.0,
        "w": 3.0,
        "k": 90.0,
        "sv": 35.0,
        "era": 2.50,
        "whip": 0.95,
        "ip": 65.0,
        "h_allowed": 50.0,
        "er": 18.0,
        "bb": 22.0,
    },
]


_INTEGRATION_LEAGUE_YAML = (
    "league:\n  team_name: Hart of the Order\n"
    "draft:\n  position: 1\n  teams:\n    1: Hart of the Order\n    2: Opp\n"
    "keepers: []\n"
    "roster_slots:\n  OF: 1\n  SP: 1\n  RP: 1\n  BN: 0\n  IL: 0\n"
)


def test_recs_returns_real_rows_with_board_and_picks(tmp_path, monkeypatch):
    """Integration: with a real board file on disk, /api/recs returns
    RecRow dicts with non-trivial immediate_delta values."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web.app import create_app

    board_path = tmp_path / "draft_state_board.json"
    write_board(_INTEGRATION_BOARD_ROWS, board_path)

    league_path = tmp_path / "league.yaml"
    league_path.write_text(_INTEGRATION_LEAGUE_YAML)
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    with a.test_client() as c:
        c.post("/api/new-draft")
        r = c.get("/api/recs?team=Hart of the Order")
    assert r.status_code == 200, r.get_json()
    body = r.get_json()
    assert isinstance(body, list)
    assert len(body) >= 1
    # Plausibility: at least one candidate has a non-zero immediate_delta.
    assert any(abs(row["immediate_delta"]) > 1e-6 for row in body), body


def test_standings_endpoint_returns_real_rows_after_pick(tmp_path, monkeypatch):
    """Integration: with a real board on disk, /api/pick refreshes
    projected_standings_cache and /api/standings renders it."""
    from fantasy_baseball.draft.state import write_board
    from fantasy_baseball.web.app import create_app

    board_path = tmp_path / "draft_state_board.json"
    write_board(_INTEGRATION_BOARD_ROWS, board_path)

    league_path = tmp_path / "league.yaml"
    league_path.write_text(_INTEGRATION_LEAGUE_YAML)
    monkeypatch.setenv("DRAFT_LEAGUE_YAML_PATH", str(league_path))

    a = create_app(state_path=tmp_path / "draft_state.json")
    a.config["TESTING"] = True
    with a.test_client() as c:
        c.post("/api/new-draft")
        c.post(
            "/api/pick",
            json={
                "player_id": "1::hitter",
                "player_name": "Slugger",
                "position": "OF",
                "team": "Hart of the Order",
            },
        )
        r = c.get("/api/standings")
    assert r.status_code == 200
    rows = r.get_json()
    assert len(rows) == 2  # two teams
    assert all(isinstance(row["total"], int | float) for row in rows)
    assert rows[0]["total"] >= rows[1]["total"]  # sorted desc
