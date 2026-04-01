import pytest
from fantasy_baseball.web.season_app import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["SECRET_KEY"] = "test"
    with app.test_client() as c:
        yield c


def test_players_page_renders(client):
    resp = client.get("/players")
    assert resp.status_code == 200
    assert b"Search players by name" in resp.data
