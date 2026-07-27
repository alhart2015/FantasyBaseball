import pytest

from fantasy_baseball.keepers.actuals import coerce_numeric, innings_to_float


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("5.1", 5 + 1 / 3),
        ("5.2", 5 + 2 / 3),
        ("7.0", 7.0),
        ("0.1", 1 / 3),
        ("12", 12.0),
        (0, 0.0),
        (None, 0.0),
        ("", 0.0),
    ],
)
def test_innings_to_float(raw: object, expected: float) -> None:
    assert innings_to_float(raw) == pytest.approx(expected)


def test_innings_rejects_impossible_outs() -> None:
    # Only .0/.1/.2 are legal; .3 would silently become a third of an inning too many.
    with pytest.raises(ValueError):
        innings_to_float("5.3")


def test_coerce_numeric_handles_api_junk() -> None:
    assert coerce_numeric("3.45") == pytest.approx(3.45)
    assert coerce_numeric(None) == 0.0
    assert coerce_numeric("-.--") == 0.0
    assert coerce_numeric("") == 0.0
