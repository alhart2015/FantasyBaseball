"""One door between the trajectory model and the projection layer, and it is `ros_anchor`.

The retired `test_no_ros_dependency.py` (#346) forbade the whole package from importing
a projection blend. #348 makes that exactly wrong for ONE module -- and deleting the file
outright dropped the guard for the other nine, which is a wider hole than the change
needed. The anchor is a deliberate, single, reviewed seam; a second module reaching for
`ros_projections` on its own is how the trajectory fit quietly acquires a forecast in a
place nobody decided to put one.

Surfaces are enumerated by name rather than matched on a "ROS" substring: a concept this
test cannot check is a test that asserts nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Import paths and symbols that pull a projection blend into the trajectory model.
FORBIDDEN = (
    "fantasy_baseball.data.ros_pipeline",
    "fantasy_baseball.data.ros_export_ingest",
    "fantasy_baseball.data.projections",
    "ROS_PROJECTIONS",
)

#: The one module allowed through, and the only one. Widening this set is a design
#: decision about where the forecast enters the model, not a test fix.
THE_DOOR = "ros_anchor.py"


def _scoped_sources() -> list[Path]:
    package = PROJECT_ROOT / "src" / "fantasy_baseball" / "trajectory"
    return sorted(p for p in package.rglob("*.py") if p.name != THE_DOOR)


def test_the_scope_is_not_empty() -> None:
    """A glob that silently matched nothing would make every assertion below vacuous."""
    files = _scoped_sources()
    assert len(files) > 5
    assert all(path.is_file() for path in files)
    assert not any(path.name == THE_DOOR for path in files), "the door is excluded by name"


def test_the_door_itself_exists() -> None:
    """If `ros_anchor.py` is ever renamed, `THE_DOOR` stops excluding anything and this
    file starts forbidding the one import that is supposed to be there -- which reads as
    a broken test rather than as the rename it is."""
    assert (PROJECT_ROOT / "src" / "fantasy_baseball" / "trajectory" / THE_DOOR).is_file()


@pytest.mark.parametrize("path", _scoped_sources(), ids=lambda p: p.name)
def test_only_the_anchor_reads_a_projection(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    hits = [name for name in FORBIDDEN if name in source]
    assert not hits, (
        f"{path.name} references {hits}. The rest-of-season projection enters the "
        f"trajectory model through {THE_DOOR} and nowhere else -- that seam is where the "
        f"combination order, the era-factor source and the no-ROS rule are enforced "
        f"(#348). A second entry point bypasses all three."
    )
