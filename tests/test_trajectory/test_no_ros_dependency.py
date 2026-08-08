"""The trajectory model reads REALIZED stats, prorated. Never a projection blend.

`board_inputs` -> `_paced` -> `prorate_partial` divides realized SGP by the elapsed
fraction, and that paced figure is what every fit is anchored on. Blending an ROS
projection into it would change what the chart's anchor MEANS -- from "on this pace"
to "on this pace, adjusted by somebody's forecast" -- while every number on the page
still rendered and every other test still passed. #346.

Surfaces are enumerated by name rather than matched on a "ROS" substring: a concept
this test cannot check is a test that asserts nothing.
"""

from __future__ import annotations

from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]

#: Import paths and symbols that would pull a projection blend into the trajectory fit.
FORBIDDEN = (
    "fantasy_baseball.data.ros_pipeline",
    "fantasy_baseball.data.ros_export_ingest",
    "fantasy_baseball.data.projections",
    "ROS_PROJECTIONS",
)


def _scoped_sources() -> list[Path]:
    files = sorted((PROJECT_ROOT / "src" / "fantasy_baseball" / "trajectory").rglob("*.py"))
    files.append(PROJECT_ROOT / "scripts" / "push_trajectory_board.py")
    return files


def test_the_scope_is_not_empty() -> None:
    """A glob that silently matched nothing would make every assertion below vacuous."""
    files = _scoped_sources()
    assert len(files) > 5
    assert all(path.is_file() for path in files)


@pytest.mark.parametrize("path", _scoped_sources(), ids=lambda p: p.name)
def test_the_trajectory_model_reads_no_ros_projection(path: Path) -> None:
    source = path.read_text(encoding="utf-8")
    hits = [name for name in FORBIDDEN if name in source]
    assert not hits, (
        f"{path.name} references {hits}. The trajectory model is anchored on PRORATED "
        "REALIZED stats (prorate_partial), and blending a projection into that changes "
        "what every fit on the keeper board means. If this is deliberate, the change "
        "belongs in the spec first -- see #346."
    )
