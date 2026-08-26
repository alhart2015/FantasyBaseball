"""The invariant that makes `-n auto` the default safe (#364).

OpenBLAS sizes its thread pool to the MACHINE, not to how many processes are
sharing it: 24 threads per process on a 32-core box. `-n auto` without a cap is
32 workers x 24 threads for 32 cores, and the result is not "slower" -- it is
starvation that surfaces as unrelated tests failing. Measured on the full suite:

    -n auto, unpinned : 14m52s, 2 failed + 3 errors
    -n auto, pinned   :  8m15s, all green
    serial            : 22m37s

If `pytest_configure` in `tests/conftest.py` is ever removed, the suite goes
back to failing in whichever test happens to be holding a browser or a fixture
when the machine runs out of room. That reads as a flaky test rather than as a
configuration bug, which is why it stayed unexplained for so long -- so it is
pinned here instead of trusted to a comment.
"""

from __future__ import annotations

import pytest


def _blas_pools() -> list[int]:
    from threadpoolctl import threadpool_info

    return [p["num_threads"] for p in threadpool_info() if p["user_api"] == "blas"]


def test_an_xdist_worker_gets_exactly_one_blas_thread(request):
    if getattr(request.config, "workerinput", None) is None:
        pytest.skip("serial run: the pool is deliberately left alone, nothing to share")

    pools = _blas_pools()

    assert pools, "no BLAS pool found -- threadpool_info could not see numpy's backend"
    assert all(n == 1 for n in pools), (
        f"an xdist worker is running with BLAS pools {pools}. Every worker "
        f"multiplies this by the worker count; see tests/conftest.py."
    )
