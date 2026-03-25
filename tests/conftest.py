import pytest
from pathlib import Path

# Ignore the old test_integration.py file to avoid naming conflict
# with the test_integration/ package directory. Tests were migrated
# to test_integration/test_sgp_pipeline.py.
collect_ignore = ["test_integration.py"]


@pytest.fixture
def fixtures_dir():
    return Path(__file__).parent / "fixtures"
