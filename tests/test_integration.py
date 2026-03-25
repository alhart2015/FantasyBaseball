# This file intentionally left empty.
# The integration tests that were here have been moved to
# tests/test_integration/test_sgp_pipeline.py.
#
# This file cannot be deleted because it would recreate the naming
# conflict between this module and the test_integration/ package.
# However, it must remain empty (no test functions) to avoid the
# "import file mismatch" error during pytest collection.
#
# TODO: Delete this file once pytest is configured with
# importmode = "importlib" in pyproject.toml, or rename
# the test_integration/ directory.
