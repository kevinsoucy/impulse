"""Test-scope overrides for BYOD demo tests.

The demo's pure-Python helpers (`lib/sensor_kpi.py` and all `adapters/<name>/`
mapper modules) don't touch Spark. We override the session-autouse Spark
fixtures from the repo-root tests/conftest.py with no-ops so these tests run
without a JVM.
"""

import pytest


@pytest.fixture(scope="session")
def spark():
    yield None


@pytest.fixture(scope="session", autouse=True)
def setup_basic_db():
    yield


@pytest.fixture(autouse=True)
def ensure_silver_basic():
    # Overrides the repo-root autouse fixture, which touches
    # spark.catalog. These pure-Python demo tests no-op Spark, so neutralize it.
    yield


@pytest.fixture(scope="function", autouse=True)
def cleanup_gold():
    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_schemas():
    yield
