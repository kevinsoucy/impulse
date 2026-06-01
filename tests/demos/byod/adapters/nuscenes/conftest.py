"""Local conftest for NuScenes BYOD adapter unit tests.

The adapter package lives under `demos/byod/adapters/nuscenes/` and imports
its helpers as `from .loader import ...`. We add the demo root to sys.path
and import the package via its full path (`adapters.nuscenes`) so its
intra-package relative imports resolve when tests are collected from the
repo root.

Also overrides the session-autouse Spark fixtures from tests/conftest.py with
no-op fixtures so these tests run without a JVM — the mapper modules under
test are pure Python and don't touch Spark.
"""

import sys
from pathlib import Path

import pytest

DEMO_ROOT = Path(__file__).resolve().parents[5] / "demos" / "byod"
if str(DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(DEMO_ROOT))


@pytest.fixture(scope="session")
def spark():
    yield None


@pytest.fixture(scope="session", autouse=True)
def setup_basic_db():
    yield


@pytest.fixture(scope="function", autouse=True)
def cleanup_gold():
    yield


@pytest.fixture(scope="session", autouse=True)
def cleanup_schemas():
    yield
