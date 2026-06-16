"""Local conftest for A2D2 BYOD adapter unit tests.

Adds `demos/byod/` to sys.path so `from adapters.a2d2 import ...` resolves
the package's intra-package relative imports. Overrides the repo-root Spark
fixtures with no-ops — the modules under test are pure Python.
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
