"""Local conftest for mdf4_sample tests.

The conversion script under test lives at
`demos/byod/adapters/mdf4_sample/convert_pandaset.py` and imports its helpers
as `from .lib_something import ...` would if it had submodules. We add the demo
root to sys.path so `from adapters.mdf4_sample import convert_pandaset` resolves
when tests are collected from the repo root.

Also overrides the session-autouse Spark fixtures from tests/conftest.py with
no-op fixtures — the conversion script and the round-trip test are pure Python
plus asammdf, no JVM/Spark needed.
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
