import pytest

from impulse_reporting.sources.registry import (
    _clear_registry,
    register_source,
    registered_sources,
    resolve_source,
)
from impulse_reporting.sources.source_adapter import SourceAdapter


class MinimalSource(SourceAdapter):
    def __init__(self, name="minimal"):
        self._name = name

    @property
    def name(self):
        return self._name

    def list_dimensions(self, spark):
        return []

    def list_dimension_values(self, spark, dimension, *, container_filters=None):
        return []

    def list_channels(self, spark, *, container_filters=None):
        return []

    def create_report(self, spark, *, name, container_filters, channels, sink=None):
        return None

    def resolve_channel_mappings(self, spark, *, container_filters, channels):
        return []


@pytest.fixture(autouse=True)
def isolated_registry():
    _clear_registry()
    yield
    _clear_registry()


def test_register_list_and_resolve_single_source():
    register_source("minimal", lambda: MinimalSource())
    assert registered_sources() == ["minimal"]
    assert resolve_source().name == "minimal"


def test_duplicate_and_default_conflicts_are_explicit():
    register_source("a", lambda: MinimalSource("a"), default=True)
    with pytest.raises(ValueError, match="already registered"):
        register_source("a", lambda: MinimalSource("replacement"))
    with pytest.raises(ValueError, match="default source"):
        register_source("b", lambda: MinimalSource("b"), default=True)


def test_missing_and_ambiguous_resolution_are_explicit():
    with pytest.raises(LookupError, match="No source adapters"):
        resolve_source()
    register_source("a", lambda: MinimalSource("a"))
    register_source("b", lambda: MinimalSource("b"))
    with pytest.raises(LookupError, match="Multiple sources"):
        resolve_source()
    with pytest.raises(KeyError, match="Unknown source"):
        resolve_source("missing")


def test_named_default_wins_when_multiple_sources_are_installed():
    register_source("a", lambda: MinimalSource("a"), default=True)
    register_source("b", lambda: MinimalSource("b"))
    assert resolve_source().name == "a"
