from unittest.mock import Mock

import pytest

from impulse_reporting.sources.source_adapter import (
    ContainerDimension,
    LogicalChannel,
    ResolvedChannelMapping,
    SourceAdapter,
    SourceSelectionError,
)


class ExampleSource(SourceAdapter):
    name = "example"

    def list_dimensions(self, spark):
        return [ContainerDimension("plant"), ContainerDimension("Plant")]

    def list_dimension_values(self, spark, dimension, *, container_filters=None):
        return ["Berlin", "Dresden"]

    def list_channels(self, spark, *, container_filters=None):
        return [
            LogicalChannel("temperature", physical_channels=("temp_a",)),
            LogicalChannel("vibration", physical_channels=("vib_a",)),
        ]

    def create_report(self, spark, *, name, container_filters, channels, sink=None):
        return Mock(name="report", sink=sink)

    def resolve_channel_mappings(self, spark, *, container_filters, channels):
        return [ResolvedChannelMapping(channels[0], "temp_a", 1, "recording-1")]


def test_selection_validation_resolves_values_and_channels():
    source = ExampleSource()
    filters = source.validate_container_filters(None, {"plant": ["berlin"]})
    channels = source.validate_channels(None, ["Temperature"], container_filters=filters)
    assert filters == {"plant": ("Berlin",)}
    assert channels == ("temperature",)


def test_selection_validation_rejects_empty_missing_and_ambiguous():
    source = ExampleSource()
    with pytest.raises(SourceSelectionError, match="requires at least one"):
        source.validate_container_filters(None, None, required=True)
    with pytest.raises(SourceSelectionError, match="non-empty"):
        source.validate_container_filters(None, {"plant": []})
    with pytest.raises(SourceSelectionError, match="Ambiguous dimension"):
        source.validate_container_filters(None, {"PLANT": ["Berlin"]})
    with pytest.raises(SourceSelectionError, match="Unknown logical channel"):
        source.validate_channels(None, ["pressure"])


def test_sinkless_report_and_mapping_diagnostics_are_public():
    source = ExampleSource()
    report = source.create_report(
        None,
        name="analysis",
        container_filters={"plant": ["Berlin"]},
        channels=["temperature"],
    )
    mappings = source.resolve_channel_mappings(
        None,
        container_filters={"plant": ["Berlin"]},
        channels=["temperature"],
    )
    assert report.sink is None
    assert mappings == [ResolvedChannelMapping("temperature", "temp_a", 1, "recording-1")]
