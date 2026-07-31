"""Neutral example: adapt factory telemetry to a standard Impulse Report."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product

import pyspark.sql.functions as F
from databricks.sdk import WorkspaceClient

from impulse_query_engine.analyze.query.solvers import (
    DefaultSolver,
    SolverConfig,
    register_solver,
)
from impulse_reporting.core.report import Report
from impulse_reporting.sources import (
    ContainerDimension,
    LogicalChannel,
    ResolvedChannelMapping,
    SourceAdapter,
    register_source,
)


class FactorySolverConfig(SolverConfig):
    """Place for factory-specific read/reshape settings."""


@register_solver("FactoryTelemetrySolver", FactorySolverConfig)
class FactoryTelemetrySolver(DefaultSolver):
    """Example custom solver; real adapters override only physical read/reshape seams."""


@dataclass(frozen=True)
class FactoryTables:
    container_metrics: str = "factory.silver.container_metrics"
    container_tags: str = "factory.silver.container_tags"
    channel_metrics: str = "factory.silver.channel_metrics"
    channel_mapping: str = "factory.silver.channel_mapping"
    channels: str = "factory.silver.channels"


@register_source("factory-telemetry")
class FactoryTelemetrySource(SourceAdapter):
    """Discover plants/lines and logical temperature/vibration channels."""

    def __init__(self, tables: FactoryTables | None = None, workspace_client=None):
        self.tables = tables or FactoryTables()
        self.workspace_client = workspace_client

    @property
    def name(self) -> str:
        return "factory-telemetry"

    def list_dimensions(self, spark):
        return [ContainerDimension("plant", required=True), ContainerDimension("line")]

    def _wide_container_tags(self, spark):
        return (
            spark.table(self.tables.container_tags)
            .groupBy("container_id")
            .pivot("key")
            .agg(F.first("value"))
        )

    def _containers(self, spark, container_filters=None):
        frame = self._wide_container_tags(spark)
        for dimension, values in (container_filters or {}).items():
            frame = frame.where(F.col(dimension).isin(list(values)))
        return frame.select("container_id").distinct()

    def list_dimension_values(self, spark, dimension, *, container_filters=None):
        dimension = self._resolve_name("dimension", dimension, ["plant", "line"])
        frame = self._wide_container_tags(spark)
        for key, values in (container_filters or {}).items():
            if key != dimension:
                frame = frame.where(F.col(key).isin(list(values)))
        return [
            row[dimension]
            for row in frame.select(dimension).distinct().orderBy(dimension).collect()
        ]

    def list_channels(self, spark, *, container_filters=None):
        containers = self._containers(spark, container_filters)
        mappings = spark.table(self.tables.channel_mapping)
        metrics = spark.table(self.tables.channel_metrics).join(containers, "container_id")
        available = (
            metrics.join(mappings, metrics.channel_id == mappings.source_channel)
            .groupBy("channel_alias")
            .agg(F.sort_array(F.collect_set("source_channel")).alias("physical_channels"))
            .orderBy("channel_alias")
            .collect()
        )
        return [
            LogicalChannel(row.channel_alias, physical_channels=tuple(row.physical_channels))
            for row in available
        ]

    @staticmethod
    def _tag_filter_dnf(container_filters):
        dimensions = sorted(container_filters)
        return [
            [
                {"tag_name": dimension, "comparator": "==", "value": value}
                for dimension, value in zip(dimensions, values, strict=True)
            ]
            for values in product(*(container_filters[dimension] for dimension in dimensions))
        ]

    def create_report(self, spark, *, name, container_filters, channels, sink=None):
        filters = self.validate_container_filters(spark, container_filters, required=True)
        self.validate_channels(spark, channels, container_filters=filters)
        config = {
            "source": {
                "container_metrics_table": self.tables.container_metrics,
                "container_tags_table": self.tables.container_tags,
                "channel_metrics_table": self.tables.channel_metrics,
                "channel_mapping_table": self.tables.channel_mapping,
                "channels_uri": self.tables.channels,
            },
            "container_filters": {"tag_filters": self._tag_filter_dnf(filters)},
            "query_engine": {"solver": "FactoryTelemetrySolver", "data_type": "RLE"},
        }
        if sink is not None:
            config["unity_sink"] = dict(sink)
        return Report(
            name=name,
            spark=spark,
            workspace_client=self.workspace_client or WorkspaceClient(),
            config=config,
        )

    def resolve_channel_mappings(self, spark, *, container_filters, channels):
        filters = self.validate_container_filters(spark, container_filters, required=True)
        selected = self.validate_channels(spark, channels, container_filters=filters)
        rows = (
            spark.table(self.tables.channel_mapping)
            .where(F.col("channel_alias").isin(list(selected)))
            .orderBy("channel_alias", "priority")
            .collect()
        )
        return [
            ResolvedChannelMapping(row.channel_alias, row.source_channel, row.priority)
            for row in rows
        ]


def configured_report(spark, plant: str, line: str) -> Report:
    """Small usage example; downstream analysis remains native TSAL."""
    source = FactoryTelemetrySource()
    report = source.create_report(
        spark,
        name="factory-analysis",
        container_filters={"plant": [plant], "line": [line]},
        channels=["temperature", "vibration"],
        sink=None,
    )
    temperature = report.get_db().query.channel_with_alias(channel_alias="temperature")
    report.get_db().query.select(temperature.mean().alias("mean_temperature")).solve(
        spark, solver=report.get_solver()
    )
    return report
