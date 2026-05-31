"""Integration test: a cross-series EntityEvent end-to-end through a Report.

Verifies that an ``EntityEvent`` registered on a ``Report`` is dispatched
through its own ``determine_events`` path (the series cogroup), not the
centralized presence ``solved_df``, and produces ``event_instance_fact`` rows
with a populated ``entity_key`` nested JSON map.
"""

from unittest.mock import create_autospec

import pyspark.sql.types as T
from databricks.sdk import WorkspaceClient

from impulse_reporting.config.config_parser import (
    Comparator,
    ContainerFilters,
    ImpulseConfig,
    MeasurementDimensions,
    MetricFilter,
    QueryEngine,
    Solvers,
    Source,
    UnitySink,
)
from impulse_reporting.core.report import Report
from impulse_reporting.events.entity_event import EntityEvent
from impulse_query_engine.surfaces.series import Series
from tests.conftest import spark  # noqa: F401  (pytest fixture)

# An RLE object-tracks series so interval synthesis needs no container_stop_ts:
# each row carries its own [tstart, tend). One entity per container — close in
# containers 1 and 3, far in container 2.
_OBJECT_TRACKS_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("tstart", T.LongType(), nullable=False),
        T.StructField("tend", T.LongType(), nullable=False),
        T.StructField("entity_id", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType()),
    ]
)
_OBJECT_TRACKS_ROWS = [
    (1, "lidar", 0, 10, 47, 5.0),
    (2, "lidar", 0, 10, 91, 99.0),
    (3, "lidar", 0, 10, 12, 4.0),
]


def _object_tracks_series() -> Series:
    return Series(
        name="object_tracks",
        schema=_OBJECT_TRACKS_SCHEMA,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="tstart",
        tend_col="tend",
        entity_key="entity_id",
    )


def test_entity_event_in_report_populates_entity_key(spark, basic_narrow_db):
    impulse_config = ImpulseConfig(
        source=Source(
            container_metrics_table="spark_catalog.silver.container_metrics",
            channel_metrics_table="spark_catalog.silver.channel_metrics",
            channels_uri="spark_catalog.silver.channels",
        ),
        unity_sink=UnitySink(
            catalog="spark_catalog",
            schema="gold",
            table_prefix="entity_event_test",
        ),
        container_filters=ContainerFilters(
            metric_filters=[
                [
                    MetricFilter(
                        column_name="vehicle_key", comparator=Comparator.EQ, value="Seat_Leon"
                    ),
                    MetricFilter(
                        column_name="start_dt",
                        comparator=Comparator.GE,
                        value="2025-07-03T07:00:00.000Z",
                    ),
                ]
            ]
        ),
        query_engine=QueryEngine(solver=Solvers.KEY_VALUE_STORE_SOLVER),
        measurement_dimensions=[
            MeasurementDimensions.CONTAINER_ID,
            MeasurementDimensions.START_TS,
            MeasurementDimensions.STOP_TS,
        ],
    )

    my_report = Report(
        name="entity_event_report",
        spark=spark,
        workspace_client=create_autospec(WorkspaceClient),
        config=dict(impulse_config),
    )

    # Register the series on the report's db, then author the EntityEvent against it.
    db = my_report.get_db()
    db.register_series(
        _object_tracks_series(),
        lambda spark: spark.createDataFrame(_OBJECT_TRACKS_ROWS, _OBJECT_TRACKS_SCHEMA),
    )
    close_object = (db.query.series("object_tracks").distance_m < 8.0).entity_condition()
    my_report.add_event(EntityEvent(name="close_object", expr=close_object))

    my_report.determine_report()

    event_dfs = my_report.event_dfs
    assert "ENTITY_EVENT" in event_dfs, "EntityEvent must dispatch under its own type"

    rows = event_dfs["ENTITY_EVENT"]["changed"].collect()
    by_container = {r.container_id: r for r in rows}

    # Containers 1 and 3 have a close object; container 2 (far) produces no row.
    assert set(by_container) == {1, 3}
    assert by_container[1].entity_key == '{"object_tracks": {"lidar": ["47"]}}'
    assert by_container[3].entity_key == '{"object_tracks": {"lidar": ["12"]}}'
    # Entity-attributed rows carry a real (non-sentinel) instance id and a valid window.
    for r in rows:
        assert r.start_ts < r.end_ts
        assert r.event_instance_id is not None

    # The dimension row is emitted exactly once for the event definition.
    dim_rows = my_report.event_metadata_dfs["ENTITY_EVENT"].collect()
    assert len(dim_rows) == 1
    assert dim_rows[0].event_name == "close_object"
