"""Integration test: an EntityEvent and a SequenceOfEvents coexisting in one Report.

EntityEvent extended the shared ``EVENT_INSTANCE_FACT_SCHEMA`` with an
``entity_key`` column (and widened ``event_instance_id`` to ``LongType``).
``SequenceOfEvents`` was updated to emit ``entity_key = NULL`` to conform. Both
event types map to the same fact table (``event_instance_fact``), so the persist
path (``Report.persist_results`` → ``reduce(unionByName)``) unions their fact
rows and writes them atomically to one table.

This test guards that coexistence: a report with both events produces two fact
DataFrames that share one schema, carry the right ``entity_key`` (populated map
for the EntityEvent, ``NULL`` for the SequenceOfEvents), and ``unionByName``
cleanly — the exact operation the shared-table write performs.
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
from impulse_reporting.events.event_types import EventType
from impulse_reporting.events.sequence_of_events import SequenceOfEvents
from impulse_query_engine.surfaces.series import Series

# RLE object-tracks series (carries its own intervals; no container_stop_ts
# needed). Close object in containers 1 and 3, far in container 2.
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


def _config() -> ImpulseConfig:
    return ImpulseConfig(
        source=Source(
            container_metrics_table="spark_catalog.silver.container_metrics",
            channel_metrics_table="spark_catalog.silver.channel_metrics",
            channels_uri="spark_catalog.silver.channels",
        ),
        unity_sink=UnitySink(
            catalog="spark_catalog", schema="gold", table_prefix="entity_seq_coexist_test"
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


def test_entity_and_sequence_events_share_one_fact_schema(spark, basic_narrow_db):
    my_report = Report(
        name="entity_seq_report",
        spark=spark,
        workspace_client=create_autospec(WorkspaceClient),
        config=dict(_config()),
    )

    db = my_report.get_db()

    # EntityEvent over a registered series (entity_key populated).
    db.register_series(
        _object_tracks_series(),
        lambda spark: spark.createDataFrame(_OBJECT_TRACKS_ROWS, _OBJECT_TRACKS_SCHEMA),
    )
    close_object = (db.query.series("object_tracks").distance_m < 8.0).each().ids(as_="object")
    my_report.add_event(EntityEvent(name="close_object", expr=close_object))

    # SequenceOfEvents over a scalar channel (entity_key must be NULL).
    veh_spd = db.query.channel(channel_name="Vehicle Speed Sensor")
    my_report.add_event(
        SequenceOfEvents(
            name="speed_transition",
            expressions=[(veh_spd > 0) & (veh_spd < 15), (veh_spd > 9) & (veh_spd < 18)],
        )
    )

    my_report.determine_report()

    event_dfs = my_report.event_dfs
    assert "ENTITY_EVENT" in event_dfs
    assert "SEQUENCE_OF_EVENTS" in event_dfs

    # Both event types target the same physical fact table — this is why the
    # entity_key column has to coexist.
    assert (
        EventType.ENTITY_EVENT.get_fact_table_name()
        == EventType.SEQUENCE_OF_EVENTS.get_fact_table_name()
        == "event_instance_fact"
    )

    entity_df = event_dfs["ENTITY_EVENT"]["changed"]
    sequence_df = event_dfs["SEQUENCE_OF_EVENTS"]["changed"]

    # EntityEvent rows carry the populated nested JSON map.
    entity_by_container = {r.container_id: r for r in entity_df.collect()}
    assert set(entity_by_container) == {1, 3}
    assert entity_by_container[1].entity_key == '{"object": {"lidar": ["47"]}}'
    assert entity_by_container[3].entity_key == '{"object": {"lidar": ["12"]}}'

    # SequenceOfEvents rows carry NULL entity_key (no per-entity scope).
    sequence_rows = sequence_df.collect()
    assert len(sequence_rows) > 0
    assert all(r.entity_key is None for r in sequence_rows)

    # The shared-table write unions both types by name — reproduce it and assert
    # the union carries both NULL and populated entity_key, and the widened
    # (LongType) instance id holds the crc32 values without truncation.
    combined = entity_df.unionByName(sequence_df)
    assert combined.count() == entity_df.count() + sequence_df.count()
    entity_keys = [r.entity_key for r in combined.collect()]
    assert None in entity_keys  # sequence rows
    assert any(v is not None for v in entity_keys)  # entity rows
    assert combined.schema["event_instance_id"].dataType == T.LongType()
