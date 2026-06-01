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

# Two distinct entities (47, 88) co-occur in the SAME container, event and window.
# They differ only by entity_key — the exact case where a 32-bit event_instance_id
# crc32 can collide.
_TWO_ENTITY_ROWS = [
    (1, "lidar", 0, 10, 47, 5.0),
    (1, "lidar", 0, 10, 88, 4.0),
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


def test_reduced_path_entity_key_matches_raw_frame_oracle(spark, basic_narrow_db):
    """The production Spark reduction path and the raw-frame oracle must produce
    identical per-entity ``entity_key`` maps for the same data.

    ``entity_key`` is rendered twice by different code: the Spark per-entity
    reduction stage (``_reduce_group_udf`` → ``render_entity_key``) on the
    production path, and ``EntityEvent.materialize_per_container`` over a raw
    ``MultiSeriesCache`` on the oracle path. ``MultiSeriesCache`` exists solely as
    that oracle, but nothing asserted the two agree — this closes that gap so a
    rendering divergence (numpy repr, ordering, scoping) can't slip through and
    desync ``entity_key`` (which is folded into ``event_instance_id``).
    """
    import json

    import pandas as pd

    from impulse_query_engine.analyze.query.solvers.empty_cache import EmptyTimeSeriesCache
    from impulse_query_engine.analyze.query.solvers.series_cache import CombinedSeriesCache
    from impulse_query_engine.surfaces import SeriesAccessor

    impulse_config = ImpulseConfig(
        source=Source(
            container_metrics_table="spark_catalog.silver.container_metrics",
            channel_metrics_table="spark_catalog.silver.channel_metrics",
            channels_uri="spark_catalog.silver.channels",
        ),
        unity_sink=UnitySink(catalog="spark_catalog", schema="gold", table_prefix="oracle_test"),
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
        name="oracle_report",
        spark=spark,
        workspace_client=create_autospec(WorkspaceClient),
        config=dict(impulse_config),
    )
    db = my_report.get_db()
    db.register_series(
        _object_tracks_series(),
        lambda spark: spark.createDataFrame(_TWO_ENTITY_ROWS, _OBJECT_TRACKS_SCHEMA),
    )
    close_object = (db.query.series("object_tracks").distance_m < 8.0).entity_condition()
    my_report.add_event(EntityEvent(name="close_object", expr=close_object))

    # Production reduced path.
    my_report.determine_report()
    reduced = {
        (int(r.start_ts), int(r.end_ts), r.entity_key)
        for r in my_report.event_dfs["ENTITY_EVENT"]["changed"].collect()
        if r.container_id == 1
    }

    # Raw-frame oracle path: same event over a MultiSeriesCache of the raw rows.
    raw_pdf = pd.DataFrame(
        [r for r in _TWO_ENTITY_ROWS if r[0] == 1],
        columns=_OBJECT_TRACKS_SCHEMA.fieldNames(),
    )
    oracle_event = EntityEvent(
        name="close_object",
        expr=(SeriesAccessor(_object_tracks_series()).distance_m < 8.0).entity_condition(),
    )
    cache = CombinedSeriesCache(EmptyTimeSeriesCache(), {"object_tracks": raw_pdf})
    oracle = {
        (int(row[1]), int(row[2]), row[3])
        for row in oracle_event.materialize_per_container(1, cache)
    }

    # Both entities (47, 88) close in container 1 → two rows, identical on both paths.
    assert reduced == oracle
    assert {json.loads(ek)["object_tracks"]["lidar"][0] for *_, ek in reduced} == {"47", "88"}


def test_colliding_entities_survive_unchanged_merge_persist(spark, basic_narrow_db, monkeypatch):
    """Two distinct entities that COLLIDE on event_instance_id must both survive
    the unchanged-definition MERGE persist — entity_key is part of the merge key.

    event_instance_id is a 32-bit crc32 that folds entity_key in, so two entities
    sharing a (container, event, window) can collide. We force that collision by
    patching the id generator to drop entity_key, then drive the real persist
    twice: the seeding (changed) write, then the unchanged-definition MERGE — the
    path a second pipeline run takes once the definition is already persisted.

    Before the fix the colliding ids made the MERGE conflate the two entities
    (with duplicate merge keys on both sides Delta raises rather than silently
    overwriting) — a per-entity detection silently lost. This is the
    regulatory-traceability guarantee for per-entity ADAS detections.
    """
    import pyspark.sql.functions as f

    from impulse_reporting.events import entity_event as entity_event_module

    def _colliding_event_instance_id(
        event_type=None,
        container_id_col="container_id",
        event_name_col="event_name",
        start_ts_col="start_ts",
        end_ts_col="end_ts",
        entity_key_col=None,  # deliberately ignored → forces a collision
    ):
        return f.crc32(
            f.concat_ws(
                "::",
                f.col(container_id_col),
                f.col(event_name_col),
                f.col(start_ts_col),
                f.col(end_ts_col),
            )
        )

    monkeypatch.setattr(
        entity_event_module,
        "generate_event_instance_id_column",
        _colliding_event_instance_id,
    )

    impulse_config = ImpulseConfig(
        source=Source(
            container_metrics_table="spark_catalog.silver.container_metrics",
            channel_metrics_table="spark_catalog.silver.channel_metrics",
            channels_uri="spark_catalog.silver.channels",
        ),
        unity_sink=UnitySink(
            catalog="spark_catalog",
            schema="gold",
            table_prefix="entity_collision_test",
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

    fact_uri = "spark_catalog.gold.entity_collision_test_event_instance_fact"
    spark.sql("CREATE SCHEMA IF NOT EXISTS spark_catalog.gold")
    spark.sql(f"DROP TABLE IF EXISTS {fact_uri}")

    my_report = Report(
        name="entity_collision_report",
        spark=spark,
        workspace_client=create_autospec(WorkspaceClient),
        config=dict(impulse_config),
    )
    db = my_report.get_db()
    db.register_series(
        _object_tracks_series(),
        lambda spark: spark.createDataFrame(_TWO_ENTITY_ROWS, _OBJECT_TRACKS_SCHEMA),
    )
    close_object = (db.query.series("object_tracks").distance_m < 8.0).entity_condition()
    my_report.add_event(EntityEvent(name="close_object", expr=close_object))

    my_report.determine_report()
    fact_df = my_report.event_dfs["ENTITY_EVENT"]["changed"]

    # Run 1 (new definition): seed the fact table via the changed-definition path.
    my_report._persist_incremental({}, my_report._changed_event_ids)

    # Run 2 (definition now unchanged): re-persist through the MERGE path — the
    # state a second pipeline run produces once the dimension already exists.
    my_report.event_dfs["ENTITY_EVENT"] = {"changed": None, "unchanged": fact_df}
    my_report._persist_incremental({}, {})

    rows = [r for r in spark.read.table(fact_uri).collect() if r.container_id == 1]

    # Both entities persisted as separate rows — neither clobbered by the other.
    assert len(rows) == 2
    assert {r.entity_key for r in rows} == {
        '{"object_tracks": {"lidar": ["47"]}}',
        '{"object_tracks": {"lidar": ["88"]}}',
    }
    # Confirm the collision actually fired: both rows share one event_instance_id,
    # so only entity_key in the merge key kept them distinct.
    assert len({r.event_instance_id for r in rows}) == 1

    spark.sql(f"DROP TABLE IF EXISTS {fact_uri}")
