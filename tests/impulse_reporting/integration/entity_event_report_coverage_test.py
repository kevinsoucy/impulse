"""Integration tests filling EntityEvent report-E2E coverage gaps.

Each test drives a full ``Report.determine_report()`` so it exercises the
production dispatch path — the series cogroup and the per-entity Spark
reduction — not just the query path. ``entity_event_report_test.py`` covers the
single RLE-series case; this file adds:

- a channel leaf ANDed with a series leaf through the report path;
- ``.any().ids()`` merged windowing (one combined union-map row);
- two EntityEvents dispatched in a single report;
- a point-in-time series through the report path (the Spark ``tend`` precompute +
  ``container_stop_ts`` last-frame close, not just the RLE path);
- a compound ``entity_key`` driven through the reduction.

Data axes (must stay unit-aligned with the container metrics, per the Series
time-axis precondition): the RLE-only tests use a synthetic ``[0, 10)`` axis (no
container_stop_ts needed); the channel-mix test places object windows inside each
container's Engine-RPM coverage (epoch micros); the point-in-time test uses the
container metrics' own ``start_ts``/``stop_ts`` axis (epoch millis).
"""

from unittest.mock import create_autospec

import pyspark.sql.types as T
from databricks.sdk import WorkspaceClient

from impulse_reporting.config.config_parser import (
    Comparator,
    ContainerFilters,
    ImpulseConfig,
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

# --- RLE object-tracks schema (synthetic [0, 10) axis) ------------------------
_RLE_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("tstart", T.LongType(), nullable=False),
        T.StructField("tend", T.LongType(), nullable=False),
        T.StructField("entity_id", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType()),
    ]
)


def _rle_series() -> Series:
    return Series(
        name="object_tracks",
        schema=_RLE_SCHEMA,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="tstart",
        tend_col="tend",
        entity_key="entity_id",
    )


def _report(spark, table_prefix: str) -> Report:
    """Build a Report over the basic silver tables, filtered to the three
    Seat_Leon containers (1, 2, 3) — the same config as entity_event_report_test."""
    impulse_config = ImpulseConfig(
        source=Source(
            container_metrics_table="spark_catalog.silver.container_metrics",
            channel_metrics_table="spark_catalog.silver.channel_metrics",
            channels_uri="spark_catalog.silver.channels",
        ),
        unity_sink=UnitySink(
            catalog="spark_catalog",
            schema="gold",
            table_prefix=table_prefix,
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
            "container_id",
            "start_ts",
            "stop_ts",
        ],
    )
    return Report(
        name=table_prefix,
        spark=spark,
        workspace_client=create_autospec(WorkspaceClient),
        config=dict(impulse_config),
    )


def _entity_rows(report: Report) -> list:
    event_dfs = report.event_dfs
    assert "ENTITY_EVENT" in event_dfs, "EntityEvent must dispatch under its own type"
    return event_dfs["ENTITY_EVENT"]["changed"].collect()


# -----------------------------------------------------------------------------
# (B.1a) channel leaf AND series leaf through the report path
# -----------------------------------------------------------------------------
# Object windows sit inside each container's Engine-RPM coverage (epoch micros)
# so the channel intervals and the series intervals share one axis. The channel
# predicate ``Engine RPM > -1`` holds over the whole coverage, so the conjunction
# reduces to the object window — which containers fire is decided by the series
# (close in 1 and 3, far in 2), proving the channel leaf rides the report path
# without disturbing entity attribution.
_MIX_ROWS = [
    (1, "lidar", 1499929250000000, 1499929260000000, 47, 5.0),  # inside c1 RPM span, close
    (2, "lidar", 1499367280000000, 1499367290000000, 91, 99.0),  # inside c2 RPM span, far
    (3, "lidar", 1499239000000000, 1499239010000000, 12, 4.0),  # inside c3 RPM span, close
]


def test_entity_event_channel_and_series_mix(spark, basic_narrow_db):
    report = _report(spark, "ee_cov_mix")
    db = report.get_db()
    db.register_series(
        _rle_series(),
        lambda spark: spark.createDataFrame(_MIX_ROWS, _RLE_SCHEMA),
    )
    eng_rpm = db.query.channel(channel_name="Engine RPM")
    close = (db.query.series("object_tracks").distance_m < 8.0).each().ids(as_="object")
    expr = close & (eng_rpm > -1)
    report.add_event(EntityEvent(name="close_with_rpm", expr=expr))

    report.determine_report()

    by_container = {r.container_id: r for r in _entity_rows(report)}
    assert set(by_container) == {1, 3}
    assert by_container[1].entity_key == '{"object": {"lidar": ["47"]}}'
    assert by_container[3].entity_key == '{"object": {"lidar": ["12"]}}'
    for r in by_container.values():
        assert r.start_ts < r.end_ts


# -----------------------------------------------------------------------------
# (B.1b) .any().ids() merged windowing — one combined union-map row per window
# -----------------------------------------------------------------------------
# Container 1 has two close entities; combined windowing must emit a SINGLE row
# whose entity_key unions both ids (per-entity windowing would emit two rows).
_COMBINED_ROWS = [
    (1, "lidar", 0, 10, 47, 5.0),
    (1, "lidar", 0, 10, 48, 6.0),
    (3, "lidar", 0, 10, 12, 4.0),
    (2, "lidar", 0, 10, 91, 99.0),  # far — no row
]


def test_entity_event_combined_windowing_unions_entities(spark, basic_narrow_db):
    report = _report(spark, "ee_cov_combined")
    db = report.get_db()
    db.register_series(
        _rle_series(),
        lambda spark: spark.createDataFrame(_COMBINED_ROWS, _RLE_SCHEMA),
    )
    close = (db.query.series("object_tracks").distance_m < 8.0).any().ids(as_="object")
    report.add_event(EntityEvent(name="close_combined", expr=close))

    report.determine_report()

    rows = _entity_rows(report)
    by_container: dict[int, list] = {}
    for r in rows:
        by_container.setdefault(r.container_id, []).append(r)
    assert set(by_container) == {1, 3}
    # Combined windowing: exactly one row for container 1, both entities unioned.
    assert len(by_container[1]) == 1
    assert by_container[1][0].entity_key == '{"object": {"lidar": ["47", "48"]}}'
    assert len(by_container[3]) == 1
    assert by_container[3][0].entity_key == '{"object": {"lidar": ["12"]}}'


# -----------------------------------------------------------------------------
# (B.1c) two EntityEvents dispatched in one report
# -----------------------------------------------------------------------------
_TWO_EVENT_ROWS = [
    (1, "lidar", 0, 10, 47, 5.0),  # close
    (2, "lidar", 0, 10, 91, 99.0),  # far
    (3, "lidar", 0, 10, 12, 4.0),  # close
]


def test_two_entity_events_one_report(spark, basic_narrow_db):
    report = _report(spark, "ee_cov_multi")
    db = report.get_db()
    db.register_series(
        _rle_series(),
        lambda spark: spark.createDataFrame(_TWO_EVENT_ROWS, _RLE_SCHEMA),
    )
    close = (db.query.series("object_tracks").distance_m < 8.0).each().ids(as_="object")
    far = (db.query.series("object_tracks").distance_m > 50.0).each().ids(as_="object")
    report.add_event(EntityEvent(name="close_object", expr=close))
    report.add_event(EntityEvent(name="far_object", expr=far))

    report.determine_report()

    # Both definitions share the ENTITY_EVENT dimension table.
    dim_rows = report.event_metadata_dfs["ENTITY_EVENT"].collect()
    names_to_id = {r.event_name: r.event_id for r in dim_rows}
    assert set(names_to_id) == {"close_object", "far_object"}

    rows = _entity_rows(report)
    fired_event_ids = {r.event_id for r in rows}
    # close fires in containers 1 & 3; far fires in container 2 — both present.
    assert names_to_id["close_object"] in fired_event_ids
    assert names_to_id["far_object"] in fired_event_ids
    far_containers = {r.container_id for r in rows if r.event_id == names_to_id["far_object"]}
    assert far_containers == {2}


# -----------------------------------------------------------------------------
# (B.1d) point-in-time series through the report path
# -----------------------------------------------------------------------------
# Point-in-time rows on the container metrics' own millis axis. With one frame
# per (container, signal), the synthesized interval is [ts, container_stop_ts) —
# so this exercises the Spark tend precompute joining container_stop_ts, end to
# end through the report path (not just RLE rows that carry their own intervals).
_PIT_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("ts", T.LongType(), nullable=False),
        T.StructField("entity_id", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType()),
    ]
)
_C1_STOP_TS = 1751528610253  # container_metrics.stop_ts for container 1
_PIT_ROWS = [
    (1, "lidar", 1751528503000, 47, 5.0),  # close; closes at c1 stop_ts
    (2, "lidar", 1751528502000, 91, 99.0),  # far
    (3, "lidar", 1751528501000, 12, 4.0),  # close
]


def _pit_series() -> Series:
    return Series(
        name="object_tracks",
        schema=_PIT_SCHEMA,
        session_col="container_id",
        signal_col="sensor_type",
        timestamp_col="ts",
        entity_key="entity_id",
    )


def test_point_in_time_series_through_report(spark, basic_narrow_db):
    report = _report(spark, "ee_cov_pit")
    db = report.get_db()
    db.register_series(
        _pit_series(),
        lambda spark: spark.createDataFrame(_PIT_ROWS, _PIT_SCHEMA),
    )
    close = (db.query.series("object_tracks").distance_m < 8.0).each().ids(as_="object")
    report.add_event(EntityEvent(name="close_pit", expr=close))

    report.determine_report()

    by_container = {r.container_id: r for r in _entity_rows(report)}
    assert set(by_container) == {1, 3}
    assert by_container[1].entity_key == '{"object": {"lidar": ["47"]}}'
    # The single frame's interval is closed at the container stop_ts (point-in-time
    # last-frame synthesis via the Spark tend precompute).
    assert by_container[1].start_ts == 1751528503000
    assert by_container[1].end_ts == _C1_STOP_TS


# -----------------------------------------------------------------------------
# (B.2) compound entity_key through the reduction
# -----------------------------------------------------------------------------
_COMPOUND_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("tstart", T.LongType(), nullable=False),
        T.StructField("tend", T.LongType(), nullable=False),
        T.StructField("entity_id", T.LongType(), nullable=False),
        T.StructField("track_idx", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType()),
    ]
)
_COMPOUND_ROWS = [
    (1, "lidar", 0, 10, 47, 0, 5.0),  # close → compound key (47, 0)
    (2, "lidar", 0, 10, 91, 0, 99.0),  # far
    (3, "lidar", 0, 10, 12, 1, 4.0),  # close → compound key (12, 1)
]


def _compound_series() -> Series:
    return Series(
        name="object_tracks",
        schema=_COMPOUND_SCHEMA,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="tstart",
        tend_col="tend",
        entity_key=("entity_id", "track_idx"),
    )


def test_compound_entity_key_through_reduction(spark, basic_narrow_db):
    report = _report(spark, "ee_cov_compound")
    db = report.get_db()
    db.register_series(
        _compound_series(),
        lambda spark: spark.createDataFrame(_COMPOUND_ROWS, _COMPOUND_SCHEMA),
    )
    close = (db.query.series("object_tracks").distance_m < 8.0).each().ids(as_="object")
    report.add_event(EntityEvent(name="close_compound", expr=close))

    report.determine_report()

    by_container = {r.container_id: r for r in _entity_rows(report)}
    assert set(by_container) == {1, 3}
    # Compound keys render as a JSON array string inside the nested entity map.
    assert by_container[1].entity_key == '{"object": {"lidar": ["[47, 0]"]}}'
    assert by_container[3].entity_key == '{"object": {"lidar": ["[12, 1]"]}}'
