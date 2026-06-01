# pylint: disable=missing-function-docstring
"""End-to-end cogroup path: queries that reference registered tabular series.

Exercises ``QueryBuilder.solve`` → ``QuerySolver.solve_with_series`` (the n-way
struct-per-series union + binary cogroup) against the key_value_store_db fixture
(containers {1, 2, 3}, channel "Engine RPM"). Series are RLE so no
``container_stop_ts`` plumbing is needed.
"""

import pyspark.sql.types as T
import pytest
from pyspark.sql import SparkSession

from impulse_query_engine.analyze.query.solvers.key_value_store_solver import (
    KeyValueStoreSolver,
)
from impulse_query_engine.analyze.query.solvers.solver_config import SolverConfig, TableConfig
from impulse_query_engine.measurement_db import MeasurementDB
from impulse_query_engine.surfaces.series import Series

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

# container 1: a close object; container 2: only a far object; container 3: close.
_OBJECT_TRACKS_ROWS = [
    (1, "lidar", 0, 10, 47, 5.0),
    (2, "lidar", 0, 10, 47, 99.0),
    (3, "lidar", 0, 10, 47, 4.0),
]


def _kvs_cfg() -> SolverConfig:
    return SolverConfig(
        project_id="SAMPLE_PROJECT",
        container_tags=TableConfig(column_name_mapping={"element_id": "key"}),
        container_metrics=TableConfig(column_name_mapping={"project": "project_id"}),
    )


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


def _register_object_tracks(db: MeasurementDB, rows=None):
    db.register_series(
        _object_tracks_series(),
        lambda spark: spark.createDataFrame(rows or _OBJECT_TRACKS_ROWS, _OBJECT_TRACKS_SCHEMA),
    )


# A second, heterogeneously-shaped RLE series (different payload column).
_TRAFFIC_SIGNS_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("tstart", T.LongType(), nullable=False),
        T.StructField("tend", T.LongType(), nullable=False),
        T.StructField("sign_id", T.LongType(), nullable=False),
        T.StructField("sign_class", T.StringType()),
    ]
)


def _traffic_signs_series() -> Series:
    return Series(
        name="traffic_signs",
        schema=_TRAFFIC_SIGNS_SCHEMA,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="tstart",
        tend_col="tend",
        entity_key="sign_id",
    )


def _register_traffic_signs(db: MeasurementDB, rows):
    db.register_series(
        _traffic_signs_series(),
        lambda spark: spark.createDataFrame(rows, _TRAFFIC_SIGNS_SCHEMA),
    )


# A point-in-time series (timestamp_col, no tstart/tend): the last frame closes
# at the container's stop_ts, which exercises container_stop_ts plumbing.
_PIT_TRACKS_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("frame_ts", T.LongType(), nullable=False),
        T.StructField("entity_id", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType()),
    ]
)


def _pit_tracks_series() -> Series:
    return Series(
        name="pit_tracks",
        schema=_PIT_TRACKS_SCHEMA,
        session_col="container_id",
        signal_col="sensor_type",
        timestamp_col="frame_ts",
        entity_key="entity_id",
    )


def _register_pit_tracks(db: MeasurementDB, rows):
    db.register_series(
        _pit_tracks_series(),
        lambda spark: spark.createDataFrame(rows, _PIT_TRACKS_SCHEMA),
    )


class TestCrossSeriesCogroup:
    def test_series_only_query_returns_per_container_intervals(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        _register_object_tracks(key_value_store_db)
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = key_value_store_db.query
        near = (query.series("object_tracks").distance_m < 8.0).entity_condition().alias("near")

        result = query.select(near).solve(spark=spark, solver=solver)
        by_container = {r.container_id: r["near"] for r in result.collect()}

        assert set(by_container) == {1, 2, 3}
        # Close objects in 1 and 3 → one interval [0, 10); none in 2.
        assert by_container[1] == [[0.0, 10.0]]
        assert by_container[3] == [[0.0, 10.0]]
        assert by_container[2] == []

    def test_bare_presence_partial_in_select_returns_intervals(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # A partial finalized as a *presence* check (no .entity_condition())
        # placed directly in select() must still resolve through the reduced
        # cogroup. Before the memoization fix, get_selectors()/build() finalized
        # different SeriesSelector instances, so the _reduce_key stamped on the
        # collected leaf was invisible at build time — the reduced lookup missed,
        # the cache fell back to an empty frame, and the predicate silently
        # returned [] for every container.
        _register_object_tracks(key_value_store_db)
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = key_value_store_db.query
        near = (query.series("object_tracks").distance_m < 8.0).alias("near")

        result = query.select(near).solve(spark=spark, solver=solver)
        by_container = {r.container_id: r["near"] for r in result.collect()}

        assert set(by_container) == {1, 2, 3}
        # Presence: a close object holds [0, 10) in 1 and 3; container 2 (99m) is empty.
        assert by_container[1] == [[0.0, 10.0]]
        assert by_container[3] == [[0.0, 10.0]]
        assert by_container[2] == []

    def test_channels_and_series_cogroup_returns_both_columns(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        _register_object_tracks(key_value_store_db)
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = key_value_store_db.query
        eng_rpm = query.channel(channel_name="Engine RPM")
        near = (query.series("object_tracks").distance_m < 8.0).entity_condition().alias("near")

        result = query.select(eng_rpm.mean().alias("rpm_mean"), near).solve(
            spark=spark, solver=solver
        )
        rows = {r.container_id: r for r in result.collect()}

        assert set(rows) == {1, 2, 3}
        # Channel aggregation present alongside the series-derived intervals.
        assert rows[1]["rpm_mean"] is not None
        assert rows[1]["near"] == [[0.0, 10.0]]
        assert rows[2]["near"] == []

    def test_container_filter_prunes_the_series_side(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        from impulse_query_engine.analyze.metadata.metric_expression import MetricSelector

        _register_object_tracks(key_value_store_db)
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = key_value_store_db.query
        # A non-matching container filter must drop every container — including
        # the series side (filters prune series, not just channels).
        query.where(MetricSelector("brand") == "NoSuchBrand")
        near = (query.series("object_tracks").distance_m < 8.0).entity_condition().alias("near")

        result = query.select(near).solve(spark=spark, solver=solver)
        assert result.count() == 0

    def test_unregistered_series_raises(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = key_value_store_db.query
        # query.series resolves at authoring; an unregistered name fails there.
        with pytest.raises(KeyError, match="object_tracks"):
            query.series("object_tracks")


class TestCrossSeriesMultiSeriesCogroup:
    """n-way: two heterogeneously-shaped registered series in one cogroup."""

    def test_two_series_conjunction_intersects_intervals(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        db = key_value_store_db
        # container 1: object close [0,10) AND sign present [5,15) → intersect [5,10).
        # container 2: object close [0,10) only, no sign → conjunction empty.
        _register_object_tracks(db, rows=[(1, "lidar", 0, 10, 47, 5.0), (2, "lidar", 0, 10, 47, 5.0)])
        _register_traffic_signs(db, rows=[(1, "camera", 5, 15, 12, "speed_30")])
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        obj = (query.series("object_tracks").distance_m < 8.0).entity_condition()
        sign = (query.series("traffic_signs").sign_class == "speed_30").entity_condition()

        result = query.select((obj & sign).alias("squeeze")).solve(spark=spark, solver=solver)
        by_container = {r.container_id: r["squeeze"] for r in result.collect()}

        assert by_container[1] == [[5.0, 10.0]]
        assert by_container[2] == []

    def test_two_series_disjunction_unions_intervals(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        db = key_value_store_db
        # container 1: object [0,10) OR sign [12,15) → union of both.
        _register_object_tracks(db, rows=[(1, "lidar", 0, 10, 47, 5.0)])
        _register_traffic_signs(db, rows=[(1, "camera", 12, 15, 12, "speed_30")])
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        obj = (query.series("object_tracks").distance_m < 8.0).entity_condition()
        sign = (query.series("traffic_signs").sign_class == "speed_30").entity_condition()

        result = query.select((obj | sign).alias("either")).solve(spark=spark, solver=solver)
        intervals = {r.container_id: r["either"] for r in result.collect()}[1]
        assert intervals == [[0.0, 10.0], [12.0, 15.0]]

    def test_heterogeneous_schemas_do_not_bleed_columns(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # object_tracks has distance_m; traffic_signs has sign_class. Each leaf
        # must resolve only its own series' columns — a struct-per-series union
        # keeps them separate. If columns bled, one of these predicates would
        # error or silently mis-evaluate.
        db = key_value_store_db
        _register_object_tracks(db, rows=[(1, "lidar", 0, 10, 47, 5.0)])
        _register_traffic_signs(db, rows=[(1, "camera", 0, 10, 12, "speed_30")])
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        obj = (query.series("object_tracks").distance_m < 8.0).entity_condition().alias("obj")
        sign = (query.series("traffic_signs").sign_class == "speed_30").entity_condition().alias(
            "sign"
        )
        result = query.select(obj, sign).solve(spark=spark, solver=solver)
        row = {r.container_id: r for r in result.collect()}[1]
        assert row["obj"] == [[0.0, 10.0]]
        assert row["sign"] == [[0.0, 10.0]]

    def test_container_in_one_series_only(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # container 2 has an object but no sign → conjunction empty there, while
        # the union still fires from the object alone.
        db = key_value_store_db
        _register_object_tracks(db, rows=[(2, "lidar", 0, 10, 47, 5.0)])
        _register_traffic_signs(db, rows=[(1, "camera", 0, 10, 12, "speed_30")])
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        obj = (query.series("object_tracks").distance_m < 8.0).entity_condition()
        sign = (query.series("traffic_signs").sign_class == "speed_30").entity_condition()
        result = query.select((obj & sign).alias("both")).solve(spark=spark, solver=solver)
        by_container = {r.container_id: r["both"] for r in result.collect()}
        # No container has both → all empty (containers 1 and 2 each have only one).
        assert all(v == [] for v in by_container.values())

    def test_four_way_channels_plus_two_series(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        db = key_value_store_db
        _register_object_tracks(db, rows=[(1, "lidar", 0, 10, 47, 5.0)])
        _register_traffic_signs(db, rows=[(1, "camera", 0, 10, 12, "speed_30")])
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        rpm = query.channel(channel_name="Engine RPM").mean().alias("rpm_mean")
        obj = (query.series("object_tracks").distance_m < 8.0).entity_condition().alias("obj")
        sign = (query.series("traffic_signs").sign_class == "speed_30").entity_condition().alias(
            "sign"
        )
        result = query.select(rpm, obj, sign).solve(spark=spark, solver=solver)
        row = {r.container_id: r for r in result.collect()}[1]
        assert row["rpm_mean"] is not None
        assert row["obj"] == [[0.0, 10.0]]
        assert row["sign"] == [[0.0, 10.0]]


class TestCrossSeriesEdgeCases:
    def test_two_predicates_on_same_series_dedupe_to_one_cogroup(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # Selecting two predicates on the SAME registered series must register it
        # once (dedup) and resolve both columns.
        db = key_value_store_db
        _register_object_tracks(db, rows=[(1, "lidar", 0, 10, 47, 5.0)])
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        near = (query.series("object_tracks").distance_m < 8.0).entity_condition().alias("near")
        far = (query.series("object_tracks").distance_m > 8.0).entity_condition().alias("far")
        result = query.select(near, far).solve(spark=spark, solver=solver)
        row = {r.container_id: r for r in result.collect()}[1]
        assert row["near"] == [[0.0, 10.0]]
        assert row["far"] == []  # the one object is close, never far

    def test_multi_interval_rle_per_entity_preserved(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # An entity present in two disjoint RLE intervals → both intervals kept.
        db = key_value_store_db
        _register_object_tracks(
            db, rows=[(1, "lidar", 0, 5, 47, 5.0), (1, "lidar", 10, 15, 47, 5.0)]
        )
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        near = (query.series("object_tracks").distance_m < 8.0).entity_condition().alias("near")
        result = query.select(near).solve(spark=spark, solver=solver)
        assert {r.container_id: r["near"] for r in result.collect()}[1] == [
            [0.0, 5.0],
            [10.0, 15.0],
        ]

    def test_session_col_not_named_container_id_is_renamed(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # A series whose session column is "vehicle_id" must be renamed to the
        # internal container_id for the cogroup key.
        db = key_value_store_db
        schema = T.StructType(
            [
                T.StructField("vehicle_id", T.LongType(), nullable=False),
                T.StructField("sensor_type", T.StringType(), nullable=False),
                T.StructField("tstart", T.LongType(), nullable=False),
                T.StructField("tend", T.LongType(), nullable=False),
                T.StructField("entity_id", T.LongType(), nullable=False),
                T.StructField("distance_m", T.DoubleType()),
            ]
        )
        series = Series(
            name="renamed_tracks",
            schema=schema,
            session_col="vehicle_id",
            signal_col="sensor_type",
            tstart_col="tstart",
            tend_col="tend",
            entity_key="entity_id",
        )
        db.register_series(
            series,
            lambda spark: spark.createDataFrame([(1, "lidar", 0, 10, 47, 5.0)], schema),
        )
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        near = (query.series("renamed_tracks").distance_m < 8.0).entity_condition().alias("near")
        result = query.select(near).solve(spark=spark, solver=solver)
        assert {r.container_id: r["near"] for r in result.collect()}[1] == [[0.0, 10.0]]

    def test_result_column_is_nested_interval_array(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # The series selection serializes to a list of [tstart, tend] float pairs.
        db = key_value_store_db
        _register_object_tracks(db, rows=[(1, "lidar", 0, 10, 47, 5.0)])
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        near = (query.series("object_tracks").distance_m < 8.0).entity_condition().alias("near")
        result = query.select(near).solve(spark=spark, solver=solver)
        val = {r.container_id: r["near"] for r in result.collect()}[1]
        assert isinstance(val, list) and isinstance(val[0], list) and len(val[0]) == 2
        assert all(isinstance(x, float) for x in val[0])

    def test_no_series_leaves_uses_plain_solve_path(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # Regression: a channel-only query (even with a series registered) must
        # NOT route through the cogroup path.
        db = key_value_store_db
        _register_object_tracks(db, rows=[(1, "lidar", 0, 10, 47, 5.0)])
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        result = query.select(
            query.channel(channel_name="Engine RPM").mean().alias("rpm_mean")
        ).solve(spark=spark, solver=solver)
        assert {r.container_id for r in result.collect()} == {1, 2, 3}

    def test_point_in_time_last_frame_closes_at_container_stop_ts(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # Two frames for one entity; the first closes at the next tick, the last
        # frame closes at the container's stop_ts (plumbed from container_metrics).
        # Without the plumbing the last frame would collapse to zero length and
        # the result would be only [f0, f1).
        db = key_value_store_db
        stop_ts = {
            r["container_id"]: r["stop_ts"]
            for r in db.container_metrics(spark).select("container_id", "stop_ts").collect()
        }[1]
        f0, f1 = stop_ts - 200, stop_ts - 100
        _register_pit_tracks(db, rows=[(1, "lidar", f0, 47, 5.0), (1, "lidar", f1, 47, 4.0)])
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        near = (query.series("pit_tracks").distance_m < 8.0).entity_condition().alias("near")

        result = query.select(near).solve(spark=spark, solver=solver)
        by_container = {r.container_id: r["near"] for r in result.collect()}

        # [f0, f1) ∪ [f1, stop_ts) merges to one interval [f0, stop_ts).
        assert by_container[1] == [[float(f0), float(stop_ts)]]

    def test_bare_presence_partial_point_in_time_closes_at_stop_ts(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # #3/#4: a bare presence partial (no .entity_condition()) on a
        # point-in-time series must resolve through the reduced cogroup AND close
        # the last frame at container_stop_ts — exercising the memoization fix on
        # the point-in-time interval-synthesis path (not just RLE).
        db = key_value_store_db
        stop_ts = {
            r["container_id"]: r["stop_ts"]
            for r in db.container_metrics(spark).select("container_id", "stop_ts").collect()
        }[1]
        f0, f1 = stop_ts - 200, stop_ts - 100
        _register_pit_tracks(db, rows=[(1, "lidar", f0, 47, 5.0), (1, "lidar", f1, 47, 4.0)])
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = db.query
        near = (query.series("pit_tracks").distance_m < 8.0).alias("near")

        result = query.select(near).solve(spark=spark, solver=solver)
        by_container = {r.container_id: r["near"] for r in result.collect()}

        assert by_container[1] == [[float(f0), float(stop_ts)]]

    def test_bare_presence_partial_with_channel_leaf_cogroups(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # #3/#4: a bare presence partial alongside a channel leaf drives the
        # has_channel_leaves cogroup branch (the channel cache path), a different
        # route than the series-only branch. The presence predicate must still
        # resolve through the reduced cache.
        _register_object_tracks(key_value_store_db)
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        query = key_value_store_db.query
        rpm = query.channel(channel_name="Engine RPM").mean().alias("rpm_mean")
        near = (query.series("object_tracks").distance_m < 8.0).alias("near")

        result = query.select(rpm, near).solve(spark=spark, solver=solver)
        rows = {r.container_id: r for r in result.collect()}

        assert rows[1]["rpm_mean"] is not None
        assert rows[1]["near"] == [[0.0, 10.0]]  # close object present
        assert rows[2]["near"] == []  # only a far object (99m)

    def test_blob_solver_rejects_series_plus_channel_query(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # BlobSolver has no cogroup machinery, so any registered-series query
        # must fail fast at the QueryBuilder boundary with a clear, actionable
        # message naming the solvers that support it — never a cryptic error deep
        # in the reduction or a silently-empty result.
        from impulse_query_engine.analyze.query.solvers.blob_solver import BlobSolver

        _register_object_tracks(key_value_store_db)
        query = key_value_store_db.query
        rpm = query.channel(channel_name="Engine RPM").mean().alias("rpm")
        near = (query.series("object_tracks").distance_m < 8.0).entity_condition().alias("near")

        with pytest.raises(NotImplementedError, match="DeltaSolver or KeyValueStoreSolver"):
            query.select(rpm, near).solve(spark=spark, solver=BlobSolver())

    def test_blob_solver_rejects_series_only_query(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # Even a series-ONLY query is rejected on BlobSolver: it has no
        # SolverConfig/reduction machinery, so it cannot resolve a registered
        # series at all. The rejection is the same clear, early error.
        from impulse_query_engine.analyze.query.solvers.blob_solver import BlobSolver

        _register_object_tracks(key_value_store_db)
        query = key_value_store_db.query
        near = (query.series("object_tracks").distance_m < 8.0).entity_condition().alias("near")

        with pytest.raises(NotImplementedError, match="DeltaSolver or KeyValueStoreSolver"):
            query.select(near).solve(spark=spark, solver=BlobSolver())

    def test_signal_name_sugar_matches_channel_by_name(
        self, spark: SparkSession, key_value_store_db: MeasurementDB
    ):
        # query.signal(name) is name-addressed sugar for query.channel(channel_name=name).
        solver = KeyValueStoreSolver(spark, config=_kvs_cfg())
        q1 = key_value_store_db.query
        r1 = q1.select(q1.signal("Engine RPM").mean().alias("m")).solve(spark=spark, solver=solver)
        q2 = key_value_store_db.query
        r2 = q2.select(
            q2.channel(channel_name="Engine RPM").mean().alias("m")
        ).solve(spark=spark, solver=solver)
        assert {r.container_id: r["m"] for r in r1.collect()} == {
            r.container_id: r["m"] for r in r2.collect()
        }
