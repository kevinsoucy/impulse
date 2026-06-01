"""Series definition registry on MeasurementDB + query.series(name) authoring.

Tabular-series *definitions* live on MeasurementDB so
that ``query.series(name)`` can build a typed accessor at authoring time (before
any solver is chosen) and the solver can resolve the ``source_factory`` at
execution time. Channels are deliberately NOT in this registry — they are
authored via ``query.channel`` / ``query.signal`` and their column roles live in
SolverConfig.
"""

from unittest.mock import create_autospec

import pyspark.sql.types as T
import pytest
from databricks.sdk import WorkspaceClient

from impulse_query_engine.measurement_db import MeasurementDB, MeasurementDBConfig
from impulse_query_engine.surfaces import SeriesAccessor
from impulse_query_engine.surfaces.series import Series


def _db() -> MeasurementDB:
    return MeasurementDB(MeasurementDBConfig(), ws=create_autospec(WorkspaceClient))


def _object_tracks(name: str = "object_tracks", *, with_schema: bool = True) -> Series:
    schema = (
        T.StructType(
            [
                T.StructField("container_id", T.LongType(), nullable=False),
                T.StructField("sensor_type", T.StringType(), nullable=False),
                T.StructField("frame_ts", T.LongType(), nullable=False),
                T.StructField("entity_id", T.LongType(), nullable=False),
                T.StructField("detection_class", T.StringType()),
                T.StructField("distance_m", T.DoubleType()),
            ]
        )
        if with_schema
        else None
    )
    return Series(
        name=name,
        schema=schema,
        session_col="container_id",
        signal_col="sensor_type",
        timestamp_col="frame_ts",
        entity_key="entity_id",
    )


def _identity_factory(spark):
    return spark


# --- registry mechanics -----------------------------------------------------


def test_register_and_snapshot_returns_definition_by_name():
    db = _db()
    series = _object_tracks()
    db.register_series(series, _identity_factory)
    snapshot = db.registered_series()
    assert snapshot == {"object_tracks": series}


def test_register_duplicate_name_raises():
    db = _db()
    db.register_series(_object_tracks(), _identity_factory)
    with pytest.raises(ValueError, match="already registered"):
        db.register_series(_object_tracks(), _identity_factory)


def test_registered_series_snapshot_is_isolated():
    # Mutating the returned snapshot must not corrupt the registry.
    db = _db()
    db.register_series(_object_tracks(), _identity_factory)
    snapshot = db.registered_series()
    snapshot.clear()
    assert "object_tracks" in db.registered_series()


def test_series_source_returns_factory():
    db = _db()
    sentinel = object()
    db.register_series(_object_tracks(), lambda spark: sentinel)  # noqa: ARG005
    factory = db.series_source("object_tracks")
    assert factory(spark=None) is sentinel


def test_series_source_unknown_raises_keyerror_naming_series():
    db = _db()
    with pytest.raises(KeyError, match="missing_series"):
        db.series_source("missing_series")


def test_two_series_resolve_independently():
    db = _db()
    a = _object_tracks("object_tracks")
    b = _object_tracks("lane_offsets")
    db.register_series(a, _identity_factory)
    db.register_series(b, _identity_factory)
    assert db.registered_series() == {"object_tracks": a, "lane_offsets": b}


# --- query.series(name) authoring -------------------------------------------


def test_query_series_returns_accessor_and_builds_leaf_keyed_by_name():
    db = _db()
    db.register_series(_object_tracks(), _identity_factory)
    acc = db.query.series("object_tracks")
    assert isinstance(acc, SeriesAccessor)
    # The authored leaf's leaf_kind is the series name (the cogroup routing key).
    leaf = (acc.distance_m < 8.0).entity_condition()
    assert leaf.leaf_kind == "object_tracks"


def test_query_series_unknown_raises_keyerror_naming_series():
    db = _db()
    with pytest.raises(KeyError, match="not_registered"):
        db.query.series("not_registered")


def test_query_series_on_schemaless_definition_raises():
    # A definition with no schema cannot build a typed accessor — fail fast at
    # authoring rather than producing a useless accessor.
    db = _db()
    db.register_series(_object_tracks(with_schema=False), _identity_factory)
    with pytest.raises(ValueError, match="no resolved schema"):
        db.query.series("object_tracks")


def test_query_series_does_not_surface_channels():
    # Channels are not in the definition registry; query.series must not resolve
    # them (they are authored via query.channel / query.signal).
    db = _db()
    with pytest.raises(KeyError, match="channels"):
        db.query.series("channels")


def test_each_query_access_sees_the_same_registry():
    # db.query builds a fresh QueryBuilder each access; both must read the same
    # db-held registry.
    db = _db()
    db.register_series(_object_tracks(), _identity_factory)
    assert db.query.series("object_tracks").series.name == "object_tracks"
    assert db.query.series("object_tracks").series.name == "object_tracks"


# --- query.signal(name) (name-addressed channel sugar) ----------------------


def test_query_signal_is_sugar_for_channel_name_tag():
    db = _db()
    sig = db.query.signal("Engine RPM")
    chan = db.query.channel(channel_name="Engine RPM")
    assert type(sig) is type(chan)
    assert sig.required_tags() == chan.required_tags() == {"channel_name"}


def test_query_signal_does_not_consult_series_registry():
    # Channels are not registered series; query.signal must work with an empty
    # registry (no KeyError).
    db = _db()
    assert db.query.signal("Vehicle Speed Sensor").required_tags() == {"channel_name"}


# --- opt-in signal-metadata validation --------------------------------------

_OBJ_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("frame_ts", T.LongType(), nullable=False),
        T.StructField("entity_id", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType()),
    ]
)


def _obj_factory(rows):
    return lambda spark: spark.createDataFrame(rows, _OBJ_SCHEMA)


def test_register_without_valid_signals_does_not_require_spark_or_metadata():
    # Default path: no signal metadata needed, no spark needed.
    db = _db()
    db.register_series(_object_tracks(), _obj_factory([(1, "lidar", 0, 47, 5.0)]))
    assert "object_tracks" in db.registered_series()


def test_valid_signals_accepts_known_signals(spark):
    db = _db()
    rows = [(1, "lidar", 0, 47, 5.0), (1, "radar", 1, 47, 4.0)]
    db.register_series(
        _object_tracks(),
        _obj_factory(rows),
        valid_signals={"lidar", "radar", "camera_front"},
        spark=spark,
    )
    assert "object_tracks" in db.registered_series()


def test_valid_signals_rejects_unknown_signal(spark):
    db = _db()
    rows = [(1, "lidar", 0, 47, 5.0), (1, "lidisar", 1, 47, 4.0)]  # typo'd signal
    with pytest.raises(ValueError, match="lidisar.*not in the provided signal metadata"):
        db.register_series(
            _object_tracks(),
            _obj_factory(rows),
            valid_signals={"lidar", "radar"},
            spark=spark,
        )
    # Failed validation must not register the series.
    assert "object_tracks" not in db.registered_series()


def test_valid_signals_flags_null_signal_as_unknown(spark):
    # A NULL signal value cannot match any allowed value, so it is flagged (it
    # would be undroppable garbage at query time). Locks the anti-join's NULL
    # handling to the prior set-difference behavior.
    nullable_schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("sensor_type", T.StringType(), nullable=True),
            T.StructField("frame_ts", T.LongType(), nullable=False),
            T.StructField("entity_id", T.LongType(), nullable=False),
            T.StructField("distance_m", T.DoubleType()),
        ]
    )
    db = _db()
    rows = [(1, "lidar", 0, 47, 5.0), (1, None, 1, 48, 4.0)]
    with pytest.raises(ValueError, match="not in the provided signal metadata"):
        db.register_series(
            _object_tracks(),
            lambda spark: spark.createDataFrame(rows, nullable_schema),
            valid_signals={"lidar"},
            spark=spark,
        )
    assert "object_tracks" not in db.registered_series()


def test_valid_signals_without_spark_raises():
    db = _db()
    with pytest.raises(ValueError, match="needs a spark session"):
        db.register_series(
            _object_tracks(),
            _obj_factory([(1, "lidar", 0, 47, 5.0)]),
            valid_signals={"lidar"},
        )
