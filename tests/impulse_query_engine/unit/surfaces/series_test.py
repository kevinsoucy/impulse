"""Construction-time validation of Series: time-axis shape, structural-role
mapping, entity-key normalization, and structural-column set."""

import pyspark.sql.types as T
import pytest

from impulse_query_engine.surfaces import Series


def _schema(extra_fields=()):
    fields = [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("ts", T.LongType(), nullable=False),
        T.StructField("object_id", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType(), nullable=True),
    ]
    fields.extend(extra_fields)
    return T.StructType(fields)


def _series(**overrides):
    kwargs = dict(
        name="object_tracks",
        schema=_schema(),
        session_col="container_id",
        signal_col="sensor_type",
        timestamp_col="ts",
        entity_key="object_id",
    )
    kwargs.update(overrides)
    return Series(**kwargs)


# --- happy paths ------------------------------------------------------------


def test_point_in_time_round_trip():
    s = _series()
    assert s.name == "object_tracks"
    assert s.is_point_in_time and not s.is_rle
    assert s.timestamp_col == "ts"
    assert s.entity_key_cols == ("object_id",)
    assert s.time_cols == ("ts",)


def test_rle_round_trip():
    schema = _schema(
        extra_fields=[
            T.StructField("seg_start", T.LongType()),
            T.StructField("seg_end", T.LongType()),
        ]
    )
    s = Series(
        name="lanes",
        schema=schema,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="seg_start",
        tend_col="seg_end",
        entity_key="object_id",
    )
    assert s.is_rle and not s.is_point_in_time
    assert s.time_cols == ("seg_start", "seg_end")


def test_defaults_session_and_signal_col():
    schema = T.StructType(
        [
            T.StructField("session_id", T.LongType()),
            T.StructField("signal_id", T.StringType()),
            T.StructField("sample_ts", T.LongType()),
            T.StructField("accel_x", T.DoubleType()),
        ]
    )
    s = Series(name="imu", schema=schema, timestamp_col="sample_ts")
    assert s.session_col == "session_id"
    assert s.signal_col == "signal_id"
    assert s.entity_key_cols == ()


def test_no_entity_key_is_supported():
    s = _series(entity_key=None)
    assert s.entity_key_cols == ()


def test_tuple_entity_key_normalizes_to_tuple():
    schema = _schema(extra_fields=[T.StructField("line_id", T.StringType(), True)])
    s = _series(schema=schema, entity_key=("object_id", "line_id"))
    assert s.entity_key_cols == ("object_id", "line_id")


def test_structural_cols_excludes_payload():
    s = _series()
    assert s.structural_cols == frozenset({"container_id", "sensor_type", "ts", "object_id"})
    assert "distance_m" not in s.structural_cols


# --- shape validation -------------------------------------------------------


def test_both_shapes_rejected():
    with pytest.raises(ValueError, match="exactly one time-axis shape"):
        _series(tstart_col="ts", tend_col="ts")


def test_no_shape_rejected():
    with pytest.raises(ValueError, match="declares no time axis"):
        _series(timestamp_col=None)


def test_rle_requires_both_endpoints():
    with pytest.raises(ValueError, match="requires both tstart_col and tend_col"):
        _series(timestamp_col=None, tstart_col="ts")


# --- structural-role validation against schema ------------------------------


def test_unknown_timestamp_col_raises():
    with pytest.raises(ValueError, match="time column 'missing' not in schema"):
        _series(timestamp_col="missing")


def test_unknown_session_col_raises():
    with pytest.raises(ValueError, match="session_col 'nope' not in schema"):
        _series(session_col="nope")


def test_unknown_signal_col_raises():
    with pytest.raises(ValueError, match="signal_col 'nope' not in schema"):
        _series(signal_col="nope")


def test_unknown_entity_key_raises():
    with pytest.raises(ValueError, match="entity_key references unknown columns"):
        _series(entity_key="missing")


# --- optional schema --------------------------------------------------------


def test_schema_optional_skips_validation_until_resolved():
    # No schema → no structural validation at construction.
    s = Series(
        name="channels",
        session_col="container_id",
        signal_col="channel_id",
        tstart_col="interval_start",
        tend_col="interval_end",
    )
    assert s.schema is None
    resolved = s.with_schema(
        T.StructType(
            [
                T.StructField("container_id", T.LongType()),
                T.StructField("channel_id", T.LongType()),
                T.StructField("interval_start", T.LongType()),
                T.StructField("interval_end", T.LongType()),
                T.StructField("value", T.DoubleType()),
            ]
        )
    )
    assert resolved.schema is not None
    assert resolved.structural_cols == frozenset(
        {"container_id", "channel_id", "interval_start", "interval_end"}
    )


def test_with_schema_validates_structure():
    s = Series(
        name="channels",
        session_col="container_id",
        signal_col="channel_id",
        timestamp_col="ts",
    )
    with pytest.raises(ValueError, match="session_col 'container_id' not in schema"):
        s.with_schema(T.StructType([T.StructField("ts", T.LongType())]))
