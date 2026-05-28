"""Construction-time validation of RowGroupedSurface."""

import pyspark.sql.types as T
import pytest

from impulse_query_engine.surfaces import RowGroupedSurface


def _schema(extra_fields=()):
    fields = [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("ts", T.LongType(), nullable=False),
        T.StructField("object_id", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType(), nullable=True),
    ]
    fields.extend(extra_fields)
    return T.StructType(fields)


def test_basic_surface_round_trip():
    s = RowGroupedSurface(
        name="object_tracks", schema=_schema(), timestamp_col="ts", group_col="object_id"
    )
    assert s.name == "object_tracks"
    assert s.timestamp_col == "ts"
    assert s.group_cols == ("object_id",)


def test_no_group_col_is_supported():
    s = RowGroupedSurface(name="snaps", schema=_schema(), timestamp_col="ts", group_col=None)
    assert s.group_cols == ()


def test_tuple_group_col_normalizes_to_tuple():
    schema = _schema(extra_fields=[T.StructField("line_id", T.StringType(), True)])
    s = RowGroupedSurface(
        name="defects",
        schema=schema,
        timestamp_col="ts",
        group_col=("object_id", "line_id"),
    )
    assert s.group_cols == ("object_id", "line_id")


def test_unknown_timestamp_col_raises():
    with pytest.raises(ValueError, match="timestamp_col 'missing' not in schema"):
        RowGroupedSurface(name="x", schema=_schema(), timestamp_col="missing")


def test_unknown_group_col_raises():
    with pytest.raises(ValueError, match="group_col references unknown columns"):
        RowGroupedSurface(
            name="x", schema=_schema(), timestamp_col="ts", group_col="missing"
        )


def test_missing_container_id_raises():
    schema = T.StructType(
        [
            T.StructField("ts", T.LongType(), nullable=False),
            T.StructField("magnitude", T.DoubleType(), nullable=True),
        ]
    )
    with pytest.raises(ValueError, match="must include a 'container_id' column"):
        RowGroupedSurface(name="x", schema=schema, timestamp_col="ts")
