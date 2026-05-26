"""Integration tests for mda_reporting.playlists.event_fact_to_playlist_items.

Requires a real SparkSession; uses the session-scoped `spark` fixture from
tests/conftest.py.
"""

from datetime import datetime, timezone

import pytest
import pyspark.sql.types as T

from mda_reporting.playlists import event_fact_to_playlist_items


_EVENT_FACT_SCHEMA = T.StructType([
    T.StructField("container_id",      T.LongType(),    nullable=False),
    T.StructField("event_instance_id", T.LongType(),    nullable=False),
    T.StructField("event_id",          T.IntegerType(), nullable=False),
    T.StructField("start_ts",          T.LongType(),    nullable=False),
    T.StructField("end_ts",            T.LongType(),    nullable=False),
])


def _events_df(spark):
    return spark.createDataFrame(
        [
            (1, 100, 1, 1_000_000, 2_000_000),
            (1, 101, 1, 5_000_000, 6_000_000),
            (2, 102, 1, 10_000_000, 11_000_000),
        ],
        _EVENT_FACT_SCHEMA,
    )


def test_row_count_preserved(spark):
    out = event_fact_to_playlist_items(_events_df(spark), playlist_id="p1")
    assert out.count() == 3


def test_event_id_is_stable_sha256_prefix(spark):
    out1 = event_fact_to_playlist_items(_events_df(spark), playlist_id="p1").orderBy("start_ts").collect()
    out2 = event_fact_to_playlist_items(_events_df(spark), playlist_id="p1").orderBy("start_ts").collect()
    assert [r.event_id for r in out1] == [r.event_id for r in out2]


def test_event_ids_unique_across_distinct_windows(spark):
    out = event_fact_to_playlist_items(_events_df(spark), playlist_id="p1").collect()
    ids = [r.event_id for r in out]
    assert len(set(ids)) == len(ids)


def test_event_id_is_32_chars(spark):
    out = event_fact_to_playlist_items(_events_df(spark), playlist_id="p1").collect()
    for r in out:
        assert len(r.event_id) == 32


def test_playlist_columns_set(spark):
    now = datetime(2026, 5, 22, 12, 0, 0, tzinfo=timezone.utc)
    out = event_fact_to_playlist_items(
        _events_df(spark),
        playlist_id="my_playlist",
        playlist_version=3,
        event_name="my_event",
        now_utc=now,
    ).collect()
    for r in out:
        assert r.playlist_id == "my_playlist"
        assert r.playlist_version == 3
        assert r.event_name == "my_event"
        assert r.created_at == now


def test_column_set_matches_playlist_items_shape(spark):
    out = event_fact_to_playlist_items(_events_df(spark), playlist_id="p1")
    assert set(out.columns) == {
        "container_id", "event_id", "event_name", "start_ts", "end_ts",
        "playlist_id", "playlist_version", "created_at",
    }


def test_start_end_ts_cast_to_long(spark):
    out = event_fact_to_playlist_items(_events_df(spark), playlist_id="p1")
    schema = {f.name: f.dataType for f in out.schema.fields}
    assert isinstance(schema["start_ts"], T.LongType)
    assert isinstance(schema["end_ts"], T.LongType)


def test_default_version_is_one(spark):
    out = event_fact_to_playlist_items(_events_df(spark), playlist_id="p1").collect()
    for r in out:
        assert r.playlist_version == 1
