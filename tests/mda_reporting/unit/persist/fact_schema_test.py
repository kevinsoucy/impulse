"""Unit tests for fact schemas in mda_reporting.persist.fact_schema."""

import pyspark.sql.types as T

from mda_reporting.persist.fact_schema import (
    EVENT_INSTANCE_FACT_SCHEMA,
    PLAYLIST_ITEMS_SCHEMA,
)


class TestPlaylistItemsSchema:
    """PLAYLIST_ITEMS_SCHEMA — named, versioned event-window collections."""

    def test_has_required_fields(self):
        required = {
            "container_id",
            "event_id",
            "event_name",
            "start_ts",
            "end_ts",
            "playlist_id",
            "playlist_version",
            "created_at",
        }
        assert required <= {f.name for f in PLAYLIST_ITEMS_SCHEMA}

    def test_event_id_is_string(self):
        f = next(f for f in PLAYLIST_ITEMS_SCHEMA if f.name == "event_id")
        assert isinstance(f.dataType, T.StringType)
        assert f.nullable is False

    def test_start_end_ts_are_long(self):
        for name in ("start_ts", "end_ts"):
            f = next(f for f in PLAYLIST_ITEMS_SCHEMA if f.name == name)
            assert isinstance(f.dataType, T.LongType)

    def test_created_at_is_timestamp(self):
        f = next(f for f in PLAYLIST_ITEMS_SCHEMA if f.name == "created_at")
        assert isinstance(f.dataType, T.TimestampType)

    def test_playlist_version_is_int(self):
        f = next(f for f in PLAYLIST_ITEMS_SCHEMA if f.name == "playlist_version")
        assert isinstance(f.dataType, T.IntegerType)


class TestEventInstanceFactSchema:
    def test_has_required_fields(self):
        required = {"container_id", "event_instance_id", "event_id", "start_ts", "end_ts"}
        assert required <= {f.name for f in EVENT_INSTANCE_FACT_SCHEMA}
