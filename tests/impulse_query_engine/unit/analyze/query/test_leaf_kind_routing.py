"""leaf_kind routes selectors away from the channel filter pipeline so a
series leaf never lands in channel_tags / channel_metrics stages."""

from types import SimpleNamespace

import pyspark.sql.types as T

from impulse_query_engine.analyze.metadata.tag_expression import TagSelector
from impulse_query_engine.analyze.metadata.time_series_expression import (
    TimeSeriesSelector,
)
from impulse_query_engine.analyze.query.query_builder import QueryBuilder
from impulse_query_engine.surfaces import SeriesAccessor, Series


def _object_series():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("sensor_type", T.StringType(), nullable=False),
            T.StructField("ts", T.LongType(), nullable=False),
            T.StructField("object_id", T.LongType(), nullable=False),
            T.StructField("distance_m", T.DoubleType()),
        ]
    )
    return Series(
        name="object_tracks",
        schema=schema,
        session_col="container_id",
        signal_col="sensor_type",
        timestamp_col="ts",
        entity_key="object_id",
    )


def _stub_query() -> QueryBuilder:
    # QueryBuilder only needs db.ws to construct; for selector collection
    # we don't touch Spark at all.
    stub_db = SimpleNamespace(ws=None)
    return QueryBuilder(stub_db)


def test_default_time_series_selector_has_channel_leaf_kind():
    sel = TimeSeriesSelector(TagSelector("name") == "speed")
    assert sel.leaf_kind == "channel"


def test_series_selector_overrides_leaf_kind_to_series_name():
    accessor = SeriesAccessor(_object_series())
    sel = accessor.distance_m < 8.0
    assert sel.leaf_kind == "object_tracks"


def test_collect_selectors_filters_by_leaf_kind():
    query = _stub_query()
    channel = TimeSeriesSelector(TagSelector("name") == "speed")
    surface_leaf = SeriesAccessor(_object_series()).distance_m < 8.0

    composed = (channel > 30) & surface_leaf
    query.select(composed)

    channel_only = query._collect_time_series_selectors(uses_alias=False, leaf_kind="channel")
    assert [s.leaf_kind for s in channel_only] == ["channel"]

    series_only = query._collect_time_series_selectors(uses_alias=False, leaf_kind="object_tracks")
    assert [s.leaf_kind for s in series_only] == ["object_tracks"]

    all_leaves = query._collect_time_series_selectors(uses_alias=False)
    assert {s.leaf_kind for s in all_leaves} == {"channel", "object_tracks"}
