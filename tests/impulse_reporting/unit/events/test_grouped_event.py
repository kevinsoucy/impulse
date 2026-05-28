"""GroupedEvent: validates group_scope consistency across leaves and
materializes per-entity rows with serialized group_value."""

import pandas as pd
import pyspark.sql.types as T
import pytest

from impulse_query_engine.surfaces import RowGroupedAccessor, RowGroupedSurface
from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.events.grouped_event import GroupedEvent, _serialize_group_value
from impulse_reporting.persist.fact_schema import EVENT_INSTANCE_FACT_SCHEMA


def _object_surface(group_col="object_id"):
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("ts", T.LongType(), nullable=False),
            T.StructField("object_id", T.LongType(), nullable=False),
            T.StructField("distance_m", T.DoubleType()),
        ]
    )
    return RowGroupedSurface(
        name="object_tracks", schema=schema, timestamp_col="ts", group_col=group_col
    )


def test_fact_schema_has_nullable_group_value_string_column():
    fields = {f.name: f for f in EVENT_INSTANCE_FACT_SCHEMA.fields}
    assert "group_value" in fields
    assert isinstance(fields["group_value"].dataType, T.StringType)
    assert fields["group_value"].nullable is True


def test_construction_requires_group_scope_true_on_every_leaf():
    accessor = RowGroupedAccessor(_object_surface())
    mixed = (accessor.distance_m < 8.0) & (
        accessor(group_scope=True).distance_m > 1.0
    )
    with pytest.raises(ValueError, match="group_scope=True"):
        GroupedEvent(name="mixed_scope", expr=mixed)


def test_construction_requires_a_row_grouped_leaf():
    # A predicate with no row-grouped leaves shouldn't pass; the channel
    # predicate path is BasicEvent's job.
    from impulse_query_engine.analyze.metadata.tag_expression import TagSelector
    from impulse_query_engine.analyze.metadata.time_series_expression import (
        TimeSeriesSelector,
    )

    ts = TimeSeriesSelector(TagSelector("name") == "speed")
    with pytest.raises(ValueError, match="at least one RowGroupedSelector"):
        GroupedEvent(name="no_surface", expr=(ts > 30))


def test_construction_rejects_multi_surface_for_now():
    surface_a = _object_surface()
    schema_b = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("ts", T.LongType(), nullable=False),
            T.StructField("lane_id", T.LongType(), nullable=False),
            T.StructField("offset_m", T.DoubleType()),
        ]
    )
    surface_b = RowGroupedSurface(
        name="lane_offsets", schema=schema_b, timestamp_col="ts", group_col="lane_id"
    )
    accessor_a = RowGroupedAccessor(surface_a)
    accessor_b = RowGroupedAccessor(surface_b)
    expr = (accessor_a(group_scope=True).distance_m < 8.0) & (
        accessor_b(group_scope=True).offset_m > 0.5
    )
    with pytest.raises(ValueError, match="single surface"):
        GroupedEvent(name="multi_surface", expr=expr)


def test_materialize_emits_one_row_per_group_per_window():
    surface = _object_surface()
    accessor = RowGroupedAccessor(surface)
    expr = accessor(group_scope=True).distance_m < 8.0
    event = GroupedEvent(name="cyclist_near_miss_per_object", expr=expr)

    df = pd.DataFrame(
        [
            (1, 0, 47, 5.0),
            (1, 1, 47, 6.0),
            (1, 2, 47, 99.0),  # off
            (1, 0, 91, 99.0),  # off
            (1, 1, 91, 4.0),
            (1, 2, 91, 4.0),
        ],
        columns=["container_id", "ts", "object_id", "distance_m"],
    )

    rows = event.materialize_per_container(
        container_id=1, surface_df=df, container_stop_ts=3
    )
    # Object 47: intervals [0,1) and [1,2) → merged to [0,2)
    # Object 91: intervals [1,2) and [2,3) → merged to [1,3)
    assert sorted(rows) == [
        (1, 0.0, 2.0, "47"),
        (1, 1.0, 3.0, "91"),
    ]


def test_materialize_skips_groups_with_no_matching_rows():
    surface = _object_surface()
    accessor = RowGroupedAccessor(surface)
    expr = accessor(group_scope=True).distance_m < 8.0
    event = GroupedEvent(name="ev", expr=expr)

    df = pd.DataFrame(
        [
            (1, 0, 47, 5.0),
            (1, 1, 47, 6.0),
            (1, 0, 91, 99.0),  # group 91 never matches
            (1, 1, 91, 99.0),
        ],
        columns=["container_id", "ts", "object_id", "distance_m"],
    )

    rows = event.materialize_per_container(1, df, container_stop_ts=2)
    assert len(rows) == 1
    assert rows[0][3] == "47"


def test_tuple_group_value_serializes_as_json():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("ts", T.LongType(), nullable=False),
            T.StructField("station_id", T.LongType(), nullable=False),
            T.StructField("line_id", T.LongType(), nullable=False),
            T.StructField("severity", T.DoubleType()),
        ]
    )
    surface = RowGroupedSurface(
        name="defects",
        schema=schema,
        timestamp_col="ts",
        group_col=("station_id", "line_id"),
    )
    accessor = RowGroupedAccessor(surface)
    expr = accessor(group_scope=True).severity > 0.7
    event = GroupedEvent(name="critical_per_station_line", expr=expr)

    df = pd.DataFrame(
        [
            (1, 0, 10, 1, 0.9),
            (1, 1, 10, 1, 0.95),
            (1, 0, 10, 2, 0.5),  # off
        ],
        columns=["container_id", "ts", "station_id", "line_id", "severity"],
    )
    rows = event.materialize_per_container(1, df, container_stop_ts=2)
    # Only the (10, 1) station-line pair triggers; serialized as JSON list.
    assert len(rows) == 1
    assert rows[0][3] == "[10, 1]"


def test_event_type_string_distinguishes_grouped_from_basic():
    surface = _object_surface()
    accessor = RowGroupedAccessor(surface)
    grouped = GroupedEvent(
        name="g", expr=accessor(group_scope=True).distance_m < 8.0
    )
    assert grouped.get_event_type_str() == "GROUPED_EVENT"
    # Sanity: still inherits BasicEvent so existing infrastructure (definition
    # hash, attributes, expression aliasing) is reused.
    assert isinstance(grouped, BasicEvent)


def test_serialize_group_value_handles_numeric_and_string_and_compound():
    assert _serialize_group_value(47) == "47"
    assert _serialize_group_value(4.5) == "4.5"
    assert _serialize_group_value("red") == "red"
    assert _serialize_group_value((10, 1)) == "[10, 1]"
    assert _serialize_group_value(["a", "b"]) == '["a", "b"]'
