"""BasicEvent is the single public event constructor; it routes to the
per-entity implementation when the expression carries an entity-scoped leaf
(``.each()`` / ``.ids()``). A plain ``.any()`` — even over an entity-bearing
series — stays a presence event with a NULL ``entity_key``."""

import pytest
import pyspark.sql.types as T

from impulse_query_engine.surfaces import Series, SeriesAccessor
from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.events.entity_event import EntityEvent


def _entity_accessor():
    """An entity-bearing series (declares an ``entity_key``)."""
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("sensor_type", T.StringType(), nullable=False),
            T.StructField("frame_ts", T.LongType(), nullable=False),
            T.StructField("entity_id", T.LongType(), nullable=False),
            T.StructField("distance_m", T.DoubleType()),
        ]
    )
    return SeriesAccessor(
        Series(
            name="object_tracks",
            schema=schema,
            session_col="container_id",
            signal_col="sensor_type",
            timestamp_col="frame_ts",
            entity_key="entity_id",
        )
    )


def _channel_accessor():
    """A series with no ``entity_key`` — there is no entity identity to project."""
    schema = T.StructType(
        [
            T.StructField("session_id", T.LongType()),
            T.StructField("signal_id", T.StringType()),
            T.StructField("sample_ts", T.LongType()),
            T.StructField("accel_x", T.DoubleType()),
        ]
    )
    return SeriesAccessor(Series(name="imu", schema=schema, timestamp_col="sample_ts"))


# --- stays a presence BasicEvent --------------------------------------------


def test_any_over_entity_series_stays_basic_event():
    acc = _entity_accessor()
    event = BasicEvent(name="near", expr=(acc.distance_m < 8.0).any())
    assert type(event) is BasicEvent
    assert event.get_event_type_str() == "BASIC_EVENT"


def test_non_entity_predicate_stays_basic_event():
    acc = _channel_accessor()
    event = BasicEvent(name="hard_brake", expr=(acc.accel_x < -4.0).any())
    assert type(event) is BasicEvent
    assert event.get_event_type_str() == "BASIC_EVENT"


# --- routes to the entity implementation ------------------------------------


def test_each_routes_to_entity_event():
    acc = _entity_accessor()
    event = BasicEvent(name="near", expr=(acc.distance_m < 8.0).each())
    assert type(event) is EntityEvent
    assert isinstance(event, BasicEvent)  # routed instance is still a BasicEvent subclass
    assert event.get_event_type_str() == "ENTITY_EVENT"


def test_any_ids_routes_to_entity_event():
    acc = _entity_accessor()
    event = BasicEvent(name="near", expr=(acc.distance_m < 8.0).any().ids(as_="obj"))
    assert type(event) is EntityEvent
    assert event.get_event_type_str() == "ENTITY_EVENT"


def test_cross_entity_with_ids_routes_to_entity_event():
    acc = _entity_accessor()
    cyclist = (acc.distance_m < 8.0).any().ids(as_="cyclist")
    car = (acc.distance_m < 15.0).any().ids(as_="car")
    event = BasicEvent(name="squeeze", expr=cyclist & car)
    assert type(event) is EntityEvent


# --- bare entity predicate must pick a windowing verb -----------------------


def test_bare_entity_predicate_requires_windowing_verb():
    acc = _entity_accessor()
    with pytest.raises(ValueError, match="windowing verb"):
        BasicEvent(name="near", expr=(acc.distance_m < 8.0))
