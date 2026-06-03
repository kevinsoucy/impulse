"""EntityEvent: per-entity / cross-entity materialization with the alias-keyed,
signal-scoped ``{alias: {signal: [ids]}}`` entity_key map, the two windowing
verbs (``.any()`` merged, ``.each()`` per-entity), and ``.ids(as_=…)`` projection."""

import json

import pandas as pd
import pyspark.sql.types as T
import pytest

from impulse_query_engine.analyze.query.solvers.empty_cache import EmptyTimeSeriesCache
from impulse_query_engine.analyze.query.solvers.series_cache import CombinedSeriesCache
from impulse_query_engine.model.series.intervals import Intervals
from impulse_query_engine.surfaces import Series, SeriesAccessor
from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.events.entity_event import EntityEvent, _eval_tree, _serialize_roster
from impulse_reporting.persist.fact_schema import EVENT_INSTANCE_FACT_SCHEMA


def _mk_cache(series_frames, container_stop_ts, channel_cache=None):
    return CombinedSeriesCache(
        channel_cache or EmptyTimeSeriesCache(),
        series_frames,
        container_stop_ts=container_stop_ts,
    )


def _object_series(entity_key="entity_id"):
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("sensor_type", T.StringType(), nullable=False),
            T.StructField("frame_ts", T.LongType(), nullable=False),
            T.StructField("entity_id", T.LongType(), nullable=False),
            T.StructField("detection_class", T.StringType()),
            T.StructField("distance_m", T.DoubleType()),
        ]
    )
    return Series(
        name="object_tracks",
        schema=schema,
        session_col="container_id",
        signal_col="sensor_type",
        timestamp_col="frame_ts",
        entity_key=entity_key,
    )


def _frame(rows, signal="fusion"):
    return pd.DataFrame(
        [(1, signal, ts, eid, cls, d) for (ts, eid, cls, d) in rows],
        columns=[
            "container_id",
            "sensor_type",
            "frame_ts",
            "entity_id",
            "detection_class",
            "distance_m",
        ],
    )


def _tracks(rows, signal="fusion"):
    """``series_frames`` map for an object_tracks-only expression."""
    return {"object_tracks": _frame(rows, signal=signal)}


def _maps(rows):
    """Parsed alias-keyed entity_key maps from materialized rows (sorted).

    A materialized row is ``(container_id, start_ts, end_ts, entity_key_json)``;
    ``entity_key_json`` is ``None`` for an unprojected window.
    """
    return sorted(
        (json.loads(entity_key) for (_cid, _start, _end, entity_key) in rows if entity_key),
        key=json.dumps,
    )


def _cyclist_pred(acc):
    return (acc.detection_class == "cyclist") & (acc.distance_m < 8.0)


def _car_pred(acc):
    return (acc.detection_class == "car") & (acc.distance_m < 8.0)


# --- fact schema ------------------------------------------------------------


def test_fact_schema_has_nullable_entity_key_string_column():
    fields = {f.name: f for f in EVENT_INSTANCE_FACT_SCHEMA.fields}
    assert "entity_key" in fields
    assert isinstance(fields["entity_key"].dataType, T.StringType)
    assert fields["entity_key"].nullable is True


def test_event_instance_id_is_long_not_int32():
    # event_instance_id is a crc32 hash in [0, 2**32 - 1], which overflows int32.
    fields = {f.name: f for f in EVENT_INSTANCE_FACT_SCHEMA.fields}
    assert isinstance(fields["event_instance_id"].dataType, T.LongType)


# --- construction validation ------------------------------------------------


def test_each_raises_on_series_without_entity_key():
    schema = T.StructType(
        [
            T.StructField("session_id", T.LongType()),
            T.StructField("signal_id", T.StringType()),
            T.StructField("sample_ts", T.LongType()),
            T.StructField("accel_x", T.DoubleType()),
        ]
    )
    imu = SeriesAccessor(Series(name="imu", schema=schema, timestamp_col="sample_ts"))
    with pytest.raises(ValueError, match="imu.*entity_key|has no entity_key"):
        (imu.accel_x < -4.0).each()


def test_ids_raises_on_series_without_entity_key():
    schema = T.StructType(
        [
            T.StructField("session_id", T.LongType()),
            T.StructField("signal_id", T.StringType()),
            T.StructField("sample_ts", T.LongType()),
            T.StructField("accel_x", T.DoubleType()),
        ]
    )
    imu = SeriesAccessor(Series(name="imu", schema=schema, timestamp_col="sample_ts"))
    with pytest.raises(ValueError, match="imu.*entity_key|has no entity_key"):
        (imu.accel_x < -4.0).any().ids(as_="imu")


def test_requires_a_series_leaf():
    from impulse_query_engine.analyze.metadata.tag_expression import TagSelector
    from impulse_query_engine.analyze.metadata.time_series_expression import (
        TimeSeriesSelector,
    )

    ts = TimeSeriesSelector(TagSelector("name") == "speed")
    with pytest.raises(ValueError, match="at least one SeriesSelector"):
        EntityEvent(name="no_series", expr=(ts > 30))


def test_requires_an_entity_scoped_leaf():
    acc = SeriesAccessor(_object_series())
    # A presence leaf (.any(), no .ids()) is not entity-scoped → EntityEvent rejects it.
    with pytest.raises(ValueError, match="entity-scoped leaf"):
        EntityEvent(name="presence_only", expr=(acc.distance_m < 8.0).any())


def test_bare_partial_in_entity_event_raises():
    acc = SeriesAccessor(_object_series())
    with pytest.raises(ValueError, match="needs an entity-scoped leaf"):
        EntityEvent(name="bare", expr=(acc.distance_m < 8.0))


def test_bare_entity_predicate_in_basic_event_raises():
    acc = SeriesAccessor(_object_series())
    with pytest.raises(ValueError, match="needs a windowing verb"):
        BasicEvent(name="bare", expr=(acc.distance_m < 8.0))


def test_entity_leaf_routes_basic_event_to_entity_event():
    # BasicEvent is the single public constructor; an entity-scoped leaf routes
    # construction to the EntityEvent implementation (see routing_test.py).
    acc = SeriesAccessor(_object_series())
    event = BasicEvent(name="ent", expr=(acc.distance_m < 8.0).each())
    assert type(event) is EntityEvent


def test_two_each_legs_raise_no_cartesian():
    acc = SeriesAccessor(_object_series())
    cyclist = _cyclist_pred(acc).each().ids(as_="cyclist")
    car = _car_pred(acc).each().ids(as_="car")
    with pytest.raises(ValueError, match="At most one .each"):
        EntityEvent(name="squeeze", expr=cyclist & car)


def test_duplicate_alias_raises():
    acc = SeriesAccessor(_object_series())
    cyclist = _cyclist_pred(acc).any().ids(as_="obj")
    car = _car_pred(acc).any().ids(as_="obj")
    with pytest.raises(ValueError, match="Duplicate .ids"):
        EntityEvent(name="squeeze", expr=cyclist & car)


# --- definition hash ---------------------------------------------------------


def test_definition_hash_is_deterministic_across_instances():
    acc = SeriesAccessor(_object_series())
    near = (acc.distance_m < 8.0).each().ids(as_="obj")
    event1 = EntityEvent(name="near", expr=near)
    event2 = EntityEvent(name="near", expr=near)
    assert event1.determine_definition_hash() == event2.determine_definition_hash()


def test_windowing_verb_changes_definition_hash():
    # The windowing axis is result-affecting: .each() emits one row per entity,
    # .any() one merged window. Flipping it must register as a redefinition
    # (different definition_hash) so the upsert replaces the prior rows.
    acc = SeriesAccessor(_object_series())
    pred = acc.distance_m < 8.0
    per_entity = EntityEvent(name="near", expr=pred.each().ids(as_="obj"))
    merged = EntityEvent(name="near", expr=pred.any().ids(as_="obj"))
    assert per_entity.determine_definition_hash() != merged.determine_definition_hash()


def test_ids_alias_changes_definition_hash():
    # The projection alias is part of the emitted facts (it keys entity_key), so
    # changing it is a redefinition.
    acc = SeriesAccessor(_object_series())
    pred = acc.distance_m < 8.0
    a = EntityEvent(name="near", expr=pred.each().ids(as_="cyclist"))
    b = EntityEvent(name="near", expr=pred.each().ids(as_="bike"))
    assert a.determine_definition_hash() != b.determine_definition_hash()


# --- .each(): per-entity fan-out --------------------------------------------


def test_each_fans_out_one_row_per_entity():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="near_miss", expr=_cyclist_pred(acc).each().ids(as_="cyclist"))
    frames = _tracks(
        [
            (0, 47, "cyclist", 5.0),
            (1, 47, "cyclist", 5.0),
            (0, 48, "cyclist", 5.0),
            (1, 48, "cyclist", 5.0),
        ]
    )
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert _maps(rows) == [{"cyclist": {"fusion": ["47"]}}, {"cyclist": {"fusion": ["48"]}}]
    # Each entity carries its own interval.
    assert all(r[1] == 0.0 and r[2] == 2.0 for r in rows)


def test_each_without_ids_emits_null_entity_key_per_entity():
    # Edge case: .each() with no .ids() — per-entity windows, NULL entity_key
    # (the windows are still distinct; the rows are deliberately unattributed).
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="near", expr=(acc.distance_m < 8.0).each())
    frames = _tracks(
        [
            (0, 47, "cyclist", 5.0),
            (1, 47, "cyclist", 5.0),
            (3, 48, "cyclist", 5.0),
            (4, 48, "cyclist", 5.0),
        ]
    )
    rows = event.materialize_per_container(1, _mk_cache(frames, 5))
    assert len(rows) == 2
    assert all(r[3] is None for r in rows)
    # Point-in-time synthesis: 47's last frame (ts 1) closes at the next fusion
    # tick (3); 48's last frame (4) closes at container_stop_ts (5).
    assert {(r[1], r[2]) for r in rows} == {(0.0, 3.0), (3.0, 5.0)}


def test_each_skips_entities_with_no_matching_rows():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="ev", expr=(acc.distance_m < 8.0).each().ids(as_="obj"))
    frames = _tracks(
        [
            (0, 47, "cyclist", 5.0),
            (1, 47, "cyclist", 6.0),
            (0, 91, "car", 99.0),  # never matches
            (1, 91, "car", 99.0),
        ]
    )
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert _maps(rows) == [{"obj": {"fusion": ["47"]}}]


def test_numeric_entity_keys_cast_to_string():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="ev", expr=(acc.distance_m < 8.0).each().ids(as_="obj"))
    frames = _tracks([(0, 47, "cyclist", 5.0)])
    rows = event.materialize_per_container(1, _mk_cache(frames, 1))
    assert json.loads(rows[0][3]) == {"obj": {"fusion": ["47"]}}


def test_same_id_across_signals_stays_distinct_by_signal():
    # The same track id under two sensors is a distinct entity: signal is part of
    # the identity, so it stays in the entity_key — lidar-47 and radar-47 land in
    # separate signal buckets, never conflated.
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="ev", expr=(acc.distance_m < 8.0).each().ids(as_="obj"))
    df = pd.concat(
        [
            _frame([(0, 47, "obj", 5.0), (1, 47, "obj", 5.0)], signal="lidar"),
            _frame([(3, 47, "obj", 5.0), (4, 47, "obj", 5.0)], signal="radar"),
        ],
        ignore_index=True,
    )
    rows = event.materialize_per_container(1, _mk_cache({"object_tracks": df}, 5))
    assert _maps(rows) == [{"obj": {"lidar": ["47"]}}, {"obj": {"radar": ["47"]}}]


# --- .any().ids(): merged roster --------------------------------------------


def test_any_ids_unions_entities_into_one_window():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="near", expr=(acc.distance_m < 8.0).any().ids(as_="obj"))
    frames = _tracks(
        [
            (0, 47, "cyclist", 5.0),
            (1, 47, "cyclist", 5.0),
            (0, 48, "cyclist", 5.0),
            (1, 48, "cyclist", 5.0),
        ]
    )
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert len(rows) == 1
    assert _maps(rows) == [{"obj": {"fusion": ["47", "48"]}}]


def test_any_ids_roster_truncates_at_limit():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="near", expr=(acc.distance_m < 8.0).any().ids(as_="obj", limit=2))
    frames = _tracks(
        [(t, eid, "cyclist", 5.0) for eid in (10, 11, 12, 13) for t in (0, 1)]
    )
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    # Four entities match, but the roster is capped at 2 (smallest ids by sort).
    assert _maps(rows) == [{"obj": {"fusion": ["10", "11"]}}]


def test_any_ids_roster_filters_to_emitted_window():
    # Roster includes only entities whose interval overlaps the emitted window —
    # not every entity that ever matched the predicate.
    acc = SeriesAccessor(_object_series())
    speed_window = Intervals([0.0], [2.0])  # channel-side window
    cyclist = (acc.distance_m < 8.0).any().ids(as_="obj")
    event = EntityEvent(name="near", expr=cyclist & speed_window)
    frames = _tracks(
        [
            (0, 47, "cyclist", 5.0),
            (1, 47, "cyclist", 5.0),
            (5, 48, "cyclist", 5.0),  # only present at frame 5, outside [0,2)
            (6, 48, "cyclist", 5.0),
        ]
    )
    rows = event.materialize_per_container(1, _mk_cache(frames, 7))
    assert _maps(rows) == [{"obj": {"fusion": ["47"]}}]  # 48 overlaps no emitted window


# --- cross-entity correlation -----------------------------------------------


def _two_cyclists_one_car():
    return _tracks(
        [
            (0, 47, "cyclist", 5.0),
            (1, 47, "cyclist", 5.0),
            (0, 48, "cyclist", 5.0),
            (1, 48, "cyclist", 5.0),
            (0, 91, "car", 5.0),
            (1, 91, "car", 5.0),
        ]
    )


def test_cross_entity_merged_rosters_keyed_by_alias():
    acc = SeriesAccessor(_object_series())
    expr = _cyclist_pred(acc).any().ids(as_="cyclist") & _car_pred(acc).any().ids(as_="car")
    event = EntityEvent(name="squeeze", expr=expr)
    rows = event.materialize_per_container(1, _mk_cache(_two_cyclists_one_car(), 2))
    assert len(rows) == 1
    assert _maps(rows) == [{"car": {"fusion": ["91"]}, "cyclist": {"fusion": ["47", "48"]}}]


def test_mixed_any_each_emits_one_row_per_each_entity():
    # .any() collapses; the single .each() leg multiplies. One car → one row,
    # carrying the full cyclist roster.
    acc = SeriesAccessor(_object_series())
    expr = _cyclist_pred(acc).any().ids(as_="cyclist") & _car_pred(acc).each().ids(as_="car")
    event = EntityEvent(name="squeeze", expr=expr)
    rows = event.materialize_per_container(1, _mk_cache(_two_cyclists_one_car(), 2))
    assert _maps(rows) == [{"car": {"fusion": ["91"]}, "cyclist": {"fusion": ["47", "48"]}}]


def test_cross_entity_merged_no_rosters_when_unprojected_is_basic_territory():
    # Without .ids() the cross-entity merged form is pure presence — it belongs to
    # BasicEvent (no entity leaves), so EntityEvent rejects it.
    acc = SeriesAccessor(_object_series())
    expr = _cyclist_pred(acc).any() & _car_pred(acc).any()
    with pytest.raises(ValueError, match="entity-scoped leaf"):
        EntityEvent(name="squeeze", expr=expr)


# --- cross-series co-occurrence ----------------------------------------------


def _sign_series():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("sensor_type", T.StringType(), nullable=False),
            T.StructField("frame_ts", T.LongType(), nullable=False),
            T.StructField("sign_id", T.LongType(), nullable=False),
            T.StructField("sign_class", T.StringType()),
        ]
    )
    return Series(
        name="traffic_signs",
        schema=schema,
        session_col="container_id",
        signal_col="sensor_type",
        timestamp_col="frame_ts",
        entity_key="sign_id",
    )


def _signs(rows, signal="camera_front"):
    return pd.DataFrame(
        [(1, signal, ts, sid, cls) for (ts, sid, cls) in rows],
        columns=["container_id", "sensor_type", "frame_ts", "sign_id", "sign_class"],
    )


def test_cross_series_co_occurrence_keys_each_side_by_its_alias():
    obj = SeriesAccessor(_object_series())
    sign = SeriesAccessor(_sign_series())
    cyclist_close = _cyclist_pred(obj).any().ids(as_="cyclist")
    sign_30 = (sign.sign_class == "speed_30").any().ids(as_="sign")
    event = EntityEvent(name="squeeze", expr=cyclist_close & sign_30)

    frames = {
        "object_tracks": _frame([(0, 47, "cyclist", 5.0), (1, 47, "cyclist", 5.0)]),
        "traffic_signs": _signs([(0, 12, "speed_30"), (1, 12, "speed_30")]),
    }
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert _maps(rows) == [{"cyclist": {"fusion": ["47"]}, "sign": {"camera_front": ["12"]}}]


def test_cross_series_conjunction_with_empty_other_series_does_not_fire():
    obj = SeriesAccessor(_object_series())
    sign = SeriesAccessor(_sign_series())
    cyclist_close = (obj.distance_m < 8.0).any().ids(as_="cyclist")
    sign_30 = (sign.sign_class == "speed_30").any().ids(as_="sign")
    event = EntityEvent(name="squeeze", expr=cyclist_close & sign_30)

    frames = {
        "object_tracks": _frame([(0, 47, "cyclist", 5.0), (1, 47, "cyclist", 5.0)]),
        "traffic_signs": _signs([]),  # no signs detected
    }
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert rows == []


def test_cross_series_construction_is_accepted():
    obj = SeriesAccessor(_object_series())
    sign = SeriesAccessor(_sign_series())
    expr = (obj.distance_m < 8.0).any().ids(as_="obj") & (
        sign.sign_class == "speed_30"
    ).any().ids(as_="sign")
    event = EntityEvent(name="multi", expr=expr)
    assert {leaf.series.name for leaf in EntityEvent._series_leaves(event.expression)} == {
        "object_tracks",
        "traffic_signs",
    }


# --- _eval_tree leaf resolution ----------------------------------------------


class _PinnedLeaf:
    """A leaf substituted from the per-entity map (not an op, not built)."""


class _ChannelLeaf:
    """A non-pinned, non-op leaf — resolved via cache.build (channel path)."""

    def __init__(self, result):
        self._result = result

    def build(self, cache):
        return self._result


def test_eval_tree_returns_pinned_leaf_from_substitution_map():
    leaf = _PinnedLeaf()
    pinned = Intervals.empty()
    assert _eval_tree(leaf, {id(leaf): pinned}, EmptyTimeSeriesCache()) is pinned


def test_eval_tree_resolves_unpinned_channel_leaf_via_cache_build():
    ivs = Intervals([0.0], [5.0])
    leaf = _ChannelLeaf(ivs)
    assert _eval_tree(leaf, {}, EmptyTimeSeriesCache()) is ivs


def test_eval_tree_memoizes_channel_leaf_build_across_calls():
    calls = []

    class _CountingLeaf:
        def build(self, cache):
            calls.append(1)
            return Intervals([0.0], [5.0])

    leaf = _CountingLeaf()
    build_cache: dict = {}
    _eval_tree(leaf, {}, EmptyTimeSeriesCache(), build_cache)
    _eval_tree(leaf, {}, EmptyTimeSeriesCache(), build_cache)
    assert len(calls) == 1  # built once, reused on the second entity iteration


def test_materialize_with_channel_cache_none_uses_empty_channel_cache():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="near", expr=(acc.distance_m < 8.0).each().ids(as_="obj"))
    rows = event.materialize_per_container(
        1, _mk_cache(_tracks([(0, 47, "cyclist", 5.0), (1, 47, "cyclist", 5.0)]), 2)
    )
    assert _maps(rows) == [{"obj": {"fusion": ["47"]}}]


# --- compound entity key ----------------------------------------------------


def test_compound_entity_key_serializes_as_json_array_string():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType()),
            T.StructField("sensor_type", T.StringType()),
            T.StructField("frame_ts", T.LongType()),
            T.StructField("station_id", T.LongType()),
            T.StructField("line_id", T.LongType()),
            T.StructField("severity", T.DoubleType()),
        ]
    )
    acc = SeriesAccessor(
        Series(
            name="defects",
            schema=schema,
            session_col="container_id",
            signal_col="sensor_type",
            timestamp_col="frame_ts",
            entity_key=("station_id", "line_id"),
        )
    )
    event = EntityEvent(name="critical", expr=(acc.severity > 0.7).each().ids(as_="defect"))
    df = pd.DataFrame(
        [
            (1, "vision", 0, 10, 1, 0.9),
            (1, "vision", 1, 10, 1, 0.95),
            (1, "vision", 0, 10, 2, 0.5),  # off
        ],
        columns=[
            "container_id",
            "sensor_type",
            "frame_ts",
            "station_id",
            "line_id",
            "severity",
        ],
    )
    rows = event.materialize_per_container(1, _mk_cache({"defects": df}, 2))
    assert _maps(rows) == [{"defect": {"vision": ["[10, 1]"]}}]


# --- misc -------------------------------------------------------------------


def test_event_type_string_and_inheritance():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="g", expr=(acc.distance_m < 8.0).each().ids(as_="obj"))
    assert event.get_event_type_str() == "ENTITY_EVENT"
    assert isinstance(event, BasicEvent)


def test_serialize_roster_sorts_aliases_and_signals():
    payload = _serialize_roster(
        {"cyclist": {"lidar": ["47", "91"]}, "car": {"fusion": ["12"]}}
    )
    parsed = json.loads(payload)
    assert parsed == {"car": {"fusion": ["12"]}, "cyclist": {"lidar": ["47", "91"]}}
    assert list(parsed) == ["car", "cyclist"]  # alias keys emitted in sorted order
