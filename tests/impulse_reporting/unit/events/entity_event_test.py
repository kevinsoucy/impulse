"""EntityEvent: per-entity / cross-entity materialization with the nested
``{table: {signal: [ids]}}`` entity_key map and per_entity_windowing."""

import json

import pandas as pd
import pyspark.sql.types as T
import pytest

from impulse_query_engine.analyze.query.solvers.empty_cache import EmptyTimeSeriesCache
from impulse_query_engine.analyze.query.solvers.series_cache import CombinedSeriesCache
from impulse_query_engine.model.series.intervals import Intervals
from impulse_query_engine.surfaces import Series, SeriesAccessor
from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.events.entity_event import EntityEvent, _eval_tree, _serialize_entity_map
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
    """Extract the parsed entity_key maps from materialized rows.

    A materialized row is ``(container_id, start_ts, end_ts, entity_key_json)``.
    """
    return sorted(
        (json.loads(entity_key) for (_cid, _start, _end, entity_key) in rows),
        key=json.dumps,
    )


# --- fact schema ------------------------------------------------------------


def test_fact_schema_has_nullable_entity_key_string_column():
    fields = {f.name: f for f in EVENT_INSTANCE_FACT_SCHEMA.fields}
    assert "entity_key" in fields
    assert isinstance(fields["entity_key"].dataType, T.StringType)
    assert fields["entity_key"].nullable is True


def test_no_entity_key_type_column():
    names = {f.name for f in EVENT_INSTANCE_FACT_SCHEMA.fields}
    assert "entity_key_type" not in names


def test_event_instance_id_is_long_not_int32():
    # event_instance_id is a crc32 hash in [0, 2**32 - 1], which overflows int32.
    # Guard against a regression to IntegerType.
    fields = {f.name: f for f in EVENT_INSTANCE_FACT_SCHEMA.fields}
    assert isinstance(fields["event_instance_id"].dataType, T.LongType)


# --- construction validation ------------------------------------------------


def test_entity_condition_raises_on_series_without_entity_key():
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
        (imu.accel_x < -4.0).entity_condition()


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
    # A presence predicate (no .entity_condition()) is not entity-scoped.
    presence = acc.distance_m < 8.0
    with pytest.raises(ValueError, match="entity-scoped leaf"):
        EntityEvent(name="presence_only", expr=presence._finalize_presence())


# --- definition hash ---------------------------------------------------------


def test_definition_hash_is_deterministic_across_instances():
    acc = SeriesAccessor(_object_series())
    near = (acc.distance_m < 8.0).entity_condition()
    event1 = EntityEvent(name="near", expr=near)
    event2 = EntityEvent(name="near", expr=near)
    assert event1.determine_definition_hash() == event2.determine_definition_hash()


def test_per_entity_windowing_changes_definition_hash():
    # per_entity_windowing is result-affecting: True emits one fact row per entity,
    # False folds them into one window with a combined entity-set map. Flipping it
    # must register as a redefinition (different definition_hash) so the upsert
    # replaces the prior rows rather than stranding them.
    acc = SeriesAccessor(_object_series())
    near = (acc.distance_m < 8.0).entity_condition()
    per_entity = EntityEvent(name="near", expr=near, per_entity_windowing=True)
    combined = EntityEvent(name="near", expr=near, per_entity_windowing=False)
    assert per_entity.determine_definition_hash() != combined.determine_definition_hash()


# --- single entity condition: per-entity fan-out ----------------------------


def test_single_condition_fans_out_one_row_per_entity():
    acc = SeriesAccessor(_object_series())
    near = ((acc.detection_class == "cyclist") & (acc.distance_m < 8.0)).entity_condition()
    event = EntityEvent(name="near_miss", expr=near)
    frames = _tracks(
        [
            (0, 47, "cyclist", 5.0),
            (1, 47, "cyclist", 5.0),
            (0, 48, "cyclist", 5.0),
            (1, 48, "cyclist", 5.0),
        ]
    )
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert _maps(rows) == [
        {"object_tracks": {"fusion": ["47"]}},
        {"object_tracks": {"fusion": ["48"]}},
    ]
    # Each entity carries its own interval.
    assert all(r[1] == 0.0 and r[2] == 2.0 for r in rows)


def test_single_entity_still_produces_nested_map_not_bare_scalar():
    acc = SeriesAccessor(_object_series())
    near = (acc.distance_m < 8.0).entity_condition()
    event = EntityEvent(name="near", expr=near)
    frames = _tracks([(0, 47, "cyclist", 5.0), (1, 47, "cyclist", 5.0)])
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert _maps(rows) == [{"object_tracks": {"fusion": ["47"]}}]


def test_skips_entities_with_no_matching_rows():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="ev", expr=(acc.distance_m < 8.0).entity_condition())
    frames = _tracks(
        [
            (0, 47, "cyclist", 5.0),
            (1, 47, "cyclist", 6.0),
            (0, 91, "car", 99.0),  # never matches
            (1, 91, "car", 99.0),
        ]
    )
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert _maps(rows) == [{"object_tracks": {"fusion": ["47"]}}]


def test_numeric_entity_keys_cast_to_string():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="ev", expr=(acc.distance_m < 8.0).entity_condition())
    frames = _tracks([(0, 47, "cyclist", 5.0)])
    rows = event.materialize_per_container(1, _mk_cache(frames, 1))
    # 47 (numeric) serializes as the string "47".
    assert json.loads(rows[0][3]) == {"object_tracks": {"fusion": ["47"]}}


# --- (signal, entity) scoping -----------------------------------------------


def test_same_id_across_signals_stays_distinct():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="ev", expr=(acc.distance_m < 8.0).entity_condition())
    df = pd.concat(
        [
            _frame([(0, 47, "obj", 5.0), (1, 47, "obj", 5.0)], signal="lidar"),
            _frame([(0, 47, "obj", 5.0), (1, 47, "obj", 5.0)], signal="radar"),
        ],
        ignore_index=True,
    )
    rows = event.materialize_per_container(1, _mk_cache({"object_tracks": df}, 2))
    assert _maps(rows) == [
        {"object_tracks": {"lidar": ["47"]}},
        {"object_tracks": {"radar": ["47"]}},
    ]


# --- cross-entity co-occurrence ---------------------------------------------


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


def _cyclist_and_car(acc):
    cyclist = ((acc.detection_class == "cyclist") & (acc.distance_m < 8.0)).entity_condition()
    car = ((acc.detection_class == "car") & (acc.distance_m < 8.0)).entity_condition()
    return cyclist & car


def test_per_entity_windowing_true_emits_one_row_per_co_occurring_pair():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="squeeze", expr=_cyclist_and_car(acc))
    rows = event.materialize_per_container(1, _mk_cache(_two_cyclists_one_car(), 2))
    assert _maps(rows) == [
        {"object_tracks": {"fusion": ["47", "91"]}},
        {"object_tracks": {"fusion": ["48", "91"]}},
    ]


def test_per_entity_windowing_false_emits_single_combined_row():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="squeeze", expr=_cyclist_and_car(acc), per_entity_windowing=False)
    rows = event.materialize_per_container(1, _mk_cache(_two_cyclists_one_car(), 2))
    assert _maps(rows) == [{"object_tracks": {"fusion": ["47", "48", "91"]}}]


def test_per_entity_windowing_false_contained_window_keeps_enclosing():
    # With per_entity_windowing=False the per-entity intervals are folded via |.
    # Entity 47 is present at frames 0..4 (window [0,5)); entity 48 only at frame 2
    # (window [2,3), contained). The combined window must enclose, [0,5).
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(
        name="near", expr=(acc.distance_m < 8.0).entity_condition(), per_entity_windowing=False
    )
    frames = _tracks(
        [
            (0, 47, "cyclist", 5.0),
            (1, 47, "cyclist", 5.0),
            (2, 47, "cyclist", 5.0),
            (3, 47, "cyclist", 5.0),
            (4, 47, "cyclist", 5.0),
            (2, 48, "cyclist", 5.0),
        ]
    )
    rows = event.materialize_per_container(1, _mk_cache(frames, 5))
    assert len(rows) == 1
    _cid, start_ts, end_ts, _ek = rows[0]
    assert start_ts == 0.0
    assert end_ts == 5.0
    assert _maps(rows) == [{"object_tracks": {"fusion": ["47", "48"]}}]


def test_two_entities_same_signal_share_one_map_entry():
    # cyclist 47 + car 91, both fusion → {"object_tracks": {"fusion": ["47", "91"]}}
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="squeeze", expr=_cyclist_and_car(acc))
    frames = _tracks(
        [
            (0, 47, "cyclist", 5.0),
            (1, 47, "cyclist", 5.0),
            (0, 91, "car", 5.0),
            (1, 91, "car", 5.0),
        ]
    )
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert _maps(rows) == [{"object_tracks": {"fusion": ["47", "91"]}}]


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


def test_cross_series_co_occurrence_emits_nested_map_across_tables():
    # A cyclist (object_tracks/fusion) close while a 30-sign (traffic_signs/
    # camera_front) is present → the flagship cross-series squeeze case.
    obj = SeriesAccessor(_object_series())
    sign = SeriesAccessor(_sign_series())
    cyclist_close = (
        (obj.detection_class == "cyclist") & (obj.distance_m < 8.0)
    ).entity_condition()
    sign_30 = (sign.sign_class == "speed_30").entity_condition()
    event = EntityEvent(name="squeeze", expr=cyclist_close & sign_30)

    frames = {
        "object_tracks": _frame([(0, 47, "cyclist", 5.0), (1, 47, "cyclist", 5.0)]),
        "traffic_signs": _signs([(0, 12, "speed_30"), (1, 12, "speed_30")]),
    }
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert _maps(rows) == [
        {
            "object_tracks": {"fusion": ["47"]},
            "traffic_signs": {"camera_front": ["12"]},
        }
    ]


def test_cross_series_conjunction_with_empty_other_series_does_not_fire():
    # The sign series has no matching rows for this container → the conjunction
    # cannot fire, and the missing/empty frame must not raise.
    obj = SeriesAccessor(_object_series())
    sign = SeriesAccessor(_sign_series())
    cyclist_close = (obj.distance_m < 8.0).entity_condition()
    sign_30 = (sign.sign_class == "speed_30").entity_condition()
    event = EntityEvent(name="squeeze", expr=cyclist_close & sign_30)

    frames = {
        "object_tracks": _frame([(0, 47, "cyclist", 5.0), (1, 47, "cyclist", 5.0)]),
        "traffic_signs": _signs([]),  # no signs detected
    }
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert rows == []


def test_combined_windowing_non_firing_returns_no_rows():
    # per_entity_windowing=False folds the entity intervals via | before evaluating
    # the tree; when the folded result does not fire, _materialize_combined returns
    # no rows (the combined-windowing counterpart to the per-entity empty case).
    obj = SeriesAccessor(_object_series())
    sign = SeriesAccessor(_sign_series())
    cyclist_close = (obj.distance_m < 8.0).entity_condition()
    sign_30 = (sign.sign_class == "speed_30").entity_condition()
    event = EntityEvent(name="squeeze", expr=cyclist_close & sign_30, per_entity_windowing=False)

    frames = {
        "object_tracks": _frame([(0, 47, "cyclist", 5.0), (1, 47, "cyclist", 5.0)]),
        "traffic_signs": _signs([]),  # no signs detected → conjunction cannot fire
    }
    rows = event.materialize_per_container(1, _mk_cache(frames, 2))
    assert rows == []


def test_cross_series_construction_is_accepted():
    # A two-series expression constructs without error (no single-series guard).
    obj = SeriesAccessor(_object_series())
    sign = SeriesAccessor(_sign_series())
    expr = (obj.distance_m < 8.0).entity_condition() & (
        sign.sign_class == "speed_30"
    ).entity_condition()
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
        # Records that the cache reached build (the channel path), returns a fixed
        # Intervals so the test can assert the value flowed through.
        return self._result


def test_eval_tree_returns_pinned_leaf_from_substitution_map():
    leaf = _PinnedLeaf()
    pinned = Intervals.empty()
    assert _eval_tree(leaf, {id(leaf): pinned}, EmptyTimeSeriesCache()) is pinned


def test_eval_tree_resolves_unpinned_channel_leaf_via_cache_build():
    # The part-E remainder: a non-SeriesSelector leaf no longer raises — it is
    # evaluated against the cache (where the cogroup carries channel rows).
    ivs = Intervals([0.0], [5.0])
    leaf = _ChannelLeaf(ivs)
    assert _eval_tree(leaf, {}, EmptyTimeSeriesCache()) is ivs


def test_eval_tree_memoizes_channel_leaf_build_across_calls():
    # Channel leaves don't vary per entity combination, so a shared build_cache
    # builds each once and reuses it across every combination in the container.
    calls = []

    class _CountingLeaf:
        def build(self, cache):
            calls.append(1)
            return Intervals([0.0], [5.0])

    leaf = _CountingLeaf()
    build_cache: dict = {}
    _eval_tree(leaf, {}, EmptyTimeSeriesCache(), build_cache)
    _eval_tree(leaf, {}, EmptyTimeSeriesCache(), build_cache)
    assert len(calls) == 1  # built once, reused on the second combination


def test_materialize_with_channel_cache_none_uses_empty_channel_cache():
    # Pure registered-series expression: channel_cache defaults to empty and the
    # series leaves still resolve from series_frames.
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="near", expr=(acc.distance_m < 8.0).entity_condition())
    rows = event.materialize_per_container(
        1, _mk_cache(_tracks([(0, 47, "cyclist", 5.0), (1, 47, "cyclist", 5.0)]), 2)
    )
    assert _maps(rows) == [{"object_tracks": {"fusion": ["47"]}}]


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
    event = EntityEvent(name="critical", expr=(acc.severity > 0.7).entity_condition())
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
    assert _maps(rows) == [{"defects": {"vision": ["[10, 1]"]}}]


# --- misc -------------------------------------------------------------------


def test_event_type_string_and_inheritance():
    acc = SeriesAccessor(_object_series())
    event = EntityEvent(name="g", expr=(acc.distance_m < 8.0).entity_condition())
    assert event.get_event_type_str() == "ENTITY_EVENT"
    assert isinstance(event, BasicEvent)


def test_serialize_entity_map_sorts_dedups_and_string_casts():
    payload = _serialize_entity_map(
        {"object_tracks": {"fusion": [91, 47, 47]}, "traffic_signs": {"camera_front": [12]}}
    )
    assert json.loads(payload) == {
        "object_tracks": {"fusion": ["47", "91"]},
        "traffic_signs": {"camera_front": ["12"]},
    }
