"""End-to-end-feel tests against a realistic ADAS object_tracks series.

Two drives with multiple object types co-existing in each frame. Tests
exercise the authoring patterns a customer would actually write:

- ``(class == 1) & (distance < 8)`` per-row fusion (close cyclist), with
  a deliberate mismatch case (far cyclist + close pedestrian at the same
  timestamp) to prove fusion evaluates per-row and not as set
  intersection of independent interval streams.
- Container-scoped near-miss intervals across two cyclists.
- Composition with a hypothetical channel-side ``Intervals`` (ego speed
  window) — the cogroup wiring lives elsewhere; this exercises the
  interval algebra side of composition.
- Per-object ``EntityEvent`` materialization (which cyclist triggered
  each window) with the alias-keyed ``{alias: {signal: [ids]}}`` entity_key map.
- Multi-container isolation — drive 99 has no close cyclists and emits
  no near-miss rows even when drive 42 does.
"""

from __future__ import annotations

import json

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pyspark.sql.types as T
import pytest

from impulse_query_engine.analyze.metadata.time_series_expression import TimeSeriesOp
from impulse_query_engine.analyze.query.solvers.empty_cache import EmptyTimeSeriesCache
from impulse_query_engine.analyze.query.solvers.series_cache import (
    CombinedSeriesCache,
    MultiSeriesCache,
)
from impulse_query_engine.model.series.intervals import Intervals
from impulse_query_engine.surfaces import Series, SeriesAccessor
from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate
from impulse_query_engine.surfaces.series_selector import SeriesSelector
from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.events.entity_event import EntityEvent, _eval_tree


def _mk_cache(series_frames, container_stop_ts, channel_cache=None):
    return CombinedSeriesCache(
        channel_cache or EmptyTimeSeriesCache(),
        series_frames,
        container_stop_ts=container_stop_ts,
    )


# ---------------------------------------------------------------------------
# Series definition + fixtures
# ---------------------------------------------------------------------------

# Class IDs in this synthetic dataset: 1 = cyclist, 2 = car, 3 = pedestrian.
# All rows come from a single fused-perception signal, so the per-(session,
# signal) frame list spans every frame timestamp in the drive.

OBJECT_TRACKS_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("frame_ts", T.LongType(), nullable=False),
        T.StructField("object_id", T.LongType(), nullable=False),
        T.StructField("detection_class", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType()),
        T.StructField("relative_velocity_ms", T.DoubleType()),
    ]
)

OBJECT_TRACKS = Series(
    name="object_tracks",
    schema=OBJECT_TRACKS_SCHEMA,
    session_col="container_id",
    signal_col="sensor_type",
    timestamp_col="frame_ts",
    entity_key="object_id",
)


_TRACK_COLS = [
    "container_id",
    "frame_ts",
    "object_id",
    "detection_class",
    "distance_m",
    "relative_velocity_ms",
]


# Drive 42 — busy intersection segment, 0..10s, ~1 Hz sampling.
DRIVE_42_ROWS = [
    (42, 0_000_000, 47, 1, 22.0, -1.2),
    (42, 1_000_000, 47, 1, 18.5, -1.3),
    (42, 2_000_000, 47, 1, 14.1, -1.4),
    (42, 3_000_000, 47, 1, 10.0, -1.5),
    (42, 4_000_000, 47, 1, 7.2, -1.6),
    (42, 5_000_000, 47, 1, 5.4, -1.5),
    (42, 6_000_000, 47, 1, 4.8, -0.5),
    (42, 7_000_000, 47, 1, 6.1, 0.8),
    (42, 8_000_000, 47, 1, 9.5, 1.2),
    (42, 9_000_000, 47, 1, 13.0, 1.5),
    (42, 0_000_000, 91, 1, 30.0, 1.0),
    (42, 1_000_000, 91, 1, 25.0, 1.1),
    (42, 2_000_000, 91, 1, 20.0, 1.2),
    (42, 3_000_000, 91, 1, 14.0, 1.3),
    (42, 4_000_000, 91, 1, 10.0, 1.4),
    (42, 5_000_000, 91, 1, 7.0, 1.5),
    (42, 6_000_000, 91, 1, 5.0, 1.0),
    (42, 7_000_000, 91, 1, 9.0, 1.2),
    (42, 8_000_000, 91, 1, 13.0, 1.3),
    (42, 9_000_000, 91, 1, 17.0, 1.4),
    (42, 1_000_000, 305, 3, 8.5, -0.3),
    (42, 2_000_000, 305, 3, 5.0, -0.3),
    (42, 3_000_000, 305, 3, 4.2, 0.0),
    (42, 4_000_000, 305, 3, 6.0, 0.5),
    (42, 5_000_000, 305, 3, 10.0, 0.5),
    (42, 0_000_000, 113, 2, 40.0, -0.5),
    (42, 5_000_000, 113, 2, 35.0, -0.6),
    (42, 9_000_000, 113, 2, 30.0, -0.7),
    (42, 3_000_000, 217, 2, 24.0, -0.3),
    (42, 4_000_000, 217, 2, 18.0, -0.4),
    (42, 5_000_000, 217, 2, 12.0, -0.8),
    (42, 6_000_000, 217, 2, 10.0, -0.9),
    (42, 7_000_000, 217, 2, 9.0, -0.4),
]
DRIVE_42_STOP_TS = 10_000_000


# Drive 99 — empty highway, one far car, no cyclists.
DRIVE_99_ROWS = [
    (99, 0_000_000, 401, 2, 60.0, 0.0),
    (99, 1_000_000, 401, 2, 58.0, 0.2),
    (99, 2_000_000, 401, 2, 55.0, 0.3),
    (99, 3_000_000, 401, 2, 52.0, 0.4),
    (99, 4_000_000, 401, 2, 50.0, 0.5),
]
DRIVE_99_STOP_TS = 5_000_000


def _track_df(rows):
    df = pd.DataFrame(rows, columns=_TRACK_COLS)
    df["sensor_type"] = "fusion"  # single fused-perception signal
    return df


@pytest.fixture
def drive_42_cache() -> MultiSeriesCache:
    return MultiSeriesCache(
        {OBJECT_TRACKS.name: _track_df(DRIVE_42_ROWS)},
        container_stop_ts=DRIVE_42_STOP_TS,
    )


@pytest.fixture
def drive_99_cache() -> MultiSeriesCache:
    return MultiSeriesCache(
        {OBJECT_TRACKS.name: _track_df(DRIVE_99_ROWS)},
        container_stop_ts=DRIVE_99_STOP_TS,
    )


@pytest.fixture
def ot() -> SeriesAccessor:
    return SeriesAccessor(OBJECT_TRACKS)


# ---------------------------------------------------------------------------
# Per-row predicate fusion on the same series
# ---------------------------------------------------------------------------


def test_same_series_and_fuses_into_single_partial(ot: SeriesAccessor):
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    assert isinstance(cyclist_close, _PartialPredicate)
    finalized = cyclist_close.each()
    assert isinstance(finalized, SeriesSelector)
    assert len(finalized.get_selectors()) == 1
    assert "detection_class" in cyclist_close.description
    assert "distance_m" in cyclist_close.description


def test_fusion_handles_far_cyclist_with_close_pedestrian_correctly(drive_42_cache, ot):
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    intervals = cyclist_close.build(drive_42_cache)
    # No row is class==1 AND distance<8 before t=4, despite a close pedestrian
    # at t=2..4 — proves per-row fusion, not interval-set intersection.
    assert intervals.tstarts[0] == 4_000_000


# ---------------------------------------------------------------------------
# Container-scoped near-miss window
# ---------------------------------------------------------------------------


def test_close_cyclist_intervals_drive_42(drive_42_cache, ot):
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    intervals = cyclist_close.build(drive_42_cache)
    nptest.assert_array_equal(intervals.tstarts, [4_000_000])
    nptest.assert_array_equal(intervals.tends, [8_000_000])


def test_close_cyclist_intervals_drive_99_is_empty(drive_99_cache, ot):
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    assert len(cyclist_close.build(drive_99_cache)) == 0


# ---------------------------------------------------------------------------
# Composition with channel-side Intervals
# ---------------------------------------------------------------------------


def test_close_cyclist_intersected_with_ego_fast_window(drive_42_cache, ot):
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    row_intervals = cyclist_close.build(drive_42_cache)
    ego_fast = Intervals(np.array([3_500_000.0]), np.array([6_500_000.0]))
    combined = row_intervals & ego_fast
    # [4s, 8s) ∩ [3.5s, 6.5s) = [4s, 6.5s).
    nptest.assert_array_equal(combined.tstarts, [4_000_000])
    nptest.assert_array_equal(combined.tends, [6_500_000])


# ---------------------------------------------------------------------------
# Per-object EntityEvent (which cyclist triggered the window?)
# ---------------------------------------------------------------------------


def _near_miss_per_object(ot):
    return ((ot.detection_class == 1) & (ot.distance_m < 8.0)).each().ids(as_="cyclist")


def test_eval_tree_op_dispatch_matches_build(drive_42_cache, ot):
    # The per-entity evaluator must dispatch ops identically to TimeSeriesOp.build
    # (both route through the shared apply_op). With no pins, walking a composed
    # op tree must equal building it — this is the guard against the two
    # evaluators drifting on optype handling.
    a = (ot.detection_class == 1).each()
    b = (ot.distance_m < 8.0).each()
    expr = a & b
    assert isinstance(expr, TimeSeriesOp)

    built = expr.build(drive_42_cache)
    walked = _eval_tree(expr, {}, drive_42_cache)

    nptest.assert_array_equal(walked.tstarts, built.tstarts)
    nptest.assert_array_equal(walked.tends, built.tends)


def test_eval_tree_routes_udf_through_apply_op(drive_42_cache, ot):
    # A TimeSeriesUDF overrides apply_op; the per-entity evaluator must invoke it
    # exactly as build() does (this is the latent drift the unification fixes —
    # previously _eval_tree never reached TimeSeriesUDF's overridden dispatch).
    leaf = (ot.distance_m < 8.0).each()
    udf = leaf.apply(lambda intervals: intervals)

    built = udf.build(drive_42_cache)
    walked = _eval_tree(udf, {}, drive_42_cache)

    # Both eval paths agree, and the identity UDF returns the leaf's own
    # presence intervals unchanged (non-empty, proving the UDF actually ran).
    leaf_built = leaf.build(drive_42_cache)
    assert len(built) > 0
    nptest.assert_array_equal(built.tstarts, leaf_built.tstarts)
    nptest.assert_array_equal(walked.tstarts, built.tstarts)
    nptest.assert_array_equal(walked.tends, built.tends)


def test_entity_event_emits_one_row_per_cyclist(ot):
    event = EntityEvent(name="cyclist_near_miss_per_object", expr=_near_miss_per_object(ot))
    rows = event.materialize_per_container(42, _mk_cache({"object_tracks": _track_df(DRIVE_42_ROWS)}, DRIVE_42_STOP_TS))
    parsed = sorted((r[0], r[1], r[2], json.loads(r[3])) for r in rows)
    # Cyclist 47 close at [4s, 8s); cyclist 91 close at [5s, 7s).
    assert parsed == [
        (42, 4_000_000.0, 8_000_000.0, {"cyclist": {"fusion": ["47"]}}),
        (42, 5_000_000.0, 7_000_000.0, {"cyclist": {"fusion": ["91"]}}),
    ]


def test_entity_event_yields_no_rows_for_drive_99(ot):
    event = EntityEvent(name="cyclist_near_miss_per_object", expr=_near_miss_per_object(ot))
    rows = event.materialize_per_container(99, _mk_cache({"object_tracks": _track_df(DRIVE_99_ROWS)}, DRIVE_99_STOP_TS))
    assert rows == []


# ---------------------------------------------------------------------------
# Separating signals — close pedestrian vs close cyclist
# ---------------------------------------------------------------------------


def test_close_pedestrian_does_not_match_close_cyclist_predicate(drive_42_cache, ot):
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    pedestrian_close = (ot.detection_class == 3) & (ot.distance_m < 8.0)
    cyclist_intervals = cyclist_close.build(drive_42_cache)
    pedestrian_intervals = pedestrian_close.build(drive_42_cache)
    nptest.assert_array_equal(cyclist_intervals.tstarts, [4_000_000])
    nptest.assert_array_equal(pedestrian_intervals.tstarts, [2_000_000])
    nptest.assert_array_equal(pedestrian_intervals.tends, [5_000_000])


# ---------------------------------------------------------------------------
# Cross-leaf-kind composition stays an op (not fused)
# ---------------------------------------------------------------------------


def test_series_predicate_does_not_fuse_across_different_leaf_kinds(ot):
    lane_schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("sensor_type", T.StringType(), nullable=False),
            T.StructField("ts", T.LongType(), nullable=False),
            T.StructField("lane_id", T.LongType(), nullable=False),
            T.StructField("offset_m", T.DoubleType()),
        ]
    )
    lanes = Series(
        name="lane_offsets",
        schema=lane_schema,
        session_col="container_id",
        signal_col="sensor_type",
        timestamp_col="ts",
        entity_key="lane_id",
    )
    la = SeriesAccessor(lanes)

    composed = (ot.detection_class == 1) & (la.offset_m > 0.5)
    assert isinstance(composed, TimeSeriesOp)
    leaf_kinds = {leaf.leaf_kind for leaf in composed.get_selectors()}
    assert leaf_kinds == {"object_tracks", "lane_offsets"}


def test_three_clause_fusion_on_same_series_keeps_one_partial(ot):
    sharply_approaching_cyclist = (
        (ot.detection_class == 1) & (ot.distance_m < 8.0) & (ot.relative_velocity_ms < -1.0)
    )
    assert isinstance(sharply_approaching_cyclist, _PartialPredicate)
    assert len(sharply_approaching_cyclist.each().get_selectors()) == 1


def test_sharply_approaching_cyclist_window_excludes_drifting_rows(drive_42_cache, ot):
    expr = (ot.detection_class == 1) & (ot.distance_m < 8.0) & (ot.relative_velocity_ms < -1.0)
    intervals = expr.build(drive_42_cache)
    # Cyclist 47 matches at t=4 (-1.6) and t=5 (-1.5); closes at t=6 → [4s, 6s).
    nptest.assert_array_equal(intervals.tstarts, [4_000_000])
    nptest.assert_array_equal(intervals.tends, [6_000_000])


# ---------------------------------------------------------------------------
# Cross-entity correlation via .each()
# ---------------------------------------------------------------------------


def _cyclist_close(ot):
    return ((ot.detection_class == 1) & (ot.distance_m < 8.0)).any()


def _car_decel_close(ot):
    return (
        (ot.detection_class == 2) & (ot.distance_m < 15.0) & (ot.relative_velocity_ms < -0.5)
    ).any()


def test_cross_entity_correlation_needs_a_finalizing_verb_on_each_side(drive_42_cache, ot):
    cyclist_close = _cyclist_close(ot)
    car_decel_close = _car_decel_close(ot)

    nptest.assert_array_equal(cyclist_close.build(drive_42_cache).tstarts, [4_000_000])
    nptest.assert_array_equal(cyclist_close.build(drive_42_cache).tends, [8_000_000])
    nptest.assert_array_equal(car_decel_close.build(drive_42_cache).tstarts, [5_000_000])
    nptest.assert_array_equal(car_decel_close.build(drive_42_cache).tends, [7_000_000])

    intervals = (cyclist_close & car_decel_close).build(drive_42_cache)
    # [4,8) ∩ [5,7) = [5,7).
    nptest.assert_array_equal(intervals.tstarts, [5_000_000])
    nptest.assert_array_equal(intervals.tends, [7_000_000])


def test_same_series_predicates_fuse_per_row_so_cross_class_matches_nothing(drive_42_cache, ot):
    fused = ((ot.detection_class == 1) & (ot.distance_m < 8.0)) & (
        (ot.detection_class == 2) & (ot.distance_m < 15.0) & (ot.relative_velocity_ms < -0.5)
    )
    assert len(fused.build(drive_42_cache)) == 0


def test_basic_event_any_finalizes_as_presence(drive_42_cache, ot):
    event = BasicEvent(
        name="cyclist_near_miss",
        expr=((ot.detection_class == 1) & (ot.distance_m < 8.0)).any(),
    )
    assert isinstance(event.expression, SeriesSelector)
    intervals = event.expression.build(drive_42_cache)
    nptest.assert_array_equal(intervals.tstarts, [4_000_000])
    nptest.assert_array_equal(intervals.tends, [8_000_000])


def test_cross_entity_event_composes_to_one_window(drive_42_cache, ot):
    event = BasicEvent(
        name="cyclist_with_decelerating_car",
        expr=_cyclist_close(ot) & _car_decel_close(ot),
    )
    intervals = event.expression.build(drive_42_cache)
    nptest.assert_array_equal(intervals.tstarts, [5_000_000])
    nptest.assert_array_equal(intervals.tends, [7_000_000])


def test_cross_entity_correlation_yields_nothing_on_drive_without_car(drive_99_cache, ot):
    co_occurrence = _cyclist_close(ot) & _car_decel_close(ot)
    assert len(co_occurrence.build(drive_99_cache)) == 0


def test_cross_entity_entity_event_maps_both_objects(ot):
    """EntityEvent over the cross-entity correlation: the single merged window
    names both the cyclist roster and the decelerating-car roster, keyed by
    their aliases."""
    event = EntityEvent(
        name="cyclist_with_decelerating_car",
        expr=_cyclist_close(ot).ids(as_="cyclist") & _car_decel_close(ot).ids(as_="car"),
    )
    rows = event.materialize_per_container(42, _mk_cache({"object_tracks": _track_df(DRIVE_42_ROWS)}, DRIVE_42_STOP_TS))
    # cyclist presence (47:[4,8) ∪ 91:[5,7)) ∩ car presence (217:[5,7)) = [5,7).
    # Both cyclists overlap that window, so the roster lists both.
    assert [(r[1], r[2], json.loads(r[3])) for r in rows] == [
        (5_000_000.0, 7_000_000.0, {"car": {"fusion": ["217"]}, "cyclist": {"fusion": ["47", "91"]}}),
    ]
