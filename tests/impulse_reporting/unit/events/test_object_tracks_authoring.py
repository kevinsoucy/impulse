"""End-to-end-feel tests against a realistic ADAS object_tracks surface.

Two drives with multiple object types co-existing in each frame. Tests
exercise the authoring patterns a customer would actually write:

- ``(class == 1) & (distance < 8)`` per-row fusion (close cyclist), with
  a deliberate mismatch case (far cyclist + close pedestrian at the same
  timestamp) to prove fusion evaluates per-row and not as set
  intersection of independent interval streams.
- Container-scoped near-miss intervals across two cyclists.
- Composition with a hypothetical channel-side ``Intervals`` (ego speed
  window) — the cogroup wiring is the follow-up; this exercises the
  interval algebra side of composition.
- Per-object ``GroupedEvent`` materialization (which cyclist triggered
  each window).
- Multi-container isolation — drive 99 has no close cyclists and emits
  no near-miss rows even when drive 42 does.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pyspark.sql.types as T
import pytest

from impulse_query_engine.analyze.metadata.time_series_expression import TimeSeriesOp
from impulse_query_engine.analyze.query.solvers.series_cache import MultiSurfaceCache
from impulse_query_engine.model.series.intervals import Intervals
from impulse_query_engine.surfaces import RowGroupedAccessor, RowGroupedSurface
from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate
from impulse_query_engine.surfaces.row_grouped_selector import RowGroupedSelector
from impulse_reporting.events.grouped_event import GroupedEvent


# ---------------------------------------------------------------------------
# Surface definition + fixtures
# ---------------------------------------------------------------------------

# Class IDs in this synthetic dataset:
#   1 = cyclist
#   2 = car
#   3 = pedestrian
# (Strings would be more natural; ADR-001 is the follow-up that adds string
# columns to scalar channels. Numeric class IDs keep this minimal branch
# honest about what's wired up today.)

OBJECT_TRACKS_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("frame_ts", T.LongType(), nullable=False),
        T.StructField("object_id", T.LongType(), nullable=False),
        T.StructField("detection_class", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType()),
        T.StructField("relative_velocity_ms", T.DoubleType()),
    ]
)

OBJECT_TRACKS = RowGroupedSurface(
    name="object_tracks",
    schema=OBJECT_TRACKS_SCHEMA,
    timestamp_col="frame_ts",
    group_col="object_id",
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
# Two cyclists, one car, one pedestrian. The pedestrian at t=2,3,4 is the
# critical mismatch case: it's close (distance < 8) but NOT a cyclist, while
# the cyclists at those same timestamps are still far. A naive
# `(class==cyclist) & (distance<8)` composition that built independent
# Intervals would incorrectly mark t=2..4 as a close-cyclist window because
# "a cyclist exists" overlaps with "a close object exists" — even though no
# row has both conditions simultaneously.
DRIVE_42_ROWS = [
    # Cyclist 47 — approaches from sidewalk, within 8m at t=4..7
    (42, 0_000_000, 47, 1, 22.0, -1.2),
    (42, 1_000_000, 47, 1, 18.5, -1.3),
    (42, 2_000_000, 47, 1, 14.1, -1.4),
    (42, 3_000_000, 47, 1, 10.0, -1.5),
    (42, 4_000_000, 47, 1,  7.2, -1.6),
    (42, 5_000_000, 47, 1,  5.4, -1.5),
    (42, 6_000_000, 47, 1,  4.8, -0.5),
    (42, 7_000_000, 47, 1,  6.1,  0.8),
    (42, 8_000_000, 47, 1,  9.5,  1.2),
    (42, 9_000_000, 47, 1, 13.0,  1.5),
    # Cyclist 91 — opposite direction, within 8m at t=5..6
    (42, 0_000_000, 91, 1, 30.0,  1.0),
    (42, 1_000_000, 91, 1, 25.0,  1.1),
    (42, 2_000_000, 91, 1, 20.0,  1.2),
    (42, 3_000_000, 91, 1, 14.0,  1.3),
    (42, 4_000_000, 91, 1, 10.0,  1.4),
    (42, 5_000_000, 91, 1,  7.0,  1.5),
    (42, 6_000_000, 91, 1,  5.0,  1.0),
    (42, 7_000_000, 91, 1,  9.0,  1.2),
    (42, 8_000_000, 91, 1, 13.0,  1.3),
    (42, 9_000_000, 91, 1, 17.0,  1.4),
    # Pedestrian 305 — close at t=2..4, but NOT a cyclist. This is the
    # mismatch case the per-row fusion must handle correctly.
    (42, 1_000_000, 305, 3, 8.5, -0.3),
    (42, 2_000_000, 305, 3, 5.0, -0.3),
    (42, 3_000_000, 305, 3, 4.2,  0.0),
    (42, 4_000_000, 305, 3, 6.0,  0.5),
    (42, 5_000_000, 305, 3, 10.0, 0.5),
    # Car 113 — passes in adjacent lane, never close
    (42, 0_000_000, 113, 2, 40.0, -0.5),
    (42, 5_000_000, 113, 2, 35.0, -0.6),
    (42, 9_000_000, 113, 2, 30.0, -0.7),
    # Car 217 — oncoming car, sharply decelerating close-in at t=5..6.
    # This is the cross-entity correlation target: it overlaps in time
    # with cyclist 47/91's close window so a co-occurrence event can fire.
    (42, 3_000_000, 217, 2, 24.0, -0.3),
    (42, 4_000_000, 217, 2, 18.0, -0.4),
    (42, 5_000_000, 217, 2, 12.0, -0.8),
    (42, 6_000_000, 217, 2, 10.0, -0.9),
    (42, 7_000_000, 217, 2,  9.0, -0.4),
]
DRIVE_42_STOP_TS = 10_000_000


# Drive 99 — empty highway, 0..5s. One car staying far throughout, no
# cyclists at all. Used to verify the predicate emits nothing for drives
# without matching objects.
DRIVE_99_ROWS = [
    (99, 0_000_000, 401, 2, 60.0, 0.0),
    (99, 1_000_000, 401, 2, 58.0, 0.2),
    (99, 2_000_000, 401, 2, 55.0, 0.3),
    (99, 3_000_000, 401, 2, 52.0, 0.4),
    (99, 4_000_000, 401, 2, 50.0, 0.5),
]
DRIVE_99_STOP_TS = 5_000_000


@pytest.fixture
def drive_42_cache() -> MultiSurfaceCache:
    df = pd.DataFrame(DRIVE_42_ROWS, columns=_TRACK_COLS)
    return MultiSurfaceCache(
        {OBJECT_TRACKS.name: df}, container_stop_ts=DRIVE_42_STOP_TS
    )


@pytest.fixture
def drive_99_cache() -> MultiSurfaceCache:
    df = pd.DataFrame(DRIVE_99_ROWS, columns=_TRACK_COLS)
    return MultiSurfaceCache(
        {OBJECT_TRACKS.name: df}, container_stop_ts=DRIVE_99_STOP_TS
    )


@pytest.fixture
def ot() -> RowGroupedAccessor:
    return RowGroupedAccessor(OBJECT_TRACKS)


# ---------------------------------------------------------------------------
# Authoring pattern: per-row predicate fusion on the same surface
# ---------------------------------------------------------------------------


def test_same_surface_and_fuses_into_single_partial(ot: RowGroupedAccessor):
    """The fusion path keeps the expression as ONE _PartialPredicate (not a
    TimeSeriesOp tree). This is the invariant that makes same-entity
    per-row semantics correct. ``.sub_event()`` then finalizes the
    partial into a single ``RowGroupedSelector`` leaf."""
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    assert isinstance(cyclist_close, _PartialPredicate)
    finalized = cyclist_close.sub_event()
    assert isinstance(finalized, RowGroupedSelector)
    leaves = finalized.get_selectors()
    assert len(leaves) == 1
    # The compound description records both clauses, which keeps the
    # selector_id stable across runs and surfaces the structure in errors.
    assert "detection_class" in cyclist_close.description
    assert "distance_m" in cyclist_close.description


def test_fusion_handles_far_cyclist_with_close_pedestrian_correctly(
    drive_42_cache, ot
):
    """At t=2..4 there's a close pedestrian (class 3) AND far cyclists
    (class 1) — no row has class==1 AND distance<8 simultaneously. The
    fused predicate must emit nothing for that window, even though a
    naive interval-set intersection would (incorrectly) include it
    because 'cyclist exists' overlaps with 'close object exists'."""
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    intervals = cyclist_close.build(drive_42_cache)
    # First close-cyclist window starts at t=4 (cyclist 47 enters <8m
    # range). If fusion were broken, the result would start at t=2 due to
    # pedestrian 305's proximity.
    assert intervals.tstarts[0] == 4_000_000


# ---------------------------------------------------------------------------
# Authoring pattern: container-scoped near-miss window
# ---------------------------------------------------------------------------


def test_close_cyclist_intervals_drive_42(drive_42_cache, ot):
    """Drive 42: cyclist 47 close at t=4..7, cyclist 91 close at t=5..6.
    The per-group interval synthesis closes each cyclist's last matching
    row at the next sample within that group. Across groups the windows
    overlap and merge to a single [4s, 8s) interval."""
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    intervals = cyclist_close.build(drive_42_cache)
    nptest.assert_array_equal(intervals.tstarts, [4_000_000])
    nptest.assert_array_equal(intervals.tends, [8_000_000])


def test_close_cyclist_intervals_drive_99_is_empty(drive_99_cache, ot):
    """Drive 99 has no cyclists at all; the predicate emits nothing."""
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    intervals = cyclist_close.build(drive_99_cache)
    assert len(intervals) == 0


# ---------------------------------------------------------------------------
# Authoring pattern: composition with channel-side Intervals
# ---------------------------------------------------------------------------


def test_close_cyclist_intersected_with_ego_fast_window(drive_42_cache, ot):
    """The classic channel + row-grouped composition. The channel side
    is mocked as a pre-built Intervals (the full cogroup wiring is the
    BL-003 follow-up); the row-grouped side returns Intervals from the
    cache, and `&` composes them via the standard interval algebra."""
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    row_intervals = cyclist_close.build(drive_42_cache)
    # ego_fast: speed > 30 km/h between t=3.5s and t=6.5s. In a real run
    # this would come from `(veh_speed > 30).build(channel_cache)`.
    ego_fast = Intervals(np.array([3_500_000.0]), np.array([6_500_000.0]))
    combined = row_intervals & ego_fast
    # Intersection: [4s, 8s) ∩ [3.5s, 6.5s) = [4s, 6.5s).
    nptest.assert_array_equal(combined.tstarts, [4_000_000])
    nptest.assert_array_equal(combined.tends, [6_500_000])


# ---------------------------------------------------------------------------
# Authoring pattern: per-object GroupedEvent (which cyclist triggered the window?)
# ---------------------------------------------------------------------------


def test_grouped_event_emits_one_row_per_cyclist(drive_42_cache, ot):
    """Per-object scope: each cyclist's window emits its own row with the
    cyclist's object_id materialized into ``group_value``."""
    expr = (ot(group_scope=True).detection_class == 1) & (
        ot(group_scope=True).distance_m < 8.0
    )
    event = GroupedEvent(name="cyclist_near_miss_per_object", expr=expr)
    df = pd.DataFrame(DRIVE_42_ROWS, columns=_TRACK_COLS)
    rows = event.materialize_per_container(
        container_id=42, surface_df=df, container_stop_ts=DRIVE_42_STOP_TS
    )
    # Cyclist 47 close at t=4..7 with last row closing at t=8 → [4s, 8s).
    # Cyclist 91 close at t=5..6 with last row closing at t=7 → [5s, 7s).
    # Pedestrian 305 and car 113 never satisfy class==1, so no rows from
    # those groups.
    assert sorted(rows) == [
        (42, 4_000_000.0, 8_000_000.0, "47"),
        (42, 5_000_000.0, 7_000_000.0, "91"),
    ]


def test_grouped_event_yields_no_rows_for_drive_99(ot):
    """Multi-container isolation: the same event, run against drive 99's
    rows, emits nothing."""
    expr = (ot(group_scope=True).detection_class == 1) & (
        ot(group_scope=True).distance_m < 8.0
    )
    event = GroupedEvent(name="cyclist_near_miss_per_object", expr=expr)
    df = pd.DataFrame(DRIVE_99_ROWS, columns=_TRACK_COLS)
    rows = event.materialize_per_container(
        container_id=99, surface_df=df, container_stop_ts=DRIVE_99_STOP_TS
    )
    assert rows == []


# ---------------------------------------------------------------------------
# Authoring pattern: separating signals — close pedestrian vs close cyclist
# ---------------------------------------------------------------------------


def test_close_pedestrian_does_not_match_close_cyclist_predicate(
    drive_42_cache, ot
):
    """A close pedestrian is its own event class. The cyclist predicate
    rejects pedestrian rows even when they're closer than the threshold."""
    cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
    pedestrian_close = (ot.detection_class == 3) & (ot.distance_m < 8.0)
    cyclist_intervals = cyclist_close.build(drive_42_cache)
    pedestrian_intervals = pedestrian_close.build(drive_42_cache)
    # Cyclist [4s, 8s). Pedestrian 305: close at t=2..4 (rows t=2,3,4
    # match; t=5 dist=10 doesn't), per-group shift produces intervals
    # [2,3), [3,4), [4,5) → merged [2s, 5s).
    nptest.assert_array_equal(cyclist_intervals.tstarts, [4_000_000])
    nptest.assert_array_equal(pedestrian_intervals.tstarts, [2_000_000])
    nptest.assert_array_equal(pedestrian_intervals.tends, [5_000_000])


# ---------------------------------------------------------------------------
# Authoring pattern: cross-leaf-kind composition stays an op (not fused)
# ---------------------------------------------------------------------------


def test_surface_predicate_does_not_fuse_across_different_leaf_kinds(ot):
    """Cross-surface composition (or surface + channel) must compose at
    the interval level via TimeSeriesOp, NOT fuse per-row. Today this is
    exercised against a fresh accessor on a different surface — different
    leaf_kind, falls through to TimeSeriesOp."""
    lane_schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("ts", T.LongType(), nullable=False),
            T.StructField("lane_id", T.LongType(), nullable=False),
            T.StructField("offset_m", T.DoubleType()),
        ]
    )
    lanes = RowGroupedSurface(
        name="lane_offsets", schema=lane_schema, timestamp_col="ts", group_col="lane_id"
    )
    la = RowGroupedAccessor(lanes)

    composed = (ot.detection_class == 1) & (la.offset_m > 0.5)
    assert isinstance(composed, TimeSeriesOp)
    leaves = composed.get_selectors()
    leaf_kinds = {leaf.leaf_kind for leaf in leaves}
    assert leaf_kinds == {"object_tracks", "lane_offsets"}


def test_three_clause_fusion_on_same_surface_keeps_one_partial(ot):
    """``(A) & (B) & (C)`` on the same surface fuses left-to-right into
    one ``_PartialPredicate``. A "sharply approaching cyclist" predicate
    is a good realistic three-clause case."""
    sharply_approaching_cyclist = (
        (ot.detection_class == 1)
        & (ot.distance_m < 8.0)
        & (ot.relative_velocity_ms < -1.0)
    )
    assert isinstance(sharply_approaching_cyclist, _PartialPredicate)
    finalized = sharply_approaching_cyclist.sub_event()
    assert len(finalized.get_selectors()) == 1


def test_sharply_approaching_cyclist_window_excludes_drifting_rows(
    drive_42_cache, ot
):
    """Realistic 3-clause predicate: a close cyclist who is also closing
    sharply (rel_vel < -1.0 m/s). Cyclist 47 enters <8m at t=4 with rel_vel
    -1.6, stays sharply approaching through t=5 (-1.5), then slows to
    -0.5 at t=6 and drifts to positive after. Cyclist 91 is always
    receding (positive rel_vel), so never matches."""
    expr = (
        (ot.detection_class == 1)
        & (ot.distance_m < 8.0)
        & (ot.relative_velocity_ms < -1.0)
    )
    intervals = expr.build(drive_42_cache)
    # Cyclist 47 matching rows at t=4 (rel_vel -1.6) and t=5 (rel_vel -1.5).
    # Per-group shift(-1): t=4→5, t=5→6. Intervals [4,5), [5,6) → [4s, 6s).
    nptest.assert_array_equal(intervals.tstarts, [4_000_000])
    nptest.assert_array_equal(intervals.tends, [6_000_000])


# ---------------------------------------------------------------------------
# Cross-entity correlation via .sub_event()
# ---------------------------------------------------------------------------


def test_cross_entity_correlation_needs_sub_event_on_each_side(
    drive_42_cache, ot
):
    """The design doc's headline use case: 'a cyclist within 8m while an
    oncoming car decelerates sharply within 15m'. The two sides match
    DIFFERENT objects (a cyclist and a car), so no single row satisfies
    both — per-row fusion would (correctly, for same-entity) emit nothing.

    Calling ``.sub_event()`` on each side flips the composition mode: each
    partial finalizes into a container-scope ``Intervals``, and ``&``
    intersects the two interval streams in time."""
    cyclist_close = (
        (ot.detection_class == 1) & (ot.distance_m < 8.0)
    ).sub_event()
    car_decel_close = (
        (ot.detection_class == 2)
        & (ot.distance_m < 15.0)
        & (ot.relative_velocity_ms < -0.5)
    ).sub_event()

    # Each side reduces to its own container-scope window.
    nptest.assert_array_equal(cyclist_close.build(drive_42_cache).tstarts, [4_000_000])
    nptest.assert_array_equal(cyclist_close.build(drive_42_cache).tends, [8_000_000])
    nptest.assert_array_equal(car_decel_close.build(drive_42_cache).tstarts, [5_000_000])
    nptest.assert_array_equal(car_decel_close.build(drive_42_cache).tends, [7_000_000])

    co_occurrence = cyclist_close & car_decel_close
    intervals = co_occurrence.build(drive_42_cache)
    # Overlap: [4,8) ∩ [5,7) = [5,7).
    nptest.assert_array_equal(intervals.tstarts, [5_000_000])
    nptest.assert_array_equal(intervals.tends, [7_000_000])


def test_without_sub_event_cross_entity_predicates_still_fuse_per_row(
    drive_42_cache, ot
):
    """Symmetric to the .sub_event() test: when the user forgets the
    boundary, the partials fuse and the result is empty because no row is
    both class==1 AND class==2. This is correct per-row semantics — the
    fix is to call .sub_event(), not change the operator."""
    fused = ((ot.detection_class == 1) & (ot.distance_m < 8.0)) & (
        (ot.detection_class == 2)
        & (ot.distance_m < 15.0)
        & (ot.relative_velocity_ms < -0.5)
    )
    intervals = fused.build(drive_42_cache)
    assert len(intervals) == 0


def test_basic_event_auto_finalizes_partial_without_sub_event(
    drive_42_cache, ot
):
    """Ergonomic check: a single-predicate event doesn't need explicit
    .sub_event(). BasicEvent's constructor auto-finalizes a partial so
    the common authoring path stays terse."""
    from impulse_reporting.events.basic_event import BasicEvent

    event = BasicEvent(
        name="cyclist_near_miss",
        expr=(ot.detection_class == 1) & (ot.distance_m < 8.0),
    )
    # After construction, the stored expression is a finalized
    # RowGroupedSelector ready for the solver.
    from impulse_query_engine.surfaces.row_grouped_selector import (
        RowGroupedSelector,
    )

    assert isinstance(event.expression, RowGroupedSelector)
    intervals = event.expression.build(drive_42_cache)
    nptest.assert_array_equal(intervals.tstarts, [4_000_000])
    nptest.assert_array_equal(intervals.tends, [8_000_000])


def test_cross_entity_event_composes_to_one_window(drive_42_cache, ot):
    """End-to-end: wrap the cross-entity correlation in a BasicEvent and
    confirm the materialized window matches the cyclist∩car overlap."""
    from impulse_reporting.events.basic_event import BasicEvent

    cyclist_close = (
        (ot.detection_class == 1) & (ot.distance_m < 8.0)
    ).sub_event()
    car_decel_close = (
        (ot.detection_class == 2)
        & (ot.distance_m < 15.0)
        & (ot.relative_velocity_ms < -0.5)
    ).sub_event()

    event = BasicEvent(
        name="cyclist_with_decelerating_car",
        expr=cyclist_close & car_decel_close,
    )
    intervals = event.expression.build(drive_42_cache)
    nptest.assert_array_equal(intervals.tstarts, [5_000_000])
    nptest.assert_array_equal(intervals.tends, [7_000_000])


def test_cross_entity_correlation_yields_nothing_on_drive_without_car(
    drive_99_cache, ot
):
    """Drive 99 has neither cyclists nor matching cars — cross-entity
    correlation produces nothing, confirming both sides need actual
    matches."""
    cyclist_close = (
        (ot.detection_class == 1) & (ot.distance_m < 8.0)
    ).sub_event()
    car_decel_close = (
        (ot.detection_class == 2)
        & (ot.distance_m < 15.0)
        & (ot.relative_velocity_ms < -0.5)
    ).sub_event()

    co_occurrence = cyclist_close & car_decel_close
    assert len(co_occurrence.build(drive_99_cache)) == 0
