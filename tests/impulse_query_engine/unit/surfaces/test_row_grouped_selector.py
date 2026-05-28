"""Selector evaluation: per-group interval synthesis, predicate, merge,
composition with other Intervals via ``&`` / ``|``."""

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pyspark.sql.types as T

from impulse_query_engine.analyze.query.solvers.series_cache import MultiSurfaceCache
from impulse_query_engine.model.series.intervals import Intervals
from impulse_query_engine.surfaces import RowGroupedAccessor, RowGroupedSurface


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
        name="object_tracks",
        schema=schema,
        timestamp_col="ts",
        group_col=group_col,
    )


def _frame(rows):
    return pd.DataFrame(
        rows,
        columns=["container_id", "ts", "object_id", "distance_m"],
    )


def test_per_group_interval_synthesis_closes_with_container_stop_ts():
    # object 1: distance < 8 at t=0,1; distance = 12 at t=2 (mask off)
    # object 2: distance = 5 at t=0; distance = 10 at t=1 (off); distance = 4 at t=2 (on)
    rows = [
        (1, 0, 1, 5.0),
        (1, 1, 1, 7.0),
        (1, 2, 1, 12.0),
        (1, 0, 2, 5.0),
        (1, 1, 2, 10.0),
        (1, 2, 2, 4.0),
    ]
    surface = _object_surface()
    accessor = RowGroupedAccessor(surface)
    cache = MultiSurfaceCache({"object_tracks": _frame(rows)}, container_stop_ts=3)

    intervals = (accessor.distance_m < 8.0).build(cache)
    # Object 1: rows at t=0..1 are "on"; intervals [0,1) and [1,2), merged to [0,2).
    # Object 2: rows at t=0 on, t=1 off, t=2 on; intervals [0,1) and [2,3).
    # Union and merge across groups: [0,2) ∪ [0,1) ∪ [2,3) = [0,3).
    nptest.assert_array_equal(intervals.tstarts, [0])
    nptest.assert_array_equal(intervals.tends, [3])


def test_disjoint_groups_remain_disjoint_when_not_overlapping():
    # Object 1 matches only at t=0; object 2 matches only at t=3.
    # Object 1 row at t=0 closes at t=1 (next row in group), so the
    # interval is [0,1). Object 2's t=3 row is the last in its group, so
    # it closes at container_stop_ts=4 — interval [3,4). A gap of (1,3)
    # separates them so merge_overlaps must not fuse them.
    rows = [
        (1, 0, 1, 5.0),
        (1, 1, 1, 99.0),
        (1, 2, 1, 99.0),
        (1, 3, 1, 99.0),
        (1, 0, 2, 99.0),
        (1, 1, 2, 99.0),
        (1, 2, 2, 99.0),
        (1, 3, 2, 5.0),
    ]
    surface = _object_surface()
    accessor = RowGroupedAccessor(surface)
    cache = MultiSurfaceCache({"object_tracks": _frame(rows)}, container_stop_ts=4)

    intervals = (accessor.distance_m < 8.0).build(cache)
    nptest.assert_array_equal(intervals.tstarts, [0, 3])
    nptest.assert_array_equal(intervals.tends, [1, 4])


def test_no_matching_rows_returns_empty_intervals():
    rows = [(1, 0, 1, 99.0), (1, 1, 1, 99.0)]
    surface = _object_surface()
    accessor = RowGroupedAccessor(surface)
    cache = MultiSurfaceCache({"object_tracks": _frame(rows)}, container_stop_ts=2)

    intervals = (accessor.distance_m < 8.0).build(cache)
    assert len(intervals) == 0


def test_missing_surface_in_cache_returns_empty():
    surface = _object_surface()
    accessor = RowGroupedAccessor(surface)
    cache = MultiSurfaceCache({}, container_stop_ts=2)

    intervals = (accessor.distance_m < 8.0).build(cache)
    assert len(intervals) == 0


def test_no_group_col_treats_surface_as_single_timeline():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("ts", T.LongType(), nullable=False),
            T.StructField("magnitude", T.DoubleType()),
        ]
    )
    surface = RowGroupedSurface(
        name="ungrouped", schema=schema, timestamp_col="ts", group_col=None
    )
    df = pd.DataFrame(
        [(1, 0, 5.0), (1, 1, 9.0), (1, 2, 4.0)],
        columns=["container_id", "ts", "magnitude"],
    )
    cache = MultiSurfaceCache({"ungrouped": df}, container_stop_ts=3)
    accessor = RowGroupedAccessor(surface)

    intervals = (accessor.magnitude < 8.0).build(cache)
    # t=0 on, t=1 off, t=2 on. Intervals [0,1) and [2,3).
    nptest.assert_array_equal(intervals.tstarts, [0, 2])
    nptest.assert_array_equal(intervals.tends, [1, 3])


def test_composes_with_external_intervals_via_and():
    # Row-grouped predicate returns Intervals; an externally produced
    # Intervals (think: built from a channel predicate) composes via &.
    rows = [(1, 0, 1, 5.0), (1, 1, 1, 5.0), (1, 2, 1, 5.0)]
    surface = _object_surface()
    accessor = RowGroupedAccessor(surface)
    cache = MultiSurfaceCache({"object_tracks": _frame(rows)}, container_stop_ts=3)

    row_grouped = (accessor.distance_m < 8.0).build(cache)
    channel_window = Intervals(np.array([1.0]), np.array([5.0]))
    composed = row_grouped & channel_window
    # Row-grouped covers [0,3); channel window covers [1,5). Intersection [1,3).
    nptest.assert_array_equal(composed.tstarts, [1])
    nptest.assert_array_equal(composed.tends, [3])


def test_selector_leaf_kind_is_surface_name():
    surface = _object_surface()
    accessor = RowGroupedAccessor(surface)
    sel = accessor.distance_m < 8.0
    assert sel.leaf_kind == "object_tracks"
