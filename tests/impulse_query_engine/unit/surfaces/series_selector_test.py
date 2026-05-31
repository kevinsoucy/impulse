"""Selector evaluation: signal-frame-list / RLE interval synthesis, predicate,
merge, per-(signal, entity) attribution, composition via ``&`` / ``|``."""

import numpy as np
import numpy.testing as nptest
import pandas as pd
import pyspark.sql.types as T

from impulse_query_engine.analyze.query.solvers.series_cache import MultiSeriesCache
from impulse_query_engine.model.series.intervals import Intervals
from impulse_query_engine.surfaces import Series, SeriesAccessor


def _object_series(entity_key="object_id"):
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
        entity_key=entity_key,
    )


def _frame(rows, signal="fusion"):
    return pd.DataFrame(
        [(c, signal, ts, oid, d) for (c, ts, oid, d) in rows],
        columns=["container_id", "sensor_type", "ts", "object_id", "distance_m"],
    )


def _ivs(iv):
    return list(zip([int(t) for t in iv.tstarts], [int(t) for t in iv.tends], strict=False))


def test_presence_build_unions_matching_rows():
    rows = [
        (1, 0, 1, 5.0),
        (1, 1, 1, 7.0),
        (1, 2, 1, 12.0),
        (1, 0, 2, 5.0),
        (1, 1, 2, 10.0),
        (1, 2, 2, 4.0),
    ]
    accessor = SeriesAccessor(_object_series())
    cache = MultiSeriesCache({"object_tracks": _frame(rows)}, container_stop_ts=3)
    intervals = (accessor.distance_m < 8.0).build(cache)
    # frame list {0,1,2}; matching rows synthesize [0,1),[1,2),[2,3) → merged [0,3).
    nptest.assert_array_equal(intervals.tstarts, [0])
    nptest.assert_array_equal(intervals.tends, [3])


def test_disjoint_windows_remain_disjoint():
    rows = [
        (1, 0, 1, 5.0),
        (1, 1, 1, 99.0),
        (1, 2, 1, 99.0),
        (1, 3, 1, 99.0),
        (1, 3, 2, 5.0),
    ]
    accessor = SeriesAccessor(_object_series())
    cache = MultiSeriesCache({"object_tracks": _frame(rows)}, container_stop_ts=4)
    intervals = (accessor.distance_m < 8.0).build(cache)
    # frame list {0,1,2,3}; match at t=0 → [0,1); match at t=3 (last) → [3,4).
    nptest.assert_array_equal(intervals.tstarts, [0, 3])
    nptest.assert_array_equal(intervals.tends, [1, 4])


def test_point_in_time_without_stop_ts_drops_final_frame():
    # No container_stop_ts: the last frame has no next tick to close against, so it
    # collapses to a zero-length interval and is dropped by del_last_empty. Earlier
    # frames still close at their next tick.
    rows = [(1, 0, 1, 5.0), (1, 1, 1, 5.0)]  # match at frames 0 and 1
    accessor = SeriesAccessor(_object_series())
    cache = MultiSeriesCache({"object_tracks": _frame(rows)})  # container_stop_ts defaults to None
    intervals = (accessor.distance_m < 8.0).build(cache)
    # frame 0 closes at frame 1 → [0,1); frame 1 is last with no stop_ts → dropped.
    assert _ivs(intervals) == [(0, 1)]


def test_no_matching_rows_returns_empty():
    rows = [(1, 0, 1, 99.0), (1, 1, 1, 99.0)]
    accessor = SeriesAccessor(_object_series())
    cache = MultiSeriesCache({"object_tracks": _frame(rows)}, container_stop_ts=2)
    assert len((accessor.distance_m < 8.0).build(cache)) == 0


def test_missing_series_in_cache_returns_empty():
    accessor = SeriesAccessor(_object_series())
    cache = MultiSeriesCache({}, container_stop_ts=2)
    assert len((accessor.distance_m < 8.0).build(cache)) == 0


def test_no_entity_key_single_timeline():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType()),
            T.StructField("signal_id", T.StringType()),
            T.StructField("ts", T.LongType()),
            T.StructField("magnitude", T.DoubleType()),
        ]
    )
    series = Series(
        name="ungrouped",
        schema=schema,
        session_col="container_id",
        signal_col="signal_id",
        timestamp_col="ts",
    )
    df = pd.DataFrame(
        [(1, "s", 0, 5.0), (1, "s", 1, 9.0), (1, "s", 2, 4.0)],
        columns=["container_id", "signal_id", "ts", "magnitude"],
    )
    cache = MultiSeriesCache({"ungrouped": df}, container_stop_ts=3)
    intervals = (SeriesAccessor(series).magnitude < 8.0).build(cache)
    nptest.assert_array_equal(intervals.tstarts, [0, 2])
    nptest.assert_array_equal(intervals.tends, [1, 3])


def test_composes_with_external_intervals_via_and():
    rows = [(1, 0, 1, 5.0), (1, 1, 1, 5.0), (1, 2, 1, 5.0)]
    accessor = SeriesAccessor(_object_series())
    cache = MultiSeriesCache({"object_tracks": _frame(rows)}, container_stop_ts=3)
    built = (accessor.distance_m < 8.0).build(cache)
    composed = built & Intervals(np.array([1.0]), np.array([5.0]))
    nptest.assert_array_equal(composed.tstarts, [1])
    nptest.assert_array_equal(composed.tends, [3])


def test_selector_leaf_kind_is_series_name():
    assert (SeriesAccessor(_object_series()).distance_m < 8.0).leaf_kind == "object_tracks"


# --- frame-list synthesis (point-in-time) -----------------------------------


def test_entity_interval_closes_at_next_signal_frame_not_own_appearance():
    # Entity 47 present at frames 0,100,200; the signal also has a frame at 300
    # (from entity 99). Entity 47 is absent at 300, so its presence closes at the
    # next signal tick (300), not at its own last appearance (200).
    rows = [
        (1, 0, 47, 5.0),
        (1, 100, 47, 5.0),
        (1, 200, 47, 5.0),
        (1, 0, 99, 5.0),
        (1, 300, 99, 5.0),
    ]
    leaf = (SeriesAccessor(_object_series()).distance_m < 8.0).entity_condition()
    cache = MultiSeriesCache({"object_tracks": _frame(rows)}, container_stop_ts=400)
    ei = leaf.entity_intervals(_frame(rows), cache)
    assert _ivs(ei[("fusion", 47)]) == [(0, 300)]
    # Entity 99's last frame (300) is the signal's last frame → closes at session end.
    assert _ivs(ei[("fusion", 99)]) == [(0, 100), (300, 400)]


def test_same_entity_id_under_two_signals_stays_distinct():
    rows = pd.DataFrame(
        [
            (1, "lidar", 0, 47, 5.0),
            (1, "lidar", 100, 47, 5.0),
            (1, "radar", 0, 47, 5.0),
        ],
        columns=["container_id", "sensor_type", "ts", "object_id", "distance_m"],
    )
    leaf = (SeriesAccessor(_object_series()).distance_m < 8.0).entity_condition()
    cache = MultiSeriesCache({"object_tracks": rows}, container_stop_ts=200)
    ei = leaf.entity_intervals(rows, cache)
    assert set(ei.keys()) == {("lidar", 47), ("radar", 47)}


# --- RLE synthesis ----------------------------------------------------------


def _rle_entity_series():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType()),
            T.StructField("sensor_type", T.StringType()),
            T.StructField("seg_start", T.LongType()),
            T.StructField("seg_end", T.LongType()),
            T.StructField("object_id", T.LongType()),
            T.StructField("magnitude", T.DoubleType()),
        ]
    )
    return Series(
        name="tracks",
        schema=schema,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="seg_start",
        tend_col="seg_end",
        entity_key="object_id",
    )


def _norm(iv):
    return sorted(zip(iv.tstarts.tolist(), iv.tends.tolist(), strict=False))


def test_presence_build_equals_union_of_entity_intervals_fuzz():
    # Invariant: presence build() (union over all matching rows) must
    # equal the union of the per-(signal, entity) entity_intervals. This failed
    # 75/400 trials before the running-max fix — always on contained windows,
    # where one entity's window enclosed another's. The per-entity intervals
    # themselves were always correct; only the union truncated.
    series = _rle_entity_series()
    acc = SeriesAccessor(series)
    rng = np.random.default_rng(20260529)
    signals = ["fusion", "lidar"]

    for _ in range(400):
        rows = []
        for signal in signals:
            for object_id in range(1, rng.integers(2, 5)):
                for _seg in range(rng.integers(1, 6)):
                    start = int(rng.integers(0, 100))
                    # strictly positive length: keeps both paths on del_last_empty
                    # semantics, isolating the contained-window behaviour under test.
                    end = start + int(rng.integers(1, 60))
                    mag = float(rng.integers(0, 16))
                    rows.append((1, signal, start, end, object_id, mag))

        df = pd.DataFrame(
            rows,
            columns=[
                "container_id",
                "sensor_type",
                "seg_start",
                "seg_end",
                "object_id",
                "magnitude",
            ],
        )
        cache = MultiSeriesCache({"tracks": df})

        sel = acc.magnitude < 8.0
        presence = sel.build(cache)

        union = Intervals.empty()
        for iv in sel.entity_condition().entity_intervals(df, cache).values():
            union = union | iv

        assert _norm(presence) == _norm(union)


def test_presence_build_equals_union_with_contained_window():
    # Deterministic witness of the fuzz invariant: entity 1's [100,500) encloses
    # entity 2's [300,400). Presence must be the enclosing [100,500), not [100,400).
    series = _rle_entity_series()
    acc = SeriesAccessor(series)
    df = pd.DataFrame(
        [
            (1, "fusion", 100, 500, 1, 5.0),
            (1, "fusion", 300, 400, 2, 5.0),
        ],
        columns=[
            "container_id",
            "sensor_type",
            "seg_start",
            "seg_end",
            "object_id",
            "magnitude",
        ],
    )
    cache = MultiSeriesCache({"tracks": df})
    sel = acc.magnitude < 8.0
    nptest.assert_array_equal(sel.build(cache).tstarts, [100])
    nptest.assert_array_equal(sel.build(cache).tends, [500])


def test_rle_uses_row_intervals_directly():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType()),
            T.StructField("sensor_type", T.StringType()),
            T.StructField("seg_start", T.LongType()),
            T.StructField("seg_end", T.LongType()),
            T.StructField("curvature", T.DoubleType()),
        ]
    )
    series = Series(
        name="lanes",
        schema=schema,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="seg_start",
        tend_col="seg_end",
    )
    df = pd.DataFrame(
        [
            (1, "camera_front", 0, 40, 0.05),
            (1, "camera_front", 60, 90, 0.001),  # below threshold
        ],
        columns=["container_id", "sensor_type", "seg_start", "seg_end", "curvature"],
    )
    cache = MultiSeriesCache({"lanes": df})
    intervals = (SeriesAccessor(series).curvature > 0.01).build(cache)
    # Uses each row's [tstart, tend) directly; no frame-list / shift.
    nptest.assert_array_equal(intervals.tstarts, [0])
    nptest.assert_array_equal(intervals.tends, [40])
