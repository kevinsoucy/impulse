"""Unit coverage for the cross-series reduction helpers on QuerySolver (no Spark).

Locks the per-``(container, signal, entity)`` reduction (the trickiest pandas
step — predicate → interval set, marker rows for non-matching groups) and the
reduced-cache reconstruction, plus the per-solver channel-cache hook.
"""

import numpy as np
import pandas as pd
import pytest

from impulse_query_engine.analyze.query.solvers.blob_solver import BlobSolver
from impulse_query_engine.analyze.query.solvers.delta_solver import DeltaSolver
from impulse_query_engine.analyze.query.solvers.empty_cache import EmptyTimeSeriesCache
from impulse_query_engine.analyze.query.solvers.key_value_store_solver import (
    KeyValueStoreSolver,
)
from impulse_query_engine.analyze.query.solvers.query_solver import QuerySolver
from impulse_query_engine.analyze.query.solvers.series_cache import ChannelTimeSeriesCache
from impulse_query_engine.surfaces import Series, SeriesAccessor

_SCHEMA_COLS = ["container_id", "sensor_type", "entity_id", "distance_m"]


def _series() -> Series:
    import pyspark.sql.types as T

    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType()),
            T.StructField("sensor_type", T.StringType()),
            T.StructField("tstart", T.LongType()),
            T.StructField("tend", T.LongType()),
            T.StructField("entity_id", T.LongType()),
            T.StructField("distance_m", T.DoubleType()),
        ]
    )
    return Series(
        name="object_tracks",
        schema=schema,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="tstart",
        tend_col="tend",
        entity_key="entity_id",
    )


def _leaf(reduce_key=0):
    leaf = (SeriesAccessor(_series()).distance_m < 8.0).each()
    leaf._reduce_key = reduce_key
    return leaf


def _group(rows):
    # rows: (entity_id, distance_m, tstart, tend) — one (container, signal) slice.
    df = pd.DataFrame(
        [(1, "lidar", e, d) for (e, d, _ts, _te) in rows],
        columns=_SCHEMA_COLS,
    )
    df["red_tstart"] = [float(ts) for (_e, _d, ts, _te) in rows]
    df["red_tend"] = [float(te) for (_e, _d, _ts, te) in rows]
    return df


def _reduce(group_df, leaves):
    return QuerySolver._reduce_group_udf(
        group_df,
        leaves=leaves,
        cid_col="container_id",
        signal_col="sensor_type",
        entity_cols=["entity_id"],
        tstart_col="red_tstart",
        tend_col="red_tend",
    )


# --- _reduce_group_udf (predicate → interval set, per group) -----------------


def test_reduce_emits_one_row_per_matching_leaf_with_merged_intervals():
    leaf = _leaf(reduce_key=3)
    # entity 47, two adjacent close segments → merge into one interval.
    out = _reduce(_group([(47, 5.0, 0, 10), (47, 4.0, 10, 20)]), [leaf])

    assert out["leaf_key"].tolist() == [3]
    assert out["signal"].tolist() == ["lidar"]
    assert out["entity"].tolist() == ["47"]
    assert out["tstarts"].iloc[0] == [0.0]
    assert out["tends"].iloc[0] == [20.0]


def test_reduce_non_matching_group_emits_marker_row():
    # No row satisfies distance < 8 → a marker row (null leaf_key) keeps the
    # container present so it resolves to an empty result downstream.
    out = _reduce(_group([(47, 99.0, 0, 10)]), [_leaf()])
    assert len(out) == 1
    assert pd.isna(out["leaf_key"].iloc[0])
    assert out["container_id"].iloc[0] == 1


def test_reduce_two_leaves_each_get_their_own_row():
    near = _leaf(reduce_key=0)
    far = (SeriesAccessor(_series()).distance_m > 50.0).each()
    far._reduce_key = 1
    out = _reduce(_group([(47, 5.0, 0, 10)]), [near, far])
    # Only `near` matches the close object → one row, keyed to near.
    assert out["leaf_key"].tolist() == [0]


def test_reduce_empty_group_returns_empty_frame_with_columns():
    out = _reduce(_group([]), [_leaf()])
    assert out.empty
    assert list(out.columns) == [
        "container_id",
        "leaf_key",
        "signal",
        "entity",
        "tstarts",
        "tends",
    ]


# --- _build_reduced_cache (reduction rows → cache keyed by _reduce_key) -------


def _reduced_pdf(records):
    # records: (leaf_key, signal, entity, tstarts, tends)
    return pd.DataFrame(
        records, columns=["container_id", "leaf_key", "signal", "entity", "tstarts", "tends"]
    )


def test_build_reduced_cache_reconstructs_presence_and_entities():
    leaf = _leaf(reduce_key=7)
    pdf = _reduced_pdf(
        [
            (1, 7, "lidar", "47", [0.0], [10.0]),
            (1, 7, "radar", "47", [12.0], [15.0]),
        ]
    )
    cache = QuerySolver._build_reduced_cache(pdf, EmptyTimeSeriesCache())

    # entity_intervals: one entry per (signal, entity).
    ents = cache.reduced_entities(leaf)
    assert set(ents) == {("lidar", "47"), ("radar", "47")}
    assert ents[("lidar", "47")].get_data() == [[0.0, 10.0]]

    # presence (build): union across the leaf's (signal, entity) rows.
    assert cache.reduced_presence(leaf).get_data() == [[0.0, 10.0], [12.0, 15.0]]


def test_build_reduced_cache_skips_marker_rows():
    leaf = _leaf(reduce_key=2)
    pdf = _reduced_pdf([(1, np.nan, None, None, [], [])])
    cache = QuerySolver._build_reduced_cache(pdf, EmptyTimeSeriesCache())
    # Marker only → no reduction for the leaf → None (raw-frame fallback).
    assert cache.reduced_presence(leaf) is None
    assert cache.reduced_entities(leaf) is None


def test_build_reduced_cache_unknown_leaf_returns_none():
    # A leaf with no _reduce_key (no reduction in play) must not match.
    leaf = _leaf(reduce_key=0)
    leaf._reduce_key = None
    pdf = _reduced_pdf([(1, 0, "lidar", "47", [0.0], [10.0])])
    cache = QuerySolver._build_reduced_cache(pdf, EmptyTimeSeriesCache())
    assert cache.reduced_presence(leaf) is None


# --- per-solver channel-cache hook -------------------------------------------


def test_channel_cache_cls_per_solver():
    # Unbound call (the method does not use self), so no Spark session needed.
    # Both grouped-map solvers now return the one shared ChannelTimeSeriesCache.
    assert DeltaSolver._channel_cache_cls(None) is ChannelTimeSeriesCache
    assert KeyValueStoreSolver._channel_cache_cls(None) is ChannelTimeSeriesCache


def test_channel_cache_cls_unsupported_solver_raises():
    # The base is an internal backstop (user-facing rejection of unsupported
    # solvers happens at QueryBuilder._require_series_support); a solver with no
    # channel cache still fails loud rather than silently dropping channel leaves.
    with pytest.raises(NotImplementedError, match="provides no channel cache"):
        BlobSolver()._channel_cache_cls()
