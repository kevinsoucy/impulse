"""Unit coverage for the cross-series reduction helpers on QuerySolver (no Spark).

Locks the reduced-cache reconstruction (reduction rows → cache keyed by
``_reduce_key``, marker rows skipped) and the per-solver channel-cache hook.

The reduction itself moved to Spark-native interval synthesis (ADR-2 Layer 2:
``_predicate_explode_udf`` + ``_coalesce_intervals`` + ``_with_markers``), so its
predicate → interval-set correctness is now covered, end-to-end against the
unreduced pandas plan, by the Spark equivalence suite
(``integration/series_reduction_equivalence_test.py``).
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
