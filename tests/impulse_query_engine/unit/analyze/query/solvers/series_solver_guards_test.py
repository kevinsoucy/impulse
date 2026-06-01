# pylint: disable=missing-function-docstring
"""Guards and edge cases for the registered-series solver paths.

* ``_container_stop_ts_df`` must collapse ``container_metrics`` to exactly one
  ``stop_ts`` per container so the point-in-time stop-ts left join in
  ``_reduce_series`` can never fan out (double-count) the series rows. Covered
  across every container_metrics data shape.
* ``BlobSolver`` cannot resolve channel leaves inside the cogroup, so a query
  that combines registered series with a channel leaf must fail fast with a
  clear, actionable message (series-only queries are unaffected).
"""

import types

import pytest

from impulse_query_engine.analyze.query.solvers.blob_solver import BlobSolver
from impulse_query_engine.analyze.query.solvers.delta_solver import DeltaSolver
from impulse_query_engine.analyze.query.solvers.key_value_store_solver import (
    KeyValueStoreSolver,
)
from impulse_query_engine.analyze.query.solvers.query_solver import QuerySolver
from impulse_query_engine.analyze.query.solvers.solver_config import SolverConfig, TableConfig


def _kvs_cfg() -> SolverConfig:
    return SolverConfig(
        project_id="SAMPLE_PROJECT",
        container_tags=TableConfig(column_name_mapping={"element_id": "key"}),
        container_metrics=TableConfig(column_name_mapping={"project": "project_id"}),
    )


def _solver(spark) -> KeyValueStoreSolver:
    # _container_stop_ts_df lives on the QuerySolver base; any concrete solver
    # exercises the same code. KVS is convenient to instantiate.
    return KeyValueStoreSolver(spark, config=_kvs_cfg())


def _stops(out) -> dict:
    return {r["container_id"]: r["__red_stop"] for r in out.collect()}


# --- _container_stop_ts_df: one row per container across every data shape -----


def test_stop_ts_one_row_per_container_is_preserved(spark):
    metrics = spark.createDataFrame(
        [(1, 100.0), (2, 300.0)], "container_id long, stop_ts double"
    )
    out = _solver(spark)._container_stop_ts_df(metrics)
    assert out.count() == 2
    assert _stops(out) == {1: 100.0, 2: 300.0}


def test_stop_ts_duplicate_identical_rows_collapse_to_one(spark):
    metrics = spark.createDataFrame(
        [(1, 100.0), (1, 100.0), (2, 300.0)], "container_id long, stop_ts double"
    )
    out = _solver(spark)._container_stop_ts_df(metrics)
    assert out.count() == 2
    assert _stops(out) == {1: 100.0, 2: 300.0}


def test_stop_ts_multiple_differing_rows_collapse_to_max_no_fanout(spark):
    # The fanout case: >1 stop_ts for one container. A left join onto this frame
    # must not duplicate the series rows, so it collapses to exactly one row —
    # the latest stop (session end).
    metrics = spark.createDataFrame(
        [(1, 100.0), (1, 200.0), (1, 150.0), (2, 300.0)],
        "container_id long, stop_ts double",
    )
    out = _solver(spark)._container_stop_ts_df(metrics)
    assert out.count() == 2  # one row per container, no fanout
    assert _stops(out) == {1: 200.0, 2: 300.0}


def test_stop_ts_null_value_is_ignored_when_a_real_value_exists(spark):
    metrics = spark.createDataFrame(
        [(1, None), (1, 120.0), (2, 300.0)], "container_id long, stop_ts double"
    )
    out = _solver(spark)._container_stop_ts_df(metrics)
    assert _stops(out) == {1: 120.0, 2: 300.0}


def test_stop_ts_all_null_for_a_container_yields_null_stop(spark):
    # A container whose only stop is NULL → one row with a NULL stop, so its last
    # frame stays open (the documented "no stop configured" behaviour) rather
    # than fanning out or erroring.
    metrics = spark.createDataFrame(
        [(1, None), (1, None), (2, 300.0)], "container_id long, stop_ts double"
    )
    out = _solver(spark)._container_stop_ts_df(metrics)
    stops = _stops(out)
    assert set(stops) == {1, 2}
    assert stops[1] is None
    assert stops[2] == 300.0


def test_stop_ts_empty_metrics_with_column_returns_empty_frame(spark):
    metrics = spark.createDataFrame([], "container_id long, stop_ts double")
    out = _solver(spark)._container_stop_ts_df(metrics)
    assert out is not None
    assert out.count() == 0


def test_stop_ts_absent_column_returns_none(spark):
    metrics = spark.createDataFrame([(1,), (2,)], "container_id long")
    assert _solver(spark)._container_stop_ts_df(metrics) is None


def test_stop_ts_none_metrics_returns_none(spark):
    assert _solver(spark)._container_stop_ts_df(None) is None


# --- BlobSolver: no channel cache in the cogroup ------------------------------


def test_only_grouped_map_solvers_provide_a_channel_cache():
    # Delta and KVS override _channel_cache_cls; BlobSolver does not, so the
    # base's NotImplementedError is what a series+channel query on the default
    # solver ultimately hits.
    assert DeltaSolver._channel_cache_cls is not QuerySolver._channel_cache_cls
    assert KeyValueStoreSolver._channel_cache_cls is not QuerySolver._channel_cache_cls
    assert BlobSolver._channel_cache_cls is QuerySolver._channel_cache_cls


def test_blob_solver_channel_cache_raises_actionable_error():
    with pytest.raises(NotImplementedError, match="DeltaSolver or KeyValueStoreSolver"):
        BlobSolver()._channel_cache_cls()


# --- _source_signal_filter: which signals to read at the source --------------


def _leaf(signal_values):
    return types.SimpleNamespace(signal_values=signal_values)


def test_source_signal_filter_unions_constrained_leaves():
    leaves = [_leaf(frozenset({"lidar"})), _leaf(frozenset({"radar", "fusion"}))]
    assert QuerySolver._source_signal_filter(leaves) == frozenset({"lidar", "radar", "fusion"})


def test_source_signal_filter_returns_none_if_any_leaf_unconstrained():
    # An unconstrained leaf could match any signal → must read all signals.
    leaves = [_leaf(frozenset({"lidar"})), _leaf(None)]
    assert QuerySolver._source_signal_filter(leaves) is None


def test_source_signal_filter_single_constrained_leaf():
    assert QuerySolver._source_signal_filter([_leaf(frozenset({"lidar"}))]) == frozenset({"lidar"})


def test_source_signal_filter_empty_isin_keeps_nothing():
    # signal.isin([]) → frozenset() → an isin([]) filter that drops every row
    # (degenerate but correct: no signal can match).
    assert QuerySolver._source_signal_filter([_leaf(frozenset())]) == frozenset()
