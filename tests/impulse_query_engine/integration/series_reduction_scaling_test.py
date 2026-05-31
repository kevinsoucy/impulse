# pylint: disable=missing-function-docstring
"""Guard that the per-entity reduction actually *engages*.

The cogroup correctness tests (``cross_series_solver_test.py``) prove the
reduction is result-identical to the old raw-frame path. They do **not** prove
it bounds memory: a regression that silently bypassed the reduction (carrying
raw frames into the cogroup) would still pass them.

This test asserts the structural property that makes the reduction a scaling
mitigation: ``QuerySolver._reduce_series`` emits **one row per
``(container, signal, entity)`` group**, regardless of how many raw frames each
entity has. So the reduced stream's row count tracks *entity* count, not *frame*
count — the reduced count is ``≪`` the raw count for a dense series, and is
**invariant** to frames-per-entity. If the reduction stopped engaging, the
reduced count would scale with frames and these assertions would fail.
"""

import pyspark.sql.types as T
from pyspark.sql import SparkSession

from impulse_query_engine.analyze.query.solvers.key_value_store_solver import (
    KeyValueStoreSolver,
)
from impulse_query_engine.analyze.query.solvers.solver_config import SolverConfig, TableConfig
from impulse_query_engine.surfaces import Series, SeriesAccessor

_DENSE_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("tstart", T.LongType(), nullable=False),
        T.StructField("tend", T.LongType(), nullable=False),
        T.StructField("entity_id", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType()),
    ]
)

_CONTAINERS = (1, 2, 3)
_ENTITIES = (10, 20, 30, 40)  # per container
_N_GROUPS = len(_CONTAINERS) * len(_ENTITIES)  # one signal ("lidar") per group


def _dense_series() -> Series:
    return Series(
        name="object_tracks",
        schema=_DENSE_SCHEMA,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="tstart",
        tend_col="tend",
        entity_key="entity_id",
    )


def _dense_rows(frames_per_entity: int):
    """Contiguous RLE frames per (container, entity); all within 8 m so the
    predicate matches every frame (each entity reduces to a single merged
    interval ``[0, frames_per_entity)``)."""
    rows = []
    for cid in _CONTAINERS:
        for eid in _ENTITIES:
            for t in range(frames_per_entity):
                rows.append((cid, "lidar", t, t + 1, eid, 5.0))
    return rows


def _kvs_solver(spark: SparkSession) -> KeyValueStoreSolver:
    cfg = SolverConfig(
        project_id="SAMPLE_PROJECT",
        container_tags=TableConfig(column_name_mapping={"element_id": "key"}),
        container_metrics=TableConfig(column_name_mapping={"project": "project_id"}),
    )
    return KeyValueStoreSolver(spark, config=cfg)


def _reduce(spark: SparkSession, frames_per_entity: int):
    """Run the per-entity reduction over a dense fixture; return (raw, reduced) DFs."""
    rows = _dense_rows(frames_per_entity)
    raw = spark.createDataFrame(rows, _DENSE_SCHEMA)
    leaf = (SeriesAccessor(_dense_series()).distance_m < 8.0).entity_condition()
    leaf._reduce_key = 0
    reduced = _kvs_solver(spark)._reduce_series(
        spark,
        _dense_series(),
        lambda _s, _rows=rows: spark.createDataFrame(_rows, _DENSE_SCHEMA),
        [leaf],
        None,  # stop_df — RLE series carries its own tend
        None,  # container_ids — no prune
        "container_id",
    )
    return raw, reduced


def test_reduction_collapses_frames_to_one_row_per_entity(spark: SparkSession):
    frames = 50
    raw, reduced = _reduce(spark, frames)

    raw_count = raw.count()
    reduced_count = reduced.count()

    # Raw stream carries every frame; reduced stream carries one row per
    # (container, signal, entity) group.
    assert raw_count == len(_CONTAINERS) * len(_ENTITIES) * frames  # 600
    assert reduced_count == _N_GROUPS  # 12
    # The reduction collapses all `frames` per-frame rows of a group to one row:
    # raw is exactly `frames`x the reduced count. This fails if raw frames are
    # carried into the cogroup instead of the reduced interval sets.
    assert raw_count == reduced_count * frames

    # Each reduced row is a single merged interval [0, frames) — the per-frame
    # rows coalesced, not 50 separate intervals.
    sample = reduced.where("entity = '10' AND container_id = 1").collect()
    assert len(sample) == 1
    assert sample[0]["tstarts"] == [0.0]
    assert sample[0]["tends"] == [float(frames)]


def test_reduced_row_count_is_invariant_to_frames_per_entity(spark: SparkSession):
    # The structural guarantee: reduced count tracks entities, not frames. A
    # 4x denser fixture yields 4x the raw rows but the SAME reduced row count.
    _, reduced_sparse = _reduce(spark, 50)
    _, reduced_dense = _reduce(spark, 200)

    assert reduced_sparse.count() == _N_GROUPS
    assert reduced_dense.count() == _N_GROUPS
