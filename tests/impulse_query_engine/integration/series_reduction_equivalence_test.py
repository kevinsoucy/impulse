# pylint: disable=missing-function-docstring
"""Equivalence gate for ADR-2 Layer 2 (Spark-native interval synthesis).

The Spark-native reduction (``QuerySolver._reduce_series`` →
``_predicate_explode_udf`` + ``_coalesce_intervals`` + ``_with_markers``) must
produce **row-for-row identical** interval sets to the unreduced pandas plan
(``SeriesSelector.entity_intervals`` / ``build`` over
``Intervals(merge_overlaps=True, del_last_empty=True)``). Passing this flips
ADR-2 from Proposed to Accepted.

Ground truth is the pandas path run directly over the full container frame; the
candidate is the real engine reduction, reconstructed per container through
``_build_reduced_cache`` (the same seam the cogroup uses). We compare both the
per-``(signal, entity)`` intervals (``entity_intervals``) and the presence union
(``build``) for every leaf.

The reduction calls each leaf's *existing* TSAL closure, so it is
predicate-kind-agnostic by construction — the equivalence risk lives entirely in
the close synthesis, the coalescing, and the marker rows. The matrix therefore
spans interval shapes (point-in-time next-frame close with/without ``stop_ts``,
same-id-two-signals, RLE touching/contained/gaps/cross-entity, zero-length
mid-stream) and a seeded fuzz, plus a representative spread of predicate kinds
(numeric ranges, ``==``/``!=``/``isin``, string ops, fused same-series ``&``/``|``)
to prove the closure runs correctly inside the batched ``mapInPandas`` UDF.

Run from the spike worktree:
    SPARK_LOCAL_IP=127.0.0.1 uv run pytest \
      tests/impulse_query_engine/integration/series_reduction_equivalence_test.py \
      -p no:cacheprovider -o addopts="" -q
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyspark.sql.types as T
import pytest

from impulse_query_engine.analyze.query.solvers.empty_cache import EmptyTimeSeriesCache
from impulse_query_engine.analyze.query.solvers.key_value_store_solver import (
    KeyValueStoreSolver,
)
from impulse_query_engine.analyze.query.solvers.query_solver import QuerySolver
from impulse_query_engine.analyze.query.solvers.series_cache import MultiSeriesCache
from impulse_query_engine.analyze.query.solvers.solver_config import SolverConfig
from impulse_query_engine.surfaces import Series, SeriesAccessor
from impulse_query_engine.surfaces.series_selector import render_entity_key

# --------------------------------------------------------------------------- #
# Series fixtures — one point-in-time and one RLE, each carrying a numeric and a
# string payload column so every authorable predicate kind is reachable.
# --------------------------------------------------------------------------- #
_PIT_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("ts", T.LongType(), nullable=False),
        T.StructField("object_id", T.LongType(), nullable=False),
        T.StructField("distance_m", T.DoubleType()),
        T.StructField("detection_class", T.StringType()),
    ]
)
_PIT_COLS = ["container_id", "sensor_type", "ts", "object_id", "distance_m", "detection_class"]

_RLE_SCHEMA = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("sensor_type", T.StringType(), nullable=False),
        T.StructField("seg_start", T.LongType(), nullable=False),
        T.StructField("seg_end", T.LongType(), nullable=False),
        T.StructField("object_id", T.LongType(), nullable=False),
        T.StructField("magnitude", T.DoubleType()),
        T.StructField("detection_class", T.StringType()),
    ]
)
_RLE_COLS = [
    "container_id",
    "sensor_type",
    "seg_start",
    "seg_end",
    "object_id",
    "magnitude",
    "detection_class",
]


def _pit_series() -> Series:
    return Series(
        name="object_tracks",
        schema=_PIT_SCHEMA,
        session_col="container_id",
        signal_col="sensor_type",
        timestamp_col="ts",
        entity_key="object_id",
    )


def _rle_series() -> Series:
    return Series(
        name="tracks",
        schema=_RLE_SCHEMA,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="seg_start",
        tend_col="seg_end",
        entity_key="object_id",
    )


def _pit_frame(rows, signal="fusion"):
    # rows: (container_id, ts, object_id, distance_m[, detection_class])
    norm = [(c, signal, ts, oid, d, cls) for (c, ts, oid, d, cls) in rows]
    return pd.DataFrame(norm, columns=_PIT_COLS)


def _rle_frame(rows):
    # rows: (container_id, signal, seg_start, seg_end, object_id, magnitude, detection_class)
    return pd.DataFrame(rows, columns=_RLE_COLS)


# --------------------------------------------------------------------------- #
# Harness: candidate (engine reduction) vs reference (pandas plan).
# --------------------------------------------------------------------------- #
def _solver() -> KeyValueStoreSolver:
    return KeyValueStoreSolver(SolverConfig(project_id="EQUIV"))


def _engine_reduced(spark, series, schema, pdf, leaves, stop_ts):
    """Run the real ``_reduce_series`` and return its reduced rows as pandas."""
    solver = _solver()
    cid = solver.config.container_id_col
    for i, leaf in enumerate(leaves):
        leaf._reduce_key = i

    def source_factory(s):
        return s.createDataFrame(pdf, schema)

    stop_df = None
    if stop_ts is not None:
        cids = sorted(pdf[cid].unique().tolist())
        metrics = spark.createDataFrame(
            pd.DataFrame({cid: cids, "stop_ts": [float(stop_ts)] * len(cids)}),
            T.StructType(
                [
                    T.StructField(cid, T.LongType()),
                    T.StructField("stop_ts", T.DoubleType()),
                ]
            ),
        )
        stop_df = solver._container_stop_ts_df(metrics)

    reduced = solver._reduce_series(spark, series, source_factory, leaves, stop_df, None, cid)
    return reduced.toPandas()


def _candidate(spark, series, schema, pdf, leaf, stop_ts):
    """Engine reduction → per-container cache → {leaf: (entities, presence)}.

    Returns ``(entities_by_container, presence_by_container)`` normalized to
    plain comparable structures.
    """
    reduced_pdf = _engine_reduced(spark, series, schema, pdf, [leaf], stop_ts)
    cid = _solver().config.container_id_col
    entities: dict = {}
    presence: dict = {}
    for cid_val, sub in reduced_pdf.groupby(cid):
        cache = QuerySolver._build_reduced_cache(
            sub.reset_index(drop=True), EmptyTimeSeriesCache()
        )
        entities[int(cid_val)] = _norm_entities(cache.reduced_entities(leaf))
        presence[int(cid_val)] = _norm_intervals(cache.reduced_presence(leaf))
    return entities, presence


def _reference(series, pdf, leaf, stop_ts):
    """Pandas ground truth per container via the unreduced plan."""
    cid = "container_id"
    entities: dict = {}
    presence: dict = {}
    for cid_val, sub in pdf.groupby(cid):
        sub = sub.reset_index(drop=True)
        cache = MultiSeriesCache({series.name: sub}, container_stop_ts=stop_ts)
        entities[int(cid_val)] = _norm_entities(leaf.entity_intervals(sub, cache))
        presence[int(cid_val)] = _norm_intervals(leaf.build(cache))
    return entities, presence


def _norm_intervals(ivs):
    if ivs is None:
        return []
    return sorted((float(a), float(b)) for a, b in ivs.get_data())


def _norm_entities(ei):
    # An empty interval set yields no event window, so it is equivalent to the
    # entity being absent. The reduction (old and new) emits *only* non-empty
    # interval sets; the raw pandas ``entity_intervals`` keeps a dict key with an
    # empty ``Intervals`` for a matched-but-zero-length entity. Drop empties so
    # the comparison is at the event-content contract, not that artifact.
    if ei is None:
        return {}
    out = {}
    for (sig, ent), ivs in ei.items():
        data = sorted((float(a), float(b)) for a, b in ivs.get_data())
        if data:
            out[(render_entity_key(sig), render_entity_key(ent))] = data
    return out


def _assert_equiv(spark, series, schema, pdf, leaf, stop_ts, label):
    cand_ent, cand_pres = _candidate(spark, series, schema, pdf, leaf, stop_ts)
    ref_ent, ref_pres = _reference(series, pdf, leaf, stop_ts)
    # Containers present on both sides (a container with no source rows produces
    # no group on either path); markers keep a data-but-no-match container.
    assert set(cand_ent) >= set(
        ref_ent
    ), f"[{label}] candidate dropped containers {set(ref_ent) - set(cand_ent)}"
    for c in ref_ent:
        assert (
            cand_ent[c] == ref_ent[c]
        ), f"\n[{label}] entity DIVERGENCE container={c}\n  ref : {ref_ent[c]}\n  cand: {cand_ent[c]}"
        assert (
            cand_pres[c] == ref_pres[c]
        ), f"\n[{label}] presence DIVERGENCE container={c}\n  ref : {ref_pres[c]}\n  cand: {cand_pres[c]}"


# --------------------------------------------------------------------------- #
# Point-in-time interval-shape matrix.
# --------------------------------------------------------------------------- #
def test_pit_close_at_next_signal_frame(spark):
    pdf = _pit_frame(
        [
            (1, 0, 47, 5.0, "ped"),
            (1, 100, 47, 5.0, "ped"),
            (1, 200, 47, 5.0, "ped"),
            (1, 0, 99, 5.0, "ped"),
            (1, 300, 99, 5.0, "ped"),
        ]
    )
    leaf = (SeriesAccessor(_pit_series()).distance_m < 8.0).each()
    _assert_equiv(spark, _pit_series(), _PIT_SCHEMA, pdf, leaf, 400, "pit/next-frame")


def test_pit_without_stop_ts_drops_final_frame(spark):
    pdf = _pit_frame([(1, 0, 1, 5.0, "ped"), (1, 1, 1, 5.0, "ped")])
    leaf = (SeriesAccessor(_pit_series()).distance_m < 8.0).each()
    _assert_equiv(spark, _pit_series(), _PIT_SCHEMA, pdf, leaf, None, "pit/no-stop")


def test_pit_same_id_two_signals_stays_distinct(spark):
    pdf = pd.DataFrame(
        [
            (1, "lidar", 0, 47, 5.0, "ped"),
            (1, "lidar", 100, 47, 5.0, "ped"),
            (1, "radar", 0, 47, 5.0, "ped"),
        ],
        columns=_PIT_COLS,
    )
    leaf = (SeriesAccessor(_pit_series()).distance_m < 8.0).each()
    _assert_equiv(spark, _pit_series(), _PIT_SCHEMA, pdf, leaf, 200, "pit/two-signals")


def test_pit_time_varying_predicate(spark):
    # The case Layer 1 cannot push but Layer 2 must reduce: predicate evaluated
    # per row on the full stream, then coalesced (gap where it fails).
    pdf = _pit_frame(
        [
            (1, 0, 47, 5.0, "ped"),
            (1, 100, 47, 20.0, "ped"),
            (1, 200, 47, 4.0, "ped"),
            (1, 300, 47, 3.0, "ped"),
        ]
    )
    leaf = (SeriesAccessor(_pit_series()).distance_m < 8.0).each()
    _assert_equiv(spark, _pit_series(), _PIT_SCHEMA, pdf, leaf, 400, "pit/time-varying")


def test_pit_marker_container_present_but_no_match(spark):
    # container 2 has rows but no match → must stay present (marker) and resolve
    # to empty; container 1 matches.
    pdf = _pit_frame(
        [
            (1, 0, 47, 5.0, "ped"),
            (1, 100, 47, 5.0, "ped"),
            (2, 0, 47, 99.0, "ped"),
            (2, 100, 47, 99.0, "ped"),
        ]
    )
    leaf = (SeriesAccessor(_pit_series()).distance_m < 8.0).each()
    cand_ent, _ = _candidate(spark, _pit_series(), _PIT_SCHEMA, pdf, leaf, 200)
    assert 2 in cand_ent and cand_ent[2] == {}, "non-matching container must be present and empty"
    assert cand_ent[1], "matching container must yield intervals"
    _assert_equiv(spark, _pit_series(), _PIT_SCHEMA, pdf, leaf, 200, "pit/marker")


# --------------------------------------------------------------------------- #
# RLE interval-shape matrix.
# --------------------------------------------------------------------------- #
def test_rle_touching_intervals_merge(spark):
    pdf = _rle_frame(
        [
            (1, "fusion", 0, 10, 1, 5.0, "ped"),
            (1, "fusion", 10, 20, 1, 5.0, "ped"),  # touches → [0,20)
            (1, "fusion", 40, 50, 1, 5.0, "ped"),  # gap → separate
        ]
    )
    leaf = (SeriesAccessor(_rle_series()).magnitude < 8.0).each()
    _assert_equiv(spark, _rle_series(), _RLE_SCHEMA, pdf, leaf, None, "rle/touching")


def test_rle_contained_window_running_max(spark):
    pdf = _rle_frame(
        [
            (1, "fusion", 100, 500, 1, 5.0, "ped"),
            (1, "fusion", 300, 400, 1, 5.0, "ped"),  # contained → keep [100,500)
        ]
    )
    leaf = (SeriesAccessor(_rle_series()).magnitude < 8.0).each()
    _assert_equiv(spark, _rle_series(), _RLE_SCHEMA, pdf, leaf, None, "rle/contained")


def test_rle_cross_entity(spark):
    pdf = _rle_frame(
        [
            (1, "fusion", 0, 10, 1, 5.0, "ped"),
            (1, "fusion", 5, 25, 2, 5.0, "ped"),
            (1, "lidar", 0, 10, 1, 5.0, "ped"),
            (1, "fusion", 100, 120, 1, 9.0, "ped"),  # no match
        ]
    )
    leaf = (SeriesAccessor(_rle_series()).magnitude < 8.0).each()
    _assert_equiv(spark, _rle_series(), _RLE_SCHEMA, pdf, leaf, None, "rle/cross-entity")


def test_rle_zero_length_mid_stream_segment(spark):
    # A zero-length segment [t,t) in the middle: del_last_empty drops *all*
    # empties, which the Spark filter(ts<te) reproduces — locks the one caveat
    # the spike flagged.
    pdf = _rle_frame(
        [
            (1, "fusion", 0, 10, 1, 5.0, "ped"),
            (1, "fusion", 10, 10, 1, 5.0, "ped"),  # zero-length, touching
            (1, "fusion", 10, 20, 1, 5.0, "ped"),
            (1, "fusion", 30, 30, 1, 5.0, "ped"),  # zero-length, isolated
        ]
    )
    leaf = (SeriesAccessor(_rle_series()).magnitude < 8.0).each()
    _assert_equiv(spark, _rle_series(), _RLE_SCHEMA, pdf, leaf, None, "rle/zero-length")


# --------------------------------------------------------------------------- #
# Predicate-kind spread (proves the closure runs in the batched UDF).
# --------------------------------------------------------------------------- #
def _pit_kind_frame():
    return _pit_frame(
        [
            (1, 0, 47, 5.0, "pedestrian"),
            (1, 100, 47, 12.0, "pedestrian"),
            (1, 200, 47, 4.0, "pedestrian"),
            (1, 0, 91, 6.0, "cyclist"),
            (1, 100, 91, 6.0, "cyclist"),
            (1, 0, 12, 30.0, "car"),
        ]
    )


@pytest.mark.parametrize(
    "make_leaf,label",
    [
        (lambda a: (a.distance_m <= 8.0).each(), "num/<="),
        (lambda a: (a.distance_m > 10.0).each(), "num/>"),
        (lambda a: (a.detection_class == "pedestrian").each(), "str/=="),
        (lambda a: (a.detection_class != "car").each(), "str/!="),
        (lambda a: (a.detection_class.isin(["cyclist", "car"])).each(), "str/isin"),
        (lambda a: (a.detection_class.startswith("ped")).each(), "str/startswith"),
        (lambda a: (a.detection_class.contains("cl")).each(), "str/contains"),
        (lambda a: (a.detection_class.matches("ped.*")).each(), "str/matches"),
        (lambda a: (a.detection_class.isnull()).each(), "str/isnull"),
        (lambda a: (a.detection_class.notnull()).each(), "str/notnull"),
        # Fused same-series AND / OR collapse to one leaf with one closure.
        (
            lambda a: ((a.distance_m < 8.0) & (a.detection_class == "pedestrian")).each(),
            "fused/and",
        ),
        (lambda a: ((a.distance_m > 20.0) | (a.detection_class == "cyclist")).each(), "fused/or"),
    ],
)
def test_predicate_kinds(spark, make_leaf, label):
    pdf = _pit_kind_frame()
    leaf = make_leaf(SeriesAccessor(_pit_series()))
    _assert_equiv(spark, _pit_series(), _PIT_SCHEMA, pdf, leaf, 300, f"kind/{label}")


def test_presence_leaf_any(spark):
    # A presence leaf (.any(), no entity scoping) — reduction still groups by
    # (signal, entity); the cache derives presence as their union.
    pdf = _pit_kind_frame()
    leaf = (SeriesAccessor(_pit_series()).distance_m < 8.0).any()
    _assert_equiv(spark, _pit_series(), _PIT_SCHEMA, pdf, leaf, 300, "presence/any")


# --------------------------------------------------------------------------- #
# Seeded fuzz — random RLE and point-in-time rosters, candidate == reference.
# --------------------------------------------------------------------------- #
def test_rle_fuzz_equivalence(spark):
    series = _rle_series()
    rng = np.random.default_rng(20260607)
    signals = ["fusion", "lidar"]
    for trial in range(60):
        rows = []
        for c in (1, 2):
            for signal in signals:
                for oid in range(1, int(rng.integers(2, 5))):
                    for _seg in range(int(rng.integers(1, 6))):
                        start = int(rng.integers(0, 100))
                        end = start + int(rng.integers(0, 60))  # 0 → zero-length
                        mag = float(rng.integers(0, 16))
                        rows.append((c, signal, start, end, oid, mag, "ped"))
        pdf = _rle_frame(rows)
        leaf = (SeriesAccessor(series).magnitude < 8.0).each()
        _assert_equiv(spark, series, _RLE_SCHEMA, pdf, leaf, None, f"rle-fuzz/{trial}")


def test_pit_fuzz_equivalence(spark):
    series = _pit_series()
    rng = np.random.default_rng(20260608)
    signals = ["fusion", "lidar"]
    for trial in range(40):
        rows = []
        for c in (1, 2):
            for signal in signals:
                # Shared per-(container, signal) tick stream across all entities;
                # an entity is present only at some ticks (partial presence), so the
                # next-frame close can be set by a *different* entity's row.
                ticks = sorted(
                    {int(rng.integers(0, 50)) * 10 for _ in range(int(rng.integers(2, 8)))}
                )
                for oid in range(1, int(rng.integers(2, 4))):
                    for t in ticks:
                        if rng.random() < 0.7:
                            rows.append((c, signal, t, oid, float(rng.integers(0, 16)), "ped"))
        if not rows:
            continue
        pdf = pd.DataFrame(rows, columns=_PIT_COLS)
        leaf = (SeriesAccessor(series).distance_m < 8.0).each()
        _assert_equiv(spark, series, _PIT_SCHEMA, pdf, leaf, 600, f"pit-fuzz/{trial}")
