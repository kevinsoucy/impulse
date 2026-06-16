"""Impulse 2.0 Series registration for the BYOD ADAS demo.

The headline of Impulse 2.0: register the per-frame per-object ``object_tracks``
table as a ``Series`` so a single predicate can span it *and* the scalar
``channels`` signal — the engine correlates them at query time via the cogroup,
no manual join, no pre-aggregated "nearest distance" channel.

``object_tracks`` is fused / sensor-agnostic, so it has no signal column of its
own. ``Series`` requires a real ``signal_col``, so we synthesize a single constant
signal ``"fusion"`` in the source_factory (D11 in impulse_2_adas_demo.md) — no
physical column and no change to the ingest write path.
"""

from __future__ import annotations

import pyspark.sql.types as T
from pyspark.sql import functions as F

from impulse_query_engine.surfaces.series import Series

from lib.silver_schema import EGO_MAP_CONTEXT, OBJECT_MAP_CONTEXT, OBJECT_TRACKS

def _with_signal(schema: T.StructType) -> T.StructType:
    """Return a copy of `schema` plus the synthesized non-null `signal_id` column.

    The physical silver tables carry no signal column (the data is fused, not
    per-sensor); the source_factory adds a constant signal at registration, and the
    Series schema must reflect that added column. Build a fresh StructType — do NOT
    mutate the shared schema constants that 01_ingest writes the physical tables from.
    """
    return T.StructType(
        list(schema.fields) + [T.StructField("signal_id", T.StringType(), nullable=False)]
    )


# Series schema = the physical object_tracks schema + the synthesized signal_id.
_OBJECT_TRACKS_SERIES_SCHEMA = _with_signal(OBJECT_TRACKS)

OBJECT_TRACKS_SERIES = Series(
    name="object_tracks",
    schema=_OBJECT_TRACKS_SERIES_SCHEMA,
    session_col="container_id",
    signal_col="signal_id",
    timestamp_col="frame_ts",  # point-in-time shape
    entity_key="object_id",  # per-object identity, scoped to (session, signal)
)


def register_object_tracks(db, object_tracks_table: str) -> None:
    """Register ``object_tracks`` as a Series on ``db``.

    The source_factory adds the constant ``signal_id="fusion"`` so the fused table
    satisfies the Series signal-column contract.
    """
    db.register_series(
        OBJECT_TRACKS_SERIES,
        source_factory=lambda spark: (
            spark.read.table(object_tracks_table).withColumn("signal_id", F.lit("fusion"))
        ),
    )


# ── map_context series (Impulse 2.0 — third queryable table) ─────────────────
#
# Both are point-in-time at the keyframe rate, sharing the time axis with
# object_tracks and channels. signal_id is synthesized as a constant "map" at
# registration (same pattern as object_tracks' "fusion") — map context is fused,
# not per-sensor. ego_map_context has no entity (one ego per frame); object_map_
# context is keyed by object_id, matching object_tracks.

EGO_MAP_CONTEXT_SERIES = Series(
    name="ego_map_context",
    schema=_with_signal(EGO_MAP_CONTEXT),
    session_col="container_id",
    signal_col="signal_id",
    timestamp_col="frame_ts",
)

OBJECT_MAP_CONTEXT_SERIES = Series(
    name="object_map_context",
    schema=_with_signal(OBJECT_MAP_CONTEXT),
    session_col="container_id",
    signal_col="signal_id",
    timestamp_col="frame_ts",
    entity_key="object_id",
)


def register_ego_map_context(db, ego_map_context_table: str) -> None:
    """Register ``ego_map_context`` as a point-in-time Series (constant signal "map")."""
    db.register_series(
        EGO_MAP_CONTEXT_SERIES,
        source_factory=lambda spark: (
            spark.read.table(ego_map_context_table).withColumn("signal_id", F.lit("map"))
        ),
    )


def register_object_map_context(db, object_map_context_table: str) -> None:
    """Register ``object_map_context`` as an entity-keyed Series (constant signal "map")."""
    db.register_series(
        OBJECT_MAP_CONTEXT_SERIES,
        source_factory=lambda spark: (
            spark.read.table(object_map_context_table).withColumn("signal_id", F.lit("map"))
        ),
    )
