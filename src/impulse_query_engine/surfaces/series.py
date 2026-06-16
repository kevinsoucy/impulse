"""Series — declarative registration of an external time-indexed table so the
engine can evaluate predicates against it and compose results with scalar
channels.

A series maps the customer's physical column names onto the engine's logical
roles — ``session_col`` → session identity, ``signal_col`` → signal identity,
and either ``timestamp_col`` (point-in-time shape) or ``(tstart_col, tend_col)``
(run-length-encoded shape) → the time axis — while keeping the customer's own
schema, column names, and payload columns. A scalar channel is the degenerate
case: an RLE series with a single ``value`` payload column and no entity key.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pyspark.sql.types as T


@dataclass(frozen=True)
class Series:
    """Declarative registration of a time-indexed table as a query series.

    Multiple rows can share the same ``(session, signal, timestamp)`` — for
    example per-frame object detections, per-cycle defect inspections, or
    per-DTC occurrences. Distinct entities within one ``(session, signal)``
    partition are identified by ``entity_key``; an entity key is scoped to its
    signal, so object ``47`` from ``lidar`` and object ``47`` from ``radar`` are
    distinct entities.

    Exactly one time-axis shape must be declared:

    - **Point-in-time** (``timestamp_col``): each row is a sample at an instant;
      the engine synthesizes ``[frame_ts_i, frame_ts_{i+1})`` intervals from the
      per-``(session, signal)`` frame list at evaluation time.
    - **Run-length encoded** (``tstart_col`` + ``tend_col``): each row already
      carries its own ``[tstart, tend)`` interval, used directly.

    **Series support predicates only, not aggregations.** A series column proxy
    builds *predicate* leaves — comparisons (``<``, ``<=``, ``==``, ``!=``, ``>=``,
    ``>``), ``isin``, the string ops (``contains`` / ``startswith`` / ``endswith``
    / ``matches``), and null checks — each of which evaluates to a set of time
    ``Intervals`` ("when did this hold?"). Channel-style *aggregations*
    (``.mean()``, ``.sum()``, histograms, …) are **not** available on series
    columns; those remain a channels-only capability. This is deliberate: because
    every series leaf is interval-valued, the cross-series cogroup can reduce a
    series to interval sets without losing information. To aggregate a series
    payload, compute it upstream and register the result as its own series/channel.

    **Time-axis precondition.** Timestamps are compared as raw integers across
    series and against the container ``stop_ts``; the engine never resamples or
    converts units at query time. Every series used together in a query — and the
    container metrics — must therefore share one time axis (same unit and epoch),
    aligned upstream at ingest. A mismatch is not detectable at runtime (both are
    ``LongType``) and yields silently wrong intervals. For dense series, shape the
    data upstream (RLE, value quantization, downsampling) so one session fits in
    memory. See the Query Engine reference, "Time-axis alignment".

    Parameters
    ----------
    name : str
        Series name. Used as the ``leaf_kind`` discriminator on selectors
        and as the registry key on the solver. Must be unique per deployment.
    schema : pyspark.sql.types.StructType, optional
        Spark schema describing the series' columns. The accessor reflects
        this schema to expose typed column proxies. When omitted, the schema
        is inferred from the ``source_factory`` DataFrame at registration and
        the same structural validation runs then; when provided, validation
        runs at construction.
    session_col : str
        Customer column mapped to the ``session_id`` role. Defaults to
        ``"session_id"``; override for non-standard names (e.g.
        ``"container_id"``).
    signal_col : str
        Customer column mapped to the ``signal_id`` role. Defaults to
        ``"signal_id"``; override for non-standard names (e.g.
        ``"sensor_type"`` → ``"lidar"`` / ``"radar"`` / ``"camera_front"`` /
        ``"fusion"``). Signal is a first-class filtering dimension — the
        equivalent of a channel in Impulse 1.0.
    timestamp_col : str, optional
        Point-in-time shape. Mutually exclusive with ``(tstart_col,
        tend_col)``; exactly one shape must be set.
    tstart_col, tend_col : str, optional
        Run-length-encoded shape; each row carries its own ``[tstart, tend)``
        interval. Both must be set together.
    entity_key : str or tuple of str, optional
        Per-entity identity column(s) (e.g. ``entity_id``). When set, the
        series carries entity multiplicity and supports ``.each()`` / ``.ids()``.
        ``None`` means one timeline per ``(session, signal)`` with no entity
        multiplicity (e.g. channels, IMU).
    """

    name: str
    schema: T.StructType | None = None
    session_col: str = "session_id"
    signal_col: str = "signal_id"
    timestamp_col: str | None = None
    tstart_col: str | None = None
    tend_col: str | None = None
    entity_key: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        has_point = self.timestamp_col is not None
        has_tstart = self.tstart_col is not None
        has_tend = self.tend_col is not None
        if has_point and (has_tstart or has_tend):
            raise ValueError(
                f"Series {self.name!r} declares both timestamp_col and "
                "tstart_col/tend_col; set exactly one time-axis shape "
                "(point-in-time OR run-length-encoded)."
            )
        if not has_point and not (has_tstart or has_tend):
            raise ValueError(
                f"Series {self.name!r} declares no time axis; set either "
                "timestamp_col (point-in-time) or both tstart_col and tend_col "
                "(run-length-encoded)."
            )
        if (has_tstart or has_tend) and not (has_tstart and has_tend):
            raise ValueError(
                f"Series {self.name!r} run-length-encoded shape requires both "
                "tstart_col and tend_col."
            )
        if self.schema is not None:
            self._validate_structure(self.schema)

    def _validate_structure(self, schema: T.StructType) -> None:
        """Validate that every structural role maps to a real schema column."""
        field_names = {f.name for f in schema.fields}
        for role, col in (
            ("session_col", self.session_col),
            ("signal_col", self.signal_col),
        ):
            if col not in field_names:
                raise ValueError(f"{role} {col!r} not in schema fields: {sorted(field_names)}")
        for col in self.time_cols:
            if col not in field_names:
                raise ValueError(
                    f"time column {col!r} not in schema fields: {sorted(field_names)}"
                )
        missing = [c for c in self.entity_key_cols if c not in field_names]
        if missing:
            raise ValueError(
                f"entity_key references unknown columns {missing}; "
                f"schema fields are {sorted(field_names)}"
            )

    def with_schema(self, schema: T.StructType) -> "Series":
        """Return a copy with ``schema`` resolved and structurally validated.

        Used at registration when a series was declared without a schema: the
        inferred schema from the source DataFrame is supplied here and the same
        structural validation runs.
        """
        return replace(self, schema=schema)

    @property
    def is_point_in_time(self) -> bool:
        return self.timestamp_col is not None

    @property
    def is_rle(self) -> bool:
        return self.tstart_col is not None

    @property
    def time_cols(self) -> tuple[str, ...]:
        """The time-axis columns for the declared shape."""
        if self.is_point_in_time:
            return (self.timestamp_col,)
        return (self.tstart_col, self.tend_col)

    @property
    def entity_key_cols(self) -> tuple[str, ...]:
        """Normalized entity-key columns as a tuple. Empty when no entity_key is set."""
        if self.entity_key is None:
            return ()
        if isinstance(self.entity_key, str):
            return (self.entity_key,)
        return tuple(self.entity_key)

    @property
    def structural_cols(self) -> frozenset[str]:
        """Columns mapped to a structural role — not exposed as predicate proxies.

        The accessor exposes every column **not** in this set. For channels the
        only remaining column is ``value``; for object tracks it is
        ``detection_class``, ``distance_m``, ``rel_velocity_ms``.
        """
        cols = {self.session_col, self.signal_col, *self.time_cols, *self.entity_key_cols}
        return frozenset(cols)
