"""Finalized leaf expression that evaluates a per-row predicate against one
series and returns ``Intervals`` for one container.

Produced by ``_PartialPredicate.any()`` (presence / merged window),
``.each()`` (per-entity), or ``.each()/.any()`` followed by ``.ids()`` (the same
windowing, plus an entity-id roster projected into ``entity_key``). Composes via
the standard ``TimeSeriesExpression`` operators — ``&`` / ``|`` intersect / union
``Intervals`` at the container scope. Two entity-scoped selectors on the same
series each reduce to container-scope ``Intervals``, and the interval algebra
correlates them in time without requiring any single row to satisfy both sides.

Interval synthesis follows the series shape:

- **Point-in-time:** for each ``(session, signal)`` partition, the sorted unique
  timestamps form the *signal frame list*. Each row's interval is
  ``[frame_ts_i, frame_ts_{i+1})`` where the close is the next tick in that
  signal's frame list — not the entity's own next appearance. The final frame
  (no subsequent tick) closes at session end (``cache.container_stop_ts``).
- **RLE:** each row already carries its ``[tstart_col, tend_col)`` interval.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyspark.sql.types as T

from impulse_query_engine.analyze.metadata.tag_expression import TagExpression
from impulse_query_engine.analyze.metadata.time_series_expression import (
    TimeSeriesExpression,
    TimeSeriesSelector,
)
from impulse_query_engine.model.series.intervals import Intervals

if TYPE_CHECKING:
    from impulse_query_engine.analyze.query.solvers.series_cache import SeriesCache
    from impulse_query_engine.surfaces.series import Series


PredicateFn = Callable[[pd.DataFrame], pd.Series]


def render_entity_key(entity: object) -> str:
    """Canonical string rendering of an entity key.

    Scalars use ``str`` (numpy scalars unwrapped via ``.item()``); compound
    (tuple) keys become a JSON array of their native components. Used both by the
    reporting ``entity_key`` serializer and by the Spark per-entity reduction
    stage, so a reduced ``(signal, entity)`` key renders identically to the
    raw-frame path. Idempotent on an already-rendered scalar string.
    """
    if isinstance(entity, tuple):
        parts = [p.item() if hasattr(p, "item") else p for p in entity]
        return json.dumps(parts)
    return str(entity.item() if hasattr(entity, "item") else entity)


class SeriesSelector(TimeSeriesSelector):
    """A finalized predicate leaf against one series."""

    def __init__(
        self,
        series: "Series",
        *,
        predicate: PredicateFn,
        description: str,
        per_entity_windowing: bool = False,
        id_alias: str | None = None,
        id_limit: int | None = None,
        signal_values: frozenset | None = None,
    ) -> None:
        self._series = series
        self._predicate = predicate
        self._description = description
        # per_entity_windowing — the leaf SPLITS output one window per entity
        #   (``.each()``). ``.any()`` / ``.any().ids()`` merge to one window.
        # The ``entity_scoped`` axis (grouped per ``(signal, entity)`` via
        # ``entity_intervals``, vs plain presence via ``build``) is derived: a
        # leaf is entity-scoped exactly when it splits per entity or projects ids
        # (``.each()`` / ``.ids()``). See the ``entity_scoped`` property.
        self._per_entity_windowing = per_entity_windowing
        # Identity projection: when set, this leaf's matched entity ids are
        # emitted into the event ``entity_key`` under ``id_alias``, capped at
        # ``id_limit``. ``None`` ⇒ the leaf contributes windowing only and is not
        # projected (``entity_key`` stays NULL for it).
        self._id_alias = id_alias
        self._id_limit = id_limit
        # Enumerable set of signal values this leaf can match (``==``/``isin`` on
        # the signal column, propagated through ``&``/``|`` fusion), or ``None``
        # when the leaf does not pin the signal. The solver lifts this to a
        # source-read filter so Delta can prune files/partitions for signals no
        # leaf can match; the per-row predicate stays the exact filter.
        self._signal_values = signal_values
        # Stable per-solve identifier assigned in the driver when a Spark
        # per-entity reduction stage runs ahead of the cogroup. It lets
        # a reduced cache map pre-synthesized intervals back to this leaf across
        # the pickle boundary (``id()`` is not stable across separately-pickled
        # references). ``None`` when no reduction is in play → raw-frame path.
        self._reduce_key: int | None = None

        TimeSeriesExpression.__init__(self, is_single_signal=True)
        self._uses_alias = False
        # selector_id machinery on TimeSeriesSelector reads str(self._expr).
        self._expr = _SeriesTagExpression(
            series.name, description, per_entity_windowing, id_alias, id_limit
        )

    @property
    def series(self) -> "Series":
        return self._series

    @property
    def description(self) -> str:
        return self._description

    @property
    def entity_scoped(self) -> bool:
        """Whether the leaf is grouped per ``(signal, entity)`` rather than plain
        presence. Derived: a leaf is entity-scoped exactly when it splits per
        entity (``.each()``) or projects entity ids (``.ids()``)."""
        return self._per_entity_windowing or self._id_alias is not None

    @property
    def per_entity_windowing(self) -> bool:
        return self._per_entity_windowing

    @property
    def id_alias(self) -> str | None:
        return self._id_alias

    @property
    def id_limit(self) -> int | None:
        return self._id_limit

    @property
    def signal_values(self) -> frozenset | None:
        return self._signal_values

    def ids(self, as_: str, limit: int = 5) -> "SeriesSelector":
        """Project this leaf's matched entity ids into the event ``entity_key``.

        Returns a new entity-scoped leaf that, in addition to its windowing
        role, emits the matched entity ids under the alias *as_* — a roster for
        ``.any()`` (every entity in the window) or the single id for ``.each()``.
        The roster is truncated to *limit* ids. Windowing (merged vs per-entity)
        is inherited from the ``.any()`` / ``.each()`` this is chained onto.

        Raises
        ------
        ValueError
            If the series has no ``entity_key`` — there is no identity to project.
        """
        if self._series.entity_key is None:
            raise ValueError(
                f"Series {self._series.name!r} has no entity_key; "
                ".ids() needs a per-entity identity to project. Use a plain "
                ".any() (no .ids()) for a presence check."
            )
        return SeriesSelector(
            self._series,
            predicate=self._predicate,
            description=self._description,
            per_entity_windowing=self._per_entity_windowing,
            id_alias=as_,
            id_limit=limit,
            signal_values=self._signal_values,
        )

    @property
    def leaf_kind(self) -> str:
        return self._series.name

    def dtype(self):
        return T.ArrayType(T.ArrayType(T.DoubleType()))

    def get_required_tag_exprs(self) -> set[TagExpression]:
        return set()

    def required_tags(self) -> set[str]:
        return set()

    def get_selector_expr(self):
        return None

    def get_selectors(self) -> list["TimeSeriesSelector"]:
        return [self]

    # ------------------------------------------------------------------
    # Interval synthesis
    # ------------------------------------------------------------------

    def _row_intervals(self, df: pd.DataFrame, cache: "SeriesCache"):
        """Per-row ``[tstart, tend)`` arrays aligned to *df* row order.

        RLE rows use their own interval columns directly. Point-in-time rows
        close at the next tick in their ``(session, signal)`` frame list, with
        the final frame closing at ``cache.container_stop_ts``.
        """
        series = self._series
        if series.is_rle:
            tstarts = df[series.tstart_col].to_numpy(dtype=np.float64)
            tends = df[series.tend_col].to_numpy(dtype=np.float64)
            return tstarts, tends

        ts_col = series.timestamp_col
        sig_col = series.signal_col
        stop_ts = getattr(cache, "container_stop_ts", None)
        tstarts = df[ts_col].to_numpy(dtype=np.float64)
        # np.full(nan) not np.empty: the per-signal loop only fills rows whose
        # signal value it visits, so a float-NaN signal value (which never equals
        # itself, so pd.unique can't route it) would otherwise leave garbage.
        # Null/NaN signal values are unsupported as frame-list keys — such a row
        # keeps a NaN end and yields no usable interval; register a real signal
        # value (see MeasurementDB.register_series valid_signals) instead.
        tends = np.full(len(df), np.nan, dtype=np.float64)
        sig_values = df[sig_col].to_numpy()
        for sig_val in pd.unique(sig_values):
            pos = np.flatnonzero(sig_values == sig_val)
            sub_ts = tstarts[pos]
            frame_list = np.unique(sub_ts)  # sorted unique frame timestamps
            next_tick = {frame_list[i]: frame_list[i + 1] for i in range(len(frame_list) - 1)}
            if stop_ts is not None:
                last_close = float(stop_ts)
            else:
                # No session end available — the final frame collapses to a
                # zero-length interval and is dropped by del_last_empty.
                last_close = float(frame_list[-1]) if len(frame_list) else np.nan
            tends[pos] = [next_tick.get(t, last_close) for t in sub_ts]
        return tstarts, tends

    def build(self, cache: "SeriesCache") -> Intervals:
        """Apply the predicate across this container's rows and return the
        union of matching-row intervals as container-scope ``Intervals``.

        This is presence semantics — "does the condition hold anywhere?".
        Per-entity attribution (for ``EntityEvent``) goes through
        ``entity_intervals``.
        """
        reduced = cache.reduced_presence(self)
        if reduced is not None:
            return reduced
        df = cache.get(self._series.name)
        if df is None or len(df) == 0:
            return Intervals.empty()
        df = df.reset_index(drop=True)

        mask = np.asarray(self._predicate(df), dtype=bool)
        if not mask.any():
            return Intervals.empty()

        tstarts, tends = self._row_intervals(df, cache)
        ts = tstarts[mask]
        te = tends[mask]
        order = np.argsort(ts, kind="mergesort")
        return Intervals(
            ts[order],
            te[order],
            merge_overlaps=True,
            del_last_empty=True,
        )

    def entity_intervals(self, df: pd.DataFrame, cache: "SeriesCache") -> dict[tuple, Intervals]:
        """Per-``(signal, entity_key)`` ``Intervals`` for an entity-scoped leaf.

        Frame-list synthesis uses the full per-signal frame list (all entities);
        matching rows are then grouped by ``(signal, entity)`` and merged. Entity
        keys are scoped to their signal, so the same id under two signals stays
        distinct. Returns ``{(signal_value, entity_value): Intervals}``.
        """
        reduced = cache.reduced_entities(self)
        if reduced is not None:
            return reduced
        series = self._series
        df = df.reset_index(drop=True)
        if len(df) == 0:
            return {}

        mask = np.asarray(self._predicate(df), dtype=bool)
        if not mask.any():
            return {}

        tstarts, tends = self._row_intervals(df, cache)
        sig_values = df[series.signal_col].to_numpy()
        entity_cols = list(series.entity_key_cols)
        if len(entity_cols) == 1:
            ent_values = df[entity_cols[0]].to_numpy()

            def entity_of(i):
                return ent_values[i]

        else:
            ent_arrays = [df[c].to_numpy() for c in entity_cols]

            def entity_of(i):
                return tuple(a[i] for a in ent_arrays)

        groups: dict[tuple, list[int]] = {}
        for i in np.flatnonzero(mask):
            groups.setdefault((sig_values[i], entity_of(i)), []).append(i)

        result: dict[tuple, Intervals] = {}
        for key, idxs in groups.items():
            ts = tstarts[idxs]
            te = tends[idxs]
            order = np.argsort(ts, kind="mergesort")
            result[key] = Intervals(
                ts[order],
                te[order],
                merge_overlaps=True,
                del_last_empty=True,
            )
        return result

    def __str__(self) -> str:
        return f"SeriesSelector<{self._series.name}: {self._description}{_scope_suffix(self)}>"


def _scope_suffix(leaf: "SeriesSelector | _SeriesTagExpression") -> str:
    """Human/identity rendering of a leaf's two axes + id projection.

    Encodes everything that affects the emitted facts so two leaves with the
    same predicate but different windowing / projection get distinct selector
    ids (no false dedup) and distinct event definition hashes.
    """
    entity_scoped = leaf._per_entity_windowing or leaf._id_alias is not None
    if not entity_scoped:
        verb = "any"
    elif leaf._per_entity_windowing:
        verb = "each"
    else:
        # entity-scoped but merged window — an ``.any().ids()`` roster leaf.
        verb = "any"
    if leaf._id_alias is None:
        return f":{verb}" if entity_scoped else ""
    return f":{verb}.ids({leaf._id_alias},{leaf._id_limit})"


class _SeriesTagExpression:
    """Lightweight stand-in for ``TagExpression``: gives the selector a
    stable string identity for ``selector_id`` / dedup without pretending
    to be a real EAV tag expression."""

    __slots__ = ("_series_name", "_description", "_per_entity_windowing", "_id_alias", "_id_limit")

    def __init__(
        self,
        series_name: str,
        description: str,
        per_entity_windowing: bool = False,
        id_alias: str | None = None,
        id_limit: int | None = None,
    ) -> None:
        self._series_name = series_name
        self._description = description
        self._per_entity_windowing = per_entity_windowing
        self._id_alias = id_alias
        self._id_limit = id_limit

    def __str__(self) -> str:
        return f"series:{self._series_name}{_scope_suffix(self)}:{self._description}"

    def required_tags(self) -> set[str]:
        return set()

    def get_selector_expr(self):
        return None
