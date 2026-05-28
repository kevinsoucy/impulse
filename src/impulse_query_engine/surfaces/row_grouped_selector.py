"""Finalized leaf expression that evaluates a per-row predicate against
one row-grouped surface and returns ``Intervals`` for one container.

Produced by ``_PartialPredicate.sub_event()``. Composes via the standard
``TimeSeriesExpression`` operators — ``&`` / ``|`` produce ``TimeSeriesOp``
instances that intersect / union ``Intervals`` at the container scope.
This is the form that makes cross-entity correlation work: two
``.sub_event()`` selectors on the same surface each reduce to
container-scope ``Intervals``, and the interval algebra correlates them
in time without requiring any single row to satisfy both sides."""

from __future__ import annotations

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
    from impulse_query_engine.surfaces.row_grouped_surface import RowGroupedSurface


PredicateFn = Callable[[pd.DataFrame], pd.Series]


class RowGroupedSelector(TimeSeriesSelector):
    """A finalized predicate leaf against one row-grouped surface."""

    def __init__(
        self,
        surface: "RowGroupedSurface",
        *,
        predicate: PredicateFn,
        description: str,
        group_scope: bool = False,
    ) -> None:
        self._surface = surface
        self._predicate = predicate
        self._description = description
        self._group_scope = group_scope

        TimeSeriesExpression.__init__(self, is_single_signal=True)
        self._uses_alias = False
        # selector_id machinery on TimeSeriesSelector reads str(self._expr).
        self._expr = _SurfaceTagExpression(surface.name, description, group_scope)

    @property
    def surface(self) -> "RowGroupedSurface":
        return self._surface

    @property
    def description(self) -> str:
        return self._description

    @property
    def group_scope(self) -> bool:
        return self._group_scope

    @property
    def leaf_kind(self) -> str:
        return self._surface.name

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

    def build(self, cache: "SeriesCache") -> Intervals:
        """Read this container's rows for the surface, synthesize per-group
        intervals, apply the (possibly compound) predicate, and return the
        union as ``Intervals``.

        Cache contract: ``cache.get(surface.name)`` returns a pandas
        DataFrame containing only this container's rows. ``cache`` may
        expose ``container_stop_ts`` to close the last row of each group;
        without it the last row collapses to a zero-length interval.
        """
        df = cache.get(self._surface.name)
        if df is None or len(df) == 0:
            return Intervals.empty()

        ts_col = self._surface.timestamp_col
        group_cols = list(self._surface.group_cols)

        sort_cols = group_cols + [ts_col] if group_cols else [ts_col]
        df = df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)

        if group_cols:
            next_ts = df.groupby(group_cols, sort=False)[ts_col].shift(-1)
        else:
            next_ts = df[ts_col].shift(-1)

        stop_ts = getattr(cache, "container_stop_ts", None)
        if stop_ts is not None:
            next_ts = next_ts.fillna(stop_ts)
        else:
            next_ts = next_ts.fillna(df[ts_col])

        mask = self._predicate(df)
        if not mask.any():
            return Intervals.empty()

        tstarts = df.loc[mask, ts_col].to_numpy(dtype=np.float64)
        tends = next_ts[mask].to_numpy(dtype=np.float64)

        sort_idx = np.argsort(tstarts, kind="mergesort")
        return Intervals(
            tstarts[sort_idx],
            tends[sort_idx],
            merge_overlaps=True,
            del_last_empty=True,
        )

    def __str__(self) -> str:
        scope = ", group_scope=True" if self._group_scope else ""
        return f"RowGroupedSelector<{self._surface.name}: {self._description}{scope}>"


class _SurfaceTagExpression:
    """Lightweight stand-in for ``TagExpression``: gives the selector a
    stable string identity for ``selector_id`` / dedup without pretending
    to be a real EAV tag expression."""

    __slots__ = ("_surface_name", "_description", "_group_scope")

    def __init__(self, surface_name: str, description: str, group_scope: bool) -> None:
        self._surface_name = surface_name
        self._description = description
        self._group_scope = group_scope

    def __str__(self) -> str:
        scope = ":grouped" if self._group_scope else ""
        return f"surface:{self._surface_name}{scope}:{self._description}"

    def required_tags(self) -> set[str]:
        return set()

    def get_selector_expr(self):
        return None
