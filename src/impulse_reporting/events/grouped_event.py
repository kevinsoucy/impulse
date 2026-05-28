"""Per-entity event: extends BasicEvent so the matching entity's identity
shows up in the fact table's ``group_value`` column."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pandas as pd

from impulse_query_engine.analyze.metadata.time_series_expression import (
    TimeSeriesExpression,
)
from impulse_query_engine.analyze.query.solvers.series_cache import MultiSurfaceCache
from impulse_query_engine.model.series.intervals import Intervals
from impulse_query_engine.surfaces.row_grouped_selector import RowGroupedSelector
from impulse_reporting.events.basic_event import BasicEvent


def _to_native(value: Any) -> Any:
    """Convert numpy scalars (int64, float64, ...) to their Python equivalents.

    Without this, ``json.dumps`` on a tuple of numpy ints emits a list of
    quoted strings (via ``default=str``) rather than a numeric JSON array.
    """
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        try:
            return value.item()
        except (ValueError, AttributeError):
            return value
    return value


def _serialize_group_value(value: Any) -> str:
    """Serialize a group identity to the StringType ``group_value`` column.

    - Numeric scalars → decimal string (``str(value)``).
    - Tuple / list (compound keys) → JSON array of native scalars.
    - Strings pass through.
    """
    if isinstance(value, (tuple, list)):
        return json.dumps([_to_native(v) for v in value])
    return str(_to_native(value))


class GroupedEvent(BasicEvent):
    """An event whose matching windows are emitted per entity.

    Where ``BasicEvent`` answers "did this combination of predicates hold
    anywhere in the recording?", ``GroupedEvent`` answers "did this hold
    for each entity, and which one(s) triggered each window?". The
    matching entity's identity is materialized into the new
    ``group_value`` column on the event fact table.

    Every ``RowGroupedSelector`` leaf in the expression tree must be
    constructed through ``accessor(group_scope=True)`` — the event
    validates this at construction time so a mixed-scope expression fails
    early rather than producing nonsensical output.

    The Spark batch-solve integration is a follow-up; today the per-group
    materialization runs through ``materialize_per_container`` and is the
    target of the unit tests.
    """

    def __init__(
        self,
        name: str,
        expr: TimeSeriesExpression,
        desc: str | None = None,
        required_channels: list[str] | None = None,
        attributes: Mapping[str, str] | None = None,
    ):
        from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate

        if isinstance(expr, _PartialPredicate):
            expr = expr.sub_event()
        self._validate_group_scope(expr)
        super().__init__(
            name=name,
            expr=expr,
            desc=desc,
            required_channels=required_channels,
            attributes=attributes,
        )

    def get_event_type_str(self) -> str:
        return "GROUPED_EVENT"

    @staticmethod
    def _validate_group_scope(expr: TimeSeriesExpression) -> None:
        leaves = [s for s in expr.get_selectors() if isinstance(s, RowGroupedSelector)]
        if not leaves:
            raise ValueError(
                "GroupedEvent requires at least one RowGroupedSelector leaf in its expression"
            )
        scopes = {leaf.group_scope for leaf in leaves}
        if scopes != {True}:
            raise ValueError(
                "GroupedEvent requires every RowGroupedSelector leaf to be authored with "
                "group_scope=True (use accessor(group_scope=True).column ...); "
                f"found leaves with group_scope values: {sorted(scopes)}"
            )
        surfaces = {leaf.surface.name for leaf in leaves}
        if len(surfaces) > 1:
            # Cross-surface per-group joins are a deferred follow-up;
            # block at construction so the failure mode is clear.
            raise ValueError(
                "GroupedEvent currently supports group-scope predicates against a "
                f"single surface; expression references {sorted(surfaces)}"
            )

    def materialize_per_container(
        self,
        container_id: int,
        surface_df: pd.DataFrame,
        container_stop_ts: float,
        channel_cache_factory=None,
    ) -> list[tuple[int, float, float, str]]:
        """Run the expression per group for one container and return rows.

        Iterates the surface's group_col values, builds a per-group
        ``MultiSurfaceCache`` (containing only that group's rows), evaluates
        the expression to ``Intervals``, and emits one tuple per matching
        window with the group identity serialized into ``group_value``.

        Parameters
        ----------
        container_id : int
            The container these rows belong to.
        surface_df : pandas.DataFrame
            All rows of the row-grouped surface that belong to this
            container. Will be partitioned by ``group_col`` here.
        container_stop_ts : float
            End timestamp used to close the last row of each group's
            window during interval synthesis.
        channel_cache_factory : callable, optional
            Hook for composing with channel predicates. When provided, the
            factory is called per container and is expected to return a
            cache-like object exposing ``resolve`` / ``load_blob``. Today
            this argument is wired through but the test path exercises
            row-grouped leaves in isolation; the Spark batch-solve will
            assemble the composite cache.

        Returns
        -------
        list of tuple
            ``(container_id, start_ts, end_ts, group_value)`` rows for each
            matching window of each group.
        """
        leaves = [
            s
            for s in self.expression.get_selectors()
            if isinstance(s, RowGroupedSelector)
        ]
        surface = leaves[0].surface  # validation guarantees exactly one surface
        group_cols = list(surface.group_cols)
        if not group_cols:
            raise ValueError(
                f"Surface {surface.name!r} has no group_col; GroupedEvent requires "
                "a group_col on the surface to identify entities."
            )

        rows: list[tuple[int, float, float, str]] = []
        for group_key, group_df in surface_df.groupby(group_cols, sort=False):
            # pandas returns a 1-tuple for single-column group_by even when
            # passed a list; unwrap so single-column surfaces serialize as a
            # scalar instead of a 1-element JSON list.
            if len(group_cols) == 1 and isinstance(group_key, tuple):
                group_key = group_key[0]
            cache = MultiSurfaceCache(
                {surface.name: group_df},
                container_stop_ts=container_stop_ts,
            )
            result = self.expression.build(cache)
            if not isinstance(result, Intervals) or len(result) == 0:
                continue
            group_value = _serialize_group_value(group_key)
            for tstart, tend in zip(result.tstarts, result.tends, strict=False):
                rows.append((container_id, float(tstart), float(tend), group_value))
        return rows
