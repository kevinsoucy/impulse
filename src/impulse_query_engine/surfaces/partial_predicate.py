"""In-progress predicate against a row-grouped surface — composes per-row
via ``&`` / ``|`` until ``.sub_event()`` finalizes it as a leaf.

Authoring story:

- Column proxies (``accessor.distance_m < 8.0``) return ``_PartialPredicate``.
- Same-surface, same-group_scope ``&`` / ``|`` between partials fuses
  clauses into a single per-row predicate. ``(class == 1) & (dist < 8)``
  evaluates as "this row is a cyclist AND this row is within 8m" — a
  same-entity compound, not set intersection of independently-built
  Intervals. A far cyclist coexisting with a close pedestrian at the
  same timestamp does NOT match.
- ``.sub_event()`` finalizes the partial as a ``RowGroupedSelector``,
  which is a regular ``TimeSeriesExpression`` leaf. After this point,
  ``&`` / ``|`` compose at the interval level — for cross-entity
  correlation::

      cyclist_close  = ((ot.detection_class == 1) & (ot.distance_m < 8)).sub_event()
      car_decel      = ((ot.detection_class == 2) & (ot.distance_m < 15)
                        & (ot.relative_velocity_ms < -0.5)).sub_event()
      co_occurrence  = cyclist_close & car_decel

  Each side reduces to container-scope ``Intervals``; the ``&`` intersects
  them in time without requiring any single row to satisfy both.
- ``BasicEvent`` / ``GroupedEvent`` constructors auto-call
  ``.sub_event()`` on partials so single-predicate events don't need the
  extra ceremony.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyspark.sql.types as T

from impulse_query_engine.analyze.metadata.tag_expression import TagExpression
from impulse_query_engine.analyze.metadata.time_series_expression import (
    TimeSeriesExpression,
)

if TYPE_CHECKING:
    from impulse_query_engine.analyze.query.solvers.series_cache import SeriesCache
    from impulse_query_engine.surfaces.row_grouped_selector import RowGroupedSelector
    from impulse_query_engine.surfaces.row_grouped_surface import RowGroupedSurface


PredicateFn = Callable[[pd.DataFrame], pd.Series]


class _PartialPredicate(TimeSeriesExpression):
    """An in-progress, fusible predicate against one row-grouped surface."""

    def __init__(
        self,
        surface: "RowGroupedSurface",
        predicate: PredicateFn,
        description: str,
        *,
        group_scope: bool = False,
    ) -> None:
        TimeSeriesExpression.__init__(self, is_single_signal=True)
        self._surface = surface
        self._predicate = predicate
        self._description = description
        self._group_scope = group_scope

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

    # ------------------------------------------------------------------
    # Per-row fusion: same target, same group_scope, fuse clauses.
    # ------------------------------------------------------------------

    def _same_target(self, other: Any) -> bool:
        return (
            isinstance(other, _PartialPredicate)
            and other._surface.name == self._surface.name
            and other._group_scope == self._group_scope
        )

    def _fuse(
        self, other: "_PartialPredicate", combiner: Callable, symbol: str
    ) -> "_PartialPredicate":
        left = self._predicate
        right = other._predicate

        def fused(df: pd.DataFrame) -> pd.Series:
            return combiner(left(df), right(df))

        return _PartialPredicate(
            self._surface,
            fused,
            f"({self._description}) {symbol} ({other._description})",
            group_scope=self._group_scope,
        )

    def __and__(self, other):
        if self._same_target(other):
            return self._fuse(other, operator.and_, "AND")
        # Different surface, different scope, or non-partial operand —
        # finalize this side so the result composes at interval level.
        return self.sub_event() & other

    def __or__(self, other):
        if self._same_target(other):
            return self._fuse(other, operator.or_, "OR")
        return self.sub_event() | other

    # ------------------------------------------------------------------
    # Finalization — the explicit per-row → container-scope boundary.
    # ------------------------------------------------------------------

    def sub_event(self) -> "RowGroupedSelector":
        """Finalize as a leaf that composes at interval level.

        Call this when authoring cross-entity correlation, where each
        side is a complete same-entity predicate and you want the
        composition between sides to be ``Intervals & Intervals`` rather
        than further per-row fusion.

        For single-predicate events this is unnecessary —
        ``BasicEvent`` / ``GroupedEvent`` constructors auto-finalize
        partials.
        """
        from impulse_query_engine.surfaces.row_grouped_selector import (
            RowGroupedSelector,
        )

        return RowGroupedSelector(
            self._surface,
            predicate=self._predicate,
            description=self._description,
            group_scope=self._group_scope,
        )

    # ------------------------------------------------------------------
    # Build path — auto-finalize when evaluated against a cache.
    # ------------------------------------------------------------------

    def build(self, cache: "SeriesCache"):
        return self.sub_event().build(cache)

    # ------------------------------------------------------------------
    # TimeSeriesExpression API surface
    # ------------------------------------------------------------------

    def get_required_tag_exprs(self) -> set[TagExpression]:
        return set()

    def required_tags(self) -> set[str]:
        return set()

    def get_selector_expr(self):
        return None

    def get_selectors(self) -> list:
        # An unfinalized partial isn't a leaf in its own right; report the
        # finalized form so leaf-walking on an unfinalized expression
        # tree still returns sensible RowGroupedSelector leaves.
        return [self.sub_event()]

    def alias(self, alias_name: str) -> "_PartialPredicate":
        # Most call paths auto-finalize before .alias() runs (e.g.
        # BasicEvent.__init__), but support it directly in case a caller
        # aliases an unfinalized partial.
        self._alias = alias_name
        return self

    def __str__(self) -> str:
        scope = ", group_scope=True" if self._group_scope else ""
        return (
            f"_PartialPredicate<{self._surface.name}: {self._description}{scope}>"
        )
