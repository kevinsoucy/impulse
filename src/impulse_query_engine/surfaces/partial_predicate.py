"""In-progress predicate against a series — composes per-row via ``&`` / ``|``
until it is finalized as a leaf.

Authoring story:

- Column proxies (``accessor.distance_m < 8.0``) return ``_PartialPredicate``.
- Same-series ``&`` / ``|`` between partials fuses clauses into a single per-row
  predicate. ``(class == "cyclist") & (dist < 8)`` evaluates as "this row is a
  cyclist AND this row is within 8m" — a same-entity compound, not set
  intersection of independently-built Intervals. A far cyclist coexisting with a
  close pedestrian at the same timestamp does NOT match.
- ``.entity_condition()`` finalizes the partial as an **entity-scoped**
  ``SeriesSelector``: per-entity coherence, evaluated per ``(signal, entity)``.
  Composing two ``.entity_condition()`` results expresses cross-entity
  co-occurrence::

      cyclist_close = ((ot.detection_class == "cyclist") & (ot.distance_m < 8)).entity_condition()
      car_close     = ((ot.detection_class == "car") & (ot.distance_m < 8)).entity_condition()
      squeeze       = cyclist_close & car_close

  Each side reduces to container-scope ``Intervals``; the ``&`` intersects them
  in time without requiring any single row to satisfy both.
- A partial finalized **without** ``.entity_condition()`` (the default when a
  ``BasicEvent`` auto-finalizes, or when a partial is built directly) is a
  presence check: "does this condition hold anywhere?" — no per-entity scope.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

import pandas as pd

from impulse_query_engine.analyze.metadata.tag_expression import TagExpression
from impulse_query_engine.analyze.metadata.time_series_expression import (
    TimeSeriesExpression,
)

if TYPE_CHECKING:
    from impulse_query_engine.analyze.query.solvers.series_cache import SeriesCache
    from impulse_query_engine.surfaces.series import Series
    from impulse_query_engine.surfaces.series_selector import SeriesSelector


PredicateFn = Callable[[pd.DataFrame], pd.Series]


class _PartialPredicate(TimeSeriesExpression):
    """An in-progress, fusible predicate against one series."""

    def __init__(
        self,
        series: "Series",
        predicate: PredicateFn,
        description: str,
    ) -> None:
        TimeSeriesExpression.__init__(self, is_single_signal=True)
        self._series = series
        self._predicate = predicate
        self._description = description

    @property
    def description(self) -> str:
        return self._description

    @property
    def leaf_kind(self) -> str:
        return self._series.name

    # ------------------------------------------------------------------
    # Per-row fusion: same target series, fuse clauses.
    # ------------------------------------------------------------------

    def _same_target(self, other: Any) -> bool:
        return isinstance(other, _PartialPredicate) and other._series.name == self._series.name

    def _fuse(
        self, other: "_PartialPredicate", combiner: Callable, symbol: str
    ) -> "_PartialPredicate":
        left = self._predicate
        right = other._predicate

        def fused(df: pd.DataFrame) -> pd.Series:
            return combiner(left(df), right(df))

        return _PartialPredicate(
            self._series,
            fused,
            f"({self._description}) {symbol} ({other._description})",
        )

    def __and__(self, other):
        if self._same_target(other):
            return self._fuse(other, operator.and_, "AND")
        # Different series or non-partial operand — finalize this side as a
        # presence leaf so the result composes at interval level.
        return self._finalize_presence() & other

    def __or__(self, other):
        if self._same_target(other):
            return self._fuse(other, operator.or_, "OR")
        return self._finalize_presence() | other

    # ------------------------------------------------------------------
    # Finalization — the per-row → container-scope boundary.
    # ------------------------------------------------------------------

    def entity_condition(self) -> "SeriesSelector":
        """Finalize as an entity-scoped leaf enforcing per-entity coherence.

        All clauses within the expression must hold for the **same entity** at
        the same timestamp; evaluation partitions rows by ``(signal, entity)``.
        Composing two entity conditions expresses cross-entity co-occurrence.

        Raises
        ------
        ValueError
            If the series has no ``entity_key`` — per-entity coherence is
            meaningless without an entity identity.
        """
        if self._series.entity_key is None:
            raise ValueError(
                f"Series {self._series.name!r} has no entity_key; "
                ".entity_condition() requires a per-entity identity. Use a plain "
                "predicate (no .entity_condition()) for a presence check."
            )
        return self._to_selector(entity_scoped=True)

    def _finalize_presence(self) -> "SeriesSelector":
        """Finalize as a presence leaf (no per-entity scope)."""
        return self._to_selector(entity_scoped=False)

    def _to_selector(self, *, entity_scoped: bool) -> "SeriesSelector":
        from impulse_query_engine.surfaces.series_selector import SeriesSelector

        return SeriesSelector(
            self._series,
            predicate=self._predicate,
            description=self._description,
            entity_scoped=entity_scoped,
        )

    # ------------------------------------------------------------------
    # Build path — auto-finalize (as presence) when evaluated against a cache.
    # ------------------------------------------------------------------

    def build(self, cache: "SeriesCache"):
        return self._finalize_presence().build(cache)

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
        # finalized presence form so leaf-walking on an unfinalized expression
        # tree still returns sensible SeriesSelector leaves.
        return [self._finalize_presence()]

    def __str__(self) -> str:
        return f"_PartialPredicate<{self._series.name}: {self._description}>"
