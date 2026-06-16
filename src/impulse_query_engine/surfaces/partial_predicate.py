"""In-progress predicate against a series — composes per-row via ``&`` / ``|``
until it is finalized as a leaf.

Authoring story:

- Column proxies (``accessor.distance_m < 8.0``) return ``_PartialPredicate``.
- Same-series ``&`` / ``|`` between partials fuses clauses into a single per-row
  predicate. ``(class == "cyclist") & (dist < 8)`` evaluates as "this row is a
  cyclist AND this row is within 8m" — a same-entity compound, not set
  intersection of independently-built Intervals. A far cyclist coexisting with a
  close pedestrian at the same timestamp does NOT match.
- A partial must be finalized by a windowing verb before it can compose across
  series or back an event. Two orthogonal axes:
    - ``.any()`` — presence / merged window: "does any matching row exist?".
      No per-entity scope. Works on any series.
    - ``.each()`` — per-entity: one window per matching ``(signal, entity)``.
      Requires an ``entity_key``.
  Either can be followed by ``.ids(as_=…)`` to also project the matched entity
  ids into the event ``entity_key`` (a roster for ``.any()``, the single id for
  ``.each()``). Composing two finalized leaves correlates them at the interval
  level — cross-entity co-occurrence::

      cyclist_close = ((ot.detection_class == "cyclist") & (ot.distance_m < 8)).any()
      car_close     = ((ot.detection_class == "car") & (ot.distance_m < 8)).any()
      squeeze       = cyclist_close & car_close

  Each side reduces to container-scope ``Intervals``; the ``&`` intersects them
  in time without requiring any single row to satisfy both.
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


def _and_signal_values(a, b):
    """Combine signal constraints under AND: a row must satisfy both clauses, so
    intersect the allowed signal sets. ``None`` means "unconstrained" (the clause
    doesn't pin the signal), so it contributes no restriction."""
    if a is None:
        return b
    if b is None:
        return a
    return a & b


def _or_signal_values(a, b):
    """Combine signal constraints under OR: a row may satisfy either clause, so
    union the allowed sets — but if either side is unconstrained, the result is
    unconstrained (a row of any signal could match)."""
    if a is None or b is None:
        return None
    return a | b


class _PartialPredicate(TimeSeriesExpression):
    """An in-progress, fusible predicate against one series."""

    def __init__(
        self,
        series: "Series",
        predicate: PredicateFn,
        description: str,
        signal_values: frozenset | None = None,
    ) -> None:
        TimeSeriesExpression.__init__(self, is_single_signal=True)
        self._series = series
        self._predicate = predicate
        self._description = description
        # Enumerable signal-column constraint for source-read pruning, or None
        # when the predicate does not pin the signal. See series_selector.py.
        self._signal_values = signal_values
        # Memoized presence finalization (see _finalize_presence).
        self._presence_selector: "SeriesSelector | None" = None

    @property
    def signal_values(self) -> frozenset | None:
        return self._signal_values

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
        self,
        other: "_PartialPredicate",
        combiner: Callable,
        symbol: str,
        sig_combine: Callable,
    ) -> "_PartialPredicate":
        left = self._predicate
        right = other._predicate

        def fused(df: pd.DataFrame) -> pd.Series:
            return combiner(left(df), right(df))

        return _PartialPredicate(
            self._series,
            fused,
            f"({self._description}) {symbol} ({other._description})",
            signal_values=sig_combine(self._signal_values, other._signal_values),
        )

    def __and__(self, other):
        if self._same_target(other):
            return self._fuse(other, operator.and_, "AND", _and_signal_values)
        # Different series or non-partial operand — finalize this side as a
        # presence leaf so the result composes at interval level.
        return self._finalize_presence() & other

    def __or__(self, other):
        if self._same_target(other):
            return self._fuse(other, operator.or_, "OR", _or_signal_values)
        return self._finalize_presence() | other

    # ------------------------------------------------------------------
    # Finalization — the per-row → container-scope boundary.
    # ------------------------------------------------------------------

    def any(self) -> "SeriesSelector":
        """Finalize as a presence leaf — "does any matching row exist?".

        One merged window over the whole population; no entity identity is
        projected (``entity_key`` stays NULL for this leaf). Works on any series.
        Chain ``.ids(as_=…)`` to also emit the matched entity ids as a roster
        (that requires an ``entity_key``).
        """
        # Reuse the memoized presence selector so leaf collection and build share
        # one instance (keeps the solver-stamped ``_reduce_key`` visible — see
        # ``_finalize_presence``).
        return self._finalize_presence()

    def each(self) -> "SeriesSelector":
        """Finalize as a per-entity leaf — one window per matching entity.

        Partitions evaluation by ``(signal, entity)`` so each entity yields its
        own window. Composing two finalized leaves correlates them in time
        without requiring any single row to satisfy both. Chain ``.ids(as_=…)``
        to emit the entity id on each row.

        Raises
        ------
        ValueError
            If the series has no ``entity_key`` — per-entity windowing needs an
            entity identity. Use ``.any()`` for a presence check.
        """
        if self._series.entity_key is None:
            raise ValueError(
                f"Series {self._series.name!r} has no entity_key; "
                ".each() requires a per-entity identity. Use .any() for a "
                "presence check."
            )
        return self._to_selector(per_entity_windowing=True)

    def _finalize_presence(self) -> "SeriesSelector":
        """Finalize as a presence leaf (no per-entity scope).

        Memoized so leaf collection (``get_selectors``) and evaluation
        (``build``) return the *same* selector instance. The cogroup stamps a
        per-solve ``_reduce_key`` on the collected leaf; if ``build`` finalized a
        fresh selector instead, its ``_reduce_key`` would be ``None`` → the
        reduced cache lookup misses → the cache falls back to an empty frame and
        the predicate silently never matches. Sharing one instance keeps the
        stamped key visible at build time (it is pickled along with this partial,
        which holds the selector as an attribute).
        """
        if self._presence_selector is None:
            self._presence_selector = self._to_selector()
        return self._presence_selector

    def _to_selector(self, *, per_entity_windowing: bool = False) -> "SeriesSelector":
        from impulse_query_engine.surfaces.series_selector import SeriesSelector

        return SeriesSelector(
            self._series,
            predicate=self._predicate,
            description=self._description,
            per_entity_windowing=per_entity_windowing,
            signal_values=self._signal_values,
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
