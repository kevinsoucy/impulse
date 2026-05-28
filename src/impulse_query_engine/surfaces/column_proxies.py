"""Typed column proxies for predicate authoring against row-grouped surfaces.

The accessor returns a proxy per column. Operator overloads on the proxy
build ``_PartialPredicate`` clauses carrying ``(column op value)``. Only
numeric columns are reachable today; non-scalar / string columns raise
``AttributeError`` from the accessor.
"""

from __future__ import annotations

import operator
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate
    from impulse_query_engine.surfaces.row_grouped_surface import RowGroupedSurface


NUMERIC_OPS: dict[str, Callable] = {
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    "ge": operator.ge,
    "gt": operator.gt,
}


def _clause_predicate(column: str, op: str, value):
    op_fn = NUMERIC_OPS[op]

    def _apply(df):
        return op_fn(df[column], value)

    return _apply


class _NumericColumn:
    """Predicate-authoring proxy for a numeric column on a row-grouped surface.

    Operator overloads return a ``_PartialPredicate`` — single-clause —
    that fuses with other same-surface, same-group_scope partials via
    ``&`` / ``|``.
    """

    __slots__ = ("_surface", "_column", "_group_scope")

    def __init__(
        self,
        surface: "RowGroupedSurface",
        column: str,
        *,
        group_scope: bool = False,
    ) -> None:
        self._surface = surface
        self._column = column
        self._group_scope = group_scope

    def _make_partial(self, op: str, value) -> "_PartialPredicate":
        from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate

        return _PartialPredicate(
            self._surface,
            _clause_predicate(self._column, op, value),
            f"{self._column} {op} {value!r}",
            group_scope=self._group_scope,
        )

    def __lt__(self, other) -> "_PartialPredicate":
        return self._make_partial("lt", other)

    def __le__(self, other) -> "_PartialPredicate":
        return self._make_partial("le", other)

    def __eq__(self, other) -> "_PartialPredicate":  # type: ignore[override]
        return self._make_partial("eq", other)

    def __ne__(self, other) -> "_PartialPredicate":  # type: ignore[override]
        return self._make_partial("ne", other)

    def __ge__(self, other) -> "_PartialPredicate":
        return self._make_partial("ge", other)

    def __gt__(self, other) -> "_PartialPredicate":
        return self._make_partial("gt", other)

    def __hash__(self) -> int:  # operator overloads disable default __hash__
        return id(self)
