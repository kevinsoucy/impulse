"""Typed column proxies for predicate authoring against series.

The accessor returns a proxy per column. Operator overloads and methods on the
proxy build ``_PartialPredicate`` clauses carrying ``(column op value)``.
Operations Python has a native operator for (``==``, ``!=``, ``<``, ``<=``,
``>``, ``>=``) use overloads; everything else (``isin``, ``contains``,
``startswith``, ``endswith``, ``matches``, ``isnull``, ``notnull``) uses methods.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate
    from impulse_query_engine.surfaces.series import Series


COMPARISON_OPS: dict[str, Callable] = {
    "lt": operator.lt,
    "le": operator.le,
    "eq": operator.eq,
    "ne": operator.ne,
    "ge": operator.ge,
    "gt": operator.gt,
}


def _comparison_predicate(column: str, op: str, value):
    op_fn = COMPARISON_OPS[op]

    def _apply(df):
        return op_fn(df[column], value)

    return _apply


class _ColumnProxy:
    """Base proxy: equality, set membership, and null checks — common to every
    scalar column type."""

    __slots__ = ("_series", "_column")

    def __init__(self, series: "Series", column: str) -> None:
        self._series = series
        self._column = column

    def _partial(self, predicate, description: str) -> "_PartialPredicate":
        from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate

        return _PartialPredicate(self._series, predicate, description)

    def _make_comparison(self, op: str, value) -> "_PartialPredicate":
        return self._partial(
            _comparison_predicate(self._column, op, value),
            f"{self._column} {op} {value!r}",
        )

    def __eq__(self, other) -> "_PartialPredicate":  # type: ignore[override]
        return self._make_comparison("eq", other)

    def __ne__(self, other) -> "_PartialPredicate":  # type: ignore[override]
        return self._make_comparison("ne", other)

    def isin(self, values) -> "_PartialPredicate":
        """Match rows where the column value is in *values*."""
        col = self._column
        values = list(values)

        def _apply(df):
            return df[col].isin(values)

        return self._partial(_apply, f"{col} isin {values!r}")

    def isnull(self) -> "_PartialPredicate":
        """Match rows where the column is null."""
        col = self._column

        def _apply(df):
            return df[col].isnull()

        return self._partial(_apply, f"{col} isnull")

    def notnull(self) -> "_PartialPredicate":
        """Match rows where the column is not null."""
        col = self._column

        def _apply(df):
            return df[col].notnull()

        return self._partial(_apply, f"{col} notnull")

    def __hash__(self) -> int:  # operator overloads disable default __hash__
        return id(self)


class _NumericColumn(_ColumnProxy):
    """Predicate-authoring proxy for a numeric column.

    Adds the four ordering operators (``<``, ``<=``, ``>``, ``>=``) on top of the
    common equality / set-membership / null-check surface.
    """

    __slots__ = ()

    def __lt__(self, other) -> "_PartialPredicate":
        return self._make_comparison("lt", other)

    def __le__(self, other) -> "_PartialPredicate":
        return self._make_comparison("le", other)

    def __ge__(self, other) -> "_PartialPredicate":
        return self._make_comparison("ge", other)

    def __gt__(self, other) -> "_PartialPredicate":
        return self._make_comparison("gt", other)


class _StringColumn(_ColumnProxy):
    """Predicate-authoring proxy for a string column.

    Adds substring / prefix / suffix / regex matching on top of the common
    equality / set-membership / null-check surface. Ordering operators are
    intentionally absent — rarely meaningful on label columns and the pandas
    behaviour is locale-sensitive.
    """

    __slots__ = ()

    def contains(self, value) -> "_PartialPredicate":
        """Match rows where the column contains *value* as a literal substring."""
        col = self._column

        def _apply(df):
            return df[col].str.contains(value, regex=False, na=False)

        return self._partial(_apply, f"{col} contains {value!r}")

    def startswith(self, value) -> "_PartialPredicate":
        """Match rows where the column starts with *value*."""
        col = self._column

        def _apply(df):
            return df[col].str.startswith(value, na=False)

        return self._partial(_apply, f"{col} startswith {value!r}")

    def endswith(self, value) -> "_PartialPredicate":
        """Match rows where the column ends with *value*."""
        col = self._column

        def _apply(df):
            return df[col].str.endswith(value, na=False)

        return self._partial(_apply, f"{col} endswith {value!r}")

    def matches(self, pattern) -> "_PartialPredicate":
        """Match rows where the column matches the regular expression *pattern*."""
        col = self._column

        def _apply(df):
            return df[col].str.contains(pattern, regex=True, na=False)

        return self._partial(_apply, f"{col} matches {pattern!r}")
