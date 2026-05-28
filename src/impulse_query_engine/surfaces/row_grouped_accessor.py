"""Schema-reflected accessor exposing predicate-authoring proxies for a
registered row-grouped surface."""

from __future__ import annotations

import pyspark.sql.types as T

from impulse_query_engine.surfaces.column_proxies import _NumericColumn
from impulse_query_engine.surfaces.row_grouped_surface import RowGroupedSurface


_NUMERIC_SPARK_TYPES: tuple[type, ...] = (
    T.LongType,
    T.IntegerType,
    T.ShortType,
    T.ByteType,
    T.DoubleType,
    T.FloatType,
    T.DecimalType,
)


class RowGroupedAccessor:
    """Predicate-authoring surface for a row-grouped table.

    Looks up the named column in the surface's schema and returns a typed
    proxy whose operator overloads construct ``RowGroupedSelector`` leaves.

    Only numeric columns are reachable in this pass. String and other
    non-scalar columns raise ``AttributeError`` naming the column and its
    Spark type so the message is informative when a user reaches for a
    column the engine can't predicate against yet.

    Per-entity windowing is opt-in via ``accessor(group_scope=True)`` which
    returns a tainted accessor whose selectors carry ``group_scope=True``;
    a ``GroupedEvent`` requires every leaf to agree.
    """

    def __init__(self, surface: RowGroupedSurface, *, group_scope: bool = False) -> None:
        self._surface = surface
        self._group_scope = group_scope
        self._fields_by_name: dict[str, T.StructField] = {
            f.name: f for f in surface.schema.fields
        }

    @property
    def surface(self) -> RowGroupedSurface:
        return self._surface

    def __call__(self, *, group_scope: bool = False) -> "RowGroupedAccessor":
        """Return a new accessor whose proxies emit ``group_scope=True`` selectors.

        Used to author per-entity predicates inside a ``GroupedEvent``::

            obj = RowGroupedAccessor(OBJECT_TRACKS)
            obj(group_scope=True).distance_m < 8.0
        """
        return RowGroupedAccessor(self._surface, group_scope=group_scope)

    def __getattr__(self, name: str):
        # __getattr__ is only called when normal lookup fails, so internal
        # attributes (_surface, _group_scope, _fields_by_name) never route here.
        fields = self.__dict__.get("_fields_by_name", {})
        if name not in fields:
            raise AttributeError(
                f"Surface {self._surface.name!r} has no column {name!r}; "
                f"available columns: {sorted(fields)}"
            )
        spark_type = fields[name].dataType
        if not isinstance(spark_type, _NUMERIC_SPARK_TYPES):
            raise AttributeError(
                f"Column {name!r} on surface {self._surface.name!r} has Spark type "
                f"{type(spark_type).__name__}; only numeric columns are reachable "
                f"through the accessor today."
            )
        return _NumericColumn(self._surface, name, group_scope=self._group_scope)
