"""Schema-reflected accessor exposing predicate-authoring proxies for a
registered series."""

from __future__ import annotations

import pyspark.sql.types as T

from impulse_query_engine.surfaces.column_proxies import _NumericColumn, _StringColumn
from impulse_query_engine.surfaces.series import Series

_NUMERIC_SPARK_TYPES: tuple[type, ...] = (
    T.LongType,
    T.IntegerType,
    T.ShortType,
    T.ByteType,
    T.DoubleType,
    T.FloatType,
    T.DecimalType,
)


class SeriesAccessor:
    """Predicate-authoring surface for a series.

    Looks up the named column in the series' schema and returns a typed
    proxy whose operator overloads construct ``SeriesSelector`` leaves.

    Payload columns **and the signal column** are exposed: the signal column is
    a primary filter (e.g. ``sensor_type == "lidar"``), so it is queryable like
    any other column. The remaining structural roles (``session_col``, the
    time-axis columns, ``entity_key``) are not predicate surfaces and raise
    ``AttributeError`` if accessed. Of the exposed columns, numeric columns
    return a ``_NumericColumn`` proxy
    and string columns a ``_StringColumn`` proxy; other non-scalar types (arrays,
    structs, maps) raise ``AttributeError`` naming the column and its Spark type.
    """

    def __init__(self, series: Series) -> None:
        if series.schema is None:
            raise ValueError(
                f"Series {series.name!r} has no resolved schema; an accessor "
                "needs the schema (provide it at construction or resolve it at "
                "registration before authoring predicates)."
            )
        self._series = series
        self._fields_by_name: dict[str, T.StructField] = {f.name: f for f in series.schema.fields}

    @property
    def series(self) -> Series:
        return self._series

    def __getattr__(self, name: str):
        # __getattr__ is only called when normal lookup fails, so internal
        # attributes (_series, _fields_by_name) never route here.
        fields = self.__dict__.get("_fields_by_name", {})
        series = self.__dict__["_series"]
        if name not in fields:
            raise AttributeError(
                f"Series {series.name!r} has no column {name!r}; "
                f"available columns: {sorted(fields)}"
            )
        # The signal column is a queryable filter; the other structural roles
        # (session / time-axis / entity_key) are not predicate surfaces.
        if name != series.signal_col and name in series.structural_cols:
            raise AttributeError(
                f"Column {name!r} on series {series.name!r} is mapped to a "
                "structural role (session/time/entity_key) and is not exposed as "
                "a predicate proxy. (The signal column is queryable.)"
            )
        spark_type = fields[name].dataType
        if isinstance(spark_type, _NUMERIC_SPARK_TYPES):
            return _NumericColumn(series, name)
        if isinstance(spark_type, T.StringType):
            return _StringColumn(series, name)
        raise AttributeError(
            f"Column {name!r} on series {series.name!r} has Spark type "
            f"{type(spark_type).__name__}; only numeric and string columns are "
            f"reachable through the accessor."
        )
