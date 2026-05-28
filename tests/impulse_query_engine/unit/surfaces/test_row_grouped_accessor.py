"""Accessor reflection: numeric columns return predicate proxies; non-numeric
columns raise informative AttributeErrors."""

import pyspark.sql.types as T
import pytest

from impulse_query_engine.surfaces import RowGroupedAccessor, RowGroupedSurface
from impulse_query_engine.surfaces.column_proxies import _NumericColumn
from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate


def _surface_with_string_column():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("ts", T.LongType(), nullable=False),
            T.StructField("object_id", T.LongType(), nullable=False),
            T.StructField("distance_m", T.DoubleType()),
            T.StructField("detection_class", T.StringType()),
            T.StructField("payload", T.ArrayType(T.DoubleType())),
        ]
    )
    return RowGroupedSurface(
        name="object_tracks",
        schema=schema,
        timestamp_col="ts",
        group_col="object_id",
    )


def test_numeric_column_returns_proxy():
    accessor = RowGroupedAccessor(_surface_with_string_column())
    proxy = accessor.distance_m
    assert isinstance(proxy, _NumericColumn)


def test_numeric_comparison_returns_partial_predicate():
    accessor = RowGroupedAccessor(_surface_with_string_column())
    sel = accessor.distance_m < 8.0
    # Column proxies emit _PartialPredicate so subsequent same-surface &
    # / | fuses per-row. Finalize with .sub_event() to get a leaf.
    assert isinstance(sel, _PartialPredicate)
    assert "distance_m" in sel.description
    assert "lt" in sel.description
    assert "8.0" in sel.description
    assert sel.group_scope is False
    assert sel.leaf_kind == "object_tracks"


def test_string_column_raises_attribute_error_naming_column_and_dtype():
    accessor = RowGroupedAccessor(_surface_with_string_column())
    with pytest.raises(AttributeError) as exc:
        _ = accessor.detection_class
    msg = str(exc.value)
    assert "detection_class" in msg
    assert "StringType" in msg


def test_array_column_raises_attribute_error():
    accessor = RowGroupedAccessor(_surface_with_string_column())
    with pytest.raises(AttributeError, match="payload"):
        _ = accessor.payload


def test_unknown_column_raises_attribute_error_listing_available():
    accessor = RowGroupedAccessor(_surface_with_string_column())
    with pytest.raises(AttributeError, match="has no column 'speed'"):
        _ = accessor.speed


def test_group_scope_call_taints_subsequent_selectors():
    surface = _surface_with_string_column()
    accessor = RowGroupedAccessor(surface)
    sel_default = accessor.distance_m < 8.0
    sel_grouped = accessor(group_scope=True).distance_m < 8.0
    assert sel_default.group_scope is False
    assert sel_grouped.group_scope is True
