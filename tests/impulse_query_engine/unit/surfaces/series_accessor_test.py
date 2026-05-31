"""Accessor reflection: only non-structural numeric/string columns are exposed
as predicate proxies; structural and non-scalar columns raise informative
AttributeErrors. Covers the full operator set."""

import pandas as pd
import pyspark.sql.types as T
import pytest

from impulse_query_engine.surfaces import Series, SeriesAccessor
from impulse_query_engine.surfaces.column_proxies import _NumericColumn, _StringColumn
from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate


def _object_tracks():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType(), nullable=False),
            T.StructField("sensor_type", T.StringType(), nullable=False),
            T.StructField("frame_ts", T.LongType(), nullable=False),
            T.StructField("entity_id", T.LongType(), nullable=False),
            T.StructField("distance_m", T.DoubleType()),
            T.StructField("detection_class", T.StringType()),
            T.StructField("payload", T.ArrayType(T.DoubleType())),
        ]
    )
    return Series(
        name="object_tracks",
        schema=schema,
        session_col="container_id",
        signal_col="sensor_type",
        timestamp_col="frame_ts",
        entity_key="entity_id",
    )


def _accessor():
    return SeriesAccessor(_object_tracks())


def _mask(partial, df):
    return list(partial._predicate(df))


# --- proxy types ------------------------------------------------------------


def test_numeric_column_returns_proxy():
    assert isinstance(_accessor().distance_m, _NumericColumn)


def test_string_column_returns_proxy():
    assert isinstance(_accessor().detection_class, _StringColumn)


def test_numeric_comparison_returns_partial_predicate():
    sel = _accessor().distance_m < 8.0
    assert isinstance(sel, _PartialPredicate)
    assert "distance_m" in sel.description and "lt" in sel.description
    assert sel.leaf_kind == "object_tracks"


def test_string_equality_returns_partial_predicate():
    sel = _accessor().detection_class == "cyclist"
    assert isinstance(sel, _PartialPredicate)
    assert "detection_class" in sel.description and "cyclist" in sel.description


# --- structural-column exclusion --------------------------------------------


@pytest.mark.parametrize("col", ["container_id", "sensor_type", "frame_ts", "entity_id"])
def test_structural_columns_not_exposed(col):
    with pytest.raises(AttributeError, match="structural role"):
        getattr(_accessor(), col)


def test_rle_time_columns_not_exposed():
    schema = T.StructType(
        [
            T.StructField("container_id", T.LongType()),
            T.StructField("sensor_type", T.StringType()),
            T.StructField("seg_start", T.LongType()),
            T.StructField("seg_end", T.LongType()),
            T.StructField("value", T.DoubleType()),
        ]
    )
    s = Series(
        name="rle",
        schema=schema,
        session_col="container_id",
        signal_col="sensor_type",
        tstart_col="seg_start",
        tend_col="seg_end",
    )
    acc = SeriesAccessor(s)
    for col in ("seg_start", "seg_end"):
        with pytest.raises(AttributeError, match="structural role"):
            getattr(acc, col)
    assert isinstance(acc.value, _NumericColumn)  # only payload column exposed


# --- numeric operator set ---------------------------------------------------


def test_numeric_comparisons_filter_correctly():
    acc = _accessor()
    df = pd.DataFrame({"distance_m": [1.0, 5.0, 8.0, 12.0]})
    assert _mask(acc.distance_m < 8.0, df) == [True, True, False, False]
    assert _mask(acc.distance_m <= 8.0, df) == [True, True, True, False]
    assert _mask(acc.distance_m > 8.0, df) == [False, False, False, True]
    assert _mask(acc.distance_m >= 8.0, df) == [False, False, True, True]
    assert _mask(acc.distance_m == 5.0, df) == [False, True, False, False]
    assert _mask(acc.distance_m != 5.0, df) == [True, False, True, True]


def test_numeric_isin_filters_correctly():
    df = pd.DataFrame({"distance_m": [1.0, 5.0, 8.0]})
    assert _mask(_accessor().distance_m.isin([1.0, 8.0]), df) == [True, False, True]


def test_numeric_null_checks():
    df = pd.DataFrame({"distance_m": [1.0, None, 3.0]})
    assert _mask(_accessor().distance_m.isnull(), df) == [False, True, False]
    assert _mask(_accessor().distance_m.notnull(), df) == [True, False, True]


# --- string operator set ----------------------------------------------------


def test_string_eq_ne_isin():
    acc = _accessor()
    df = pd.DataFrame({"detection_class": ["cyclist", "car", "pedestrian", "truck"]})
    assert _mask(acc.detection_class == "cyclist", df) == [True, False, False, False]
    assert _mask(acc.detection_class != "car", df) == [True, False, True, True]
    assert _mask(acc.detection_class.isin(["cyclist", "pedestrian"]), df) == [
        True,
        False,
        True,
        False,
    ]


def test_string_substring_prefix_suffix_regex():
    acc = _accessor()
    df = pd.DataFrame({"detection_class": ["cyclist", "pedestrian", "P0420", "post"]})
    assert _mask(acc.detection_class.contains("ed"), df) == [False, True, False, False]
    assert _mask(acc.detection_class.startswith("cy"), df) == [
        True,
        False,
        False,
        False,
    ]
    assert _mask(acc.detection_class.endswith("st"), df) == [True, False, False, True]
    assert _mask(acc.detection_class.matches("^P0[0-9]"), df) == [
        False,
        False,
        True,
        False,
    ]


def test_string_null_checks():
    df = pd.DataFrame({"detection_class": ["cyclist", None, "car"]})
    acc = _accessor()
    assert _mask(acc.detection_class.isnull(), df) == [False, True, False]
    assert _mask(acc.detection_class.notnull(), df) == [True, False, True]


# --- error cases ------------------------------------------------------------


def test_array_column_raises_attribute_error_naming_type():
    with pytest.raises(AttributeError) as exc:
        _ = _accessor().payload
    msg = str(exc.value)
    assert "payload" in msg and "ArrayType" in msg


def test_unknown_column_raises_attribute_error_listing_available():
    with pytest.raises(AttributeError, match="has no column 'speed'"):
        _ = _accessor().speed


def test_string_column_has_no_ordering_operators():
    # Ordering (<, <=, >, >=) is intentionally absent on string columns; attempting
    # it falls through to Python's default and raises, rather than silently building
    # a locale-sensitive comparison.
    acc = _accessor()
    for op in (lambda c: c < "x", lambda c: c <= "x", lambda c: c > "x", lambda c: c >= "x"):
        with pytest.raises(TypeError):
            op(acc.detection_class)
