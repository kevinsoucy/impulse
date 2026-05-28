"""MultiSurfaceCache holds per-surface DataFrames and exposes the container's
stop timestamp for interval closure."""

import pandas as pd

from impulse_query_engine.analyze.query.solvers.empty_cache import EmptyTimeSeriesCache
from impulse_query_engine.analyze.query.solvers.series_cache import MultiSurfaceCache


def test_get_returns_per_surface_dataframe():
    pdf_a = pd.DataFrame({"x": [1, 2]})
    pdf_b = pd.DataFrame({"y": [3, 4]})
    cache = MultiSurfaceCache({"a": pdf_a, "b": pdf_b})
    assert cache.get("a").equals(pdf_a)
    assert cache.get("b").equals(pdf_b)


def test_missing_surface_returns_empty_dataframe():
    cache = MultiSurfaceCache({"a": pd.DataFrame({"x": [1]})})
    out = cache.get("unknown")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_put_after_construction_overrides():
    cache = MultiSurfaceCache()
    new_pdf = pd.DataFrame({"x": [9]})
    cache.put("a", new_pdf)
    assert cache.get("a").equals(new_pdf)


def test_container_stop_ts_is_optional_attribute():
    cache = MultiSurfaceCache(container_stop_ts=42.0)
    assert cache.container_stop_ts == 42.0
    # Default cache (e.g. EmptyTimeSeriesCache) shouldn't expose one, but the
    # default get() on SeriesCache returns an empty DataFrame so row-grouped
    # selectors fall through gracefully.
    empty = EmptyTimeSeriesCache()
    assert empty.get("x").empty
