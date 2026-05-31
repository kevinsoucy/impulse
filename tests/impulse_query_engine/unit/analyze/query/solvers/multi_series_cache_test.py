"""MultiSeriesCache holds per-series DataFrames and exposes the container's
stop timestamp for interval closure."""

import pandas as pd

from impulse_query_engine.analyze.query.solvers.empty_cache import EmptyTimeSeriesCache
from impulse_query_engine.analyze.query.solvers.series_cache import (
    CombinedSeriesCache,
    MultiSeriesCache,
    SeriesCache,
)


def test_get_returns_per_series_dataframe():
    pdf_a = pd.DataFrame({"x": [1, 2]})
    pdf_b = pd.DataFrame({"y": [3, 4]})
    cache = MultiSeriesCache({"a": pdf_a, "b": pdf_b})
    assert cache.get("a").equals(pdf_a)
    assert cache.get("b").equals(pdf_b)


def test_missing_series_returns_empty_dataframe():
    cache = MultiSeriesCache({"a": pd.DataFrame({"x": [1]})})
    out = cache.get("unknown")
    assert isinstance(out, pd.DataFrame)
    assert out.empty


def test_put_after_construction_overrides():
    cache = MultiSeriesCache()
    new_pdf = pd.DataFrame({"x": [9]})
    cache.put("a", new_pdf)
    assert cache.get("a").equals(new_pdf)


def test_container_stop_ts_is_optional_attribute():
    cache = MultiSeriesCache(container_stop_ts=42.0)
    assert cache.container_stop_ts == 42.0


def test_default_get_returns_empty_dataframe():
    # A cache that carries no series data (the SeriesCache default, e.g.
    # EmptyTimeSeriesCache) returns an empty DataFrame from get() so series
    # selectors fall through gracefully rather than raising.
    empty = EmptyTimeSeriesCache()
    assert empty.get("x").empty


# --- CombinedSeriesCache (channel leaves + registered-series leaves) ---------


class _RecordingChannelCache(SeriesCache):
    """Channel cache stand-in that records delegation."""

    def __init__(self):
        self.resolved = []
        self.loaded = []

    def resolve(self, selection):
        self.resolved.append(selection)
        return pd.DataFrame({"candidate": [selection]})

    def load_blob(self, mid, cid):
        self.loaded.append((mid, cid))
        return f"blob:{mid}:{cid}"


def test_combined_routes_resolve_and_load_blob_to_channel_cache():
    channel = _RecordingChannelCache()
    cache = CombinedSeriesCache(channel, {"object_tracks": pd.DataFrame({"x": [1]})})
    assert cache.resolve("sel").iloc[0]["candidate"] == "sel"
    assert cache.load_blob(7, 3) == "blob:7:3"
    assert channel.resolved == ["sel"]
    assert channel.loaded == [(7, 3)]


def test_combined_routes_get_to_series_frames_not_channel_cache():
    channel = _RecordingChannelCache()
    pdf = pd.DataFrame({"distance_m": [5.0]})
    cache = CombinedSeriesCache(channel, {"object_tracks": pdf})
    assert cache.get("object_tracks").equals(pdf)
    # Channel cache must not have been consulted for a series leaf.
    assert channel.resolved == []


def test_combined_missing_series_returns_empty_frame():
    cache = CombinedSeriesCache(_RecordingChannelCache(), {"object_tracks": pd.DataFrame({"x": [1]})})
    out = cache.get("traffic_signs")
    assert isinstance(out, pd.DataFrame) and out.empty


def test_combined_exposes_container_stop_ts_and_put():
    cache = CombinedSeriesCache(_RecordingChannelCache(), container_stop_ts=99.0)
    assert cache.container_stop_ts == 99.0
    pdf = pd.DataFrame({"y": [2]})
    cache.put("lanes", pdf)
    assert cache.get("lanes").equals(pdf)


def test_combined_with_no_series_still_resolves_channels():
    channel = _RecordingChannelCache()
    cache = CombinedSeriesCache(channel, None)
    assert cache.get("anything").empty
    cache.load_blob(1, 1)
    assert channel.loaded == [(1, 1)]
