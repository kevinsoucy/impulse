from abc import ABC, abstractmethod

import pandas as pd

from impulse_query_engine.model.series.sample_series import SampleSeries


class SeriesCache(ABC):
    @abstractmethod
    def resolve(self, selection) -> pd.DataFrame:
        """
        Resolve selected tags/metrics to a list of candidates.

        Parameters
        ----------
        selection : Any
            The selection object specifying tags or metrics.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the resolved candidates.
        """
        pass

    @abstractmethod
    def load_blob(self, mid, cid, uses_alias: bool = False) -> SampleSeries:
        """
        Resolve given mid and cid to a series.

        Parameters
        ----------
        mid : Any
            Container or measurement ID.
        cid : Any
            Channel ID.
        uses_alias : bool, optional
            ``True`` when the calling selector resolves the channel via a
            ``channel_mapping`` alias.  Caches that perform unit conversion
            (:class:`ChannelTimeSeriesCache` on the KVS path) only apply the
            per-channel conversion factor when this is ``True``, so a direct selector
            on the same physical channel always returns raw values.
            Defaults to ``False`` (direct / no-conversion semantics).

        Returns
        -------
        SampleSeries
            The loaded sample series object.
        """
        pass

    def get(self, series_name: str) -> pd.DataFrame:
        """Return the per-container DataFrame for a registered series.

        Default returns an empty DataFrame so dtype probing through
        ``EmptyTimeSeriesCache`` and channel-only caches doesn't blow up on
        expressions that reference a series leaf. Concrete caches that
        carry series data (``MultiSeriesCache``) override this.
        """
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Pre-reduced interval lookups
    # ------------------------------------------------------------------
    #
    # When a Spark per-entity reduction stage runs ahead of the cogroup, the
    # per-container worker no longer holds a series' raw rows — it holds the
    # already-synthesized interval sets, keyed by each leaf's stable
    # ``_reduce_key``. ``SeriesSelector.build`` / ``entity_intervals`` consult
    # these first and only fall back to raw-frame synthesis when a cache returns
    # ``None`` (the default — every non-reduced cache path is unchanged).

    def reduced_presence(self, leaf) -> "object | None":
        """Pre-reduced presence ``Intervals`` for *leaf*, or ``None`` if this
        cache carries no reduction for it (fall back to raw-frame synthesis)."""
        return None

    def reduced_entities(self, leaf) -> "dict | None":
        """Pre-reduced ``{(signal, entity): Intervals}`` for an entity-scoped
        *leaf*, or ``None`` if this cache carries no reduction for it."""
        return None


class ChannelTimeSeriesCache(SeriesCache):
    """In-memory channel cache for the cogroup path, shared by all grouped-map
    solvers.

    Holds one container's channel rows as a metadata frame (``mdf``, one row per
    ``(container, channel)``) plus the sorted sample frame (``pdf``), and resolves
    selections / loads blobs against them. The backend (Delta vs key-value store)
    only affects how the rows are read upstream; once they are a pandas frame the
    caching behaviour is identical, so a single class serves every solver
    (``DeltaSolver`` / ``KeyValueStoreSolver`` return it from
    ``_channel_cache_cls``).
    """

    def __init__(self, pdf, col_map: dict[str, str]):
        """
        Initialize the ChannelTimeSeriesCache.

        Parameters
        ----------
        pdf : pd.DataFrame
            DataFrame containing time series data.
        col_map : dict[str, str]
            Mapping with keys ``"cid"``, ``"ch"``, ``"ts"``, ``"te"``,
            ``"val"`` to the actual column names in *pdf*.
        """
        self._cid_col = col_map["cid"]
        self._ch_col = col_map["ch"]
        self._ts_col = col_map["ts"]
        self._te_col = col_map["te"]
        self._val_col = col_map["val"]
        # Optional per-channel unit-conversion factor. Present only on the KVS
        # solve path with a unit_conversion table configured; the column rides
        # in the channel metadata, so it stays in mdf/pdf (not in the drop list).
        self._conv_col = col_map.get("conv")
        self._has_conversion = self._conv_col is not None and self._conv_col in pdf.columns

        meta = pdf.drop(columns=[self._ts_col, self._te_col, self._val_col])
        self.mdf = meta.drop_duplicates(subset=[self._cid_col, self._ch_col]).reset_index()
        self.pdf = pdf.sort_values([self._cid_col, self._ch_col, self._ts_col]).reset_index()

    def resolve(self, selection) -> pd.DataFrame:
        """
        Resolve selected tags/metrics to a list of candidates.

        Parameters
        ----------
        selection : Any
            The selection object specifying tags or metrics.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the resolved candidates.
        """
        if "selector_ids" in self.mdf.columns:
            idx = self.mdf["selector_ids"].apply(
                lambda arr: arr is not None and selection.selector_id in arr
            )
            return self.mdf[idx]
        idx = selection._expr.build_pandas(self.mdf)
        return self.mdf[idx]

    def load_blob(self, mid, cid, uses_alias: bool = False) -> SampleSeries:
        """
        Load a time series blob from the DataFrame.

        When the rows carry a unit-conversion factor (``col_map["conv"]``) **and**
        the caller resolved the channel through an alias (``uses_alias=True``),
        values are multiplied by that per-channel factor. A direct selector on
        the same physical channel (``uses_alias=False``) always returns raw
        values — conversion is a property of the alias, not the channel.

        Parameters
        ----------
        mid : Any
            Container or measurement ID.
        cid : Any
            Channel ID.
        uses_alias : bool, optional
            ``True`` when the calling selector resolved via ``channel_mapping``.
            Gates the per-channel conversion factor; defaults to ``False``.

        Returns
        -------
        SampleSeries
            The loaded sample series object.
        """
        s = self.pdf[(self.pdf[self._cid_col] == mid) & (self.pdf[self._ch_col] == cid)]
        values = s[self._val_col]
        if self._has_conversion and len(s) > 0 and uses_alias:
            factor = s[self._conv_col].iloc[0]
            if pd.notna(factor):
                values = values * factor
        return SampleSeries(s[self._ts_col], s[self._te_col], values)


class MultiSeriesCache(SeriesCache):
    """Cache that holds one pandas DataFrame per registered series.

    This is the reference implementation of series-leaf resolution: it carries a
    series' raw per-container rows so ``SeriesSelector.build`` can synthesize
    intervals directly from them (``cache.get(series_name)``). The production
    cogroup does **not** take this path — it reduces each series to interval sets
    in a distributed Spark stage and resolves leaves via
    ``reduced_presence`` / ``reduced_entities`` (see :class:`CombinedSeriesCache`);
    the integration tests assert that path is result-identical to building from
    raw frames here. As the cache backing that raw-frame oracle, this class is
    exercised by the surface/authoring unit tests, not the live report pipeline.

    ``resolve`` / ``load_blob`` are not the access path for series leaves
    and return empty values; channel leaves go through their own cache
    implementation.
    """

    def __init__(
        self,
        series: dict[str, pd.DataFrame] | None = None,
        *,
        container_stop_ts: float | None = None,
    ):
        self._series: dict[str, pd.DataFrame] = dict(series) if series else {}
        # container_stop_ts is read by SeriesSelector.build to close the
        # last row of each entity; without it, the last row collapses to a
        # zero-length interval and drops out of the result.
        self.container_stop_ts = container_stop_ts

    def get(self, series_name: str) -> pd.DataFrame:
        return self._series.get(series_name, pd.DataFrame())

    def put(self, series_name: str, df: pd.DataFrame) -> None:
        self._series[series_name] = df

    def resolve(self, selection) -> pd.DataFrame:
        return pd.DataFrame()

    def load_blob(self, mid, cid, uses_alias: bool = False) -> SampleSeries:
        return SampleSeries.empty()


class CombinedSeriesCache(SeriesCache):
    """Resolve channel leaves *and* registered-series leaves from one cache.

    In the cogroup path a container's rows arrive from two places: the channels
    table (wrapped by the shared ``ChannelTimeSeriesCache``) and one or more
    registered series (one pandas
    frame each). A single expression may reference both kinds of leaf, but
    ``selection.build`` takes one cache. This cache routes by access path:

    - ``resolve`` / ``load_blob`` (channel leaves) delegate to the channel cache;
    - ``get(series_name)`` (registered-series leaves) reads the per-series frame.

    A series with no rows for this container yields an empty frame, so a
    conjunction over it correctly does not fire (rather than raising).
    """

    def __init__(
        self,
        channel_cache: SeriesCache,
        series: dict[str, pd.DataFrame] | None = None,
        *,
        container_stop_ts: float | None = None,
        reduced_presence: dict | None = None,
        reduced_entities: dict | None = None,
    ):
        self._channel_cache = channel_cache
        self._series: dict[str, pd.DataFrame] = dict(series) if series else {}
        # Read by SeriesSelector.build to close the last row of each entity;
        # without it the last row collapses to a zero-length interval.
        self.container_stop_ts = container_stop_ts
        # Pre-reduced interval sets keyed by leaf ``_reduce_key``. When
        # present, SeriesSelector short-circuits to these instead of synthesizing
        # from raw frames. Empty/None means "no reduction" → raw-frame path.
        self._reduced_presence: dict = reduced_presence or {}
        self._reduced_entities: dict = reduced_entities or {}

    def resolve(self, selection) -> pd.DataFrame:
        return self._channel_cache.resolve(selection)

    def load_blob(self, mid, cid, uses_alias: bool = False) -> SampleSeries:
        return self._channel_cache.load_blob(mid, cid, uses_alias)

    def get(self, series_name: str) -> pd.DataFrame:
        return self._series.get(series_name, pd.DataFrame())

    def put(self, series_name: str, df: pd.DataFrame) -> None:
        self._series[series_name] = df

    def reduced_presence(self, leaf):
        key = getattr(leaf, "_reduce_key", None)
        return self._reduced_presence.get(key) if key is not None else None

    def reduced_entities(self, leaf):
        key = getattr(leaf, "_reduce_key", None)
        return self._reduced_entities.get(key) if key is not None else None
