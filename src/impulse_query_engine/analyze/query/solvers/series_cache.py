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
    def load_blob(self, mid, cid) -> SampleSeries:
        """
        Resolve given mid and cid to a series.

        Parameters
        ----------
        mid : Any
            Container or measurement ID.
        cid : Any
            Channel ID.

        Returns
        -------
        SampleSeries
            The loaded sample series object.
        """
        pass

    def get(self, surface_name: str) -> pd.DataFrame:
        """Return the per-container DataFrame for a registered row-grouped surface.

        Default returns an empty DataFrame so dtype probing through
        ``EmptyTimeSeriesCache`` and channel-only caches doesn't blow up on
        expressions that reference a row-grouped leaf. Concrete caches that
        carry surface data (``MultiSurfaceCache``) override this.
        """
        return pd.DataFrame()


class MultiSurfaceCache(SeriesCache):
    """Cache that holds one pandas DataFrame per registered surface.

    The per-container UDF wraps the slice of each surface's source DataFrame
    that belongs to one container in a ``MultiSurfaceCache`` and hands it to
    the expression tree. ``RowGroupedSelector.build`` reads
    ``cache.get(surface_name)`` to find its rows.

    ``resolve`` / ``load_blob`` are not the access path for row-grouped
    leaves and return empty values; channel leaves go through their own
    cache implementation.
    """

    def __init__(
        self,
        surfaces: dict[str, pd.DataFrame] | None = None,
        *,
        container_stop_ts: float | None = None,
    ):
        self._surfaces: dict[str, pd.DataFrame] = dict(surfaces) if surfaces else {}
        # container_stop_ts is read by RowGroupedSelector.build to close the
        # last row of each group; without it, the last row collapses to a
        # zero-length interval and drops out of the result.
        self.container_stop_ts = container_stop_ts

    def get(self, surface_name: str) -> pd.DataFrame:
        return self._surfaces.get(surface_name, pd.DataFrame())

    def put(self, surface_name: str, df: pd.DataFrame) -> None:
        self._surfaces[surface_name] = df

    def resolve(self, selection) -> pd.DataFrame:
        return pd.DataFrame()

    def load_blob(self, mid, cid) -> SampleSeries:
        return SampleSeries.empty()
