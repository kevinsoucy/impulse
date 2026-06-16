"""Series — declarative registration and predicate authoring for external
tables shaped as ``(container_id, timestamp, [entity_key], wide row)``."""

from impulse_query_engine.surfaces.series import Series
from impulse_query_engine.surfaces.series_accessor import SeriesAccessor
from impulse_query_engine.surfaces.series_selector import SeriesSelector

__all__ = [
    "Series",
    "SeriesAccessor",
    "SeriesSelector",
]
