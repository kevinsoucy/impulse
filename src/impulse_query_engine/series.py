"""Public import path for the ``Series`` model.

The whitepaper authors series as ``from impulse_query_engine.series import
Series``. The implementation lives under ``surfaces/`` (the registration and
predicate-authoring package); this module re-exports the public names so the
documented import path resolves regardless of internal layout.
"""

from impulse_query_engine.surfaces.series import Series
from impulse_query_engine.surfaces.series_accessor import SeriesAccessor
from impulse_query_engine.surfaces.series_selector import SeriesSelector

__all__ = ["Series", "SeriesAccessor", "SeriesSelector"]
