"""Row-grouped query surfaces — declarative registration and predicate
authoring for external tables shaped as ``(container_id, timestamp,
[group_col], wide row)``."""

from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate
from impulse_query_engine.surfaces.row_grouped_accessor import RowGroupedAccessor
from impulse_query_engine.surfaces.row_grouped_selector import RowGroupedSelector
from impulse_query_engine.surfaces.row_grouped_surface import RowGroupedSurface

__all__ = [
    "RowGroupedAccessor",
    "RowGroupedSelector",
    "RowGroupedSurface",
    "_PartialPredicate",
]
