"""QueryBuilder that buckets selectors into channel vs perception leaves
before the solver runs.

Channel selectors flow through the existing 6-stage filter pipeline
(``filter_container_tags`` → ``filter_container_metrics`` → …).  Perception
selectors short-circuit that path (their ``get_selector_expr()`` returns
``F.lit(False)`` so they never match a channel) and are routed to
``PerceptionSolver`` in stage 6 alongside the cogrouped ``object_tracks``
rows.
"""

from __future__ import annotations

from mda_query_engine.analyze.metadata.time_series_expression import (
    TimeSeriesExpression,
    TimeSeriesSelector,
)
from mda_query_engine.analyze.query.query_builder import QueryBuilder

from lakevision.tsal.perception_selector import PerceptionSelector


class PerceptionQueryBuilder(QueryBuilder):
    """QueryBuilder that knows about ``PerceptionSelector`` leaves.

    Exposes ``object_track`` so authors can write ``ot = db.query.object_track``
    and compose predicates without importing ``ObjectTrackAccessor`` directly.
    """

    @property
    def object_track(self):
        from lakevision.tsal.object_track_accessor import ObjectTrackAccessor

        return ObjectTrackAccessor()

    def _collect_time_series_selectors(self, uses_alias=None) -> list[TimeSeriesSelector]:
        """Override the base collector to exclude perception leaves.

        Perception leaves are tracked separately via
        ``_collect_perception_selectors`` and routed to ``PerceptionSolver``.
        Without this filter, the channel-side filter pipeline would try to
        match them against ``channel_metrics`` (and get an empty set from
        ``F.lit(False)``, but at the cost of a join per leaf).
        """
        selectors: list[TimeSeriesSelector] = []
        seen_selector_ids: set[int] = set()
        for expression in self.selections:
            if not isinstance(expression, TimeSeriesExpression):
                continue
            for selector in expression.get_selectors():
                if isinstance(selector, PerceptionSelector):
                    continue
                if uses_alias is not None and selector.uses_alias != uses_alias:
                    continue
                if selector.selector_id in seen_selector_ids:
                    continue
                seen_selector_ids.add(selector.selector_id)
                selectors.append(selector)
        return selectors

    def _collect_perception_selectors(self) -> list[PerceptionSelector]:
        """Return the deduplicated perception leaves across all selections."""
        selectors: list[PerceptionSelector] = []
        seen: set[int] = set()
        for expression in self.selections:
            if not isinstance(expression, TimeSeriesExpression):
                continue
            for selector in expression.get_selectors():
                if not isinstance(selector, PerceptionSelector):
                    continue
                if selector.selector_id in seen:
                    continue
                seen.add(selector.selector_id)
                selectors.append(selector)
        return selectors

    def has_perception(self) -> bool:
        return len(self._collect_perception_selectors()) > 0
