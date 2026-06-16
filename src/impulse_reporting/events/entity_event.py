"""Per-entity event: extends BasicEvent so the matching entities' identities
show up in the fact table's ``entity_key`` column as an alias-keyed, signal-
scoped JSON map ``{alias: {signal: [entity_key, ...]}}`` — one bucket per
``.ids(as_=alias)`` leaf, with signal retained as part of the entity identity."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import partial
from typing import Any

import numpy as np
import pandas as pd
import pyspark.sql.functions as f
import pyspark.sql.types as T

from impulse_query_engine.analyze.metadata.time_series_expression import (
    TimeSeriesExpression,
    TimeSeriesOp,
)
from impulse_query_engine.analyze.query.solvers.series_cache import SeriesCache
from impulse_query_engine.model.series.intervals import Intervals
from impulse_query_engine.surfaces.series_selector import SeriesSelector, render_entity_key
from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.persist.fact_schema import EVENT_INSTANCE_FACT_SCHEMA
from impulse_reporting.util.event_instance_util import generate_event_instance_id_column
from impulse_reporting.util.report_entity_util import ReportEntityUtil


def _id_sort_key(value: str):
    """Sort numeric-looking ids numerically, everything else lexicographically."""
    body = value[1:] if value.startswith("-") else value
    if body.isdigit():
        return (0, int(value))
    return (1, value)


def _serialize_roster(roster: Mapping[str, Mapping[str, list[str]]]) -> str:
    """Serialize ``{alias: {signal: [ids]}}`` to the JSON ``entity_key`` column.

    Alias and signal keys are emitted in sorted order for determinism; the id
    lists are already rendered, de-duplicated, sorted, and truncated by
    ``_roster``. Signal stays in the key because it is part of the entity
    identity — the same id under two sensors must not conflate.
    """
    return json.dumps(
        {
            alias: {signal: roster[alias][signal] for signal in sorted(roster[alias])}
            for alias in sorted(roster)
        }
    )


def _eval_tree(
    expr: TimeSeriesExpression,
    leaf_intervals: dict[int, Intervals],
    cache: "SeriesCache",
    build_cache: dict[int, Any] | None = None,
) -> Any:
    """Evaluate an expression tree under a per-entity interval assignment.

    Leaves that are pinned for the current entity combination (the entity-scoped
    ``SeriesSelector`` leaves, and any series-presence leaves) are substituted
    from *leaf_intervals* (keyed by ``id``). Any other leaf — a channel /
    scalar-signal leaf (``veh_speed > 50``, ``sign_30_present``) — is evaluated
    normally against *cache*, which (in the cogroup path) is a
    ``CombinedSeriesCache`` carrying the channel rows. Channel leaves do not vary
    per entity, so their ``build`` results are memoized in *build_cache* (keyed by
    ``id``) and reused across every entity combination in the same container.
    """
    if id(expr) in leaf_intervals:
        return leaf_intervals[id(expr)]
    if isinstance(expr, TimeSeriesOp):
        args = [
            (
                _eval_tree(a, leaf_intervals, cache, build_cache)
                if isinstance(a, TimeSeriesExpression)
                else a
            )
            for a in expr.args
        ]
        kwargs = {
            k: (
                _eval_tree(v, leaf_intervals, cache, build_cache)
                if isinstance(v, TimeSeriesExpression)
                else v
            )
            for k, v in expr.kwargs.items()
        }
        # Reuse the op's own dispatch (shared with TimeSeriesOp.build) so this
        # evaluator can't drift on optype handling, and a build-overriding op
        # like TimeSeriesUDF is invoked the same way build() would.
        return expr.apply_op(args, kwargs)
    # Not a pinned leaf and not an op: a channel/scalar leaf → evaluate via cache.
    # These do not vary per entity, so memoize across combinations when asked.
    if build_cache is None:
        return expr.build(cache)
    if id(expr) not in build_cache:
        build_cache[id(expr)] = expr.build(cache)
    return build_cache[id(expr)]


class EntityEvent(BasicEvent):
    """An event whose matching windows carry the participating entities.

    Where ``BasicEvent`` answers "did this combination of predicates hold
    anywhere in this session?", ``EntityEvent`` answers "which entities, and
    with what windowing?". It is selected by composing ``.each()`` and/or
    ``.ids()`` leaves; the matched ids land in the ``entity_key`` column as an
    alias-keyed, signal-scoped JSON map ``{alias: {signal: [entity_key, ...]}}``
    (NULL for a window whose leaves are not projected with ``.ids()``).

    Windowing is per-leaf, not per-event:

    - ``.any()`` / ``.any().ids()`` — one merged window (union of the matching
      entities).
    - ``.each()`` — one window per entity. At most **one** ``.each()`` leaf per
      event: two would enumerate the entity cross-product. ``.any()`` legs
      collapse, so an ``.any() & .each()`` event emits one row per ``.each()``
      entity, each carrying the ``.any()`` roster.
    """

    def get_event_type_str(self) -> str:
        return "ENTITY_EVENT"

    def _finalize_bare(self, expr: TimeSeriesExpression) -> TimeSeriesExpression:
        """An EntityEvent never auto-finalizes a bare partial — the windowing
        verb is the whole point. Tell the author which one to add."""
        from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate

        if isinstance(expr, _PartialPredicate):
            raise ValueError(
                "EntityEvent needs an entity-scoped leaf: finalize the predicate "
                "with .each() (per-entity) or .any().ids(as_=…) (roster) before "
                "building the event."
            )
        return expr

    def _validate_expression(self, expr: TimeSeriesExpression) -> None:
        self._assert_no_unfinalized_partials(expr)
        series_leaves = self._series_leaves(expr)
        if not series_leaves:
            raise ValueError(
                "EntityEvent requires at least one SeriesSelector leaf in its expression"
            )
        entity_leaves = self._entity_leaves(expr)
        if not entity_leaves:
            raise ValueError(
                "EntityEvent requires at least one entity-scoped leaf (.each() or "
                ".ids()); for a presence-only condition use BasicEvent."
            )
        split_leaves = [leaf for leaf in entity_leaves if leaf.per_entity_windowing]
        if len(split_leaves) > 1:
            names = [leaf.id_alias or leaf.series.name for leaf in split_leaves]
            raise ValueError(
                f"At most one .each() leaf per event (got {len(split_leaves)}: "
                f"{names}). Multiple .each() legs would enumerate the entity "
                "cross-product; use .any() on all but one side, or enumerate the "
                "pairs in SQL downstream."
            )
        projected = [leaf.id_alias for leaf in entity_leaves if leaf.id_alias is not None]
        dupes = sorted({a for a in projected if projected.count(a) > 1})
        if dupes:
            raise ValueError(
                f"Duplicate .ids(as_=…) alias(es) {dupes} in one event; each "
                "projected leaf needs a distinct alias (it keys entity_key)."
            )

    @staticmethod
    def _assert_no_unfinalized_partials(expr: TimeSeriesExpression) -> None:
        """Guard the leaf-substitution invariant.

        ``materialize_per_container`` pins each leaf's intervals by ``id(leaf)``
        and substitutes them into the tree in ``_eval_tree``. That only works
        because every series leaf in the tree is a finalized ``SeriesSelector``
        with stable identity. A surviving ``_PartialPredicate`` would hand back a
        *fresh* selector from ``get_selectors()`` whose ``id()`` never matches the
        tree node, so its entity scope would be silently lost. Composition and the
        constructor always finalize partials before this point; assert it so a
        future change that breaks that invariant fails loudly here instead.
        """
        from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate

        if isinstance(expr, _PartialPredicate):
            raise AssertionError(
                "EntityEvent expression still holds an unfinalized _PartialPredicate; "
                "it must be finalized via .each()/.any()/.ids() before materialization."
            )
        if isinstance(expr, TimeSeriesOp):
            for arg in (*expr.args, *expr.kwargs.values()):
                if isinstance(arg, TimeSeriesExpression):
                    EntityEvent._assert_no_unfinalized_partials(arg)

    def materialize_per_container(
        self,
        container_id: int,
        cache: SeriesCache,
    ) -> list[tuple[int, float, float, str | None]]:
        """Evaluate the expression for one container and return its fact rows.

        Returns ``(container_id, start_ts, end_ts, entity_key_json)`` tuples.
        ``entity_key_json`` is the alias-keyed roster, or ``None`` when no leaf in
        the matched window is projected with ``.ids()``. A ``.each()`` leaf emits
        one window per entity; otherwise one merged window.

        *cache* resolves every leaf for this container — registered-series leaves
        via ``cache.get(series_name)`` (raw frames) or the pre-reduced cogroup
        lookups, and channel / scalar-signal leaves via the wrapped channel cache.
        ``container_stop_ts`` (point-in-time frame synthesis) lives on the cache.
        """
        expr = self.expression
        series_leaves = self._series_leaves(expr)
        entity_leaves = [leaf for leaf in series_leaves if leaf.entity_scoped]
        presence_leaves = [leaf for leaf in series_leaves if not leaf.entity_scoped]
        split_leaves = [leaf for leaf in entity_leaves if leaf.per_entity_windowing]
        merged_leaves = [leaf for leaf in entity_leaves if not leaf.per_entity_windowing]

        presence_iv = {id(leaf): leaf.build(cache) for leaf in presence_leaves}
        entity_iv = {
            id(leaf): leaf.entity_intervals(cache.get(leaf.series.name), cache)
            for leaf in entity_leaves
        }

        # Base substitution shared by every emitted window: presence leaves as
        # built, merged entity leaves as the union of their per-entity intervals.
        base_sub = dict(presence_iv)
        for leaf in merged_leaves:
            union = Intervals.empty()
            for iv in entity_iv[id(leaf)].values():
                union = union | iv
            base_sub[id(leaf)] = union

        if not split_leaves:
            return self._emit_windows(
                container_id, expr, base_sub, None, None, merged_leaves, entity_iv, cache
            )

        # Exactly one .each() leg (validation guarantees ≤ 1): iterate its
        # entities, each producing its own window(s). Channel-leaf builds don't
        # vary per entity, so memoize them across iterations.
        split = split_leaves[0]
        rows: list[tuple[int, float, float, str | None]] = []
        build_cache: dict[int, Any] = {}
        for split_key, iv in entity_iv[id(split)].items():
            sub = dict(base_sub)
            sub[id(split)] = iv
            rows.extend(
                self._emit_windows(
                    container_id,
                    expr,
                    sub,
                    split,
                    split_key,
                    merged_leaves,
                    entity_iv,
                    cache,
                    build_cache,
                )
            )
        return rows

    def _emit_windows(
        self,
        container_id,
        expr,
        sub,
        split_leaf,
        split_key,
        merged_leaves,
        entity_iv,
        cache,
        build_cache=None,
    ) -> list[tuple[int, float, float, str | None]]:
        """Evaluate the tree under *sub* and emit one row per result window, each
        with its roster computed against that window."""
        result = _eval_tree(expr, sub, cache, build_cache)
        if not isinstance(result, Intervals) or len(result) == 0:
            return []
        rows: list[tuple[int, float, float, str | None]] = []
        for tstart, tend in zip(result.tstarts, result.tends, strict=False):
            window = Intervals(np.array([tstart]), np.array([tend]))
            roster = self._roster(window, split_leaf, split_key, merged_leaves, entity_iv)
            payload = _serialize_roster(roster) if roster else None
            rows.append((container_id, float(tstart), float(tend), payload))
        return rows

    @staticmethod
    def _roster(
        window, split_leaf, split_key, merged_leaves, entity_iv
    ) -> dict[str, dict[str, list[str]]]:
        """Alias-keyed, signal-scoped entity ids for one emitted *window*:
        ``{alias: {signal: [ids]}}``.

        The ``.each()`` leg (if projected) contributes its single iteration
        entity under its signal; each merged ``.any().ids()`` leg contributes the
        entities whose interval overlaps the window, capped at ``id_limit`` ids
        total for the alias (sorted deterministically, then regrouped by signal).
        Signal stays in the key — it is part of the entity identity, so the same
        id under two sensors stays distinct. Leaves with no ``id_alias`` (and
        aliases that match nothing in this window) contribute nothing; a window
        with no contribution is emitted with a NULL ``entity_key``.
        """
        roster: dict[str, dict[str, list[str]]] = {}
        if split_leaf is not None and split_leaf.id_alias is not None:
            signal, entity = split_key
            roster[split_leaf.id_alias] = {str(signal): [render_entity_key(entity)]}
        for leaf in merged_leaves:
            if leaf.id_alias is None:
                continue
            hits = {
                (str(signal), render_entity_key(entity))
                for (signal, entity), iv in entity_iv[id(leaf)].items()
                if len(iv & window) > 0
            }
            capped = sorted(hits, key=lambda se: (se[0], _id_sort_key(se[1])))[: leaf.id_limit]
            by_signal: dict[str, list[str]] = {}
            for signal, ent in capped:
                by_signal.setdefault(signal, []).append(ent)
            if by_signal:
                roster[leaf.id_alias] = by_signal
        return roster

    # ------------------------------------------------------------------
    # Production report pipeline integration
    # ------------------------------------------------------------------
    #
    # EntityEvent does not ride the centralized ``solved_df`` (one interval-array
    # column per event) like BasicEvent, because it emits *entity-attributed*
    # rows, not a single presence interval set. Like ContainerEvent it has its
    # own ``determine_events`` path: it runs the series cogroup
    # (``QueryBuilder.solve_series_apply``) and, per container, calls
    # ``materialize_per_container`` to produce fact rows with a populated
    # ``entity_key``. The cogroup assembly (per-container series frames + channel
    # cache + ``container_stop_ts``) lives in the query engine; the per-container
    # row producer is supplied here, so the layering stays one-directional.

    @classmethod
    def determine_events(
        cls,
        spark,
        events: list["EntityEvent"],
        *,
        solved_df=None,
        query=None,
        solver=None,
        pre_filtered_containers_df=None,
    ):
        """Materialize entity-attributed fact rows for a list of EntityEvents.

        Runs one cogroup over the union of every event's referenced series; each
        event reads only the frames its expression references, so a single pass
        serves all events. ``solved_df`` is unused (EntityEvent does not ride the
        centralized presence solve); ``query``/``solver`` drive the cogroup.
        """
        cid_col = solver.config.container_id_col
        schema = T.StructType(
            [
                T.StructField(cid_col, T.LongType()),
                T.StructField("event_name", T.StringType()),
                T.StructField("start_ts", T.DoubleType()),
                T.StructField("end_ts", T.DoubleType()),
                T.StructField("entity_key", T.StringType()),
            ]
        )

        per_container = partial(cls._materialize_fact_rows, events=events, cid_col=cid_col)
        event_query = query.select(*[event.expression for event in events])
        df = event_query.solve_series_apply(
            spark,
            solver,
            per_container=per_container,
            schema=schema,
            pre_filtered_containers_df=pre_filtered_containers_df,
        )

        return (
            df.withColumn("start_ts", f.col("start_ts").cast("long"))
            .withColumn("end_ts", f.col("end_ts").cast("long"))
            .withColumn(
                "event_instance_id",
                generate_event_instance_id_column(event_type=cls, entity_key_col="entity_key"),
            )
            .withColumn(
                "event_id",
                ReportEntityUtil.get_event_id_column(elements=events, element_name="event_name"),
            )
            .select(EVENT_INSTANCE_FACT_SCHEMA.fieldNames())
            .where(f.col("start_ts") < f.col("end_ts"))
        )

    @staticmethod
    def _materialize_fact_rows(container_id, cache, *, events, cid_col) -> pd.DataFrame:
        """``per_container`` producer for the cogroup: evaluate every event for one
        container and return its entity-attributed fact rows.

        A static function bound via ``functools.partial`` (not a local closure) so
        it pickles cleanly into the Spark UDF. *cache* resolves every leaf — the
        reduced series interval sets and any channel leaves.
        """
        columns = [cid_col, "event_name", "start_ts", "end_ts", "entity_key"]
        rows: list[tuple] = []
        for event in events:
            name = event.get_name()
            for cid, start_ts, end_ts, entity_key in event.materialize_per_container(
                container_id, cache
            ):
                rows.append((cid, name, float(start_ts), float(end_ts), entity_key))
        return pd.DataFrame(rows, columns=columns)
