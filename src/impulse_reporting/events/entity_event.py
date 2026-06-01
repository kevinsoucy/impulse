"""Per-entity event: extends BasicEvent so the matching entity's identity
shows up in the fact table's ``entity_key`` column as a nested JSON map
``{table_name: {signal: [entity_key, ...]}}``."""

from __future__ import annotations

import itertools
import json
from collections.abc import Mapping
from functools import partial
from typing import Any

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


def _entity_id_str(entity: Any) -> str:
    """Cast an entity key to string. Delegates to the canonical query-engine
    renderer so the raw-frame and Spark-reduced paths agree."""
    return render_entity_key(entity)


def _id_sort_key(value: str):
    """Sort numeric-looking ids numerically, everything else lexicographically."""
    body = value[1:] if value.startswith("-") else value
    if body.isdigit():
        return (0, int(value))
    return (1, value)


def _serialize_entity_map(entity_map: Mapping[str, Mapping[str, list]]) -> str:
    """Serialize ``{table: {signal: [ids]}}`` to the JSON ``entity_key`` column.

    Every id is cast to string; per-signal id lists are de-duplicated and sorted;
    table and signal keys are emitted in sorted order for determinism.
    """
    out: dict[str, dict[str, list[str]]] = {}
    for table in sorted(entity_map):
        out[table] = {}
        for signal in sorted(entity_map[table]):
            ids = {str(v) for v in entity_map[table][signal]}
            out[table][signal] = sorted(ids, key=_id_sort_key)
    return json.dumps(out)


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
            _eval_tree(a, leaf_intervals, cache, build_cache)
            if isinstance(a, TimeSeriesExpression)
            else a
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
    """An event whose matching windows are emitted per entity.

    Where ``BasicEvent`` answers "did this combination of predicates hold
    anywhere in this session?", ``EntityEvent`` answers "which specific
    entity / entity set triggered each matched window?". The matching entities
    are materialized into the ``entity_key`` column on the event fact table as a
    nested JSON map ``{table_name: {signal: [entity_key, ...]}}``.

    The expression must contain at least one entity-scoped leaf (a
    ``.entity_condition()``); presence and scalar leaves may participate too.

    ``per_entity_windowing`` (default ``True``) emits one row per entity
    participation / co-occurring entity set, each with its own interval. ``False``
    emits one combined row per matched window with the union map of all
    participating entities.
    """

    def __init__(
        self,
        name: str,
        expr: TimeSeriesExpression,
        desc: str | None = None,
        required_channels: list[str] | None = None,
        attributes: Mapping[str, str] | None = None,
        per_entity_windowing: bool = True,
    ):
        from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate

        if isinstance(expr, _PartialPredicate):
            # A bare partial in an EntityEvent is per-entity by intent; finalize
            # as an entity condition (raises if the series has no entity_key).
            expr = expr.entity_condition()
        self._per_entity_windowing = per_entity_windowing
        self._validate(expr)
        super().__init__(
            name=name,
            expr=expr,
            desc=desc,
            required_channels=required_channels,
            attributes=attributes,
        )

    def get_event_type_str(self) -> str:
        return "ENTITY_EVENT"

    @property
    def per_entity_windowing(self) -> bool:
        return self._per_entity_windowing

    def _definition_str(self) -> str:
        """Fold ``per_entity_windowing`` into the definition string.

        It is result-affecting: ``True`` emits one fact row per entity with a
        per-entity ``entity_key``; ``False`` emits one row per window with the
        combined entity-set map. Two events with the same expression but
        different windowing produce different facts and different
        ``event_instance_id`` values, so flipping it must register as a
        redefinition (different ``definition_hash``) rather than an unchanged
        upsert that strands the prior rows.
        """
        return f"{super()._definition_str()}|per_entity_windowing={self._per_entity_windowing}"

    @staticmethod
    def _series_leaves(expr: TimeSeriesExpression) -> list[SeriesSelector]:
        return [s for s in expr.get_selectors() if isinstance(s, SeriesSelector)]

    @staticmethod
    def _entity_leaves(expr: TimeSeriesExpression) -> list[SeriesSelector]:
        return [s for s in EntityEvent._series_leaves(expr) if s.entity_scoped]

    def _validate(self, expr: TimeSeriesExpression) -> None:
        self._assert_no_unfinalized_partials(expr)
        series_leaves = self._series_leaves(expr)
        if not series_leaves:
            raise ValueError(
                "EntityEvent requires at least one SeriesSelector leaf in its expression"
            )
        if not self._entity_leaves(expr):
            raise ValueError(
                "EntityEvent requires at least one entity-scoped leaf "
                "(use .entity_condition()); for a presence-only condition use BasicEvent."
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
                "it must be finalized via .entity_condition() before materialization."
            )
        if isinstance(expr, TimeSeriesOp):
            for arg in (*expr.args, *expr.kwargs.values()):
                if isinstance(arg, TimeSeriesExpression):
                    EntityEvent._assert_no_unfinalized_partials(arg)

    def materialize_per_container(
        self,
        container_id: int,
        cache: SeriesCache,
    ) -> list[tuple[int, float, float, str]]:
        """Evaluate the expression per entity for one container and return rows.

        *cache* resolves every leaf of the expression for this container:

        - registered-series leaves via ``cache.get(series_name)`` (raw frames) or,
          when a Spark per-entity reduction ran ahead of the cogroup, via
          the cache's pre-reduced interval lookups — ``SeriesSelector`` consults
          the reduction first and falls back to raw-frame synthesis;
        - channel / scalar-signal leaves (e.g. ``veh_speed > 50``) via the channel
          cache wrapped inside it.

        Build the cache with :class:`CombinedSeriesCache` — for raw frames pass the
        per-series frames and ``container_stop_ts``; the cogroup builds the reduced
        variant. ``container_stop_ts`` (read for point-in-time frame synthesis)
        lives on the cache.

        Returns ``(container_id, start_ts, end_ts, entity_key_json)`` tuples — one
        per matched window per entity (``per_entity_windowing=True``) or one per
        matched window with the union map (``per_entity_windowing=False``).
        """
        expr = self.expression
        entity_leaves = self._entity_leaves(expr)
        presence_leaves = [leaf for leaf in self._series_leaves(expr) if not leaf.entity_scoped]

        presence_iv = {id(leaf): leaf.build(cache) for leaf in presence_leaves}
        entity_iv = {
            id(leaf): leaf.entity_intervals(cache.get(leaf.series.name), cache)
            for leaf in entity_leaves
        }

        if self._per_entity_windowing:
            return self._materialize_per_entity(
                container_id, expr, entity_leaves, entity_iv, presence_iv, cache
            )
        return self._materialize_combined(
            container_id, expr, entity_leaves, entity_iv, presence_iv, cache
        )

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
                generate_event_instance_id_column(
                    event_type=cls, entity_key_col="entity_key"
                ),
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

    def _materialize_per_entity(
        self, container_id, expr, entity_leaves, entity_iv, presence_iv, cache
    ) -> list[tuple[int, float, float, str]]:
        rows: list[tuple[int, float, float, str]] = []
        leaf_keys = [list(entity_iv[id(leaf)].keys()) for leaf in entity_leaves]
        if any(len(keys) == 0 for keys in leaf_keys):
            # An entity condition matched no entities — the conjunction cannot fire.
            return rows
        # Channel leaves don't vary per entity combination; memoize their build
        # results once per container and reuse across every combination below.
        build_cache: dict[int, Any] = {}
        for combo in itertools.product(*leaf_keys):
            sub = dict(presence_iv)
            entity_map: dict[str, dict[str, list[str]]] = {}
            for leaf, key in zip(entity_leaves, combo, strict=False):
                sub[id(leaf)] = entity_iv[id(leaf)][key]
                signal, entity = key
                entity_map.setdefault(leaf.series.name, {}).setdefault(str(signal), []).append(
                    _entity_id_str(entity)
                )
            result = _eval_tree(expr, sub, cache, build_cache)
            if not isinstance(result, Intervals) or len(result) == 0:
                continue
            payload = _serialize_entity_map(entity_map)
            for tstart, tend in zip(result.tstarts, result.tends, strict=False):
                rows.append((container_id, float(tstart), float(tend), payload))
        return rows

    def _materialize_combined(
        self, container_id, expr, entity_leaves, entity_iv, presence_iv, cache
    ) -> list[tuple[int, float, float, str]]:
        sub = dict(presence_iv)
        entity_map: dict[str, dict[str, list[str]]] = {}
        for leaf in entity_leaves:
            combined = Intervals.empty()
            for (signal, entity), iv in entity_iv[id(leaf)].items():
                combined = combined | iv
                entity_map.setdefault(leaf.series.name, {}).setdefault(str(signal), []).append(
                    _entity_id_str(entity)
                )
            sub[id(leaf)] = combined
        result = _eval_tree(expr, sub, cache)
        if not isinstance(result, Intervals) or len(result) == 0:
            return []
        payload = _serialize_entity_map(entity_map)
        return [
            (container_id, float(tstart), float(tend), payload)
            for tstart, tend in zip(result.tstarts, result.tends, strict=False)
        ]
