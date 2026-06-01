from __future__ import annotations

import abc
from abc import ABC
from functools import partial
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyspark.sql.functions as F
import pyspark.sql.types as T
from pyspark.sql import DataFrame
from pyspark.sql.column import Column
from pyspark.sql.window import Window

from impulse_query_engine.model.series.intervals import Intervals
from impulse_query_engine.surfaces.series_selector import render_entity_key

from .empty_cache import EmptyTimeSeriesCache
from .series_cache import CombinedSeriesCache, SeriesCache

if TYPE_CHECKING:
    from impulse_query_engine.measurement_db import MeasurementDB

from impulse_query_engine.analyze.metadata.time_series_expression import (
    TimeSeriesSelector,
)

from .solver_config import SolverConfig


class QuerySolver(ABC):
    """Abstract base class for query solvers.

    Defines a 6-stage filter pipeline that all solvers must implement:
    filter_container_tags -> filter_container_metrics -> filter_channel_tags ->
    filter_channel_metrics -> filter_candidates -> solve.

    ``filter_container_metrics`` must return a DataFrame that includes **all
    columns** needed for container dimensions and event bounds (e.g.
    ``container_id``, ``start_ts``/``stop_ts`` or ``start_dt``/``stop_dt``),
    not only ``container_id``.
    """

    # Whether this solver can resolve registered-series leaves (the cogroup
    # path: per-series reduction + channel cache, keyed off a SolverConfig). The
    # grouped-map solvers (Delta, KVS) set this True; BlobSolver leaves it False
    # — it has none of that machinery — so a registered-series query is rejected
    # at the QueryBuilder boundary with a clear message instead of failing deep
    # in the reduction.
    supports_registered_series: bool = False

    def __init__(self, config: SolverConfig = None):
        self.config = config or SolverConfig()

    @staticmethod
    def _apply_column_mapping(df: DataFrame, mapping: dict[str, str]) -> DataFrame:
        """Rename DataFrame columns according to a physical → internal mapping."""
        for physical, internal in mapping.items():
            if physical != internal:
                df = df.withColumnRenamed(physical, internal)
        return df

    def _build_expr(self, filters):
        """
        Build a combined selector expression from a list of filter expressions.

        Parameters
        ----------
        filters : list
            List of filter expressions.
            Example: [MetricOp<or_(MetricOp<and_(MetricOp<eq(MetricSelector<vehicle_key>,Seat_Leon)>]
        Returns
        -------
        pyspark.sql.Column or None
            Combined selector expression or None if no filters.
        """
        expr = None
        for filt in filters:
            if expr is None:
                expr = filt.get_selector_expr()
            else:
                expr = expr | filt.get_selector_expr()
        return expr

    def _empty_channel_match_df(self, spark) -> DataFrame:
        return spark.createDataFrame(
            [],
            schema=T.StructType(
                [
                    T.StructField(self.config.container_id_col, T.LongType()),
                    T.StructField(self.config.channel_id_col, T.LongType()),
                    T.StructField("selector_ids", T.ArrayType(T.IntegerType())),
                ]
            ),
        )

    def _build_selector_id_expr(self, filters) -> Column:
        """Build a Spark ``Column`` that maps rows to their ``selector_id``.

        Produces a chained ``F.when`` expression: for each selector in
        *filters*, if the row satisfies the selector's tag expression the
        column evaluates to that selector's ``selector_id``.

        Parameters
        ----------
        filters : Iterable[TimeSeriesSelector]
            Selectors whose ``get_selector_expr()`` and ``selector_id`` are
            used to build the ``WHEN … THEN …`` chain.

        Returns
        -------
        pyspark.sql.Column
            A column expression suitable for ``df.withColumn("selector_id", …)``.
        """
        selector_expr = None
        for selection in filters:
            if selector_expr is None:
                selector_expr = F.when(selection.get_selector_expr(), F.lit(selection.selector_id))
            else:
                selector_expr = selector_expr.when(
                    selection.get_selector_expr(), F.lit(selection.selector_id)
                )
        return selector_expr

    # ------------------------------------------------------------------
    # Shared single-table (grouped-map) solve path
    # ------------------------------------------------------------------
    #
    # The Delta and KVS solvers run an identical per-container
    # ``groupBy(container_id).apply(pandas_udf)``: build one ``SeriesCache``
    # over the container's channel rows, evaluate each selection against it,
    # and emit one result row. The only thing that differs is *which*
    # ``SeriesCache`` subclass wraps the rows. That is the per-solver hook
    # (``_build_channel_cache``); everything else lives here so the two
    # solvers — and the cogroup path that will reuse this same cache hook —
    # do not duplicate the UDF, schema, and serialization wiring.

    @staticmethod
    def _serialize_built(res):
        """Serialize a built selection result to its DataFrame column payload.

        Built results such as ``Intervals`` expose ``serialize`` / ``get_data``;
        plain scalar values pass through unchanged.
        """
        if hasattr(res, "serialize") and callable(res.serialize):
            return res.serialize()
        if hasattr(res, "get_data") and callable(res.get_data):
            return res.get_data()
        return res

    @staticmethod
    def _grouped_map_solve_udf(pdf, selections, col_map, cache_cls):
        """Per-container UDF body: wrap rows in *cache_cls* and evaluate each
        selection. Shared verbatim by every grouped-map solver."""
        cache = cache_cls(pdf, col_map=col_map)
        cid_col = col_map["cid"]
        result = {cid_col: [pdf[cid_col].iloc[0]]}
        for s in selections:
            result[s._alias] = [QuerySolver._serialize_built(s.build(cache))]
        return pd.DataFrame(result)

    def _result_schema(self, selections, dtypes) -> T.StructType:
        """``(container_id, <one column per selection alias>)`` output schema."""
        entries = [T.StructField(self.config.container_id_col, T.LongType())]
        for s, dtype in zip(selections, dtypes, strict=False):
            entries.append(T.StructField(s._alias, dtype))
        return T.StructType(entries)

    def _grouped_map_udf(self, selections, dtypes, cache_cls: type[SeriesCache]):
        """Build the grouped-map ``pandas_udf`` and its output schema.

        Parameters
        ----------
        selections, dtypes
            The query's selections and their resolved Spark dtypes.
        cache_cls
            The per-solver :class:`SeriesCache` subclass (the only thing that
            varies between the Delta and KVS solve paths).

        Returns
        -------
        tuple[pandas_udf, StructType]
            The UDF and its output schema (the schema is returned so callers
            can build an empty result frame without rebuilding it).
        """
        schema = self._result_schema(selections, dtypes)
        solve_udf = F.pandas_udf(
            partial(
                QuerySolver._grouped_map_solve_udf,
                selections=selections,
                col_map=self.config.col_map,
                cache_cls=cache_cls,
            ),
            schema,
            F.PandasUDFType.GROUPED_MAP,
        )
        return solve_udf, schema

    # ------------------------------------------------------------------
    # Cross-series cogroup solve path (registered tabular series)
    # ------------------------------------------------------------------
    #
    # When a query references one or more registered series (``SeriesSelector``
    # leaves), the per-container rows of every active series must reach the same
    # pandas worker as the channel rows. Spark ``cogroup`` is binary, so the
    # channels side is cogrouped against a single reduced-series stream: each
    # series is first reduced to a flat per-leaf interval schema (see the
    # dense-series note below), the per-series reduced frames are ``unionByName``'d
    # into one stream, and inside the ``applyInPandas`` UDF a
    # :class:`CombinedSeriesCache` resolves channel *and* series leaves from the
    # cogrouped rows.

    def _channel_cache_cls(self) -> type[SeriesCache]:
        """The per-solver channel :class:`SeriesCache` subclass (cogroup path).

        Overridden by grouped-map solvers (Delta, KVS). The base raises so a
        solver that has no inline channel cache (e.g. the RDD-based Blob solver)
        fails loudly rather than silently dropping channel leaves.
        """
        raise NotImplementedError(
            f"{type(self).__name__} cannot resolve channel leaves alongside "
            "registered series in a single query. Use DeltaSolver or "
            "KeyValueStoreSolver for queries that combine channels and registered "
            "series, or drop the channel leaf to run a series-only query."
        )

    def _read_prepared_channels(self, spark, query) -> DataFrame:
        """Read the channels table, apply column mapping, and RLE-encode raw data.

        Identical for the grouped-map solvers; reused by both the single-table
        and cogroup paths. Requires ``self.config.channels`` and (for raw data)
        ``self.interval_encoder`` — present on Delta/KVS.
        """
        q = query.db.channels(spark)
        q = self._apply_column_mapping(q, self.config.channels.column_name_mapping)
        if getattr(self, "is_raw_data", False):
            q = self.interval_encoder.prepare_channels_df(q)
        return q

    # The dense-series concern: a whole session's rows for a series must
    # not land in one Python worker. So instead of co-locating raw rows, each
    # series is *reduced* in a distributed Spark stage — grouped by
    # ``(container, signal, entity)`` (each group is one entity's rows, small) —
    # to the per-leaf interval sets its predicates produce. Point-in-time series
    # first get a per-row ``tend`` computed in Spark (the next distinct tick in
    # the ``(container, signal)`` frame list, last → ``container_stop_ts``), so a
    # per-entity slice no longer needs the shared frame list. Because every series
    # leaf is interval-valued (predicates only — no aggregations), the cogroup can
    # carry these reduced interval sets with no loss; the per-container worker then
    # only assembles already-tiny interval sets, not raw frames.

    REDUCED_TSTART = "__red_tstart"
    REDUCED_TEND = "__red_tend"

    def _reduced_schema(self) -> T.StructType:
        """Schema of the per-series reduction output (one row per
        ``(container, leaf, signal, entity)`` with a non-empty interval set)."""
        return T.StructType(
            [
                T.StructField(self.config.container_id_col, T.LongType()),
                T.StructField("leaf_key", T.IntegerType()),
                T.StructField("signal", T.StringType()),
                T.StructField("entity", T.StringType()),
                T.StructField("tstarts", T.ArrayType(T.DoubleType())),
                T.StructField("tends", T.ArrayType(T.DoubleType())),
            ]
        )

    def _reduce_series(
        self, spark, series, source_factory, leaves, stop_df, container_ids, cid
    ) -> DataFrame:
        """Reduce one series to ``(container, leaf_key, signal, entity, intervals)``
        rows, applying each leaf's predicate per ``(container, signal, entity)``."""
        df = source_factory(spark)
        if series.session_col != cid:
            df = df.withColumnRenamed(series.session_col, cid)
        if container_ids is not None:
            df = df.join(F.broadcast(container_ids), on=cid, how="inner")

        signal_col = series.signal_col
        entity_cols = list(series.entity_key_cols)

        if series.is_rle:
            df = df.withColumn(self.REDUCED_TSTART, F.col(series.tstart_col).cast("double"))
            df = df.withColumn(self.REDUCED_TEND, F.col(series.tend_col).cast("double"))
        else:
            ts = series.timestamp_col
            ticks = df.select(cid, signal_col, ts).distinct()
            w = Window.partitionBy(cid, signal_col).orderBy(ts)
            ticks = ticks.withColumn("__red_next", F.lead(F.col(ts)).over(w))
            df = df.join(ticks, on=[cid, signal_col, ts], how="left")
            df = df.withColumn(self.REDUCED_TSTART, F.col(ts).cast("double"))
            next_close = F.col("__red_next").cast("double")
            if stop_df is not None:
                df = df.join(F.broadcast(stop_df), on=cid, how="left")
                df = df.withColumn(
                    self.REDUCED_TEND, F.coalesce(next_close, F.col("__red_stop"))
                )
            else:
                df = df.withColumn(self.REDUCED_TEND, next_close)

        udf = partial(
            QuerySolver._reduce_group_udf,
            leaves=leaves,
            cid_col=cid,
            signal_col=signal_col,
            entity_cols=entity_cols,
            tstart_col=self.REDUCED_TSTART,
            tend_col=self.REDUCED_TEND,
        )
        group_cols = [cid, signal_col, *entity_cols]
        return df.groupBy(*group_cols).applyInPandas(udf, self._reduced_schema())

    @staticmethod
    def _reduce_group_udf(group_df, *, leaves, cid_col, signal_col, entity_cols, tstart_col, tend_col):
        """Per-``(container, signal, entity)`` body: apply each leaf's predicate,
        synthesize its interval set, and emit one reduced row per matching leaf."""
        cols = ["leaf_key", "signal", "entity", "tstarts", "tends"]
        out_cols = [cid_col, *cols]
        if group_df is None or len(group_df) == 0:
            return pd.DataFrame(columns=out_cols)

        cid_val = int(group_df[cid_col].iloc[0])
        signal_str = render_entity_key(group_df[signal_col].iloc[0])
        if len(entity_cols) == 0:
            entity_str = ""
        elif len(entity_cols) == 1:
            entity_str = render_entity_key(group_df[entity_cols[0]].iloc[0])
        else:
            entity_str = render_entity_key(tuple(group_df[c].iloc[0] for c in entity_cols))

        ts = group_df[tstart_col].to_numpy(dtype=np.float64)
        te = group_df[tend_col].to_numpy(dtype=np.float64)
        rows = []
        for leaf in leaves:
            mask = np.asarray(leaf._predicate(group_df), dtype=bool)
            if not mask.any():
                continue
            sub_ts = ts[mask]
            sub_te = te[mask]
            order = np.argsort(sub_ts, kind="mergesort")
            ivs = Intervals(
                sub_ts[order], sub_te[order], merge_overlaps=True, del_last_empty=True
            )
            if len(ivs) == 0:
                continue
            rows.append(
                (
                    cid_val,
                    leaf._reduce_key,
                    signal_str,
                    entity_str,
                    [float(x) for x in ivs.tstarts],
                    [float(x) for x in ivs.tends],
                )
            )
        if not rows:
            # The group has rows but no leaf matched. Emit a marker (null
            # leaf_key) so the container still reaches the per-container worker
            # and resolves to an empty result — matching the pre-reduction
            # behaviour where every container *with data* produced a row. A
            # container with no series rows at all has no group, so it stays
            # absent (the series-side filter prune relies on this).
            rows.append((cid_val, None, None, None, [], []))
        return pd.DataFrame(rows, columns=out_cols)

    @staticmethod
    def _build_reduced_cache(reduced_pdf, channel_cache):
        """Reconstruct the per-leaf reduced interval sets for one container from
        the reduction rows and wrap them (plus the channel cache) in a
        :class:`CombinedSeriesCache` keyed by each leaf's ``_reduce_key``."""
        presence: dict = {}
        entities: dict = {}
        if reduced_pdf is not None and len(reduced_pdf):
            keys = reduced_pdf["leaf_key"].tolist()
            sigs = reduced_pdf["signal"].tolist()
            ents = reduced_pdf["entity"].tolist()
            tss = reduced_pdf["tstarts"].tolist()
            tes = reduced_pdf["tends"].tolist()
            for key, sig, ent, ts, te in zip(keys, sigs, ents, tss, tes, strict=False):
                if pd.isna(key):
                    continue  # marker row: container had data but no leaf matched
                key = int(key)
                ivs = Intervals(
                    np.asarray(ts, dtype=np.float64),
                    np.asarray(te, dtype=np.float64),
                    merge_overlaps=True,
                )
                entities.setdefault(key, {})[(sig, ent)] = ivs
                presence[key] = presence.get(key, Intervals.empty()) | ivs
        return CombinedSeriesCache(
            channel_cache, reduced_presence=presence, reduced_entities=entities
        )

    @staticmethod
    def _reduced_core(channels_pdf, reduced_pdf, *, per_container, col_map, channel_cache):
        """Per-container body shared by the cogroup and series-only UDFs: pick the
        container id, build the reduced cache, hand it to *per_container*."""
        cid_col = col_map["cid"]
        cid_val = None
        if channels_pdf is not None and len(channels_pdf):
            cid_val = channels_pdf[cid_col].iloc[0]
        elif reduced_pdf is not None and len(reduced_pdf):
            cid_val = reduced_pdf[cid_col].iloc[0]
        if cid_val is None:
            return pd.DataFrame()
        cache = QuerySolver._build_reduced_cache(reduced_pdf, channel_cache)
        return per_container(cid_val, cache)

    @staticmethod
    def _reduced_cogroup_udf(channels_pdf, reduced_pdf, *, per_container, col_map, cache_cls):
        channel_cache = cache_cls(channels_pdf, col_map=col_map)
        return QuerySolver._reduced_core(
            channels_pdf,
            reduced_pdf,
            per_container=per_container,
            col_map=col_map,
            channel_cache=channel_cache,
        )

    @staticmethod
    def _reduced_series_only_udf(reduced_pdf, *, per_container, col_map):
        return QuerySolver._reduced_core(
            None,
            reduced_pdf,
            per_container=per_container,
            col_map=col_map,
            channel_cache=EmptyTimeSeriesCache(),
        )

    def run_series_cogroup(
        self,
        spark,
        query,
        channel_metrics_df,
        container_metrics_df,
        active_series,
        has_channel_leaves: bool,
        *,
        series_leaves,
        per_container,
        schema,
    ) -> DataFrame:
        """Reduce each active series per ``(container, signal, entity)`` and bring
        the reduced interval sets together per container, invoking *per_container*
        once per container.

        This is the single cross-series topology. ``solve_with_series`` uses it to
        emit one result row per container; the reporting layer uses it (via
        ``QueryBuilder.solve_series_apply``) to emit per-entity event fact rows —
        the only difference is the *per_container* row producer and *schema*.

        Parameters
        ----------
        series_leaves : list[SeriesSelector]
            The registered-series leaves referenced by the query. Each is assigned
            a stable ``_reduce_key`` (driver-side, on the shared leaf objects) so
            the reduced rows map back to the leaf across the pickle boundary.
        per_container : Callable
            ``(container_id, cache) → pandas.DataFrame`` matching *schema*. The
            cache resolves every leaf (reduced series intervals + channel leaves).
            Must be picklable (a ``functools.partial`` of a static/module function).
        schema : StructType
            Output schema of the ``applyInPandas`` result.
        """
        cid = self.config.container_id_col
        col_map = self.config.col_map

        # Resolve the channel cache up front so a solver that cannot resolve
        # channel leaves in the cogroup (e.g. BlobSolver) fails fast with a clear
        # message, before any reduction work — rather than deep inside the
        # has_channel_leaves branch below. Series-only queries never need it.
        channel_cache_cls = self._channel_cache_cls() if has_channel_leaves else None

        # Stable per-leaf key shared by the reduction stage and per_container (same
        # leaf objects are referenced by both; mutated once here in the driver).
        for i, leaf in enumerate(series_leaves):
            leaf._reduce_key = i
        leaves_by_series: dict = {}
        for leaf in series_leaves:
            leaves_by_series.setdefault(leaf.series.name, []).append(leaf)

        container_ids = None
        if container_metrics_df is not None and cid in container_metrics_df.columns:
            container_ids = container_metrics_df.select(cid).distinct()
        stop_df = self._container_stop_ts_df(container_metrics_df)

        reduced_frames = []
        for name, (series, source_factory) in active_series.items():
            leaves = leaves_by_series.get(name)
            if not leaves:
                continue
            reduced_frames.append(
                self._reduce_series(
                    spark, series, source_factory, leaves, stop_df, container_ids, cid
                )
            )
        if not reduced_frames:
            # No active series contributed a leaf — nothing to cogroup. Callers
            # only invoke this with series leaves present, so this is a defensive
            # guard returning an empty result rather than an IndexError.
            return spark.createDataFrame([], schema)
        reduced_stream = reduced_frames[0]
        for frame in reduced_frames[1:]:
            reduced_stream = reduced_stream.unionByName(frame)

        if has_channel_leaves:
            channels_df = self._read_prepared_channels(spark, query).join(
                F.broadcast(channel_metrics_df),
                on=[cid, self.config.channel_id_col],
            )
            cogroup_udf = partial(
                QuerySolver._reduced_cogroup_udf,
                per_container=per_container,
                col_map=col_map,
                cache_cls=channel_cache_cls,
            )
            return (
                channels_df.groupBy(cid)
                .cogroup(reduced_stream.groupBy(cid))
                .applyInPandas(cogroup_udf, schema)
            )

        series_only_udf = partial(
            QuerySolver._reduced_series_only_udf,
            per_container=per_container,
            col_map=col_map,
        )
        return reduced_stream.groupBy(cid).applyInPandas(series_only_udf, schema)

    @staticmethod
    def _query_result_row(cid_val, cache, *, selections, col_map):
        """``per_container`` producer for the query path: one result row whose
        columns are the query selections (each serialized to its dtype)."""
        result = {col_map["cid"]: [cid_val]}
        for s in selections:
            result[s._alias] = [QuerySolver._serialize_built(s.build(cache))]
        return pd.DataFrame(result)

    def solve_with_series(
        self,
        spark,
        query,
        channel_metrics_df,
        container_metrics_df,
        selections,
        dtypes,
        active_series,
        has_channel_leaves: bool,
        series_leaves,
    ) -> DataFrame:
        """Solve a query referencing one or more registered series.

        Emits one result row per container, one column per selection. Delegates
        the cogroup topology to :meth:`run_series_cogroup`.

        Parameters
        ----------
        spark : SparkSession
        query : QueryBuilder
        channel_metrics_df : DataFrame
            The resolved ``(container_id, channel_id, selector_ids)`` matches
            (empty when the query has no channel leaves).
        container_metrics_df : DataFrame
            The containers surviving the tag/metric filter stages. The series
            side is restricted to these container ids so query filters prune
            series rows too.
        selections, dtypes
            Query selections and resolved dtypes.
        active_series : dict[str, tuple[Series, Callable]]
            The registered series referenced by the query, with source factories.
        has_channel_leaves : bool
            Whether the query references any channel leaf. When ``False`` the
            channels side is skipped entirely (a registered-series-only
            deployment need not configure a channels table).
        series_leaves : list[SeriesSelector]
            The registered-series leaves in the selections (drive the reduction).
        """
        per_container = partial(
            QuerySolver._query_result_row,
            selections=selections,
            col_map=self.config.col_map,
        )
        return self.run_series_cogroup(
            spark,
            query,
            channel_metrics_df,
            container_metrics_df,
            active_series,
            has_channel_leaves,
            series_leaves=series_leaves,
            per_container=per_container,
            schema=self._result_schema(selections, dtypes),
        )

    def _container_stop_ts_df(self, container_metrics_df) -> DataFrame | None:
        """``(container_id, __red_stop)`` frame of per-container session ends, or
        ``None`` when no metrics frame carries ``container_stop_ts_col``.

        Joined onto a point-in-time series in :meth:`_reduce_series` so the last
        frame closes at session end. ``None`` leaves the last frame open (it
        collapses to zero length and drops) — the pre-existing behaviour when no
        stop timestamp is configured.

        Aggregates to exactly one row per container (``max`` = the latest stop,
        i.e. session end). A ``.distinct()`` here would keep every distinct stop
        value, so a ``container_metrics`` frame with more than one row per
        container (e.g. resolved at a finer grain, or duplicated upstream) would
        fan out the left join in :meth:`_reduce_series` and double-count the
        series rows. ``max`` ignores NULL stops; a container whose only stop is
        NULL yields a NULL ``__red_stop`` (its last frame stays open, as if no
        stop were configured).
        """
        cid = self.config.container_id_col
        stop_col = self.config.container_stop_ts_col
        if container_metrics_df is None or stop_col not in container_metrics_df.columns:
            return None
        return container_metrics_df.groupBy(F.col(cid)).agg(
            F.max(F.col(stop_col).cast("double")).alias("__red_stop")
        )

    @abc.abstractmethod
    def filter_container_tags(self, spark, query) -> DataFrame:
        """
        Abstract method to filter containers by tags.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        query : QueryBuilder
            Query object containing filters and db info.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame filtered by container tags.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Each solver must implement the filter_container_tags method.")

    @abc.abstractmethod
    def filter_container_metrics(
        self, spark, query, container_df, pre_filtered_containers_df=None
    ) -> DataFrame:
        """
        Abstract method to filter containers by metrics.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        query : QueryBuilder
            Query object containing filters and db info.
        container_df : pyspark.sql.DataFrame
            DataFrame from filter_container_tags stage.
        pre_filtered_containers_df : pyspark.sql.DataFrame, optional
            Pre-filtered containers for incremental processing.
            When provided, restricts processing to only these containers.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing filtered container metrics.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError(
            "Each solver must implement the filter_container_metrics method."
        )

    @abc.abstractmethod
    def filter_channel_tags(self, spark, db: MeasurementDB, container_df, selectors) -> DataFrame:
        """
        Stage 3: Filter channels by measurements and tags.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        db : MeasurementDB
            Measurement database for table access.
        container_df : pyspark.sql.DataFrame
            DataFrame containing container information.
        selectors : list[TimeSeriesSelector]
            Non-aliased (direct) selectors extracted from the query.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing filtered channel tags.

        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Each solver must implement the filter_channel_tags method.")

    @abc.abstractmethod
    def filter_channel_metrics(self, spark, db: MeasurementDB, channel_df, selectors) -> DataFrame:
        """
        Stage 4: Filter channels by metrics.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        db : MeasurementDB
            Measurement database for table access.
        channel_df : pyspark.sql.DataFrame
            DataFrame containing channel information.
        selectors : list[TimeSeriesSelector]
            Non-aliased (direct) selectors extracted from the query.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with ``(container_id, channel_id, selector_ids)``
            where ``selector_ids`` is an array column.
        Raises
        ------
        NotImplementedError
            If not implemented by subclass.
        """
        raise NotImplementedError("Each solver must implement the filter_channel_metrics method.")

    def filter_aliased_channel_metrics(
        self, spark, db: MeasurementDB, container_df, selectors
    ) -> DataFrame:
        """
        Resolve aliased channel selections via the channel_mapping table.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        db : MeasurementDB
            Measurement database for table access.
        container_df : pyspark.sql.DataFrame
            DataFrame containing filtered container IDs.
        selectors : list[TimeSeriesSelector]
            Aliased selectors extracted from the query.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with ``(container_id, channel_id, selector_ids)``
            where ``selector_ids`` is an array column.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support aliased channel resolution"
        )

    def resolve_channel_selections(
        self, spark, channel_metrics_df, aliased_channel_metrics_df
    ) -> DataFrame:
        """
        Union direct and aliased channel metrics, combining selector_ids.

        Only called when aliased selectors are present.  The default
        implementation raises ``NotImplementedError``; solvers that support
        aliasing (e.g. ``KeyValueStoreSolver``) must override this.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        channel_metrics_df : pyspark.sql.DataFrame
            Direct channel metrics with ``selector_ids`` array column.
        aliased_channel_metrics_df : pyspark.sql.DataFrame
            Aliased channel metrics with ``selector_ids`` array column.

        Returns
        -------
        pyspark.sql.DataFrame
            Merged DataFrame with ``(container_id, channel_id, selector_ids)``.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support aliased channel resolution"
        )

    def filter_candidates(self, query, channel_df) -> DataFrame:
        """
        Stage 5: Select best channel candidate.

        Parameters
        ----------
        query : QueryBuilder
            Query object containing filters and db info.
        channel_df : pyspark.sql.DataFrame
            DataFrame containing channel information.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing selected channel candidates.
        """
        pass

    @abc.abstractmethod
    def solve(self, query, channels_df, selections, dtypes):
        """
        Stage 6: Solve query.

        Parameters
        ----------
        query : QueryBuilder
            Query object containing database and filter information.
        channels_df : pyspark.sql.DataFrame
            DataFrame containing channel information.
        selections : list
            List of selection expressions to apply.
        dtypes : list
            List of data types for each selection.

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing results for each container.
        """
        pass
