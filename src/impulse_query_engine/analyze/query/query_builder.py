from typing import Self

import pandas as pd
import pyspark.sql.types as T
from pyspark.sql import DataFrame

from impulse_query_engine.analyze.metadata.metric_expression import MetricSelector
from impulse_query_engine.analyze.metadata.tag_expression import TagSelector
from impulse_query_engine.analyze.metadata.time_series_expression import (
    RequiresDeserialization,
    TimeSeriesExpression,
    TimeSeriesSelector,
)
from impulse_query_engine.analyze.query.solvers.empty_cache import EmptyTimeSeriesCache
from impulse_query_engine.surfaces import SeriesAccessor
from impulse_query_engine.surfaces.series_selector import SeriesSelector
from impulse_query_engine.telemetry import telemetry_logger

from .solvers.blob_solver import BlobSolver
from .solvers.query_solver import QuerySolver


class QueryBuilder:
    def __init__(self, db: "impulse_query_engine.analyze.MeasurementDB"):
        """
        Initialize the QueryBuilder.

        Parameters
        ----------
        db : impulse_query_engine.analyze.MeasurementDB
            Measurement database object.
        """
        self.db = db
        self.ws = db.ws
        self.filters = []
        self.selections = []
        self.result_objects = []
        self.result_dtypes = []

    def where(self, *args):
        """
        Add filter expressions to the query.

        Parameters
        ----------
        *args : list
            Filter expressions to be added.
        Returns
        -------
        QueryBuilder
            The updated QueryBuilder instance.
        """
        if len(args) == 0:
            return self
        filtered_args = [arg for arg in args if arg is not None]
        self.filters.extend(filtered_args)
        return self

    def filter(self, *args):
        """
        Alias for where().

        Parameters
        ----------
        *args : list
            Filter expressions to be added.

        Returns
        -------
        QueryBuilder
            The updated QueryBuilder instance.
        """
        return self.where(*args)

    def havingTag(self, **kwargs):
        """
        Add tag-based filters to the query.

        Parameters
        ----------
        **kwargs : dict
            Tag-value pairs to filter by.

        Returns
        -------
        QueryBuilder
            The updated QueryBuilder instance.
        """
        for k, arg in kwargs.items():
            self.filters.append(TagSelector(k) == arg)
        return self

    def tag(self, key: str, cast_type: str | None = None) -> TagSelector:
        """
        Create a tag selector for the given key.

        Parameters
        ----------
        key : str
            Name of the tag (element_id in the EAV table).
        cast_type : str or None, optional
            Spark type to cast the tag value to before comparison
            (e.g. ``"int"``, ``"double"``, ``"string"``).

        Returns
        -------
        TagSelector
            Tag selector object.
        """
        return TagSelector(key, cast_type=cast_type)

    def metric(self, name) -> MetricSelector:
        """
        Create a metric selector for the given name.

        Parameters
        ----------
        name : str
           Name of the metric.

        Returns
        -------
        MetricSelector
           Metric selector object.
        """
        return MetricSelector(name)

    def series(self, name: str) -> SeriesAccessor:
        """Return a predicate-authoring accessor for a registered tabular series.

        Looks the series up by name in the :class:`MeasurementDB` definition
        registry and returns a :class:`SeriesAccessor` over its schema, from
        which typed column predicates are authored (e.g.
        ``query.series("object_tracks").distance_m < 8.0``).

        Parameters
        ----------
        name : str
            The registered series name.

        Returns
        -------
        SeriesAccessor
            Accessor over the registered series' schema.

        Raises
        ------
        KeyError
            If *name* is not registered on the database.
        """
        registry = self.db.registered_series()
        if name not in registry:
            raise KeyError(
                f"Series {name!r} is not registered; register it on the "
                "MeasurementDB with register_series(...)."
            )
        return SeriesAccessor(registry[name])

    def channel(self, **kwargs) -> TimeSeriesSelector:
        """
        Create a time series selector for the given channel tags.

        Parameters
        ----------
        **kwargs : dict
            Channel tag-value pairs.

        Returns
        -------
        TimeSeriesSelector
            Time series selector object.
        """
        expr = None
        for k, arg in kwargs.items():
            if not expr:
                expr = TagSelector(k) == str(arg)
            else:
                expr = expr & (TagSelector(k) == str(arg))
        return TimeSeriesSelector(expr)

    def signal(self, name: str) -> TimeSeriesSelector:
        """Select a scalar channel by name — the name-addressed counterpart to
        the tag-addressed :meth:`channel`.

        Sugar for ``channel(channel_name=name)``: it resolves the channel whose
        ``channel_name`` tag equals *name*. Deployments that key channels on a
        different tag should call :meth:`channel` directly. Channels are resolved
        natively by the solver (the channels side of the cogroup / the
        single-table path), so they are not held in the series definition
        registry and ``query.series`` does not surface them.

        Parameters
        ----------
        name : str
            The channel name (value of the ``channel_name`` tag).

        Returns
        -------
        TimeSeriesSelector
            Selector for the named scalar channel.
        """
        return self.channel(channel_name=name)

    def channel_with_alias(self, **kwargs) -> TimeSeriesSelector:
        if self.db.config.channel_mapping_table is None:
            raise ValueError("channel_mapping_table is not configured")

        expr = None
        for k, arg in kwargs.items():
            if not expr:
                expr = TagSelector(k) == str(arg)
            else:
                expr = expr & (TagSelector(k) == str(arg))
        return TimeSeriesSelector(expr, uses_alias=True)

    def select(self, *args) -> Self:
        """
        Set the selection expressions for the query.

        Parameters
        ----------
        *args : list
            Selection expressions.

        Returns
        -------
        QueryBuilder
            The updated QueryBuilder instance.
        """
        self.selections = list(args)
        return self

    def _collect_time_series_selectors(
        self, uses_alias=None, leaf_kind: str | None = None
    ) -> list[TimeSeriesSelector]:
        """Collect deduplicated leaf selectors from this query's selections.

        Parameters
        ----------
        uses_alias : bool or None, optional
            When ``True``, keep only alias selectors; when ``False``, keep
            only direct selectors; when ``None`` (default), keep all.
        leaf_kind : str or None, optional
            When set, keep only selectors whose ``leaf_kind`` matches.
            Channel-side filter stages pass ``"channel"`` so series leaves
            are excluded from channel-tag / channel-metric filtering.

        Returns
        -------
        list of TimeSeriesSelector
            Deduplicated selectors in discovery order.
        """
        selectors = []
        seen_selector_ids = set()
        for expression in self.selections:
            if not isinstance(expression, TimeSeriesExpression):
                continue
            for selector in expression.get_selectors():
                if uses_alias is not None and selector.uses_alias != uses_alias:
                    continue
                if leaf_kind is not None and selector.leaf_kind != leaf_kind:
                    continue
                if selector.selector_id in seen_selector_ids:
                    continue
                seen_selector_ids.add(selector.selector_id)
                selectors.append(selector)
        return selectors

    def _collect_series_selectors(self) -> list[SeriesSelector]:
        """Collect deduplicated registered-series leaves from the selections.

        These are the ``SeriesSelector`` leaves (``leaf_kind`` == series name)
        that drive the cogroup path; channel leaves (``leaf_kind == "channel"``)
        are handled by the existing channel-side stages.
        """
        selectors: list[SeriesSelector] = []
        seen: set[int] = set()
        for expression in self.selections:
            if not isinstance(expression, TimeSeriesExpression):
                continue
            for selector in expression.get_selectors():
                if isinstance(selector, SeriesSelector) and id(selector) not in seen:
                    seen.add(id(selector))
                    selectors.append(selector)
        return selectors

    def _determine_result_objects_dtypes(self, default_dtype: T = T.DoubleType()):
        """
        Determine result objects and their data types for the selections.

        Parameters
        ----------
        default_dtype : pyspark.sql.types.DataType, optional
            Default data type to use if not specified (default is DoubleType).

        Returns
        -------
        tuple
            Tuple of (result_objects, result_dtypes).
        """
        result_objects = []
        result_dtypes = []
        for s in self.selections:
            result_object = s.build(EmptyTimeSeriesCache())
            result_objects.append(result_object)
            dtype = default_dtype
            if hasattr(result_object, "dtype") and callable(result_object.dtype):
                dtype = result_object.dtype()
            elif hasattr(s, "dtype") and callable(s.dtype):
                dtype = s.dtype()
            result_dtypes.append(dtype)
        return (result_objects, result_dtypes)

    @telemetry_logger("query", "solve")
    def solve(
        self,
        spark,
        solver: QuerySolver = BlobSolver(),
        pre_filtered_containers_df: DataFrame = None,
    ) -> DataFrame:
        """
        Execute the query using the specified solver and return a Spark DataFrame.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        solver : QuerySolver, optional
            Query solver to use (default is BlobSolver).
        pre_filtered_containers_df : DataFrame, optional
            Pre-filtered container metrics DataFrame for incremental processing.
            When provided, only these containers will be processed.
            When None, all containers matching query filters are processed (full mode).

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing query results.
        """  # determining result types
        (
            self.result_objects,
            self.result_dtypes,
        ) = self._determine_result_objects_dtypes()

        metrics_df, channel_metrics_df, direct_selectors, aliased_selectors = (
            self._run_filter_stages(spark, solver, pre_filtered_containers_df)
        )

        active_series = self._collect_active_series()
        if active_series:
            has_channel_leaves = bool(direct_selectors) or bool(aliased_selectors)
            return solver.solve_with_series(
                spark,
                self,
                channel_metrics_df,
                metrics_df,
                self.selections,
                self.result_dtypes,
                active_series,
                has_channel_leaves,
                self._collect_series_selectors(),
            )

        return solver.solve(self, channel_metrics_df, self.selections, self.result_dtypes)

    def _run_filter_stages(self, spark, solver, pre_filtered_containers_df):
        """Run the container/channel filter stages shared by every solve path.

        Returns ``(metrics_df, channel_metrics_df, direct_selectors,
        aliased_selectors)``. Channel-side stages only see channel leaves; series
        leaves are resolved separately by :meth:`_collect_active_series`.
        """
        direct_selectors = self._collect_time_series_selectors(
            uses_alias=False, leaf_kind="channel"
        )
        aliased_selectors = self._collect_time_series_selectors(
            uses_alias=True, leaf_kind="channel"
        )

        tags_df = solver.filter_container_tags(spark, self)
        metrics_df = solver.filter_container_metrics(
            spark, self, tags_df, pre_filtered_containers_df
        )
        channel_tags_df = solver.filter_channel_tags(spark, self.db, metrics_df, direct_selectors)
        channel_metrics_df = solver.filter_channel_metrics(
            spark, self.db, channel_tags_df, direct_selectors
        )

        if len(aliased_selectors) > 0:
            aliased_channel_metrics_df = solver.filter_aliased_channel_metrics(
                spark, self.db, channel_tags_df, aliased_selectors
            )
            channel_metrics_df = solver.resolve_channel_selections(
                spark, channel_metrics_df, aliased_channel_metrics_df
            )

        return metrics_df, channel_metrics_df, direct_selectors, aliased_selectors

    def _collect_active_series(self) -> dict:
        """Registered series referenced by the selections, as
        ``{name: (Series, source_factory)}``.

        Empty when the query has no series leaves. Raises ``KeyError`` naming a
        series that is referenced but not registered on the ``MeasurementDB``.
        """
        active_series: dict = {}
        registry = self.db.registered_series()
        for sel in self._collect_series_selectors():
            name = sel.leaf_kind
            if name in active_series:
                continue
            if name not in registry:
                raise KeyError(
                    f"Series {name!r} is referenced by the query but not "
                    "registered on the MeasurementDB."
                )
            active_series[name] = (registry[name], self.db.series_source(name))
        return active_series

    def solve_series_apply(
        self,
        spark,
        solver: QuerySolver,
        *,
        per_container,
        schema,
        pre_filtered_containers_df: DataFrame = None,
    ) -> DataFrame:
        """Run the filter pipeline + series cogroup, invoking *per_container* once
        per container to emit rows matching *schema*.

        The query's selections (set via :meth:`select`) are the expressions whose
        leaves drive channel/series resolution. This is the generic entry the
        reporting layer uses to materialize per-entity event facts, so the
        query-engine layer carries no dependency on it.

        Parameters
        ----------
        per_container : Callable
            ``(container_id, cache)`` → pandas DataFrame matching *schema*, where
            *cache* is the :class:`CombinedSeriesCache` that resolves every leaf
            for that container (reduced series interval sets + channel leaves).
            Must be picklable (a ``functools.partial`` of a static/module
            function, not a local closure).
        schema : StructType
            Output schema of the cogroup result.
        """
        metrics_df, channel_metrics_df, direct_selectors, aliased_selectors = (
            self._run_filter_stages(spark, solver, pre_filtered_containers_df)
        )
        active_series = self._collect_active_series()
        has_channel_leaves = bool(direct_selectors) or bool(aliased_selectors)
        return solver.run_series_cogroup(
            spark,
            self,
            channel_metrics_df,
            metrics_df,
            active_series,
            has_channel_leaves,
            series_leaves=self._collect_series_selectors(),
            per_container=per_container,
            schema=schema,
        )

    @telemetry_logger("query", "to_pandas")
    def toPandas(self, spark, solver: QuerySolver = BlobSolver()) -> pd.DataFrame:
        """
        Execute the query and collect results into a Pandas DataFrame.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        solver : QuerySolver, optional
            Query solver to use (default is BlobSolver).

        Returns
        -------
        pd.DataFrame
            Pandas DataFrame containing query results.
        """
        df = self.solve(spark, solver)
        pdf = df.toPandas()
        for selection, result_object in zip(self.selections, self.result_objects, strict=False):
            if isinstance(selection, RequiresDeserialization):
                pdf[selection._alias] = pdf[selection._alias].apply(
                    lambda x: selection.deserialize(x)
                )
            elif hasattr(result_object, "requires_deserialization"):
                pdf[selection._alias] = pdf[selection._alias].apply(
                    lambda x: result_object.deserialize(x)
                )
        return pdf
