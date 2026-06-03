import pyspark.sql.functions as F
from pyspark.sql import DataFrame

from impulse_query_engine.analyze.metadata.metric_expression import MetricExpression
from impulse_query_engine.analyze.metadata.tag_expression import TagExpression

from .query_solver import QuerySolver
from .series_cache import ChannelTimeSeriesCache
from .solver_config import SolverConfig
from .utils.interval_encoder import IntervalEncoder


class DeltaSolver(QuerySolver):
    supports_registered_series = True

    def __init__(
        self,
        spark,
        config: SolverConfig = None,
        is_raw_data: bool = True,
        drop_implausible_data: bool = False,
    ):
        """
        Initialize the DeltaSolver.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        config : SolverConfig, optional
            Solver configuration.  When *None* a default :class:`SolverConfig`
            is used (backward-compatible column names).
        is_raw_data : bool
            Indicates whether the input data is raw point data (with a timestamp column) or already in RLE format
            (with tstart and tend columns).
        drop_implausible_data: bool
            Specifies whether we should drop implausible data points before RLE encoding.
            IMPORTANT: The silver layer needs the is_plausible column for this to work.
            If this is set to True, all data points which are marked as implausible will be dropped before RLE encoding.
        """
        super().__init__(config)
        self.spark = spark
        self.is_raw_data: bool = is_raw_data
        self.drop_implausible_data: bool = drop_implausible_data

        self.interval_encoder: IntervalEncoder = IntervalEncoder(
            timestamp_col_name="timestamp",
            drop_implausible_data_points=self.drop_implausible_data,
        )

    def _channel_cache_cls(self):
        return ChannelTimeSeriesCache

    def filter_container_tags(self, spark, query) -> DataFrame:
        """
        Stage 1: Generate DataFrame filtered by container tags.

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
        """
        filters = []
        required_tags = []
        for filt in query.filters:
            if isinstance(filt, TagExpression):
                filters.append(filt)
                required_tags.extend(filt.required_tags())
        required_tags = set(required_tags)
        tags = query.db.container_tags(spark)
        tags = self._apply_column_mapping(tags, self.config.container_tags.column_name_mapping)
        # apply filters
        if len(filters) > 0:
            tags = tags.where(F.col("key").isin(required_tags))
            tags = tags.groupBy("container_id")
            tags = tags.pivot("key", list(required_tags)).agg({"value": "first"})
            expr = self._build_expr(filters)
            tags = tags.where(expr)
            for tag in required_tags:
                tags = tags.withColumnRenamed(tag, f"mt_{tag}")
        else:
            tags = tags.select("container_id").distinct()
        return tags

    def filter_container_metrics(
        self, spark, query, container_df, pre_filtered_containers_df=None
    ) -> DataFrame:
        """
        Stage 2: Filter containers by metrics.

        Returns full container metrics (not just container_id) so that
        ContainerDimension and ContainerEvent can access start_ts/stop_ts.

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

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame containing filtered container metrics.
        """
        filters = []
        tag_count = 0
        for filt in query.filters:
            if isinstance(filt, MetricExpression):
                filters.append(filt)
            if isinstance(filt, TagExpression):
                tag_count += 1
        cm_mapping = self.config.container_metrics.column_name_mapping
        if len(filters) > 0:
            if pre_filtered_containers_df is not None:
                metrics = pre_filtered_containers_df
            else:
                metrics = query.db.container_metrics(self.spark)
            metrics = self._apply_column_mapping(metrics, cm_mapping)
            expr = self._build_expr(filters)
            metrics = metrics.where(expr)
            if tag_count > 0:
                container_ids = container_df.select("container_id").distinct()
                return metrics.join(F.broadcast(container_ids), on=["container_id"], how="inner")
            else:
                return metrics
        else:
            if pre_filtered_containers_df is not None:
                metrics = pre_filtered_containers_df
            else:
                metrics = self._apply_column_mapping(query.db.container_metrics(spark), cm_mapping)
            container_ids = container_df.select("container_id").distinct()
            return metrics.join(F.broadcast(container_ids), on=["container_id"], how="inner")

    def filter_channel_tags(self, spark, db, container_df, selectors) -> DataFrame:
        """
        Stage 3: Filter channels by tags and compute ``selector_id``.

        Extracts leaf selectors, pivots the channel-tags table, filters
        matching channels, and assigns each row its ``selector_id`` so
        that Stage 4 can be a simple passthrough.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        db : MeasurementDB
            Measurement database for table access.
        container_df : pyspark.sql.DataFrame
            DataFrame containing container information.
        selectors : list[TimeSeriesSelector]
            Non-aliased (direct) selectors.

        Returns
        -------
        pyspark.sql.DataFrame
            ``(container_id, channel_id, selector_id)``
        """
        container_id_col = self.config.container_id_col
        channel_id_col = self.config.channel_id_col
        mids = container_df.select(container_id_col).distinct()

        if len(selectors) == 0:
            return self._empty_channel_match_df(spark)

        required_tags = set()
        for selector in selectors:
            required_tags.update(selector.required_tags())

        tbl = db.channel_tags(spark)
        tbl = self._apply_column_mapping(tbl, self.config.channel_tags.column_name_mapping)
        expr = self._build_expr(selectors)

        tags = (
            tbl.where(F.col("key").isin(required_tags))
            .join(F.broadcast(mids), on=[container_id_col], how="inner")
            .groupBy(container_id_col, channel_id_col)
            .pivot("key", list(required_tags))
            .agg(F.first(F.col("value")))
            .where(expr)
        )
        tags = tags.withColumn("selector_id", self._build_selector_id_expr(selectors))
        return tags.select(container_id_col, channel_id_col, "selector_id")

    def filter_channel_metrics(self, spark, db, channel_df, selectors) -> DataFrame:
        """
        Stage 4: Join with ``channel_metrics`` to restrict to channels that
        have metric entries.

        The input *channel_df* already carries a ``selector_id`` column
        from Stage 3.  This stage inner-joins it with the channel-metrics
        table so that channels without any recorded samples are excluded,
        then wraps ``selector_id`` into an array ``selector_ids``.

        Parameters
        ----------
        spark : SparkSession
            Spark session used for query execution.
        db : MeasurementDB
            Measurement database for table access.
        channel_df : pyspark.sql.DataFrame
            DataFrame from :meth:`filter_channel_tags` with columns
            ``(container_id, channel_id, selector_id)``.
        selectors : list[TimeSeriesSelector]
            Non-aliased selectors (unused — selector_id comes from Stage 3).

        Returns
        -------
        pyspark.sql.DataFrame
            DataFrame with ``(container_id, channel_id, selector_ids)``
            where ``selector_ids`` is an array column.
        """
        container_id_col = self.config.container_id_col
        channel_id_col = self.config.channel_id_col
        tbl = db.channel_metrics(spark)
        tbl = self._apply_column_mapping(tbl, self.config.channel_metrics.column_name_mapping)
        metrics = tbl.select(container_id_col, channel_id_col).join(
            F.broadcast(channel_df),
            on=[container_id_col, channel_id_col],
            how="inner",
        )
        metrics = metrics.withColumn("selector_ids", F.array(F.col("selector_id"))).drop(
            "selector_id"
        )
        return metrics

    def solve(self, query, channels_df, selections, dtypes):
        """
        Solve the query by grouping channels and applying selections.

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
        q = query.db.channels(self.spark)
        q = self._apply_column_mapping(q, self.config.channels.column_name_mapping)

        if self.is_raw_data:
            # Calculate the tend info and prepare the data for the solving step.
            q = self.interval_encoder.prepare_channels_df(q)

        solve_udf, _schema = self._grouped_map_udf(selections, dtypes, ChannelTimeSeriesCache)
        df = q.join(
            F.broadcast(channels_df), on=[self.config.container_id_col, self.config.channel_id_col]
        )
        return df.groupBy(self.config.container_id_col).apply(solve_udf)
