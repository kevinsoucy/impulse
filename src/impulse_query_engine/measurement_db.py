from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import pyspark.sql.functions as F
import pyspark.sql.types as T
from databricks.sdk import WorkspaceClient
from pyspark.sql import DataFrame, SparkSession

# Cap on how many offending signal values the validation error lists. Bounds the
# driver collect — we only need a few examples to make the message actionable.
_MAX_UNKNOWN_SIGNALS_IN_ERROR = 20

from impulse_query_engine import __version__
from impulse_query_engine.telemetry import verify_workspace_client
from .analyze.query.query_builder import QueryBuilder

if TYPE_CHECKING:
    from impulse_query_engine.surfaces.series import Series


class MeasurementDBConfig:
    def __init__(
        self,
        container_tags_table=None,
        container_metrics_table=None,
        channel_tags_table=None,
        channel_metrics_table=None,
        channels_uri=None,
        channel_mapping_table=None,
        table_locations: str = "external_locations",
    ):
        self.container_tags_table = container_tags_table
        self.container_metrics_table = container_metrics_table
        self.channel_tags_table = channel_tags_table
        self.channel_metrics_table = channel_metrics_table
        self.channels_uri = channels_uri
        self.channel_mapping_table = channel_mapping_table
        self.table_locations = table_locations
        self.debug_tables = None

    @staticmethod
    def for_unity_catalog(
        catalog_name: str,
        core_schema_name: str = "core",
        channel_mapping_table: str | None = None,
    ):
        return MeasurementDBConfig(
            container_tags_table=f"{catalog_name}.{core_schema_name}.container_tags",
            container_metrics_table=f"{catalog_name}.{core_schema_name}.container_metrics",
            channel_tags_table=f"{catalog_name}.{core_schema_name}.channel_tags",
            channel_metrics_table=f"{catalog_name}.{core_schema_name}.channel_metrics",
            channels_uri=f"{catalog_name}.{core_schema_name}.channels",
            channel_mapping_table=channel_mapping_table,
            table_locations="unity_catalog",
        )

    @staticmethod
    def for_debug(debug_tables):
        cfg = MeasurementDBConfig(
            container_tags_table="container_tags" if "container_tags" in debug_tables else None,
            container_metrics_table=(
                "container_metrics" if "container_metrics" in debug_tables else None
            ),
            channel_tags_table="channel_tags" if "channel_tags" in debug_tables else None,
            channel_metrics_table=(
                "channel_metrics" if "channel_metrics" in debug_tables else None
            ),
            channels_uri="channels" if "channels" in debug_tables else None,
            channel_mapping_table=(
                "channel_mapping" if "channel_mapping" in debug_tables else None
            ),
            table_locations="debug",
        )
        cfg.debug_tables = debug_tables
        return cfg


class MeasurementDB:
    def __init__(self, config: MeasurementDBConfig, ws: WorkspaceClient):
        self.config = config
        self.ws = verify_workspace_client(ws, "databricks-impulse", __version__)
        # name -> (Series definition, source_factory). The definition is needed
        # at *authoring* time (query.series(name) builds a typed accessor from the
        # schema, before any solver is chosen); the source_factory is needed at
        # *execution* time (the solver loads the backing DataFrame per solve).
        # Both live here so QueryBuilder (authoring) and the solver (execution)
        # reach the same registry. Built-in channels are not registered here —
        # their column roles live in SolverConfig, so the solver contributes the
        # channels Series into its effective registry at solve time.
        self._series_registry: dict[
            str, tuple["Series", Callable[[SparkSession], DataFrame]]
        ] = {}

    def register_series(
        self,
        series: "Series",
        source_factory: Callable[[SparkSession], DataFrame],
        *,
        valid_signals: "Iterable[object] | None" = None,
        spark: SparkSession | None = None,
    ) -> None:
        """Register a tabular series definition and its DataFrame factory.

        Parameters
        ----------
        series : Series
            The declarative series definition. Must carry a resolved ``schema``
            so ``query.series(name)`` can build a typed accessor.
        source_factory : Callable[[SparkSession], DataFrame]
            Called per solve to produce the Spark DataFrame backing this series.
            The DataFrame must include the series' ``session_col`` so the cogroup
            can partition on it.
        valid_signals : Iterable, optional
            **Opt-in** signal-metadata validation. When supplied, the set
            of known ``signal_col`` values for this series. Registration then reads
            the source DataFrame's distinct ``signal_col`` values and raises if any
            are **not** in *valid_signals* — fail-fast, so a typo'd or unregistered
            signal cannot be silently dropped at query time (queries filter by
            signal). When omitted, registration requires no signal metadata at all
            (the default; signal values live in the data and are queried directly).
        spark : SparkSession, optional
            Required only when *valid_signals* is given — used to materialize the
            source DataFrame for validation. Ignored otherwise.

        Raises
        ------
        ValueError
            If the series name is already registered; if *valid_signals* is given
            without *spark*; or if the data carries signal value(s) absent from
            *valid_signals* (the message names the series and the unknown signals).
        """
        if series.name in self._series_registry:
            raise ValueError(f"Series {series.name!r} is already registered.")
        if valid_signals is not None:
            self._validate_signal_values(series, source_factory, valid_signals, spark)
        self._series_registry[series.name] = (series, source_factory)

    @staticmethod
    def _validate_signal_values(
        series: "Series",
        source_factory: Callable[[SparkSession], DataFrame],
        valid_signals: "Iterable[object]",
        spark: SparkSession | None,
    ) -> None:
        """Fail-fast check that every ``signal_col`` value in the source data is a
        known signal. See :meth:`register_series`."""
        if spark is None:
            raise ValueError(
                f"register_series({series.name!r}, valid_signals=...) needs a spark "
                "session to read the source data for signal-metadata validation."
            )
        allowed = list(valid_signals)
        df = source_factory(spark)
        sig = series.signal_col
        # Anti-join the source's signal column against the allowed set instead of
        # collecting the full distinct set to the driver: Spark can stop after the
        # first few offending rows (limit), there is no distinct shuffle, and only
        # a bounded sample is materialized. The check stays eager and exact —
        # fail-fast is preserved. A NULL signal never matches an allowed value, so
        # it surfaces as unknown, exactly as the prior set-difference did.
        sig_type = df.select(sig).schema[0].dataType
        allowed_df = spark.createDataFrame(
            [(value,) for value in allowed],
            T.StructType([T.StructField(sig, sig_type, True)]),
        )
        unknown = {
            row[0]
            for row in (
                df.select(F.col(sig))
                .join(F.broadcast(allowed_df), on=sig, how="left_anti")
                .limit(_MAX_UNKNOWN_SIGNALS_IN_ERROR)
                .collect()
            )
        }
        if unknown:
            raise ValueError(
                f"Series {series.name!r} carries signal value(s) "
                f"{sorted(map(str, unknown))} on column {series.signal_col!r} that are "
                f"not in the provided signal metadata {sorted(map(str, allowed))}; "
                "they would be silently dropped at query time."
            )

    def registered_series(self) -> dict[str, "Series"]:
        """Snapshot of registered series definitions, keyed by name."""
        return {name: pair[0] for name, pair in self._series_registry.items()}

    def series_source(self, name: str) -> Callable[[SparkSession], DataFrame]:
        """Return the ``source_factory`` for a registered series.

        Raises ``KeyError`` naming the series if it is not registered.
        """
        if name not in self._series_registry:
            raise KeyError(f"Series {name!r} is not registered on this MeasurementDB.")
        return self._series_registry[name][1]

    @property
    def query(self):
        return QueryBuilder(db=self)

    def _read_table(self, spark, table_name):
        # if not DeltaTable.isDeltaTable(spark, table_name):
        #    raise Exception(f"Table not found: `{table_name}`")
        if self.config.table_locations == "unity_catalog":
            return spark.read.table(table_name)
        elif self.config.table_locations == "debug":
            return self.config.debug_tables[table_name]
        return spark.read.format("delta").load(table_name)

    def container_tags(self, spark) -> DataFrame:
        return self._read_table(spark, self.config.container_tags_table)

    def container_metrics(self, spark) -> DataFrame:
        return self._read_table(spark, self.config.container_metrics_table)

    def channel_tags(self, spark) -> DataFrame:
        return self._read_table(spark, self.config.channel_tags_table)

    def channel_metrics(self, spark) -> DataFrame:
        return self._read_table(spark, self.config.channel_metrics_table)

    def channels(self, spark) -> DataFrame:
        return self._read_table(spark, self.config.channels_uri)

    def channel_mapping(self, spark) -> DataFrame:
        if self.config.channel_mapping_table is None:
            raise ValueError("channel_mapping_table is not configured")
        return self._read_table(spark, self.config.channel_mapping_table)

    def channel_uri(self):
        return self.config.channels_uri


class InMemoryMeasurementDB(MeasurementDB):
    @property
    def query(self):
        pass

    def add(self, ts, container_tags, measurement_tags):
        pass

    def container_tags(self, spark) -> DataFrame:
        return self._read_table(spark, self.config.container_tags_table)

    def container_metrics(self, spark) -> DataFrame:
        return self._read_table(spark, self.config.container_metrics_table)

    def channel_tags(self, spark) -> DataFrame:
        return self._read_table(spark, self.config.channel_tags_table)

    def channel_metrics(self, spark) -> DataFrame:
        return self._read_table(spark, self.config.channel_metrics_table)

    def channels(self, spark) -> DataFrame:
        return self._read_table(spark, self.config.channels_uri)

    def channel_uri(self):
        return self.config.channels_uri
