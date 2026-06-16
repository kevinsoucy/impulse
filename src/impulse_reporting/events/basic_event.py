from __future__ import annotations

import hashlib
from collections.abc import Mapping

import pyspark.sql.functions as f
import zlib
from pyspark.sql import Row, SparkSession

from impulse_query_engine.analyze.metadata.time_series_expression import (
    TimeSeriesExpression,
)
from impulse_query_engine.analyze.query.query_builder import QueryBuilder
from impulse_query_engine.analyze.query.solvers.query_solver import QuerySolver
from impulse_reporting.events.event import Event
from impulse_reporting.persist.dimension_schema import EVENT_DIMENSION_SCHEMA
from impulse_reporting.persist.fact_schema import EVENT_INSTANCE_FACT_SCHEMA
from impulse_reporting.util.event_instance_util import generate_event_instance_id_column
from impulse_reporting.util.report_entity_util import ReportEntityUtil


class BasicEvent(Event):
    """Class representing a basic event in a report.

    ``BasicEvent`` is the single public event constructor. When its expression
    carries an entity-scoped leaf (``.each()`` / ``.ids()``), construction routes
    to the per-entity :class:`EntityEvent` implementation so the matched ids land
    in ``entity_key``; otherwise it stays a session-scoped presence event. A plain
    ``.any()`` (no ``.ids()``) over an entity-bearing series is still presence, so
    it remains a ``BasicEvent`` with a NULL ``entity_key``.
    """

    def __new__(cls, *args, **kwargs):
        # Route only when called as BasicEvent(...) with an expression that carries
        # an entity-scoped leaf; the returned instance is a BasicEvent subclass, so
        # Python runs __init__ once with the original args, dispatching the routed
        # class's _finalize_bare / _validate_expression hooks (no double init). A
        # direct EntityEvent(...) (or any other subclass) keeps its own class.
        #
        # __new__ must also be callable with no args: pickle reconstructs an
        # instance via cls.__new__(cls) (no __init__, __dict__ restored directly),
        # which is how an EntityEvent crosses the Spark UDF boundary. Pull expr
        # positionally or by keyword and only route when it is actually present.
        expr = kwargs.get("expr", args[1] if len(args) >= 2 else None)
        if cls is BasicEvent and expr is not None and cls._entity_leaves(expr):
            from impulse_reporting.events.entity_event import EntityEvent

            cls = EntityEvent
        return super().__new__(cls)

    def __init__(
        self,
        name: str,
        expr: TimeSeriesExpression,
        desc: str = None,
        required_channels: list[str] = None,
        attributes: Mapping[str, str] = None,
    ):
        """
        Initialize a BasicEvent object.

        Parameters
        ----------
        name : str
            Name of the event.
        expr : TimeSeriesExpression
            Time series expression for the event.
        desc : str, optional
            Description of the event.
        required_channels : list of str, optional
            List of required channels for the event.
        attributes : Mapping[str, str], optional
            Key-value metadata for the event (e.g. limit_type, limit_direction).
        """
        Event.__init__(self, name)
        expr = self._finalize_bare(expr)
        self._validate_expression(expr)
        self.expression = expr.alias(name)
        self.description = desc
        self.required_channels = required_channels
        normalized_attributes: dict[str, str] = {}
        if attributes is not None:
            normalized_attributes = {str(k): str(v) for k, v in attributes.items()}
        self.attributes = normalized_attributes

    @staticmethod
    def _series_leaves(expr: TimeSeriesExpression) -> list:
        from impulse_query_engine.surfaces.series_selector import SeriesSelector

        return [s for s in expr.get_selectors() if isinstance(s, SeriesSelector)]

    @staticmethod
    def _entity_leaves(expr: TimeSeriesExpression) -> list:
        return [s for s in BasicEvent._series_leaves(expr) if s.entity_scoped]

    def _finalize_bare(self, expr: TimeSeriesExpression) -> TimeSeriesExpression:
        """Finalize a bare ``_PartialPredicate`` as a presence leaf.

        A predicate over an entity-bearing series must pick a windowing verb
        first (``.any()`` / ``.each()``); only a non-entity predicate has a
        single sensible meaning (presence) and may be auto-finalized.
        """
        from impulse_query_engine.surfaces.partial_predicate import _PartialPredicate

        if isinstance(expr, _PartialPredicate):
            if expr._series.entity_key is not None:
                raise ValueError(
                    f"Predicate over entity series {expr._series.name!r} needs a "
                    "windowing verb: .any() for a presence window, or .each() for "
                    "per-entity (add .ids(as_=…) to project the matched ids)."
                )
            expr = expr._finalize_presence()
        return expr

    def _validate_expression(self, expr: TimeSeriesExpression) -> None:
        """A BasicEvent is session-scoped presence; it must not carry
        entity-scoped leaves — those need EntityEvent's per-entity path."""
        if self._entity_leaves(expr):
            raise ValueError(
                "Expression has entity-scoped leaves (.each() / .ids()); "
                "use EntityEvent, not BasicEvent, for per-entity reporting."
            )

    def get_id(self) -> int:
        """
        Returns a unique identifier for the event.

        Returns
        -------
        int
            Unique positive 32-bit integer identifier for the event.
        """
        hash_input = f"{self.name}"
        return zlib.crc32(hash_input.encode()) & 0x7FFFFFFF  # Ensures positive 32-bit int

    def get_expression(self) -> TimeSeriesExpression | None:
        """
        Get the time series expression associated with the event.

        Returns
        -------
        TimeSeriesExpression or None
            The time series expression for the event.
        """
        return self.expression

    def get_event_type_str(self) -> str:
        """Get the event type string for BasicEvent.

        Returns
        -------
        str
            Event type string.
        """
        return "BASIC_EVENT"

    def _definition_str(self) -> str:
        """The string whose hash defines this event's computation.

        For a basic event only the expression affects results. Subclasses that
        carry additional result-affecting parameters extend this so those
        parameters participate in the definition hash and a redefinition is
        detected as a change.
        """
        return self.get_expression_str()

    def determine_definition_hash(self) -> int:
        """
        Calculate definition hash for the event.

        Hashes :meth:`_definition_str` — the expression plus any subclass
        parameters that affect results. Excludes name, description,
        required_channels, report_id.

        Returns
        -------
        int
            Hash value representing the computation definition.
        """
        # Use SHA-256 and return as int (truncated to fit LongType)
        hash_bytes = hashlib.sha256(self._definition_str().encode()).digest()
        return int.from_bytes(hash_bytes[:8], byteorder="big", signed=True)

    def as_dict(self) -> dict:
        """
        Get a dictionary representation of the event.

        Returns
        -------
        dict
            Dictionary containing event metadata.
        """
        return {
            "event_id": self.get_id(),
            "report_id": self.report_id,
            "event_type": self.get_event_type_str(),
            "event_name": self.name,
            "event_description": self.description,
            "required_channels": self.required_channels,
            "event_expression": self.get_expression_str(),
            "definition_hash": self.determine_definition_hash(),
            "attributes": self.attributes,
        }

    def as_spark_row(self) -> Row:
        """
        Get a Spark Row representation of the event.

        Returns
        -------
        Row
            Spark Row containing event metadata.
        """
        return Row(**self.as_dict())

    @classmethod
    def determine_events(
        cls,
        spark: SparkSession,
        events: list[BasicEvent],
        *,
        solved_df: "DataFrame" = None,
        query: QueryBuilder = None,
        solver: QuerySolver = None,
        pre_filtered_containers_df=None,
    ):
        """
        Extract event fact table for the given list of BasicEvent objects.

        Parameters
        ----------
        spark : SparkSession
            Spark session for data processing.
        events : list of BasicEvent
            List of BasicEvent objects to process.
        solved_df : DataFrame, optional
            Pre-solved wide DataFrame from centralized batch solve. Required.
        query : QueryBuilder, optional
            Query builder (unused, kept for interface compatibility).
        solver : QuerySolver, optional
            Query solver (unused, kept for interface compatibility).
        pre_filtered_containers_df : DataFrame, optional
            Pre-filtered containers for incremental processing.

        Returns
        -------
        DataFrame
            Spark DataFrame containing event instance facts.
        """
        if solved_df is None:
            raise ValueError(
                "BasicEvent.determine_events requires solved_df. "
                "Provide a pre-solved DataFrame from the centralized batch-solve flow."
            )

        event_names = [event.get_name() for event in events]

        df = (
            solved_df.select("container_id", *event_names)
            .unpivot(
                f.col("container_id"),
                event_names,
                variableColumnName="event_name",
                valueColumnName="value",
            )
            .select(
                "container_id",
                "event_name",
                f.explode(f.col("value")).alias("event_instance"),
            )
            .withColumn("start_ts", f.col("event_instance").getItem(0))
            .withColumn("end_ts", f.col("event_instance").getItem(1))
            .withColumn(
                "event_instance_id",
                generate_event_instance_id_column(event_type=BasicEvent),
            )
            .withColumn(
                "event_id",
                ReportEntityUtil.get_event_id_column(elements=events, element_name="event_name"),
            )
            # BasicEvent has no per-entity scope; entity_key is NULL for these rows.
            # EntityEvent overrides the materialization path to populate it.
            .withColumn("entity_key", f.lit(None).cast("string"))
            .select(EVENT_INSTANCE_FACT_SCHEMA.fieldNames())
            .where(f.col("start_ts") < f.col("end_ts"))  # Ensure valid time intervals
        )
        return df

    @classmethod
    def determine_metadata_df(cls, spark: SparkSession, events: list[BasicEvent]):
        """
        Create a Spark DataFrame containing event metadata.

        Parameters
        ----------
        spark : SparkSession
            Spark session for data processing.
        events : list of BasicEvent
            List of BasicEvent objects.

        Returns
        -------
        DataFrame
            Spark DataFrame containing event metadata.
        """
        events = [event.as_spark_row() for event in events]
        return spark.createDataFrame(events, schema=EVENT_DIMENSION_SCHEMA)
