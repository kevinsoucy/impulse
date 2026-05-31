from enum import Enum

from pyspark.sql.types import StructType

from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.events.container_event import ContainerEvent
from impulse_reporting.events.entity_event import EntityEvent
from impulse_reporting.events.sequence_of_events import SequenceOfEvents
from impulse_reporting.persist.dimension_schema import EVENT_DIMENSION_SCHEMA
from impulse_reporting.persist.fact_schema import EVENT_INSTANCE_FACT_SCHEMA


class EventType(Enum):
    """
    Enumeration of available event types.

    Defines the supported event types and their associated metadata
    including table names and schemas.

    Attributes
    ----------
    ENTITY_EVENT : EntityEvent
        Per-entity event type. Declared before ``BASIC_EVENT`` because
        ``EntityEvent`` subclasses ``BasicEvent``: the ``isinstance``-based
        grouping in ``Report._group_events_by_type`` matches the first enum
        member an event is an instance of, so the more specific type must come
        first or every ``EntityEvent`` would be misfiled as a ``BasicEvent``.
    BASIC_EVENT : BasicEvent
        Basic event type for standard event processing.
    CONTAINER_EVENT : ContainerEvent
        Container event type spanning the full measurement container.
    SEQUENCE_OF_EVENTS : SequenceOfEvents
        Sequence-of-events type for ordered interval sequence detection.

    """

    ENTITY_EVENT = EntityEvent
    BASIC_EVENT = BasicEvent
    CONTAINER_EVENT = ContainerEvent
    SEQUENCE_OF_EVENTS = SequenceOfEvents

    def get_fact_table_name(self) -> str:
        """
        Get the fact table name for the event type.

        Returns
        -------
        str
            The name of the fact table associated with this event type.

        Raises
        ------
        ValueError
            If the event type is not supported.
        """
        match self:
            case (
                EventType.ENTITY_EVENT
                | EventType.BASIC_EVENT
                | EventType.CONTAINER_EVENT
                | EventType.SEQUENCE_OF_EVENTS
            ):
                return "event_instance_fact"
            case _:
                raise ValueError(f"Unsupported event type: {self}")

    def get_fact_schema(self) -> StructType:
        """
        Get the fact schema for the event type.

        Returns
        -------
        StructType
            The PySpark schema structure for this event type.

        Raises
        ------
        ValueError
            If the event type is not supported.
        """
        match self:
            case (
                EventType.ENTITY_EVENT
                | EventType.BASIC_EVENT
                | EventType.CONTAINER_EVENT
                | EventType.SEQUENCE_OF_EVENTS
            ):
                return EVENT_INSTANCE_FACT_SCHEMA
            case _:
                raise ValueError(f"Unsupported event type: {self}")

    def get_dimension_table_name(self) -> str:
        """
        Get the dimension table name for the event type.

        Returns
        -------
        str
            The name of the dimension table associated with this event type.

        Raises
        ------
        ValueError
            If the event type is not supported.
        """
        match self:
            case (
                EventType.ENTITY_EVENT
                | EventType.BASIC_EVENT
                | EventType.CONTAINER_EVENT
                | EventType.SEQUENCE_OF_EVENTS
            ):
                return "event_dimension"
            case _:
                raise ValueError(f"Unsupported aggregation type: {self}")

    def get_dimension_schema(self) -> StructType:
        """
        Get the dimension schema for the event type.
        Returns
        -------
        StructType
            The PySpark schema structure for this event type.
        Raises
        ------
        ValueError
            If the event type is not supported.
        """
        match self:
            case (
                EventType.ENTITY_EVENT
                | EventType.BASIC_EVENT
                | EventType.CONTAINER_EVENT
                | EventType.SEQUENCE_OF_EVENTS
            ):
                return EVENT_DIMENSION_SCHEMA
            case _:
                raise ValueError(f"Unsupported event type: {self}")

    @classmethod
    def get_any_for_fact_table(cls, table_name: str) -> "EventType":
        """Return the first EventType whose fact table name matches.

        Multiple event types share one fact table, and callers use the returned
        type's fact schema to write *all* rows on that table. This asserts every
        matching type declares the same fact schema, so a new event type with a
        divergent schema on a shared table fails loudly here instead of silently
        writing rows under the wrong schema.

        Parameters
        ----------
        table_name : str
            Fact table name to look up.

        Returns
        -------
        EventType

        Raises
        ------
        ValueError
            If no EventType matches the given table name, or if matching types
            declare divergent fact schemas.
        """
        matches = [et for et in cls if et.get_fact_table_name() == table_name]
        if not matches:
            raise ValueError(f"No EventType found for fact table: {table_name}")
        first, *rest = matches
        for et in rest:
            if et.get_fact_schema() != first.get_fact_schema():
                raise ValueError(
                    f"EventTypes on fact table {table_name!r} declare divergent fact "
                    f"schemas ({first.name} vs {et.name}); the shared-table writer "
                    f"requires one schema per fact table."
                )
        return first

    @classmethod
    def get_any_for_dimension_table(cls, table_name: str) -> "EventType":
        """Return the first EventType whose dimension table name matches.

        Parameters
        ----------
        table_name : str
            Dimension table name to look up.

        Returns
        -------
        EventType

        Raises
        ------
        ValueError
            If no EventType matches the given table name.
        """
        for et in cls:
            if et.get_dimension_table_name() == table_name:
                return et
        raise ValueError(f"No EventType found for dimension table: {table_name}")
