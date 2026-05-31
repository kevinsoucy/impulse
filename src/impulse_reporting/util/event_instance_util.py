"""Utility functions for event instance ID generation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pyspark.sql.functions as f
from pyspark.sql import Column

if TYPE_CHECKING:
    from impulse_reporting.events.event import Event


def generate_event_instance_id_column(
    event_type: type[Event] | None = None,
    container_id_col: str = "container_id",
    event_name_col: str = "event_name",
    start_ts_col: str = "start_ts",
    end_ts_col: str = "end_ts",
    entity_key_col: str | None = None,
) -> Column:
    """
    Generate an event_instance_id column.

    For ``ContainerEvent`` the sentinel value ``-1`` is returned because a
    container event produces exactly one instance per container.
    For all other event types a CRC32 hash of
    ``container_id::event_name::start_ts::end_ts`` is used.

    Parameters
    ----------
    event_type : type[Event] or None, optional
        The event class.  When the class is ``ContainerEvent``, ``container_id``
          column is used for CRC32 hash. For any other value (including ``None``
        for backward-compatibility) the CRC32 hash column is returned.
    container_id_col : str, optional
        Name of the container ID column, defaults to "container_id".
    event_name_col : str, optional
        Name of the event name column, defaults to "event_name".
    start_ts_col : str, optional
        Name of the start timestamp column, defaults to "start_ts".
    end_ts_col : str, optional
        Name of the end timestamp column, defaults to "end_ts".
    entity_key_col : str or None, optional
        Name of the entity_key column. When set (the ``EntityEvent`` path) it is
        folded into the hash so two distinct entities matching the *same* window
        of the *same* event get distinct instance ids. When ``None`` (default)
        the hash is unchanged, preserving every existing caller.

    Returns
    -------
    pyspark.sql.Column
        A column expression for the event_instance_id.
    """
    from impulse_reporting.events.container_event import ContainerEvent

    if event_type is ContainerEvent:
        return f.crc32(f.col(container_id_col).cast("string"))

    parts = [
        f.col(container_id_col),
        f.col(event_name_col),
        f.col(start_ts_col),
        f.col(end_ts_col),
    ]
    if entity_key_col is not None:
        parts.append(f.coalesce(f.col(entity_key_col), f.lit("")))
    return f.crc32(f.concat_ws("::", *parts))
