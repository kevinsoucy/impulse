"""Row-grouped query surfaces — declarative registration of external tables
shaped as (container_id, timestamp, [group_col], wide row) so the engine can
evaluate predicates against them and compose results with scalar channels."""

from __future__ import annotations

from dataclasses import dataclass, field

import pyspark.sql.types as T


@dataclass(frozen=True)
class RowGroupedSurface:
    """Declarative registration of a row-grouped table as a query surface.

    A row-grouped surface is a table where multiple rows can share the same
    ``(container_id, timestamp)`` — for example per-frame object detections,
    per-cycle defect inspections, or per-DTC occurrences. The natural key
    includes a per-entity ``group_col`` (or the table allows multiplicity
    without one). Each row carries a wide row of mixed-type attributes.

    This is the property that distinguishes a row-grouped surface from a
    scalar channel: channels assume one value per ``(channel, interval)``,
    surfaces do not.

    Point-in-time shape only in this minimal pass: each row is a sample at
    ``timestamp_col`` and the engine synthesizes ``[timestamp_col, next_ts)``
    intervals via per-group ``shift(-1)`` at evaluation time.

    Parameters
    ----------
    name : str
        Surface name. Used as the ``leaf_kind`` discriminator on selectors
        and as the registry key on the solver. Must be unique per deployment.
    schema : pyspark.sql.types.StructType
        Spark schema describing the surface's columns. The accessor reflects
        this schema to expose typed column proxies; only numeric columns
        are reachable through the accessor today.
    timestamp_col : str
        Name of the timestamp column. Each row carries a sample at this
        moment; the selector synthesizes intervals via per-group
        ``shift(-1)``.
    group_col : str or tuple of str, optional
        Name of the per-entity identity column (e.g. ``object_id``,
        ``station_id``). When set, interval synthesis partitions by this
        column so each entity's timeline closes at its own next sample
        rather than at the next sample globally. ``None`` means no
        partitioning — the surface is treated as one timeline per
        container.
    """

    name: str
    schema: T.StructType
    timestamp_col: str
    group_col: str | tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        field_names = {f.name for f in self.schema.fields}
        if self.timestamp_col not in field_names:
            raise ValueError(
                f"timestamp_col {self.timestamp_col!r} not in schema fields: "
                f"{sorted(field_names)}"
            )
        if self.group_col is not None:
            cols = (self.group_col,) if isinstance(self.group_col, str) else self.group_col
            missing = [c for c in cols if c not in field_names]
            if missing:
                raise ValueError(
                    f"group_col references unknown columns {missing}; "
                    f"schema fields are {sorted(field_names)}"
                )
        if "container_id" not in field_names:
            raise ValueError(
                "row-grouped surfaces must include a 'container_id' column "
                "(partition key shared with channels)"
            )

    @property
    def group_cols(self) -> tuple[str, ...]:
        """Normalized group columns as a tuple. Empty when no group_col is set."""
        if self.group_col is None:
            return ()
        if isinstance(self.group_col, str):
            return (self.group_col,)
        return tuple(self.group_col)
