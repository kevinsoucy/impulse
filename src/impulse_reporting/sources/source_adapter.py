"""Public source-adapter contract for intent-first Impulse setup."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from impulse_reporting.core.report import Report


class SourceSelectionError(ValueError):
    """A source, dimension, value, or logical-channel selection is invalid."""


@dataclass(frozen=True)
class ContainerDimension:
    """A business dimension by which a source can scope containers."""

    name: str
    label: str | None = None
    required: bool = False
    multi_value: bool = True


@dataclass(frozen=True)
class LogicalChannel:
    """A stable logical channel exposed independently of physical storage names."""

    name: str
    label: str | None = None
    unit: str | None = None
    physical_channels: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResolvedChannelMapping:
    """One logical-to-physical resolution selected by an adapter."""

    logical_channel: str
    physical_channel: str
    priority: int | None = None
    container_id: str | int | None = None


class SourceAdapter(ABC):
    """Discover source data and construct a standard configured Impulse report."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable registry name for this source."""

    @abstractmethod
    def list_dimensions(self, spark: Any) -> Sequence[ContainerDimension]:
        """Return the source's self-described container dimensions."""

    @abstractmethod
    def list_dimension_values(
        self,
        spark: Any,
        dimension: str,
        *,
        container_filters: Mapping[str, Sequence[str]] | None = None,
    ) -> Sequence[str]:
        """Return valid values for a dimension, optionally under other filters."""

    @abstractmethod
    def list_channels(
        self,
        spark: Any,
        *,
        container_filters: Mapping[str, Sequence[str]] | None = None,
    ) -> Sequence[LogicalChannel]:
        """Return logical channels available under the selected container scope."""

    @abstractmethod
    def create_report(
        self,
        spark: Any,
        *,
        name: str,
        container_filters: Mapping[str, Sequence[str]] | None,
        channels: Sequence[str],
        sink: Mapping[str, str] | None = None,
    ) -> Report:
        """Return the ordinary Impulse Report; omit *sink* for sinkless operation."""

    @abstractmethod
    def resolve_channel_mappings(
        self,
        spark: Any,
        *,
        container_filters: Mapping[str, Sequence[str]] | None,
        channels: Sequence[str],
    ) -> Sequence[ResolvedChannelMapping]:
        """Expose the logical-to-physical mappings used for diagnostics."""

    def validate_container_filters(
        self,
        spark: Any,
        container_filters: Mapping[str, Sequence[str]] | None,
        *,
        required: bool = False,
    ) -> dict[str, tuple[str, ...]]:
        """Resolve names case-insensitively and reject empty or missing selections."""
        if not container_filters:
            if required:
                raise SourceSelectionError(
                    f"Source {self.name!r} requires at least one container-dimension selection."
                )
            return {}

        dimensions = [dimension.name for dimension in self.list_dimensions(spark)]
        normalized: dict[str, tuple[str, ...]] = {}
        for requested_dimension, requested_values in container_filters.items():
            dimension = self._resolve_name("dimension", requested_dimension, dimensions)
            if isinstance(requested_values, str) or not requested_values:
                raise SourceSelectionError(
                    f"Dimension {dimension!r} requires a non-empty sequence of values."
                )
            available = list(self.list_dimension_values(spark, dimension))
            values = tuple(
                self._resolve_name(f"value for dimension {dimension!r}", value, available)
                for value in requested_values
            )
            normalized[dimension] = tuple(dict.fromkeys(values))
        return normalized

    def validate_channels(
        self,
        spark: Any,
        channels: Sequence[str],
        *,
        container_filters: Mapping[str, Sequence[str]] | None = None,
    ) -> tuple[str, ...]:
        """Resolve logical channels and reject empty, missing, or ambiguous names."""
        if isinstance(channels, str) or not channels:
            raise SourceSelectionError("Select at least one logical channel.")
        available = [
            channel.name
            for channel in self.list_channels(spark, container_filters=container_filters)
        ]
        resolved = tuple(
            self._resolve_name("logical channel", value, available) for value in channels
        )
        return tuple(dict.fromkeys(resolved))

    @staticmethod
    def _resolve_name(kind: str, requested: str, available: Sequence[str]) -> str:
        if requested in available:
            return requested
        matches = [value for value in available if value.casefold() == requested.casefold()]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise SourceSelectionError(
                f"Ambiguous {kind} {requested!r}; matches {sorted(matches)!r}. Use the exact name."
            )
        preview = sorted(available)[:20]
        raise SourceSelectionError(
            f"Unknown {kind} {requested!r}. Available values (first 20): {preview!r}."
        )
