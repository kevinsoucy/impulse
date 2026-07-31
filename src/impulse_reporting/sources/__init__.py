"""Public source-adapter API."""

from .registry import register_source, registered_sources, resolve_source
from .source_adapter import (
    ContainerDimension,
    LogicalChannel,
    ResolvedChannelMapping,
    SourceAdapter,
    SourceSelectionError,
)

__all__ = [
    "ContainerDimension",
    "LogicalChannel",
    "ResolvedChannelMapping",
    "SourceAdapter",
    "SourceSelectionError",
    "register_source",
    "registered_sources",
    "resolve_source",
]
