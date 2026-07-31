"""Deterministic registry for installed/configured source adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .source_adapter import SourceAdapter

SourceFactory = Callable[[], SourceAdapter]


@dataclass(frozen=True)
class SourceRegistration:
    """A registered source factory and whether it is the configured default."""

    factory: SourceFactory
    default: bool = False


_REGISTRY: dict[str, SourceRegistration] = {}


def register_source(
    name: str,
    source: SourceAdapter | SourceFactory | type[SourceAdapter] | None = None,
    *,
    default: bool = False,
    overwrite: bool = False,
):
    """Register an adapter instance, factory, or class under a stable name.

    With ``source=None`` this acts as a class decorator. Registration is explicit
    and import-driven; the framework never scans source files or guesses imports.
    """

    def apply(candidate):
        if isinstance(candidate, SourceAdapter):
            factory = lambda: candidate
        elif callable(candidate):
            factory = candidate
        else:
            raise TypeError("register_source expects a SourceAdapter, factory, or adapter class.")

        existing = _REGISTRY.get(name)
        if existing is not None and existing.factory is not factory and not overwrite:
            raise ValueError(
                f"Source {name!r} is already registered; pass overwrite=True to replace it."
            )
        if default:
            other_defaults = [
                key for key, value in _REGISTRY.items() if value.default and key != name
            ]
            if other_defaults and not overwrite:
                raise ValueError(
                    f"Cannot register {name!r} as default; default source(s) already exist: "
                    f"{sorted(other_defaults)!r}."
                )
        _REGISTRY[name] = SourceRegistration(factory=factory, default=default)
        return candidate

    return apply if source is None else apply(source)


def registered_sources() -> list[str]:
    """Return source names in deterministic order."""
    return sorted(_REGISTRY)


def resolve_source(name: str | None = None) -> SourceAdapter:
    """Resolve a named source, or the unambiguous configured/default source."""
    if name is not None:
        try:
            registration = _REGISTRY[name]
        except KeyError:
            raise KeyError(
                f"Unknown source {name!r}. Registered sources: {registered_sources()}. "
                "Import the package that registers it before resolving the source."
            ) from None
        adapter = registration.factory()
        if not isinstance(adapter, SourceAdapter):
            raise TypeError(f"Source factory {name!r} returned {adapter!r}, not a SourceAdapter.")
        return adapter

    defaults = sorted(key for key, value in _REGISTRY.items() if value.default)
    if len(defaults) == 1:
        return resolve_source(defaults[0])
    if len(defaults) > 1:
        raise LookupError(f"Ambiguous default source; registered defaults: {defaults!r}.")
    names = registered_sources()
    if len(names) == 1:
        return resolve_source(names[0])
    if not names:
        raise LookupError(
            "No source adapters are registered. Import or install a source package first."
        )
    raise LookupError(
        f"Multiple sources are registered and none is default: {names!r}. Select one by name."
    )


def _clear_registry() -> None:
    """Clear registrations for isolated unit tests."""
    _REGISTRY.clear()
