"""Deterministic registry for built-in and customer query solvers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

from .query_solver import QuerySolver
from .solver_config import SolverConfig

_SolverT = TypeVar("_SolverT", bound=type[QuerySolver])


@dataclass(frozen=True)
class SolverRegistration:
    """A solver class and the config class used to validate its settings."""

    solver_cls: type[QuerySolver]
    config_cls: type[SolverConfig]


_REGISTRY: dict[str, SolverRegistration] = {}


def register_solver(
    name: str,
    config_cls: type[SolverConfig] = SolverConfig,
    *,
    aliases: tuple[str, ...] = (),
    overwrite: bool = False,
) -> Callable[[_SolverT], _SolverT]:
    """Register a solver class under a stable configuration name."""

    def decorator(solver_cls: _SolverT) -> _SolverT:
        if not (isinstance(solver_cls, type) and issubclass(solver_cls, QuerySolver)):
            raise TypeError(f"register_solver expects a QuerySolver subclass, got {solver_cls!r}.")
        registration = SolverRegistration(solver_cls, config_cls)
        for key in (name, *aliases):
            existing = _REGISTRY.get(key)
            if existing is not None and existing.solver_cls is not solver_cls and not overwrite:
                raise ValueError(
                    f"Solver {key!r} is already registered to {existing.solver_cls.__name__}; "
                    "pass overwrite=True to replace it."
                )
            _REGISTRY[key] = registration
        return solver_cls

    return decorator


def is_registered(name: str) -> bool:
    """Return whether *name* resolves to a registered solver."""
    return name in _REGISTRY


def registered_names() -> list[str]:
    """Return registered solver names in deterministic order."""
    return sorted(_REGISTRY)


def resolve_registration(name: str) -> SolverRegistration:
    """Resolve a solver name or raise with the available names."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"Unknown solver {name!r}. Registered solvers: {registered_names()}. "
            "Import the package that registers it before building the report."
        ) from None
