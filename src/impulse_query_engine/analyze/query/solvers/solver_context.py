"""Construction inputs shared by registered query solvers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .solver_config import SolverConfig

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


@dataclass(frozen=True)
class SolverBuildContext:
    """Inputs resolved from a report configuration for solver construction."""

    spark: SparkSession
    solver_config: SolverConfig | None = None
    is_raw_data: bool = False
    drop_implausible_data: bool = False
