"""Adapter Protocol + registry for the BYOD (Bring Your Own Data) demo.

Every dataset (NuScenes, A2D2, customer MDF4) plugs into the demo via an
adapter implementing this Protocol. Notebooks 03–09 are byte-identical across
adapters; notebooks 00–02 dispatch through the adapter for dataset-specific
download, ingest, and scalar derivation. Notebooks 04, 06, 06b, 07 call into
the adapter only for per-scene row assembly — all geometry, schema encoding,
window-gating, and OpenLABEL export live in `mda_query_engine.perception.*`
core modules.

Adapter authorship contract:
- Each adapter is a Python package under `demos/byod/adapters/<name>/`.
- The package's `__init__.py` exposes an `Adapter` class implementing this
  Protocol and calls `register("<name>", Adapter)` at import time.
- The adapter ships a `config.yaml` describing its `dataset_versions`,
  `dataroot_template`, `visualize_format`, and `openlabel_metadata`.

The generic notebooks resolve the adapter by name from the `BYODConfig`:
    adapter_cls = resolve(cfg.adapter_name)
    adapter = adapter_cls(cfg)
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


# A Scene is whatever the adapter wants it to be — a frozen dataclass, a dict,
# anything. The generic notebooks only require it expose `.container_id: int`
# and `.name: str` for logging and join keys.
SceneLike = Any


@runtime_checkable
class Adapter(Protocol):
    """The contract every BYOD adapter implements.

    Adapter instances hold a per-session dataset loader. The generic notebooks
    construct one adapter per notebook run and call its methods directly; they
    never touch dataset-specific types (e.g. `nuscenes-devkit`) themselves.
    """

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def download(self, dataroot: Path) -> None:
        """Download or validate the dataset under `dataroot`. Optional — the
        default registered adapters either no-op (data already provisioned) or
        delegate to a CLI tool. Must be idempotent."""

    # ── Foundation (notebook 01) ────────────────────────────────────────────

    def ingest_metadata(self, spark) -> None:
        """Write the dataset-specific `container_tags`, `container_metrics`,
        and `channel_tags` rows. The generic notebook 01 has already created
        empty LakeVision tables — this method only writes data."""

    # ── Phase 1 (notebook 02) ───────────────────────────────────────────────

    def scalar_source(self, spark):
        """Return a Spark DataFrame matching `mda_query_engine.schema.CHANNELS_SCHEMA`
        (container_id LONG, channel_id INT, tstart LONG, tend LONG, value DOUBLE).

        Adapters synthesize (NuScenes), bus-decode (A2D2), or MDF4-decode rows.
        The generic notebook 02 writes the DataFrame to `cfg.t_channels`."""

    # ── Foundation (notebook 03) ────────────────────────────────────────────

    def perception_paths(self) -> Iterable[dict]:
        """Yield `perception_channels` rows: dicts with keys
        `container_id`, `channel_id`, `timestamp`, `file_path`, `format`.
        One row per camera frame and LiDAR scan; radar typically omitted."""

    # ── Phase 2 (notebook 04) ───────────────────────────────────────────────

    def scenes(self) -> list[SceneLike]:
        """Return all scenes in the dataset. Each must expose
        `container_id: int` and `name: str`."""

    def map_to_object_tracks(
        self, scene: SceneLike, min_confidence: float
    ) -> list[dict]:
        """Return `object_tracks` rows for one scene. Thin row assembly only —
        geometry/encoding helpers live in `mda_query_engine.perception.geometry`."""

    # ── Phase 4 (notebooks 06, 06b) ─────────────────────────────────────────

    def map_to_lidar_detections(
        self, scene: SceneLike, event_windows: list[tuple[int, int]]
    ) -> list[dict]:
        """Return `lidar_object_detections` rows for one scene, restricted to
        annotations whose timestamp falls inside any of `event_windows`."""

    def map_to_camera_detections(
        self, scene: SceneLike, event_windows: list[tuple[int, int]]
    ) -> list[dict]:
        """Return `camera_object_detections` rows for one scene, same windowing
        contract as `map_to_lidar_detections`."""

    # ── Export + visualize (notebooks 07, 08) ───────────────────────────────

    def openlabel_metadata(self) -> dict:
        """Per-dataset strings injected into the OpenLABEL package metadata.
        Keys: `annotator`, `exporter`, `stream_description_prefix`."""

    def visualize_format(self) -> dict:
        """Per-dataset format hints used by notebook 08. Keys:
        - `lidar_dtype`: numpy dtype name (e.g. "float32")
        - `lidar_stride`: floats per point (e.g. 5 for NuScenes)
        - `camera_reader`: "pil" (Pillow) | "opencv"
        - `lidar_reader`: "fromfile" | "npz"
        """


# ── Registry ────────────────────────────────────────────────────────────────

_REGISTRY: dict[str, type] = {}


def register(name: str, adapter_cls: type) -> None:
    """Register an adapter class under a string name. Idempotent on re-register
    with the same class object (lets adapter modules be imported twice safely)."""
    existing = _REGISTRY.get(name)
    if existing is not None and existing is not adapter_cls:
        raise ValueError(
            f"adapter '{name}' already registered as {existing.__module__}.{existing.__qualname__}; "
            f"refusing to overwrite with {adapter_cls.__module__}.{adapter_cls.__qualname__}"
        )
    _REGISTRY[name] = adapter_cls


def resolve(name: str) -> type:
    """Return the adapter class registered under `name`. Adapter packages
    register themselves on first import — this function imports the package
    `demos.byod.adapters.<name>` if it isn't already in the registry."""
    if name in _REGISTRY:
        return _REGISTRY[name]
    # Lazy import — the package's __init__ calls register().
    import importlib
    importlib.import_module(f"adapters.{name}")
    if name not in _REGISTRY:
        raise KeyError(
            f"adapter '{name}' did not register itself on import — check "
            f"adapters/{name}/__init__.py for a `register('{name}', Adapter)` call"
        )
    return _REGISTRY[name]


def registered_names() -> list[str]:
    """Snapshot of currently-registered adapter names."""
    return sorted(_REGISTRY.keys())
