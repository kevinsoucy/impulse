"""ZOD adapter for the BYOD demo (BL-008).

Implements the `lib.adapter.Adapter` Protocol. Registers itself as `"zod"`.
The generic notebooks instantiate this class via:

    from lib.adapter import resolve
    from lib.byod_config import BYODConfig

    cfg = BYODConfig.for_adapter("zod")
    adapter = resolve("zod")(cfg)

What's different from the NuScenes and A2D2 adapters:

  * `scalar_source()` reads `oxts.hdf5` via h5py — third scalar-ingestion
    format after NuScenes synthesis and A2D2's bus_signals.json.
  * Annotations carry stable `uuid` across frames, so
    `object_tracks.relative_velocity_ms` IS computed (not None like A2D2).
  * Each annotation has `lidar_attribute` + `radar_attribute` + occlusion, so
    `source` becomes "lidar|radar|camera" — the only adapter where notebook 09's
    radar modality populates non-empty in `distance_stats_by_modality`.
  * Single fused 3-LiDAR stream → `sensor_id = "LIDAR_FUSED"` (same as A2D2).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from pyspark.sql import Row

import mda_query_engine.schema as core_schema

from lib.adapter import register

from . import camera as _camera
from . import ingest as _ingest
from . import lidar as _lidar
from . import object_tracks as _object_tracks
from . import scalar_source as _scalar_source
from .loader import Scene, ZodLoader


class Adapter:
    """ZOD adapter — implementation of `lib.adapter.Adapter` Protocol."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._loader: ZodLoader | None = None

    def _get_loader(self) -> ZodLoader:
        if self._loader is None:
            self._loader = ZodLoader(
                dataroot=self._cfg.dataroot,
                dataset_version=self._cfg.dataset_version,
            )
        return self._loader

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def download(self, dataroot: Path) -> None:
        from .download import validate_dataroot
        validate_dataroot(Path(dataroot), self._cfg.dataset_version)

    # ── Notebook 01 ─────────────────────────────────────────────────────────

    def ingest_metadata(self, spark) -> None:
        _ingest.ingest_metadata(spark, self._get_loader(), self._cfg)

    # ── Notebook 02 ─────────────────────────────────────────────────────────

    def scalar_source(self, spark):
        loader = self._get_loader()
        rows = [
            Row(**asdict(cv))
            for scene in loader.scenes()
            for cv in _scalar_source.derive_for_scene(loader, scene)
        ]
        return spark.createDataFrame(rows, core_schema.CHANNELS_SCHEMA)

    # ── Notebook 03 ─────────────────────────────────────────────────────────

    def perception_paths(self) -> Iterable[dict]:
        return _ingest.perception_paths(self._get_loader())

    # ── Notebook 04 ─────────────────────────────────────────────────────────

    def scenes(self) -> list[Scene]:
        return self._get_loader().list_scenes()

    def map_to_object_tracks(self, scene: Scene, min_confidence: float) -> list[dict]:
        return _object_tracks.map_scene_to_object_tracks(
            self._get_loader(), scene, min_confidence=min_confidence
        )

    # ── Notebooks 06, 06b ───────────────────────────────────────────────────

    def map_to_lidar_detections(
        self, scene: Scene, event_windows: list[tuple[int, int]]
    ) -> list[dict]:
        return _lidar.map_scene_for_event_windows(self._get_loader(), scene, event_windows)

    def map_to_camera_detections(
        self, scene: Scene, event_windows: list[tuple[int, int]]
    ) -> list[dict]:
        return _camera.map_scene_for_event_windows(self._get_loader(), scene, event_windows)

    # ── Notebook 07 ─────────────────────────────────────────────────────────

    def openlabel_metadata(self) -> dict:
        return self._cfg.openlabel_metadata()

    # ── Notebook 08 ─────────────────────────────────────────────────────────

    def visualize_format(self) -> dict:
        return self._cfg.visualize_format()


register("zod", Adapter)
