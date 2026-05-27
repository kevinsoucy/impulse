"""NuScenes adapter for the BYOD demo.

Implements the `lib.adapter.Adapter` Protocol. Registers itself as `"nuscenes"`
on import. The generic notebooks instantiate this class with a BYODConfig:

    from lib.adapter import resolve
    from lib.byod_config import BYODConfig

    cfg = BYODConfig.for_adapter("nuscenes", dataset_version="v1.0-mini")
    adapter = resolve("nuscenes")(cfg)
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
from .loader import NuScenesLoader, Scene


class Adapter:
    """NuScenes adapter — implementation of `lib.adapter.Adapter` Protocol."""

    def __init__(self, cfg) -> None:
        self._cfg = cfg
        self._loader: NuScenesLoader | None = None

    # ── Internal loader factory (lazy) ──────────────────────────────────────

    def _get_loader(self) -> NuScenesLoader:
        if self._loader is None:
            self._loader = NuScenesLoader(
                dataroot=self._cfg.dataroot,
                dataset_version=self._cfg.dataset_version,
                verbose=False,
            )
        return self._loader

    # ── Lifecycle ───────────────────────────────────────────────────────────

    def download(self, dataroot: Path) -> None:
        """No-op — NuScenes requires manual download (EULA acceptance).

        The script `demos/byod/00_download.py` handles the interactive flow
        for NuScenes. Programmatic dispatch here just validates the dataroot.
        """
        from .download import validate_dataroot
        validate_dataroot(Path(dataroot), self._cfg.dataset_version)

    # ── Notebook 01 ─────────────────────────────────────────────────────────

    def ingest_metadata(self, spark) -> None:
        _ingest.ingest_metadata(spark, self._get_loader(), self._cfg)

    # ── Notebook 02 ─────────────────────────────────────────────────────────

    def scalar_source(self, spark):
        """Return a Spark DataFrame matching CHANNELS_SCHEMA, derived from
        ego pose + annotations across every scene."""
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

    def map_to_object_tracks(
        self, scene: Scene, min_confidence: float
    ) -> list[dict]:
        return _object_tracks.map_scene_to_object_tracks(
            self._get_loader(), scene, min_confidence=min_confidence
        )

    # ── Notebooks 06, 06b ───────────────────────────────────────────────────

    def map_to_lidar_detections(
        self, scene: Scene, event_windows: list[tuple[int, int]]
    ) -> list[dict]:
        return _lidar.map_scene_for_event_windows(
            self._get_loader(), scene, event_windows
        )

    def map_to_camera_detections(
        self, scene: Scene, event_windows: list[tuple[int, int]]
    ) -> list[dict]:
        return _camera.map_scene_for_event_windows(
            self._get_loader(), scene, event_windows
        )

    # ── Notebook 07 ─────────────────────────────────────────────────────────

    def openlabel_metadata(self) -> dict:
        return self._cfg.openlabel_metadata()

    # ── Notebook 08 ─────────────────────────────────────────────────────────

    def visualize_format(self) -> dict:
        return self._cfg.visualize_format()


register("nuscenes", Adapter)
