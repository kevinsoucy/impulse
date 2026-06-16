"""BYODConfig — runtime configuration for the BYOD demo.

One config object holds the UC catalog/schema/volume layout, the selected
adapter name, and the adapter's per-dataset settings loaded from its
`config.yaml`. Every generic notebook constructs a `BYODConfig` from widget
values and the resolved adapter's YAML.

Layout produced (with default args, `adapter=nuscenes`, `dataset_version=v1.0-mini`):
  Catalog:                  main
  Silver schema:            main.demo_silver
  Perception silver schema: main.demo_perception_silver
  Gold schema:              main.demo_gold
  Raw data volume:          /Volumes/main/demo_silver/raw/<dataset_version>/
  Camera frames volume:     /Volumes/main/demo_silver/camera_frames/
  LiDAR scans volume:       /Volumes/main/demo_silver/lidar_scans/
  OpenLABEL exports volume: /Volumes/main/demo_silver/openlabel_packages/
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _load_yaml(path: Path) -> dict:
    """Load a YAML file without requiring PyYAML at import time.

    PyYAML is an optional dependency in the test environment; it is always
    available in the deployed Databricks runtime."""
    import yaml  # local import — not available in every test environment
    with open(path) as f:
        return yaml.safe_load(f) or {}


@dataclass(frozen=True)
class BYODConfig:
    # Adapter selection
    adapter_name: str  # e.g. "nuscenes", "a2d2"
    dataset_version: str  # adapter-specific identifier (e.g. "v1.0-mini")

    # Filesystem
    dataroot: str  # absolute path the adapter loader reads from

    # Unity Catalog targets
    catalog: str
    schema_silver: str
    schema_perception: str
    schema_gold: str

    # UC Volume targets for binary artifacts (paths only — ADR-P3)
    volume_camera_frames: str
    volume_lidar_scans: str
    volume_openlabel: str

    # Per-adapter settings loaded from adapter's config.yaml
    adapter_settings: dict = field(default_factory=dict)

    # ── Factory methods ─────────────────────────────────────────────────────

    @classmethod
    def for_adapter(
        cls,
        adapter_name: str,
        dataset_version: str | None = None,
        catalog: str = "main",
        schema_prefix: str = "demo",
        volume_root: str | None = None,
        adapter_yaml_path: Path | str | None = None,
    ) -> "BYODConfig":
        """Build a BYODConfig from an adapter name + UC namespace overrides.

        Loads the adapter's `config.yaml` and substitutes
        `{catalog}`, `{schema_prefix}`, `{version}` into its `dataroot_template`.

        If `adapter_yaml_path` is None, defaults to
        `demos/byod/adapters/<adapter_name>/config.yaml` relative to this file.
        """
        if adapter_yaml_path is None:
            adapter_yaml_path = (
                Path(__file__).resolve().parent.parent
                / "adapters" / adapter_name / "config.yaml"
            )
        adapter_settings = _load_yaml(Path(adapter_yaml_path))

        if dataset_version is None:
            dataset_version = adapter_settings.get("default_version") or adapter_settings.get(
                "dataset_versions", [None]
            )[0]
            if dataset_version is None:
                raise ValueError(
                    f"adapter '{adapter_name}' yaml has neither `default_version` nor "
                    f"`dataset_versions` — pass dataset_version explicitly"
                )

        # The bundle's `resources/catalog.yml` creates all UC Volumes under the
        # silver schema (`{schema_prefix}_silver`). `vroot` is the resulting
        # `/Volumes/{catalog}/{schema_prefix}_silver` namespace so the deployed
        # paths and the runtime dataroot_template substitution match.
        vroot = volume_root or f"/Volumes/{catalog}/{schema_prefix}_silver"
        dataroot_template: str = adapter_settings.get(
            "dataroot_template", "{vroot}/raw/{version}"
        )
        dataroot = dataroot_template.format(
            vroot=vroot,
            catalog=catalog,
            schema_prefix=schema_prefix,
            version=dataset_version,
        )

        return cls(
            adapter_name=adapter_name,
            dataset_version=dataset_version,
            dataroot=dataroot,
            catalog=catalog,
            schema_silver=f"{schema_prefix}_silver",
            schema_perception=f"{schema_prefix}_perception_silver",
            schema_gold=f"{schema_prefix}_gold",
            volume_camera_frames=f"{vroot}/camera_frames",
            volume_lidar_scans=f"{vroot}/lidar_scans",
            volume_openlabel=f"{vroot}/openlabel_packages",
            adapter_settings=adapter_settings,
        )

    @classmethod
    def from_yaml(cls, path: Path | str) -> "BYODConfig":
        """Build a BYODConfig from a `config.yaml` whose top-level includes
        `adapter_name` (in addition to the per-adapter settings). Useful for
        customer-curated config files that point at a specific dataroot."""
        settings = _load_yaml(Path(path))
        adapter_name = settings["adapter_name"]
        return cls.for_adapter(
            adapter_name=adapter_name,
            dataset_version=settings.get("dataset_version"),
            catalog=settings.get("catalog", "main"),
            schema_prefix=settings.get("schema_prefix", "demo"),
            volume_root=settings.get("volume_root"),
            adapter_yaml_path=path,
        )

    # ── Fully-qualified table names ─────────────────────────────────────────

    @property
    def t_channels(self) -> str:
        return f"{self.catalog}.{self.schema_silver}.channels"

    @property
    def t_channel_tags(self) -> str:
        return f"{self.catalog}.{self.schema_silver}.channel_tags"

    @property
    def t_container_tags(self) -> str:
        return f"{self.catalog}.{self.schema_silver}.container_tags"

    @property
    def t_perception_channels(self) -> str:
        return f"{self.catalog}.{self.schema_silver}.perception_channels"

    @property
    def t_object_tracks(self) -> str:
        return f"{self.catalog}.{self.schema_perception}.object_tracks"

    @property
    def t_ego_map_context(self) -> str:
        return f"{self.catalog}.{self.schema_perception}.ego_map_context"

    @property
    def t_object_map_context(self) -> str:
        return f"{self.catalog}.{self.schema_perception}.object_map_context"

    @property
    def map_expansion_dir(self) -> str:
        """Directory holding the nuScenes map-expansion `<location>.json` layer files.

        Staged by `stage_nuscenes_trainval.py` (with_map_expansion=true) under the
        silver volume. Optional — present only when map context is in use."""
        return f"/Volumes/{self.catalog}/{self.schema_silver}/raw/map-expansion-v1.3/expansion"

    @property
    def t_camera_object_detections(self) -> str:
        return f"{self.catalog}.{self.schema_perception}.camera_object_detections"

    @property
    def t_lidar_object_detections(self) -> str:
        return f"{self.catalog}.{self.schema_perception}.lidar_object_detections"

    @property
    def event_sink_prefix(self) -> str:
        """Gold table prefix for the 02_detect_events sink (what 03/04 read).

        Impulse writes one fact + one dimension table per Report, named
        `{prefix}_event_instance_fact` / `{prefix}_event_dimension`. The detect
        step uses `{adapter}_demo`; the explorer (05) uses its own `_explore`
        sink inline.
        """
        return f"{self.adapter_name}_demo"

    @property
    def t_event_instance_fact(self) -> str:
        return f"{self.catalog}.{self.schema_gold}.{self.event_sink_prefix}_event_instance_fact"

    @property
    def t_event_dimension(self) -> str:
        return f"{self.catalog}.{self.schema_gold}.{self.event_sink_prefix}_event_dimension"

    @property
    def t_container_metrics(self) -> str:
        return f"{self.catalog}.{self.schema_silver}.container_metrics"

    @property
    def t_channel_metrics(self) -> str:
        return f"{self.catalog}.{self.schema_silver}.channel_metrics"

    # ── DDL helpers ─────────────────────────────────────────────────────────

    def schemas_to_create(self) -> list[str]:
        return [self.schema_silver, self.schema_perception, self.schema_gold]

    def volumes_to_create(self) -> list[str]:
        return ["raw", "camera_frames", "lidar_scans", "openlabel_packages"]

    # ── Adapter settings access ─────────────────────────────────────────────

    def openlabel_metadata(self) -> dict:
        """Defaults for OpenLABEL export when adapter does not override."""
        return self.adapter_settings.get("openlabel", {
            "annotator": "ground_truth",
            "exporter": "demos/byod/lib/openlabel.py",
            "stream_description_prefix": "",
        })

    def visualize_format(self) -> dict:
        return self.adapter_settings.get("visualize_format", {
            "lidar_dtype": "float32",
            "lidar_stride": 5,
            "camera_reader": "pil",
            "lidar_reader": "fromfile",
        })

    def discover_signals(self, adapter: Any) -> list[str]:
        """Ask the adapter what signals it sees in the configured `dataroot`.

        Used by the validate-config step (notebook 00 in some adapters, or an
        ad-hoc dev session) to see which canonical channels can be filled
        before any data is written. Adapter must expose a `discover_signals`
        method; if it doesn't, returns the empty list.
        """
        discover = getattr(adapter, "discover_signals", None)
        if discover is None:
            return []
        return list(discover())
