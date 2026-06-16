"""NuScenes-specific writes for the foundation tables.

`ingest_metadata` populates `container_tags`, `container_metrics`, and
`channel_tags` with NuScenes-specific content. The generic notebook 01 calls
this after creating empty ADAS tables.

`perception_paths` yields one dict per sensor file (camera + LiDAR; radar
intentionally omitted from the demo's scenario-search path). The generic
notebook 03 collects these into a DataFrame and writes `perception_channels`.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

from pyspark.sql import Row

import impulse_query_engine.schema as core_schema

from .loader import DERIVED_CHANNEL_IDS, SENSOR_CHANNEL_IDS, NuScenesLoader


# Static metadata for derived channels — needed to populate detection_class /
# detection_aggregate.
DERIVED_CHANNEL_META: dict[str, dict[str, str]] = {
    "Vehicle_Speed_kph":              {"unit": "kph",  "kind": "kinematic"},
    "Vehicle_Accel_Longitudinal_ms2": {"unit": "m/s2", "kind": "kinematic"},
    "Steering_Angle_deg":             {"unit": "deg",  "kind": "kinematic"},
    "Pedestrian_Count":               {"unit": "count", "kind": "detection_aggregate", "detection_class": "pedestrian", "detection_aggregate": "count"},
    "Pedestrian_Nearest_Distance_m":  {"unit": "m",    "kind": "detection_aggregate", "detection_class": "pedestrian", "detection_aggregate": "nearest_distance"},
    "Vehicle_Count_Front":            {"unit": "count", "kind": "detection_aggregate", "detection_class": "car",        "detection_aggregate": "count_front"},
    "Vehicle_Nearest_Distance_m":     {"unit": "m",    "kind": "detection_aggregate", "detection_class": "car",        "detection_aggregate": "nearest_distance"},
    "Cyclist_Count":                  {"unit": "count", "kind": "detection_aggregate", "detection_class": "cyclist",    "detection_aggregate": "count"},
    "Cyclist_Nearest_Distance_m":     {"unit": "m",    "kind": "detection_aggregate", "detection_class": "cyclist",    "detection_aggregate": "nearest_distance"},
}


def _sensor_type(name: str) -> str:
    if name.startswith("CAM"):
        return "camera"
    if name.startswith("LIDAR"):
        return "lidar"
    if name.startswith("RADAR"):
        return "radar"
    return "other"


def ingest_metadata(spark, loader: NuScenesLoader, cfg) -> None:
    """Write container_tags, container_metrics, and channel_tags for every scene.

    `cfg` is a BYODConfig instance; only `t_container_tags`, `t_container_metrics`,
    `t_channel_tags`, and `dataset_version` are read."""
    scenes = list(loader.scenes())

    # ── container_tags ──
    container_tag_rows: list[Row] = []
    for s in scenes:
        container_tag_rows.extend([
            Row(container_id=s.container_id, key="scene_name", value=s.name),
            Row(container_id=s.container_id, key="scene_description", value=s.description),
            Row(container_id=s.container_id, key="scene_token", value=s.scene_token),
            Row(container_id=s.container_id, key="log_token", value=s.log_token),
            Row(container_id=s.container_id, key="nbr_samples", value=str(s.nbr_samples)),
            Row(container_id=s.container_id, key="dataset_source", value="nuscenes"),
            Row(container_id=s.container_id, key="dataset_version", value=cfg.dataset_version),
        ])
    container_tags_df = spark.createDataFrame(container_tag_rows, core_schema.CONTAINER_TAGS)
    container_tags_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_container_tags)

    # ── container_metrics ──
    container_metrics_rows: list[Row] = []
    for s in scenes:
        samples = list(loader.samples_in_scene(s))
        if not samples:
            continue
        start_us = samples[0].timestamp_us
        stop_us = samples[-1].timestamp_us
        duration_ms = (stop_us - start_us) // 1000
        container_metrics_rows.append(Row(
            container_id=s.container_id,
            start_dt=datetime.fromtimestamp(start_us / 1e6, tz=timezone.utc),
            stop_dt=datetime.fromtimestamp(stop_us / 1e6, tz=timezone.utc),
            duration_ms=int(duration_ms),
            num_channels=len(SENSOR_CHANNEL_IDS) + len(DERIVED_CHANNEL_IDS),
        ))
    container_metrics_df = spark.createDataFrame(container_metrics_rows, core_schema.CONTAINER_METRICS)
    container_metrics_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_container_metrics)

    # ── channel_tags ──
    channel_tag_rows: list[Row] = []
    for s in scenes:
        for sensor_name, channel_id in SENSOR_CHANNEL_IDS.items():
            sensor_type = _sensor_type(sensor_name)
            channel_tag_rows.extend([
                Row(container_id=s.container_id, channel_id=channel_id, key="channel_name", value=sensor_name),
                Row(container_id=s.container_id, channel_id=channel_id, key="sensor_type", value=sensor_type),
                Row(container_id=s.container_id, channel_id=channel_id, key="modality", value="binary_file"),
            ])
        for channel_name, channel_id in DERIVED_CHANNEL_IDS.items():
            meta = DERIVED_CHANNEL_META[channel_name]
            channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="channel_name", value=channel_name))
            channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="unit", value=meta["unit"]))
            channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="kind", value=meta["kind"]))
            channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="source", value="synthesized_from_nuscenes"))
            if "detection_class" in meta:
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="detection_class", value=meta["detection_class"]))
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="detection_aggregate", value=meta["detection_aggregate"]))
    channel_tags_df = spark.createDataFrame(channel_tag_rows, core_schema.CHANNEL_TAGS)
    channel_tags_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_channel_tags)


# ── perception_channels (notebook 03 dispatch) ──────────────────────────────

ALLOWED_SENSORS = {"camera", "lidar"}  # skip radar — not used in the demo's scenario search


def perception_paths(loader: NuScenesLoader) -> Iterator[dict]:
    """Yield perception_channels rows for every camera + LiDAR keyframe file.

    Keyframe-only: object_tracks and the LiDAR/camera detections are all derived
    from keyframe annotations (2 Hz), and notebook 04 only visualizes frames inside
    detected event windows — all keyframes. Indexing the non-keyframe `sweeps/`
    files would point perception_channels at paths that need not exist on disk (the
    demo stages only the keyframe `samples/`), so we walk keyframe sample_data only.
    """
    for scene in loader.scenes():
        for sample in loader.samples_in_scene(scene):
            for sd in loader.sample_data_for_sample(sample):
                if _sensor_type(sd.sensor_name) not in ALLOWED_SENSORS:
                    continue
                yield {
                    "container_id": sd.container_id,
                    "channel_id": sd.channel_id,
                    "timestamp": sd.timestamp_us,
                    "file_path": sd.file_path,
                    "format": sd.fileformat,
                }
