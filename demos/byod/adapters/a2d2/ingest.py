"""A2D2-specific writes for the foundation tables.

Same shape as the NuScenes adapter's `ingest.py`:
  - `ingest_metadata` writes container_tags, container_metrics, channel_tags.
  - `perception_paths` yields one perception_channels row per camera + LiDAR file.

The notebook 01 dispatcher provisions empty ADAS tables before this
runs; this module only writes data.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

from pyspark.sql import Row

import impulse_query_engine.schema as core_schema

from .loader import (
    BUS_SIGNAL_TO_CHANNEL,
    DERIVED_CHANNEL_IDS,
    SENSOR_CHANNEL_IDS,
    A2D2Loader,
    FRAME_DT_US,
)


# Static channel metadata. Distinct from NuScenes' `DERIVED_CHANNEL_META`
# because A2D2's bus signals are real measurements, not synthesized.
DERIVED_CHANNEL_META: dict[str, dict[str, str]] = {
    "Vehicle_Speed_kph":              {"unit": "kph",   "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Vehicle_Accel_Longitudinal_ms2": {"unit": "m/s2",  "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Vehicle_Accel_Lateral_ms2":      {"unit": "m/s2",  "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Vehicle_Accel_Vertical_ms2":     {"unit": "m/s2",  "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Yaw_Rate_rads":                  {"unit": "rad/s", "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Roll_Rate_rads":                 {"unit": "rad/s", "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Pitch_Rate_rads":                {"unit": "rad/s", "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Steering_Angle_deg":             {"unit": "deg",   "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Brake_Pressure_pct":             {"unit": "%",     "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Accelerator_Pedal_pct":          {"unit": "%",     "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Pitch_Angle_deg":                {"unit": "deg",   "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Roll_Angle_deg":                 {"unit": "deg",   "kind": "kinematic", "source": "a2d2_bus_signal"},
    "Latitude_deg":                   {"unit": "deg",   "kind": "gnss",      "source": "a2d2_bus_signal"},
    "Longitude_deg":                  {"unit": "deg",   "kind": "gnss",      "source": "a2d2_bus_signal"},
    # Detection-aggregate channels
    "Pedestrian_Count":               {"unit": "count", "kind": "detection_aggregate", "source": "a2d2_annotations",
                                       "detection_class": "pedestrian", "detection_aggregate": "count"},
    "Pedestrian_Nearest_Distance_m":  {"unit": "m",     "kind": "detection_aggregate", "source": "a2d2_annotations",
                                       "detection_class": "pedestrian", "detection_aggregate": "nearest_distance"},
    "Vehicle_Count_Front":            {"unit": "count", "kind": "detection_aggregate", "source": "a2d2_annotations",
                                       "detection_class": "car",        "detection_aggregate": "count_front"},
    "Vehicle_Nearest_Distance_m":     {"unit": "m",     "kind": "detection_aggregate", "source": "a2d2_annotations",
                                       "detection_class": "car",        "detection_aggregate": "nearest_distance"},
    "Cyclist_Count":                  {"unit": "count", "kind": "detection_aggregate", "source": "a2d2_annotations",
                                       "detection_class": "cyclist",    "detection_aggregate": "count"},
    "Cyclist_Nearest_Distance_m":     {"unit": "m",     "kind": "detection_aggregate", "source": "a2d2_annotations",
                                       "detection_class": "cyclist",    "detection_aggregate": "nearest_distance"},
}


def _sensor_type(name: str) -> str:
    if name.startswith("cam_"):
        return "camera"
    if name.startswith("lidar_"):
        return "lidar"
    return "other"


def ingest_metadata(spark, loader: A2D2Loader, cfg) -> None:
    """Write container_tags, container_metrics, and channel_tags for every A2D2 scene."""
    scenes = list(loader.scenes())

    # ── container_tags ──
    container_tag_rows: list[Row] = []
    for s in scenes:
        container_tag_rows.extend([
            Row(container_id=s.container_id, key="scene_name", value=s.name),
            Row(container_id=s.container_id, key="scene_description", value=s.description),
            Row(container_id=s.container_id, key="scene_id", value=s.scene_id),
            Row(container_id=s.container_id, key="nbr_samples", value=str(s.nbr_samples)),
            Row(container_id=s.container_id, key="dataset_source", value="a2d2"),
            Row(container_id=s.container_id, key="dataset_version", value=cfg.dataset_version),
            Row(container_id=s.container_id, key="dataset_license", value="CC BY-ND 4.0"),
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
        stop_us = samples[-1].timestamp_us + FRAME_DT_US
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
    bus_keys_by_adas_name = {v: k for k, v in BUS_SIGNAL_TO_CHANNEL.items()}
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
            channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="source", value=meta["source"]))
            raw_bus_key = bus_keys_by_adas_name.get(channel_name)
            if raw_bus_key:
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="bus_signal_key", value=raw_bus_key))
            if "detection_class" in meta:
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="detection_class", value=meta["detection_class"]))
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="detection_aggregate", value=meta["detection_aggregate"]))
    channel_tags_df = spark.createDataFrame(channel_tag_rows, core_schema.CHANNEL_TAGS)
    channel_tags_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_channel_tags)


# ── perception_channels (notebook 03 dispatch) ──────────────────────────────

ALLOWED_SENSORS = {"camera", "lidar"}


def perception_paths(loader: A2D2Loader) -> Iterator[dict]:
    """Yield perception_channels rows for every camera + LiDAR file in every scene."""
    for scene in loader.scenes():
        for sd in loader.all_sample_data_in_scene(scene):
            if _sensor_type(sd.sensor_name) not in ALLOWED_SENSORS:
                continue
            yield {
                "container_id": sd.container_id,
                "channel_id": sd.channel_id,
                "timestamp": sd.timestamp_us,
                "file_path": sd.file_path,
                "format": sd.fileformat,
            }
