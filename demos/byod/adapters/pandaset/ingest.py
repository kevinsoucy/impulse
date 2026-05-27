"""PandaSet-specific writes for the foundation tables.

Same shape as the NuScenes / A2D2 / ZOD ingest modules.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timezone

from pyspark.sql import Row

import mda_query_engine.schema as core_schema

from .loader import (
    DERIVED_CHANNEL_IDS,
    GPS_COLUMN_TO_CHANNEL,
    SENSOR_CHANNEL_IDS,
    FRAME_DT_US,
    PandaSetLoader,
)


DERIVED_CHANNEL_META: dict[str, dict[str, str]] = {
    "Vehicle_Speed_kph": {"unit": "kph", "kind": "kinematic", "source": "pandaset_gps"},
    "Heading_deg":       {"unit": "deg", "kind": "kinematic", "source": "pandaset_gps"},
    "Latitude_deg":      {"unit": "deg", "kind": "gnss",      "source": "pandaset_gps"},
    "Longitude_deg":     {"unit": "deg", "kind": "gnss",      "source": "pandaset_gps"},
    "Pedestrian_Count":              {"unit": "count", "kind": "detection_aggregate", "source": "pandaset_annotations",
                                      "detection_class": "pedestrian", "detection_aggregate": "count"},
    "Pedestrian_Nearest_Distance_m": {"unit": "m",     "kind": "detection_aggregate", "source": "pandaset_annotations",
                                      "detection_class": "pedestrian", "detection_aggregate": "nearest_distance"},
    "Vehicle_Count_Front":           {"unit": "count", "kind": "detection_aggregate", "source": "pandaset_annotations",
                                      "detection_class": "car",        "detection_aggregate": "count_front"},
    "Vehicle_Nearest_Distance_m":    {"unit": "m",     "kind": "detection_aggregate", "source": "pandaset_annotations",
                                      "detection_class": "car",        "detection_aggregate": "nearest_distance"},
    "Cyclist_Count":                 {"unit": "count", "kind": "detection_aggregate", "source": "pandaset_annotations",
                                      "detection_class": "cyclist",    "detection_aggregate": "count"},
    "Cyclist_Nearest_Distance_m":    {"unit": "m",     "kind": "detection_aggregate", "source": "pandaset_annotations",
                                      "detection_class": "cyclist",    "detection_aggregate": "nearest_distance"},
}


# LakeVision channel name → comma-joined raw GPS columns that feed it.
_CHANNEL_TO_GPS_COLS: dict[str, str] = {}
for col, name in GPS_COLUMN_TO_CHANNEL.items():
    _CHANNEL_TO_GPS_COLS.setdefault(name, "")
    if _CHANNEL_TO_GPS_COLS[name]:
        _CHANNEL_TO_GPS_COLS[name] += "," + col
    else:
        _CHANNEL_TO_GPS_COLS[name] = col


def _sensor_type(name: str) -> str:
    if name.endswith("_camera"):
        return "camera"
    if name.startswith("lidar"):
        return "lidar"
    return "other"


def ingest_metadata(spark, loader: PandaSetLoader, cfg) -> None:
    scenes = list(loader.scenes())

    # ── container_tags ──
    container_tag_rows: list[Row] = []
    for s in scenes:
        container_tag_rows.extend([
            Row(container_id=s.container_id, key="scene_name", value=s.name),
            Row(container_id=s.container_id, key="scene_description", value=s.description),
            Row(container_id=s.container_id, key="sequence_id", value=s.sequence_id),
            Row(container_id=s.container_id, key="nbr_samples", value=str(s.nbr_samples)),
            Row(container_id=s.container_id, key="dataset_source", value="pandaset"),
            Row(container_id=s.container_id, key="dataset_version", value=cfg.dataset_version),
            Row(container_id=s.container_id, key="dataset_license", value="CC BY 4.0"),
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
    for s in scenes:
        for sensor_name, channel_id in SENSOR_CHANNEL_IDS.items():
            sensor_type = _sensor_type(sensor_name)
            channel_tag_rows.extend([
                Row(container_id=s.container_id, channel_id=channel_id, key="channel_name", value=sensor_name),
                Row(container_id=s.container_id, channel_id=channel_id, key="sensor_type", value=sensor_type),
                Row(container_id=s.container_id, channel_id=channel_id, key="modality", value="binary_file"),
            ])
            # Tag the per-LiDAR channels with their physical sensor model so the
            # BL-009 differentiator is discoverable from channel_tags directly.
            if sensor_name == "lidar_pandar64":
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="lidar_model", value="Pandar64"))
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="lidar_type", value="spinning"))
            elif sensor_name == "lidar_pandargt":
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="lidar_model", value="PandarGT"))
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="lidar_type", value="solid_state"))
        for channel_name, channel_id in DERIVED_CHANNEL_IDS.items():
            meta = DERIVED_CHANNEL_META[channel_name]
            channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="channel_name", value=channel_name))
            channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="unit", value=meta["unit"]))
            channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="kind", value=meta["kind"]))
            channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="source", value=meta["source"]))
            gps_col = _CHANNEL_TO_GPS_COLS.get(channel_name)
            if gps_col:
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="gps_columns", value=gps_col))
            if "detection_class" in meta:
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="detection_class", value=meta["detection_class"]))
                channel_tag_rows.append(Row(container_id=s.container_id, channel_id=channel_id, key="detection_aggregate", value=meta["detection_aggregate"]))
    channel_tags_df = spark.createDataFrame(channel_tag_rows, core_schema.CHANNEL_TAGS)
    channel_tags_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_channel_tags)


# ── perception_channels ─────────────────────────────────────────────────────

ALLOWED_SENSORS = {"camera", "lidar"}


def perception_paths(loader: PandaSetLoader) -> Iterator[dict]:
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
