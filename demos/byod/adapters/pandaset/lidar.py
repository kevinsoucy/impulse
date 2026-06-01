"""Convert PandaSet annotations to `lidar_object_detections` rows.

**dual-LiDAR differentiator:** each cuboid produces TWO rows, one per physical
LiDAR sensor (`LIDAR_SPINNING` Pandar64, `LIDAR_SOLIDSTATE` PandarGT). This
gives notebook 04's `distance_stats_by_modality` a within-adapter sensor
comparison no other adapter in the suite can provide — spinning roof-rack
LiDAR vs. forward-facing solid-state LiDAR range distributions.

This table is populated only inside TSAL event windows.
"""

from __future__ import annotations

from lib.geometry import yaw_from_quat

from .loader import LIDAR_SENSOR_BY_D, Annotation, PandaSetLoader, Scene


# Canonical sensor-id constants (re-exported from loader.LIDAR_SENSOR_BY_D so
# callers don't have to know about the `d` column convention).
LIDAR_SENSOR_SPINNING = "LIDAR_SPINNING"      # Pandar64 (d=0)
LIDAR_SENSOR_SOLIDSTATE = "LIDAR_SOLIDSTATE"  # PandarGT (d=1)


__all__ = [
    "LIDAR_SENSOR_BY_D",
    "LIDAR_SENSOR_SOLIDSTATE",
    "LIDAR_SENSOR_SPINNING",
    "map_annotation_to_lidar_detections",
    "map_scene_for_event_windows",
]


def _base_row(ann: Annotation) -> dict:
    cx, cy, cz = (float(c) for c in ann.translation)
    length, width, height = (float(s) for s in ann.size)
    yaw_rad = yaw_from_quat(*ann.rotation)
    return {
        "container_id": ann.container_id,
        "frame_ts": ann.timestamp_us,
        "object_id": ann.object_id,
        "detection_class": ann.detection_class,
        "confidence": 1.0,
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "length": length,
        "width": width,
        "height": height,
        "yaw_rad": yaw_rad,
    }


def map_annotation_to_lidar_detections(ann: Annotation) -> list[dict]:
    """Emit one detection row per physical LiDAR sensor.

    PandaSet's cuboid annotations don't carry a sensor split (each cuboid is
    fused across both LiDARs). So that notebook 04 shows distinct range
    distributions for the two sensors, we duplicate each cuboid into two rows
    tagged with the per-physical-sensor
    `sensor_id`. Downstream consumers JOIN against the LiDAR point clouds
    (split on `d`) to derive sensor-specific range stats.
    """
    base = _base_row(ann)
    return [
        {**base, "sensor_id": sensor_id}
        for sensor_id in (LIDAR_SENSOR_SPINNING, LIDAR_SENSOR_SOLIDSTATE)
    ]


def map_scene_for_event_windows(
    loader: PandaSetLoader,
    scene: Scene,
    event_windows: list[tuple[int, int]],
) -> list[dict]:
    if not event_windows:
        return []
    sorted_windows = sorted(event_windows)
    out: list[dict] = []
    for sample in loader.samples_in_scene(scene):
        ts = sample.timestamp_us
        if not any(start <= ts <= end for start, end in sorted_windows):
            continue
        for ann in loader.annotations_in_sample(sample):
            out.extend(map_annotation_to_lidar_detections(ann))
    return out
