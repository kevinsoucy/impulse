"""Convert ZOD annotations to `lidar_object_detections` rows.

Same shape as the A2D2 adapter — ZOD boxes are also in the vehicle frame,
so no global→ego transform. The single 3-LiDAR fused stream is encoded as
`sensor_id = "LIDAR_FUSED"`.

This table is populated only inside TSAL event windows.
"""

from __future__ import annotations

from lib.geometry import yaw_from_quat

from .loader import Annotation, Scene, ZodLoader


LIDAR_SENSOR_ID = "LIDAR_FUSED"


__all__ = [
    "LIDAR_SENSOR_ID",
    "map_annotation_to_lidar_detection",
    "map_scene_for_event_windows",
]


def map_annotation_to_lidar_detection(ann: Annotation) -> dict:
    cx, cy, cz = (float(c) for c in ann.translation)
    length, width, height = (float(s) for s in ann.size)
    yaw_rad = yaw_from_quat(*ann.rotation)
    return {
        "container_id": ann.container_id,
        "frame_ts": ann.timestamp_us,
        "object_id": ann.object_id,
        "detection_class": ann.detection_class,
        "confidence": 1.0,
        "sensor_id": LIDAR_SENSOR_ID,
        "cx": cx,
        "cy": cy,
        "cz": cz,
        "length": length,
        "width": width,
        "height": height,
        "yaw_rad": yaw_rad,
    }


def map_scene_for_event_windows(
    loader: ZodLoader,
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
            out.append(map_annotation_to_lidar_detection(ann))
    return out
