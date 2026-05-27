"""Convert A2D2 annotations to `lidar_object_detections` rows.

A2D2 ground-truth boxes are already in the vehicle (ego) frame, so the
NuScenes-style global→ego→box-yaw transform is unnecessary. The yaw is just
the angle of the box's local x-axis after rotation by the stored quaternion.

Per ADR-7, this table is populated only for frames inside TSAL event windows.
"""

from __future__ import annotations

import numpy as np

from lakevision.geometry import yaw_from_quat

from .loader import A2D2Loader, Annotation, Scene


# All A2D2 box annotations are referenced against the vehicle frame; we tag
# the synthetic "sensor" as the fused LiDAR cluster rather than picking one
# of the five physical LiDARs.
LIDAR_SENSOR_ID = "LIDAR_FUSED"


__all__ = [
    "LIDAR_SENSOR_ID",
    "map_annotation_to_lidar_detection",
    "map_scene_for_event_windows",
]


def map_annotation_to_lidar_detection(ann: Annotation) -> dict:
    """One A2D2 annotation → one lidar_object_detections row.

    No ego pose argument because A2D2 boxes are already in the vehicle frame.
    """
    cx, cy, cz = (float(c) for c in ann.translation)

    # A2D2 size convention is (length, width, height) — same as the LakeVision
    # schema, so no reordering is needed (unlike NuScenes' (w, l, h)).
    length, width, height = (float(s) for s in ann.size)

    # `rotation` is a (w, x, y, z) quaternion derived from A2D2's (axis, angle)
    # representation in the loader. Yaw is the rotation around the vehicle's
    # vertical (z) axis.
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
    loader: A2D2Loader,
    scene: Scene,
    event_windows: list[tuple[int, int]],
) -> list[dict]:
    """Emit rows for every annotation whose sample timestamp falls inside any
    of the (start_us, end_us) windows."""
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
