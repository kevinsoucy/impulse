"""Convert NuScenes 3D annotations to `lidar_object_detections` rows.

NuScenes annotations carry full 3D geometry in the global frame. We:
1. Transform translation into ego frame (x=forward, y=left, z=up — vehicle frame)
2. Unpack size as (length, width, height) from NuScenes' (width, length, height) convention
3. Extract yaw in ego frame from the quaternion rotation

This table is populated **only for frames inside TSAL event windows** —
the OpenLABEL export (notebook 03) is the scoped downstream consumer.
"""

from __future__ import annotations

import numpy as np

from lib.geometry import (
    ego_yaw_from_global_yaw as _ego_yaw_from_global_yaw,
    yaw_from_quat,
)

from .loader import Annotation, NuScenesLoader, Scene
from .scalar_source import global_to_ego


LIDAR_SENSOR_ID = "LIDAR_TOP"  # NuScenes reference frame for 3D annotations


# Re-exported for the migrated test modules.
__all__ = [
    "LIDAR_SENSOR_ID",
    "_ego_yaw_from_global_yaw",
    "map_annotation_to_lidar_detection",
    "map_scene_for_event_windows",
]


def map_annotation_to_lidar_detection(ann: Annotation, ego) -> dict:
    """One NuScenes annotation → one lidar_object_detections row dict."""
    ann_global = np.array(ann.translation)
    ego_frame = global_to_ego(ann_global, ego)
    cx, cy, cz = float(ego_frame[0]), float(ego_frame[1]), float(ego_frame[2])

    # NuScenes size convention: [width, length, height]
    nu_w, nu_l, nu_h = ann.size
    length, width, height = float(nu_l), float(nu_w), float(nu_h)

    global_yaw = yaw_from_quat(*ann.rotation)
    yaw_rad = _ego_yaw_from_global_yaw(global_yaw, ego.rotation)

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
    loader: NuScenesLoader,
    scene: Scene,
    event_windows: list[tuple[int, int]],
) -> list[dict]:
    """For each (start_us, end_us) window, emit rows for every annotation whose
    sample timestamp falls inside it. Idempotent — duplicate windows are deduped on write."""
    if not event_windows:
        return []
    sorted_windows = sorted(event_windows)
    out: list[dict] = []
    samples = list(loader.samples_in_scene(scene))
    for sample in samples:
        ts = sample.timestamp_us
        if not any(start <= ts <= end for start, end in sorted_windows):
            continue
        ego = loader.ego_pose_for_sample(sample)
        for ann in loader.annotations_in_sample(sample):
            out.append(map_annotation_to_lidar_detection(ann, ego))
    return out
