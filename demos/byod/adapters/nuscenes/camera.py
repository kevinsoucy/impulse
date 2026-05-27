"""Convert NuScenes 3D annotations to `camera_object_detections` rows.

Each annotation is projected into every camera sensor present at the keyframe.
Projection uses the calibrated sensor extrinsics and camera intrinsics stored
in NuScenes' `calibrated_sensor` and `ego_pose` records.

Per ADR-7, this table is populated only for frames inside TSAL event windows.

Geometry notes:
- NuScenes annotations are in the global frame (x=east, y=north, z=up).
- Each camera has an ego_pose (ego→global) and a calibrated_sensor (sensor→ego).
- Transform chain: global → ego frame → sensor frame → project with intrinsics.
- Only corners with sensor-frame z > 0 (in front of camera) contribute to the bbox.
- NuScenes size convention: (width, length, height).
"""

from __future__ import annotations

import numpy as np

from mda_query_engine.perception.geometry import rotation_matrix_from_quat as _rotation_matrix_from_quat

from .loader import Annotation, NuScenesLoader, Scene


__all__ = [
    "_box_corners_global",
    "_global_to_sensor",
    "_project_to_2d",
    "map_scene_for_event_windows",
    "project_annotation_to_camera",
]


def _global_to_sensor(
    point_global: np.ndarray,
    ego_translation: list,
    ego_rotation_wxyz: list,
    sensor_translation: list,
    sensor_rotation_wxyz: list,
) -> np.ndarray:
    """Transform a 3D point from global frame → ego frame → sensor frame."""
    R_ego = _rotation_matrix_from_quat(*ego_rotation_wxyz)
    R_sensor = _rotation_matrix_from_quat(*sensor_rotation_wxyz)
    p_ego = R_ego.T @ (point_global - np.array(ego_translation))
    return R_sensor.T @ (p_ego - np.array(sensor_translation))


def _box_corners_global(
    translation: tuple,
    size_wlh: tuple,
    rotation_wxyz: tuple,
) -> np.ndarray:
    """Return (8, 3) array of 3D box corner positions in the global frame.

    NuScenes size convention is (width, length, height); box axes in local frame:
    x=forward(length), y=left(width), z=up(height).
    """
    w, l, h = size_wlh
    dx, dy, dz = l / 2, w / 2, h / 2
    corners_local = np.array([
        [ dx,  dy,  dz], [ dx,  dy, -dz],
        [ dx, -dy,  dz], [ dx, -dy, -dz],
        [-dx,  dy,  dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [-dx, -dy, -dz],
    ], dtype=np.float64)
    R = _rotation_matrix_from_quat(*rotation_wxyz)
    return (R @ corners_local.T).T + np.array(translation, dtype=np.float64)


def _project_to_2d(
    corners_sensor: np.ndarray,
    intrinsic: list,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Project sensor-frame corners to a 2D bounding box.

    Returns (x1, y1, x2, y2) clamped to image bounds, or None if all corners
    are behind the camera or the projected bbox has zero area.
    """
    in_front = corners_sensor[:, 2] > 0
    if not in_front.any():
        return None
    K = np.array(intrinsic, dtype=np.float64)
    valid = corners_sensor[in_front]
    coords = (K @ valid.T).T
    pts = coords[:, :2] / coords[:, 2:3]
    x1 = max(0, int(np.floor(pts[:, 0].min())))
    y1 = max(0, int(np.floor(pts[:, 1].min())))
    x2 = min(width - 1, int(np.ceil(pts[:, 0].max())))
    y2 = min(height - 1, int(np.ceil(pts[:, 1].max())))
    if x1 >= x2 or y1 >= y2:
        return None
    return x1, y1, x2, y2


def project_annotation_to_camera(
    ann: Annotation,
    calib_info: dict,
) -> dict | None:
    """Project one NuScenes annotation into a camera's image plane.

    Returns a camera_object_detections row dict, or None when the annotation
    falls entirely outside or behind the camera.
    """
    corners_global = _box_corners_global(ann.translation, ann.size, ann.rotation)
    corners_sensor = np.array([
        _global_to_sensor(
            c,
            calib_info["ego_translation"],
            calib_info["ego_rotation"],
            calib_info["sensor_translation"],
            calib_info["sensor_rotation"],
        )
        for c in corners_global
    ])
    bbox = _project_to_2d(
        corners_sensor,
        calib_info["intrinsic"],
        calib_info["width"],
        calib_info["height"],
    )
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return {
        "container_id": ann.container_id,
        "frame_ts": ann.timestamp_us,
        "object_id": ann.object_id,
        "detection_class": ann.detection_class,
        "confidence": 1.0,
        "sensor_id": calib_info["sensor_id"],
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
    }


def map_scene_for_event_windows(
    loader: NuScenesLoader,
    scene: Scene,
    event_windows: list[tuple[int, int]],
) -> list[dict]:
    """Emit camera_object_detections rows for every annotation that falls inside
    a TSAL event window, projected into all camera sensors at that keyframe.
    """
    if not event_windows:
        return []
    sorted_windows = sorted(event_windows)
    out: list[dict] = []
    for sample in loader.samples_in_scene(scene):
        ts = sample.timestamp_us
        if not any(start <= ts <= end for start, end in sorted_windows):
            continue
        cam_calibs = loader.camera_calib_for_sample(sample)
        for ann in loader.annotations_in_sample(sample):
            for calib in cam_calibs.values():
                row = project_annotation_to_camera(ann, calib)
                if row is not None:
                    out.append(row)
    return out
