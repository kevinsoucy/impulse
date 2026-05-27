"""Convert PandaSet annotations to `camera_object_detections` rows.

Identical projection chain to A2D2 and ZOD — vehicle-frame box → camera
sensor frame → image plane via intrinsic K. PandaSet ships 6 cameras with
per-frame poses (so we look up the entry matching the sample's frame index).

Per ADR-7, only frames inside TSAL event windows are projected.
"""

from __future__ import annotations

import numpy as np

from lakevision.geometry import rotation_matrix_from_quat as _rotation_matrix_from_quat

from .loader import Annotation, PandaSetLoader, Scene


__all__ = [
    "_box_corners_vehicle",
    "_project_corners_to_image",
    "_vehicle_to_sensor",
    "map_scene_for_event_windows",
    "project_annotation_to_camera",
]


def _vehicle_to_sensor(
    point_vehicle: np.ndarray,
    sensor_translation,
    sensor_rotation_wxyz,
) -> np.ndarray:
    R_sensor = _rotation_matrix_from_quat(*sensor_rotation_wxyz)
    return R_sensor.T @ (point_vehicle - np.array(sensor_translation, dtype=float))


def _box_corners_vehicle(translation, size_lwh, rotation_wxyz) -> np.ndarray:
    l, w, h = size_lwh
    dx, dy, dz = l / 2, w / 2, h / 2
    corners_local = np.array([
        [ dx,  dy,  dz], [ dx,  dy, -dz],
        [ dx, -dy,  dz], [ dx, -dy, -dz],
        [-dx,  dy,  dz], [-dx,  dy, -dz],
        [-dx, -dy,  dz], [-dx, -dy, -dz],
    ], dtype=np.float64)
    R = _rotation_matrix_from_quat(*rotation_wxyz)
    return (R @ corners_local.T).T + np.array(translation, dtype=np.float64)


def _project_corners_to_image(corners_sensor, intrinsic, width, height):
    in_front = corners_sensor[:, 2] > 0
    if not in_front.any():
        return None
    K = np.asarray(intrinsic, dtype=np.float64)
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


def project_annotation_to_camera(ann: Annotation, calib_info: dict) -> dict | None:
    corners_vehicle = _box_corners_vehicle(ann.translation, ann.size, ann.rotation)
    corners_sensor = np.array([
        _vehicle_to_sensor(c, calib_info["sensor_translation"], calib_info["sensor_rotation"])
        for c in corners_vehicle
    ])
    bbox = _project_corners_to_image(
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
        cam_calibs = loader.camera_calib_for_sample(sample)
        for ann in loader.annotations_in_sample(sample):
            for calib in cam_calibs.values():
                row = project_annotation_to_camera(ann, calib)
                if row is not None:
                    out.append(row)
    return out
