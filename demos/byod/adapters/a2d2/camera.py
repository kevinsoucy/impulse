"""Convert A2D2 annotations to `camera_object_detections` rows.

A2D2 ground-truth 3D boxes are in the vehicle (ego) frame. Each of A2D2's six
cameras has a `view` definition in `cams_lidars.json` that lets us project the
box corners into image space using the same numpy-only projection chain the
NuScenes adapter uses — the only difference is that the ego→sensor step is
identity-on-ego (boxes are in the vehicle frame already).

Only frames inside TSAL event windows are projected.
"""

from __future__ import annotations

import numpy as np

from lib.geometry import rotation_matrix_from_quat as _rotation_matrix_from_quat

from .loader import A2D2Loader, Annotation, Scene


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
    """Transform a 3D point in the vehicle frame to a camera's sensor frame.

    `sensor_rotation_wxyz` is the cam→vehicle rotation built in the loader.
    To go vehicle→cam we apply its transpose.
    """
    R_sensor = _rotation_matrix_from_quat(*sensor_rotation_wxyz)
    return R_sensor.T @ (point_vehicle - np.array(sensor_translation, dtype=float))


def _box_corners_vehicle(
    translation: tuple,
    size_lwh: tuple,
    rotation_wxyz: tuple,
) -> np.ndarray:
    """Return (8, 3) array of 3D box corner positions in the vehicle frame.

    A2D2 size convention is (length, width, height) and the box's local axes
    are x=forward (length), y=left (width), z=up (height) — the canonical
    LakeVision convention. No reordering vs. NuScenes' (width, length, height).
    """
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


def _project_corners_to_image(
    corners_sensor: np.ndarray,
    intrinsic,
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    """Project sensor-frame corners to a 2D bounding box.

    A2D2 camera local convention: z forward (optical axis), x right, y down —
    same convention as the canonical pinhole model used by the NuScenes
    adapter. Returns clamped (x1, y1, x2, y2) or None for boxes that fall
    entirely outside or behind the camera.
    """
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
    """Project one A2D2 box into a camera. Returns a row dict or None when
    the box is fully outside or behind the camera."""
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
    loader: A2D2Loader,
    scene: Scene,
    event_windows: list[tuple[int, int]],
) -> list[dict]:
    """Emit camera_object_detections rows for every annotation inside an
    event window, projected into all six A2D2 cameras at that keyframe."""
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
