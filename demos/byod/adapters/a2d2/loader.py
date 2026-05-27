"""Filesystem loader for the A2D2 `camera_lidar_semantic_bboxes` subset.

A2D2 has no official Python SDK. Its public release is a flat directory tree of
JSON (annotations + bus signals + calibration), NPZ (LiDAR point clouds), and
PNG (camera frames). This module wraps that tree behind the same Loader surface
the NuScenes adapter exposes — `scenes()`, `samples_in_scene()`,
`ego_pose_for_sample()`, `annotations_in_sample()`, `camera_calib_for_sample()`,
`all_sample_data_in_scene()` — so the BYOD generic notebooks don't see any
A2D2-specific types.

A2D2 vocab → LakeVision vocab mapping:
  scene directory ({YYYYMMDD_HHMMSS})    → container (one container_id per scene)
  per-camera frame at ~10 Hz             → sample (one keyframe = one Sample)
  per-camera image / lidar / label file  → SampleData (one per sensor file)
  ego pose                               → derived from bus signals (no separate ego_pose stream)
  3D label JSON (label3D/cam_front_center/*.json) → Annotation

A2D2 specifics that shape this module:
  - Annotation coordinates are in the **vehicle (ego) frame** already.
    Unlike NuScenes (global frame), no global→ego transform is needed downstream.
  - There is no cross-frame instance tracking — each frame's box ids are local
    to that frame. `relative_velocity_ms` is therefore `None` for every track.
  - There is no radar — `source` is constantly `"ground_truth_camera_lidar"`.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


# ── Channel ID scheme ───────────────────────────────────────────────────────
# Sensor channel IDs are stable across scenes — same physical sensor, same
# channel_id. Derived bus-signal channels are shared across all A2D2 scenes.

SENSOR_CHANNEL_IDS: dict[str, int] = {
    # Cameras
    "cam_front_center": 101,
    "cam_front_left":   102,
    "cam_front_right":  103,
    "cam_side_left":    104,
    "cam_side_right":   105,
    "cam_rear_center":  106,
    # LiDARs (one NPZ per camera in A2D2's `camera_lidar_*` release)
    "lidar_front_center": 200,
    "lidar_front_left":   201,
    "lidar_front_right":  202,
    "lidar_side_left":    203,
    "lidar_side_right":   204,
    "lidar_rear_center":  205,
}

# Real bus signals + detection-aggregate channels. The first block decodes
# directly from `bus_signals.json`; the second is computed in scalar_source.py.
DERIVED_CHANNEL_IDS: dict[str, int] = {
    # Real bus signals (decoded — no synthesis)
    "Vehicle_Speed_kph":              1001,
    "Vehicle_Accel_Longitudinal_ms2": 1002,
    "Vehicle_Accel_Lateral_ms2":      1003,
    "Vehicle_Accel_Vertical_ms2":     1004,
    "Yaw_Rate_rads":                  1005,
    "Roll_Rate_rads":                 1006,
    "Pitch_Rate_rads":                1007,
    "Steering_Angle_deg":             1008,
    "Brake_Pressure_pct":             1009,
    "Accelerator_Pedal_pct":          1010,
    "Pitch_Angle_deg":                1011,
    "Roll_Angle_deg":                 1012,
    "Latitude_deg":                   1013,
    "Longitude_deg":                  1014,
    # Detection-aggregate channels (per ADR-2, derived from annotations)
    "Pedestrian_Count":               2001,
    "Pedestrian_Nearest_Distance_m":  2002,
    "Vehicle_Count_Front":            2003,
    "Vehicle_Nearest_Distance_m":     2004,
    "Cyclist_Count":                  2005,
    "Cyclist_Nearest_Distance_m":     2006,
}


# Maps the raw A2D2 bus signal key in `bus_signals.json` to one of the
# DERIVED_CHANNEL_IDS above. Signals not in this map are ignored by scalar_source.
BUS_SIGNAL_TO_CHANNEL: dict[str, str] = {
    "vehicle_speed":              "Vehicle_Speed_kph",
    "acceleration_x":             "Vehicle_Accel_Longitudinal_ms2",
    "acceleration_y":             "Vehicle_Accel_Lateral_ms2",
    "acceleration_z":             "Vehicle_Accel_Vertical_ms2",
    "angular_velocity_z":         "Yaw_Rate_rads",
    "angular_velocity_x":         "Roll_Rate_rads",
    "angular_velocity_y":         "Pitch_Rate_rads",
    "steering_angle_calculated":  "Steering_Angle_deg",
    "brake_pressure":             "Brake_Pressure_pct",
    "accelerator_pedal":          "Accelerator_Pedal_pct",
    "pitch_angle":                "Pitch_Angle_deg",
    "roll_angle":                 "Roll_Angle_deg",
    "latitude_degree":            "Latitude_deg",
    "longitude_degree":           "Longitude_deg",
}


# A2D2's published 3D-bbox class names. Reduced to the same LakeVision
# detection_class vocabulary the NuScenes adapter uses (pedestrian, cyclist,
# car, truck, bus, van, animal, other).
def simplify_class(a2d2_class: str) -> str:
    c = (a2d2_class or "").lower()
    if c == "pedestrian":
        return "pedestrian"
    if c in ("bicycle", "motorcycle", "motorbiker"):
        return "cyclist"
    if c == "car":
        return "car"
    if c == "bus":
        return "bus"
    if c in ("truck", "trailer", "caravantransporter", "utilityvehicle", "emergencyvehicle"):
        return "truck"
    if c == "vansuv":
        return "van"
    if c == "animal":
        return "animal"
    return c or "other"


# ── Camera frame rate ────────────────────────────────────────────────────────
# A2D2 cameras log at ~10 Hz. The labeled subset's frame index is monotonic
# inside a scene; we derive a per-sample microsecond timestamp as
#   scene_start_us + frame_index * FRAME_DT_US
# which is the same simplification used in A2D2's reference notebooks.
FRAME_DT_US: int = 100_000  # 10 Hz


# ── Typed records ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scene:
    """One A2D2 scene = one timestamped directory under camera_lidar_semantic_bboxes/."""
    container_id: int
    scene_id: str          # e.g. "20180807_145028"
    name: str              # = scene_id
    description: str       # short summary derived from scene_id
    nbr_samples: int
    scene_dir: str         # absolute filesystem path
    start_ts_us: int       # synthesized from scene_id (date+time → us) for consistent ordering


@dataclass(frozen=True)
class Sample:
    """One A2D2 keyframe = one synchronized set of camera + LiDAR + label files."""
    container_id: int
    sample_token: str      # "{scene_id}#{frame_index:09d}"
    timestamp_us: int
    scene_id: str
    frame_index: int       # 0-based monotonic index inside scene
    sensor_files: dict[str, str]  # sensor_name → absolute file path


@dataclass(frozen=True)
class SampleData:
    """One sensor file at a given sample timestamp."""
    container_id: int
    sample_token: str
    sensor_name: str
    channel_id: int
    timestamp_us: int
    file_path: str
    fileformat: str        # "png", "npz", "json"
    width: int             # 0 for non-image
    height: int            # 0 for non-image


@dataclass(frozen=True)
class EgoPose:
    """Ego pose synthesized from bus signals.

    A2D2 does NOT publish an absolute ego pose stream — there is no
    LiDAR-SLAM-resolved ego pose like NuScenes' `ego_pose`. The adapter does
    not need one because A2D2 annotations are already in the vehicle frame.
    This dataclass exists so the call sites that the BYOD pipeline shares with
    NuScenes (e.g. `ego_pose_for_sample`) keep the same return shape; the
    translation is always (0, 0, 0) and rotation is identity.
    """
    container_id: int
    timestamp_us: int
    translation: tuple[float, float, float]  # always (0, 0, 0) for A2D2
    rotation: tuple[float, float, float, float]  # always identity (1, 0, 0, 0)


@dataclass(frozen=True)
class Annotation:
    """A2D2 3D bounding box in the **vehicle (ego) frame**."""
    container_id: int
    sample_token: str
    timestamp_us: int
    object_id: int
    box_key: str           # "box_0", "box_1", … from the JSON
    category_name: str     # raw A2D2 class name (e.g. "VanSUV")
    detection_class: str   # simplified (e.g. "van")
    translation: tuple[float, float, float]  # vehicle-frame xyz of box center
    size: tuple[float, float, float]         # length, width, height (A2D2 native order)
    rotation: tuple[float, float, float, float]  # quaternion (w, x, y, z) derived from axis+angle
    truncation: float
    occlusion: float


# ── Helpers ─────────────────────────────────────────────────────────────────


def stable_int_id(text: str) -> int:
    """Stable non-negative 63-bit int from any string identifier."""
    import hashlib
    h = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
    return int(h, 16) & 0x7FFFFFFFFFFFFFFF


_SCENE_DIR_RE = re.compile(r"^\d{8}_\d{6}$")


def _scene_start_us(scene_id: str) -> int:
    """Convert "YYYYMMDD_HHMMSS" → POSIX microseconds.

    A2D2 scenes don't carry an absolute capture timestamp in their filenames —
    just the date+time the recording session started. We use that as the
    scene's start time so cross-scene ordering is deterministic.
    """
    from datetime import datetime, timezone
    try:
        dt = datetime.strptime(scene_id, "%Y%m%d_%H%M%S").replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1_000_000)
    except ValueError:
        return stable_int_id(scene_id) & 0xFFFFFFFF  # fallback for non-conforming dirs


def _axis_angle_to_quat(axis: list[float], angle_rad: float) -> tuple[float, float, float, float]:
    """A2D2 stores per-box rotation as `axis` (unit vector) + `angle` (radians).
    Convert to (w, x, y, z) quaternion to match the rest of the pipeline.
    """
    import math
    ax, ay, az = axis
    norm = math.sqrt(ax * ax + ay * ay + az * az)
    if norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    ax, ay, az = ax / norm, ay / norm, az / norm
    half = angle_rad / 2.0
    s = math.sin(half)
    return (math.cos(half), ax * s, ay * s, az * s)


# ── Loader ──────────────────────────────────────────────────────────────────


class A2D2Loader:
    """File-tree loader for A2D2 `camera_lidar_semantic_bboxes`.

    Lazy and memory-light: scene/sample enumeration walks the filesystem
    on demand; no devkit equivalent to pre-load.
    """

    # The 6 cameras in A2D2's published release.
    CAMERAS: tuple[str, ...] = (
        "cam_front_center",
        "cam_front_left",
        "cam_front_right",
        "cam_side_left",
        "cam_side_right",
        "cam_rear_center",
    )

    # The LiDAR streams are co-named after the camera they're co-aligned with
    # in the published `camera_lidar_*` release (one NPZ per camera per frame).
    LIDARS: tuple[str, ...] = (
        "lidar_front_center",
        "lidar_front_left",
        "lidar_front_right",
        "lidar_side_left",
        "lidar_side_right",
        "lidar_rear_center",
    )

    # The 3D box label files live under label3D/cam_front_center/ only — all
    # boxes are already in the vehicle frame and the labeling tool emits the
    # canonical copy under the front-center camera's directory.
    LABEL_REF_CAMERA = "cam_front_center"

    def __init__(self, dataroot: str, dataset_version: str = "camera_lidar_semantic_bboxes") -> None:
        self._dataroot = dataroot
        self._dataset_version = dataset_version
        self._subset_root = Path(dataroot) / dataset_version

    # ── Public surfaces ─────────────────────────────────────────────────────

    @property
    def dataroot(self) -> str:
        return self._dataroot

    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    # ── Calibration (read once, memoized) ───────────────────────────────────

    @lru_cache(maxsize=1)
    def _cams_lidars(self) -> dict:
        path = Path(self._dataroot) / "cams_lidars.json"
        if not path.is_file():
            return {}
        with open(path) as f:
            return json.load(f)

    def camera_calib_for_sample(self, sample: Sample) -> dict[str, dict]:
        """Per-camera calibration dict shaped for `mda_query_engine.perception`-side projection.

        Returns the same keys as the NuScenes adapter's `camera_calib_for_sample`:
          intrinsic, sensor_rotation, sensor_translation, ego_rotation,
          ego_translation, width, height, sensor_id.

        A2D2 expresses each camera as a "view" with `origin`, `x-axis`, `y-axis`
        in the vehicle frame; the third axis is derived from the cross product.
        We translate that into a (translation, quaternion) pair so the existing
        `camera.py` projection logic works unchanged. `ego_*` is always identity
        because annotations are already in the vehicle frame.
        """
        calib = self._cams_lidars()
        cameras = calib.get("cameras", {})
        out: dict[str, dict] = {}
        for cam in self.CAMERAS:
            entry = cameras.get(cam)
            if not entry:
                continue
            view = entry.get("view", {})
            origin = view.get("origin", [0.0, 0.0, 0.0])
            x_axis = view.get("x-axis", [1.0, 0.0, 0.0])
            y_axis = view.get("y-axis", [0.0, 1.0, 0.0])
            intrinsic = entry.get("CamMatrix", entry.get("matrix", _DEFAULT_INTRINSIC))
            resolution = entry.get("Resolution", [1920, 1208])
            quat_wxyz = _axes_to_quat(x_axis, y_axis)
            out[cam] = {
                "intrinsic": intrinsic,
                "sensor_rotation": list(quat_wxyz),
                "sensor_translation": origin,
                "ego_rotation": [1.0, 0.0, 0.0, 0.0],
                "ego_translation": [0.0, 0.0, 0.0],
                "width": int(resolution[0]),
                "height": int(resolution[1]),
                "sensor_id": cam,
            }
        return out

    # ── Scenes ──────────────────────────────────────────────────────────────

    def scenes(self) -> Iterator[Scene]:
        if not self._subset_root.is_dir():
            return
        for entry in sorted(self._subset_root.iterdir()):
            if not entry.is_dir() or not _SCENE_DIR_RE.match(entry.name):
                continue
            cam_dir = entry / "camera" / self.LABEL_REF_CAMERA
            nbr_samples = sum(1 for _ in cam_dir.glob("*.png")) if cam_dir.is_dir() else 0
            yield Scene(
                container_id=stable_int_id(entry.name),
                scene_id=entry.name,
                name=entry.name,
                description=f"A2D2 scene {entry.name}",
                nbr_samples=nbr_samples,
                scene_dir=str(entry),
                start_ts_us=_scene_start_us(entry.name),
            )

    def list_scenes(self) -> list[Scene]:
        return list(self.scenes())

    # ── Samples (per-frame) ─────────────────────────────────────────────────

    def samples_in_scene(self, scene: Scene) -> Iterator[Sample]:
        """Yield Samples in monotonic frame-index order.

        The frame index is parsed from the canonical front-center PNG filename
        (`*_camera_frontcenter_NNNNNNNNN.png`). For each frame, we resolve the
        sibling camera, lidar, and label paths.
        """
        cam_dir = Path(scene.scene_dir) / "camera" / self.LABEL_REF_CAMERA
        if not cam_dir.is_dir():
            return
        for png in sorted(cam_dir.glob("*.png")):
            frame_index = _frame_index_from_filename(png.name)
            if frame_index is None:
                continue
            sensor_files = {}
            # Cameras
            for cam in self.CAMERAS:
                cam_path = (
                    Path(scene.scene_dir) / "camera" / cam
                    / _camera_filename(scene.scene_id, cam, frame_index)
                )
                if cam_path.is_file():
                    sensor_files[cam] = str(cam_path)
            # LiDARs
            for lidar, cam in zip(self.LIDARS, self.CAMERAS):
                lidar_path = (
                    Path(scene.scene_dir) / "lidar" / cam
                    / _lidar_filename(scene.scene_id, cam, frame_index)
                )
                if lidar_path.is_file():
                    sensor_files[lidar] = str(lidar_path)
            yield Sample(
                container_id=scene.container_id,
                sample_token=f"{scene.scene_id}#{frame_index:09d}",
                timestamp_us=scene.start_ts_us + frame_index * FRAME_DT_US,
                scene_id=scene.scene_id,
                frame_index=frame_index,
                sensor_files=sensor_files,
            )

    # ── Sample data (for perception_channels) ───────────────────────────────

    def all_sample_data_in_scene(self, scene: Scene) -> Iterator[SampleData]:
        """Yield one SampleData per sensor file in the scene. Used by
        `perception_paths()` to populate the `perception_channels` table."""
        for sample in self.samples_in_scene(scene):
            for sensor_name, file_path in sample.sensor_files.items():
                fileformat = Path(file_path).suffix.lstrip(".").lower()
                yield SampleData(
                    container_id=sample.container_id,
                    sample_token=sample.sample_token,
                    sensor_name=sensor_name,
                    channel_id=SENSOR_CHANNEL_IDS.get(sensor_name, 0),
                    timestamp_us=sample.timestamp_us,
                    file_path=file_path,
                    fileformat=fileformat,
                    width=0,
                    height=0,
                )

    # ── Ego pose (always identity in A2D2) ──────────────────────────────────

    def ego_pose_for_sample(self, sample: Sample) -> EgoPose:
        """A2D2 annotations are already in the vehicle frame. Returning an
        identity pose here keeps the call shape identical to NuScenes so the
        BYOD pipeline can share helpers (`global_to_ego` becomes a no-op when
        ego is identity)."""
        return EgoPose(
            container_id=sample.container_id,
            timestamp_us=sample.timestamp_us,
            translation=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
        )

    # ── Annotations ─────────────────────────────────────────────────────────

    def annotations_in_sample(self, sample: Sample) -> Iterator[Annotation]:
        label_path = (
            Path(self._subset_root) / sample.scene_id / "label3D" / self.LABEL_REF_CAMERA
            / _label_filename(sample.scene_id, self.LABEL_REF_CAMERA, sample.frame_index)
        )
        if not label_path.is_file():
            return
        try:
            with open(label_path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        # A2D2's 3D-box label JSON is a flat dict of {"box_0": {...}, "box_1": {...}}
        # with no nested wrapper. Tolerate either flat or "boxes"-wrapped shapes.
        boxes = payload.get("boxes", payload) if isinstance(payload, dict) else {}
        for box_key, box in boxes.items():
            if not isinstance(box, dict):
                continue
            center = box.get("center", [0.0, 0.0, 0.0])
            size = box.get("size", [0.0, 0.0, 0.0])
            axis = box.get("axis", [0.0, 0.0, 1.0])
            angle = float(box.get("angle", 0.0))
            klass = str(box.get("class", "Other"))
            yield Annotation(
                container_id=sample.container_id,
                sample_token=sample.sample_token,
                timestamp_us=sample.timestamp_us,
                object_id=stable_int_id(f"{sample.sample_token}/{box_key}"),
                box_key=box_key,
                category_name=klass,
                detection_class=simplify_class(klass),
                translation=tuple(float(c) for c in center),
                size=tuple(float(s) for s in size),
                rotation=_axis_angle_to_quat(axis, angle),
                truncation=float(box.get("truncation", 0.0)),
                occlusion=float(box.get("occlusion", 0.0)),
            )

    # ── Bus signals (per-scene file) ────────────────────────────────────────

    def bus_signals_for_scene(self, scene: Scene) -> dict[str, list[tuple[int, float]]]:
        """Decode `bus/bus_signals.json` into {a2d2_signal_key: [(ts_us, value)]}.

        Returns the raw A2D2 keys (e.g. `vehicle_speed`, `acceleration_x`).
        Caller maps them to LakeVision channel names via `BUS_SIGNAL_TO_CHANNEL`.
        Missing files or signals → empty result; not an error.
        """
        bus_path = Path(scene.scene_dir) / "bus" / "bus_signals.json"
        if not bus_path.is_file():
            return {}
        try:
            with open(bus_path) as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

        result: dict[str, list[tuple[int, float]]] = {}
        for sig_key, sig in raw.items():
            if not isinstance(sig, dict):
                continue
            values = sig.get("values", [])
            series: list[tuple[int, float]] = []
            for entry in values:
                ts = entry.get("timestamp")
                v = entry.get("value")
                if ts is None or v is None:
                    continue
                try:
                    series.append((int(ts), float(v)))
                except (TypeError, ValueError):
                    continue
            if series:
                result[sig_key] = series
        return result

    # ── Counts ──────────────────────────────────────────────────────────────

    def count_summary(self) -> dict[str, int]:
        scenes = self.list_scenes()
        return {
            "scenes": len(scenes),
            "samples": sum(s.nbr_samples for s in scenes),
        }


# ── Filename helpers (A2D2's published convention) ───────────────────────────

# Camera frame:  {date}_camera_{cam_short}_{frame_index:09d}.png
# LiDAR scan:    {date}_lidar_{cam_short}_{frame_index:09d}.npz
# 3D label:      {date}_label3D_{cam_short}_{frame_index:09d}.json
# where `{cam_short}` = camera name with underscores stripped after "cam_"
# (e.g. "cam_front_center" → "frontcenter").

_CAM_SHORT = {
    "cam_front_center": "frontcenter",
    "cam_front_left":   "frontleft",
    "cam_front_right":  "frontright",
    "cam_side_left":    "sideleft",
    "cam_side_right":   "sideright",
    "cam_rear_center":  "rearcenter",
}


def _camera_filename(scene_id: str, camera: str, frame_index: int) -> str:
    date_part = scene_id.replace("_", "")
    return f"{date_part}_camera_{_CAM_SHORT[camera]}_{frame_index:09d}.png"


def _lidar_filename(scene_id: str, camera: str, frame_index: int) -> str:
    date_part = scene_id.replace("_", "")
    return f"{date_part}_lidar_{_CAM_SHORT[camera]}_{frame_index:09d}.npz"


def _label_filename(scene_id: str, camera: str, frame_index: int) -> str:
    date_part = scene_id.replace("_", "")
    return f"{date_part}_label3D_{_CAM_SHORT[camera]}_{frame_index:09d}.json"


_FRAME_INDEX_RE = re.compile(r"_(\d{9})\.")


def _frame_index_from_filename(name: str) -> int | None:
    m = _FRAME_INDEX_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# ── Camera-axes → quaternion ────────────────────────────────────────────────

_DEFAULT_INTRINSIC = [
    [1687.3, 0.0, 965.4],
    [0.0,    1687.3, 569.4],
    [0.0,    0.0,    1.0],
]


def _axes_to_quat(x_axis: list[float], y_axis: list[float]) -> tuple[float, float, float, float]:
    """Build a (w, x, y, z) quaternion from A2D2's `x-axis` and `y-axis` view vectors.

    A2D2 specifies each camera's pose by an `origin` + an `x-axis` and `y-axis`
    expressed in the vehicle frame. The rotation matrix whose columns are
    [x_hat, y_hat, z_hat=x×y] takes a point from the camera's local frame to
    the vehicle frame. We invert the standard quaternion-from-matrix formula
    to recover the unit quaternion.
    """
    import math

    import numpy as np

    x = np.asarray(x_axis, dtype=float)
    y = np.asarray(y_axis, dtype=float)
    x_norm = np.linalg.norm(x)
    y_norm = np.linalg.norm(y)
    if x_norm == 0.0 or y_norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    x = x / x_norm
    y = y / y_norm
    z = np.cross(x, y)
    z_norm = np.linalg.norm(z)
    if z_norm == 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    z = z / z_norm
    # Re-orthogonalize y in case the published axes weren't perfectly perpendicular.
    y = np.cross(z, x)

    R = np.column_stack([x, y, z])  # 3×3 rotation: cam → vehicle
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    else:
        # Pick the largest diagonal element for numerical stability.
        if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    return (float(qw), float(qx), float(qy), float(qz))
