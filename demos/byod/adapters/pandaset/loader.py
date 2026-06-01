"""Filesystem loader for PandaSet (Hesai / Scale AI).

PandaSet's `pandaset` SDK on PyPI wraps the same on-disk layout this adapter
reads directly. We mirror the A2D2 / ZOD pattern — pure-filesystem reads — so
unit tests can build synthetic pandas+gzip+CSV fixtures without depending on
the SDK or a live download.

PandaSet vocab → LakeVision vocab mapping:
  sequence directory ({3-digit-id})    → container
  per-frame LiDAR/camera/annotation    → sample (~10 Hz keyframe)
  per-sensor file (jpg, pkl.gz)        → SampleData
  ego pose                             → derived from `meta/gps.csv` heading/lat/lon
  cuboid annotation                    → Annotation (one per object per frame)

PandaSet specifics that shape this module:
  - Annotations are in the **vehicle (ego) frame** already (`position.x/y/z`
    are documented as ego-frame in the pandaset-devkit). Same as A2D2 / ZOD —
    no global→ego transform.
  - Cross-frame instance tracking IS preserved via the cuboid `uuid` column,
    so `relative_velocity_ms` is computable.
  - No radar sensor → `source = "lidar|camera"` (no radar modality).
  - **Dual heterogeneous LiDAR.** Each cuboid is duplicated into two
    `lidar_object_detections` rows — one per physical sensor (`LIDAR_SPINNING`
    Pandar64, `LIDAR_SOLIDSTATE` PandarGT). This is the dual-LiDAR differentiator.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


# ── Channel ID scheme ───────────────────────────────────────────────────────

SENSOR_CHANNEL_IDS: dict[str, int] = {
    # PandaSet ships 6 cameras
    "front_camera":         101,
    "front_left_camera":    102,
    "front_right_camera":   103,
    "left_camera":          104,
    "right_camera":         105,
    "back_camera":          106,
    # Two physical LiDARs co-recorded; the adapter splits the fused
    # `.pkl.gz` on the `d` column to populate per-sensor rows.
    "lidar_pandar64":       200,
    "lidar_pandargt":       201,
}

DERIVED_CHANNEL_IDS: dict[str, int] = {
    # Real GPS/IMU-derived scalars from meta/gps.csv (no accelerometer/gyro
    # in the PandaSet public release — fewer kinematic channels than ZOD/A2D2)
    "Vehicle_Speed_kph":              1001,
    "Heading_deg":                    1002,
    "Latitude_deg":                   1003,
    "Longitude_deg":                  1004,
    # Detection-aggregate channels (derived from annotations)
    "Pedestrian_Count":               2001,
    "Pedestrian_Nearest_Distance_m":  2002,
    "Vehicle_Count_Front":            2003,
    "Vehicle_Nearest_Distance_m":     2004,
    "Cyclist_Count":                  2005,
    "Cyclist_Nearest_Distance_m":     2006,
}


# Sensor-id values for the two physical LiDARs in a fused PandaSet pickle.
# The `d` column in PandaSet's LiDAR DataFrame is 0=Pandar64, 1=PandarGT.
LIDAR_SENSOR_BY_D: dict[int, str] = {
    0: "LIDAR_SPINNING",    # Pandar64, 360° roof-rack
    1: "LIDAR_SOLIDSTATE",  # PandarGT, forward-facing windshield-mounted
}


# meta/gps.csv column → LakeVision channel name. Columns kept defensive
# against subtle field-name variants in PandaSet's published CSVs.
GPS_COLUMN_TO_CHANNEL: dict[str, str] = {
    "speed":             "Vehicle_Speed_kph",     # m/s in CSV; scaled below
    "velocity":          "Vehicle_Speed_kph",
    "heading":           "Heading_deg",
    "altitude_heading":  "Heading_deg",
    "lat":               "Latitude_deg",
    "latitude":          "Latitude_deg",
    "lon":               "Longitude_deg",
    "lng":               "Longitude_deg",
    "longitude":         "Longitude_deg",
}


# m/s → kph for the speed column; heading column in PandaSet CSVs is already
# in degrees, so no scale.
GPS_UNIT_SCALES: dict[str, float] = {
    "Vehicle_Speed_kph": 3.6,
}


def simplify_class(pandaset_class: str) -> str:
    """Map PandaSet's published class names to the LakeVision detection_class
    vocabulary the other adapters use."""
    c = (pandaset_class or "").lower()
    if "pedestrian" in c:
        return "pedestrian"
    if any(s in c for s in ("bicycle", "motorcycle", "rider")):
        return "cyclist"
    if "car" in c or c == "personal mobility device":
        return "car"
    if "truck" in c or "trailer" in c or "construction" in c:
        return "truck"
    if "bus" in c:
        return "bus"
    if "vansuv" in c.replace(" ", "") or "van" in c or "suv" in c:
        return "van"
    if "animal" in c:
        return "animal"
    return c or "other"


def stable_int_id(text: str) -> int:
    h = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
    return int(h, 16) & 0x7FFFFFFFFFFFFFFF


# ── Typed records ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scene:
    """One PandaSet sequence = one container."""
    container_id: int
    sequence_id: str             # 3-digit numeric id, e.g. "001"
    name: str                    # = sequence_id
    description: str
    nbr_samples: int
    sequence_dir: str
    start_ts_us: int


@dataclass(frozen=True)
class Sample:
    """One PandaSet keyframe (10 Hz). PandaSet keeps frame indices 00–79
    inside an 8-second sequence."""
    container_id: int
    sample_token: str            # "{sequence_id}#{frame_index:02d}"
    timestamp_us: int
    sequence_id: str
    frame_index: int
    sensor_files: dict[str, str]


@dataclass(frozen=True)
class SampleData:
    container_id: int
    sample_token: str
    sensor_name: str
    channel_id: int
    timestamp_us: int
    file_path: str
    fileformat: str
    width: int
    height: int


@dataclass(frozen=True)
class EgoPose:
    """Identity pose. PandaSet cuboids are in the vehicle frame already."""
    container_id: int
    timestamp_us: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class Annotation:
    """One PandaSet 3D cuboid annotation."""
    container_id: int
    sample_token: str
    timestamp_us: int
    object_id: int               # stable hash of `uuid`
    uuid: str                    # PandaSet's per-instance identifier (persistent across frames)
    category_name: str           # raw label string from the pickle
    detection_class: str
    translation: tuple[float, float, float]
    size: tuple[float, float, float]   # length, width, height (PandaSet native order)
    rotation: tuple[float, float, float, float]  # quaternion (w, x, y, z) derived from yaw
    yaw_rad: float


# ── Loader ──────────────────────────────────────────────────────────────────


_SEQUENCE_DIR_RE = re.compile(r"^\d{3}$")
FRAME_DT_US: int = 100_000  # 10 Hz


class PandaSetLoader:
    """Filesystem loader for the PandaSet directory tree."""

    CAMERAS: tuple[str, ...] = (
        "front_camera",
        "front_left_camera",
        "front_right_camera",
        "left_camera",
        "right_camera",
        "back_camera",
    )

    def __init__(self, dataroot: str, dataset_version: str = "v1") -> None:
        self._dataroot = dataroot
        self._dataset_version = dataset_version
        self._root = Path(dataroot)

    @property
    def dataroot(self) -> str:
        return self._dataroot

    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    # ── Scenes ──────────────────────────────────────────────────────────────

    def scenes(self) -> Iterator[Scene]:
        if not self._root.is_dir():
            return
        for entry in sorted(self._root.iterdir()):
            if not entry.is_dir() or not _SEQUENCE_DIR_RE.match(entry.name):
                continue
            lidar_dir = entry / "lidar"
            nbr_samples = (
                sum(1 for _ in lidar_dir.glob("*.pkl.gz")) if lidar_dir.is_dir() else 0
            )
            ts_us = _scene_start_us(entry)
            yield Scene(
                container_id=stable_int_id(f"pandaset/{entry.name}"),
                sequence_id=entry.name,
                name=entry.name,
                description=f"PandaSet sequence {entry.name}",
                nbr_samples=nbr_samples,
                sequence_dir=str(entry),
                start_ts_us=ts_us,
            )

    def list_scenes(self) -> list[Scene]:
        return list(self.scenes())

    # ── Samples ─────────────────────────────────────────────────────────────

    def samples_in_scene(self, scene: Scene) -> Iterator[Sample]:
        lidar_dir = Path(scene.sequence_dir) / "lidar"
        if not lidar_dir.is_dir():
            return
        per_frame_ts = self._frame_timestamps(scene.sequence_id)
        for pkl in sorted(lidar_dir.glob("*.pkl.gz")):
            frame_index = _frame_index_from_filename(pkl.name)
            if frame_index is None:
                continue
            sensor_files: dict[str, str] = {"lidar": str(pkl)}
            for cam in self.CAMERAS:
                cam_path = Path(scene.sequence_dir) / "camera" / cam / f"{frame_index:02d}.jpg"
                if cam_path.is_file():
                    sensor_files[cam] = str(cam_path)
            ts_us = per_frame_ts[frame_index] if frame_index < len(per_frame_ts) else (
                scene.start_ts_us + frame_index * FRAME_DT_US
            )
            yield Sample(
                container_id=scene.container_id,
                sample_token=f"{scene.sequence_id}#{frame_index:02d}",
                timestamp_us=ts_us,
                sequence_id=scene.sequence_id,
                frame_index=frame_index,
                sensor_files=sensor_files,
            )

    @lru_cache(maxsize=64)
    def _frame_timestamps(self, sequence_id: str) -> tuple[int, ...]:
        """Read `meta/timestamps.json` → list of per-frame microsecond timestamps.

        PandaSet ships timestamps as floats in seconds; we convert to µs.
        """
        path = self._root / sequence_id / "meta" / "timestamps.json"
        if not path.is_file():
            return ()
        try:
            with open(path) as f:
                values = json.load(f)
        except (OSError, json.JSONDecodeError):
            return ()
        if not isinstance(values, list):
            return ()
        return tuple(int(float(v) * 1_000_000) for v in values)

    # ── Sample data (for perception_channels) ───────────────────────────────

    def all_sample_data_in_scene(self, scene: Scene) -> Iterator[SampleData]:
        for sample in self.samples_in_scene(scene):
            for sensor_name, file_path in sample.sensor_files.items():
                fileformat = Path(file_path).suffix.lstrip(".").lower()
                channel_id = SENSOR_CHANNEL_IDS.get(sensor_name, 0)
                if sensor_name == "lidar":
                    # Emit one perception_channels row per physical LiDAR sensor
                    # so the table can be partitioned by sensor downstream.
                    for sid, lakevision_name in (("lidar_pandar64", "lidar_pandar64"),
                                                  ("lidar_pandargt", "lidar_pandargt")):
                        yield SampleData(
                            container_id=sample.container_id,
                            sample_token=sample.sample_token,
                            sensor_name=lakevision_name,
                            channel_id=SENSOR_CHANNEL_IDS[lakevision_name],
                            timestamp_us=sample.timestamp_us,
                            file_path=file_path,
                            fileformat=fileformat,
                            width=0,
                            height=0,
                        )
                    continue
                yield SampleData(
                    container_id=sample.container_id,
                    sample_token=sample.sample_token,
                    sensor_name=sensor_name,
                    channel_id=channel_id,
                    timestamp_us=sample.timestamp_us,
                    file_path=file_path,
                    fileformat=fileformat,
                    width=0,
                    height=0,
                )

    # ── Ego pose (always identity) ──────────────────────────────────────────

    def ego_pose_for_sample(self, sample: Sample) -> EgoPose:
        return EgoPose(
            container_id=sample.container_id,
            timestamp_us=sample.timestamp_us,
            translation=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
        )

    # ── Annotations ─────────────────────────────────────────────────────────

    def annotations_in_sample(self, sample: Sample) -> Iterator[Annotation]:
        ann_path = (
            self._root / sample.sequence_id / "annotations" / "cuboids"
            / f"{sample.frame_index:02d}.pkl.gz"
        )
        if not ann_path.is_file():
            return
        records = _read_pickle_gz(ann_path)
        if records is None:
            return
        for row in _iter_records(records):
            yield _annotation_from_row(row, sample.container_id, sample.sample_token, sample.timestamp_us)

    # ── Camera calibration ──────────────────────────────────────────────────

    @lru_cache(maxsize=64)
    def _camera_intrinsics(self, sequence_id: str, camera: str) -> dict:
        path = self._root / sequence_id / "camera" / camera / "intrinsics.json"
        if not path.is_file():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    @lru_cache(maxsize=64)
    def _camera_poses(self, sequence_id: str, camera: str) -> list[dict]:
        path = self._root / sequence_id / "camera" / camera / "poses.json"
        if not path.is_file():
            return []
        try:
            with open(path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return []
        return payload if isinstance(payload, list) else []

    def camera_calib_for_sample(self, sample: Sample) -> dict[str, dict]:
        """Per-camera calibration dict shaped for the generic projection pipeline.

        PandaSet `intrinsics.json` carries fx/fy/cx/cy as separate fields; we
        assemble the 3×3 K matrix here so `camera.py` can stay generic. Poses
        are per-frame; we use the per-frame entry matching `sample.frame_index`.
        """
        out: dict[str, dict] = {}
        for cam in self.CAMERAS:
            intrinsics = self._camera_intrinsics(sample.sequence_id, cam)
            poses = self._camera_poses(sample.sequence_id, cam)
            if not intrinsics or not poses or sample.frame_index >= len(poses):
                continue
            pose = poses[sample.frame_index]
            K = [
                [float(intrinsics.get("fx", 0.0)), 0.0,                              float(intrinsics.get("cx", 0.0))],
                [0.0,                              float(intrinsics.get("fy", 0.0)), float(intrinsics.get("cy", 0.0))],
                [0.0,                              0.0,                              1.0],
            ]
            position = pose.get("position", {})
            heading = pose.get("heading", {})
            out[cam] = {
                "intrinsic": K,
                "sensor_rotation": [
                    float(heading.get("w", 1.0)),
                    float(heading.get("x", 0.0)),
                    float(heading.get("y", 0.0)),
                    float(heading.get("z", 0.0)),
                ],
                "sensor_translation": [
                    float(position.get("x", 0.0)),
                    float(position.get("y", 0.0)),
                    float(position.get("z", 0.0)),
                ],
                "ego_rotation": [1.0, 0.0, 0.0, 0.0],
                "ego_translation": [0.0, 0.0, 0.0],
                "width": int(intrinsics.get("width", 1920)),
                "height": int(intrinsics.get("height", 1080)),
                "sensor_id": cam,
            }
        return out

    # ── GPS / per-scene scalars ─────────────────────────────────────────────

    def gps_for_scene(self, scene: Scene) -> dict[str, list[tuple[int, float]]]:
        """Read `meta/gps.csv` → {LakeVision channel name: [(ts_us, value)]}.

        PandaSet GPS CSVs typically include a `timestamp` column (seconds)
        plus speed/heading/lat/lon. Missing files or malformed rows return an
        empty dict.
        """
        path = self._root / scene.sequence_id / "meta" / "gps.csv"
        if not path.is_file():
            return {}
        result: dict[str, list[tuple[int, float]]] = {}
        try:
            with open(path, newline="") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ts_raw = row.get("timestamp") or row.get("time")
                    if ts_raw is None:
                        continue
                    try:
                        ts_us = int(float(ts_raw) * 1_000_000)
                    except ValueError:
                        continue
                    for col, value in row.items():
                        if value is None or value == "" or col is None:
                            continue
                        channel = GPS_COLUMN_TO_CHANNEL.get(col.lower())
                        if not channel:
                            continue
                        try:
                            v = float(value)
                        except ValueError:
                            continue
                        scale = GPS_UNIT_SCALES.get(channel, 1.0)
                        result.setdefault(channel, []).append((ts_us, v * scale))
        except OSError:
            return {}
        return result

    # ── Counts ──────────────────────────────────────────────────────────────

    def count_summary(self) -> dict[str, int]:
        scenes = self.list_scenes()
        return {
            "scenes": len(scenes),
            "samples": sum(s.nbr_samples for s in scenes),
        }


# ── Helpers ─────────────────────────────────────────────────────────────────


def _scene_start_us(scene_dir: Path) -> int:
    """Read the first timestamp from `meta/timestamps.json` for ordering.
    Falls back to a stable hash if absent so cross-scene order is deterministic.
    """
    path = scene_dir / "meta" / "timestamps.json"
    if path.is_file():
        try:
            with open(path) as f:
                values = json.load(f)
            if isinstance(values, list) and values:
                return int(float(values[0]) * 1_000_000)
        except (OSError, json.JSONDecodeError, ValueError):
            pass
    return stable_int_id(str(scene_dir)) & 0xFFFFFFFF


_FRAME_INDEX_RE = re.compile(r"^(\d{2})\.")


def _frame_index_from_filename(name: str) -> int | None:
    m = _FRAME_INDEX_RE.match(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


def _read_pickle_gz(path: Path):
    """Read a gzip-compressed pickle. Returns whatever the pickle holds — a
    `pd.DataFrame`, list of dicts, or numpy array. Caller normalises."""
    try:
        with gzip.open(path, "rb") as f:
            return pickle.load(f)  # noqa: S301 — adapter explicitly loads internal-format files
    except (OSError, pickle.UnpicklingError, EOFError):
        return None


def _iter_records(payload) -> Iterator[dict]:
    """Normalize the pickle payload into an iterator of dict rows.

    PandaSet's published cuboid pickle is a `pd.DataFrame` with one row per
    annotation; we coerce to dict-per-row so the consumer is pandas-agnostic.
    """
    if payload is None:
        return
    # Pandas DataFrame path (without importing pandas at module top so tests
    # don't require pandas for the non-annotation surfaces).
    if hasattr(payload, "to_dict") and hasattr(payload, "columns"):
        for record in payload.to_dict(orient="records"):
            yield record
        return
    if isinstance(payload, list):
        for row in payload:
            if isinstance(row, dict):
                yield row
        return


def _quat_from_yaw(yaw_rad: float) -> tuple[float, float, float, float]:
    """Convert a yaw angle (rad) around +Z to a (w, x, y, z) quaternion."""
    import math
    half = yaw_rad / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _get(row: dict, *keys, default=None):
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return default


def _annotation_from_row(
    row: dict,
    container_id: int,
    sample_token: str,
    timestamp_us: int,
) -> Annotation:
    # PandaSet cuboids use `position.x/y/z`, `dimensions.x/y/z`, `yaw`, `uuid`,
    # `label` (high-level), `attributes.object_motion` etc. The exact column
    # names sometimes appear as `position` dict vs. flat fields; tolerate both.
    px = _get(row, "position.x") if "position.x" in row else None
    py = _get(row, "position.y") if "position.y" in row else None
    pz = _get(row, "position.z") if "position.z" in row else None
    if px is None and "position" in row and isinstance(row["position"], dict):
        pos = row["position"]
        px, py, pz = pos.get("x"), pos.get("y"), pos.get("z")

    dx = _get(row, "dimensions.x", "size.x", default=None)
    dy = _get(row, "dimensions.y", "size.y", default=None)
    dz = _get(row, "dimensions.z", "size.z", default=None)
    if dx is None and "dimensions" in row and isinstance(row["dimensions"], dict):
        d = row["dimensions"]
        dx, dy, dz = d.get("x"), d.get("y"), d.get("z")

    yaw = float(_get(row, "yaw", "heading.yaw", default=0.0))
    uuid = str(_get(row, "uuid", "id", default=sample_token))
    label = str(_get(row, "label", "class", default="Other"))

    return Annotation(
        container_id=container_id,
        sample_token=sample_token,
        timestamp_us=timestamp_us,
        object_id=stable_int_id(uuid),
        uuid=uuid,
        category_name=label,
        detection_class=simplify_class(label),
        translation=(float(px or 0.0), float(py or 0.0), float(pz or 0.0)),
        size=(float(dx or 0.0), float(dy or 0.0), float(dz or 0.0)),
        rotation=_quat_from_yaw(yaw),
        yaw_rad=yaw,
    )
