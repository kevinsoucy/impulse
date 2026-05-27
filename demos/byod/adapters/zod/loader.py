"""Filesystem loader for ZOD (Zenseact Open Dataset) Sequences.

ZOD publishes an official Python SDK (`pip install zod`) plus a documented
on-disk layout. This adapter reads the directory tree directly — same pattern
as the A2D2 adapter — so unit tests can use synthetic numpy + h5py fixtures
without depending on the SDK or a live download.

ZOD vocab → LakeVision vocab mapping:
  sequence (~20 s clip)        → container (one container_id per sequence)
  per-frame camera/lidar/radar → sample (one keyframe = one Sample)
  per-sensor file              → SampleData (one per sensor file)
  ego pose                     → derived from `oxts.hdf5` (filtered IMU + GNSS)
  3D annotation                → Annotation (one per object per frame)

ZOD specifics that shape this module:
  - Annotation coordinates are in the **vehicle (ego) frame** already.
    Same as A2D2 — no global→ego transform needed downstream.
  - Annotation `uuid` is stable across frames within a sequence, so
    `relative_velocity_ms` IS computable (unlike A2D2).
  - Radar has per-detection Doppler (`radar_attribute.doppler` in m/s) —
    unique among the four BYOD adapters; visible in `object_tracks.source`.
  - 3-LiDAR roof rack is pre-fused at annotation time → `sensor_id = "LIDAR_FUSED"`.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


# ── Channel ID scheme ───────────────────────────────────────────────────────

SENSOR_CHANNEL_IDS: dict[str, int] = {
    # ZOD ships several camera variants (blur, dnat, original) of the same
    # forward-facing camera. We index the privacy-blurred variant — that's the
    # one Zenseact recommends for downstream use.
    "camera_front_blur":   101,
    "lidar_velodyne":      200,  # 3-LiDAR fused, single logical stream
    "radar_front":         301,
}

# Real OXTS bus signals + detection-aggregate channels.
DERIVED_CHANNEL_IDS: dict[str, int] = {
    # Real OXTS-decoded scalars (filtered IMU + GNSS in oxts.hdf5)
    "Vehicle_Speed_kph":              1001,
    "Vehicle_Accel_Longitudinal_ms2": 1002,
    "Vehicle_Accel_Lateral_ms2":      1003,
    "Vehicle_Accel_Vertical_ms2":     1004,
    "Yaw_Rate_rads":                  1005,
    "Roll_Rate_rads":                 1006,
    "Pitch_Rate_rads":                1007,
    "Heading_deg":                    1008,
    "Pitch_Angle_deg":                1009,
    "Roll_Angle_deg":                 1010,
    "Latitude_deg":                   1011,
    "Longitude_deg":                  1012,
    # Detection-aggregate channels (per ADR-2, derived from annotations)
    "Pedestrian_Count":               2001,
    "Pedestrian_Nearest_Distance_m":  2002,
    "Vehicle_Count_Front":            2003,
    "Vehicle_Nearest_Distance_m":     2004,
    "Cyclist_Count":                  2005,
    "Cyclist_Nearest_Distance_m":     2006,
}


# OXTS HDF5 dataset path → LakeVision channel name. The (path, axis) tuple
# disambiguates the multi-axis arrays (e.g. `/acceleration` shape (N, 3) → 3
# separate channels). Axis = None for 1-D datasets.
OXTS_DATASET_TO_CHANNEL: dict[tuple[str, int | None], str] = {
    ("/speed",         None): "Vehicle_Speed_kph",
    ("/acceleration",  0):    "Vehicle_Accel_Longitudinal_ms2",
    ("/acceleration",  1):    "Vehicle_Accel_Lateral_ms2",
    ("/acceleration",  2):    "Vehicle_Accel_Vertical_ms2",
    ("/angular_rate",  0):    "Roll_Rate_rads",
    ("/angular_rate",  1):    "Pitch_Rate_rads",
    ("/angular_rate",  2):    "Yaw_Rate_rads",
    ("/heading",       None): "Heading_deg",
    ("/pitch",         None): "Pitch_Angle_deg",
    ("/roll",          None): "Roll_Angle_deg",
    ("/latitude",      None): "Latitude_deg",
    ("/longitude",     None): "Longitude_deg",
}


# Unit conversion to apply when emitting `channels` rows. OXTS speed is logged
# in m/s but LakeVision's canonical kinematic channel is kph; ZOD heading is
# in radians in the HDF5 but the canonical channel is degrees.
OXTS_UNIT_SCALES: dict[str, float] = {
    "Vehicle_Speed_kph":   3.6,        # m/s → kph
    "Heading_deg":         57.2957795, # rad → deg
    "Pitch_Angle_deg":     57.2957795,
    "Roll_Angle_deg":      57.2957795,
}


# ZOD published annotation `name` (or `subclass`) → simplified LakeVision class.
def simplify_class(zod_name: str, zod_subclass: str | None = None) -> str:
    """Map a ZOD annotation name/subclass to the LakeVision detection_class vocab.

    Public ZOD annotations use a coarse `name` (Vehicle | VulnerableVehicle |
    Pedestrian | …) plus a finer `subclass` (Car | Truck | Bicycle | …). We
    prefer `subclass` when present and informative.
    """
    if zod_subclass:
        s = zod_subclass.lower()
        if s in ("car",):
            return "car"
        if s in ("truck", "trailer", "construction_vehicle"):
            return "truck"
        if s in ("bus",):
            return "bus"
        if s in ("vansuv", "van"):
            return "van"
        if s in ("bicycle", "bicyclerider", "motorbike", "motorbiker",
                 "motorcycle", "motorcyclerider"):
            return "cyclist"
    n = (zod_name or "").lower()
    if n == "pedestrian":
        return "pedestrian"
    if n in ("vulnerablevehicle", "vulnerable_vehicle"):
        return "cyclist"
    if n == "vehicle":
        return "car"  # generic fallback when subclass missing
    return n or "other"


def stable_int_id(text: str) -> int:
    """Stable non-negative 63-bit int from any string identifier."""
    h = hashlib.blake2b(text.encode(), digest_size=8).hexdigest()
    return int(h, 16) & 0x7FFFFFFFFFFFFFFF


# ── Typed records ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scene:
    """One ZOD sequence = one LakeVision container."""
    container_id: int
    sequence_id: str            # zero-padded numeric id, e.g. "000001"
    name: str                   # = sequence_id
    description: str
    nbr_samples: int
    sequence_dir: str           # absolute path under {dataroot}/sequences/{sequence_id}
    start_ts_us: int            # earliest frame timestamp, used for ordering


@dataclass(frozen=True)
class Sample:
    """One ZOD keyframe at LiDAR rate (~10 Hz)."""
    container_id: int
    sample_token: str           # "{sequence_id}#{frame_id}"
    timestamp_us: int
    sequence_id: str
    frame_id: str               # filename stem (matches the lidar/camera/radar file)
    sensor_files: dict[str, str]


@dataclass(frozen=True)
class SampleData:
    """One sensor file at a given sample timestamp."""
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
    """Ego pose — identity in vehicle frame; OXTS provides position but ZOD
    annotations are already in the vehicle frame so we don't need a non-identity
    pose for the projection pipeline. Same role as the A2D2 EgoPose."""
    container_id: int
    timestamp_us: int
    translation: tuple[float, float, float]
    rotation: tuple[float, float, float, float]


@dataclass(frozen=True)
class Annotation:
    """ZOD 3D bounding box in the **vehicle (ego) frame**."""
    container_id: int
    sample_token: str
    timestamp_us: int
    object_id: int              # stable hash of `uuid` so the schema's LONG type holds
    uuid: str                   # ZOD's persistent instance identifier (across frames)
    category_name: str          # raw ZOD name + subclass (e.g. "Vehicle/Car")
    detection_class: str        # simplified
    translation: tuple[float, float, float]
    size: tuple[float, float, float]  # length, width, height (ZOD native order)
    rotation: tuple[float, float, float, float]
    num_lidar_pts: int          # `lidar_attribute.num_points` if present
    num_radar_pts: int          # `radar_attribute.num_points` if present
    occlusion: float
    radar_doppler_ms: float | None  # ZOD-unique; None if no radar coverage


# ── Loader ──────────────────────────────────────────────────────────────────


# Frame filenames are content-token strings, not numeric indices. ZOD's stem
# pattern is roughly `{17-digit-timestamp-ns}_{8-char-token}`; we don't parse
# the token, only the timestamp.
_FRAME_TS_RE = re.compile(r"^(\d{16,19})")


class ZodLoader:
    """Filesystem loader for `{dataroot}/sequences/{sequence_id}/` trees.

    Lazy by design: scene enumeration walks the filesystem; no devkit warmup.
    The optional `zod` SDK is never imported here — keeps unit tests SDK-free.
    """

    PRIMARY_CAMERA = "camera_front_blur"
    LIDAR_SUBDIR = "lidar_velodyne"
    RADAR_SUBDIR = "radar"

    def __init__(self, dataroot: str, dataset_version: str = "sequences-mini") -> None:
        self._dataroot = dataroot
        self._dataset_version = dataset_version
        # ZOD's published layout puts the version (mini/full) at the top level
        # via a separate dataroot — not nested under the dataroot like A2D2. We
        # support both shapes: if `dataroot/sequences/` exists, use it directly;
        # otherwise resolve `dataroot/<version>/sequences/`.
        if (Path(dataroot) / "sequences").is_dir():
            self._sequences_root = Path(dataroot) / "sequences"
        else:
            self._sequences_root = Path(dataroot) / dataset_version / "sequences"

    @property
    def dataroot(self) -> str:
        return self._dataroot

    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    # ── Scenes ──────────────────────────────────────────────────────────────

    def scenes(self) -> Iterator[Scene]:
        if not self._sequences_root.is_dir():
            return
        for entry in sorted(self._sequences_root.iterdir()):
            if not entry.is_dir():
                continue
            cam_dir = entry / self.PRIMARY_CAMERA
            jpgs = sorted(cam_dir.glob("*.jpg")) if cam_dir.is_dir() else []
            start_ts = _ts_from_filename(jpgs[0].name) if jpgs else 0
            yield Scene(
                container_id=stable_int_id(f"zod/{entry.name}"),
                sequence_id=entry.name,
                name=entry.name,
                description=f"ZOD sequence {entry.name}",
                nbr_samples=len(jpgs),
                sequence_dir=str(entry),
                start_ts_us=start_ts,
            )

    def list_scenes(self) -> list[Scene]:
        return list(self.scenes())

    # ── Samples ─────────────────────────────────────────────────────────────

    def samples_in_scene(self, scene: Scene) -> Iterator[Sample]:
        """Yield Samples in ascending-timestamp order.

        Samples are keyed off the LiDAR cadence (~10 Hz). For each LiDAR file
        we pair the closest-timestamp camera + radar files.
        """
        lidar_dir = Path(scene.sequence_dir) / self.LIDAR_SUBDIR
        if not lidar_dir.is_dir():
            return

        cam_files = _index_files(Path(scene.sequence_dir) / self.PRIMARY_CAMERA, "*.jpg")
        radar_files = _index_files(Path(scene.sequence_dir) / self.RADAR_SUBDIR, "*.npy")

        for lidar_path in sorted(lidar_dir.glob("*.npy")):
            ts_us = _ts_from_filename(lidar_path.name)
            if ts_us == 0:
                continue
            frame_id = lidar_path.stem
            sensor_files: dict[str, str] = {self.LIDAR_SUBDIR: str(lidar_path)}
            cam = _closest_file(cam_files, ts_us)
            if cam:
                sensor_files[self.PRIMARY_CAMERA] = str(cam)
            radar = _closest_file(radar_files, ts_us)
            if radar:
                sensor_files[self.RADAR_SUBDIR] = str(radar)
            yield Sample(
                container_id=scene.container_id,
                sample_token=f"{scene.sequence_id}#{frame_id}",
                timestamp_us=ts_us,
                sequence_id=scene.sequence_id,
                frame_id=frame_id,
                sensor_files=sensor_files,
            )

    # ── Sample data ─────────────────────────────────────────────────────────

    def all_sample_data_in_scene(self, scene: Scene) -> Iterator[SampleData]:
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

    # ── Ego pose ────────────────────────────────────────────────────────────

    def ego_pose_for_sample(self, sample: Sample) -> EgoPose:
        """Identity pose. ZOD annotations are in the vehicle frame, so the
        pipeline's `global_to_ego` reduces to a no-op."""
        return EgoPose(
            container_id=sample.container_id,
            timestamp_us=sample.timestamp_us,
            translation=(0.0, 0.0, 0.0),
            rotation=(1.0, 0.0, 0.0, 0.0),
        )

    # ── Annotations ─────────────────────────────────────────────────────────

    @lru_cache(maxsize=64)
    def _annotations_by_frame(self, sequence_id: str) -> dict[str, list[dict]]:
        """Read `annotations/object_detection_3d.json` once per sequence,
        bucket by `frame_id`. Returns a dict so per-sample lookup is O(1)."""
        ann_path = self._sequences_root / sequence_id / "annotations" / "object_detection_3d.json"
        if not ann_path.is_file():
            return {}
        try:
            with open(ann_path) as f:
                payload = json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}
        rows = payload if isinstance(payload, list) else payload.get("annotations", [])
        bucket: dict[str, list[dict]] = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            fid = row.get("frame_id")
            if fid is None:
                continue
            bucket.setdefault(str(fid), []).append(row)
        return bucket

    def annotations_in_sample(self, sample: Sample) -> Iterator[Annotation]:
        scene_id = sample.sequence_id
        for row in self._annotations_by_frame(scene_id).get(sample.frame_id, []):
            yield _annotation_from_dict(row, sample.container_id, sample.sample_token, sample.timestamp_us)

    # ── Camera calibration ──────────────────────────────────────────────────

    @lru_cache(maxsize=64)
    def _calibration(self, sequence_id: str) -> dict:
        path = self._sequences_root / sequence_id / "calibration.json"
        if not path.is_file():
            return {}
        try:
            with open(path) as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError):
            return {}

    def camera_calib_for_sample(self, sample: Sample) -> dict[str, dict]:
        """Camera calibration dict shaped for `mda_query_engine.perception`-side projection.

        Returns the same keys as the A2D2 adapter so `camera.py` can stay
        nearly identical: intrinsic, sensor_rotation, sensor_translation,
        ego_rotation, ego_translation, width, height, sensor_id.
        """
        calib = self._calibration(sample.sequence_id)
        cameras = calib.get("cameras", {})
        out: dict[str, dict] = {}
        entry = cameras.get(self.PRIMARY_CAMERA) or cameras.get("front_blur") or cameras.get("front")
        if not entry:
            return out
        intrinsic = entry.get("intrinsic", {})
        extrinsic = entry.get("extrinsic", {})
        K = intrinsic.get("K", intrinsic.get("matrix", _DEFAULT_INTRINSIC))
        dims = intrinsic.get("image_dimensions", intrinsic.get("resolution", [3848, 2168]))
        out[self.PRIMARY_CAMERA] = {
            "intrinsic": K,
            "sensor_rotation": extrinsic.get("rotation", [1.0, 0.0, 0.0, 0.0]),
            "sensor_translation": extrinsic.get("translation", [0.0, 0.0, 0.0]),
            "ego_rotation": [1.0, 0.0, 0.0, 0.0],
            "ego_translation": [0.0, 0.0, 0.0],
            "width": int(dims[0]),
            "height": int(dims[1]),
            "sensor_id": self.PRIMARY_CAMERA,
        }
        return out

    # ── OXTS (per-scene HDF5) ───────────────────────────────────────────────

    def oxts_for_scene(self, scene: Scene) -> dict[str, list[tuple[int, float]]]:
        """Read `oxts.hdf5` → {LakeVision channel name: [(ts_us, value)]}.

        Returns an empty dict for sequences missing OXTS or with unreadable
        HDF5 — not an error, since the bundle should still run with partial
        bus signal coverage.
        """
        path = Path(scene.sequence_dir) / "oxts.hdf5"
        if not path.is_file():
            return {}
        try:
            import h5py
        except ImportError:
            return {}
        result: dict[str, list[tuple[int, float]]] = {}
        try:
            with h5py.File(path, "r") as f:
                timestamps = _read_oxts_timestamps(f)
                if not timestamps:
                    return {}
                for (ds_path, axis), channel_name in OXTS_DATASET_TO_CHANNEL.items():
                    if ds_path not in f:
                        continue
                    arr = f[ds_path][...]
                    if arr.ndim == 0:
                        continue
                    if axis is None:
                        values = arr.tolist()
                    else:
                        if arr.ndim < 2 or arr.shape[1] <= axis:
                            continue
                        values = arr[:, axis].tolist()
                    if len(values) != len(timestamps):
                        # Skip rather than truncate — alignment matters.
                        continue
                    scale = OXTS_UNIT_SCALES.get(channel_name, 1.0)
                    result[channel_name] = [
                        (int(ts), float(v) * scale)
                        for ts, v in zip(timestamps, values)
                    ]
        except (OSError, KeyError):
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


def _ts_from_filename(name: str) -> int:
    """Extract the leading timestamp from a ZOD frame filename.

    ZOD's filename stem is `{epoch_nanoseconds}_{token}`; we accept 16–19 digit
    leading numbers to be defensive against minor format variations. Returns
    microseconds; falls back to 0 if no parse is possible (caller skips those).
    """
    m = _FRAME_TS_RE.match(name)
    if not m:
        return 0
    digits = m.group(1)
    n = int(digits)
    # Distinguish ns from µs by magnitude: any 2020+ POSIX µs timestamp has
    # 16 digits (< 10^17); the same instant in ns has 19 digits (> 10^18).
    # Threshold > 10^17 cleanly separates the two regimes.
    if n > 10**17:
        return n // 1000
    return n


def _index_files(directory: Path, pattern: str) -> list[tuple[int, Path]]:
    """Index `(timestamp_us, path)` pairs in `directory`, sorted by timestamp.
    Used to do nearest-timestamp matching across sensor streams."""
    if not directory.is_dir():
        return []
    out: list[tuple[int, Path]] = []
    for p in directory.glob(pattern):
        ts = _ts_from_filename(p.name)
        if ts > 0:
            out.append((ts, p))
    out.sort(key=lambda x: x[0])
    return out


def _closest_file(indexed: list[tuple[int, Path]], target_us: int) -> Path | None:
    """Return the file with the timestamp closest to `target_us`. None if empty."""
    if not indexed:
        return None
    best = indexed[0]
    best_delta = abs(best[0] - target_us)
    for ts, path in indexed[1:]:
        delta = abs(ts - target_us)
        if delta < best_delta:
            best, best_delta = (ts, path), delta
    return best[1]


def _read_oxts_timestamps(h5_file) -> list[int]:
    """Read the timestamp 1-D dataset that aligns with every OXTS signal array.

    ZOD's `oxts.hdf5` stores per-sample microsecond timestamps under `/timestamp`
    in the published layout. Accept a couple of plausible variants defensively.
    """
    for ds_path in ("/timestamp", "/timestamps", "/time"):
        if ds_path in h5_file:
            arr = h5_file[ds_path][...]
            return [int(x) for x in arr.tolist()]
    return []


def _quat_from_dict_or_list(value) -> tuple[float, float, float, float]:
    """Coerce ZOD's quaternion encoding into (w, x, y, z) tuple.

    ZOD uses either an inline list `[qw, qx, qy, qz]` or an object
    `{"qw": ..., "qx": ..., "qy": ..., "qz": ...}`. Default to identity if
    the structure isn't recognised."""
    if isinstance(value, dict):
        return (
            float(value.get("qw", 1.0)),
            float(value.get("qx", 0.0)),
            float(value.get("qy", 0.0)),
            float(value.get("qz", 0.0)),
        )
    if isinstance(value, (list, tuple)) and len(value) >= 4:
        return tuple(float(v) for v in value[:4])  # type: ignore[return-value]
    return (1.0, 0.0, 0.0, 0.0)


def _annotation_from_dict(
    row: dict,
    container_id: int,
    sample_token: str,
    timestamp_us: int,
) -> Annotation:
    box = row.get("box3d") or row
    center = box.get("center") or box.get("location") or [0.0, 0.0, 0.0]
    size = box.get("size") or [0.0, 0.0, 0.0]
    rotation = _quat_from_dict_or_list(box.get("orientation") or box.get("rotation"))

    name = str(row.get("name", ""))
    subclass = row.get("subclass") or row.get("class")
    detection_class = simplify_class(name, str(subclass) if subclass else None)
    category_name = f"{name}/{subclass}" if subclass else name

    lidar_attr = row.get("lidar_attribute") or {}
    radar_attr = row.get("radar_attribute") or {}
    doppler = radar_attr.get("doppler")
    radar_doppler_ms = float(doppler) if doppler is not None else None

    uuid = str(row.get("uuid", row.get("object_id", sample_token)))

    return Annotation(
        container_id=container_id,
        sample_token=sample_token,
        timestamp_us=timestamp_us,
        object_id=stable_int_id(uuid),
        uuid=uuid,
        category_name=category_name,
        detection_class=detection_class,
        translation=tuple(float(c) for c in center),
        size=tuple(float(s) for s in size),
        rotation=rotation,
        num_lidar_pts=int(lidar_attr.get("num_points", 0)),
        num_radar_pts=int(radar_attr.get("num_points", 0)),
        occlusion=float(row.get("occlusion", 0.0)),
        radar_doppler_ms=radar_doppler_ms,
    )


# Fallback intrinsic if the calibration.json doesn't ship one. Values reflect
# ZOD's published forward-camera resolution + a rough focal estimate; only
# used to keep the projection pipeline running when calibration is absent.
_DEFAULT_INTRINSIC = [
    [2000.0, 0.0, 1924.0],
    [0.0,    2000.0, 1084.0],
    [0.0,    0.0,    1.0],
]
