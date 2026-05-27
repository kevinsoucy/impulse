"""Thin wrapper around nuscenes-devkit exposing scene/sample/annotation iterables.

The devkit's API is token-based and verbose — every navigation step is a `.get()`
by token. This module hides that behind plain Python iterators returning typed
dataclasses so notebook code stays focused on the LakeVision data model rather
than the dataset's quirks.

NuScenes vocab → LakeVision vocab mapping:
  scene        → container (one container_id per scene)
  sample       → keyframe @ 2 Hz; sample.timestamp aligns to Impulse's microsecond clock
  sample_data  → per-sensor file at non-keyframe rates (cameras ~12 Hz, LiDAR ~20 Hz)
  channel      → sensor name (CAM_FRONT, LIDAR_TOP, etc.) → channel_id per (container, channel)
  sample_annotation → ground-truth 3D bbox; becomes one object_tracks / lidar_object_detections row
  instance_token → stable across-frame identity → object_tracks.object_id
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


# ── Channel ID scheme ───────────────────────────────────────────────────────
# Stable integer IDs per sensor name. NuScenes has a fixed sensor set, so we
# can hardcode the mapping (no need for a dynamic registry like in real MDF4 ingest).
SENSOR_CHANNEL_IDS: dict[str, int] = {
    # Cameras
    "CAM_FRONT":         101,
    "CAM_FRONT_LEFT":    102,
    "CAM_FRONT_RIGHT":   103,
    "CAM_BACK":          104,
    "CAM_BACK_LEFT":     105,
    "CAM_BACK_RIGHT":    106,
    # LiDAR
    "LIDAR_TOP":         200,
    # Radars
    "RADAR_FRONT":       301,
    "RADAR_FRONT_LEFT":  302,
    "RADAR_FRONT_RIGHT": 303,
    "RADAR_BACK_LEFT":   304,
    "RADAR_BACK_RIGHT":  305,
}

# Derived (synthesized) scalar channels — Phase 1.
DERIVED_CHANNEL_IDS: dict[str, int] = {
    "Vehicle_Speed_kph":              1001,
    "Vehicle_Accel_Longitudinal_ms2": 1002,
    "Steering_Angle_deg":             1003,
    "Pedestrian_Count":               2001,
    "Pedestrian_Nearest_Distance_m":  2002,
    "Vehicle_Count_Front":            2003,
    "Vehicle_Nearest_Distance_m":     2004,
    "Cyclist_Count":                  2005,
    "Cyclist_Nearest_Distance_m":     2006,
}


# ── Typed records ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Scene:
    """A NuScenes scene = one LakeVision container."""
    container_id: int  # stable integer hash of the scene token
    scene_token: str
    name: str
    description: str
    log_token: str
    nbr_samples: int
    first_sample_token: str
    last_sample_token: str


@dataclass(frozen=True)
class Sample:
    """A NuScenes keyframe @ 2 Hz. timestamp is microseconds (matches Impulse tstart)."""
    container_id: int
    sample_token: str
    timestamp_us: int
    scene_token: str
    sensor_data_tokens: dict[str, str]  # sensor_name → sample_data_token


@dataclass(frozen=True)
class SampleData:
    """One sensor file at a non-keyframe timestamp."""
    container_id: int
    sample_token: str  # parent keyframe
    sensor_name: str  # e.g. "CAM_FRONT", "LIDAR_TOP"
    channel_id: int
    timestamp_us: int
    file_path: str  # absolute path under dataroot (or UC volume after copy)
    fileformat: str  # "jpg", "pcd", "bin", …
    width: int  # 0 for non-image
    height: int  # 0 for non-image


@dataclass(frozen=True)
class EgoPose:
    """Ego vehicle pose in the global frame at a given timestamp."""
    container_id: int
    timestamp_us: int
    translation: tuple[float, float, float]  # x, y, z meters in global frame
    rotation: tuple[float, float, float, float]  # quaternion (w, x, y, z)


@dataclass(frozen=True)
class Annotation:
    """A ground-truth 3D object annotation. Always in the global frame in NuScenes."""
    container_id: int
    sample_token: str
    timestamp_us: int
    instance_token: str
    object_id: int
    category_name: str
    detection_class: str
    translation: tuple[float, float, float]
    size: tuple[float, float, float]  # NuScenes order: (width, length, height)
    rotation: tuple[float, float, float, float]
    num_lidar_pts: int
    num_radar_pts: int
    visibility_token: str  # "1"..."4" → 0-40%, 40-60%, 60-80%, 80-100% visible


# ── Category simplification ──────────────────────────────────────────────────

def simplify_category(nuscenes_category: str) -> str:
    """Reduce NuScenes hierarchical category to a single LakeVision detection_class."""
    parts = nuscenes_category.split(".")
    if parts[0] == "human":
        return "pedestrian"
    if parts[0] == "vehicle":
        if len(parts) >= 2:
            v = parts[1]
            if v in ("car", "truck", "bus", "motorcycle", "bicycle"):
                return "cyclist" if v == "bicycle" else v
            return v
    if parts[0] == "movable_object":
        return parts[1] if len(parts) >= 2 else "movable_object"
    return parts[0]


def stable_int_id(token: str) -> int:
    """Map a NuScenes token (32-char hex) to a stable non-negative 63-bit int."""
    return int(token[:16], 16) & 0x7FFFFFFFFFFFFFFF


# ── Loader ──────────────────────────────────────────────────────────────────


class NuScenesLoader:
    """Wraps `nuscenes.NuScenes` and exposes iterables over typed records.

    Lazy: does NOT eagerly enumerate everything; iterates via the devkit's
    token navigation. Safe to use on trainval — memory usage tracks `nbr_samples`.
    """

    def __init__(self, dataroot: str, dataset_version: str, verbose: bool = False) -> None:
        # Import here so the module can be imported (e.g. for unit tests) without
        # nuscenes-devkit installed. The deployed bundle installs the dep.
        from nuscenes.nuscenes import NuScenes

        self._dataroot = dataroot
        self._dataset_version = dataset_version
        self._nusc = NuScenes(
            version=dataset_version,
            dataroot=dataroot,
            verbose=verbose,
        )

    @property
    def dataroot(self) -> str:
        return self._dataroot

    @property
    def dataset_version(self) -> str:
        return self._dataset_version

    # ── Scenes ──────────────────────────────────────────────────────────────

    def scenes(self) -> Iterator[Scene]:
        for raw in self._nusc.scene:
            yield Scene(
                container_id=stable_int_id(raw["token"]),
                scene_token=raw["token"],
                name=raw["name"],
                description=raw["description"],
                log_token=raw["log_token"],
                nbr_samples=raw["nbr_samples"],
                first_sample_token=raw["first_sample_token"],
                last_sample_token=raw["last_sample_token"],
            )

    def list_scenes(self) -> list[Scene]:
        return list(self.scenes())

    # ── Samples (keyframes at 2 Hz) ─────────────────────────────────────────

    def samples_in_scene(self, scene: Scene) -> Iterator[Sample]:
        tok = scene.first_sample_token
        while tok:
            s = self._nusc.get("sample", tok)
            yield Sample(
                container_id=scene.container_id,
                sample_token=s["token"],
                timestamp_us=int(s["timestamp"]),
                scene_token=scene.scene_token,
                sensor_data_tokens=dict(s["data"]),
            )
            tok = s["next"]

    # ── Sample data (per-sensor files) ──────────────────────────────────────

    def sample_data_for_sample(self, sample: Sample) -> Iterator[SampleData]:
        for sensor_name, sd_token in sample.sensor_data_tokens.items():
            sd = self._nusc.get("sample_data", sd_token)
            yield SampleData(
                container_id=sample.container_id,
                sample_token=sample.sample_token,
                sensor_name=sensor_name,
                channel_id=SENSOR_CHANNEL_IDS.get(sensor_name, 0),
                timestamp_us=int(sd["timestamp"]),
                file_path=f"{self._dataroot}/{sd['filename']}",
                fileformat=sd["fileformat"],
                width=int(sd.get("width", 0)),
                height=int(sd.get("height", 0)),
            )

    def all_sample_data_in_scene(self, scene: Scene) -> Iterator[SampleData]:
        """Iterate every sample_data in the scene, including non-keyframe sensor sweeps."""
        seen: set[str] = set()
        for sample in self.samples_in_scene(scene):
            for sensor_name, sd_token in sample.sensor_data_tokens.items():
                tok = sd_token
                while tok and tok not in seen:
                    seen.add(tok)
                    sd = self._nusc.get("sample_data", tok)
                    yield SampleData(
                        container_id=scene.container_id,
                        sample_token=sd["sample_token"],
                        sensor_name=sensor_name,
                        channel_id=SENSOR_CHANNEL_IDS.get(sensor_name, 0),
                        timestamp_us=int(sd["timestamp"]),
                        file_path=f"{self._dataroot}/{sd['filename']}",
                        fileformat=sd["fileformat"],
                        width=int(sd.get("width", 0)),
                        height=int(sd.get("height", 0)),
                    )
                    tok = sd["next"]

    # ── Ego poses ───────────────────────────────────────────────────────────

    def ego_pose_for_sample(self, sample: Sample) -> EgoPose:
        sd_token = sample.sensor_data_tokens["LIDAR_TOP"]
        sd = self._nusc.get("sample_data", sd_token)
        ep = self._nusc.get("ego_pose", sd["ego_pose_token"])
        return EgoPose(
            container_id=sample.container_id,
            timestamp_us=int(ep["timestamp"]),
            translation=tuple(ep["translation"]),
            rotation=tuple(ep["rotation"]),
        )

    # ── Annotations ─────────────────────────────────────────────────────────

    def annotations_in_sample(self, sample: Sample) -> Iterator[Annotation]:
        s = self._nusc.get("sample", sample.sample_token)
        for ann_token in s["anns"]:
            a = self._nusc.get("sample_annotation", ann_token)
            yield Annotation(
                container_id=sample.container_id,
                sample_token=sample.sample_token,
                timestamp_us=sample.timestamp_us,
                instance_token=a["instance_token"],
                object_id=stable_int_id(a["instance_token"]),
                category_name=a["category_name"],
                detection_class=simplify_category(a["category_name"]),
                translation=tuple(a["translation"]),
                size=tuple(a["size"]),
                rotation=tuple(a["rotation"]),
                num_lidar_pts=int(a.get("num_lidar_pts", 0)),
                num_radar_pts=int(a.get("num_radar_pts", 0)),
                visibility_token=a.get("visibility_token", ""),
            )

    # ── Camera calibration ──────────────────────────────────────────────────

    def camera_calib_for_sample(self, sample: Sample) -> dict[str, dict]:
        """Return {sensor_name: calib_info} for all camera sensors at this keyframe."""
        result: dict[str, dict] = {}
        for sensor_name, sd_token in sample.sensor_data_tokens.items():
            if not sensor_name.startswith("CAM_"):
                continue
            sd = self._nusc.get("sample_data", sd_token)
            cs = self._nusc.get("calibrated_sensor", sd["calibrated_sensor_token"])
            ep = self._nusc.get("ego_pose", sd["ego_pose_token"])
            result[sensor_name] = {
                "intrinsic": cs["camera_intrinsic"],
                "sensor_rotation": cs["rotation"],
                "sensor_translation": cs["translation"],
                "ego_rotation": ep["rotation"],
                "ego_translation": ep["translation"],
                "width": int(sd.get("width", 1600)),
                "height": int(sd.get("height", 900)),
                "sensor_id": sensor_name.lower(),
            }
        return result

    # ── Counts ──────────────────────────────────────────────────────────────

    def count_summary(self) -> dict[str, int]:
        return {
            "scenes": len(self._nusc.scene),
            "samples": len(self._nusc.sample),
            "sample_data": len(self._nusc.sample_data),
            "annotations": len(self._nusc.sample_annotation),
            "ego_poses": len(self._nusc.ego_pose),
        }
