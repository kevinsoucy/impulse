"""Convert NuScenes annotations to LakeVision `object_tracks` rows.

Each NuScenes ground-truth annotation becomes one `object_tracks` row:
- `distance_m`, `azimuth`, `lane_offset` computed in ego frame
- `relative_velocity_ms` tracked across consecutive keyframes via `instance_token`
- `confidence` = 1.0 (ground truth, not a detection)
- `source` derived from per-annotation sensor coverage metadata

NuScenes annotations are at 2 Hz (keyframe rate). This matches the
Nyquist floor for ADAS event durations, so we populate every annotation. The
`object_tracks` table itself is then implicitly "full_stride" at 2 Hz —
TSAL-gating happens via downstream queries that JOIN against `event_instance_fact`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lib.geometry import LANE_WIDTH_M, azimuth_sector, lane_offset

from .loader import Annotation, EgoPose, NuScenesLoader, Scene
from .scalar_source import global_to_ego


# Re-exported for the migrated test modules.
__all__ = [
    "LANE_WIDTH_M",
    "_CAMERA_VISIBLE_TOKENS",
    "_compute_relative_velocity",
    "azimuth_sector",
    "lane_offset",
    "map_all_scenes",
    "map_scene_to_object_tracks",
    "source_from_annotation",
]


# NuScenes visibility tokens where camera detection is considered reliable (≥ 40% visible).
# Token "1" = 0–40% — object is mostly occluded; we exclude camera from source for those.
_CAMERA_VISIBLE_TOKENS = frozenset({"2", "3", "4"})


def source_from_annotation(
    num_lidar_pts: int,
    num_radar_pts: int,
    visibility_token: str,
) -> str:
    """Derive pipe-delimited sensor-coverage string from NuScenes annotation metadata.

    Uses the per-annotation sensor statistics NuScenes records at labeling time:
      num_lidar_pts  — LiDAR returns inside the 3D cuboid (> 0 → LiDAR coverage confirmed)
      num_radar_pts  — radar returns inside the 3D cuboid (> 0 → radar coverage confirmed)
      visibility_token — image-plane visibility: "1"=0-40%, "2"=40-60%, "3"=60-80%, "4"=80-100%

    This is sensor *coverage* at annotation time, not the output of a per-sensor
    detector. The honest framing is documented in the notebook that populates
    object_tracks.
    """
    parts: list[str] = []
    if num_lidar_pts > 0:
        parts.append("lidar")
    if num_radar_pts > 0:
        parts.append("radar")
    if visibility_token in _CAMERA_VISIBLE_TOKENS:
        parts.append("camera")
    # Fallback: the object was annotated by a human so it was perceivable by something.
    return "|".join(parts) if parts else "camera"


def map_scene_to_object_tracks(
    loader: NuScenesLoader,
    scene: Scene,
    min_confidence: float = 0.5,
) -> list[dict]:
    """Return one object_tracks dict per annotation in this scene."""
    samples = list(loader.samples_in_scene(scene))
    if not samples:
        return []

    ego_by_sample = {s.sample_token: loader.ego_pose_for_sample(s) for s in samples}
    prev_position_by_instance: dict[str, tuple[int, np.ndarray]] = {}

    rows: list[dict] = []
    for sample in samples:
        ego = ego_by_sample[sample.sample_token]
        ego_pos_global = np.array(ego.translation)

        for ann in loader.annotations_in_sample(sample):
            confidence = 1.0  # ground truth
            if confidence < min_confidence:
                continue

            ann_global = np.array(ann.translation)
            ego_frame_xyz = global_to_ego(ann_global, ego)
            forward_x = float(ego_frame_xyz[0])
            left_y = float(ego_frame_xyz[1])
            distance_m = float(np.linalg.norm(ego_frame_xyz))

            rel_vel_ms = _compute_relative_velocity(
                ann=ann,
                ann_global_xy=ann_global[:2],
                ego_pos_global_xy=ego_pos_global[:2],
                ego=ego,
                prev_position_by_instance=prev_position_by_instance,
            )

            rows.append({
                "container_id": ann.container_id,
                "frame_ts": ann.timestamp_us,
                "object_id": ann.object_id,
                "detection_class": ann.detection_class,
                "distance_m": distance_m,
                "lane_offset": lane_offset(left_y),
                "relative_velocity_ms": rel_vel_ms,
                "azimuth": azimuth_sector(forward_x, left_y),
                "confidence": confidence,
                "source": source_from_annotation(
                    ann.num_lidar_pts, ann.num_radar_pts, ann.visibility_token
                ),
            })

    return rows


def _compute_relative_velocity(
    ann: Annotation,
    ann_global_xy: np.ndarray,
    ego_pos_global_xy: np.ndarray,
    ego: EgoPose,
    prev_position_by_instance: dict,
) -> float | None:
    """Compute relative velocity along ego's forward axis (negative = approaching).

    Updates the cache so the next keyframe can compute against this one.
    Returns None for the first observation of an instance.
    """
    prev = prev_position_by_instance.get(ann.instance_token)
    if prev is None:
        prev_position_by_instance[ann.instance_token] = (ann.timestamp_us, ann_global_xy)
        return None

    prev_ts, prev_xy = prev
    dt_s = (ann.timestamp_us - prev_ts) / 1e6
    if dt_s <= 0:
        prev_position_by_instance[ann.instance_token] = (ann.timestamp_us, ann_global_xy)
        return None

    prev_distance = float(np.linalg.norm(prev_xy - ego_pos_global_xy))
    curr_distance = float(np.linalg.norm(ann_global_xy - ego_pos_global_xy))
    rel_vel_ms = (curr_distance - prev_distance) / dt_s

    prev_position_by_instance[ann.instance_token] = (ann.timestamp_us, ann_global_xy)
    return rel_vel_ms


def map_all_scenes(
    loader: NuScenesLoader,
    scenes: Iterable[Scene],
    min_confidence: float = 0.5,
) -> Iterable[dict]:
    for scene in scenes:
        yield from map_scene_to_object_tracks(loader, scene, min_confidence=min_confidence)
