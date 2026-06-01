"""Convert ZOD annotations to LakeVision `object_tracks` rows.

ZOD-specific properties vs A2D2:
  - Annotation `uuid` is stable across frames → cross-frame instance tracking
    works, so `relative_velocity_ms` IS computed (not always None).
  - Each annotation has separate `lidar_attribute.num_points` and
    `radar_attribute.num_points` plus an inferred camera coverage flag, so
    `source` becomes the pipe-delimited string "lidar|radar|camera" that
    notebook 04's `distance_stats_by_modality` partitions on. **This is the
    radar differentiator** — ZOD is the only adapter where the radar modality
    populates non-empty in notebook 04's KPI tables.

Same as A2D2: annotations are in the vehicle frame, so no `global_to_ego`.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lib.geometry import LANE_WIDTH_M, azimuth_sector, lane_offset

from .loader import Annotation, Scene, ZodLoader


__all__ = [
    "LANE_WIDTH_M",
    "azimuth_sector",
    "lane_offset",
    "map_all_scenes",
    "map_scene_to_object_tracks",
    "source_from_annotation",
]


def source_from_annotation(num_lidar_pts: int, num_radar_pts: int, occlusion: float) -> str:
    """ZOD-specific sensor coverage encoding.

    LiDAR: `num_lidar_pts > 0` → "lidar"
    Radar: `num_radar_pts > 0` → "radar"  (this is ZOD's distinguishing feature)
    Camera: occlusion < 0.8 → "camera" (heavily-occluded objects excluded from camera modality)
    Fallback: if all flags are off, return "camera" (annotator saw the object somehow).
    """
    parts: list[str] = []
    if num_lidar_pts > 0:
        parts.append("lidar")
    if num_radar_pts > 0:
        parts.append("radar")
    if occlusion < 0.8:
        parts.append("camera")
    return "|".join(parts) if parts else "camera"


def map_scene_to_object_tracks(
    loader: ZodLoader,
    scene: Scene,
    min_confidence: float = 0.5,
) -> list[dict]:
    """Return one object_tracks dict per ZOD annotation in this scene.

    `relative_velocity_ms` is computed across consecutive frames per `uuid`,
    using the same approach as the NuScenes adapter.
    """
    samples = list(loader.samples_in_scene(scene))
    if not samples:
        return []

    prev_position_by_uuid: dict[str, tuple[int, np.ndarray]] = {}

    rows: list[dict] = []
    for sample in samples:
        for ann in loader.annotations_in_sample(sample):
            confidence = 1.0  # ground truth
            if confidence < min_confidence:
                continue

            ego_frame_xyz = np.asarray(ann.translation, dtype=float)
            forward_x = float(ego_frame_xyz[0])
            left_y = float(ego_frame_xyz[1])
            distance_m = float(np.linalg.norm(ego_frame_xyz))

            rel_vel_ms = _compute_relative_velocity(
                uuid=ann.uuid,
                ego_frame_xy=ego_frame_xyz[:2],
                timestamp_us=ann.timestamp_us,
                cache=prev_position_by_uuid,
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
                    ann.num_lidar_pts, ann.num_radar_pts, ann.occlusion
                ),
            })
    return rows


def _compute_relative_velocity(
    uuid: str,
    ego_frame_xy: np.ndarray,
    timestamp_us: int,
    cache: dict[str, tuple[int, np.ndarray]],
) -> float | None:
    """Relative-velocity magnitude (signed by approaching/receding) across two
    consecutive observations of the same instance.

    Computed in the vehicle frame: positive Δ‖position‖/Δt = receding.
    Returns None on the first observation of an instance or zero Δt.
    """
    prev = cache.get(uuid)
    if prev is None:
        cache[uuid] = (timestamp_us, ego_frame_xy.copy())
        return None
    prev_ts, prev_xy = prev
    dt_s = (timestamp_us - prev_ts) / 1e6
    if dt_s <= 0:
        cache[uuid] = (timestamp_us, ego_frame_xy.copy())
        return None
    prev_dist = float(np.linalg.norm(prev_xy))
    curr_dist = float(np.linalg.norm(ego_frame_xy))
    rel_vel = (curr_dist - prev_dist) / dt_s
    cache[uuid] = (timestamp_us, ego_frame_xy.copy())
    return rel_vel


def map_all_scenes(
    loader: ZodLoader,
    scenes: Iterable[Scene],
    min_confidence: float = 0.5,
) -> Iterable[dict]:
    for scene in scenes:
        yield from map_scene_to_object_tracks(loader, scene, min_confidence=min_confidence)
