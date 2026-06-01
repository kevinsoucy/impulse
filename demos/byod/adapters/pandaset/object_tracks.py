"""Convert PandaSet annotations to LakeVision `object_tracks` rows.

Closest precedent: the ZOD adapter (vehicle-frame boxes + persistent uuid).
Key differences:

  - `source = "lidar|camera"` — PandaSet has no radar sensor, so the radar
    modality never appears (vs. ZOD where it does).
  - `relative_velocity_ms` IS computed from cross-frame `uuid` tracking.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lib.geometry import LANE_WIDTH_M, azimuth_sector, lane_offset

from .loader import Annotation, PandaSetLoader, Scene


__all__ = [
    "LANE_WIDTH_M",
    "PANDASET_SOURCE",
    "azimuth_sector",
    "lane_offset",
    "map_all_scenes",
    "map_scene_to_object_tracks",
]


# PandaSet has no radar; cuboid labels were created from LiDAR + camera fusion.
PANDASET_SOURCE = "lidar|camera"


def map_scene_to_object_tracks(
    loader: PandaSetLoader,
    scene: Scene,
    min_confidence: float = 0.5,
) -> list[dict]:
    samples = list(loader.samples_in_scene(scene))
    if not samples:
        return []

    prev_position_by_uuid: dict[str, tuple[int, np.ndarray]] = {}

    rows: list[dict] = []
    for sample in samples:
        for ann in loader.annotations_in_sample(sample):
            confidence = 1.0
            if confidence < min_confidence:
                continue

            ego_xyz = np.asarray(ann.translation, dtype=float)
            forward_x = float(ego_xyz[0])
            left_y = float(ego_xyz[1])
            distance_m = float(np.linalg.norm(ego_xyz))

            rel_vel_ms = _compute_relative_velocity(
                uuid=ann.uuid,
                ego_xy=ego_xyz[:2],
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
                "source": PANDASET_SOURCE,
            })
    return rows


def _compute_relative_velocity(
    uuid: str,
    ego_xy: np.ndarray,
    timestamp_us: int,
    cache: dict[str, tuple[int, np.ndarray]],
) -> float | None:
    prev = cache.get(uuid)
    if prev is None:
        cache[uuid] = (timestamp_us, ego_xy.copy())
        return None
    prev_ts, prev_xy = prev
    dt_s = (timestamp_us - prev_ts) / 1e6
    if dt_s <= 0:
        cache[uuid] = (timestamp_us, ego_xy.copy())
        return None
    rel_vel = (float(np.linalg.norm(ego_xy)) - float(np.linalg.norm(prev_xy))) / dt_s
    cache[uuid] = (timestamp_us, ego_xy.copy())
    return rel_vel


def map_all_scenes(
    loader: PandaSetLoader,
    scenes: Iterable[Scene],
    min_confidence: float = 0.5,
) -> Iterable[dict]:
    for scene in scenes:
        yield from map_scene_to_object_tracks(loader, scene, min_confidence=min_confidence)
