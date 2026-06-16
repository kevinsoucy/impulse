"""Convert A2D2 annotations to ADAS `object_tracks` rows.

Two A2D2-specific simplifications vs. NuScenes:

1. **Vehicle frame already.** A2D2 publishes box centers in the vehicle (ego)
   frame, so the ego→global→ego round-trip is skipped — coordinates go
   straight into `azimuth_sector` and `lane_offset`.

2. **No cross-frame instance tracking.** A2D2 labels each frame independently;
   `box_0` in frame N is not the same physical object as `box_0` in frame N+1.
   `relative_velocity_ms` is therefore `None` for every row — A2D2 just can't
   support that field. Documented in the adapter README so reviewers don't
   look for the gap as a bug.

3. **No radar.** A2D2 has no radar sensor, so `source = "ground_truth_camera_lidar"`
   is constant for every row.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from lib.geometry import LANE_WIDTH_M, azimuth_sector, lane_offset

from .loader import A2D2Loader, Annotation, Scene


__all__ = [
    "A2D2_SOURCE",
    "LANE_WIDTH_M",
    "azimuth_sector",
    "lane_offset",
    "map_all_scenes",
    "map_scene_to_object_tracks",
]


# A2D2 annotations are ground-truth labels created from camera + LiDAR fusion.
# There is no radar in A2D2, so the source string is constant.
A2D2_SOURCE = "ground_truth_camera_lidar"


def map_scene_to_object_tracks(
    loader: A2D2Loader,
    scene: Scene,
    min_confidence: float = 0.5,
) -> list[dict]:
    """Return one object_tracks dict per A2D2 annotation in this scene."""
    samples = list(loader.samples_in_scene(scene))
    if not samples:
        return []

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

            rows.append({
                "container_id": ann.container_id,
                "frame_ts": ann.timestamp_us,
                "object_id": ann.object_id,
                "detection_class": ann.detection_class,
                "distance_m": distance_m,
                "lane_offset": lane_offset(left_y),
                "relative_velocity_ms": None,  # A2D2 has no cross-frame tracking
                "azimuth": azimuth_sector(forward_x, left_y),
                "confidence": confidence,
                "source": A2D2_SOURCE,
            })
    return rows


def map_all_scenes(
    loader: A2D2Loader,
    scenes: Iterable[Scene],
    min_confidence: float = 0.5,
) -> Iterable[dict]:
    for scene in scenes:
        yield from map_scene_to_object_tracks(loader, scene, min_confidence=min_confidence)
