"""Derive per-keyframe `map_context` rows from nuScenes by joining positions
against the HD map-expansion layers.

Two grains, both point-in-time at the 2 Hz keyframe rate (so they share the time
axis with `object_tracks` and `channels`):

- **ego_map_context** — one row per keyframe: the ego vehicle's map situation.
- **object_map_context** — one row per keyframe per annotated object: each object's
  map situation, plus `same_lane_as_ego` (precomputed because the predicate DSL
  correlates entities by time-overlap, not by comparing a column across entities).

The geometry lives in `lib.map_layers`; this module is thin row assembly, mirroring
`object_tracks.py`. nuScenes ego-pose and annotation translations are in the map's
global frame, so positions query the map directly.
"""

from __future__ import annotations

import hashlib

from lib.map_layers import MapLayers

from .loader import NuScenesLoader, Scene

__all__ = [
    "MAP_CONTEXT_VERSION",
    "map_context_lineage",
    "map_scene_to_ego_map_context",
    "map_scene_to_object_map_context",
]

# Derivation version + the definition the hash is taken over. Bump VERSION when the
# derivation logic changes; the hash makes a silent redefinition visible downstream
# (regulatory traceability for derived ADAS products).
MAP_CONTEXT_VERSION = "1.0"
_DEFINITION = (
    "ego{location,on_ped_crossing,on_walkway,on_drivable_area,in_intersection,"
    "dist_to_ped_crossing_m,dist_to_stop_line_m,lane_id};"
    "object{detection_class,on_ped_crossing,on_walkway,in_intersection,"
    "same_lane_as_ego,dist_to_ped_crossing_m,lane_id};"
    "source=nuScenes-map-expansion-v1.3;contains=intersects;dist=nearest"
)


def map_context_lineage(name: str) -> dict[str, str]:
    """Lineage tag for a derived map_context table: (name, version, definition_hash)."""
    h = hashlib.sha256(f"{MAP_CONTEXT_VERSION}:{_DEFINITION}".encode()).hexdigest()[:16]
    return {"derivation_name": name, "derivation_version": MAP_CONTEXT_VERSION, "definition_hash": h}


def _ego_xy(loader: NuScenesLoader, sample) -> tuple[float, float]:
    ego = loader.ego_pose_for_sample(sample)
    return float(ego.translation[0]), float(ego.translation[1])


def map_scene_to_ego_map_context(
    loader: NuScenesLoader, scene: Scene, layers: MapLayers, location: str
) -> list[dict]:
    """One ego_map_context row per keyframe in the scene."""
    rows: list[dict] = []
    for sample in loader.samples_in_scene(scene):
        x, y = _ego_xy(loader, sample)
        rows.append({
            "container_id": scene.container_id,
            "frame_ts": sample.timestamp_us,
            "location": location,
            "on_ped_crossing": int(layers.on_ped_crossing(x, y)),
            "on_walkway": int(layers.on_walkway(x, y)),
            "on_drivable_area": int(layers.on_drivable_area(x, y)),
            "in_intersection": int(layers.in_intersection(x, y)),
            "dist_to_ped_crossing_m": layers.dist_to_ped_crossing_m(x, y),
            "dist_to_stop_line_m": layers.dist_to_stop_line_m(x, y),
            "lane_id": layers.lane_id_at(x, y),
        })
    return rows


def map_scene_to_object_map_context(
    loader: NuScenesLoader, scene: Scene, layers: MapLayers
) -> list[dict]:
    """One object_map_context row per annotation per keyframe.

    `same_lane_as_ego` is 1 when the object and the ego occupy the same map lane at
    that keyframe (both lane ids resolved and equal), else 0.
    """
    rows: list[dict] = []
    for sample in loader.samples_in_scene(scene):
        ego_x, ego_y = _ego_xy(loader, sample)
        ego_lane = layers.lane_id_at(ego_x, ego_y)
        for ann in loader.annotations_in_sample(sample):
            x, y = float(ann.translation[0]), float(ann.translation[1])
            obj_lane = layers.lane_id_at(x, y)
            rows.append({
                "container_id": ann.container_id,
                "frame_ts": ann.timestamp_us,
                "object_id": ann.object_id,
                "detection_class": ann.detection_class,
                "on_ped_crossing": int(layers.on_ped_crossing(x, y)),
                "on_walkway": int(layers.on_walkway(x, y)),
                "in_intersection": int(layers.in_intersection(x, y)),
                "same_lane_as_ego": int(ego_lane is not None and obj_lane == ego_lane),
                "dist_to_ped_crossing_m": layers.dist_to_ped_crossing_m(x, y),
                "lane_id": obj_lane,
            })
    return rows
