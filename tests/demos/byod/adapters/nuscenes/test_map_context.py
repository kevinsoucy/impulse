"""Unit tests for demos/byod/adapters/nuscenes/map_context.py.

FakeLoader (same pattern as test_object_tracks) + a stub MapLayers, so these test
the row-assembly logic — field population, the int 0/1 encoding, same_lane_as_ego,
and schema conformance — in isolation from the shapely geometry (covered in
test_map_layers).
"""

import pytest

from adapters.nuscenes.loader import Annotation, EgoPose, Sample, Scene
from adapters.nuscenes.map_context import (
    map_context_lineage,
    map_scene_to_ego_map_context,
    map_scene_to_object_map_context,
)
from lib.silver_schema import EGO_MAP_CONTEXT, OBJECT_MAP_CONTEXT


# ── fakes ────────────────────────────────────────────────────────────────────


class FakeLoader:
    def __init__(self, samples, ego_by_token, anns_by_token):
        self._samples = samples
        self._ego = ego_by_token
        self._anns = anns_by_token

    def samples_in_scene(self, scene):
        return iter(self._samples)

    def ego_pose_for_sample(self, sample):
        return self._ego[sample.sample_token]

    def annotations_in_sample(self, sample):
        return iter(self._anns.get(sample.sample_token, []))


class StubMapLayers:
    """Position-keyed stub. lane_by_xy / ped_xy / etc. are dicts/sets of (x, y)."""

    def __init__(self, *, lane_by_xy=None, ped_xy=None, walk_xy=None,
                 drivable_xy=None, intersection_xy=None,
                 ped_dist=None, stop_dist=None):
        self._lane = lane_by_xy or {}
        self._ped = ped_xy or set()
        self._walk = walk_xy or set()
        self._drivable = drivable_xy or set()
        self._intersection = intersection_xy or set()
        self._ped_dist = ped_dist or {}
        self._stop_dist = stop_dist or {}

    def on_ped_crossing(self, x, y):
        return (x, y) in self._ped

    def on_walkway(self, x, y):
        return (x, y) in self._walk

    def on_drivable_area(self, x, y):
        return (x, y) in self._drivable

    def in_intersection(self, x, y):
        return (x, y) in self._intersection

    def lane_id_at(self, x, y):
        return self._lane.get((x, y))

    def dist_to_ped_crossing_m(self, x, y):
        return self._ped_dist.get((x, y))

    def dist_to_stop_line_m(self, x, y):
        return self._stop_dist.get((x, y))


def _scene():
    return Scene(container_id=1, scene_token="t", name="s", description="",
                 log_token="lg", nbr_samples=0, first_sample_token="", last_sample_token="")


def _sample(ts, name):
    return Sample(container_id=1, sample_token=name, timestamp_us=ts, scene_token="t",
                  sensor_data_tokens={})


def _ego(ts, x, y):
    return EgoPose(container_id=1, timestamp_us=ts, translation=(x, y, 0.0),
                   rotation=(1.0, 0.0, 0.0, 0.0))


def _ann(ts, instance, x, y, cls="pedestrian"):
    return Annotation(
        container_id=1, sample_token="a", timestamp_us=ts, instance_token=instance,
        object_id=abs(hash(instance)) & 0x7FFFFFFFFFFFFFFF,
        category_name="human.pedestrian.adult", detection_class=cls,
        translation=(x, y, 0.0), size=(0.7, 0.7, 1.8), rotation=(1.0, 0.0, 0.0, 0.0),
        num_lidar_pts=5, num_radar_pts=0, visibility_token="4",
    )


# ── ego_map_context ───────────────────────────────────────────────────────────


class TestEgoMapContext:
    def test_empty_scene(self):
        loader = FakeLoader(samples=[], ego_by_token={}, anns_by_token={})
        layers = StubMapLayers()
        assert map_scene_to_ego_map_context(loader, _scene(), layers, "town") == []

    def test_one_row_per_keyframe_with_int_flags(self):
        s1, s2 = _sample(0, "a"), _sample(500_000, "b")
        loader = FakeLoader(
            samples=[s1, s2],
            ego_by_token={"a": _ego(0, 1.0, 1.0), "b": _ego(500_000, 50.0, 50.0)},
            anns_by_token={},
        )
        layers = StubMapLayers(
            ped_xy={(1.0, 1.0)}, drivable_xy={(1.0, 1.0), (50.0, 50.0)},
            intersection_xy={(1.0, 1.0)}, lane_by_xy={(50.0, 50.0): "L9"},
            ped_dist={(1.0, 1.0): 0.0, (50.0, 50.0): 12.5},
            stop_dist={(1.0, 1.0): 3.0, (50.0, 50.0): 99.0},
        )
        rows = map_scene_to_ego_map_context(loader, _scene(), layers, "town")
        assert len(rows) == 2
        r0, r1 = rows
        assert (r0["frame_ts"], r0["location"]) == (0, "town")
        assert r0["on_ped_crossing"] == 1 and r0["in_intersection"] == 1
        assert r0["on_drivable_area"] == 1 and r0["on_walkway"] == 0
        assert r0["dist_to_ped_crossing_m"] == 0.0 and r0["lane_id"] is None
        # second frame: off crossing, in a lane
        assert r1["on_ped_crossing"] == 0 and r1["lane_id"] == "L9"
        assert r1["dist_to_ped_crossing_m"] == 12.5

    def test_rows_conform_to_schema(self):
        loader = FakeLoader(samples=[_sample(0, "a")],
                            ego_by_token={"a": _ego(0, 1.0, 1.0)}, anns_by_token={})
        rows = map_scene_to_ego_map_context(loader, _scene(), StubMapLayers(), "town")
        assert set(rows[0].keys()) == {f.name for f in EGO_MAP_CONTEXT.fields}


# ── object_map_context ──────────────────────────────────────────────────────────


class TestObjectMapContext:
    def test_same_lane_as_ego_true_when_lanes_match(self):
        s1 = _sample(0, "a")
        loader = FakeLoader(
            samples=[s1],
            ego_by_token={"a": _ego(0, 0.0, 0.0)},
            anns_by_token={"a": [_ann(0, "ped1", 10.0, 0.0)]},
        )
        # ego at (0,0) → lane "L1"; object at (10,0) → also "L1".
        layers = StubMapLayers(lane_by_xy={(0.0, 0.0): "L1", (10.0, 0.0): "L1"})
        rows = map_scene_to_object_map_context(loader, _scene(), layers)
        assert len(rows) == 1
        assert rows[0]["same_lane_as_ego"] == 1
        assert rows[0]["lane_id"] == "L1"
        assert rows[0]["detection_class"] == "pedestrian"

    def test_same_lane_false_when_lanes_differ(self):
        loader = FakeLoader(
            samples=[_sample(0, "a")],
            ego_by_token={"a": _ego(0, 0.0, 0.0)},
            anns_by_token={"a": [_ann(0, "ped1", 10.0, 0.0)]},
        )
        layers = StubMapLayers(lane_by_xy={(0.0, 0.0): "L1", (10.0, 0.0): "L2"})
        assert map_scene_to_object_map_context(loader, _scene(), layers)[0]["same_lane_as_ego"] == 0

    def test_same_lane_false_when_ego_lane_unknown(self):
        # ego not in any lane → None; even if object resolves a lane, not "same".
        loader = FakeLoader(
            samples=[_sample(0, "a")],
            ego_by_token={"a": _ego(0, 0.0, 0.0)},
            anns_by_token={"a": [_ann(0, "ped1", 10.0, 0.0)]},
        )
        layers = StubMapLayers(lane_by_xy={(10.0, 0.0): "L2"})  # ego (0,0) absent → None
        assert map_scene_to_object_map_context(loader, _scene(), layers)[0]["same_lane_as_ego"] == 0

    def test_both_lanes_none_is_not_same(self):
        loader = FakeLoader(
            samples=[_sample(0, "a")],
            ego_by_token={"a": _ego(0, 0.0, 0.0)},
            anns_by_token={"a": [_ann(0, "ped1", 10.0, 0.0)]},
        )
        layers = StubMapLayers(lane_by_xy={})  # both None
        assert map_scene_to_object_map_context(loader, _scene(), layers)[0]["same_lane_as_ego"] == 0

    def test_rows_conform_to_schema(self):
        loader = FakeLoader(
            samples=[_sample(0, "a")],
            ego_by_token={"a": _ego(0, 0.0, 0.0)},
            anns_by_token={"a": [_ann(0, "ped1", 10.0, 0.0)]},
        )
        rows = map_scene_to_object_map_context(loader, _scene(), StubMapLayers())
        assert set(rows[0].keys()) == {f.name for f in OBJECT_MAP_CONTEXT.fields}

    def test_empty_scene(self):
        loader = FakeLoader(samples=[], ego_by_token={}, anns_by_token={})
        assert map_scene_to_object_map_context(loader, _scene(), StubMapLayers()) == []


# ── lineage ──────────────────────────────────────────────────────────────────


class TestLineage:
    def test_lineage_has_required_keys(self):
        lin = map_context_lineage("ego_map_context")
        assert set(lin) == {"derivation_name", "derivation_version", "definition_hash"}
        assert lin["derivation_name"] == "ego_map_context"

    def test_definition_hash_is_stable(self):
        assert map_context_lineage("x")["definition_hash"] == map_context_lineage("y")["definition_hash"]
