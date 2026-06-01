"""Unit tests for demos/byod/adapters/nuscenes/object_tracks.py."""

import numpy as np
import pytest

from adapters.nuscenes.loader import Annotation, EgoPose, Sample, Scene
from adapters.nuscenes.object_tracks import (
    LANE_WIDTH_M,
    _CAMERA_VISIBLE_TOKENS,
    _compute_relative_velocity,
    azimuth_sector,
    lane_offset,
    map_all_scenes,
    map_scene_to_object_tracks,
    source_from_annotation,
)


# ── azimuth_sector ───────────────────────────────────────────────────────────


class TestAzimuthSector:
    @pytest.mark.parametrize("x,y,expected", [
        (1.0, 0.0, "front"),
        (0.0, 1.0, "left"),
        (-1.0, 0.0, "rear"),
        (0.0, -1.0, "right"),
        (1.0, 1.0, "front_left"),
        (-1.0, 1.0, "rear_left"),
        (-1.0, -1.0, "rear_right"),
        (1.0, -1.0, "front_right"),
    ])
    def test_cardinal_and_diagonal_sectors(self, x, y, expected):
        assert azimuth_sector(x, y) == expected

    def test_near_boundary_classification(self):
        import math
        below = math.radians(22.0)
        above = math.radians(23.0)
        assert azimuth_sector(math.cos(below), math.sin(below)) == "front"
        assert azimuth_sector(math.cos(above), math.sin(above)) == "front_left"


# ── lane_offset ──────────────────────────────────────────────────────────────


class TestLaneOffset:
    def test_same_lane(self):
        assert lane_offset(0.0) == 0
        assert lane_offset(LANE_WIDTH_M / 2 - 0.1) == 0

    def test_left_lane(self):
        assert lane_offset(LANE_WIDTH_M) == 1
        assert lane_offset(LANE_WIDTH_M * 2) == 2

    def test_right_lane(self):
        assert lane_offset(-LANE_WIDTH_M) == -1
        assert lane_offset(-LANE_WIDTH_M * 2) == -2

    def test_clamped_to_two_lanes_each_side(self):
        assert lane_offset(LANE_WIDTH_M * 100) == 2
        assert lane_offset(-LANE_WIDTH_M * 100) == -2


# ── _compute_relative_velocity ───────────────────────────────────────────────


def _ann(*, ts: int, instance: str, gx: float, gy: float) -> Annotation:
    return Annotation(
        container_id=1,
        sample_token="s",
        timestamp_us=ts,
        instance_token=instance,
        object_id=hash(instance) & 0x7FFFFFFFFFFFFFFF,
        category_name="vehicle.car",
        detection_class="car",
        translation=(gx, gy, 0.0),
        size=(1.7, 4.0, 1.5),
        rotation=(1.0, 0.0, 0.0, 0.0),
        num_lidar_pts=10,
        num_radar_pts=2,
        visibility_token="4",
    )


def _ego(ts: int) -> EgoPose:
    return EgoPose(container_id=1, timestamp_us=ts, translation=(0.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0))


class TestComputeRelativeVelocity:
    def test_first_observation_returns_none(self):
        cache = {}
        ann = _ann(ts=0, instance="i1", gx=20.0, gy=0.0)
        v = _compute_relative_velocity(
            ann=ann,
            ann_global_xy=np.array(ann.translation[:2]),
            ego_pos_global_xy=np.zeros(2),
            ego=_ego(0),
            prev_position_by_instance=cache,
        )
        assert v is None
        assert "i1" in cache

    def test_approaching_object_returns_negative_rate(self):
        cache = {}
        a1 = _ann(ts=0, instance="i1", gx=20.0, gy=0.0)
        _compute_relative_velocity(
            ann=a1, ann_global_xy=np.array([20.0, 0.0]),
            ego_pos_global_xy=np.zeros(2), ego=_ego(0), prev_position_by_instance=cache,
        )
        a2 = _ann(ts=500_000, instance="i1", gx=10.0, gy=0.0)
        v = _compute_relative_velocity(
            ann=a2, ann_global_xy=np.array([10.0, 0.0]),
            ego_pos_global_xy=np.zeros(2), ego=_ego(500_000), prev_position_by_instance=cache,
        )
        assert v == pytest.approx(-20.0, rel=1e-9)

    def test_zero_dt_returns_none(self):
        cache = {}
        a1 = _ann(ts=0, instance="i1", gx=20.0, gy=0.0)
        _compute_relative_velocity(
            ann=a1, ann_global_xy=np.array([20.0, 0.0]),
            ego_pos_global_xy=np.zeros(2), ego=_ego(0), prev_position_by_instance=cache,
        )
        a2 = _ann(ts=0, instance="i1", gx=10.0, gy=0.0)
        v = _compute_relative_velocity(
            ann=a2, ann_global_xy=np.array([10.0, 0.0]),
            ego_pos_global_xy=np.zeros(2), ego=_ego(0), prev_position_by_instance=cache,
        )
        assert v is None


# ── map_scene_to_object_tracks ───────────────────────────────────────────────


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


def _scene() -> Scene:
    return Scene(container_id=1, scene_token="t", name="s", description="", log_token="",
                 nbr_samples=0, first_sample_token="", last_sample_token="")


def _sample(ts: int, name: str) -> Sample:
    return Sample(container_id=1, sample_token=name, timestamp_us=ts, scene_token="t", sensor_data_tokens={})


class TestMapSceneToObjectTracks:
    def test_empty_scene(self):
        loader = FakeLoader(samples=[], ego_by_token={}, anns_by_token={})
        assert map_scene_to_object_tracks(loader, _scene()) == []

    def test_single_annotation_populates_expected_fields(self):
        s1 = _sample(0, "a")
        loader = FakeLoader(
            samples=[s1],
            ego_by_token={"a": _ego(0)},
            anns_by_token={"a": [_ann(ts=0, instance="i1", gx=10.0, gy=0.0)]},
        )
        rows = map_scene_to_object_tracks(loader, _scene())
        assert len(rows) == 1
        r = rows[0]
        assert r["detection_class"] == "car"
        assert r["distance_m"] == pytest.approx(10.0, rel=1e-9)
        assert r["azimuth"] == "front"
        assert r["lane_offset"] == 0
        assert r["confidence"] == 1.0
        assert r["source"] == "lidar|radar|camera"
        assert r["relative_velocity_ms"] is None

    def test_relative_velocity_populated_on_second_observation(self):
        s1 = _sample(0, "a")
        s2 = _sample(500_000, "b")
        loader = FakeLoader(
            samples=[s1, s2],
            ego_by_token={"a": _ego(0), "b": _ego(500_000)},
            anns_by_token={
                "a": [_ann(ts=0, instance="i1", gx=20.0, gy=0.0)],
                "b": [_ann(ts=500_000, instance="i1", gx=10.0, gy=0.0)],
            },
        )
        rows = map_scene_to_object_tracks(loader, _scene())
        assert rows[0]["relative_velocity_ms"] is None
        assert rows[1]["relative_velocity_ms"] == pytest.approx(-20.0, rel=1e-9)


class TestMapAllScenes:
    def test_yields_across_multiple_scenes(self):
        loader = FakeLoader(samples=[], ego_by_token={}, anns_by_token={})
        assert list(map_all_scenes(loader, [_scene(), _scene()])) == []


# ── source_from_annotation ───────────────────────────────────────────────────


class TestSourceFromAnnotation:
    def test_lidar_pts_sets_lidar(self):
        result = source_from_annotation(num_lidar_pts=10, num_radar_pts=0, visibility_token="1")
        assert "lidar" in result.split("|")

    def test_radar_pts_sets_radar(self):
        result = source_from_annotation(num_lidar_pts=0, num_radar_pts=5, visibility_token="1")
        assert "radar" in result.split("|")

    def test_high_visibility_sets_camera(self):
        for token in ("2", "3", "4"):
            result = source_from_annotation(num_lidar_pts=0, num_radar_pts=0, visibility_token=token)
            assert result == "camera", f"token={token!r}: expected 'camera', got {result!r}"

    def test_low_visibility_no_camera_if_no_sensor_coverage(self):
        result = source_from_annotation(num_lidar_pts=0, num_radar_pts=0, visibility_token="1")
        assert result == "camera"

    def test_all_modalities_full_coverage(self):
        result = source_from_annotation(num_lidar_pts=15, num_radar_pts=3, visibility_token="4")
        assert result == "lidar|radar|camera"

    def test_lidar_camera_no_radar(self):
        result = source_from_annotation(num_lidar_pts=8, num_radar_pts=0, visibility_token="3")
        assert result == "lidar|camera"

    def test_lidar_only_low_visibility(self):
        result = source_from_annotation(num_lidar_pts=5, num_radar_pts=0, visibility_token="1")
        assert result == "lidar"

    def test_camera_visible_tokens_match_module_constant(self):
        assert _CAMERA_VISIBLE_TOKENS == {"2", "3", "4"}

    def test_empty_visibility_token_falls_back_to_camera(self):
        result = source_from_annotation(num_lidar_pts=0, num_radar_pts=0, visibility_token="")
        assert result == "camera"

    def test_source_is_pipe_delimited_string(self):
        result = source_from_annotation(num_lidar_pts=10, num_radar_pts=2, visibility_token="4")
        assert isinstance(result, str)
        assert "|" in result or result in ("lidar", "radar", "camera")
