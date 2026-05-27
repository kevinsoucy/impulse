"""Unit tests for demos/byod/adapters/nuscenes/lidar.py."""

import math

import pytest

from adapters.nuscenes.lidar import (
    LIDAR_SENSOR_ID,
    _ego_yaw_from_global_yaw,
    map_annotation_to_lidar_detection,
    map_scene_for_event_windows,
)
from adapters.nuscenes.loader import Annotation, EgoPose, Sample, Scene


# ── _ego_yaw_from_global_yaw ────────────────────────────────────────────────


class TestEgoYawFromGlobalYaw:
    def test_identity_ego_returns_global_yaw(self):
        identity_quat = (1.0, 0.0, 0.0, 0.0)
        assert _ego_yaw_from_global_yaw(math.pi / 4, identity_quat) == pytest.approx(math.pi / 4, abs=1e-12)

    def test_wraps_to_negative_pi_pi(self):
        identity_quat = (1.0, 0.0, 0.0, 0.0)
        rel = _ego_yaw_from_global_yaw(3 * math.pi / 2, identity_quat)
        assert rel == pytest.approx(-math.pi / 2, abs=1e-12)

    def test_wraps_negative_below_minus_pi(self):
        identity_quat = (1.0, 0.0, 0.0, 0.0)
        rel = _ego_yaw_from_global_yaw(-3 * math.pi / 2, identity_quat)
        assert rel == pytest.approx(math.pi / 2, abs=1e-12)

    def test_ego_yaw_subtracted(self):
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        ego_quat = (c, 0.0, 0.0, s)
        assert _ego_yaw_from_global_yaw(math.pi / 2, ego_quat) == pytest.approx(0.0, abs=1e-12)


# ── map_annotation_to_lidar_detection ────────────────────────────────────────


def _ann(*, ts: int, instance: str, gx: float, gy: float, gz: float = 0.0,
         nu_size=(1.7, 4.0, 1.5), rotation=(1.0, 0.0, 0.0, 0.0)) -> Annotation:
    return Annotation(
        container_id=1,
        sample_token="s",
        timestamp_us=ts,
        instance_token=instance,
        object_id=hash(instance) & 0x7FFFFFFFFFFFFFFF,
        category_name="vehicle.car",
        detection_class="car",
        translation=(gx, gy, gz),
        size=nu_size,
        rotation=rotation,
        num_lidar_pts=10,
        num_radar_pts=2,
        visibility_token="4",
    )


def _ego(ts: int, x: float = 0.0, y: float = 0.0, rotation=(1.0, 0.0, 0.0, 0.0)) -> EgoPose:
    return EgoPose(container_id=1, timestamp_us=ts, translation=(x, y, 0.0), rotation=rotation)


class TestMapAnnotationToLidarDetection:
    def test_translation_into_ego_frame(self):
        ann = _ann(ts=0, instance="i1", gx=15.0, gy=3.0)
        det = map_annotation_to_lidar_detection(ann, _ego(0))
        assert det["cx"] == pytest.approx(15.0, abs=1e-9)
        assert det["cy"] == pytest.approx(3.0, abs=1e-9)
        assert det["cz"] == pytest.approx(0.0, abs=1e-9)

    def test_size_swap_from_nuscenes_to_schema(self):
        ann = _ann(ts=0, instance="i1", gx=0.0, gy=0.0, nu_size=(1.7, 4.0, 1.5))
        det = map_annotation_to_lidar_detection(ann, _ego(0))
        assert det["length"] == 4.0
        assert det["width"] == 1.7
        assert det["height"] == 1.5

    def test_fixed_fields(self):
        ann = _ann(ts=12345, instance="i1", gx=0.0, gy=0.0)
        det = map_annotation_to_lidar_detection(ann, _ego(0))
        assert det["container_id"] == 1
        assert det["frame_ts"] == 12345
        assert det["object_id"] == ann.object_id
        assert det["detection_class"] == "car"
        assert det["confidence"] == 1.0
        assert det["sensor_id"] == LIDAR_SENSOR_ID

    def test_yaw_relative_to_ego(self):
        ann = _ann(ts=0, instance="i1", gx=10.0, gy=0.0, rotation=(1.0, 0.0, 0.0, 0.0))
        det = map_annotation_to_lidar_detection(ann, _ego(0))
        assert det["yaw_rad"] == pytest.approx(0.0, abs=1e-12)


# ── map_scene_for_event_windows ──────────────────────────────────────────────


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


class TestMapSceneForEventWindows:
    def test_empty_windows_returns_empty(self):
        loader = FakeLoader(samples=[_sample(0, "a")], ego_by_token={"a": _ego(0)}, anns_by_token={"a": []})
        assert map_scene_for_event_windows(loader, _scene(), []) == []

    def test_only_samples_inside_windows_are_emitted(self):
        s_in = _sample(1000, "in")
        s_out = _sample(50_000, "out")
        loader = FakeLoader(
            samples=[s_in, s_out],
            ego_by_token={"in": _ego(1000), "out": _ego(50_000)},
            anns_by_token={
                "in": [_ann(ts=1000, instance="i1", gx=5.0, gy=0.0)],
                "out": [_ann(ts=50_000, instance="i2", gx=5.0, gy=0.0)],
            },
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(500, 2000)])
        assert len(rows) == 1
        assert rows[0]["frame_ts"] == 1000

    def test_inclusive_window_boundaries(self):
        s_start = _sample(500, "a")
        s_end = _sample(2000, "b")
        loader = FakeLoader(
            samples=[s_start, s_end],
            ego_by_token={"a": _ego(500), "b": _ego(2000)},
            anns_by_token={
                "a": [_ann(ts=500, instance="i1", gx=5.0, gy=0.0)],
                "b": [_ann(ts=2000, instance="i2", gx=5.0, gy=0.0)],
            },
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(500, 2000)])
        assert {r["frame_ts"] for r in rows} == {500, 2000}

    def test_multiple_windows_independent(self):
        s1 = _sample(100, "a")
        s2 = _sample(5000, "b")
        s3 = _sample(20_000, "c")
        loader = FakeLoader(
            samples=[s1, s2, s3],
            ego_by_token={"a": _ego(100), "b": _ego(5000), "c": _ego(20_000)},
            anns_by_token={
                "a": [_ann(ts=100, instance="x", gx=1.0, gy=0.0)],
                "b": [_ann(ts=5000, instance="y", gx=1.0, gy=0.0)],
                "c": [_ann(ts=20_000, instance="z", gx=1.0, gy=0.0)],
            },
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(0, 200), (15_000, 25_000)])
        assert {r["frame_ts"] for r in rows} == {100, 20_000}
