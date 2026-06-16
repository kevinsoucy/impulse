"""Unit tests for demos/byod/adapters/nuscenes/scalar_source.py.

Covers the geometry helpers and per-scene derivation logic. Uses a small
FakeLoader stub so we don't need nuscenes-devkit or any dataset on disk.
"""

import math

import numpy as np
import pytest

from adapters.nuscenes.loader import (
    DERIVED_CHANNEL_IDS,
    Annotation,
    EgoPose,
    Sample,
    Scene,
)
from adapters.nuscenes.scalar_source import (
    ChannelValue,
    _rotation_matrix_from_quat,
    derive_all,
    derive_for_scene,
    global_to_ego,
    yaw_from_quat,
)


# ── Geometry helpers ─────────────────────────────────────────────────────────


class TestRotationMatrix:
    def test_identity_quaternion_returns_identity_matrix(self):
        R = _rotation_matrix_from_quat(1.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90deg_yaw_rotates_x_to_y(self):
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        R = _rotation_matrix_from_quat(c, 0.0, 0.0, s)
        np.testing.assert_allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-12)


class TestGlobalToEgo:
    def test_identity_pose_only_translates(self):
        ego = EgoPose(container_id=1, timestamp_us=0, translation=(10.0, 0.0, 0.0), rotation=(1.0, 0.0, 0.0, 0.0))
        ego_frame = global_to_ego(np.array([12.0, 3.0, 0.0]), ego)
        np.testing.assert_allclose(ego_frame, [2.0, 3.0, 0.0], atol=1e-12)

    def test_90deg_rotation_rotates_into_ego_frame(self):
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        ego = EgoPose(container_id=1, timestamp_us=0, translation=(0.0, 0.0, 0.0), rotation=(c, 0.0, 0.0, s))
        ego_frame = global_to_ego(np.array([0.0, 5.0, 0.0]), ego)
        np.testing.assert_allclose(ego_frame, [5.0, 0.0, 0.0], atol=1e-12)


class TestYawFromQuat:
    def test_identity_quaternion_yaw_is_zero(self):
        assert yaw_from_quat(1.0, 0.0, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)

    def test_90deg_yaw_returns_pi_over_two(self):
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        assert yaw_from_quat(c, 0.0, 0.0, s) == pytest.approx(math.pi / 2, abs=1e-12)

    def test_negative_yaw(self):
        c = math.cos(-math.pi / 4)
        s = math.sin(-math.pi / 4)
        assert yaw_from_quat(c, 0.0, 0.0, s) == pytest.approx(-math.pi / 2, abs=1e-12)


# ── ChannelValue dataclass ───────────────────────────────────────────────────


class TestChannelValue:
    def test_frozen(self):
        cv = ChannelValue(container_id=1, channel_id=1001, tstart=0, tend=500_000, value=12.3)
        with pytest.raises(Exception):  # FrozenInstanceError
            cv.value = 99.0  # type: ignore[misc]


# ── Per-scene derivation (with a fake loader) ────────────────────────────────


def _make_scene(container_id: int = 42) -> Scene:
    return Scene(
        container_id=container_id,
        scene_token="tok",
        name="scene",
        description="",
        log_token="",
        nbr_samples=0,
        first_sample_token="",
        last_sample_token="",
    )


def _make_sample(container_id: int, ts: int, name: str) -> Sample:
    return Sample(
        container_id=container_id,
        sample_token=name,
        timestamp_us=ts,
        scene_token="tok",
        sensor_data_tokens={},
    )


def _make_ego(ts: int, x: float, y: float = 0.0) -> EgoPose:
    return EgoPose(
        container_id=42,
        timestamp_us=ts,
        translation=(x, y, 0.0),
        rotation=(1.0, 0.0, 0.0, 0.0),
    )


def _make_annotation(*, container_id: int, ts: int, instance: str, klass: str, gx: float, gy: float) -> Annotation:
    return Annotation(
        container_id=container_id,
        sample_token="s",
        timestamp_us=ts,
        instance_token=instance,
        object_id=hash(instance) & 0x7FFFFFFFFFFFFFFF,
        category_name=f"vehicle.{klass}" if klass == "car" else f"human.{klass}",
        detection_class=klass,
        translation=(gx, gy, 0.0),
        size=(1.7, 4.0, 1.5),
        rotation=(1.0, 0.0, 0.0, 0.0),
        num_lidar_pts=10,
        num_radar_pts=2,
        visibility_token="4",
    )


class FakeLoader:
    """Duck-typed NuScenesLoader exposing only the methods derive_for_scene uses."""

    def __init__(self, samples, ego_by_sample_token, anns_by_sample_token):
        self._samples = samples
        self._ego = ego_by_sample_token
        self._anns = anns_by_sample_token

    def samples_in_scene(self, scene):
        return iter(self._samples)

    def ego_pose_for_sample(self, sample):
        return self._ego[sample.sample_token]

    def annotations_in_sample(self, sample):
        return iter(self._anns.get(sample.sample_token, []))


class TestDeriveForScene:
    def test_returns_empty_when_fewer_than_two_samples(self):
        scene = _make_scene()
        loader = FakeLoader(samples=[_make_sample(42, 0, "a")],
                            ego_by_sample_token={"a": _make_ego(0, 0.0)},
                            anns_by_sample_token={})
        assert derive_for_scene(loader, scene) == []

    def test_speed_kph_matches_ego_translation_delta(self):
        scene = _make_scene()
        s1 = _make_sample(42, 0, "a")
        s2 = _make_sample(42, 500_000, "b")
        loader = FakeLoader(
            samples=[s1, s2],
            ego_by_sample_token={"a": _make_ego(0, 0.0), "b": _make_ego(500_000, 10.0)},
            anns_by_sample_token={},
        )
        rows = derive_for_scene(loader, scene)
        speed_rows = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Vehicle_Speed_kph"]]
        assert len(speed_rows) == 2
        assert speed_rows[0].value == pytest.approx(72.0, rel=1e-9)
        assert speed_rows[0].container_id == 42
        assert speed_rows[0].tstart == 0
        assert speed_rows[0].tend == 500_000

    def test_last_interval_extended_by_median_dt(self):
        s1 = _make_sample(42, 0, "a")
        s2 = _make_sample(42, 500_000, "b")
        loader = FakeLoader(
            samples=[s1, s2],
            ego_by_sample_token={"a": _make_ego(0, 0.0), "b": _make_ego(500_000, 5.0)},
            anns_by_sample_token={},
        )
        rows = derive_for_scene(loader, _make_scene())
        last = [r for r in rows if r.tstart == 500_000][0]
        assert last.tend == 1_000_000

    def test_detection_aggregate_channels_emit_per_class_counts(self):
        scene = _make_scene()
        s1 = _make_sample(42, 0, "a")
        s2 = _make_sample(42, 500_000, "b")
        anns = {
            "a": [
                _make_annotation(container_id=42, ts=0, instance="p1", klass="pedestrian", gx=10.0, gy=0.0),
                _make_annotation(container_id=42, ts=0, instance="p2", klass="pedestrian", gx=20.0, gy=1.0),
                _make_annotation(container_id=42, ts=0, instance="c1", klass="car", gx=15.0, gy=0.0),
            ],
            "b": [],
        }
        loader = FakeLoader(
            samples=[s1, s2],
            ego_by_sample_token={"a": _make_ego(0, 0.0), "b": _make_ego(500_000, 0.0)},
            anns_by_sample_token=anns,
        )
        rows = derive_for_scene(loader, scene)

        ped_count = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Pedestrian_Count"]]
        ped_dist = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Pedestrian_Nearest_Distance_m"]]
        front_cars = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Vehicle_Count_Front"]]

        sample_a_ped_count = [r for r in ped_count if r.tstart == 0]
        assert sample_a_ped_count[0].value == 2.0
        sample_a_ped_dist = [r for r in ped_dist if r.tstart == 0]
        assert sample_a_ped_dist[0].value == pytest.approx(10.0, rel=1e-9)
        sample_a_front_cars = [r for r in front_cars if r.tstart == 0]
        assert sample_a_front_cars[0].value == 1.0

        sample_b_ped_count = [r for r in ped_count if r.tstart == 500_000]
        assert sample_b_ped_count[0].value == 0.0
        sample_b_ped_dist = [r for r in ped_dist if r.tstart == 500_000]
        assert sample_b_ped_dist == []


class TestDeriveAll:
    def test_yields_concatenation_across_scenes(self):
        scene_a = _make_scene(container_id=1)
        scene_b = _make_scene(container_id=2)
        loader = FakeLoader(samples=[], ego_by_sample_token={}, anns_by_sample_token={})
        assert list(derive_all(loader, [scene_a, scene_b])) == []
