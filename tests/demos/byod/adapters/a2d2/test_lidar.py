"""Unit tests for demos/byod/adapters/a2d2/lidar.py."""

import math

import pytest

from adapters.a2d2.lidar import (
    LIDAR_SENSOR_ID,
    map_annotation_to_lidar_detection,
    map_scene_for_event_windows,
)
from adapters.a2d2.loader import Annotation, Sample, Scene, _axis_angle_to_quat


def _ann(*, ts: int = 0, klass: str = "car", gx: float = 10.0, gy: float = 0.0, gz: float = 0.0,
         size_lwh=(4.0, 1.8, 1.5), rotation=(1.0, 0.0, 0.0, 0.0)) -> Annotation:
    return Annotation(
        container_id=1,
        sample_token="s",
        timestamp_us=ts,
        object_id=42,
        box_key="box_0",
        category_name=klass.capitalize(),
        detection_class=klass,
        translation=(gx, gy, gz),
        size=size_lwh,
        rotation=rotation,
        truncation=0.0,
        occlusion=0.0,
    )


def _sample(ts: int, name: str) -> Sample:
    return Sample(container_id=1, sample_token=name, timestamp_us=ts,
                  scene_id="t", frame_index=0, sensor_files={})


def _scene() -> Scene:
    return Scene(container_id=1, scene_id="t", name="s", description="",
                 nbr_samples=0, scene_dir="/tmp", start_ts_us=0)


class FakeLoader:
    def __init__(self, samples, anns_by_token):
        self._samples = samples
        self._anns = anns_by_token

    def samples_in_scene(self, scene):
        return iter(self._samples)

    def annotations_in_sample(self, sample):
        return iter(self._anns.get(sample.sample_token, []))


# ── map_annotation_to_lidar_detection ────────────────────────────────────────


class TestMapAnnotationToLidarDetection:
    def test_translation_passthrough(self):
        ann = _ann(gx=15.0, gy=3.0, gz=0.5)
        det = map_annotation_to_lidar_detection(ann)
        assert det["cx"] == pytest.approx(15.0, abs=1e-9)
        assert det["cy"] == pytest.approx(3.0, abs=1e-9)
        assert det["cz"] == pytest.approx(0.5, abs=1e-9)

    def test_size_passthrough(self):
        ann = _ann(size_lwh=(4.0, 1.8, 1.5))
        det = map_annotation_to_lidar_detection(ann)
        assert det["length"] == 4.0
        assert det["width"] == 1.8
        assert det["height"] == 1.5

    def test_fixed_fields(self):
        ann = _ann(ts=12345, klass="pedestrian", gx=0.0)
        det = map_annotation_to_lidar_detection(ann)
        assert det["frame_ts"] == 12345
        assert det["detection_class"] == "pedestrian"
        assert det["confidence"] == 1.0
        assert det["sensor_id"] == LIDAR_SENSOR_ID

    def test_yaw_from_quaternion(self):
        # 90° yaw around +Z
        quat = _axis_angle_to_quat([0.0, 0.0, 1.0], math.pi / 2)
        ann = _ann(rotation=quat)
        det = map_annotation_to_lidar_detection(ann)
        assert det["yaw_rad"] == pytest.approx(math.pi / 2, abs=1e-9)

    def test_identity_rotation_yields_zero_yaw(self):
        ann = _ann(rotation=(1.0, 0.0, 0.0, 0.0))
        det = map_annotation_to_lidar_detection(ann)
        assert det["yaw_rad"] == pytest.approx(0.0, abs=1e-12)


# ── map_scene_for_event_windows ──────────────────────────────────────────────


class TestMapSceneForEventWindows:
    def test_empty_windows_returns_empty(self):
        loader = FakeLoader(samples=[_sample(0, "a")], anns_by_token={"a": [_ann(ts=0)]})
        assert map_scene_for_event_windows(loader, _scene(), []) == []

    def test_only_samples_inside_windows_are_emitted(self):
        s_in = _sample(1000, "in")
        s_out = _sample(50_000, "out")
        loader = FakeLoader(
            samples=[s_in, s_out],
            anns_by_token={
                "in":  [_ann(ts=1000)],
                "out": [_ann(ts=50_000)],
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
            anns_by_token={"a": [_ann(ts=500)], "b": [_ann(ts=2000)]},
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(500, 2000)])
        assert {r["frame_ts"] for r in rows} == {500, 2000}

    def test_multiple_windows_independent(self):
        s1 = _sample(100, "a")
        s2 = _sample(5000, "b")
        s3 = _sample(20_000, "c")
        loader = FakeLoader(
            samples=[s1, s2, s3],
            anns_by_token={
                "a": [_ann(ts=100)],
                "b": [_ann(ts=5000)],
                "c": [_ann(ts=20_000)],
            },
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(0, 200), (15_000, 25_000)])
        assert {r["frame_ts"] for r in rows} == {100, 20_000}
