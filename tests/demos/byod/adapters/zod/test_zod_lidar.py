"""Unit tests for demos/byod/adapters/zod/lidar.py."""

import math

import pytest

from adapters.zod.lidar import (
    LIDAR_SENSOR_ID,
    map_annotation_to_lidar_detection,
    map_scene_for_event_windows,
)
from adapters.zod.loader import Annotation, Sample, Scene


def _ann(*, ts: int = 0, klass: str = "car", uuid: str = "u",
         gx: float = 10.0, gy: float = 0.0, gz: float = 0.0,
         size_lwh=(4.0, 1.8, 1.5), rotation=(1.0, 0.0, 0.0, 0.0)) -> Annotation:
    return Annotation(
        container_id=1, sample_token="s", timestamp_us=ts,
        object_id=hash(uuid) & 0x7FFFFFFFFFFFFFFF, uuid=uuid,
        category_name=klass, detection_class=klass,
        translation=(gx, gy, gz), size=size_lwh, rotation=rotation,
        num_lidar_pts=10, num_radar_pts=2, occlusion=0.1, radar_doppler_ms=None,
    )


def _sample(ts: int, fid: str) -> Sample:
    return Sample(container_id=1, sample_token=fid, timestamp_us=ts,
                  sequence_id="t", frame_id=fid, sensor_files={})


def _scene() -> Scene:
    return Scene(container_id=1, sequence_id="t", name="s", description="",
                 nbr_samples=0, sequence_dir="/tmp", start_ts_us=0)


class FakeLoader:
    def __init__(self, samples, anns):
        self._s = samples
        self._a = anns

    def samples_in_scene(self, scene): return iter(self._s)
    def annotations_in_sample(self, sample): return iter(self._a.get(sample.sample_token, []))


# ── map_annotation_to_lidar_detection ────────────────────────────────────────


class TestMapAnnotationToLidarDetection:
    def test_translation_passthrough(self):
        det = map_annotation_to_lidar_detection(_ann(gx=15.0, gy=3.0, gz=0.5))
        assert det["cx"] == pytest.approx(15.0, abs=1e-9)
        assert det["cy"] == pytest.approx(3.0, abs=1e-9)
        assert det["cz"] == pytest.approx(0.5, abs=1e-9)

    def test_size_passthrough(self):
        det = map_annotation_to_lidar_detection(_ann(size_lwh=(4.0, 1.8, 1.5)))
        assert det["length"] == 4.0
        assert det["width"] == 1.8
        assert det["height"] == 1.5

    def test_sensor_id_is_lidar_fused(self):
        det = map_annotation_to_lidar_detection(_ann())
        assert det["sensor_id"] == LIDAR_SENSOR_ID == "LIDAR_FUSED"

    def test_identity_rotation_yields_zero_yaw(self):
        det = map_annotation_to_lidar_detection(_ann(rotation=(1.0, 0.0, 0.0, 0.0)))
        assert det["yaw_rad"] == pytest.approx(0.0, abs=1e-12)

    def test_90deg_yaw_quat(self):
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        det = map_annotation_to_lidar_detection(_ann(rotation=(c, 0.0, 0.0, s)))
        assert det["yaw_rad"] == pytest.approx(math.pi / 2, abs=1e-9)


# ── map_scene_for_event_windows ──────────────────────────────────────────────


class TestMapSceneForEventWindows:
    def test_empty_windows_returns_empty(self):
        loader = FakeLoader(samples=[_sample(0, "a")], anns={"a": [_ann()]})
        assert map_scene_for_event_windows(loader, _scene(), []) == []

    def test_only_samples_inside_windows_are_emitted(self):
        s_in = _sample(1000, "in")
        s_out = _sample(50_000, "out")
        loader = FakeLoader(
            samples=[s_in, s_out],
            anns={
                "in":  [_ann(ts=1000)],
                "out": [_ann(ts=50_000)],
            },
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(500, 2000)])
        assert len(rows) == 1
        assert rows[0]["frame_ts"] == 1000

    def test_inclusive_window_boundaries(self):
        loader = FakeLoader(
            samples=[_sample(500, "a"), _sample(2000, "b")],
            anns={"a": [_ann(ts=500)], "b": [_ann(ts=2000)]},
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(500, 2000)])
        assert {r["frame_ts"] for r in rows} == {500, 2000}
