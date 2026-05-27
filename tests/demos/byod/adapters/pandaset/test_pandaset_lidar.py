"""Unit tests for demos/byod/adapters/pandaset/lidar.py.

The BL-009 acceptance criterion #1 is verified here: each cuboid produces
TWO `lidar_object_detections` rows — one for `LIDAR_SPINNING` (Pandar64) and
one for `LIDAR_SOLIDSTATE` (PandarGT).
"""

import math

import pytest

from adapters.pandaset.lidar import (
    LIDAR_SENSOR_BY_D,
    LIDAR_SENSOR_SOLIDSTATE,
    LIDAR_SENSOR_SPINNING,
    map_annotation_to_lidar_detections,
    map_scene_for_event_windows,
)
from adapters.pandaset.loader import Annotation, Sample, Scene, _quat_from_yaw


def _ann(*, ts: int = 0, klass: str = "car", uuid: str = "u",
         gx: float = 10.0, gy: float = 0.0, gz: float = 0.0,
         size_lwh=(4.0, 1.8, 1.5), yaw_rad: float = 0.0) -> Annotation:
    return Annotation(
        container_id=1, sample_token="s", timestamp_us=ts,
        object_id=hash(uuid) & 0x7FFFFFFFFFFFFFFF,
        uuid=uuid, category_name=klass, detection_class=klass,
        translation=(gx, gy, gz), size=size_lwh,
        rotation=_quat_from_yaw(yaw_rad), yaw_rad=yaw_rad,
    )


def _sample(ts: int, fid: str) -> Sample:
    return Sample(container_id=1, sample_token=fid, timestamp_us=ts,
                  sequence_id="t", frame_index=0, sensor_files={})


def _scene() -> Scene:
    return Scene(container_id=1, sequence_id="t", name="s", description="",
                 nbr_samples=0, sequence_dir="/tmp", start_ts_us=0)


class FakeLoader:
    def __init__(self, samples, anns):
        self._s = samples
        self._a = anns

    def samples_in_scene(self, scene): return iter(self._s)
    def annotations_in_sample(self, sample): return iter(self._a.get(sample.sample_token, []))


# ── LIDAR_SENSOR_BY_D constants ──────────────────────────────────────────────


class TestSensorConstants:
    def test_spinning_is_d0(self):
        assert LIDAR_SENSOR_BY_D[0] == LIDAR_SENSOR_SPINNING == "LIDAR_SPINNING"

    def test_solid_state_is_d1(self):
        assert LIDAR_SENSOR_BY_D[1] == LIDAR_SENSOR_SOLIDSTATE == "LIDAR_SOLIDSTATE"


# ── map_annotation_to_lidar_detections ───────────────────────────────────────


class TestMapAnnotationToLidarDetections:
    def test_emits_two_rows_per_cuboid(self):
        # BL-009 acceptance criterion #1.
        dets = map_annotation_to_lidar_detections(_ann())
        assert len(dets) == 2
        sensor_ids = {d["sensor_id"] for d in dets}
        assert sensor_ids == {"LIDAR_SPINNING", "LIDAR_SOLIDSTATE"}

    def test_same_geometry_in_both_rows(self):
        ann = _ann(gx=15.0, gy=3.0, gz=0.5, size_lwh=(4.0, 1.8, 1.5))
        dets = map_annotation_to_lidar_detections(ann)
        for det in dets:
            assert det["cx"] == pytest.approx(15.0, abs=1e-9)
            assert det["cy"] == pytest.approx(3.0, abs=1e-9)
            assert det["cz"] == pytest.approx(0.5, abs=1e-9)
            assert det["length"] == 4.0
            assert det["width"] == 1.8
            assert det["height"] == 1.5

    def test_yaw_extracted_from_quat(self):
        ann = _ann(yaw_rad=math.pi / 2)
        dets = map_annotation_to_lidar_detections(ann)
        for det in dets:
            assert det["yaw_rad"] == pytest.approx(math.pi / 2, abs=1e-9)

    def test_object_id_same_across_sensor_rows(self):
        ann = _ann(uuid="obj-X")
        dets = map_annotation_to_lidar_detections(ann)
        assert dets[0]["object_id"] == dets[1]["object_id"]


# ── map_scene_for_event_windows ──────────────────────────────────────────────


class TestMapSceneForEventWindows:
    def test_empty_windows_returns_empty(self):
        loader = FakeLoader(samples=[_sample(0, "a")], anns={"a": [_ann()]})
        assert map_scene_for_event_windows(loader, _scene(), []) == []

    def test_each_annotation_in_window_produces_two_rows(self):
        loader = FakeLoader(
            samples=[_sample(1000, "in")],
            anns={"in": [_ann(ts=1000), _ann(ts=1000, uuid="obj-Y")]},
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(500, 2000)])
        # 2 annotations × 2 sensor rows = 4 rows total.
        assert len(rows) == 4
        sensor_counts = {}
        for r in rows:
            sensor_counts[r["sensor_id"]] = sensor_counts.get(r["sensor_id"], 0) + 1
        assert sensor_counts == {"LIDAR_SPINNING": 2, "LIDAR_SOLIDSTATE": 2}

    def test_out_of_window_samples_excluded(self):
        s_in = _sample(1000, "in")
        s_out = _sample(50_000, "out")
        loader = FakeLoader(
            samples=[s_in, s_out],
            anns={"in": [_ann(ts=1000)], "out": [_ann(ts=50_000)]},
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(500, 2000)])
        # Only s_in's annotation contributes — 1 × 2 sensor rows.
        assert len(rows) == 2
        assert all(r["frame_ts"] == 1000 for r in rows)
