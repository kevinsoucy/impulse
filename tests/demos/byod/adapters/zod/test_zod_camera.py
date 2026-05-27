"""Unit tests for demos/byod/adapters/zod/camera.py."""

import numpy as np
import pytest

from adapters.zod.camera import (
    _box_corners_vehicle,
    _project_corners_to_image,
    _vehicle_to_sensor,
    map_scene_for_event_windows,
    project_annotation_to_camera,
)
from adapters.zod.loader import Annotation, Sample, Scene


# Forward-facing camera at vehicle origin; quaternion = the same A2D2-test
# rotation (cam-Z = veh-X, cam-X = -veh-Y, cam-Y = -veh-Z).
_FORWARD_CAM_CALIB = {
    "intrinsic": [[2000.0, 0.0, 1924.0], [0.0, 2000.0, 1084.0], [0.0, 0.0, 1.0]],
    "sensor_rotation": [0.5, -0.5, 0.5, -0.5],
    "sensor_translation": [0.0, 0.0, 0.0],
    "ego_rotation": [1.0, 0.0, 0.0, 0.0],
    "ego_translation": [0.0, 0.0, 0.0],
    "width": 3848,
    "height": 2168,
    "sensor_id": "camera_front_blur",
}


def _ann(ts=1_000_000, gx=20.0, gy=0.0, gz=0.0) -> Annotation:
    return Annotation(
        container_id=1, sample_token="s", timestamp_us=ts,
        object_id=42, uuid="u", category_name="Vehicle/Car",
        detection_class="car",
        translation=(gx, gy, gz), size=(4.0, 1.8, 1.5),
        rotation=(1.0, 0.0, 0.0, 0.0),
        num_lidar_pts=10, num_radar_pts=2, occlusion=0.1, radar_doppler_ms=-1.5,
    )


def _sample(ts: int, fid: str) -> Sample:
    return Sample(container_id=1, sample_token=fid, timestamp_us=ts,
                  sequence_id="t", frame_id=fid, sensor_files={})


def _scene() -> Scene:
    return Scene(container_id=1, sequence_id="t", name="s", description="",
                 nbr_samples=0, sequence_dir="/tmp", start_ts_us=0)


# ── _vehicle_to_sensor ──────────────────────────────────────────────────────


class TestVehicleToSensor:
    def test_identity_calibration_passes_through(self):
        pt = np.array([5.0, 3.0, 1.0])
        np.testing.assert_allclose(
            _vehicle_to_sensor(pt, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            pt, atol=1e-10,
        )

    def test_translation_subtracted(self):
        result = _vehicle_to_sensor(
            np.array([10.0, 0.0, 0.0]), [2.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result, [8.0, 0.0, 0.0], atol=1e-10)


# ── _box_corners_vehicle ────────────────────────────────────────────────────


class TestBoxCornersVehicle:
    def test_returns_eight_corners(self):
        corners = _box_corners_vehicle((0.0, 0.0, 0.0), (4.0, 1.8, 1.5), (1.0, 0.0, 0.0, 0.0))
        assert corners.shape == (8, 3)

    def test_lwh_order(self):
        l, w, h = 4.0, 1.8, 1.5
        corners = _box_corners_vehicle((0.0, 0.0, 0.0), (l, w, h), (1.0, 0.0, 0.0, 0.0))
        assert corners[:, 0].max() == pytest.approx(l / 2)
        assert corners[:, 1].max() == pytest.approx(w / 2)
        assert corners[:, 2].max() == pytest.approx(h / 2)


# ── _project_corners_to_image ────────────────────────────────────────────────


class TestProjectCornersToImage:
    def test_returns_none_when_all_corners_behind(self):
        corners = np.array([
            [1.0, 1.0, -1.0], [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0],
        ])
        assert _project_corners_to_image(corners, _FORWARD_CAM_CALIB["intrinsic"], 3848, 2168) is None

    def test_clamps_to_image_bounds(self):
        corners = np.array([
            [1000.0, 1000.0, 10.0], [-1000.0, -1000.0, 10.0],
            [0.0, 0.0, 10.0], [0.5, 0.5, 10.0],
        ])
        bbox = _project_corners_to_image(corners, _FORWARD_CAM_CALIB["intrinsic"], 3848, 2168)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert 0 <= x1 < x2 < 3848
        assert 0 <= y1 < y2 < 2168


# ── project_annotation_to_camera ─────────────────────────────────────────────


class TestProjectAnnotationToCamera:
    def test_forward_box_yields_row(self):
        row = project_annotation_to_camera(_ann(gx=20.0), _FORWARD_CAM_CALIB)
        assert row is not None
        assert row["sensor_id"] == "camera_front_blur"
        assert row["detection_class"] == "car"
        assert row["x1"] < row["x2"]
        assert row["y1"] < row["y2"]

    def test_behind_box_yields_none(self):
        assert project_annotation_to_camera(_ann(gx=-20.0), _FORWARD_CAM_CALIB) is None


# ── map_scene_for_event_windows ──────────────────────────────────────────────


class FakeLoader:
    def __init__(self, samples, anns, calib):
        self._s = samples
        self._a = anns
        self._c = calib

    def samples_in_scene(self, scene): return iter(self._s)
    def annotations_in_sample(self, sample): return iter(self._a.get(sample.sample_token, []))
    def camera_calib_for_sample(self, sample): return {"camera_front_blur": self._c}


class TestMapSceneForEventWindows:
    def test_empty_windows_returns_empty(self):
        loader = FakeLoader(samples=[_sample(0, "a")], anns={"a": [_ann(ts=0)]}, calib=_FORWARD_CAM_CALIB)
        assert map_scene_for_event_windows(loader, _scene(), []) == []

    def test_only_samples_inside_windows_emitted(self):
        s_in = _sample(1000, "in")
        s_out = _sample(50_000, "out")
        loader = FakeLoader(
            samples=[s_in, s_out],
            anns={"in": [_ann(ts=1000)], "out": [_ann(ts=50_000)]},
            calib=_FORWARD_CAM_CALIB,
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(500, 2000)])
        assert len(rows) >= 1
        assert all(r["frame_ts"] == 1000 for r in rows)
