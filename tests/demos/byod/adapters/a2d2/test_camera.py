"""Unit tests for demos/byod/adapters/a2d2/camera.py."""

import numpy as np
import pytest

from adapters.a2d2.camera import (
    _box_corners_vehicle,
    _project_corners_to_image,
    _vehicle_to_sensor,
    map_scene_for_event_windows,
    project_annotation_to_camera,
)
from adapters.a2d2.loader import Annotation, Sample, Scene


# Forward-facing camera at the vehicle origin with the standard pinhole optical
# convention (z forward, x right, y down). Built by hand so the test cases
# isolate the projection math rather than the calibration parsing.
_FORWARD_CAM_CALIB = {
    "intrinsic": [[400.0, 0.0, 400.0], [0.0, 400.0, 300.0], [0.0, 0.0, 1.0]],
    # sensor_rotation: cam→vehicle. For a forward-facing camera with cam-Z = veh-X,
    # cam-X = -veh-Y, cam-Y = -veh-Z, the rotation matrix has columns
    # [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]. Quaternion below realizes that rotation.
    "sensor_rotation": [0.5, -0.5, 0.5, -0.5],
    "sensor_translation": [0.0, 0.0, 0.0],
    "ego_rotation": [1.0, 0.0, 0.0, 0.0],
    "ego_translation": [0.0, 0.0, 0.0],
    "width": 800,
    "height": 600,
    "sensor_id": "cam_front_center",
}


def _ann(ts=1_000_000, gx=15.0, gy=0.0, gz=0.0, size_lwh=(4.0, 1.8, 1.5)) -> Annotation:
    return Annotation(
        container_id=1,
        sample_token="s",
        timestamp_us=ts,
        object_id=42,
        box_key="box_0",
        category_name="Car",
        detection_class="car",
        translation=(gx, gy, gz),
        size=size_lwh,
        rotation=(1.0, 0.0, 0.0, 0.0),
        truncation=0.0,
        occlusion=0.0,
    )


def _scene() -> Scene:
    return Scene(container_id=1, scene_id="t", name="s", description="",
                 nbr_samples=0, scene_dir="/tmp", start_ts_us=0)


def _sample(ts: int, name: str) -> Sample:
    return Sample(container_id=1, sample_token=name, timestamp_us=ts,
                  scene_id="t", frame_index=0, sensor_files={})


# ── _vehicle_to_sensor ──────────────────────────────────────────────────────


class TestVehicleToSensor:
    def test_identity_calibration_passes_point_through(self):
        pt = np.array([5.0, 3.0, 1.0])
        result = _vehicle_to_sensor(
            pt,
            sensor_translation=[0.0, 0.0, 0.0],
            sensor_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result, pt, atol=1e-10)

    def test_sensor_translation_subtracted(self):
        result = _vehicle_to_sensor(
            np.array([10.0, 0.0, 0.0]),
            sensor_translation=[2.0, 0.0, 0.0],
            sensor_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result, [8.0, 0.0, 0.0], atol=1e-10)


# ── _box_corners_vehicle ────────────────────────────────────────────────────


class TestBoxCornersVehicle:
    def test_returns_eight_corners(self):
        corners = _box_corners_vehicle((0.0, 0.0, 0.0), (4.0, 2.0, 1.5), (1.0, 0.0, 0.0, 0.0))
        assert corners.shape == (8, 3)

    def test_axis_aligned_box_spans_correct_extent(self):
        l, w, h = 4.0, 2.0, 1.5
        corners = _box_corners_vehicle((0.0, 0.0, 0.0), (l, w, h), (1.0, 0.0, 0.0, 0.0))
        # x: ±l/2, y: ±w/2, z: ±h/2
        assert corners[:, 0].max() == pytest.approx(l / 2)
        assert corners[:, 0].min() == pytest.approx(-l / 2)
        assert corners[:, 1].max() == pytest.approx(w / 2)
        assert corners[:, 1].min() == pytest.approx(-w / 2)
        assert corners[:, 2].max() == pytest.approx(h / 2)
        assert corners[:, 2].min() == pytest.approx(-h / 2)

    def test_translation_offsets_corners(self):
        corners = _box_corners_vehicle((10.0, 0.0, 0.0), (4.0, 2.0, 1.5), (1.0, 0.0, 0.0, 0.0))
        assert corners[:, 0].min() == pytest.approx(10.0 - 2.0)
        assert corners[:, 0].max() == pytest.approx(10.0 + 2.0)


# ── _project_corners_to_image ────────────────────────────────────────────────


class TestProjectCornersToImage:
    def test_returns_none_when_all_corners_behind(self):
        # All z < 0 → behind the camera.
        corners = np.array([
            [1.0, 1.0, -1.0],
            [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0],
            [-1.0, -1.0, -1.0],
        ])
        result = _project_corners_to_image(corners, _FORWARD_CAM_CALIB["intrinsic"], 800, 600)
        assert result is None

    def test_returns_bbox_when_in_front(self):
        corners = np.array([
            [0.5, 0.5, 5.0],
            [0.5, -0.5, 5.0],
            [-0.5, 0.5, 5.0],
            [-0.5, -0.5, 5.0],
        ])
        bbox = _project_corners_to_image(corners, _FORWARD_CAM_CALIB["intrinsic"], 800, 600)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert x1 < x2 and y1 < y2

    def test_clamps_to_image_bounds(self):
        # A wildly off-axis corner should still clamp inside [0, width-1] / [0, height-1].
        corners = np.array([
            [1000.0, 1000.0, 5.0],
            [-1000.0, -1000.0, 5.0],
            [0.0, 0.0, 5.0],
            [0.5, 0.5, 5.0],
        ])
        bbox = _project_corners_to_image(corners, _FORWARD_CAM_CALIB["intrinsic"], 800, 600)
        assert bbox is not None
        x1, y1, x2, y2 = bbox
        assert 0 <= x1 < 800
        assert 0 <= x2 < 800
        assert 0 <= y1 < 600
        assert 0 <= y2 < 600


# ── project_annotation_to_camera ─────────────────────────────────────────────


class TestProjectAnnotationToCamera:
    def test_forward_box_yields_row(self):
        # Box 15 m forward of vehicle. The forward camera should see it.
        row = project_annotation_to_camera(_ann(gx=15.0, gy=0.0, gz=0.0), _FORWARD_CAM_CALIB)
        assert row is not None
        assert row["sensor_id"] == "cam_front_center"
        assert row["detection_class"] == "car"
        assert row["confidence"] == 1.0
        assert row["x1"] < row["x2"]
        assert row["y1"] < row["y2"]

    def test_behind_box_yields_none(self):
        # Box 15 m *behind* the vehicle — forward camera shouldn't see it.
        row = project_annotation_to_camera(_ann(gx=-15.0, gy=0.0), _FORWARD_CAM_CALIB)
        assert row is None


# ── map_scene_for_event_windows ──────────────────────────────────────────────


class FakeLoader:
    def __init__(self, samples, anns_by_token, calib):
        self._samples = samples
        self._anns = anns_by_token
        self._calib = calib

    def samples_in_scene(self, scene):
        return iter(self._samples)

    def annotations_in_sample(self, sample):
        return iter(self._anns.get(sample.sample_token, []))

    def camera_calib_for_sample(self, sample):
        return {"cam_front_center": self._calib}


class TestMapSceneForEventWindows:
    def test_empty_windows_returns_empty(self):
        loader = FakeLoader(
            samples=[_sample(0, "a")],
            anns_by_token={"a": [_ann(ts=0)]},
            calib=_FORWARD_CAM_CALIB,
        )
        assert map_scene_for_event_windows(loader, _scene(), []) == []

    def test_only_samples_inside_windows_are_emitted(self):
        s_in = _sample(1000, "in")
        s_out = _sample(50_000, "out")
        loader = FakeLoader(
            samples=[s_in, s_out],
            anns_by_token={"in": [_ann(ts=1000)], "out": [_ann(ts=50_000)]},
            calib=_FORWARD_CAM_CALIB,
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(500, 2000)])
        assert len(rows) >= 1
        assert all(r["frame_ts"] == 1000 for r in rows)
