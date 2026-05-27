"""Unit tests for demos/byod/adapters/pandaset/camera.py."""

import numpy as np
import pytest

from adapters.pandaset.camera import (
    _box_corners_vehicle,
    _project_corners_to_image,
    _vehicle_to_sensor,
    map_scene_for_event_windows,
    project_annotation_to_camera,
)
from adapters.pandaset.loader import Annotation, Sample, Scene


# Forward-facing camera quaternion (same as A2D2/ZOD tests): cam-Z = veh-X.
_FORWARD_CAM_CALIB = {
    "intrinsic": [[1900.0, 0.0, 960.0], [0.0, 1900.0, 540.0], [0.0, 0.0, 1.0]],
    "sensor_rotation": [0.5, -0.5, 0.5, -0.5],
    "sensor_translation": [0.0, 0.0, 0.0],
    "ego_rotation": [1.0, 0.0, 0.0, 0.0],
    "ego_translation": [0.0, 0.0, 0.0],
    "width": 1920,
    "height": 1080,
    "sensor_id": "front_camera",
}


def _ann(ts=1_000_000, gx=20.0, gy=0.0, gz=0.0) -> Annotation:
    return Annotation(
        container_id=1, sample_token="s", timestamp_us=ts,
        object_id=42, uuid="u", category_name="Car",
        detection_class="car",
        translation=(gx, gy, gz), size=(4.5, 1.8, 1.6),
        rotation=(1.0, 0.0, 0.0, 0.0), yaw_rad=0.0,
    )


def _sample(ts: int, fid: str) -> Sample:
    return Sample(container_id=1, sample_token=fid, timestamp_us=ts,
                  sequence_id="t", frame_index=0, sensor_files={})


def _scene() -> Scene:
    return Scene(container_id=1, sequence_id="t", name="s", description="",
                 nbr_samples=0, sequence_dir="/tmp", start_ts_us=0)


# ── _vehicle_to_sensor ──────────────────────────────────────────────────────


class TestVehicleToSensor:
    def test_identity_passes_through(self):
        pt = np.array([5.0, 3.0, 1.0])
        np.testing.assert_allclose(
            _vehicle_to_sensor(pt, [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]),
            pt, atol=1e-10,
        )


# ── _box_corners_vehicle ────────────────────────────────────────────────────


class TestBoxCornersVehicle:
    def test_returns_eight_corners(self):
        corners = _box_corners_vehicle((0.0, 0.0, 0.0), (4.0, 1.8, 1.5), (1.0, 0.0, 0.0, 0.0))
        assert corners.shape == (8, 3)


# ── _project_corners_to_image ────────────────────────────────────────────────


class TestProjectCornersToImage:
    def test_returns_none_when_all_corners_behind(self):
        corners = np.array([
            [1.0, 1.0, -1.0], [1.0, -1.0, -1.0],
            [-1.0, 1.0, -1.0], [-1.0, -1.0, -1.0],
        ])
        assert _project_corners_to_image(corners, _FORWARD_CAM_CALIB["intrinsic"], 1920, 1080) is None


# ── project_annotation_to_camera ─────────────────────────────────────────────


class TestProjectAnnotationToCamera:
    def test_forward_box_yields_row(self):
        row = project_annotation_to_camera(_ann(gx=20.0), _FORWARD_CAM_CALIB)
        assert row is not None
        assert row["sensor_id"] == "front_camera"
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
    def camera_calib_for_sample(self, sample): return {"front_camera": self._c}


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
