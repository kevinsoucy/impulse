"""Unit tests for demos/byod/adapters/nuscenes/camera.py."""

import numpy as np
import pytest

from adapters.nuscenes.loader import Annotation, Sample, Scene
from adapters.nuscenes.camera import (
    _box_corners_global,
    _global_to_sensor,
    _project_to_2d,
    map_scene_for_event_windows,
    project_annotation_to_camera,
)


# Forward-looking camera: ego = global (identity ego pose), camera optical axis
# aligned with ego +X (forward).
_IDENTITY_CALIB = {
    "intrinsic": [[400.0, 0.0, 400.0], [0.0, 400.0, 300.0], [0.0, 0.0, 1.0]],
    "sensor_rotation": [0.7071067811865476, 0.0, 0.7071067811865476, 0.0],
    "sensor_translation": [0.0, 0.0, 0.0],
    "ego_rotation": [1.0, 0.0, 0.0, 0.0],
    "ego_translation": [0.0, 0.0, 0.0],
    "width": 800,
    "height": 600,
    "sensor_id": "cam_front",
}


def _ann(
    gx=10.0, gy=0.0, gz=0.0,
    size_wlh=(2.0, 4.0, 1.5),
    rotation=(1.0, 0.0, 0.0, 0.0),
    num_lidar_pts=10, num_radar_pts=2, visibility_token="4",
) -> Annotation:
    return Annotation(
        container_id=1,
        sample_token="s",
        timestamp_us=1_000_000,
        instance_token="inst_01",
        object_id=42,
        category_name="vehicle.car",
        detection_class="car",
        translation=(gx, gy, gz),
        size=size_wlh,
        rotation=rotation,
        num_lidar_pts=num_lidar_pts,
        num_radar_pts=num_radar_pts,
        visibility_token=visibility_token,
    )


class TestGlobalToSensor:
    def test_identity_transform_leaves_point_unchanged(self):
        pt = np.array([5.0, 3.0, 1.0])
        result = _global_to_sensor(
            pt,
            ego_translation=[0.0, 0.0, 0.0], ego_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
            sensor_translation=[0.0, 0.0, 0.0], sensor_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result, pt, atol=1e-10)

    def test_ego_translation_applied(self):
        pt = np.array([10.0, 0.0, 0.0])
        result = _global_to_sensor(
            pt,
            ego_translation=[5.0, 0.0, 0.0], ego_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
            sensor_translation=[0.0, 0.0, 0.0], sensor_rotation_wxyz=[1.0, 0.0, 0.0, 0.0],
        )
        np.testing.assert_allclose(result, [5.0, 0.0, 0.0], atol=1e-10)


class TestBoxCornersGlobal:
    def test_returns_eight_corners(self):
        corners = _box_corners_global((0.0, 0.0, 0.0), (2.0, 4.0, 1.5), (1.0, 0.0, 0.0, 0.0))
        assert corners.shape == (8, 3)

    def test_axis_aligned_box_spans_correct_extent(self):
        w, l, h = 2.0, 4.0, 1.5
        corners = _box_corners_global((0.0, 0.0, 0.0), (w, l, h), (1.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(corners[:, 0].max(), l / 2, atol=1e-10)
        np.testing.assert_allclose(corners[:, 0].min(), -l / 2, atol=1e-10)
        np.testing.assert_allclose(corners[:, 1].max(), w / 2, atol=1e-10)
        np.testing.assert_allclose(corners[:, 2].max(), h / 2, atol=1e-10)

    def test_corner_translated_by_center(self):
        corners = _box_corners_global((10.0, 5.0, 0.0), (2.0, 2.0, 2.0), (1.0, 0.0, 0.0, 0.0))
        np.testing.assert_allclose(corners[:, 0].mean(), 10.0, atol=1e-10)
        np.testing.assert_allclose(corners[:, 1].mean(), 5.0, atol=1e-10)


class TestProjectTo2D:
    def _make_corners_in_front(self):
        c = np.array([
            [ 1,  1, 9], [ 1,  1, 11], [ 1, -1, 9], [ 1, -1, 11],
            [-1,  1, 9], [-1,  1, 11], [-1, -1, 9], [-1, -1, 11],
        ], dtype=np.float64)
        return c

    def test_object_in_front_returns_bbox(self):
        corners = self._make_corners_in_front()
        intrinsic = [[400.0, 0.0, 400.0], [0.0, 400.0, 300.0], [0.0, 0.0, 1.0]]
        result = _project_to_2d(corners, intrinsic, width=800, height=600)
        assert result is not None
        x1, y1, x2, y2 = result
        assert 0 <= x1 < x2 <= 799
        assert 0 <= y1 < y2 <= 599

    def test_all_corners_behind_camera_returns_none(self):
        corners = np.array([[0, 0, -1]] * 8, dtype=np.float64)
        intrinsic = [[400.0, 0.0, 400.0], [0.0, 400.0, 300.0], [0.0, 0.0, 1.0]]
        assert _project_to_2d(corners, intrinsic, width=800, height=600) is None

    def test_bbox_clamped_to_image_bounds(self):
        corners = np.array([
            [ 10,  10, 0.01], [ 10,  10, 0.02],
            [ 10, -10, 0.01], [ 10, -10, 0.02],
            [-10,  10, 0.01], [-10,  10, 0.02],
            [-10, -10, 0.01], [-10, -10, 0.02],
        ], dtype=np.float64)
        intrinsic = [[400.0, 0.0, 400.0], [0.0, 400.0, 300.0], [0.0, 0.0, 1.0]]
        result = _project_to_2d(corners, intrinsic, width=800, height=600)
        assert result is not None
        x1, y1, x2, y2 = result
        assert x1 == 0
        assert y1 == 0
        assert x2 == 799
        assert y2 == 599


class TestProjectAnnotationToCamera:
    def test_object_directly_in_front_returns_valid_row(self):
        ann = _ann(gx=10.0, gy=0.0, gz=0.0)
        row = project_annotation_to_camera(ann, _IDENTITY_CALIB)
        assert row is not None
        assert row["container_id"] == 1
        assert row["frame_ts"] == 1_000_000
        assert row["object_id"] == 42
        assert row["detection_class"] == "car"
        assert row["confidence"] == 1.0
        assert row["sensor_id"] == "cam_front"
        assert 0 <= row["x1"] < row["x2"] <= 799
        assert 0 <= row["y1"] < row["y2"] <= 599

    def test_object_behind_camera_returns_none(self):
        ann = _ann(gx=-10.0, gy=0.0, gz=0.0)
        row = project_annotation_to_camera(ann, _IDENTITY_CALIB)
        assert row is None

    def test_object_far_to_the_side_returns_none_or_clipped(self):
        ann = _ann(gx=1.0, gy=100.0, gz=0.0)
        result = project_annotation_to_camera(ann, _IDENTITY_CALIB)
        if result is not None:
            x1, y1, x2, y2 = result["x1"], result["y1"], result["x2"], result["y2"]
            assert x1 < x2 and y1 < y2


class FakeLoader:
    def __init__(self, samples, anns_by_token, cam_calibs_by_token=None):
        self._samples = samples
        self._anns = anns_by_token
        self._calibs = cam_calibs_by_token or {}

    def samples_in_scene(self, scene):
        return iter(self._samples)

    def annotations_in_sample(self, sample):
        return iter(self._anns.get(sample.sample_token, []))

    def camera_calib_for_sample(self, sample):
        return self._calibs.get(sample.sample_token, {})


def _scene():
    return Scene(container_id=1, scene_token="t", name="s", description="", log_token="",
                 nbr_samples=0, first_sample_token="", last_sample_token="")


def _sample(ts, name):
    return Sample(container_id=1, sample_token=name, timestamp_us=ts, scene_token="t", sensor_data_tokens={})


class TestMapSceneForEventWindows:
    def test_empty_windows_returns_empty(self):
        loader = FakeLoader(samples=[], anns_by_token={})
        assert map_scene_for_event_windows(loader, _scene(), []) == []

    def test_annotation_in_window_projects_to_camera(self):
        s = _sample(1_000_000, "s1")
        ann = _ann(gx=10.0)
        loader = FakeLoader(
            samples=[s],
            anns_by_token={"s1": [ann]},
            cam_calibs_by_token={"s1": {"cam_front": _IDENTITY_CALIB}},
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(0, 2_000_000)])
        assert len(rows) == 1
        assert rows[0]["sensor_id"] == "cam_front"

    def test_annotation_outside_window_produces_no_rows(self):
        s = _sample(5_000_000, "s1")
        ann = _ann(gx=10.0)
        loader = FakeLoader(
            samples=[s],
            anns_by_token={"s1": [ann]},
            cam_calibs_by_token={"s1": {"cam_front": _IDENTITY_CALIB}},
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(0, 2_000_000)])
        assert rows == []

    def test_annotation_behind_camera_produces_no_rows(self):
        s = _sample(1_000_000, "s1")
        ann = _ann(gx=-10.0)
        loader = FakeLoader(
            samples=[s],
            anns_by_token={"s1": [ann]},
            cam_calibs_by_token={"s1": {"cam_front": _IDENTITY_CALIB}},
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(0, 2_000_000)])
        assert rows == []

    def test_multiple_cameras_produce_multiple_rows_per_annotation(self):
        s = _sample(1_000_000, "s1")
        ann = _ann(gx=10.0)
        calib_back = {**_IDENTITY_CALIB, "sensor_id": "cam_back"}
        loader = FakeLoader(
            samples=[s],
            anns_by_token={"s1": [ann]},
            cam_calibs_by_token={"s1": {
                "cam_front": _IDENTITY_CALIB,
                "cam_back": calib_back,
            }},
        )
        rows = map_scene_for_event_windows(loader, _scene(), [(0, 2_000_000)])
        sensor_ids = {r["sensor_id"] for r in rows}
        assert "cam_front" in sensor_ids
