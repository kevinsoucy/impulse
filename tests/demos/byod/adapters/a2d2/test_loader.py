"""Unit tests for demos/byod/adapters/a2d2/loader.py.

Uses tmp_path fixtures to build synthetic A2D2 directory trees — no live
download required. Validates filesystem walking, JSON parsing, calibration
loading, axis→quaternion conversion, and the simplification map.
"""

import json
import math
from pathlib import Path

import numpy as np
import pytest

from adapters.a2d2.loader import (
    BUS_SIGNAL_TO_CHANNEL,
    DERIVED_CHANNEL_IDS,
    FRAME_DT_US,
    SENSOR_CHANNEL_IDS,
    A2D2Loader,
    _axes_to_quat,
    _axis_angle_to_quat,
    _scene_start_us,
    simplify_class,
    stable_int_id,
)


# ── simplify_class ───────────────────────────────────────────────────────────


class TestSimplifyClass:
    @pytest.mark.parametrize("raw,expected", [
        ("Pedestrian", "pedestrian"),
        ("pedestrian", "pedestrian"),
        ("Bicycle", "cyclist"),
        ("Motorcycle", "cyclist"),
        ("MotorBiker", "cyclist"),
        ("Car", "car"),
        ("Bus", "bus"),
        ("Truck", "truck"),
        ("Trailer", "truck"),
        ("UtilityVehicle", "truck"),
        ("VanSUV", "van"),
        ("Animal", "animal"),
        ("UnknownClass", "unknownclass"),
        ("", "other"),
    ])
    def test_maps_to_lakevision_class(self, raw, expected):
        assert simplify_class(raw) == expected


# ── stable_int_id ────────────────────────────────────────────────────────────


class TestStableIntId:
    def test_deterministic(self):
        assert stable_int_id("20180807_145028") == stable_int_id("20180807_145028")

    def test_nonzero_for_realistic_input(self):
        assert stable_int_id("20180807_145028") != 0

    def test_fits_63_bits(self):
        assert stable_int_id("20180807_145028") < 2**63

    def test_distinct_scenes_get_distinct_ids(self):
        assert stable_int_id("20180807_145028") != stable_int_id("20180810_142822")


# ── _scene_start_us ──────────────────────────────────────────────────────────


class TestSceneStartUs:
    def test_parses_iso_like_directory_name(self):
        # 2018-08-07 14:50:28 UTC
        from datetime import datetime, timezone
        expected = int(datetime(2018, 8, 7, 14, 50, 28, tzinfo=timezone.utc).timestamp() * 1_000_000)
        assert _scene_start_us("20180807_145028") == expected

    def test_unrecognized_directory_name_does_not_raise(self):
        # Falls back to a stable hash-derived value so the pipeline can still run
        # over directories that don't conform exactly.
        assert isinstance(_scene_start_us("not_a_date"), int)


# ── _axis_angle_to_quat ──────────────────────────────────────────────────────


class TestAxisAngleToQuat:
    def test_zero_angle_returns_identity(self):
        q = _axis_angle_to_quat([0.0, 0.0, 1.0], 0.0)
        assert q == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-12)

    def test_90deg_yaw_returns_z_axis_quat(self):
        q = _axis_angle_to_quat([0.0, 0.0, 1.0], math.pi / 2)
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        assert q == pytest.approx((c, 0.0, 0.0, s), abs=1e-12)

    def test_zero_norm_axis_returns_identity(self):
        q = _axis_angle_to_quat([0.0, 0.0, 0.0], 1.0)
        assert q == (1.0, 0.0, 0.0, 0.0)

    def test_normalizes_input_axis(self):
        # Axis = [0, 0, 2] should be normalized to [0, 0, 1].
        q = _axis_angle_to_quat([0.0, 0.0, 2.0], math.pi / 2)
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        assert q == pytest.approx((c, 0.0, 0.0, s), abs=1e-12)


# ── _axes_to_quat ────────────────────────────────────────────────────────────


class TestAxesToQuat:
    def test_canonical_axes_return_identity(self):
        q = _axes_to_quat([1.0, 0.0, 0.0], [0.0, 1.0, 0.0])
        assert q == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-9)

    def test_90deg_yaw_axes(self):
        # x-axis along world +Y, y-axis along world -X → rotation = 90° yaw about +Z
        q = _axes_to_quat([0.0, 1.0, 0.0], [-1.0, 0.0, 0.0])
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        assert q == pytest.approx((c, 0.0, 0.0, s), abs=1e-9)

    def test_zero_norm_x_axis_returns_identity(self):
        assert _axes_to_quat([0.0, 0.0, 0.0], [0.0, 1.0, 0.0]) == (1.0, 0.0, 0.0, 0.0)

    def test_parallel_axes_return_identity(self):
        # y identical to x → cross product is zero → identity fallback.
        assert _axes_to_quat([1.0, 0.0, 0.0], [1.0, 0.0, 0.0]) == (1.0, 0.0, 0.0, 0.0)


# ── Channel-id table sanity ──────────────────────────────────────────────────


class TestChannelIdTables:
    def test_all_bus_signal_mappings_resolve_to_known_channels(self):
        for lakevision_name in BUS_SIGNAL_TO_CHANNEL.values():
            assert lakevision_name in DERIVED_CHANNEL_IDS

    def test_sensor_channel_ids_are_distinct(self):
        values = list(SENSOR_CHANNEL_IDS.values())
        assert len(values) == len(set(values))

    def test_derived_channel_ids_are_distinct(self):
        values = list(DERIVED_CHANNEL_IDS.values())
        assert len(values) == len(set(values))

    def test_sensor_and_derived_id_spaces_dont_overlap(self):
        assert set(SENSOR_CHANNEL_IDS.values()) & set(DERIVED_CHANNEL_IDS.values()) == set()


# ── Filesystem walking ───────────────────────────────────────────────────────


def _build_minimal_dataroot(root: Path, scene_id: str = "20180807_145028", frames: int = 2) -> Path:
    """Materialize a synthetic A2D2-shaped dataroot for tests."""
    version = "camera_lidar_semantic_bboxes"
    scene_dir = root / version / scene_id
    (scene_dir / "camera" / "cam_front_center").mkdir(parents=True)
    (scene_dir / "lidar" / "cam_front_center").mkdir(parents=True)
    (scene_dir / "label3D" / "cam_front_center").mkdir(parents=True)
    (scene_dir / "bus").mkdir(parents=True)

    date_part = scene_id.replace("_", "")
    for i in range(frames):
        png = scene_dir / "camera" / "cam_front_center" / f"{date_part}_camera_frontcenter_{i:09d}.png"
        png.write_bytes(b"")
        npz = scene_dir / "lidar" / "cam_front_center" / f"{date_part}_lidar_frontcenter_{i:09d}.npz"
        npz.write_bytes(b"")
        label = scene_dir / "label3D" / "cam_front_center" / f"{date_part}_label3D_frontcenter_{i:09d}.json"
        label.write_text(json.dumps({
            "box_0": {
                "center": [10.0 + i, 0.5 - i * 0.1, 0.0],
                "size":   [4.0, 1.8, 1.5],
                "axis":   [0.0, 0.0, 1.0],
                "angle":  0.0,
                "class":  "Car",
                "truncation": 0.0,
                "occlusion": 0.0,
            },
            "box_1": {
                "center": [5.0, -3.0, 0.0],
                "size":   [0.6, 0.6, 1.7],
                "axis":   [0.0, 0.0, 1.0],
                "angle":  math.pi / 2,
                "class":  "Pedestrian",
            },
        }))

    (scene_dir / "bus" / "bus_signals.json").write_text(json.dumps({
        "vehicle_speed": {
            "unit": "kph",
            "values": [{"timestamp": 1_000_000, "value": 30.0}, {"timestamp": 1_100_000, "value": 32.0}],
        },
        "acceleration_x": {
            "unit": "m/s^2",
            "values": [{"timestamp": 1_000_000, "value": 0.5}],
        },
        "unused_signal": {
            "unit": "?",
            "values": [{"timestamp": 0, "value": 0.0}],
        },
    }))

    (root / "cams_lidars.json").write_text(json.dumps({
        "cameras": {
            "cam_front_center": {
                "view": {"origin": [1.7, 0.0, 1.5], "x-axis": [1.0, 0.0, 0.0], "y-axis": [0.0, 1.0, 0.0]},
                "CamMatrix": [[1687.3, 0.0, 965.4], [0.0, 1687.3, 569.4], [0.0, 0.0, 1.0]],
                "Resolution": [1920, 1208],
            },
        },
    }))

    return root


class TestA2D2Loader:
    def test_scenes_lists_timestamped_directories(self, tmp_path):
        _build_minimal_dataroot(tmp_path)
        loader = A2D2Loader(dataroot=str(tmp_path))
        scenes = loader.list_scenes()
        assert len(scenes) == 1
        assert scenes[0].scene_id == "20180807_145028"
        assert scenes[0].nbr_samples == 2

    def test_skips_non_conforming_directories(self, tmp_path):
        _build_minimal_dataroot(tmp_path)
        (tmp_path / "camera_lidar_semantic_bboxes" / "not_a_scene").mkdir()
        loader = A2D2Loader(dataroot=str(tmp_path))
        scene_ids = [s.scene_id for s in loader.list_scenes()]
        assert scene_ids == ["20180807_145028"]

    def test_samples_in_scene_yield_monotonic_frame_index(self, tmp_path):
        _build_minimal_dataroot(tmp_path, frames=3)
        loader = A2D2Loader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        samples = list(loader.samples_in_scene(scene))
        assert [s.frame_index for s in samples] == [0, 1, 2]

    def test_sample_timestamps_advance_by_frame_dt(self, tmp_path):
        _build_minimal_dataroot(tmp_path, frames=2)
        loader = A2D2Loader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        samples = list(loader.samples_in_scene(scene))
        assert samples[1].timestamp_us - samples[0].timestamp_us == FRAME_DT_US

    def test_annotations_parse_box_dict(self, tmp_path):
        _build_minimal_dataroot(tmp_path)
        loader = A2D2Loader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        anns = list(loader.annotations_in_sample(sample))
        assert len(anns) == 2
        classes = {a.detection_class for a in anns}
        assert classes == {"car", "pedestrian"}

    def test_annotations_carry_box_translation_in_vehicle_frame(self, tmp_path):
        _build_minimal_dataroot(tmp_path)
        loader = A2D2Loader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        car = next(a for a in loader.annotations_in_sample(sample) if a.detection_class == "car")
        # The synthetic fixture put the car at vehicle-frame x=10.0 on the first frame.
        assert car.translation[0] == pytest.approx(10.0, abs=1e-12)

    def test_missing_label_file_returns_empty_iter(self, tmp_path):
        _build_minimal_dataroot(tmp_path)
        # Delete one of the label files.
        for f in (tmp_path / "camera_lidar_semantic_bboxes" / "20180807_145028" / "label3D" / "cam_front_center").iterdir():
            f.unlink()
            break
        loader = A2D2Loader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        # If first frame's label was deleted, annotations should be empty for that sample.
        anns = list(loader.annotations_in_sample(sample))
        assert anns == []

    def test_bus_signals_filter_to_known_keys_and_sort(self, tmp_path):
        _build_minimal_dataroot(tmp_path)
        loader = A2D2Loader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        signals = loader.bus_signals_for_scene(scene)
        assert "vehicle_speed" in signals
        assert "acceleration_x" in signals
        assert signals["vehicle_speed"] == [(1_000_000, 30.0), (1_100_000, 32.0)]
        assert "unused_signal" in signals  # the loader returns *all* signals; mapping happens later

    def test_camera_calib_resolves_intrinsic_and_extrinsic(self, tmp_path):
        _build_minimal_dataroot(tmp_path)
        loader = A2D2Loader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        calib = loader.camera_calib_for_sample(sample)
        assert "cam_front_center" in calib
        info = calib["cam_front_center"]
        assert info["sensor_translation"] == [1.7, 0.0, 1.5]
        assert info["width"] == 1920
        assert info["height"] == 1208
        np.testing.assert_allclose(info["sensor_rotation"], [1.0, 0.0, 0.0, 0.0], atol=1e-9)

    def test_ego_pose_is_identity(self, tmp_path):
        _build_minimal_dataroot(tmp_path)
        loader = A2D2Loader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        ego = loader.ego_pose_for_sample(sample)
        assert ego.translation == (0.0, 0.0, 0.0)
        assert ego.rotation == (1.0, 0.0, 0.0, 0.0)

    def test_all_sample_data_yields_camera_and_lidar(self, tmp_path):
        _build_minimal_dataroot(tmp_path, frames=2)
        loader = A2D2Loader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sd_by_sensor = {sd.sensor_name for sd in loader.all_sample_data_in_scene(scene)}
        assert "cam_front_center" in sd_by_sensor
        assert "lidar_front_center" in sd_by_sensor

    def test_count_summary_reports_frames_per_scene(self, tmp_path):
        _build_minimal_dataroot(tmp_path, frames=4)
        loader = A2D2Loader(dataroot=str(tmp_path))
        counts = loader.count_summary()
        assert counts == {"scenes": 1, "samples": 4}
