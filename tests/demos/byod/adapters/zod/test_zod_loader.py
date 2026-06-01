"""Unit tests for demos/byod/adapters/zod/loader.py.

Builds synthetic ZOD-shaped directory trees in tmp_path — no `zod` SDK or
HDF5 file required (`oxts_for_scene` tests are isolated in a separate file
so they can skip when h5py is missing).
"""

import json
from pathlib import Path

import numpy as np
import pytest

from adapters.zod.loader import (
    DERIVED_CHANNEL_IDS,
    OXTS_DATASET_TO_CHANNEL,
    OXTS_UNIT_SCALES,
    SENSOR_CHANNEL_IDS,
    ZodLoader,
    _annotation_from_dict,
    _closest_file,
    _index_files,
    _quat_from_dict_or_list,
    _ts_from_filename,
    simplify_class,
    stable_int_id,
)


# ── simplify_class ───────────────────────────────────────────────────────────


class TestSimplifyClass:
    @pytest.mark.parametrize("name,subclass,expected", [
        ("Vehicle", "Car", "car"),
        ("Vehicle", "Truck", "truck"),
        ("Vehicle", "Bus", "bus"),
        ("Vehicle", "VanSUV", "van"),
        ("Vehicle", None, "car"),  # fallback when subclass missing
        ("Pedestrian", None, "pedestrian"),
        ("VulnerableVehicle", "Bicycle", "cyclist"),
        ("VulnerableVehicle", "Motorbike", "cyclist"),
        ("VulnerableVehicle", None, "cyclist"),
        ("", "", "other"),
        ("Custom", None, "custom"),
    ])
    def test_maps_to_lakevision_class(self, name, subclass, expected):
        assert simplify_class(name, subclass) == expected


# ── stable_int_id ────────────────────────────────────────────────────────────


class TestStableIntId:
    def test_deterministic(self):
        assert stable_int_id("zod-uuid-abc") == stable_int_id("zod-uuid-abc")

    def test_distinct_inputs_distinct_outputs(self):
        assert stable_int_id("uuid-a") != stable_int_id("uuid-b")

    def test_fits_63_bits(self):
        assert stable_int_id("anything") < 2**63


# ── _ts_from_filename ────────────────────────────────────────────────────────


class TestTsFromFilename:
    def test_microsecond_timestamp_passthrough(self):
        assert _ts_from_filename("1672534800123456_abc.npy") == 1672534800123456

    def test_nanosecond_timestamp_divided(self):
        assert _ts_from_filename("1672534800123456789_abc.npy") == 1672534800123456

    def test_invalid_prefix_returns_zero(self):
        assert _ts_from_filename("not_a_timestamp.npy") == 0


# ── _index_files / _closest_file ─────────────────────────────────────────────


class TestNearestFileMatching:
    def test_empty_directory_returns_empty_index(self, tmp_path):
        assert _index_files(tmp_path, "*.jpg") == []

    def test_closest_returns_none_on_empty(self):
        assert _closest_file([], 100) is None

    def test_closest_picks_smallest_delta(self, tmp_path):
        # Use realistic 16-digit µs timestamps (the parser requires that range).
        for ts in (1672534800000100, 1672534800000500, 1672534800001000):
            (tmp_path / f"{ts}_x.jpg").write_bytes(b"")
        idx = _index_files(tmp_path, "*.jpg")
        chosen = _closest_file(idx, 1672534800000480)
        assert chosen is not None
        assert chosen.name.startswith("1672534800000500_")


# ── _quat_from_dict_or_list ──────────────────────────────────────────────────


class TestQuatCoercion:
    def test_dict_form(self):
        q = _quat_from_dict_or_list({"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0})
        assert q == (1.0, 0.0, 0.0, 0.0)

    def test_list_form(self):
        q = _quat_from_dict_or_list([0.7071, 0.0, 0.0, 0.7071])
        assert q == pytest.approx((0.7071, 0.0, 0.0, 0.7071), abs=1e-9)

    def test_missing_keys_default_to_identity(self):
        assert _quat_from_dict_or_list({"qw": 0.5}) == (0.5, 0.0, 0.0, 0.0)

    def test_garbage_input_returns_identity(self):
        assert _quat_from_dict_or_list(None) == (1.0, 0.0, 0.0, 0.0)
        assert _quat_from_dict_or_list([1.0, 2.0]) == (1.0, 0.0, 0.0, 0.0)


# ── Channel-id table sanity ──────────────────────────────────────────────────


class TestChannelIdTables:
    def test_oxts_targets_resolve_to_known_channels(self):
        for lakevision_name in OXTS_DATASET_TO_CHANNEL.values():
            assert lakevision_name in DERIVED_CHANNEL_IDS

    def test_unit_scale_targets_are_known_channels(self):
        for name in OXTS_UNIT_SCALES:
            assert name in DERIVED_CHANNEL_IDS

    def test_sensor_and_derived_id_spaces_dont_overlap(self):
        assert set(SENSOR_CHANNEL_IDS.values()) & set(DERIVED_CHANNEL_IDS.values()) == set()


# ── _annotation_from_dict ────────────────────────────────────────────────────


class TestAnnotationFromDict:
    def _row(self, **overrides):
        base = {
            "uuid": "uuid-1",
            "name": "Vehicle",
            "subclass": "Car",
            "frame_id": "frame-1",
            "box3d": {
                "center": [10.0, 0.5, 0.2],
                "size": [4.5, 1.8, 1.6],
                "orientation": {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
            },
            "lidar_attribute": {"num_points": 150},
            "radar_attribute": {"num_points": 4, "doppler": -3.2},
            "occlusion": 0.1,
        }
        base.update(overrides)
        return base

    def test_basic_fields_populated(self):
        ann = _annotation_from_dict(self._row(), container_id=1, sample_token="s", timestamp_us=12345)
        assert ann.uuid == "uuid-1"
        assert ann.detection_class == "car"
        assert ann.translation == (10.0, 0.5, 0.2)
        assert ann.size == (4.5, 1.8, 1.6)
        assert ann.num_lidar_pts == 150
        assert ann.num_radar_pts == 4
        assert ann.radar_doppler_ms == pytest.approx(-3.2, abs=1e-12)
        assert ann.occlusion == pytest.approx(0.1, abs=1e-12)

    def test_object_id_is_stable_hash_of_uuid(self):
        ann = _annotation_from_dict(self._row(), container_id=1, sample_token="s", timestamp_us=0)
        assert ann.object_id == stable_int_id("uuid-1")

    def test_subclass_drives_simplification(self):
        row = self._row(subclass="Truck")
        ann = _annotation_from_dict(row, container_id=1, sample_token="s", timestamp_us=0)
        assert ann.detection_class == "truck"

    def test_radar_doppler_none_when_attr_absent(self):
        row = self._row(radar_attribute={"num_points": 0})
        ann = _annotation_from_dict(row, container_id=1, sample_token="s", timestamp_us=0)
        assert ann.radar_doppler_ms is None

    def test_category_name_combines_name_and_subclass(self):
        ann = _annotation_from_dict(self._row(), container_id=1, sample_token="s", timestamp_us=0)
        assert ann.category_name == "Vehicle/Car"


# ── Filesystem walking ───────────────────────────────────────────────────────


def _build_minimal_zod(root: Path, sequence_id: str = "000001", frames: int = 2) -> Path:
    seq_dir = root / "sequences" / sequence_id
    (seq_dir / "camera_front_blur").mkdir(parents=True)
    (seq_dir / "lidar_velodyne").mkdir(parents=True)
    (seq_dir / "radar").mkdir(parents=True)
    (seq_dir / "annotations").mkdir(parents=True)

    base_ts_ns = 1672534800_000_000_000  # 2023-01-01 in ns
    frame_ids: list[str] = []
    for i in range(frames):
        ts_ns = base_ts_ns + i * 100_000_000  # 10 Hz
        stem = f"{ts_ns:019d}_abcd1234"
        frame_ids.append(stem)
        (seq_dir / "camera_front_blur" / f"{stem}.jpg").write_bytes(b"")
        (seq_dir / "lidar_velodyne" / f"{stem}.npy").write_bytes(b"")
        (seq_dir / "radar" / f"{stem}.npy").write_bytes(b"")

    annotations = []
    for i, fid in enumerate(frame_ids):
        annotations.append({
            "uuid": "obj-A",
            "name": "Vehicle",
            "subclass": "Car",
            "frame_id": fid,
            "box3d": {
                "center": [20.0 - i * 5.0, 0.0, 0.0],
                "size": [4.5, 1.8, 1.6],
                "orientation": {"qw": 1.0, "qx": 0.0, "qy": 0.0, "qz": 0.0},
            },
            "lidar_attribute": {"num_points": 120},
            "radar_attribute": {"num_points": 5, "doppler": -2.0},
            "occlusion": 0.1,
        })
        annotations.append({
            "uuid": "obj-B",
            "name": "Pedestrian",
            "frame_id": fid,
            "box3d": {
                "center": [5.0, 3.0, 0.0],
                "size": [0.6, 0.6, 1.7],
                "orientation": [1.0, 0.0, 0.0, 0.0],
            },
            "lidar_attribute": {"num_points": 8},
            "radar_attribute": {"num_points": 0},
            "occlusion": 0.85,  # heavily occluded — camera modality excluded
        })
    (seq_dir / "annotations" / "object_detection_3d.json").write_text(json.dumps(annotations))

    (seq_dir / "calibration.json").write_text(json.dumps({
        "cameras": {
            "camera_front_blur": {
                "intrinsic": {
                    "K": [[2000.0, 0.0, 1924.0], [0.0, 2000.0, 1084.0], [0.0, 0.0, 1.0]],
                    "image_dimensions": [3848, 2168],
                },
                "extrinsic": {
                    "translation": [1.7, 0.0, 1.5],
                    "rotation": [1.0, 0.0, 0.0, 0.0],
                },
            }
        }
    }))
    return root


class TestZodLoader:
    def test_scenes_lists_sequences(self, tmp_path):
        _build_minimal_zod(tmp_path)
        loader = ZodLoader(dataroot=str(tmp_path))
        scenes = loader.list_scenes()
        assert len(scenes) == 1
        assert scenes[0].sequence_id == "000001"
        assert scenes[0].nbr_samples == 2

    def test_samples_in_scene_have_aligned_sensor_files(self, tmp_path):
        _build_minimal_zod(tmp_path, frames=2)
        loader = ZodLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        samples = list(loader.samples_in_scene(scene))
        assert len(samples) == 2
        for sample in samples:
            assert "lidar_velodyne" in sample.sensor_files
            assert "camera_front_blur" in sample.sensor_files
            assert "radar" in sample.sensor_files

    def test_sample_timestamps_monotonic(self, tmp_path):
        _build_minimal_zod(tmp_path, frames=3)
        loader = ZodLoader(dataroot=str(tmp_path))
        samples = list(loader.samples_in_scene(loader.list_scenes()[0]))
        assert len(samples) == 3
        assert samples[0].timestamp_us < samples[1].timestamp_us < samples[2].timestamp_us

    def test_annotations_parse_with_separate_lidar_radar_attrs(self, tmp_path):
        _build_minimal_zod(tmp_path)
        loader = ZodLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        anns = list(loader.annotations_in_sample(sample))
        assert len(anns) == 2
        car = next(a for a in anns if a.detection_class == "car")
        assert car.num_lidar_pts == 120
        assert car.num_radar_pts == 5
        assert car.radar_doppler_ms == pytest.approx(-2.0, abs=1e-12)

    def test_missing_annotations_file_returns_empty(self, tmp_path):
        _build_minimal_zod(tmp_path)
        # Delete the annotations file.
        (tmp_path / "sequences" / "000001" / "annotations" / "object_detection_3d.json").unlink()
        loader = ZodLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        assert list(loader.annotations_in_sample(sample)) == []

    def test_camera_calib_resolves_intrinsic_and_extrinsic(self, tmp_path):
        _build_minimal_zod(tmp_path)
        loader = ZodLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        calib = loader.camera_calib_for_sample(sample)
        assert "camera_front_blur" in calib
        info = calib["camera_front_blur"]
        assert info["sensor_translation"] == [1.7, 0.0, 1.5]
        assert info["width"] == 3848
        assert info["height"] == 2168
        np.testing.assert_allclose(info["sensor_rotation"], [1.0, 0.0, 0.0, 0.0], atol=1e-9)

    def test_ego_pose_is_identity(self, tmp_path):
        _build_minimal_zod(tmp_path)
        loader = ZodLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        ego = loader.ego_pose_for_sample(sample)
        assert ego.translation == (0.0, 0.0, 0.0)
        assert ego.rotation == (1.0, 0.0, 0.0, 0.0)

    def test_all_sample_data_yields_camera_lidar_radar(self, tmp_path):
        _build_minimal_zod(tmp_path)
        loader = ZodLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sensors = {sd.sensor_name for sd in loader.all_sample_data_in_scene(scene)}
        assert sensors == {"camera_front_blur", "lidar_velodyne", "radar"}

    def test_version_prefixed_layout_also_works(self, tmp_path):
        # Build a version-prefixed dataroot: {root}/sequences-mini/sequences/...
        version_root = tmp_path / "sequences-mini"
        version_root.mkdir()
        _build_minimal_zod(version_root)
        # `dataroot=tmp_path` should still discover sequences via the version prefix.
        loader = ZodLoader(dataroot=str(tmp_path), dataset_version="sequences-mini")
        scenes = loader.list_scenes()
        assert len(scenes) == 1

    def test_count_summary(self, tmp_path):
        _build_minimal_zod(tmp_path, frames=4)
        loader = ZodLoader(dataroot=str(tmp_path))
        assert loader.count_summary() == {"scenes": 1, "samples": 4}
