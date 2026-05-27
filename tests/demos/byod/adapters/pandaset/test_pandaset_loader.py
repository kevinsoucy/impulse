"""Unit tests for demos/byod/adapters/pandaset/loader.py.

Synthetic gzip-pickle + CSV fixtures (no live PandaSet download). Covers
filesystem walking, annotation parsing, GPS CSV decode, per-frame camera
pose lookup, and the dual-LiDAR channel-id table sanity.
"""

import csv
import gzip
import json
import math
import pickle
from pathlib import Path

import pytest

from adapters.pandaset.loader import (
    DERIVED_CHANNEL_IDS,
    FRAME_DT_US,
    GPS_COLUMN_TO_CHANNEL,
    LIDAR_SENSOR_BY_D,
    SENSOR_CHANNEL_IDS,
    Annotation,
    PandaSetLoader,
    _annotation_from_row,
    _quat_from_yaw,
    simplify_class,
    stable_int_id,
)


# ── simplify_class ───────────────────────────────────────────────────────────


class TestSimplifyClass:
    @pytest.mark.parametrize("raw,expected", [
        ("Car", "car"),
        ("Pedestrian", "pedestrian"),
        ("Pickup Truck", "truck"),
        ("Bicycle", "cyclist"),
        ("Motorcycle", "cyclist"),
        ("Bicycle Rider", "cyclist"),
        ("Bus", "bus"),
        ("VanSUV", "van"),
        ("Animal - Other", "animal"),
        ("", "other"),
    ])
    def test_maps_to_lakevision_class(self, raw, expected):
        assert simplify_class(raw) == expected


# ── stable_int_id ────────────────────────────────────────────────────────────


class TestStableIntId:
    def test_deterministic(self):
        assert stable_int_id("uuid-1") == stable_int_id("uuid-1")

    def test_fits_63_bits(self):
        assert stable_int_id("uuid-1") < 2**63

    def test_distinct(self):
        assert stable_int_id("uuid-A") != stable_int_id("uuid-B")


# ── Sensor-id tables ────────────────────────────────────────────────────────


class TestSensorTables:
    def test_two_lidar_sensors_spinning_and_solid_state(self):
        assert LIDAR_SENSOR_BY_D == {0: "LIDAR_SPINNING", 1: "LIDAR_SOLIDSTATE"}

    def test_gps_columns_resolve_to_known_channels(self):
        for channel_name in GPS_COLUMN_TO_CHANNEL.values():
            assert channel_name in DERIVED_CHANNEL_IDS

    def test_sensor_and_derived_id_spaces_dont_overlap(self):
        assert set(SENSOR_CHANNEL_IDS.values()) & set(DERIVED_CHANNEL_IDS.values()) == set()

    def test_both_lidar_channels_registered(self):
        assert "lidar_pandar64" in SENSOR_CHANNEL_IDS
        assert "lidar_pandargt" in SENSOR_CHANNEL_IDS


# ── _quat_from_yaw ──────────────────────────────────────────────────────────


class TestQuatFromYaw:
    def test_zero_yaw_returns_identity(self):
        assert _quat_from_yaw(0.0) == pytest.approx((1.0, 0.0, 0.0, 0.0), abs=1e-12)

    def test_90deg_yaw(self):
        q = _quat_from_yaw(math.pi / 2)
        c = math.cos(math.pi / 4)
        s = math.sin(math.pi / 4)
        assert q == pytest.approx((c, 0.0, 0.0, s), abs=1e-12)


# ── _annotation_from_row ────────────────────────────────────────────────────


class TestAnnotationFromRow:
    def test_dot_notation_columns(self):
        row = {
            "position.x": 10.0, "position.y": 0.5, "position.z": 0.0,
            "dimensions.x": 4.5, "dimensions.y": 1.8, "dimensions.z": 1.6,
            "yaw": math.pi / 4,
            "uuid": "uuid-1",
            "label": "Car",
        }
        ann = _annotation_from_row(row, container_id=1, sample_token="s", timestamp_us=12345)
        assert ann.translation == (10.0, 0.5, 0.0)
        assert ann.size == (4.5, 1.8, 1.6)
        assert ann.detection_class == "car"
        assert ann.yaw_rad == pytest.approx(math.pi / 4, abs=1e-12)

    def test_nested_dict_columns(self):
        row = {
            "position": {"x": 5.0, "y": -1.0, "z": 0.5},
            "dimensions": {"x": 0.6, "y": 0.6, "z": 1.7},
            "yaw": 0.0,
            "uuid": "uuid-2",
            "label": "Pedestrian",
        }
        ann = _annotation_from_row(row, container_id=1, sample_token="s", timestamp_us=0)
        assert ann.translation == (5.0, -1.0, 0.5)
        assert ann.detection_class == "pedestrian"

    def test_object_id_hashed_from_uuid(self):
        row = {"position.x": 0.0, "position.y": 0.0, "position.z": 0.0,
               "dimensions.x": 1.0, "dimensions.y": 1.0, "dimensions.z": 1.0,
               "yaw": 0.0, "uuid": "uuid-X", "label": "Car"}
        ann = _annotation_from_row(row, container_id=1, sample_token="s", timestamp_us=0)
        assert ann.object_id == stable_int_id("uuid-X")


# ── Filesystem walking ───────────────────────────────────────────────────────


def _build_minimal_pandaset(root: Path, sequence_id: str = "001", frames: int = 2) -> Path:
    seq_dir = root / sequence_id
    (seq_dir / "annotations" / "cuboids").mkdir(parents=True)
    (seq_dir / "lidar").mkdir(parents=True)
    (seq_dir / "meta").mkdir(parents=True)
    for cam in ("front_camera", "front_left_camera", "front_right_camera",
                "left_camera", "right_camera", "back_camera"):
        (seq_dir / "camera" / cam).mkdir(parents=True)
        (seq_dir / "camera" / cam / "intrinsics.json").write_text(json.dumps({
            "fx": 1900.0, "fy": 1900.0, "cx": 960.0, "cy": 540.0,
            "width": 1920, "height": 1080,
        }))
        # Per-frame poses.
        poses = [
            {"position": {"x": 1.7, "y": 0.0, "z": 1.5},
             "heading": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}}
            for _ in range(frames)
        ]
        (seq_dir / "camera" / cam / "poses.json").write_text(json.dumps(poses))
        for i in range(frames):
            (seq_dir / "camera" / cam / f"{i:02d}.jpg").write_bytes(b"")

    # Per-frame timestamps in seconds (10 Hz).
    base_ts = 1577836800.0  # 2020-01-01 UTC
    timestamps = [base_ts + i * 0.1 for i in range(frames)]
    (seq_dir / "meta" / "timestamps.json").write_text(json.dumps(timestamps))

    # Synthetic GPS CSV.
    with open(seq_dir / "meta" / "gps.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "speed", "heading", "lat", "lon"])
        w.writeheader()
        for i in range(frames):
            w.writerow({
                "timestamp": str(base_ts + i * 0.1),
                "speed": str(10.0 + i * 0.5),       # m/s
                "heading": str(45.0 + i),           # deg
                "lat": "37.7749",
                "lon": "-122.4194",
            })

    # Synthetic annotations: one car + one pedestrian per frame, with persistent uuids.
    for i in range(frames):
        records = [
            {
                "position.x": 20.0 - i * 5.0, "position.y": 0.0, "position.z": 0.0,
                "dimensions.x": 4.5, "dimensions.y": 1.8, "dimensions.z": 1.6,
                "yaw": 0.0,
                "uuid": "obj-A",
                "label": "Car",
            },
            {
                "position.x": 5.0, "position.y": 3.0, "position.z": 0.0,
                "dimensions.x": 0.6, "dimensions.y": 0.6, "dimensions.z": 1.7,
                "yaw": math.pi / 2,
                "uuid": "obj-B",
                "label": "Pedestrian",
            },
        ]
        pkl_path = seq_dir / "annotations" / "cuboids" / f"{i:02d}.pkl.gz"
        with gzip.open(pkl_path, "wb") as f:
            pickle.dump(records, f)
        # Empty LiDAR file (the test path doesn't read its contents).
        (seq_dir / "lidar" / f"{i:02d}.pkl.gz").write_bytes(b"")

    return root


class TestPandaSetLoader:
    def test_scenes_lists_3digit_directories(self, tmp_path):
        _build_minimal_pandaset(tmp_path)
        scenes = PandaSetLoader(dataroot=str(tmp_path)).list_scenes()
        assert len(scenes) == 1
        assert scenes[0].sequence_id == "001"
        assert scenes[0].nbr_samples == 2

    def test_skips_non_conforming_directories(self, tmp_path):
        _build_minimal_pandaset(tmp_path)
        (tmp_path / "not_a_seq").mkdir()
        (tmp_path / "9999").mkdir()  # 4-digit, also skipped
        scenes = PandaSetLoader(dataroot=str(tmp_path)).list_scenes()
        assert [s.sequence_id for s in scenes] == ["001"]

    def test_samples_use_timestamps_json(self, tmp_path):
        _build_minimal_pandaset(tmp_path, frames=2)
        loader = PandaSetLoader(dataroot=str(tmp_path))
        samples = list(loader.samples_in_scene(loader.list_scenes()[0]))
        # Timestamps come from meta/timestamps.json (1577836800.0 + N*0.1) in µs.
        assert samples[0].timestamp_us == 1577836800_000_000
        assert samples[1].timestamp_us == 1577836800_100_000

    def test_samples_fall_back_when_timestamps_missing(self, tmp_path):
        _build_minimal_pandaset(tmp_path)
        # Delete timestamps file to force fallback.
        (tmp_path / "001" / "meta" / "timestamps.json").unlink()
        loader = PandaSetLoader(dataroot=str(tmp_path))
        # Clear the lru_cache so the fallback is re-tested fresh.
        loader._frame_timestamps.cache_clear()
        samples = list(loader.samples_in_scene(loader.list_scenes()[0]))
        # Fallback: scene.start_ts_us + i * FRAME_DT_US.
        assert samples[1].timestamp_us - samples[0].timestamp_us == FRAME_DT_US

    def test_annotations_parse_persistent_uuids(self, tmp_path):
        _build_minimal_pandaset(tmp_path, frames=2)
        loader = PandaSetLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        samples = list(loader.samples_in_scene(scene))
        anns_f0 = list(loader.annotations_in_sample(samples[0]))
        anns_f1 = list(loader.annotations_in_sample(samples[1]))
        # Same uuids across frames.
        assert {a.uuid for a in anns_f0} == {"obj-A", "obj-B"}
        assert {a.uuid for a in anns_f1} == {"obj-A", "obj-B"}
        # Object IDs match by uuid across frames (cross-frame tracking).
        obj_a_f0 = next(a for a in anns_f0 if a.uuid == "obj-A")
        obj_a_f1 = next(a for a in anns_f1 if a.uuid == "obj-A")
        assert obj_a_f0.object_id == obj_a_f1.object_id

    def test_gps_csv_decode(self, tmp_path):
        _build_minimal_pandaset(tmp_path, frames=2)
        loader = PandaSetLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        gps = loader.gps_for_scene(scene)
        assert "Vehicle_Speed_kph" in gps
        # Speed scaled from m/s to kph: 10.0 m/s = 36.0 kph.
        assert gps["Vehicle_Speed_kph"][0][1] == pytest.approx(36.0, rel=1e-9)
        assert gps["Heading_deg"][0][1] == pytest.approx(45.0, abs=1e-9)
        assert gps["Latitude_deg"][0][1] == pytest.approx(37.7749, abs=1e-9)

    def test_camera_calib_resolves_intrinsics_and_per_frame_pose(self, tmp_path):
        _build_minimal_pandaset(tmp_path)
        loader = PandaSetLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        calib = loader.camera_calib_for_sample(sample)
        assert "front_camera" in calib
        info = calib["front_camera"]
        assert info["width"] == 1920
        assert info["height"] == 1080
        assert info["sensor_translation"] == [1.7, 0.0, 1.5]
        # K matrix assembled from fx/fy/cx/cy.
        assert info["intrinsic"][0][0] == 1900.0
        assert info["intrinsic"][0][2] == 960.0

    def test_ego_pose_is_identity(self, tmp_path):
        _build_minimal_pandaset(tmp_path)
        loader = PandaSetLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sample = next(iter(loader.samples_in_scene(scene)))
        ego = loader.ego_pose_for_sample(sample)
        assert ego.translation == (0.0, 0.0, 0.0)
        assert ego.rotation == (1.0, 0.0, 0.0, 0.0)

    def test_all_sample_data_emits_both_lidar_sensors(self, tmp_path):
        # The BL-009 differentiator at the perception_channels-source level.
        _build_minimal_pandaset(tmp_path)
        loader = PandaSetLoader(dataroot=str(tmp_path))
        scene = loader.list_scenes()[0]
        sensors = {sd.sensor_name for sd in loader.all_sample_data_in_scene(scene)}
        assert "lidar_pandar64" in sensors
        assert "lidar_pandargt" in sensors
        assert "front_camera" in sensors

    def test_count_summary(self, tmp_path):
        _build_minimal_pandaset(tmp_path, frames=3)
        loader = PandaSetLoader(dataroot=str(tmp_path))
        assert loader.count_summary() == {"scenes": 1, "samples": 3}
