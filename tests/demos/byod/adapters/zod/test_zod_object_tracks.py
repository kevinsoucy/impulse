"""Unit tests for demos/byod/adapters/zod/object_tracks.py."""

import pytest

from adapters.zod.loader import Annotation, Sample, Scene
from adapters.zod.object_tracks import (
    LANE_WIDTH_M,
    azimuth_sector,
    lane_offset,
    map_all_scenes,
    map_scene_to_object_tracks,
    source_from_annotation,
)


def _ann(*, ts: int, klass: str, uuid: str, gx: float, gy: float = 0.0,
         num_lidar=10, num_radar=2, occlusion=0.1, doppler: float | None = -1.5) -> Annotation:
    return Annotation(
        container_id=1, sample_token="s", timestamp_us=ts,
        object_id=hash(uuid) & 0x7FFFFFFFFFFFFFFF,
        uuid=uuid, category_name=klass, detection_class=klass,
        translation=(gx, gy, 0.0), size=(4.0, 1.8, 1.5),
        rotation=(1.0, 0.0, 0.0, 0.0),
        num_lidar_pts=num_lidar, num_radar_pts=num_radar,
        occlusion=occlusion, radar_doppler_ms=doppler,
    )


def _scene() -> Scene:
    return Scene(container_id=1, sequence_id="t", name="s", description="",
                 nbr_samples=0, sequence_dir="/tmp", start_ts_us=0)


def _sample(ts: int, fid: str) -> Sample:
    return Sample(container_id=1, sample_token=fid, timestamp_us=ts,
                  sequence_id="t", frame_id=fid, sensor_files={})


class FakeLoader:
    def __init__(self, samples, anns):
        self._s = samples
        self._a = anns

    def samples_in_scene(self, scene): return iter(self._s)
    def annotations_in_sample(self, sample): return iter(self._a.get(sample.sample_token, []))


# ── source_from_annotation ───────────────────────────────────────────────────


class TestSourceFromAnnotation:
    def test_all_three_modalities(self):
        assert source_from_annotation(10, 5, occlusion=0.1) == "lidar|radar|camera"

    def test_lidar_camera_no_radar(self):
        assert source_from_annotation(10, 0, occlusion=0.1) == "lidar|camera"

    def test_radar_camera_no_lidar(self):
        assert source_from_annotation(0, 5, occlusion=0.1) == "radar|camera"

    def test_heavy_occlusion_drops_camera(self):
        assert source_from_annotation(10, 5, occlusion=0.9) == "lidar|radar"

    def test_all_off_falls_back_to_camera(self):
        assert source_from_annotation(0, 0, occlusion=0.95) == "camera"

    def test_radar_modality_appears_only_with_radar_points(self):
        # The BL-008 differentiator. ZOD is the only adapter where this
        # invariant produces non-empty radar coverage at object_tracks-time.
        assert "radar" in source_from_annotation(0, 3, occlusion=0.1).split("|")
        assert "radar" not in source_from_annotation(0, 0, occlusion=0.1).split("|")


# ── Geometry re-exports ──────────────────────────────────────────────────────


class TestGeometryReexports:
    def test_azimuth_basic(self):
        assert azimuth_sector(1.0, 0.0) == "front"

    def test_lane_basic(self):
        assert lane_offset(LANE_WIDTH_M) == 1


# ── map_scene_to_object_tracks ───────────────────────────────────────────────


class TestMapSceneToObjectTracks:
    def test_empty_scene(self):
        loader = FakeLoader(samples=[], anns={})
        assert map_scene_to_object_tracks(loader, _scene()) == []

    def test_single_annotation_populates_expected_fields(self):
        s1 = _sample(0, "f1")
        loader = FakeLoader(samples=[s1], anns={"f1": [_ann(ts=0, klass="car", uuid="A", gx=10.0)]})
        rows = map_scene_to_object_tracks(loader, _scene())
        assert len(rows) == 1
        r = rows[0]
        assert r["detection_class"] == "car"
        assert r["distance_m"] == pytest.approx(10.0, rel=1e-9)
        assert r["azimuth"] == "front"
        assert r["lane_offset"] == 0
        assert r["confidence"] == 1.0
        assert r["source"] == "lidar|radar|camera"
        assert r["relative_velocity_ms"] is None  # first observation

    def test_relative_velocity_populated_on_second_observation(self):
        s1 = _sample(0, "f1")
        s2 = _sample(500_000, "f2")
        loader = FakeLoader(
            samples=[s1, s2],
            anns={
                "f1": [_ann(ts=0, klass="car", uuid="A", gx=20.0)],
                "f2": [_ann(ts=500_000, klass="car", uuid="A", gx=10.0)],
            },
        )
        rows = map_scene_to_object_tracks(loader, _scene())
        assert rows[0]["relative_velocity_ms"] is None
        # Δ‖position‖ = -10 m over 0.5 s = -20 m/s (approaching).
        assert rows[1]["relative_velocity_ms"] == pytest.approx(-20.0, rel=1e-9)

    def test_distinct_uuids_track_independently(self):
        s1 = _sample(0, "f1")
        s2 = _sample(500_000, "f2")
        loader = FakeLoader(
            samples=[s1, s2],
            anns={
                "f1": [_ann(ts=0, klass="car", uuid="A", gx=20.0),
                       _ann(ts=0, klass="car", uuid="B", gx=30.0)],
                "f2": [_ann(ts=500_000, klass="car", uuid="A", gx=10.0)],
            },
        )
        rows = map_scene_to_object_tracks(loader, _scene())
        # A's second observation gets a relative velocity; B's first observation does not.
        assert rows[0]["relative_velocity_ms"] is None  # A first frame
        assert rows[1]["relative_velocity_ms"] is None  # B first frame
        assert rows[2]["relative_velocity_ms"] == pytest.approx(-20.0, rel=1e-9)  # A second frame


class TestMapAllScenes:
    def test_yields_across_scenes(self):
        loader = FakeLoader(samples=[], anns={})
        assert list(map_all_scenes(loader, [_scene(), _scene()])) == []
