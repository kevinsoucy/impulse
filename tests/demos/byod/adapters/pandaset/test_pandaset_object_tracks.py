"""Unit tests for demos/byod/adapters/pandaset/object_tracks.py."""

import pytest

from adapters.pandaset.loader import Annotation, Sample, Scene
from adapters.pandaset.object_tracks import (
    PANDASET_SOURCE,
    azimuth_sector,
    lane_offset,
    map_all_scenes,
    map_scene_to_object_tracks,
)


def _ann(*, ts: int, klass: str, uuid: str, gx: float, gy: float = 0.0) -> Annotation:
    return Annotation(
        container_id=1, sample_token="s", timestamp_us=ts,
        object_id=hash(uuid) & 0x7FFFFFFFFFFFFFFF,
        uuid=uuid, category_name=klass, detection_class=klass,
        translation=(gx, gy, 0.0), size=(4.0, 1.8, 1.5),
        rotation=(1.0, 0.0, 0.0, 0.0), yaw_rad=0.0,
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


# ── PANDASET_SOURCE constant ─────────────────────────────────────────────────


class TestPandasetSource:
    def test_constant_is_lidar_camera_no_radar(self):
        assert PANDASET_SOURCE == "lidar|camera"
        assert "radar" not in PANDASET_SOURCE.split("|")


class TestGeometryReexports:
    def test_azimuth_basic(self):
        assert azimuth_sector(1.0, 0.0) == "front"

    def test_lane_basic(self):
        assert lane_offset(0.0) == 0


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
        assert r["source"] == "lidar|camera"
        assert r["relative_velocity_ms"] is None

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
        assert rows[1]["relative_velocity_ms"] == pytest.approx(-20.0, rel=1e-9)


class TestMapAllScenes:
    def test_yields_across_scenes(self):
        loader = FakeLoader(samples=[], anns={})
        assert list(map_all_scenes(loader, [_scene(), _scene()])) == []
