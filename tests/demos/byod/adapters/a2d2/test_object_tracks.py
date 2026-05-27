"""Unit tests for demos/byod/adapters/a2d2/object_tracks.py."""

import pytest

from adapters.a2d2.loader import Annotation, Sample, Scene
from adapters.a2d2.object_tracks import (
    A2D2_SOURCE,
    LANE_WIDTH_M,
    azimuth_sector,
    lane_offset,
    map_all_scenes,
    map_scene_to_object_tracks,
)


def _ann(*, ts: int, klass: str, gx: float, gy: float = 0.0, gz: float = 0.0) -> Annotation:
    return Annotation(
        container_id=1,
        sample_token="s",
        timestamp_us=ts,
        object_id=42,
        box_key="box_0",
        category_name=klass.capitalize(),
        detection_class=klass,
        translation=(gx, gy, gz),
        size=(4.0, 1.8, 1.5),
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


class FakeLoader:
    def __init__(self, samples, anns_by_token):
        self._samples = samples
        self._anns = anns_by_token

    def samples_in_scene(self, scene):
        return iter(self._samples)

    def annotations_in_sample(self, sample):
        return iter(self._anns.get(sample.sample_token, []))


# ── A2D2_SOURCE constant ─────────────────────────────────────────────────────


class TestA2D2Source:
    def test_constant_is_pipe_free(self):
        # A2D2 has no radar — source string is single-valued by design.
        assert "|" not in A2D2_SOURCE
        assert A2D2_SOURCE == "ground_truth_camera_lidar"


# ── Geometry helpers re-exported from lakevision.geometry ────────────────────


class TestGeometryReexports:
    def test_azimuth_sector_basic(self):
        assert azimuth_sector(1.0, 0.0) == "front"
        assert azimuth_sector(-1.0, 0.0) == "rear"

    def test_lane_offset_basic(self):
        assert lane_offset(0.0) == 0
        assert lane_offset(LANE_WIDTH_M) == 1
        assert lane_offset(-LANE_WIDTH_M) == -1


# ── map_scene_to_object_tracks ───────────────────────────────────────────────


class TestMapSceneToObjectTracks:
    def test_empty_scene(self):
        loader = FakeLoader(samples=[], anns_by_token={})
        assert map_scene_to_object_tracks(loader, _scene()) == []

    def test_single_annotation_populates_expected_fields(self):
        s1 = _sample(0, "a")
        loader = FakeLoader(
            samples=[s1],
            anns_by_token={"a": [_ann(ts=0, klass="car", gx=10.0)]},
        )
        rows = map_scene_to_object_tracks(loader, _scene())
        assert len(rows) == 1
        r = rows[0]
        assert r["detection_class"] == "car"
        assert r["distance_m"] == pytest.approx(10.0, rel=1e-9)
        assert r["azimuth"] == "front"
        assert r["lane_offset"] == 0
        assert r["confidence"] == 1.0
        assert r["source"] == A2D2_SOURCE
        assert r["relative_velocity_ms"] is None

    def test_relative_velocity_is_always_none(self):
        s1 = _sample(0, "a")
        s2 = _sample(500_000, "b")
        loader = FakeLoader(
            samples=[s1, s2],
            anns_by_token={
                "a": [_ann(ts=0, klass="car", gx=20.0)],
                "b": [_ann(ts=500_000, klass="car", gx=10.0)],
            },
        )
        rows = map_scene_to_object_tracks(loader, _scene())
        # A2D2 has no cross-frame tracking — both rows must remain None.
        assert all(r["relative_velocity_ms"] is None for r in rows)

    def test_source_is_constant_across_classes(self):
        s1 = _sample(0, "a")
        loader = FakeLoader(
            samples=[s1],
            anns_by_token={
                "a": [
                    _ann(ts=0, klass="car", gx=10.0),
                    _ann(ts=0, klass="pedestrian", gx=5.0, gy=2.0),
                    _ann(ts=0, klass="cyclist", gx=8.0, gy=-1.0),
                ],
            },
        )
        rows = map_scene_to_object_tracks(loader, _scene())
        assert {r["source"] for r in rows} == {A2D2_SOURCE}

    def test_min_confidence_does_not_filter_ground_truth(self):
        s1 = _sample(0, "a")
        loader = FakeLoader(
            samples=[s1],
            anns_by_token={"a": [_ann(ts=0, klass="car", gx=10.0)]},
        )
        rows = map_scene_to_object_tracks(loader, _scene(), min_confidence=0.99)
        # confidence is hard-coded to 1.0 for ground truth, so high min still passes.
        assert len(rows) == 1


class TestMapAllScenes:
    def test_yields_across_multiple_scenes(self):
        loader = FakeLoader(samples=[], anns_by_token={})
        assert list(map_all_scenes(loader, [_scene(), _scene()])) == []
