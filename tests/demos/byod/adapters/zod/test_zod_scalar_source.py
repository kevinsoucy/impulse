"""Unit tests for demos/byod/adapters/zod/scalar_source.py.

OXTS HDF5 decode goes through `ZodLoader.oxts_for_scene` which needs h5py at
runtime. Tests that exercise the real HDF5 path are gated by `h5py` availability;
the rest use a FakeLoader stub so the bulk of the suite is dependency-free.
"""

import pytest

from adapters.zod.loader import (
    DERIVED_CHANNEL_IDS,
    OXTS_UNIT_SCALES,
    Annotation,
    Sample,
    Scene,
)
from adapters.zod.scalar_source import (
    ChannelValue,
    derive_all,
    derive_for_scene,
)


def _scene() -> Scene:
    return Scene(
        container_id=1,
        sequence_id="000001",
        name="000001",
        description="",
        nbr_samples=0,
        sequence_dir="/tmp/dummy",
        start_ts_us=1_000_000,
    )


def _sample(ts: int, frame_id: str = "f") -> Sample:
    return Sample(
        container_id=1,
        sample_token=f"000001#{frame_id}",
        timestamp_us=ts,
        sequence_id="000001",
        frame_id=frame_id,
        sensor_files={},
    )


def _ann(*, ts: int, klass: str, gx: float, gy: float = 0.0) -> Annotation:
    return Annotation(
        container_id=1, sample_token="s", timestamp_us=ts,
        object_id=42, uuid=f"u-{klass}", category_name=klass,
        detection_class=klass,
        translation=(gx, gy, 0.0), size=(4.0, 1.8, 1.5),
        rotation=(1.0, 0.0, 0.0, 0.0),
        num_lidar_pts=10, num_radar_pts=2, occlusion=0.1, radar_doppler_ms=-1.5,
    )


class FakeLoader:
    def __init__(self, samples, anns, oxts):
        self._s = samples
        self._a = anns
        self._o = oxts

    def samples_in_scene(self, scene): return iter(self._s)
    def annotations_in_sample(self, sample): return iter(self._a.get(sample.sample_token, []))
    def oxts_for_scene(self, scene): return self._o


# ── ChannelValue ─────────────────────────────────────────────────────────────


class TestChannelValue:
    def test_frozen(self):
        cv = ChannelValue(container_id=1, channel_id=1001, tstart=0, tend=100_000, value=30.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            cv.value = 99.0  # type: ignore[misc]


# ── OXTS decode (via stub) ───────────────────────────────────────────────────


class TestOxtsDecode:
    def test_speed_emits_rows(self):
        loader = FakeLoader(
            samples=[],
            anns={},
            oxts={
                "Vehicle_Speed_kph": [(1_000_000, 36.0), (1_100_000, 39.0)],
            },
        )
        rows = derive_for_scene(loader, _scene())
        speed = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Vehicle_Speed_kph"]]
        assert len(speed) == 2
        assert speed[0].tstart == 1_000_000
        assert speed[0].tend == 1_100_000
        assert speed[0].value == 36.0

    def test_unknown_channel_keys_ignored(self):
        loader = FakeLoader(samples=[], anns={}, oxts={"Unknown_Channel": [(0, 0.0)]})
        assert derive_for_scene(loader, _scene()) == []

    def test_single_point_extends_with_default_interval(self):
        loader = FakeLoader(samples=[], anns={}, oxts={"Heading_deg": [(1_000_000, 90.0)]})
        rows = derive_for_scene(loader, _scene())
        head = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Heading_deg"]]
        assert len(head) == 1
        assert head[0].tend - head[0].tstart > 0

    def test_unsorted_input_sorted_before_intervals(self):
        loader = FakeLoader(
            samples=[],
            anns={},
            oxts={"Vehicle_Speed_kph": [(1_100_000, 39.0), (1_000_000, 36.0)]},
        )
        rows = derive_for_scene(loader, _scene())
        speed = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Vehicle_Speed_kph"]]
        assert speed[0].tstart == 1_000_000
        assert speed[0].value == 36.0


# ── Detection aggregates ─────────────────────────────────────────────────────


class TestDetectionAggregates:
    def test_per_frame_counts(self):
        s1 = _sample(1_000_000, "f1")
        s2 = _sample(1_100_000, "f2")
        loader = FakeLoader(
            samples=[s1, s2],
            anns={
                s1.sample_token: [
                    _ann(ts=1_000_000, klass="pedestrian", gx=10.0),
                    _ann(ts=1_000_000, klass="pedestrian", gx=20.0),
                    _ann(ts=1_000_000, klass="car", gx=15.0),
                ],
                s2.sample_token: [],
            },
            oxts={},
        )
        rows = derive_for_scene(loader, _scene())
        ped = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Pedestrian_Count"]]
        first_sample = [r for r in ped if r.tstart == 1_000_000]
        assert first_sample[0].value == 2.0

    def test_no_targets_emit_no_distance_rows(self):
        s1 = _sample(1_000_000, "f1")
        s2 = _sample(1_100_000, "f2")
        loader = FakeLoader(
            samples=[s1, s2],
            anns={s1.sample_token: [], s2.sample_token: []},
            oxts={},
        )
        rows = derive_for_scene(loader, _scene())
        nearest = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Pedestrian_Nearest_Distance_m"]]
        assert nearest == []

    def test_vehicle_count_front_filters_forward_x(self):
        s1 = _sample(1_000_000, "f1")
        loader = FakeLoader(
            samples=[s1],
            anns={s1.sample_token: [
                _ann(ts=1_000_000, klass="car", gx=10.0),
                _ann(ts=1_000_000, klass="car", gx=-5.0),
            ]},
            oxts={},
        )
        rows = derive_for_scene(loader, _scene())
        front = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Vehicle_Count_Front"]]
        assert front[0].value == 1.0


# ── derive_all ───────────────────────────────────────────────────────────────


class TestDeriveAll:
    def test_yields_concatenation_across_scenes(self):
        loader = FakeLoader(samples=[], anns={}, oxts={})
        assert list(derive_all(loader, [_scene(), _scene()])) == []


# ── Unit-scale sanity ────────────────────────────────────────────────────────


class TestUnitScales:
    def test_speed_scale_is_kph(self):
        assert OXTS_UNIT_SCALES["Vehicle_Speed_kph"] == pytest.approx(3.6, abs=1e-9)

    def test_heading_scale_converts_radians_to_degrees(self):
        import math
        assert OXTS_UNIT_SCALES["Heading_deg"] == pytest.approx(180.0 / math.pi, rel=1e-4)
