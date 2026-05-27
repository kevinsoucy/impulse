"""Unit tests for demos/byod/adapters/pandaset/scalar_source.py."""

import pytest

from adapters.pandaset.loader import (
    DERIVED_CHANNEL_IDS,
    Annotation,
    Sample,
    Scene,
)
from adapters.pandaset.scalar_source import (
    ChannelValue,
    derive_all,
    derive_for_scene,
)


def _scene() -> Scene:
    return Scene(container_id=1, sequence_id="001", name="001", description="",
                 nbr_samples=0, sequence_dir="/tmp", start_ts_us=1_000_000)


def _sample(ts: int, idx: int) -> Sample:
    return Sample(container_id=1, sample_token=f"001#{idx:02d}", timestamp_us=ts,
                  sequence_id="001", frame_index=idx, sensor_files={})


def _ann(*, ts: int, klass: str, gx: float, gy: float = 0.0) -> Annotation:
    return Annotation(
        container_id=1, sample_token="s", timestamp_us=ts,
        object_id=hash(klass) & 0x7FFFFFFFFFFFFFFF,
        uuid=f"u-{klass}", category_name=klass, detection_class=klass,
        translation=(gx, gy, 0.0), size=(4.0, 1.8, 1.5),
        rotation=(1.0, 0.0, 0.0, 0.0), yaw_rad=0.0,
    )


class FakeLoader:
    def __init__(self, samples, anns, gps):
        self._s = samples
        self._a = anns
        self._g = gps

    def samples_in_scene(self, scene): return iter(self._s)
    def annotations_in_sample(self, sample): return iter(self._a.get(sample.sample_token, []))
    def gps_for_scene(self, scene): return self._g


# ── ChannelValue ─────────────────────────────────────────────────────────────


class TestChannelValue:
    def test_frozen(self):
        cv = ChannelValue(container_id=1, channel_id=1001, tstart=0, tend=100_000, value=36.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            cv.value = 99.0  # type: ignore[misc]


# ── GPS decode ───────────────────────────────────────────────────────────────


class TestGpsDecode:
    def test_speed_emits_rows(self):
        loader = FakeLoader(
            samples=[], anns={},
            gps={"Vehicle_Speed_kph": [(1_000_000, 36.0), (1_100_000, 40.0)]},
        )
        rows = derive_for_scene(loader, _scene())
        speed = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Vehicle_Speed_kph"]]
        assert len(speed) == 2
        assert speed[0].tstart == 1_000_000
        assert speed[0].tend == 1_100_000
        assert speed[0].value == 36.0

    def test_unknown_channel_keys_ignored(self):
        loader = FakeLoader(samples=[], anns={}, gps={"Unknown": [(0, 0.0)]})
        assert derive_for_scene(loader, _scene()) == []

    def test_unsorted_input_sorted_before_intervals(self):
        loader = FakeLoader(
            samples=[], anns={},
            gps={"Heading_deg": [(1_100_000, 50.0), (1_000_000, 45.0)]},
        )
        rows = derive_for_scene(loader, _scene())
        heading = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Heading_deg"]]
        assert heading[0].tstart == 1_000_000
        assert heading[0].value == 45.0


# ── Detection aggregates ─────────────────────────────────────────────────────


class TestDetectionAggregates:
    def test_per_frame_counts(self):
        s1 = _sample(1_000_000, 0)
        s2 = _sample(1_100_000, 1)
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
            gps={},
        )
        rows = derive_for_scene(loader, _scene())
        ped = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Pedestrian_Count"]]
        first_sample = [r for r in ped if r.tstart == 1_000_000]
        assert first_sample[0].value == 2.0


# ── derive_all ───────────────────────────────────────────────────────────────


class TestDeriveAll:
    def test_yields_across_scenes(self):
        loader = FakeLoader(samples=[], anns={}, gps={})
        assert list(derive_all(loader, [_scene(), _scene()])) == []
