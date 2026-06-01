"""Unit tests for demos/byod/adapters/a2d2/scalar_source.py."""

import pytest

from adapters.a2d2.loader import (
    BUS_SIGNAL_TO_CHANNEL,
    DERIVED_CHANNEL_IDS,
    Annotation,
    Sample,
    Scene,
)
from adapters.a2d2.scalar_source import (
    ChannelValue,
    derive_all,
    derive_for_scene,
)


def _scene() -> Scene:
    return Scene(
        container_id=1,
        scene_id="20180807_145028",
        name="20180807_145028",
        description="",
        nbr_samples=0,
        scene_dir="/tmp/dummy",
        start_ts_us=1_000_000,
    )


def _sample(ts: int, idx: int) -> Sample:
    return Sample(
        container_id=1,
        sample_token=f"20180807_145028#{idx:09d}",
        timestamp_us=ts,
        scene_id="20180807_145028",
        frame_index=idx,
        sensor_files={},
    )


def _ann(*, ts: int, klass: str, gx: float, gy: float = 0.0) -> Annotation:
    return Annotation(
        container_id=1,
        sample_token="s",
        timestamp_us=ts,
        object_id=42,
        box_key="box_0",
        category_name=klass.capitalize(),
        detection_class=klass,
        translation=(gx, gy, 0.0),
        size=(4.0, 1.8, 1.5),
        rotation=(1.0, 0.0, 0.0, 0.0),
        truncation=0.0,
        occlusion=0.0,
    )


class FakeLoader:
    def __init__(self, samples, anns_by_sample, bus_signals):
        self._samples = samples
        self._anns = anns_by_sample
        self._bus = bus_signals

    def samples_in_scene(self, scene):
        return iter(self._samples)

    def annotations_in_sample(self, sample):
        return iter(self._anns.get(sample.sample_token, []))

    def bus_signals_for_scene(self, scene):
        return self._bus


# ── Channel value record ────────────────────────────────────────────────────


class TestChannelValue:
    def test_frozen(self):
        cv = ChannelValue(container_id=1, channel_id=1001, tstart=0, tend=100_000, value=30.0)
        with pytest.raises(Exception):  # FrozenInstanceError
            cv.value = 99.0  # type: ignore[misc]


# ── Bus-signal decode ───────────────────────────────────────────────────────


class TestBusSignalDecode:
    def test_known_signal_maps_to_adas_channel(self):
        loader = FakeLoader(
            samples=[],
            anns_by_sample={},
            bus_signals={
                "vehicle_speed": [(1_000_000, 30.0), (1_100_000, 32.0)],
            },
        )
        rows = derive_for_scene(loader, _scene())
        speed_rows = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Vehicle_Speed_kph"]]
        assert len(speed_rows) == 2
        assert speed_rows[0].tstart == 1_000_000
        assert speed_rows[0].tend == 1_100_000
        assert speed_rows[0].value == 30.0

    def test_unknown_signal_keys_ignored(self):
        loader = FakeLoader(
            samples=[],
            anns_by_sample={},
            bus_signals={
                "unused_signal": [(0, 0.0)],
            },
        )
        rows = derive_for_scene(loader, _scene())
        assert rows == []

    def test_single_value_signal_extends_by_default_interval(self):
        loader = FakeLoader(
            samples=[],
            anns_by_sample={},
            bus_signals={"vehicle_speed": [(1_000_000, 30.0)]},
        )
        rows = derive_for_scene(loader, _scene())
        speed_rows = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Vehicle_Speed_kph"]]
        assert len(speed_rows) == 1
        assert speed_rows[0].tend - speed_rows[0].tstart > 0

    def test_signals_emit_distinct_channel_ids(self):
        loader = FakeLoader(
            samples=[],
            anns_by_sample={},
            bus_signals={
                "vehicle_speed":  [(0, 10.0)],
                "acceleration_x": [(0, 0.5)],
                "steering_angle_calculated": [(0, -5.0)],
            },
        )
        rows = derive_for_scene(loader, _scene())
        ids_seen = {r.channel_id for r in rows}
        assert DERIVED_CHANNEL_IDS["Vehicle_Speed_kph"] in ids_seen
        assert DERIVED_CHANNEL_IDS["Vehicle_Accel_Longitudinal_ms2"] in ids_seen
        assert DERIVED_CHANNEL_IDS["Steering_Angle_deg"] in ids_seen

    def test_unsorted_input_is_sorted_before_intervals(self):
        loader = FakeLoader(
            samples=[],
            anns_by_sample={},
            bus_signals={"vehicle_speed": [(1_100_000, 32.0), (1_000_000, 30.0)]},
        )
        rows = derive_for_scene(loader, _scene())
        speed_rows = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Vehicle_Speed_kph"]]
        # The first row (sorted by ts) should carry the earlier timestamp + value.
        assert speed_rows[0].tstart == 1_000_000
        assert speed_rows[0].value == 30.0


# ── Detection aggregates ─────────────────────────────────────────────────────


class TestDetectionAggregates:
    def test_per_frame_counts(self):
        s1 = _sample(1_000_000, 0)
        s2 = _sample(1_100_000, 1)
        loader = FakeLoader(
            samples=[s1, s2],
            anns_by_sample={
                s1.sample_token: [
                    _ann(ts=s1.timestamp_us, klass="pedestrian", gx=10.0),
                    _ann(ts=s1.timestamp_us, klass="pedestrian", gx=20.0),
                    _ann(ts=s1.timestamp_us, klass="car", gx=15.0),
                ],
                s2.sample_token: [],
            },
            bus_signals={},
        )
        rows = derive_for_scene(loader, _scene())
        ped = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Pedestrian_Count"]]
        sample_a_ped = [r for r in ped if r.tstart == 1_000_000]
        assert sample_a_ped[0].value == 2.0

    def test_nearest_distance_uses_vehicle_frame_norm(self):
        s1 = _sample(1_000_000, 0)
        s2 = _sample(1_100_000, 1)
        loader = FakeLoader(
            samples=[s1, s2],
            anns_by_sample={
                s1.sample_token: [
                    _ann(ts=s1.timestamp_us, klass="pedestrian", gx=10.0, gy=0.0),
                    _ann(ts=s1.timestamp_us, klass="pedestrian", gx=20.0, gy=0.0),
                ],
                s2.sample_token: [],
            },
            bus_signals={},
        )
        rows = derive_for_scene(loader, _scene())
        nearest = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Pedestrian_Nearest_Distance_m"]]
        sample_a_nearest = [r for r in nearest if r.tstart == 1_000_000]
        assert sample_a_nearest[0].value == pytest.approx(10.0, rel=1e-9)

    def test_frames_with_no_targets_emit_no_distance_rows(self):
        s1 = _sample(1_000_000, 0)
        s2 = _sample(1_100_000, 1)
        loader = FakeLoader(
            samples=[s1, s2],
            anns_by_sample={s1.sample_token: [], s2.sample_token: []},
            bus_signals={},
        )
        rows = derive_for_scene(loader, _scene())
        nearest = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Pedestrian_Nearest_Distance_m"]]
        assert nearest == []

    def test_vehicle_count_front_uses_forward_x_filter(self):
        s1 = _sample(1_000_000, 0)
        loader = FakeLoader(
            samples=[s1],
            anns_by_sample={
                s1.sample_token: [
                    _ann(ts=s1.timestamp_us, klass="car", gx=10.0),
                    _ann(ts=s1.timestamp_us, klass="car", gx=-5.0),
                ],
            },
            bus_signals={},
        )
        rows = derive_for_scene(loader, _scene())
        front = [r for r in rows if r.channel_id == DERIVED_CHANNEL_IDS["Vehicle_Count_Front"]]
        assert front[0].value == 1.0


# ── derive_all ───────────────────────────────────────────────────────────────


class TestDeriveAll:
    def test_yields_concatenation_across_scenes(self):
        loader = FakeLoader(samples=[], anns_by_sample={}, bus_signals={})
        assert list(derive_all(loader, [_scene(), _scene()])) == []


# ── Mapping sanity ───────────────────────────────────────────────────────────


class TestMapping:
    def test_every_bus_signal_target_has_an_id(self):
        for raw_key, adas_name in BUS_SIGNAL_TO_CHANNEL.items():
            assert adas_name in DERIVED_CHANNEL_IDS, (
                f"bus signal {raw_key} maps to {adas_name} which has no DERIVED_CHANNEL_ID"
            )
