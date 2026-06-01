"""Decode A2D2 Phase 1 scalar channels.

Unlike NuScenes (where scalars are synthesized from ego pose because there's
no bus stream), A2D2 publishes the real recorded ECU signals in
`{scene}/bus/bus_signals.json`. This module:

1. **Bus signals → channels:** reads the bus_signals.json file once per scene
   and emits one `ChannelValue` per (signal, timestamp) pair, mapped through
   `BUS_SIGNAL_TO_CHANNEL` to the canonical LakeVision channel ids.
   No synthesis, no derivatives — values come straight from the recording.

2. **Detection-aggregate channels:** computed exactly like NuScenes,
   aggregated per keyframe from the per-frame annotations.

The output schema matches Impulse's `channels`:
    container_id LONG, channel_id INT, tstart LONG, tend LONG, value DOUBLE.
Adjacent identical values are NOT RLE-merged here — Impulse's downstream
compaction handles that.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .loader import (
    BUS_SIGNAL_TO_CHANNEL,
    DERIVED_CHANNEL_IDS,
    A2D2Loader,
    Annotation,
    EgoPose,
    Sample,
    Scene,
)


@dataclass(frozen=True)
class ChannelValue:
    """Row to write to Impulse `channels`. tstart/tend in microseconds; value DOUBLE."""
    container_id: int
    channel_id: int
    tstart: int
    tend: int
    value: float


# ── Per-scene derivation ────────────────────────────────────────────────────


def derive_for_scene(loader: A2D2Loader, scene: Scene) -> list[ChannelValue]:
    """Return all Phase 1 channel rows for one A2D2 scene."""
    rows: list[ChannelValue] = []
    rows.extend(_decode_bus_signals(loader, scene))
    rows.extend(_compute_detection_aggregates(loader, scene))
    return rows


def _decode_bus_signals(loader: A2D2Loader, scene: Scene) -> list[ChannelValue]:
    """Decode bus_signals.json → ChannelValue rows.

    Each (timestamp, value) becomes one row spanning [ts, next_ts) per signal.
    The last sample's interval is extended by the median Δt of its signal.
    """
    signals = loader.bus_signals_for_scene(scene)
    out: list[ChannelValue] = []
    for raw_key, lakevision_name in BUS_SIGNAL_TO_CHANNEL.items():
        series = signals.get(raw_key)
        if not series:
            continue
        channel_id = DERIVED_CHANNEL_IDS[lakevision_name]
        sorted_series = sorted(series, key=lambda p: p[0])
        if len(sorted_series) == 1:
            ts, value = sorted_series[0]
            out.append(ChannelValue(scene.container_id, channel_id, ts, ts + 20_000, value))
            continue
        intervals: list[tuple[int, int]] = []
        for i, (ts, _v) in enumerate(sorted_series):
            if i + 1 < len(sorted_series):
                intervals.append((ts, sorted_series[i + 1][0]))
            else:
                # Extend the last interval by the median Δt to avoid a zero-length tail.
                deltas = [intervals[k][1] - intervals[k][0] for k in range(len(intervals))]
                median_dt = sorted(deltas)[len(deltas) // 2] if deltas else 20_000
                intervals.append((ts, ts + median_dt))
        for (tstart, tend), (_, value) in zip(intervals, sorted_series):
            out.append(ChannelValue(scene.container_id, channel_id, tstart, tend, value))
    return out


def _compute_detection_aggregates(loader: A2D2Loader, scene: Scene) -> list[ChannelValue]:
    """Per-keyframe detection counts and nearest-distance summaries.

    A2D2 annotations are already in the vehicle (ego) frame so distances are
    just the Euclidean norm of the box center. Output mirrors the NuScenes
    adapter's detection-aggregate channels for cross-adapter comparability.
    """
    samples: list[Sample] = list(loader.samples_in_scene(scene))
    if not samples:
        return []

    annotations_by_sample: list[list[Annotation]] = [
        list(loader.annotations_in_sample(s)) for s in samples
    ]
    timestamps = [s.timestamp_us for s in samples]
    intervals: list[tuple[int, int]] = []
    for i, ts in enumerate(timestamps):
        if i + 1 < len(timestamps):
            intervals.append((ts, timestamps[i + 1]))
        else:
            if intervals:
                median_dt = intervals[len(intervals) // 2][1] - intervals[len(intervals) // 2][0]
            else:
                median_dt = 100_000  # 10 Hz default
            intervals.append((ts, ts + median_dt))

    out: list[ChannelValue] = []
    for anns, (tstart, tend) in zip(annotations_by_sample, intervals):
        ego_frame = [(a, np.array(a.translation, dtype=float)) for a in anns]

        def stats(class_name: str, front_only: bool = False) -> tuple[int, float | None]:
            ds = [
                float(np.linalg.norm(p))
                for a, p in ego_frame
                if a.detection_class == class_name and (not front_only or p[0] > 0)
            ]
            return len(ds), (min(ds) if ds else None)

        ped = stats("pedestrian")
        car_all = stats("car")
        car_front = stats("car", front_only=True)
        cyc = stats("cyclist")
        rows = {
            "Pedestrian_Count":              float(ped[0]),
            "Pedestrian_Nearest_Distance_m": ped[1],
            "Vehicle_Count_Front":           float(car_front[0]),
            "Vehicle_Nearest_Distance_m":    car_all[1],
            "Cyclist_Count":                 float(cyc[0]),
            "Cyclist_Nearest_Distance_m":    cyc[1],
        }
        for channel_name, value in rows.items():
            if value is None:
                continue
            out.append(ChannelValue(
                container_id=scene.container_id,
                channel_id=DERIVED_CHANNEL_IDS[channel_name],
                tstart=tstart,
                tend=tend,
                value=float(value),
            ))
    return out


def derive_all(loader: A2D2Loader, scenes: Iterable[Scene]) -> Iterable[ChannelValue]:
    """Yield channel rows for every scene. Streaming-friendly."""
    for scene in scenes:
        yield from derive_for_scene(loader, scene)


# `EgoPose` is imported only so external test modules can re-use the typed
# dataclass under a stable namespace; this adapter does not compute kinematic
# scalars from ego pose (A2D2 publishes them directly).
__all__ = [
    "BUS_SIGNAL_TO_CHANNEL",
    "ChannelValue",
    "EgoPose",
    "derive_all",
    "derive_for_scene",
]
