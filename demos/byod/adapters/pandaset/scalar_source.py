"""Decode PandaSet Phase 1 scalar channels from `meta/gps.csv`.

PandaSet's public release exposes only speed + heading + lat + lon in its GPS
CSV — no accelerometer/gyro. That's a smaller scalar surface than A2D2 or ZOD
but it still demonstrates TSAL speed thresholding and TSAL heading-change
event detection in notebook 05.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from .loader import (
    DERIVED_CHANNEL_IDS,
    Annotation,
    PandaSetLoader,
    Sample,
    Scene,
)


@dataclass(frozen=True)
class ChannelValue:
    container_id: int
    channel_id: int
    tstart: int
    tend: int
    value: float


def derive_for_scene(loader: PandaSetLoader, scene: Scene) -> list[ChannelValue]:
    out: list[ChannelValue] = []
    out.extend(_decode_gps(loader, scene))
    out.extend(_compute_detection_aggregates(loader, scene))
    return out


def _decode_gps(loader: PandaSetLoader, scene: Scene) -> list[ChannelValue]:
    series_by_channel = loader.gps_for_scene(scene)
    out: list[ChannelValue] = []
    for channel_name, series in series_by_channel.items():
        channel_id = DERIVED_CHANNEL_IDS.get(channel_name)
        if channel_id is None or not series:
            continue
        sorted_series = sorted(series, key=lambda p: p[0])
        if len(sorted_series) == 1:
            ts, value = sorted_series[0]
            out.append(ChannelValue(scene.container_id, channel_id, ts, ts + 100_000, value))
            continue
        intervals: list[tuple[int, int]] = []
        for i, (ts, _) in enumerate(sorted_series):
            if i + 1 < len(sorted_series):
                intervals.append((ts, sorted_series[i + 1][0]))
            else:
                deltas = [intervals[k][1] - intervals[k][0] for k in range(len(intervals))]
                median_dt = sorted(deltas)[len(deltas) // 2] if deltas else 100_000
                intervals.append((ts, ts + median_dt))
        for (tstart, tend), (_, value) in zip(intervals, sorted_series):
            out.append(ChannelValue(scene.container_id, channel_id, tstart, tend, value))
    return out


def _compute_detection_aggregates(loader: PandaSetLoader, scene: Scene) -> list[ChannelValue]:
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
                median_dt = 100_000
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


def derive_all(loader: PandaSetLoader, scenes: Iterable[Scene]) -> Iterable[ChannelValue]:
    for scene in scenes:
        yield from derive_for_scene(loader, scene)


__all__ = ["ChannelValue", "derive_all", "derive_for_scene"]
