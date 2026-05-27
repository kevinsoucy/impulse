"""Derive Phase 1 scalar channels from NuScenes data.

NuScenes has no CAN bus / MDF4. This module synthesizes equivalent scalar
channels from what NuScenes does provide:

  Kinematic channels         ← differentiated ego pose (IMU+GPS measurements)
  Detection-aggregate channels ← aggregated from ground-truth annotations

Output rows match Impulse's `channels` schema:
    container_id LONG, channel_id INT, tstart LONG, tend LONG, value DOUBLE

Each channel is RLE-encoded: one row per sample interval. For this demo we
sample at NuScenes keyframe rate (2 Hz) — real CAN ingest would sample at
the ECU's native rate (typically 10–100 Hz).
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from mda_query_engine.perception.geometry import (
    rotation_matrix_from_quat as _rotation_matrix_from_quat,
    yaw_from_quat,
)
from mda_query_engine.perception.geometry import global_to_ego as _global_to_ego_core

from .loader import (
    DERIVED_CHANNEL_IDS,
    Annotation,
    EgoPose,
    NuScenesLoader,
    Sample,
    Scene,
)


def global_to_ego(point_global: np.ndarray, ego: EgoPose) -> np.ndarray:
    """NuScenes-shaped wrapper around `mda_query_engine.perception.geometry.global_to_ego`.

    Accepts an `EgoPose` so existing call sites don't have to unpack translation
    and rotation. The core function takes them separately so it has no
    dependency on the demo's `EgoPose` type.
    """
    return _global_to_ego_core(
        point_global=point_global,
        ego_translation=ego.translation,
        ego_rotation_wxyz=ego.rotation,
    )


# ── Channel value record ────────────────────────────────────────────────────


@dataclass(frozen=True)
class ChannelValue:
    """Row to write to Impulse `channels`. tstart/tend in microseconds; value is DOUBLE."""
    container_id: int
    channel_id: int
    tstart: int
    tend: int
    value: float


# ── Per-scene derivation ────────────────────────────────────────────────────


def derive_for_scene(loader: NuScenesLoader, scene: Scene) -> list[ChannelValue]:
    """Return all Phase 1 channel rows for one scene at keyframe rate (2 Hz)."""
    out: list[ChannelValue] = []

    samples: list[Sample] = list(loader.samples_in_scene(scene))
    if len(samples) < 2:
        return out

    ego_poses: list[EgoPose] = [loader.ego_pose_for_sample(s) for s in samples]
    annotations_by_sample: list[list[Annotation]] = [
        list(loader.annotations_in_sample(s)) for s in samples
    ]

    timestamps = [s.timestamp_us for s in samples]
    intervals: list[tuple[int, int]] = []
    for i, ts in enumerate(timestamps):
        if i + 1 < len(timestamps):
            intervals.append((ts, timestamps[i + 1]))
        else:
            if len(intervals) > 0:
                median_dt = intervals[len(intervals) // 2][1] - intervals[len(intervals) // 2][0]
            else:
                median_dt = 500_000
            intervals.append((ts, ts + median_dt))

    # ── Kinematic channels ──
    for i, (ego, (tstart, tend)) in enumerate(zip(ego_poses, intervals)):
        dt_s = max((tend - tstart) / 1e6, 1e-6)

        if i + 1 < len(ego_poses):
            other = ego_poses[i + 1]
        else:
            other = ego_poses[i - 1]
        delta = np.array(other.translation) - np.array(ego.translation)
        speed_ms = float(np.linalg.norm(delta)) / max(abs(other.timestamp_us - ego.timestamp_us) / 1e6, 1e-6)
        speed_kph = speed_ms * 3.6
        out.append(ChannelValue(scene.container_id, DERIVED_CHANNEL_IDS["Vehicle_Speed_kph"], tstart, tend, speed_kph))

        if 0 < i < len(ego_poses) - 1:
            p_prev = np.array(ego_poses[i - 1].translation)
            p_curr = np.array(ego.translation)
            p_next = np.array(ego_poses[i + 1].translation)
            dt_prev = max((ego.timestamp_us - ego_poses[i - 1].timestamp_us) / 1e6, 1e-6)
            dt_next = max((ego_poses[i + 1].timestamp_us - ego.timestamp_us) / 1e6, 1e-6)
            v_prev = (p_curr - p_prev) / dt_prev
            v_next = (p_next - p_curr) / dt_next
            accel_vec = (v_next - v_prev) / ((dt_prev + dt_next) / 2)
            R = _rotation_matrix_from_quat(*ego.rotation)
            heading_global = R[:, 0]
            accel_long = float(np.dot(accel_vec, heading_global))
        else:
            accel_long = 0.0
        out.append(ChannelValue(scene.container_id, DERIVED_CHANNEL_IDS["Vehicle_Accel_Longitudinal_ms2"], tstart, tend, accel_long))

        # NuScenes vehicle: Renault Zoe, wheelbase ≈ 2.59 m. Crude approximation, sufficient for demo.
        WHEELBASE_M = 2.59
        if i + 1 < len(ego_poses) and speed_ms > 1.0:
            yaw_curr = yaw_from_quat(*ego.rotation)
            yaw_next = yaw_from_quat(*ego_poses[i + 1].rotation)
            d_yaw = yaw_next - yaw_curr
            while d_yaw > math.pi:
                d_yaw -= 2 * math.pi
            while d_yaw < -math.pi:
                d_yaw += 2 * math.pi
            yaw_rate = d_yaw / dt_s
            steering_rad = math.atan2(WHEELBASE_M * yaw_rate, max(speed_ms, 1.0))
            steering_deg = math.degrees(steering_rad)
        else:
            steering_deg = 0.0
        out.append(ChannelValue(scene.container_id, DERIVED_CHANNEL_IDS["Steering_Angle_deg"], tstart, tend, steering_deg))

    # ── Detection-aggregate channels ──
    for ego, anns, (tstart, tend) in zip(ego_poses, annotations_by_sample, intervals):
        ego_frame: list[tuple[Annotation, np.ndarray]] = [
            (a, global_to_ego(np.array(a.translation), ego)) for a in anns
        ]

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
                value=value,
            ))

    return out


def derive_all(loader: NuScenesLoader, scenes: Iterable[Scene]) -> Iterable[ChannelValue]:
    """Yield channel rows for every scene. Streaming-friendly for trainval."""
    for scene in scenes:
        yield from derive_for_scene(loader, scene)
