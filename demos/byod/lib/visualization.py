"""Visualization helpers for the BYOD demo.

These helpers live with the demo (rather than in the core engine) because they
consume the BYOD adapter Protocol's `visualize_format()` contract — the
per-adapter dispatch keys
(`camera_reader`, `lidar_reader`, `lidar_dtype`, `lidar_stride`) are part of
the BYOD contract, not part of the LakeVision data model.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


# ── Per-format readers ───────────────────────────────────────────────────────


def read_camera_image(file_path: str, reader: str) -> np.ndarray:
    """Decode a camera frame from disk into an `(H, W, 3)` RGB array.

    `reader` must match one of the values an adapter's
    `visualize_format()['camera_reader']` may return (`pil`, `opencv`).
    """
    if reader == "pil":
        from PIL import Image
        return np.array(Image.open(file_path))
    if reader == "opencv":
        import cv2
        img = cv2.imread(file_path)
        return img[:, :, ::-1] if img is not None else None
    raise ValueError(f"unknown camera_reader: {reader!r}")


def read_lidar_xyzi(
    file_path: str,
    reader: str,
    dtype: str = "float32",
    stride: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode a LiDAR scan from disk into `(x, y, z, intensity)` arrays.

    `reader` must match one of the values an adapter's
    `visualize_format()['lidar_reader']` may return. The `dtype` and
    `stride` parameters are read from the same `visualize_format()` dict
    for the binary `fromfile` path; `npz` ignores them.
    """
    if reader == "fromfile":
        arr = np.fromfile(file_path, dtype=np.dtype(dtype)).reshape(-1, stride)
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if reader == "npz":
        loaded = np.load(file_path)
        if isinstance(loaded, np.lib.npyio.NpzFile):
            arr = loaded[loaded.files[0]]
        else:
            arr = loaded
        return arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
    if reader == "pickle_gz":
        import gzip
        import pickle
        with gzip.open(file_path, "rb") as f:
            df = pickle.load(f)
        return df["x"].to_numpy(), df["y"].to_numpy(), df["z"].to_numpy(), df["i"].to_numpy()
    raise ValueError(f"unknown lidar_reader: {reader!r}")


# ── channel_tags-based channel resolution ────────────────────────────────────


def resolve_front_camera_channel_id(
    spark: SparkSession,
    channel_tags_table: str,
    container_id: int,
) -> int:
    """Pick the channel_id of the forward-facing camera for `container_id`.

    Walks `channel_tags` for `channel_name` values containing `CAM` and
    prefers a name with `FRONT` (excluding `FRONT_LEFT`/`FRONT_RIGHT`). Falls
    back to the first camera channel if no front-facing match is found.
    """
    cam_channels = (
        spark.read.table(channel_tags_table)
        .filter((F.col("container_id") == container_id) & (F.col("key") == "channel_name"))
        .filter(F.upper(F.col("value")).contains("CAM"))
        .select("channel_id", "value")
        .collect()
    )
    front = [
        c for c in cam_channels
        if "FRONT" in c.value.upper()
        and "LEFT" not in c.value.upper()
        and "RIGHT" not in c.value.upper()
    ]
    if front:
        return int(front[0].channel_id)
    if cam_channels:
        return int(cam_channels[0].channel_id)
    raise RuntimeError(
        f"No camera channel found in {channel_tags_table} for container_id={container_id}"
    )


def resolve_lidar_channel_id(
    spark: SparkSession,
    channel_tags_table: str,
    container_id: int,
) -> int:
    """Pick the channel_id of a LiDAR sensor for `container_id`.

    Picks the first channel whose `channel_name` contains `LIDAR`. For
    adapters with multiple physical LiDARs (e.g. PandaSet's spinning +
    solid-state), the first matching channel is returned.
    """
    row = (
        spark.read.table(channel_tags_table)
        .filter((F.col("container_id") == container_id) & (F.col("key") == "channel_name"))
        .filter(F.upper(F.col("value")).contains("LIDAR"))
        .select("channel_id", "value")
        .first()
    )
    if row is None:
        raise RuntimeError(
            f"No LiDAR channel found in {channel_tags_table} for container_id={container_id}"
        )
    return int(row.channel_id)


# ── Adapter visualize_format helper ──────────────────────────────────────────


def adapter_visualize_format(adapter: Any) -> dict:
    """Return an adapter's `visualize_format()` dict with safe defaults.

    Centralizes the default values so notebooks and helpers don't repeat the
    fallbacks.
    """
    fmt = adapter.visualize_format() or {}
    return {
        "camera_reader": fmt.get("camera_reader", "pil"),
        "lidar_reader":  fmt.get("lidar_reader",  "fromfile"),
        "lidar_dtype":   fmt.get("lidar_dtype",   "float32"),
        "lidar_stride":  int(fmt.get("lidar_stride", 5)),
    }
