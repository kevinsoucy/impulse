"""KPI computation helpers for the sensor-coverage comparison notebook.

Adapter-independent: operates on the `source` column convention defined in
`object_tracks` (pipe-delimited modality string, e.g. "lidar|radar|camera").
Every adapter that populates `object_tracks.source` with that convention can
use these helpers as-is.

All functions operate on pandas DataFrames collected from Spark — suitable for
the NuScenes mini dataset. For trainval-scale, convert to Spark aggregations.
"""

from __future__ import annotations

import pandas as pd


def sensor_flags(source: str) -> dict[str, bool]:
    """Parse a pipe-delimited source string into per-modality booleans.

    >>> sensor_flags("lidar|radar|camera")
    {'lidar': True, 'radar': True, 'camera': True}
    >>> sensor_flags("lidar|camera")
    {'lidar': True, 'radar': False, 'camera': True}
    >>> sensor_flags("camera")
    {'lidar': False, 'radar': False, 'camera': True}
    """
    parts = set(source.split("|")) if source else set()
    return {
        "lidar": "lidar" in parts,
        "radar": "radar" in parts,
        "camera": "camera" in parts,
    }


def with_sensor_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with three added bool columns: lidar_seen, radar_seen, camera_seen."""
    dicts = df["source"].apply(sensor_flags)
    if dicts.empty:
        flags = pd.DataFrame(
            {"lidar": pd.Series(dtype=bool), "radar": pd.Series(dtype=bool), "camera": pd.Series(dtype=bool)},
            index=df.index,
        )
    else:
        flags = dicts.apply(pd.Series)
    return pd.concat(
        [
            df,
            flags.rename(columns={
                "lidar": "lidar_seen",
                "radar": "radar_seen",
                "camera": "camera_seen",
            }),
        ],
        axis=1,
    )


def detection_counts_by_source(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["container_id", "source"])
        .size()
        .reset_index(name="detection_count")
        .sort_values("detection_count", ascending=False)
        .reset_index(drop=True)
    )


def distance_stats_by_modality(df: pd.DataFrame) -> pd.DataFrame:
    """df must already have lidar_seen / radar_seen / camera_seen columns."""
    records = []
    for modality, col in [("lidar", "lidar_seen"), ("radar", "radar_seen"), ("camera", "camera_seen")]:
        if col not in df.columns:
            continue
        sub = df[df[col]]["distance_m"].dropna()
        if sub.empty:
            continue
        records.append({
            "modality": modality,
            "n_detections": int(len(sub)),
            "mean_m": round(float(sub.mean()), 1),
            "p50_m": round(float(sub.median()), 1),
            "p90_m": round(float(sub.quantile(0.9)), 1),
            "max_m": round(float(sub.max()), 1),
        })
    return pd.DataFrame(records)


def class_coverage_by_modality(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for modality, col in [("lidar", "lidar_seen"), ("radar", "radar_seen"), ("camera", "camera_seen")]:
        if col not in df.columns:
            continue
        counts = (
            df[df[col]]
            .groupby("detection_class")
            .size()
            .reset_index(name="count")
        )
        counts.insert(0, "modality", modality)
        rows.append(counts)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def detection_gaps(
    tracks_df: pd.DataFrame,
    events_df: pd.DataFrame,
) -> pd.DataFrame:
    """Find event windows where lidar or radar had zero detections."""
    records = []
    for _, ev in events_df.iterrows():
        cid = int(ev["container_id"])
        start = int(ev["start_ts"])
        end = int(ev["end_ts"])
        window = tracks_df[
            (tracks_df["container_id"] == cid)
            & (tracks_df["frame_ts"] >= start)
            & (tracks_df["frame_ts"] <= end)
        ]
        lidar_n = int(window["lidar_seen"].sum()) if "lidar_seen" in window.columns else 0
        radar_n = int(window["radar_seen"].sum()) if "radar_seen" in window.columns else 0
        records.append({
            "container_id": cid,
            "start_ts": start,
            "end_ts": end,
            "total_detections": len(window),
            "lidar_detections": lidar_n,
            "radar_detections": radar_n,
            "lidar_gap": lidar_n == 0,
            "radar_gap": radar_n == 0,
        })
    return pd.DataFrame(records)
