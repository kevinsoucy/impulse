# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Visualize: see one event, then compare across modalities
# MAGIC
# MAGIC The "see and compare" beat. Two halves:
# MAGIC
# MAGIC 1. **Event walkthrough** — pick an event from the playlist, render the front-camera
# MAGIC    frame at the event's midpoint, plot the LiDAR scan top-down, overlay the
# MAGIC    `object_tracks` rows that fired the event.
# MAGIC 2. **Sensor KPI comparison** — across every event in the playlist, compare detection
# MAGIC    counts and distance distributions by sensor modality (LiDAR / radar / camera).
# MAGIC
# MAGIC Format dispatching (camera reader, LiDAR dtype/stride) comes from
# MAGIC `adapter.visualize_format()`, so this notebook works for any adapter that registers
# MAGIC sensible defaults.

# COMMAND ----------

# Initialize Spark as the first notebook command (serverless requirement).
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()

import os
import sys

try:
    _nb_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
    _demo_dir = os.path.dirname(_nb_path)
except Exception:
    _demo_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
if _demo_dir not in sys.path:
    sys.path.insert(0, _demo_dir)

import numpy as np
import matplotlib.pyplot as plt

from lib.adapter import resolve
from lib.byod_config import BYODConfig
from lib.sensor_kpi import (
    class_coverage_by_modality,
    detection_counts_by_source,
    detection_gaps,
    distance_stats_by_modality,
    with_sensor_flags,
)
from lib.visualization import (
    adapter_visualize_format,
    read_camera_image,
    read_lidar_xyzi,
    resolve_front_camera_channel_id,
    resolve_lidar_channel_id,
)
from lakevision import frame_nearest_to
from lakevision.geometry import azimuth_label_to_xy
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

dbutils.widgets.text("adapter",         "nuscenes",                   "Adapter name")
dbutils.widgets.text("dataset_version", "v1.0-mini",                  "Adapter-specific variant")
dbutils.widgets.text("catalog",         "main",                       "UC catalog")
dbutils.widgets.text("schema_prefix",   "lakevision_demo",            "Schema prefix")
dbutils.widgets.text("playlist_id",     "pedestrian_high_speed_v1",   "Playlist ID")
dbutils.widgets.text("event_id",        "",                           "Event ID (blank = first event in playlist)")

_adapter_name = dbutils.widgets.get("adapter")
_dataset_version = dbutils.widgets.get("dataset_version") or None
_catalog = dbutils.widgets.get("catalog")
_schema_prefix = dbutils.widgets.get("schema_prefix")
_playlist_id = dbutils.widgets.get("playlist_id")
_event_id_override = dbutils.widgets.get("event_id")

cfg = BYODConfig.for_adapter(
    adapter_name=_adapter_name,
    dataset_version=_dataset_version,
    catalog=_catalog,
    schema_prefix=_schema_prefix,
)
adapter = resolve(_adapter_name)(cfg)
fmt = adapter_visualize_format(adapter)

# COMMAND ----------

# MAGIC %md ## Part 1 — Event walkthrough
# MAGIC
# MAGIC After `run_all_job` completes, this notebook shows what the pipeline found —
# MAGIC first a gallery of every event in the playlist, then a single-event deep-dive,
# MAGIC then a frame-strip playback of that event over time.
# MAGIC
# MAGIC The single-event panel auto-picks the event with the most `object_tracks` rows
# MAGIC (the densest "scene"). To inspect a specific event, set the `event_id` widget
# MAGIC at the top of the notebook.

# COMMAND ----------

# Pull every event in the playlist as a Pandas frame for convenience.
events_df = (
    spark.read.table(cfg.t_playlist_items)
    .filter(F.col("playlist_id") == _playlist_id)
    .orderBy("container_id", "start_ts")
    .toPandas()
)

if events_df.empty:
    raise RuntimeError(
        f"No events found in playlist '{_playlist_id}'. "
        "Re-run 02_detect_events.py if the playlist is empty."
    )

# Object-track count per event window — used to pick the "most interesting" event.
object_tracks_pdf = (
    spark.sql(f"""
    SELECT ot.container_id, ot.frame_ts, ot.detection_class, ot.distance_m, ot.azimuth
    FROM   {cfg.t_object_tracks} ot
    JOIN   {cfg.t_playlist_items} pi
      ON   ot.container_id = pi.container_id
     AND   ot.frame_ts BETWEEN pi.start_ts AND pi.end_ts
    WHERE  pi.playlist_id = '{_playlist_id}'
    """)
    .toPandas()
)

def _objects_in_event(row):
    return int(
        (
            (object_tracks_pdf["container_id"] == int(row["container_id"]))
            & (object_tracks_pdf["frame_ts"] >= int(row["start_ts"]))
            & (object_tracks_pdf["frame_ts"] <= int(row["end_ts"]))
        ).sum()
    )

events_df["n_objects"] = events_df.apply(_objects_in_event, axis=1)

print(f"Playlist '{_playlist_id}': {len(events_df)} events, "
      f"{int(events_df['n_objects'].sum()):,} total object_tracks rows in their windows")

# COMMAND ----------

# MAGIC %md ### Pick the focal event
# MAGIC
# MAGIC `event_id` widget (if non-empty) → that exact event; otherwise → the event with
# MAGIC the most `object_tracks` rows in its window.

# COMMAND ----------

if _event_id_override:
    focal = events_df[events_df["event_id"] == _event_id_override]
    if focal.empty:
        raise RuntimeError(
            f"event_id={_event_id_override!r} not found in playlist '{_playlist_id}'"
        )
    focal_row = focal.iloc[0]
else:
    focal_row = events_df.loc[events_df["n_objects"].idxmax()]

event_id = focal_row["event_id"]
container_id = int(focal_row["container_id"])
start_us = int(focal_row["start_ts"])
end_us = int(focal_row["end_ts"])
mid_us = (start_us + end_us) // 2

print(f"Event ID:     {event_id}")
print(f"Container:    {container_id}")
print(f"Window:       [{start_us}, {end_us}]  ({(end_us - start_us) / 1e6:.2f}s)")
print(f"Objects in window: {int(focal_row['n_objects'])}")

# COMMAND ----------

# MAGIC %md ### Thumbnail gallery — every event in the playlist
# MAGIC
# MAGIC Front-camera frame at each event's midpoint, with the in-window object count
# MAGIC and window duration below. Grid is 3 columns wide.

# COMMAND ----------

import math as _math

_GRID_COLS = 3
_n_events = len(events_df)
_n_rows = _math.ceil(_n_events / _GRID_COLS)

fig_gallery, axes_gallery = plt.subplots(
    _n_rows, _GRID_COLS,
    figsize=(4.5 * _GRID_COLS, 3.5 * _n_rows),
    squeeze=False,
)
for ax in axes_gallery.flat:
    ax.axis("off")

for idx, (_, ev) in enumerate(events_df.iterrows()):
    ax = axes_gallery[idx // _GRID_COLS][idx % _GRID_COLS]
    ev_cid = int(ev["container_id"])
    ev_mid = (int(ev["start_ts"]) + int(ev["end_ts"])) // 2

    try:
        ev_cam_id = resolve_front_camera_channel_id(spark, cfg.t_channel_tags, ev_cid)
        ev_cam_frame = frame_nearest_to(spark, cfg.t_perception_channels, ev_cid, ev_cam_id, ev_mid)
        ev_img = read_camera_image(ev_cam_frame.file_path, fmt["camera_reader"])
        ax.imshow(ev_img)
    except Exception as e:
        ax.text(0.5, 0.5, f"frame unavailable\n{type(e).__name__}", ha="center", va="center", fontsize=9)

    ev_duration_s = (int(ev["end_ts"]) - int(ev["start_ts"])) / 1e6
    title = (
        f"container {ev_cid}  ·  {int(ev['n_objects'])} objects\n"
        f"window {ev_duration_s:.1f}s"
    )
    if ev["event_id"] == event_id:
        title = f"★ FOCAL ★\n{title}"
    ax.set_title(title, fontsize=9)

plt.tight_layout()
display(fig_gallery)
plt.close(fig_gallery)

# COMMAND ----------

# MAGIC %md ### Find the closest front-camera frame and LiDAR scan

# COMMAND ----------

cam_channel_id = resolve_front_camera_channel_id(spark, cfg.t_channel_tags, container_id)
cam_frame = frame_nearest_to(spark, cfg.t_perception_channels, container_id, cam_channel_id, mid_us)
print(f"Front camera @ {cam_frame.timestamp}: {cam_frame.file_path}")

lidar_channel_id = resolve_lidar_channel_id(spark, cfg.t_channel_tags, container_id)
lidar_scan = frame_nearest_to(spark, cfg.t_perception_channels, container_id, lidar_channel_id, mid_us)
print(f"LiDAR @ {lidar_scan.timestamp}: {lidar_scan.file_path}")

# COMMAND ----------

# MAGIC %md ### Get the `object_tracks` rows that fired the event

# COMMAND ----------

objects_in_window = (
    spark.read.table(cfg.t_object_tracks)
    .filter(
        (F.col("container_id") == container_id)
        & (F.col("frame_ts") >= start_us)
        & (F.col("frame_ts") <= end_us)
    )
    .toPandas()
)
print(f"object_tracks rows in window: {len(objects_in_window)}")
display(objects_in_window.head(20))

# COMMAND ----------

# MAGIC %md ### Render — camera frame + LiDAR top-down overlay

# COMMAND ----------

fig, (ax_cam, ax_lidar) = plt.subplots(1, 2, figsize=(18, 7))

img = read_camera_image(cam_frame.file_path, fmt["camera_reader"])
ax_cam.imshow(img)
ax_cam.set_title(f"Front camera @ t={cam_frame.timestamp}")
ax_cam.axis("off")

xs, ys, zs, intensity = read_lidar_xyzi(
    lidar_scan.file_path,
    reader=fmt["lidar_reader"],
    dtype=fmt["lidar_dtype"],
    stride=fmt["lidar_stride"],
)
if len(xs) > 50000:
    idx = np.random.choice(len(xs), 50000, replace=False)
    xs, ys, zs, intensity = xs[idx], ys[idx], zs[idx], intensity[idx]

ax_lidar.scatter(xs, ys, s=0.5, c=intensity, cmap="viridis", alpha=0.6)
ax_lidar.scatter([0], [0], marker="^", c="red", s=200, label="ego", zorder=5)

_CLASS_COLOR = {"pedestrian": "yellow", "car": "orange", "cyclist": "lime"}
for _, obj in objects_in_window.iterrows():
    x, y = azimuth_label_to_xy(obj["distance_m"], obj["azimuth"])
    color = _CLASS_COLOR.get(obj["detection_class"], "white")
    ax_lidar.scatter([x], [y], s=120, marker="o", facecolor="none", edgecolor=color, linewidth=2.0, zorder=4)
    ax_lidar.annotate(
        f"{obj['detection_class'][:3]}\n{obj['distance_m']:.1f}m",
        (x, y), xytext=(5, 5), textcoords="offset points",
        fontsize=7, color=color,
    )

ax_lidar.set_xlim(-50, 80)
ax_lidar.set_ylim(-30, 30)
ax_lidar.set_aspect("equal")
ax_lidar.set_xlabel("forward (m)")
ax_lidar.set_ylabel("left (m)")
ax_lidar.set_title(f"LiDAR @ t={lidar_scan.timestamp}  +  object_tracks overlay")
ax_lidar.legend(loc="upper right")
ax_lidar.grid(True, alpha=0.3)

plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md ### Matching TSAL scalars at event midpoint

# COMMAND ----------

display(spark.sql(f"""
SELECT
  ct.value AS channel_name,
  ROUND(c.value, 2) AS value
FROM   {cfg.t_channels} c
JOIN   {cfg.t_channel_tags} ct
  ON   c.container_id = ct.container_id
 AND   c.channel_id   = ct.channel_id
 AND   ct.key = 'channel_name'
WHERE  c.container_id = {container_id}
  AND  c.tstart <= {mid_us}
  AND  c.tend   >  {mid_us}
  AND  ct.value IN ('Pedestrian_Nearest_Distance_m', 'Vehicle_Speed_kph', 'Pedestrian_Count')
"""))

# COMMAND ----------

# MAGIC %md ### Frame-strip playback — focal event over time
# MAGIC
# MAGIC Five camera frames around the event midpoint (t-1.5s, t-0.5s, t, t+0.5s, t+1.5s)
# MAGIC with the matching LiDAR top-down + `object_tracks` overlay underneath each. Gives
# MAGIC temporal context for the single midpoint frame above — you can see the scene evolve.

# COMMAND ----------

_FRAME_OFFSETS_US = [int(-1.5e6), int(-0.5e6), 0, int(0.5e6), int(1.5e6)]
_FRAME_LABELS = ["t-1.5s", "t-0.5s", "t", "t+0.5s", "t+1.5s"]

# Pre-load the object_tracks for this container — used for the overlay on each frame.
container_tracks = (
    spark.read.table(cfg.t_object_tracks)
    .filter(F.col("container_id") == container_id)
    .toPandas()
)

fig_strip, axes_strip = plt.subplots(2, len(_FRAME_OFFSETS_US), figsize=(4.5 * len(_FRAME_OFFSETS_US), 8))
for ax in axes_strip.flat:
    ax.axis("off")

for col, (offset_us, label) in enumerate(zip(_FRAME_OFFSETS_US, _FRAME_LABELS)):
    target_ts = mid_us + offset_us
    cam_row = frame_nearest_to(spark, cfg.t_perception_channels, container_id, cam_channel_id, target_ts)
    lidar_row = frame_nearest_to(spark, cfg.t_perception_channels, container_id, lidar_channel_id, target_ts)

    # Top row — camera frame.
    ax_cam = axes_strip[0][col]
    try:
        img_strip = read_camera_image(cam_row.file_path, fmt["camera_reader"])
        ax_cam.imshow(img_strip)
    except Exception as e:
        ax_cam.text(0.5, 0.5, f"frame unavailable\n{type(e).__name__}",
                    ha="center", va="center", fontsize=9)
    ax_cam.set_title(label, fontsize=11)
    ax_cam.axis("off")

    # Bottom row — LiDAR top-down + object overlay near this timestamp.
    ax_lid = axes_strip[1][col]
    ax_lid.axis("on")
    try:
        xs_s, ys_s, zs_s, intensity_s = read_lidar_xyzi(
            lidar_row.file_path,
            reader=fmt["lidar_reader"],
            dtype=fmt["lidar_dtype"],
            stride=fmt["lidar_stride"],
        )
        # Subsample to 20k for tighter visuals at slide-projection size.
        if len(xs_s) > 20000:
            idx_s = np.random.choice(len(xs_s), 20000, replace=False)
            xs_s, ys_s, intensity_s = xs_s[idx_s], ys_s[idx_s], intensity_s[idx_s]
        ax_lid.scatter(xs_s, ys_s, s=0.4, c=intensity_s, cmap="viridis", alpha=0.5)
        ax_lid.scatter([0], [0], marker="^", c="red", s=120, zorder=5)
    except Exception as e:
        ax_lid.text(0.5, 0.5, f"LiDAR unavailable\n{type(e).__name__}",
                    transform=ax_lid.transAxes, ha="center", va="center", fontsize=9)

    # Objects within ~200 ms of this offset.
    near_ts_mask = (container_tracks["frame_ts"] - target_ts).abs() <= 200_000
    for _, obj in container_tracks[near_ts_mask].iterrows():
        ox, oy = azimuth_label_to_xy(obj["distance_m"], obj["azimuth"])
        col_o = _CLASS_COLOR.get(obj["detection_class"], "white")
        ax_lid.scatter([ox], [oy], s=80, marker="o", facecolor="none",
                       edgecolor=col_o, linewidth=1.5, zorder=4)
    ax_lid.set_xlim(-50, 80)
    ax_lid.set_ylim(-30, 30)
    ax_lid.set_aspect("equal")
    ax_lid.grid(True, alpha=0.3)
    ax_lid.tick_params(labelsize=8)

plt.suptitle(
    f"Frame strip — container {container_id}, event {event_id} (window {(end_us - start_us) / 1e6:.2f}s)",
    fontsize=12,
)
plt.tight_layout()
display(fig_strip)
plt.close(fig_strip)

# COMMAND ----------

# MAGIC %md ## Part 2 — Sensor KPI comparison
# MAGIC
# MAGIC Across **every** event in the playlist, compare detection performance across
# MAGIC LiDAR / radar / camera using only data already produced by the pipeline.
# MAGIC
# MAGIC This part is byte-identical across BYOD adapters — it queries the
# MAGIC `object_tracks.source` pipe-delimited convention populated by every adapter.

# COMMAND ----------

# MAGIC %md ### Load `object_tracks` rows for the playlist event windows

# COMMAND ----------

tracks_df = spark.sql(f"""
SELECT
  ot.container_id,
  ot.frame_ts,
  ot.object_id,
  ot.detection_class,
  ot.distance_m,
  ot.azimuth,
  ot.lane_offset,
  ot.relative_velocity_ms,
  ot.source
FROM {cfg.t_object_tracks} ot
JOIN {cfg.t_playlist_items} pi
  ON ot.container_id = pi.container_id
 AND ot.frame_ts BETWEEN pi.start_ts AND pi.end_ts
WHERE pi.playlist_id = '{_playlist_id}'
""").toPandas()

events_df = (
    spark.read.table(cfg.t_playlist_items)
    .filter(F.col("playlist_id") == _playlist_id)
    .select("container_id", "start_ts", "end_ts")
    .toPandas()
)

print(f"object_tracks rows in playlist windows: {len(tracks_df):,}")
print(f"Event windows: {len(events_df):,}")
print(f"Source values present: {sorted(tracks_df['source'].unique())}")

# COMMAND ----------

# MAGIC %md ### Expand source flags

# COMMAND ----------

tracks_df = with_sensor_flags(tracks_df)
print(f"LiDAR-seen:  {tracks_df['lidar_seen'].sum():,} detections")
print(f"Radar-seen:  {tracks_df['radar_seen'].sum():,} detections")
print(f"Camera-seen: {tracks_df['camera_seen'].sum():,} detections")

# COMMAND ----------

# MAGIC %md ### KPI 1 — Detection counts by source pattern

# COMMAND ----------

display(detection_counts_by_source(tracks_df))

# COMMAND ----------

# MAGIC %md ### KPI 2 — Distance distributions by modality

# COMMAND ----------

dist_stats = distance_stats_by_modality(tracks_df)
display(dist_stats)

# COMMAND ----------

# MAGIC %md ### Distance distribution plots

# COMMAND ----------

fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
colors = {"lidar": "#2196F3", "radar": "#FF9800", "camera": "#4CAF50"}
bins = list(range(0, 82, 4))

for ax, (modality, col) in zip(axes, [("lidar", "lidar_seen"), ("radar", "radar_seen"), ("camera", "camera_seen")]):
    sub = tracks_df[tracks_df[col]]["distance_m"].dropna()
    ax.hist(sub, bins=bins, color=colors[modality], edgecolor="white", linewidth=0.5)
    ax.set_title(f"{modality.capitalize()} detections\n(n={len(sub):,})")
    ax.set_xlabel("Distance from ego (m)")
    ax.set_ylabel("Detection count")
    if len(sub):
        ax.axvline(float(sub.median()), color="black", linestyle="--", linewidth=1.5, label=f"median {sub.median():.1f}m")
        ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.suptitle("Detection distance distributions by sensor modality", fontsize=13)
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md ### KPI 3 — Object-class coverage by modality

# COMMAND ----------

display(class_coverage_by_modality(tracks_df))

# COMMAND ----------

# MAGIC %md ### KPI 4 — Detection gap analysis

# COMMAND ----------

gaps_df = detection_gaps(tracks_df, events_df)
display(gaps_df)

n_lidar_gaps = int(gaps_df["lidar_gap"].sum())
n_radar_gaps = int(gaps_df["radar_gap"].sum())
print(f"\nLiDAR gaps: {n_lidar_gaps} event windows with zero LiDAR detections")
print(f"Radar gaps: {n_radar_gaps} event windows with zero radar detections")

# COMMAND ----------

# MAGIC %md ## Acceptance — KPI ran on real data

# COMMAND ----------

n_windows = len(events_df)
assert n_windows >= 1, f"playlist has no windows — upstream t02_detect_events must populate playlist_items before t04_visualize ({cfg.t_playlist_items})"
assert not dist_stats.empty, "No distance stats produced — check object_tracks source values"
assert n_lidar_gaps >= 0 and n_radar_gaps >= 0

at_least_one_gap = (n_lidar_gaps + n_radar_gaps) > 0
if at_least_one_gap:
    print(f"✓ acceptance criterion met — {n_lidar_gaps} LiDAR + {n_radar_gaps} radar gaps found")
else:
    print("⚠️  no detection gaps found — all windows had both LiDAR and radar coverage")
    print("   Try a larger dataset variant for a wider sensor-coverage spectrum")

# COMMAND ----------

# MAGIC %md ## ✓ Demo complete
