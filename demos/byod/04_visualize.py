# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Visualize: see one event, then compare across modalities
# MAGIC
# MAGIC The "see and compare" stage. Two halves:
# MAGIC
# MAGIC 1. **Event walkthrough** — pick an event window (by `event_name`), render the
# MAGIC    front-camera frame at the event's midpoint, plot the LiDAR scan top-down, overlay
# MAGIC    the `object_tracks` rows that fired the event.
# MAGIC 2. **Sensor KPI comparison** — across every window of that event, compare detection
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

from lib.demo_setup import connect
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
from lib.frames import frame_nearest_to
from lib.geometry import azimuth_label_to_xy
from pyspark.sql import functions as F

from databricks.sdk import WorkspaceClient
from impulse_reporting.core.report import Report
from impulse_reporting.core.page import Page
from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.aggregations.histogram import HistogramDuration
from impulse_reporting.aggregations.histogram2d import Histogram2DDuration

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

ctx = connect(dbutils, with_adapter=True, report_table_suffix="viz_agg", extra_widgets=[
    ("event_name", "pedestrian_high_speed_proximity", "Event name (from 02_detect_events)"),
    ("event_instance_id", "", "Event instance ID (blank = densest window)"),
])
cfg = ctx["cfg"]
adapter = ctx["adapter"]
_event_name = ctx["event_name"]
_event_instance_override = ctx["event_instance_id"]
fmt = adapter_visualize_format(adapter)

# COMMAND ----------

# MAGIC %md ## Part 1 — Event walkthrough
# MAGIC
# MAGIC After `run_all_job` completes, this notebook shows what the pipeline found —
# MAGIC first a gallery of every window of this event, then a single-window deep-dive,
# MAGIC then a frame-strip playback of that window over time.
# MAGIC
# MAGIC The single-window panel auto-picks the window with the most `object_tracks` rows
# MAGIC (the densest "scene"). To inspect a specific window, set the `event_instance_id`
# MAGIC widget at the top of the notebook.

# COMMAND ----------

# Pull every window of this event as a Pandas frame for convenience. Windows come
# from the detect step's `event_instance_fact` (joined to `event_dimension` for the
# name).
# NOTE: container_id / object_id are 63-bit hashes (stable_int_id) that exceed
# float64's exact range (2^53). pandas would silently round them, so we carry
# them as STRING across every Spark→pandas boundary and int() them only when
# handing a value back to a Spark filter.
events_df = (
    spark.read.table(cfg.t_event_instance_fact)
    .join(spark.read.table(cfg.t_event_dimension), "event_id")
    .filter(F.col("event_name") == _event_name)
    .select(
        F.col("container_id").cast("string").alias("container_id"),
        F.col("event_instance_id").cast("string").alias("event_instance_id"),
        "start_ts", "end_ts",
    )
    .orderBy("container_id", "start_ts")
    .toPandas()
)

if events_df.empty:
    raise RuntimeError(
        f"No windows found for event '{_event_name}'. "
        "Re-run 02_detect_events.py if the event fact table is empty."
    )

# Object-track count per event window — used to pick the "most interesting" event.
object_tracks_pdf = (
    spark.sql(f"""
    SELECT CAST(ot.container_id AS STRING) AS container_id,
           ot.frame_ts, ot.detection_class, ot.distance_m, ot.azimuth
    FROM   {cfg.t_object_tracks} ot
    JOIN   {cfg.t_event_instance_fact} f
      ON   ot.container_id = f.container_id
     AND   ot.frame_ts BETWEEN f.start_ts AND f.end_ts
    JOIN   {cfg.t_event_dimension} d USING (event_id)
    WHERE  d.event_name = '{_event_name}'
    """)
    .toPandas()
)

def _objects_in_event(row):
    return int(
        (
            (object_tracks_pdf["container_id"] == row["container_id"])
            & (object_tracks_pdf["frame_ts"] >= int(row["start_ts"]))
            & (object_tracks_pdf["frame_ts"] <= int(row["end_ts"]))
        ).sum()
    )

events_df["n_objects"] = events_df.apply(_objects_in_event, axis=1)

print(f"Event '{_event_name}': {len(events_df)} windows, "
      f"{int(events_df['n_objects'].sum()):,} total object_tracks rows in their windows")

# COMMAND ----------

# MAGIC %md ### Pick the focal event
# MAGIC
# MAGIC `event_instance_id` widget (if non-empty) → that exact window; otherwise → the
# MAGIC window with the most `object_tracks` rows.

# COMMAND ----------

if _event_instance_override:
    focal = events_df[events_df["event_instance_id"] == _event_instance_override]
    if focal.empty:
        raise RuntimeError(
            f"event_instance_id={_event_instance_override!r} not found for event '{_event_name}'"
        )
    focal_row = focal.iloc[0]
else:
    focal_row = events_df.loc[events_df["n_objects"].idxmax()]

event_instance_id = focal_row["event_instance_id"]
container_id = int(focal_row["container_id"])
start_us = int(focal_row["start_ts"])
end_us = int(focal_row["end_ts"])
mid_us = (start_us + end_us) // 2

print(f"Event instance: {event_instance_id}")
print(f"Container:    {container_id}")
print(f"Window:       [{start_us}, {end_us}]  ({(end_us - start_us) / 1e6:.2f}s)")
print(f"Objects in window: {int(focal_row['n_objects'])}")

# COMMAND ----------

# MAGIC %md ### Thumbnail gallery — every window of this event
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
    if ev["event_instance_id"] == event_instance_id:
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

# MAGIC %md ### Map context at the event (where on the map)
# MAGIC
# MAGIC The third Series. For the focal frame, `ego_map_context` says *where the ego is on
# MAGIC the HD map* (on a ped-crossing? in an intersection? which lane? how far to the next
# MAGIC stop line), and `object_map_context` says which detected objects share that
# MAGIC context (on a crossing, in the ego's lane). This is the visual payoff for the
# MAGIC three-table map event — you can see the scene above and read the map situation that
# MAGIC fired it. Presence-gated: shown only when the map_context tables were ingested.

# COMMAND ----------

if not spark.catalog.tableExists(cfg.t_ego_map_context):
    print(f"⏭️  No map context ({cfg.t_ego_map_context}) — skipping map panel.")
else:
    ego_ctx = spark.sql(f"""
        SELECT location, on_ped_crossing, on_walkway, on_drivable_area, in_intersection,
               ROUND(dist_to_ped_crossing_m, 1) AS dist_to_ped_crossing_m,
               ROUND(dist_to_stop_line_m, 1)    AS dist_to_stop_line_m, lane_id
        FROM {cfg.t_ego_map_context}
        WHERE container_id = {container_id}
        ORDER BY ABS(frame_ts - {mid_us}) LIMIT 1
    """).toPandas()
    print("Ego map context at the focal frame:")
    display(ego_ctx)
    if len(ego_ctx):
        e = ego_ctx.iloc[0]
        flags = [n for n in ("on_ped_crossing", "on_walkway", "in_intersection") if int(e[n] or 0) == 1]
        print(f"→ {e['location']}: ego is " + (", ".join(flags) if flags else "on open road")
              + (f"; nearest stop line {e['dist_to_stop_line_m']} m" if e['dist_to_stop_line_m'] is not None else ""))

    # Objects in the window with their map context (on a crossing / sharing ego's lane).
    obj_ctx = spark.sql(f"""
        SELECT detection_class, COUNT(*) AS rows,
               SUM(on_ped_crossing)  AS on_ped_crossing,
               SUM(same_lane_as_ego) AS same_lane_as_ego
        FROM {cfg.t_object_map_context}
        WHERE container_id = {container_id} AND frame_ts BETWEEN {start_us} AND {end_us}
        GROUP BY detection_class ORDER BY rows DESC
    """).toPandas()
    print("Objects in the event window, by map context:")
    display(obj_ctx)

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
    f"Frame strip — container {container_id}, event instance {event_instance_id} (window {(end_us - start_us) / 1e6:.2f}s)",
    fontsize=12,
)
plt.tight_layout()
display(fig_strip)
plt.close(fig_strip)

# COMMAND ----------

# MAGIC %md ## Part 2 — Sensor KPI comparison
# MAGIC
# MAGIC Across **every** window of this event, compare detection performance across
# MAGIC LiDAR / radar / camera using only data already produced by the pipeline.
# MAGIC
# MAGIC This part is byte-identical across BYOD adapters — it queries the
# MAGIC `object_tracks.source` pipe-delimited convention populated by every adapter.

# COMMAND ----------

# MAGIC %md ### Load `object_tracks` rows for the event windows

# COMMAND ----------

# Part 2 keeps container_id numeric: it's only ever compared tracks_df↔events_df
# (both from the same toPandas, so any float rounding is identical on both sides
# and they still match) — never against an exact external id, unlike Part 1.
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
JOIN {cfg.t_event_instance_fact} f
  ON ot.container_id = f.container_id
 AND ot.frame_ts BETWEEN f.start_ts AND f.end_ts
JOIN {cfg.t_event_dimension} d USING (event_id)
WHERE d.event_name = '{_event_name}'
""").toPandas()

events_df = (
    spark.read.table(cfg.t_event_instance_fact)
    .join(spark.read.table(cfg.t_event_dimension), "event_id")
    .filter(F.col("event_name") == _event_name)
    .select("container_id", "start_ts", "end_ts")
    .toPandas()
)

print(f"object_tracks rows in event windows: {len(tracks_df):,}")
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

# MAGIC %md ### Distance distribution — built-in engine aggregations, event-gated
# MAGIC
# MAGIC Rather than hand-rolling histograms, register the engine's built-in **duration
# MAGIC aggregations** over the scalar channels and let it compute them — each gated to
# MAGIC the intervals where the vehicle was moving faster than 30 kph:
# MAGIC
# MAGIC - **`HistogramDuration`** — how long the nearest pedestrian spent in each
# MAGIC   distance band while the vehicle was at speed.
# MAGIC - **`Histogram2DDuration`** — a speed × pedestrian-distance density heatmap.
# MAGIC
# MAGIC Aggregations run over **channels** (single-valued scalar signals); the per-object
# MAGIC modality breakdown above comes from the `object_tracks` Series instead.

# COMMAND ----------

agg_report = Report(
    name=f"{cfg.adapter_name}_viz_aggregations",
    spark=spark,
    workspace_client=WorkspaceClient(),
    config=ctx["report_config"],
)
agg_db = agg_report.get_db()
speed = agg_db.query.channel(channel_name="Vehicle_Speed_kph")
ped_distance = agg_db.query.channel(channel_name="Pedestrian_Nearest_Distance_m")

# Event gate: the intervals where the vehicle exceeds 30 kph.
high_speed = BasicEvent(
    name="vehicle_over_30kph",
    expr=speed > 30.0,
    desc="Vehicle speed above 30 kph",
)
agg_report.add_event(high_speed)

page = Page(page_number=1)
page.add_aggregation(HistogramDuration(
    name="pedestrian_distance_at_speed",
    base_expr=ped_distance,
    bins=[float(b) for b in range(0, 85, 5)],
    event=high_speed,
    channel_name="Pedestrian_Nearest_Distance_m",
    values_unit="s",
    bins_unit="m",
))
page.add_aggregation(Histogram2DDuration(
    name="speed_x_pedestrian_distance",
    x_expr=speed,
    y_expr=ped_distance,
    x_bins=[float(b) for b in range(0, 90, 10)],
    y_bins=[float(b) for b in range(0, 85, 5)],
    event=high_speed,
    x_channel_name="Vehicle_Speed_kph",
    y_channel_name="Pedestrian_Nearest_Distance_m",
    x_bins_unit="kph",
    y_bins_unit="m",
))
agg_report.add_page(page)

agg_report.determine_report()
print("✓ engine computed HistogramDuration + Histogram2DDuration (gated to vehicle > 30 kph)")


def _agg_fact(agg_type: str):
    """The in-memory fact DataFrame for an aggregation type (no persist needed)."""
    dfs = agg_report.aggregation_dfs.get(agg_type, {})
    return dfs.get("changed") if dfs.get("changed") is not None else dfs.get("unchanged")

# COMMAND ----------

# MAGIC %md ### HistogramDuration — pedestrian distance while the vehicle is at speed

# COMMAND ----------

hist1d = (
    _agg_fact("HISTOGRAM")
    .groupBy("bin_id", "bin_name", "lower_bound")
    .agg(F.sum("hist_value").alias("duration_s"))
    .orderBy("lower_bound")
    .toPandas()
)

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(hist1d["bin_name"], hist1d["duration_s"], color="#2196F3", edgecolor="white")
ax.set_title("Time spent by nearest-pedestrian distance (vehicle > 30 kph)")
ax.set_xlabel("Pedestrian distance (m)")
ax.set_ylabel("Duration (s)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
display(fig)
plt.close(fig)

# COMMAND ----------

# MAGIC %md ### Histogram2DDuration — speed × pedestrian-distance density

# COMMAND ----------

hist2d = (
    _agg_fact("HISTOGRAM2D")
    .groupBy("x_bin_id", "y_bin_id", "x_bin_name", "y_bin_name")
    .agg(F.sum("hist_value").alias("duration_s"))
    .toPandas()
)

x_order = hist2d.sort_values("x_bin_id")["x_bin_name"].unique()
y_order = hist2d.sort_values("y_bin_id")["y_bin_name"].unique()
grid = (
    hist2d.pivot_table(index="y_bin_name", columns="x_bin_name", values="duration_s", aggfunc="sum", fill_value=0.0)
    .reindex(index=y_order, columns=x_order, fill_value=0.0)
)

fig, ax = plt.subplots(figsize=(9, 5))
im = ax.imshow(grid.values, origin="lower", aspect="auto", cmap="viridis")
ax.set_xticks(range(len(x_order))); ax.set_xticklabels(x_order, rotation=45, ha="right")
ax.set_yticks(range(len(y_order))); ax.set_yticklabels(y_order)
ax.set_xlabel("Vehicle speed (kph)")
ax.set_ylabel("Pedestrian distance (m)")
ax.set_title("Speed × pedestrian-distance — duration (s), vehicle > 30 kph")
fig.colorbar(im, ax=ax, label="Duration (s)")
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
assert n_windows >= 1, f"event '{_event_name}' has no windows — upstream 02_detect_events must populate event_instance_fact before 04_visualize ({cfg.t_event_instance_fact})"
assert not dist_stats.empty, "No distance stats produced — check object_tracks source values"
assert n_lidar_gaps >= 0 and n_radar_gaps >= 0

at_least_one_gap = (n_lidar_gaps + n_radar_gaps) > 0
if at_least_one_gap:
    print(f"✓ Validated — {n_lidar_gaps} LiDAR + {n_radar_gaps} radar gaps found")
else:
    print("⚠️  no detection gaps found — all windows had both LiDAR and radar coverage")
    print("   Try a larger dataset variant for a wider sensor-coverage spectrum")

# COMMAND ----------

# MAGIC %md ## ✓ Demo complete
