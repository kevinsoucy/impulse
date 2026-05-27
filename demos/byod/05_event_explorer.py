# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Event Explorer: live TSAL event definition and comparison
# MAGIC
# MAGIC **The live-demo beat.** Notebooks 01–04 run a fixed pipeline with one pre-defined event.
# MAGIC This notebook shows how fast you can define *new* events — and compare them — without touching
# MAGIC any pipeline code.
# MAGIC
# MAGIC Four events are defined below, ordered by severity — a natural progressive refinement:
# MAGIC
# MAGIC | # | Event | Condition | What it answers |
# MAGIC |---|---|---|---|
# MAGIC | 1 | `ped_approach_30m` | Pedestrian ≤ 30 m | Broad proximity — any pedestrian nearby |
# MAGIC | 2 | `ped_hazard_15m_30kph` | Ped ≤ 15 m **AND** speed > 30 kph | Dangerous subset of #1 |
# MAGIC | 3 | `cyclist_hazard_10m` | Cyclist ≤ 10 m **AND** speed > 20 kph | Same shape, different object class |
# MAGIC | 4 | `dense_urban` | Vehicle ahead **AND** pedestrian present **AND** moving | Multi-object scene complexity |
# MAGIC
# MAGIC Events 1 and 2 demonstrate TSAL refinement: `ped_hazard ⊆ ped_approach` by construction —
# MAGIC every interval that satisfies the narrower predicate also satisfies the broader one.
# MAGIC The cross-surface query at the end drills from *when* (scalar event windows) to *what*
# MAGIC (`object_tracks` — per-frame object detail) to show the full two-surface query pattern.
# MAGIC
# MAGIC This notebook is byte-identical across BYOD adapters.

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

from databricks.sdk import WorkspaceClient
from lib.byod_config import BYODConfig

from mda_reporting.core.page import Page
from mda_reporting.core.report import Report
from mda_reporting.aggregations.histogram import HistogramDuration
from mda_reporting.aggregations.histogram2d import Histogram2DDuration
from mda_reporting.aggregations.stats_aggregator import StatsAggregator
from mda_reporting.events.basic_event import BasicEvent
from mda_reporting.playlists import event_fact_to_playlist_items
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

dbutils.widgets.text("adapter",         "nuscenes",         "Adapter name")
dbutils.widgets.text("dataset_version", "v1.0-mini",        "Adapter-specific variant")
dbutils.widgets.text("catalog",         "main",             "UC catalog")
dbutils.widgets.text("schema_prefix",   "lakevision_demo",  "Schema prefix")
dbutils.widgets.text("playlist_id",     "explore_v1",       "Playlist ID for explorer events")

_adapter_name    = dbutils.widgets.get("adapter")
_dataset_version = dbutils.widgets.get("dataset_version") or None
_catalog         = dbutils.widgets.get("catalog")
_schema_prefix   = dbutils.widgets.get("schema_prefix")
_playlist_id     = dbutils.widgets.get("playlist_id")

cfg = BYODConfig.for_adapter(
    adapter_name=_adapter_name,
    dataset_version=_dataset_version,
    catalog=_catalog,
    schema_prefix=_schema_prefix,
)

_table_prefix = f"{cfg.adapter_name}_explore"
report_config = {
    "source": {
        "container_metrics_table": cfg.t_container_metrics,
        "container_tags_table":    cfg.t_container_tags,
        "channel_metrics_table":   cfg.t_channel_metrics,
        "channel_tags_table":      cfg.t_channel_tags,
        "channels_uri":            cfg.t_channels,
    },
    "unity_sink": {
        "catalog": cfg.catalog,
        "schema":  cfg.schema_gold,
        "table_prefix": _table_prefix,
    },
    "query_engine": {"solver": "DeltaSolver", "data_type": "RAW"},
    # Only container_id — same reasoning as notebook 02 (start_dt/stop_dt vs start_ts/stop_ts mismatch).
    "measurement_dimensions": ["container_id"],
}

print(f"Adapter:       {cfg.adapter_name}")
print(f"Channels from: {report_config['source']['channels_uri']}")
print(f"Event sink:    {cfg.catalog}.{cfg.schema_gold}.{_table_prefix}_*")
print(f"Playlist:      {_playlist_id}")

# COMMAND ----------

# MAGIC %md ## Channel overview
# MAGIC
# MAGIC Nine scalar channels are synthesized from ego pose and ground-truth annotations.
# MAGIC Three are kinematic (speed, acceleration, steering); six are detection aggregates
# MAGIC (nearest distance and count per class). All nine are available as TSAL channel expressions.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ct.channel_name,
# MAGIC   COUNT(*)               AS intervals,
# MAGIC   ROUND(MIN(c.value), 2) AS min_val,
# MAGIC   ROUND(AVG(c.value), 2) AS avg_val,
# MAGIC   ROUND(MAX(c.value), 2) AS max_val
# MAGIC FROM   ${catalog}.${schema_prefix}_silver.channels c
# MAGIC JOIN   ${catalog}.${schema_prefix}_silver.channel_tags ct USING (channel_id)
# MAGIC GROUP  BY ct.channel_name
# MAGIC ORDER  BY ct.channel_name

# COMMAND ----------

# MAGIC %md ## Build TSAL channel expressions
# MAGIC
# MAGIC Each call to `db.query.channel()` returns a lazy expression — no data is read until
# MAGIC `report.determine_report()` is called. The channel name resolves via `channel_tags`.

# COMMAND ----------

report = Report(
    name=f"{cfg.adapter_name}_event_explorer",
    spark=spark,
    workspace_client=WorkspaceClient(),
    config=report_config,
)
db = report.get_db()

pedestrian_distance = db.query.channel(channel_name="Pedestrian_Nearest_Distance_m")
cyclist_distance    = db.query.channel(channel_name="Cyclist_Nearest_Distance_m")
vehicle_speed       = db.query.channel(channel_name="Vehicle_Speed_kph")
vehicle_accel       = db.query.channel(channel_name="Vehicle_Accel_Longitudinal_ms2")
vehicle_count_front = db.query.channel(channel_name="Vehicle_Count_Front")
pedestrian_count    = db.query.channel(channel_name="Pedestrian_Count")

print("Channel expressions built — no data read yet")

# COMMAND ----------

# MAGIC %md ## Event 1 — Pedestrian approach: wide net
# MAGIC
# MAGIC > *"In which intervals was any pedestrian within 30 m of ego?"*
# MAGIC
# MAGIC A single-channel threshold — the broadest cut. Every interval matching event 2
# MAGIC (`ped_hazard`) is guaranteed to appear here too.

# COMMAND ----------

ped_approach = BasicEvent(
    name="ped_approach_30m",
    expr=pedestrian_distance <= 30.0,
    desc="Any pedestrian within 30 m of ego",
    required_channels=["Pedestrian_Nearest_Distance_m"],
)

# COMMAND ----------

# MAGIC %md ## Event 2 — Pedestrian hazard: compound refinement
# MAGIC
# MAGIC > *"When was a pedestrian within 15 m AND the vehicle faster than 30 kph?"*
# MAGIC
# MAGIC Two channels, one expression. Comparing the window counts from events 1 and 2 shows
# MAGIC how often simple proximity escalates into an actual hazard scenario.

# COMMAND ----------

ped_hazard = BasicEvent(
    name="ped_hazard_15m_30kph",
    expr=(pedestrian_distance <= 15.0) & (vehicle_speed > 30.0),
    desc="Pedestrian within 15 m while speed exceeds 30 kph",
    required_channels=["Pedestrian_Nearest_Distance_m", "Vehicle_Speed_kph"],
)

# COMMAND ----------

# MAGIC %md ## Event 3 — Cyclist hazard: same pattern, different class
# MAGIC
# MAGIC > *"When was a cyclist within 10 m while the vehicle was moving?"*
# MAGIC
# MAGIC Structurally identical to event 2 — only the channel name changes.
# MAGIC Adding a new object class to the search costs one `BasicEvent`.

# COMMAND ----------

cyclist_hazard = BasicEvent(
    name="cyclist_hazard_10m",
    expr=(cyclist_distance <= 10.0) & (vehicle_speed > 20.0),
    desc="Cyclist within 10 m while vehicle is moving faster than 20 kph",
    required_channels=["Cyclist_Nearest_Distance_m", "Vehicle_Speed_kph"],
)

# COMMAND ----------

# MAGIC %md ## Event 4 — Dense urban scenario: multi-channel complexity
# MAGIC
# MAGIC > *"When was at least one vehicle ahead AND a pedestrian present AND the vehicle moving?"*
# MAGIC
# MAGIC Three channels combined with `&`. Captures mixed-traffic moments — higher cognitive
# MAGIC load for the perception stack than any single-object event.

# COMMAND ----------

dense_urban = BasicEvent(
    name="dense_urban",
    expr=(vehicle_count_front >= 1.0) & (pedestrian_count >= 1.0) & (vehicle_speed > 10.0),
    desc="Vehicle ahead + pedestrian present + moving — mixed-traffic urban scenario",
    required_channels=["Vehicle_Count_Front", "Pedestrian_Count", "Vehicle_Speed_kph"],
)

# COMMAND ----------

# MAGIC %md ## Register events and define aggregations

# COMMAND ----------

report.add_event(ped_approach)
report.add_event(ped_hazard)
report.add_event(cyclist_hazard)
report.add_event(dense_urban)

page = Page(page_number=1)
report.add_page(page)

# Speed distribution during pedestrian hazard windows.
page.add_aggregation(HistogramDuration(
    name="speed_during_ped_hazard",
    base_expr=vehicle_speed,
    bins=[float(b) for b in range(0, 90, 10)],
    event=ped_hazard,
    desc="Vehicle speed distribution during pedestrian hazard events",
    channel_name="Vehicle_Speed_kph",
    bins_unit="kph",
    values_unit="s",
))

# Pedestrian distance vs. speed heatmap — how close is the pedestrian across speed regimes?
page.add_aggregation(Histogram2DDuration(
    name="ped_distance_vs_speed",
    x_expr=vehicle_speed,
    y_expr=pedestrian_distance,
    x_bins=[float(b) for b in range(0, 90, 10)],
    y_bins=[float(b) for b in range(0, 35, 5)],
    event=ped_approach,
    desc="Pedestrian nearest distance vs. vehicle speed — during any proximity",
    x_channel_name="Vehicle_Speed_kph",
    y_channel_name="Pedestrian_Nearest_Distance_m",
    x_bins_unit="kph",
    y_bins_unit="m",
    values_unit="s",
))

# Per-event statistics: speed and acceleration within each event type.
page.add_aggregation(StatsAggregator(
    name="ped_hazard_stats",
    input_expressions=[vehicle_speed, vehicle_accel, pedestrian_distance],
    channel_names=["Vehicle_Speed_kph", "Vehicle_Accel_Longitudinal_ms2", "Pedestrian_Nearest_Distance_m"],
    statistics=["min", "median", "mean", "max"],
    event=ped_hazard,
    desc="Speed, deceleration, and pedestrian distance statistics within hazard events",
))
page.add_aggregation(StatsAggregator(
    name="cyclist_hazard_stats",
    input_expressions=[vehicle_speed, cyclist_distance],
    channel_names=["Vehicle_Speed_kph", "Cyclist_Nearest_Distance_m"],
    statistics=["min", "median", "mean", "max"],
    event=cyclist_hazard,
    desc="Speed and cyclist distance statistics within cyclist hazard events",
))

print(f"4 events and {len(page.aggregations)} aggregations registered")

# COMMAND ----------

# MAGIC %md ## Run the TSAL queries and persist results

# COMMAND ----------

report.determine_report()
report.persist_results()
print("✓ event_instance_fact and aggregation tables written to gold")

# COMMAND ----------

# MAGIC %md ## Event comparison
# MAGIC
# MAGIC How many windows did each event capture? Events 1 and 2 should show the refinement:
# MAGIC more total time in `ped_approach_30m`, a narrower subset in `ped_hazard_15m_30kph`.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   d.event_name,
# MAGIC   COUNT(*)                                        AS event_windows,
# MAGIC   COUNT(DISTINCT f.container_id)                 AS scenes_covered,
# MAGIC   ROUND(SUM(f.end_ts - f.start_ts) / 1e6, 1)    AS total_s,
# MAGIC   ROUND(AVG((f.end_ts - f.start_ts) / 1e6), 2)  AS avg_duration_s
# MAGIC FROM   ${catalog}.${schema_prefix}_gold.${adapter}_explore_event_instance_fact f
# MAGIC JOIN   ${catalog}.${schema_prefix}_gold.${adapter}_explore_event_dimension     d  USING (event_id)
# MAGIC GROUP  BY d.event_name
# MAGIC ORDER  BY total_s DESC

# COMMAND ----------

# MAGIC %md ## Cross-surface query — event windows → object_tracks
# MAGIC
# MAGIC TSAL found the *when* by scanning `channels`. Now we join against `object_tracks`
# MAGIC to see the *what*: which objects were around the ego during each `ped_hazard` window,
# MAGIC their distance, azimuth, and whether they were approaching or receding.

# COMMAND ----------

_event_fact_table = f"{cfg.catalog}.{cfg.schema_gold}.{_table_prefix}_event_instance_fact"
_event_dim_table  = f"{cfg.catalog}.{cfg.schema_gold}.{_table_prefix}_event_dimension"

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   d.event_name,
# MAGIC   ot.container_id,
# MAGIC   ot.frame_ts,
# MAGIC   ot.detection_class,
# MAGIC   ROUND(ot.distance_m, 1)           AS distance_m,
# MAGIC   ot.azimuth,
# MAGIC   ot.lane_offset,
# MAGIC   ROUND(ot.relative_velocity_ms, 2) AS rel_vel_ms
# MAGIC FROM   ${catalog}.${schema_prefix}_perception_silver.object_tracks ot
# MAGIC JOIN   ${catalog}.${schema_prefix}_gold.${adapter}_explore_event_instance_fact f
# MAGIC          ON  ot.container_id = f.container_id
# MAGIC         AND  ot.frame_ts BETWEEN f.start_ts AND f.end_ts
# MAGIC JOIN   ${catalog}.${schema_prefix}_gold.${adapter}_explore_event_dimension d  USING (event_id)
# MAGIC WHERE  d.event_name = 'ped_hazard_15m_30kph'
# MAGIC ORDER  BY ot.container_id, ot.frame_ts, ot.distance_m
# MAGIC LIMIT  50

# COMMAND ----------

# MAGIC %md ## Save matched windows as a playlist
# MAGIC
# MAGIC The `ped_hazard_15m_30kph` windows are appended to `playlist_items` so notebooks 03 and 04
# MAGIC can pick them up by `playlist_id`. Uses `mode("append")` to coexist with the playlist
# MAGIC already written by notebook 02.

# COMMAND ----------

events_df = spark.read.table(_event_fact_table)
ped_hazard_event_id = (
    spark.read.table(_event_dim_table)
    .filter(F.col("event_name") == "ped_hazard_15m_30kph")
    .select("event_id")
    .first()["event_id"]
)

playlist_df = event_fact_to_playlist_items(
    events_df.filter(F.col("event_id") == ped_hazard_event_id),
    playlist_id=_playlist_id,
    playlist_version=1,
    event_name="ped_hazard_15m_30kph",
)
playlist_df.write.format("delta").mode("append").saveAsTable(cfg.t_playlist_items)
print(f"Appended {playlist_df.count():,} playlist_items rows (playlist_id={_playlist_id} v1)")

# COMMAND ----------

# MAGIC %md ## Acceptance check

# COMMAND ----------

n_events = spark.read.table(_event_fact_table).count()
assert n_events > 0, (
    f"event_instance_fact is empty — no events matched. "
    f"Check that channel_tags has the correct channel_name values. Table: {_event_fact_table}"
)

per_event = (
    spark.read.table(_event_fact_table)
    .join(spark.read.table(_event_dim_table), "event_id")
    .groupBy("event_name")
    .agg(F.count("*").alias("n"))
    .collect()
)
counts = {r["event_name"]: r["n"] for r in per_event}
print(f"✓ acceptance — {n_events} total event windows across {len(counts)} events")
for name, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {name}: {n} windows")

if counts.get("cyclist_hazard_10m", 0) == 0:
    print("  ℹ  cyclist_hazard_10m: 0 windows — no cyclists in this dataset variant (expected for v1.0-mini)")

# COMMAND ----------

# MAGIC %md ## What's next
# MAGIC
# MAGIC - **03** (`03_per_event_detail.py`) accepts any `playlist_id` — point it at `explore_v1`
# MAGIC   to generate LiDAR + camera detections and OpenLABEL packages for these windows.
# MAGIC - **04** (`04_visualize.py`) can visualize the same playlist for sensor-level review and KPI comparison.
# MAGIC - To add another event: one `BasicEvent(name=..., expr=...)` line and one `report.add_event(...)` call.
