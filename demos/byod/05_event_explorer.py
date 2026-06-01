# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Event Explorer: live TSAL event definition and comparison
# MAGIC
# MAGIC **The live-demo stage.** Notebooks 01–04 run a fixed pipeline with one pre-defined event.
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
# MAGIC This notebook is generic across BYOD adapters.

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
from lib.demo_setup import connect

from impulse_reporting.core.page import Page
from impulse_reporting.core.report import Report
from impulse_reporting.aggregations.histogram import HistogramDuration
from impulse_reporting.aggregations.histogram2d import Histogram2DDuration
from impulse_reporting.aggregations.stats_aggregator import StatsAggregator
from impulse_reporting.events.basic_event import BasicEvent
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

ctx = connect(dbutils, report_table_suffix="explore")
cfg = ctx["cfg"]
report_config = ctx["report_config"]
_table_prefix = report_config["unity_sink"]["table_prefix"]

# COMMAND ----------

# MAGIC %md ## Channel overview
# MAGIC
# MAGIC Nine scalar channels are synthesized from ego pose and ground-truth annotations.
# MAGIC Three are kinematic (speed, acceleration, steering); six are detection aggregates
# MAGIC (nearest distance and count per class). All nine are available as TSAL channel expressions.

# COMMAND ----------

# MAGIC %sql
# MAGIC -- channel_tags is key/value: the channel name is `value` where key='channel_name'.
# MAGIC SELECT
# MAGIC   ct.value               AS channel_name,
# MAGIC   COUNT(*)               AS intervals,
# MAGIC   ROUND(MIN(c.value), 2) AS min_val,
# MAGIC   ROUND(AVG(c.value), 2) AS avg_val,
# MAGIC   ROUND(MAX(c.value), 2) AS max_val
# MAGIC FROM   ${catalog}.${schema_prefix}_silver.channels c
# MAGIC JOIN   ${catalog}.${schema_prefix}_silver.channel_tags ct
# MAGIC   ON   c.container_id = ct.container_id
# MAGIC  AND   c.channel_id   = ct.channel_id
# MAGIC  AND   ct.key = 'channel_name'
# MAGIC GROUP  BY ct.value
# MAGIC ORDER  BY ct.value

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

# COMMAND ----------

# MAGIC %md ## Series-based events (object_tracks + map_context)
# MAGIC
# MAGIC The four events above are scalar-channel TSAL. On the **same report**, the
# MAGIC explorer can also author events over the per-object **`object_tracks`** Series
# MAGIC and the per-frame **`map_context`** Series — including *cross-entity correlation*
# MAGIC (two independent objects overlapping in time) and a *three-table* scenario
# MAGIC (object_tracks + map + the speed channel). Presence-gated on the Series tables, so
# MAGIC the notebook stays generic across adapters.

# COMMAND ----------

from impulse_reporting.events.entity_event import EntityEvent
from lib.series_defs import (
    register_ego_map_context,
    register_object_map_context,
    register_object_tracks,
)

if not spark.catalog.tableExists(cfg.t_object_tracks):
    print("object_tracks not present — Series events skipped (channel events only).")
else:
    register_object_tracks(db, cfg.t_object_tracks)
    ot = db.query.series("object_tracks")

    # Per-object proximity — EntityEvent names *which* pedestrian (entity_key=object_id).
    report.add_event(EntityEvent(
        name="ped_close_per_object",
        expr=(ot.detection_class == "pedestrian") & (ot.distance_m <= 15.0),
        desc="Series: each pedestrian within 15 m (entity_key = object_id)",
    ))

    # Cross-entity correlation — a VRU close while a car closes from ahead on radar.
    report.add_event(EntityEvent(
        name="vru_while_car_closing",
        expr=(
            (ot.detection_class.isin(["cyclist", "motorcycle"]) & (ot.distance_m <= 8.0)).entity_condition()
            & ((ot.detection_class == "car") & ot.azimuth.startswith("front")
               & ot.source.contains("radar") & (ot.distance_m <= 15.0)
               & (ot.relative_velocity_ms < -0.5)).entity_condition()
        ),
        per_entity_windowing=False,
        desc="Series cross-entity: two-wheeled VRU within 8 m while a car closes from ahead",
    ))

    if spark.catalog.tableExists(cfg.t_ego_map_context):
        register_ego_map_context(db, cfg.t_ego_map_context)
        register_object_map_context(db, cfg.t_object_map_context)
        ego_mapc = db.query.series("ego_map_context")
        # Three tables in one predicate: object_tracks + map_context + speed channel.
        report.add_event(BasicEvent(
            name="ped_on_crossing_at_speed",
            expr=(ego_mapc.on_ped_crossing == 1)
                 & ((ot.detection_class == "pedestrian") & (ot.distance_m <= 15.0)).entity_condition()
                 & (vehicle_speed > 20.0),
            desc="Series three-table: pedestrian within 15 m + ego on a ped_crossing + speed > 20 kph",
        ))
        print("Series events added: ped_close_per_object, vru_while_car_closing, ped_on_crossing_at_speed")
    else:
        print("Series events added: ped_close_per_object, vru_while_car_closing (no map_context → map event skipped)")

# COMMAND ----------

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
print(f"✓ Validated — {n_events} total event windows across {len(counts)} events")
for name, n in sorted(counts.items(), key=lambda x: -x[1]):
    print(f"  {name}: {n} windows")

if counts.get("cyclist_hazard_10m", 0) == 0:
    print("  ℹ  cyclist_hazard_10m: 0 windows — no cyclists in this dataset variant (expected for v1.0-mini)")

# COMMAND ----------

# MAGIC %md ## What's next
# MAGIC
# MAGIC - **03** (`03_per_event_detail.py`) reads event windows from `event_instance_fact`
# MAGIC   (filter by `event_name`) to generate LiDAR + camera detections and OpenLABEL packages.
# MAGIC - **04** (`04_visualize.py`) visualizes the same windows for sensor-level review and KPI comparison.
# MAGIC - To add another event: one `BasicEvent(name=..., expr=...)` line and one `report.add_event(...)` call.
