# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Detect events: TSAL scenario search + playlist creation
# MAGIC
# MAGIC **The core value-prop beat.**
# MAGIC
# MAGIC 1. Configures Impulse's `Report` against the demo's silver schema.
# MAGIC 2. Defines a TSAL `BasicEvent`: *pedestrian within 15 m AND vehicle speed > 30 kph*.
# MAGIC 3. Runs the event detection — Impulse compiles TSAL to Spark, scans `channels`,
# MAGIC    writes matched intervals to `event_instance_fact`.
# MAGIC 4. Saves the matched windows as a versioned `playlist_items` row set.
# MAGIC 5. Demonstrates the **cross-surface query** pattern: TSAL event window → drill into `object_tracks`.
# MAGIC
# MAGIC **Two query surfaces, one search:** TSAL finds the *when* (over `channels`,
# MAGIC the per-channel scalar table); the cross-surface join into `object_tracks`
# MAGIC shows the *what*. Searching over aggregated scalars keeps the predicate
# MAGIC compact; the per-object detail is only needed to explain a match.
# MAGIC
# MAGIC This notebook is byte-identical across BYOD adapters — it operates on the `channels` schema only.

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
from mda_reporting.events.basic_event import BasicEvent
from mda_reporting.playlists import event_fact_to_playlist_items
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

dbutils.widgets.text("adapter",         "nuscenes",                   "Adapter name")
dbutils.widgets.text("dataset_version", "v1.0-mini",                  "Adapter-specific variant")
dbutils.widgets.text("catalog",         "main",                       "UC catalog")
dbutils.widgets.text("schema_prefix",   "lakevision_demo",            "Schema prefix")
dbutils.widgets.text("playlist_id",     "pedestrian_high_speed_v1",   "Playlist ID")

_adapter_name = dbutils.widgets.get("adapter")
_dataset_version = dbutils.widgets.get("dataset_version") or None
_catalog = dbutils.widgets.get("catalog")
_schema_prefix = dbutils.widgets.get("schema_prefix")
_playlist_id = dbutils.widgets.get("playlist_id")

cfg = BYODConfig.for_adapter(
    adapter_name=_adapter_name,
    dataset_version=_dataset_version,
    catalog=_catalog,
    schema_prefix=_schema_prefix,
)

# Source-table config for Impulse Report.
_table_prefix = f"{cfg.adapter_name}_demo"
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
    # Only container_id — Impulse's CONTAINER_METRICS schema names timestamps
    # `start_dt`/`stop_dt`, but MeasurementDimensions.{START_TS,STOP_TS} expect
    # `start_ts`/`stop_ts` (no gold→silver remap). Requesting those raises
    # UNRESOLVED_COLUMN. start_dt/stop_dt remain readable directly from the
    # silver container_metrics table when needed.
    "measurement_dimensions": ["container_id"],
}

print(f"Adapter:          {cfg.adapter_name}")
print(f"Source channels:  {report_config['source']['channels_uri']}")
print(f"Event sink:       {cfg.catalog}.{cfg.schema_gold}.{_table_prefix}_*")
print(f"Playlist:         {_playlist_id}")

# COMMAND ----------

# MAGIC %md ## Define the TSAL event
# MAGIC
# MAGIC **Question we're asking:**
# MAGIC > *"In which time intervals across our recordings was a pedestrian within 15 m AND the vehicle was moving faster than 30 kph?"*
# MAGIC
# MAGIC This is the canonical TSAL pattern — interval algebra over multiple scalar channels.

# COMMAND ----------

report = Report(
    name=f"{cfg.adapter_name}_pedestrian_proximity",
    spark=spark,
    workspace_client=WorkspaceClient(),
    config=report_config,
)
db = report.get_db()

pedestrian_distance = db.query.channel(channel_name="Pedestrian_Nearest_Distance_m")
vehicle_speed       = db.query.channel(channel_name="Vehicle_Speed_kph")

pedestrian_proximity_event = BasicEvent(
    name="pedestrian_high_speed_proximity",
    expr=(pedestrian_distance <= 15.0) & (vehicle_speed > 30.0),
    desc="Pedestrian within 15 m of ego while vehicle speed exceeds 30 kph",
)
report.add_event(pedestrian_proximity_event)

page = Page(page_number=1)
report.add_page(page)

print("TSAL event registered: pedestrian_high_speed_proximity")

# COMMAND ----------

# MAGIC %md ## Run the TSAL query and persist `event_instance_fact`

# COMMAND ----------

report.determine_report()
report.persist_results()
print("✓ event_instance_fact written to gold")

# COMMAND ----------

# MAGIC %md ## Inspect detected events

# COMMAND ----------

event_fact_table = f"{cfg.catalog}.{cfg.schema_gold}.{_table_prefix}_event_instance_fact"
event_dimension_table = f"{cfg.catalog}.{cfg.schema_gold}.{_table_prefix}_event_dimension"

# COMMAND ----------

# MAGIC %sql
# MAGIC -- event_instance_fact columns: container_id, event_instance_id, event_id, start_ts, end_ts.
# MAGIC -- The human-readable event name lives on event_dimension (joined on event_id).
# MAGIC SELECT f.container_id, d.event_name, f.start_ts, f.end_ts,
# MAGIC        ROUND((f.end_ts - f.start_ts) / 1000.0, 2) AS duration_s
# MAGIC FROM   ${catalog}.${schema_prefix}_gold.${adapter}_demo_event_instance_fact f
# MAGIC JOIN   ${catalog}.${schema_prefix}_gold.${adapter}_demo_event_dimension     d  USING (event_id)
# MAGIC ORDER  BY f.container_id, f.start_ts

# COMMAND ----------

# MAGIC %md ## Save matched windows as `playlist_items` v1

# COMMAND ----------

# Only one event registered (pedestrian_high_speed_proximity) — every row in
# event_instance_fact belongs to it. If multiple events are added in future,
# filter via JOIN on event_dimension.event_name.
events_df = spark.read.table(event_fact_table)
playlist_rows_df = event_fact_to_playlist_items(
    events_df,
    playlist_id=_playlist_id,
    playlist_version=1,
    event_name="pedestrian_high_speed_proximity",
)
playlist_rows_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_playlist_items)
print(f"wrote {playlist_rows_df.count():,} playlist_items rows (playlist_id={_playlist_id} v1)")

# COMMAND ----------

# MAGIC %md ## Verification — playlist summary

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   playlist_id,
# MAGIC   playlist_version,
# MAGIC   COUNT(*) AS event_windows,
# MAGIC   COUNT(DISTINCT container_id) AS scenes_covered,
# MAGIC   ROUND(SUM(end_ts - start_ts) / 1e6, 1) AS total_seconds
# MAGIC FROM ${catalog}.${schema_prefix}_perception_silver.playlist_items
# MAGIC WHERE playlist_id = '${playlist_id}'
# MAGIC GROUP BY playlist_id, playlist_version

# COMMAND ----------

# MAGIC %md ## Cross-surface query
# MAGIC
# MAGIC TSAL found the *when*. Now we drill into `object_tracks` to see the *what*.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   ot.container_id,
# MAGIC   ot.frame_ts,
# MAGIC   ot.object_id,
# MAGIC   ot.detection_class,
# MAGIC   ROUND(ot.distance_m, 1) AS distance_m,
# MAGIC   ot.azimuth,
# MAGIC   ot.lane_offset
# MAGIC FROM ${catalog}.${schema_prefix}_perception_silver.object_tracks ot
# MAGIC JOIN ${catalog}.${schema_prefix}_perception_silver.playlist_items pi
# MAGIC   ON ot.container_id = pi.container_id
# MAGIC  AND ot.frame_ts BETWEEN pi.start_ts AND pi.end_ts
# MAGIC WHERE pi.playlist_id = '${playlist_id}'
# MAGIC   AND ot.detection_class = 'pedestrian'
# MAGIC   AND ot.distance_m <= 15.0
# MAGIC ORDER BY ot.container_id, ot.frame_ts, ot.distance_m
# MAGIC LIMIT 30

# COMMAND ----------

# MAGIC %md ## Acceptance check
# MAGIC
# MAGIC Hard guard: `event_instance_fact` and `playlist_items` must be non-empty.
# MAGIC An empty `event_instance_fact` with a green job typically means the search
# MAGIC ran but produced no matches — `channel_metrics` was unpopulated, the channel
# MAGIC name tags didn't match, or the predicate doesn't fit any data ranges. Soft
# MAGIC check: ≥ 10 windows is the dataset-quality bar;
# MAGIC fewer means the predicate didn't hit much, not that the pipeline is broken.

# COMMAND ----------

n_events = spark.read.table(event_fact_table).count()
n_windows = spark.read.table(cfg.t_playlist_items).filter(F.col("playlist_id") == _playlist_id).count()

assert n_events > 0, (
    f"event_instance_fact is empty — event search produced no matches. Common causes: "
    f"channel_metrics not populated upstream, channel_name tags missing, or "
    f"predicate doesn't match any data ranges. Table: {event_fact_table}"
)
assert n_windows > 0, f"playlist_items has 0 rows for playlist_id={_playlist_id} (table: {cfg.t_playlist_items})"

print(f"✓ acceptance — {n_events} events; {n_windows} playlist windows")
if n_windows < 10:
    print(f"⚠️  Fewer than 10 windows ({n_windows}) — dataset has limited pedestrian-proximity-at-speed examples; consider a larger variant.")

# COMMAND ----------

# MAGIC %md ## What's next
# MAGIC
# MAGIC - `03_per_event_detail.py` — per-window LiDAR + camera detections + OpenLABEL JSON packages
# MAGIC - `04_visualize.py` — event walkthrough + sensor KPI comparison
