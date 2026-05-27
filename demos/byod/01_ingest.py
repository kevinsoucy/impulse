# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest: load source-dataset into the LakeVision silver tables
# MAGIC
# MAGIC The first beat of the demo. Four steps, run in sequence:
# MAGIC
# MAGIC 1. **Provision** — schemas + volumes + empty LakeVision Delta tables (idempotent).
# MAGIC 2. **Metadata** — adapter writes `container_tags`, `container_metrics`, `channel_tags` rows.
# MAGIC 3. **Scalars** — adapter populates `channels` (and `channel_metrics` is derived from it).
# MAGIC 4. **Perception channels** — adapter registers per-frame camera + LiDAR file paths.
# MAGIC 5. **Object tracks** — adapter assembles per-frame `object_tracks` rows from annotations.
# MAGIC
# MAGIC By the end, every silver table that downstream notebooks depend on is populated.
# MAGIC
# MAGIC **Prereqs:** the source dataset has already been staged into the raw UC Volume —
# MAGIC see `00_download.py` for the per-adapter staging steps.
# MAGIC
# MAGIC **bootstrap_only mode:** set `bootstrap_only=true` to stop after metadata ingest.
# MAGIC The `bootstrap_job` runs in this mode to (re-)initialize tables without re-running
# MAGIC the full scalar/perception/object-track ingest.

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

from pyspark.sql import Row

from lib.adapter import resolve
from lib.byod_config import BYODConfig
from mda_query_engine.perception import ObjectTracksConfig, derive_channel_metrics_from_channels
from mda_query_engine.perception.schema import (
    OBJECT_TRACKS,
    PERCEPTION_CHANNELS,
)
import mda_query_engine.schema as core_schema
from mda_query_engine.schema import CHANNEL_VALUE_LABELS

# COMMAND ----------

# MAGIC %md ## Configuration via widgets

# COMMAND ----------

dbutils.widgets.text("adapter",         "nuscenes",                "Adapter name (e.g. nuscenes, a2d2)")
dbutils.widgets.text("dataset_version", "v1.0-mini",               "Adapter-specific variant")
dbutils.widgets.text("catalog",         "main",                    "UC catalog (must exist)")
dbutils.widgets.text("schema_prefix",   "lakevision_demo",         "Schema prefix")
dbutils.widgets.text("min_confidence",  "0.5",                     "Min confidence floor for object_tracks (ObjectTracksConfig)")
dbutils.widgets.dropdown("bootstrap_only", "false", ["false", "true"],
                         "If true, stop after metadata ingest (used by bootstrap_job)")

_adapter_name = dbutils.widgets.get("adapter")
_dataset_version = dbutils.widgets.get("dataset_version") or None
_catalog = dbutils.widgets.get("catalog")
_schema_prefix = dbutils.widgets.get("schema_prefix")
_min_confidence = float(dbutils.widgets.get("min_confidence"))
_bootstrap_only = dbutils.widgets.get("bootstrap_only").lower() == "true"

cfg = BYODConfig.for_adapter(
    adapter_name=_adapter_name,
    dataset_version=_dataset_version,
    catalog=_catalog,
    schema_prefix=_schema_prefix,
)
adapter = resolve(_adapter_name)(cfg)

print(f"adapter:         {cfg.adapter_name}")
print(f"dataset_version: {cfg.dataset_version}")
print(f"dataroot:        {cfg.dataroot}")
print(f"silver schema:   {cfg.catalog}.{cfg.schema_silver}")
print(f"perception:      {cfg.catalog}.{cfg.schema_perception}")
print(f"gold:            {cfg.catalog}.{cfg.schema_gold}")
print(f"bootstrap_only:  {_bootstrap_only}")

# COMMAND ----------

# MAGIC %md ## Step 1 — Provision schemas, volumes, and empty tables

# COMMAND ----------

for s in cfg.schemas_to_create():
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {cfg.catalog}.{s}")
    print(f"✓ schema {cfg.catalog}.{s}")
for v in cfg.volumes_to_create():
    spark.sql(f"CREATE VOLUME IF NOT EXISTS {cfg.catalog}.{cfg.schema_silver}.{v}")
    print(f"✓ volume {cfg.catalog}.{cfg.schema_silver}.{v}")

# COMMAND ----------

# MAGIC %md ### Empty LakeVision tables (from `mda_query_engine.perception.schema`)

# COMMAND ----------

def _create_empty_table(name: str, schema) -> None:
    spark.createDataFrame([], schema).write.format("delta").mode("ignore").saveAsTable(name)
    print(f"✓ table {name}")

# Impulse core tables (silver).
_create_empty_table(cfg.t_container_tags, core_schema.CONTAINER_TAGS)
_create_empty_table(cfg.t_container_metrics, core_schema.CONTAINER_METRICS)
_create_empty_table(cfg.t_channel_tags, core_schema.CHANNEL_TAGS)
_create_empty_table(cfg.t_channel_metrics, core_schema.CHANNEL_METRICS)
_create_empty_table(cfg.t_channels, core_schema.CHANNELS_SCHEMA)

# LakeVision silver.
_create_empty_table(cfg.t_perception_channels, PERCEPTION_CHANNELS)
_create_empty_table(cfg.t_channel_value_labels, CHANNEL_VALUE_LABELS)

# COMMAND ----------

# MAGIC %md ## Step 2 — Metadata ingest (adapter)
# MAGIC
# MAGIC Writes `container_tags`, `container_metrics`, `channel_tags` rows for every
# MAGIC container (scene / drive segment / log).

# COMMAND ----------

adapter.ingest_metadata(spark)
print("✓ adapter.ingest_metadata complete")

# COMMAND ----------

# MAGIC %md ### Verification — container_tags by key

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT key, COUNT(*) AS row_count, COUNT(DISTINCT container_id) AS containers
# MAGIC FROM   ${catalog}.${schema_prefix}_silver.container_tags
# MAGIC GROUP  BY key
# MAGIC ORDER  BY key

# COMMAND ----------

# MAGIC %md ### Verification — channel_tags by key

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT key, COUNT(*) AS row_count
# MAGIC FROM   ${catalog}.${schema_prefix}_silver.channel_tags
# MAGIC GROUP  BY key
# MAGIC ORDER  BY key

# COMMAND ----------

# MAGIC %md ### Acceptance — metadata ingest produced non-empty rows
# MAGIC
# MAGIC Every notebook ends sections with `assert`-based row-count checks so the
# MAGIC task fails fast if the notebook ran cleanly but wrote no data.

# COMMAND ----------

n_ctags = spark.read.table(cfg.t_container_tags).count()
n_cmetrics = spark.read.table(cfg.t_container_metrics).count()
n_chtags = spark.read.table(cfg.t_channel_tags).count()
n_containers = spark.read.table(cfg.t_container_tags).select("container_id").distinct().count()

assert n_ctags > 0,    f"container_tags empty after adapter.ingest_metadata (table: {cfg.t_container_tags})"
assert n_cmetrics > 0, f"container_metrics empty after adapter.ingest_metadata (table: {cfg.t_container_metrics})"
assert n_chtags > 0,   f"channel_tags empty after adapter.ingest_metadata (table: {cfg.t_channel_tags})"
assert n_containers > 0, "no distinct container_id values written"

print(f"✓ acceptance — {n_containers} containers; {n_ctags:,} container_tags; {n_cmetrics:,} container_metrics; {n_chtags:,} channel_tags")

# COMMAND ----------

# MAGIC %md ### Stop here if bootstrap_only=true
# MAGIC
# MAGIC The `bootstrap_job` runs the notebook with `bootstrap_only=true` as a recoverable
# MAGIC re-initialization step. The full `run_all_job` runs with the default
# MAGIC `bootstrap_only=false` and continues through scalars / perception / object_tracks.

# COMMAND ----------

if _bootstrap_only:
    dbutils.notebook.exit("bootstrap_only=true — stopped after metadata ingest")

# COMMAND ----------

# MAGIC %md ## Step 3 — Scalars ingest (adapter → `channels`)
# MAGIC
# MAGIC Each adapter chooses how to populate `channels`:
# MAGIC - **NuScenes** synthesizes 9 scalars per scene from ego pose + annotations (no CAN bus).
# MAGIC - **A2D2** decodes bus-signal JSON into rows (real ECU signals).
# MAGIC - **MDF4-based adapters** decode binary CAN/ECU recordings via Impulse's MDF4 reader.
# MAGIC
# MAGIC The output schema is identical across adapters: every downstream notebook is byte-identical.

# COMMAND ----------

channels_df = adapter.scalar_source(spark)
total = channels_df.count()
print(f"Adapter produced {total:,} channel rows")

channels_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_channels)
print(f"wrote {total:,} rows to {cfg.t_channels}")

# COMMAND ----------

# MAGIC %md ### Derive `channel_metrics` from `channels`
# MAGIC
# MAGIC The event-detection step (`02_detect_events.py`) joins `channel_metrics` against
# MAGIC `channels` on `(container_id, channel_id)` to scope its scan — any channel
# MAGIC without a matching `channel_metrics` row is dropped from event search and would
# MAGIC return zero events. Per-channel summary statistics are computed too so the
# MAGIC table is also useful for ad-hoc inspection.

# COMMAND ----------

channel_metrics_df = derive_channel_metrics_from_channels(spark, cfg.t_channels)
channel_metrics_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_channel_metrics)
print(f"wrote {channel_metrics_df.count():,} rows to {cfg.t_channel_metrics}")

# COMMAND ----------

# MAGIC %md ### Verification — per-channel summary stats

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   c.channel_id,
# MAGIC   ct.value AS channel_name,
# MAGIC   COUNT(*) AS rows,
# MAGIC   COUNT(DISTINCT c.container_id) AS containers,
# MAGIC   ROUND(MIN(c.value), 2) AS v_min,
# MAGIC   ROUND(MAX(c.value), 2) AS v_max,
# MAGIC   ROUND(AVG(c.value), 2) AS v_avg
# MAGIC FROM   ${catalog}.${schema_prefix}_silver.channels c
# MAGIC JOIN   ${catalog}.${schema_prefix}_silver.channel_tags ct
# MAGIC   ON   c.container_id = ct.container_id
# MAGIC  AND   c.channel_id   = ct.channel_id
# MAGIC  AND   ct.key         = 'channel_name'
# MAGIC GROUP  BY c.channel_id, ct.value
# MAGIC ORDER  BY c.channel_id

# COMMAND ----------

# MAGIC %md ### Acceptance — channels + channel_metrics

# COMMAND ----------

n_channels = spark.read.table(cfg.t_channels).count()
n_metrics = spark.read.table(cfg.t_channel_metrics).count()
n_distinct = spark.sql(f"""
    SELECT COUNT(*) AS n FROM (
      SELECT DISTINCT container_id, channel_id FROM {cfg.t_channels}
    )
""").collect()[0]["n"]

assert n_channels > 0, f"channels empty after adapter.scalar_source (table: {cfg.t_channels})"
assert n_metrics > 0, f"channel_metrics empty — event detection in 02_detect_events.py would return 0 events (table: {cfg.t_channel_metrics})"
assert n_metrics == n_distinct, (
    f"channel_metrics row count ({n_metrics}) does not match distinct "
    f"(container_id, channel_id) in channels ({n_distinct}) — derivation is incomplete"
)

print(f"✓ acceptance — {n_channels:,} channels rows; {n_metrics:,} channel_metrics rows (= {n_distinct} distinct (container_id, channel_id) pairs)")

# COMMAND ----------

# MAGIC %md ## Step 4 — Perception channels (camera + LiDAR file paths)
# MAGIC
# MAGIC One `perception_channels` row per camera frame and per LiDAR scan. The
# MAGIC `file_path` column points at a UC Volume that holds the source files —
# MAGIC image and LiDAR bytes never enter Delta, so the table stays small and the
# MAGIC raw artifacts remain governed by the volume's permissions.

# COMMAND ----------

rows = [Row(**d) for d in adapter.perception_paths()]
print(f"Adapter yielded {len(rows):,} perception_channels rows")

perception_df = spark.createDataFrame(rows, PERCEPTION_CHANNELS)
perception_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_perception_channels)
print(f"wrote {perception_df.count():,} rows to {cfg.t_perception_channels}")

# COMMAND ----------

# MAGIC %md ### Verification — sensor rate per container

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   pc.container_id,
# MAGIC   ct.value AS sensor,
# MAGIC   COUNT(*) AS rows,
# MAGIC   ROUND((MAX(pc.timestamp) - MIN(pc.timestamp)) / 1e6, 1) AS span_seconds,
# MAGIC   ROUND(COUNT(*) / NULLIF((MAX(pc.timestamp) - MIN(pc.timestamp)) / 1e6, 0), 1) AS effective_hz
# MAGIC FROM ${catalog}.${schema_prefix}_silver.perception_channels pc
# MAGIC JOIN ${catalog}.${schema_prefix}_silver.channel_tags ct
# MAGIC   ON pc.container_id = ct.container_id
# MAGIC  AND pc.channel_id   = ct.channel_id
# MAGIC  AND ct.key = 'channel_name'
# MAGIC GROUP BY pc.container_id, ct.value
# MAGIC ORDER BY pc.container_id, ct.value

# COMMAND ----------

# MAGIC %md ### Sanity check — UC Volume paths only, no bytes in Delta

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*) AS total_rows,
# MAGIC   SUM(CASE WHEN file_path IS NULL THEN 1 ELSE 0 END) AS null_paths,
# MAGIC   SUM(CASE WHEN file_path NOT LIKE '/Volumes/%' THEN 1 ELSE 0 END) AS non_uc_paths,
# MAGIC   COUNT(DISTINCT format) AS distinct_formats
# MAGIC FROM ${catalog}.${schema_prefix}_silver.perception_channels

# COMMAND ----------

# MAGIC %md ### Acceptance — perception_channels

# COMMAND ----------

stats = spark.sql(f"""
    SELECT COUNT(*) AS n_rows,
           SUM(CASE WHEN file_path IS NULL THEN 1 ELSE 0 END) AS n_null,
           SUM(CASE WHEN file_path NOT LIKE '/Volumes/%' THEN 1 ELSE 0 END) AS n_non_uc
    FROM {cfg.t_perception_channels}
""").collect()[0]

assert stats["n_rows"] > 0, f"perception_channels empty after adapter.perception_paths (table: {cfg.t_perception_channels})"
assert stats["n_null"] == 0, f"perception_channels has {stats['n_null']} rows with NULL file_path"
assert stats["n_non_uc"] == 0, f"perception_channels has {stats['n_non_uc']} rows with non-UC-Volume paths (bytes-in-Delta violation)"

print(f"✓ acceptance — {stats['n_rows']:,} perception_channels rows; all file_paths are UC Volume paths")

# COMMAND ----------

# MAGIC %md ## Step 5 — Object tracks (per-frame fused object table)
# MAGIC
# MAGIC One row per object per frame. Sensor provenance lives in the `source` column,
# MAGIC not in separate per-sensor tables. `ObjectTracksConfig` enforces the TSAL-gated
# MAGIC default + 2 Hz Nyquist floor for full-stride mode.

# COMMAND ----------

scenes = adapter.scenes()
otc = ObjectTracksConfig.full_stride(stride_hz=2.0, min_confidence=_min_confidence)
print(f"Populating object_tracks for {len(scenes)} scenes via adapter={cfg.adapter_name}")
print(f"Mode:              {otc.mode}")
print(f"Stride:            {otc.full_stride_hz} Hz")
print(f"Min confidence:    {otc.min_confidence}")

# COMMAND ----------

# MAGIC %md ### Smoke test — first scene

# COMMAND ----------

smoke = adapter.map_to_object_tracks(scenes[0], min_confidence=otc.min_confidence)
print(f"scene 0 ({scenes[0].name}): {len(smoke):,} object_tracks rows")
if smoke:
    print("\nFirst 5 rows:")
    for r in smoke[:5]:
        print(
            f"  ts={r['frame_ts']}  obj={r['object_id']:<20}  cls={r['detection_class']:<12}  "
            f"d={r['distance_m']:5.1f}m  lane={r['lane_offset']:+d}  az={r['azimuth']:<11}  "
            f"rv={r['relative_velocity_ms']}"
        )

# COMMAND ----------

# MAGIC %md ### Map every scene

# COMMAND ----------

all_rows = []
for s in scenes:
    rows = adapter.map_to_object_tracks(s, min_confidence=otc.min_confidence)
    all_rows.extend(rows)
    print(f"  {s.name}: {len(rows):,} rows")

print(f"\nTotal: {len(all_rows):,} object_tracks rows across {len(scenes)} scenes")

object_tracks_df = spark.createDataFrame(all_rows, OBJECT_TRACKS)
object_tracks_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_object_tracks)
print(f"wrote {object_tracks_df.count():,} rows to {cfg.t_object_tracks}")

# COMMAND ----------

# MAGIC %md ### Verification — class distribution

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   detection_class,
# MAGIC   COUNT(*) AS rows,
# MAGIC   COUNT(DISTINCT object_id) AS distinct_objects,
# MAGIC   ROUND(AVG(distance_m), 1) AS avg_distance_m,
# MAGIC   ROUND(MIN(distance_m), 1) AS min_distance_m
# MAGIC FROM ${catalog}.${schema_prefix}_perception_silver.object_tracks
# MAGIC GROUP BY detection_class
# MAGIC ORDER BY rows DESC

# COMMAND ----------

# MAGIC %md ### Verification — azimuth + lane_offset distribution

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   azimuth,
# MAGIC   lane_offset,
# MAGIC   COUNT(*) AS rows
# MAGIC FROM ${catalog}.${schema_prefix}_perception_silver.object_tracks
# MAGIC WHERE detection_class IN ('pedestrian', 'car', 'cyclist')
# MAGIC GROUP BY azimuth, lane_offset
# MAGIC ORDER BY rows DESC

# COMMAND ----------

# MAGIC %md ### Sample query — object-tracks use case
# MAGIC
# MAGIC "Find every pedestrian within 10m of ego in the same lane."

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT container_id, frame_ts, object_id, distance_m, azimuth
# MAGIC FROM ${catalog}.${schema_prefix}_perception_silver.object_tracks
# MAGIC WHERE detection_class = 'pedestrian'
# MAGIC   AND distance_m <= 10.0
# MAGIC   AND lane_offset = 0
# MAGIC ORDER BY distance_m
# MAGIC LIMIT 20

# COMMAND ----------

# MAGIC %md ### Acceptance — object_tracks

# COMMAND ----------

stats = spark.sql(f"""
    SELECT COUNT(*)                      AS n_rows,
           COUNT(DISTINCT container_id)  AS n_containers,
           COUNT(DISTINCT object_id)     AS n_objects,
           COUNT(DISTINCT detection_class) AS n_classes
    FROM {cfg.t_object_tracks}
""").collect()[0]

assert stats["n_rows"] > 0, f"object_tracks empty after adapter.map_to_object_tracks (table: {cfg.t_object_tracks})"
assert stats["n_containers"] > 0, "object_tracks has no distinct container_id values"
assert stats["n_classes"] > 0, "object_tracks has no detection_class values"

print(f"✓ acceptance — {stats['n_rows']:,} object_tracks rows across {stats['n_containers']} containers, {stats['n_objects']:,} distinct objects, {stats['n_classes']} detection classes")

# COMMAND ----------

# MAGIC %md ## What's next
# MAGIC
# MAGIC - `02_detect_events.py` — TSAL scenario search → `event_instance_fact`, `playlist_items`
# MAGIC - `03_per_event_detail.py` — per-window LiDAR + camera detections + OpenLABEL export
# MAGIC - `04_visualize.py` — event walkthrough + sensor KPI comparison
