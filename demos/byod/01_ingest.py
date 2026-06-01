# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest: load source-dataset into the LakeVision silver tables
# MAGIC
# MAGIC The first stage of the demo. Five steps, run in sequence:
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

from lib.demo_setup import connect
from lib.object_tracks_config import ObjectTracksConfig
from lib.scalar_metrics import derive_channel_metrics_from_channels
from lib.silver_schema import EGO_MAP_CONTEXT, OBJECT_MAP_CONTEXT, OBJECT_TRACKS, PERCEPTION_CHANNELS
import impulse_query_engine.schema as core_schema

# COMMAND ----------

# MAGIC %md ## Configuration via widgets

# COMMAND ----------

ctx = connect(dbutils, with_adapter=True, extra_widgets=[
    ("min_confidence", "0.5", "Min confidence floor for object_tracks"),
    ("bootstrap_only", "false", "Stop after metadata ingest (used by bootstrap_job)", ["false", "true"]),
])
cfg = ctx["cfg"]
adapter = ctx["adapter"]
_min_confidence = float(ctx["min_confidence"])
_bootstrap_only = ctx["bootstrap_only"].lower() == "true"
print(f"dataroot: {cfg.dataroot}")

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

# MAGIC %md ### Empty LakeVision tables (from `impulse_query_engine.schema`)

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
# channel_value_labels intentionally omitted — the demo never reads it.

# COMMAND ----------

# MAGIC %md ## Step 2 — Metadata ingest (adapter)
# MAGIC
# MAGIC Writes `container_tags`, `container_metrics`, `channel_tags` rows for every
# MAGIC container (scene / drive segment / log).

# COMMAND ----------

adapter.ingest_metadata(spark)
print("✓ adapter.ingest_metadata complete")

# COMMAND ----------

# MAGIC %md ### Check — metadata ingest produced non-empty rows

# COMMAND ----------

n_ctags = spark.read.table(cfg.t_container_tags).count()
n_cmetrics = spark.read.table(cfg.t_container_metrics).count()
n_chtags = spark.read.table(cfg.t_channel_tags).count()
n_containers = spark.read.table(cfg.t_container_tags).select("container_id").distinct().count()

assert n_ctags > 0,    f"container_tags empty after adapter.ingest_metadata (table: {cfg.t_container_tags})"
assert n_cmetrics > 0, f"container_metrics empty after adapter.ingest_metadata (table: {cfg.t_container_metrics})"
assert n_chtags > 0,   f"channel_tags empty after adapter.ingest_metadata (table: {cfg.t_channel_tags})"
assert n_containers > 0, "no distinct container_id values written"

print(f"✓ Validated — {n_containers} containers; {n_ctags:,} container_tags; {n_cmetrics:,} container_metrics; {n_chtags:,} channel_tags")

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

# MAGIC %md ### Inspect — per-channel summary stats

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

# MAGIC %md ### Check — channels + channel_metrics

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

print(f"✓ Validated — {n_channels:,} channels rows; {n_metrics:,} channel_metrics rows (= {n_distinct} distinct (container_id, channel_id) pairs)")

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

# MAGIC %md ### Inspect — sensor rate per container

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

# MAGIC %md ### Check — UC Volume paths only, no bytes in Delta

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   COUNT(*) AS total_rows,
# MAGIC   SUM(CASE WHEN file_path IS NULL THEN 1 ELSE 0 END) AS null_paths,
# MAGIC   SUM(CASE WHEN file_path NOT LIKE '/Volumes/%' THEN 1 ELSE 0 END) AS non_uc_paths,
# MAGIC   COUNT(DISTINCT format) AS distinct_formats
# MAGIC FROM ${catalog}.${schema_prefix}_silver.perception_channels

# COMMAND ----------

# MAGIC %md ### Check — perception_channels

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

print(f"✓ Validated — {stats['n_rows']:,} perception_channels rows; all file_paths are UC Volume paths")

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

# MAGIC %md ### Check — first scene

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

# MAGIC %md ## Step 5b — Map context (optional third Series)
# MAGIC
# MAGIC Derives per-keyframe `ego_map_context` and `object_map_context` by joining ego
# MAGIC / object positions against the HD map layers — giving the query engine a third
# MAGIC table to compose with `channels` and `object_tracks`. **Presence-gated:** runs
# MAGIC only when the adapter supports map context *and* its map data is staged;
# MAGIC otherwise it is skipped (no failure, logged). Each table is tagged with its
# MAGIC derivation lineage `(name, version, definition_hash)`.

# COMMAND ----------

_map_available = (
    callable(getattr(adapter, "map_context_available", None)) and adapter.map_context_available()
)

if not _map_available:
    reason = (
        "adapter does not support map context"
        if not hasattr(adapter, "map_context_available")
        else f"no map-expansion layers staged at {cfg.map_expansion_dir}"
    )
    print(f"⏭️  Step 5b skipped — {reason}. (map_context tables not written.)")
else:
    def _tag_lineage(table: str, name: str) -> None:
        lin = adapter.map_context_lineage(name)
        props = ", ".join(f"'{k}' = '{v}'" for k, v in lin.items())
        spark.sql(f"ALTER TABLE {table} SET TBLPROPERTIES ({props})")
        print(f"    lineage: {lin}")

    ego_rows, obj_rows = [], []
    for s in scenes:
        ego_rows.extend(adapter.map_to_ego_map_context(s))
        obj_rows.extend(adapter.map_to_object_map_context(s))
    print(f"map_context: {len(ego_rows):,} ego rows, {len(obj_rows):,} object rows across {len(scenes)} scenes")

    ego_df = spark.createDataFrame(ego_rows, EGO_MAP_CONTEXT)
    ego_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_ego_map_context)
    _tag_lineage(cfg.t_ego_map_context, "ego_map_context")
    print(f"wrote {ego_df.count():,} rows to {cfg.t_ego_map_context}")

    obj_df = spark.createDataFrame(obj_rows, OBJECT_MAP_CONTEXT)
    obj_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_object_map_context)
    _tag_lineage(cfg.t_object_map_context, "object_map_context")
    print(f"wrote {obj_df.count():,} rows to {cfg.t_object_map_context}")

    # Acceptance — non-empty, and ego positions resolve into the map (drivable area
    # should be common for a driving recording).
    ego_stats = spark.sql(f"""
        SELECT COUNT(*) AS n, SUM(on_drivable_area) AS on_road,
               COUNT(DISTINCT location) AS locations
        FROM {cfg.t_ego_map_context}
    """).collect()[0]
    assert ego_stats["n"] > 0, f"ego_map_context empty (table: {cfg.t_ego_map_context})"
    assert ego_stats["on_road"] and ego_stats["on_road"] > 0, (
        "ego_map_context has no on_drivable_area frames — map join likely misaligned "
        "(coordinate frame mismatch between ego pose and map layers)."
    )
    print(f"✓ Validated — {ego_stats['n']:,} ego frames, {ego_stats['on_road']:,} on drivable area, {ego_stats['locations']} location(s)")

# COMMAND ----------

# MAGIC %md ### Inspect — class distribution

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

# MAGIC %md ### Inspect — azimuth + lane_offset distribution

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

# MAGIC %md ### Check — object_tracks

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

print(f"✓ Validated — {stats['n_rows']:,} object_tracks rows across {stats['n_containers']} containers, {stats['n_objects']:,} distinct objects, {stats['n_classes']} detection classes")

# COMMAND ----------

# MAGIC %md ## What's next
# MAGIC
# MAGIC - `02_detect_events.py` — cross-series predicate search → `event_instance_fact`
# MAGIC - `03_per_event_detail.py` — per-window LiDAR + camera detections + OpenLABEL export
# MAGIC - `04_visualize.py` — event walkthrough + sensor KPI comparison
