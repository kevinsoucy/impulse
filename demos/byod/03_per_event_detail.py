# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Per-event detail: cuboids, bboxes, and OpenLABEL packages
# MAGIC
# MAGIC The "package the matches" beat. For every TSAL event window in the playlist:
# MAGIC
# MAGIC 1. **LiDAR cuboids** — adapter projects 3D ground-truth boxes inside the window.
# MAGIC 2. **Camera bboxes** — adapter projects the same 3D boxes into each camera plane.
# MAGIC 3. **OpenLABEL export** — one JSON document per event window, written to a UC volume.
# MAGIC
# MAGIC The per-event-detail tables (`lidar_object_detections`, `camera_object_detections`)
# MAGIC are populated only for the matching event windows. This keeps per-frame perception
# MAGIC data scoped to the events that actually use it downstream, rather than materializing
# MAGIC every annotation in the dataset.

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

from collections import defaultdict

from lib.adapter import resolve
from lib.byod_config import BYODConfig
from mda_query_engine.perception.openlabel import build_openlabel_for_event, serialize
from mda_query_engine.perception.schema import CAMERA_OBJECT_DETECTIONS, LIDAR_OBJECT_DETECTIONS

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
adapter = resolve(_adapter_name)(cfg)
print(f"Processing playlist '{_playlist_id}' via adapter={cfg.adapter_name}")

# COMMAND ----------

# MAGIC %md ## Load event windows from `playlist_items`

# COMMAND ----------

playlist_rows = (
    spark.read.table(cfg.t_playlist_items)
    .filter(f"playlist_id = '{_playlist_id}'")
    .select("container_id", "start_ts", "end_ts")
    .collect()
)

windows_by_container: dict[int, list[tuple[int, int]]] = defaultdict(list)
for r in playlist_rows:
    windows_by_container[int(r.container_id)].append((int(r.start_ts), int(r.end_ts)))

print(f"Playlist windows: {len(playlist_rows):,} across {len(windows_by_container):,} containers")

scenes = adapter.scenes()
scenes_by_container = {s.container_id: s for s in scenes}

# COMMAND ----------

# MAGIC %md ## Step 1 — LiDAR cuboids per event window
# MAGIC
# MAGIC Adapter projects every in-window 3D annotation as one `lidar_object_detections`
# MAGIC row. Cuboid geometry lives in typed Delta columns (cx/cy/cz, length/width/height,
# MAGIC yaw); the OpenLABEL JSON for these same rows is generated on demand below.

# COMMAND ----------

all_lidar = []
for container_id, windows in windows_by_container.items():
    scene = scenes_by_container.get(container_id)
    if scene is None:
        print(f"  ⚠️  container {container_id} not found in loader — skipping")
        continue
    rows = adapter.map_to_lidar_detections(scene, windows)
    all_lidar.extend(rows)
    print(f"  {scene.name}: {len(windows):,} windows → {len(rows):,} cuboid rows")

print(f"\nTotal: {len(all_lidar):,} lidar_object_detections rows")

lidar_df = spark.createDataFrame(all_lidar, LIDAR_OBJECT_DETECTIONS)
lidar_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_lidar_object_detections)
print(f"wrote {lidar_df.count():,} rows to {cfg.t_lidar_object_detections}")

# COMMAND ----------

# MAGIC %md ### Verification — cuboid dimensions plausibility

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   detection_class,
# MAGIC   COUNT(*) AS rows,
# MAGIC   ROUND(AVG(length), 2) AS avg_length_m,
# MAGIC   ROUND(AVG(width),  2) AS avg_width_m,
# MAGIC   ROUND(AVG(height), 2) AS avg_height_m,
# MAGIC   ROUND(AVG(SQRT(cx*cx + cy*cy)), 1) AS avg_xy_distance_m
# MAGIC FROM ${catalog}.${schema_prefix}_perception_silver.lidar_object_detections
# MAGIC GROUP BY detection_class
# MAGIC ORDER BY rows DESC

# COMMAND ----------

# MAGIC %md ### Playlist-window sanity check — rows only within playlist windows

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH playlist_coverage AS (
# MAGIC   SELECT container_id, MIN(start_ts) AS plist_min, MAX(end_ts) AS plist_max
# MAGIC   FROM   ${catalog}.${schema_prefix}_perception_silver.playlist_items
# MAGIC   WHERE  playlist_id = '${playlist_id}'
# MAGIC   GROUP  BY container_id
# MAGIC ),
# MAGIC detection_span AS (
# MAGIC   SELECT container_id, MIN(frame_ts) AS det_min, MAX(frame_ts) AS det_max, COUNT(*) AS rows
# MAGIC   FROM   ${catalog}.${schema_prefix}_perception_silver.lidar_object_detections
# MAGIC   GROUP  BY container_id
# MAGIC )
# MAGIC SELECT
# MAGIC   d.container_id,
# MAGIC   d.rows,
# MAGIC   d.det_min, p.plist_min,
# MAGIC   d.det_max, p.plist_max,
# MAGIC   CASE WHEN d.det_min >= p.plist_min AND d.det_max <= p.plist_max THEN '✓' ELSE '⚠️ leak' END AS gated
# MAGIC FROM detection_span d
# MAGIC JOIN playlist_coverage p USING (container_id)
# MAGIC ORDER BY d.container_id

# COMMAND ----------

# MAGIC %md ### Acceptance — lidar_object_detections gated to playlist windows

# COMMAND ----------

n_playlist = spark.read.table(cfg.t_playlist_items).filter(f"playlist_id = '{_playlist_id}'").count()
n_lidar = spark.read.table(cfg.t_lidar_object_detections).count()
n_lidar_leaks = spark.sql(f"""
    WITH p AS (
      SELECT container_id, MIN(start_ts) AS plist_min, MAX(end_ts) AS plist_max
      FROM {cfg.t_playlist_items} WHERE playlist_id = '{_playlist_id}'
      GROUP BY container_id
    ),
    d AS (
      SELECT container_id, MIN(frame_ts) AS det_min, MAX(frame_ts) AS det_max
      FROM {cfg.t_lidar_object_detections} GROUP BY container_id
    )
    SELECT COUNT(*) AS n FROM d JOIN p USING (container_id)
    WHERE d.det_min < p.plist_min OR d.det_max > p.plist_max
""").collect()[0]["n"]

assert n_playlist > 0, "playlist_items has no rows for this playlist — upstream t02_detect_events must populate windows first"
assert n_lidar > 0, f"lidar_object_detections empty despite non-empty playlist ({n_playlist} windows) — table: {cfg.t_lidar_object_detections}"
assert n_lidar_leaks == 0, f"{n_lidar_leaks} containers have lidar detections outside their playlist windows (temporal leak)"

print(f"✓ acceptance — {n_lidar:,} lidar_object_detections rows; all within playlist windows ({n_playlist} windows total)")

# COMMAND ----------

# MAGIC %md ## Step 2 — Camera bboxes per event window
# MAGIC
# MAGIC Adapter projects every in-window 3D cuboid into each camera plane. Note: the
# MAGIC dataset's 3D ground truth produces *projections* of 3D boxes, not per-camera
# MAGIC detector output. This is the correct representation for a demo without a live
# MAGIC inference pipeline.

# COMMAND ----------

all_camera = []
for container_id, windows in windows_by_container.items():
    scene = scenes_by_container.get(container_id)
    if scene is None:
        continue  # already warned in step 1
    rows = adapter.map_to_camera_detections(scene, windows)
    all_camera.extend(rows)
    print(f"  {scene.name}: {len(windows):,} windows → {len(rows):,} camera bbox rows")

print(f"\nTotal: {len(all_camera):,} camera_object_detections rows")

cam_df = spark.createDataFrame(all_camera, CAMERA_OBJECT_DETECTIONS)
cam_df.write.format("delta").mode("overwrite").saveAsTable(cfg.t_camera_object_detections)
print(f"wrote {cam_df.count():,} rows to {cfg.t_camera_object_detections}")

# COMMAND ----------

# MAGIC %md ### Verification — per-camera detection counts

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   sensor_id,
# MAGIC   detection_class,
# MAGIC   COUNT(*) AS rows,
# MAGIC   ROUND(AVG(x2 - x1), 0) AS avg_bbox_width_px,
# MAGIC   ROUND(AVG(y2 - y1), 0) AS avg_bbox_height_px
# MAGIC FROM ${catalog}.${schema_prefix}_perception_silver.camera_object_detections
# MAGIC GROUP BY sensor_id, detection_class
# MAGIC ORDER BY sensor_id, rows DESC

# COMMAND ----------

# MAGIC %md ### Playlist-window sanity check — camera detections within playlist windows

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH playlist_coverage AS (
# MAGIC   SELECT container_id, MIN(start_ts) AS plist_min, MAX(end_ts) AS plist_max
# MAGIC   FROM   ${catalog}.${schema_prefix}_perception_silver.playlist_items
# MAGIC   WHERE  playlist_id = '${playlist_id}'
# MAGIC   GROUP  BY container_id
# MAGIC ),
# MAGIC detection_span AS (
# MAGIC   SELECT container_id, MIN(frame_ts) AS det_min, MAX(frame_ts) AS det_max, COUNT(*) AS rows
# MAGIC   FROM   ${catalog}.${schema_prefix}_perception_silver.camera_object_detections
# MAGIC   GROUP  BY container_id
# MAGIC )
# MAGIC SELECT
# MAGIC   d.container_id,
# MAGIC   d.rows,
# MAGIC   CASE WHEN d.det_min >= p.plist_min AND d.det_max <= p.plist_max THEN '✓' ELSE '⚠️ leak' END AS gated
# MAGIC FROM detection_span d
# MAGIC JOIN playlist_coverage p USING (container_id)
# MAGIC ORDER BY d.container_id

# COMMAND ----------

# MAGIC %md ### Acceptance — camera_object_detections gated to playlist windows

# COMMAND ----------

n_camera = spark.read.table(cfg.t_camera_object_detections).count()
n_cam_leaks = spark.sql(f"""
    WITH p AS (
      SELECT container_id, MIN(start_ts) AS plist_min, MAX(end_ts) AS plist_max
      FROM {cfg.t_playlist_items} WHERE playlist_id = '{_playlist_id}'
      GROUP BY container_id
    ),
    d AS (
      SELECT container_id, MIN(frame_ts) AS det_min, MAX(frame_ts) AS det_max
      FROM {cfg.t_camera_object_detections} GROUP BY container_id
    )
    SELECT COUNT(*) AS n FROM d JOIN p USING (container_id)
    WHERE d.det_min < p.plist_min OR d.det_max > p.plist_max
""").collect()[0]["n"]

assert n_camera > 0, f"camera_object_detections empty despite non-empty playlist ({n_playlist} windows) — table: {cfg.t_camera_object_detections}"
assert n_cam_leaks == 0, f"{n_cam_leaks} containers have camera detections outside their playlist windows (temporal leak)"

print(f"✓ acceptance — {n_camera:,} camera_object_detections rows; all within playlist windows")

# COMMAND ----------

# MAGIC %md ## Step 3 — OpenLABEL packages (one JSON per event window)
# MAGIC
# MAGIC OpenLABEL is the **export format**, not the storage format. JSON files are
# MAGIC derived artifacts; Delta tables remain the source of truth. Adapter-specific
# MAGIC metadata strings (annotator, exporter, stream prefix) come from
# MAGIC `adapter.openlabel_metadata()`.

# COMMAND ----------

ol_meta = adapter.openlabel_metadata()
print(f"Export destination: {cfg.volume_openlabel}/{_playlist_id}/<event_id>.json")
print(f"Annotator:          {ol_meta.get('annotator')}")
print(f"Exporter:           {ol_meta.get('exporter')}")

playlist = (
    spark.read.table(cfg.t_playlist_items)
    .filter(f"playlist_id = '{_playlist_id}'")
    .orderBy("container_id", "start_ts")
    .collect()
)

perception_with_names = spark.sql(f"""
SELECT pc.container_id, pc.channel_id, ct.value AS channel_name,
       pc.timestamp, pc.file_path, pc.format
FROM   {cfg.t_perception_channels} pc
JOIN   {cfg.t_channel_tags} ct
  ON   pc.container_id = ct.container_id
 AND   pc.channel_id   = ct.channel_id
 AND   ct.key = 'channel_name'
""").collect()

lidar_rows = spark.read.table(cfg.t_lidar_object_detections).collect()

scene_name_rows = spark.sql(f"""
SELECT container_id, value AS scene_name
FROM   {cfg.t_container_tags}
WHERE  key = 'scene_name'
""").collect()
scene_name_by_container = {int(r.container_id): r.scene_name for r in scene_name_rows}

print(f"Playlist windows: {len(playlist)}")
print(f"Lidar detections: {len(lidar_rows):,}")
print(f"Perception channel rows: {len(perception_with_names):,}")

# COMMAND ----------

# MAGIC %md ### Build and write the JSON packages

# COMMAND ----------

lidar_by_container = defaultdict(list)
for r in lidar_rows:
    lidar_by_container[int(r.container_id)].append(r.asDict())

media_by_container = defaultdict(list)
for r in perception_with_names:
    media_by_container[int(r.container_id)].append(r.asDict())

written = 0
output_dir = f"{cfg.volume_openlabel}/{_playlist_id}"
dbutils.fs.mkdirs(output_dir)
print(f"Writing to {output_dir}")

for pi_row in playlist:
    cid       = int(pi_row.container_id)
    eid       = pi_row.event_id
    start_us  = int(pi_row.start_ts)
    end_us    = int(pi_row.end_ts)
    version   = int(pi_row.playlist_version)
    event_nm  = pi_row.event_name

    in_window_lidar = [
        d for d in lidar_by_container[cid]
        if start_us <= int(d["frame_ts"]) <= end_us
    ]
    in_window_media = [
        m for m in media_by_container[cid]
        if start_us <= int(m["timestamp"]) <= end_us
    ]
    if not in_window_lidar:
        continue

    doc = build_openlabel_for_event(
        event_id=eid,
        event_name=event_nm,
        container_id=cid,
        scene_name=scene_name_by_container.get(cid, f"scene_{cid}"),
        start_ts_us=start_us,
        end_ts_us=end_us,
        lidar_detections=in_window_lidar,
        media_paths=in_window_media,
        playlist_id=_playlist_id,
        playlist_version=version,
        annotator=ol_meta.get("annotator", "ground_truth"),
        exporter=ol_meta.get("exporter", "mda_query_engine/perception/openlabel.py"),
        stream_description_prefix=ol_meta.get("stream_description_prefix", ""),
    )

    file_path = f"{output_dir}/{eid}.json"
    with open(file_path, "w") as f:
        f.write(serialize(doc))
    written += 1

print(f"✓ wrote {written} OpenLABEL packages to {output_dir}")

# COMMAND ----------

# MAGIC %md ### Verification — package contents

# COMMAND ----------

package_files = [
    f.path
    for f in dbutils.fs.ls(output_dir)
    if f.path.endswith(".json")
]
print(f"{len(package_files)} packages in {output_dir}\n")

if package_files:
    sample = package_files[0]
    with open(sample.replace("dbfs:", "")) as f:
        import json as _json
        doc = _json.load(f)
    print(f"Sample: {sample}")
    print(f"  schema_version: {doc['openlabel']['metadata']['schema_version']}")
    print(f"  name:           {doc['openlabel']['metadata']['name']}")
    print(f"  streams:        {list(doc['openlabel']['streams'].keys())}")
    print(f"  object count:   {len(doc['openlabel']['objects'])}")
    print(f"  frame count:    {len(doc['openlabel']['frames'])}")
    print(f"  first frame:    {next(iter(doc['openlabel']['frames'].values()))['frame_properties']['timestamp']}")

# COMMAND ----------

# MAGIC %md ### Acceptance — one JSON per playlist window

# COMMAND ----------

n_windows = spark.read.table(cfg.t_playlist_items).filter(f"playlist_id = '{_playlist_id}'").count()

assert n_windows > 0, "playlist_items has no rows for this playlist — upstream t02_detect_events must populate windows first"
assert len(package_files) > 0, f"no OpenLABEL JSON files written to {output_dir} despite {n_windows} playlist windows"
assert len(package_files) == n_windows, (
    f"OpenLABEL package count ({len(package_files)}) does not match playlist window count ({n_windows}) — "
    f"expected one JSON per window"
)

print(f"✓ acceptance — {len(package_files)} OpenLABEL packages in {output_dir} (= {n_windows} playlist windows)")

# COMMAND ----------

# MAGIC %md ## What's next
# MAGIC
# MAGIC - `04_visualize.py` — event walkthrough (camera + LiDAR) + sensor KPI comparison
