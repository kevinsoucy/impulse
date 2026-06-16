# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Per-event detail: cuboids, bboxes, and OpenLABEL packages
# MAGIC
# MAGIC The "package the matches" step. For every event window the detect step wrote to
# MAGIC `event_instance_fact` (selected by `event_name`):
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

from lib.demo_setup import connect
from lib.openlabel import build_openlabel_for_event, serialize
from lib.annotation_schema import CAMERA_OBJECT_DETECTIONS, LIDAR_OBJECT_DETECTIONS
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

ctx = connect(dbutils, with_adapter=True, extra_widgets=[
    ("event_name", "pedestrian_high_speed_proximity", "Event name (from 02_detect_events)"),
])
cfg = ctx["cfg"]
adapter = ctx["adapter"]
_event_name = ctx["event_name"]
print(f"Processing event '{_event_name}' via adapter={cfg.adapter_name}")

# COMMAND ----------

# MAGIC %md ## Load event windows from `event_instance_fact`
# MAGIC
# MAGIC The detect step (02) persisted one fact row per window for the presence
# MAGIC `BasicEvent`. We read those windows by joining `event_instance_fact` to
# MAGIC `event_dimension` and filtering on `event_name`.

# COMMAND ----------

window_rows = (
    spark.read.table(cfg.t_event_instance_fact)
    .join(spark.read.table(cfg.t_event_dimension), "event_id")
    .filter(F.col("event_name") == _event_name)
    .select("container_id", "event_instance_id", "start_ts", "end_ts")
    .collect()
)

windows_by_container: dict[int, list[tuple[int, int]]] = defaultdict(list)
for r in window_rows:
    windows_by_container[int(r.container_id)].append((int(r.start_ts), int(r.end_ts)))

print(f"Event windows: {len(window_rows):,} across {len(windows_by_container):,} containers")

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

# MAGIC %md ### Inspect — cuboid dimensions plausibility

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

# MAGIC %md ### Check — rows only within event windows

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH window_coverage AS (
# MAGIC   SELECT f.container_id, MIN(f.start_ts) AS win_min, MAX(f.end_ts) AS win_max
# MAGIC   FROM   ${catalog}.${schema_prefix}_gold.${adapter}_demo_event_instance_fact f
# MAGIC   JOIN   ${catalog}.${schema_prefix}_gold.${adapter}_demo_event_dimension     d  USING (event_id)
# MAGIC   WHERE  d.event_name = '${event_name}'
# MAGIC   GROUP  BY f.container_id
# MAGIC ),
# MAGIC detection_span AS (
# MAGIC   SELECT container_id, MIN(frame_ts) AS det_min, MAX(frame_ts) AS det_max, COUNT(*) AS rows
# MAGIC   FROM   ${catalog}.${schema_prefix}_perception_silver.lidar_object_detections
# MAGIC   GROUP  BY container_id
# MAGIC )
# MAGIC SELECT
# MAGIC   s.container_id,
# MAGIC   s.rows,
# MAGIC   s.det_min, w.win_min,
# MAGIC   s.det_max, w.win_max,
# MAGIC   CASE WHEN s.det_min >= w.win_min AND s.det_max <= w.win_max THEN '✓' ELSE '⚠️ leak' END AS gated
# MAGIC FROM detection_span s
# MAGIC JOIN window_coverage w USING (container_id)
# MAGIC ORDER BY s.container_id

# COMMAND ----------

# MAGIC %md ### Check — lidar_object_detections gated to event windows

# COMMAND ----------

n_windows = len(window_rows)
n_lidar = spark.read.table(cfg.t_lidar_object_detections).count()
n_lidar_leaks = spark.sql(f"""
    WITH w AS (
      SELECT f.container_id, MIN(f.start_ts) AS win_min, MAX(f.end_ts) AS win_max
      FROM {cfg.t_event_instance_fact} f
      JOIN {cfg.t_event_dimension}     d USING (event_id)
      WHERE d.event_name = '{_event_name}'
      GROUP BY f.container_id
    ),
    s AS (
      SELECT container_id, MIN(frame_ts) AS det_min, MAX(frame_ts) AS det_max
      FROM {cfg.t_lidar_object_detections} GROUP BY container_id
    )
    SELECT COUNT(*) AS n FROM s JOIN w USING (container_id)
    WHERE s.det_min < w.win_min OR s.det_max > w.win_max
""").collect()[0]["n"]

assert n_windows > 0, f"event_instance_fact has no '{_event_name}' windows — upstream 02_detect_events must populate windows first"
assert n_lidar > 0, f"lidar_object_detections empty despite {n_windows} event windows — table: {cfg.t_lidar_object_detections}"
assert n_lidar_leaks == 0, f"{n_lidar_leaks} containers have lidar detections outside their event windows (temporal leak)"

print(f"✓ Validated — {n_lidar:,} lidar_object_detections rows; all within event windows ({n_windows} windows total)")

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

# MAGIC %md ### Inspect — per-camera detection counts

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

# MAGIC %md ### Check — camera detections within event windows

# COMMAND ----------

# MAGIC %sql
# MAGIC WITH window_coverage AS (
# MAGIC   SELECT f.container_id, MIN(f.start_ts) AS win_min, MAX(f.end_ts) AS win_max
# MAGIC   FROM   ${catalog}.${schema_prefix}_gold.${adapter}_demo_event_instance_fact f
# MAGIC   JOIN   ${catalog}.${schema_prefix}_gold.${adapter}_demo_event_dimension     d  USING (event_id)
# MAGIC   WHERE  d.event_name = '${event_name}'
# MAGIC   GROUP  BY f.container_id
# MAGIC ),
# MAGIC detection_span AS (
# MAGIC   SELECT container_id, MIN(frame_ts) AS det_min, MAX(frame_ts) AS det_max, COUNT(*) AS rows
# MAGIC   FROM   ${catalog}.${schema_prefix}_perception_silver.camera_object_detections
# MAGIC   GROUP  BY container_id
# MAGIC )
# MAGIC SELECT
# MAGIC   s.container_id,
# MAGIC   s.rows,
# MAGIC   CASE WHEN s.det_min >= w.win_min AND s.det_max <= w.win_max THEN '✓' ELSE '⚠️ leak' END AS gated
# MAGIC FROM detection_span s
# MAGIC JOIN window_coverage w USING (container_id)
# MAGIC ORDER BY s.container_id

# COMMAND ----------

# MAGIC %md ### Check — camera_object_detections gated to event windows

# COMMAND ----------

n_camera = spark.read.table(cfg.t_camera_object_detections).count()
n_cam_leaks = spark.sql(f"""
    WITH w AS (
      SELECT f.container_id, MIN(f.start_ts) AS win_min, MAX(f.end_ts) AS win_max
      FROM {cfg.t_event_instance_fact} f
      JOIN {cfg.t_event_dimension}     d USING (event_id)
      WHERE d.event_name = '{_event_name}'
      GROUP BY f.container_id
    ),
    s AS (
      SELECT container_id, MIN(frame_ts) AS det_min, MAX(frame_ts) AS det_max
      FROM {cfg.t_camera_object_detections} GROUP BY container_id
    )
    SELECT COUNT(*) AS n FROM s JOIN w USING (container_id)
    WHERE s.det_min < w.win_min OR s.det_max > w.win_max
""").collect()[0]["n"]

assert n_camera > 0, f"camera_object_detections empty despite {n_windows} event windows — table: {cfg.t_camera_object_detections}"
assert n_cam_leaks == 0, f"{n_cam_leaks} containers have camera detections outside their event windows (temporal leak)"

print(f"✓ Validated — {n_camera:,} camera_object_detections rows; all within event windows")

# COMMAND ----------

# MAGIC %md ## Step 3 — OpenLABEL packages (one JSON per event window)
# MAGIC
# MAGIC OpenLABEL is the **export format**, not the storage format. JSON files are
# MAGIC derived artifacts; Delta tables remain the source of truth. Adapter-specific
# MAGIC metadata strings (annotator, exporter, stream prefix) come from
# MAGIC `adapter.openlabel_metadata()`.

# COMMAND ----------

ol_meta = adapter.openlabel_metadata()
print(f"Export destination: {cfg.volume_openlabel}/{_event_name}/<event_instance_id>.json")
print(f"Annotator:          {ol_meta.get('annotator')}")
print(f"Exporter:           {ol_meta.get('exporter')}")

event_windows = (
    spark.read.table(cfg.t_event_instance_fact)
    .join(spark.read.table(cfg.t_event_dimension), "event_id")
    .filter(F.col("event_name") == _event_name)
    .select("container_id", "event_instance_id", "start_ts", "end_ts")
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

print(f"Event windows: {len(event_windows)}")
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
output_dir = f"{cfg.volume_openlabel}/{_event_name}"
dbutils.fs.mkdirs(output_dir)
print(f"Writing to {output_dir}")

for win_row in event_windows:
    cid       = int(win_row.container_id)
    eid       = win_row.event_instance_id
    start_us  = int(win_row.start_ts)
    end_us    = int(win_row.end_ts)

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
        event_name=_event_name,
        container_id=cid,
        scene_name=scene_name_by_container.get(cid, f"scene_{cid}"),
        start_ts_us=start_us,
        end_ts_us=end_us,
        lidar_detections=in_window_lidar,
        media_paths=in_window_media,
        annotator=ol_meta.get("annotator", "ground_truth"),
        exporter=ol_meta.get("exporter", "demos/byod/lib/openlabel.py"),
        stream_description_prefix=ol_meta.get("stream_description_prefix", ""),
    )

    file_path = f"{output_dir}/{eid}.json"
    with open(file_path, "w") as f:
        f.write(serialize(doc))
    written += 1

print(f"✓ wrote {written} OpenLABEL packages to {output_dir}")

# COMMAND ----------

# MAGIC %md ### Inspect — package contents

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

# MAGIC %md ### Check — one JSON per event window

# COMMAND ----------

n_windows = len(event_windows)

assert n_windows > 0, f"event_instance_fact has no '{_event_name}' windows — upstream 02_detect_events must populate windows first"
assert len(package_files) > 0, f"no OpenLABEL JSON files written to {output_dir} despite {n_windows} event windows"
assert len(package_files) == n_windows, (
    f"OpenLABEL package count ({len(package_files)}) does not match event window count ({n_windows}) — "
    f"expected one JSON per window"
)

print(f"✓ Validated — {len(package_files)} OpenLABEL packages in {output_dir} (= {n_windows} event windows)")

# COMMAND ----------

# MAGIC %md ## What's next
# MAGIC
# MAGIC - `04_visualize.py` — event walkthrough (camera + LiDAR) + sensor KPI comparison
