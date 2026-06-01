# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — Detect events: one predicate across multiple tables
# MAGIC
# MAGIC Write a single predicate that spans a per-object detection table *and* a
# MAGIC scalar signal channel, and Impulse returns every time window where it holds:
# MAGIC
# MAGIC > *"a pedestrian within 15 m **AND** vehicle speed > 30 kph"*
# MAGIC
# MAGIC `object_tracks` (the per-frame per-object detections) is registered as a
# MAGIC **Series**; `vehicle_speed` is a scalar **channel**. Both resolve to
# MAGIC time-intervals, so they combine with `&` in one expression — the engine
# MAGIC correlates the two tables per recording in a single step, with no manual join
# MAGIC and no pre-aggregation. An `EntityEvent` additionally records *which*
# MAGIC pedestrian triggered each window in the `entity_key` column. The final event
# MAGIC goes further — **cross-entity correlation**: two independent objects (a
# MAGIC vulnerable road user and a car) overlapping in time, authored with string
# MAGIC predicate operators (`isin` / `startswith` / `contains`).

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
from lib.series_defs import register_object_tracks

from impulse_reporting.core.page import Page
from impulse_reporting.core.report import Report
from impulse_reporting.events.basic_event import BasicEvent
from impulse_reporting.events.entity_event import EntityEvent
from pyspark.sql import functions as F

# COMMAND ----------

# MAGIC %md ## Configuration

# COMMAND ----------

ctx = connect(dbutils, report_table_suffix="demo")
cfg = ctx["cfg"]
report_config = ctx["report_config"]

# COMMAND ----------

# MAGIC %md ## Register `object_tracks` as a Series
# MAGIC
# MAGIC `object_tracks` is the per-frame per-object detection
# MAGIC table; registering it lets a query reference its columns (`detection_class`,
# MAGIC `distance_m`, `lane_offset`, …) directly. It is fused/sensor-agnostic, so the
# MAGIC registration synthesizes a single constant signal `"fusion"` (see
# MAGIC `lib/series_defs.py`).

# COMMAND ----------

report = Report(
    name=f"{cfg.adapter_name}_pedestrian_proximity",
    spark=spark,
    workspace_client=WorkspaceClient(),
    config=report_config,
)
db = report.get_db()
register_object_tracks(db, cfg.t_object_tracks)
print("Series registered: object_tracks (signal_id='fusion')")

# COMMAND ----------

# MAGIC %md ## Define the event — one predicate across two tables
# MAGIC
# MAGIC **Question:**
# MAGIC > *"In which time intervals was a pedestrian within 15 m AND the vehicle moving faster than 30 kph?"*
# MAGIC
# MAGIC `ot.distance_m` is a per-object **series** column; `vehicle_speed` is a scalar
# MAGIC **channel**. Both produce time-intervals, so they combine with `&` in one
# MAGIC expression — the engine brings the two tables together per recording.

# COMMAND ----------

ot = db.query.series("object_tracks")
vehicle_speed = db.query.channel(channel_name="Vehicle_Speed_kph")


def risky_predicate():
    """The risky-driving predicate: a pedestrian within 15 m while the vehicle is
    moving faster than 30 kph. The class and distance clauses must hold for the
    *same* object, so they are grouped with `.entity_condition()`. Built fresh per
    event so each event owns its own predicate object.
    """
    pedestrian_close = (
        (ot.detection_class == "pedestrian") & (ot.distance_m <= 15.0)
    ).entity_condition()
    return pedestrian_close & (vehicle_speed > 30.0)


# Presence — "did it happen in this recording?"
report.add_event(
    BasicEvent(
        name="pedestrian_high_speed_proximity",
        expr=risky_predicate(),
        desc="Pedestrian within 15 m of ego while vehicle speed exceeds 30 kph",
    )
)

# Which pedestrian — one output row per triggering object (entity_key populated).
report.add_event(
    EntityEvent(
        name="pedestrian_high_speed_per_object",
        expr=risky_predicate(),
        desc="Per-object: one row per pedestrian that triggered the window",
    )
)

# Combined — one output row per window, entity_key carries the union of objects.
report.add_event(
    EntityEvent(
        name="pedestrian_high_speed_combined",
        expr=risky_predicate(),
        per_entity_windowing=False,
        desc="Per-window: one row per window, entity_key unions all triggering pedestrians",
    )
)

# COMMAND ----------

# MAGIC %md ## Cross-entity correlation — two independent objects, overlapping in time
# MAGIC
# MAGIC The events above ask about *one* object (or an object combined with the ego
# MAGIC speed channel). The next step is correlating **two independent
# MAGIC entities**: *did a vulnerable road user pass close **while** a car closed on ego
# MAGIC from ahead?* No single detection row is both a VRU and a car — so each side is
# MAGIC finalized with `.entity_condition()` ("all my clauses hold for the SAME object;
# MAGIC from here treat me as a session-scope window") and the outer `&` is the
# MAGIC time-overlap of the two windows.
# MAGIC
# MAGIC This event also exercises the **string predicate** operators on the
# MAGIC per-object columns — `isin` (a class set), `startswith` (the azimuth sector
# MAGIC enum), `contains` (the pipe-delimited sensor provenance) — alongside the numeric
# MAGIC `<=` / `<` clauses.

# COMMAND ----------

# A two-wheeled vulnerable road user (VRU) within 8 m. `isin` matches a class set.
vru_close = (
    ot.detection_class.isin(["cyclist", "motorcycle"]) & (ot.distance_m <= 8.0)
).entity_condition()

# A car closing on ego from the forward arc, radar-confirmed. `startswith` on the
# azimuth sector enum (front, front_left, front_right); `contains` on the
# pipe-delimited `source` provenance (e.g. "lidar|radar|camera").
car_closing_ahead = (
    (ot.detection_class == "car")
    & ot.azimuth.startswith("front")
    & ot.source.contains("radar")
    & (ot.distance_m <= 15.0)
    & (ot.relative_velocity_ms < -0.5)  # negative = approaching
).entity_condition()

# Outer & = interval intersection: both windows overlap in the same recording,
# regardless of which VRU and which car triggered each side. With
# per_entity_windowing=False, one row per window carries a union entity_key naming
# both the VRU and the car.
report.add_event(
    EntityEvent(
        name="vru_while_car_closing",
        expr=vru_close & car_closing_ahead,
        per_entity_windowing=False,
        desc="A two-wheeled VRU within 8 m while a car closed from ahead on radar — two distinct objects overlapping in the same recording",
    )
)
print("Cross-entity event registered: vru_while_car_closing (string ops: isin/startswith/contains)")

# COMMAND ----------

# COMMAND ----------

# MAGIC %md ## Map-aware scenario — three tables in one predicate (presence-gated)
# MAGIC
# MAGIC The multi-table scenario: compose the scalar **speed channel**, the
# MAGIC per-object **object_tracks** series, AND the per-frame **ego_map_context** series
# MAGIC — three tables — in a single predicate, resolved per recording in one step.
# MAGIC
# MAGIC > *"A pedestrian within 15 m while the ego vehicle is on a pedestrian crossing
# MAGIC > and still moving faster than 20 kph."*
# MAGIC
# MAGIC `ego_map_context` has no entity (one ego per frame), so `on_ped_crossing == 1`
# MAGIC is a **presence** leaf — placed first so the `&` auto-finalizes it — while the
# MAGIC pedestrian clause stays per-object via `.entity_condition()`.
# MAGIC
# MAGIC Runs only if the map_context tables exist (built by `01_ingest` Step 5b). On an
# MAGIC adapter/run without map data, the block is skipped — the notebook stays generic.

# COMMAND ----------

if not spark.catalog.tableExists(cfg.t_ego_map_context):
    print(f"⏭️  Map context not present ({cfg.t_ego_map_context}) — skipping map showcase events.")
else:
    from lib.series_defs import register_ego_map_context, register_object_map_context

    register_ego_map_context(db, cfg.t_ego_map_context)
    register_object_map_context(db, cfg.t_object_map_context)
    print("Series registered: ego_map_context, object_map_context")

    ego_mapc = db.query.series("ego_map_context")

    def map_scenario():
        """The three-table scenario predicate, built fresh per event."""
        ped_close = (
            (ot.detection_class == "pedestrian") & (ot.distance_m <= 15.0)
        ).entity_condition()
        ego_on_crossing = ego_mapc.on_ped_crossing == 1  # presence leaf (no entity_key)
        # ego_on_crossing first → _PartialPredicate.__and__ finalizes it as presence.
        return ego_on_crossing & ped_close & (vehicle_speed > 20.0)

    report.add_event(
        BasicEvent(
            name="pedestrian_on_crossing_at_speed",
            expr=map_scenario(),
            desc="Pedestrian within 15 m while ego is on a ped_crossing and speed > 20 kph — channel + object_tracks + map_context in one predicate",
        )
    )
    report.add_event(
        EntityEvent(
            name="pedestrian_on_crossing_at_speed_per_object",
            expr=map_scenario(),
            desc="Per-pedestrian attribution for the three-table map scenario",
        )
    )
    print("Map showcase events registered: BasicEvent + EntityEvent (three-table)")

# COMMAND ----------

report.add_page(Page(page_number=1))
print("Events registered: pedestrian-proximity + cross-entity + (map showcase if present)")

# COMMAND ----------

# MAGIC %md ## Run the query and persist `event_instance_fact`

# COMMAND ----------

report.determine_report()
report.persist_results()
print("✓ event_instance_fact written to gold")

# COMMAND ----------

# MAGIC %md ## Inspect detected events + the entity_key map
# MAGIC
# MAGIC `entity_key` is a JSON map identifying the triggering object(s), e.g.
# MAGIC `{"object_tracks": {"fusion": ["47"]}}`. `BasicEvent` rows leave it `NULL`;
# MAGIC the `EntityEvent`s populate it. The cross-entity `vru_while_car_closing` row
# MAGIC carries a **union** map naming both the VRU and the car that co-occurred, e.g.
# MAGIC `{"object_tracks": {"fusion": ["12", "47"]}}`.

# COMMAND ----------

event_fact_table = cfg.t_event_instance_fact
event_dimension_table = cfg.t_event_dimension

# COMMAND ----------

# MAGIC %sql
# MAGIC -- event_instance_fact columns: container_id, event_instance_id, event_id,
# MAGIC -- start_ts, end_ts, entity_key. Event name lives on event_dimension.
# MAGIC SELECT d.event_name,
# MAGIC        f.container_id, f.start_ts, f.end_ts,
# MAGIC        ROUND((f.end_ts - f.start_ts) / 1000.0, 2) AS duration_s,
# MAGIC        f.entity_key
# MAGIC FROM   ${catalog}.${schema_prefix}_gold.${adapter}_demo_event_instance_fact f
# MAGIC JOIN   ${catalog}.${schema_prefix}_gold.${adapter}_demo_event_dimension     d  USING (event_id)
# MAGIC ORDER  BY d.event_name, f.container_id, f.start_ts

# COMMAND ----------

# MAGIC %md ## Validation
# MAGIC
# MAGIC Guards the run: `event_instance_fact` must be non-empty and the per-object
# MAGIC `EntityEvent` must populate `entity_key`. The cross-entity and map-scenario
# MAGIC events are reported but not required — they are genuinely sparse on
# MAGIC `v1.0-mini` and fire on `v1.0-trainval`.

# COMMAND ----------

fact = spark.read.table(event_fact_table)
dim = spark.read.table(event_dimension_table)

n_events = fact.count()
assert n_events > 0, (
    f"event_instance_fact is empty — event search produced no matches. Common causes: "
    f"channel_metrics not populated upstream, Vehicle_Speed_kph channel name missing, "
    f"or predicate doesn't match any data ranges. Table: {event_fact_table}"
)

# entity_key must be present on the per-object EntityEvent rows.
per_obj_with_key = (
    fact.join(dim, "event_id")
    .filter(F.col("event_name") == "pedestrian_high_speed_per_object")
    .filter(F.col("entity_key").isNotNull())
    .count()
)
assert per_obj_with_key > 0, (
    "EntityEvent 'pedestrian_high_speed_per_object' produced no rows with a populated "
    "entity_key — cross-series entity attribution did not resolve."
)

n_windows = (
    fact.join(dim, "event_id")
    .filter(F.col("event_name") == "pedestrian_high_speed_proximity")
    .count()
)

# Cross-entity correlation — soft on small variants. The strict VRU-while-car-closing
# co-occurrence is genuinely sparse on v1.0-mini (a close two-wheeler overlapping a
# car closing from ahead on radar may not occur in 10 scenes); it fires natively on
# v1.0-trainval. So we report it but don't fail the run when mini has none.
n_xentity = (
    fact.join(dim, "event_id")
    .filter(F.col("event_name") == "vru_while_car_closing")
    .filter(F.col("entity_key").isNotNull())
    .count()
)

# Map showcase (three-table) — present only when map_context was ingested. Soft for
# the same small-variant reason; the hard correctness guard for the map join lives in
# 01_ingest Step 5b (ego frames must resolve onto the drivable area).
map_event_names = ["pedestrian_on_crossing_at_speed", "pedestrian_on_crossing_at_speed_per_object"]
n_map = (
    fact.join(dim, "event_id")
    .filter(F.col("event_name").isin(map_event_names))
    .count()
)

print(f"✓ Validated — {n_events} fact rows; {per_obj_with_key} per-object rows carry entity_key; {n_windows} presence windows")
print(f"   cross-entity windows: {n_xentity}   |   map-scenario fact rows: {n_map}")
if n_windows < 10:
    print(f"⚠️  Fewer than 10 windows ({n_windows}) — dataset has limited pedestrian-proximity-at-speed examples; consider a larger variant.")
if n_xentity == 0:
    print("ℹ️  vru_while_car_closing produced 0 windows — expected on v1.0-mini; fires on v1.0-trainval.")

# COMMAND ----------

# MAGIC %md ## What's next
# MAGIC
# MAGIC - `03_per_event_detail.py` — per-window LiDAR + camera detections + OpenLABEL JSON packages
# MAGIC - `04_visualize.py` — event walkthrough + sensor KPI comparison
# MAGIC
# MAGIC Downstream notebooks read event windows from `event_instance_fact` directly.
