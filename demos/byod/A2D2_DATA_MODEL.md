# Impulse ADAS demo — A2D2 data model

**Adapter:** `a2d2` (Audi Autonomous Driving Dataset) · **Variant:** `camera_lidar_semantic_bboxes`

> **Not currently deployed** — there is no live A2D2 workspace, so this carries **no sample data**. Column schemas are taken from the demo's schema definitions (adapter-independent — identical to the deployed nuScenes tables) and overlaid with which tables the A2D2 adapter populates vs. leaves empty.

Schema names follow the deploy-time `<schema_prefix>` convention: `<prefix>_silver`, `<prefix>_perception_silver`, `<prefix>_gold`. Gold event tables are prefixed `a2d2_demo_` (the pipeline Report sink, `{adapter}_demo`).

## A2D2 channel catalog (`<prefix>_silver.channels`)

A2D2's distinguishing input: **real recorded ECU bus signals** (nuScenes has none — it synthesizes kinematics from ego pose). These are decoded straight from `bus_signals.json`, no synthesis.

**Bus signals**

| Channel | channel_id | A2D2 bus key | Unit |
|---|---:|---|---|
| `Vehicle_Speed_kph` | 1001 | `vehicle_speed` | kph |
| `Vehicle_Accel_Longitudinal_ms2` | 1002 | `acceleration_x` | m/s^2 |
| `Vehicle_Accel_Lateral_ms2` | 1003 | `acceleration_y` | m/s^2 |
| `Vehicle_Accel_Vertical_ms2` | 1004 | `acceleration_z` | m/s^2 |
| `Yaw_Rate_rads` | 1005 | `angular_velocity_z` | rad/s |
| `Roll_Rate_rads` | 1006 | `angular_velocity_x` | rad/s |
| `Pitch_Rate_rads` | 1007 | `angular_velocity_y` | rad/s |
| `Steering_Angle_deg` | 1008 | `steering_angle_calculated` | deg |
| `Brake_Pressure_pct` | 1009 | `brake_pressure` | % |
| `Accelerator_Pedal_pct` | 1010 | `accelerator_pedal` | % |
| `Pitch_Angle_deg` | 1011 | `pitch_angle` | deg |
| `Roll_Angle_deg` | 1012 | `roll_angle` | deg |
| `Latitude_deg` | 1013 | `latitude_degree` | deg |
| `Longitude_deg` | 1014 | `longitude_degree` | deg |

**Detection aggregates** (per-keyframe, derived from annotations — mirrors the nuScenes adapter for comparability)

| Channel | channel_id | Class |
|---|---:|---|
| `Pedestrian_Count` | 2001 | pedestrian |
| `Pedestrian_Nearest_Distance_m` | 2002 | pedestrian |
| `Vehicle_Count_Front` | 2003 | car (forward only) |
| `Vehicle_Nearest_Distance_m` | 2004 | car |
| `Cyclist_Count` | 2005 | cyclist |
| `Cyclist_Nearest_Distance_m` | 2006 | cyclist |

## Tables at a glance

| Schema | Table | Status | Notes |
|---|---|---|---|
| `<prefix>_silver` | `channels` | Populated | Real A2D2 bus signals + per-keyframe detection aggregates (see channel catalog). |
| `<prefix>_silver` | `channel_tags` | Populated | Per-channel metadata (sensor name, unit). |
| `<prefix>_silver` | `channel_metrics` | Populated | Per-channel summary stats. |
| `<prefix>_silver` | `container_tags` | Populated | Per-scene (drive) key/value metadata. |
| `<prefix>_silver` | `container_metrics` | Populated | Per-scene start/stop/duration/channel-count. |
| `<prefix>_silver` | `perception_channels` | Populated | Index of camera PNGs + LiDAR NPZs (one row per file). |
| `<prefix>_perception_silver` | `object_tracks` | Populated | 3D bbox JSON, vehicle frame. `relative_velocity_ms` always NULL; `source` constant `ground_truth_camera_lidar`. |
| `<prefix>_perception_silver` | `lidar_object_detections` | Populated | 3D cuboids, vehicle frame, `sensor_id=LIDAR_FUSED`. |
| `<prefix>_perception_silver` | `camera_object_detections` | Populated | 3D bbox projected to 6 cameras via `cams_lidars.json`. |
| `<prefix>_perception_silver` | `ego_map_context` | EMPTY | No HD map in A2D2 (this is derived from the nuScenes map-expansion layers). |
| `<prefix>_perception_silver` | `object_map_context` | EMPTY | No HD map in A2D2. |
| `<prefix>_perception_silver` | `radar_object_detections` | EMPTY | A2D2 has no radar sensor. |
| `<prefix>_perception_silver` | `lane_markings` | EMPTY | No lane-line annotations in the bbox subset. |
| `<prefix>_perception_silver` | `free_space` | EMPTY | Drivable masks live only in the segmentation subset, not the bbox subset this adapter loads. |
| `<prefix>_perception_silver` | `predicted_trajectories` | EMPTY | A2D2 has no planner output / future-state labels. |
| `<prefix>_gold` | `a2d2_demo_event_instance_fact` | Populated | Detected event instances (Impulse Report sink; prefix `{adapter}_demo`). |
| `<prefix>_gold` | `a2d2_demo_event_dimension` | Populated | Event catalog for the pipeline Report. |
| `<prefix>_gold` | `a2d2_demo_measurement_dimension` | Populated | Measurement config dimension. |

Gold also contains the Event Explorer sink (`a2d2_explore_*`: event fact/dimension, 1D/2D histograms, stats aggregator) when notebook 05 runs — same structure as the nuScenes `*_explore_*` tables.

## Silver — ingested signals & metadata  (`<prefix>_silver`)

### `channels`

Real A2D2 bus signals + per-keyframe detection aggregates (see channel catalog).

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `channel_id` | `int` |
| `tstart` | `bigint` |
| `tend` | `bigint` |
| `value` | `double` |

### `channel_tags`

Per-channel metadata (sensor name, unit).

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `channel_id` | `int` |
| `key` | `string` |
| `value` | `string` |

### `channel_metrics`

Per-channel summary stats.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `channel_id` | `int` |
| `value_type` | `string` |
| `sample_count` | `int` |
| `nan_ratio` | `float` |
| `begin_s` | `float` |
| `end_s` | `float` |
| `duration_ms` | `int` |
| `original_sample_count` | `int` |
| `original_sr` | `float` |
| `min` | `float` |
| `max` | `float` |
| `mean` | `float` |
| `std` | `float` |
| `pz1` | `float` |
| `pz10` | `float` |
| `pz90` | `float` |
| `pz99` | `float` |

### `container_tags`

Per-scene (drive) key/value metadata.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `key` | `string` |
| `value` | `string` |

### `container_metrics`

Per-scene start/stop/duration/channel-count.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `start_dt` | `timestamp` |
| `stop_dt` | `timestamp` |
| `duration_ms` | `int` |
| `num_channels` | `int` |

### `perception_channels`

Index of camera PNGs + LiDAR NPZs (one row per file).

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `channel_id` | `int` |
| `timestamp` | `bigint` |
| `file_path` | `string` |
| `format` | `string` |

## Perception silver — derived per-frame / per-object products  (`<prefix>_perception_silver`)

### `object_tracks`

3D bbox JSON, vehicle frame. `relative_velocity_ms` always NULL; `source` constant `ground_truth_camera_lidar`.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `frame_ts` | `bigint` |
| `object_id` | `bigint` |
| `detection_class` | `string` |
| `distance_m` | `double` |
| `lane_offset` | `int` |
| `relative_velocity_ms` | `double` |
| `azimuth` | `string` |
| `confidence` | `double` |
| `source` | `string` |

### `lidar_object_detections`

3D cuboids, vehicle frame, `sensor_id=LIDAR_FUSED`.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `frame_ts` | `bigint` |
| `object_id` | `bigint` |
| `detection_class` | `string` |
| `confidence` | `double` |
| `sensor_id` | `string` |
| `cx` | `double` |
| `cy` | `double` |
| `cz` | `double` |
| `length` | `double` |
| `width` | `double` |
| `height` | `double` |
| `yaw_rad` | `double` |

### `camera_object_detections`

3D bbox projected to 6 cameras via `cams_lidars.json`.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `frame_ts` | `bigint` |
| `object_id` | `bigint` |
| `detection_class` | `string` |
| `confidence` | `double` |
| `sensor_id` | `string` |
| `x1` | `int` |
| `y1` | `int` |
| `x2` | `int` |
| `y2` | `int` |

### `ego_map_context`  ·  **EMPTY for A2D2**

No HD map in A2D2 (this is derived from the nuScenes map-expansion layers).

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `frame_ts` | `bigint` |
| `location` | `string` |
| `on_ped_crossing` | `int` |
| `on_walkway` | `int` |
| `on_drivable_area` | `int` |
| `in_intersection` | `int` |
| `dist_to_ped_crossing_m` | `double` |
| `dist_to_stop_line_m` | `double` |
| `lane_id` | `string` |

### `object_map_context`  ·  **EMPTY for A2D2**

No HD map in A2D2.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `frame_ts` | `bigint` |
| `object_id` | `bigint` |
| `detection_class` | `string` |
| `on_ped_crossing` | `int` |
| `on_walkway` | `int` |
| `in_intersection` | `int` |
| `same_lane_as_ego` | `int` |
| `dist_to_ped_crossing_m` | `double` |
| `lane_id` | `string` |

### `radar_object_detections`  ·  **EMPTY for A2D2**

A2D2 has no radar sensor.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `frame_ts` | `bigint` |
| `object_id` | `bigint` |
| `detection_class` | `string` |
| `confidence` | `double` |
| `sensor_id` | `string` |
| `range_m` | `double` |
| `azimuth_rad` | `double` |
| `radial_velocity_ms` | `double` |

### `lane_markings`  ·  **EMPTY for A2D2**

No lane-line annotations in the bbox subset.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `frame_ts` | `bigint` |
| `sensor_id` | `string` |
| `boundary` | `string` |
| `marking_type` | `string` |
| `c0` | `double` |
| `c1` | `double` |
| `c2` | `double` |
| `c3` | `double` |

### `free_space`  ·  **EMPTY for A2D2**

Drivable masks live only in the segmentation subset, not the bbox subset this adapter loads.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `frame_ts` | `bigint` |
| `sensor_id` | `string` |
| `boundary_pts` | `array<struct<x:double,y:double>>` |

### `predicted_trajectories`  ·  **EMPTY for A2D2**

A2D2 has no planner output / future-state labels.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `frame_ts` | `bigint` |
| `object_id` | `bigint` |
| `detection_class` | `string` |
| `horizon_ms` | `int` |
| `waypoints` | `array<struct<x:double,y:double,z:double,t:bigint>>` |

## Gold — Impulse event Report sink  (`<prefix>_gold`)

### `a2d2_demo_event_instance_fact`

Detected event instances (Impulse Report sink; prefix `{adapter}_demo`).

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `event_instance_id` | `bigint` |
| `event_id` | `int` |
| `start_ts` | `double` |
| `end_ts` | `double` |
| `entity_key` | `string` |
| `_created_at` | `timestamp` |

### `a2d2_demo_event_dimension`

Event catalog for the pipeline Report.

**Schema**

| Column | Type |
|---|---|
| `event_id` | `int` |
| `report_id` | `int` |
| `event_type` | `string` |
| `event_name` | `string` |
| `event_description` | `string` |
| `required_channels` | `array<string>` |
| `event_expression` | `string` |
| `definition_hash` | `bigint` |
| `attributes` | `map<string,string>` |
| `_created_at` | `timestamp` |

### `a2d2_demo_measurement_dimension`

Measurement config dimension.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `config_hash` | `int` |
| `_created_at` | `timestamp` |

## A2D2 field caveats

- **`object_tracks.relative_velocity_ms` is always NULL** — A2D2 labels each frame independently (no instance token), so cross-frame velocity can't be computed. Scenario search depending on this field silently excludes A2D2.
- **`object_tracks.source` is constant `ground_truth_camera_lidar`** — labels are camera+LiDAR fusion; no per-annotation sensor-coverage metadata is published.
- **Ego pose is identity** — A2D2 publishes box translations already in the vehicle frame, so the global↔ego transform is a no-op.
- **Sample timestamps are synthesized** — frame filenames carry a monotonic index, not microseconds; the loader assigns `scene_start + frame_index × 100,000 µs` (10 Hz). Δt is frame-accurate; absolute clock matches only at scene start.
