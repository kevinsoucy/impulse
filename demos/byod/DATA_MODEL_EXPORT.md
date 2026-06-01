# Impulse ADAS demo — data model export

**Workspace:** `fevm-adas-engine-validate` · **Catalog:** `adas_engine_validate_catalog` · **Source:** NuScenes v1.0-mini (`nuscenes` adapter)

Exported live from the deployed workspace. The demo deploys two parallel variants — `demo_mini_*` (NuScenes v1.0-mini, the complete verified run, documented here) and `demo_trainval_*` (v1.0-trainval, a partial run). Schema names follow `<schema_prefix>_silver` / `_perception_silver` / `_gold`; gold event tables are prefixed `nuscenes_demo_` (pipeline sink) and `nuscenes_explore_` (notebook 05 Event Explorer sink).

## Tables at a glance

| Schema | Table | Rows | Purpose |
|---|---|---:|---|
| `demo_mini_silver` | `channel_metrics` | 85 | Per-channel summary statistics (min/max/mean/std/percentiles, sample counts, time bounds) computed during ingest. |
| `demo_mini_silver` | `channel_tags` | 840 | Key/value metadata attached to each (container, channel) |
| `demo_mini_silver` | `channels` | 3300 | Time-series signal samples: one row per (container, channel, timestamp) with a scalar value. The core ingested signal table. |
| `demo_mini_silver` | `container_metrics` | 10 | Per-container (per-scene) summary: start/stop time, duration, channel count. |
| `demo_mini_silver` | `container_tags` | 70 | Key/value metadata at the container (scene/drive) level. |
| `demo_mini_silver` | `perception_channels` | 2828 | Index of perception artifacts (camera frames, LiDAR scans) |
| `demo_mini_perception_silver` | `camera_object_detections` | 2478 | Per-frame 2D bounding-box detections from camera sensors (class, confidence, pixel box x1/y1/x2/y2). |
| `demo_mini_perception_silver` | `ego_map_context` | 404 | Per-frame ego-vehicle map context (location, on crosswalk/walkway/drivable area, intersection, distances, lane). |
| `demo_mini_perception_silver` | `lidar_object_detections` | 2074 | Per-frame 3D bounding-box detections from LiDAR (class, confidence, center cx/cy/cz, L/W/H, yaw). |
| `demo_mini_perception_silver` | `object_map_context` | 18538 | Per-object, per-frame map context (on crosswalk/walkway, intersection, same-lane-as-ego, distances, lane). |
| `demo_mini_perception_silver` | `object_tracks` | 18538 | Fused per-object, per-frame tracks: class, range, lane offset, relative velocity, azimuth, confidence, source. |
| `demo_mini_gold` | `nuscenes_demo_event_dimension` | 6 | Event catalog (dimension): one row per event type with its expression, required channels, definition hash, attributes. |
| `demo_mini_gold` | `nuscenes_demo_event_instance_fact` | 74 | Detected event instances (fact): one row per occurrence with container, event_id, start/end timestamp, entity key. |
| `demo_mini_gold` | `nuscenes_demo_measurement_dimension` | 10 | Measurement config dimension keyed by container + config hash. |
| `demo_mini_gold` | `nuscenes_explore_event_dimension` | 7 | Event Explorer (notebook 05) event catalog |
| `demo_mini_gold` | `nuscenes_explore_event_instance_fact` | 119 | Event Explorer detected event instances. |
| `demo_mini_gold` | `nuscenes_explore_histogram2d_dimension` | 1 | Event Explorer 2D-histogram visual definitions (axes, bins, expressions). |
| `demo_mini_gold` | `nuscenes_explore_histogram2d_fact` | 480 | Event Explorer 2D-histogram bin values. |
| `demo_mini_gold` | `nuscenes_explore_histogram_dimension` | 1 | Event Explorer 1D-histogram visual definitions. |
| `demo_mini_gold` | `nuscenes_explore_histogram_fact` | 80 | Event Explorer 1D-histogram bin values. |
| `demo_mini_gold` | `nuscenes_explore_measurement_dimension` | 10 | Event Explorer measurement config dimension. |
| `demo_mini_gold` | `nuscenes_explore_stats_aggregator_dimension` | 2 | Event Explorer stats-aggregator visual definitions. |
| `demo_mini_gold` | `nuscenes_explore_stats_aggregator_fact` | 124 | Event Explorer per-event aggregated statistic values. |

## Silver — ingested signals & metadata (`demo_mini_silver`)

### `channel_metrics`  ·  85 rows

Per-channel summary statistics (min/max/mean/std/percentiles, sample counts, time bounds) computed during ingest.

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

**Sample rows**

| container_id | channel_id | value_type | sample_count | nan_ratio | begin_s | end_s | duration_ms | original_sample_count | original_sr | min | max | mean | std | pz1 | pz10 | pz90 | pz99 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 8035291023832990565 | 1001 | double | 41 | 0.0 | 1.5354893E9 | 1.5354893E9 | 20399 | 41 | *null* | 4.706724E-7 | 1.3498514E-4 | 5.3074E-5 | 3.3214234E-5 | 4.706724E-7 | 1.1354728E-5 | 9.814226E-5 | 1.3498514E-4 |
| 8035291023832990565 | 1003 | double | 41 | 0.0 | 1.5354893E9 | 1.5354893E9 | 20399 | 41 | *null* | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| 5515796792657594645 | 1002 | double | 39 | 0.0 | 1.532403E9 | 1.532403E9 | 19648 | 39 | *null* | -0.97281325 | 0.37408254 | -0.3481737 | 0.3072661 | -0.97281325 | -0.74371964 | 0.04433392 | 0.37408254 |

### `channel_tags`  ·  840 rows

Key/value metadata attached to each (container, channel) — sensor names, units, modality hints.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `channel_id` | `int` |
| `key` | `string` |
| `value` | `string` |

**Sample rows**

| container_id | channel_id | key | value |
|---|---|---|---|
| 5515796792657594645 | 101 | channel_name | CAM_FRONT |
| 5515796792657594645 | 101 | sensor_type | camera |
| 5515796792657594645 | 101 | modality | binary_file |

### `channels`  ·  3300 rows

Time-series signal samples: one row per (container, channel, timestamp) with a scalar value. The core ingested signal table.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `channel_id` | `int` |
| `tstart` | `bigint` |
| `tend` | `bigint` |
| `value` | `double` |

**Sample rows**

| container_id | channel_id | tstart | tend | value |
|---|---|---|---|---|
| 5933237715990037811 | 2004 | 1542800368447460 | 1542800368948430 | 11.123531868852163 |
| 5933237715990037811 | 2005 | 1542800368447460 | 1542800368948430 | 0.0 |
| 5933237715990037811 | 2001 | 1542800368948430 | 1542800369448339 | 0.0 |

### `container_metrics`  ·  10 rows

Per-container (per-scene) summary: start/stop time, duration, channel count.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `start_dt` | `timestamp` |
| `stop_dt` | `timestamp` |
| `duration_ms` | `int` |
| `num_channels` | `int` |

**Sample rows**

| container_id | start_dt | stop_dt | duration_ms | num_channels |
|---|---|---|---|---|
| 5515796792657594645 | 2018-07-24T03:28:47.647Z | 2018-07-24T03:29:06.797Z | 19149 | 21 |
| 8988286406539497243 | 2018-08-01T19:26:43.547Z | 2018-08-01T19:27:02.948Z | 19400 | 21 |
| 8035291023832990565 | 2018-08-28T20:48:16.047Z | 2018-08-28T20:48:35.948Z | 19900 | 21 |

### `container_tags`  ·  70 rows

Key/value metadata at the container (scene/drive) level.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `key` | `string` |
| `value` | `string` |

**Sample rows**

| container_id | key | value |
|---|---|---|
| 5515796792657594645 | scene_name | scene-0061 |
| 5515796792657594645 | scene_description | Parked truck, construction, intersection, turn left, following a van |
| 5515796792657594645 | scene_token | cc8c0bf57f984915a77078b10eb33198 |

### `perception_channels`  ·  2828 rows

Index of perception artifacts (camera frames, LiDAR scans) — file path + format per (container, channel, timestamp).

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `channel_id` | `int` |
| `timestamp` | `bigint` |
| `file_path` | `string` |
| `format` | `string` |

**Sample rows**

| container_id | channel_id | timestamp | file_path | format |
|---|---|---|---|---|
| 3629038630498356309 | 103 | 1538984246920339 | /Volumes/adas_engine_validate_catalog/demo_mini_silver/raw/v1.0-mini/samples/... | jpg |
| 3629038630498356309 | 106 | 1538984246927893 | /Volumes/adas_engine_validate_catalog/demo_mini_silver/raw/v1.0-mini/samples/... | jpg |
| 3629038630498356309 | 104 | 1538984246937525 | /Volumes/adas_engine_validate_catalog/demo_mini_silver/raw/v1.0-mini/samples/... | jpg |

## Perception silver — derived per-frame/per-object products (`demo_mini_perception_silver`)

### `camera_object_detections`  ·  2478 rows

Per-frame 2D bounding-box detections from camera sensors (class, confidence, pixel box x1/y1/x2/y2).

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

**Sample rows**

| container_id | frame_ts | object_id | detection_class | confidence | sensor_id | x1 | y1 | x2 | y2 |
|---|---|---|---|---|---|---|---|---|---|
| 6808739644759034942 | 1542800861447555 | 3189925987310847455 | bus | 1.0 | cam_back | 775 | 441 | 823 | 492 |
| 6808739644759034942 | 1542800861447555 | 4576195470809646097 | car | 1.0 | cam_front | 815 | 453 | 916 | 556 |
| 6808739644759034942 | 1542800861447555 | 185588036796170715 | pedestrian | 1.0 | cam_back | 172 | 499 | 236 | 571 |

### `ego_map_context`  ·  404 rows

Per-frame ego-vehicle map context (location, on crosswalk/walkway/drivable area, intersection, distances, lane).

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

**Sample rows**

| container_id | frame_ts | location | on_ped_crossing | on_walkway | on_drivable_area | in_intersection | dist_to_ped_crossing_m | dist_to_stop_line_m | lane_id |
|---|---|---|---|---|---|---|---|---|---|
| 5515796792657594645 | 1532402927647951 | singapore-onenorth | 0 | 0 | 1 | 1 | 57.422364157906735 | 49.04724835347871 | *null* |
| 5515796792657594645 | 1532402928147847 | singapore-onenorth | 0 | 0 | 1 | 1 | 52.97477055623929 | 44.57168452203449 | *null* |
| 5515796792657594645 | 1532402928698048 | singapore-onenorth | 0 | 0 | 1 | 0 | 48.21395543890509 | 39.777859890922954 | 7b690728-97e7-4fe9-9116-a4a41ab39e41 |

### `lidar_object_detections`  ·  2074 rows

Per-frame 3D bounding-box detections from LiDAR (class, confidence, center cx/cy/cz, L/W/H, yaw).

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

**Sample rows**

| container_id | frame_ts | object_id | detection_class | confidence | sensor_id | cx | cy | cz | length | width | height | yaw_rad |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4521437396061341233 | 1535385097901019 | 5277483219836422329 | car | 1.0 | LIDAR_TOP | -23.952680312486056 | -7.761008354701009 | 1.0460809593506437 | 4.893 | 2.065 | 1.938 | -1.5923832555549489 |
| 4521437396061341233 | 1535385097901019 | 5561092214351874304 | car | 1.0 | LIDAR_TOP | -8.449435844463782 | 8.257822152085604 | 0.7362985202460397 | 4.959 | 1.943 | 1.499 | 3.1377382696897342 |
| 4521437396061341233 | 1535385097901019 | 6866937574538694407 | car | 1.0 | LIDAR_TOP | -2.834273359105573 | 8.092672408978094 | 0.9314031726232126 | 4.791 | 1.872 | 1.698 | 3.120284977169791 |

### `object_map_context`  ·  18538 rows

Per-object, per-frame map context (on crosswalk/walkway, intersection, same-lane-as-ego, distances, lane).

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

**Sample rows**

| container_id | frame_ts | object_id | detection_class | on_ped_crossing | on_walkway | in_intersection | same_lane_as_ego | dist_to_ped_crossing_m | lane_id |
|---|---|---|---|---|---|---|---|---|---|
| 6808739644759034942 | 1542800851450143 | 1691007756295292709 | pedestrian | 0 | 1 | 0 | 0 | 1.5666633999262454 | *null* |
| 6808739644759034942 | 1542800851450143 | 5662150181538252793 | pedestrian | 0 | 0 | 0 | 0 | 8.899447940465905 | *null* |
| 6808739644759034942 | 1542800851450143 | 3189925987310847455 | bus | 0 | 0 | 1 | 0 | 23.030506751149105 | *null* |

### `object_tracks`  ·  18538 rows

Fused per-object, per-frame tracks: class, range, lane offset, relative velocity, azimuth, confidence, source.

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

**Sample rows**

| container_id | frame_ts | object_id | detection_class | distance_m | lane_offset | relative_velocity_ms | azimuth | confidence | source |
|---|---|---|---|---|---|---|---|---|---|
| 3629038630498356309 | 1538984243946219 | 9028842191423490771 | car | 20.716033961618777 | -2 | -0.00382044162758995 | front | 1.0 | lidar\|radar\|camera |
| 3629038630498356309 | 1538984243946219 | 7383333818545684600 | car | 57.22901045846195 | -2 | 0.04271843293325898 | front | 1.0 | camera |
| 3629038630498356309 | 1538984243946219 | 5942220647650118353 | car | 13.254514540854768 | -2 | -0.0016788634523321338 | right | 1.0 | lidar |

## Gold — Impulse event/visual report sinks (`demo_mini_gold`)

### `nuscenes_demo_event_dimension`  ·  6 rows

Event catalog (dimension): one row per event type with its expression, required channels, definition hash, attributes.

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

**Sample rows**

| event_id | report_id | event_type | event_name | event_description | required_channels | event_expression | definition_hash | attributes | _created_at |
|---|---|---|---|---|---|---|---|---|---|
| 1452681454 | 1191369856 | ENTITY_EVENT | pedestrian_high_speed_per_object | Per-object: one row per pedestrian that triggered the window | *null* | TimeSeriesOp<and_(SeriesSelector<object_tracks: (detection_class eq 'pedestri... | -2225513469215004027 | {} | 2026-06-01T12:12:29.783Z |
| 276168696 | 1191369856 | ENTITY_EVENT | pedestrian_high_speed_combined | Per-window: one row per window, entity_key unions all triggering pedestrians | *null* | TimeSeriesOp<and_(SeriesSelector<object_tracks: (detection_class eq 'pedestri... | -6669090994882521186 | {} | 2026-06-01T12:12:29.783Z |
| 312529516 | 1191369856 | ENTITY_EVENT | vru_while_car_closing | A two-wheeled VRU within 8 m while a car closed from ahead on radar — two dis... | *null* | TimeSeriesOp<and_(SeriesSelector<object_tracks: (detection_class isin ['cycli... | -5833973350441440027 | {} | 2026-06-01T12:12:29.783Z |

### `nuscenes_demo_event_instance_fact`  ·  74 rows

Detected event instances (fact): one row per occurrence with container, event_id, start/end timestamp, entity key.

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

**Sample rows**

| container_id | event_instance_id | event_id | start_ts | end_ts | entity_key | _created_at |
|---|---|---|---|---|---|---|
| 4521437396061341233 | 2768859968 | 1452681454 | 1.535385108951051E15 | 1.535385110449087E15 | {"object_tracks": {"fusion": ["2543507779130445364"]}} | 2026-06-01T12:12:16.393Z |
| 4521437396061341233 | 1342795791 | 1452681454 | 1.535385097901019E15 | 1.535385098400887E15 | {"object_tracks": {"fusion": ["4845355294897292846"]}} | 2026-06-01T12:12:16.393Z |
| 4521437396061341233 | 2900792067 | 1452681454 | 1.535385100398781E15 | 1.535385100898103E15 | {"object_tracks": {"fusion": ["4845355294897292846"]}} | 2026-06-01T12:12:16.393Z |

### `nuscenes_demo_measurement_dimension`  ·  10 rows

Measurement config dimension keyed by container + config hash.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `config_hash` | `int` |
| `_created_at` | `timestamp` |

**Sample rows**

| container_id | config_hash | _created_at |
|---|---|---|
| 5515796792657594645 | -828821608 | 2026-06-01T12:12:33.183Z |
| 8988286406539497243 | -828821608 | 2026-06-01T12:12:33.183Z |
| 8035291023832990565 | -828821608 | 2026-06-01T12:12:33.183Z |

### `nuscenes_explore_event_dimension`  ·  7 rows

Event Explorer (notebook 05) event catalog — separate _explore sink.

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

**Sample rows**

| event_id | report_id | event_type | event_name | event_description | required_channels | event_expression | definition_hash | attributes | _created_at |
|---|---|---|---|---|---|---|---|---|---|
| 643796907 | 462496676 | ENTITY_EVENT | ped_close_per_object | Series: each pedestrian within 15 m (entity_key = object_id) | *null* | SeriesSelector<object_tracks: (detection_class eq 'pedestrian') AND (distance... | -4421573687668005466 | {} | 2026-06-01T12:40:34.069Z |
| 312529516 | 462496676 | ENTITY_EVENT | vru_while_car_closing | Series cross-entity: two-wheeled VRU within 8 m while a car closes from ahead | *null* | TimeSeriesOp<and_(SeriesSelector<object_tracks: (detection_class isin ['cycli... | -5833973350441440027 | {} | 2026-06-01T12:40:34.069Z |
| 1174939160 | 462496676 | BASIC_EVENT | ped_approach_30m | Any pedestrian within 30 m of ego | ["Pedestrian_Nearest_Distance_m"] | TimeSeriesOp<le(TimeSeriesSelector<TagOp<eq(TagSelector<channel_name>,Pedestr... | 5647899485662100392 | {} | 2026-06-01T12:40:34.069Z |

### `nuscenes_explore_event_instance_fact`  ·  119 rows

Event Explorer detected event instances.

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

**Sample rows**

| container_id | event_instance_id | event_id | start_ts | end_ts | entity_key | _created_at |
|---|---|---|---|---|---|---|
| 3441723421252272626 | 833419727 | 1174939160 | 1.535657116649423E15 | 1.535657128649774E15 | *null* | 2026-06-01T12:40:23.111Z |
| 3629038630498356309 | 544903802 | 1174939160 | 1.538984233547259E15 | 1.538984245947391E15 | *null* | 2026-06-01T12:40:23.111Z |
| 4521437396061341233 | 3490166142 | 1174939160 | 1.535385095449675E15 | 1.535385112449187E15 | *null* | 2026-06-01T12:40:23.111Z |

### `nuscenes_explore_histogram2d_dimension`  ·  1 rows

Event Explorer 2D-histogram visual definitions (axes, bins, expressions).

**Schema**

| Column | Type |
|---|---|
| `visual_id` | `int` |
| `report_id` | `int` |
| `page_number` | `int` |
| `name` | `string` |
| `description` | `string` |
| `agg_type` | `string` |
| `x_bins` | `array<double>` |
| `y_bins` | `array<double>` |
| `x_channel_name` | `string` |
| `x_signal_expression` | `string` |
| `y_channel_name` | `string` |
| `y_signal_expression` | `string` |
| `weights_channel_name` | `string` |
| `weights_expression` | `string` |
| `values_unit` | `string` |
| `x_bins_unit` | `string` |
| `y_bins_unit` | `string` |
| `definition_hash` | `bigint` |
| `_created_at` | `timestamp` |

**Sample rows**

| visual_id | report_id | page_number | name | description | agg_type | x_bins | y_bins | x_channel_name | x_signal_expression | y_channel_name | y_signal_expression | weights_channel_name | weights_expression | values_unit | x_bins_unit | y_bins_unit | definition_hash | _created_at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1337004958 | 462496676 | 1 | ped_distance_vs_speed | Pedestrian nearest distance vs. vehicle speed — during any proximity | histogram_duration | ["0.0","10.0","20.0","30.0","40.0","50.0","60.0","70.0","80.0"] | ["0.0","5.0","10.0","15.0","20.0","25.0","30.0"] | Vehicle_Speed_kph | TimeSeriesSelector<TagOp<eq(TagSelector<channel_name>,Vehicle_Speed_kph)>> | Pedestrian_Nearest_Distance_m | TimeSeriesSelector<TagOp<eq(TagSelector<channel_name>,Pedestrian_Nearest_Dist... | *null* | *null* | s | kph | m | 3335933525397721290 | 2026-06-01T12:40:18.098Z |

### `nuscenes_explore_histogram2d_fact`  ·  480 rows

Event Explorer 2D-histogram bin values.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `visual_id` | `int` |
| `event_id` | `int` |
| `x_bin_id` | `int` |
| `y_bin_id` | `int` |
| `hist_value` | `double` |
| `x_lower_bound` | `double` |
| `x_upper_bound` | `double` |
| `y_lower_bound` | `double` |
| `y_upper_bound` | `double` |
| `x_bin_name` | `string` |
| `y_bin_name` | `string` |
| `_created_at` | `timestamp` |

**Sample rows**

| container_id | visual_id | event_id | x_bin_id | y_bin_id | hist_value | x_lower_bound | x_upper_bound | y_lower_bound | y_upper_bound | x_bin_name | y_bin_name | _created_at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 3441723421252272626 | 1337004958 | 1174939160 | 0 | 0 | 0.0 | 0.0 | 10.0 | 0.0 | 5.0 | 0.0-10.0 | 0.0-5.0 | 2026-06-01T12:40:10.258Z |
| 3441723421252272626 | 1337004958 | 1174939160 | 0 | 1 | 0.0 | 0.0 | 10.0 | 5.0 | 10.0 | 0.0-10.0 | 5.0-10.0 | 2026-06-01T12:40:10.258Z |
| 3441723421252272626 | 1337004958 | 1174939160 | 0 | 2 | 0.0 | 0.0 | 10.0 | 10.0 | 15.0 | 0.0-10.0 | 10.0-15.0 | 2026-06-01T12:40:10.258Z |

### `nuscenes_explore_histogram_dimension`  ·  1 rows

Event Explorer 1D-histogram visual definitions.

**Schema**

| Column | Type |
|---|---|
| `visual_id` | `int` |
| `report_id` | `int` |
| `name` | `string` |
| `page_number` | `int` |
| `description` | `string` |
| `agg_type` | `string` |
| `bins` | `array<double>` |
| `channel_name` | `string` |
| `signal_expression` | `string` |
| `weights_channel_name` | `string` |
| `weights_expression` | `string` |
| `values_unit` | `string` |
| `bins_unit` | `string` |
| `definition_hash` | `bigint` |
| `_created_at` | `timestamp` |

**Sample rows**

| visual_id | report_id | name | page_number | description | agg_type | bins | channel_name | signal_expression | weights_channel_name | weights_expression | values_unit | bins_unit | definition_hash | _created_at |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 840966261 | 462496676 | speed_during_ped_hazard | 1 | Vehicle speed distribution during pedestrian hazard events | histogram_duration | ["0.0","10.0","20.0","30.0","40.0","50.0","60.0","70.0","80.0"] | Vehicle_Speed_kph | TimeSeriesSelector<TagOp<eq(TagSelector<channel_name>,Vehicle_Speed_kph)>> | *null* | *null* | s | kph | 4674195254603957857 | 2026-06-01T12:40:15.489Z |

### `nuscenes_explore_histogram_fact`  ·  80 rows

Event Explorer 1D-histogram bin values.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `visual_id` | `int` |
| `event_id` | `int` |
| `bin_id` | `int` |
| `hist_value` | `double` |
| `lower_bound` | `double` |
| `upper_bound` | `double` |
| `bin_name` | `string` |
| `_created_at` | `timestamp` |

**Sample rows**

| container_id | visual_id | event_id | bin_id | hist_value | lower_bound | upper_bound | bin_name | _created_at |
|---|---|---|---|---|---|---|---|---|
| 3441723421252272626 | 840966261 | 311318222 | 0 | 0.0 | 0.0 | 10.0 | 0.0-10.0 | 2026-06-01T12:40:12.929Z |
| 3441723421252272626 | 840966261 | 311318222 | 1 | 0.0 | 10.0 | 20.0 | 10.0-20.0 | 2026-06-01T12:40:12.929Z |
| 3441723421252272626 | 840966261 | 311318222 | 2 | 0.0 | 20.0 | 30.0 | 20.0-30.0 | 2026-06-01T12:40:12.929Z |

### `nuscenes_explore_measurement_dimension`  ·  10 rows

Event Explorer measurement config dimension.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `config_hash` | `int` |
| `_created_at` | `timestamp` |

**Sample rows**

| container_id | config_hash | _created_at |
|---|---|---|
| 5515796792657594645 | -1792932892 | 2026-06-01T12:40:36.391Z |
| 8988286406539497243 | -1792932892 | 2026-06-01T12:40:36.391Z |
| 8035291023832990565 | -1792932892 | 2026-06-01T12:40:36.391Z |

### `nuscenes_explore_stats_aggregator_dimension`  ·  2 rows

Event Explorer stats-aggregator visual definitions.

**Schema**

| Column | Type |
|---|---|
| `visual_id` | `int` |
| `report_id` | `int` |
| `name` | `string` |
| `page_number` | `int` |
| `description` | `string` |
| `agg_type` | `string` |
| `statistics` | `array<string>` |
| `channel_names` | `array<string>` |
| `signal_expressions` | `array<string>` |
| `values_unit` | `string` |
| `definition_hash` | `bigint` |
| `_created_at` | `timestamp` |

**Sample rows**

| visual_id | report_id | name | page_number | description | agg_type | statistics | channel_names | signal_expressions | values_unit | definition_hash | _created_at |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 296347236 | 462496676 | ped_hazard_stats | 1 | Speed, deceleration, and pedestrian distance statistics within hazard events | stats_aggregator | ["min","median","mean","max"] | ["Vehicle_Speed_kph","Vehicle_Accel_Longitudinal_ms2","Pedestrian_Nearest_Dis... | ["TimeSeriesSelector<TagOp<eq(TagSelector<channel_name>,Vehicle_Speed_kph)>>"... | *null* | -3710015527847539469 | 2026-06-01T12:40:20.612Z |
| 69327484 | 462496676 | cyclist_hazard_stats | 1 | Speed and cyclist distance statistics within cyclist hazard events | stats_aggregator | ["min","median","mean","max"] | ["Vehicle_Speed_kph","Cyclist_Nearest_Distance_m"] | ["TimeSeriesSelector<TagOp<eq(TagSelector<channel_name>,Vehicle_Speed_kph)>>"... | *null* | 9158076129358227068 | 2026-06-01T12:40:20.612Z |

### `nuscenes_explore_stats_aggregator_fact`  ·  124 rows

Event Explorer per-event aggregated statistic values.

**Schema**

| Column | Type |
|---|---|
| `container_id` | `bigint` |
| `visual_id` | `int` |
| `channel_name` | `string` |
| `event_id` | `int` |
| `event_instance_id` | `bigint` |
| `aggregation_label` | `string` |
| `statistic_value` | `double` |
| `_created_at` | `timestamp` |

**Sample rows**

| container_id | visual_id | channel_name | event_id | event_instance_id | aggregation_label | statistic_value | _created_at |
|---|---|---|---|---|---|---|---|
| 4521437396061341233 | 296347236 | Vehicle_Speed_kph | 311318222 | 688528628 | min | 30.66152666373413 | 2026-06-01T12:40:07.341Z |
| 4521437396061341233 | 296347236 | Vehicle_Speed_kph | 311318222 | 688528628 | median | 30.66152666373413 | 2026-06-01T12:40:07.341Z |
| 4521437396061341233 | 296347236 | Vehicle_Speed_kph | 311318222 | 688528628 | mean | 30.66152666373413 | 2026-06-01T12:40:07.341Z |
