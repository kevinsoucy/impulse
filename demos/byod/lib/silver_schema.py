"""Silver/scenario-layer schemas for the BYOD ADAS demo.

These table *schemas* are demo-owned data definitions, so they live with the demo.

- ``PERCEPTION_CHANNELS`` — file-path index for camera frames / LiDAR scans in UC
  Volumes. Keyed by ``(container_id, channel_id, timestamp)``; joins to
  ``channel_tags`` on ``channel_id``.
- ``OBJECT_TRACKS`` — one fused row per tracked object per frame. In Impulse 2.0
  this table is registered as a ``Series`` (``session_col="container_id"``,
  ``timestamp_col="frame_ts"``, ``entity_key="object_id"``) so predicates can span
  it and the scalar ``channels`` signal in one expression via the cogroup.
"""

import pyspark.sql.types as T

# File-path index for camera frames, LiDAR scans, and other binary media in UC
# Volumes. Same container_id + channel_id abstraction as Impulse core channels;
# channel_id maps to a channel_tags row with key=sensor_type.
PERCEPTION_CHANNELS = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("channel_id", T.IntegerType(), nullable=False),
        T.StructField("timestamp", T.LongType(), nullable=False),  # microseconds
        T.StructField("file_path", T.StringType(), nullable=False),  # UC Volume path
        T.StructField("format", T.StringType()),  # jpeg, png, mp4, h264, pcd, …
    ]
)

# One fused row per tracked object per frame. Sensor-agnostic (populated from
# fused output, not per-sensor raw geometry). Stores only attributes needed for
# scenario search. Ingestion mode is controlled by ObjectTracksConfig.
OBJECT_TRACKS = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("frame_ts", T.LongType(), nullable=False),  # microseconds; aligns to channels.tstart
        T.StructField("object_id", T.LongType(), nullable=False),  # stable tracked identity across frames
        T.StructField("detection_class", T.StringType()),  # pedestrian, car, cyclist, truck, motorcycle, bus
        T.StructField("distance_m", T.DoubleType()),  # range from ego; primary source: LiDAR
        T.StructField("lane_offset", T.IntegerType()),  # relative to ego lane: −2, −1, 0, +1, +2
        T.StructField("relative_velocity_ms", T.DoubleType()),  # negative = approaching; primary: radar Doppler
        T.StructField("azimuth", T.StringType()),  # sector enum: front, front_left, …, rear
        T.StructField("confidence", T.DoubleType()),
        T.StructField("source", T.StringType()),  # pipe-delimited sensor provenance, e.g. lidar|radar|camera
    ]
)

# Per-keyframe map context — the ego vehicle's situation on the HD map, derived by
# joining the ego pose against the nuScenes map-expansion layers (lib/map_layers.py).
# Registered as a point-in-time Series (session=container_id, timestamp=frame_ts;
# signal_id synthesized at registration). frame_ts is microseconds, aligned to
# object_tracks.frame_ts and channels.tstart so the three compose in one query.
# Membership is INT 0/1 (the series accessor exposes numeric + string columns only,
# not boolean); distances are meters (0.0 when on the feature, NULL if the layer is
# empty for that location); lane_id/location are strings.
EGO_MAP_CONTEXT = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("frame_ts", T.LongType(), nullable=False),  # microseconds; keyframe timestamp
        T.StructField("location", T.StringType()),  # e.g. singapore-onenorth, boston-seaport
        T.StructField("on_ped_crossing", T.IntegerType()),  # 0/1
        T.StructField("on_walkway", T.IntegerType()),  # 0/1
        T.StructField("on_drivable_area", T.IntegerType()),  # 0/1
        T.StructField("in_intersection", T.IntegerType()),  # 0/1
        T.StructField("dist_to_ped_crossing_m", T.DoubleType()),  # 0.0 = on it; NULL if no layer
        T.StructField("dist_to_stop_line_m", T.DoubleType()),
        T.StructField("lane_id", T.StringType()),  # map lane token under ego, NULL if none
    ]
)

# Per-keyframe per-object map context — each detected object's situation on the map,
# plus ego-relative facts the predicate DSL can't express across entities (the engine
# correlates entities by time-overlap, not column equality), pre-computed here:
# `same_lane_as_ego`. Registered as an entity-keyed Series (entity_key=object_id),
# matching object_tracks' (container_id, frame_ts, object_id) grain so map and motion
# predicates compose per object.
OBJECT_MAP_CONTEXT = T.StructType(
    [
        T.StructField("container_id", T.LongType(), nullable=False),
        T.StructField("frame_ts", T.LongType(), nullable=False),
        T.StructField("object_id", T.LongType(), nullable=False),
        T.StructField("detection_class", T.StringType()),
        T.StructField("on_ped_crossing", T.IntegerType()),  # 0/1
        T.StructField("on_walkway", T.IntegerType()),  # 0/1
        T.StructField("in_intersection", T.IntegerType()),  # 0/1
        T.StructField("same_lane_as_ego", T.IntegerType()),  # 0/1; object lane == ego lane
        T.StructField("dist_to_ped_crossing_m", T.DoubleType()),
        T.StructField("lane_id", T.StringType()),
    ]
)
