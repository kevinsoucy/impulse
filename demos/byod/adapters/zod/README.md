# ZOD adapter

[ZOD — Zenseact Open Dataset](https://zod.zenseact.com/) plugged into the BYOD demo. Released by Zenseact (Volvo Cars spinout) in 2023. ZOD Sequences contains 1,473 20-second clips covering European motorway + urban driving across 14 countries.

This adapter is the third reference adapter after NuScenes and A2D2. Its value is **proving the abstraction on a dataset with radar Doppler and HDF5 scalar source**: ZOD is the only public ADAS dataset with both a commercially usable license AND per-detection Doppler radial velocity. Notebook 04's KPI tables populate a non-empty `radar` modality only when running with `adapter=zod`.

---

## License

CC BY-SA 4.0. ShareAlike applies to *published* derivative datasets — internal Databricks demo use (data consumed, tables written into a private namespace) is unencumbered. **Confirm with legal before publishing any derived ZOD data** (customer-facing Delta Share, public demo workspace, etc.).

## Prerequisites

1. Register at https://zod.zenseact.com/download/.
2. Download `sequences-mini` (~20 GB) or `sequences-full` (~250 GB).
3. Extract into `/Volumes/<catalog>/lakevision_demo_silver/raw/zod/sequences-mini/`.

After extraction, the dataroot must contain at least one sequence directory:

```
/Volumes/<catalog>/lakevision_demo_silver/raw/zod/sequences-mini/
└── sequences/
    ├── 000001/
    │   ├── annotations/object_detection_3d.json
    │   ├── camera_front_blur/{timestamp}.jpg
    │   ├── lidar_velodyne/{timestamp}.npy
    │   ├── radar/{timestamp}.npy
    │   ├── oxts.hdf5
    │   ├── calibration.json
    │   └── metadata.json
    ├── 000002/
    └── ...
```

The adapter also accepts a flat `{dataroot}/sequences/...` layout for cases where the version dir is omitted.

`adapter.download(dataroot)` validates this layout but does not fetch — ZOD requires registration.

## Deploy

> **Note:** the ZOD adapter is deprecated (CC BY-SA license posture, see engine CHANGELOG 2026-05-22) and is not included in any deployed wheel. The commands below are kept for reference only — if you re-enable the adapter, you also need to extend `demos/byod/pyproject.toml`'s `packages.find.include` glob to add `adapters.zod*`.

```bash
cd repos/impulse
databricks bundle deploy --target shared --profile <your-workspace-profile> --var "catalog=<your-catalog>" --var "adapter=zod"
databricks bundle run bootstrap_job --target shared --profile <your-workspace-profile> --var "adapter=zod"
databricks bundle run run_all_job   --target shared --profile <your-workspace-profile> --var "adapter=zod"
```

ZOD's adapter requires `h5py>=3.0` (declared in `config.yaml`).

---

## Coverage

| Phase | LakeVision table | ZOD source |
|---|---|---|
| Phase 1 | `channels` | `oxts.hdf5` (HDF5 IMU + GNSS) + per-frame annotation aggregates |
| Foundation | `perception_channels` | Camera JPEGs + LiDAR .npy (radar `.npy` intentionally excluded from this table) |
| Phase 2 | `object_tracks` | `annotations/object_detection_3d.json`, vehicle frame, `relative_velocity_ms` computed across frames via `uuid` |
| Phase 4 | `lidar_object_detections` | Same JSON; sensor_id `LIDAR_FUSED` |
| Phase 4 | `camera_object_detections` | Same JSON projected to `camera_front_blur` via `calibration.json` |

### Bus signals decoded from `oxts.hdf5`

| LakeVision channel | HDF5 dataset path | Unit applied |
|---|---|---|
| Vehicle_Speed_kph | `/speed` | × 3.6 (m/s → kph) |
| Vehicle_Accel_Longitudinal_ms2 | `/acceleration[0]` | (no scale) |
| Vehicle_Accel_Lateral_ms2 | `/acceleration[1]` | (no scale) |
| Vehicle_Accel_Vertical_ms2 | `/acceleration[2]` | (no scale) |
| Roll_Rate_rads | `/angular_rate[0]` | (no scale) |
| Pitch_Rate_rads | `/angular_rate[1]` | (no scale) |
| Yaw_Rate_rads | `/angular_rate[2]` | (no scale) |
| Heading_deg | `/heading` | × 57.296 (rad → deg) |
| Pitch_Angle_deg | `/pitch` | × 57.296 |
| Roll_Angle_deg | `/roll` | × 57.296 |
| Latitude_deg | `/latitude` | (no scale) |
| Longitude_deg | `/longitude` | (no scale) |

Each derived channel carries an `oxts_path` tag in `channel_tags` pointing back to the source HDF5 path.

### `object_tracks.source` semantics

| ZOD annotation state | Resulting source |
|---|---|
| `lidar_attribute.num_points > 0` and `radar_attribute.num_points > 0` and `occlusion < 0.8` | `lidar\|radar\|camera` |
| LiDAR + camera, no radar | `lidar\|camera` |
| Radar + camera, no LiDAR | `radar\|camera` |
| All flags off (rare, heavily occluded) | `camera` (fallback) |

**Notebook 04's `distance_stats_by_modality` will show non-empty `radar` rows only with `adapter=zod`** — this is the radar differentiator.

---

## Unpopulated tables

| Table | Why empty for ZOD |
|---|---|
| `radar_object_detections` | Per-detection radar emission is a planned ZOD-specific extension. Doppler is captured per annotation in `Annotation.radar_doppler_ms` (loader.py); a future notebook can lift these into a dedicated radar table. |
| `predicted_trajectories` | ZOD has no planner output / future-state labels. |
| `lane_markings` | ZOD Frames has lane-line annotations; ZOD Sequences (what this adapter targets) does not. |
| `free_space` | ZOD Frames has drivable-area segmentation masks; ZOD Sequences does not. |

---

## Field-by-field caveats

- **`relative_velocity_ms` is populated.** Unlike A2D2, ZOD ships persistent `uuid` per object across frames within a sequence. The adapter caches the previous `(timestamp, vehicle_frame_xy)` per uuid and emits a signed Δ‖position‖/Δt.
- **`sensor_id = "LIDAR_FUSED"`.** ZOD ships a single pre-fused 3-LiDAR roof-rack point cloud per frame — there are no per-physical-sensor LiDAR rows.
- **Ego pose is identity.** ZOD annotations are in the vehicle frame; the OXTS-derived ego pose is used for scalar channels, not for box transforms.
- **Sample cadence is LiDAR-driven (~10 Hz).** The loader walks the LiDAR directory and nearest-timestamp-matches camera + radar files. Mismatched per-sensor cadence is tolerated; misaligned timestamps within ~50 ms is the typical Zenseact convention.
- **Filename timestamps may be nanoseconds or microseconds.** The loader auto-detects (any 16+ digit number > 10^15 is treated as ns and divided by 1000); ZOD's published filenames use ns.
- **Camera variant.** ZOD ships multiple processings of the front camera (`blur`, `dnat`, `original`). The adapter targets `camera_front_blur` — Zenseact's recommended variant for downstream use. Switching variants is a 1-line change to `ZodLoader.PRIMARY_CAMERA`.

---

## File layout

| File | Purpose |
|---|---|
| `__init__.py` | `Adapter` class + `register("zod", Adapter)`. |
| `config.yaml` | `dataset_versions`, `dataroot_template`, OXTS + visualize hints, `h5py` dep. |
| `loader.py` | Filesystem walker; OXTS HDF5 reader; calibration + annotation parsers. |
| `scalar_source.py` | OXTS HDF5 decode + per-frame detection aggregates. |
| `object_tracks.py` | Per-uuid relative-velocity tracking + tri-modal source encoding. |
| `lidar.py` | 3D box → lidar_object_detections (`LIDAR_FUSED`). |
| `camera.py` | 3D box → camera_object_detections via the front-blur projection chain. |
| `ingest.py` | container_tags / container_metrics / channel_tags writes; `perception_paths`. |
| `download.py` | dataroot completeness validator (no scripted download). |
