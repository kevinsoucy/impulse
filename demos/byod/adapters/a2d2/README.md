# A2D2 adapter

[A2D2 — Audi Autonomous Driving Dataset](https://www.a2d2.audi/a2d2/en/) plugged into the BYOD demo. Released by Audi in March 2020; 12,499 labeled frames across ~38 scenes recorded on German roads.

This adapter is the second reference adapter after NuScenes. Its value is **proving the adapter abstraction on a structurally different dataset**: A2D2 has real CAN bus signals (NuScenes has none), annotations in the vehicle frame (NuScenes uses global), no cross-frame instance tracking (NuScenes does), and no radar (NuScenes has five). Everything downstream of `scalar_source()` and the per-frame mapper functions is unchanged.

---

## License

CC BY-ND 4.0. NoDerivatives applies to *redistributed derivative datasets* — the demo pipeline reads the data and writes ADAS Delta tables in a separate namespace; that's not redistribution. Internal Databricks demo use is unencumbered. **Confirm with legal before publishing any derived A2D2 data (e.g. Delta Share to a customer).**

## Prerequisites

1. Register at https://www.a2d2.audi/a2d2/en/download.html.
2. Download the `camera_lidar_semantic_bboxes` subset (~400 GB).
3. Extract into `/Volumes/<catalog>/demo_silver/raw/a2d2/camera_lidar_semantic_bboxes/`.

After extraction, the dataroot must contain:

```
/Volumes/<catalog>/demo_silver/raw/a2d2/
├── cams_lidars.json
└── camera_lidar_semantic_bboxes/
    ├── 20180807_145028/
    │   ├── camera/{cam_*}/*.png
    │   ├── lidar/{cam_*}/*.npz
    │   ├── label3D/{cam_*}/*.json
    │   └── bus/bus_signals.json
    ├── 20180810_142822/
    ├── ...
```

`adapter.download(dataroot)` validates this layout — it does not fetch the data, since Audi requires registration.

## Deploy

```bash
cd repos/impulse
databricks bundle deploy --target shared --profile <your-workspace-profile> --var "catalog=<your-catalog>" --var "adapter=a2d2"
databricks bundle run bootstrap_job --target shared --profile <your-workspace-profile> --var "adapter=a2d2"
databricks bundle run run_all_job   --target shared --profile <your-workspace-profile> --var "adapter=a2d2"
```

Deploy with `--target shared` on a shared workspace so every SA sees the same bundle (`dev` works too — it just deploys to a per-user path). See `demos/byod/bundle/README.md` for the target details.

The bundle dependency list in `resources/jobs.yml` does not include any A2D2-specific package — the adapter uses only `numpy`, Pillow, and the stdlib (`json`, `pathlib`).

---

## Coverage

| Phase | ADAS table | A2D2 source |
|---|---|---|
| Phase 1 | `channels` | `bus_signals.json` (real bus signals) + per-frame annotation aggregates |
| Foundation | `perception_channels` | Camera PNGs + LiDAR NPZs (one row per file) |
| Phase 2 | `object_tracks` | A2D2 3D bbox JSON, vehicle-frame coordinates |
| Phase 4 | `lidar_object_detections` | 3D bbox JSON, vehicle frame, sensor_id `LIDAR_FUSED` |
| Phase 4 | `camera_object_detections` | 3D bbox JSON projected to 6 cameras via `cams_lidars.json` |

### Channels populated

**Bus signals** (real ECU readings, no synthesis):

| ADAS channel | A2D2 bus key | Unit |
|---|---|---|
| Vehicle_Speed_kph | `vehicle_speed` | kph |
| Vehicle_Accel_Longitudinal_ms2 | `acceleration_x` | m/s² |
| Vehicle_Accel_Lateral_ms2 | `acceleration_y` | m/s² |
| Vehicle_Accel_Vertical_ms2 | `acceleration_z` | m/s² |
| Yaw_Rate_rads | `angular_velocity_z` | rad/s |
| Roll_Rate_rads | `angular_velocity_x` | rad/s |
| Pitch_Rate_rads | `angular_velocity_y` | rad/s |
| Steering_Angle_deg | `steering_angle_calculated` | deg |
| Brake_Pressure_pct | `brake_pressure` | % |
| Accelerator_Pedal_pct | `accelerator_pedal` | % |
| Pitch_Angle_deg | `pitch_angle` | deg |
| Roll_Angle_deg | `roll_angle` | deg |
| Latitude_deg | `latitude_degree` | deg |
| Longitude_deg | `longitude_degree` | deg |

**Detection aggregates** (derived from per-frame annotations):

| Channel | Class |
|---|---|
| Pedestrian_Count, Pedestrian_Nearest_Distance_m | pedestrian |
| Vehicle_Count_Front, Vehicle_Nearest_Distance_m | car |
| Cyclist_Count, Cyclist_Nearest_Distance_m | cyclist |

---

## Unpopulated tables

These tables remain empty when the bundle runs with `adapter=a2d2`. The pipeline does **not** fall back, synthesize, or mark them as TODO — the gap is a property of the source dataset, documented here so reviewers and customers don't look for it as a bug.

| Table | Why empty for A2D2 |
|---|---|
| `radar_object_detections` | A2D2 has no radar sensor. |
| `lane_markings` | A2D2 has no lane-line annotations. |
| `predicted_trajectories` | A2D2 has no planner output / future-state labels. |

`free_space` *can* be populated from A2D2's semantic-segmentation drivable-class masks (a per-adapter capability that NuScenes can't match without the map extension), but the bbox subset doesn't include the per-pixel labels — only the `camera_lidar_semantic_segmentation` subset does. Adding that requires loading a second A2D2 subset; deferred to a future backlog item.

---

## Field-by-field caveats

- **`object_tracks.relative_velocity_ms` is always `None`.** A2D2's per-frame box keys (`box_0`, `box_1`, …) are local to the frame; the same physical object gets a different key in the next frame. There is no instance token, so cross-frame velocity cannot be computed. TSAL scenario search that depends on `relative_velocity_ms` will silently exclude A2D2 — a real limitation customers should understand if they're using A2D2 for that workload.
- **`object_tracks.source` is constant `"ground_truth_camera_lidar"`.** Annotations were created by labelers fusing camera + LiDAR. No per-annotation sensor-coverage metadata is published, so we encode it as a single string.
- **Ego pose is identity.** A2D2 does not publish a separate global ego pose stream. Box translations are already in the vehicle frame, so `global_to_ego` becomes a no-op and the rest of the geometry pipeline is unchanged.
- **Camera calibration interpretation.** The loader translates A2D2's `view = {origin, x-axis, y-axis}` into the (translation, quaternion) shape that `adas.geometry.rotation_matrix_from_quat` consumes. The published axes are assumed to define a right-handed `[x, y, z=x×y]` basis (the loader re-orthogonalizes `y` to enforce this when necessary).
- **Sample timestamps are synthesized.** A2D2 frame filenames carry a monotonic 9-digit index, not microseconds. The loader synthesizes a microsecond timestamp as `scene_start + frame_index × 100,000 µs` (10 Hz). This keeps Δt accurate at the frame level even though the absolute clock matches only the scene start.

---

## File layout

| File | Purpose |
|---|---|
| `__init__.py` | `Adapter` class + `register("a2d2", Adapter)`. |
| `config.yaml` | `dataset_versions`, `dataroot_template`, `openlabel`, `visualize_format`. |
| `loader.py` | Filesystem walker; typed dataclasses; calibration parser. |
| `scalar_source.py` | Bus-signal decode + per-frame detection aggregates. |
| `object_tracks.py` | Vehicle-frame → object_tracks row assembly. |
| `lidar.py` | 3D bbox → lidar_object_detections (vehicle frame, no transform). |
| `camera.py` | 3D bbox → camera_object_detections via 6-camera projection. |
| `ingest.py` | container_tags / container_metrics / channel_tags writes; `perception_paths`. |
| `download.py` | dataroot completeness validator (no scripted download). |
