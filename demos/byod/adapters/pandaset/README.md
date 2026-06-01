# PandaSet adapter

[PandaSet](https://pandaset.org/) from Hesai Technology and Scale AI, plugged into the BYOD demo. Released 2020, then re-hosted by Hesai in 2022 after Scale AI ended public hosting. 103 sequences × 8 seconds × 10 Hz ≈ ~80 GB.

This adapter is the fourth and final reference adapter. Its value is twofold:

1. **Cleanest license.** CC BY 4.0 (no ShareAlike, no NoDerivatives, no NC clause) — a strict upgrade over ZOD's BY-SA 4.0 and A2D2's BY-ND 4.0 for any scenario involving customer-facing derived data.
2. **Dual heterogeneous LiDAR.** Pandar64 spinning roof-rack + PandarGT solid-state forward-facing. The adapter emits TWO `lidar_object_detections` rows per cuboid (one per physical sensor), giving notebook 04's `distance_stats_by_modality` a within-adapter sensor comparison no other adapter in the suite can provide.

---

## License

CC BY 4.0. Attribution required when distributing derived material; no other restrictions. **The cleanest license in the four-adapter suite** — recommend this dataset whenever a customer demo will involve Delta Sharing derived data.

## Prerequisites

1. Register at https://pandaset.org/ (Hesai data portal).
2. Download the v1 release (~80 GB).
3. Extract into `/Volumes/<catalog>/lakevision_demo_silver/raw/pandaset/`.

After extraction:

```
/Volumes/<catalog>/lakevision_demo_silver/raw/pandaset/
├── 001/
│   ├── annotations/
│   │   └── cuboids/00.pkl.gz, 01.pkl.gz, ..., 79.pkl.gz
│   ├── camera/
│   │   ├── front_camera/{00.jpg ... 79.jpg, intrinsics.json, poses.json}
│   │   ├── front_left_camera/
│   │   └── …
│   ├── lidar/{00.pkl.gz ... 79.pkl.gz, poses.json}
│   └── meta/
│       ├── gps.csv
│       └── timestamps.json
├── 002/
└── ...
```

`adapter.download(dataroot)` validates this layout; the Hesai portal handles the actual download.

## Deploy

```bash
cd repos/impulse
databricks bundle deploy --target shared --profile <your-workspace-profile> --var "catalog=<your-catalog>" --var "adapter=pandaset"
databricks bundle run bootstrap_job --target shared --profile <your-workspace-profile> --var "adapter=pandaset"
databricks bundle run run_all_job   --target shared --profile <your-workspace-profile> --var "adapter=pandaset"
```

Deploy with `--target shared` on a shared workspace so every SA sees the same bundle (`dev` works too — it just deploys to a per-user path). See `demos/byod/bundle/README.md` for the target details.

PandaSet's adapter has no dataset-specific pip dependencies — the `.pkl.gz` reads use stdlib `gzip` + `pickle`, with pandas (already in the Databricks runtime) for DataFrame coercion. Lightest adapter dependency surface of the four.

---

## Coverage

| Phase | LakeVision table | PandaSet source |
|---|---|---|
| Phase 1 | `channels` | `meta/gps.csv` (speed, heading, lat, lon) + per-frame annotation aggregates |
| Foundation | `perception_channels` | 6 camera JPEGs + 2 LiDAR `.pkl.gz` rows per frame (one per physical sensor) |
| Phase 2 | `object_tracks` | `annotations/cuboids/*.pkl.gz`, vehicle frame, `relative_velocity_ms` computed via persistent `uuid` |
| Phase 4 | `lidar_object_detections` | **Two rows per cuboid**: `LIDAR_SPINNING` (Pandar64) + `LIDAR_SOLIDSTATE` (PandarGT) |
| Phase 4 | `camera_object_detections` | Cuboids projected to 6 cameras via per-camera `intrinsics.json` + `poses.json` |

### Scalar channels from `meta/gps.csv`

| LakeVision channel | CSV column(s) | Unit applied |
|---|---|---|
| Vehicle_Speed_kph | `speed` / `velocity` | × 3.6 (m/s → kph) |
| Heading_deg | `heading` / `altitude_heading` | (already deg) |
| Latitude_deg | `lat` / `latitude` | — |
| Longitude_deg | `lon` / `lng` / `longitude` | — |

PandaSet's GPS CSV has **no accelerometer or gyro data**, so kinematic coverage is thinner than A2D2 or ZOD. Sufficient for TSAL speed thresholding and heading-change event detection.

### `object_tracks.source` semantics

Constant `"lidar|camera"` for every cuboid — PandaSet has no radar sensor. Notebook 04's radar modality is unpopulated for `adapter=pandaset` (vs. non-empty for `adapter=zod`).

### Dual-LiDAR row emission (the dual-LiDAR differentiator)

For each cuboid the adapter emits:

```
{... cuboid fields ..., sensor_id: "LIDAR_SPINNING"}      # Pandar64
{... cuboid fields ..., sensor_id: "LIDAR_SOLIDSTATE"}    # PandarGT
```

These rows are populated only inside TSAL event windows. Notebook 04's `distance_stats_by_modality` then partitions on `sensor_id` to show two separate range distributions per scene — a sensor evolution story (spinning roof-rack is the current OEM norm; solid-state forward-facing is where platform hardware is heading) that's not derivable from any other adapter.

`channel_tags` also carries `lidar_model = Pandar64|PandarGT` and `lidar_type = spinning|solid_state` for the two LiDAR channel IDs, so analysts can filter on the physical sensor properties directly.

---

## Unpopulated tables

| Table | Why empty for PandaSet |
|---|---|
| `radar_object_detections` | PandaSet has no radar sensor. |
| `predicted_trajectories` | PandaSet has no planner output. |
| `lane_markings` | PandaSet has no lane-line annotations. |
| `free_space` | PandaSet has point-cloud semseg labels (drivable surface) accessible via `.semseg`; loading them is a future extension because it requires reading additional `.pkl.gz` files per frame that this adapter does not yet load. |

---

## Field-by-field caveats

- **Annotations are read as `pd.DataFrame` rows.** PandaSet's published cuboid pickle is a pandas DataFrame; the loader coerces it to dict-per-row so downstream code stays pandas-agnostic. Both `position.x/y/z` (dot-notation columns) and `position: {x, y, z}` (nested dict) forms are tolerated — the published format uses dot-notation, but the dataset documentation has historically been inconsistent.
- **`yaw_rad` is the only rotation freedom.** PandaSet cuboids carry only a yaw angle (rotation around vertical Z) — no roll/pitch — so the adapter's `_quat_from_yaw` helper builds a (w, 0, 0, sin(yaw/2)) quaternion. The other adapters carry full 3-DoF rotations via quaternion or axis-angle.
- **`object_id` is hashed from `uuid`.** PandaSet's per-instance UUID is a string; we hash it to a 63-bit non-negative int for the LONG-typed `object_id` column. Persistence across frames is preserved — same UUID maps to same int.
- **Frame timestamps come from `meta/timestamps.json`.** Each sequence ships a per-frame timestamp list (seconds since recording start). If the file is missing, the loader falls back to `scene.start_ts_us + frame_index × 100,000 µs` (10 Hz).
- **Camera poses are per-frame.** Unlike NuScenes (per-keyframe) or A2D2 (single static pose per camera), PandaSet's `camera/<cam>/poses.json` ships a pose entry for every frame. The loader looks up the entry at `sample.frame_index` to project cuboids correctly.

---

## File layout

| File | Purpose |
|---|---|
| `__init__.py` | `Adapter` class + `register("pandaset", Adapter)`. |
| `config.yaml` | `dataset_versions: [v1]`, `dataroot_template`, OpenLABEL + visualize hints. |
| `loader.py` | Filesystem walker; gzip-pickle reader; CSV parser; per-frame camera-pose lookup. |
| `scalar_source.py` | GPS CSV decode + per-frame detection aggregates. |
| `object_tracks.py` | Vehicle-frame row assembly with per-uuid relative-velocity tracking. |
| `lidar.py` | **Dual-LiDAR row emission** (`LIDAR_SPINNING` + `LIDAR_SOLIDSTATE`). |
| `camera.py` | Cuboid → 6-camera 2D bbox projection. |
| `ingest.py` | container_tags / container_metrics / channel_tags writes; `perception_paths` (per-physical-LiDAR rows). |
| `download.py` | dataroot completeness validator (no scripted download). |
