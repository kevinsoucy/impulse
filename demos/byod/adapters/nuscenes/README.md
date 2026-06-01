# NuScenes adapter

[NuScenes](https://www.nuscenes.org/) plugged into the BYOD demo. The reference adapter — NuScenes has no CAN bus, so `scalar_source()` synthesizes 9 scalars per scene from ego pose + annotations. The four pipeline notebooks (`01`–`04`) are byte-identical across adapters; all NuScenes-specific logic lives in this package.

---

## License

[NuScenes Terms of Use](https://www.nuscenes.org/terms-of-use) — free for non-commercial research and product evaluation; commercial use requires written permission from Motional. Free Databricks-internal demo and customer-evaluation use is covered; **do not publish derived NuScenes data** (Delta Share, customer-facing dataset, public demo workspace) without a commercial agreement.

## Prerequisites

1. Accept the EULA at https://www.nuscenes.org/sign-up (one-time registration, free).
2. Populate the dataroot at `/Volumes/<catalog>/<schema_prefix>_silver/raw/v1.0-mini/` — either path works:

   **A. Server-side download (fastest — recommended).** From a notebook attached to serverless or any cluster with internet:
   ```python
   import tarfile, urllib.request
   from pathlib import Path
   URL      = "https://www.nuscenes.org/data/v1.0-mini.tgz"   # ~4.2 GB
   TARBALL  = Path("/Volumes/<catalog>/<schema_prefix>_silver/raw/v1.0-mini.tgz")
   DATAROOT = Path("/Volumes/<catalog>/<schema_prefix>_silver/raw/v1.0-mini")
   DATAROOT.mkdir(parents=True, exist_ok=True)
   with urllib.request.urlopen(URL) as r, open(TARBALL, "wb") as f:
       while chunk := r.read(8 * 1024 * 1024):
           f.write(chunk)
   with tarfile.open(TARBALL, "r:gz") as tf:
       tf.extractall(DATAROOT, filter="data")
   ```

   **B. Local upload (slower — only if outbound network is restricted on the workspace).** Download to your laptop, then:
   ```bash
   databricks fs cp v1.0-mini.tgz dbfs:/Volumes/<catalog>/<schema_prefix>_silver/raw/v1.0-mini.tgz
   # Extract via a serverless notebook (same tarfile snippet as path A, without the urlopen).
   ```

After extraction, the dataroot must contain four subdirectories — the layout `adapter.download()` validates:

```
/Volumes/<catalog>/<schema_prefix>_silver/raw/v1.0-mini/
├── maps/
├── samples/
├── sweeps/
└── v1.0-mini/
```

`v1.0-trainval` is the same flow with a different tarball (~350 GB across multiple files — see [the NuScenes download page](https://www.nuscenes.org/nuscenes#download) for the file list). Default is `v1.0-mini`.

## Deploy

```bash
cd repos/impulse
databricks bundle deploy --target shared --profile <your-workspace-profile> --var "catalog=<your-catalog>"
databricks bundle run bootstrap_job --target shared --profile <your-workspace-profile> --var "catalog=<your-catalog>"
databricks bundle run run_all_job   --target shared --profile <your-workspace-profile> --var "catalog=<your-catalog>"
```

See `demos/byod/bundle/README.md` for the full prerequisites + first-time-deploy walkthrough.

---

## Coverage

| Phase | LakeVision table | NuScenes source |
|---|---|---|
| Phase 1 | `channels` | Synthesized 9 scalars/scene from ego pose deltas + annotation aggregates (no CAN bus). See `scalar_source.py` for the exact list. |
| Foundation | `perception_channels` | Camera JPEGs (front + 5 surround) + LiDAR `.bin` from `samples/` |
| Phase 2 | `object_tracks` | `sample_annotation` table joined with `instance` for per-track persistence; `category` filtered to a Phase 2 vocabulary |
| Phase 4 | `lidar_object_detections` | Same annotations, sensor_id `LIDAR_TOP` |
| Phase 4 | `camera_object_detections` | Same annotations projected per camera using the `calibrated_sensor` + `ego_pose` transforms (`lib/geometry.py`) |

### Synthesized scalar channels

NuScenes ships no CAN bus, so `scalar_source.py` derives the following from ego pose deltas + per-frame annotation aggregates:

| Channel | Source |
|---|---|
| `Vehicle_Speed_kph` | Ego pose translation Δ / Δt × 3.6 |
| `Yaw_Rate_rads` | Ego pose rotation Δ / Δt |
| `Pedestrian_Nearest_Distance_m` | min(distance) over `human.pedestrian.*` annotations |
| `Vehicle_Nearest_Distance_m` | min(distance) over `vehicle.*` annotations |
| `Vehicle_Count_Front` | count(`vehicle.*`) with positive x-axis projection |
| `Vehicle_Count_Surround` | count(`vehicle.*`) total |
| `Pedestrian_Count` | count(`human.pedestrian.*`) |
| `Cyclist_Count` | count(`human.cyclist.*`) |
| `Lane_Change_Heading_Delta_deg` | abs(yaw Δ over a 1 s window) × 180/π |

Each derived channel carries `synthesized=true` and `derivation` tags in `channel_tags`.

---

## File layout

| File | Purpose |
|---|---|
| `__init__.py` | `Adapter` class + `register("nuscenes", Adapter)`. |
| `config.yaml` | `dataset_versions`, `dataroot_template`, OpenLABEL + visualize hints. |
| `loader.py` | Reads the nuScenes metadata JSON directly (`_NuScenesMeta`); `Scene` dataclass. |
| `scalar_source.py` | Per-scene 9-scalar derivation. |
| `object_tracks.py` | `sample_annotation` → Phase 2 schema with instance-id persistence. |
| `lidar.py` | 3D box → `lidar_object_detections` (sensor_id `LIDAR_TOP`). |
| `camera.py` | 3D box → `camera_object_detections` via `calibrated_sensor` + `ego_pose` transforms (`lib/geometry.py`). |
| `ingest.py` | container_tags / container_metrics / channel_tags writes; `perception_paths`. |
| `download.py` | dataroot completeness validator (no scripted download — EULA required). |
