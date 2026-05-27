# BYOD — Bring Your Own Data

**Local verification:** integration n/a (Java not installed — skipped), bundle n/a (no local bundle test).
**Workspace E2E:** 2026-05-26 (green) — nuscenes/v1.0-mini on fevm-adas-engine-validate / adas_engine_validate_catalog.

The reference demo for **how to add a new dataset to LakeVision**. Every dataset (NuScenes, A2D2, customer MDF4, …) plugs in as an *adapter*. The four pipeline notebooks (`01`–`04`) are byte-identical across adapters; the per-dataset logic lives entirely inside `adapters/<name>/`. Adding a new dataset means writing one Python package — zero notebook edits.

> **What this proves:** every layer downstream of dataset-specific ingest works identically regardless of source. TSAL scenario search, OpenLABEL export, KPI comparison, and visualization are all generic.
>
> **What it does not prove:** MDF4 decode itself — the bundled NuScenes adapter synthesizes scalars from ego pose + annotations because NuScenes has no CAN bus. Real MDF4 ingest plugs in as another adapter (same shape) using Impulse's MDF4 reader inside its `scalar_source()`.

---

## Architecture — adapter Protocol + registry

| Layer | What lives here | New-dataset work |
|---|---|---|
| **Adapter package** (`adapters/<name>/`) | Loader, dataset-specific row assembly, `config.yaml` | **Write this** |
| **`lib/adapter.py`** | `Adapter` Protocol + `register()` / `resolve()` registry | Don't edit |
| **`lib/byod_config.py`** | `BYODConfig` — UC paths, adapter settings loader | Don't edit |
| **`lib/visualization.py`** | Camera / LiDAR readers + channel-resolution helpers | Don't edit |
| **`lib/sensor_kpi.py`** | KPI helpers used by `04_visualize.py` | Don't edit |
| **Generic notebooks** (`01`–`04`) | Provisioning, ingest, TSAL, OpenLABEL, KPI, visualize | Don't edit |
| **Core helpers** (`src/lakevision/`) | `geometry`, `windowing`, `openlabel`, `scalar_metrics`, `playlists`, schemas | Don't edit |

### Adapter Protocol contract

See `lib/adapter.py` for the canonical Protocol. Summary:

| Method | Called by | Purpose |
|---|---|---|
| `download(dataroot)` | `00_download.py` | Validate or fetch source data. Optional — may raise FileNotFoundError to signal "manual step required". |
| `ingest_metadata(spark)` | `01_ingest.py` (Step 2) | Write `container_tags`, `container_metrics`, `channel_tags` rows. |
| `scalar_source(spark)` | `01_ingest.py` (Step 3) | Return a Spark DataFrame matching `CHANNELS_SCHEMA`. Implementations: synthesize, bus-decode, MDF4-decode. |
| `perception_paths()` | `01_ingest.py` (Step 4) | Yield `perception_channels` rows (one per camera frame / LiDAR scan). |
| `scenes()` | `01_ingest.py` (Step 5), `03_per_event_detail.py` | List all scenes; each must expose `.container_id` and `.name`. |
| `map_to_object_tracks(scene, min_confidence)` | `01_ingest.py` (Step 5) | Per-frame row assembly. Geometry/encoding helpers live in `lakevision.geometry`. |
| `map_to_lidar_detections(scene, event_windows)` | `03_per_event_detail.py` | Phase 4 cuboids, restricted to TSAL event windows. |
| `map_to_camera_detections(scene, event_windows)` | `03_per_event_detail.py` | 2D bbox projections, same windowing contract. |
| `openlabel_metadata()` | `03_per_event_detail.py` | `{annotator, exporter, stream_description_prefix}` strings injected into OpenLABEL output. |
| `visualize_format()` | `04_visualize.py` | Hints: `camera_reader`, `lidar_reader`, `lidar_dtype`, `lidar_stride`. |

### Adapter `config.yaml` schema

Each adapter ships a `config.yaml` (read by `BYODConfig.for_adapter()`):

```yaml
adapter_name: nuscenes
dataset_versions: [v1.0-mini, v1.0-trainval]
default_version: v1.0-mini

# {vroot} = /Volumes/{catalog}/{schema_prefix}_silver
dataroot_template: "{vroot}/raw/{version}"

openlabel:
  annotator: nuscenes_ground_truth
  exporter: lakevision-demo/byod/adapters/nuscenes/openlabel
  stream_description_prefix: "NuScenes "

visualize_format:
  lidar_dtype: float32
  lidar_stride: 5
  camera_reader: pil       # pil | opencv
  lidar_reader: fromfile   # fromfile | npz | pickle_gz

python_deps:
  - nuscenes-devkit==1.1.11
```

---

## The four pipeline beats

The customer story arc: **load → find → package → show.**

| # | Notebook | Story beat | Inputs | Outputs |
|---|---|---|---|---|
| 00 | `00_download.py` (script) | Stage | adapter, variant | Validates dataroot or downloads |
| 01 | `01_ingest.py` | Load | dataset metadata + scalars + perception paths + annotations | `container_tags`, `container_metrics`, `channel_tags`, `channels`, `channel_metrics`, `perception_channels`, `object_tracks` |
| 02 | `02_detect_events.py` | Find | `channels` | `event_instance_fact`, `playlist_items` |
| 03 | `03_per_event_detail.py` | Package | playlist + per-window adapter mappings | `lidar_object_detections`, `camera_object_detections`, OpenLABEL JSON packages |
| 04 | `04_visualize.py` | Show | playlist + perception_channels + object_tracks | Event walkthrough figures + sensor KPI tables/histograms |

`01_ingest.py` accepts a `bootstrap_only=true` widget to stop after metadata ingest — used by `bootstrap_job` for table re-initialization without re-running the full pipeline.

Notebooks 02, 03, and 04 are byte-identical across BYOD adapters; the entire per-dataset surface lives inside `adapters/<name>/`.

---

## Bundled adapters

### `nuscenes`

The reference adapter — proves the data model end-to-end on the public NuScenes dataset. NuScenes has no CAN bus; `scalar_source()` synthesizes 9 scalars from ego pose + annotations. See [`adapters/nuscenes/README.md`](adapters/nuscenes/README.md) for the full data-acquisition flow (public URL, both server-side-download and local-upload paths) and the per-channel coverage table.

Quick start:
- [ ] Accept the EULA at https://www.nuscenes.org/sign-up (one-time, free).
- [ ] Server-side download from `https://www.nuscenes.org/data/v1.0-mini.tgz` into `/Volumes/<catalog>/lakevision_demo_silver/raw/v1.0-mini/` — snippet in the adapter README.
- [ ] Deploy the bundle with `databricks bundle deploy --target shared --profile <your-workspace-profile> --var "catalog=<your-catalog>"`.

### `a2d2`

Audi Autonomous Driving Dataset — real bus signals (no synthesis), camera + LiDAR + 3D box annotations in the vehicle frame. The reference adapter for non-global box frames and dataset-shipped CAN bus data. See `adapters/a2d2/README.md`.

### `pandaset`

PandaSet from Hesai / Scale AI — cleanest license (CC BY 4.0) and dual heterogeneous LiDAR (Pandar64 spinning + PandarGT solid-state). The only adapter that emits per-physical-LiDAR rows, so notebook 04's KPI comparison can compare spinning vs. solid-state range distributions within a single adapter. See `adapters/pandaset/README.md`.

---

## Adding a new adapter

1. **Create the package directory:**
   ```
   demos/byod/adapters/<your_name>/
   ├── __init__.py       ← Adapter class + register("<your_name>", Adapter)
   ├── config.yaml       ← dataset_versions, dataroot_template, openlabel, visualize_format
   ├── loader.py         ← thin wrapper around your dataset's SDK
   ├── scalar_source.py  ← decode bus signals or synthesize scalars
   ├── object_tracks.py  ← per-frame row assembly (uses lakevision.geometry)
   ├── lidar.py          ← 3D cuboid → lidar_object_detections row assembly
   ├── camera.py         ← 3D box → camera_object_detections row assembly
   ├── ingest.py         ← container_tags / channel_tags / container_metrics writes
   └── download.py       ← optional CLI download helper
   ```

2. **Implement the `Adapter` class** in `__init__.py`. Use `adapters/nuscenes/__init__.py` as the template. Register on import: `register("<your_name>", Adapter)`.

3. **Test before deploy.** Tests live under `tests/demos/byod/adapters/<your_name>/`. The NuScenes adapter ships with mapper unit tests that use `FakeLoader` stubs — copy the pattern.

4. **Deploy the bundle** with `--var "adapter=<your_name>"`. The four generic notebooks are unchanged.

5. **Acceptance:**
   - [ ] Bundle deploys without notebook edits
   - [ ] Notebooks 01–04 run end-to-end on at least one dataset variant
   - [ ] `tests/demos/byod/adapters/<your_name>/` passes

---

## Relationship to the broader engine

The `BACKLOG.md` and `CHANGELOG.md` at the engine root track open and completed implementation work. The technical decisions that shape the demo's structure live under `engines/adas-on-databricks/technical/` (ADRs).

This BYOD demo is the **integration test** for the engine's perception data model — if a new dataset adapter compiles, the data model carries it.
