# ADAS demo bundle (ADAS data domain)

Databricks Asset Bundle for the Impulse + ADAS BYOD demo. Two structural choices vs. the older `byod_demo` bundle:

- **Build and upload wheels automatically** via DAB's `artifacts` block — no manual `uv build && databricks fs cp` step.
- **Ship a `notebook_env.yml`** so SAs can attach the same dependency set to interactive serverless notebook runs (the Environment side panel's "Custom" option).

Same notebooks (`demos/byod/*.py`), same adapter packages — different deploy and runtime story.

---

## Prerequisites

- Databricks CLI v1.0.0 or newer (`databricks --version`).
- `uv` 0.4 or newer (`uv --version`). DAB invokes `uv build --wheel` during deploy.
- A workspace where you have `USE CATALOG` + `CREATE SCHEMA` on the target catalog.
- An authenticated profile in `~/.databrickscfg` for your target workspace. If you don't have one yet:
  ```bash
  databricks auth login --host <your-workspace-url> --profile <name>
  ```
  This opens a browser, completes OAuth, and writes the profile. See [Databricks CLI authentication](https://docs.databricks.com/dev-tools/cli/authentication.html) for token-based and service-principal alternatives.

---

## First-time deploy

```bash
# From the impulse repo root:
databricks bundle deploy \
  --target shared \
  --profile <your-workspace-profile> \
  --var "catalog=<your-catalog>"
```

`catalog` is required and has no default — workspaces vary on what catalogs are provisioned. Other defaults: `schema_prefix=demo`, `adapter=nuscenes`, `dataset_version=v1.0-mini`. Override any of these with additional `--var "name=value"` flags.

> **Prefer `--target shared` on a shared workspace.** The target controls only *where the bundle's code and jobs live* — not where data lands. Both targets write to the same Unity Catalog tables, because the schemas come from `--var "schema_prefix=…"` (default `demo`) and are created by `01_ingest.py`. The difference: the default `dev` target is DAB development mode — per-user, with files under `/Workspace/Users/<you>/.bundle/...` and jobs named `[dev <username>] …`. `--target shared` writes to `/Workspace/Shared/.bundle/demo/...` so every SA on the workspace sees the same deployed bundle, and the absolute paths in the "Interactive notebook use" section resolve. `dev` works fine for solo iteration; use `shared` for a team workspace.

What this does in one shot:

1. Builds two wheels via `uv build --wheel`:
   - `databricks_impulse-*.whl` from the impulse repo root
   - `byod_demo-*.whl` from `demos/byod/`
2. Uploads both to `/Workspace/Shared/.bundle/demo/artifacts/.internal/`.
3. Registers two jobs:
   - `adas — bootstrap ADAS tables + ingest metadata`
   - `adas — end-to-end (ingest → detect → per-event detail → visualize)`
4. Syncs notebooks (`demos/byod/*.py`), `lib/`, `adapters/`, and this `notebook_env.yml` to `/Workspace/Shared/.bundle/demo/files/`.

No `databricks fs cp` step. No manual wheel handling.

**Schemas and volumes are not bundle-managed.** They are created on first run by the first cell of `01_ingest.py` via `CREATE SCHEMA IF NOT EXISTS` + `CREATE VOLUME IF NOT EXISTS`. Schemas: `<schema_prefix>_silver`, `<schema_prefix>_perception_silver`, `<schema_prefix>_gold`. Volumes (under `<schema_prefix>_silver`): `raw`, `camera_frames`, `lidar_scans`, `openlabel_packages`. This separation is intentional: bundle-managed UC resources are subject to destructive Terraform recreates on state disturbance, which would nuke staged data.

---

## First run — bootstrap, then stage, then run_all_job

The bundle does not download datasets (registration-gated for most, signed-URL-expiring for NuScenes), and the schemas + volumes are not bundle-managed either. The first cell of `01_ingest.py` provisions them, so the order is:

```bash
# 1. Provision schemas, volumes, and empty ADAS tables, then write metadata.
databricks bundle run bootstrap_job --target shared --profile <your-workspace-profile>

# 2. Stage the dataset into the raw volume. NuScenes mini (or whichever variant the adapter expects):
databricks fs cp -r \
  /path/to/local/v1.0-mini \
  /Volumes/<catalog>/demo_silver/raw/v1.0-mini \
  --profile <your-workspace-profile>

# 3. Run the full pipeline.
databricks bundle run run_all_job --target shared --profile <your-workspace-profile>
```

`bootstrap_job` runs `01_ingest.py` with `bootstrap_only=true`: provisions the three schemas + four volumes (idempotent `CREATE … IF NOT EXISTS`), creates the empty ADAS tables, writes metadata via the selected adapter, then exits. Re-runnable idempotently as a recovery / table-reset step.

`run_all_job` runs four notebooks in sequence — `01_ingest` → `02_detect_events` → `03_per_event_detail` → `04_visualize` — recreating the silver + perception_silver + gold tables on every run. (`01_ingest.py` repeats the `CREATE … IF NOT EXISTS` cells so `run_all_job` is self-sufficient even on a workspace where `bootstrap_job` was never run — but the dataset still has to be staged before the scalars-ingest cell of `01_ingest.py` reads it.)

The adapter's `config.yaml` controls `dataroot_template`, which determines the expected layout under `<raw>/<dataset_version>/`.

Every notebook ends with an `assert`-based acceptance cell, so a SUCCESS at the job level means the notebook actually wrote what it claims. A regression that silently writes zero rows fails the task on the spot rather than reporting green. If you ever see a green job with downstream emptiness, that's a missing assertion — file it.

---

## Verify the run

Quick SQL spot-checks after `run_all_job` completes. Run via DBSQL or `databricks sql query --warehouse-id <id>`. Schemas resolve from `--var "catalog=<your-catalog>"` plus the default `schema_prefix=demo`.

The block below assumes the defaults (`adapter=nuscenes`, `schema_prefix=demo`). If you overrode either at deploy time, substitute the table prefix accordingly — Impulse persists event tables under `<adapter>_demo_` in the gold schema (`impulse_reporting`'s `UnitySink` requires a `table_prefix`).

```sql
-- t01_ingest cross-table invariant (the channel_metrics index event search inner-joins on)
SELECT
  (SELECT COUNT(*) FROM demo_silver.channel_metrics)                     AS metrics_rows,
  (SELECT COUNT(*) FROM (SELECT DISTINCT container_id, channel_id
                         FROM demo_silver.channels))                     AS distinct_pairs;
-- expect:  metrics_rows == distinct_pairs

-- t02_detect_events — event instances per event definition
SELECT d.event_name, COUNT(*) AS n
FROM   demo_gold.nuscenes_demo_event_instance_fact f
JOIN   demo_gold.nuscenes_demo_event_dimension   d USING (event_id)
GROUP BY d.event_name ORDER BY d.event_name;
-- expect for NuScenes v1.0-mini:
--   pedestrian_high_speed_proximity = 9   (windows: one per pedestrian-close-at-speed match)
--   pedestrian_high_speed_combined  = 9   (one row per window)
--   pedestrian_high_speed_per_object = 42 (one row per triggering pedestrian)
-- Larger variants and other adapters yield more.

-- t03_per_event_detail — one OpenLABEL JSON per detected window
LIST '/Volumes/<catalog>/demo_silver/openlabel_packages/pedestrian_high_speed_proximity/';
-- expect for NuScenes v1.0-mini: 9 .json files (= the proximity window count above).
```

Non-zero everywhere ⇒ the run produced what the demo claims. Zero with a green job ⇒ regression: a notebook's acceptance assert didn't cover the failure mode — add one.

---

## Interactive notebook use

When you open a deployed notebook in the workspace UI, the job's environment spec is **not** applied — interactive runs get the bare serverless default and the first cell fails with `ModuleNotFoundError: impulse_reporting`. To fix:

1. Open the notebook (e.g., `/Workspace/Shared/.bundle/demo/files/demos/byod/02_detect_events`).
2. Open the **Environment** side panel (top-right of the notebook editor).
3. **Base environment → Custom → file picker**.
4. Select `/Workspace/Shared/.bundle/demo/files/demos/byod/bundle/notebook_env.yml`.
5. **Apply**. The kernel restarts with the same deps the job uses; the first cell now succeeds.

> **Path assumes `--target shared`.** The absolute paths in steps 1 and 4 are only valid after a `--target shared` deploy. If you deployed with the `dev` default, both files live under `/Workspace/Users/${workspace.current_user.userName}/.bundle/...` and copy-pasting the paths above into the Environment side panel will fail silently (the file picker just won't find them). Re-deploy with `--target shared` per the note under "First-time deploy".

This is per-notebook, per-SA. Alternatively, a **workspace admin** can ⭐ `notebook_env.yml` in the Base Environments admin page → every new serverless notebook in the workspace defaults to it.

---

## Switching adapters

Set `--var "adapter=<name>"` at deploy time. The selected adapter's `config.yaml` controls `dataroot_template`, `dataset_versions`, OpenLABEL metadata, and visualization format. If a new adapter needs extra pip deps, add them to **both**:

- `demos/byod/bundle/resources/jobs.yml` (both jobs' `environments.serverless.dependencies` blocks)
- `demos/byod/bundle/notebook_env.yml` (the `dependencies:` list)

Keep these two files in lockstep — they're the same dep list in two shapes.

---

## Tear down

```bash
databricks bundle destroy --target shared --profile <your-workspace-profile>
```

Destroys the two jobs, the DAB-managed artifacts at `${workspace.root_path}/artifacts/.internal/`, and the workspace files at `${workspace.root_path}/files/`. **Schemas, volumes, and all staged data survive** — they are not bundle-managed. This is deliberate: `bundle destroy` is now safe to run as a cleanup of code/artifacts without losing the dataset.

If you want a full reset, run these by hand after `bundle destroy`:

```sql
DROP SCHEMA IF EXISTS <catalog>.demo_silver CASCADE;
DROP SCHEMA IF EXISTS <catalog>.demo_perception_silver CASCADE;
DROP SCHEMA IF EXISTS <catalog>.demo_gold CASCADE;
```

`DROP SCHEMA … CASCADE` removes every table and volume in the schema, including the staged dataset. Catalog-level state (the catalog itself, any IAM grants, etc.) is left intact.

---

## What's where

```
demos/byod/bundle/
├── databricks.yml          # bundle name, sync paths, artifacts block, targets
├── notebook_env.yml        # base environment for interactive serverless notebooks
├── README.md               # this file
└── resources/
    └── jobs.yml            # bootstrap_job + run_all_job, env deps via ${workspace.root_path}/artifacts/.internal/
```

Schemas and volumes are intentionally not in `resources/` — they're created by `01_ingest.py` so a destructive Terraform recreate can't propose to nuke them.

Notebooks, library code, and adapter packages live under `demos/byod/` at the repo root and are synced into the bundle by `sync.paths: [../..]` in `databricks.yml`.
