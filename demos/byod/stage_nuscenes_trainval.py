# Databricks notebook source
# MAGIC %md
# MAGIC # Stage nuScenes v1.0-trainval into the demo volume (keyframes only)
# MAGIC
# MAGIC One-time staging for the full nuScenes dataset. The four pipeline notebooks
# MAGIC (`01`–`04`) are unchanged — this only populates the dataroot the loader reads.
# MAGIC
# MAGIC **Source:** the public, anonymously-readable S3 mirror `s3://motional-nuscenes/`.
# MAGIC No EULA URL juggling and no credentials — anonymous (unsigned) reads.
# MAGIC
# MAGIC **Keyframes only, by construction.** The mirror ships purpose-built
# MAGIC `v1.0-trainvalNN_keyframes.tgz` archives (~45 GB total for all 10) that contain
# MAGIC the annotated 2 Hz `samples/` and **no** `sweeps/`. So we never download the
# MAGIC ~300 GB of un-annotated in-between frames the demo doesn't read — we stream each
# MAGIC archive straight from S3 and extract it to the volume without ever storing the
# MAGIC tarball.
# MAGIC
# MAGIC Companion code changes that make this safe (already applied to the adapter):
# MAGIC `download.EXPECTED_SUBDIRS` no longer requires `sweeps/`, and
# MAGIC `ingest.perception_paths` indexes keyframe sample_data only.

# COMMAND ----------

import tarfile
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

# COMMAND ----------

dbutils.widgets.text("catalog",       "adas_engine_validate_catalog", "UC catalog")
dbutils.widgets.text("schema_prefix", "lakevision_demo",              "Schema prefix")
dbutils.widgets.text("version",       "v1.0-trainval",                "Dataset version")
dbutils.widgets.dropdown("with_map_expansion", "true", ["true", "false"], "Also stage map expansion v1.3")
dbutils.widgets.dropdown("keyframes", "true", ["true", "false"], "Stage keyframe sensor blobs")

_catalog = dbutils.widgets.get("catalog")
_schema_prefix = dbutils.widgets.get("schema_prefix")
_version = dbutils.widgets.get("version")
_with_map = dbutils.widgets.get("with_map_expansion") == "true"
_with_keyframes = dbutils.widgets.get("keyframes") == "true"

BUCKET = "motional-nuscenes"
PREFIX = "public/v1.0"

# The metadata tarball (v1.0-trainval/*.json + maps/) is all the ingest+detection
# path (notebooks 01–02) ever reads — scalars, object_tracks, and map_context are
# derived from annotations + ego pose, not from the sensor files. The 10 keyframe
# blob archives (~45 GB) are only needed by 03/04 (visualization). Set
# keyframes=false to stage metadata only (~462 MB) for a full-scale 01→02 run.
KEYS = [f"{PREFIX}/{_version}_meta.tgz"]
if _with_keyframes:
    KEYS += [f"{PREFIX}/{_version}{n:02d}_keyframes.tgz" for n in range(1, 11)]

# Ensure the silver schema + raw volume exist before writing into them (staging may
# run against a brand-new schema_prefix, e.g. demo_trainval).
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {_catalog}.{_schema_prefix}_silver")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {_catalog}.{_schema_prefix}_silver.raw")

_vroot = f"/Volumes/{_catalog}/{_schema_prefix}_silver"
DATAROOT = Path(f"{_vroot}/raw/{_version}")
DATAROOT.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
print(f"Dataroot: {DATAROOT}")
print(f"Archives to stream: {len(KEYS)}")

# COMMAND ----------

# MAGIC %md ## Stream each archive from S3 → extract to the volume (no tarball on disk)

# COMMAND ----------

for i, key in enumerate(KEYS, 1):
    size_gb = s3.head_object(Bucket=BUCKET, Key=key)["ContentLength"] / 1e9
    print(f"[{i}/{len(KEYS)}] {key}  (~{size_gb:.1f} GB) → streaming…")
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"]
    # mode="r|gz" is the streaming reader: sequential, no seek, nothing buffered to disk.
    with tarfile.open(fileobj=body, mode="r|gz") as tf:
        tf.extractall(DATAROOT, filter="data")
    print(f"    extracted → {DATAROOT}")

print("✓ all keyframe archives staged")

# COMMAND ----------

# MAGIC %md ## (Optional) Map expansion v1.3 — vectorized map layers for the map Series

# COMMAND ----------

if _with_map:
    import zipfile
    import io

    map_dir = Path(f"{_vroot}/raw/map-expansion-v1.3")
    map_dir.mkdir(parents=True, exist_ok=True)
    key = f"{PREFIX}/nuScenes-map-expansion-v1.3.zip"
    print(f"Streaming {key} → {map_dir}")
    blob = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()  # ~0.4 GB, small enough to buffer
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        zf.extractall(map_dir)
    print(f"✓ map expansion staged: {len(list(map_dir.glob('*.json')))} layer files")
else:
    print("skipped map expansion (with_map_expansion=false)")

# COMMAND ----------

# MAGIC %md ## Validate the extraction
# MAGIC
# MAGIC The metadata dir + `maps/` must always be present. `samples/` (keyframe sensor
# MAGIC files) is required only when `keyframes=true`; in meta-only mode it is
# MAGIC intentionally absent (notebooks 01–02 read only metadata). `sweeps/` is never
# MAGIC staged.

# COMMAND ----------

required = ["maps", _version] + (["samples"] if _with_keyframes else [])
for sub in required:
    exists = (DATAROOT / sub).is_dir()
    print(f"  {'✓' if exists else '✗'} {sub}/")
    assert exists, f"missing {sub}/ — staging incomplete"
if not _with_keyframes:
    print("  ⏭️  samples/ skipped (keyframes=false) — metadata-only run for notebooks 01–02")

n_meta = len(list((DATAROOT / _version).glob("*.json")))
n_scenes = len((DATAROOT / _version / "scene.json").read_text().split('"token"')) - 1
print(f"✓ staged {_version}: {n_meta} metadata tables, ~{n_scenes} scenes; sweeps/ skipped by design")
