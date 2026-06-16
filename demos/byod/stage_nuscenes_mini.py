# Databricks notebook source
# MAGIC %md
# MAGIC # Stage nuScenes v1.0-mini (full sensor data) + map expansion
# MAGIC
# MAGIC The mini split is ~4 GB and ships **camera + LiDAR + sweeps + metadata** in a
# MAGIC single tarball — everything `04_visualize` needs to render event overlays. Use
# MAGIC this to stand up a viz-capable deployment (e.g. `schema_prefix=demo_mini`)
# MAGIC alongside a metadata-only trainval deployment.
# MAGIC
# MAGIC Self-sufficient: creates the `<prefix>_silver` schema + `raw` volume, then
# MAGIC streams the data straight from the public S3 mirror (anonymous, no download to
# MAGIC the driver).

# COMMAND ----------

import io
import tarfile
import zipfile

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

# COMMAND ----------

dbutils.widgets.text("catalog",       "adas_engine_validate_catalog", "UC catalog")
dbutils.widgets.text("schema_prefix", "demo_mini",                    "Schema prefix")

_catalog = dbutils.widgets.get("catalog")
_schema_prefix = dbutils.widgets.get("schema_prefix")
_silver = f"{_catalog}.{_schema_prefix}_silver"

# COMMAND ----------

# MAGIC %md ## Ensure the silver schema + raw volume exist (staging writes into them)

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {_silver}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {_silver}.raw")
print(f"✓ {_silver}.raw ready")

_vroot = f"/Volumes/{_catalog}/{_schema_prefix}_silver"
DATAROOT = Path(f"{_vroot}/raw/v1.0-mini")
DATAROOT.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
BUCKET = "motional-nuscenes"

# COMMAND ----------

# MAGIC %md ## Stream v1.0-mini.tgz (full sensors) → extract to the volume

# COMMAND ----------

def _without_sweeps(tf: tarfile.TarFile):
    """Skip `sweeps/` members. mini.tgz bundles the un-annotated non-keyframe
    sweeps (the bulk of the file count); 01-04 read only keyframe `samples/`, and
    extracting tens of thousands of small sweep files over the volume FUSE mount is
    pathologically slow. Keep samples/ + maps/ + metadata only."""
    kept = skipped = 0
    for m in tf:
        if m.name.split("/", 1)[0] == "sweeps":
            skipped += 1
            continue
        kept += 1
        yield m
    print(f"    members: kept {kept}, skipped {skipped} sweeps")


key = "public/v1.0/v1.0-mini.tgz"
print(f"Streaming s3://{BUCKET}/{key} (~4.2 GB, sweeps skipped) → {DATAROOT}")
body = s3.get_object(Bucket=BUCKET, Key=key)["Body"]
with tarfile.open(fileobj=body, mode="r|gz") as tf:
    tf.extractall(DATAROOT, members=_without_sweeps(tf), filter="data")
print("✓ mini extracted (keyframe samples only)")

# COMMAND ----------

# MAGIC %md ## Map expansion v1.3 (for map_context)

# COMMAND ----------

map_dir = Path(f"{_vroot}/raw/map-expansion-v1.3")
map_dir.mkdir(parents=True, exist_ok=True)
blob = s3.get_object(Bucket=BUCKET, Key="public/v1.0/nuScenes-map-expansion-v1.3.zip")["Body"].read()
with zipfile.ZipFile(io.BytesIO(blob)) as zf:
    zf.extractall(map_dir)
print(f"✓ map expansion: {len(list((map_dir / 'expansion').glob('*.json')))} layers")

# COMMAND ----------

# MAGIC %md ## Validate

# COMMAND ----------

for sub in ("maps", "samples", "v1.0-mini"):
    ok = (DATAROOT / sub).is_dir()
    print(f"  {'✓' if ok else '✗'} {sub}/")
    assert ok, f"missing {sub}/ — staging incomplete"
assert (map_dir / "expansion").is_dir(), "map expansion missing"
print(f"✓ demo_mini staged at {DATAROOT} (full sensors) + map expansion")
