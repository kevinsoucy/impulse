# Databricks notebook source
# MAGIC %md
# MAGIC # Stage nuScenes map-expansion v1.3 (vectorized map layers)
# MAGIC
# MAGIC Small (~0.4 GB), version-independent. Streams the map-expansion zip from the
# MAGIC public S3 mirror and extracts the `expansion/<location>.json` layer files to the
# MAGIC silver volume, where `BYODConfig.map_expansion_dir` expects them. Needed for the
# MAGIC `map_context` Series — works on any variant (mini or trainval).

# COMMAND ----------

import io
import zipfile

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path

# COMMAND ----------

dbutils.widgets.text("catalog",       "adas_engine_validate_catalog", "UC catalog")
dbutils.widgets.text("schema_prefix", "lakevision_demo",              "Schema prefix")

_catalog = dbutils.widgets.get("catalog")
_schema_prefix = dbutils.widgets.get("schema_prefix")

_vroot = f"/Volumes/{_catalog}/{_schema_prefix}_silver"
DEST = Path(f"{_vroot}/raw/map-expansion-v1.3")
DEST.mkdir(parents=True, exist_ok=True)

s3 = boto3.client("s3", config=Config(signature_version=UNSIGNED))
KEY = "public/v1.0/nuScenes-map-expansion-v1.3.zip"

# COMMAND ----------

print(f"Streaming s3://motional-nuscenes/{KEY} → {DEST}")
blob = s3.get_object(Bucket="motional-nuscenes", Key=KEY)["Body"].read()  # ~0.4 GB
with zipfile.ZipFile(io.BytesIO(blob)) as zf:
    zf.extractall(DEST)

layers = sorted((DEST / "expansion").glob("*.json"))
print(f"✓ staged {len(layers)} map layers under {DEST}/expansion:")
for p in layers:
    print(f"   {p.name}  ({p.stat().st_size / 1e6:.1f} MB)")

# COMMAND ----------

# MAGIC %md The `map_context` ingest (01_ingest Step 5b) reads from
# MAGIC `BYODConfig.map_expansion_dir` = `<vroot>/raw/map-expansion-v1.3/expansion`.

# COMMAND ----------

assert (DEST / "expansion").is_dir() and layers, "map-expansion layers not extracted"
print(f"✓ map_expansion_dir ready: {DEST}/expansion")
