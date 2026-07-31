---
name: impulse-config
description: >
  Configure an Impulse report or source adapter. Use when selecting a registered source, filtering
  containers by adapter dimensions, choosing RAW/RLE behavior, running sinkless, or—only when no
  adapter exists—manually supplying silver tables, solver mappings, incremental settings, and a
  persistence destination.
---

# Impulse — configuration

## Prefer the configured source

Let the adapter own source tables, physical mappings, solver choice, and report construction:

```python
from impulse_reporting.sources import resolve_source

source = resolve_source()
report = source.create_report(
    spark,
    name="analysis",
    container_filters={"plant": ["Berlin"]},
    channels=["temperature"],
    sink=None,
)
```

Discover dimensions, values, and logical channels before calling `create_report`. Do not reproduce
the adapter's config or replace its solver. `sink=None` is the default exploration posture.

## Manual fallback when no adapter is installed

Use a config dict only when the deployment has no adapter and the user provides the source tables:

```python
config = {
    "source": {
        "container_metrics_table": "catalog.silver.container_metrics",
        "channel_metrics_table": "catalog.silver.channel_metrics",
        "channels_uri": "catalog.silver.channels",
        "container_tags_table": "catalog.silver.container_tags",
        "channel_tags_table": "catalog.silver.channel_tags",
        "channel_mapping_table": "catalog.silver.channel_mapping",
    },
    "query_engine": {"solver": "DefaultSolver", "data_type": "RLE"},
}
```

Omit `unity_sink` for sinkless mode. Add it only for an explicit persistence request:

```python
config["unity_sink"] = {
    "catalog": "approved_catalog",
    "schema": "approved_schema",
    "table_prefix": "analysis",
}
```

## Native container filters

Filters are DNF: outer groups are OR; filters inside a group are AND. Use tag filters for adapter
business dimensions and metric filters for known internal metric columns.

```python
"container_filters": {
    "tag_filters": [
        [{"tag_name": "plant", "comparator": "==", "value": "Berlin"}],
        [{"tag_name": "plant", "comparator": "==", "value": "Dresden"}],
    ]
}
```

Never collect container IDs merely to recreate a business-dimension filter.

## Manual solver settings

- `data_type`: `RLE` for `[tstart, tend)` intervals; `RAW` for timestamp/value points.
- `solver_config`: physical-to-internal column mappings and solver-specific settings.
- `incremental`: scheduled reporting behavior.
- `measurement_dimensions`: post-mapping container columns written to gold.

Treat these as adapter implementation details when an adapter exists. Never infer them from table
names, and never guess a custom solver import.
