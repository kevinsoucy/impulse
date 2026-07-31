---
name: impulse
description: >
  Route business-language measurement and telemetry requests through Impulse. Use when a user asks
  what recordings or signals exist, selects business dimensions, requests time-series statistics,
  events, histograms, derived signals, reports, or mentions Impulse/TSAL without knowing the source
  configuration. Prefer a registered source adapter and return native Impulse analysis.
---

# Impulse — intent-first entry point

Translate the user's intent into a configured native Impulse `Report`, then route the analysis to
the specialist skill. Do not ask the user for physical tables, solver mappings, or signal IDs when a
source adapter is installed.

## Start with source discovery

```python
from impulse_reporting.sources import registered_sources, resolve_source

print(registered_sources())
source = resolve_source()  # configured default, or the only registered source
dimensions = source.list_dimensions(spark)
```

If several sources are registered and none is default, show their names and ask the user to select
one. Never guess an import path or scan source files. Importing the deployment's adapter package is
environment setup, not a recipe to infer from customer data.

## Build the analysis context

1. Show the adapter's dimensions.
2. For relevant dimensions, call `source.list_dimension_values(...)`; never invent values.
3. Ask for a selection when the user has not provided one.
4. Call `source.list_channels(..., container_filters=selection)`; never guess physical names.
5. Create a sinkless report unless persistence was explicitly requested and a destination exists.

```python
scope = {"plant": ["Berlin"], "line": ["L1", "L2"]}
available = source.list_channels(spark, container_filters=scope)
report = source.create_report(
    spark,
    name="analysis",
    container_filters=scope,
    channels=["temperature", "vibration"],
    sink=None,
)
db = report.get_db()
solver = report.get_solver()
temperature = db.query.channel_with_alias(channel_alias="temperature")
```

Use `source.resolve_channel_mappings(...)` only for diagnostics. Keep logical-to-physical mapping
inside the adapter and native alias resolution.

## Route the requested analysis

| Intent | Skill |
|---|---|
| Select/derive signals or define conditions | `impulse-tsal` |
| Define event windows | `impulse-events` |
| Histograms or event-scoped statistics | `impulse-aggregations` |
| Interactive DataFrame, no writes | `impulse-analyze` |
| Persist a report to an approved destination | `impulse-reporting` |
| Understand adapter/table responsibilities | `impulse-data-model` |
| No adapter is installed and tables are known | `impulse-config` |

## Non-negotiable rules

- Prefer the configured source adapter.
- Reuse its `Report`, `MeasurementDB`, and solver.
- Never replace its solver with `DefaultSolver(spark)`.
- Never reconstruct source tables or solver configuration from physical data.
- Never guess dimension values or physical channel names.
- Remain sinkless unless persistence is explicit and `sink` is provided.
- After setup, use ordinary TSAL, events, aggregations, ad-hoc solving, and reporting.

Use manual `MeasurementDB` or report configuration only as an advanced fallback when no adapter is
registered and the user supplies the source contract.
