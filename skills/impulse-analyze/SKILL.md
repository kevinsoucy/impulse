---
name: impulse-analyze
description: >
  Run sinkless ad-hoc Impulse analysis from business intent and return Spark or pandas results. Use
  for interactive statistics, distributions, derived signals, or quick per-container summaries.
  Reuse a source-adapter Report and its configured solver whenever available; use manual
  MeasurementDB construction only when no adapter exists.
---

# Impulse — ad-hoc analysis

Ad-hoc mode evaluates TSAL and returns one row per container. It does not persist results.

## Reuse the adapter report

```python
from impulse_reporting.sources import resolve_source

source = resolve_source()
report = source.create_report(
    spark,
    name="scratch",
    container_filters={"plant": ["Berlin"]},
    channels=["temperature", "vibration"],
    sink=None,
)
db = report.get_db()
solver = report.get_solver()

temperature = db.query.channel_with_alias(channel_alias="temperature")
vibration = db.query.channel_with_alias(channel_alias="vibration")
result = db.query.select(
    temperature.mean().alias("temperature_mean"),
    vibration.max().alias("vibration_max"),
).solve(spark, solver=solver)
```

For pandas, call `.toPandas(spark, solver=solver)` on the same query.

## Rules

- Use logical channels returned by `source.list_channels`; do not guess physical IDs.
- Use `channel_with_alias(...)` when the adapter exposes logical aliases.
- Always pass `report.get_solver()` to `solve()` or `toPandas()`.
- Never instantiate `DefaultSolver` over an adapter-created report.
- Do not reconstruct `MeasurementDB`, source tables, or solver config.
- Do not call `persist_results()` in ad-hoc mode.

## Advanced fallback

If no source adapter is registered, construct `MeasurementDB` from user-provided tables and create a
matching solver as described in `impulse-config`. This is a fallback, not the first path.

Use `impulse-tsal` for expressions and derived channels. Use `impulse-events` plus
`impulse-aggregations` when the request is scoped to event windows.
