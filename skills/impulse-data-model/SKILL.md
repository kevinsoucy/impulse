---
name: impulse-data-model
description: >
  Explain how a deployment's physical telemetry model maps to Impulse containers, business
  dimensions, logical channels, solvers, and the standard Report. Use for source-adapter design,
  silver input contracts, logical-to-physical channel identity, RAW/RLE formats, or the gold output
  model. Keep customer layouts inside adapters rather than generic skills.
---

# Impulse — data model and adapter boundary

## Responsibility split

| Layer | Responsibility |
|---|---|
| Source adapter | Discover dimensions/values/channels, validate selection, construct `Report` |
| Solver | Read physical tables and reshape them into Impulse's internal model |
| Native Impulse | TSAL, events, aggregations, ad-hoc solving, reporting |
| Skills | Route user intent through the public APIs |

Keep domain concepts such as plant, project, machine, vehicle, or fleet as dimension data. Do not add
domain-specific methods to the generic contract.

## Logical and physical channel identity

Expose stable logical names to users and resolve them through `channel_with_alias(...)`. Preserve an
unambiguous physical channel identity in `channels`, `channel_metrics`, and `channel_tags`. Put
ordered fallbacks in `channel_mapping` with priority; the solver selects the first available physical
channel independently per container.

Never teach a generic skill customer table names, paths, or signal identifiers.

## Standard silver model

The manual/default-solver path uses:

| Table | Purpose |
|---|---|
| `container_metrics` | One row per container/recording |
| `channel_metrics` | One row per `(container_id, channel_id)` |
| `channels` | RAW points or RLE intervals |
| `container_tags` | Optional EAV business dimensions |
| `channel_tags` | Optional EAV channel metadata |
| `channel_mapping` | Optional logical alias → physical channel + priority |
| `unit_conversion` | Optional conversion factors |

RAW rows use `(container_id, channel_id, timestamp, value)`. RLE rows use
`(container_id, channel_id, tstart, tend, value)`. IDs must have consistent types and meanings across
all channel surfaces.

Use `SolverConfig` for column-name differences. Use a registered custom solver for structural reads
and reshaping. Neither choice belongs in agent skill prose when an adapter already encapsulates it.

## Gold output

An ordinary `Report` can emit event, histogram, 2D histogram, statistics, and measurement dimension
tables when a sink is configured. Adapter-created reports remain sinkless unless persistence is
explicitly requested with a destination.
