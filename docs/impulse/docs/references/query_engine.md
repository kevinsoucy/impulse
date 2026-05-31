---
sidebar_position: 2
title: Query Engine
---

# Query Engine

The query engine resolves channel selections and evaluates events and
aggregations against silver-layer data. It does this through a *solver*:
the component that knows how your silver tables are physically laid out
and how to read them. Two solvers ship with Impulse — `DeltaSolver` and
`KeyValueStoreSolver` — and the right one depends on which silver tables
you have.

Both solvers read tag tables in the same narrow EAV layout `(key, value)`
and pivot them on the fly; the practical difference is *which* silver
tables each one consumes.

## Available solvers

**`DeltaSolver`** consumes the full default silver-layer schema —
`container_metrics`, `container_tags`, `channel_metrics`, `channel_tags`,
and `channels` — and uses `channel_tags` for channel selection (pivoting
its narrow EAV rows on the fly). It is the right pick whenever your data
matches the [default silver-layer
shape](../data_model/silver_layer_schema.md) and you don't need channel
aliasing.

**`KeyValueStoreSolver`** does not consume `channel_tags` at all —
channel selection runs directly against columns of `channel_metrics`, so
attributes such as `channel_name` must live as columns on
`channel_metrics`. It needs only three silver tables —
`container_metrics`, `channel_metrics`, and `channels` — and supports two
optional add-ons: a narrow EAV `container_tags` table for tag-based
container filtering, and a `channel_mapping` table to resolve channel
aliases (logical names that map to physical channels). It is the right
pick when you don't have a `channel_tags` table, or when container
attributes are already wide on `container_metrics` itself.

## Table requirements

| Silver table        | `DeltaSolver`          | `KeyValueStoreSolver`                              |
|---------------------|------------------------|----------------------------------------------------|
| `container_metrics` | required               | required                                           |
| `channel_metrics`   | required               | required (also carries channel selection columns)  |
| `channels`          | required               | required                                           |
| `container_tags`    | required (narrow EAV)  | optional (narrow EAV)                              |
| `channel_tags`      | required (narrow EAV)  | not used                                           |
| `channel_mapping`   | not used               | optional (channel aliases)                         |

See the [Silver Layer Schema](../data_model/silver_layer_schema.md) for
the columns each table is expected to carry.

## Which solver should I use?

- Do you have all five silver tables in the default shape?
  → **`DeltaSolver`**.
- Otherwise → **`KeyValueStoreSolver`**.

`KeyValueStoreSolver` also covers the wide-only case where container
attributes live directly on `container_metrics` and no `container_tags`
table exists; just omit `source.container_tags_table` from your config.
See the
[`query_engine.solver` field](../config/configuration.md#query_engine-optional)
for details.

`KeyValueStoreSolver` is the default — if `query_engine` is omitted from
your config entirely, the engine runs with `KeyValueStoreSolver` and
`data_type = "RLE"`.

## Time-axis alignment (precondition for cross-series queries)

When a query spans more than one series — for example correlating
`object_tracks` with `traffic_signs`, or a registered series with channels —
the engine compares their timestamps **directly, as raw integers**. It never
resamples a series to a common frequency and never converts between time units
at query time. This keeps evaluation at native resolution and avoids hidden
interpolation, but it carries one hard precondition:

> **Every series in a query must share one time axis: the same unit and the
> same epoch (zero point).** This includes the container's `stop_ts` in
> `container_metrics`, which the engine uses to close the final frame of a
> point-in-time series at session end.

The unit itself is unconstrained — microseconds, milliseconds, nanoseconds, or
any other integer base — as long as it is *identical* across every series and
the container metrics in the query. The engine cannot detect a mismatch: two
series in different units are both just `LongType` columns, so a misaligned
series produces silently wrong intervals rather than an error. Getting the axes
onto a common base is an **upstream responsibility**, done once at ingest /
silver before the series is registered, not a runtime conversion.

### Keeping dense series tractable

The same upstream prep step is also where you make dense series (e.g. per-frame
ADAS object detections — hundreds of objects per frame at 10+ Hz) scale. A query
evaluates one session's rows for a series together, so a session that is dense
enough to not fit in memory will fail. Shape the data at ingest with three
levers that compound:

- **Run-length encoding (RLE)** — store one row per `[tstart, tend)` segment
  during which a value holds, rather than one row per sample. For a value that
  changes rarely this is a 100–1000× row reduction.
- **Value quantization** — round continuous payloads (e.g. `distance_m`) to the
  coarsest precision the query actually needs. Adjacent frames then share a
  value, so RLE runs coalesce into longer segments — quantize too finely and you
  defeat RLE.
- **Downsampling** — drop the sample rate where the question tolerates it (a
  "close cyclist" window does not need 100 Hz). This is a linear row reduction.

These are data-model decisions, not engine settings: they belong in the
pipeline that produces the silver tables the series reads from.

## Configuring the solver

Solver selection and tuning live under the `query_engine` section of your
report config:

- [`query_engine.solver`](../config/configuration.md#query_engine-optional)
  — which solver to use.
- [Solver column mappings and filters](../config/configuration.md#solver-column-mappings-and-filters)
  — adapt either solver to a silver layer whose physical column names
  diverge from Impulse's internal names, scope reads by `project_id`, or
  apply per-table equality filters.

## API reference

Auto-generated symbol-level docs for each solver class:

- [`DeltaSolver`](api/impulse_query_engine/analyze/query/solvers/delta_solver.md)
- [`KeyValueStoreSolver`](api/impulse_query_engine/analyze/query/solvers/key_value_store_solver.md)
- [`QuerySolver`](api/impulse_query_engine/analyze/query/solvers/query_solver.md)
  — abstract base class defining the six-stage solver pipeline.
- [`SolverConfig`](api/impulse_query_engine/analyze/query/solvers/solver_config.md)
  — per-table column mappings, filters, and project scoping.
