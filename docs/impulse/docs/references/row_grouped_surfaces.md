---
sidebar_position: 7
title: Row-grouped surfaces
---

# Row-grouped surfaces

Impulse models measurement recordings as **scalar time-series channels** —
one numeric value per `(channel, interval)`, RLE-encoded, with the query
language evaluating predicates against `SampleSeries`. That model handles
CAN signals, ECU parameters, and analog sensor traces cleanly.

It does not handle a second data shape that shows up across the same
customers: **per-frame or per-event tables** where many entities can
co-exist at the same timestamp and each entity carries a wide row of
typed attributes. ADAS object detections, industrial defect inspections,
motorsport lap events, ECU diagnostic trouble codes — all share this
shape.

**Row-grouped surfaces** add first-class support for that second shape.
A team registers their own table once, in about five lines, and gets the
full predicate authoring DSL, composition with existing channel
predicates via `&` / `|`, and optional per-entity reporting that names
which object triggered each matching window.

---

## When to use one

A row-grouped surface is the right tool when **multiple rows can exist
for the same `(container_id, timestamp)`**. If your data doesn't have
that shape, channels remain the right answer:

| Your data shape | What to use |
|-----------------|-------------|
| One scalar value per timestamp per signal | Scalar channels (one channel per signal) |
| Wide row, one row per timestamp, no per-entity multiplicity | N parallel scalar channels — one per column |
| Multiple rows per `(container_id, timestamp)`, each row is one detected entity | Row-grouped surface |
| Pre-aggregated intervals (one row per `[tstart, tend)`) | Load directly as `Intervals` |
| Lookup / dimension tables (no timestamp) | Join target, not a query surface |

Tables that fit the row-grouped shape typically also share these
correlates: each entity carries a wide row of mixed types; entity
lifetime is transient (a detected object exists for a few frames and is
gone, a defect fires and clears); and the schema varies across customers
and use cases.

---

## Registering a surface

A surface is a declarative pointer to a Spark table plus the columns
Impulse needs to evaluate predicates against it. ADAS object detections
make a good example:

```python
from pyspark.sql.types import (
    StructType, StructField, LongType, DoubleType,
)
from impulse_query_engine.surfaces import RowGroupedSurface

OBJECT_TRACKS = RowGroupedSurface(
    name="object_tracks",
    schema=StructType([
        StructField("container_id",         LongType(),   nullable=False),
        StructField("frame_ts",             LongType(),   nullable=False),
        StructField("object_id",            LongType(),   nullable=False),
        StructField("detection_class",      LongType()),
        StructField("distance_m",           DoubleType()),
        StructField("relative_velocity_ms", DoubleType()),
    ]),
    timestamp_col="frame_ts",
    group_col="object_id",
)

db.register_surface(
    OBJECT_TRACKS,
    source_factory=lambda spark: spark.table("adas_demo.silver.object_tracks"),
)
```

The required pieces are:

- **`name`** — a unique identifier used to address this surface in
  the query language and in the fact-table output.
- **`schema`** — the Spark schema of the underlying table. Every
  registered surface must include a `container_id` column (the same
  partition key channels use) and a timestamp column.
- **`timestamp_col`** — names the column that carries each row's
  point-in-time sample. Impulse synthesizes intervals from these
  samples at evaluation time, closing each entity's last frame at
  the container's end timestamp.
- **`group_col`** — names the column that identifies a single
  entity (an object ID, a station, a driver, an ECU). When present,
  interval synthesis runs per entity rather than across the whole
  surface. Compound keys (`("station_id", "line_id")`) are also
  supported.

Once registered, the surface is addressable through an accessor whose
attributes match the schema's columns.

---

## Authoring predicates

Predicates on a row-grouped surface use the same Python operators as
predicates on channels. They return values composable with everything
else in the query language:

```python
from impulse_query_engine.surfaces import RowGroupedAccessor

ot = RowGroupedAccessor(OBJECT_TRACKS)

# A close cyclist — type 1 in this deployment's class catalog.
cyclist_close = (ot.detection_class == 1) & (ot.distance_m < 8.0)
```

A few things to notice:

- Each column access (`ot.distance_m`) reflects the schema and returns
  a typed proxy. Numeric columns support `<`, `<=`, `==`, `!=`, `>=`,
  `>`. Non-numeric columns raise a clear error naming the column and
  its Spark type.
- `&` and `|` on the same surface **compose per-row** — each row is
  checked against the combined predicate. A close pedestrian sharing a
  timestamp with a far cyclist does *not* match the close-cyclist
  predicate, because no single row satisfies both clauses.
- The result is a regular Impulse expression. It builds to `Intervals`,
  composes with channel predicates, and feeds into events without any
  special handling.

### Composing with channel predicates

The payoff is composition. A row-grouped predicate produces the same
`Intervals` shape a channel predicate does, so the two combine in one
expression:

```python
ego_fast = db.query.channel(channel_name="Vehicle Speed Sensor") > 30  # km/h

near_miss_at_speed = ego_fast & (ot.detection_class == 1) & (ot.distance_m < 8.0)
```

The result is the set of windows when ego speed exceeded 30 km/h and a
cyclist was within 8 meters. The author writes channel-side and
surface-side clauses side by side; the engine routes each leaf to its
own evaluation path and intersects the results in time.

---

## Per-entity reporting

The composition above answers "did this combination ever hold?" — a
recording-scoped answer. Many investigations need a finer-grained
answer: *which* cyclist, *which* station, *which* ECU.

`GroupedEvent` materializes that identity. It validates that every
row-grouped clause in its expression is per-entity, runs the predicate
per entity, and writes the matching entity's identity into a new
`group_value` column on the event fact table:

```python
from impulse_reporting.events.grouped_event import GroupedEvent

near_miss_per_object = GroupedEvent(
    name="cyclist_near_miss_per_object",
    expr=(ot(group_scope=True).detection_class == 1)
       & (ot(group_scope=True).distance_m < 8.0),
)
```

When two cyclists trigger the predicate during the same window of a
recording, the fact table carries two rows — one per cyclist — both
overlapping in time. A `BasicEvent` would have collapsed them to a
single recording-scoped row.

`group_value` is a string column. Single-column group identities
serialize as their decimal representation (`"47"`); compound keys
serialize as a JSON array (`"[10, 1]"`). Existing event types
(`BasicEvent`, `ContainerEvent`, `SequenceOfEvents`) populate the
column with `NULL`, so existing reports are unchanged.

---

## Cross-entity correlation

A different question shows up across the same domains: *did entity A
do X while entity B did Y in the same recording?* — a close cyclist
while an oncoming car decelerated sharply, a defect on Unit A while a
process anomaly fired on Unit B, an emissions DTC on the engine ECU
while a torque DTC fired on the transmission ECU.

This is *not* a per-row predicate. No single row is both a cyclist and
a car. The author wants the *time-overlap* of two independent
same-entity windows.

Express this by calling `.sub_event()` on each side. That signals "I'm
done composing this predicate per-row; treat it as a recording-scope
window from here":

```python
cyclist_close = (
    (ot.detection_class == 1) & (ot.distance_m < 8.0)
).sub_event()

car_decel_close = (
    (ot.detection_class == 2)
    & (ot.distance_m < 15.0)
    & (ot.relative_velocity_ms < -0.5)
).sub_event()

co_occurrence = cyclist_close & car_decel_close
```

Each `.sub_event()` finalizes its side as a recording-scope `Intervals`.
The outer `&` is interval intersection — the time-window overlap. The
result fires only when both conditions hold in the same recording,
regardless of which entities triggered each side.

When you're authoring a single predicate (no cross-entity correlation),
you don't need `.sub_event()`. `BasicEvent` and `GroupedEvent`
constructors finalize the predicate for you. The explicit call is only
necessary when the *outer* `&` is across two independently-authored
predicates.

---

## Where else this fits

The engine's role is identical across domains: schema reflection drives
the accessor, the per-container evaluation pipeline applies the
predicate, and the result composes with channel predicates. Only the
registration differs.

- **Industrial test cells — per-cycle defect inspection.** Multiple
  units flow through one production cycle. *"Cycles where unit A
  failed dimensional check while unit B on the adjacent line ran at
  high pressure"* — defect inspection records keyed by station, with
  channel composition against the line's pressure sensor.

- **Motorsport telemetry — per-lap event logs.** Multiple drivers in
  one race session. *"Laps where driver A took sector 3 within 0.2s
  of fastest while driver B was on a slow lap"* — lap events keyed
  by driver, with per-driver event materialization.

- **Fleet diagnostics — per-DTC occurrences.** Multiple ECUs reporting
  faults during one drive. *"Drives where the engine ECU raised an
  emissions DTC while the transmission ECU raised a torque-converter
  DTC within the same 30-second window"* — DTC records keyed by ECU,
  cross-entity correlation across ECUs.

The pattern repeats. Customers whose primary data is row-grouped (a
labeling team, a defect tracker, a fleet diagnostics archive) can use
Impulse end-to-end without channel data — the recording is still the
unit of partitioning, but the channels table can be empty.

---

## Technical implementation

A surface is a frozen dataclass holding a Spark schema, the timestamp
column, and an optional group column. Registration on the solver
stores it alongside a `source_factory` that produces the backing
`DataFrame` per solve.

Predicates are authored through `RowGroupedAccessor`. Each numeric
column on the accessor returns a typed proxy whose comparison operators
build a `_PartialPredicate` — a fusible per-row predicate against the
surface. Same-surface `&` and `|` between partials fuse the clauses
into a single row-level callable; cross-surface or surface-plus-channel
composition falls through to the standard interval algebra.

`.sub_event()` finalizes a partial as a `RowGroupedSelector`, which is
a regular `TimeSeriesExpression` leaf with `leaf_kind = surface.name`.
The channel-side filter pipeline only walks selectors with
`leaf_kind == "channel"`, so row-grouped leaves never reach the
channel-tag / channel-metric stages. Event constructors auto-finalize
partials, so the explicit call is only needed for cross-entity
correlation.

At evaluation time, each selector reads its container's rows from the
solver's cache, sorts by `(group_col, timestamp_col)`, computes the
next-sample timestamp per entity via `shift(-1)`, applies the
predicate, synthesizes `[timestamp, next_ts)` intervals for matching
rows, closes the last row of each entity with the recording's stop
timestamp, and merges overlaps. `GroupedEvent` runs the same
machinery once per entity and emits one fact row per matching window
with the entity's identity serialized into `group_value`.

Scalar channels are untouched. Channel-only queries continue on the
existing fast path. The surface registry is consulted only when a
query's selections include row-grouped leaves.
