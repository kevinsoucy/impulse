---
sidebar_position: 4
title: User Guide (v0.0.5)
---

# Impulse v0.0.5 User Guide

This guide is the map. It explains how Impulse v0.0.5 thinks about your data — the
difference between channels and series, how the metadata tables fit together,
how to query across several tables at once, and what's required versus nice to
have. For the deep dive on series specifically, see the
[Series reference](references/series.mdx).

---

## The core idea

Impulse turns a **question about time** into a **set of time intervals**.

You write a predicate — "vehicle speed over 30", "a cyclist within 8 meters" —
and the engine gives you back the windows when it was true. Predicates combine
with `&` (both true at once) and `|` (either true), and the answer is always
intervals of time, per recording.

Channels and series are just two kinds of data you write predicates against.

---

## Channels vs. series

A **channel** is one number that changes over time — engine RPM, vehicle speed,
a temperature. One value per moment. This is what Impulse 0.x handled, and it
still works the same.

A **series** is your own table where each row is a thing observed at a time, and
**many rows can share the same moment**. A camera frame holds twenty detected
objects; a production cycle holds several units; a drive holds faults from
several control units. Each row carries a wide set of typed columns.

Use this table to decide:

| Your data | Use |
|---|---|
| One number per signal per moment | **Channel** |
| One wide row per moment, no repeats | **Series** with no entity key |
| Many rows per moment, each a distinct thing | **Series** with an entity key |
| Already stored as `[start, end)` intervals | **Series** in run-length form (`tstart_col` / `tend_col`) |
| A lookup table with no timestamp | Not a query source — join it upstream |

An **entity key** is the column that says which thing a row is about — an object
ID, a station, a driver — so you can ask *which one* matched, not only that
something did. An entity key is scoped to its signal: object `47` seen by the
lidar and object `47` seen by the radar are two different things.

A channel is simply the smallest possible series — one value column, no
entities. Nothing in v0.0.5 is bolted on; series are the general case and channels
are the simple one.

Reach for channels when you have **many sensors each sampling on its own
timeline**: every channel carries its own timestamps and is resolved
independently, so they don't have to share a single clock the way series in one
query do (see [Best practices](#best-practices)).

---

## The data model: what's required, what's optional

Impulse reads a few tables. In v0.0.5, fewer of them are mandatory than you might
expect.

| Table | What it holds | Required? |
|---|---|---|
| **Recording (container) metadata** | One row per recording: start/stop time, vehicle, campaign, geography | **Always required.** It's how recordings are filtered and how point-in-time series close their last interval. |
| **Series tables** | Your own data, registered with `register_series(...)` | Required if you query series (one per series you use). |
| **Channels table** | The scalar measurement data | Optional. If you only use series, it can be empty. |
| **Signal/channel metadata + tags** | Tags and metrics that describe each signal | Optional. Only needed for *tag-addressed* channel queries (see below). |

The practical upshot: a team whose data is entirely series (a labeling team, a
defect tracker) needs only **recording metadata + their series tables**. No
channels, no channel-metadata tables.

---

## Three ways to name a signal

All three coexist in v0.0.5. None is deprecated — they're just different ways to
point at data.

```python
# 1. A registered series, by name — returns an accessor for its columns
ot = db.query.series("object_tracks")
close = ot.distance_m < 8.0

# 2. A scalar signal, by name
speed = db.query.signal("Vehicle Speed Sensor")

# 3. A channel, by its tags (the 0.x tag-addressed path) — unchanged
speed = db.query.channel(channel_name="Vehicle Speed Sensor")
```

`query.signal(name)` is just shorthand for `query.channel(channel_name=name)`.
`query.channel(**tags)` is the original tag-based lookup and is kept as-is — use
it when you select a signal by metadata rather than by name. Channels are **not**
registered as a series; the engine knows about them natively.

### Filtering by signal

A series' **signal column** (e.g. `object_tracks.sensor_type` ∈ `lidar` /
`radar` / `fusion`) is queryable like any other column — and it's a primary
filter, just as it was in 0.x:

```python
ot = db.query.series("object_tracks")

lidar_only = (ot.sensor_type == "lidar") & (ot.distance_m < 8.0)
lidar_or_fusion = ot.sensor_type.isin(["lidar", "fusion"]) & (ot.distance_m < 8.0)
```

A signal filter prunes at the **source read** — the engine never loads signals no
clause can match — and `isin` only widens which signals you consider; it never
merges an entity id across them (object `47` on lidar ≠ on fusion, as above).

---

## Querying across several tables at once

A series predicate produces the same time-intervals a channel predicate does, so
you write them side by side and combine them:

```python
ot   = db.query.series("object_tracks")
sign = db.query.series("traffic_signs")
ego  = db.query.signal("Vehicle Speed Sensor")

# A 30 km/h speed-limit sign, a close cyclist, and ego speed over 50 — all at once.
speed_30      = (sign.sign_class == "speed_30")
cyclist_close = ((ot.detection_class == "cyclist") & (ot.distance_m < 8.0)).any()
risky         = speed_30 & cyclist_close & (ego > 50)
```

You can mix any number of series with channels in one expression. The engine
figures out which tables a query touches and brings only those together, per
recording, in a single step — you don't manage joins or partitioning.

---

## Asking "which one": entity-aware events

A normal event answers "did this happen in this recording?" Entity awareness adds
finer answers — but only when you ask for them. It comes in tiers; **use only the
one your question needs**, and a report that never asks "which entity?" never meets
an entity concept.

**Presence — "did any match?"** Over a series with entities, finalize the predicate
with `.any()`. One window, no entity in the output. You author every event through
the one `BasicEvent(...)` constructor.

```python
from impulse_reporting.events.basic_event import BasicEvent

any_close = BasicEvent(
    name="cyclist_near",
    expr=((ot.detection_class == "cyclist") & (ot.distance_m < 8.0)).any(),
)
```

**Which ones — add `.ids()`.** Project the matched ids into an `entity_key` column,
under an alias you name. Same `BasicEvent(...)` call.

```python
close = BasicEvent(
    name="cyclist_near",
    expr=((ot.detection_class == "cyclist") & (ot.distance_m < 8.0)).any().ids(as_="cyclist"),
)
# entity_key → {"cyclist": {"fusion": ["47", "48"]}}
```

The identity lands as a small JSON map `{alias: {signal: [ids]}}`. Events without
`.ids()` leave the column `NULL`, so existing reports are unaffected.

**One window per entity — use `.each()` instead of `.any()`.** Two cyclists in the
same window then produce two rows, one per cyclist, instead of one merged row.

**Two different entities — correlate with `&`.** "A close cyclist while a car
braked": finalize each side and intersect them.

```python
cyclist = ((ot.detection_class == "cyclist") & (ot.distance_m < 8.0)).any().ids(as_="cyclist")
car = ((ot.detection_class == "car") & (ot.distance_m < 15.0)).any().ids(as_="car")
squeeze = BasicEvent(name="cyclist_and_car", expr=cyclist & car)
```

At most one side may use `.each()` (two would enumerate the entity cross-product).
**Pull the entity's payload** by joining the fact rows back to the series table on
the entity id. For the full tier-by-tier walkthrough and the registration and
operator details, see the [Series reference](references/series.mdx).

---

## Best practices

**Put every table on the same clock.** A cross-table query compares timestamps as
plain integers — no unit conversion, no resampling — so every series and the
recording metadata must share the **same unit and epoch**. A mismatch isn't caught
at runtime; it quietly gives wrong answers. Align once, upstream at ingest — the
single most important rule in v0.0.5.

**Keep dense series small at ingest.** A camera running at 10 Hz for 30 minutes
with 200 objects per frame is millions of rows per recording. Three upstream
levers keep that tractable, and they compound:

- **Run-length encode** — one row per "object present from t1 to t2", not one row
  per frame.
- **Quantize values** — round payloads (distance, speed) to the precision your
  question actually needs, so adjacent frames collapse into one interval.
- **Downsample** — drop the frame rate where the question tolerates it.

**Validate signals if typos would hurt.** `register_series(...)` takes an
optional `valid_signals` set. Pass it and registration fails fast on an unknown
signal value instead of silently dropping it. Leave it off and registration needs
no signal metadata at all.

```python
db.register_series(
    object_tracks,
    source,
    valid_signals={"lidar", "radar", "fusion"},  # a stray "lidat" fails registration
)
```

**Series are for filtering, not math.** Series predicates produce time intervals;
there is no `.mean()` or histogram on a series column (those stay channel-only).
To aggregate a series payload, compute it upstream and register the result as its
own series or channel.

---

## Where to go next

- **[Series reference](references/series.mdx)** — registration options, the full
  operator set, per-entity and cross-entity reporting, the technical details.
- **[Query Engine reference](references/query_engine.mdx)** — choosing a solver
  and the time-axis precondition in full.
- **[Getting Started](getting_started.md)** — run a report end-to-end in five
  minutes.
- **[What's New & Upgrading in v0.0.5](whats_new_v0_0_5.md)** — the new capabilities
  and the (safe) 0.x → v0.0.5 upgrade.
