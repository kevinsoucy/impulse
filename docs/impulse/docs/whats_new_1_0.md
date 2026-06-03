---
sidebar_position: 3
title: What's New & Upgrading in 1.0
---

# What's New in Impulse 1.0

Impulse 1.0 adds **series**: support for time-series tables where
many things can exist at the same instant, each carrying a wide row of typed
columns. ADAS object detections, defect inspections, lap events, diagnostic
trouble codes — all fit this shape.

The driving motivation was **ADAS scenario search**: finding the moments in a
recording where a particular driving situation occurred — a cyclist cutting in
close, a specific sign present while the car was speeding — across perception
data (camera, lidar, radar detections) that 0.x's scalar channels couldn't
represent. Series make that data queryable, and the rest of this page is the
generalization that fell out of it.

**Everything from 0.x still works exactly as before — 1.0 is purely additive.**
It opens a new data shape without changing anything you already do. The
[Upgrading from 0.x](#upgrading-from-0x) section below is the short answer to
"is it safe to bump the version?" (yes).

---

## In one sentence

In 0.x you could query **scalar channels** — one number per signal per moment.
In 1.0 you can also query **series** — your own tables, of any shape — and
combine the two in a single expression.

A series doesn't require multiple rows per instant: a wide, single-row table
like an IMU — several values per instant, some non-numeric — is a series with no
`entity_key`.

---

## What you can do now that you couldn't before

**1. Query data where many entities share a timestamp.**
A 0.x channel holds one value per `(signal, time)`. A camera frame holds twenty
objects at once. 1.0 lets you register that table as-is and query it.

**2. Query string and multi-column data.**
0.x predicates were numeric only. Series columns can be strings too, with
`==`, `isin`, `contains`, `startswith`, `endswith`, and regex `matches`.

**3. Ask *which one* — not just *whether*.**
A 0.x event answers "did this ever happen in this recording?" Add `.ids(as_=…)` to a
predicate and the event also records *which* object / unit / driver triggered it
(use `.each()` for one window per entity).

**4. Correlate two independent entities in time.**
"A cyclist was close *while* a car decelerated sharply" — two different objects,
overlapping in time. 0.x had no way to express this; 1.0 does, by composing two
finalized predicates with `&`.

**5. Query across several time-series tables at once.**
Mix channels and multiple series in one expression with `&` and `|`. The engine
brings them together per recording in a single step. For example, an ADAS
scenario search combining a signs series, the speed channel, and lidar detections:

```python
sign  = db.query.series("traffic_signs")     # a series table
lidar = db.query.series("lidar_detections")  # another series table
speed = db.query.signal("Vehicle Speed Sensor")  # a scalar channel

# Speeding past a 30 sign with a cyclist close on lidar — all true at once.
scenario = (sign.value == "30") & (speed > 30) & (
    (lidar.object_class == "cyclist") & (lidar.distance_m < 8.0)
).any()
```

**Is there a limit on how many tables?** No fixed limit — add as many series as
your question needs; the engine resolves them in **one pass per recording** (table
count adds no stages). See
[How a cross-series query runs](references/query_engine.mdx#how-a-cross-series-query-runs)
for the map-reduce picture. The only real limit is how much data one recording
holds, so compact dense detection data upstream (see
[Best practices](user_guide_1_0.md#best-practices)).

**6. Bring your own table — no reshaping.**
Register a table in about five lines by mapping your column names to the engine's
roles. You keep your schema. You can even run end-to-end with **no channels at
all** — the recording is still the unit of analysis, but the channels table can
be empty.

---

## What stayed the same

| 0.x capability | Status in 1.0 |
|---|---|
| Scalar channels and the channels table | Unchanged |
| `query.channel(...)` | Unchanged — **not** deprecated |
| `BasicEvent`, `ContainerEvent`, `SequenceOfEvents` | Unchanged |
| Aggregations (`.mean()`, histograms, statistics) | Unchanged (channels only) |
| `DeltaSolver` / `KeyValueStoreSolver` | Unchanged |
| Existing reports and their output | Unchanged |

A scalar channel is just the simplest kind of series — one value column, no
entities — so nothing had to be rebuilt. Series sit alongside channels.

---

## Upgrading from 0.x

**Short version: it's safe. Upgrade without changing any of your code.** Your
channel queries, your events, and your reports run unchanged and produce the same
results. There are **no renamed APIs** to chase and **no deprecation warnings** to
fix — everything in the table above behaves exactly as it did in 0.x.

### Will anything behave differently if I just bump the version?

Your **query results are identical.** There is one correctness fix in 1.0
(interval union with a fully-contained interval), but it lives on a code path
that channel data never reaches — channel intervals always have non-decreasing
ends, so the old bug could not occur for channel queries.

The **one thing that changes is the shape of the event output table**, and it
changes in a way that doesn't break existing logic:

1. **A new `entity_key` column** appears on the event fact table
   (`event_instance_fact`). For all of your existing events it is always
   `NULL`. It only carries a value when an event projects entity identity with
   `.ids()` / `.each()`.
2. **`event_instance_id` is now a `bigint`** (it was an `int`). The values were
   always produced as 64-bit anyway; the declared type now matches.

Impulse evolves your existing table automatically on the first 1.0 write — **both**
persist paths handle it: the `replaceWhere` write (changed event definitions) and
the `MERGE` write (unchanged / incremental definitions, the steady-state path).
There is no manual migration: existing rows read back with `entity_key = NULL`,
and the widened column is reconciled before the write so existing values are
preserved (they were always 64-bit, so nothing truncates).

**Incremental and repeated runs are safe.** The event instance IDs that match a
row to its previous version on re-run are computed exactly as in 0.x for all
existing events — the new `entity_key` is folded in only when an event projects
entity identity (`.ids()` / `.each()`).
Re-running a 0.x report won't duplicate rows or break the merge.

### Upgrade checklist

- [ ] Bump the Impulse version.
- [ ] Run your existing reports — results match 0.x.
- [ ] If anything *downstream* of Impulse pins the `event_instance_fact` schema (a
      typed Spark view, a BI model), add the nullable `entity_key` column and
      declare `event_instance_id` as `bigint`. If your downstream reads the table
      as-is, there's nothing to do.
- [ ] That's it. When you're ready for the new capabilities, see the
      [User Guide](user_guide_1_0.md).

---

## Where to go next

- **[User Guide](user_guide_1_0.md)** — channels vs series, the metadata tables,
  querying multiple tables, what's required vs optional.
- **[Series reference](references/series.mdx)** — the deep dive: registration
  options, predicate operators, per-entity reporting, cross-entity correlation.
