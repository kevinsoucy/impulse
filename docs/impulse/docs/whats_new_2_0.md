---
sidebar_position: 3
title: What's New in 2.0
---

# What's New in Impulse 2.0

Impulse 2.0 adds **series**: first-class support for time-series tables where
many things can exist at the same instant, each carrying a wide row of typed
columns. ADAS object detections, defect inspections, lap events, diagnostic
trouble codes — all fit this shape.

The driving motivation was **ADAS scenario search**: finding the moments in a
recording where a particular driving situation occurred — a cyclist cutting in
close, a specific sign present while the car was speeding — across perception
data (camera, lidar, radar detections) that 1.0's scalar channels couldn't
represent. Series make that data queryable, and the rest of this page is the
generalization that fell out of it.

Everything from 1.0 still works exactly as before. 2.0 is purely additive: it
opens up a new data shape without changing anything you already do. (If you just
want to know whether upgrading is safe, see the [Upgrade Guide](upgrade_1_to_2.md).)

---

## In one sentence

In 1.0 you could query **scalar channels** — one number per signal per moment.
In 2.0 you can also query **series** — your own tables, of any shape — and
combine the two in a single expression.

---

## What you can do now that you couldn't before

**1. Query data where many entities share a timestamp.**
A 1.0 channel holds one value per `(signal, time)`. A camera frame holds twenty
objects at once. 2.0 lets you register that table as-is and query it.

**2. Query string and multi-column data.**
1.0 predicates were numeric only. Series columns can be strings too, with
`==`, `isin`, `contains`, `startswith`, `endswith`, and regex `matches`.

**3. Ask *which one* — not just *whether*.**
A 1.0 event answers "did this ever happen in this recording?" An `EntityEvent`
answers "*which* object / unit / driver triggered it?" and writes that identity
into the output.

**4. Correlate two independent entities in time.**
"A cyclist was close *while* a car decelerated sharply" — two different objects,
overlapping in time. 1.0 had no way to express this; 2.0 does, with
`.entity_condition()`.

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
).entity_condition()
```

**Is there a limit on how many tables?** No fixed limit. A realistic ADAS query
might span a camera table, a lidar table, a signs table, and a lanes table all at
once — add as many series as your question needs. The engine still resolves them
in a **single step** per recording (the number of tables doesn't add processing
stages). The practical limit is how much data one recording holds, not how many
tables you reference — which is why dense detection data should be compacted
upstream (see the [User Guide](user_guide_2_0.md), "Best practices").

**6. Bring your own table — no reshaping.**
Register a table in about five lines by mapping your column names to the engine's
roles. You keep your schema. You can even run end-to-end with **no channels at
all** — the recording is still the unit of analysis, but the channels table can
be empty.

---

## What stayed exactly the same

| 1.0 capability | Status in 2.0 |
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

## Where to go next

- **[Upgrade Guide](upgrade_1_to_2.md)** — is it safe to upgrade without
  touching my code? (Short answer: yes.)
- **[User Guide](user_guide_2_0.md)** — channels vs series, the metadata tables,
  querying multiple tables, what's required vs optional.
- **[Series reference](references/series.md)** — the deep dive: registration
  options, predicate operators, per-entity reporting, cross-entity correlation.
