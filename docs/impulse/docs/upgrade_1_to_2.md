---
sidebar_position: 4
title: Upgrading from 1.0 to 2.0
---

# Upgrading from Impulse 1.0 to 2.0

**Short version: it's safe. Upgrade without changing any of your code.** Your
channel queries, your events, and your reports run unchanged and produce the same
results. 2.0 only *adds* things; it doesn't change how 1.0 works.

This page answers the two questions every 1.0 user asks: *do I need to change my
code?* and *will anything behave differently if I don't?*

---

## Do I need to change my code?

No. Everything 1.0 did, 2.0 does the same way.

| What you have | What you do |
|---|---|
| `query.channel(...)` calls | Nothing — same signature, same behavior, not deprecated |
| `BasicEvent`, `ContainerEvent`, `SequenceOfEvents` | Nothing — unchanged |
| Aggregations (`.mean()`, histograms, statistics) | Nothing — unchanged |
| Your report configs and solvers | Nothing — unchanged |

There are **no renamed APIs** to chase and **no deprecation warnings** to fix.

---

## Will anything behave differently if I just bump the version?

Your **query results are identical.** There is one correctness fix in 2.0
(interval union with a fully-contained interval), but it lives on a code path
that channel data never reaches — channel intervals always have non-decreasing
ends, so the old bug could not occur for channel queries. Nothing you get back
from a channel query changes.

The **one thing that changes is the shape of the event output table**, and it
changes in a way that doesn't break existing logic:

1. **A new `entity_key` column** appears on the event fact table
   (`event_instance_fact`). For all of your existing event types it is always
   `NULL`. It only carries a value for the new `EntityEvent` type.
2. **`event_instance_id` is now a `bigint`** (it was an `int`). The values were
   always produced as 64-bit anyway; the declared type now matches.

Impulse adds the column to your existing table automatically on the first 2.0
write. **Both** persist paths evolve the schema — the `replaceWhere` write (for
changed event definitions) and the `MERGE` write (for unchanged / incremental
definitions, the steady-state path). There is no manual migration: existing rows
read back with `entity_key = NULL`. And because `event_instance_id` was always
produced as a 64-bit value (Spark `crc32` yields `bigint`), the change is in the
*declared* schema and existing values fit without truncation.

**Incremental and repeated runs are safe.** Event instance IDs (the keys that
match a row to its previous version on re-run) are computed exactly as in 1.0 for
all existing event types — the new `entity_key` is folded in only for the new
`EntityEvent`. So re-running a 1.0 report won't duplicate rows or break the merge.

**The only thing to check:** if something *downstream* of Impulse reads
`event_instance_fact` with a hard-coded schema (a pinned Spark schema, a typed
view, a BI model), update it to allow the extra nullable `entity_key` column. If
your downstream reads the table as-is, there's nothing to do.

---

## Upgrade checklist

- [ ] Bump the Impulse version.
- [ ] Run your existing reports — results match 1.0.
- [ ] If anything downstream pins the `event_instance_fact` schema, add the
      nullable `entity_key` column (and declare `event_instance_id` as `bigint`
      if it was pinned to `int` — the data was always 64-bit).
- [ ] That's it. When you're ready to use the new capabilities, see the
      [User Guide](user_guide_2_0.md).

---

## When you're ready for the new features

Nothing forces you to adopt series. They're there when you have data that doesn't
fit the scalar-channel shape — tables where many entities share a timestamp, or
string/multi-column payloads. See **[What's New](whats_new_2_0.md)** for the
capabilities and the **[User Guide](user_guide_2_0.md)** to get started.
