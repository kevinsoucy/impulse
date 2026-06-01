# Backlog

Deferred ideas that are out of scope for the current work but worth keeping.
"Interesting but not now" lives here, not in the code.

## Series / query engine

- **Derive a `Series` schema from Unity Catalog metadata.** Today `register_series`
  takes a hand-written `StructType` plus explicit column roles. Auto-construct the
  schema (and infer roles where possible) from the registered UC table so callers
  don't restate what the catalog already knows. *(PR #4 review note on
  `series_selector_test.py`.)*

- **String / `sha2` `event_instance_id`.** `event_instance_id` is a 32-bit `crc32`,
  so distinct per-entity rows can collide in the same `(container, event, window)`.
  This is currently made safe by including `entity_key` in the event-fact merge key
  (shipped). A wider hash (or a string id) would make collisions effectively
  impossible and let the merge key shrink back. Keep the `entity_key` merge key
  regardless — it stays correct and cheap. *(PR #4 review note on `fact_schema.py`.)*

- **Signal source-prune covers only conjunctive constraints.** A signal filter
  pushes down to the source read when the signal is constrained by `==`/`isin`
  (incl. AND-ed with payload clauses). When the signal clause is OR-ed with a
  non-signal clause — e.g. `(sensor_type == "lidar") | (distance_m < 8)` — the
  constraint becomes unconstrained (a close row of any signal could match), so
  all signals are read (still correct, just not pruned). A smarter rewrite could
  push a per-branch filter; deferred until a real query needs it.

- **Per-entity scope for `SequenceOfEvents`.** `SequenceOfEvents` currently emits
  `entity_key = NULL`. A per-entity sequence (which *entity* completed the ordered
  sequence) is a natural extension of the `EntityEvent` model. *(PR #4 review note
  on `sequence_of_events.py`.)*

## Query engine API hygiene

- **`solve()` / `toPandas()` solver default.** The default is `BlobSolver()` — a
  mutable default argument and a solver that cannot run registered-series queries.
  Switching to `solver=None` + `solver = solver or BlobSolver()` removes the smell
  with zero behaviour change; deriving the default from a configured solver needs
  `MeasurementDB` to carry a solver concept first. Left unchanged to avoid a
  public-signature break. See the maintainer note in `query_builder.py`.
