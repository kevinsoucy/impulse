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

- **`TimeSeriesExpression.alias()` is mutate-and-return.** *(Risk: low — one
  caller relies on the side effect.)* `alias()` sets `self._alias` and returns
  `self`, so aliasing one expression more than once clobbers the earlier names —
  every "copy" is the same object. Consumers work around it by rebuilding a fresh
  expression per use (e.g. the ADAS demo's `risky_predicate()`: three rebuilds +
  a docstring explaining the trap). Fix: make `alias()` copy-on-write — return a
  shallow clone with `_alias` set, leaving `self` untouched. Use
  `clone = self.__class__.__new__(self.__class__); clone.__dict__.update(self.__dict__); clone._alias = alias_name`
  rather than `copy.copy` — the class's `__getattr__` (manufactures a
  `TimeSeriesOp` for any missing attr) is a landmine for the copy/pickle
  machinery; the manual clone sidesteps it, shares child operands by reference
  (correct — only the top-level `_alias` changes), and round-trips through the
  existing `__getstate__`/`__setstate__`. **Required companion fix:** the
  `TimeSeriesSelector` `from_dict` path (~line 708–711 in
  `time_series_expression.py`) discards the return and depends on the in-place
  mutation — change `m.alias(obj["alias"])` to `m = m.alias(obj["alias"])`. Every
  other caller already consumes the return value (`basic_event`,
  `sequence_of_events`, the histogram/stats aggregators), so they're transparent;
  re-grep `\.alias(` to confirm. Tests: identity (returns a new object),
  non-mutation of the source, two independent aliases from one base expr (the
  regression the demo dodges), state preserved on a `TimeSeriesOp` (not just a
  leaf), and a `from_dict`/pickle round-trip. Purely additive behavior; no public
  signature change.

- **`solve()` / `toPandas()` solver default.** The default is `BlobSolver()` — a
  mutable default argument and a solver that cannot run registered-series queries.
  Switching to `solver=None` + `solver = solver or BlobSolver()` removes the smell
  with zero behaviour change; deriving the default from a configured solver needs
  `MeasurementDB` to carry a solver concept first. Left unchanged to avoid a
  public-signature break. See the maintainer note in `query_builder.py`.

## Performance

- **Prune non-overlapping combinations in cross-entity correlation.** *(Risk:
  medium-high — silent fact correctness; large test surface.)*
  `EntityEvent._materialize_per_entity` (per_entity_windowing=True only) takes the
  full `itertools.product` over the **entity-scoped leaves** (one product axis per
  leaf, not per entity) and evaluates the expression for every combination. A
  single `entity_condition()` is already linear; the blowup only appears with
  **two or more** entity-scoped leaves (e.g. `cyclist_close & car_close`). Skipping
  combos whose per-leaf intervals don't overlap is correct **only** when the path
  from those leaves' lowest common ancestor to the root is an unbroken chain of
  `&` (intersection) — a top-level `&` is necessary but **not** sufficient. Any
  `|`, arithmetic, `.where`, aggregation, UDF, or presence/channel leaf on that
  path re-admits non-overlapping combos (an OR fires from one side alone), so
  pruning there would silently drop valid facts. Implementation must: detect the
  pure-AND spine over the specific leaf pair; reuse `Intervals.plane_sweep`
  overlap semantics (incl. zero-length/point-in-time edges); skip before building
  `entity_map`; and ship a randomized equivalence test proving the pruned output
  is row-for-row identical to the full product. The existing `any(len==0)` guard
  already covers the whole-leaf-empty case.

  *(Note: the `valid_signals` eager-scan and the `_eval_tree` dispatch-drift items
  that lived here are done — see the git history. The `_eval_tree` fix extracted a
  shared `TimeSeriesOp.apply_op` used by both `build` and the per-entity evaluator,
  rather than the cache-injection originally sketched: it removes the dispatch
  duplication that drifts while keeping the per-entity leaf substitution and
  channel-build memo byte-identical — lower risk.)*
