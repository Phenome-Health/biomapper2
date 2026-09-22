---
title: "An opt-in observability side channel must be provably inert: separate its calls, counters, and provenance from the pipeline it observes, and pass external rows through verbatim"
date: 2026-09-21
category: best-practices
module: biomapper2
problem_type: best_practice
component: service_object
related_components:
  - tooling
  - documentation
  - testing_framework
applies_when:
  - "adding an opt-in flag that exposes intermediate/raw data (e.g. kestrel_top_n) alongside a normal result"
  - "the pipeline being observed produces numbers that must stay reproducible (selection output, certificates, benchmark counters)"
  - "the side channel re-issues requests to a shared external client that has instrumentation, caches, or retry counters"
  - "returning raw external rows the caller is meant to treat as untrusted, verbatim evidence"
tags:
  - observability
  - side-channel
  - reproducibility
  - kestrel
  - passthrough
  - provenance
  - pydantic
severity: high
---

# An opt-in observability side channel must be provably inert to the pipeline it observes

## Context

PR [#87](https://github.com/trentleslie/biomapper2/pull/87) (`feat/kestrel-raw-passthrough`) added an opt-in `kestrel_top_n` option that returns the top-N raw rows Kestrel returned per search endpoint the mapping pipeline actually used, alongside the normal `EntityMappingResult`. The whole point of the feature is to be a **read-only side channel**: turning it on must not change `chosen_kg_id`, `assigned_ids`, or the resolution certificate, and must not contaminate the measured pipeline it observes.

The design got the hard part right — passthrough collection runs in a second pass *after* `disarm_batch_deadline()` so its latency never enters the armed RefMet window, and failures are classified into an enumerated error that never propagates. But adversarial review (Greptile, confidence 3/5) caught three places where the "inert" promise still leaked. All three are the same failure mode wearing different hats: **a side channel that shares state with the thing it measures is not inert.** This is the reusable lesson worth compounding.

## Guidance

When you add an opt-in channel that surfaces intermediate or raw data from a pipeline whose outputs must stay reproducible, treat "the flag changes nothing measurable" as a **testable invariant**, not a design intention. Enforce it along three axes:

1. **Verbatim fidelity for pass-through data.** If you promise callers the *raw* upstream rows, do not round-trip them through a strict typed model that adds/reorders/coerces fields. `KestrelRow(**row)` re-materializes every row: missing optional fields reappear as explicit `null`, Pydantic can coerce or reject known-field values, and serialization without `exclude_unset` changes the shape. Preserve the original dict (or serialize with `model_dump(exclude_unset=True)` / keep an untyped `extra="allow"` passthrough), and never let a validation rejection silently convert a whole endpoint's rows into empty-rows-plus-error.

2. **Record what actually happened, not what was planned.** The endpoint-provenance list must come from Kestrel calls that were *actually issued*, not from the selected-annotator list. An accepted empty/whitespace entity name makes annotators return early without issuing a request, yet their endpoints were still recorded — so with the flag on, the collector fires new empty-search requests and reports endpoints the pipeline never used. Capture provenance at the call site (after the request is issued), not at plan time.

3. **Isolate the side channel's footprint on shared clients.** The passthrough calls reused the normal `kestrel_request` instrumentation and the same endpoint keys as selection calls, so enabling the flag inflated the shared request/cache/retry/failure counters. Because benchmark manifests persist those counters, passthrough traffic became indistinguishable from mapping traffic — provenance and cache-rate comparisons would then depend on a response-only option. Route side-channel calls through a distinct instrumentation tag/counter (or bypass the shared counters entirely) so the observed metrics are invariant to whether anyone is observing.

## Why This Matters

The failure is insidious because the feature "works" in every functional test — you get rows back, selection is unchanged. The damage is to things you only notice later: benchmark counters drift, provenance lists lie, and "raw" rows aren't actually raw. In a project where the numbers back a preprint and reproducibility is the product, a side channel that quietly perturbs cache rates or request counts is worse than no side channel. The invariant "turning on observation does not change the observed system" is the entire contract; if it isn't tested, it isn't true.

Note also *what validation caught this*: not the unit tests (they mock all Kestrel access and assert selection parity), but adversarial code review reasoning about states the tests didn't construct — empty-name entities, upstream schema drift, and cross-run counter persistence. Design-by-invariant needs tests that adversarially try to *violate* the invariant, not just confirm the happy path.

## When to Apply

- Adding any opt-in flag that returns intermediate/raw pipeline data alongside the normal result
- The pipeline's outputs (selection, certificates, benchmark manifests, cache/request counters) must be reproducible and are compared across runs
- The side channel re-issues calls through a shared, instrumented external client
- You are handing callers "raw" or "verbatim" external data and typing it on the way out

## Examples

**Fidelity — don't rebuild rows through a strict model:**

```python
# Perturbs: missing optionals reappear as null, values coerced, one bad row
# nukes the whole endpoint into empty-rows+error.
rows = [KestrelRow(**row) for row in raw_rows]

# Inert: preserve the upstream dict verbatim (or exclude_unset on dump).
rows = list(raw_rows)  # or KestrelRow(**row).model_dump(exclude_unset=True)
```

**Provenance — record at the call site, not from the plan:**

```python
# Wrong: records endpoints for annotators that may have returned early
# (empty/whitespace name) without issuing a request.
recorded_endpoints = [a.endpoint for a in selected_annotators]

# Right: append only after the request is actually issued.
resp = await kestrel_request(endpoint, ...)
recorded_endpoints.append(endpoint)
```

**Isolation — separate the side channel's counter footprint:**

```python
# Wrong: passthrough reuses selection's instrumentation + endpoint keys,
# inflating shared request/cache/retry counters persisted in benchmark manifests.
await kestrel_request(endpoint, limit=n)

# Right: tag side-channel traffic so observed metrics stay invariant.
await kestrel_request(endpoint, limit=n, instrument_as="passthrough")
```

## Related

- [Measurement code that backs a published claim is production code](audit-instruments-backing-published-claims-2026-08-05.md) — same spirit (protect the numbers), different axis (backing a claim vs. not perturbing the pipeline)
- [Trustworthy gates invoke tests on real, shape-faithful fallbacks](trustworthy-gates-invoke-test-real-shape-faithful-fallbacks-2026-08-04.md) — adversarial validation over happy-path confirmation
- PR [#87](https://github.com/trentleslie/biomapper2/pull/87) — the Kestrel raw-passthrough side channel and the three Greptile findings
