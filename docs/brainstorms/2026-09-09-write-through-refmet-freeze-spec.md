---
title: "Write-through RefMet freeze (live-first + freeze-backup + write-back) — design spec"
status: draft (open decisions flagged for sign-off)
created: 2026-09-09
scope: production RefMet resolution only (small-molecule / biolink:SmallMolecule); benchmark freeze-first mode unchanged
---

# Write-through RefMet freeze — design spec

## Problem / goal
RefMet resolution calls a LIVE Metabolomics Workbench `/match` endpoint behind a process-global circuit
breaker. Under load the breaker trips and RefMet returns nothing, flipping the committed node to a wrong
value (the retinol CHEBI:12777↔132246 flip). The current fix is a **static, immutable, freeze-first**
snapshot — great for benchmark reproducibility, but corpus-bound and not prod-shaped (novel names get no
RefMet vote, and a static snapshot goes stale vs RefMet releases).

Goal (Trent's C design): a **live-first, self-healing** RefMet source for prod —
1. **Always try live `/match` first.**
2. **Live succeeds** → use it; if the result **differs** from the freeze, **write it back** (upsert).
3. **Live fails** (breaker open / timeout / transport) → serve the **freeze as backup** if present
   (`freeze_backup`); else `unavailable`.

This keeps full live coverage + freshness (no corpus bound), self-maintains vs RefMet releases, and still
kills the flip: a frozen name resolves correctly even when live is down (last-known-good backup). It is a
recognized pattern — **circuit-breaker-with-cache-fallback + stale-if-error + write-through**, where "the
cache is a safety net, not a performance optimization."

## Two modes (this is ADDITIVE, not a replacement)
- **Mode `frozen`** (current #70/#71 behavior): freeze-first, deterministic, breaker out of the path. For
  **benchmark reproducibility** (pure function of the pinned snapshot). Keep as-is.
- **Mode `live_backup`** (this spec): live-first + freeze-backup + write-back. For **prod** (coverage,
  freshness, outage-resilience). NOT run-to-run deterministic by design (which is fine for prod).
- **Mode `off`**: live only, no freeze (today's pre-#70 behavior).

Selected by config (D5). Mode `frozen` stays the benchmark tool; `live_backup` is prod.

## Control flow (mode `live_backup`)
```
resolve(name):
    try:
        data = live_match(name)              # breaker-open raises immediately (no call)
        if data is voted:  result = VOTED(refmet_id); src = live_api
        else:              result = NO_MATCH; src = live_api      # negative result is authoritative too
        if store.get(name) != result:        # "different" = absent OR changed
            store.upsert(name, result, updated_at=now)            # write-back (D6: includes NO_MATCH)
        return result, src
    except (CircuitBreakerError, timeout, transport):
        hit = store.get(name)
        if hit is not None:  return hit, src=freeze_backup        # last-known-good (D3: staleness)
        return UNAVAILABLE, src=unavailable
```

## Design decisions

### D1 — Store: SQLite (WAL) [recommend]
Move the mutable store to a **SQLite DB** (`refmet_store.sqlite`), WAL mode, one table
`refmet(name_norm PK, refmet_id, refmet_name, inchi_key, formula, exactmass, status, source, updated_at,
snapshot_version)`. Reads = one indexed `SELECT` (µs); writes = `INSERT ... ON CONFLICT DO UPDATE`.
Rationale: multi-process safe (prod = 2 uvicorn workers), atomic upserts, no compaction. Alternative
considered: append-only JSONL + in-memory index (crash-safe, but needs compaction + cross-worker reload).
The immutable-TSV freeze stays the artifact format for mode `frozen`; `live_backup` seeds a SQLite store
from it (D8).

### D2 — Concurrency [recommend: SQLite-direct, no long-lived in-memory copy]
Prod runs multiple worker PROCESSES. Read AND write go to SQLite directly (WAL + `busy_timeout≈2s`), so
there is no cross-worker in-memory staleness. Upserts are idempotent last-writer-wins; two workers writing
the same name write the same value. `PRAGMA journal_mode=WAL; synchronous=NORMAL`. (An in-memory read cache
would reintroduce cross-worker staleness — avoid, or bound it to a very short TTL.) **Open:** is
SELECT-per-resolution latency acceptable, or do we want a small per-worker LRU with a short TTL?

### D3 — Staleness of a backup value [OPEN — needs your call]
A `freeze_backup` value is served only when live is DOWN, and write-back keeps it fresh whenever live
answers. Options:
- **(a) unbounded** — always serve last-known-good regardless of age (stale > nothing during an outage);
  record `updated_at` + report age. [recommend]
- **(b) bounded** — a configurable `max_backup_age`; beyond it, refuse (`unavailable`) rather than serve
  ancient data.
Recommend (a) with age surfaced in provenance; add (b) as an optional bound if you want a hard ceiling.

### D4 — Provenance
`refmet_source` in this mode ∈ `live_api` | `freeze_backup` | `unavailable`. Add a run/counter metric:
`refmet_store_writebacks` (upserts this run), `refmet_backup_served` (times the backup answered). Keep the
`frozen`-mode values (`local_snapshot` / `not_in_snapshot`) for that mode.

### D5 — Mode selection / config [OPEN — naming]
`REFMET_FREEZE_MODE` = `off` | `frozen` | `live_backup` (default `off` → today's behavior). `frozen` uses
`REFMET_SNAPSHOT_PATH` (immutable TSV, deterministic). `live_backup` uses a store path
`REFMET_STORE_PATH` (SQLite, mutable), seeded from `REFMET_SNAPSHOT_PATH` if the store is empty.
**Open:** confirm the env names + that we keep all three modes (vs collapsing `off` into `frozen`-unset).

### D6 — Negative caching
Live `NO_MATCH` is a real, authoritative result → write it back (store `status=no_match`). On backup a
stored `no_match` deterministically returns NO_MATCH. Prevents re-hammering `/match` for known-absent names
and keeps backup behavior correct.

### D7 — "Different" definition
Write back when `store.get(name)` is **absent** OR its `(status, refmet_id)` **differs** from the live
result. No-op when identical (avoids write amplification).

### D8 — Seeding
On first start in `live_backup`, if `REFMET_STORE_PATH` is empty/absent, seed it from the existing static
freeze TSV (`REFMET_SNAPSHOT_PATH`) so the common corpus is pre-covered from day one; then it grows from
live traffic.

### D9 — Refresh
Mode `live_backup` is **self-maintaining**: live-first means RefMet releases are picked up automatically and
write-back propagates them. No periodic re-freeze job needed (unlike `frozen`, which needs one).

## Falsifiable acceptance gate (a no-op / net-negative is an acceptable STOP)
- **Positive control (flip killed via backup):** retinol seeded; simulate breaker-open → resolves
  `freeze_backup` → CHEBI:12777, NOT 132246.
- **Write-back:** a name whose live result differs from the seed → store is upserted (assert the row
  changed) with `source=live_api`.
- **Negative cache:** live `no_match` → stored `no_match`; on breaker-open → deterministic NO_MATCH backup.
- **Concurrency:** N threads/processes upserting the same name → no corruption, last-writer-wins, single row.
- **Mode isolation:** `off`/`frozen` behavior byte-identical to today (no regression); `live_backup` only
  active when configured.
- **Small-molecule scope:** gene/protein/disease paths never touch the store.

## Test plan (offline, monkeypatch/fixture; no live calls; ≤8/file)
- store unit: upsert/get/seed-from-TSV/WAL-concurrency (threads).
- annotator (`live_backup`): live-success→live_api(+writeback-if-different); live-fail+present→freeze_backup;
  live-fail+absent→unavailable; no_match writeback; breaker-open path.
- mode/config: `REFMET_FREEZE_MODE` selection; `off`/`frozen` unchanged.

## Scope / non-goals
- Small-molecule / RefMet only. Kestrel and other annotators untouched.
- Benchmark `frozen` mode unchanged (deterministic).
- Not a distributed store — per-deployment SQLite (prod = single box, 2 workers). A shared/remote store is
  future work if biomapper2 goes multi-host.

## Rollout
Fork PR on `trentleslie/biomapper2` (base `dev`) → Greptile → promote to org → deploy. Prod enables via
`REFMET_FREEZE_MODE=live_backup` + `REFMET_STORE_PATH` (seeded from the current freeze). `frozen` remains the
benchmark mode.

## Open decisions to sign off before implementation
1. **D3** staleness: (a) unbounded last-known-good [recommend] vs (b) configurable `max_backup_age`.
2. **D5** config: confirm `REFMET_FREEZE_MODE` {off,frozen,live_backup} + `REFMET_STORE_PATH`, and that we
   keep all three modes.
3. **D2** read path: SQLite-direct [recommend] vs a short-TTL per-worker read cache.
4. **D8** seed the store from the current freeze TSV on first run? [recommend yes]
