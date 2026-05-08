# Testing Guide

## Quick reference

```bash
./scripts/test-fast.sh          # Unit tests only — no API, ~18s
./scripts/test-full.sh          # Full correctness suite — needs Kestrel, ~45s (same as CI)
./scripts/test-performance.sh   # Cold-cache benchmarks — needs Kestrel, ~70s
./scripts/test-all.sh           # Everything: performance first, then full suite
```

Single file or test:
```bash
uv run pytest tests/test_normalizer.py
uv run pytest tests/test_entity_kg_mapping.py::test_map_entity_multiple_identifiers
```

Benchmark a specific KG build:
```bash
./scripts/test-performance.sh --kestrel-url https://staging.example.com/api --tag spoke-merged
./scripts/test-performance.sh --tag production
diff reports/*spoke-merged*.json reports/*production*.json
```

---

## Test scripts

| Script | Tier | Marker filter | When to use |
|--------|------|---------------|-------------|
| `test-fast.sh` | Fast | `not requires_api and not third_party and not performance` | Inner dev loop — pure unit tests, offline |
| `test-full.sh` | Full | `not third_party and not performance` | Before committing; mirrors CI exactly |
| `test-performance.sh` | Performance | `performance` | Benchmarking pipeline step latency; cold cache |
| `test-all.sh` | All | performance first, then `not performance` | Nightly or full pre-release sweep |

`test-performance.sh` passes through any extra args to pytest, so `--kestrel-url` and `--kg-version` work directly.

`test-all.sh` runs performance first so the HTTP cache is warmed for the correctness suite that follows — avoiding duplicate cold API calls.

---

## Markers

Markers are on two orthogonal axes: **pyramid level** (what the test exercises) and **infrastructure dependency** (what it requires to run).

### Pyramid levels

| Marker | Tier | Meaning |
|--------|------|---------|
| `unit` | A | Pure logic, no I/O — always fast, always deterministic |
| `component` | A | Single pipeline step, mocks external deps |
| `integration` | B/C | Multiple steps or real API calls |
| `e2e` | C | Full pipeline against a live KG |

### Infrastructure dependencies

| Marker | Meaning | Excluded from |
|--------|---------|---------------|
| `requires_api` | Needs live Kestrel (our infrastructure — has CI secret) | `test-fast` |
| `third_party` | Calls APIs we don't own or control (Metabolomics Workbench, etc.) | `test-fast`, `test-full`, CI |
| `slow` | Individually takes >10s | — (not filtered by tier scripts) |
| `performance` | Cold-cache timing benchmarks; clears HTTP cache before running | `test-fast`, `test-full`, CI |
| `kg_regression` | KG version change detection (not yet implemented) | — |

**Why `third_party` ≠ `requires_api`:** Both make network calls, but Kestrel is our infrastructure with a guaranteed API key in CI. Third-party services (Metabolomics Workbench, etc.) have independent uptime and rate limits we can't control, so they're excluded from automated runs.

### CI gating

| Context | Script / command | Excludes |
|---------|-----------------|---------|
| GitHub Actions (every PR) | `pytest -m "not third_party and not performance"` | Third-party APIs, benchmarks |
| `./scripts/check.sh` (local pre-commit) | same | same |
| `./scripts/test-fast.sh` | `pytest -m "not requires_api and not third_party and not performance"` | All API calls |
| `./scripts/test-full.sh` | `pytest -m "not third_party and not performance"` | Third-party, benchmarks |
| `./scripts/test-performance.sh` | `pytest -m performance` | Everything except benchmarks |
| `./scripts/test-all.sh` | performance, then `not performance` | Nothing |

---

## Test files

| File | Tests | Markers | Focus |
|------|------:|---------|-------|
| `test_normalizer.py` | 18 | `unit` | ID validation, CURIE formatting, vocab config |
| `test_batching.py` | 17 | `unit` | Kestrel API request batching and chunking |
| `test_visualizer.py` | 48 | `unit` | P/R/F1 heatmaps, scatter plots, breakdown rendering |
| `test_validators_kraken.py` | 13 | `unit` | KRAKEN harmonizer schema validation |
| `test_dataset_analysis.py` | 13 | `unit` | Summary stats, miss/unmapped calculations |
| `test_api_unit.py` | 12 | `unit` | API route logic (mocked) |
| `test_entity_model.py` | 12 | `unit` | `Entity` model fields, serialization |
| `test_equivalent_ids.py` | 8 | `unit` | `Linker.get_equivalent_ids()`, `kg_equivalent_ids` field |
| `test_api.py` | 16 | mixed `unit` / `integration + requires_api` | API auth, health, mapping endpoints |
| `test_entity_kg_mapping.py` | 10 | `integration + requires_api` | Single-entity pipeline: annotation → kg_equivalent_ids |
| `test_dataset_kg_mapping.py` | 7 | `integration + requires_api + slow` | Bulk dataset mapping, output file structure |
| `test_example_scripts.py` | 1 | `integration + requires_api + slow` | `examples/` scripts run end-to-end |
| `test_metabolomics_workbench.py` | 8 | `integration + third_party` | Metabolomics Workbench annotator (live third-party API) |
| `test_performance.py` | 4 | `performance + requires_api` | Per-step timing benchmarks (see below) |

---

## Performance tests

`test_performance.py` times each pipeline step in isolation. A `clear_kestrel_cache` autouse fixture deletes the HTTP cache at the start of the performance session so timings reflect real Kestrel latency, not SQLite reads. Tests are excluded from CI and `check.sh` — run via `./scripts/test-performance.sh`.

| Test | Dataset | Items | Scenario |
|------|---------|------:|---------|
| `test_step_timings_olink_proteins` | OLink protein metadata | 2,923 | All rows have UniProt IDs — annotation skipped |
| `test_step_timings_olink_proteins_name_only` | OLink (name column only) | 2,923 | No provided IDs — annotation runs hybrid-search on all rows |
| `test_step_timings_metabolites_synthetic` | Synthetic metabolites | 30 | Multi-vocab IDs (INCHIKEY, HMDB, KEGG, PUBCHEM, CHEBI) |
| `test_normalizer_throughput_metabolites` | Synthetic metabolites | 30 | Normalizer only — no API calls |

Observed cold-cache timings (2026-05-05, production KG):

| Step | Proteins w/ IDs | Proteins name-only | Metabolites (30) |
|------|-----------:|----------:|----------:|
| annotation | 3ms | **29,128ms** | 1ms |
| normalization | 180ms | 159ms | 3ms |
| linking | 744ms | 743ms | 158ms |
| resolution | 156ms | 141ms | 2ms |
| equivalent_ids | **1,555ms** | 1,050ms | 343ms |
| **TOTAL** | **2,639ms** | **31,220ms** | **506ms** |

---

## CLI options

| Option | Default | Effect |
|--------|---------|--------|
| `--kestrel-url URL` | env `KESTREL_API_URL` | Override the Kestrel API endpoint for all tests in the session |
| `--tag LABEL` | `production` | Human-readable label for this run — used in the report filename and metadata alongside semantic versions |

Both are passed through by `test-performance.sh` via `"$@"`. Available to all tests via session-scoped fixtures (`kestrel_url`, `tag`).

---

## Test reports

Every test session writes `reports/{timestamp}_{kg_version}.json` (gitignored):

```jsonc
{
  "metadata": {
    "biomapper2_version": "0.1.0",       // from pyproject.toml via importlib.metadata
    "kestrel_version": "0.1.0",          // from Kestrel /health
    "kg_build": {                         // stamped by KRAKEN at graph build time
      "kraken_version": "0.1.0",
      "build_timestamp": "2026-05-01T03:14:00Z",
      "build_sha": "abc123"
    },
    "kestrel_url": "https://...",
    "git_commit": "0c50cb8...",          // biomapper2 commit SHA
    "tag": "production",                  // --tag value (human label for the run)
    "timestamp": "2026-05-06T00:25:08Z"
  },
  "test_counts": { "passed": 4, "failed": 0, "error": 0, "skipped": 0 },
  "performance": {
    "olink_proteins": {
      "items": 2923,
      "total_ms": 2691.4,
      "steps": [
        { "step": "annotation", "duration_ms": 20.5, "ms_per_item": 0.007 },
        { "step": "normalization", "duration_ms": 159.8, "ms_per_item": 0.055 },
        { "step": "linking", "duration_ms": 760.8, "ms_per_item": 0.26 },
        { "step": "resolution", "duration_ms": 131.8, "ms_per_item": 0.045 },
        { "step": "equivalent_ids", "duration_ms": 1618.4, "ms_per_item": 0.554 }
      ]
    }
  }
}
```

`performance` is empty `{}` when no performance tests ran.

---

## Conftest fixtures (session-scoped)

| Fixture | autouse | What it does |
|---------|---------|-------------|
| `shared_mapper` | no | Single `Mapper` instance shared across all tests; respects `--kestrel-url` |
| `test_run_metadata` | yes | Fetches versions from Kestrel `/health`, captures git commit + timestamp; triggers report write at session end |
| `kestrel_url` | no | Reads `--kestrel-url` CLI option |
| `tag` | no | Reads `--tag` CLI option (report filename label) |
| `clear_kestrel_cache` | yes | (`test_performance.py` only) Deletes HTTP cache before **each** benchmark so tests don't warm the cache for each other |
