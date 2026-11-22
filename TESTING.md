# Testing Guide

## Standard Test Flow

1. **Before you push**, run the fast tests: `./scripts/test-fast.sh` (~18s, no network). Or install
   the pre-push hook once (`cp scripts/hooks/pre-push .git/hooks/pre-push`) and it runs automatically
   — lint + type-check + fast tests — on every `git push`.
2. **Opening/updating a PR** triggers two GitHub Actions jobs automatically:
   - **`gate`** — lint, type-check, fast tests. Must pass. This is what blocks merge.
   - **`live-kestrel`** — the full test suite against the real Kestrel API. Informational only;
     it can fail without blocking your PR in the case Kestrel is unreachable.
3. **Merging to `main`** doesn't ship a release by itself. A bot called **release-please** reads your
   commit messages (`feat:`, `fix:`, etc.) and keeps a standing "Release PR" up to date with the next
   version number and changelog. Nothing is released until *that* PR gets merged — merging it bumps
   the version and publishes the GitHub Release. You don't need to do anything to trigger this; just
   use [Conventional Commits](https://www.conventionalcommits.org/) (`feat:`, `fix:`, `docs:`, `chore:`, etc.).

That's the whole loop. Everything below is reference material for when you need more control — running
a single test, understanding why a test is excluded from CI, benchmarking a KG build, etc.

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

| Context | Marker filter | Blocks merge? |
|---------|---------------|---------------|
| **CI — `gate`** (Python 3.10 & 3.12) | `not requires_api and not third_party and not performance` | ✅ yes |
| **CI — `live-kestrel`** (`continue-on-error`) | `not third_party and not performance` | no — informational |
| pre-push hook (local) | `not requires_api and not third_party and not performance` | yes, locally (`--no-verify` skips) |
| `./scripts/check.sh` (local) | `not third_party and not performance` | — |

See [CI/CD](#cicd-github-actions) for what each CI job does. Tier scripts are in the [Test scripts](#test-scripts) table above.

---

## CI/CD (GitHub Actions)

biomapper2 runs two jobs on every PR and push to `main` (`.github/workflows/ci.yml`):

- **`gate`** — ruff → black → pyright → fast offline tests on Python 3.10 & 3.12. **Blocks merge.** No external services, so a Kestrel outage can't red it.
- **`live-kestrel`** — the full correctness suite against the hosted Kestrel. **Informational** (`continue-on-error`): it never blocks an unrelated PR, but renders the test report (counts + run provenance) to the run's step summary and uploads `reports/*.json` as an artifact.

A separate **`kg-regression.yml`** (manual `workflow_dispatch`) runs the correctness suite against a candidate Kestrel/KG URL, tagged and stamped with that build's provenance. Run it before pointing production at a new KG build, then compare reports by `kg_version`.

The sibling repos guard the cross-repo build-metadata contract with their own CI: **KRAKEN** fails if `build_info.schema.json` is stale (drift guard), and **Kestrel** tests the `/health` build-info loader.

### Releases (`release-please.yml`)

Runs on every push to `main` (i.e. every merged PR) via [release-please](https://github.com/googleapis/release-please).
It doesn't run tests — it parses Conventional Commits since the last release and keeps a single
standing **"chore(main): release X.Y.Z"** PR up to date with the next version bump and a generated
`CHANGELOG.md`. `feat:` commits bump minor, `fix:` bumps patch (see `release-please-config.json`,
`release-type: python`). No release happens until that PR is merged: merging it tags the release,
publishes it on GitHub, and updates `pyproject.toml` + `.release-please-manifest.json`. If your commit
messages don't follow Conventional Commits, release-please simply won't pick them up for the changelog.

**When the version changes on prod, check that the API reports it.** `/api/v1/health` and the OpenAPI
document report the installed package's version
(`src/biomapper2/api/constants.py`), which comes from `pyproject.toml` as of the last `uv sync`. The
prod deploy (`deploy-api.yml`) syncs before restarting, so a merged release PR should show its new
version there. If you bump the version any other way, or deploy without syncing, health keeps
reporting the old release: after a release deploys, confirm `curl -s <api>/api/v1/health` shows the
new version. Do not hardcode a version string in the API again.

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
| `test_metabolomics_workbench.py` | 8 | `unit` / `integration + third_party` | Metabolomics Workbench annotator; most tests use mocks, 2 call the live API |
| `test_performance.py` | 5 | `performance + requires_api` | Per-step timing benchmarks (see below) |
| `test_provenance.py` | 4 | `unit` (+1 `integration + requires_api`) | `RunProvenance` / `KgBuildInfo`, `/health` fetch |
| `test_report_to_summary.py` | 4 | `unit` | CI report → markdown step-summary renderer |

---

## Performance tests

`test_performance.py` times each pipeline step in isolation. A `clear_kestrel_cache` autouse fixture deletes the HTTP cache at the start of the performance session so timings reflect real Kestrel latency, not SQLite reads. Tests are excluded from CI and `check.sh` — run via `./scripts/test-performance.sh`.

| Test | Dataset | Items | Scenario |
|------|---------|------:|---------|
| `test_step_timings_olink_proteins` | OLink protein metadata | 2,923 | All rows have UniProt IDs — annotation skipped |
| `test_step_timings_olink_proteins_name_only` | OLink (name column only) | 2,923 | No provided IDs — annotation runs hybrid-search on all rows |
| `test_step_timings_metabolites_synthetic` | Synthetic metabolites | 30 | Multi-vocab IDs (INCHIKEY, HMDB, KEGG, PUBCHEM, CHEBI) |
| `test_step_timings_milestone` | All milestone datasets | varies | Parametrized sweep over every milestone dataset; skips missing files |
| `test_normalizer_throughput_metabolites` | Synthetic metabolites | 30 | Normalizer only — no API calls |

Timings are written to `reports/{timestamp}_{tag}.json` (not committed) — they depend on
the KG build and network latency, so compare runs by `kg_version` rather than against
fixed numbers.

---

## CLI options

| Option | Default | Effect |
|--------|---------|--------|
| `--kestrel-url URL` | env `KESTREL_API_URL` | Override the Kestrel API endpoint for all tests in the session |
| `--tag LABEL` | `production` | Human-readable label for this run — used in the report filename and metadata alongside semantic versions |

Both are passed through by `test-performance.sh` via `"$@"`. Available to all tests via session-scoped fixtures (`kestrel_url`, `tag`).

---

## Test reports

Every test session writes `reports/{timestamp}_{tag}.json` (gitignored):

```jsonc
{
  "metadata": {
    "biomapper2_version": "0.1.0",       // from importlib.metadata
    "kestrel_version": "0.2.0",          // from Kestrel /health ("unknown" offline)
    "kestrel_url": "https://...",
    "run_timestamp": "2026-06-16T14:47:19+00:00",
    "kg_build": {                         // from /health → KRAKEN build_info.json (see kraken/build_info.schema.json)
      "kg_version": "2026.06.0",
      "kraken_package_version": "0.1.0",
      "biolink_version": "4.2.5",
      "build_timestamp": "2026-06-15T00:00:00+00:00",
      "git_commit": "abc123",
      "sources": ["kg2", "umls"],
      "kg_label": "kraken-no-spoke"
    },
    "git_commit": "0c50cb8...",          // biomapper2 commit SHA
    "tag": "production"                   // --tag value
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

When running offline (`test-fast.sh`), `kestrel_version` is `"unknown"` and `kg_build` fields are all `"unknown"` / `[]` / `null` — the block is always present with the full key set, never `{}`.

---

## Analyst run provenance

Every `map_dataset_to_kg` run stamps a `run_provenance` block into its
`*_a_summary_stats.json` so the result file is self-describing and reproducible:

```jsonc
"run_provenance": {
  "biomapper2_version": "0.1.0",
  "kestrel_version": "0.4.1",
  "kestrel_url": "https://kestrel.nathanpricelab.com/api",
  "run_timestamp": "2026-06-16T20:20:48.923707+00:00",
  "kg_build": {                          // from /health → KRAKEN build_info.json
    "kg_version": "2026.06.0",
    "kraken_package_version": "0.3.2",
    "biolink_version": "4.2.5",
    "build_timestamp": "2026-06-14T03:11:52+00:00",
    "git_commit": "a1b9f3c",             // exact KRAKEN commit → reproducibility
    "sources": ["kg2", "spoke", "umls", "lipidmaps", "refmet"],
    "kg_label": "kraken-2026.06.0",      // optional enrichments ↓
    "node_count": 9214877,
    "edge_count": 41663902
  }
}
```

`kg_label`, `node_count`, and `edge_count` are optional — present only when the KG build
emits them. When Kestrel is unreachable or predates the build-info feature, every value
degrades to `"unknown"` / `[]` / `null` rather than being omitted.

| Field | Source |
|-------|--------|
| `biomapper2_version` | `importlib.metadata` |
| `kestrel_version` | Kestrel `/health` |
| `kg_build` | Kestrel `/health` → KRAKEN `build_info.json` (see `kraken/build_info.schema.json`) |
| `kestrel_url`, `run_timestamp` | the run |

The same `RunProvenance` model (`src/biomapper2/provenance.py`) backs both the analyst
stats block and the pytest test report. The `/health` fetch degrades gracefully to
`"unknown"` if Kestrel is unreachable, so a stats file always records provenance
(or explicitly that it was unavailable) rather than omitting it.

---

## Conftest fixtures (session-scoped)

| Fixture | autouse | What it does |
|---------|---------|-------------|
| `shared_mapper` | no | Single `Mapper` instance shared across all tests; respects `--kestrel-url` |
| `test_run_metadata` | yes | Fetches versions from Kestrel `/health`, captures git commit + timestamp; triggers report write at session end |
| `kestrel_url` | no | Reads `--kestrel-url` CLI option |
| `tag` | no | Reads `--tag` CLI option (report filename label) |
| `clear_kestrel_cache` | yes | (`test_performance.py` only) Deletes HTTP cache before **each** benchmark so tests don't warm the cache for each other |
