# biomapper2

![CI](https://github.com/Phenome-Health/biomapper2/actions/workflows/ci.yml/badge.svg)

This is a package for mapping **biomedical entities** to the KRAKEN knowledge graph, whether starting from text names or vocabulary/ontology IDs (local IDs or CURIEs).

It supports both **single-entity** lookups and **dataset-level** batch processing, and does:

1. **entity linking** (text name → CURIE)
2. **ID normalization** (messy local ID → CURIE)
3. **entity resolution** (CURIE → canonical CURIE, by leveraging the CURIE equivalencies in the KRAKEN knowledge graph)

All CURIEs are represented in [Biolink](https://github.com/biolink/biolink-model/tree/master/src/biolink_model/prefixmaps)-standard format.

⚠️ **Note**: This package is in active development. Feedback and issues welcome!

### Quick access: the hosted API

If you just want to **map entities against the hosted production API** without running your own server, you have two options. Neither needs an API key.

- **Over plain HTTP, from any language:** the API is served at `https://biomapper.expertintheloop.io/api/v1`. The [REST API tutorial notebook](examples/biomapper_api_tutorial.ipynb) walks through health and provenance pins, single and batch resolution across entity types, target namespaces, reading the resolution certificate, and batching safely.
- **From Python:** install the client package, `pip install biomapper` ([repository and README](https://github.com/Phenome-Health/biomapper), [tutorial notebook](https://github.com/Phenome-Health/biomapper/blob/main/notebooks/biomapper_tutorial.ipynb)). It wraps the same REST API.

(`ddharmon` on PyPI is a separate project, a data-dictionary harmonization tool; it is not a client for this API.)

## Setup

### Install uv (if not already installed)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other platforms, see [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

### Clone and install
```bash
git clone https://github.com/Phenome-Health/biomapper2.git
cd biomapper2
uv sync --dev
```

This will create a virtual environment and install all dependencies.

Then create a `.env` file from the template:
```bash
cp .env.example .env
```
Edit `.env` to fill in your `KESTREL_API_KEY`. The file also contains `KESTREL_API_URL` and optional API authentication settings — see the comments in `.env.example` for details.

Then [run the pytest suite](#run-tests) to confirm all is working.

## Usage

### Map a single entity to knowledge graph

```python
from biomapper2.mapper import Mapper

mapper = Mapper()

item = {
    'name': 'carnitine',
    'kegg': ['C00487'],
    'pubchem': '10917'
}

mapped_item = mapper.map_entity_to_kg(
    item=item,
    name_field='name',
    provided_id_fields=['kegg', 'pubchem'],
    entity_type='metabolite'
)
```

### Map a dataset to knowledge graph

```python
from biomapper2.mapper import Mapper

mapper = Mapper()

mapper.map_dataset_to_kg(
    dataset='data/examples/olink_protein_metadata.tsv',
    entity_type='protein',
    name_column='Assay',
    provided_id_columns=['UniProt'],
    array_delimiters=['_']
)
```
See `examples/` for complete working examples.

## REST API

biomapper2 includes a FastAPI server that exposes the mapping pipeline over HTTP.

### Run the server

```bash
# Local development (with hot reload)
uv run uvicorn biomapper2.api.main:app --reload --port 8001

# Or via Docker
docker compose --profile prod up -d
```

The hosted service is at `https://biomapper.expertintheloop.io/api/v1` (Swagger UI at `/api/v1/docs`). For a worked, executed walkthrough of these endpoints over plain HTTP, see the [REST API tutorial notebook](examples/biomapper_api_tutorial.ipynb).

For a local server, the API docs are available at:
- **Swagger UI**: http://localhost:8001/api/v1/docs
- **ReDoc**: http://localhost:8001/api/v1/redoc
- **OpenAPI spec**: http://localhost:8001/api/v1/openapi.json

### Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/v1/health` | Health check (no auth required) |
| GET | `/api/v1/entity-types` | List supported entity types |
| GET | `/api/v1/annotators` | List available annotators |
| GET | `/api/v1/vocabularies` | List supported vocabularies |
| POST | `/api/v1/map/entity` | Map a single entity |
| POST | `/api/v1/map/batch` | Map multiple entities (max 1000) |
| POST | `/api/v1/map/dataset` | Map an uploaded TSV/CSV file |
| POST | `/api/v1/map/dataset/stream` | Stream mapping results as NDJSON |

### Reading a mapping result

- **`chosen_kg_id`** is the answer: the knowledge-graph node the resolver committed. `kg_equivalent_ids`
  lists that node's identifiers by prefix.
- **`primary_curie`** is the one CURIE to report for the entity, and it always identifies the
  `chosen_kg_id` node: the first CURIE in `curies` that is the chosen node, links to it
  (`kg_ids[chosen_kg_id]`), or is among its `kg_equivalent_ids`; `chosen_kg_id` itself if none is.
  When nothing was chosen it is `curies[0]` (or null).
- **`curies`** is the alphabetically sorted union of every source's CURIEs, including hits for nodes
  that lost the vote. Do not read `curies[0]` as the answer.
- **Provenance:** `curies_provided` holds the CURIEs from identifiers you supplied, and
  `curies_assigned` maps each annotator slug to the CURIEs it produced (`assigned_ids` keeps the raw
  per-annotator local IDs and scores).
- **ICD-10-CM codes:** the knowledge graph has no `ICD10CM:` nodes yet, so a provided `ICD10CM:X` that
  misses is retried as `ICD10:X` (`icd10cm_route = twin`), and if that also misses, climbs to the most
  specific existing parent code in a pinned CDC FY2027 snapshot (`icd10cm_route = walkup:<parent>`,
  e.g. `E11.65` resolves via `walkup:E11`). Both are fallbacks used only when no other identifier on the
  row links. Read `mapping_relation`: `broad` means the node is less specific than the code you supplied
  (every walk-up hit, and twin hits until the twin gate passes). Walk-up rows also carry
  `icd10cm_snapshot_version`; fallback rows carry `kg_build`. `BIOMAPPER2_ICD10CM_FALLBACK_ENABLED=0`
  turns off both; `BIOMAPPER2_ICD10CM_WALKUP_ENABLED=0` turns off the walk-up only. Refresh the snapshot
  for a new fiscal year with `scripts/build_icd10cm_hierarchy.py` (pinned URL + SHA-256).

### Raw Kestrel passthrough (`kestrel_top_n`)

`kestrel_top_n` is an opt-in request option that returns the top-N **raw** rows Kestrel returned for
each search endpoint the pipeline actually used, alongside the normal result, so downstream consumers
can audit why a `chosen_kg_id` won or inspect near-misses without re-querying Kestrel.

**`candidate_limit` vs `kestrel_top_n` — they are not the same knob:**

| Option | What it controls | Can it change `chosen_kg_id`? |
|--------|------------------|-------------------------------|
| `candidate_limit` | The **selection window** — how many candidates each Kestrel search annotator retrieves and re-ranks. | **Yes.** Widening the window can change which node wins. |
| `kestrel_top_n` | **Passthrough only** — how many raw rows are returned in `kestrel_results`. | **No.** `chosen_kg_id`, `assigned_ids`, the resolver vote, and the certificate are byte-identical across any `kestrel_top_n` (including `None`). |

Details:

- Set on the request body for `/map/entity` and `/map/batch` (`options.kestrel_top_n`, 1..100), or as a
  query param on `/map/dataset/stream`. **`/map/dataset` (non-streaming) rejects it with a 422** — it
  returns a TSV path, not per-entity JSON, so use `/map/dataset/stream`.
- The response carries `kestrel_results`: one entry per used endpoint, each with the exact `request`
  params, the raw `rows` (Kestrel's order, truncated to N, **before** the hybrid `score>=0.5` filter and
  any re-ranking), `fetch_strategy` (`separate_call`), and a classified `error` (`timeout` |
  `upstream_error` | `malformed_response` | `other`) when that endpoint's passthrough call failed. A
  passthrough failure never turns a successful mapping into an error, and empty when the entity made no
  Kestrel call.
- **Rows are untrusted external data** (passed through verbatim, not biomapper-attested) — any renderer
  must treat every field as unescaped.
- **Payload cap:** the worst case is `len(entities) × kestrel_top_n × 3 endpoints` raw rows
  (e.g. `1000 × 100 × 3 = 300,000`). `/map/batch` and `/map/dataset/stream` hard-enforce a cap
  (`KESTREL_PASSTHROUGH_MAX_ROWS`, default 100,000 worst-case rows) and reject an over-cap request with
  a 422; reduce `kestrel_top_n` or the batch size, or page the request.

Example (single entity, 10 raw rows per used endpoint):

```bash
curl -X POST http://localhost:8001/api/v1/map/entity \
  -H "Content-Type: application/json" \
  -d '{"name": "glucose", "entity_type": "metabolite", "options": {"kestrel_top_n": 10}}'
```

### Authentication

Set `BIOMAPPER_API_KEY` or `BIOMAPPER2_API_KEYS` (comma-separated) in your `.env` file to require API key authentication via the `X-API-Key` header (a missing key returns 401, a wrong one 403; `/health` never requires a key). If no keys are configured, the API runs in open-access mode. The hosted service at `https://biomapper.expertintheloop.io/api/v1` currently runs in open-access mode.

## Docker

### Quick start

```bash
cp .env.example .env    # then edit .env with your KESTREL_API_KEY

docker compose --profile prod up -d
curl http://localhost:8001/api/v1/health
```

### Development

```bash
# Start with live code reload (mounts src/ and tests/)
docker compose --profile dev up

# Run tests inside the container
docker compose --profile dev run --rm biomapper2-dev uv run pytest -m "not integration" -v

# Run quality checks
docker compose --profile dev run --rm biomapper2-dev ./scripts/check.sh
```

### Building manually

```bash
docker build --target prod -t biomapper2 .       # Production image
docker build --target dev -t biomapper2:dev .     # Development image
```

### Generate KG-performance across datasets
```python

from biomapper2.visualizer import Visualizer

viz = Visualizer()

# collect metrics from jsons named {dataset}_{entity}_MAPPED_a_summary_stats.json
stats_df = viz.aggregate_stats(
    stats_dir='data/examples/synthetic_stats/'
)

viz.render_heatmap(
    df=stats_df,
    output_path='docs/assets/comparison_viz' # defaults to producing pdf and png, configurable via Visualizer(
)
```
<p align="center">
    <img src='docs/assets/comparison_viz.png' width="500">
</p>

## Run examples
```bash
uv run python examples/basic_entity_kg_mapping.py
uv run python examples/basic_dataset_kg_mapping.py

# Notebooks (the REST API tutorial needs only requests and pandas, both in this environment)
uv run --with jupyter jupyter lab examples/biomapper_api_tutorial.ipynb
```

## Run tests
```bash
uv run pytest          # Run all tests
uv run pytest -v       # Run with verbose output
uv run pytest -vs      # Run with verbose output and logging/prints displayed
```

**Note:** Tests run automatically on every commit via GitHub Actions (CI/CD).

**New contributor?** See [TESTING.md](TESTING.md) for a concise walkthrough of the test scripts and what happens on push/merge (CI gating, release-please), plus the full reference for test tiers and markers.

## Development

### Quick Start

Run all code quality checks before committing:
```bash
./scripts/check.sh     # Run ruff, black, pyright, and pytests
./scripts/fix.sh       # Auto-fix formatting and linting issues
```

**For detailed contribution guidelines, code style standards, and workflow practices, see [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md).**

## Project structure
```
src/biomapper2/
├── mapper.py                   # Main Mapper class - entry point for entity/dataset mapping
├── models.py                   # Pydantic Entity model for type-safe pipeline processing
├── config.py                   # Configuration (KG API endpoint, logging, etc.)
├── api/                        # FastAPI REST API
│   ├── main.py                 # Application setup, middleware, lifespan
│   ├── auth.py                 # API key authentication
│   ├── models/                 # Request/response Pydantic models
│   └── routes/                 # Endpoint implementations (mapping, discovery)
├── core/
│   ├── annotation_engine.py    # Orchestrates annotation of entities with ontology local IDs
│   ├── annotators/             # Individual annotator implementations (Kestrel text search, etc.)
│   │   ├── base.py             # Base annotator interface
│   │   ├── kestrel_text.py     # Kestrel text search annotator
│   │   ├── kestrel_vector.py   # Kestrel vector search annotator
│   │   ├── kestrel_hybrid.py   # Kestrel hybrid search annotator
│   │   ├── metabolomics_workbench.py  # RefMet match API annotator
│   │   └── goslin_lipid.py     # Lipid shorthand normalization via Goslin
│   ├── normalizer/             # ID normalization package
│   │   ├── normalizer.py       # Main Normalizer class
│   │   ├── validators.py       # ID validation functions for different vocabularies
│   │   ├── cleaners.py         # ID cleaning/standardization functions
│   │   └── vocab_config.py     # Biolink prefix mappings and validator configurations
│   ├── linker.py               # Links curies to knowledge graph nodes
│   └── resolver.py             # Resolves one-to-many entity→KG matches
├── utils.py                    # Utility functions
└── visualizer.py               # Visualize KG performance across datasets

Dockerfile                      # Multi-stage build (builder → dev → prod)
compose.yaml                    # Docker Compose with prod and dev profiles
examples/                       # Working code examples and notebooks (incl. the REST API tutorial)
tests/                          # Pytest test suite
data/                           # Example and groundtruth datasets
scripts/                        # Development scripts (check.sh, fix.sh)
```

### Configuration

Environment variables (set in `.env`):
- `KESTREL_API_URL` - Knowledge graph API endpoint. Defaults to the public Kestrel
  (`https://kestrel.krakenkg.com/api`). Point it at
  `https://kestrel.nathanpricelab.com/api` for the internal endpoint, which serves a
  different KRAKEN build. Pin the build from `GET $KESTREL_API_URL/health` (keyless), which reports
  `kg_version`, `kraken_package_version`, `biolink_version`, the build `git_commit` and the ingested
  `sources`; `/metagraph` alone does not identify the build.
- `KESTREL_API_KEY` - API key for the Kestrel API. A key issued by the public endpoint (prefix
  `kp_`) is sent to it; any other key is sent only to a non-public endpoint and is withheld from the
  public one. `GET $KESTREL_API_URL/health` reports `auth_required`

- `REFMET_FREEZE_MODE` - Where RefMet name matches come from: `off` (live Metabolomics Workbench
  `/match` only), `frozen` (a pinned freeze TSV at `REFMET_SNAPSHOT_PATH`, no network) or
  `live_backup` (live first, with a SQLite store at `REFMET_STORE_PATH` as backup). See
  [RefMet in `live_backup` mode](#refmet-in-live_backup-mode).

#### RefMet in `live_backup` mode

Live `/match` is asked first and its answer is written to the store at `REFMET_STORE_PATH`. Each
result says which source served it in `refmet_source`:

| `refmet_source` | Meaning |
|---|---|
| `live_api` | Live `/match` answered (a match or a no-match). |
| `freeze_backup` | Live was unreachable; the store's last-known value was served. |
| `kept_live_no_match` | Live answered no-match, but the store holds an earlier vote for this exact query. The stored vote was served and the store row was left alone. |
| `unavailable` | Live was unreachable and the store has no row for the query. |

`kept_live_no_match` exists because RefMet `/match` has stopped matching names it used to match
(for example `docosahexaenoate (dha; 22:6n3)`, around 2026-11-12) while the RefMet entries
themselves still exist. A kept vote has `refmet_availability` `voted`, raises no review flag, and is
counted under its own label in a batch's `refmet_source_counts`, so a run summary shows how many
votes were kept. A live match with a different id still replaces the stored vote. The stored vote is
kept until its store row is deleted or edited; it does not expire, and nothing checks that the id
still exists in RefMet. Live answers are cached over HTTP for 7 days, so a change upstream can take
up to a week to show.

Every change in what live says for a query is recorded once in the store. List the changes, back up
the store, or restore rows that an earlier release overwrote with `scripts/refmet_store_admin.py`,
run from the repository root as the user the API runs as:

```bash
uv run python scripts/refmet_store_admin.py history --limit 50
uv run python scripts/refmet_store_admin.py backup /path/to/refmet_store.backup.sqlite
uv run python scripts/refmet_store_admin.py repair --freeze /path/to/freeze.tsv --since 2026-11-12
uv run python scripts/refmet_store_admin.py repair --freeze /path/to/freeze.tsv --since 2026-11-12 --apply --expect-count 5
```

`repair` lists store rows that are a no-match where the freeze file has a vote for the same exact
query. It writes nothing without `--apply`, and `--apply` refuses to run unless the list has exactly
`--expect-count` rows, the number reviewed in the dry run. `backup` uses SQLite's online backup; do
not copy a store file the API is writing to with `cp`.

Additional settings in `src/biomapper2/config.py`:
- `BIOLINK_VERSION_DEFAULT` - Default Biolink model version
- `LOG_LEVEL` - Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)

## Future work

Known follow-ups, mostly around the cross-repo build-provenance chain (KRAKEN
`build_info.json` → Kestrel `/health` → biomapper2 `run_provenance`):

- **Cross-repo round-trip test.** Each repo tests its own leg of the provenance chain in
  isolation; the schema-drift guard in KRAKEN is the only thing tying them together. A
  committed golden `build_info.json` fixture, shared and asserted in all three repos,
  would catch a field that is spelled consistently in the schema but consumed wrongly.
- **Wire up the `kg_regression` marker.** It is registered (`pyproject.toml`) but unused —
  the `kg-regression.yml` workflow currently runs the whole correctness suite against a
  candidate KG. Marking a focused subset would let regression runs target just the tests
  that detect KG-version drift.
- **Populate `node_count` / `edge_count`.** These are reserved in
  `kraken/build_info.schema.json` and surface in `run_provenance`, but KRAKEN does not yet
  emit them (they serialize as `null`). Wiring them in would give analysts graph-scale at a
  glance.
- **Pre-1.0 release cadence.** release-please currently bumps the minor on every `feat:`
  (0.1.0 → 0.2.0 → …). If we'd rather keep features as patch bumps until a deliberate 1.0,
  set `bump-patch-for-minor-pre-major` (and `bump-minor-pre-major`) in each repo's
  `release-please-config.json`.
