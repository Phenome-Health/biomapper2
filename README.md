# biomapper2

![CI](https://github.com/Phenome-Health/biomapper2/actions/workflows/ci.yml/badge.svg)

This is a package for mapping **biomedical entities** to the KRAKEN knowledge graph, whether starting from text names or vocabulary/ontology IDs (local IDs or CURIEs).

It supports both **single-entity** lookups and **dataset-level** batch processing, and does:

1. **entity linking** (text name → CURIE)
2. **ID normalization** (messy local ID → CURIE)
3. **entity resolution** (CURIE → canonical CURIE, by leveraging the CURIE equivalencies in the KRAKEN knowledge graph)

All CURIEs are represented in [Biolink](https://github.com/biolink/biolink-model/tree/master/src/biolink_model/prefixmaps)-standard format.

⚠️ **Note**: This package is in active development. Feedback and issues welcome!

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

Then just create a `.env` file with the proper secrets:
```bash
cd biomapper2
cp .env.example .env
```
And edit `.env` so that it has the actual secrets instead of placeholders.

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

### Generate KG-performance across datasets
```python

from biomapper.visualizer import Visualizer

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
```

## Run tests
```bash
uv run pytest          # Run all tests
uv run pytest -v       # Run with verbose output
uv run pytest -vs      # Run with verbose output and logging/prints displayed
```

**Note:** Tests run automatically on every commit via GitHub Actions (CI/CD).

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
├── config.py                   # Configuration (KG API endpoint, logging, etc.)
├── core/
│   ├── annotation_engine.py    # Orchestrates annotation of entities with ontology local IDs
│   ├── annotators/             # Individual annotator implementations (Kestrel text search, etc.)
│   │   ├── base.py             # Base annotator interface
│   │   └── kestrel_text.py     # Kestrel text search annotator
│   ├── normalizer/             # ID normalization package
│   │   ├── normalizer.py       # Main Normalizer class
│   │   ├── validators.py       # ID validation functions for different vocabularies
│   │   ├── cleaners.py         # ID cleaning/standardization functions
│   │   └── vocab_config.py     # Biolink prefix mappings and validator configurations
│   ├── linker.py               # Links curies to knowledge graph nodes
│   └── resolver.py             # Resolves one-to-many entity→KG matches
└── utils.py                    # Utility functions
└── visualizer.py               # Visualize KG performance across datasets

examples/                       # Working code examples
tests/                          # Pytest test suite
data/                           # Example and groundtruth datasets
scripts/                        # Development scripts (check.sh, fix.sh)
```

### Configuration

Edit `src/biomapper2/config.py` to customize:
- `KESTREL_API_URL` - Knowledge graph API endpoint (default: production server)
- `BIOLINK_VERSION_DEFAULT` - Default Biolink model version
- `LOG_LEVEL` - Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
