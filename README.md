# biomapper2

![Tests](https://github.com/arpanauts/biomapper2/actions/workflows/test.yml/badge.svg)

Unified toolkit for multiomics data harmonization, from entity linking and identifier standardization to knowledge graph mapping and cross-dataset alignment.

## Setup

### Install uv (if not already installed)

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other platforms, see [uv installation docs](https://docs.astral.sh/uv/getting-started/installation/).

### Clone and install

```bash
git clone https://github.com/arpanauts/biomapper2.git
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
    dataset_tsv_path='data/examples/olink_protein_metadata.tsv',
    entity_type='protein',
    name_column='Assay',
    provided_id_columns=['UniProt'],
    array_delimiters=['_']
)
```

See `examples/` for complete working examples.

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

## Project structure

```
src/biomapper2/
├── mapper.py           # Main Mapper class - entry point for entity/dataset mapping
├── config.py           # Configuration (KG API endpoint, logging, etc.)
├── core/
│   ├── annotator.py    # Annotates entities with ontology local IDs
│   ├── normalizer.py   # Normalizes local IDs to form Biolink-standard curies
│   ├── linker.py       # Links curies to knowledge graph nodes
│   └── resolver.py     # Resolves one-to-many entity-->KG matches
└── utils.py            # Utility functions

examples/               # Working code examples
tests/                  # Pytest test suite
data/                   # Example and groundtruth datasets
```

### Configuration

Edit `src/biomapper2/config.py` to customize:
- `KESTREL_API_URL` - Knowledge graph API endpoint (default: production server)
- `BIOLINK_VERSION_DEFAULT` - Default Biolink model version
- `LOG_LEVEL` - Logging verbosity (DEBUG, INFO, WARNING, ERROR, CRITICAL)
