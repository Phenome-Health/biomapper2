# Visualizer Usage Guide

## Overview

The `Visualizer` is a helper class, used to visually compare KG-harmonization performance across multiple datasets. Currently, it produces two visualization types:

1. **Coverage heatmap** – a matrix showing mapping success rates across dataset/entity combinations
2. **Breakdown chart** – a grid of stacked bar charts showing the pipeline stages (total → valid IDs → mapped to KG)

---

## Instantiation

```python
from biomapper2.visualizer import Visualizer

# Default configuration
viz = Visualizer()

# With custom config - creates a 3 x 2 (entity x dataset) grid
viz = Visualizer(config={
    "row_order": ["proteins", "metabolites", "lipids"],
    "col_order": ["cohort1", "cohort2"],
}) 
```

### Key Configuration Options

| Option | Default | Description |
|--------|---------|-------------|
| `row_order` | `None` (alphabetical) | Entity display order (top to bottom) |
| `col_order` | `None` (alphabetical) | Dataset display order (left to right) |
| `entity_labels` | See code | Dict mapping internal names → display names |
| `dpi` | 300 | Output resolution |
| `output_formats` | `["pdf", "png"]` | Formats to save |

---

## Data Aggregation

Before rendering, aggregate your stats JSONs into a tidy DataFrame:

```python
df = viz.aggregate_stats("path/to/stats_dir/")
```

This expects files matching `*_MAPPED_a_summary_stats.json`, which should be created automatically when mapper is run via:

```python
mapper.map_dataset_to_kg()
```

## Rendering Methods

Both methods share a common signature pattern:

```python
fig = viz.render_*(
    df,                          # Required: the aggregated DataFrame
    output_path=None,            # Optional: saves to this path (no extension)
    title=None,                  # Optional: overrides default title
    dataset_labels=None,         # Optional: {"internal": "Display Name"}
    entity_labels=None,          # Optional: merged with config defaults
    figsize=None,                # Optional: (width, height) override
)
```

### Coverage Heatmap

```python
fig = viz.render_heatmap(
    df,
    output_path="figures/coverage_heatmap",
    title="Mapping Coverage by Dataset",
    dataset_labels={"cohort1": "My Custom Dataset"},
)
```

**Output**: Matrix where each cell shows the coverage percentage (bold) and raw counts. Missing combinations appear as "N/A" cells.

### Breakdown Chart

```python
fig = viz.render_breakdown(
    df,
    output_path="figures/pipeline_breakdown",
    title="Harmonization Breakdown",
)
```

**Output**: Grid of subplots. Each cell contains three bars showing the funnel from total items → valid IDs (broken down by source) → mapped to KG (broken down by 1:1 vs multi-mappings).

---

## Typical Workflow

```python
from visualizer import Visualizer

viz = Visualizer(config={
    "row_order": ["proteins", "metabolites", "lipids", "clinical-labs"],
    "col_order": ["arivale", "ukb"],
})

# Aggregate
df = viz.aggregate_stats("results/stats/")

# Generate both figures
viz.render_heatmap(df, output_path="figures/coverage")
viz.render_breakdown(df, output_path="figures/breakdown")
```

---

## Notes

- **Figure sizing**: If you don't pass `figsize`, the methods calculate dimensions based on grid size using `figsize_per_cell` (breakdown) or `figsize_per_cell_heatmap` (heatmap) from config.

- **Label precedence**: `entity_labels` passed to a render method are merged with (and override) those in config. Same for `dataset_labels`, though those have no config defaults.

- **Saving**: When `output_path` is provided, figures are saved in all formats listed in `output_formats`. The path should omit the extension.