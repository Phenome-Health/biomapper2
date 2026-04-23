# Visualizer Usage Guide

## Overview

The `Visualizer` is a helper class, used to visually compare KG-harmonization performance across multiple datasets. It produces four visualization types:

1. **Coverage heatmap** – a matrix showing mapping success rates across dataset/entity combinations
2. **Breakdown chart** – a grid of stacked bar charts showing the pipeline stages (total → valid IDs → mapped to KG)
3. **Metric heatmaps** – faceted heatmaps showing precision, recall, and F1 score (overall or per-annotator)
4. **Precision-recall scatter** – scatter plot with iso-F1 contour lines, colored by entity type

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

This expects files matching `*_MAPPED_a_summary_stats.json`, which are created automatically by `mapper.map_dataset_to_kg()`.

The returned DataFrame includes an `annotator` column: one `_overall` row per dataset/entity combination, plus one row per annotator found in the `performance.per_annotator` section of the stats JSON. Precision, recall, F1 (and their post-one-to-many-resolution adjusted variants) are extracted from `performance.assigned_ids.per_provided_ids`.

## Rendering Methods

### Coverage & Breakdown

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

### Metric Heatmaps (Precision / Recall / F1)

```python
# Overall metrics
fig = viz.render_metric_heatmaps(df, annotator="_overall")

# Per-annotator metrics
fig = viz.render_metric_heatmaps(df, annotator="kestrel-hybrid-search")
```

**Output**: Three side-by-side heatmaps (Precision, Recall, F1 Score) for the selected annotator. Each cell shows the metric as a percentage.

### Precision-Recall Scatter

```python
fig = viz.render_pr_scatter(df, annotator="_overall")
```

**Output**: Scatter plot with precision on the x-axis and recall on the y-axis. Points are colored by entity type and labeled with dataset names (labels are automatically repositioned to avoid overlap). Dashed iso-F1 contour lines at 0.3, 0.5, 0.7, and 0.9 provide reference.

---

## Typical Workflow

```python
from biomapper2.visualizer import Visualizer

viz = Visualizer(config={
    "row_order": ["proteins", "metabolites", "lipids", "clinical-labs"],
    "col_order": ["arivale", "ukb"],
})

# Aggregate
df = viz.aggregate_stats("results/stats/")

# Coverage & breakdown
viz.render_heatmap(df, output_path="figures/coverage")
viz.render_breakdown(df, output_path="figures/breakdown")

# Precision / Recall / F1
viz.render_metric_heatmaps(df, output_path="figures/prf1_overall")
viz.render_metric_heatmaps(df, annotator="kestrel-hybrid-search", output_path="figures/prf1_kestrel")
viz.render_pr_scatter(df, output_path="figures/pr_scatter")
```

---

## Notes

- **Figure sizing**: If you don't pass `figsize`, the methods calculate dimensions based on grid size using `figsize_per_cell` (breakdown) or `figsize_per_cell_heatmap` (heatmap) from config.

- **Label precedence**: `entity_labels` passed to a render method are merged with (and override) those in config. Same for `dataset_labels`, though those have no config defaults.

- **Saving**: When `output_path` is provided, figures are saved in all formats listed in `output_formats`. The path should omit the extension.