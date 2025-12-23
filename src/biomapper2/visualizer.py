import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure


class HarmonizationVisualizer:
    """Aggregates mapping stats and renders visualizations."""

    DEFAULT_CONFIG = {
        # Ordering
        "row_order": None,
        "col_order": None,
        # Display labels
        "entity_labels": {
            "proteins": "Proteins",
            "metabolites": "Metabolites",
            "lipids": "Lipids",
            "clinical-labs": "Labs",
            "questionnaire": "Questionnaires",
        },
        # Figure settings
        "figsize_per_cell": (2.2, 1.6),
        "figsize_per_cell_heatmap": (1.6, 1.2),
        "dpi": 300,
        "output_formats": ["pdf", "png"],
        "title": "Harmonization Coverage",
        # Heatmap settings
        "heatmap_vmin": 0,
        "heatmap_vmax": 100,
        "heatmap_na_color": "#ffffff",
        "heatmap_annot_fontsize": 9,
        # Breakdown chart settings
        "breakdown_colors": {
            "total": "lightgray",
            "only_provided": "lightblue",
            "both": "#D2C3DA",
            "only_assigned": "#f4cbd5",
            "one_to_one": "#a6f1a6",
            "multi": "khaki",
        },
        "breakdown_label_threshold_pct": 8,
    }

    def __init__(self, config: dict | None = None, parse_filename: Callable | None = None):
        self.config = {**self.DEFAULT_CONFIG, **(config or {})}
        self.parse_filename = parse_filename or self._parse_filename_default

    def _parse_filename_default(self, filename: str) -> dict:
        """Parse dataset_entity from filename. Assumes no underscores in names."""
        stem = filename.replace("_MAPPED_a_summary_stats.json", "")
        parts = stem.split("_")

        if len(parts) != 2:
            raise ValueError(f"Cannot parse '{filename}': expected 'dataset_entity_MAPPED_a_summary_stats.json'")
        return {"dataset": parts[0], "entity": parts[1]}

    def aggregate_stats(self, stats_dir: str | Path, fill_missing: bool = True) -> pd.DataFrame:
        """
        Aggregate stats JSONs into tidy DataFrame.
        Single pass: extracts metadata, stats, and tracks combinations for gap-filling.
        """
        stats_dir = Path(stats_dir)
        json_files = list(stats_dir.glob("*_MAPPED_a_summary_stats.json"))

        if not json_files:
            raise ValueError(f"No stats JSON files found in {stats_dir}")

        records = []
        all_datasets, all_entities = set(), set()

        for json_file in json_files:
            with open(json_file) as f:
                data = json.load(f)

            # Prefer embedded _meta, fall back to filename
            if "_meta" in data:
                dataset = data["_meta"].get("dataset")
                entity = data["_meta"].get("entity")
            else:
                parsed = self.parse_filename(json_file.name)
                dataset, entity = parsed["dataset"], parsed["entity"]

            all_datasets.add(dataset)
            all_entities.add(entity)

            n_total = data.get("total_items")
            n_mapped = data.get("mapped_to_kg")

            if n_total and n_total > 0 and n_mapped is not None:
                coverage = n_mapped / n_total
                coverage_explanation = f"{n_mapped:,} / {n_total:,}"
            else:
                coverage, coverage_explanation = None, None

            records.append(
                {
                    "dataset": dataset,
                    "entity": entity,
                    "coverage": coverage,
                    "coverage_explanation": coverage_explanation,
                    "n_total": n_total,
                    "n_mapped": n_mapped,
                    "one_to_one": data.get("one_to_one_mappings"),
                    "multi_mappings": data.get("multi_mappings"),
                    # Breakdown-specific fields
                    "has_valid_ids": data.get("has_valid_ids"),
                    "has_only_provided_ids": data.get("has_only_provided_ids"),
                    "has_only_assigned_ids": data.get("has_only_assigned_ids"),
                    "has_both_provided_and_assigned_ids": data.get("has_both_provided_and_assigned_ids"),
                    "_source_file": json_file.name,
                }
            )

        df = pd.DataFrame(records)

        if fill_missing:
            existing = set(zip(df["dataset"], df["entity"]))
            missing = set(itertools.product(all_datasets, all_entities)) - existing

            if missing:
                missing_records = [
                    {
                        "dataset": d,
                        "entity": e,
                        "coverage": None,
                        "coverage_explanation": "N/A",
                        "n_total": None,
                        "n_mapped": None,
                        "one_to_one": None,
                        "multi_mappings": None,
                        "has_valid_ids": None,
                        "has_only_provided_ids": None,
                        "has_only_assigned_ids": None,
                        "has_both_provided_and_assigned_ids": None,
                        "_source_file": "MISSING",
                    }
                    for d, e in missing
                ]
                missing_df = pd.DataFrame(missing_records).astype(df.dtypes, errors="ignore")
                df = pd.concat([df, missing_df], ignore_index=True)

        return df

    def render_heatmap(
        self,
        df: pd.DataFrame,
        output_path: str | Path | None = None,
        title: str | None = None,
        dataset_labels: dict[str, str] | None = None,
        entity_labels: dict[str, str] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Render coverage heatmap from tidy DataFrame.
        Args:
            df: Tidy DataFrame with dataset, entity, coverage columns
            output_path: Optional path to save figure (without extension)
            title: Optional title override
            dataset_labels: Optional display names for datasets
            entity_labels: Optional display names for entities
            figsize: Optional (width, height) override; if None, calculated from grid size
        """
        _dataset_labels = dataset_labels or {}
        _entity_labels = {**self.config["entity_labels"], **(entity_labels or {})}

        matrix = df.pivot(index="entity", columns="dataset", values="coverage")

        if self.config["row_order"]:
            matrix = matrix.reindex(self.config["row_order"])
        if self.config["col_order"]:
            matrix = matrix.reindex(columns=self.config["col_order"])

        # Build annotations (before renaming index/columns)
        annot = np.empty(matrix.shape, dtype=object)
        for i, entity in enumerate(matrix.index):
            for j, dataset in enumerate(matrix.columns):
                val = matrix.iloc[i, j]
                if pd.isna(val):
                    annot[i, j] = "N/A"
                else:
                    expl_series = cast(
                        pd.Series,  # type: ignore[type-arg]
                        df.loc[
                            (df["entity"] == entity) & (df["dataset"] == dataset),
                            "coverage_explanation",
                        ],
                    )
                    expl = expl_series.iloc[0]
                    val_float = cast(float, val)
                    annot[i, j] = f"$\\mathbf{{{val_float*100:.1f}\\%}}$\n\n({expl})"

        # Apply display labels (after ordering, after building annotations)
        matrix.index = matrix.index.map(lambda x: _entity_labels.get(x, x))
        matrix.columns = matrix.columns.map(lambda x: _dataset_labels.get(x, x))

        # Calculate figure size based on grid dimensions, or use override
        n_rows, n_cols = matrix.shape
        if figsize is None:
            cell_w, cell_h = self.config["figsize_per_cell_heatmap"]
            figsize = (n_cols * cell_w + 1.5, n_rows * cell_h + 1)  # extra space for colorbar and title

        fig, ax = plt.subplots(figsize=figsize, dpi=self.config["dpi"])

        cmap = sns.diverging_palette(10, 130, as_cmap=True)
        cmap.set_bad(color=self.config["heatmap_na_color"])

        sns.heatmap(
            matrix * 100,
            annot=annot,
            fmt="",
            cmap=cmap,
            vmin=self.config["heatmap_vmin"],
            vmax=self.config["heatmap_vmax"],
            cbar_kws={"label": "Coverage (%)"},
            annot_kws={
                "fontsize": self.config["heatmap_annot_fontsize"],
                "ha": "center",
                "va": "center",
            },
            square=True,
            linewidths=0.5,
            linecolor="white",
            ax=ax,
        )

        # Manually add N/A text for NaN cells
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if pd.isna(matrix.iloc[i, j]):
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        "N/A",
                        ha="center",
                        va="center",
                        fontsize=self.config["heatmap_annot_fontsize"],
                        color="black",
                    )

        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha="right")
        ax.tick_params(axis="both", which="both", length=4, width=1, bottom=True, left=True)
        ax.set_title(title or self.config["title"], fontweight="bold", pad=15)
        ax.set_ylabel("")
        ax.set_xlabel("Dataset", fontweight="bold")
        plt.tight_layout()

        if output_path:
            self._save_fig(fig, output_path)

        return fig

    def render_breakdown(
        self,
        df: pd.DataFrame,
        output_path: str | Path | None = None,
        title: str | None = None,
        dataset_labels: dict[str, str] | None = None,
        entity_labels: dict[str, str] | None = None,
        figsize: tuple[float, float] | None = None,
    ) -> Figure:
        """Render stacked bar breakdown grid.
        Args:
            df: Tidy DataFrame with dataset, entity, and breakdown columns
            output_path: Optional path to save figure (without extension)
            title: Optional title override
            dataset_labels: Optional display names for datasets
            entity_labels: Optional display names for entities
            figsize: Optional (width, height) override; if None, calculated from grid size
        Each cell shows 3 bars: Total -> Valid IDs -> Mapped to KG
        """
        _dataset_labels = dataset_labels or {}
        _entity_labels = {**self.config["entity_labels"], **(entity_labels or {})}

        entities = self._get_ordered_values(df, "entity", self.config["row_order"])
        datasets = self._get_ordered_values(df, "dataset", self.config["col_order"])

        entity_display = [_entity_labels.get(e, e.title()) for e in entities]
        dataset_display = [_dataset_labels.get(d, d.title()) for d in datasets]

        # Calculate figure size based on grid dimensions, or use override
        if figsize is None:
            cell_w, cell_h = self.config["figsize_per_cell"]
            figsize = (len(datasets) * cell_w, len(entities) * cell_h + 1.2)

        fig, axes = plt.subplots(
            len(entities),
            len(datasets),
            figsize=figsize,
            squeeze=False,
            layout="constrained",
        )

        colors = self.config["breakdown_colors"]
        label_thresh = self.config["breakdown_label_threshold_pct"]

        for i, (entity, entity_disp) in enumerate(zip(entities, entity_display)):
            for j, (dataset, dataset_disp) in enumerate(zip(datasets, dataset_display)):
                ax = axes[i, j]

                mask = (df["dataset"] == dataset) & (df["entity"] == entity)
                n_total_series = cast(pd.Series, df.loc[mask, "n_total"])  # type: ignore[type-arg]
                if not mask.any() or n_total_series.isna().all():
                    ax.text(
                        0.5, 0.5, "N/A", ha="center", va="center", fontsize=12, transform=ax.transAxes, color="gray"
                    )
                    self._style_breakdown_cell(ax, i, j, entity_disp, dataset_disp)
                    continue

                row = df[mask].iloc[0]
                self._draw_breakdown_bars(ax, row, colors, label_thresh)
                self._style_breakdown_cell(ax, i, j, entity_disp, dataset_disp)

        fig.suptitle(title or self.config["title"], fontsize=18, fontweight="bold")

        legend_elements = [
            mpatches.Patch(color=colors["only_provided"], alpha=0.8, label="Provided IDs"),
            mpatches.Patch(color=colors["both"], alpha=0.8, label="Both Provided & Assigned"),
            mpatches.Patch(color=colors["only_assigned"], alpha=0.8, label="Assigned IDs"),
            mpatches.Patch(color=colors["one_to_one"], alpha=0.8, label="1:1 Mappings"),
            mpatches.Patch(color=colors["multi"], alpha=0.8, label="Multi-Mappings"),
        ]
        fig.legend(handles=legend_elements, loc="outside lower center", ncol=3, fontsize=9)

        if output_path:
            self._save_fig(fig, output_path)

        return fig

    def _get_ordered_values(self, df: pd.DataFrame, col: str, order: list | None) -> list:
        """Get unique values in specified order, or alphabetically if None."""
        unique_vals = df[col].dropna().unique()
        if order is None:
            return sorted(unique_vals)
        return [v for v in order if v in unique_vals]

    def _draw_breakdown_bars(self, ax, row, colors, label_thresh):
        """Draw the three stacked bars for a single breakdown cell."""
        bar_width = 0.25
        x_pos = [0.1, 0.4, 0.7]

        total = row["n_total"] or 0
        valid = row["has_valid_ids"] or 0
        only_provided = row["has_only_provided_ids"] or 0
        both = row["has_both_provided_and_assigned_ids"] or 0
        only_assigned = row["has_only_assigned_ids"] or 0
        mapped = row["n_mapped"] or 0
        one_to_one = row["one_to_one"] or 0
        multi = row["multi_mappings"] or 0

        # Percentages relative to total
        valid_pct = (valid / total * 100) if total > 0 else 0
        only_provided_pct = (only_provided / total * 100) if total > 0 else 0
        both_pct = (both / total * 100) if total > 0 else 0
        only_assigned_pct = (only_assigned / total * 100) if total > 0 else 0
        mapped_pct = (mapped / total * 100) if total > 0 else 0

        # Bar 3 breakdown (relative to mapped, scaled to total)
        one_to_one_pct_of_mapped = (one_to_one / mapped * 100) if mapped > 0 else 0
        multi_pct_of_mapped = (multi / mapped * 100) if mapped > 0 else 0
        one_to_one_portion = (one_to_one_pct_of_mapped / 100) * mapped_pct
        multi_portion = (multi_pct_of_mapped / 100) * mapped_pct

        # Bar 1: Total (baseline 100%)
        ax.bar(x_pos[0], 100, bar_width, color=colors["total"], alpha=0.7)
        ax.text(x_pos[0], 50, f"{total:,}", ha="center", va="center", fontsize=7, fontweight="bold")

        # Bar 2: Valid IDs breakdown
        ax.bar(x_pos[1], only_provided_pct, bar_width, color=colors["only_provided"], alpha=0.8)
        ax.bar(x_pos[1], both_pct, bar_width, color=colors["both"], alpha=0.8, bottom=only_provided_pct)
        ax.bar(
            x_pos[1],
            only_assigned_pct,
            bar_width,
            color=colors["only_assigned"],
            alpha=0.8,
            bottom=only_provided_pct + both_pct,
        )

        # Labels for bar 2 chunks
        if only_provided_pct > label_thresh:
            ax.text(
                x_pos[1],
                only_provided_pct / 2,
                f"{only_provided:,}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
        if both_pct > label_thresh:
            ax.text(
                x_pos[1],
                only_provided_pct + both_pct / 2,
                f"{both:,}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
        if only_assigned_pct > label_thresh:
            ax.text(
                x_pos[1],
                only_provided_pct + both_pct + only_assigned_pct / 2,
                f"{only_assigned:,}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )

        # Bar 3: Mapped to KG breakdown
        ax.bar(x_pos[2], one_to_one_portion, bar_width, color=colors["one_to_one"], alpha=0.8)
        ax.bar(x_pos[2], multi_portion, bar_width, color=colors["multi"], alpha=0.8, bottom=one_to_one_portion)

        if one_to_one_portion > label_thresh:
            ax.text(
                x_pos[2],
                one_to_one_portion / 2,
                f"{one_to_one:,}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )
        if multi_portion > label_thresh:
            ax.text(
                x_pos[2],
                one_to_one_portion + multi_portion / 2,
                f"{multi:,}",
                ha="center",
                va="center",
                fontsize=7,
                fontweight="bold",
            )

        # Bar labels belowrender_breakdown
        ax.text(x_pos[0], -5, "Total", ha="center", va="top", fontsize=8)
        ax.text(x_pos[1], -5, "Valid\nIDs", ha="center", va="top", fontsize=8)
        ax.text(x_pos[2], -5, "Mapped\nto KG", ha="center", va="top", fontsize=8)

        # Count labels above bars
        ax.text(x_pos[0], 104, f"{total:,}", ha="center", va="bottom", fontsize=6.5, color="grey")
        ax.text(x_pos[1], valid_pct + 4, f"{valid:,}", ha="center", va="bottom", fontsize=6.5, color="grey")
        ax.text(x_pos[2], mapped_pct + 4, f"{mapped:,}", ha="center", va="bottom", fontsize=6.5, color="grey")

        # Step-down line
        heights = [100 + 0.8, valid_pct + 0.8, mapped_pct + 0.8]
        left_edges = [p - bar_width / 2 for p in x_pos]
        extended_x = left_edges + [x_pos[2] + bar_width / 2]
        extended_h = heights + [mapped_pct + 0.8]
        ax.step(extended_x, extended_h, "k:", linewidth=1, alpha=0.7, zorder=10, where="post")

    def _style_breakdown_cell(self, ax, row_idx, col_idx, entity_disp, dataset_disp):
        """Apply consistent styling to a breakdown cell."""
        ax.set_ylim(0, 120)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"], fontsize=8)
        ax.grid(True, alpha=0.3, axis="y")

        if row_idx == 0:
            ax.set_title(dataset_disp, fontsize=10, fontweight="bold", pad=10)
        if col_idx == 0:
            ax.set_ylabel(entity_disp, fontsize=10, fontweight="bold")

    def _save_fig(self, fig: Figure, output_path: str | Path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for fmt in self.config["output_formats"]:
            path = output_path.with_suffix(f".{fmt}")
            fig.savefig(path, dpi=self.config["dpi"], bbox_inches="tight")
            print(f"Saved: {path}")
