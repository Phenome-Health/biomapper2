"""Tests for the Visualizer class."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.figure import Figure

from biomapper2.visualizer import (
    REQUIRED_STATS_FIELDS,
    StatsParseError,
    StatsValidationError,
    Visualizer,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def cleanup_figures():
    """Close all matplotlib figures after each test to prevent memory leaks."""
    yield
    plt.close("all")


def make_valid_stats(
    total_items: int = 100,
    mapped_to_kg: int = 80,
    one_to_one_mappings: int = 70,
    multi_mappings: int = 10,
    has_valid_ids: int = 90,
    has_only_provided_ids: int = 30,
    has_only_assigned_ids: int = 20,
    has_both_provided_and_assigned_ids: int = 40,
) -> dict:
    """Create a valid stats dict with all required fields."""
    return {
        "total_items": total_items,
        "mapped_to_kg": mapped_to_kg,
        "one_to_one_mappings": one_to_one_mappings,
        "multi_mappings": multi_mappings,
        "has_valid_ids": has_valid_ids,
        "has_only_provided_ids": has_only_provided_ids,
        "has_only_assigned_ids": has_only_assigned_ids,
        "has_both_provided_and_assigned_ids": has_both_provided_and_assigned_ids,
    }


@pytest.fixture
def stats_dir(tmp_path: Path) -> Path:
    """Create a temporary stats directory."""
    return tmp_path / "stats"


@pytest.fixture(scope="session")
def visualizer() -> Visualizer:
    """Create a default Visualizer instance (session-scoped since it's stateless)."""
    return Visualizer()


# =============================================================================
# Aggregation Logic Tests
# =============================================================================


class TestCoverageCalculation:
    """Tests for coverage calculation math and explanation formatting."""

    def test_coverage_basic_calculation(self, stats_dir: Path, visualizer: Visualizer):
        """Verify coverage = mapped_to_kg / total_items."""
        stats_dir.mkdir()
        stats = make_valid_stats(total_items=100, mapped_to_kg=75)
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)

        assert len(df) == 1
        assert df.iloc[0]["coverage"] == pytest.approx(0.75)

    def test_coverage_explanation_formatting(self, stats_dir: Path, visualizer: Visualizer):
        """Verify explanation string has correct format with comma separators."""
        stats_dir.mkdir()
        stats = make_valid_stats(total_items=10000, mapped_to_kg=8500)
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)

        assert df.iloc[0]["coverage_explanation"] == "8,500 / 10,000"

    def test_coverage_100_percent(self, stats_dir: Path, visualizer: Visualizer):
        """Verify 100% coverage is calculated correctly."""
        stats_dir.mkdir()
        stats = make_valid_stats(total_items=50, mapped_to_kg=50)
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)

        assert df.iloc[0]["coverage"] == pytest.approx(1.0)
        assert df.iloc[0]["coverage_explanation"] == "50 / 50"

    def test_coverage_0_percent(self, stats_dir: Path, visualizer: Visualizer):
        """Verify 0% coverage (mapped=0) is handled correctly."""
        stats_dir.mkdir()
        stats = make_valid_stats(total_items=100, mapped_to_kg=0)
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)

        assert df.iloc[0]["coverage"] == pytest.approx(0.0)
        assert df.iloc[0]["coverage_explanation"] == "0 / 100"


class TestMissingCombinationFilling:
    """Tests for sparse dataset/entity matrix gap-filling."""

    def test_missing_combinations_filled_with_na(self, stats_dir: Path, visualizer: Visualizer):
        """Sparse dataset/entity matrix gets correctly expanded with N/A rows."""
        stats_dir.mkdir()

        # Create stats for datasetA/proteins and datasetB/metabolites only
        stats_a_prot = make_valid_stats(total_items=100, mapped_to_kg=80)
        stats_b_met = make_valid_stats(total_items=200, mapped_to_kg=150)

        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats_a_prot))
        (stats_dir / "datasetB_metabolites_MAPPED_a_summary_stats.json").write_text(json.dumps(stats_b_met))

        df = visualizer.aggregate_stats(stats_dir, fill_missing=True)

        # Should have 4 rows: 2 datasets x 2 entities
        assert len(df) == 4

        # Check original entries exist
        a_prot = df[(df["dataset"] == "datasetA") & (df["entity"] == "proteins")]
        assert len(a_prot) == 1
        assert a_prot.iloc[0]["coverage"] == pytest.approx(0.8)

        b_met = df[(df["dataset"] == "datasetB") & (df["entity"] == "metabolites")]
        assert len(b_met) == 1
        assert b_met.iloc[0]["coverage"] == pytest.approx(0.75)

        # Check filled entries have None/N/A
        a_met = df[(df["dataset"] == "datasetA") & (df["entity"] == "metabolites")]
        assert len(a_met) == 1
        assert pd.isna(a_met.iloc[0]["coverage"])
        assert a_met.iloc[0]["coverage_explanation"] == "N/A"

        b_prot = df[(df["dataset"] == "datasetB") & (df["entity"] == "proteins")]
        assert len(b_prot) == 1
        assert pd.isna(b_prot.iloc[0]["coverage"])

    def test_fill_missing_disabled(self, stats_dir: Path, visualizer: Visualizer):
        """When fill_missing=False, no extra rows are added."""
        stats_dir.mkdir()

        stats_a = make_valid_stats()
        stats_b = make_valid_stats()

        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats_a))
        (stats_dir / "datasetB_metabolites_MAPPED_a_summary_stats.json").write_text(json.dumps(stats_b))

        df = visualizer.aggregate_stats(stats_dir, fill_missing=False)

        # Should have only 2 rows (no filling)
        assert len(df) == 2

    def test_no_filling_needed_when_complete(self, stats_dir: Path, visualizer: Visualizer):
        """When all combinations exist, no extra rows are added."""
        stats_dir.mkdir()

        stats = make_valid_stats()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "datasetA_metabolites_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "datasetB_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "datasetB_metabolites_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)

        # All 4 combinations already exist
        assert len(df) == 4
        # None should be N/A
        assert df["coverage"].notna().all()


class TestEmptyDirectory:
    """Tests for empty directory handling."""

    def test_empty_directory_raises_error(self, stats_dir: Path, visualizer: Visualizer):
        """Empty stats directory raises a clear ValueError."""
        stats_dir.mkdir()

        with pytest.raises(ValueError, match="No stats JSON files matching"):
            visualizer.aggregate_stats(stats_dir)

    def test_no_matching_files_raises_error(self, stats_dir: Path, visualizer: Visualizer):
        """Directory with non-matching files raises ValueError."""
        stats_dir.mkdir()
        (stats_dir / "some_other_file.json").write_text("{}")
        (stats_dir / "readme.txt").write_text("hello")

        with pytest.raises(ValueError, match="No stats JSON files matching"):
            visualizer.aggregate_stats(stats_dir)


class TestZeroTotalItems:
    """Tests for zero total_items edge case."""

    def test_zero_total_items_no_division_error(self, stats_dir: Path, visualizer: Visualizer):
        """Zero total_items should be handled gracefully without division by zero."""
        stats_dir.mkdir()
        stats = make_valid_stats(total_items=0, mapped_to_kg=0)
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)

        assert len(df) == 1
        # Coverage should be None when total is 0
        assert pd.isna(df.iloc[0]["coverage"])
        assert df.iloc[0]["coverage_explanation"] is None


# =============================================================================
# Schema Validation Tests
# =============================================================================


class TestJSONParsing:
    """Tests for JSON file parsing and error handling."""

    def test_malformed_json_raises_parse_error(self, stats_dir: Path, visualizer: Visualizer):
        """Malformed JSON should raise StatsParseError with filename context."""
        stats_dir.mkdir()
        (stats_dir / "broken_proteins_MAPPED_a_summary_stats.json").write_text("{ invalid json }")

        with pytest.raises(StatsParseError, match="Failed to parse JSON file 'broken_proteins"):
            visualizer.aggregate_stats(stats_dir)

    def test_truncated_json_raises_parse_error(self, stats_dir: Path, visualizer: Visualizer):
        """Truncated JSON should raise StatsParseError."""
        stats_dir.mkdir()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text('{"total_items": 100')

        with pytest.raises(StatsParseError, match="Failed to parse JSON"):
            visualizer.aggregate_stats(stats_dir)

    def test_empty_json_file_raises_parse_error(self, stats_dir: Path, visualizer: Visualizer):
        """Empty JSON file should raise StatsParseError."""
        stats_dir.mkdir()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text("")

        with pytest.raises(StatsParseError, match="Failed to parse JSON"):
            visualizer.aggregate_stats(stats_dir)


class TestSchemaValidation:
    """Tests for required field validation in stats JSON files.

    Validation checks for key PRESENCE only - values can be null/None/empty.
    This ensures the upstream schema hasn't changed unexpectedly.
    """

    @pytest.mark.parametrize("missing_field", list(REQUIRED_STATS_FIELDS))
    def test_missing_key_raises_error(self, stats_dir: Path, visualizer: Visualizer, missing_field: str):
        """Each required key, when absent from the JSON, should raise StatsValidationError."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        del stats[missing_field]

        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        with pytest.raises(StatsValidationError, match=f"missing required fields.*{missing_field}"):
            visualizer.aggregate_stats(stats_dir)

    def test_multiple_missing_keys_listed(self, stats_dir: Path, visualizer: Visualizer):
        """Error message should list all missing keys."""
        stats_dir.mkdir()
        stats = {"total_items": 100}  # Missing most keys

        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        with pytest.raises(StatsValidationError, match="missing required fields"):
            visualizer.aggregate_stats(stats_dir)

    def test_null_values_allowed(self, stats_dir: Path, visualizer: Visualizer):
        """Keys with null/None values should pass validation (only key presence matters)."""
        stats_dir.mkdir()
        stats = {
            "total_items": None,
            "mapped_to_kg": None,
            "has_valid_ids": None,
            "has_only_provided_ids": None,
            "has_only_assigned_ids": None,
            "has_both_provided_and_assigned_ids": None,
            "one_to_one_mappings": None,
            "multi_mappings": None,
        }

        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        # Should not raise - keys are present even though values are null
        df = visualizer.aggregate_stats(stats_dir)
        assert len(df) == 1
        assert pd.isna(df.iloc[0]["coverage"])

    def test_all_required_keys_present_passes(self, stats_dir: Path, visualizer: Visualizer):
        """Valid JSON with all required keys should not raise."""
        stats_dir.mkdir()
        stats = make_valid_stats()

        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)
        assert len(df) == 1

    def test_extra_fields_allowed(self, stats_dir: Path, visualizer: Visualizer):
        """Extra fields beyond required ones should be ignored."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        stats["extra_field"] = "some value"
        stats["another_extra"] = 42

        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)
        assert len(df) == 1


# =============================================================================
# Filename Parsing Tests
# =============================================================================


class TestDefaultFilenameParsing:
    """Tests for the default filename parser."""

    def test_valid_filename_parsing(self, visualizer: Visualizer):
        """Default parser correctly extracts dataset and entity."""
        result = visualizer.parse_filename("ARIC_proteins_MAPPED_a_summary_stats.json")

        assert result["dataset"] == "ARIC"
        assert result["entity"] == "proteins"

    def test_valid_filename_various_names(self, visualizer: Visualizer):
        """Parser works with various valid dataset/entity names."""
        test_cases = [
            ("ukb_metabolites_MAPPED_a_summary_stats.json", "ukb", "metabolites"),
            ("HELIX_lipids_MAPPED_a_summary_stats.json", "HELIX", "lipids"),
            ("dataset123_clinical-labs_MAPPED_a_summary_stats.json", "dataset123", "clinical-labs"),
        ]

        for filename, expected_dataset, expected_entity in test_cases:
            result = visualizer.parse_filename(filename)
            assert result["dataset"] == expected_dataset
            assert result["entity"] == expected_entity

    def test_ambiguous_filename_raises_error(self, visualizer: Visualizer):
        """Filenames with too many underscores should raise ValueError."""
        # This has 3 parts before the suffix, which is ambiguous
        with pytest.raises(ValueError, match="Cannot parse"):
            visualizer.parse_filename("dataset_with_underscore_proteins_MAPPED_a_summary_stats.json")

    def test_single_part_filename_raises_error(self, visualizer: Visualizer):
        """Filename without underscore separator should raise ValueError."""
        with pytest.raises(ValueError, match="Cannot parse"):
            visualizer.parse_filename("datasetproteins_MAPPED_a_summary_stats.json")


class TestCustomParser:
    """Tests for custom filename parser integration."""

    def test_custom_parser_used_end_to_end(self, stats_dir: Path):
        """Custom parser function is used for filename parsing."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        # Use a different filename format
        (stats_dir / "STUDY-ABC--proteomics_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        def custom_parser(filename: str) -> dict:
            stem = filename.replace("_MAPPED_a_summary_stats.json", "")
            parts = stem.split("--")
            return {"dataset": parts[0], "entity": parts[1]}

        visualizer = Visualizer(parse_filename=custom_parser)
        df = visualizer.aggregate_stats(stats_dir)

        assert len(df) == 1
        assert df.iloc[0]["dataset"] == "STUDY-ABC"
        assert df.iloc[0]["entity"] == "proteomics"

    def test_custom_parser_error_propagates(self, stats_dir: Path):
        """Errors from custom parser should propagate up."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        (stats_dir / "bad_filename_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        def failing_parser(filename: str) -> dict:
            raise ValueError("Custom parser error!")

        visualizer = Visualizer(parse_filename=failing_parser)

        with pytest.raises(ValueError, match="Custom parser error!"):
            visualizer.aggregate_stats(stats_dir)


class TestConfigurableFileGlob:
    """Tests for configurable file_glob pattern."""

    def test_custom_file_glob_finds_files(self, stats_dir: Path):
        """Custom file_glob config allows finding files with different naming patterns."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        # Use a different filename format
        (stats_dir / "datasetA_proteins_stats.json").write_text(json.dumps(stats))
        (stats_dir / "datasetB_metabolites_stats.json").write_text(json.dumps(stats))

        def custom_parser(filename: str) -> dict:
            stem = filename.replace("_stats.json", "")
            parts = stem.split("_")
            return {"dataset": parts[0], "entity": parts[1]}

        visualizer = Visualizer(
            config={"file_glob": "*_stats.json"},
            parse_filename=custom_parser,
        )
        df = visualizer.aggregate_stats(stats_dir)

        assert len(df) == 4  # 2 datasets x 2 entities (with fill_missing)
        datasets = set(df["dataset"])
        assert datasets == {"datasetA", "datasetB"}

    def test_custom_file_glob_error_message_includes_pattern(self, stats_dir: Path):
        """Error message should include the custom file_glob pattern."""
        stats_dir.mkdir()
        (stats_dir / "some_file.json").write_text("{}")

        visualizer = Visualizer(config={"file_glob": "*.custom_pattern.json"})

        with pytest.raises(ValueError, match=r"\*\.custom_pattern\.json"):
            visualizer.aggregate_stats(stats_dir)

    def test_default_file_glob_works(self, stats_dir: Path, visualizer: Visualizer):
        """Default file_glob pattern finds standard naming convention files."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)

        assert len(df) == 1
        assert df.iloc[0]["dataset"] == "datasetA"


# =============================================================================
# Rendering Smoke Tests
# =============================================================================


class TestHeatmapRendering:
    """Smoke tests for heatmap rendering."""

    def test_heatmap_returns_figure(self, stats_dir: Path, visualizer: Visualizer):
        """render_heatmap should return a matplotlib Figure without crashing."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "datasetB_metabolites_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)
        fig = visualizer.render_heatmap(df)

        assert isinstance(fig, Figure)

    def test_heatmap_with_custom_title(self, stats_dir: Path, visualizer: Visualizer):
        """Custom title parameter should be accepted."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)
        fig = visualizer.render_heatmap(df, title="Custom Title")

        assert isinstance(fig, Figure)

    def test_heatmap_with_label_overrides(self, stats_dir: Path, visualizer: Visualizer):
        """Dataset and entity label overrides should be accepted."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)
        fig = visualizer.render_heatmap(
            df, dataset_labels={"datasetA": "Dataset A"}, entity_labels={"proteins": "Proteomics"}
        )

        assert isinstance(fig, Figure)


class TestBreakdownRendering:
    """Smoke tests for breakdown chart rendering."""

    def test_breakdown_returns_figure(self, stats_dir: Path, visualizer: Visualizer):
        """render_breakdown should return a matplotlib Figure without crashing."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "datasetB_metabolites_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)
        fig = visualizer.render_breakdown(df)

        assert isinstance(fig, Figure)

    def test_breakdown_with_custom_title(self, stats_dir: Path, visualizer: Visualizer):
        """Custom title parameter should be accepted."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)
        fig = visualizer.render_breakdown(df, title="Breakdown Chart")

        assert isinstance(fig, Figure)


class TestOutputFileWriting:
    """Tests for output file generation."""

    def test_heatmap_writes_output_files(self, stats_dir: Path, tmp_path: Path, visualizer: Visualizer):
        """render_heatmap with output_path should create PDF and PNG files."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)
        output_path = tmp_path / "output" / "heatmap"

        visualizer.render_heatmap(df, output_path=output_path)

        assert (tmp_path / "output" / "heatmap.pdf").exists()
        assert (tmp_path / "output" / "heatmap.png").exists()

    def test_breakdown_writes_output_files(self, stats_dir: Path, tmp_path: Path, visualizer: Visualizer):
        """render_breakdown with output_path should create PDF and PNG files."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)
        output_path = tmp_path / "output" / "breakdown"

        visualizer.render_breakdown(df, output_path=output_path)

        assert (tmp_path / "output" / "breakdown.pdf").exists()
        assert (tmp_path / "output" / "breakdown.png").exists()

    def test_output_creates_parent_directories(self, stats_dir: Path, tmp_path: Path, visualizer: Visualizer):
        """Output path with non-existent parent directories should be created."""
        stats_dir.mkdir()
        stats = make_valid_stats()
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir)
        output_path = tmp_path / "deeply" / "nested" / "output" / "heatmap"

        visualizer.render_heatmap(df, output_path=output_path)

        assert (tmp_path / "deeply" / "nested" / "output" / "heatmap.pdf").exists()


class TestMixedNARendering:
    """Tests for rendering with some N/A values in the data."""

    def test_heatmap_renders_with_partial_na_coverage(self, stats_dir: Path, visualizer: Visualizer):
        """Heatmap should render correctly when some cells have N/A coverage."""
        stats_dir.mkdir()
        stats = make_valid_stats(total_items=100, mapped_to_kg=75)

        # Create only 2 of 4 possible combinations - missing combos will be N/A
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "datasetB_metabolites_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir, fill_missing=True)

        # Should have 4 rows: 2 real + 2 filled N/A
        assert len(df) == 4

        fig = visualizer.render_heatmap(df)

        assert isinstance(fig, Figure)

    def test_heatmap_with_sparse_grid(self, stats_dir: Path, visualizer: Visualizer):
        """Heatmap renders with many N/A cells (3 datasets x 3 entities, only 3 have data)."""
        stats_dir.mkdir()
        stats = make_valid_stats()

        # Only populate diagonal
        (stats_dir / "datasetA_proteins_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "datasetB_metabolites_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "datasetC_lipids_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        df = visualizer.aggregate_stats(stats_dir, fill_missing=True)

        # 3x3 grid = 9 cells, only 3 have data
        assert len(df) == 9

        fig = visualizer.render_heatmap(df)

        assert isinstance(fig, Figure)

    def test_breakdown_renders_with_all_na_data(self, visualizer: Visualizer):
        """Breakdown should render even when all data values are N/A."""
        df = pd.DataFrame(
            [
                {
                    "dataset": "A",
                    "entity": "proteins",
                    "n_total": None,
                    "has_valid_ids": None,
                    "n_mapped": None,
                    "has_only_provided_ids": None,
                    "has_only_assigned_ids": None,
                    "has_both_provided_and_assigned_ids": None,
                    "one_to_one": None,
                    "multi_mappings": None,
                },
                {
                    "dataset": "B",
                    "entity": "metabolites",
                    "n_total": None,
                    "has_valid_ids": None,
                    "n_mapped": None,
                    "has_only_provided_ids": None,
                    "has_only_assigned_ids": None,
                    "has_both_provided_and_assigned_ids": None,
                    "one_to_one": None,
                    "multi_mappings": None,
                },
            ]
        )

        fig = visualizer.render_breakdown(df)

        assert isinstance(fig, Figure)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Additional edge case tests."""

    def test_config_override(self):
        """Custom config values should override defaults."""
        custom_config = {"title": "My Custom Title", "dpi": 150}
        visualizer = Visualizer(config=custom_config)

        assert visualizer.config["title"] == "My Custom Title"
        assert visualizer.config["dpi"] == 150
        # Default values should still be present
        assert "heatmap_vmin" in visualizer.config

    def test_row_and_col_ordering(self, stats_dir: Path):
        """Custom row/col ordering should be respected."""
        stats_dir.mkdir()
        stats = make_valid_stats()

        (stats_dir / "A_x_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "A_y_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "B_x_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))
        (stats_dir / "B_y_MAPPED_a_summary_stats.json").write_text(json.dumps(stats))

        visualizer = Visualizer(config={"row_order": ["y", "x"], "col_order": ["B", "A"]})

        df = visualizer.aggregate_stats(stats_dir)
        fig = visualizer.render_heatmap(df)

        assert isinstance(fig, Figure)
