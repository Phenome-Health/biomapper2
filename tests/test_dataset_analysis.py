"""Tests for the analysis module."""

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from biomapper2.core.analysis import analyze_dataset_mapping


@pytest.fixture
def mock_linker():
    """Create a mock linker that returns IDs unchanged."""
    linker = MagicMock()
    linker.get_kg_ids = lambda ids: {id_: id_ for id_ in ids}
    return linker


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def make_test_df(
    total: int,
    with_valid_provided: int = 0,
    with_invalid_provided: int = 0,
    with_assigned: int = 0,
    assigned_match_provided: int = 0,
) -> pd.DataFrame:
    """
    Build a minimal dataframe for testing analysis.

    Args:
        total: Total number of rows
        with_valid_provided: Number of rows with valid provided IDs that map to KG
        with_invalid_provided: Number of rows with invalid provided IDs
        with_assigned: Number of rows with assigned IDs (from eligible items)
        assigned_match_provided: Number of assigned IDs that match provided (for mode='all')
    """
    rows = []
    for i in range(total):
        has_valid_provided = i < with_valid_provided
        has_invalid_provided = with_valid_provided <= i < (with_valid_provided + with_invalid_provided)
        has_provided = has_valid_provided or has_invalid_provided

        # For mode='missing', assigned IDs go to items WITHOUT provided
        # For mode='all', we might assign to items WITH provided too
        eligible_idx = i - (with_valid_provided + with_invalid_provided)
        has_assigned = not has_provided and eligible_idx < with_assigned

        # For mode='all' testing: some assigned match provided
        assigned_matches = has_valid_provided and i < assigned_match_provided

        kg_id = f"KG:{i:04d}"
        curie = f"CURIE:{i:04d}"

        row = {
            "curies": [curie] if (has_valid_provided or has_assigned or assigned_matches) else [],
            "curies_provided": [curie] if has_valid_provided else [],
            "curies_assigned": {"test-annotator": [curie]} if (has_assigned or assigned_matches) else {},
            "invalid_ids_provided": [f"INVALID:{i}"] if has_invalid_provided else [],
            "invalid_ids_assigned": [],
            "unrecognized_vocabs_provided": [],
            "unrecognized_vocabs_assigned": [],
            "kg_ids": {kg_id: [curie]} if (has_valid_provided or has_assigned or assigned_matches) else {},
            "kg_ids_provided": {kg_id: [curie]} if has_valid_provided else {},
            "kg_ids_assigned": {"test-annotator": {kg_id: [curie]}} if (has_assigned or assigned_matches) else {},
            "chosen_kg_id": kg_id if (has_valid_provided or has_assigned or assigned_matches) else None,
            "chosen_kg_id_provided": kg_id if has_valid_provided else None,
            "chosen_kg_id_assigned": kg_id if (has_assigned or assigned_matches) else None,
        }
        rows.append(row)

    return pd.DataFrame(rows)


def save_test_df(df: pd.DataFrame, path: Path) -> str:
    """Save dataframe to TSV with proper string representation of complex columns."""
    df_copy = df.copy()
    for col in df_copy.columns:
        if df_copy[col].apply(lambda x: isinstance(x, (dict, list))).any():
            df_copy[col] = df_copy[col].apply(repr)
    filepath = str(path / "test_data.tsv")
    df_copy.to_csv(filepath, sep="\t", index=False)
    return filepath


class TestAnnotationModeCoverage:
    """Tests for coverage calculation based on annotation mode."""

    def test_mode_missing_coverage_all_eligible_assigned(self, mock_linker, temp_dir):
        """Coverage should be 100% when all eligible items get assigned IDs."""
        # 10 items: 7 have provided IDs, 3 don't (eligible)
        # All 3 eligible items get assigned IDs
        df = make_test_df(total=10, with_valid_provided=7, with_assigned=3)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="missing")

        assert stats["performance"]["assigned_ids"]["eligible_entities"] == 3
        assert stats["performance"]["assigned_ids"]["coverage"] == 1.0
        assert stats["performance"]["assigned_ids"]["coverage_explanation"] == "3 / 3"

    def test_mode_missing_coverage_partial_assigned(self, mock_linker, temp_dir):
        """Coverage should reflect partial success."""
        # 10 items: 5 have provided IDs, 5 don't (eligible)
        # Only 2 eligible items get assigned IDs
        df = make_test_df(total=10, with_valid_provided=5, with_assigned=2)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="missing")

        assert stats["performance"]["assigned_ids"]["eligible_entities"] == 5
        assert stats["performance"]["assigned_ids"]["coverage"] == 0.4
        assert stats["performance"]["assigned_ids"]["coverage_explanation"] == "2 / 5"

    def test_mode_all_coverage(self, mock_linker, temp_dir):
        """Coverage should be relative to total items when mode='all'."""
        df = make_test_df(total=10, with_valid_provided=7, with_assigned=3)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="all")

        assert stats["performance"]["assigned_ids"]["eligible_entities"] == 10
        # 7 from provided + 3 from assigned = 10, but assigned only has 3
        # Actually mapped_to_kg_assigned counts items with assigned IDs = 3
        assert stats["performance"]["assigned_ids"]["coverage_explanation"] == "3 / 10"

    def test_mode_none_coverage(self, mock_linker, temp_dir):
        """Coverage should be null when mode='none'."""
        df = make_test_df(total=10, with_valid_provided=7)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="none")

        assert stats["performance"]["assigned_ids"]["eligible_entities"] == 0
        assert stats["performance"]["assigned_ids"]["coverage"] is None


class TestPerProvidedStats:
    """Tests for per_provided_ids statistics."""

    def test_per_provided_skipped_when_mode_missing(self, mock_linker, temp_dir):
        """per_provided_ids should be null when mode='missing'."""
        df = make_test_df(total=10, with_valid_provided=7, with_assigned=3)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="missing")

        assert stats["performance"]["assigned_ids"]["per_provided_ids"] is None

    def test_per_provided_skipped_when_mode_none(self, mock_linker, temp_dir):
        """per_provided_ids should be null when mode='none'."""
        df = make_test_df(total=10, with_valid_provided=7)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="none")

        assert stats["performance"]["assigned_ids"]["per_provided_ids"] is None

    def test_per_provided_calculated_when_mode_all(self, mock_linker, temp_dir):
        """per_provided_ids should have metrics when mode='all' and there are provided IDs."""
        df = make_test_df(total=10, with_valid_provided=7, with_assigned=3, assigned_match_provided=5)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="all")

        per_provided = stats["performance"]["assigned_ids"]["per_provided_ids"]
        assert per_provided is not None
        assert "precision" in per_provided
        assert "recall" in per_provided
        assert "f1_score" in per_provided

    def test_per_provided_null_when_no_provided_ids(self, mock_linker, temp_dir):
        """per_provided_ids should be null when no items have provided IDs."""
        df = make_test_df(total=10, with_valid_provided=0, with_assigned=5)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="all")

        assert stats["performance"]["assigned_ids"]["per_provided_ids"] is None


class TestEligibleEntitiesCalculation:
    """Tests for eligible_entities calculation."""

    def test_invalid_provided_ids_count_as_having_provided(self, mock_linker, temp_dir):
        """Items with invalid provided IDs should not be eligible when mode='missing'."""
        # 10 items: 5 valid provided, 3 invalid provided, 2 no provided
        # eligible = 2 (only items with NO provided IDs at all)
        df = make_test_df(total=10, with_valid_provided=5, with_invalid_provided=3, with_assigned=2)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="missing")

        assert stats["has_provided_ids"] == 8  # 5 valid + 3 invalid
        assert stats["performance"]["assigned_ids"]["eligible_entities"] == 2

    def test_no_eligible_entities_when_all_have_provided(self, mock_linker, temp_dir):
        """When all items have provided IDs and mode='missing', eligible should be 0."""
        df = make_test_df(total=10, with_valid_provided=10)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="missing")

        assert stats["performance"]["assigned_ids"]["eligible_entities"] == 0
        assert stats["performance"]["assigned_ids"]["coverage"] is None
        assert stats["performance"]["assigned_ids"]["mapped_to_kg"] == 0


class TestPerAnnotatorStats:
    """Tests for per-annotator statistics."""

    def test_per_annotator_empty_when_no_annotations(self, mock_linker, temp_dir):
        """per_annotator should be empty when no annotators ran."""
        df = make_test_df(total=10, with_valid_provided=10)  # no eligible items
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="missing")

        assert stats["performance"]["per_annotator"] == {}

    def test_per_annotator_has_correct_eligible_entities(self, mock_linker, temp_dir):
        """Each annotator should report same eligible_entities as overall."""
        df = make_test_df(total=10, with_valid_provided=7, with_assigned=3)
        filepath = save_test_df(df, temp_dir)

        stats = analyze_dataset_mapping(filepath, mock_linker, annotation_mode="missing")

        overall_eligible = stats["performance"]["assigned_ids"]["eligible_entities"]
        assert overall_eligible == 3
        for annotator_stats in stats["performance"]["per_annotator"].values():
            assert annotator_stats["eligible_entities"] == overall_eligible


class TestOutputFiles:
    """Tests for output file generation."""

    def test_creates_summary_stats_json(self, mock_linker, temp_dir):
        """Should create a summary stats JSON file."""
        df = make_test_df(total=10, with_valid_provided=7, with_assigned=3)
        filepath = save_test_df(df, temp_dir)

        analyze_dataset_mapping(filepath, mock_linker, annotation_mode="missing")

        json_path = Path(filepath.replace(".tsv", "_a_summary_stats.json"))
        assert json_path.exists()

        with open(json_path) as f:
            saved_stats = json.load(f)
        assert saved_stats["total_items"] == 10
        assert saved_stats["annotation_mode"] == "missing"
