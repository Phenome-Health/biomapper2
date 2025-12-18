"""Unit tests for MetabolomicsWorkbenchAnnotator.

Consolidated test suite with 8 tests covering:
- Basic annotator structure
- Core annotation behavior
- Edge cases
- Bulk operations
- Engine integration
- Real API calls (integration tests)
- Full Mapper pipeline (end-to-end test)
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from biomapper2.core.annotators.base import BaseAnnotator
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator


class TestMetabolomicsWorkbenchAnnotator:
    """Consolidated test suite for MetabolomicsWorkbenchAnnotator."""

    # =========================================================================
    # Test 1: Basic annotator structure
    # =========================================================================
    def test_annotator_basics(self):
        """Test annotator has correct slug and inherits from BaseAnnotator."""
        annotator = MetabolomicsWorkbenchAnnotator()
        assert annotator.slug == "metabolomics-workbench"
        assert isinstance(annotator, BaseAnnotator)

    # =========================================================================
    # Test 2: Core annotation behavior
    # =========================================================================
    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_get_annotations_returns_raw_field_names(self, mock_get: MagicMock):
        """Test that annotations use raw API field names (Normalizer handles mapping)."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "Carnitine",
            "pubchem_cid": "10917",
            "inchi_key": "PHIQHXFUZVPYII-ZCFIWIBFSA-N",
            "smiles": "C[N+](C)(C)C[C@@H](CC(=O)[O-])O",
            "refmet_id": "RM0008606",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "Carnitine"}

        # Act
        result = annotator.get_annotations(entity, name_field="name")

        # Assert - raw API field names preserved
        annotations = result["metabolomics-workbench"]
        assert "pubchem_cid" in annotations
        assert "10917" in annotations["pubchem_cid"]
        assert "inchi_key" in annotations
        assert "PHIQHXFUZVPYII-ZCFIWIBFSA-N" in annotations["inchi_key"]
        assert "smiles" in annotations
        # refmet_id preserved with RM prefix (Normalizer handles cleaning)
        assert "refmet_id" in annotations
        assert "RM0008606" in annotations["refmet_id"]

    # =========================================================================
    # Test 3: Edge case - not found
    # =========================================================================
    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_empty_response(self, mock_get: MagicMock):
        """Test that empty API response (metabolite not found) returns empty annotations."""
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "NonexistentMetabolite"}

        result = annotator.get_annotations(entity, name_field="name")

        assert result == {"metabolomics-workbench": {}}

    # =========================================================================
    # Test 4: Edge case - missing input
    # =========================================================================
    def test_missing_name_field(self):
        """Test that entity without name field returns empty dict."""
        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"other_field": "value"}

        result = annotator.get_annotations(entity, name_field="name")

        assert result == {}

    # =========================================================================
    # Test 5: Bulk operations
    # =========================================================================
    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_bulk_returns_series(self, mock_get: MagicMock):
        """Test that get_annotations_bulk returns Series with matching index."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "Carnitine",
            "pubchem_cid": "10917",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        entities = pd.DataFrame(
            {"name": ["Carnitine", "Glucose", "Alanine"]},
            index=[10, 20, 30],
        )

        result = annotator.get_annotations_bulk(entities, name_field="name")

        assert isinstance(result, pd.Series)
        assert list(result.index) == [10, 20, 30]
        assert len(result) == 3

    # =========================================================================
    # Test 6: Engine integration
    # =========================================================================
    def test_engine_selects_annotator_for_metabolite(self):
        """Test that AnnotationEngine selects MetabolomicsWorkbenchAnnotator for metabolite."""
        from biomapper2.core.annotation_engine import AnnotationEngine

        engine = AnnotationEngine()
        annotators = engine._select_annotators("metabolite")

        annotator_types = [type(a).__name__ for a in annotators]
        assert "MetabolomicsWorkbenchAnnotator" in annotator_types

    # =========================================================================
    # Test 7: Real API call (integration)
    # =========================================================================
    @pytest.mark.integration
    def test_real_api_carnitine(self):
        """Test real API call for known metabolite: Carnitine."""
        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "Carnitine"}

        result = annotator.get_annotations(entity, name_field="name")

        # Verify structure
        assert "metabolomics-workbench" in result
        annotations = result["metabolomics-workbench"]

        # Verify raw API field names
        assert "pubchem_cid" in annotations
        assert "inchi_key" in annotations
        assert "smiles" in annotations
        assert "refmet_id" in annotations

        # Verify known values
        assert "10917" in annotations["pubchem_cid"]
        assert "PHIQHXFUZVPYII-ZCFIWIBFSA-N" in annotations["inchi_key"]

    # =========================================================================
    # Test 8: Full Mapper pipeline (end-to-end)
    # =========================================================================
    @pytest.mark.integration
    def test_mapper_end_to_end(self):
        """Test full pipeline using Mapper.map_entity_to_kg() with MW annotator."""
        from biomapper2.mapper import Mapper

        mapper = Mapper()
        # Use pd.Series to get a pd.Series back (dict input returns dict)
        entity = pd.Series({"name": "Carnitine"})

        # Run full pipeline: annotation -> normalization -> linking -> resolution
        result = mapper.map_entity_to_kg(
            item=entity,
            name_field="name",
            provided_id_fields=[],
            entity_type="metabolite",
            annotation_mode="all",
        )

        # Verify result is a Series (since input was Series)
        assert isinstance(result, pd.Series)

        # Verify assigned_ids contain MW annotations
        assert "assigned_ids" in result.index
        assigned_ids = result["assigned_ids"]
        assert "metabolomics-workbench" in assigned_ids

        # Verify MW annotations have expected fields (raw API field names)
        mw_annotations = assigned_ids["metabolomics-workbench"]
        assert "pubchem_cid" in mw_annotations
        assert "10917" in mw_annotations["pubchem_cid"]
