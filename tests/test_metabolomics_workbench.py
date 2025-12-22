"""Unit tests for MetabolomicsWorkbenchAnnotator.

Test suite (8 tests) covering:
- Basic annotator structure (1 test)
- Core annotation behavior with /match endpoint (1 test)
- Edge cases: no match, missing input (2 tests)
- Bulk operations (1 test)
- Engine integration (1 test)
- Real API calls: exact + fuzzy match (1 integration test)
- Full Mapper pipeline (1 end-to-end test)
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
    def test_get_annotations_returns_refmet_id(self, mock_get: MagicMock):
        """Test that annotations return refmet_id (KRAKEN has all equivalencies)."""
        # Arrange - /match endpoint response format
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "refmet_name": "Carnitine",
            "formula": "C7H15NO3",
            "exactmass": "161.1052",
            "super_class": "Fatty Acyls",
            "main_class": "Fatty acyl carnitines",
            "sub_class": "Acyl carnitines",
            "refmet_id": "RM0008606",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "Carnitine"}

        # Act
        result = annotator.get_annotations(entity, name_field="name", category="biolink:SmallMolecule")

        # Assert - only refmet_id is extracted (KRAKEN has all equivalencies)
        annotations = result["metabolomics-workbench"]
        assert "refmet_id" in annotations
        assert "RM0008606" in annotations["refmet_id"]
        # Other fields NOT extracted since we only need refmet_id
        assert "pubchem_cid" not in annotations

    # =========================================================================
    # Test 3: Edge case - not found
    # =========================================================================
    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_no_match_response(self, mock_get: MagicMock):
        """Test that /match 'no match' response (dash values) returns empty annotations."""
        # /match endpoint returns dashes when no match found
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "refmet_name": "-",
            "formula": "-",
            "exactmass": "-",
            "super_class": "-",
            "main_class": "-",
            "sub_class": "-",
            "refmet_id": "-",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "NonexistentMetabolite"}

        result = annotator.get_annotations(entity, name_field="name", category="biolink:SmallMolecule")

        assert result == {"metabolomics-workbench": {}}

    # =========================================================================
    # Test 4: Edge case - missing input
    # =========================================================================
    def test_missing_name_field(self):
        """Test that entity without name field returns empty dict."""
        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"other_field": "value"}

        result = annotator.get_annotations(entity, name_field="name", category="biolink:SmallMolecule")

        assert result == {}

    # =========================================================================
    # Test 5: Bulk operations
    # =========================================================================
    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_bulk_returns_series(self, mock_get: MagicMock):
        """Test that get_annotations_bulk returns Series with matching index."""
        # /match endpoint response format
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "refmet_name": "Carnitine",
            "refmet_id": "RM0008606",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        entities = pd.DataFrame(
            {"name": ["Carnitine", "Glucose", "Alanine"]},
            index=[10, 20, 30],
        )

        result = annotator.get_annotations_bulk(entities, name_field="name", category="biolink:SmallMolecule")

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
        annotator_slugs = engine._select_annotators("biolink:SmallMolecule")

        assert MetabolomicsWorkbenchAnnotator.slug in annotator_slugs

    # =========================================================================
    # Test 7: Real API call (integration)
    # =========================================================================
    @pytest.mark.integration
    def test_real_api_match_endpoint(self):
        """Test real API call: exact match and fuzzy match."""
        annotator = MetabolomicsWorkbenchAnnotator()

        # Test 1: Exact match - "Carnitine"
        result = annotator.get_annotations({"name": "Carnitine"}, name_field="name", category="biolink:SmallMolecule")
        assert "metabolomics-workbench" in result
        annotations = result["metabolomics-workbench"]
        assert "refmet_id" in annotations
        assert "RM0008606" in annotations["refmet_id"]

        # Test 2: Fuzzy match - "cholate" → "Cholic acid" (RM0135798)
        result = annotator.get_annotations({"name": "cholate"}, name_field="name", category="biolink:SmallMolecule")
        assert "metabolomics-workbench" in result
        annotations = result["metabolomics-workbench"]
        assert "refmet_id" in annotations
        assert "RM0135798" in annotations["refmet_id"]

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

        # Verify MW annotations have refmet_id (only field extracted now)
        mw_annotations = assigned_ids["metabolomics-workbench"]
        assert "refmet_id" in mw_annotations
        assert "RM0008606" in mw_annotations["refmet_id"]
