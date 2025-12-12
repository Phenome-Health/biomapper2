"""Unit tests for MetabolomicsWorkbenchAnnotator."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests

from biomapper2.core.annotators.base import BaseAnnotator
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator


class TestMetabolomicsWorkbenchAnnotator:
    """Test suite for MetabolomicsWorkbenchAnnotator basic structure."""

    def test_annotator_slug(self):
        """Test that annotator has correct slug identifier."""
        annotator = MetabolomicsWorkbenchAnnotator()
        assert annotator.slug == "metabolomics-workbench"

    def test_annotator_inheritance(self):
        """Test that annotator inherits from BaseAnnotator."""
        annotator = MetabolomicsWorkbenchAnnotator()
        assert isinstance(annotator, BaseAnnotator)


class TestGetAnnotations:
    """Test suite for get_annotations() method."""

    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_get_annotations_returns_correct_structure(self, mock_get: MagicMock):
        """Test that get_annotations returns correctly structured AssignedIDsDict."""
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

        # Assert
        assert "metabolomics-workbench" in result
        assert isinstance(result["metabolomics-workbench"], dict)

    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_vocabulary_mappings(self, mock_get: MagicMock):
        """Test that API fields are returned with raw field names (Normalizer handles mapping)."""
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

        # Assert - verify raw API field names are used (Normalizer handles mapping)
        annotations = result["metabolomics-workbench"]

        # Raw field names preserved
        assert "pubchem_cid" in annotations
        assert "10917" in annotations["pubchem_cid"]

        assert "inchi_key" in annotations
        assert "PHIQHXFUZVPYII-ZCFIWIBFSA-N" in annotations["inchi_key"]

        assert "smiles" in annotations
        assert "C[N+](C)(C)C[C@@H](CC(=O)[O-])O" in annotations["smiles"]

        # refmet_id preserved with RM prefix (Normalizer handles cleaning)
        assert "refmet_id" in annotations
        assert "RM0008606" in annotations["refmet_id"]


class TestEdgeCases:
    """Test suite for edge cases and error handling."""

    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_empty_response(self, mock_get: MagicMock):
        """Test that empty API response (metabolite not found) returns empty annotations."""
        # Arrange - API returns empty list when metabolite not found
        mock_response = MagicMock()
        mock_response.json.return_value = []
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "NonexistentMetabolite"}

        # Act
        result = annotator.get_annotations(entity, name_field="name")

        # Assert - should return slug with empty dict
        assert result == {"metabolomics-workbench": {}}

    def test_missing_name_field(self):
        """Test that entity without name field returns empty dict."""
        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"other_field": "value"}

        # Act
        result = annotator.get_annotations(entity, name_field="name")

        # Assert - no API call, returns empty dict
        assert result == {}

    def test_none_name_value(self):
        """Test that entity with None name value returns empty dict."""
        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": None}

        # Act
        result = annotator.get_annotations(entity, name_field="name")

        # Assert - no API call, returns empty dict
        assert result == {}

    def test_empty_name_value(self):
        """Test that entity with empty string name returns empty dict."""
        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": ""}

        # Act
        result = annotator.get_annotations(entity, name_field="name")

        # Assert - no API call, returns empty dict
        assert result == {}

    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_special_characters_in_name(self, mock_get: MagicMock):
        """Test that special characters in name are URL encoded."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "5-hydroxyindoleacetic acid",
            "pubchem_cid": "1826",
            "inchi_key": "DUUGKQCEGZLZNO-UHFFFAOYSA-N",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "5-hydroxyindoleacetic acid"}

        # Act
        result = annotator.get_annotations(entity, name_field="name")

        # Assert - verify API was called with URL-encoded name
        mock_get.assert_called_once()
        call_url = mock_get.call_args[0][0]
        assert "5-hydroxyindoleacetic%20acid" in call_url

        # Assert - verify annotations returned with raw field names
        assert "metabolomics-workbench" in result
        assert "pubchem_cid" in result["metabolomics-workbench"]

    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_api_http_error(self, mock_get: MagicMock):
        """Test that HTTP errors are raised."""
        # Arrange
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500 Server Error")
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "Carnitine"}

        # Act & Assert
        with pytest.raises(requests.exceptions.HTTPError):
            annotator.get_annotations(entity, name_field="name")

    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_api_timeout(self, mock_get: MagicMock):
        """Test that timeout errors are raised."""
        # Arrange
        mock_get.side_effect = requests.exceptions.Timeout("Connection timed out")

        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "Carnitine"}

        # Act & Assert
        with pytest.raises(requests.exceptions.Timeout):
            annotator.get_annotations(entity, name_field="name")

    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_partial_api_response(self, mock_get: MagicMock):
        """Test that partial API response (missing some fields) is handled gracefully."""
        # Arrange - API returns response with only some fields
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "SomeMetabolite",
            "pubchem_cid": "12345",
            # Missing: inchi_key, smiles, refmet_id
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "SomeMetabolite"}

        # Act
        result = annotator.get_annotations(entity, name_field="name")

        # Assert - should only include available fields (raw API field names)
        annotations = result["metabolomics-workbench"]
        assert "pubchem_cid" in annotations
        assert "12345" in annotations["pubchem_cid"]

        # Missing fields should not be in annotations
        assert "inchi_key" not in annotations
        assert "smiles" not in annotations
        assert "refmet_id" not in annotations


class TestBulkOperations:
    """Test suite for get_annotations_bulk() method."""

    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_bulk_returns_series_with_matching_index(self, mock_get: MagicMock):
        """Test that get_annotations_bulk returns Series with same index as input DataFrame."""
        # Arrange
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
            index=[10, 20, 30],  # Custom index
        )

        # Act
        result = annotator.get_annotations_bulk(entities, name_field="name")

        # Assert
        assert isinstance(result, pd.Series)
        assert list(result.index) == [10, 20, 30]
        assert len(result) == 3

    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_bulk_caches_api_calls(self, mock_get: MagicMock):
        """Test that duplicate names only result in one API call each (deduplication)."""
        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "name": "Carnitine",
            "pubchem_cid": "10917",
        }
        mock_response.status_code = 200
        mock_response.raise_for_status = MagicMock()
        mock_get.return_value = mock_response

        annotator = MetabolomicsWorkbenchAnnotator()
        # DataFrame with duplicate names
        entities = pd.DataFrame(
            {"name": ["Carnitine", "Carnitine", "Glucose", "Carnitine"]},
        )

        # Act
        result = annotator.get_annotations_bulk(entities, name_field="name")

        # Assert - should only call API twice (once for Carnitine, once for Glucose)
        assert mock_get.call_count == 2
        assert len(result) == 4  # But still returns 4 results

    @patch("biomapper2.core.annotators.metabolomics_workbench.requests.get")
    def test_get_annotations_uses_cache_when_provided(self, mock_get: MagicMock):
        """Test that get_annotations uses cache instead of making API call."""
        # Arrange - provide a cache with pre-fetched data
        cache = {
            "Carnitine": {
                "name": "Carnitine",
                "pubchem_cid": "10917",
                "inchi_key": "PHIQHXFUZVPYII-ZCFIWIBFSA-N",
            }
        }

        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "Carnitine"}

        # Act
        result = annotator.get_annotations(entity, name_field="name", cache=cache)

        # Assert - no API call should be made
        mock_get.assert_not_called()

        # Assert - annotations should still be returned from cache (raw field names)
        assert "metabolomics-workbench" in result
        assert "pubchem_cid" in result["metabolomics-workbench"]
        assert "10917" in result["metabolomics-workbench"]["pubchem_cid"]


class TestAnnotationEngineIntegration:
    """Test suite for AnnotationEngine integration with MetabolomicsWorkbenchAnnotator."""

    def test_engine_selects_metabolomics_workbench_for_metabolite(self):
        """Test that AnnotationEngine selects MetabolomicsWorkbenchAnnotator for metabolite entity type."""
        from biomapper2.core.annotation_engine import AnnotationEngine

        engine = AnnotationEngine()

        # Act
        annotators = engine._select_annotators("metabolite")

        # Assert - should include MetabolomicsWorkbenchAnnotator
        annotator_types = [type(a).__name__ for a in annotators]
        assert "MetabolomicsWorkbenchAnnotator" in annotator_types

    def test_engine_selects_metabolomics_workbench_for_lipid(self):
        """Test that AnnotationEngine selects MetabolomicsWorkbenchAnnotator for lipid entity type."""
        from biomapper2.core.annotation_engine import AnnotationEngine

        engine = AnnotationEngine()

        # Act
        annotators = engine._select_annotators("lipid")

        # Assert - should include MetabolomicsWorkbenchAnnotator
        annotator_types = [type(a).__name__ for a in annotators]
        assert "MetabolomicsWorkbenchAnnotator" in annotator_types

    def test_engine_selects_metabolomics_workbench_for_smallmolecule(self):
        """Test that AnnotationEngine selects MetabolomicsWorkbenchAnnotator for smallmolecule entity type."""
        from biomapper2.core.annotation_engine import AnnotationEngine

        engine = AnnotationEngine()

        # Act
        annotators = engine._select_annotators("smallmolecule")

        # Assert - should include MetabolomicsWorkbenchAnnotator
        annotator_types = [type(a).__name__ for a in annotators]
        assert "MetabolomicsWorkbenchAnnotator" in annotator_types


@pytest.mark.integration
class TestEndToEndPipeline:
    """End-to-end tests for the full annotation pipeline with MetabolomicsWorkbenchAnnotator."""

    def test_annotation_engine_annotates_metabolite_with_real_api(self):
        """Test that AnnotationEngine correctly annotates a metabolite using real MW API."""
        from biomapper2.core.annotation_engine import AnnotationEngine

        engine = AnnotationEngine()
        entity = {"name": "Carnitine"}

        # Act - annotate using the full pipeline
        result = engine.annotate(
            item=entity,
            name_field="name",
            provided_id_fields=[],
            entity_type="metabolite",
            mode="all",
        )

        # Assert - should have assigned_ids with MW annotations
        assert "assigned_ids" in result.index
        assigned_ids = result["assigned_ids"]

        # Should contain metabolomics-workbench annotations
        assert "metabolomics-workbench" in assigned_ids
        mw_annotations = assigned_ids["metabolomics-workbench"]

        # Verify expected fields are populated (raw API field names)
        assert "pubchem_cid" in mw_annotations
        assert "10917" in mw_annotations["pubchem_cid"]
        assert "inchi_key" in mw_annotations
        assert "PHIQHXFUZVPYII-ZCFIWIBFSA-N" in mw_annotations["inchi_key"]

    def test_annotation_engine_annotates_dataframe_with_real_api(self):
        """Test that AnnotationEngine correctly annotates a DataFrame using real MW API."""
        from biomapper2.core.annotation_engine import AnnotationEngine

        engine = AnnotationEngine()
        entities = pd.DataFrame({"name": ["Carnitine", "Glucose"]})

        # Act - annotate using the full pipeline
        result = engine.annotate(
            item=entities,
            name_field="name",
            provided_id_fields=[],
            entity_type="metabolite",
            mode="all",
        )

        # Assert - should be a DataFrame with assigned_ids column
        assert isinstance(result, pd.DataFrame)
        assert "assigned_ids" in result.columns
        assert len(result) == 2

        # Verify first row (Carnitine) has MW annotations (raw field names)
        carnitine_ids = result.iloc[0]["assigned_ids"]
        assert "metabolomics-workbench" in carnitine_ids
        assert "pubchem_cid" in carnitine_ids["metabolomics-workbench"]

        # Verify second row (Glucose) has MW annotations (raw field names)
        glucose_ids = result.iloc[1]["assigned_ids"]
        assert "metabolomics-workbench" in glucose_ids
        assert "pubchem_cid" in glucose_ids["metabolomics-workbench"]
