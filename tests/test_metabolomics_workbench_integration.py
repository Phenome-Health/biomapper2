"""Integration tests for MetabolomicsWorkbenchAnnotator using real API.

These tests make actual HTTP requests to the Metabolomics Workbench RefMet API.
Run with: uv run pytest -m integration -v
Skip with: uv run pytest -m "not integration"
"""

import pytest

from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator


@pytest.mark.integration
class TestRealAPI:
    """Integration tests that hit the real Metabolomics Workbench API."""

    def test_real_api_carnitine(self):
        """Test real API call for known metabolite: Carnitine."""
        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "Carnitine"}

        # Act
        result = annotator.get_annotations(entity, name_field="name")

        # Assert - verify structure
        assert "metabolomics-workbench" in result
        annotations = result["metabolomics-workbench"]

        # Verify expected vocabularies are present
        assert "pubchem.compound" in annotations
        assert "inchikey" in annotations
        assert "smiles" in annotations
        assert "rm" in annotations

        # Verify specific known values for Carnitine
        assert "10917" in annotations["pubchem.compound"]
        assert "PHIQHXFUZVPYII-ZCFIWIBFSA-N" in annotations["inchikey"]

    def test_real_api_nonexistent(self):
        """Test real API call for nonexistent metabolite."""
        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "ThisMetaboliteDoesNotExist12345"}

        # Act
        result = annotator.get_annotations(entity, name_field="name")

        # Assert - should return slug with empty dict
        assert result == {"metabolomics-workbench": {}}

    def test_real_api_special_characters(self):
        """Test real API call for metabolite with special characters."""
        annotator = MetabolomicsWorkbenchAnnotator()
        entity = {"name": "5-hydroxyindoleacetic acid"}

        # Act
        result = annotator.get_annotations(entity, name_field="name")

        # Assert - verify structure
        assert "metabolomics-workbench" in result
        annotations = result["metabolomics-workbench"]

        # Verify expected vocabularies are present (this is a known metabolite)
        assert "pubchem.compound" in annotations
        assert "inchikey" in annotations

        # Verify specific known values for 5-hydroxyindoleacetic acid
        assert "1826" in annotations["pubchem.compound"]
