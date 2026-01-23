"""Tests for Pydantic Entity model."""

import pandas as pd
import pytest

from biomapper2.mapper import Mapper
from biomapper2.models import Entity


def test_entity_creation_minimal():
    """Entity can be created with just a name."""
    entity = Entity(name="glucose")
    assert entity.name == "glucose"
    assert entity.curies == []
    assert entity.kg_ids == {}
    assert entity.chosen_kg_id is None


def test_entity_creation_with_fields():
    """Entity can be created with pipeline fields."""
    entity = Entity(
        name="creatinine",
        curies=["KEGG.COMPOUND:C00791"],
        kg_ids={"CHEBI:16737": ["KEGG.COMPOUND:C00791"]},
        chosen_kg_id="CHEBI:16737",
    )
    assert entity.name == "creatinine"
    assert len(entity.curies) == 1
    assert "CHEBI:16737" in entity.kg_ids
    assert entity.chosen_kg_id == "CHEBI:16737"


def test_entity_allows_extra_fields():
    """Entity allows user-provided extra fields (provided_id columns)."""
    # Extra kwargs allowed by ConfigDict(extra="allow")
    entity = Entity(name="glucose", kegg_id="C00031", pubchem_cid="5793")  # type: ignore[call-arg]
    assert entity.name == "glucose"
    assert entity.model_extra is not None
    assert entity.model_extra["kegg_id"] == "C00031"
    assert entity.model_extra["pubchem_cid"] == "5793"


def test_entity_from_dict():
    """Entity can be created from a dict."""
    data = {"name": "glucose", "kegg_id": "C00031"}
    entity = Entity.from_input(data)
    assert entity.name == "glucose"
    assert entity.model_extra is not None
    assert entity.model_extra["kegg_id"] == "C00031"


def test_entity_from_series():
    """Entity can be created from a pandas Series."""
    series = pd.Series({"name": "glucose", "kegg_id": "C00031"})
    entity = Entity.from_input(series)
    assert entity.name == "glucose"
    assert entity.model_extra is not None
    assert entity.model_extra["kegg_id"] == "C00031"


def test_entity_to_dict():
    """Entity can be converted to dict."""
    entity = Entity(name="glucose", kegg_id="C00031", curies=["KEGG.COMPOUND:C00031"])  # type: ignore[call-arg]
    result = entity.to_dict()
    assert isinstance(result, dict)
    assert result["name"] == "glucose"
    assert result["kegg_id"] == "C00031"
    assert result["curies"] == ["KEGG.COMPOUND:C00031"]


def test_entity_to_series():
    """Entity can be converted to pandas Series."""
    entity = Entity(name="glucose", kegg_id="C00031", curies=["KEGG.COMPOUND:C00031"])  # type: ignore[call-arg]
    result = entity.to_series()
    assert isinstance(result, pd.Series)
    assert result["name"] == "glucose"
    assert result["kegg_id"] == "C00031"
    assert result["curies"] == ["KEGG.COMPOUND:C00031"]


def test_entity_update_from_series():
    """Entity can be updated with fields from a pandas Series (pipeline step output)."""
    entity = Entity.from_input({"name": "glucose", "kegg_id": "C00031"})

    # Simulate normalization step output
    normalization_result = pd.Series(
        {
            "curies": ["KEGG.COMPOUND:C00031"],
            "curies_provided": ["KEGG.COMPOUND:C00031"],
            "curies_assigned": {},
            "invalid_ids_provided": {},
            "invalid_ids_assigned": {},
            "unrecognized_vocabs_provided": [],
            "unrecognized_vocabs_assigned": [],
        }
    )

    updated = entity.update_from(normalization_result)

    # Original unchanged
    assert entity.curies == []

    # Updated has new fields
    assert updated.curies == ["KEGG.COMPOUND:C00031"]
    assert updated.name == "glucose"
    assert updated.model_extra is not None
    assert updated.model_extra["kegg_id"] == "C00031"


def test_entity_update_preserves_extra():
    """Entity update preserves user-provided extra fields."""
    entity = Entity.from_input({"name": "glucose", "kegg_id": "C00031", "pubchem_cid": "5793"})

    linking_result = pd.Series(
        {
            "kg_ids": {"CHEBI:17234": ["KEGG.COMPOUND:C00031"]},
            "kg_ids_provided": {"CHEBI:17234": ["KEGG.COMPOUND:C00031"]},
            "kg_ids_assigned": {},
        }
    )

    updated = entity.update_from(linking_result)

    assert updated.model_extra is not None
    assert updated.model_extra["kegg_id"] == "C00031"
    assert updated.model_extra["pubchem_cid"] == "5793"
    assert "CHEBI:17234" in updated.kg_ids


def test_entity_from_input_with_name_field():
    """Entity.from_input can rename a field to 'name'."""
    data = {"chemical_name": "glucose", "kegg_id": "C00031"}
    entity = Entity.from_input(data, name_field="chemical_name")

    assert entity.name == "glucose"
    assert entity.model_extra is not None
    assert entity.model_extra["chemical_name"] == "glucose"  # Original preserved
    assert entity.model_extra["kegg_id"] == "C00031"


def test_entity_from_input_name_field_default():
    """Entity.from_input uses 'name' field by default."""
    data = {"name": "glucose", "kegg_id": "C00031"}
    entity = Entity.from_input(data)

    assert entity.name == "glucose"


# Integration tests with Mapper


@pytest.fixture
def mapper():
    """Create a Mapper instance for testing."""
    return Mapper()


@pytest.mark.integration
def test_map_entity_returns_dict(mapper: Mapper):
    """map_entity_to_kg still returns dict for API compatibility."""
    entity = {"name": "creatinine", "kegg_ids": "C00791"}
    result = mapper.map_entity_to_kg(
        item=entity,
        name_field="name",
        provided_id_fields=["kegg_ids"],
        entity_type="metabolite",
    )

    # External API returns dict
    assert isinstance(result, dict)
    assert "name" in result
    assert "curies" in result
    assert "kg_ids" in result
