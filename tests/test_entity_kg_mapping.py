"""Tests for mapping individual entities to the KG."""

import json

import pandas as pd

from biomapper2.mapper import Mapper


def print_entity(entity: dict | pd.Series):
    print(json.dumps(entity, indent=2))


def test_map_entity_basic(shared_mapper: Mapper):
    """Test basic entity mapping to KG."""
    entity = {"name": "creatinine", "kegg_ids": "C00791"}
    mapped_entity = shared_mapper.map_entity_to_kg(
        item=entity, name_field="name", provided_id_fields=["kegg_ids"], entity_type="metabolite"
    )

    # Check that original entity wasn't modified
    assert "curies" not in entity

    # Check that mapped_entity has expected fields
    assert "name" in mapped_entity
    assert "kegg_ids" in mapped_entity
    assert "curies" in mapped_entity
    assert isinstance(mapped_entity["curies"], list)
    assert "kg_ids" in mapped_entity


def test_map_entity_preserves_input(shared_mapper: Mapper):
    """Test that input entity is not modified."""
    entity = {"name": "glucose", "kegg": "C00031"}
    original = entity.copy()

    mapped_entity = shared_mapper.map_entity_to_kg(
        item=entity, name_field="name", provided_id_fields=["kegg"], entity_type="metabolite"
    )

    assert entity == original
    assert isinstance(mapped_entity, dict)
    assert mapped_entity != entity  # Different object


def test_map_entity_name_only(shared_mapper: Mapper):
    """Test handling of minimal entity."""
    entity = {"chemical_name": "carnitine"}

    mapped_entity = shared_mapper.map_entity_to_kg(
        item=entity, name_field="chemical_name", provided_id_fields=[], entity_type="metabolite"
    )

    assert "chemical_name" in mapped_entity
    assert mapped_entity["chemical_name"] == "carnitine"
    # TODO: once have lexical mappings handled, check that mapping is there


def test_map_entity_multiple_identifiers(shared_mapper: Mapper):
    """Test entity with multiple identifier types."""
    entity = {"name": "aspirin", "kegg.compound": "C01405", "drugbank": "DB00945"}
    mapped_entity = shared_mapper.map_entity_to_kg(
        item=entity, name_field="name", provided_id_fields=["kegg.compound", "drugbank"], entity_type="drug"
    )

    assert "curies" in mapped_entity
    assert len(mapped_entity["curies"]) > 1


def test_map_entity_id_field_is_list(shared_mapper: Mapper):
    """Test entity with a list value for one of the vocab ID fields."""
    entity = {"name": "carnitine", "kegg": ["C00487"], "pubchem_cid": "10917"}
    mapped_entity = shared_mapper.map_entity_to_kg(
        item=entity, name_field="name", provided_id_fields=["kegg", "pubchem_cid"], entity_type="metabolite"
    )

    assert "curies" in mapped_entity
    assert len(mapped_entity["curies"]) > 1


def test_custom_biolink_version(shared_mapper: Mapper):
    test_mapper = Mapper(biolink_version="4.2.4")
    assert test_mapper.normalizer.biolink_version == "4.2.4"
    entity = {"name": "glucose", "kegg": "C00031"}

    mapped_entity = test_mapper.map_entity_to_kg(
        item=entity, name_field="name", provided_id_fields=["kegg"], entity_type="metabolite"
    )
    assert len(mapped_entity["curies"]) == 1


def test_smiles_canonical_conversion(shared_mapper: Mapper):
    canonical_smiles = r"CCCCC/C=C\C/C=C\C/C=C\C/C=C\CCCC(=O)OCCN"  # This is already in canonical form
    non_canonical_smiles = r"C(OCCN)(=O)CCC/C=C\C/C=C\C/C=C\C/C=C\CCCCC"  # This is in NON-canonical form

    # Make sure that when we input a non-canonical smiles string it's converted to canonical form
    mapped_entity = shared_mapper.map_entity_to_kg(
        item={"name": "Virodhamine", "smiles": non_canonical_smiles},
        name_field="name",
        provided_id_fields=["smiles"],
        entity_type="lipid",
    )
    assert len(mapped_entity["curies_provided"]) == 1
    curie = mapped_entity["curies_provided"][0]
    local_id = curie.split(":")[1]
    assert local_id == canonical_smiles
    print_entity(mapped_entity)

    # Then make sure that when we input a canonical smiles string, it remains in canonical form
    mapped_entity = shared_mapper.map_entity_to_kg(
        item={"name": "Virodhamine", "smiles": canonical_smiles},
        name_field="name",
        provided_id_fields=["smiles"],
        entity_type="lipid",
    )
    assert len(mapped_entity["curies_provided"]) == 1
    curie = mapped_entity["curies_provided"][0]
    local_id = curie.split(":")[1]
    assert local_id == canonical_smiles


def test_dash_id_handling(shared_mapper: Mapper):
    """Test entity with a list value for one of the vocab ID fields."""
    entity = {"name": "parkinson's disease", "mesh": "-", "umls": ["-"], "doid": "14330"}
    mapped_entity = shared_mapper.map_entity_to_kg(
        item=entity,
        name_field="name",
        provided_id_fields=["mesh", "umls", "doid"],
        entity_type="disease",
        stop_on_invalid_id=True,
    )

    assert "curies" in mapped_entity
    assert len(mapped_entity["curies"]) >= 1
    print_entity(mapped_entity)


def test_annotation_mode_parameter(shared_mapper: Mapper):
    entity = {"name": "parkinson's disease", "doid": "14330"}
    mapped_entity_all = shared_mapper.map_entity_to_kg(
        item=entity, name_field="name", provided_id_fields=["doid"], entity_type="disease", annotation_mode="all"
    )
    print_entity(mapped_entity_all)
    assert mapped_entity_all["assigned_ids"]

    mapped_entity_none = shared_mapper.map_entity_to_kg(
        item=entity, name_field="name", provided_id_fields=["doid"], entity_type="disease", annotation_mode="none"
    )
    print_entity(mapped_entity_none)
    assert not mapped_entity_none["assigned_ids"]

    mapped_entity_missing_1 = shared_mapper.map_entity_to_kg(
        item=entity, name_field="name", provided_id_fields=["doid"], entity_type="disease", annotation_mode="missing"
    )
    print_entity(mapped_entity_missing_1)
    assert not mapped_entity_missing_1["assigned_ids"]

    mapped_entity_missing_2 = shared_mapper.map_entity_to_kg(
        item=entity, name_field="name", provided_id_fields=[], entity_type="disease", annotation_mode="missing"
    )
    print_entity(mapped_entity_missing_2)
    assert mapped_entity_missing_2["assigned_ids"]


def test_user_provided_annotators(shared_mapper: Mapper):
    entity = {"name": "cholate"}
    mapped_entity = shared_mapper.map_entity_to_kg(
        item=entity,
        name_field="name",
        provided_id_fields=[],
        entity_type="metabolite",
        annotators=["kestrel-vector-search", "metabolomics-workbench"],
    )
    print_entity(mapped_entity)
    assert mapped_entity["kg_ids_assigned"]
    assert "kestrel-vector-search" in mapped_entity["kg_ids_assigned"]
