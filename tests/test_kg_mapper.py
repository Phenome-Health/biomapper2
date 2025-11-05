"""Tests for kg_mapper module."""
import pytest

from biomapper2.kg_mapper import map_to_kg


def test_map_to_kg_basic():
    """Test basic entity mapping to KG."""
    entity = {
        'name': 'creatinine',
        'kegg_ids': 'C00791'
    }

    mapped_entity = map_to_kg(entity)

    # Check that original entity wasn't modified
    assert 'curies' not in entity

    # Check that mapped_entity has expected fields
    assert 'name' in mapped_entity
    assert 'kegg_ids' in mapped_entity
    assert 'curies' in mapped_entity
    assert isinstance(mapped_entity['curies'], list)
    assert 'kg_ids' in mapped_entity


def test_map_to_kg_preserves_input():
    """Test that input entity is not modified."""
    entity = {'name': 'glucose', 'kegg': 'C00031'}
    original = entity.copy()

    mapped_entity = map_to_kg(entity)

    assert entity == original
    assert mapped_entity != entity  # Different object


def test_map_to_kg_name_only():
    """Test handling of minimal entity."""
    entity = {'name': 'unknown'}

    mapped_entity = map_to_kg(entity)

    assert 'name' in mapped_entity
    assert mapped_entity['name'] == 'unknown'
    # TODO: once have lexical mappings handled, check that mapping is there


def test_map_to_kg_multiple_identifiers():
    """Test entity with multiple identifier types."""
    entity = {
        'name': 'aspirin',
        'kegg.compound': 'C01405',
        'drugbank': 'DB00945'
    }

    mapped_entity = map_to_kg(entity)

    assert 'curies' in mapped_entity
    assert len(mapped_entity['curies']) > 1
