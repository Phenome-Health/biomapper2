"""Tests for mapping individual entities to the KG."""
from biomapper2.mapper import Mapper


def test_map_to_kg_basic(shared_mapper: Mapper):
    """Test basic entity mapping to KG."""
    entity = {
        'name': 'creatinine',
        'kegg_ids': 'C00791'
    }
    mapped_entity = shared_mapper.map_entity_to_kg(item=entity,
                                                   name_field='name',
                                                   provided_id_fields=['kegg_ids'],
                                                   entity_type='metabolite')

    # Check that original entity wasn't modified
    assert 'curies' not in entity

    # Check that mapped_entity has expected fields
    assert 'name' in mapped_entity
    assert 'kegg_ids' in mapped_entity
    assert 'curies' in mapped_entity
    assert isinstance(mapped_entity['curies'], list)
    assert 'kg_ids' in mapped_entity


def test_map_to_kg_preserves_input(shared_mapper: Mapper):
    """Test that input entity is not modified."""
    entity = {'name': 'glucose', 'kegg': 'C00031'}
    original = entity.copy()

    mapped_entity = shared_mapper.map_entity_to_kg(item=entity,
                                                   name_field='name',
                                                   provided_id_fields=['kegg'],
                                                   entity_type='metabolite')

    assert entity == original
    assert mapped_entity != entity  # Different object


def test_map_to_kg_name_only(shared_mapper: Mapper):
    """Test handling of minimal entity."""
    entity = {'chemical_name': 'carnitine'}

    mapped_entity = shared_mapper.map_entity_to_kg(item=entity,
                                                   name_field='chemical_name',
                                                   provided_id_fields=[],
                                                   entity_type='metabolite')

    assert 'chemical_name' in mapped_entity
    assert mapped_entity['chemical_name'] == 'carnitine'
    # TODO: once have lexical mappings handled, check that mapping is there


def test_map_to_kg_multiple_identifiers(shared_mapper: Mapper):
    """Test entity with multiple identifier types."""
    entity = {
        'name': 'aspirin',
        'kegg.compound': 'C01405',
        'drugbank': 'DB00945'
    }
    mapped_entity = shared_mapper.map_entity_to_kg(item=entity,
                                                   name_field='name',
                                                   provided_id_fields=['kegg.compound', 'drugbank'],
                                                   entity_type='drug')

    assert 'curies' in mapped_entity
    assert len(mapped_entity['curies']) > 1
