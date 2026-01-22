"""Tests for the semantic query module."""

import pytest

from biomapper2.core.semantic_query import (
    SemanticQueryResult,
    SemanticRelation,
    classify_predicate,
    get_all_relations,
    get_associated_diseases,
    get_available_predicates,
    get_pathways_for_entity,
    get_semantic_relations,
    is_semantic_predicate,
    query_with_semantic_expansion,
)

# ==============================================================================
# UNIT TESTS: Predicate Classification (no API calls)
# ==============================================================================


def test_classify_predicate_semantic():
    """Test classification of semantic predicates."""
    assert classify_predicate("biolink:participates_in") == "semantic"
    assert classify_predicate("biolink:treats") == "semantic"
    assert classify_predicate("biolink:causes") == "semantic"
    assert classify_predicate("biolink:interacts_with") == "semantic"


def test_classify_predicate_equivalency():
    """Test classification of equivalency predicates."""
    assert classify_predicate("biolink:same_as") == "equivalency"
    assert classify_predicate("owl:sameAs") == "equivalency"
    assert classify_predicate("skos:exactMatch") == "equivalency"


def test_classify_predicate_unknown():
    """Test classification of unknown predicates."""
    assert classify_predicate("biolink:totally_made_up") == "unknown"
    assert classify_predicate("random_predicate") == "unknown"


def test_is_semantic_predicate():
    """Test semantic predicate helper function."""
    assert is_semantic_predicate("biolink:participates_in") is True
    assert is_semantic_predicate("biolink:same_as") is False
    # Unknown predicates are treated as semantic (conservative)
    assert is_semantic_predicate("unknown_pred") is True


def test_get_available_predicates():
    """Test that available predicates are returned correctly."""
    predicates = get_available_predicates()
    assert "semantic" in predicates
    assert "equivalency" in predicates
    assert len(predicates["semantic"]) > 0
    assert len(predicates["equivalency"]) > 0
    assert "biolink:participates_in" in predicates["semantic"]
    assert "biolink:same_as" in predicates["equivalency"]


def test_data_structures():
    """Test data structure creation and methods."""
    relation = SemanticRelation(
        subject_id="CHEBI:4167",
        subject_name="glucose",
        predicate="biolink:participates_in",
        object_id="REACT:123",
        object_name="Glycolysis",
        object_category="Pathway",
    )
    rel_dict = relation.to_dict()
    assert rel_dict["subject_name"] == "glucose"
    assert rel_dict["predicate"] == "biolink:participates_in"

    result = SemanticQueryResult(
        resolved_entities=[{"id": "CHEBI:4167", "name": "glucose"}],
        semantic_relations=[{"predicate": "participates_in"}],
        stats={"total_relations": 1},
    )
    result_dict = result.to_dict()
    assert result_dict["stats"]["total_relations"] == 1


# ==============================================================================
# INTEGRATION TESTS: API Calls (require KESTREL_API_KEY)
# ==============================================================================


@pytest.mark.integration
def test_get_semantic_relations_glucose():
    """Test getting semantic relations for glucose (CHEBI:4167)."""
    relations = get_semantic_relations("CHEBI:4167", limit=10)

    # Should return some relations (glucose has many semantic connections)
    assert isinstance(relations, list)
    # Relations should have expected keys
    if relations:
        assert "predicate" in relations[0]
        assert "object_id" in relations[0] or "end_node_id" in relations[0]


@pytest.mark.integration
def test_get_all_relations_grouped():
    """Test getting all relations grouped by predicate type."""
    grouped = get_all_relations("CHEBI:4167", limit=50)

    assert "semantic" in grouped
    assert "equivalency" in grouped
    assert "unknown" in grouped
    assert isinstance(grouped["semantic"], list)


@pytest.mark.integration
def test_get_pathways_for_entity():
    """Test convenience function for pathway lookup."""
    pathways = get_pathways_for_entity("CHEBI:4167", limit=5)
    assert isinstance(pathways, list)


@pytest.mark.integration
def test_get_associated_diseases():
    """Test convenience function for disease associations."""
    diseases = get_associated_diseases("CHEBI:4167", limit=5)
    assert isinstance(diseases, list)


@pytest.mark.integration
def test_query_with_semantic_expansion():
    """End-to-end test: search + semantic expansion."""
    result = query_with_semantic_expansion(
        "glucose",
        predicates=["participates_in"],
        search_limit=2,
        relation_limit=5,
    )

    assert isinstance(result, SemanticQueryResult)
    assert "search_hits" in result.stats
    assert "total_relations" in result.stats
    assert isinstance(result.resolved_entities, list)
    assert isinstance(result.semantic_relations, list)
