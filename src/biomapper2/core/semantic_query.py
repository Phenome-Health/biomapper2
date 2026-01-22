"""
Semantic query module for KRAKEN knowledge graph traversal.

This module provides access to KRAKEN's semantic multi-hop reasoning capability
via the /one-hop endpoint, complementing existing search-based entity resolution.

Key Discovery (KG-o1 v3 exploration):
- KRAKEN has 26+ semantic predicates with 89% semantic edge coverage
- Graph traversal achieves 99.4% recall on 1-hop semantic queries
- Search achieves 95.2% EM on entity resolution (vocabulary mapping)
- Both approaches are complementary: search for entities, graph for relationships

Usage:
    # Direct semantic relation lookup
    from biomapper2.core.semantic_query import get_semantic_relations
    relations = get_semantic_relations("CHEBI:4167")  # glucose

    # Convenience functions
    from biomapper2.core.semantic_query import get_pathways_for_entity
    pathways = get_pathways_for_entity("CHEBI:4167")

    # Hybrid: search + semantic expansion
    from biomapper2.core.semantic_query import query_with_semantic_expansion
    results = query_with_semantic_expansion("glucose", predicates=["participates_in"])
"""

import logging
from dataclasses import asdict, dataclass, field
from typing import Any

from ..utils import kestrel_request

# ==============================================================================
# PREDICATE CLASSIFICATION
# ==============================================================================

EQUIVALENCY_PREDICATES = {
    "biolink:same_as",
    "owl:sameAs",
    "skos:exactMatch",
    "oboInOwl:hasDbXref",
    "biolink:xref",
    "equivalent_to",
    "biolink:equivalent_to",
    "same_as",
    "skos:closeMatch",
    "biolink:has_equivalent_class",
}

SEMANTIC_PREDICATES = {
    # Participation
    "biolink:participates_in",
    "biolink:has_participant",
    "biolink:actively_involved_in",
    "biolink:capable_of",
    # Catalysis/Metabolism
    "biolink:catalyzes",
    "biolink:is_substrate_of",
    "biolink:has_metabolite",
    "biolink:metabolite_of",
    "biolink:affects",
    "biolink:affected_by",
    "biolink:has_output",
    "biolink:has_input",
    # Association
    "biolink:associated_with",
    "biolink:related_to",
    "biolink:correlated_with",
    "biolink:coexists_with",
    # Causal
    "biolink:causes",
    "biolink:caused_by",
    "biolink:contributes_to",
    "biolink:ameliorates",
    # Therapeutic / Clinical
    "biolink:treats",
    "biolink:treated_by",
    "biolink:prevents",
    "biolink:predisposes",
    "biolink:applied_to_treat",
    "biolink:in_clinical_trials_for",
    "biolink:mentioned_in_clinical_trials_for",
    # Structural / Ontological
    "biolink:part_of",
    "biolink:has_part",
    "biolink:component_of",
    "biolink:contains",
    "biolink:subclass_of",
    "biolink:superclass_of",
    "biolink:has_chemical_role",
    # Location
    "biolink:located_in",
    "biolink:location_of",
    "biolink:expressed_in",
    "biolink:expresses",
    # Interaction
    "biolink:interacts_with",
    # Similarity (semantic, not equivalency)
    "biolink:chemically_similar_to",
    "biolink:close_match",
}

# ==============================================================================
# DATA STRUCTURES
# ==============================================================================


@dataclass
class SemanticRelation:
    """A single semantic relation from the knowledge graph."""

    subject_id: str
    subject_name: str
    predicate: str
    object_id: str
    object_name: str
    object_category: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SemanticQueryResult:
    """Result of a semantic expansion query."""

    resolved_entities: list[dict[str, Any]] = field(default_factory=list)
    semantic_relations: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "resolved_entities": self.resolved_entities,
            "semantic_relations": self.semantic_relations,
            "stats": self.stats,
        }


# ==============================================================================
# PREDICATE UTILITIES
# ==============================================================================


def classify_predicate(predicate: str) -> str:
    """
    Classify a predicate as 'semantic', 'equivalency', or 'unknown'.

    Args:
        predicate: The predicate string (e.g., "biolink:participates_in")

    Returns:
        Classification: 'semantic', 'equivalency', or 'unknown'
    """
    pred_lower = predicate.lower()

    # Check both with and without biolink prefix
    variants = [predicate, pred_lower]
    if not predicate.startswith("biolink:"):
        variants.append(f"biolink:{predicate}")
        variants.append(f"biolink:{pred_lower}")

    semantic_lower = {p.lower() for p in SEMANTIC_PREDICATES}
    equivalency_lower = {p.lower() for p in EQUIVALENCY_PREDICATES}

    for variant in variants:
        if variant in SEMANTIC_PREDICATES or variant.lower() in semantic_lower:
            return "semantic"
        if variant in EQUIVALENCY_PREDICATES or variant.lower() in equivalency_lower:
            return "equivalency"

    return "unknown"


def is_semantic_predicate(predicate: str) -> bool:
    """Check if a predicate is semantic (not equivalency)."""
    return classify_predicate(predicate) in ("semantic", "unknown")


# ==============================================================================
# ONE-HOP API WRAPPER
# ==============================================================================


def _one_hop_request(
    start_node_id: str,
    direction: str = "both",
    predicate_filter: str | None = None,
    end_category_filter: str | None = None,
    limit: int = 50,
    mode: str = "slim",
) -> dict[str, Any]:
    """
    Query KRAKEN's /one-hop endpoint for graph traversal.

    This is the core API wrapper for semantic graph traversal.
    Returns raw API response; use _parse_one_hop_edges() to process.

    Args:
        start_node_id: Starting node ID (e.g., "CHEBI:4167")
        direction: "forward", "reverse", or "both"
        predicate_filter: Filter by predicate (e.g., "participates_in")
        end_category_filter: Filter by end node category (e.g., "Pathway")
        limit: Maximum number of results
        mode: Response mode ("slim", "full", or "preview")

    Returns:
        API response dict with keys: edge_schema, results, nodes
    """
    payload: dict[str, Any] = {
        "start_node_ids": start_node_id,
        "direction": direction,
        "limit": limit,
        "mode": mode,
    }

    if predicate_filter:
        payload["predicate_filter"] = predicate_filter
    if end_category_filter:
        payload["end_category_filter"] = end_category_filter

    try:
        return kestrel_request("POST", "one-hop", json=payload)
    except Exception as e:
        logging.warning(f"One-hop request failed for {start_node_id}: {e}")
        return {}


def _parse_one_hop_edges(response: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Parse one-hop response into a flat list of edge dictionaries.

    CRITICAL: This function properly extracts node names and categories
    from the 'nodes' dictionary, fixing the metadata loss bug discovered
    during KG-o1 v3 testing.

    Args:
        response: Raw response from _one_hop_request()

    Returns:
        List of edge dicts with keys: subject_id, predicate, object_id,
        object_name, object_category
    """
    if not isinstance(response, dict):
        return []

    schema = response.get("edge_schema", [])
    results = response.get("results", [])
    nodes = response.get("nodes", {})  # Node info lookup table

    parsed_edges = []

    for result in results:
        end_node_id = result.get("end_node_id", "")
        edges = result.get("edges", [])

        # Get node info from the nodes dictionary
        node_info = nodes.get(end_node_id, {})
        node_name = node_info.get("name", "")
        node_category = node_info.get("category", "")

        for edge_tuple in edges:
            # Convert tuple to dict using schema
            edge_dict: dict[str, Any] = {}
            for i, field_name in enumerate(schema):
                if i < len(edge_tuple):
                    edge_dict[field_name] = edge_tuple[i]

            # Add standardized field names
            edge_dict["subject_id"] = edge_dict.get("subject", "")
            edge_dict["object_id"] = edge_dict.get("object", end_node_id)
            edge_dict["object_name"] = node_name
            edge_dict["object_category"] = node_category
            edge_dict["end_node_id"] = end_node_id
            edge_dict["end_node_name"] = node_name

            parsed_edges.append(edge_dict)

    return parsed_edges


# ==============================================================================
# PUBLIC API: SEMANTIC RELATION QUERIES
# ==============================================================================


def get_semantic_relations(
    entity_id: str,
    direction: str = "both",
    predicate_filter: str | None = None,
    category_filter: str | None = None,
    limit: int = 50,
    include_equivalency: bool = False,
) -> list[dict[str, Any]]:
    """
    Get semantic relations for an entity via one-hop graph traversal.

    Use this for semantic queries like:
    - "What pathways does glucose participate in?"
    - "What diseases are associated with this compound?"
    - "What genes interact with this metabolite?"

    For entity resolution (vocabulary mapping), use hybrid_search instead.

    Args:
        entity_id: Entity ID (e.g., "CHEBI:4167")
        direction: "forward" (outgoing), "reverse" (incoming), or "both"
        predicate_filter: Filter by predicate (e.g., "participates_in")
        category_filter: Filter by end node category (e.g., "Pathway")
        limit: Maximum results
        include_equivalency: If True, include equivalency predicates

    Returns:
        List of relation dictionaries with keys: predicate, object_id,
        object_name, object_category
    """
    response = _one_hop_request(
        entity_id,
        direction=direction,
        predicate_filter=predicate_filter,
        end_category_filter=category_filter,
        limit=limit,
    )

    edges = _parse_one_hop_edges(response)

    if include_equivalency:
        return edges

    # Filter to semantic predicates only
    return [edge for edge in edges if is_semantic_predicate(edge.get("predicate", ""))]


def get_all_relations(
    entity_id: str,
    direction: str = "both",
    limit: int = 100,
) -> dict[str, list[dict[str, Any]]]:
    """
    Get all relations for an entity, grouped by predicate type.

    Args:
        entity_id: Entity ID
        direction: "forward", "reverse", or "both"
        limit: Maximum results

    Returns:
        Dict with keys 'semantic', 'equivalency', 'unknown', each containing
        a list of edge dictionaries
    """
    response = _one_hop_request(entity_id, direction=direction, limit=limit)
    edges = _parse_one_hop_edges(response)

    grouped: dict[str, list[dict[str, Any]]] = {"semantic": [], "equivalency": [], "unknown": []}

    for edge in edges:
        classification = classify_predicate(edge.get("predicate", ""))
        grouped[classification].append(edge)

    return grouped


# ==============================================================================
# CONVENIENCE FUNCTIONS
# ==============================================================================


def get_pathways_for_entity(entity_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Get pathways that an entity participates in.

    Args:
        entity_id: Entity ID (e.g., "CHEBI:4167" for glucose)
        limit: Maximum pathways to return

    Returns:
        List of pathway relations with pathway ID, name, and predicate
    """
    return get_semantic_relations(
        entity_id,
        direction="forward",
        predicate_filter="participates_in",
        category_filter="Pathway",
        limit=limit,
    )


def get_associated_diseases(entity_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Get diseases associated with an entity.

    Args:
        entity_id: Entity ID
        limit: Maximum diseases to return

    Returns:
        List of disease relations
    """
    # Try multiple predicates that indicate disease associations
    relations = []

    for predicate in ["associated_with", "treats", "causes", "predisposes"]:
        results = get_semantic_relations(
            entity_id,
            direction="both",
            predicate_filter=predicate,
            category_filter="Disease",
            limit=limit,
        )
        relations.extend(results)

        # Break early if we have enough
        if len(relations) >= limit:
            break

    return relations[:limit]


def get_interacting_genes(entity_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Get genes that interact with an entity.

    Args:
        entity_id: Entity ID
        limit: Maximum genes to return

    Returns:
        List of gene relations
    """
    return get_semantic_relations(
        entity_id,
        direction="both",
        predicate_filter="interacts_with",
        category_filter="Gene",
        limit=limit,
    )


def get_related_metabolites(entity_id: str, limit: int = 20) -> list[dict[str, Any]]:
    """
    Get metabolites related to an entity.

    Args:
        entity_id: Entity ID
        limit: Maximum metabolites to return

    Returns:
        List of metabolite relations
    """
    return get_semantic_relations(
        entity_id,
        direction="both",
        category_filter="SmallMolecule",
        limit=limit,
    )


# ==============================================================================
# HYBRID QUERY: SEARCH + SEMANTIC EXPANSION
# ==============================================================================


def _hybrid_search(query: str, limit: int = 10, category: str | None = None) -> list[dict[str, Any]]:
    """
    Perform hybrid search via Kestrel API.

    Internal helper - wraps the hybrid-search endpoint.
    """
    payload: dict[str, Any] = {"search_text": query, "limit": limit}
    if category:
        payload["category_filter"] = category

    try:
        response = kestrel_request("POST", "hybrid-search", json=payload)

        # Handle various response formats
        if isinstance(response, list):
            return response
        if isinstance(response, dict):
            # Try direct lookup first
            if query in response:
                result = response[query]
                return result if isinstance(result, list) else [result]
            # Try case-insensitive lookup
            query_lower = query.lower()
            for key, value in response.items():
                if key.lower() == query_lower:
                    return value if isinstance(value, list) else [value]
            # Fallback to common keys
            for fallback in ["results", "data", "items"]:
                if fallback in response:
                    return response[fallback] if isinstance(response[fallback], list) else [response[fallback]]
        return []
    except Exception as e:
        logging.warning(f"Hybrid search failed: {e}")
        return []


def query_with_semantic_expansion(
    query: str,
    predicates: list[str] | None = None,
    search_limit: int = 5,
    relation_limit: int = 20,
    category: str | None = None,
) -> SemanticQueryResult:
    """
    Execute a hybrid query: search for entity, then expand semantically.

    This is the recommended approach for complex queries that need both
    entity resolution and semantic relationship discovery.

    Example workflow:
        1. Search: "glucose" -> resolves to CHEBI:4167
        2. Expand: get_semantic_relations(CHEBI:4167, predicate="participates_in")
        3. Return: Entity info + pathways it participates in

    Args:
        query: Natural language query or entity name
        predicates: Predicates to expand (default: participates_in, associated_with, affects)
        search_limit: Max entities to consider from search
        relation_limit: Max relations per entity
        category: Optional category filter for search

    Returns:
        SemanticQueryResult with resolved_entities and semantic_relations
    """
    if predicates is None:
        predicates = ["participates_in", "associated_with", "affects"]

    # Step 1: Resolve entity via search
    search_results = _hybrid_search(query, limit=search_limit, category=category)

    if not search_results:
        return SemanticQueryResult(
            stats={
                "search_hits": 0,
                "total_relations": 0,
                "predicates_queried": predicates,
            }
        )

    # Step 2: Get semantic relations for top entities
    all_relations: list[dict[str, Any]] = []
    entities_with_relations = 0

    for entity in search_results[:search_limit]:
        entity_id = entity.get("id", "")
        entity_name = entity.get("name", entity_id)

        if not entity_id:
            continue

        entity_relations_found = False

        for predicate in predicates:
            relations = get_semantic_relations(
                entity_id,
                predicate_filter=predicate,
                limit=relation_limit,
            )

            for rel in relations:
                entity_relations_found = True
                all_relations.append(
                    {
                        "source_entity": entity_name,
                        "source_id": entity_id,
                        "predicate": predicate,
                        "target_entity": rel.get("object_name", ""),
                        "target_id": rel.get("object_id", ""),
                        "target_category": rel.get("object_category", ""),
                    }
                )

        if entity_relations_found:
            entities_with_relations += 1

    return SemanticQueryResult(
        resolved_entities=search_results[:search_limit],
        semantic_relations=all_relations,
        stats={
            "search_hits": len(search_results),
            "entities_processed": min(len(search_results), search_limit),
            "entities_with_relations": entities_with_relations,
            "total_relations": len(all_relations),
            "predicates_queried": predicates,
        },
    )


# ==============================================================================
# UTILITY: AVAILABLE PREDICATES
# ==============================================================================


def get_available_predicates() -> dict[str, list[str]]:
    """
    Get available predicates, classified by type.

    Returns:
        Dict with 'semantic' and 'equivalency' keys, each containing
        a list of predicate strings
    """
    return {
        "semantic": sorted(SEMANTIC_PREDICATES),
        "equivalency": sorted(EQUIVALENCY_PREDICATES),
    }
