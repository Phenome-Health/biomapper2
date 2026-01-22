"""
KG-o1 v3 Utilities Module

Production-ready helper functions for semantic multi-hop reasoning in KRAKEN.
v3 focuses on TRUE semantic relations via the /one-hop endpoint, unlike v2's
vocabulary-transition approach.

Key Components:
1. One-hop API wrapper with predicate filtering
2. Predicate classification (semantic vs equivalency)
3. BFS path finding with explosion safeguards
4. External validation (Reactome/KEGG)
5. Semantic QA pair generation

Author: Generated for biomapper2 project
"""

import json
import logging
import sys
import time
import urllib.parse
from collections import deque
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import requests

# Add biomapper2 to path for imports
PROJECT_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from biomapper2.utils import kestrel_request

# ============================================================================
# CONSTANTS & TERMINOLOGY
# ============================================================================

TERMINOLOGY = {
    'semantic_hop': (
        "Entity-relation-entity traversal in a typed knowledge graph. "
        "Example: glucose --participates_in--> Glycolysis"
    ),
    'vocab_transition': (
        "Same entity with different identifiers across databases. "
        "Example: CHEBI:4167 == HMDB:HMDB0000122 (both are glucose)"
    ),
    'v2_vs_v3': (
        "v2: Tested vocabulary transitions (95.2% EM on entity resolution). "
        "v3: Tests semantic relations via /one-hop endpoint."
    ),
}

# Predicate classification - CRITICAL for determining if KRAKEN has semantic capability
EQUIVALENCY_PREDICATES = {
    'biolink:same_as', 'owl:sameAs', 'skos:exactMatch',
    'oboInOwl:hasDbXref', 'biolink:xref', 'equivalent_to',
    'biolink:equivalent_to', 'same_as', 'skos:closeMatch',
    'biolink:has_equivalent_class',
}

SEMANTIC_PREDICATES = {
    # Participation
    'biolink:participates_in', 'biolink:has_participant',
    'biolink:actively_involved_in', 'biolink:capable_of',
    # Catalysis/Metabolism
    'biolink:catalyzes', 'biolink:is_substrate_of',
    'biolink:has_metabolite', 'biolink:metabolite_of',
    'biolink:affects', 'biolink:affected_by',
    'biolink:has_output', 'biolink:has_input',
    # Association
    'biolink:associated_with', 'biolink:related_to',
    'biolink:correlated_with', 'biolink:coexists_with',
    # Causal
    'biolink:causes', 'biolink:caused_by',
    'biolink:contributes_to', 'biolink:ameliorates',
    # Therapeutic / Clinical
    'biolink:treats', 'biolink:treated_by',
    'biolink:prevents', 'biolink:predisposes',
    'biolink:applied_to_treat', 'biolink:in_clinical_trials_for',
    'biolink:mentioned_in_clinical_trials_for',
    # Structural / Ontological
    'biolink:part_of', 'biolink:has_part',
    'biolink:component_of', 'biolink:contains',
    'biolink:subclass_of', 'biolink:superclass_of',
    'biolink:has_chemical_role',
    # Location
    'biolink:located_in', 'biolink:location_of',
    'biolink:expressed_in', 'biolink:expresses',
    # Similarity (semantic, not equivalency)
    'biolink:chemically_similar_to', 'biolink:close_match',
}

# Priority predicates for BFS expansion (explore these first)
PRIORITY_PREDICATES = {
    'biolink:participates_in', 'biolink:catalyzes', 'biolink:treats',
    'biolink:associated_with', 'biolink:causes', 'biolink:has_participant',
    'biolink:affects', 'biolink:metabolite_of',
}

LOW_PRIORITY_PREDICATES = EQUIVALENCY_PREDICATES

# BFS Safeguards
MAX_VISITED_NODES = 1000
TIMEOUT_SECONDS = 30
MAX_QUEUE_SIZE = 5000

# Minimum samples for statistical claims
MIN_SAMPLES_FOR_CI = 5
MIN_SAMPLES_FOR_CLAIMS = 10
MIN_SAMPLES_FOR_RELIABLE = 30

# ============================================================================
# DATA STRUCTURES
# ============================================================================


@dataclass
class SemanticTriple:
    """A single entity-relation-entity relationship."""
    subject_id: str
    subject_name: str
    predicate: str
    object_id: str
    object_name: str
    object_category: str = ""
    source: str = "kestrel"  # kestrel, reactome, kegg


@dataclass
class SemanticSubgraph:
    """A subgraph centered on an entity with semantic relations."""
    center_entity_id: str
    center_entity_name: str
    center_entity_category: str
    outgoing_relations: list[SemanticTriple] = field(default_factory=list)
    incoming_relations: list[SemanticTriple] = field(default_factory=list)

    def all_relations(self) -> list[SemanticTriple]:
        return self.outgoing_relations + self.incoming_relations

    def to_dict(self) -> dict:
        return {
            'center_entity_id': self.center_entity_id,
            'center_entity_name': self.center_entity_name,
            'center_entity_category': self.center_entity_category,
            'outgoing_relations': [asdict(r) for r in self.outgoing_relations],
            'incoming_relations': [asdict(r) for r in self.incoming_relations],
        }


@dataclass
class MultiHopPath:
    """A multi-hop path between two entities."""
    start_id: str
    end_id: str
    path: list[SemanticTriple]
    num_hops: int
    stats: dict = field(default_factory=dict)
    termination_reason: str = ""

    def to_dict(self) -> dict:
        return {
            'start_id': self.start_id,
            'end_id': self.end_id,
            'path': [asdict(t) for t in self.path],
            'num_hops': self.num_hops,
            'stats': self.stats,
            'termination_reason': self.termination_reason,
        }


@dataclass
class SemanticQAPair:
    """A semantic QA pair with validation."""
    question: str
    answer: str
    answer_id: str
    source_entity_id: str
    source_entity_name: str
    reasoning_chain: list[dict]  # List of {subject, predicate, object}
    num_hops: int
    qa_type: str  # semantic_1_hop, semantic_2_hop, path_finding, intersection
    validation: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PathFindingResult:
    """Result of a BFS path-finding operation."""
    path: list[SemanticTriple]
    stats: dict
    termination_reason: str  # 'found', 'max_visited_reached', 'timeout', 'queue_overflow', 'exhausted'


# ============================================================================
# ONE-HOP API WRAPPER
# ============================================================================


def test_one_hop(
    start_node_id: str,
    direction: str = "both",
    predicate_filter: str | None = None,
    end_category_filter: str | None = None,
    limit: int = 20,
    mode: str = "slim",
) -> dict:
    """
    Test the /one-hop endpoint for semantic graph traversal.

    This is the KEY function for v3 - if it returns semantic relations,
    KRAKEN has multi-hop reasoning capability.

    Args:
        start_node_id: Starting node ID (e.g., "CHEBI:4167")
        direction: "forward", "reverse", or "both"
        predicate_filter: Optional predicate to filter by (e.g., "participates_in")
        end_category_filter: Optional category filter (e.g., "Pathway")
        limit: Maximum number of results
        mode: "slim", "full", or "preview"

    Returns:
        API response dictionary with keys: edge_schema, results, nodes
        Results are grouped by end_node_id, each containing edges as tuples.
    """
    payload = {
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
    except requests.exceptions.HTTPError as e:
        if e.response.status_code in (400, 404, 422):
            return {"error": f"api_error_{e.response.status_code}", "status_code": e.response.status_code}
        raise


def parse_one_hop_edges(response: dict) -> list[dict]:
    """
    Parse one-hop response into a flat list of edge dictionaries.

    The API returns edges grouped by end_node_id with edges as tuples.
    This function flattens and converts to dicts for easier processing.

    Args:
        response: Raw response from test_one_hop()

    Returns:
        List of edge dicts with keys: subject_id, predicate, object_id,
        object_name, object_category, etc.
    """
    if not isinstance(response, dict) or 'error' in response:
        return []

    schema = response.get('edge_schema', [])
    results = response.get('results', [])
    nodes = response.get('nodes', {})  # Node info lookup

    parsed_edges = []

    for result in results:
        end_node_id = result.get('end_node_id', '')
        edges = result.get('edges', [])

        # Get node info for this end_node
        node_info = nodes.get(end_node_id, {})
        node_name = node_info.get('name', '')
        node_category = node_info.get('category', '')

        for edge_tuple in edges:
            # Convert tuple to dict using schema
            edge_dict = {}
            for i, field in enumerate(schema):
                if i < len(edge_tuple):
                    edge_dict[field] = edge_tuple[i]

            # Add friendly names
            edge_dict['subject_id'] = edge_dict.get('subject', '')
            edge_dict['object_id'] = edge_dict.get('object', '')
            edge_dict['end_node_id'] = end_node_id
            edge_dict['object_name'] = node_name
            edge_dict['end_node_name'] = node_name
            edge_dict['object_category'] = node_category
            edge_dict['category'] = node_category

            parsed_edges.append(edge_dict)

    return parsed_edges


def get_semantic_edges(
    entity_id: str,
    direction: str = "both",
    predicate_filter: str | None = None,
    limit: int = 50,
) -> list[dict]:
    """
    Get semantic edges for an entity, filtering out equivalency predicates.

    This is a convenience wrapper around test_one_hop + parse_one_hop_edges
    that only returns edges with semantic predicates.

    Args:
        entity_id: Entity ID (e.g., "CHEBI:4167")
        direction: "forward", "reverse", or "both"
        predicate_filter: Optional specific predicate to filter by
        limit: Maximum results

    Returns:
        List of semantic edge dictionaries
    """
    response = test_one_hop(entity_id, direction, predicate_filter, limit=limit)
    all_edges = parse_one_hop_edges(response)

    # Filter to semantic predicates only
    semantic_edges = []
    for edge in all_edges:
        predicate = edge.get('predicate', '')
        classification = classify_predicate(predicate)
        if classification in ('semantic', 'unknown'):
            semantic_edges.append(edge)

    return semantic_edges


def get_predicates() -> list[str]:
    """
    Fetch all available predicates from the Kestrel API.

    Returns:
        List of predicate strings
    """
    try:
        response = kestrel_request("GET", "predicates")
        if isinstance(response, list):
            return response
        elif isinstance(response, dict):
            return response.get('predicates', [])
        return []
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            logging.warning("GET /predicates endpoint not found")
            return []
        raise


def get_edges(
    node_id: str,
    direction: str = "both",
    limit: int = 50,
) -> list[dict]:
    """
    Get edges connected to a node.

    Args:
        node_id: Node ID
        direction: "forward", "reverse", or "both"
        limit: Maximum edges to return

    Returns:
        List of edge dictionaries
    """
    try:
        payload = {
            "node_id": node_id,
            "direction": direction,
            "limit": limit,
        }
        response = kestrel_request("POST", "get-edges", json=payload)
        if isinstance(response, list):
            return response
        elif isinstance(response, dict):
            return response.get('edges', [])
        return []
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return []
        raise


# ============================================================================
# PREDICATE CLASSIFICATION
# ============================================================================


def classify_predicate(predicate: str) -> str:
    """
    Classify a predicate as 'semantic', 'equivalency', or 'unknown'.

    Args:
        predicate: The predicate string

    Returns:
        Classification string
    """
    pred_lower = predicate.lower()

    # Normalize to check both with and without biolink prefix
    variants = [predicate, pred_lower]
    if not predicate.startswith('biolink:'):
        variants.append(f'biolink:{predicate}')
        variants.append(f'biolink:{pred_lower}')

    for variant in variants:
        if variant in SEMANTIC_PREDICATES or variant.lower() in {p.lower() for p in SEMANTIC_PREDICATES}:
            return 'semantic'
        if variant in EQUIVALENCY_PREDICATES or variant.lower() in {p.lower() for p in EQUIVALENCY_PREDICATES}:
            return 'equivalency'

    return 'unknown'


def classify_all_predicates(predicates: list[str]) -> dict[str, list[str]]:
    """
    Classify a list of predicates.

    Args:
        predicates: List of predicate strings

    Returns:
        Dictionary with 'semantic', 'equivalency', 'unknown' keys
    """
    result = {'semantic': [], 'equivalency': [], 'unknown': []}
    for pred in predicates:
        classification = classify_predicate(pred)
        result[classification].append(pred)
    return result


# ============================================================================
# SEARCH FUNCTIONS (inherited from v2)
# ============================================================================


def _normalize_key(key: str) -> str:
    """Normalize a key for case-insensitive matching."""
    return key.lower().replace("-", " ").replace("_", " ").strip()


def _robust_get_results(response: dict | list, query: str) -> list[dict]:
    """
    Robustly extract results from Kestrel API response.
    Fixes the vector search 0% recall bug from v2.
    """
    if isinstance(response, list):
        return response
    if not isinstance(response, dict):
        return []

    # Direct lookup
    if query in response:
        return response[query] if isinstance(response[query], list) else [response[query]]

    # Case-insensitive lookup
    query_lower = query.lower()
    for key, value in response.items():
        if key.lower() == query_lower:
            return value if isinstance(value, list) else [value]

    # Normalized lookup
    query_norm = _normalize_key(query)
    for key, value in response.items():
        if _normalize_key(key) == query_norm:
            return value if isinstance(value, list) else [value]

    # Fallback keys
    for fallback_key in ['results', 'data', 'items', 'entities']:
        if fallback_key in response:
            fallback_value = response[fallback_key]
            return fallback_value if isinstance(fallback_value, list) else [fallback_value]

    return []


def hybrid_search(query: str, limit: int = 10, category: str | None = None) -> list[dict]:
    """Combined text + vector search via Kestrel API."""
    payload = {'search_text': query, 'limit': limit}
    if category:
        payload['category_filter'] = category

    try:
        response = kestrel_request('POST', 'hybrid-search', json=payload)
        return _robust_get_results(response, query)
    except Exception as e:
        logging.warning(f"Hybrid search failed: {e}")
        return []


def text_search(query: str, limit: int = 10, category: str | None = None) -> list[dict]:
    """BM25-based lexical search via Kestrel API."""
    payload = {'search_text': query, 'limit': limit}
    if category:
        payload['category_filter'] = category

    try:
        response = kestrel_request('POST', 'text-search', json=payload)
        return _robust_get_results(response, query)
    except Exception as e:
        logging.warning(f"Text search failed: {e}")
        return []


# ============================================================================
# BFS PATH FINDING WITH EXPLOSION SAFEGUARDS
# ============================================================================


def find_path_bfs(
    start_id: str,
    end_id: str,
    max_hops: int = 3,
    max_visited: int = MAX_VISITED_NODES,
    timeout_sec: float = TIMEOUT_SECONDS,
    semantic_only: bool = False,
) -> PathFindingResult:
    """
    BFS to find semantic path between entities with explosion safeguards.

    Args:
        start_id: Starting entity ID
        end_id: Target entity ID
        max_hops: Maximum path length
        max_visited: Maximum nodes to visit before stopping
        timeout_sec: Maximum time in seconds
        semantic_only: If True, skip equivalency edges

    Returns:
        PathFindingResult with path, stats, and termination reason
    """
    start_time = time.time()
    visited = {start_id}
    queue = deque([(start_id, [])])  # (node, path_so_far)
    stats = {
        'nodes_visited': 0,
        'api_calls': 0,
        'edges_examined': 0,
        'semantic_edges': 0,
        'equivalency_edges': 0,
    }

    while queue:
        # Check safeguards
        if len(visited) >= max_visited:
            return PathFindingResult(
                path=[],
                stats=stats,
                termination_reason='max_visited_reached'
            )

        if time.time() - start_time > timeout_sec:
            return PathFindingResult(
                path=[],
                stats=stats,
                termination_reason='timeout'
            )

        if len(queue) > MAX_QUEUE_SIZE:
            return PathFindingResult(
                path=[],
                stats=stats,
                termination_reason='queue_overflow'
            )

        current, path = queue.popleft()
        stats['nodes_visited'] += 1

        if len(path) >= max_hops:
            continue

        # Get neighbors via one-hop
        try:
            response = test_one_hop(current, direction="both")
            stats['api_calls'] += 1
        except Exception as e:
            logging.warning(f"One-hop failed for {current}: {e}")
            continue

        # Parse edges using the proper parser (fixes predicate metadata loss)
        neighbors = parse_one_hop_edges(response)
        if not neighbors:
            continue

        # Sort edges: semantic predicates first
        priority_edges = []
        low_priority_edges = []

        for edge in neighbors:
            pred = edge.get('predicate', '')
            stats['edges_examined'] += 1

            classification = classify_predicate(pred)
            if classification == 'semantic':
                stats['semantic_edges'] += 1
            elif classification == 'equivalency':
                stats['equivalency_edges'] += 1

            if semantic_only and classification == 'equivalency':
                continue

            if pred in PRIORITY_PREDICATES:
                priority_edges.append(edge)
            else:
                low_priority_edges.append(edge)

        # Process edges
        for edge in priority_edges + low_priority_edges:
            neighbor_id = edge.get('object_id') or edge.get('end_node_id') or edge.get('target_id')

            if not neighbor_id:
                continue

            # Create triple for path
            triple = SemanticTriple(
                subject_id=current,
                subject_name=edge.get('subject_name', current),
                predicate=edge.get('predicate', 'unknown'),
                object_id=neighbor_id,
                object_name=edge.get('object_name', neighbor_id),
                object_category=edge.get('object_category', ''),
            )

            if neighbor_id == end_id:
                return PathFindingResult(
                    path=path + [triple],
                    stats=stats,
                    termination_reason='found'
                )

            if neighbor_id not in visited:
                visited.add(neighbor_id)
                queue.append((neighbor_id, path + [triple]))

    return PathFindingResult(
        path=[],
        stats=stats,
        termination_reason='exhausted'
    )


# ============================================================================
# EXTERNAL VALIDATION (Reactome/KEGG)
# ============================================================================

_last_external_request_time = 0
_MIN_REQUEST_INTERVAL = 0.15  # 150ms between external requests


def _rate_limited_get(url: str, timeout: int = 10) -> requests.Response | None:
    """Rate-limited HTTP GET for external APIs."""
    global _last_external_request_time

    elapsed = time.time() - _last_external_request_time
    if elapsed < _MIN_REQUEST_INTERVAL:
        time.sleep(_MIN_REQUEST_INTERVAL - elapsed)
    _last_external_request_time = time.time()

    try:
        response = requests.get(url, timeout=timeout)
        return response
    except Exception as e:
        logging.warning(f"External request failed: {url} - {e}")
        return None


@lru_cache(maxsize=1000)
def validate_with_reactome(metabolite_chebi: str, expected_pathway: str) -> dict:
    """
    Validate metabolite-pathway relationship against Reactome.

    Args:
        metabolite_chebi: CHEBI ID (e.g., "CHEBI:4167")
        expected_pathway: Expected pathway identifier

    Returns:
        Validation result dictionary
    """
    encoded_id = urllib.parse.quote(metabolite_chebi)
    url = f"https://reactome.org/ContentService/data/participants/{encoded_id}/pathways"

    response = _rate_limited_get(url)
    if response is None or response.status_code != 200:
        return {'validated': None, 'reason': 'api_error', 'source': 'reactome'}

    try:
        pathways = response.json()
        pathway_ids = [p.get('stId', '') for p in pathways]
        pathway_names = [p.get('displayName', '').lower() for p in pathways]

        # Check if expected pathway matches any result
        expected_lower = expected_pathway.lower()
        validated = (
            expected_pathway in pathway_ids or
            any(expected_lower in name for name in pathway_names)
        )

        return {
            'validated': validated,
            'source': 'reactome',
            'num_pathways_found': len(pathway_ids),
            'sample_pathways': pathway_ids[:5],
        }
    except Exception as e:
        return {'validated': None, 'reason': str(e), 'source': 'reactome'}


@lru_cache(maxsize=1000)
def validate_with_kegg(compound_id: str, expected_pathway: str) -> dict:
    """
    Validate compound-pathway relationship against KEGG.

    Args:
        compound_id: KEGG compound ID (e.g., "C00031" or "KEGG.COMPOUND:C00031")
        expected_pathway: Expected pathway identifier

    Returns:
        Validation result dictionary
    """
    # Extract just the compound ID
    if ':' in compound_id:
        compound_id = compound_id.split(':')[-1]

    url = f"https://rest.kegg.jp/link/pathway/cpd:{compound_id}"

    response = _rate_limited_get(url)
    if response is None or response.status_code != 200:
        return {'validated': None, 'reason': 'api_error', 'source': 'kegg'}

    try:
        pathways = []
        for line in response.text.strip().split('\n'):
            if '\t' in line:
                _, pathway_id = line.split('\t')
                pathways.append(pathway_id)

        validated = any(expected_pathway in p for p in pathways)

        return {
            'validated': validated,
            'source': 'kegg',
            'num_pathways_found': len(pathways),
            'sample_pathways': pathways[:5],
        }
    except Exception as e:
        return {'validated': None, 'reason': str(e), 'source': 'kegg'}


def validate_qa_pair(qa_pair: SemanticQAPair) -> SemanticQAPair:
    """
    Validate a QA pair against external sources.

    Args:
        qa_pair: The QA pair to validate

    Returns:
        Updated QA pair with validation results
    """
    entity_id = qa_pair.source_entity_id
    expected = qa_pair.answer_id

    validations = {}

    # Try Reactome validation
    if entity_id.startswith('CHEBI:'):
        validations['reactome'] = validate_with_reactome(entity_id, expected)

    # Try KEGG validation
    if 'KEGG' in entity_id.upper() or entity_id.startswith('C'):
        kegg_id = entity_id.split(':')[-1] if ':' in entity_id else entity_id
        validations['kegg'] = validate_with_kegg(kegg_id, expected)

    qa_pair.validation = {
        'sources_checked': list(validations.keys()),
        'confirmed_by': [k for k, v in validations.items() if v.get('validated')],
        'is_valid': any(v.get('validated') for v in validations.values()),
        'details': validations,
    }

    return qa_pair


# ============================================================================
# SEMANTIC QA GENERATION
# ============================================================================

QA_TEMPLATES = {
    'direct_relation': [
        ("What {category} does {entity} {predicate}?", 1),
        ("Which {category} is {entity} associated with?", 1),
        ("What does {entity} participate in?", 1),
    ],
    'path_finding': [
        ("Through what does {entity1} connect to {entity2}?", 2),
        ("How is {entity1} related to {entity2}?", 2),
    ],
    'intersection': [
        ("What is related to both {entity1} and {entity2}?", 2),
        ("What do {entity1} and {entity2} have in common?", 2),
    ],
}


def generate_qa_from_triple(triple: SemanticTriple) -> SemanticQAPair | None:
    """
    Generate a QA pair from a semantic triple.

    Args:
        triple: A SemanticTriple

    Returns:
        SemanticQAPair or None if cannot generate
    """
    # Map predicate to natural language
    pred_to_verb = {
        'biolink:participates_in': 'participate in',
        'biolink:has_participant': 'have as participant',
        'biolink:catalyzes': 'catalyze',
        'biolink:treats': 'treat',
        'biolink:causes': 'cause',
        'biolink:associated_with': 'associate with',
        'biolink:part_of': 'be part of',
        'biolink:located_in': 'be located in',
    }

    verb = pred_to_verb.get(triple.predicate, triple.predicate.replace('biolink:', '').replace('_', ' '))
    category = triple.object_category or 'entity'

    question = f"What {category} does {triple.subject_name} {verb}?"

    return SemanticQAPair(
        question=question,
        answer=triple.object_name,
        answer_id=triple.object_id,
        source_entity_id=triple.subject_id,
        source_entity_name=triple.subject_name,
        reasoning_chain=[{
            'subject': triple.subject_name,
            'predicate': triple.predicate,
            'object': triple.object_name,
        }],
        num_hops=1,
        qa_type='semantic_1_hop',
    )


def generate_qa_from_path(path: list[SemanticTriple]) -> SemanticQAPair | None:
    """
    Generate a QA pair from a multi-hop path.

    Args:
        path: List of SemanticTriples forming a path

    Returns:
        SemanticQAPair or None if cannot generate
    """
    if not path:
        return None

    start = path[0]
    end = path[-1]

    question = f"How is {start.subject_name} connected to {end.object_name}?"

    reasoning_chain = [
        {
            'subject': t.subject_name,
            'predicate': t.predicate,
            'object': t.object_name,
        }
        for t in path
    ]

    return SemanticQAPair(
        question=question,
        answer=end.object_name,
        answer_id=end.object_id,
        source_entity_id=start.subject_id,
        source_entity_name=start.subject_name,
        reasoning_chain=reasoning_chain,
        num_hops=len(path),
        qa_type=f'semantic_{len(path)}_hop',
    )


# ============================================================================
# EVALUATION METRICS
# ============================================================================


def find_answer_rank(results: list[dict], answer_id: str) -> int:
    """
    Find the rank of the correct answer in search results.

    Args:
        results: List of search result dictionaries
        answer_id: Expected answer ID

    Returns:
        1-indexed rank if found, 0 if not found
    """
    answer_lower = answer_id.lower()

    for i, result in enumerate(results, 1):
        result_id = result.get('id', '').lower()
        equiv_ids = [eid.lower() for eid in result.get('equivalent_ids', [])]

        if answer_lower == result_id or answer_lower in equiv_ids:
            return i

    return 0


def compute_recall_at_k(ranks: list[int], k: int) -> float:
    """Compute Recall@k: fraction of queries with answer in top k."""
    if not ranks:
        return 0.0
    hits = sum(1 for r in ranks if 0 < r <= k)
    return hits / len(ranks)


def compute_mrr(ranks: list[int]) -> float:
    """Compute Mean Reciprocal Rank."""
    if not ranks:
        return 0.0
    reciprocals = [1.0 / r if r > 0 else 0.0 for r in ranks]
    return sum(reciprocals) / len(reciprocals)


def compute_exact_match(ranks: list[int]) -> float:
    """Compute Exact Match: fraction with answer at rank 1."""
    return compute_recall_at_k(ranks, 1)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def save_json(data: Any, path: Path | str) -> None:
    """Save data to JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(data, f, indent=2, default=str)


def load_json(path: Path | str) -> Any:
    """Load data from JSON file."""
    with open(path) as f:
        return json.load(f)


def assess_statistical_validity(n: int) -> dict:
    """Assess statistical validity based on sample size."""
    return {
        'n': n,
        'can_compute_ci': n >= MIN_SAMPLES_FOR_CI,
        'per_group_valid': n >= MIN_SAMPLES_FOR_CLAIMS,
        'fully_reliable': n >= MIN_SAMPLES_FOR_RELIABLE,
        'interpretation': (
            'Results are illustrative only' if n < MIN_SAMPLES_FOR_CI
            else 'CIs computed but interpret with caution' if n < MIN_SAMPLES_FOR_CLAIMS
            else 'Per-group claims valid' if n < MIN_SAMPLES_FOR_RELIABLE
            else 'Statistically reliable claims supported'
        ),
    }


# ============================================================================
# SEED ENTITIES FOR TESTING
# ============================================================================

# Known entities for testing one-hop
TEST_ENTITIES = [
    ("CHEBI:4167", "glucose"),
    ("CHEBI:15846", "NAD+"),
    ("CHEBI:16113", "cholesterol"),
    ("CHEBI:17234", "alanine"),
    ("CHEBI:30616", "ATP"),
    ("CHEBI:15377", "water"),
]

# Expected semantic relations (for validation)
EXPECTED_SEMANTIC_RELATIONS = {
    "glucose": ["Glycolysis", "Gluconeogenesis", "Pentose phosphate pathway"],
    "ATP": ["Oxidative phosphorylation", "Glycolysis"],
    "cholesterol": ["Steroid biosynthesis"],
}
