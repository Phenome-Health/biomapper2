"""
Curie linking module for mapping to knowledge graph nodes.

Queries the knowledge graph API to find canonical node IDs for normalized curies.
"""
import logging
from collections import defaultdict
from typing import Dict, Any, Tuple, List

import requests
import pandas as pd

from ..config import KESTREL_API_URL, KESTREL_API_KEY


def link(item: pd.Series | Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Link entity curies to knowledge graph node IDs.

    Args:
        item: Entity containing curies

    Returns:
        Tuple of (kg_ids, kg_ids_provided, kg_ids_assigned)
    """
    logging.debug(f"Beginning link step (curies-->KG)..")
    curie_to_kg_id_map = get_kg_ids(item['curies'])
    kg_ids, kg_ids_provided, kg_ids_assigned = get_kg_id_fields(item, curie_to_kg_id_map)
    return kg_ids, kg_ids_provided, kg_ids_assigned


def get_kg_ids(curies: List[str]) -> Dict[str, str]:
    """
    Query knowledge graph API for canonical node IDs.

    Args:
        curies: List of curies to look up

    Returns:
        Dictionary mapping curies to canonical KG node IDs
    """
    curie_to_kg_id_map = dict()
    if curies:
        # Get the canonical curies from the KG  # TODO: expose streamlined get_canonical_ids dict endpoint in kestrel
        try:
            response = requests.post(f"{KESTREL_API_URL}/get-nodes",
                                     json={'curies': curies},
                                     headers={'X-API-Key': KESTREL_API_KEY})
            response.raise_for_status()  # Raises HTTPError for bad status codes
            result = response.json()
            curie_to_kg_id_map = {input_curie: node['id'] for input_curie, node in result.items() if node}
        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error occurred: {e}", exc_info=True)
            # Optional: re-raise if you want calling code to handle it
            raise
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}", exc_info=True)
            raise
    return curie_to_kg_id_map


def get_kg_id_fields(item: pd.Series | Dict[str, Any],
                          curie_to_kg_id_map: Dict[str, str]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    """
    Organize KG IDs by source (overall, provided, assigned) and record their corresponding curie 'votes'.

    Args:
        item: Entity with curie fields
        curie_to_kg_id_map: Mapping from curies to KG node IDs

    Returns:
        Tuple of (kg_ids_dict, kg_ids_provided_dict, kg_ids_assigned_dict)
    """
    curies = item['curies']
    curies_provided = item['curies_provided']
    curies_assigned = item['curies_assigned']

    kg_ids = _reverse_curie_map(curie_to_kg_id_map, curie_subset=curies)
    kg_ids_provided = _reverse_curie_map(curie_to_kg_id_map, curie_subset=curies_provided)
    kg_ids_assigned = _reverse_curie_map(curie_to_kg_id_map, curie_subset=curies_assigned)

    return kg_ids, kg_ids_provided, kg_ids_assigned


def _reverse_curie_map(curie_map: Dict[str, str], curie_subset: List[str]) -> Dict[str, List[str]]:
    """
    Reverse curie-to-kg-id mapping for a subset of curies.

    Args:
        curie_map: Dictionary mapping curies to KG IDs
        curie_subset: Subset of curies to include

    Returns:
        Dictionary mapping KG IDs to lists of curies
    """
    reversed_dict = defaultdict(list)
    for curie in curie_subset:
        if curie in curie_map:  # Curies that didn't match to a KG node won't be in the curie map
            kg_id = curie_map[curie]
            reversed_dict[kg_id].append(curie)
    return dict(reversed_dict)
