import logging
from collections import defaultdict
from typing import Dict, Any, Tuple, List

import requests
import pandas as pd

from ..config import KESTREL_API_URL


def link(item: pd.Series | Dict[str, Any]) -> Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, List[str]]]:
    logging.debug(f"Beginning link step (curies-->KG)..")
    curie_to_kg_id_map = get_kg_ids(item['curies'])
    kg_ids, kg_ids_provided, kg_ids_assigned = get_kg_id_fields(item, curie_to_kg_id_map)
    return kg_ids, kg_ids_provided, kg_ids_assigned


def get_kg_ids(curies: List[str]) -> Dict[str, str]:
    curie_to_kg_id_map = dict()
    if curies:
        # Get the canonical curies from the KG  # TODO: expose streamlined get_canonical_ids dict endpoint in kestrel
        try:
            response = requests.post(f"{KESTREL_API_URL}/get-nodes", json={'curies': curies})
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
    # Restructure so canonical IDs are keys, with separate overall vs. provided vs. assigned dicts
    curies = item['curies']
    curies_provided = item['curies_provided']
    curies_assigned = item['curies_assigned']

    kg_ids = _reverse_curie_map(curie_to_kg_id_map, curie_subset=curies)
    kg_ids_provided = _reverse_curie_map(curie_to_kg_id_map, curie_subset=curies_provided)
    kg_ids_assigned = _reverse_curie_map(curie_to_kg_id_map, curie_subset=curies_assigned)

    return kg_ids, kg_ids_provided, kg_ids_assigned


def _reverse_curie_map(curie_map: Dict[str, str], curie_subset: List[str]) -> Dict[str, List[str]]:
    reversed_dict = defaultdict(list)
    for curie in curie_subset:
        if curie in curie_map:  # Curies that didn't match to a KG node won't be in the curie map
            kg_id = curie_map[curie]
            reversed_dict[kg_id].append(curie)
    return dict(reversed_dict)
