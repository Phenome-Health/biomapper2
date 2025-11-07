"""
Entity annotation module for assigning ontology local IDs.

Queries external APIs or uses other creative approaches to retrieve additional identifiers for biological entities.
"""
import logging
from typing import Dict, Any, List, Set, Optional

import pandas as pd
import requests

from ..config import KESTREL_API_URL


def annotate(item: pd.Series | Dict[str, Any],
             name_field: str,
             provided_id_fields: List[str],
             entity_type: str) -> Dict[str, str]:
    """
    Annotate entity with additional local IDs, obtained using various internal or external methods.

    NOTE: This is a placeholder!!

    Args:
        item: Entity to annotate
        name_field: Field containing entity name
        provided_id_fields: Fields containing existing IDs
        entity_type: Type of entity (e.g., 'metabolite', 'protein')

    Returns:
        Dictionary of assigned identifiers by source
    """
    logging.debug(f"Beginning annotation step..")
    entity_type_cleaned = ''.join(c for c in entity_type.lower() if c.isalpha())

    assigned_ids: Dict[str, str] = dict()  # All annotations (aka assigned IDs) should go in here
    # TODO: later organize by source..? need to keep straight when multiple annotators adding

    if entity_type_cleaned in {'metabolite', 'smallmolecule', 'lipid'}:
        # TODO: call metabolomics workbench api, passing in name...
        pass

    if entity_type_cleaned in {'disease', 'phenotypicfeature'} and item.get(name_field):
        curie = run_kestrel_fuzzy_search(item[name_field])
        if curie:
            vocab_prefix = curie.split(':')[0]
            assigned_ids[vocab_prefix] = curie


    # TODO: add in different annotation submodules/methods..

    return assigned_ids


def run_kestrel_fuzzy_search(search_text: str) -> Optional[str]:
    try:
        response = requests.post(f"{KESTREL_API_URL}/search-node", json={'search_term': search_text, 'limit': 1})
        response.raise_for_status()  # Raises HTTPError for bad status codes
        result = response.json()
        if result:
            return result[0]['id']
    except requests.exceptions.HTTPError as e:
        logging.error(f"HTTP error occurred: {e}", exc_info=True)
        # Optional: re-raise if you want calling code to handle it
        raise
    except requests.exceptions.RequestException as e:
        logging.error(f"Request failed: {e}", exc_info=True)
        raise
    return None

