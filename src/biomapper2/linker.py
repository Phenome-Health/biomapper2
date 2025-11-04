import logging
from collections import defaultdict
from typing import Dict, Any

import requests


def link(entity: Dict[str, Any]) -> Dict[str, Any]:
    curies = entity['curies']
    if curies:
        # Get the canonical curies from the KG  # TODO: expose streamlined get_canonical_ids dict endpoint in kestrel
        try:
            response = requests.post('https://kestrel.nathanpricelab.com/api/get-nodes', json={'curies': curies})  # TODO: move kestrel url to config
            response.raise_for_status()  # Raises HTTPError for bad status codes
            result = response.json()
            input_to_canonical_map = {input_curie: node['id'] for input_curie, node in result.items() if node}

            # Restructure so canonical IDs are keys
            canonical_to_inputs_map = defaultdict(list)
            for input_curie, canonical_id in input_to_canonical_map.items():
                canonical_to_inputs_map[canonical_id].append(input_curie)

            entity['kg_ids'] = canonical_to_inputs_map

        except requests.exceptions.HTTPError as e:
            logging.error(f"HTTP error occurred: {e}", exc_info=True)
            # Optional: re-raise if you want calling code to handle it
            raise
        except requests.exceptions.RequestException as e:
            logging.error(f"Request failed: {e}", exc_info=True)
            raise

    return entity
