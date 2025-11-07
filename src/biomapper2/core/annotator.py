"""
Entity annotation module for assigning ontology local IDs.

Queries external APIs or uses other creative approaches to retrieve additional identifiers for biological entities.
"""
import logging
from typing import Dict, Any, List, Set

import pandas as pd


def annotate(item: pd.Series | Dict[str, Any],
             name_field: str,
             provided_id_fields: List[str],
             entity_type: str) -> Dict[str, str]:
    """
    Annotate entity with additional local IDs, obtained using various internal or external methods.

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

    # TODO: add in different annotation submodules/methods..

    return assigned_ids

