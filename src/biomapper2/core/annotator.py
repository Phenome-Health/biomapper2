import logging
from typing import Dict, Any, List, Set

import pandas as pd


def annotate(item: pd.Series | Dict[str, Any],
             name_field: str,
             provided_id_fields: List[str],
             entity_type: str) -> Dict[str, str]:
    logging.debug(f"Beginning annotation step..")
    entity_type_cleaned = ''.join(c for c in entity_type.lower() if c.isalpha())

    assigned_ids: Dict[str, str] = dict()  # All annotations (aka assigned IDs) should go in here
    # TODO: later organize by source..? need to keep straight when multiple annotators adding

    if entity_type_cleaned in {'metabolite', 'smallmolecule', 'lipid'}:
        # TODO: call metabolomics workbench api, passing in name...
        pass

    # TODO: add in different annotation submodules/methods..

    return assigned_ids

