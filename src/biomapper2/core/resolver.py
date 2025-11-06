import logging
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd


def resolve(item: pd.Series | Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    logging.debug(f"Beginning one-to-many resolution step..")

    chosen_kg_id = _get_chosen_kg_id(item['kg_ids'])
    chosen_kg_id_provided = _get_chosen_kg_id(item['kg_ids_provided'])
    chosen_kg_id_assigned = _get_chosen_kg_id(item['kg_ids_assigned'])

    return chosen_kg_id, chosen_kg_id_provided, chosen_kg_id_assigned


def _get_chosen_kg_id(kg_ids_dict: Dict[str, List[str]]) -> Optional[str]:
    # For now, use a voting approach # TODO: later use more advanced methods, like depending on source/using LLMs/other
    if kg_ids_dict:
        majority_kg_id = max(kg_ids_dict, key=lambda k: len(kg_ids_dict[k]))
        return majority_kg_id
    else:
        return None
