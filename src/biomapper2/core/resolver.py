import logging
from typing import Dict, Any


def resolve(entity: Dict[str, Any]) -> Dict[str, Any]:
    logging.debug(f"Beginning one-to-many resolution step..")

    if entity.get('kg_ids'):
        # For now, use a voting approach # TODO: later use more advanced methods, like depending on source/using LLMs/other
        kg_ids = entity['kg_ids']
        majority_kg_id = max(kg_ids, key=lambda k: len(kg_ids[k]))
        entity['chosen_kg_id'] = majority_kg_id

    return entity
