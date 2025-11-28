"""
One-to-many resolution module for selecting single KG nodes.

Resolves cases where multiple KG nodes match an entity by selecting the best candidate.
"""

import logging
from typing import Dict, Any, List, Optional

import pandas as pd


class Resolver:
    """Resolves one-to-many KG mappings to single chosen nodes."""

    def resolve(self, item: pd.Series | Dict[str, Any] | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Resolve one-to-many KG mappings to single chosen node.

        Args:
            item: Entity or entities with kg_ids fields

        Returns:
            Named Series for single entity or DataFrame for multiple entities,
            containing fields: chosen_kg_id, chosen_kg_id_provided, chosen_kg_id_assigned
        """
        logging.debug(f"Beginning one-to-many resolution step..")

        if isinstance(item, pd.DataFrame):
            return item.apply(self._resolve_entity, axis=1, result_type="expand")
        else:
            return self._resolve_entity(item)

    def _resolve_entity(self, entity: pd.Series | Dict[str, Any]) -> pd.Series:
        """
        Resolve one-to-many KG mappings for a single entity.

        Args:
            entity: Entity with kg_ids fields

        Returns:
            Named Series with fields: chosen_kg_id, chosen_kg_id_provided, chosen_kg_id_assigned
        """
        chosen_kg_id = self._choose_best_kg_id(entity["kg_ids"])
        chosen_kg_id_provided = self._choose_best_kg_id(entity["kg_ids_provided"])
        chosen_kg_id_assigned = self._choose_best_kg_id(entity["kg_ids_assigned"])

        return pd.Series(
            {
                "chosen_kg_id": chosen_kg_id,
                "chosen_kg_id_provided": chosen_kg_id_provided,
                "chosen_kg_id_assigned": chosen_kg_id_assigned,
            }
        )

    @staticmethod
    def _choose_best_kg_id(kg_ids_dict: Dict[str, List[str]]) -> Optional[str]:
        """
        Select single KG ID from multiple candidates using voting.

        Args:
            kg_ids_dict: Dictionary mapping KG IDs to supporting curies

        Returns:
            KG ID with most supporting curies, or None if no candidates
        """
        # For now, use a voting approach # TODO: later use more advanced methods, like depending on source/using LLMs/other
        if kg_ids_dict:
            majority_kg_id = max(kg_ids_dict, key=lambda k: len(kg_ids_dict[k]))
            return majority_kg_id
        else:
            return None
