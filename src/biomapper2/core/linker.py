"""
Curie linking module for mapping to knowledge graph nodes.

Queries the knowledge graph API to find canonical node IDs for normalized curies.
"""

import logging
from collections import defaultdict
from typing import Any

import pandas as pd

from ..utils import kestrel_request


class Linker:
    """Links normalized curies to knowledge graph node IDs."""

    def link(self, item: pd.Series | dict[str, Any] | pd.DataFrame) -> pd.Series | pd.DataFrame:
        """
        Link entity curies to knowledge graph node IDs.

        Args:
            item: Entity or entities containing curies

        Returns:
            Named Series for single entity or DataFrame for multiple entities,
            containing fields: kg_ids, kg_ids_provided, kg_ids_assigned
        """
        logging.debug("Beginning link step (curies-->KG)..")

        if isinstance(item, pd.DataFrame):
            return self._link_dataframe(item)
        else:
            return self._link_entity(item)

    def _link_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Link curies to KG IDs for all entities in a DataFrame (bulk request).

        Args:
            df: DataFrame containing curies columns

        Returns:
            DataFrame with columns: kg_ids, kg_ids_provided, kg_ids_assigned
        """
        # Collect all unique curies across all entities
        all_curies = set()
        for curies_list in df["curies"]:
            all_curies.update(curies_list)

        # Single bulk request for all curies
        curie_to_kg_id_cache = self.get_kg_ids(list(all_curies))

        # Apply per-entity processing using the shared cache
        return df.apply(
            lambda entity: self._link_entity(entity, curie_to_kg_id_cache=curie_to_kg_id_cache),
            axis=1,
            result_type="expand",  # Expands Series into columns
        )

    def _link_entity(
        self, entity: pd.Series | dict[str, Any], curie_to_kg_id_cache: dict[str, str] | None = None
    ) -> pd.Series:
        """
        Link a single entity's curies to knowledge graph node IDs.

        Args:
            entity: Entity containing curies
            curie_to_kg_id_cache: Optional pre-computed mapping from curies to KG node IDs.
                                  If not provided, will make API request for this entity's curies.

        Returns:
            Named Series with fields: kg_ids, kg_ids_provided, kg_ids_assigned
        """
        # Use cache if provided, otherwise fetch KG IDs for this entity
        if curie_to_kg_id_cache is None:
            curie_to_kg_id_cache = self.get_kg_ids(entity["curies"])

        kg_ids, kg_ids_provided, kg_ids_assigned = self._format_kg_id_fields(entity, curie_to_kg_id_cache)

        return pd.Series({"kg_ids": kg_ids, "kg_ids_provided": kg_ids_provided, "kg_ids_assigned": kg_ids_assigned})

    @staticmethod
    def get_kg_ids(curies: list[str]) -> dict[str, str]:
        """
        Query knowledge graph API for canonical node IDs (in bulk).

        Args:
            curies: List of curies to look up

        Returns:
            Dictionary mapping curies to canonical KG node IDs
        """
        if curies:
            results = kestrel_request("POST", "canonicalize", json={"curies": curies})
        else:
            results = dict()

        return results

    def _format_kg_id_fields(
        self, entity: pd.Series | dict[str, Any], curie_to_kg_id_map: dict[str, str]
    ) -> tuple[dict[str, list[str]], dict[str, list[str]], dict[str, dict[str, list[str]]]]:
        """
        Organize KG IDs by source (overall, provided, assigned) and record their corresponding curie 'votes'.

        Args:
            entity: Entity with curie fields
            curie_to_kg_id_map: Mapping from curies to KG node IDs

        Returns:
            Tuple of (kg_ids_dict, kg_ids_provided_dict, kg_ids_assigned_dict)
        """
        curies = entity["curies"]
        curies_provided = entity["curies_provided"]
        curies_assigned = entity["curies_assigned"]

        kg_ids = self._reverse_curie_map(curie_to_kg_id_map, curie_subset=curies)
        kg_ids_provided = self._reverse_curie_map(curie_to_kg_id_map, curie_subset=curies_provided)

        # Build KG IDs assigned in nested fashion (per annotator)
        kg_ids_assigned = dict()
        for annotator_slug, annotator_curies in curies_assigned.items():
            kg_ids_assigned[annotator_slug] = self._reverse_curie_map(curie_to_kg_id_map, curie_subset=annotator_curies)

        return kg_ids, kg_ids_provided, kg_ids_assigned

    @staticmethod
    def _reverse_curie_map(curie_map: dict[str, str], curie_subset: list[str]) -> dict[str, list[str]]:
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
            kg_id = curie_map.get(curie)
            if kg_id:
                reversed_dict[kg_id].append(curie)
        return dict(reversed_dict)
