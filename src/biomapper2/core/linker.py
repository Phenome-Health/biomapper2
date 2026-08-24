"""
Curie linking module for mapping to knowledge graph nodes.

Queries the knowledge graph API to find canonical node IDs for normalized curies.
"""

import logging
from collections import defaultdict
from typing import Any

import pandas as pd

from ..config import KESTREL_BATCH_SIZE_CANONICALIZE
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
        Query knowledge graph API for canonical node IDs (in bulk, with batching).

        Args:
            curies: List of curies to look up

        Returns:
            Dictionary mapping curies to canonical KG node IDs
        """
        return kestrel_request(
            method="POST",
            endpoint="canonicalize",
            batch_field="curies",
            batch_items=curies,
            batch_size=KESTREL_BATCH_SIZE_CANONICALIZE,
        )

    @staticmethod
    def get_node_records(kg_node_ids: list[str]) -> dict[str, dict[str, Any]]:
        """Fetch node records ``{curie: {"name": str|None, "equivalent_ids": {prefix: [local_id, ...]}}}``.

        A superset of :meth:`get_equivalent_ids` used by the resolver's connectivity test, which needs
        the node ``name`` for the Metabolomics Workbench / PubChem fallback. Non-critical enrichment:
        returns ``{}`` on API failure rather than raising (mirrors :meth:`get_equivalent_ids`).
        """
        if not kg_node_ids:
            return {}
        try:
            raw_results = kestrel_request(
                method="POST",
                endpoint="get-nodes",
                batch_field="curies",
                batch_items=kg_node_ids,
                batch_size=KESTREL_BATCH_SIZE_CANONICALIZE,
                json={"slim": False, "truncate_long_fields": False},
            )
        except Exception:
            logging.warning("Failed to fetch node records from Kestrel /get-nodes; returning empty", exc_info=True)
            return {}

        result: dict[str, dict[str, Any]] = {}
        for curie, node_obj in raw_results.items():
            if not isinstance(node_obj, dict):
                continue
            grouped: dict[str, list[str]] = {}
            for equiv_id in node_obj.get("equivalent_ids", []):
                if ":" not in equiv_id:
                    continue
                prefix, local_id = equiv_id.split(":", 1)
                grouped.setdefault(prefix, []).append(local_id)
            result[curie] = {"name": node_obj.get("name"), "equivalent_ids": grouped}
        return result

    @staticmethod
    def get_equivalent_ids(
        kg_node_ids: list[str],
        prefixes: list[str] | None = None,
    ) -> dict[str, dict[str, list[str]]]:
        """
        Fetch equivalent IDs for KG nodes from the Kestrel /get-nodes endpoint.

        Returns all IDs grouped by CURIE prefix. Pass a list of prefixes to
        filter to specific vocabularies only.

        This is a non-critical enrichment step. On API failure, logs a warning
        and returns an empty dict rather than raising.

        Callers that need to tell an API failure apart from a node the graph genuinely lists
        nothing for should use :meth:`get_equivalent_ids_checked`.

        Args:
            kg_node_ids: List of KG node CURIEs to look up
            prefixes: Optional CURIE prefixes to include. When None (default),
                      all prefixes are returned.

        Returns:
            Dictionary mapping each node CURIE to a dict of {prefix: [local_ids]},
            e.g. {"CHEBI:15365": {"HMDB": ["HMDB0001879"], "KEGG.COMPOUND": ["C01405"]}}
        """
        return Linker.get_equivalent_ids_checked(kg_node_ids, prefixes)[0]

    @staticmethod
    def get_equivalent_ids_checked(
        kg_node_ids: list[str],
        prefixes: list[str] | None = None,
    ) -> tuple[dict[str, dict[str, list[str]]], bool]:
        """As :meth:`get_equivalent_ids`, plus whether the lookup actually succeeded.

        The flag exists because the two failure shapes are otherwise identical downstream. An empty
        payload means either "the graph lists no equivalent ids for this node" or "the /get-nodes
        call raised and we swallowed it", and the resolution certificate must not read the second as
        the first: doing so would mark an entire outage-affected run ``structure_absent`` and no
        offline rerun on the resulting TSV could tell.

        Returns:
            ``(mapping, ok)`` -- ``ok`` is False only when the API call itself failed.
        """
        if not kg_node_ids:
            return {}, True

        try:
            raw_results = kestrel_request(
                method="POST",
                endpoint="get-nodes",
                batch_field="curies",
                batch_items=kg_node_ids,
                batch_size=KESTREL_BATCH_SIZE_CANONICALIZE,
                json={"slim": False, "truncate_long_fields": False},
            )
        except Exception:
            logging.warning("Failed to fetch equivalent IDs from Kestrel /get-nodes; returning empty", exc_info=True)
            return {}, False

        result: dict[str, dict[str, list[str]]] = {}
        for curie, node_obj in raw_results.items():
            if not isinstance(node_obj, dict):
                continue
            raw_ids = node_obj.get("equivalent_ids", [])
            grouped: dict[str, list[str]] = {}
            for equiv_id in raw_ids:
                if ":" not in equiv_id:
                    continue
                prefix, local_id = equiv_id.split(":", 1)
                # By default all prefixes are returned — KG nodes naturally carry only
                # entity-type-relevant vocabularies (e.g. genes get HGNC/ENSEMBL,
                # metabolites get LM/HMDB). The prefixes param is an opt-in hook for
                # callers that need to narrow further (e.g. an API query param).
                if prefixes and prefix not in prefixes:
                    continue
                grouped.setdefault(prefix, []).append(local_id)
            # Sort local IDs within each prefix for deterministic output
            result[curie] = {prefix: sorted(ids) for prefix, ids in sorted(grouped.items())}

        return result, True

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
