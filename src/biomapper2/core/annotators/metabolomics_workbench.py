"""Metabolomics Workbench RefMet API annotator for metabolite entities."""

import logging
from collections import defaultdict
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests

from ...utils import AssignedIDsDict
from .base import BaseAnnotator


class MetabolomicsWorkbenchAnnotator(BaseAnnotator):
    """Annotator that queries the Metabolomics Workbench RefMet API.

    Retrieves vocabulary IDs for metabolite entities. Returns raw API field names;
    the Normalizer handles mapping to standard vocab names and ID cleaning.

    API Endpoint: GET https://www.metabolomicsworkbench.org/rest/refmet/name/{metabolite_name}/all/
    """

    slug = "metabolomics-workbench"
    BASE_URL = "https://www.metabolomicsworkbench.org/rest/refmet/name"

    # API fields to extract (raw field names - Normalizer handles mapping)
    API_FIELDS = [
        "pubchem_cid",
        "inchi_key",
        "smiles",
        "refmet_id",
        "ChEBI_ID",
        "HMDB_ID",
        "LM_ID",
        "KEGG_ID",
    ]

    def get_annotations(self, entity: dict | pd.Series, name_field: str, cache: dict | None = None) -> AssignedIDsDict:
        """Get annotations for a single entity.

        Args:
            entity: Entity to annotate (dict or DataFrame row)
            name_field: Name of the field containing the entity name
            cache: Optional pre-fetched results from bulk API call

        Returns:
            Dict with annotation results using raw API field names
        """
        # Extract the entity name
        name = entity.get(name_field)

        if not name:
            # No name provided, cannot annotate
            return {}

        # Use cache if available, otherwise fetch from API
        if cache is not None:
            api_data = cache.get(name)
        else:
            api_data = self._fetch_refmet_data(name)

        if not api_data:
            # No data returned from API
            return {self.slug: {}}

        # Build the annotations structure using raw API field names
        annotations: dict[str, dict[str, dict[str, Any]]] = defaultdict(lambda: defaultdict(dict))

        for api_field in self.API_FIELDS:
            value = api_data.get(api_field)
            if value:
                # Use raw field name and value - Normalizer handles mapping and cleaning
                annotations[api_field][value] = {}

        return {self.slug: dict(annotations)}

    def get_annotations_bulk(self, entities: pd.DataFrame, name_field: str) -> pd.Series:
        """Get annotations for multiple entities with bulk API call.

        Args:
            entities: DataFrame where each row is an entity
            name_field: Name of the column containing entity names

        Returns:
            Column (Series) of annotation results (same index as input)
        """
        # Extract unique names to avoid duplicate API calls
        names = entities[name_field].dropna().unique().tolist()

        # Build cache with API results
        logging.info(f"Fetching RefMet data for {len(names)} unique metabolite names")
        cache: dict[str, dict[str, Any] | None] = {}
        for name in names:
            cache[name] = self._fetch_refmet_data(name)

        # Apply get_annotations to each row using the cache
        assigned_ids_col = entities.apply(self.get_annotations, axis=1, cache=cache, name_field=name_field)

        return assigned_ids_col

    def _fetch_refmet_data(self, metabolite_name: str) -> dict[str, Any] | None:
        """Fetch RefMet data from the Metabolomics Workbench API.

        Args:
            metabolite_name: Name of the metabolite to look up

        Returns:
            API response dict or None if not found
        """
        url = f"{self.BASE_URL}/{quote(metabolite_name)}/all/"

        response = requests.get(url, timeout=30)
        response.raise_for_status()

        data = response.json()

        # API returns empty list [] when metabolite not found
        if isinstance(data, list):
            return None

        # API returns a dict when metabolite is found
        if isinstance(data, dict):
            return data

        return None
