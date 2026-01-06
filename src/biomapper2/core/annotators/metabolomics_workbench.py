"""Metabolomics Workbench RefMet API annotator for metabolite entities."""

import logging
from datetime import timedelta
from typing import Any
from urllib.parse import quote

import pandas as pd
import requests
import requests_cache
from circuitbreaker import CircuitBreakerError, circuit

from ...config import CACHE_DIR
from ...utils import AssignedIDsDict
from .base import BaseAnnotator


class MetabolomicsWorkbenchAnnotator(BaseAnnotator):
    """Annotator that queries the Metabolomics Workbench RefMet match API.

    Retrieves RefMet IDs for metabolite entities using fuzzy name matching.
    Returns raw API field names; the Normalizer handles mapping and ID cleaning.

    The /match endpoint handles non-standard names (e.g., "cholate" → "Cholic acid").
    Only refmet_id is returned since KRAKEN has all RefMet equivalencies.

    API Endpoint: GET https://www.metabolomicsworkbench.org/rest/refmet/match/{metabolite_name}
    """

    slug = "metabolomics-workbench"
    BASE_URL = "https://www.metabolomicsworkbench.org/rest/refmet/match"

    # Only extract refmet_id - KRAKEN has all RefMet equivalencies
    API_FIELDS = ["refmet_id"]

    def __init__(self):
        self._session = requests_cache.CachedSession(
            CACHE_DIR / "metabolomics_workbench_http",
            expire_after=timedelta(days=7),
        )

    def get_annotations(
        self,
        entity: dict | pd.Series,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        cache: dict | None = None,
    ) -> AssignedIDsDict:
        """Implements BaseAnnotator.get_annotations"""

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
        annotations: dict[str, dict[str, dict[str, Any]]] = {}

        for api_field in self.API_FIELDS:
            value = api_data.get(api_field)
            if value:
                # Use raw field name and value - Normalizer handles mapping and cleaning
                annotations.setdefault(api_field, {})[value] = {}

        return {self.slug: annotations}

    def get_annotations_bulk(
        self,
        entities: pd.DataFrame,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
    ) -> pd.Series:
        """Implements BaseAnnotator.get_annotations_bulk"""

        # Extract unique names to avoid duplicate API calls
        names = entities[name_field].dropna().unique().tolist()

        # Build cache with API results
        logging.info(f"Fetching RefMet data for {len(names)} unique metabolite names")
        cache = {name: self._fetch_refmet_data(name) for name in names}

        # Apply get_annotations to each row using the cache
        assigned_ids_col = entities.apply(
            self.get_annotations, axis=1, cache=cache, name_field=name_field, category=category, prefixes=prefixes
        )

        return assigned_ids_col

    def _fetch_refmet_data(self, metabolite_name: str) -> dict[str, Any] | None:
        """Fetch RefMet data from the Metabolomics Workbench match API.

        Args:
            metabolite_name: Name of the metabolite to look up

        Returns:
            API response dict or None if not found
        """
        try:
            return self._do_refmet_request(metabolite_name)
        except CircuitBreakerError:
            logging.debug(f"RefMet API outage, skipping '{metabolite_name}' (circuit open)")
            return None
        except requests.RequestException as e:
            logging.warning(f"Failed to fetch RefMet data for '{metabolite_name}': {e}")
            return None

    @circuit(failure_threshold=3, recovery_timeout=300)
    def _do_refmet_request(self, metabolite_name: str) -> dict[str, Any] | None:
        """
        Make the actual HTTP request. Protected by circuit breaker - trips on repeated failures,
        meaning it will skip requests until a cooldown period is over (recovery_timeout).
        """
        url = f"{self.BASE_URL}/{quote(metabolite_name)}"

        response = self._session.get(url, timeout=3)
        response.raise_for_status()
        data = response.json()

        if isinstance(data, dict):
            if data.get("refmet_id") == "-":
                # /match endpoint returns dict with "-" values when no match found
                return None
            return data

        return None
