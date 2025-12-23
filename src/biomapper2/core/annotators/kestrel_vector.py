import logging
from typing import Any

import pandas as pd

from ...utils import AssignedIDsDict, kestrel_request, text_is_not_empty
from .base import BaseAnnotator


class KestrelVectorSearchAnnotator(BaseAnnotator):

    slug = "kestrel-vector-search"

    def get_annotations(
        self, entity: dict | pd.Series, name_field: str, category: str, cache: dict | None = None
    ) -> AssignedIDsDict:
        """Implements BaseAnnotator.get_annotations"""

        # Extract the value to search
        search_term = entity.get(name_field)
        if text_is_not_empty(search_term):
            # Use cache if available, otherwise make API call
            if cache:
                term_results = cache.get(search_term)
            else:
                results = self._kestrel_vector_search(search_term, category, limit=1)
                term_results = results[search_term]

            annotations: dict[str, dict[str, dict[str, Any]]] = {}
            if term_results:
                first_result = term_results[0]
                node_id = first_result["id"]
                score = first_result["score"]
                vocab, local_id = node_id.split(":", 1)
                annotations.setdefault(vocab, {})[local_id] = {"score": score}

            return {self.slug: annotations}
        else:
            # This entity didn't have a name, so we can't use this annotator on it
            return dict()

    def get_annotations_bulk(
        self, entities: pd.DataFrame, name_field: str, category: str
    ) -> pd.Series:  # Series of AssignedIDsDicts
        """Implements BaseAnnotator.get_annotations_bulk"""

        # Filter out any empty/NaN entity names
        search_terms = [t for t in entities[name_field].tolist() if text_is_not_empty(t)]

        logging.info(f"Getting vector search results from Kestrel API for {len(entities)} entities")
        results = self._kestrel_vector_search(search_terms, category, limit=1)
        logging.debug(f"Kestrel results: {results}")

        # Annotate each entity using the results from the bulk request
        assigned_ids_col = entities.apply(
            self.get_annotations, axis=1, cache=results, name_field=name_field, category=category
        )

        return assigned_ids_col

    # ----------------------------------------- Helper methods ----------------------------------------------- #

    @staticmethod
    def _kestrel_vector_search(search_text: str | list[str], category: str, limit: int = 10) -> dict[str, list[dict]]:
        """Call Kestrel vector search endpoint."""
        payload = {"search_text": search_text, "limit": limit, "category_filter": category}
        return kestrel_request("POST", "vector-search", json=payload)
