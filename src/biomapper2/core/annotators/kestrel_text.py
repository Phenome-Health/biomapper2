import logging
import sys
from collections import defaultdict
from typing import Optional

from .base import BaseAnnotator
from ...utils import kestrel_request, AssignedIDsDict
import pandas as pd


class KestrelTextSearchAnnotator(BaseAnnotator):

    slug = 'kestrel-text-search'

    def get_annotations(
            self,
            entity: dict | pd.Series,
            name_field: str,
            cache: Optional[dict] = None
    ) -> AssignedIDsDict:
        """Get annotations via Kestrel text search."""

        # Extract the value to search
        search_term = entity.get(name_field)
        if search_term:
            # Use cache if available, otherwise make API call
            if cache:
                results = cache.get(search_term)
            else:
                results = self._kestrel_text_search(search_term, limit=1)

            annotations = defaultdict(lambda: defaultdict(dict))
            if results:
                result = results[0]
                node_id = result['id']
                score = result['score']
                vocab, local_id = node_id.split(':', 1)
                annotations[vocab][local_id]['score'] = score

            return {self.slug: annotations}
        else:
            # This entity didn't have a name, so we can't use this annotator on it
            return dict()


    def get_annotations_bulk(
            self,
            entities: pd.DataFrame,
            name_field: str
    ) -> pd.Series:  # Series of AssignedIDsDicts
        """Get annotations for multiple entities with bulk optimization."""

        # Extract all search terms (adjust column name as needed)
        search_terms = entities['name'].tolist()

        # Build cache with all API calls  TODO: use actual bulk endpoint once stand that up...
        logging.info(f"Getting results from Kestrel API for {len(entities)} entities")
        cache = {}
        for term in search_terms:
            cache[term] = self._kestrel_text_search(term, limit=1)

        # Annotate each entity using the cache
        assigned_ids_col = entities.apply(
            self.get_annotations,
            axis=1,
            cache=cache,
            name_field=name_field
        )

        return assigned_ids_col


    # ----------------------------------------- Helper methods ----------------------------------------------- #

    @staticmethod
    def _kestrel_text_search(query: str, limit: int = 10) -> list[dict]:
        """Call Kestrel text search endpoint."""
        return kestrel_request('POST', 'text-search',
                               json={'search_term': query, 'limit': limit})
