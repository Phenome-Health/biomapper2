import logging
from typing import Any, cast

import pandas as pd

from ...config import HYBRID_SEARCH_LIMIT, KESTREL_BATCH_SIZE_SEARCH
from ...utils import AssignedIDsDict, kestrel_request, text_is_not_empty
from .base import BaseAnnotator, is_on_category


class KestrelVectorSearchAnnotator(BaseAnnotator):

    slug = "kestrel-vector-search"

    def get_annotations(
        self,
        entity: dict | pd.Series,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        prefer_human: bool = True,  # accepted for interface parity; not applicable to vector search
        preferred_prefixes: set[str] | None = None,  # accepted for interface parity; not applicable
        accepted_categories: set[str] | None = None,
        cache: dict | None = None,
    ) -> AssignedIDsDict:
        """Implements BaseAnnotator.get_annotations"""

        # Extract the value to search
        search_term = entity.get(name_field)
        if text_is_not_empty(search_term):
            # Use cache if available, otherwise make API call
            if cache:
                term_results = cache.get(search_term)
            else:
                results = self._kestrel_vector_search(search_term, category, prefixes, limit=HYBRID_SEARCH_LIMIT)
                term_results = results[search_term]

            annotations: dict[str, dict[str, dict[str, Any]]] = {}
            # Commit-point category validator, same as kestrel-hybrid. `category_filter` is advisory:
            # the endpoint returns rows outside the requested category, and an off-category row can land
            # at rank 1 (e.g. a CHV/GO/UMLS node named like the query). Scan the ranked window for the
            # first ON-category candidate instead of committing/refusing on rank 1 alone; the guard still
            # refuses when NOTHING in the window is on-category. `annotators` is API-exposed
            # (api/models/requests.py), so without this guard a caller requesting
            # annotators=['kestrel-vector-search'] could commit a node the default set refuses.
            chosen = next((r for r in (term_results or []) if is_on_category(r, accepted_categories)), None)
            if chosen is not None:
                node_id = chosen["id"]
                score = chosen["score"]
                vocab, local_id = node_id.split(":", 1)
                annotations.setdefault(vocab, {})[local_id] = {"score": score}
            elif term_results:
                logging.info(
                    "off_category_refusal: annotator=%s term=%r window=%d top=%s categories=%s",
                    self.slug,
                    search_term,
                    len(term_results),
                    term_results[0].get("id"),
                    term_results[0].get("categories"),
                )

            return {self.slug: annotations}
        else:
            # This entity didn't have a name, so we can't use this annotator on it
            return dict()

    def get_annotations_bulk(
        self,
        entities: pd.DataFrame,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        prefer_human: bool = True,  # accepted for interface parity; not applicable to vector search
        preferred_prefixes: set[str] | None = None,  # accepted for interface parity; not applicable
        accepted_categories: set[str] | None = None,
    ) -> pd.Series:  # Series of AssignedIDsDicts
        """Implements BaseAnnotator.get_annotations_bulk"""

        # Filter out any empty/NaN entity names
        search_terms = [t for t in entities[name_field].tolist() if text_is_not_empty(t)]

        logging.info(f"Getting vector search results from Kestrel API for {len(entities)} entities")
        results = self._kestrel_vector_search(search_terms, category, prefixes, limit=HYBRID_SEARCH_LIMIT)

        # Annotate each entity using the results from the bulk request
        assigned_ids_col = entities.apply(
            self.get_annotations,
            axis=1,
            cache=results,
            name_field=name_field,
            category=category,
            prefixes=prefixes,
            prefer_human=prefer_human,
            # MUST be forwarded: the bulk path re-dispatches into get_annotations, so omitting this
            # would silently drop the category guard on every dataset job while keeping it on the
            # single-entity path.
            accepted_categories=accepted_categories,
        )

        return cast(pd.Series, assigned_ids_col)

    # ----------------------------------------- Helper methods ----------------------------------------------- #

    @staticmethod
    def _kestrel_vector_search(
        search_text: str | list[str], category: str, prefixes: list[str] | None, limit: int = 10
    ) -> dict[str, list[dict]]:
        """Call Kestrel vector search endpoint (with batching for large lists)."""
        search_list = [search_text] if isinstance(search_text, str) else list(search_text)

        return kestrel_request(
            method="POST",
            endpoint="vector-search",
            batch_field="search_text",
            batch_items=search_list,
            batch_size=KESTREL_BATCH_SIZE_SEARCH,
            json={"limit": limit, "category_filter": category, "prefix_filter": prefixes},
        )
