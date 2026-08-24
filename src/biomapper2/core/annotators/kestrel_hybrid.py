import logging
from typing import Any, cast

import pandas as pd

from ...config import (
    GENE_SYMBOL_FALLBACK_ENABLED,
    HUMAN_MARKER_PREFIXES,
    HYBRID_SEARCH_LIMIT,
    KESTREL_BATCH_SIZE_SEARCH,
)
from ...utils import AssignedIDsDict, kestrel_request, text_is_not_empty
from ..gene_symbol_resolver import GeneSymbolResolver
from .base import BaseAnnotator, is_on_category

# Score assigned to a node recovered by the deterministic symbol fallback. Modest and fixed: the result
# is a verified identity match, but it bypassed competitive search, so it must not be reported as a top
# search hit. The `resolved_via` provenance marker is the authoritative signal, not this score.
_FALLBACK_SCORE = 1.0


class KestrelHybridSearchAnnotator(BaseAnnotator):

    slug = "kestrel-hybrid-search"

    def __init__(self) -> None:
        # Annotator-owned resolver (arg-free construction preserves the registry pattern; no network here).
        self._resolver = GeneSymbolResolver()

    def get_annotations(
        self,
        entity: dict | pd.Series,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        prefer_human: bool = True,
        preferred_prefixes: set[str] | None = None,
        accepted_categories: set[str] | None = None,
        cache: dict | None = None,
    ) -> AssignedIDsDict:
        """Implements BaseAnnotator.get_annotations.

        Two mutually-exclusive re-ranking policies, both gated by the engine (which sets at most one):
        - ``prefer_human`` (gene/protein): select the human (HGNC-bearing) candidate matching the queried
          symbol, with the curated drug-conflated fallback on a miss (see ``_select_result``).
        - ``preferred_prefixes`` (other categories, e.g. metabolite/disease): prefer the canonical-namespace
          candidate that matches the query, falling back to the top-scored canonical or — on an empty
          canonical pool — the overall top-scored row (see ``_select_canonical``).
        With neither active, the legacy top-1 behavior is kept.

        ``accepted_categories`` is orthogonal to both: a *validator* applied to the single node the
        selectors committed (see ``_is_on_category``). It never touches the candidate pool, so no
        selector's behavior changes and no vacancy is created for a runner-up to fill.
        """

        # Extract the value to search
        search_term = entity.get(name_field)
        if text_is_not_empty(search_term):
            # Use cache if available, otherwise make API call. The wider candidate window is needed for
            # either re-ranking policy (the canonical/human node often ranks below the conflated top hit).
            if cache:
                term_results = cache.get(search_term)
            else:
                limit = HYBRID_SEARCH_LIMIT if (prefer_human or preferred_prefixes) else 1
                results = self._kestrel_hybrid_search(search_term, category, prefixes, limit=limit)
                term_results = results[search_term]

            annotations: dict[str, dict[str, dict[str, Any]]] = {}
            chosen, matched = self._select_result(term_results, search_term, prefer_human)

            # Miss path: no HGNC + exact-symbol match was found among the search rows (ortholog fallback,
            # empty results, or a best-HGNC paralog). For the curated drug-conflated symbols, the human
            # node is unreachable by search, so resolve it deterministically (non-search) and verify.
            if prefer_human and GENE_SYMBOL_FALLBACK_ENABLED and not matched:
                resolved_curie = self._resolver.resolve(search_term)
                if resolved_curie is not None:
                    chosen = {"id": resolved_curie, "score": _FALLBACK_SCORE, "resolved_via": "symbol_fallback"}
            # Canonical-namespace preference (non-gene categories). Mutual exclusivity with the gene path
            # is enforced by the *engine*, which sets preferred_prefixes only when prefer_human is off for
            # this category — not by this branch structure. (As an `elif` on the gene `if`, a hypothetical
            # caller passing both an effective prefer_human and preferred_prefixes would, on a genuine gene
            # match, fall through to here and override it; that combination never occurs via the engine.)
            elif preferred_prefixes:
                chosen, is_canonical = self._select_canonical(term_results, preferred_prefixes, search_term)
                if chosen is not None and is_canonical:
                    # Tag only a genuine canonical pick (the selector reports it — not the empty-pool
                    # fallback to the top-scored non-canonical row).
                    chosen = {**chosen, "resolved_via": "canonical_preference"}

            # Category validation at the single commit point. Applied AFTER selection, deliberately: a
            # pool filter would promote the runner-up into the vacancy, and that cannot recover a right
            # answer (`_select_canonical` already preferred the canonical rows, so a promotion only
            # happens when none were there). It only substitutes a different wrong node that now passes
            # the type test — harder to audit than one announcing itself as not-a-molecule. Refuse
            # instead; this can only turn wrong->refuse. See `_is_on_category`.
            if chosen is not None and not self._is_on_category(chosen, accepted_categories):
                logging.info(
                    "off_category_refusal: term=%r node=%s categories=%s",
                    search_term,
                    chosen.get("id"),
                    chosen.get("categories"),
                )
                chosen = None

            if chosen is not None:
                node_id = chosen["id"]
                score = chosen["score"]
                vocab, local_id = node_id.split(":", 1)
                metadata: dict[str, Any] = {"score": score}
                if chosen.get("resolved_via"):
                    metadata["resolved_via"] = chosen["resolved_via"]
                annotations.setdefault(vocab, {})[local_id] = metadata

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
        prefer_human: bool = True,
        preferred_prefixes: set[str] | None = None,
        accepted_categories: set[str] | None = None,
    ) -> pd.Series:  # Series of AssignedIDsDicts
        """Implements BaseAnnotator.get_annotations_bulk"""

        # Filter out any empty/NaN entity names
        search_terms = [t for t in entities[name_field].tolist() if text_is_not_empty(t)]

        logging.info(f"Getting hybrid search results from Kestrel API for {len(entities)} entities")
        # Enlarge the candidate window when either re-ranking policy is active (keeps payloads small for
        # bulk jobs where no re-ranking applies).
        limit = HYBRID_SEARCH_LIMIT if (prefer_human or preferred_prefixes) else 1
        results = self._kestrel_hybrid_search(search_terms, category, prefixes, limit=limit)

        # Annotate each entity using the results from the bulk request. The internal re-dispatch MUST
        # forward prefer_human, preferred_prefixes AND accepted_categories, otherwise the bulk path would
        # silently use the get_annotations defaults (a silent no-op on dataset jobs — which is every
        # benchmark run — for the canonical re-rank and for the category guard alike).
        assigned_ids_col = entities.apply(
            self.get_annotations,
            axis=1,
            cache=results,
            name_field=name_field,
            category=category,
            prefixes=prefixes,
            prefer_human=prefer_human,
            preferred_prefixes=preferred_prefixes,
            accepted_categories=accepted_categories,
        )

        return cast(pd.Series, assigned_ids_col)

    # ----------------------------------------- Helper methods ----------------------------------------------- #

    @staticmethod
    def _select_result(
        term_results: list[dict] | None, search_term: str, prefer_human: bool
    ) -> tuple[dict | None, bool]:
        """Select one candidate row, and report whether it is a genuine human-gene match.

        Returns ``(chosen_row, matched)`` where ``matched`` is True only when ``chosen_row`` is an
        HGNC-bearing row that exact-symbol-matches the query. ``matched`` is False for an empty result,
        the legacy top-1, the ortholog fallback, and the best-HGNC-but-no-symbol-match (paralog) case —
        all of which the caller treats as a miss eligible for the deterministic symbol fallback.

        Two-tier human preference (finalized from the 2026-06-15 live spike):
        1. Filter to human (HGNC-bearing) rows; if none, fall back to the top-scored candidate.
        2. Among human rows, prefer those whose symbol matches the query (``name`` or a synonym). A
           symbol-matched pick is the only genuine "match"; the no-symbol-match fallback is not.
        3. Return the highest-scoring row of the chosen pool.

        The HGNC filter must precede the symbol match because orthologs share the human row's ``name``
        (same symbol, different species) — HGNC separates species, the symbol match picks the right gene.
        """
        if not term_results:
            return None, False
        if not prefer_human:
            return term_results[0], False

        human = [r for r in term_results if HUMAN_MARKER_PREFIXES & set(r.get("prefixes") or [])]
        if not human:
            return term_results[0], False

        exact = [r for r in human if KestrelHybridSearchAnnotator._symbol_matches(r, search_term)]
        if exact:
            return max(exact, key=lambda r: r.get("score", 0)), True
        # HGNC rows exist but none symbol-match the query (e.g. a paralog) — not a genuine match.
        return max(human, key=lambda r: r.get("score", 0)), False

    @staticmethod
    def _select_canonical(
        term_results: list[dict] | None, preferred_prefixes: set[str], search_term: str
    ) -> tuple[dict | None, bool]:
        """Select the canonical-namespace candidate for a non-gene category, else the honest top-1.

        Returns ``(chosen_row, is_canonical)`` where ``is_canonical`` is True only when ``chosen_row`` came
        from the preferred-namespace pool — so the caller can tag provenance without re-deriving the
        namespace it already filtered on. It is False for an empty result and for the empty-pool fallback.

        Validated against live Kestrel data (2026-06-18): within a category the canonical node (CHEBI/HMDB/
        RM for metabolites, MONDO for disease) routinely scores well *below* a conflated non-canonical node
        (UMLS/ICD/KEGG/PANTHER), so score is not a gate — the namespace filter plus an identity match is the
        discriminator. There is deliberately **no score-margin guard**.

        1. Filter to rows whose ``id`` namespace is in ``preferred_prefixes``. Filtering on the ``id``
           (the assigned CURIE) — not the broader ``prefixes`` xref list ``_select_result`` uses — guarantees
           the *assigned* result is canonical, not merely cross-referenced. The comparison is case-insensitive
           so a differently-cased KG prefix can't silently empty the pool.
        2. Empty pool → honest fallback to the overall top-scored row, ``is_canonical=False`` (never
           fabricate a canonical CURIE).
        3. Among the pool, prefer the rows whose ``name``/synonym matches the query (reuses ``_symbol_matches``
           directly), then take the highest-scoring of that sub-pool; if none match, the top-scored canonical
           row. This picks the right concept among same-namespace homonyms (e.g. CHEBI isomers) and the right
           specific node when the query does not exact-match any name (e.g. the top-scored MONDO for a disease).
        """
        if not term_results:
            return None, False
        preferred_cf = {p.casefold() for p in preferred_prefixes}
        pool = [r for r in term_results if str(r.get("id", "")).split(":", 1)[0].casefold() in preferred_cf]
        if not pool:
            return term_results[0], False
        exact = [r for r in pool if KestrelHybridSearchAnnotator._symbol_matches(r, search_term)]
        candidates = exact if exact else pool
        return max(candidates, key=lambda r: r.get("score", 0)), True

    @staticmethod
    def _is_on_category(row: dict, accepted: set[str] | None) -> bool:
        """Delegates to :func:`base.is_on_category`.

        Kept as a name on this class because it is the annotator most reached for, but the
        implementation is shared: kestrel-text and kestrel-vector apply the SAME predicate at
        their own commit points, so a caller cannot bypass the guard by naming an annotator.
        """
        return is_on_category(row, accepted)

    @staticmethod
    def _symbol_matches(row: dict, search_term: str) -> bool:
        """True if the row's ``name`` equals the query (case-insensitive) or the query is a synonym."""
        term = str(search_term).strip().lower()
        if str(row.get("name", "")).strip().lower() == term:
            return True
        return any(str(s).strip().lower() == term for s in row.get("synonyms", []) or [])

    @staticmethod
    def _kestrel_hybrid_search(
        search_text: str | list[str], category: str, prefixes: list[str] | None, limit: int = 10
    ) -> dict[str, list[dict]]:
        """Call Kestrel hybrid search endpoint (with batching for large lists)."""
        search_list = [search_text] if isinstance(search_text, str) else list(search_text)

        results = kestrel_request(
            method="POST",
            endpoint="hybrid-search",
            batch_field="search_text",
            batch_items=search_list,
            batch_size=KESTREL_BATCH_SIZE_SEARCH,
            json={"limit": limit, "category_filter": category, "prefix_filter": prefixes},
        )
        # Filter out very low-scoring results (hybrid search scores range from 0-5)
        return {s: [match for match in matches if match["score"] >= 0.5] for s, matches in results.items()}
