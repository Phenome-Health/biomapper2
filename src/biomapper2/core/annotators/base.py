from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from ...utils import AssignedIDsDict

# Availability of an annotator's vote for a single row, surfaced ALONGSIDE the vote (never inside it)
# so a consumer can tell a genuine no-match from a service that never answered. Plain strings (not an
# Enum) so the value passes unchanged through a pandas column, ``Entity.model_extra`` and the JSON
# API. ``not_queried`` is the total-map default: an annotator that did not run for the row.
AVAILABILITY_VOTED = "voted"
AVAILABILITY_NO_MATCH = "no_match"
AVAILABILITY_UNAVAILABLE = "unavailable"
AVAILABILITY_NOT_QUERIED = "not_queried"

# WHICH RefMet source served a row's vote, surfaced ALONGSIDE availability on a parallel provenance
# channel (never folded into it): availability answers "did a source answer?", source answers "which
# source". Plain strings for the same reasons as the AVAILABILITY_* values (pandas column,
# ``Entity.model_extra``, JSON API). ``not_queried`` is the total-map default: an annotator that did
# not run for the row (e.g. every gene row, and every RefMet-not-selected row).
#   local_snapshot  : the pinned local freeze answered (voted or no_match) — the breaker was OUT of
#                     the path, so the vote is deterministic.
#   not_in_snapshot : a snapshot IS loaded but this exact query name was not in the freeze; the
#                     default deterministic outcome (NO_MATCH) without a network call.
#   live_api        : the live Metabolomics Workbench /match endpoint served the row (no snapshot,
#                     or a snapshot miss with live_api_fallback enabled) and answered.
#   unavailable     : the live service did not answer (breaker open / transport error / timeout).
#   not_queried     : RefMet was not selected for the row.
REFMET_SOURCE_LOCAL = "local_snapshot"
REFMET_SOURCE_NOT_IN_SNAPSHOT = "not_in_snapshot"
REFMET_SOURCE_LIVE = "live_api"
REFMET_SOURCE_UNAVAILABLE = "unavailable"
REFMET_SOURCE_NOT_QUERIED = "not_queried"

# A node typed only at the top of the Biolink hierarchy is an ABSENT type assertion,
# not an off-category claim, so the category validator lets it through (see `is_on_category`).
TOP_OF_HIERARCHY_SENTINELS = frozenset({"biolink:NamedThing", "biolink:Entity"})


def stable_result_order(rows: list[dict] | None) -> list[dict]:
    """Deterministic total order over Kestrel candidate rows: higher score first, then the ``id`` CURIE.

    The single-node selection in each Kestrel annotator (``term_results[0]``, ``max(..., key=score)``,
    and the first-on-category scan) trusts the order candidates arrive in. Kestrel returns them
    score-descending, but breaks *exact* score ties by response order, which varies run-to-run and so
    flipped which node an annotator committed (the residual "Axis 3" non-determinism the resolver's own
    tie-break cannot reach, because it acts on the candidate SET the annotators already reduced to one
    node each). Sorting once at ingestion resolves an exact score tie by the ``id`` CURIE. This is a
    determinism fix, not a ranking change: rows with distinct scores keep their order, and a
    missing/None score sorts as ``0.0``. Applied at ingestion so the pure selection helpers stay
    order-trusting and their unit tests stay valid.
    """

    def _key(row: dict) -> tuple[float, str]:
        try:
            score = float(row.get("score"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            score = 0.0
        # A NaN score coerces fine but is unorderable (every NaN comparison is False), so timsort would
        # leave NaN rows in arrival order — the very non-determinism this helper removes. Fold NaN to 0.0.
        if score != score:  # noqa: PLR0124 — NaN check
            score = 0.0
        return (-score, str(row.get("id") or ""))

    return sorted(rows or [], key=_key)


def is_on_category(row: dict, accepted: set[str] | None) -> bool:
    """True if the committed node's Biolink type is compatible with the queried category.

    Values are deliberately not restated here. ``studies/analysis/off_category_audit.py`` emits every
    figure behind this function into its committed artifact; the relevant fields are named inline below.

    **This is a CATEGORY check, never a NAMESPACE check.** Writing it as "the committed node must be
    in a canonical namespace" looks nearly identical in a diff, but it would additionally refuse a
    substantial population of on-category commits in non-canonical namespaces that this check keeps —
    LIPID MAPS, UMLS, UNII, MESH, PubChem, KEGG and others — including plainly-correct ones such as
    ``S-adenosylhomocysteine -> UNII:8K31Q2S66S``. Size and per-namespace breakdown, at one stated
    scope: artifact field ``namespace_whitelist_cost``. Namespace preference is ``_select_canonical``'s
    job and stays there.

    Failure-open in two shapes, because an absent type assertion is not a wrong type assertion:
    - no ``categories`` at all (missing, None, or empty), and
    - a *pure* top-of-hierarchy sentinel. A scan of live candidate rows found the empty/missing case
      does not occur in practice — that clause is belt-and-braces — while the pure-``biolink:NamedThing``
      case does, on nodes that are legitimate chemicals the KG simply failed to type. Counts and
      examples: artifact field ``failure_open_candidate_scan``. ``biolink:NamedThing`` is not among the
      descendants of ``biolink:ChemicalEntity``, so without this clause the guard would drop exactly
      the case it exists to protect. "Pure" matters: a sentinel alongside a real off-category type
      (``['biolink:NamedThing', 'biolink:Pathway']``) IS a type assertion and is judged normally.

    **Why a validator here and not a filter on the candidate pool.** Filtering the pool and letting
    the selector promote the best surviving chemical cannot improve a wrong answer into a right one:
    ``_select_canonical`` already prefers CHEBI/HMDB/RM, so a promotion can only happen when no
    canonical node was in the pool at all — otherwise the pool filter would be promoting a node the
    selector had already declined. What promotion does produce is a *different* wrong node that now
    passes the type test, which is strictly harder to audit than the ``EFO:...measurement`` node it
    replaced. Refusing is the honest outcome, so the check runs on the committed node.

    ``accepted=None`` disables the guard entirely — the byte-for-byte guarantee for the gene path
    and for every category with no configured acceptance root.
    """
    if accepted is None:
        return True
    categories = set(row.get("categories") or [])
    if not categories or categories <= TOP_OF_HIERARCHY_SENTINELS:
        return True
    return bool(categories & accepted)


class BaseAnnotator(ABC):  # Inherit from ABC

    # Subclasses must define this
    slug: str = NotImplemented

    def prepare(
        self, item: dict | pd.Series | pd.DataFrame, provided_id_fields: list[str]
    ) -> dict | pd.Series | pd.DataFrame:
        """
        Prepare entity/entities before annotation. Override to customize.

        Common use: removing provided_ids to prevent annotators from "cheating"
        by seeing "ground truth" IDs during evaluation.

        Args:
            item: Entity or entities to prepare
            provided_id_fields: List of field names containing ground truth IDs

        Returns:
            Prepared entity/entities (default: unchanged)
        """
        return item

    def build_availability_cache(
        self, items: dict | pd.Series | pd.DataFrame, name_field: str
    ) -> dict[str, Any] | None:
        """Prefetch external data once so the vote and the availability signal share ONE fetch.

        Default: ``None`` — an annotator with no external dependency needs no cache and reports
        ``not_queried`` regardless. Only the RefMet annotator (Metabolomics Workbench) overrides
        this, because it alone can distinguish a genuine no-match from an unreachable service.
        Building the cache once and threading it to both ``get_annotations`` and ``get_availability``
        is what keeps a degraded row from being fetched (and breaker-counted) twice.
        """
        return None

    def get_availability(self, entity: dict | pd.Series, name_field: str, cache: dict | None = None) -> dict[str, str]:
        """This annotator's availability for one row, keyed by slug.

        Default ``not_queried``: only an annotator that observes service availability reports
        anything else. Kept SEPARATE from ``get_annotations`` on purpose — an UNAVAILABLE outcome
        must never be folded into an empty vote, which is indistinguishable from a genuine no-match.
        """
        return {self.slug: AVAILABILITY_NOT_QUERIED}

    def get_source(self, entity: dict | pd.Series, name_field: str, cache: dict | None = None) -> dict[str, str]:
        """WHICH source served this annotator's vote for one row, keyed by slug.

        A PARALLEL provenance channel to ``get_availability`` (same shape, never folded into it):
        availability records whether a source answered, source records which one. Default
        ``not_queried`` — only the RefMet annotator, which can serve a row from a pinned local freeze
        instead of the live endpoint, reports anything else. Kept as its own method so the engine can
        accumulate a total per-row source map exactly the way it accumulates availability.
        """
        return {self.slug: REFMET_SOURCE_NOT_QUERIED}

    @abstractmethod
    def get_annotations(
        self,
        entity: dict | pd.Series,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        prefer_human: bool = True,
        preferred_prefixes: set[str] | None = None,
        accepted_categories: set[str] | None = None,
        candidate_limit: int | None = None,
        cache: dict | None = None,
    ) -> AssignedIDsDict:
        """
        Get annotations for a single entity.

        Args:
            entity: Entity to annotate (dict or DataFrame row)
            name_field: Name of the field containing the entity name
            category: Biolink category (standardized entity type)
            prefixes: Allowed (standardized) curie prefixes to map to (e.g., 'CHEBI', 'MONDO')
            prefer_human: When True (and the category is gene/protein-applicable, as gated by the
                engine), prefer the human (HGNC-bearing) candidate. Honored only by annotators where
                a human marker applies; others accept and ignore it.
            preferred_prefixes: When set (the engine resolves it for non-gene categories with a configured
                canonical-namespace policy), prefer the candidate in that namespace set. Honored only by
                annotators that re-rank (hybrid search); others accept and ignore it.
            accepted_categories: When set (the engine resolves it for categories with a configured
                acceptance root), the committed node's Biolink ``categories`` must intersect this set or
                the annotator refuses rather than committing an off-category node. A correctness guard,
                not a preference: it is independent of prefer_canonical/prefer_human, and None means
                unfiltered. Honored only by annotators that filter candidates; others accept and ignore it.
            candidate_limit: When set, the search ``limit`` each Kestrel annotator uses directly,
                overriding the adaptive default (20 with a re-ranking policy, else 1 for hybrid). None
                means "use the adaptive default". Honored only by the Kestrel search annotators; others
                accept and ignore it.
            cache: Optional pre-fetched results from bulk API call

        Returns:
            Dict with annotation results
        """
        pass

    @abstractmethod
    def get_annotations_bulk(
        self,
        entities: pd.DataFrame,
        name_field: str,
        category: str,
        prefixes: list[str] | None = None,
        prefer_human: bool = True,
        preferred_prefixes: set[str] | None = None,
        accepted_categories: set[str] | None = None,
        candidate_limit: int | None = None,
    ) -> pd.Series:  # Series of AssignedIdsDicts
        """
        Get annotations for multiple entities with bulk API call.

        Args:
            entities: DataFrame where each row is an entity
            name_field: Name of the column containing entity names
            category: Biolink category (standardized entity type)
            prefixes: Allowed (standardized) curie prefixes to map to (e.g., 'CHEBI', 'MONDO')
            prefer_human: See get_annotations. Accepted by all annotators; honored where applicable.
            preferred_prefixes: See get_annotations. Accepted by all annotators; honored where applicable.
            accepted_categories: See get_annotations. Accepted by all annotators; honored where applicable.
            candidate_limit: See get_annotations. Accepted by all annotators; honored where applicable.

        Returns:
            Column (Series) of annotation results (same index as input)
        """
        pass
