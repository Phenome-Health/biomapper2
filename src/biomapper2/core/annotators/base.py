from abc import ABC, abstractmethod

import pandas as pd

from ...utils import AssignedIDsDict

# A node typed only at the top of the Biolink hierarchy is an ABSENT type assertion,
# not an off-category claim, so the category validator lets it through (see `is_on_category`).
TOP_OF_HIERARCHY_SENTINELS = frozenset({"biolink:NamedThing", "biolink:Entity"})


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

        Returns:
            Column (Series) of annotation results (same index as input)
        """
        pass
