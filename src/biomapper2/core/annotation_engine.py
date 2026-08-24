"""
Entity annotation module for assigning ontology/vocabulary local IDs.

Queries external APIs or uses other creative approaches to retrieve additional identifiers for biological entities.
"""

import logging
from copy import deepcopy
from functools import cached_property
from typing import Any

import pandas as pd

from ..biolink_client import BiolinkClient
from ..config import CATEGORY_ACCEPTED_ROOTS, CATEGORY_PREFERRED_NAMESPACES
from ..utils import AnnotationMode, AssignedIDsDict
from .annotators.base import BaseAnnotator
from .annotators.goslin_lipid import GoslinLipidAnnotator
from .annotators.kestrel_hybrid import KestrelHybridSearchAnnotator
from .annotators.kestrel_text import KestrelTextSearchAnnotator
from .annotators.kestrel_vector import KestrelVectorSearchAnnotator
from .annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator


class AnnotationEngine:
    """Engine for annotating biological entities with additional ontology IDs."""

    def __init__(self, biolink_client: BiolinkClient | None = None):
        """Initialize the annotation engine and set up available annotators."""
        self.annotator_registry: dict[str, BaseAnnotator] = {
            annotator.slug: annotator
            for annotator in [
                KestrelHybridSearchAnnotator(),
                KestrelTextSearchAnnotator(),
                KestrelVectorSearchAnnotator(),
                MetabolomicsWorkbenchAnnotator(),
                GoslinLipidAnnotator(),
            ]
        }
        self.biolink_client = biolink_client if biolink_client else BiolinkClient()

    def annotate(
        self,
        item: pd.Series | dict[str, Any] | pd.DataFrame,
        name_field: str,
        provided_id_fields: list[str],
        category: str,
        prefixes: list[str],
        mode: AnnotationMode = "missing",
        annotators: list[str] | None = None,
        prefer_human: bool = True,
        prefer_canonical: bool = True,
    ) -> pd.DataFrame | pd.Series:
        """
        Annotate entity with additional vocab IDs, obtained using various internal or external methods.

        Args:
            item: Entity or entities to annotate
            name_field: Field containing entity name
            provided_id_fields: Fields containing existing IDs
            category: Biolink category (standardized entity type - e.g., 'biolink:SmallMolecule')
            prefixes: Allowed (standardized) curie prefixes to map to (e.g., 'CHEBI', 'MONDO')
            mode: When to annotate
                - 'all': Annotate all entities
                - 'missing': Only annotate entities without provided_ids (default)
                - 'none': Skip annotation entirely (returns empty)
            annotators: Optional list of annotators to use (by slug). If None, annotators are selected automatically.
            prefer_human: When True, prefer the human (HGNC-bearing) candidate for gene/protein
                categories. This engine gates applicability by category (see below) and passes the
                resolved, effective flag to the annotators.
            prefer_canonical: When True, prefer the canonical-namespace candidate for non-gene
                categories with a configured policy (e.g. CHEBI/HMDB/RM for metabolites, MONDO for
                disease). The engine resolves the category's preferred-prefix set and passes it down;
                gene/protein categories never receive a set (they use prefer_human).

        Note: the engine also resolves an ``accepted_categories`` set from ``CATEGORY_ACCEPTED_ROOTS``
        and passes it down. It has no request-level flag on purpose — it is a correctness guard on the
        committed node's Biolink category, not a re-ranking preference, so it is independent of both
        ``prefer_canonical`` and ``prefer_human``.

        Returns:
            AssignedIDsDict (in a named Series) for single entity, and in a single-column
            pd.DataFrame for multiple entities.
        """
        valid_modes = {"all", "missing", "none"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        # Human-preference applies only to gene/protein categories (and their Biolink descendants).
        # Resolve the effective flag here so annotators receive an already-gated boolean.
        effective_prefer_human = prefer_human and self._is_human_applicable_category(category)

        # Canonical-namespace preference applies to non-gene categories with a configured policy. The
        # `not effective_prefer_human` clause enforces the mutually-exclusive partition (a category that
        # is gene/protein-applicable never also receives a canonical set), even if the descendant sets
        # overlap via mixins. None means "no canonical re-rank for this category".
        effective_preferred_prefixes = (
            self._category_preferred_prefixes.get(category) if prefer_canonical and not effective_prefer_human else None
        )

        # Category acceptance is a CORRECTNESS GUARD, not a re-ranking preference, so it is resolved
        # independently of both prefer_canonical and prefer_human — it must not inherit the canonical
        # policy's kill switch. None means "this category is unfiltered" (the gene path's guarantee).
        effective_accepted_categories = self._category_accepted_categories.get(category)

        logging.info(
            f"Beginning annotation step.. (mode={mode}, annotators={annotators}, "
            f"prefer_human={prefer_human}, effective_prefer_human={effective_prefer_human}, "
            f"prefer_canonical={prefer_canonical}, effective_preferred_prefixes={effective_preferred_prefixes}, "
            f"effective_accepted_categories={effective_accepted_categories})"
        )

        # Skip annotation if user requested it
        if mode == "none":
            logging.info(f"Skipping all annotation since mode={mode}")
            return self._get_empty_assigned_ids(item)

        if annotators is None:
            annotator_slugs = self._select_annotators(category)
        else:
            if not annotators:
                logging.warning(
                    "Empty annotators list provided; skipping annotation "
                    "(consider using annotation_mode='none' instead)"
                )
            annotator_slugs = annotators

        invalid_slugs = set(annotator_slugs).difference(set(self.annotator_registry.keys()))
        if invalid_slugs:
            raise ValueError(
                f"Invalid annotator slug(s) provided: {invalid_slugs}. "
                f"Valid options are: {list(self.annotator_registry.keys())}"
            )
        annotators_to_use = [self.annotator_registry[slug] for slug in annotator_slugs]

        if annotators_to_use:
            # Get annotations using selected annotators
            logging.info(f"Using Annotators: {annotator_slugs}")

            if isinstance(item, pd.DataFrame):
                return self._annotate_dataframe(
                    item,
                    name_field,
                    provided_id_fields,
                    mode,
                    category,
                    prefixes,
                    annotators_to_use,
                    effective_prefer_human,
                    effective_preferred_prefixes,
                    effective_accepted_categories,
                )
            else:
                return self._annotate_single(
                    item,
                    name_field,
                    provided_id_fields,
                    mode,
                    category,
                    prefixes,
                    annotators_to_use,
                    effective_prefer_human,
                    effective_preferred_prefixes,
                    effective_accepted_categories,
                )
        else:
            return self._get_empty_assigned_ids(item)

    # ------------------------------------- Helper methods --------------------------------------- #

    @cached_property
    def _human_applicable_categories(self) -> set[str]:
        """Gene/Protein and their Biolink descendants — categories where human-preference applies.

        Cached per instance: the Biolink hierarchy is static at runtime, so this avoids recomputing the
        descendant union on every annotate() call (which happens once per entity on the single/stream paths).
        Lazy (computed on first use) so constructing the engine doesn't force the lookup eagerly.
        """
        return self.biolink_client.get_descendants("biolink:Gene") | self.biolink_client.get_descendants(
            "biolink:Protein"
        )

    def _is_human_applicable_category(self, category: str) -> bool:
        """True if the category is a gene/protein (or Biolink descendant), where human-preference applies."""
        return category in self._human_applicable_categories

    @cached_property
    def _category_preferred_prefixes(self) -> dict[str, set[str]]:
        """Map every category in the canonical-namespace policy (incl. Biolink descendants) to its set.

        Expands each configured key (e.g. ``biolink:SmallMolecule``, ``biolink:Disease``) via
        ``get_descendants`` so subcategories inherit the policy, mirroring ``_human_applicable_categories``.
        Cached per instance: the Biolink hierarchy is static at runtime. If two configured keys' descendant
        sets overlap (e.g. via a shared mixin), the shared category inherits the **union** of both policies'
        prefixes rather than silently dropping one.
        """
        resolved: dict[str, set[str]] = {}
        for category, prefixes in CATEGORY_PREFERRED_NAMESPACES.items():
            for descendant in self.biolink_client.get_descendants(category):
                resolved.setdefault(descendant, set()).update(prefixes)
        return resolved

    @cached_property
    def _category_accepted_categories(self) -> dict[str, set[str]]:
        """Map every category in the acceptance policy (incl. Biolink descendants) to its accepted set.

        Both sides of ``CATEGORY_ACCEPTED_ROOTS`` are expanded via ``get_descendants``, but for different
        reasons: the **key** so subcategories of a configured job category inherit the policy (mirroring
        ``_category_preferred_prefixes``), and the **value** because the acceptance root stands for its
        whole subtree — ``biolink:ChemicalEntity`` admits all 12 chemical categories, so a legitimately
        broader typing survives the guard. On overlap the category inherits the **union** of both roots'
        subtrees (accept-wider), which keeps the guard failure-open in the ambiguous case.

        Cached per instance: the Biolink hierarchy is static at runtime.
        """
        resolved: dict[str, set[str]] = {}
        for category, root in CATEGORY_ACCEPTED_ROOTS.items():
            accepted = self.biolink_client.get_descendants(root)
            for descendant in self.biolink_client.get_descendants(category):
                resolved.setdefault(descendant, set()).update(accepted)
        return resolved

    def _select_annotators(self, category: str) -> list[str]:
        """Select appropriate annotators based on entity type (returns their slugs)."""
        logging.info(f"Selecting annotators for category '{category}'")
        annotators: list[str] = []

        # Choose which annotators to use considering biolink category and its descendants
        category_with_descendants = self.biolink_client.get_descendants(category)
        if category_with_descendants.intersection({"biolink:SmallMolecule"}):
            annotators.append(MetabolomicsWorkbenchAnnotator.slug)
            annotators.append(GoslinLipidAnnotator.slug)

        # Always include fallback annotator (temp: orchestration will become more advanced later)
        annotators.append(KestrelHybridSearchAnnotator.slug)

        return annotators

    def _annotate_dataframe(
        self,
        df: pd.DataFrame,
        name_field: str,
        provided_id_fields: list[str],
        mode: AnnotationMode,
        category: str,
        prefixes: list[str],
        annotators: list,
        prefer_human: bool = True,
        preferred_prefixes: set[str] | None = None,
        accepted_categories: set[str] | None = None,
    ) -> pd.DataFrame:
        """Annotate an entire DataFrame. Returns a single-column DataFrame containing AssignedIDsDicts."""
        if mode == "missing":
            # Identify rows that need annotation (no provided_ids)
            has_ids_mask = df[provided_id_fields].notna().any(axis=1)
            needs_annotation_mask = ~has_ids_mask
            items_to_annotate = df[needs_annotation_mask]
            logging.info(
                f"Annotation mode is set to 'missing': annotating {needs_annotation_mask.sum()} of {len(df)} rows"
            )
        else:  # mode == 'all'
            items_to_annotate = df
            needs_annotation_mask = pd.Series([True] * len(df), index=df.index)

        # Initialize results column with empty dicts for all rows
        assigned_ids_col = pd.Series([{} for _ in range(len(df))], index=df.index)

        # Only annotate rows that need it
        if not items_to_annotate.empty:
            annotated_rows = pd.Series([{} for _ in range(len(items_to_annotate))], index=items_to_annotate.index)

            for annotator in annotators:
                prepared_df = annotator.prepare(items_to_annotate, provided_id_fields)
                annotations_col = annotator.get_annotations_bulk(
                    prepared_df,
                    name_field,
                    category,
                    prefixes,
                    prefer_human=prefer_human,
                    preferred_prefixes=preferred_prefixes,
                    accepted_categories=accepted_categories,
                )
                annotated_rows = pd.Series(
                    [self._merge_nested_dicts(d1, d2) for d1, d2 in zip(annotated_rows, annotations_col)],
                    index=annotated_rows.index,
                )

            # Merge partial results back into full results
            assigned_ids_col[needs_annotation_mask] = annotated_rows

        return pd.DataFrame({"assigned_ids": assigned_ids_col})

    def _annotate_single(
        self,
        item: pd.Series | dict[str, Any],
        name_field: str,
        provided_id_fields: list[str],
        mode: AnnotationMode,
        category: str,
        prefixes: list[str],
        annotators: list,
        prefer_human: bool = True,
        preferred_prefixes: set[str] | None = None,
        accepted_categories: set[str] | None = None,
    ) -> pd.Series:
        """Annotate a single entity. Returns named series containing AssignedIDsDict."""
        # If user requested it, skip entities that have any provided IDs
        if mode == "missing":
            has_provided_ids = any(pd.notna(item.get(field)) for field in provided_id_fields)
            if has_provided_ids:
                logging.debug(f"Entity has provided_ids and mode={mode}, skipping annotation")
                return self._get_empty_assigned_ids_for_entity(item)

        # Otherwise get assigned IDs for the entity
        assigned_ids = dict()  # All annotations will be merged into this
        for annotator in annotators:
            prepared_entity = annotator.prepare(item, provided_id_fields)
            entity_annotations = annotator.get_annotations(
                prepared_entity,
                name_field,
                category,
                prefixes,
                prefer_human=prefer_human,
                preferred_prefixes=preferred_prefixes,
                accepted_categories=accepted_categories,
            )
            assigned_ids: AssignedIDsDict = self._merge_nested_dicts(assigned_ids, entity_annotations)

        return pd.Series({"assigned_ids": assigned_ids})  # Named Series

    @staticmethod
    def _merge_nested_dicts(d1: AssignedIDsDict, d2: AssignedIDsDict) -> AssignedIDsDict:
        """
        Merge two nested dicts with structure:
        Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]

        Structure: {annotator: {vocabulary: {id: result_metadata_dict}}}
        """
        result = deepcopy(d1)

        for annotator, vocab_dict in d2.items():
            if annotator in result:
                # Annotator exists, merge vocabularies
                for vocab, id_dict in vocab_dict.items():
                    if vocab in result[annotator]:
                        # Vocabulary exists, merge IDs
                        for id_str, result_metadata in id_dict.items():
                            if id_str in result[annotator][vocab]:
                                # ID exists, update/merge the metadata dict
                                result[annotator][vocab][id_str].update(result_metadata)
                            else:
                                # New ID
                                result[annotator][vocab][id_str] = result_metadata
                    else:
                        # New vocabulary
                        result[annotator][vocab] = deepcopy(id_dict)
            else:
                # New annotator
                result[annotator] = deepcopy(vocab_dict)

        return result

    def _get_empty_assigned_ids(self, item: pd.Series | dict[str, Any] | pd.DataFrame) -> pd.DataFrame | pd.Series:
        """Return empty assigned_ids in appropriate format."""
        if isinstance(item, pd.DataFrame):
            return self._get_empty_assigned_ids_for_dataset(item)
        else:
            return self._get_empty_assigned_ids_for_entity(item)

    @staticmethod
    def _get_empty_assigned_ids_for_dataset(item: pd.DataFrame) -> pd.DataFrame:
        # Return single-column DataFrame with empty dicts
        empty_col = pd.Series([{} for _ in range(len(item))], index=item.index)
        return pd.DataFrame({"assigned_ids": empty_col})

    @staticmethod
    def _get_empty_assigned_ids_for_entity(item: pd.Series | dict[str, Any]) -> pd.Series:
        # Return named Series with empty dict
        return pd.Series({"assigned_ids": {}})
