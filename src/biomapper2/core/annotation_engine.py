"""
Entity annotation module for assigning ontology/vocabulary local IDs.

Queries external APIs or uses other creative approaches to retrieve additional identifiers for biological entities.
"""

import logging
from copy import deepcopy
from typing import Any, Literal

import pandas as pd

from ..utils import AssignedIDsDict, get_descendants, initialize_biolink_model_toolkit, standardize_entity_type
from .annotators.base import BaseAnnotator
from .annotators.kestrel_text import KestrelTextSearchAnnotator
from .annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator


class AnnotationEngine:
    """Engine for annotating biological entities with additional ontology IDs."""

    def __init__(self, biolink_version: str | None = None):
        """Initialize the annotation engine and set up available annotators."""
        self.kestrel_text_search_annotator = KestrelTextSearchAnnotator()
        self.metabolomics_workbench_annotator = MetabolomicsWorkbenchAnnotator()

        self.bmt = initialize_biolink_model_toolkit(biolink_version)

    def annotate(
        self,
        item: pd.Series | dict[str, Any] | pd.DataFrame,
        name_field: str,
        provided_id_fields: list[str],
        entity_type: str,
        mode: Literal["all", "missing", "none"] = "missing",
    ) -> pd.DataFrame | pd.Series:
        """
        Annotate entity with additional vocab IDs, obtained using various internal or external methods.

        Args:
            item: Entity or entities to annotate
            name_field: Field containing entity name
            provided_id_fields: Fields containing existing IDs
            entity_type: Type of entity (e.g., 'metabolite', 'protein')
            mode: When to annotate
                - 'all': Annotate all entities
                - 'missing': Only annotate entities without provided_ids (default)
                - 'none': Skip annotation entirely (returns empty)

        Returns:
            AssignedIDsDict (in a named Series) for single entity, and in a single-column
            pd.DataFrame for multiple entities.
        """
        valid_modes = {"all", "missing", "none"}
        if mode not in valid_modes:
            raise ValueError(f"Invalid mode '{mode}'. Must be one of: {valid_modes}")

        # Skip annotation if user requested it
        if mode == "none":
            logging.debug(f"Skipping all annotation since mode={mode}")
            return self._get_empty_assigned_ids(item)

        logging.debug("Beginning annotation step..")

        biolink_category, annotators = self._select_annotators(entity_type)

        if annotators:
            logging.debug(f"Using annotators: {annotators}")
        else:
            logging.warning("Did not identify any annotators to use for input item. Proceeding without assigned IDs.")
            return self._get_empty_assigned_ids(item)

        # Get annotations using selected annotators
        if isinstance(item, pd.DataFrame):
            return self._annotate_dataframe(item, name_field, provided_id_fields, mode, biolink_category, annotators)
        else:
            return self._annotate_single(item, name_field, provided_id_fields, mode, biolink_category, annotators)

    # ------------------------------------- Helper methods --------------------------------------- #

    def _select_annotators(self, entity_type: str) -> tuple[str, list[BaseAnnotator]]:
        """Select appropriate annotators based on entity type."""

        # Validate the entity type and convert it into a standard biolink category
        category = standardize_entity_type(entity_type, self.bmt)
        logging.debug(f"Biolink category for entity type '{entity_type}' is: {category}")

        # Choose which annotators to use considering biolink category and its descendants
        category_with_descendants = get_descendants(category, self.bmt)
        annotators: list[BaseAnnotator] = []
        if category_with_descendants.intersection({"biolink:SmallMolecule"}):
            annotators.append(self.metabolomics_workbench_annotator)

        # Always include fallback annotator (temp: orchestration will become more advanced later)
        annotators.append(self.kestrel_text_search_annotator)

        return category, annotators

    def _annotate_dataframe(
        self,
        df: pd.DataFrame,
        name_field: str,
        provided_id_fields: list[str],
        mode: str,
        category: str,
        annotators: list,
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
                annotations_col = annotator.get_annotations_bulk(prepared_df, name_field, category)
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
        mode: str,
        category: str,
        annotators: list,
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
            entity_annotations = annotator.get_annotations(prepared_entity, name_field, category)
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
