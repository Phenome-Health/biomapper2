from abc import ABC, abstractmethod

import pandas as pd

from ...utils import AssignedIDsDict


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
        cache: dict | None = None,
    ) -> AssignedIDsDict:
        """
        Get annotations for a single entity.

        Args:
            entity: Entity to annotate (dict or DataFrame row)
            name_field: Name of the field containing the entity name
            category: Biolink category (standardized entity type)
            prefixes: Allowed (standardized) curie prefixes to map to (e.g., 'CHEBI', 'MONDO')
            cache: Optional pre-fetched results from bulk API call

        Returns:
            Dict with annotation results
        """
        pass

    @abstractmethod
    def get_annotations_bulk(
        self, entities: pd.DataFrame, name_field: str, category: str, prefixes: list[str] | None = None
    ) -> pd.Series:  # Series of AssignedIdsDicts
        """
        Get annotations for multiple entities with bulk API call.

        Args:
            entities: DataFrame where each row is an entity
            name_field: Name of the column containing entity names
            category: Biolink category (standardized entity type)
            prefixes: Allowed (standardized) curie prefixes to map to (e.g., 'CHEBI', 'MONDO')

        Returns:
            Column (Series) of annotation results (same index as input)
        """
        pass
