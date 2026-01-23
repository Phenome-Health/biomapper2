"""
Pydantic models for biomapper2 entities.

Provides type-safe internal representation of entities flowing through the mapping pipeline.
External API remains dict-based for data scientist friendliness.
"""

from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict

__all__ = ["AssignedIDsDict", "Entity"]

if TYPE_CHECKING:
    import pandas as pd

# Type alias for annotation results structure (matches Entity.assigned_ids)
# Structure: {annotator: {vocabulary: {local_id: result_metadata_dict}}}
AssignedIDsDict = dict[str, dict[str, dict[str, dict[str, Any]]]]


class Entity(BaseModel):
    """
    Represents a biological entity being mapped to a knowledge graph.

    Fields are accumulated through the four-stage pipeline:
    1. Input: name + user-provided ID fields (stored in model_extra)
    2. Annotation: assigned_ids from annotator APIs
    3. Normalization: curies, curies_provided, curies_assigned, invalid_ids
    4. Linking: kg_ids, kg_ids_provided, kg_ids_assigned
    5. Resolution: chosen_kg_id, chosen_kg_id_provided, chosen_kg_id_assigned
    """

    model_config = ConfigDict(extra="allow")

    # Required input field
    name: str

    # Annotation step output
    assigned_ids: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}

    # Normalization step outputs
    curies: list[str] = []
    curies_provided: list[str] = []
    curies_assigned: dict[str, list[str]] = {}
    invalid_ids_provided: dict[str, list[str]] = {}
    invalid_ids_assigned: dict[str, dict[str, list[str]]] = {}
    unrecognized_vocabs_provided: list[str] = []
    unrecognized_vocabs_assigned: list[str] = []

    # Linking step outputs
    kg_ids: dict[str, list[str]] = {}
    kg_ids_provided: dict[str, list[str]] = {}
    kg_ids_assigned: dict[str, dict[str, list[str]]] = {}

    # Resolution step outputs
    chosen_kg_id: str | None = None
    chosen_kg_id_provided: str | None = None
    chosen_kg_id_assigned: str | None = None

    @classmethod
    def from_input(cls, item: "pd.Series | dict[str, Any]", name_field: str = "name") -> "Entity":
        """
        Create an Entity from a dict or pandas Series.

        Args:
            item: Input data (dict or Series)
            name_field: Field containing entity name (copied to 'name' if different)

        User-provided fields (like kegg_id, pubchem_cid) are stored in model_extra.
        """
        import pandas as pd

        if isinstance(item, pd.Series):
            data = item.to_dict()
        else:
            data = dict(item)

        # Copy name_field value to 'name' if different
        if name_field != "name" and name_field in data:
            data["name"] = data[name_field]

        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """
        Convert Entity to a dict, including extra fields.

        Returns dict suitable for external API output.
        """
        result = self.model_dump()
        if self.model_extra:
            result.update(self.model_extra)
        return result

    def to_series(self) -> "pd.Series":
        """
        Convert Entity to a pandas Series.

        Returns Series suitable for DataFrame operations.
        """
        import pandas as pd

        return pd.Series(self.to_dict())

    def update_from(self, series: "pd.Series") -> "Entity":
        """
        Create a new Entity with fields updated from a pandas Series.

        Used to incorporate pipeline step outputs. Returns a new Entity
        (immutable update pattern).

        Args:
            series: pandas Series with fields to merge

        Returns:
            New Entity with merged fields
        """
        current = self.to_dict()
        current.update(series.to_dict())
        return Entity(**current)
