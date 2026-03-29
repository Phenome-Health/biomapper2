"""Pydantic request models for biomapper2 API."""

from typing import Literal

from pydantic import BaseModel, Field


class MappingOptions(BaseModel):
    """Options for entity mapping."""

    annotation_mode: Literal["all", "missing", "none"] = Field(
        default="missing",
        description="When to annotate: 'all' annotates everything, 'missing' only entities without IDs, 'none' skips",
    )
    annotators: list[str] | None = Field(
        default=None,
        description="List of annotator slugs to use. If None, annotators are selected automatically.",
    )
    array_delimiters: list[str] = Field(
        default=[",", ";"],
        description="Characters used to split delimited ID strings",
    )
    vocab: str | list[str] | None = Field(
        default=None,
        description="Allowed vocabulary name(s) to map to (e.g., 'refmet', 'chebi')",
    )


class EntityMappingRequest(BaseModel):
    """Request to map a single entity to knowledge graph nodes."""

    name: str = Field(..., description="Entity name (e.g., 'carnitine', 'glucose')")
    entity_type: str = Field(
        ...,
        description="Type of entity (e.g., 'metabolite', 'protein', 'SmallMolecule')",
    )
    identifiers: dict[str, str | list[str]] = Field(
        default_factory=dict,
        description="Known identifiers for the entity (e.g., {'kegg': 'C00487', 'pubchem': '10917'})",
    )
    options: MappingOptions = Field(default_factory=MappingOptions)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "carnitine",
                    "entity_type": "metabolite",
                    "identifiers": {"kegg": "C00487", "pubchem": "10917"},
                    "options": {"annotation_mode": "missing"},
                }
            ]
        }
    }


class BatchMappingRequest(BaseModel):
    """Request to map multiple entities at once."""

    entities: list[EntityMappingRequest] = Field(
        ...,
        description="List of entities to map",
        min_length=1,
        max_length=1000,
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "entities": [
                        {
                            "name": "carnitine",
                            "entity_type": "metabolite",
                            "identifiers": {"kegg": "C00487"},
                        },
                        {
                            "name": "glucose",
                            "entity_type": "metabolite",
                            "identifiers": {"pubchem": "5793"},
                        },
                    ]
                }
            ]
        }
    }


class DatasetMappingRequest(BaseModel):
    """Request to map a dataset (via file upload or JSON)."""

    entity_type: str = Field(
        ...,
        description="Type of entities in the dataset",
    )
    name_column: str = Field(
        ...,
        description="Column containing entity names",
    )
    provided_id_columns: list[str] = Field(
        ...,
        description="Columns containing existing identifiers",
    )
    options: MappingOptions = Field(default_factory=MappingOptions)
