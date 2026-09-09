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
    prefer_human: bool = Field(
        default=True,
        description=(
            "For gene/protein entities, prefer the human (HGNC-bearing) candidate that matches the "
            "queried symbol over a wrong-species ortholog. Default True. Set False to restore legacy "
            "top-scored selection. No effect on metabolites or other non-gene/protein categories."
        ),
    )
    prefer_canonical: bool = Field(
        default=True,
        description=(
            "For non-gene categories with a configured canonical-namespace policy (e.g. CHEBI/HMDB/RefMet "
            "for metabolites, MONDO for disease), prefer the canonical-namespace node that matches the "
            "query over a same-text node from a non-canonical vocabulary (UMLS/ICD/KEGG/...). Default True. "
            "Set False to restore legacy top-scored selection. No effect on gene/protein (which use "
            "prefer_human)."
        ),
    )
    candidate_limit: int | None = Field(
        default=None,
        ge=1,
        le=100,
        description=(
            "Number of candidate nodes each Kestrel search annotator retrieves (the search 'limit'). "
            "When set, this value is used directly, overriding the adaptive default (20 when a "
            "re-ranking policy is active, else 1 for hybrid). When None (default), behavior is "
            "unchanged. Bounded to 1..100; out-of-range values are rejected with a 422."
        ),
    )

    # Ignore unknown fields so a newer client sending extra options to an older server does not 422.
    # (Pydantic v2 already defaults to ignore; set explicitly as documentation of the contract.)
    model_config = {"extra": "ignore"}


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
