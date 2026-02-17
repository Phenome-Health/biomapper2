"""Pydantic response models for biomapper2 API."""

from typing import Any

from pydantic import BaseModel, Field


class RequestMetadata(BaseModel):
    """Metadata about the API request."""

    request_id: str = Field(..., description="Unique identifier for this request")
    processing_time_ms: float = Field(..., description="Time taken to process the request in milliseconds")


class EntityMappingResult(BaseModel):
    """Result of mapping a single entity to knowledge graph nodes."""

    name: str = Field(..., description="Entity name")
    curies: list[str] = Field(default_factory=list, description="Normalized CURIEs for the entity")
    chosen_kg_id: str | None = Field(default=None, description="Best knowledge graph node ID chosen by resolution")
    kg_ids: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Map of KG node IDs to the CURIEs that linked to them",
    )
    assigned_ids: dict[str, Any] = Field(
        default_factory=dict,
        description="IDs assigned during annotation (raw API results)",
    )
    error: str | None = Field(default=None, description="Error message if mapping failed")


class EntityMappingResponse(BaseModel):
    """Response for single entity mapping."""

    result: EntityMappingResult
    metadata: RequestMetadata


class BatchMappingResponse(BaseModel):
    """Response for batch entity mapping."""

    results: list[EntityMappingResult]
    metadata: RequestMetadata
    summary: dict[str, int] = Field(
        default_factory=dict,
        description="Summary statistics (total, successful, failed)",
    )


class DatasetMappingResponse(BaseModel):
    """Response for dataset mapping."""

    output_file: str = Field(..., description="Path to the output TSV file")
    stats: dict[str, Any] = Field(..., description="Statistics about the mapping results")
    metadata: RequestMetadata


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    mapper_initialized: bool = Field(..., description="Whether Mapper is ready")


class AnnotatorInfo(BaseModel):
    """Information about an annotator."""

    slug: str = Field(..., description="Unique identifier for the annotator")
    name: str = Field(..., description="Human-readable name")
    description: str | None = Field(default=None, description="Description of what this annotator does")


class AnnotatorsResponse(BaseModel):
    """Response listing available annotators."""

    annotators: list[AnnotatorInfo]


class EntityTypesResponse(BaseModel):
    """Response listing supported entity types."""

    entity_types: list[str] = Field(..., description="Biolink entity types supported")
    aliases: dict[str, str] = Field(..., description="Common aliases mapped to Biolink types")


class VocabularyInfo(BaseModel):
    """Information about a vocabulary."""

    prefix: str = Field(..., description="Standard CURIE prefix")
    iri: str | None = Field(default=None, description="Base IRI for the vocabulary")
    aliases: list[str] = Field(default_factory=list, description="Alternative names for this vocabulary")


class VocabulariesResponse(BaseModel):
    """Response listing supported vocabularies."""

    vocabularies: list[VocabularyInfo]
    count: int = Field(..., description="Total number of vocabularies")


class ErrorResponse(BaseModel):
    """Error response model."""

    detail: str = Field(..., description="Error message")
    error_type: str | None = Field(default=None, description="Type of error")
