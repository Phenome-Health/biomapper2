"""Pydantic models for API requests and responses."""

from .requests import BatchMappingRequest, DatasetMappingRequest, EntityMappingRequest, MappingOptions
from .responses import (
    AnnotatorInfo,
    AnnotatorsResponse,
    BatchMappingResponse,
    DatasetMappingResponse,
    EntityMappingResponse,
    EntityMappingResult,
    EntityTypesResponse,
    ErrorResponse,
    HealthResponse,
    RequestMetadata,
    VocabulariesResponse,
    VocabularyInfo,
)

__all__ = [
    # Requests
    "BatchMappingRequest",
    "DatasetMappingRequest",
    "EntityMappingRequest",
    "MappingOptions",
    # Responses
    "AnnotatorInfo",
    "AnnotatorsResponse",
    "BatchMappingResponse",
    "DatasetMappingResponse",
    "EntityMappingResponse",
    "EntityMappingResult",
    "EntityTypesResponse",
    "ErrorResponse",
    "HealthResponse",
    "RequestMetadata",
    "VocabulariesResponse",
    "VocabularyInfo",
]
