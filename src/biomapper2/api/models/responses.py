"""Pydantic response models for biomapper2 API."""

from typing import Any

from pydantic import BaseModel, Field


class RequestMetadata(BaseModel):
    """Metadata about the API request."""

    request_id: str = Field(..., description="Unique identifier for this request")
    processing_time_ms: float = Field(..., description="Time taken to process the request in milliseconds")


class ResolutionCertificateModel(BaseModel):
    """What the graph asserts about the chosen node, and what independent evidence says about it.

    Scope: this describes ``chosen_kg_id`` and nothing else. ``chosen_kg_id_provided`` and
    ``chosen_kg_id_assigned`` are resolved without a category and carry no certificate.
    """

    state: str = Field(
        ...,
        description=(
            "'corroborated' | 'uncorroborated' | 'contradicted' | 'unavailable' | 'not_applicable'. "
            "'contradicted' means A HUMAN SHOULD LOOK — an independent registry returned a different "
            "structure for the query name — never that the resolver is wrong: name lookup at an "
            "external registry can itself return a related-but-different compound. 'unavailable' "
            "means no structure was available to check against, which is unverifiable, NOT wrong. "
            "'not_applicable' means the entity is outside the small-molecule population this "
            "certificate is defined for (e.g. a gene)."
        ),
    )
    structure_status: str = Field(
        ...,
        description="'structure_present' | 'structure_absent' | 'not_applicable' — what the KG "
        "asserts about the chosen node's InChIKey. Never an external lookup.",
    )
    node_inchikey_blocks: list[str] = Field(
        default_factory=list, description="Sorted InChIKey first blocks the KG asserts for the chosen node"
    )
    comparison_rule: str = Field(..., description="Identifier of the rule that produced the verdict")
    equivalent_ids_lookup_ok: bool = Field(
        ..., description="False when the /get-nodes enrichment call failed; a failed lookup is not 'no structure'"
    )
    selection_conflict: str | None = Field(
        default=None,
        description="Intra-KG selection conflict ('divergent_refmet' | 'conflict_no_structure'). A "
        "DIFFERENT axis from 'state': both sides come from the graph, so it is not a contradiction.",
    )
    independent_source: str | None = Field(default=None, description="Registry consulted for independent evidence")
    independent_inchikey_block: str | None = Field(
        default=None, description="InChIKey first block that registry returned FOR THE QUERY NAME"
    )
    independent_of_selection: bool | None = Field(
        default=None,
        description="False when the independent source is the same registry that supplied the chosen "
        "node — corroboration there is circular. None when no independent lookup was made.",
    )
    tier_b_outcome: str = Field(
        default="off",
        description="'off' | 'resolved' | 'unresolvable' | 'lookup_failed'. A failed lookup is kept "
        "distinct from an unresolvable name so a throttled service is never read as name difficulty.",
    )
    refusal_reason: str | None = Field(
        default=None,
        description="Reserved. Until the refusal-reason change ships, an off-category refusal and a "
        "no-match are not distinguishable in this response.",
    )
    provenance: dict[str, Any] = Field(default_factory=dict, description="Tier B state, cache stores and expiry policy")


class EntityMappingResult(BaseModel):
    """Result of mapping a single entity to knowledge graph nodes."""

    name: str = Field(..., description="Entity name")
    curies: list[str] = Field(default_factory=list, description="Normalized CURIEs for the entity")
    chosen_kg_id: str | None = Field(default=None, description="Best knowledge graph node ID chosen by resolution")
    chosen_kg_id_review: str | None = Field(
        default=None,
        description="DEPRECATED — read resolution_certificate.selection_conflict instead, which this "
        "field is now derived from. Human-review flag for source-weighted small-molecule ChEBI "
        "conflicts ('divergent_refmet' | 'conflict_no_structure'); None when no review is warranted",
    )
    resolution_certificate: ResolutionCertificateModel | None = Field(
        default=None,
        description="Structural certificate for chosen_kg_id (and only chosen_kg_id — "
        "chosen_kg_id_provided and chosen_kg_id_assigned carry none). Null when mapping failed.",
    )
    kg_equivalent_ids: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Equivalent identifiers from the resolved KG node, grouped by CURIE prefix",
    )
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


class EntityType(BaseModel):
    """A single entity type with optional aliases and default vocabulary prefixes."""

    type: str = Field(..., description="Biolink category string (e.g. 'biolink:SmallMolecule')")
    aliases: list[str] | None = Field(default=None, description="Human-friendly alias names for this type")
    default_prefixes: list[str] | None = Field(
        default=None, serialization_alias="defaultPrefixes", description="Default vocabulary prefixes for this type"
    )


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
