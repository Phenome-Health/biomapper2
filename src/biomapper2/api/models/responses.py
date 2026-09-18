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
        description="'off' | 'resolved' | 'unresolvable' | 'lookup_failed' | 'ambiguous' | "
        "'out_of_scope'. A failed lookup is kept distinct from an unresolvable name so a throttled "
        "service is never read as name difficulty. 'off' means Tier B was disabled for the run; "
        "'out_of_scope' means it was enabled but this row is not one an independent lookup can "
        "adjudicate, so the two are never conflated.",
    )
    lipid_resolution_level: str = Field(
        default="unavailable",
        description="Graded agreement of a STRUCTURE-FREE lipid check ('lipid_species' | 'contradicted' "
        "| 'unavailable'), on an axis PARALLEL to the InChIKey-block 'resolution_level'. Set only for a "
        "committed lipid node the graph lists no InChIKey for; 'unavailable' on every other row.",
    )
    refusal_reason: str | None = Field(
        default=None,
        description="Reserved. Until the refusal-reason change ships, an off-category refusal and a "
        "no-match are not distinguishable in this response.",
    )
    refmet_availability: str = Field(
        default="not_queried",
        description="Whether the RefMet source (Metabolomics Workbench) answered for this row: "
        "'voted' | 'no_match' | 'unavailable' | 'not_queried'. A runtime availability signal like "
        "'equivalent_ids_lookup_ok', NOT a verdict — 'unavailable' means the service did not answer, "
        "distinct from 'no_match' (it answered, no such metabolite). 'not_queried' when RefMet was "
        "not selected for the row.",
    )
    refmet_source: str = Field(
        default="not_queried",
        description="WHICH RefMet source served the row, on a parallel axis to 'refmet_availability': "
        "'local_snapshot' (the pinned freeze answered) | 'not_in_snapshot' (freeze loaded, name absent, "
        "deterministic no-match, no network) | 'live_api' (live /match answered) | 'unavailable' (live "
        "service did not answer) | 'not_queried'. With a freeze present the circuit breaker is out of "
        "the default path.",
    )
    refmet_snapshot_version: str | None = Field(
        default=None,
        description="Version of the pinned freeze that served (or was consulted for) the row, else "
        "None. Set only when 'refmet_source' is 'local_snapshot' or 'not_in_snapshot'.",
    )
    tier_b_snapshot_version: str | None = Field(
        default=None,
        description="Version of the Tier B freeze that produced a frozen independent result, else "
        "None for a live result. A first-class field mirroring 'refmet_snapshot_version' so frozen "
        "Tier B evidence is auditable.",
    )
    provenance: dict[str, Any] = Field(default_factory=dict, description="Tier B state, cache stores and expiry policy")


class LipidResolution(BaseModel):
    """Lipid hierarchy-aware resolution detail for a committed node. Null for non-lipid rows.

    Additive (R10): assembled from metadata the lipid units already produce (goslin parse + the
    committed node's matched level). ``mapping_relation`` is the canonical successor of the flat
    ``chosen_kg_id_lipid_hint`` field: ``mapping_relation == 'broad'`` is exactly the
    ``lipid_generalized`` case that hint marks.
    """

    query_lipid_level_asserted: str | None = Field(
        default=None, description="The input's real lipid level (goslin LipidLevel, lowercased), before trust policy"
    )
    query_lipid_level_effective: str | None = Field(
        default=None,
        description="The level actually queried after the sn-position trust policy (Decision 1): equals "
        "'query_lipid_level_asserted' unless a trust-off downgrade capped a slash-bearing input to a coarser level",
    )
    matched_lipid_level: str | None = Field(
        default=None,
        description="The level at which the committed node matched, joined to the goslin votes by the same "
        "raw-id vs curie reconciliation the resolver tie-break uses; None when no goslin vote backs the node",
    )
    mapping_relation: str = Field(
        default="unknown",
        description="Relation of the committed node's matched level to the effective query level: "
        "'exact' | 'broad' | 'narrow' | 'unknown'. 'broad' means the committed node is coarser than the "
        "query asked for (a generalization); 'narrow' should not occur under Decision 2 but is represented "
        "honestly if seen; 'unknown' when a level is missing",
    )
    mapping_predicate: str | None = Field(
        default=None,
        description="SKOS predicate for 'mapping_relation': 'skos:exactMatch' | 'skos:broadMatch' | "
        "'skos:narrowMatch'; None when the relation is 'unknown'",
    )
    query_transformed: str | None = Field(
        default=None,
        description="How the query name was transformed before lookup: 'slash_to_underscore' when a "
        "trust-off downgrade rewrote a slash-bearing input, else 'goslin_species_canonical'",
    )
    ambiguous: bool = Field(
        default=False,
        description="Whether the committed node came from a structurally ambiguous candidate set. Defaults "
        "to false: the LIPID MAPS / Tier B ambiguity signal is not yet threaded to this surface, so no "
        "ambiguity is asserted rather than fabricated",
    )
    candidate_structure_count: int | None = Field(
        default=None,
        description="Number of distinct structures in the candidate set when an ambiguity signal is present; "
        "None when that signal does not reach this row",
    )
    ambiguity_basis: str | None = Field(
        default=None,
        description="What the ambiguity is grounded in (e.g. 'lipidmaps_abbrev_chains') when 'ambiguous' is "
        "true; None otherwise",
    )
    goslin_dialect: str | None = Field(default=None, description="Goslin grammar dialect that parsed the input name")
    goslin_formula: str | None = Field(default=None, description="Sum formula from the goslin parse")
    goslin_mass: float | None = Field(default=None, description="Monoisotopic mass from the goslin parse")


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
    chosen_kg_id_lipid_hint: str | None = Field(
        default=None,
        description="Additive lipid review hint for chosen_kg_id, on its own axis from the closed "
        "selection_conflict whitelist: 'lipid_generalized' when the committed lipid node is BROADER "
        "than the effective lipid query level (a generalization the resolver could not avoid); None "
        "otherwise and for non-lipid rows. Now folded into lipid_resolution.mapping_relation "
        "('broad' == this hint's 'lipid_generalized'); kept for one release for backward compatibility.",
    )
    lipid_resolution: LipidResolution | None = Field(
        default=None,
        description="Lipid hierarchy-aware resolution detail for chosen_kg_id; null for non-lipid rows. "
        "Additive object assembled from the goslin parse and the committed node's matched level.",
    )
    refmet_availability: str = Field(
        default="not_queried",
        description="Per-row RefMet (Metabolomics Workbench) availability, mirrored from the "
        "certificate so a consumer can flag/exclude rows a degraded RefMet service left uncovered: "
        "'voted' | 'no_match' | 'unavailable' | 'not_queried'. Always present (never None).",
    )
    refmet_source: str = Field(
        default="not_queried",
        description="Per-row RefMet source, mirrored from the certificate: 'local_snapshot' | "
        "'not_in_snapshot' | 'live_api' | 'unavailable' | 'not_queried'. Always present (never None).",
    )
    refmet_snapshot_version: str | None = Field(
        default=None,
        description="Version of the pinned freeze that served the row, mirrored from the certificate; "
        "None when the row was not served by (or consulted against) a freeze.",
    )
    tier_b_snapshot_version: str | None = Field(
        default=None,
        description="Version of the Tier B freeze that produced a frozen independent result for the "
        "row, mirrored from the certificate; None for a live (non-frozen) result.",
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
    summary: dict[str, int | dict[str, int]] = Field(
        default_factory=dict,
        description="Summary statistics (total, successful, failed, refmet_unavailable) plus "
        "'refmet_source_counts', a per-source tally of which RefMet source served each row.",
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
