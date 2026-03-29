"""Discovery and health check endpoints."""

from fastapi import APIRouter, Depends, Request

from ..auth import validate_api_key
from ..constants import API_VERSION
from ..models import AnnotatorInfo, AnnotatorsResponse, EntityTypesResponse, HealthResponse, VocabulariesResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check(request: Request) -> HealthResponse:
    """
    Health check endpoint.

    Returns service status, version, and whether the Mapper is initialized.
    This endpoint does not require authentication.
    """
    mapper = getattr(request.app.state, "mapper", None)
    mapper_error = getattr(request.app.state, "mapper_error", None)

    if mapper is not None:
        status = "healthy"
    elif mapper_error:
        status = f"degraded: {mapper_error}"
    else:
        status = "initializing"

    return HealthResponse(
        status=status,
        version=API_VERSION,
        mapper_initialized=mapper is not None,
    )


@router.get("/entity-types", response_model=EntityTypesResponse)
async def list_entity_types(
    request: Request,
    _api_key: str = Depends(validate_api_key),
) -> EntityTypesResponse:
    """
    List supported entity types.

    Returns Biolink entity types and common aliases that biomapper2 supports.
    """
    # Common entity types and their Biolink mappings
    aliases = {
        "metabolite": "biolink:SmallMolecule",
        "lipid": "biolink:SmallMolecule",
        "protein": "biolink:Protein",
        "gene": "biolink:Gene",
        "disease": "biolink:Disease",
        "phenotype": "biolink:PhenotypicFeature",
        "pathway": "biolink:Pathway",
        "drug": "biolink:Drug",
        "clinicallab": "biolink:ClinicalFinding",
        "lab": "biolink:ClinicalFinding",
    }

    # Entity types are based on Biolink categories
    entity_types = sorted(set(aliases.values()))

    return EntityTypesResponse(
        entity_types=entity_types,
        aliases=aliases,
    )


@router.get("/annotators", response_model=AnnotatorsResponse)
async def list_annotators(
    request: Request,
    _api_key: str = Depends(validate_api_key),
) -> AnnotatorsResponse:
    """
    List available annotators.

    Returns annotators that can be used to fetch additional identifiers for entities.
    """
    mapper = getattr(request.app.state, "mapper", None)

    if mapper is None:
        # Return static list if mapper not available
        return AnnotatorsResponse(
            annotators=[
                AnnotatorInfo(
                    slug="kestrel_hybrid",
                    name="Kestrel Hybrid Search",
                    description="Combined text+vector search via Kestrel API",
                ),
                AnnotatorInfo(
                    slug="kestrel_text", name="Kestrel Text Search", description="Text-based search via Kestrel API"
                ),
                AnnotatorInfo(
                    slug="kestrel_vector",
                    name="Kestrel Vector Search",
                    description="Embedding-based search via Kestrel API",
                ),
                AnnotatorInfo(
                    slug="metabolomics_workbench",
                    name="Metabolomics Workbench",
                    description="RefMet annotations from Metabolomics Workbench",
                ),
            ]
        )

    # Get annotators from the annotation engine
    annotators = []
    for slug, annotator in mapper.annotation_engine.annotator_registry.items():
        annotators.append(
            AnnotatorInfo(
                slug=slug,
                name=annotator.__class__.__name__,
                description=annotator.__doc__ or None,
            )
        )

    return AnnotatorsResponse(annotators=annotators)


@router.get("/vocabularies", response_model=VocabulariesResponse)
async def list_vocabularies(
    request: Request,
    _api_key: str = Depends(validate_api_key),
) -> VocabulariesResponse:
    """
    List supported vocabularies.

    Returns vocabulary prefixes that biomapper2 can normalize and link.
    """
    mapper = getattr(request.app.state, "mapper", None)

    if mapper is None:
        # Return minimal info if mapper not available
        from ..models import VocabularyInfo

        common_vocabs = [
            VocabularyInfo(prefix="CHEBI", iri="http://purl.obolibrary.org/obo/CHEBI_"),
            VocabularyInfo(prefix="PUBCHEM.COMPOUND", iri="https://pubchem.ncbi.nlm.nih.gov/compound/"),
            VocabularyInfo(prefix="KEGG.COMPOUND", iri="https://www.kegg.jp/entry/"),
            VocabularyInfo(prefix="HMDB", iri="https://hmdb.ca/metabolites/"),
            VocabularyInfo(prefix="UNIPROT", iri="https://www.uniprot.org/uniprot/"),
        ]
        return VocabulariesResponse(vocabularies=common_vocabs, count=len(common_vocabs))

    # Get vocabularies from the normalizer's vocab info
    from ...utils import ALIASES_PROP
    from ..models import VocabularyInfo

    vocab_info_map = mapper.normalizer.vocab_info_map
    vocab_validator_map = mapper.normalizer.vocab_validator_map

    vocabularies = []
    for key, info in vocab_info_map.items():
        prefix = info.get("prefix", key.upper())
        iri = info.get("iri")

        # Get aliases from validator map
        aliases = []
        if key in vocab_validator_map:
            aliases = vocab_validator_map[key].get(ALIASES_PROP, [])

        vocabularies.append(
            VocabularyInfo(
                prefix=prefix,
                iri=iri if iri else None,
                aliases=aliases,
            )
        )

    return VocabulariesResponse(
        vocabularies=sorted(vocabularies, key=lambda v: v.prefix),
        count=len(vocabularies),
    )
