"""Entity mapping endpoints."""

import io
import logging
import tempfile
import time
import uuid
from typing import Any

import pandas as pd
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from fastapi.responses import StreamingResponse

from ..auth import validate_api_key
from ..models import (
    BatchMappingRequest,
    BatchMappingResponse,
    DatasetMappingResponse,
    EntityMappingRequest,
    EntityMappingResponse,
    EntityMappingResult,
    RequestMetadata,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def get_mapper(request: Request):
    """Get the Mapper instance from app state."""
    mapper = getattr(request.app.state, "mapper", None)
    if mapper is None:
        raise HTTPException(
            status_code=503,
            detail="Mapper not initialized. Service is starting up or encountered an error.",
        )
    return mapper


def extract_mapping_result(mapped_item: dict[str, Any] | pd.Series, original_name: str) -> EntityMappingResult:
    """Extract mapping result from mapped item."""
    if isinstance(mapped_item, pd.Series):
        mapped_item = mapped_item.to_dict()

    return EntityMappingResult(
        name=original_name,
        curies=mapped_item.get("curies", []) or [],
        chosen_kg_id=mapped_item.get("chosen_kg_id"),
        kg_ids=mapped_item.get("kg_ids", {}) or {},
        assigned_ids=mapped_item.get("assigned_ids", {}) or {},
        error=None,
    )


@router.post("/entity", response_model=EntityMappingResponse)
async def map_entity(
    request: Request,
    body: EntityMappingRequest,
    _api_key: str = Depends(validate_api_key),
) -> EntityMappingResponse:
    """
    Map a single entity to knowledge graph nodes.

    Takes an entity name, type, and optional identifiers, then runs the full
    biomapper2 pipeline: annotation → normalization → linking → resolution.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    mapper = get_mapper(request)

    try:
        # Build entity dict from request
        entity: dict[str, Any] = {"name": body.name}

        # Add identifiers as separate fields
        provided_id_fields = []
        for vocab, ids in body.identifiers.items():
            field_name = vocab.lower()
            # Handle list of IDs by joining
            if isinstance(ids, list):
                entity[field_name] = ",".join(str(i) for i in ids)
            else:
                entity[field_name] = str(ids)
            provided_id_fields.append(field_name)

        # Run mapping
        mapped_item = mapper.map_entity_to_kg(
            item=entity,
            name_field="name",
            provided_id_fields=provided_id_fields,
            entity_type=body.entity_type,
            vocab=body.options.vocab,
            array_delimiters=body.options.array_delimiters,
            annotation_mode=body.options.annotation_mode,
            annotators=body.options.annotators,
        )

        result = extract_mapping_result(mapped_item, body.name)

    except Exception as e:
        logger.exception(f"Error mapping entity: {e}")
        result = EntityMappingResult(
            name=body.name,
            error=str(e),
        )

    processing_time = (time.time() - start_time) * 1000

    return EntityMappingResponse(
        result=result,
        metadata=RequestMetadata(
            request_id=request_id,
            processing_time_ms=round(processing_time, 2),
        ),
    )


@router.post("/batch", response_model=BatchMappingResponse)
async def map_batch(
    request: Request,
    body: BatchMappingRequest,
    _api_key: str = Depends(validate_api_key),
) -> BatchMappingResponse:
    """
    Map multiple entities in a single request.

    Processes each entity through the mapping pipeline and returns results.
    Maximum 1000 entities per request.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    mapper = get_mapper(request)

    results: list[EntityMappingResult] = []
    successful = 0
    failed = 0

    for entity_req in body.entities:
        try:
            # Build entity dict
            entity: dict[str, Any] = {"name": entity_req.name}
            provided_id_fields = []

            for vocab, ids in entity_req.identifiers.items():
                field_name = vocab.lower()
                if isinstance(ids, list):
                    entity[field_name] = ",".join(str(i) for i in ids)
                else:
                    entity[field_name] = str(ids)
                provided_id_fields.append(field_name)

            # Run mapping
            mapped_item = mapper.map_entity_to_kg(
                item=entity,
                name_field="name",
                provided_id_fields=provided_id_fields,
                entity_type=entity_req.entity_type,
                vocab=entity_req.options.vocab,
                array_delimiters=entity_req.options.array_delimiters,
                annotation_mode=entity_req.options.annotation_mode,
                annotators=entity_req.options.annotators,
            )

            result = extract_mapping_result(mapped_item, entity_req.name)
            results.append(result)
            successful += 1

        except Exception as e:
            logger.exception(f"Error mapping entity '{entity_req.name}': {e}")
            results.append(
                EntityMappingResult(
                    name=entity_req.name,
                    error=str(e),
                )
            )
            failed += 1

    processing_time = (time.time() - start_time) * 1000

    return BatchMappingResponse(
        results=results,
        metadata=RequestMetadata(
            request_id=request_id,
            processing_time_ms=round(processing_time, 2),
        ),
        summary={
            "total": len(body.entities),
            "successful": successful,
            "failed": failed,
        },
    )


@router.post("/dataset", response_model=DatasetMappingResponse)
async def map_dataset(
    request: Request,
    entity_type: str,
    name_column: str,
    provided_id_columns: str,  # Comma-separated
    file: UploadFile = File(...),
    annotation_mode: str = "missing",
    annotators: str | None = None,  # Comma-separated
    vocab: str | None = None,
    _api_key: str = Depends(validate_api_key),
) -> DatasetMappingResponse:
    """
    Map a dataset from an uploaded TSV/CSV file.

    Upload a file and specify column mappings. Returns statistics about the mapping.
    """
    start_time = time.time()
    request_id = str(uuid.uuid4())

    mapper = get_mapper(request)

    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename = file.filename.lower()
    if not (filename.endswith(".tsv") or filename.endswith(".csv")):
        raise HTTPException(status_code=400, detail="File must be .tsv or .csv")

    try:
        # Read file content
        content = await file.read()

        # Create temp file to pass to mapper
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=f".{filename.split('.')[-1]}",
            delete=False,
        ) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        # Parse columns
        id_columns = [col.strip() for col in provided_id_columns.split(",")]
        annotator_list = [a.strip() for a in annotators.split(",")] if annotators else None

        # Run dataset mapping
        output_path, stats = mapper.map_dataset_to_kg(
            dataset=tmp_path,
            entity_type=entity_type,
            name_column=name_column,
            provided_id_columns=id_columns,
            vocab=vocab,
            annotation_mode=annotation_mode,  # type: ignore
            annotators=annotator_list,
        )

    except Exception as e:
        logger.exception(f"Error mapping dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    processing_time = (time.time() - start_time) * 1000

    return DatasetMappingResponse(
        output_file=output_path,
        stats=stats,
        metadata=RequestMetadata(
            request_id=request_id,
            processing_time_ms=round(processing_time, 2),
        ),
    )


@router.post("/dataset/stream")
async def map_dataset_stream(
    request: Request,
    entity_type: str,
    name_column: str,
    provided_id_columns: str,
    file: UploadFile = File(...),
    annotation_mode: str = "missing",
    annotators: str | None = None,
    vocab: str | None = None,
    _api_key: str = Depends(validate_api_key),
) -> StreamingResponse:
    """
    Map a dataset and stream results as NDJSON.

    Useful for large datasets - streams results row by row instead of waiting
    for full processing.
    """
    mapper = get_mapper(request)

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    filename = file.filename.lower()
    if not (filename.endswith(".tsv") or filename.endswith(".csv")):
        raise HTTPException(status_code=400, detail="File must be .tsv or .csv")

    # Read file content
    content = await file.read()

    # Parse file based on extension
    if filename.endswith(".tsv"):
        df = pd.read_csv(io.BytesIO(content), sep="\t")
    else:
        df = pd.read_csv(io.BytesIO(content))

    id_columns = [col.strip() for col in provided_id_columns.split(",")]
    annotator_list = [a.strip() for a in annotators.split(",")] if annotators else None

    async def generate_ndjson():
        """Generator that yields NDJSON lines."""
        import json

        for idx, row in df.iterrows():
            try:
                entity = row.to_dict()
                mapped = mapper.map_entity_to_kg(
                    item=entity,
                    name_field=name_column,
                    provided_id_fields=id_columns,
                    entity_type=entity_type,
                    vocab=vocab,
                    annotation_mode=annotation_mode,  # type: ignore
                    annotators=annotator_list,
                )

                result = {
                    "row_index": idx,
                    "name": entity.get(name_column, ""),
                    "chosen_kg_id": mapped.get("chosen_kg_id"),
                    "curies": mapped.get("curies", []),
                    "kg_ids": mapped.get("kg_ids", {}),
                }

            except Exception as e:
                result = {
                    "row_index": idx,
                    "name": row.get(name_column, ""),
                    "error": str(e),
                }

            yield json.dumps(result) + "\n"

    return StreamingResponse(
        generate_ndjson(),
        media_type="application/x-ndjson",
        headers={"X-Content-Type-Options": "nosniff"},
    )
