"""Regression tests for API response JSON schemas and conditional field omission.

Ensures ``EntityMappingResult`` and embedding response models preserve their full
serialization and OpenAPI property sets, ``kestrel_degraded`` is omitted when ``None``
without altering any other ``None``-valued field across routes, and no response model
serialization schema ever collapses to an empty-properties object.
"""

from __future__ import annotations

import inspect
import io
import json
import types
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import BaseModel

import biomapper2.api.models.responses as responses_mod
from biomapper2.api.main import app
from biomapper2.api.models.responses import (
    BatchMappingResponse,
    EntityMappingResponse,
    EntityMappingResult,
)

ORIGIN_DEV_ENTITY_MAPPING_RESULT_PROPERTIES = frozenset(
    {
        "name",
        "curies",
        "primary_curie",
        "curies_provided",
        "curies_assigned",
        "chosen_kg_id",
        "chosen_kg_id_review",
        "resolution_certificate",
        "chosen_kg_id_lipid_hint",
        "lipid_resolution",
        "refmet_availability",
        "refmet_source",
        "refmet_snapshot_version",
        "tier_b_snapshot_version",
        "mapping_relation",
        "mapping_predicate",
        "icd10cm_route",
        "kg_build",
        "icd10cm_snapshot_version",
        "kg_equivalent_ids",
        "kg_ids",
        "assigned_ids",
        "kestrel_results",
        "error",
    }
)

EXPECTED_ENTITY_MAPPING_RESULT_PROPERTIES = ORIGIN_DEV_ENTITY_MAPPING_RESULT_PROPERTIES | {"kestrel_degraded"}

NULLABLE_ORIGIN_DEV_FIELDS = frozenset(
    {
        "primary_curie",
        "chosen_kg_id",
        "chosen_kg_id_review",
        "resolution_certificate",
        "chosen_kg_id_lipid_hint",
        "lipid_resolution",
        "refmet_snapshot_version",
        "tier_b_snapshot_version",
        "mapping_relation",
        "mapping_predicate",
        "icd10cm_route",
        "kg_build",
        "icd10cm_snapshot_version",
        "kestrel_results",
        "error",
    }
)


def _fresh_openapi() -> dict[str, Any]:
    app.openapi_schema = None
    return app.openapi()


@pytest.mark.unit
def test_entity_mapping_result_and_embedding_schemas_preserve_property_sets() -> None:
    """EntityMappingResult, embedding models, and OpenAPI list all origin/dev fields plus optional kestrel_degraded."""
    ser_schema = EntityMappingResult.model_json_schema(mode="serialization")
    assert set(ser_schema["properties"]) == EXPECTED_ENTITY_MAPPING_RESULT_PROPERTIES
    assert "kestrel_degraded" not in ser_schema.get("required", [])
    kd_ser = ser_schema["properties"]["kestrel_degraded"]
    assert {"type": "boolean", "const": True} in kd_ser["anyOf"]
    assert {"type": "null"} in kd_ser["anyOf"]

    # Embedding model 1: EntityMappingResponse
    entity_resp_ser = EntityMappingResponse.model_json_schema(mode="serialization")
    assert set(entity_resp_ser["properties"]) == {"result", "metadata"}
    assert set(entity_resp_ser["$defs"]["EntityMappingResult"]["properties"]) == (
        EXPECTED_ENTITY_MAPPING_RESULT_PROPERTIES
    )

    # Embedding model 2: BatchMappingResponse
    batch_resp_ser = BatchMappingResponse.model_json_schema(mode="serialization")
    assert set(batch_resp_ser["properties"]) == {"results", "metadata", "summary"}
    assert set(batch_resp_ser["$defs"]["EntityMappingResult"]["properties"]) == (
        EXPECTED_ENTITY_MAPPING_RESULT_PROPERTIES
    )

    # Published OpenAPI schemas
    oa_schemas = _fresh_openapi()["components"]["schemas"]
    oa_emr = oa_schemas["EntityMappingResult"]
    assert set(oa_emr["properties"]) == EXPECTED_ENTITY_MAPPING_RESULT_PROPERTIES
    assert "kestrel_degraded" not in oa_emr.get("required", [])
    assert {"type": "boolean", "const": True} in oa_emr["properties"]["kestrel_degraded"]["anyOf"]
    assert {"type": "null"} in oa_emr["properties"]["kestrel_degraded"]["anyOf"]

    assert set(oa_schemas["EntityMappingResponse"]["properties"]) == {"result", "metadata"}
    assert oa_schemas["EntityMappingResponse"]["properties"]["result"]["$ref"] == (
        "#/components/schemas/EntityMappingResult"
    )
    assert set(oa_schemas["BatchMappingResponse"]["properties"]) == {"results", "metadata", "summary"}
    assert oa_schemas["BatchMappingResponse"]["properties"]["results"]["items"]["$ref"] == (
        "#/components/schemas/EntityMappingResult"
    )


@pytest.mark.unit
def test_routes_omit_kestrel_degraded_when_none_and_keep_all_other_null_fields() -> None:
    """Routes /map/entity, /map/batch, and /map/dataset/stream omit kestrel_degraded=None while keeping other nulls."""

    class _StubMapper:
        def __init__(self, degraded: bool) -> None:
            self.degraded = degraded
            self.annotation_engine = types.SimpleNamespace(annotator_registry={})

        def map_entity_to_kg(self, item: dict[str, Any], **_: Any) -> dict[str, Any]:
            out: dict[str, Any] = {"name": item["name"]}
            if self.degraded:
                out["chosen_kg_id"] = "CHEBI:17234"
                out["kestrel_degraded"] = True
            return out

        def map_entities_to_kg(self, items: list[dict[str, Any]], **_: Any) -> pd.DataFrame:
            return pd.DataFrame([self.map_entity_to_kg(it) for it in items])

    prev_mapper = getattr(app.state, "mapper", None)
    client = TestClient(app, raise_server_exceptions=True)
    tsv_bytes = b"name\tkegg\nglucose\t\n"
    stream_params = {"entity_type": "metabolite", "name_column": "name", "provided_id_columns": "kegg"}

    try:
        # 1. Healthy (kestrel_degraded None/absent): omitted everywhere, every other None field preserved
        app.state.mapper = _StubMapper(degraded=False)

        r_ent = client.post("/api/v1/map/entity", json={"name": "glucose", "entity_type": "metabolite"})
        assert r_ent.status_code == 200
        ent_result = r_ent.json()["result"]
        assert "kestrel_degraded" not in ent_result
        assert "kestrel_degraded" not in r_ent.text
        assert set(ent_result) == ORIGIN_DEV_ENTITY_MAPPING_RESULT_PROPERTIES
        for field_name in NULLABLE_ORIGIN_DEV_FIELDS:
            assert field_name in ent_result and ent_result[field_name] is None

        r_batch = client.post(
            "/api/v1/map/batch",
            json={"entities": [{"name": "glucose", "entity_type": "metabolite"}]},
        )
        assert r_batch.status_code == 200
        batch_item = r_batch.json()["results"][0]
        assert "kestrel_degraded" not in batch_item
        assert "kestrel_degraded" not in r_batch.text
        assert set(batch_item) == ORIGIN_DEV_ENTITY_MAPPING_RESULT_PROPERTIES
        for field_name in NULLABLE_ORIGIN_DEV_FIELDS:
            assert field_name in batch_item and batch_item[field_name] is None

        r_stream = client.post(
            "/api/v1/map/dataset/stream",
            params=stream_params,
            files={"file": ("healthy.tsv", io.BytesIO(tsv_bytes), "text/tab-separated-values")},
        )
        assert r_stream.status_code == 200
        stream_row = json.loads(r_stream.text.strip())
        assert "kestrel_degraded" not in stream_row
        assert "kestrel_degraded" not in r_stream.text
        for stream_null_field in (
            "primary_curie",
            "chosen_kg_id",
            "chosen_kg_id_review",
            "chosen_kg_id_lipid_hint",
            "lipid_resolution",
            "mapping_relation",
            "mapping_predicate",
            "icd10cm_route",
            "kg_build",
            "icd10cm_snapshot_version",
            "resolution_certificate",
        ):
            assert stream_null_field in stream_row and stream_row[stream_null_field] is None

        # 2. Degraded (kestrel_degraded=True): emitted across all three routes without dropping other nulls
        app.state.mapper = _StubMapper(degraded=True)

        r_ent_deg = client.post("/api/v1/map/entity", json={"name": "glucose", "entity_type": "metabolite"})
        ent_deg = r_ent_deg.json()["result"]
        assert ent_deg["kestrel_degraded"] is True
        assert set(ent_deg) == EXPECTED_ENTITY_MAPPING_RESULT_PROPERTIES
        assert ent_deg["error"] is None

        r_batch_deg = client.post(
            "/api/v1/map/batch",
            json={"entities": [{"name": "glucose", "entity_type": "metabolite"}]},
        )
        assert r_batch_deg.json()["results"][0]["kestrel_degraded"] is True

        r_stream_deg = client.post(
            "/api/v1/map/dataset/stream",
            params=stream_params,
            files={"file": ("deg.tsv", io.BytesIO(tsv_bytes), "text/tab-separated-values")},
        )
        assert json.loads(r_stream_deg.text.strip())["kestrel_degraded"] is True
    finally:
        app.state.mapper = prev_mapper


@pytest.mark.unit
def test_no_response_model_serialization_schema_collapses() -> None:
    """Every response BaseModel and OpenAPI object schema exposes non-empty field properties."""
    response_models = [
        cls
        for _, cls in inspect.getmembers(responses_mod, inspect.isclass)
        if issubclass(cls, BaseModel) and cls is not BaseModel and cls.__module__ == responses_mod.__name__
    ]
    assert len(response_models) > 0

    for model_cls in response_models:
        ser = model_cls.model_json_schema(mode="serialization")
        props = ser.get("properties")
        expected_fields = {
            (field_info.serialization_alias or field_name)
            for field_name, field_info in model_cls.model_fields.items()
            if not field_info.exclude
        }
        assert (
            isinstance(props, dict) and len(props) > 0
        ), f"{model_cls.__name__} serialization schema collapsed to empty properties: {ser}"
        assert (
            set(props) == expected_fields
        ), f"{model_cls.__name__} serialization properties {set(props)} != {expected_fields}"
        for def_name, def_schema in ser.get("$defs", {}).items():
            if def_schema.get("type") == "object":
                def_props = def_schema.get("properties")
                assert (
                    isinstance(def_props, dict) and len(def_props) > 0
                ), f"{model_cls.__name__}.$defs.{def_name} collapsed to empty properties: {def_schema}"

    oa_schemas = _fresh_openapi()["components"]["schemas"]
    for schema_name, schema_def in oa_schemas.items():
        if schema_def.get("type") == "object":
            oa_props = schema_def.get("properties")
            assert (
                isinstance(oa_props, dict) and len(oa_props) > 0
            ), f"OpenAPI components.schemas.{schema_name} collapsed to empty properties: {schema_def}"


@pytest.mark.unit
def test_pyproject_requires_pydantic_with_exclude_if() -> None:
    """pyproject.toml enforces pydantic >= 2.12.0 so Field(exclude_if=...) is never silently ignored."""
    import ast
    from pathlib import Path

    pyproject_text = (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text(encoding="utf-8")
    _, after = pyproject_text.split("\ndependencies = ", 1)
    raw_block = after[: after.index("\n]") + 2]
    cleaned_lines = [line.split("#", 1)[0] for line in raw_block.splitlines()]
    deps: list[str] = ast.literal_eval("\n".join(cleaned_lines))
    pydantic_specs = [d for d in deps if d.startswith("pydantic")]
    assert pydantic_specs == ["pydantic>=2.12.0"]


@pytest.mark.unit
def test_entity_mapping_result_kestrel_degraded_literal_true_type() -> None:
    """EntityMappingResult only allows Literal true or None for kestrel_degraded."""
    import pydantic

    valid_true = EntityMappingResult(name="glucose", kestrel_degraded=True)
    assert valid_true.kestrel_degraded is True
    valid_none = EntityMappingResult(name="glucose", kestrel_degraded=None)
    assert valid_none.kestrel_degraded is None

    with pytest.raises(pydantic.ValidationError):
        EntityMappingResult(name="glucose", kestrel_degraded=False)  # type: ignore[arg-type]
