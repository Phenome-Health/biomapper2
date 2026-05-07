"""Unit tests for biomapper2 REST API (mocked Mapper, no external deps)."""

from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomapper2.api.main import app

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    """Test client with Mapper available in app.state."""
    with TestClient(app) as c:
        yield c


def _app(client: TestClient) -> FastAPI:
    """Get the FastAPI app from a TestClient, typed for pyright."""
    assert isinstance(client.app, FastAPI)
    return client.app


@pytest.fixture
def mock_mapper():
    """A MagicMock that mimics Mapper.map_entity_to_kg return value."""
    mapper = MagicMock()
    mapper.map_entity_to_kg.return_value = {
        "name": "glucose",
        "curies": ["KEGG.COMPOUND:C00031"],
        "curies_provided": ["KEGG.COMPOUND:C00031"],
        "curies_assigned": {},
        "kg_ids": {"CHEBI:17234": ["KEGG.COMPOUND:C00031"]},
        "kg_ids_provided": {"CHEBI:17234": ["KEGG.COMPOUND:C00031"]},
        "kg_ids_assigned": {},
        "chosen_kg_id": "CHEBI:17234",
        "chosen_kg_id_provided": "CHEBI:17234",
        "chosen_kg_id_assigned": None,
        "assigned_ids": {},
        "invalid_ids_provided": {},
        "invalid_ids_assigned": {},
        "unrecognized_vocabs_provided": [],
        "unrecognized_vocabs_assigned": [],
        "kegg": "C00031",
    }
    return mapper


class TestMapperUnavailable:
    """Tests for when Mapper fails to initialize."""

    def test_health_returns_degraded_when_mapper_none(self, client: TestClient):
        """Health endpoint returns degraded status when mapper is None."""
        fa = _app(client)
        original_mapper = getattr(fa.state, "mapper", None)
        original_error = getattr(fa.state, "mapper_error", None)
        fa.state.mapper = None
        fa.state.mapper_error = "Init failed"
        try:
            response = client.get("/api/v1/health")
            assert response.status_code == 200
            data = response.json()
            assert data["mapper_initialized"] is False
        finally:
            fa.state.mapper = original_mapper
            if original_error is None and hasattr(fa.state, "mapper_error"):
                del fa.state.mapper_error
            else:
                fa.state.mapper_error = original_error

    def test_mapping_returns_503_when_mapper_none(self, client: TestClient):
        """Mapping endpoint returns 503 when mapper is not initialized."""
        fa = _app(client)
        original = getattr(fa.state, "mapper", None)
        fa.state.mapper = None
        try:
            response = client.post(
                "/api/v1/map/entity",
                json={
                    "name": "test",
                    "entity_type": "metabolite",
                },
            )
            assert response.status_code == 503
        finally:
            fa.state.mapper = original

    def test_batch_returns_503_when_mapper_none(self, client: TestClient):
        """Batch endpoint returns 503 when mapper is not initialized."""
        fa = _app(client)
        original = getattr(fa.state, "mapper", None)
        fa.state.mapper = None
        try:
            response = client.post(
                "/api/v1/map/batch",
                json={"entities": [{"name": "test", "entity_type": "metabolite"}]},
            )
            assert response.status_code == 503
        finally:
            fa.state.mapper = original


class TestEntityMappingMocked:
    """Entity mapping with mocked Mapper (no Kestrel dependency)."""

    def test_single_entity_mapping_response_structure(self, client: TestClient, mock_mapper: MagicMock):
        """Verify response structure when mapper returns Entity.to_dict() output."""
        fa = _app(client)
        original = fa.state.mapper
        fa.state.mapper = mock_mapper
        try:
            response = client.post(
                "/api/v1/map/entity",
                json={
                    "name": "glucose",
                    "entity_type": "metabolite",
                    "identifiers": {"kegg": "C00031"},
                },
            )
            assert response.status_code == 200
            data = response.json()

            # Verify API response envelope
            assert "result" in data
            assert "metadata" in data
            assert "request_id" in data["metadata"]
            assert isinstance(data["metadata"]["processing_time_ms"], float)

            # Verify result maps correctly from Entity.to_dict() output
            result = data["result"]
            assert result["name"] == "glucose"
            assert result["curies"] == ["KEGG.COMPOUND:C00031"]
            assert result["chosen_kg_id"] == "CHEBI:17234"
            assert "CHEBI:17234" in result["kg_ids"]
            assert result["error"] is None
        finally:
            fa.state.mapper = original

    def test_batch_mapping_with_mocked_mapper(self, client: TestClient, mock_mapper: MagicMock):
        """Batch mapping returns correct summary stats."""
        fa = _app(client)
        original = fa.state.mapper
        fa.state.mapper = mock_mapper
        try:
            response = client.post(
                "/api/v1/map/batch",
                json={
                    "entities": [
                        {"name": "glucose", "entity_type": "metabolite"},
                        {"name": "fructose", "entity_type": "metabolite"},
                    ]
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["results"]) == 2
            assert data["summary"]["total"] == 2
            assert data["summary"]["successful"] == 2
            assert data["summary"]["failed"] == 0
        finally:
            fa.state.mapper = original

    def test_mapper_exception_populates_error_field(self, client: TestClient):
        """When mapper raises, error field is populated (not a 500)."""
        error_mapper = MagicMock()
        error_mapper.map_entity_to_kg.side_effect = ValueError("Kestrel timeout")
        fa = _app(client)
        original = fa.state.mapper
        fa.state.mapper = error_mapper
        try:
            response = client.post(
                "/api/v1/map/entity",
                json={"name": "bad", "entity_type": "metabolite"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["result"]["error"] == "Kestrel timeout"
            assert data["result"]["name"] == "bad"
        finally:
            fa.state.mapper = original


class TestBatchValidation:
    """Batch endpoint validation edge cases."""

    def test_empty_batch_returns_422(self, client: TestClient):
        """Empty entities list is rejected (min_length=1)."""
        response = client.post("/api/v1/map/batch", json={"entities": []})
        assert response.status_code == 422

    def test_batch_over_limit_returns_422(self, client: TestClient):
        """More than 1000 entities is rejected (max_length=1000)."""
        entities = [{"name": f"entity_{i}", "entity_type": "metabolite"} for i in range(1001)]
        response = client.post("/api/v1/map/batch", json={"entities": entities})
        assert response.status_code == 422


class TestMultiKeyAuth:
    """Tests for BIOMAPPER2_API_KEYS (comma-separated multi-key)."""

    def test_multi_key_accepts_any_valid_key(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        """Any key in the comma-separated list is accepted."""
        monkeypatch.setenv("BIOMAPPER2_API_KEYS", "key-alpha,key-beta,key-gamma")
        monkeypatch.delenv("BIOMAPPER_API_KEY", raising=False)

        for key in ["key-alpha", "key-beta", "key-gamma"]:
            response = client.get("/api/v1/entity-types", headers={"X-API-Key": key})
            assert response.status_code == 200, f"Key {key} should be accepted"

    def test_multi_key_rejects_unknown_key(self, client: TestClient, monkeypatch: pytest.MonkeyPatch):
        """A key not in the list is rejected."""
        monkeypatch.setenv("BIOMAPPER2_API_KEYS", "key-alpha,key-beta")
        monkeypatch.delenv("BIOMAPPER_API_KEY", raising=False)

        response = client.get("/api/v1/entity-types", headers={"X-API-Key": "key-unknown"})
        assert response.status_code == 403


class TestEntityApiCompatibility:
    """Verify Entity.to_dict() output feeds correctly into API response construction."""

    def test_extract_mapping_result_from_entity_dict(self):
        """extract_mapping_result correctly reads Entity.to_dict() output."""
        from biomapper2.api.routes.mapping import extract_mapping_result
        from biomapper2.models import Entity

        # Simulate a fully-processed entity — use **kwargs for extra fields (pyright-safe)
        extra: dict[str, Any] = {"kegg_id": "C00791"}
        entity = Entity(
            name="creatinine",
            curies=["KEGG.COMPOUND:C00791", "HMDB:HMDB0000562"],
            curies_provided=["KEGG.COMPOUND:C00791"],
            curies_assigned={"hmdb_api": ["HMDB:HMDB0000562"]},
            kg_ids={"CHEBI:16737": ["KEGG.COMPOUND:C00791"]},
            kg_ids_provided={"CHEBI:16737": ["KEGG.COMPOUND:C00791"]},
            kg_ids_assigned={},
            chosen_kg_id="CHEBI:16737",
            assigned_ids={"hmdb_api": {"hmdb": {"HMDB0000562": {"score": 1.0}}}},
            **extra,
        )

        result = extract_mapping_result(entity.to_dict(), "creatinine")

        assert result.name == "creatinine"
        assert result.curies == ["KEGG.COMPOUND:C00791", "HMDB:HMDB0000562"]
        assert result.chosen_kg_id == "CHEBI:16737"
        assert "CHEBI:16737" in result.kg_ids
        assert result.error is None

    def test_extract_mapping_result_from_entity_series(self):
        """extract_mapping_result correctly reads Entity.to_series() output."""
        from biomapper2.api.routes.mapping import extract_mapping_result
        from biomapper2.models import Entity

        entity = Entity(name="glucose", curies=["KEGG.COMPOUND:C00031"], chosen_kg_id="CHEBI:17234")
        result = extract_mapping_result(entity.to_series(), "glucose")

        assert result.name == "glucose"
        assert result.curies == ["KEGG.COMPOUND:C00031"]
        assert result.chosen_kg_id == "CHEBI:17234"
