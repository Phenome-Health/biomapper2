"""Tests for the biomapper2 REST API."""

import pytest
from fastapi.testclient import TestClient

from biomapper2.api.main import app


@pytest.fixture(scope="module")
def client():
    """Create a test client for the API."""
    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    def test_health_check(self, client: TestClient):
        """Health endpoint returns status and version."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200

        data = response.json()
        assert "status" in data
        assert "version" in data
        assert "mapper_initialized" in data

    def test_health_no_auth_required(self, client: TestClient):
        """Health endpoint doesn't require authentication."""
        response = client.get("/api/v1/health")
        assert response.status_code == 200


class TestDiscoveryEndpoints:
    """Tests for discovery endpoints."""

    def test_list_entity_types(self, client: TestClient):
        """Entity types endpoint returns valid types and aliases."""
        response = client.get("/api/v1/entity-types")
        assert response.status_code == 200

        data = response.json()
        assert "entity_types" in data
        assert "aliases" in data
        assert len(data["entity_types"]) > 0
        assert "metabolite" in data["aliases"]

    def test_list_annotators(self, client: TestClient):
        """Annotators endpoint returns list of available annotators."""
        response = client.get("/api/v1/annotators")
        assert response.status_code == 200

        data = response.json()
        assert "annotators" in data
        assert len(data["annotators"]) > 0

        # Check annotator structure
        annotator = data["annotators"][0]
        assert "slug" in annotator
        assert "name" in annotator

    def test_list_vocabularies(self, client: TestClient):
        """Vocabularies endpoint returns list of supported vocabs."""
        response = client.get("/api/v1/vocabularies")
        assert response.status_code == 200

        data = response.json()
        assert "vocabularies" in data
        assert "count" in data
        assert data["count"] > 0


class TestMappingEndpoints:
    """Tests for mapping endpoints."""

    @pytest.mark.integration
    def test_map_single_entity(self, client: TestClient):
        """Map a single entity with known identifiers."""
        response = client.post(
            "/api/v1/map/entity",
            json={
                "name": "carnitine",
                "entity_type": "metabolite",
                "identifiers": {"kegg": "C00487"},
                "options": {"annotation_mode": "none"},
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "result" in data
        assert "metadata" in data
        assert data["result"]["name"] == "carnitine"
        assert "curies" in data["result"]
        assert "request_id" in data["metadata"]
        assert "processing_time_ms" in data["metadata"]

    @pytest.mark.integration
    def test_map_entity_without_identifiers(self, client: TestClient):
        """Map an entity by name only (triggers annotation)."""
        response = client.post(
            "/api/v1/map/entity",
            json={
                "name": "glucose",
                "entity_type": "metabolite",
                "identifiers": {},
                "options": {"annotation_mode": "all"},
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert data["result"]["name"] == "glucose"

    @pytest.mark.integration
    def test_map_batch_entities(self, client: TestClient):
        """Map multiple entities in a batch."""
        response = client.post(
            "/api/v1/map/batch",
            json={
                "entities": [
                    {
                        "name": "carnitine",
                        "entity_type": "metabolite",
                        "identifiers": {"kegg": "C00487"},
                        "options": {"annotation_mode": "none"},
                    },
                    {
                        "name": "glucose",
                        "entity_type": "metabolite",
                        "identifiers": {"pubchem": "5793"},
                        "options": {"annotation_mode": "none"},
                    },
                ]
            },
        )
        assert response.status_code == 200

        data = response.json()
        assert "results" in data
        assert "summary" in data
        assert len(data["results"]) == 2
        assert data["summary"]["total"] == 2

    def test_map_entity_invalid_request(self, client: TestClient):
        """Invalid request returns 422."""
        response = client.post(
            "/api/v1/map/entity",
            json={
                # Missing required fields
                "name": "test",
            },
        )
        assert response.status_code == 422


class TestAPIAuthentication:
    """Tests for API authentication."""

    def test_auth_not_required_when_no_key_set(self, client: TestClient, monkeypatch):
        """When BIOMAPPER_API_KEY is not set, auth is not required."""
        monkeypatch.delenv("BIOMAPPER_API_KEY", raising=False)

        response = client.get("/api/v1/entity-types")
        assert response.status_code == 200

    def test_auth_required_when_key_set(self, client: TestClient, monkeypatch):
        """When BIOMAPPER_API_KEY is set, requests without key fail."""
        monkeypatch.setenv("BIOMAPPER_API_KEY", "test-secret-key")

        # Request without API key should fail
        response = client.get("/api/v1/entity-types")
        assert response.status_code == 401

    def test_auth_succeeds_with_valid_key(self, client: TestClient, monkeypatch):
        """Request with valid API key succeeds."""
        monkeypatch.setenv("BIOMAPPER_API_KEY", "test-secret-key")

        response = client.get(
            "/api/v1/entity-types",
            headers={"X-API-Key": "test-secret-key"},
        )
        assert response.status_code == 200

    def test_auth_fails_with_invalid_key(self, client: TestClient, monkeypatch):
        """Request with invalid API key fails."""
        monkeypatch.setenv("BIOMAPPER_API_KEY", "test-secret-key")

        response = client.get(
            "/api/v1/entity-types",
            headers={"X-API-Key": "wrong-key"},
        )
        assert response.status_code == 403


class TestAPIMetadata:
    """Tests for API metadata and headers."""

    def test_response_includes_timing_header(self, client: TestClient):
        """Responses include X-Process-Time-Ms header."""
        response = client.get("/api/v1/health")
        assert "X-Process-Time-Ms" in response.headers

    def test_openapi_docs_available(self, client: TestClient):
        """OpenAPI docs are available."""
        response = client.get("/api/v1/openapi.json")
        assert response.status_code == 200
        assert "openapi" in response.json()

    def test_root_redirects_to_docs(self, client: TestClient):
        """Root path redirects to docs."""
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert "/api/v1/docs" in response.headers["location"]
