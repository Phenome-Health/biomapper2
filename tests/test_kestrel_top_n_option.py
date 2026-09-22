"""Unit tests for the opt-in ``kestrel_top_n`` request knob (R1).

``kestrel_top_n`` turns on the raw-Kestrel passthrough side channel: it controls how many raw rows
each used search endpoint returns, and it CANNOT change ``chosen_kg_id`` (that is ``candidate_limit``).
These tests exercise the model bound and the OpenAPI surface only — no Kestrel call is ever made.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from biomapper2.api.main import app
from biomapper2.api.models.requests import MappingOptions

pytestmark = pytest.mark.unit


# --------------------------------------- Model validation ----------------------------------------- #


def test_mapping_options_kestrel_top_n_defaults_none():
    """The knob is opt-in: default is None (feature disabled)."""
    assert MappingOptions().kestrel_top_n is None


@pytest.mark.parametrize("good", [1, 20, 100])
def test_mapping_options_accepts_in_range(good):
    """In-range values (1..100) are accepted and stored verbatim."""
    assert MappingOptions(kestrel_top_n=good).kestrel_top_n == good


@pytest.mark.parametrize("bad", [0, 101, -1])
def test_mapping_options_rejects_out_of_range(bad):
    """Values outside the bound raise a pydantic ValidationError (the HTTP validation boundary) (T1)."""
    with pytest.raises(ValidationError):
        MappingOptions(kestrel_top_n=bad)


def test_mapping_options_rejects_non_integer():
    """A non-integer value is rejected (T1)."""
    with pytest.raises(ValidationError):
        MappingOptions(kestrel_top_n="lots")  # type: ignore[arg-type]


def test_mapping_options_ignores_unknown_extra_option():
    """A newer client sending an unknown option still parses (extra='ignore') (T13)."""
    opts = MappingOptions(kestrel_top_n=10, some_future_option="x")  # type: ignore[call-arg]
    assert opts.kestrel_top_n == 10


# ------------------------------------------ OpenAPI surface ---------------------------------------- #


def test_openapi_exposes_kestrel_top_n_with_bound():
    """OpenAPI schema advertises kestrel_top_n on MappingOptions with the 1..100 bound (T1)."""
    with TestClient(app) as client:
        schema = client.get("/api/v1/openapi.json").json()
    options_schema = schema["components"]["schemas"]["MappingOptions"]["properties"]
    assert "kestrel_top_n" in options_schema
    field = options_schema["kestrel_top_n"]
    # int | None → anyOf with an integer branch carrying the bounds
    branches = field.get("anyOf", [field])
    int_branch = next(b for b in branches if b.get("type") == "integer")
    assert int_branch["minimum"] == 1
    assert int_branch["maximum"] == 100


def test_out_of_range_kestrel_top_n_returns_422_via_testclient():
    """POST /map/entity with an out-of-range kestrel_top_n is rejected with a validation error (T1)."""
    with TestClient(app) as client:
        response = client.post(
            "/api/v1/map/entity",
            json={"name": "glucose", "entity_type": "metabolite", "options": {"kestrel_top_n": 0}},
        )
    assert response.status_code == 422
