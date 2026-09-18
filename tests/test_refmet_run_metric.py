"""Run-level RefMet availability metric on the batch path (U5 / D5).

A batch with K rows the RefMet service could not answer reports K in its summary, alongside the
total. Cold-run-attributable only: the RefMet HTTP cache serves successes, so a warm rerun would
understate this. Mocked mapper — no external calls.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomapper2.api.main import app

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _app(client: TestClient) -> FastAPI:
    assert isinstance(client.app, FastAPI)
    return client.app


def test_batch_summary_counts_refmet_unavailable_rows(client: TestClient):
    statuses = ["unavailable", "voted", "unavailable", "no_match"]
    mapper = MagicMock()
    mapper.map_entity_to_kg.side_effect = [
        {"name": f"m{i}", "chosen_kg_id": "CHEBI:1", "refmet_availability": status} for i, status in enumerate(statuses)
    ]

    fa = _app(client)
    original = fa.state.mapper
    fa.state.mapper = mapper
    try:
        response = client.post(
            "/api/v1/map/batch",
            json={"entities": [{"name": f"m{i}", "entity_type": "metabolite"} for i in range(len(statuses))]},
        )
        assert response.status_code == 200
        summary = response.json()["summary"]
        assert summary["total"] == len(statuses)
        assert summary["refmet_unavailable"] == 2  # two 'unavailable' rows; voted/no_match excluded
    finally:
        fa.state.mapper = original
