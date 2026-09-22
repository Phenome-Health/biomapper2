"""Live-Kestrel integration tests for the raw passthrough side channel (T14, T15).

These hit the REAL (keyless public) Kestrel and are marked ``integration`` so the unit CI
(``-m "not integration"``) skips them. Run explicitly with ``pytest -m integration``.

T14 — a caller setting kestrel_top_n=10 gets <=10 raw rows per used endpoint whose IDs match a direct
Kestrel call with the same parameters.
T15 — selection invariance against the real backend, PLUS a timed multi-entity /batch with a realistic
armed RefMet BATCH_DEADLINE_S and induced passthrough latency, asserting late-batch
refmet_availability/certificate are unchanged across kestrel_top_n (the case the zero-latency mocked
T6 cannot reach — see the Unit 5 note).
"""

from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient

from biomapper2.api.main import app

pytestmark = pytest.mark.integration

_NAMES = ["glucose", "creatinine", "carnitine"]


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_passthrough_ids_match_direct_kestrel_call(client):
    """T14: passthrough rows for kestrel_top_n=10 match a direct hybrid-search call, same params."""
    from biomapper2.core.annotators.kestrel_hybrid import KestrelHybridSearchAnnotator

    for name in _NAMES:
        resp = client.post(
            "/api/v1/map/entity",
            json={"name": name, "entity_type": "metabolite", "options": {"kestrel_top_n": 10}},
        )
        assert resp.status_code == 200
        ksr = resp.json()["result"]["kestrel_results"]
        # There should be a hybrid-search entry (the default small-molecule path uses hybrid).
        hybrid = next((k for k in ksr if k["endpoint"] == "hybrid-search"), None)
        assert hybrid is not None
        assert len(hybrid["rows"]) <= 10

        # A direct call with the SAME params should return the same top-N ids (raw, unfiltered).
        direct = KestrelHybridSearchAnnotator._kestrel_hybrid_search(name, "biolink:SmallMolecule", None, limit=10)
        # _kestrel_hybrid_search applies the >=0.5 filter, so compare only the >=0.5 passthrough rows.
        passthrough_ids = [r["id"] for r in hybrid["rows"] if r.get("score", 0) >= 0.5]
        direct_ids = [r["id"] for r in direct.get(name, [])]
        assert passthrough_ids == direct_ids[: len(passthrough_ids)]


@pytest.mark.parametrize("kestrel_top_n", [None, 1, 20, 100])
def test_selection_invariance_against_real_kestrel(client, kestrel_top_n):
    """T15 (part 1): chosen_kg_id is unchanged across kestrel_top_n against the real backend."""
    baseline = client.post("/api/v1/map/entity", json={"name": "glucose", "entity_type": "metabolite"}).json()[
        "result"
    ]["chosen_kg_id"]

    opts = {} if kestrel_top_n is None else {"kestrel_top_n": kestrel_top_n}
    result = client.post(
        "/api/v1/map/entity", json={"name": "glucose", "entity_type": "metabolite", "options": opts}
    ).json()["result"]
    assert result["chosen_kg_id"] == baseline


def test_timed_batch_deadline_invariance(client, monkeypatch):
    """T15 (part 2): a timed /batch with an armed RefMet deadline + induced passthrough latency keeps
    late-batch refmet_availability/certificate independent of kestrel_top_n (the deadline case)."""
    from biomapper2.core import kestrel_passthrough

    # Induce passthrough latency so, if collection ran INSIDE the armed window, it would tip late-batch
    # entities past the deadline. It runs AFTER disarm, so it must not.
    real_collect = kestrel_passthrough.collect

    def slow_collect(*args, **kwargs):
        time.sleep(0.2)
        return real_collect(*args, **kwargs)

    monkeypatch.setattr("biomapper2.api.routes.mapping.collect_kestrel_passthrough", slow_collect)

    def _run(top_n):
        options: dict = {} if top_n is None else {"kestrel_top_n": top_n}
        entities = [{"name": n, "entity_type": "metabolite", "options": options} for n in _NAMES * 4]
        resp = client.post("/api/v1/map/batch", json={"entities": entities})
        assert resp.status_code == 200
        return resp.json()["results"]

    baseline = _run(None)
    withn = _run(20)
    for b, w in zip(baseline, withn):
        assert b["refmet_availability"] == w["refmet_availability"]
        assert b["resolution_certificate"] == w["resolution_certificate"]
