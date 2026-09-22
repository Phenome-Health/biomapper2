"""End-to-end (Kestrel MOCKED) tests for threading kestrel_top_n through the pipeline and routes.

Covers R4 (selection invariance), R5 (single/bulk parity + per-entity isolation), R6 (no-call → []),
R7 (passthrough failure isolation), R8 (route coverage incl. the /map/dataset rejection), and the
/batch post-window collection ordering. Every Kestrel call — selection AND passthrough — is patched at
the module boundary; no live/paid call is made.
"""

from __future__ import annotations

import io

import pytest
import requests
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomapper2.api.main import app
from biomapper2.api.routes import mapping as mapping_route
from biomapper2.core import kestrel_passthrough
from biomapper2.core import linker as linker_module
from biomapper2.core.annotators import kestrel_hybrid, kestrel_text, kestrel_vector

pytestmark = pytest.mark.unit

_ONCAT = {
    "id": "CHEBI:17234",
    "score": 3.2,
    "name": "glucose",
    "prefixes": ["CHEBI"],
    "categories": ["biolink:SmallMolecule"],
}
_SUBTHRESHOLD = {
    "id": "CHEBI:9999",
    "score": 0.3,
    "name": "glucose-ish",
    "prefixes": ["CHEBI"],
    "categories": ["biolink:SmallMolecule"],
}


def _make_fake_kestrel(calls: list[dict]):
    """A fake kestrel_request that records every call and returns canned per-endpoint data."""

    def fake(*, method, endpoint, batch_field, batch_items, batch_size, **kwargs):  # noqa: ANN001
        calls.append({"endpoint": endpoint, "json": kwargs.get("json"), "batch_items": list(batch_items)})
        if endpoint in ("hybrid-search", "text-search", "vector-search"):
            return {t: [dict(_ONCAT), dict(_SUBTHRESHOLD)] for t in batch_items}
        if endpoint == "canonicalize":
            return {c: c for c in batch_items}
        if endpoint == "get-nodes":
            return {c: {"name": "glucose", "equivalent_ids": [c]} for c in batch_items}
        return {t: [] for t in batch_items}

    return fake


@pytest.fixture
def mock_kestrel(monkeypatch):
    """Patch kestrel_request in every module that calls it (selection, linking, passthrough)."""
    calls: list[dict] = []
    fake = _make_fake_kestrel(calls)
    for module in (kestrel_hybrid, kestrel_text, kestrel_vector, linker_module, kestrel_passthrough):
        monkeypatch.setattr(module, "kestrel_request", fake)
    return calls


@pytest.fixture
def client(mock_kestrel):
    with TestClient(app) as c:
        yield c


def _app(client: TestClient) -> FastAPI:
    assert isinstance(client.app, FastAPI)
    return client.app


def _entity_body(**opts):
    body = {"name": "glucose", "entity_type": "metabolite", "options": {"annotators": ["kestrel-hybrid-search"]}}
    body["options"].update(opts)
    return body


# ------------------------------------ R4: selection invariance (T6) ------------------------------- #


@pytest.mark.parametrize("kestrel_top_n", [None, 1, 20, 100])
@pytest.mark.parametrize("candidate_limit", [None, 1, 50])
def test_selection_invariant_across_kestrel_top_n(client, kestrel_top_n, candidate_limit):
    """chosen_kg_id / assigned_ids / resolution_certificate are identical across the matrix (T6, R4)."""
    opts = {}
    if kestrel_top_n is not None:
        opts["kestrel_top_n"] = kestrel_top_n
    if candidate_limit is not None:
        opts["candidate_limit"] = candidate_limit
    resp = client.post("/api/v1/map/entity", json=_entity_body(**opts))
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["chosen_kg_id"] == "CHEBI:17234"
    assert result["assigned_ids"] == {
        "kestrel-hybrid-search": {"CHEBI": {"17234": {"score": 3.2, "resolved_via": "canonical_preference"}}}
    }
    # Certificate present and independent of kestrel_top_n (compare against the None-baseline below).
    assert "resolution_certificate" in result


def test_certificate_byte_identical_to_baseline(client):
    """The full certificate is byte-identical with and without kestrel_top_n (R4)."""
    base = client.post("/api/v1/map/entity", json=_entity_body()).json()["result"]
    withn = client.post("/api/v1/map/entity", json=_entity_body(kestrel_top_n=50)).json()["result"]
    assert base["resolution_certificate"] == withn["resolution_certificate"]
    assert base["chosen_kg_id"] == withn["chosen_kg_id"]
    # Baseline (no option) omits the passthrough field entirely (R2).
    assert base["kestrel_results"] is None


# --------------------------- separate_call proof + raw fidelity (T8, R3) --------------------------- #


def test_passthrough_is_a_separate_limit_n_call(client, mock_kestrel):
    """A distinct passthrough hybrid-search call with limit=N is issued; rows are raw (incl. <0.5) (T8, R3)."""
    resp = client.post("/api/v1/map/entity", json=_entity_body(kestrel_top_n=20))
    assert resp.status_code == 200
    ksr = resp.json()["result"]["kestrel_results"]
    assert len(ksr) == 1
    assert ksr[0]["endpoint"] == "hybrid-search"
    assert ksr[0]["request"]["limit"] == 20
    assert ksr[0]["fetch_strategy"] == "separate_call"
    # Raw fidelity: the sub-threshold row that selection's >=0.5 filter drops is present in passthrough.
    scores = [r["score"] for r in ksr[0]["rows"]]
    assert 0.3 in scores
    # A passthrough hybrid-search call carrying limit=20 was actually made.
    assert any(c["endpoint"] == "hybrid-search" and (c["json"] or {}).get("limit") == 20 for c in mock_kestrel)


def test_selection_call_args_unchanged_by_option(client, mock_kestrel):
    """The selection hybrid-search call args are byte-identical with vs without the option (T8)."""
    client.post("/api/v1/map/entity", json=_entity_body())
    baseline_selection = [c for c in mock_kestrel if c["endpoint"] == "hybrid-search"]
    mock_kestrel.clear()
    client.post("/api/v1/map/entity", json=_entity_body(kestrel_top_n=20))
    with_option_selection = [c for c in mock_kestrel if c["endpoint"] == "hybrid-search"]
    # The default hybrid limit (20) selection call is present and identical in both runs.
    assert baseline_selection[0]["json"] == with_option_selection[0]["json"]


# ------------------------------- R5: single/bulk parity + isolation (T9) --------------------------- #


def test_single_and_batch_parity(client):
    """The same entity via /entity and /batch yields the same kestrel_results (R5)."""
    single = client.post("/api/v1/map/entity", json=_entity_body(kestrel_top_n=10)).json()["result"]
    batch = client.post(
        "/api/v1/map/batch",
        json={
            "entities": [
                {
                    "name": "glucose",
                    "entity_type": "metabolite",
                    "options": {"annotators": ["kestrel-hybrid-search"], "kestrel_top_n": 10},
                }
            ]
        },
    ).json()["results"][0]
    assert single["kestrel_results"] == batch["kestrel_results"]


def test_batch_per_entity_isolation_with_duplicate_names(client):
    """Each entity receives only its own rows; duplicates do not cross-contaminate (R5, T9)."""
    resp = client.post(
        "/api/v1/map/batch",
        json={
            "entities": [
                {
                    "name": "glucose",
                    "entity_type": "metabolite",
                    "options": {"annotators": ["kestrel-hybrid-search"], "kestrel_top_n": 5},
                },
                {
                    "name": "glucose",
                    "entity_type": "metabolite",
                    "options": {"annotators": ["kestrel-hybrid-search"], "kestrel_top_n": 5},
                },
            ]
        },
    )
    assert resp.status_code == 200
    results = resp.json()["results"]
    assert len(results) == 2
    for r in results:
        assert r["kestrel_results"][0]["request"]["search_text"] == "glucose"
        assert r["kestrel_results"][0]["endpoint"] == "hybrid-search"


# ---------------------------------------- R6: no-call cases (T10) ---------------------------------- #


def test_annotation_mode_none_yields_empty_passthrough(client):
    """annotation_mode='none' makes no Kestrel call → kestrel_results == [] (R6)."""
    resp = client.post(
        "/api/v1/map/entity",
        json={
            "name": "glucose",
            "entity_type": "metabolite",
            "options": {"annotation_mode": "none", "kestrel_top_n": 10},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["result"]["kestrel_results"] == []


def test_provided_id_missing_mode_yields_empty_passthrough(client):
    """mode='missing' with a provided ID skips annotation → kestrel_results == [] (R6)."""
    resp = client.post(
        "/api/v1/map/entity",
        json={
            "name": "glucose",
            "entity_type": "metabolite",
            "identifiers": {"kegg": "C00031"},
            "options": {"annotation_mode": "missing", "kestrel_top_n": 10},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["result"]["kestrel_results"] == []


def test_non_kestrel_annotator_yields_empty_passthrough(client):
    """Only a non-Kestrel annotator selected → kestrel_results == [] (R6)."""
    resp = client.post(
        "/api/v1/map/entity",
        json={
            "name": "glucose",
            "entity_type": "metabolite",
            "options": {"annotators": ["goslin-lipid"], "kestrel_top_n": 10},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["result"]["kestrel_results"] == []


def test_whitespace_name_yields_empty_passthrough_and_no_call(mock_kestrel, client):
    """A whitespace-only name makes the Kestrel annotators return before issuing a request, so the
    endpoint is NOT recorded and the collector fires NO phantom call → kestrel_results == [] (R6).

    Regression for the provenance bug where endpoints were recorded from the selected annotator list
    rather than the calls actually issued, causing empty-search passthrough requests.
    """
    resp = client.post(
        "/api/v1/map/entity",
        json={
            "name": "   ",
            "entity_type": "metabolite",
            "options": {"annotators": ["kestrel-hybrid-search"], "kestrel_top_n": 10},
        },
    )
    assert resp.status_code == 200
    assert resp.json()["result"]["kestrel_results"] == []
    # No search call was made for the whitespace term (neither selection nor passthrough).
    assert not any(c["endpoint"] == "hybrid-search" for c in mock_kestrel)


# --------------------------------- R7: passthrough failure isolation (T11) ------------------------- #


def test_passthrough_failure_is_isolated(client, monkeypatch):
    """A passthrough call failure sets a classified error, empty rows, and HTTP 200 (R7, T11)."""

    def boom(*, endpoint, **kwargs):  # noqa: ANN001
        if endpoint in ("hybrid-search",) and kwargs.get("json", {}).get("limit") == 15:
            raise requests.exceptions.Timeout("slow")
        # selection + linking still succeed
        if endpoint == "canonicalize":
            return {c: c for c in kwargs["batch_items"]}
        if endpoint == "get-nodes":
            return {c: {"name": "glucose", "equivalent_ids": [c]} for c in kwargs["batch_items"]}
        return {t: [dict(_ONCAT)] for t in kwargs["batch_items"]}

    # Passthrough uses limit=15; selection hybrid uses limit=20, so only passthrough trips the boom.
    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", boom)
    resp = client.post("/api/v1/map/entity", json=_entity_body(kestrel_top_n=15))
    assert resp.status_code == 200
    result = resp.json()["result"]
    assert result["chosen_kg_id"] == "CHEBI:17234"  # mapping unaffected
    ksr = result["kestrel_results"]
    assert ksr[0]["error"] == "timeout"
    assert ksr[0]["rows"] == []


# ------------------------------- R8: routes + /dataset rejection ----------------------------------- #


def test_dataset_stream_carries_rows(client):
    """/map/dataset/stream returns kestrel_results per NDJSON line (R8)."""
    csv = "name,kegg\nglucose,\n"
    resp = client.post(
        "/api/v1/map/dataset/stream?entity_type=metabolite&name_column=name&provided_id_columns=kegg&annotators=kestrel-hybrid-search&kestrel_top_n=10",
        files={"file": ("in.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    assert resp.status_code == 200
    import json as _json

    lines = [_json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert lines[0]["kestrel_results"][0]["endpoint"] == "hybrid-search"
    assert lines[0]["kestrel_results"][0]["request"]["limit"] == 10


def test_dataset_stream_without_option_omits_rows(client):
    """/map/dataset/stream without the option has no kestrel_results key (byte-unchanged) (R8)."""
    csv = "name,kegg\nglucose,\n"
    resp = client.post(
        "/api/v1/map/dataset/stream?entity_type=metabolite&name_column=name&provided_id_columns=kegg&annotators=kestrel-hybrid-search",
        files={"file": ("in.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    assert resp.status_code == 200
    import json as _json

    lines = [_json.loads(line) for line in resp.text.splitlines() if line.strip()]
    assert "kestrel_results" not in lines[0]


def test_dataset_nonstream_rejects_option_with_422(client):
    """/map/dataset (non-streaming) rejects kestrel_top_n, pointing to the stream route (R8)."""
    csv = "name,kegg\nglucose,\n"
    resp = client.post(
        "/api/v1/map/dataset?entity_type=metabolite&name_column=name&provided_id_columns=kegg&kestrel_top_n=10",
        files={"file": ("in.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    assert resp.status_code == 422
    assert "/map/dataset/stream" in resp.json()["detail"]


def test_dataset_nonstream_without_option_unchanged(client):
    """/map/dataset without the option still works (not rejected) (R8)."""
    csv = "name,kegg\nglucose,\n"
    resp = client.post(
        "/api/v1/map/dataset?entity_type=metabolite&name_column=name&provided_id_columns=kegg&annotators=kestrel-hybrid-search",
        files={"file": ("in.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    assert resp.status_code == 200


# ------------------------------- /batch collects AFTER disarm (deadline safety) -------------------- #


def test_batch_collects_after_disarm(client, monkeypatch):
    """In /batch, passthrough collection happens AFTER disarm_batch_deadline() (R4 deadline safety)."""
    order: list[str] = []

    mapper = _app(client).state.mapper
    from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator

    mw = mapper.annotation_engine.annotator_registry[MetabolomicsWorkbenchAnnotator.slug]
    real_disarm = mw.disarm_batch_deadline

    def spy_disarm():
        order.append("disarm")
        return real_disarm()

    def spy_collect(**kwargs):
        order.append("collect")
        return []

    monkeypatch.setattr(mw, "disarm_batch_deadline", spy_disarm)
    monkeypatch.setattr(mapping_route, "collect_kestrel_passthrough", spy_collect)

    resp = client.post(
        "/api/v1/map/batch",
        json={
            "entities": [
                {
                    "name": "glucose",
                    "entity_type": "metabolite",
                    "options": {"annotators": ["kestrel-hybrid-search"], "kestrel_top_n": 5},
                },
                {
                    "name": "glucose",
                    "entity_type": "metabolite",
                    "options": {"annotators": ["kestrel-hybrid-search"], "kestrel_top_n": 5},
                },
            ]
        },
    )
    assert resp.status_code == 200
    assert "disarm" in order and "collect" in order
    assert order.index("disarm") < order.index("collect"), order


# ------------------------------------------ R9: payload cap --------------------------------------- #


def test_batch_over_cap_rejected_with_422(client, monkeypatch):
    """A batch whose projected passthrough rows exceed the cap is rejected before mapping (R9)."""
    monkeypatch.setattr("biomapper2.config.KESTREL_PASSTHROUGH_MAX_ROWS", 100)
    # Two entities at the max kestrel_top_n over the worst-case endpoint count exceed the tiny cap.
    resp = client.post(
        "/api/v1/map/batch",
        json={
            "entities": [
                {"name": "glucose", "entity_type": "metabolite", "options": {"kestrel_top_n": 100}},
                {"name": "fructose", "entity_type": "metabolite", "options": {"kestrel_top_n": 100}},
            ]
        },
    )
    assert resp.status_code == 422
    assert "cap" in resp.json()["detail"].lower()


def test_batch_under_cap_allowed(client, monkeypatch):
    """A batch within the cap still succeeds (R9)."""
    monkeypatch.setattr("biomapper2.config.KESTREL_PASSTHROUGH_MAX_ROWS", 100_000)
    resp = client.post(
        "/api/v1/map/batch",
        json={
            "entities": [
                {
                    "name": "glucose",
                    "entity_type": "metabolite",
                    "options": {"annotators": ["kestrel-hybrid-search"], "kestrel_top_n": 10},
                }
            ]
        },
    )
    assert resp.status_code == 200


def test_stream_over_cap_rejected_with_422(client, monkeypatch):
    """A dataset-stream request whose projected passthrough rows exceed the cap is rejected (R9)."""
    monkeypatch.setattr("biomapper2.config.KESTREL_PASSTHROUGH_MAX_ROWS", 100)
    # A multi-row file at the max kestrel_top_n over the worst-case endpoint count exceeds the tiny cap.
    csv = "name,kegg\n" + "".join(f"m{i},\n" for i in range(10))
    resp = client.post(
        "/api/v1/map/dataset/stream?entity_type=metabolite&name_column=name&provided_id_columns=kegg&annotators=kestrel-hybrid-search&kestrel_top_n=100",
        files={"file": ("in.csv", io.BytesIO(csv.encode()), "text/csv")},
    )
    assert resp.status_code == 422


def test_entity_never_hits_cap(client, monkeypatch):
    """A single entity's worst-case passthrough is tiny, so it never trips a reasonable cap (R9)."""
    monkeypatch.setattr("biomapper2.config.KESTREL_PASSTHROUGH_MAX_ROWS", 100_000)
    resp = client.post("/api/v1/map/entity", json=_entity_body(kestrel_top_n=100))
    assert resp.status_code == 200
