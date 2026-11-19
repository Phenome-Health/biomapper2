"""Unit tests covering concurrency, breaker caching, startup deadlines, and dataset fault boundaries."""

from __future__ import annotations

import contextvars
import io
import json
import threading
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
import requests
import requests_cache
from fastapi import FastAPI
from fastapi.testclient import TestClient

from biomapper2.api.auth import validate_api_key
from biomapper2.api.kestrel_discovery import derive_all_presets
from biomapper2.api.routes.mapping import router as mapping_router
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator
from biomapper2.core.gene_symbol_resolver import GeneSymbolResolver
from biomapper2.mapper import Mapper
from biomapper2.utils import (
    _CIRCUIT_BREAKER,
    CircuitBreakerOpenError,
    kestrel_request,
    reset_api_context,
    reset_circuit_breaker,
    set_api_context,
)


@pytest.mark.unit
def test_refmet_batch_deadline_isolated_across_contexts() -> None:
    """Concurrent request contexts keep independent RefMet batch deadlines."""
    clock_val = 10.0
    ann = MetabolomicsWorkbenchAnnotator.__new__(MetabolomicsWorkbenchAnnotator)
    ann._clock = lambda: clock_val  # type: ignore[attr-defined]
    ann.BATCH_DEADLINE_S = 30.0

    ctx_a = contextvars.copy_context()
    ctx_b = contextvars.copy_context()

    assert ctx_a.run(ann.arm_batch_deadline) is True
    clock_val = 25.0
    assert ctx_b.run(ann.arm_batch_deadline) is True

    assert ctx_a.run(lambda: ann._batch_deadline) == 40.0
    assert ctx_b.run(lambda: ann._batch_deadline) == 55.0
    assert ann._batch_deadline is None

    ctx_a.run(ann.disarm_batch_deadline)
    assert ctx_a.run(lambda: ann._batch_deadline) is None
    assert ctx_b.run(lambda: ann._batch_deadline) == 55.0
    ctx_b.run(ann.disarm_batch_deadline)


@pytest.mark.unit
def test_local_batch_and_stream_route_to_fast_executor() -> None:
    """Wholly local /batch and /dataset/stream work runs on fast_executor rather than kestrel_executor."""
    app = FastAPI()
    app.include_router(mapping_router, prefix="/map")
    app.dependency_overrides[validate_api_key] = lambda: "test-key"

    stub_mapper = MagicMock()
    stub_mapper.annotation_engine.annotator_registry = {}
    stub_mapper.map_entity_to_kg.return_value = {"name": "glucose", "chosen_kg_id": None}
    app.state.mapper = stub_mapper

    used_executors: list[str] = []

    class _TrackingExecutor:
        def __init__(self, label: str) -> None:
            self.label = label
            self._shutdown = False

        def submit(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
            from concurrent.futures import Future

            used_executors.append(self.label)
            fut: Future[Any] = Future()
            fut.set_result(fn(*args, **kwargs))
            return fut

    app.state.kestrel_executor = _TrackingExecutor("kestrel")
    app.state.fast_executor = _TrackingExecutor("fast")

    with TestClient(app) as client:
        rb = client.post(
            "/map/batch",
            json={
                "entities": [
                    {"name": "glucose", "entity_type": "metabolite", "options": {"annotation_mode": "none"}},
                ]
            },
        )
        assert rb.status_code == 200

        rs = client.post(
            "/map/dataset/stream",
            params={
                "entity_type": "metabolite",
                "name_column": "name",
                "provided_id_columns": "chebi",
                "annotation_mode": "none",
            },
            files={"file": ("local.tsv", io.BytesIO(b"name\tchebi\nglucose\t\n"), "text/tab-separated-values")},
        )
        assert rs.status_code == 200
        lines = [json.loads(line) for line in rs.text.splitlines() if line.strip()]
        assert len(lines) == 1

    assert used_executors == ["fast", "fast"]


@pytest.mark.unit
def test_cached_replies_do_not_reset_or_close_circuit_breaker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Cached Kestrel responses neither reset failure counts nor satisfy the recovery check."""
    monkeypatch.setenv("BIOMAPPER2_KESTREL_BREAKER_FAILURE_THRESHOLD", "2")
    monkeypatch.setenv("BIOMAPPER2_KESTREL_BREAKER_COOLDOWN_S", "0.03")
    token = set_api_context(timeout_s=0.5, max_retries=0, retry_backoff_base=0.01, use_circuit_breaker=True)
    reset_circuit_breaker()

    def _resp(status_code: int, *, from_cache: bool, payload: dict[str, Any]) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status_code
        resp.from_cache = from_cache
        if status_code >= 400:
            resp.raise_for_status.side_effect = requests.HTTPError(f"HTTP {status_code}", response=resp)
        else:
            resp.raise_for_status.return_value = None
            resp.json.return_value = payload
        return resp

    try:
        mock_sess = MagicMock()
        mock_sess.request.side_effect = [
            requests.Timeout("first timeout"),
            _resp(200, from_cache=True, payload={"cached": []}),
            requests.Timeout("second timeout"),
        ]
        with patch("biomapper2.utils.get_session", return_value=mock_sess):
            with pytest.raises(requests.Timeout):
                kestrel_request("POST", "text-search", "items", ["x"], 10, json={})
            assert _CIRCUIT_BREAKER._consecutive_failures == 1

            # Cached reply while CLOSED serves data without erasing the failure count
            out = kestrel_request("POST", "text-search", "items", ["cached"], 10, json={})
            assert out == {"cached": []}
            assert _CIRCUIT_BREAKER._consecutive_failures == 1

            # Next live failure reaches threshold=2 and opens the breaker
            with pytest.raises(requests.Timeout):
                kestrel_request("POST", "text-search", "items", ["y"], 10, json={})
            assert _CIRCUIT_BREAKER.state == "open"

        # After cooldown, a cached reply must NOT close the breaker; only a live reply may close it
        time.sleep(0.05)
        mock_sess2 = MagicMock()
        mock_sess2.request.side_effect = [
            _resp(200, from_cache=True, payload={"cached": []}),
            _resp(200, from_cache=False, payload={"live": []}),
        ]
        with patch("biomapper2.utils.get_session", return_value=mock_sess2):
            cached_out = kestrel_request("POST", "text-search", "items", ["cached"], 10, json={})
            assert cached_out == {"cached": []}
            assert _CIRCUIT_BREAKER.state == "open"

            live_out = kestrel_request("POST", "text-search", "items", ["live"], 10, json={})
            assert live_out == {"live": []}
            assert _CIRCUIT_BREAKER.state == "closed"

        # Real CachedSession also serves cached entries while breaker is OPEN during cooldown
        cached_session = requests_cache.CachedSession(
            str(tmp_path / "breaker_cache"),
            allowable_methods=["GET", "POST"],
        )
        try:
            with patch.object(
                requests.Session,
                "send",
                side_effect=lambda req, **kw: _real_response(req, {"warm": [{"id": "CHEBI:1"}]}),
            ):
                warm = kestrel_request("POST", "text-search", "search_text", ["warm"], 10, session=cached_session)
                assert "warm" in warm

            _CIRCUIT_BREAKER.record_failure()
            _CIRCUIT_BREAKER.record_failure()
            assert _CIRCUIT_BREAKER.state == "open"

            served_while_open = kestrel_request(
                "POST", "text-search", "search_text", ["warm"], 10, session=cached_session
            )
            assert "warm" in served_while_open
            assert _CIRCUIT_BREAKER.state == "open"

            with pytest.raises(CircuitBreakerOpenError):
                kestrel_request("POST", "text-search", "search_text", ["cold"], 10, session=cached_session)
        finally:
            cached_session.close()
    finally:
        reset_api_context(token)
        reset_circuit_breaker()


def _real_response(request: requests.PreparedRequest, body: dict[str, Any]) -> requests.Response:
    import urllib3

    payload = json.dumps(body).encode("utf-8")
    resp = requests.Response()
    resp.status_code = 200
    resp._content = payload
    resp.headers["Content-Type"] = "application/json"
    resp.request = request
    resp.url = str(request.url)
    resp.raw = urllib3.HTTPResponse(
        body=io.BytesIO(payload),
        headers={"Content-Type": "application/json"},
        status=200,
        preload_content=False,
        request_url=str(request.url),
    )
    return resp


@pytest.mark.unit
def test_get_nodes_fault_rejects_kestrel_only_match_and_degrades_non_kestrel_match() -> None:
    """Per R1 / Rule 3.2-3.3, /get-nodes fault rejects Kestrel-only matches and marks non-Kestrel matches degraded."""
    from biomapper2.core.fault_tracking import KestrelFaultError, record_kestrel_fault

    mapper = Mapper.__new__(Mapper)
    mapper.tier_b = None
    mapper.lipid_resolver = None
    mapper.biolink_client = MagicMock()
    mapper.biolink_client.standardize_entity_type.return_value = "biolink:Gene"
    mapper.biolink_client.is_small_molecule.return_value = False
    mapper.normalizer = MagicMock()
    mapper.normalizer.get_standard_prefix.return_value = None
    mapper.normalizer.normalize.return_value = pd.Series(
        {"curies": ["NCBIGene:1"], "curies_provided": [], "curies_assigned": {"kestrel-hybrid-search": ["NCBIGene:1"]}}
    )
    mapper.annotation_engine = MagicMock()
    mapper.annotation_engine.annotate.return_value = pd.Series(
        {"assigned_ids": {"kestrel-hybrid-search": {"NCBIGene": {"1": {"score": 0.9}}}}}
    )
    mapper.linker = MagicMock()
    mapper.linker.link.return_value = pd.Series(
        {
            "kg_ids": {"NCBIGene:1": ["NCBIGene:1"]},
            "kg_ids_provided": {},
            "kg_ids_assigned": {"kestrel-hybrid-search": {"NCBIGene:1": ["NCBIGene:1"]}},
        }
    )

    def _failing_equiv(_ids: list[str]) -> tuple[dict[str, Any], bool]:
        record_kestrel_fault(None, "get-nodes: simulated outage")
        return {}, False

    mapper.linker.get_equivalent_ids_checked.side_effect = _failing_equiv
    mapper.resolver = MagicMock()
    mapper.resolver.resolve.return_value = pd.Series({"chosen_kg_id": "NCBIGene:1", "chosen_kg_id_review": None})
    mapper.resolver.structure_resolver = None
    mapper.resolver.node_lipid_readings = None

    with pytest.raises(KestrelFaultError):
        mapper.map_entity_to_kg(
            item={"name": "GH1"},
            name_field="name",
            provided_id_fields=[],
            entity_type="gene",
        )

    mapper.linker.link.return_value = pd.Series(
        {
            "kg_ids": {"NCBIGene:1": ["NCBIGene:1"]},
            "kg_ids_provided": {"NCBIGene:1": ["NCBIGene:1"]},
            "kg_ids_assigned": {},
        }
    )
    degraded = mapper.map_entity_to_kg(
        item={"name": "GH1", "ncbigene": "1"},
        name_field="name",
        provided_id_fields=["ncbigene"],
        entity_type="gene",
    )
    assert isinstance(degraded, dict)
    assert degraded["chosen_kg_id"] == "NCBIGene:1"
    assert degraded["kestrel_degraded"] is True


@pytest.mark.unit
def test_dataset_link_fault_preserves_input_columns_and_clears_rejected_certificates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """API dataset mapping handles bulk canonicalize faults, input column collisions, and certificate clearing."""
    import biomapper2.mapper as mapper_module
    from biomapper2.core.fault_tracking import record_kestrel_fault

    monkeypatch.setattr(mapper_module, "analyze_dataset_mapping", lambda *a, **k: {})
    monkeypatch.setattr(
        mapper_module,
        "build_run_provenance",
        lambda **kw: MagicMock(model_dump=lambda: {}),
    )

    mapper = Mapper.__new__(Mapper)
    mapper.tier_b = None
    mapper.lipid_resolver = None
    mapper.biolink_client = MagicMock()
    mapper.biolink_client.standardize_entity_type.return_value = "biolink:Gene"
    mapper.biolink_client.is_small_molecule.return_value = False
    mapper.normalizer = MagicMock()
    mapper.normalizer.get_standard_prefix.return_value = None
    mapper.normalizer.normalize.return_value = pd.DataFrame(
        [
            {"curies": ["NCBIGene:1"], "curies_provided": ["NCBIGene:1"], "curies_assigned": {}},
            {"curies": [], "curies_provided": [], "curies_assigned": {}},
        ]
    )
    mapper.annotation_engine = MagicMock()
    mapper.annotation_engine.annotate.return_value = pd.DataFrame([{"assigned_ids": {}}, {"assigned_ids": {}}])

    from biomapper2.core.linker import Linker

    mapper.linker = Linker()
    monkeypatch.setattr(
        "biomapper2.core.linker.kestrel_request",
        MagicMock(side_effect=requests.Timeout("canonicalize timed out")),
    )
    mapper.resolver = MagicMock()
    mapper.resolver.resolve.return_value = pd.DataFrame(
        [
            {"chosen_kg_id": None, "chosen_kg_id_review": None},
            {"chosen_kg_id": None, "chosen_kg_id_review": None},
        ]
    )
    mapper.resolver.structure_resolver = None
    mapper.resolver.node_lipid_readings = None

    token = set_api_context(timeout_s=1.0, max_retries=0)
    try:
        out_path, _ = mapper.map_dataset_to_kg(
            dataset=pd.DataFrame({"name": ["BRCA1", "blank"], "ncbigene": ["1", ""], "error": ["keep_a", "keep_b"]}),
            entity_type="gene",
            name_column="name",
            provided_id_columns=["ncbigene"],
            output_dir=tmp_path,
            annotation_mode="none",
        )
        out_df = pd.read_csv(out_path, sep="\t", keep_default_na=False)
        assert list(out_df["error_input"]) == ["keep_a", "keep_b"]
        assert "canonicalize" in str(out_df.loc[0, "error"])
        assert out_df.loc[1, "error"] == ""

        # Verify a pre-enrichment rejected match clears chosen_kg_id BEFORE certificate emission
        mapper.biolink_client.standardize_entity_type.return_value = "biolink:SmallMolecule"
        mapper.biolink_client.is_small_molecule.return_value = True

        def _annot_with_gene_fault(item: pd.DataFrame, **kw: Any) -> pd.DataFrame:
            record_kestrel_fault(0, "gene-symbol resolver get-nodes fault for 'GH1'")
            return pd.DataFrame([{"assigned_ids": {"kestrel-text-search": {"CHEBI": {"1": {}}}}}], index=item.index)

        mapper.annotation_engine.annotate.side_effect = _annot_with_gene_fault
        mapper.normalizer.normalize.return_value = pd.DataFrame(
            [{"curies": ["CHEBI:1"], "curies_provided": [], "curies_assigned": {"kestrel-text-search": ["CHEBI:1"]}}]
        )
        mapper.linker = MagicMock()
        mapper.linker.link.return_value = pd.DataFrame(
            [
                {
                    "kg_ids": {"CHEBI:1": ["CHEBI:1"]},
                    "kg_ids_provided": {},
                    "kg_ids_assigned": {"kestrel-text-search": {"CHEBI:1": ["CHEBI:1"]}},
                }
            ]
        )
        mapper.linker.get_equivalent_ids_checked.return_value = ({"CHEBI:1": {"INCHIKEY": ["AAAA"]}}, True)
        mapper.resolver.resolve.return_value = pd.DataFrame(
            [{"chosen_kg_id": "CHEBI:1", "chosen_kg_id_review": "conflict"}]
        )

        out_path2, _ = mapper.map_dataset_to_kg(
            dataset=pd.DataFrame({"name": ["GH1"], "chebi": [""], "kestrel_degraded": ["orig"]}),
            entity_type="metabolite",
            name_column="name",
            provided_id_columns=["chebi"],
            output_dir=tmp_path,
            output_prefix="rejected_cert",
        )
        out_df2 = pd.read_csv(out_path2, sep="\t", keep_default_na=False)
        assert not mapper.linker.get_equivalent_ids_checked.called
        assert out_df2.loc[0, "chosen_kg_id"] == ""
        assert out_df2.loc[0, "chosen_kg_id_review"] == ""
        assert out_df2.loc[0, "certificate_state"] == "unavailable"
        assert "gene-symbol" in str(out_df2.loc[0, "error"])

        # Verify RERESOLUTION_ENABLED swapping a provided-ID match to a Kestrel-only candidate on a
        # faulted row also clears chosen_kg_id and rebuilds an unresolved certificate
        monkeypatch.setattr(mapper_module.config, "RERESOLUTION_ENABLED", True)

        def _annot_with_search_fault(item: pd.DataFrame, **kw: Any) -> pd.DataFrame:
            record_kestrel_fault(0, "kestrel-vector-search: simulated outage")
            return pd.DataFrame([{"assigned_ids": {"kestrel-hybrid-search": {"CHEBI": {"2": {}}}}}], index=item.index)

        mapper.annotation_engine.annotate.side_effect = _annot_with_search_fault
        mapper.normalizer.normalize.return_value = pd.DataFrame(
            [
                {
                    "curies": ["CHEBI:1", "CHEBI:2"],
                    "curies_provided": ["CHEBI:1"],
                    "curies_assigned": {"kestrel-hybrid-search": ["CHEBI:2"]},
                }
            ]
        )
        mapper.linker.link.return_value = pd.DataFrame(
            [
                {
                    "kg_ids": {"CHEBI:1": ["CHEBI:1"], "CHEBI:2": ["CHEBI:2"]},
                    "kg_ids_provided": {"CHEBI:1": ["CHEBI:1"]},
                    "kg_ids_assigned": {"kestrel-hybrid-search": {"CHEBI:2": ["CHEBI:2"]}},
                }
            ]
        )
        mapper.linker.get_equivalent_ids_checked.return_value = ({"CHEBI:1": {"INCHIKEY": ["AAAA"]}}, True)
        mapper.resolver.resolve.return_value = pd.DataFrame(
            [{"chosen_kg_id": "CHEBI:1", "chosen_kg_id_review": None, "chosen_kg_id_lipid_hint": "hint"}]
        )
        real_certify = mapper._certify_and_reresolve

        def _swap_to_kestrel_only(**ckw: Any) -> Any:
            if ckw.get("chosen_kg_id") == "CHEBI:1":
                cert, _, _, _ = real_certify(**ckw)
                return cert, "CHEBI:2", {"INCHIKEY": ["BBBB"]}, None
            return real_certify(**ckw)

        monkeypatch.setattr(mapper, "_certify_and_reresolve", _swap_to_kestrel_only)

        out_path3, _ = mapper.map_dataset_to_kg(
            dataset=pd.DataFrame({"name": ["glucose"], "chebi": ["1"]}),
            entity_type="metabolite",
            name_column="name",
            provided_id_columns=["chebi"],
            output_dir=tmp_path,
            output_prefix="reresolved_fault",
        )
        out_df3 = pd.read_csv(out_path3, sep="\t", keep_default_na=False)
        assert out_df3.loc[0, "chosen_kg_id"] == ""
        assert out_df3.loc[0, "chosen_kg_id_lipid_hint"] == ""
        assert out_df3.loc[0, "certificate_state"] == "unavailable"
        assert "kestrel-vector-search" in str(out_df3.loc[0, "error"])
    finally:
        reset_api_context(token)


@pytest.mark.unit
def test_gene_symbol_resolver_fault_isolated_across_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    """Concurrent GeneSymbolResolver calls do not overwrite each other's fault state."""
    resolver = GeneSymbolResolver()
    barrier = threading.Barrier(2)

    def _mock_kestrel(**kwargs: Any) -> dict[str, Any]:
        curies = kwargs.get("batch_items") or []
        if curies == ["NCBIGene:2688"]:
            raise requests.Timeout("get-nodes timeout")
        return {}

    monkeypatch.setattr("biomapper2.core.gene_symbol_resolver.kestrel_request", _mock_kestrel)
    outcomes: dict[str, tuple[str | None, bool, bool]] = {}

    def _worker_fault() -> None:
        curie, faulted = resolver.resolve_checked("GH1")
        barrier.wait(timeout=2.0)
        time.sleep(0.02)
        outcomes["gh1"] = (curie, faulted, resolver.last_call_faulted("GH1"))

    def _worker_clean() -> None:
        barrier.wait(timeout=2.0)
        curie, faulted = resolver.resolve_checked("NOT_CURATED")
        outcomes["clean"] = (curie, faulted, resolver.last_call_faulted("NOT_CURATED"))

    t1 = threading.Thread(target=_worker_fault)
    t2 = threading.Thread(target=_worker_clean)
    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert outcomes["gh1"] == (None, True, True)
    assert outcomes["clean"] == (None, False, False)


@pytest.mark.unit
def test_startup_discovery_enforces_total_budget_across_categories_and_terms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """derive_all_presets starts the total budget before fetch_categories and clamps per-term timeouts."""
    fake_now = 0.0
    observed_timeouts: list[float] = []

    monkeypatch.setattr("biomapper2.api.kestrel_discovery.time.monotonic", lambda: fake_now)
    monkeypatch.setattr("biomapper2.api.kestrel_discovery.config.get_kestrel_startup_total_timeout_s", lambda: 8.0)
    monkeypatch.setattr("biomapper2.api.kestrel_discovery.config.get_kestrel_startup_request_timeout_s", lambda: 2.5)

    def _fake_bulk(method: str, endpoint: str, **kwargs: Any) -> Any:
        nonlocal fake_now
        req_timeout = float(kwargs["timeout"])
        observed_timeouts.append(round(req_timeout, 3))
        if endpoint == "categories":
            fake_now += 2.0
            return ["biolink:SmallMolecule", "biolink:Protein"]
        fake_now += req_timeout
        raise requests.Timeout("simulated startup hang")

    monkeypatch.setattr("biomapper2.api.kestrel_discovery.bulk_kestrel_request", _fake_bulk)

    presets, completed = derive_all_presets({"metabolite": "biolink:SmallMolecule", "protein": "biolink:Protein"})
    assert completed is False
    assert fake_now <= 8.0
    assert observed_timeouts == [2.5, 2.5, 2.5, 1.0]
    assert "biolink:SmallMolecule" in presets
