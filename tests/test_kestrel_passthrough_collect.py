"""Unit tests for the passthrough collector ``kestrel_passthrough.collect`` (R3, R5, R6, R7).

Every Kestrel call is MOCKED via ``kestrel_passthrough.kestrel_request`` — no live/paid call is made.
The collector fetches raw rows from the RECORDED endpoint plan (no annotator re-derivation), truncates
to N, and preserves rows byte-for-byte including hybrid rows the >=0.5 selection filter would drop.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import requests

from biomapper2.config import KESTREL_BATCH_SIZE_SEARCH
from biomapper2.core import kestrel_passthrough
from biomapper2.utils import BisectBudgetExceeded

pytestmark = pytest.mark.unit

_FIXTURE_DIR = Path(__file__).parent / "fixtures"


def _fixture_rows(name: str) -> tuple[str, list[dict]]:
    data = json.loads((_FIXTURE_DIR / name).read_text())
    search_text = data["search_text"]
    return search_text, data["response"][search_text]


def test_collect_truncates_to_n_and_preserves_order(monkeypatch):
    """n=1 returns the first row only, in Kestrel's order; request.limit reflects n (T5)."""
    rows = [{"id": f"CHEBI:{i}", "score": 5.0 - i} for i in range(100)]
    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", lambda **kw: {"glucose": rows})

    results = kestrel_passthrough.collect("glucose", "biolink:SmallMolecule", ["CHEBI"], ["hybrid-search"], n=10)

    assert len(results) == 1
    result = results[0]
    assert len(result.rows) == 10
    assert [r["id"] for r in result.rows] == [f"CHEBI:{i}" for i in range(10)]
    assert result.request.limit == 10
    assert result.request.search_text == "glucose"
    assert result.fetch_strategy == "separate_call"
    assert result.error is None


def test_collect_sends_limit_n_and_selection_shaped_payload(monkeypatch):
    """The passthrough call replicates the selection payload shape with limit=N (cache-key parity)."""
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        return {"glucose": []}

    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", fake)
    kestrel_passthrough.collect("glucose", "biolink:SmallMolecule", ["CHEBI", "HMDB"], ["text-search"], n=7)

    assert captured["method"] == "POST"
    assert captured["endpoint"] == "text-search"
    assert captured["batch_field"] == "search_text"
    assert captured["batch_items"] == ["glucose"]
    assert captured["batch_size"] == KESTREL_BATCH_SIZE_SEARCH
    assert captured["json"] == {"limit": 7, "category": "biolink:SmallMolecule", "prefix": ["CHEBI", "HMDB"]}


def test_collect_omits_prefix_when_none(monkeypatch):
    """No prefixes → the payload omits 'prefix' entirely, matching the selection call shape."""
    captured: dict = {}
    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", lambda **kw: captured.update(kw) or {"x": []})
    kestrel_passthrough.collect("x", "biolink:SmallMolecule", [], ["hybrid-search"], n=3)
    assert "prefix" not in captured["json"]


def test_collect_raw_fidelity_keeps_sub_threshold_hybrid_row(monkeypatch):
    """A hybrid row with score<0.5 is passed through raw — the selection >=0.5 filter is NOT applied (R3, T4)."""
    _search_text, rows = _fixture_rows("kestrel_hybrid_search_glucose.json")
    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", lambda **kw: {"glucose": rows})

    results = kestrel_passthrough.collect("glucose", "biolink:SmallMolecule", ["CHEBI"], ["hybrid-search"], n=100)

    scores = [r["score"] for r in results[0].rows]
    assert any(s < 0.5 for s in scores), "sub-threshold hybrid row must survive raw passthrough"


def test_collect_rows_are_verbatim_dicts_not_coerced(monkeypatch):
    """Rows are the raw Kestrel dicts, untouched — no model round-trip, coercion, or null-fill (R3)."""
    raw = [{"id": "CHEBI:17234", "score": "3.21", "weird": {"n": [1]}}]  # score is a str; no name/synonyms
    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", lambda **kw: {"glucose": raw})
    results = kestrel_passthrough.collect("glucose", "c", None, ["hybrid-search"], n=10)
    assert results[0].rows == raw
    assert results[0].rows[0]["score"] == "3.21"  # NOT coerced
    assert "name" not in results[0].rows[0]  # NOT null-filled


def test_collect_does_not_count_requests(monkeypatch):
    """Passthrough calls pass count_requests=False so shared benchmark counters stay untouched.

    A response-only option must not move request_counter_snapshot, or enabling kestrel_top_n would
    make passthrough traffic indistinguishable from mapping traffic in benchmark manifests.
    """
    captured: dict = {}
    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", lambda **kw: captured.update(kw) or {"glucose": []})
    kestrel_passthrough.collect("glucose", "biolink:SmallMolecule", ["CHEBI"], ["hybrid-search"], n=5)
    assert captured["count_requests"] is False


def test_collect_empty_endpoints_returns_empty(monkeypatch):
    """No recorded endpoints → [] (R6). No Kestrel call is made."""
    called = False

    def fake(**kw):
        nonlocal called
        called = True
        return {}

    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", fake)
    assert kestrel_passthrough.collect("glucose", "biolink:SmallMolecule", ["CHEBI"], [], n=10) == []
    assert called is False


def test_collect_all_three_endpoints(monkeypatch):
    """Recorded [text, vector, hybrid] → three results, one per endpoint, in order."""
    monkeypatch.setattr(
        kestrel_passthrough, "kestrel_request", lambda **kw: {"glucose": [{"id": "CHEBI:17234", "score": 1.0}]}
    )
    results = kestrel_passthrough.collect(
        "glucose", "biolink:SmallMolecule", ["CHEBI"], ["text-search", "vector-search", "hybrid-search"], n=5
    )
    assert [r.endpoint for r in results] == ["text-search", "vector-search", "hybrid-search"]


def test_collect_missing_search_text_key_yields_empty_rows(monkeypatch):
    """Kestrel returning no entry for the term → rows=[] with no error (not a failure)."""
    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", lambda **kw: {})
    results = kestrel_passthrough.collect("glucose", "c", None, ["hybrid-search"], n=10)
    assert results[0].rows == []
    assert results[0].error is None


# ------------------------------------------ Error paths (R7) --------------------------------------- #


@pytest.mark.parametrize(
    "exc,expected",
    [
        (requests.exceptions.Timeout("slow"), "timeout"),
        (requests.exceptions.ConnectionError("down"), "upstream_error"),
        (BisectBudgetExceeded("budget"), "upstream_error"),
        (ValueError("No JSON object could be decoded"), "malformed_response"),
        (RuntimeError("weird"), "other"),
    ],
)
def test_collect_classifies_errors_and_never_propagates(monkeypatch, exc, expected):
    """Each raised exception is classified, rows=[], and nothing propagates (R7, T11)."""

    def boom(**kw):
        raise exc

    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", boom)
    results = kestrel_passthrough.collect("glucose", "c", None, ["hybrid-search"], n=10)
    assert results[0].error == expected
    assert results[0].rows == []


def test_collect_http_error_5xx_is_upstream_error(monkeypatch):
    """An HTTPError (5xx) from raise_for_status classifies as upstream_error (R7, T11)."""
    response = requests.Response()
    response.status_code = 503

    def boom(**kw):
        raise requests.exceptions.HTTPError("503 Server Error", response=response)

    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", boom)
    results = kestrel_passthrough.collect("glucose", "c", None, ["hybrid-search"], n=10)
    assert results[0].error == "upstream_error"
    assert results[0].rows == []


def test_collect_one_endpoint_fails_others_succeed(monkeypatch):
    """A failure on one endpoint is isolated; the other endpoint still returns rows."""

    def selective(**kw):
        if kw["endpoint"] == "text-search":
            raise requests.exceptions.Timeout("slow")
        return {"glucose": [{"id": "CHEBI:17234", "score": 1.0}]}

    monkeypatch.setattr(kestrel_passthrough, "kestrel_request", selective)
    results = kestrel_passthrough.collect(
        "glucose", "biolink:SmallMolecule", ["CHEBI"], ["text-search", "hybrid-search"], n=5
    )
    by_ep = {r.endpoint: r for r in results}
    assert by_ep["text-search"].error == "timeout"
    assert by_ep["text-search"].rows == []
    assert by_ep["hybrid-search"].error is None
    assert len(by_ep["hybrid-search"].rows) == 1
