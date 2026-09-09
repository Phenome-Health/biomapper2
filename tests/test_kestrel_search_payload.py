"""Kestrel search annotators must send the schema field names the server actually honors.

Regression guard for the field-name fix: the annotators previously sent ``category_filter`` /
``prefix_filter``, which Kestrel's TextSearchRequest schema ignores (its fields are ``category`` /
``prefix``), so search ran effectively unfiltered. These tests capture the outgoing request body and
assert the correct keys. Each is a positive control that FAILS against the pre-fix payload literal.
No live Kestrel call is made — ``kestrel_request`` / ``bulk_kestrel_request`` are monkeypatched.
"""

from __future__ import annotations

from typing import Any

import biomapper2.api.kestrel_discovery as discovery
from biomapper2.core.annotators.kestrel_hybrid import KestrelHybridSearchAnnotator
from biomapper2.core.annotators.kestrel_text import KestrelTextSearchAnnotator
from biomapper2.core.annotators.kestrel_vector import KestrelVectorSearchAnnotator


def _capture(monkeypatch, module) -> dict[str, Any]:
    """Patch ``kestrel_request`` in the annotator module; return a dict the call records ``json`` into."""
    seen: dict[str, Any] = {}

    def fake(*_args, **kwargs):
        seen["json"] = kwargs.get("json")
        return {}

    monkeypatch.setattr(module, "kestrel_request", fake)
    return seen


def test_text_search_sends_category_not_category_filter(monkeypatch):
    import biomapper2.core.annotators.kestrel_text as mod

    seen = _capture(monkeypatch, mod)
    KestrelTextSearchAnnotator._kestrel_text_search("glucose", "biolink:SmallMolecule", None)
    body = seen["json"]
    assert body["category"] == "biolink:SmallMolecule"
    assert "category_filter" not in body and "prefix_filter" not in body


def test_vector_search_sends_category_not_category_filter(monkeypatch):
    import biomapper2.core.annotators.kestrel_vector as mod

    seen = _capture(monkeypatch, mod)
    KestrelVectorSearchAnnotator._kestrel_vector_search("glucose", "biolink:SmallMolecule", None)
    body = seen["json"]
    assert body["category"] == "biolink:SmallMolecule"
    assert "category_filter" not in body and "prefix_filter" not in body


def test_hybrid_search_sends_category_not_category_filter(monkeypatch):
    import biomapper2.core.annotators.kestrel_hybrid as mod

    seen = _capture(monkeypatch, mod)
    KestrelHybridSearchAnnotator._kestrel_hybrid_search("glucose", "biolink:SmallMolecule", None)
    body = seen["json"]
    assert body["category"] == "biolink:SmallMolecule"
    assert "category_filter" not in body and "prefix_filter" not in body


def test_prefix_omitted_when_empty_and_forwarded_when_populated(monkeypatch):
    import biomapper2.core.annotators.kestrel_text as mod

    seen = _capture(monkeypatch, mod)
    # Empty/None prefixes: no server-side namespace hard-filter (verified name-path no-op).
    KestrelTextSearchAnnotator._kestrel_text_search("glucose", "biolink:SmallMolecule", None)
    assert "prefix" not in seen["json"]
    # Populated prefixes: forwarded under the honored key so the vocab restriction actually applies.
    KestrelTextSearchAnnotator._kestrel_text_search("glucose", "biolink:SmallMolecule", ["CHEBI", "HMDB"])
    assert seen["json"]["prefix"] == ["CHEBI", "HMDB"]
    assert "prefix_filter" not in seen["json"]


def test_discovery_text_search_sends_category(monkeypatch):
    seen: dict[str, Any] = {}

    def fake_bulk(*_args, **kwargs):
        seen["json"] = kwargs.get("json")
        return {}

    monkeypatch.setattr(discovery, "bulk_kestrel_request", fake_bulk)
    discovery.sample_prefixes_for_category("biolink:SmallMolecule", ["glucose"])
    body = seen["json"]
    assert body["category"] == "biolink:SmallMolecule"
    assert "category_filter" not in body
