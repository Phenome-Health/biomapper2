"""Unit tests for the optional ``candidate_limit`` request knob.

``candidate_limit`` lets a caller set the Kestrel search ``limit`` directly. An explicit value always
wins over the adaptive default (20 when a re-ranking policy is active, else 1 for hybrid); None keeps
the current adaptive behavior unchanged. These tests monkeypatch the private ``_kestrel_*_search``
helpers to capture the outgoing ``limit`` — no live Kestrel call is ever made.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from biomapper2.api.models.requests import MappingOptions
from biomapper2.config import HYBRID_SEARCH_LIMIT
from biomapper2.core.annotators.kestrel_hybrid import KestrelHybridSearchAnnotator
from biomapper2.core.annotators.kestrel_text import KestrelTextSearchAnnotator
from biomapper2.core.annotators.kestrel_vector import KestrelVectorSearchAnnotator

_ENTITY = {"name": "glucose"}
_CATEGORY = "biolink:SmallMolecule"


def _capture_limit(monkeypatch, annotator_cls: type, method_name: str) -> dict[str, Any]:
    """Patch a ``_kestrel_*_search`` staticmethod to record its ``limit`` and return an empty result."""
    captured: dict[str, Any] = {}

    def fake(search_text, category, prefixes, limit: int = 10):
        captured["limit"] = limit
        term = search_text if isinstance(search_text, str) else (search_text[0] if search_text else None)
        return {term: []}

    monkeypatch.setattr(annotator_cls, method_name, staticmethod(fake))
    return captured


# --------------------------------------- Model validation ----------------------------------------- #


def test_mapping_options_accepts_candidate_limit():
    """An in-range candidate_limit is accepted and stored; the default is None."""
    assert MappingOptions().candidate_limit is None
    assert MappingOptions(candidate_limit=5).candidate_limit == 5


@pytest.mark.parametrize("bad", [0, 101])
def test_mapping_options_rejects_out_of_range(bad):
    """Values below the floor or above the ceiling raise a pydantic ValidationError (an HTTP 4xx boundary)."""
    with pytest.raises(ValidationError):
        MappingOptions(candidate_limit=bad)


# ------------------------------------------ Hybrid limit ------------------------------------------- #


def test_hybrid_explicit_limit_overrides_prefer_human_false(monkeypatch):
    """candidate_limit wins even with prefer_human=False (which would otherwise force limit=1)."""
    captured = _capture_limit(monkeypatch, KestrelHybridSearchAnnotator, "_kestrel_hybrid_search")
    KestrelHybridSearchAnnotator().get_annotations(_ENTITY, "name", _CATEGORY, prefer_human=False, candidate_limit=7)
    assert captured["limit"] == 7


def test_hybrid_none_prefer_human_false_is_one(monkeypatch):
    """With no candidate_limit and no re-ranking policy, hybrid keeps its narrow window of 1."""
    captured = _capture_limit(monkeypatch, KestrelHybridSearchAnnotator, "_kestrel_hybrid_search")
    KestrelHybridSearchAnnotator().get_annotations(_ENTITY, "name", _CATEGORY, prefer_human=False, candidate_limit=None)
    assert captured["limit"] == 1


def test_hybrid_none_prefer_human_true_is_default(monkeypatch):
    """With no candidate_limit but prefer_human active, hybrid uses the adaptive default (20)."""
    captured = _capture_limit(monkeypatch, KestrelHybridSearchAnnotator, "_kestrel_hybrid_search")
    KestrelHybridSearchAnnotator().get_annotations(_ENTITY, "name", _CATEGORY, prefer_human=True, candidate_limit=None)
    assert captured["limit"] == HYBRID_SEARCH_LIMIT == 20


# --------------------------------------- Text / vector limit --------------------------------------- #


def test_text_explicit_limit_reaches_search(monkeypatch):
    """candidate_limit overrides the text-search default window and reaches the search limit."""
    captured = _capture_limit(monkeypatch, KestrelTextSearchAnnotator, "_kestrel_text_search")
    KestrelTextSearchAnnotator().get_annotations(_ENTITY, "name", _CATEGORY, candidate_limit=5)
    assert captured["limit"] == 5


def test_vector_none_uses_default_window(monkeypatch):
    """With no candidate_limit, vector search keeps the adaptive default (HYBRID_SEARCH_LIMIT)."""
    captured = _capture_limit(monkeypatch, KestrelVectorSearchAnnotator, "_kestrel_vector_search")
    KestrelVectorSearchAnnotator().get_annotations(_ENTITY, "name", _CATEGORY, candidate_limit=None)
    assert captured["limit"] == HYBRID_SEARCH_LIMIT
