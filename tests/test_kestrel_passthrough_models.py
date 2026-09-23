"""Unit tests for the Kestrel passthrough response models (R2, R3).

``KestrelSearchResult`` carries the raw rows Kestrel returned as VERBATIM dicts — no coercion, no
null-filling, no field added or dropped. These tests validate every Unit 1 fixture through the model
and assert zero field change, the raw-fidelity contract (a row missing a known field or carrying an
odd-typed value survives untouched), plus the backward-compat contract on ``EntityMappingResult``
(R2/T13).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from biomapper2.api.models.responses import (
    EntityMappingResult,
    KestrelRequestParams,
    KestrelSearchResult,
)

pytestmark = pytest.mark.unit

_FIXTURE_DIR = Path(__file__).parent / "fixtures"
_FIXTURES = sorted(_FIXTURE_DIR.glob("kestrel_*_search_*.json"))


def _load(path: Path) -> tuple[str, str, list[dict]]:
    data = json.loads(path.read_text())
    endpoint = data["endpoint"]
    search_text = data["search_text"]
    rows = data["response"][search_text]
    return endpoint, search_text, rows


def _result_for(endpoint: str, rows: list[dict]) -> KestrelSearchResult:
    return KestrelSearchResult(
        endpoint=endpoint,  # type: ignore[arg-type]
        request=KestrelRequestParams(search_text="x", limit=10, category="biolink:SmallMolecule", prefix=None),
        rows=rows,
        fetch_strategy="separate_call",
    )


def test_fixtures_are_present():
    """The Unit 1 fixtures exist and cover all three endpoints."""
    endpoints = {_load(p)[0] for p in _FIXTURES}
    assert endpoints == {"text-search", "vector-search", "hybrid-search"}


@pytest.mark.parametrize("fixture", _FIXTURES, ids=lambda p: p.name)
def test_all_fixtures_pass_through_with_zero_field_change(fixture: Path):
    """Every fixture row survives the model verbatim — identical dict in, identical dict out (R3, T3)."""
    endpoint, _search_text, rows = _load(fixture)
    assert rows, f"{fixture.name} has no rows"
    result = _result_for(endpoint, rows)
    dumped = result.model_dump()["rows"]
    # Byte-for-byte: the serialized rows equal the raw input rows exactly (order, keys, values).
    assert dumped == rows


def test_rows_are_not_coerced_or_null_filled():
    """A row MISSING a known field / carrying an odd-typed value passes through untouched (R3 regression).

    This is the raw-fidelity contract a typed row model broke: it would null-fill the absent
    ``synonyms``/``name`` and could coerce ``score``. The verbatim dict must add nothing and change
    nothing.
    """
    raw = {"id": "CHEBI:17234", "score": "3.21", "extra_only": {"nested": [1, 2]}}  # no name/synonyms; score is a str
    result = _result_for("hybrid-search", [raw])
    out = result.model_dump()["rows"][0]
    assert out == raw
    assert out["score"] == "3.21"  # NOT coerced to float
    assert "name" not in out and "synonyms" not in out  # NOT null-filled


def test_curie_preserved_byte_for_byte():
    """IDs/CURIEs are preserved as strings without coercion."""
    result = _result_for("hybrid-search", [{"id": "CHEBI:17234", "score": 3.21}])
    assert result.model_dump()["rows"][0]["id"] == "CHEBI:17234"


def test_hybrid_fixture_carries_a_sub_threshold_row():
    """The hybrid fixture keeps a score<0.5 row — the raw fidelity the >=0.5 filter would drop (R3)."""
    fixture = _FIXTURE_DIR / "kestrel_hybrid_search_glucose.json"
    endpoint, _search_text, rows = _load(fixture)
    result = _result_for(endpoint, rows)
    assert any(r["score"] < 0.5 for r in result.model_dump()["rows"])


def test_search_result_nests_rows_and_request():
    """KestrelSearchResult nests request params + raw rows and defaults error to None."""
    result = KestrelSearchResult(
        endpoint="hybrid-search",
        request=KestrelRequestParams(
            search_text="glucose", limit=10, category="biolink:SmallMolecule", prefix=["CHEBI"]
        ),
        rows=[{"id": "CHEBI:17234", "score": 3.21}],
        fetch_strategy="separate_call",
    )
    assert result.error is None
    assert result.fetch_strategy == "separate_call"
    assert result.request.limit == 10
    assert result.rows[0]["id"] == "CHEBI:17234"


def test_search_result_rejects_unknown_endpoint():
    """endpoint is a closed Literal set."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KestrelSearchResult(
            endpoint="graph-search",  # type: ignore[arg-type]
            request=KestrelRequestParams(search_text="x", limit=1, category="c", prefix=None),
            rows=[],
            fetch_strategy="separate_call",
        )


@pytest.mark.parametrize("bad", ["str_error", "not_a_class"])
def test_search_result_error_is_enumerated(bad):
    """error is a closed Literal set, not free text (avoids leaking str(exc))."""
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        KestrelSearchResult(
            endpoint="hybrid-search",
            request=KestrelRequestParams(search_text="x", limit=1, category="c", prefix=None),
            rows=[],
            fetch_strategy="separate_call",
            error=bad,  # type: ignore[arg-type]
        )


# --------------------------------- EntityMappingResult wiring (R2) -------------------------------- #


def test_entity_result_omits_kestrel_results_by_default():
    """Default is None so an existing response is byte-unchanged (R2, T2)."""
    result = EntityMappingResult(name="glucose")
    assert result.kestrel_results is None
    # Serialized with exclude_none (as the API does), the key is absent entirely.
    assert "kestrel_results" not in result.model_dump(exclude_none=True)


def test_entity_result_carries_kestrel_results_when_set():
    """When set, kestrel_results is a list of KestrelSearchResult (R2)."""
    ksr = KestrelSearchResult(
        endpoint="hybrid-search",
        request=KestrelRequestParams(search_text="glucose", limit=5, category="biolink:SmallMolecule", prefix=None),
        rows=[{"id": "CHEBI:17234", "score": 3.21}],
        fetch_strategy="separate_call",
    )
    result = EntityMappingResult(name="glucose", kestrel_results=[ksr])
    assert result.kestrel_results is not None
    assert result.kestrel_results[0].endpoint == "hybrid-search"


def test_pre_change_response_still_parses_new_response():
    """A response WITH the new field still validates as EntityMappingResult (backward compat, T13)."""
    payload = EntityMappingResult(name="glucose", chosen_kg_id="CHEBI:17234").model_dump()
    payload["kestrel_results"] = [
        {
            "endpoint": "hybrid-search",
            "request": {"search_text": "glucose", "limit": 5, "category": "biolink:SmallMolecule", "prefix": None},
            "rows": [{"id": "CHEBI:17234", "score": 3.21, "surprise_field": "kept"}],
            "fetch_strategy": "separate_call",
            "error": None,
        }
    ]
    reparsed = EntityMappingResult(**payload)
    assert reparsed.kestrel_results is not None
    assert reparsed.kestrel_results[0].rows[0]["surprise_field"] == "kept"
