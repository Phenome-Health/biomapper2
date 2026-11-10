"""RefMet ``/match`` stopped returning ``refmet_id`` (observed 2026-11-10).

``/match`` still answers with ``refmet_name`` (plus formula, mass and classes) but no id; the
``/refmet/name/{refmet_name}/refmet_id`` endpoint still returns the id. These tests pin the backfill,
the no-match shape without the id field, the degraded-answer classification, and that the
``live_backup`` store never overwrites a last-known-good id with an id-less vote.
"""

from __future__ import annotations

import pytest
from circuitbreaker import CircuitBreakerMonitor

from biomapper2.core.annotators import refmet_store
from biomapper2.core.annotators.base import (
    AVAILABILITY_NO_MATCH,
    AVAILABILITY_UNAVAILABLE,
    AVAILABILITY_VOTED,
)
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator, RefMetResult

MATCH_NO_ID = {
    "refmet_name": "CAR 2:0",
    "formula": "C9H17NO4",
    "exactmass": "203.1158",
    "super_class": "Fatty Acyls",
    "main_class": "Fatty esters",
    "sub_class": "Acyl carnitines",
}


_BREAKER_NAME = "MetabolomicsWorkbenchAnnotator._do_refmet_request"


@pytest.fixture(autouse=True)
def _reset_breaker():
    """The /match breaker is shared across instances; other tests trip it on purpose."""
    breaker = CircuitBreakerMonitor.get(_BREAKER_NAME)
    if breaker is not None:
        breaker.reset()
    yield
    if breaker is not None:
        breaker.reset()


class _Resp:
    def __init__(self, body):
        self._body = body
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._body


class _Session:
    def __init__(self, match_body, name_body):
        self.match_body = match_body
        self.name_body = name_body
        self.urls: list[str] = []

    def get(self, url, timeout=None):
        self.urls.append(url)
        if url.startswith(MetabolomicsWorkbenchAnnotator.NAME_URL):
            return _Resp(self.name_body)
        return _Resp(self.match_body)


def _annotator(match_body, name_body=None) -> tuple[MetabolomicsWorkbenchAnnotator, _Session]:
    ann = MetabolomicsWorkbenchAnnotator(max_retries=0, freeze_mode="off")
    session = _Session(match_body, name_body)
    ann._session = session  # type: ignore[assignment]
    return ann, session


@pytest.mark.unit
def test_a_match_without_an_id_is_backfilled_from_the_name_endpoint() -> None:
    ann, session = _annotator(MATCH_NO_ID, {"name": "CAR 2:0", "refmet_id": "RM0154009"})
    result = ann._fetch_live("acetylcarnitine (C2)")
    assert result.status == AVAILABILITY_VOTED
    assert result.data is not None and result.data["refmet_id"] == "RM0154009"
    assert session.urls[-1] == f"{MetabolomicsWorkbenchAnnotator.NAME_URL}/CAR%202%3A0/refmet_id"
    votes = ann.get_annotations({"name": "acetylcarnitine (C2)"}, name_field="name", category="biolink:SmallMolecule")
    assert "RM0154009" in votes["metabolomics-workbench"]["refmet_id"]


@pytest.mark.unit
def test_a_lipid_refmet_name_keeps_its_chain_slash_literal_on_the_name_lookup() -> None:
    match = {"refmet_name": "PC 16:0/18:1", "formula": "C42H82NO8P"}
    ann, session = _annotator(match, {"name": "PC 16:0/18:1", "refmet_id": "RM0015001"})
    assert ann._fetch_live("PC(16:0/18:1)").data["refmet_id"] == "RM0015001"  # type: ignore[index]
    assert "%2F" not in session.urls[-1].upper()


@pytest.mark.unit
def test_a_dash_refmet_name_without_an_id_field_is_a_no_match_and_skips_the_name_lookup() -> None:
    dashes = {k: "-" for k in MATCH_NO_ID}
    ann, session = _annotator(dashes)
    assert ann._fetch_live("zzqq not a metabolite").status == AVAILABILITY_NO_MATCH
    assert not any(u.startswith(MetabolomicsWorkbenchAnnotator.NAME_URL) for u in session.urls)


@pytest.mark.unit
@pytest.mark.parametrize("name_body", [[], {}, {"name": "CAR 2:0"}, {"refmet_id": "-"}, None])
def test_a_match_whose_id_cannot_be_backfilled_is_unavailable_not_a_vote(name_body) -> None:
    ann, _ = _annotator(MATCH_NO_ID, name_body)
    result = ann._fetch_live("acetylcarnitine (C2)")
    assert result.status == AVAILABILITY_UNAVAILABLE
    assert not result.data


@pytest.mark.unit
def test_live_backup_never_overwrites_a_stored_id_with_an_id_less_vote(monkeypatch) -> None:
    ann = MetabolomicsWorkbenchAnnotator(max_retries=0, freeze_mode="live_backup")
    monkeypatch.setattr(ann, "_fetch_live", lambda name: RefMetResult(AVAILABILITY_VOTED, {"refmet_name": "CAR 2:0"}))
    writes: list[tuple] = []
    monkeypatch.setattr(refmet_store, "get", lambda name: None)
    monkeypatch.setattr(refmet_store, "upsert", lambda *a, **k: writes.append((a, k)))
    ann._fetch_live_backup("acetylcarnitine (C2)")
    assert writes == []


@pytest.mark.integration
@pytest.mark.requires_api
def test_canary_live_refmet_still_yields_ids_for_known_names() -> None:
    """Fails loudly if RefMet stops supplying ids through both /match and /name."""
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="off")
    for name, expected in (("Carnitine", "RM0008606"), ("acetylcarnitine (C2)", "RM0154009")):
        result = ann._fetch_live(name)
        assert result.status == AVAILABILITY_VOTED, (name, result)
        assert result.data is not None and result.data.get("refmet_id") == expected, (name, result.data)
