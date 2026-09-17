"""Unit 0: slash names never trip the RefMet circuit breaker.

The defect (F1): a slash-bearing name (e.g. "PC 16:0/18:1") makes MW /match return a 404, which was
raised as an HTTPError and counted as a circuit-breaker failure. After three such names the breaker
opened for the recovery window and RefMet /match was skipped for EVERY name and EVERY annotator
instance in the process. The fix classifies a per-name Bad-Request / Not-Found from /match as
``no_match`` -- a normal result, never raised -- so only 5xx / timeouts / transport errors can trip
the breaker. A slash name is an honest ``no_match``: MW's web server rejects the encoded slash
outright (%2F -> 404), so RefMet has no entry for that raw string. No name is rewritten to any other
form (the semantically-correct slash->queryable-level transform is deferred to later units).

Fake session only -- nothing here may touch a live service. The breaker is process-global (decorated
once at class definition), so it is reset around every test.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from circuitbreaker import CircuitBreakerMonitor

from biomapper2.core.annotators.base import AVAILABILITY_NO_MATCH, AVAILABILITY_VOTED
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator
from biomapper2.core.certificate import TierBOutcome
from biomapper2.core.structure_resolver import StructureResolver
from biomapper2.core.tier_b import IndependentStructureLookup

SLUG = MetabolomicsWorkbenchAnnotator.slug
_BREAKER_NAME = "MetabolomicsWorkbenchAnnotator._do_refmet_request"


def _breaker():
    return CircuitBreakerMonitor.get(_BREAKER_NAME)


@pytest.fixture(autouse=True)
def _reset_breaker():
    """Close the shared breaker and zero its failure count around every test."""
    breaker = _breaker()
    if breaker is not None:
        breaker.reset()
    yield
    if breaker is not None:
        breaker.reset()


class _FakeResponse:
    def __init__(self, payload: Any, status: int = 200) -> None:
        self._payload = payload
        self.status_code = status
        self.from_cache = False

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            import requests

            err = requests.HTTPError(f"HTTP {self.status_code}")
            err.response = type("R", (), {"status_code": self.status_code})()  # type: ignore[attr-defined]
            raise err

    def json(self) -> Any:
        return self._payload


class _FakeSession:
    """Records every URL requested so a test can assert on what was and was not called."""

    def __init__(self, responder) -> None:
        self._responder = responder
        self.calls: list[str] = []

    def get(self, url: str, timeout: float | None = None) -> _FakeResponse:  # noqa: ARG002
        self.calls.append(url)
        return self._responder(url)


def _annotator(session: _FakeSession, monkeypatch: pytest.MonkeyPatch) -> MetabolomicsWorkbenchAnnotator:
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="off", sleep=lambda _s: None, max_retries=0)
    monkeypatch.setattr(ann, "_session", session)
    return ann


def _name_segment_has_no_raw_slash(url: str) -> bool:
    """The name's slash must be percent-encoded (%2F), never sent as a raw extra path segment."""
    tail = url.split("/refmet/match/", 1)[-1] if "/refmet/match/" in url else url.split("/refmet/name/", 1)[-1]
    name_part = tail.split("/inchi_key", 1)[0]
    return "/" not in name_part


# -- annotator (core/annotators/metabolomics_workbench.py) ------------------------------------


def test_slash_name_is_no_match_not_unavailable(monkeypatch: pytest.MonkeyPatch):
    """Call site 1/3: a slash name 404s -> no_match (not unavailable), the slash is encoded, breaker stays closed."""
    session = _FakeSession(lambda _url: _FakeResponse({}, status=404))
    ann = _annotator(session, monkeypatch)
    result = ann._fetch_refmet_data("PC 16:0/18:1")
    assert result.status == AVAILABILITY_NO_MATCH
    assert _breaker().failure_count == 0
    assert len(session.calls) == 1
    assert "%2F" in session.calls[0] and _name_segment_has_no_raw_slash(session.calls[0])


def test_slash_names_do_not_trip_the_breaker_for_other_names(monkeypatch: pytest.MonkeyPatch):
    """Three slash names (each a 404 -> no_match) then 'carnitine' still VOTES."""

    def respond(url: str) -> _FakeResponse:
        if "carnitine" in url.lower():
            return _FakeResponse({"refmet_id": "RM0008606"})
        return _FakeResponse({}, status=404)

    ann = _annotator(_FakeSession(respond), monkeypatch)
    for name in ("PC 16:0/18:1", "PE 18:0/20:4", "SM d18:1/16:0"):
        assert ann.get_availability({"name": name}, "name") == {SLUG: AVAILABILITY_NO_MATCH}
    assert ann.get_availability({"name": "carnitine"}, "name") == {SLUG: AVAILABILITY_VOTED}
    assert _breaker().failure_count == 0


def test_http_404_is_no_match_not_unavailable(monkeypatch: pytest.MonkeyPatch):
    """A per-name 404 is a definitive no-match, not a raised outage."""
    ann = _annotator(_FakeSession(lambda _url: _FakeResponse({}, status=404)), monkeypatch)
    result = ann._fetch_refmet_data("definitely-not-a-metabolite")
    assert result.status == AVAILABILITY_NO_MATCH
    assert result.data is None
    assert _breaker().failure_count == 0


def test_http_400_is_no_match_not_unavailable(monkeypatch: pytest.MonkeyPatch):
    """A per-name Bad Request is handled on the same no-match branch as a Not Found, so the Bad
    Request path also never increments the shared breaker."""
    ann = _annotator(_FakeSession(lambda _url: _FakeResponse({}, status=400)), monkeypatch)
    result = ann._fetch_refmet_data("a-name-the-server-rejects")
    assert result.status == AVAILABILITY_NO_MATCH
    assert result.data is None
    assert _breaker().failure_count == 0


def test_repeated_404s_do_not_open_the_breaker(monkeypatch: pytest.MonkeyPatch):
    """Four 404s in a row leave the breaker closed (4xx is per-name, not an outage)."""

    def respond(url: str) -> _FakeResponse:
        if "good" in url:
            return _FakeResponse({"refmet_id": "RM1"})
        return _FakeResponse({}, status=404)

    ann = _annotator(_FakeSession(respond), monkeypatch)
    for name in ("miss1", "miss2", "miss3", "miss4"):
        assert ann._fetch_refmet_data(name).status == AVAILABILITY_NO_MATCH
    # The breaker never opened, so a subsequent good name still votes.
    assert ann._fetch_refmet_data("good").status == AVAILABILITY_VOTED
    assert _breaker().failure_count == 0


def test_a_second_annotator_instance_is_unaffected_by_slash_names(monkeypatch: pytest.MonkeyPatch):
    """The shared function-level breaker stays closed, so a separate instance still votes."""
    first = _annotator(_FakeSession(lambda _url: _FakeResponse({}, status=404)), monkeypatch)
    for name in ("PC 16:0/18:1", "PE 18:0/20:4", "SM d18:1/16:0"):
        assert first.get_availability({"name": name}, "name") == {SLUG: AVAILABILITY_NO_MATCH}

    second = _annotator(_FakeSession(lambda _url: _FakeResponse({"refmet_id": "RM_TAURINE"})), monkeypatch)
    assert second.get_availability({"name": "taurine"}, "name") == {SLUG: AVAILABILITY_VOTED}
    assert _breaker().failure_count == 0


# -- Tier B _fetch_mw (core/tier_b.py) --------------------------------------------------------


def test_tier_b_mw_slash_name_is_a_clean_not_found(monkeypatch: pytest.MonkeyPatch):
    """Call site 2/3: Tier B's MW hop 404s on a slash name -> clean unknown (not a lookup failure)."""
    session = _FakeSession(lambda _url: _FakeResponse({}, status=404))
    lookup = IndependentStructureLookup(session=session, sleep=lambda _s: None, clock=lambda: 0.0)
    result = lookup.lookup("PC 16:0/18:1")
    mw_calls = [url for url in session.calls if "refmet/name" in url]
    assert mw_calls and all("%2F" in url and _name_segment_has_no_raw_slash(url) for url in mw_calls)
    assert result.outcome is TierBOutcome.UNRESOLVABLE  # a 404 is a clean "unknown", not lookup_failed
    assert lookup.stats()["n_tier_b_lookup_failed"] == 0


# -- structure resolver _fetch_mw_inchikey (core/structure_resolver.py) -----------------------


def test_structure_resolver_mw_slash_name_is_no_match(monkeypatch: pytest.MonkeyPatch):
    """Call site 3/3: the candidate-side MW fetch 404s on a slash name -> None, no raise, slash encoded."""
    resolver = StructureResolver(linker=MagicMock())
    session = _FakeSession(lambda _url: _FakeResponse({}, status=404))
    monkeypatch.setattr(resolver, "_session", session)
    assert resolver._fetch_mw_inchikey("PC 16:0/18:1") is None
    assert len(session.calls) == 1
    assert "%2F" in session.calls[0] and _name_segment_has_no_raw_slash(session.calls[0])
