"""Tier B: opt-in, default-off, and never reached by accident.

Every test here runs against a fake session. Nothing in this file may touch Metabolomics Workbench,
PubChem or Kestrel -- the single committed Tier-B sweep that produces the published curve is a
separate supervised step, not something a test suite fires.
"""

from __future__ import annotations

from typing import Any

import pytest

from biomapper2.config import MW_INCHIKEY_URL
from biomapper2.core.certificate import TierBOutcome
from biomapper2.core.tier_b import IndependentStructureLookup

MW_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"
PUBCHEM_KEY = "QNAYBMKLOCPYGJ-REOHCLBHSA-N"


class _FakeResponse:
    def __init__(self, payload: Any, status: int = 200, from_cache: bool = False) -> None:
        self._payload = payload
        self.status_code = status
        self.from_cache = from_cache

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise _HTTPError(self.status_code)

    def json(self) -> Any:
        return self._payload


class _HTTPError(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.status = status
        self.response = type("R", (), {"status_code": status})()


class _FakeSession:
    """Records every URL requested so a test can assert on what was and was not called."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, timeout: float | None = None) -> _FakeResponse:  # noqa: ARG002
        self.calls.append(url)
        for fragment, response in self.responses.items():
            if fragment in url:
                if isinstance(response, Exception):
                    raise response
                return response
        return _FakeResponse({}, status=404)


def _lookup(responses: dict[str, Any], **kwargs) -> tuple[IndependentStructureLookup, _FakeSession]:
    session = _FakeSession(responses)
    slept: list[float] = []
    return (
        IndependentStructureLookup(session=session, sleep=slept.append, clock=lambda: 0.0, **kwargs),
        session,
    )


def test_default_is_off_in_config() -> None:
    """Default-off is the contract, not a deployment convention. Changing this fires network calls
    for every unique query name across every arm."""
    from biomapper2 import config

    assert config.TIER_B_ENABLED is False


def test_mw_hit_resolves_the_query_name() -> None:
    lookup, session = _lookup({"refmet/name": _FakeResponse({"inchi_key": MW_KEY})})
    result = lookup.lookup("glucose")
    assert result.outcome is TierBOutcome.RESOLVED
    assert result.source == "metabolomics-workbench"
    assert result.inchikey_block == "BSYNRYMUTXBXSQ"
    assert all("pubchem" not in url for url in session.calls), "PubChem must not be called after an MW hit"


def test_pubchem_is_the_second_hop() -> None:
    lookup, session = _lookup(
        {
            "refmet/name": _FakeResponse({"inchi_key": "-"}),
            "pubchem": _FakeResponse({"PropertyTable": {"Properties": [{"InChIKey": PUBCHEM_KEY}]}}),
        }
    )
    result = lookup.lookup("some obscure name")
    assert result.outcome is TierBOutcome.RESOLVED
    assert result.source == "pubchem"
    assert result.inchikey_block == "QNAYBMKLOCPYGJ"
    assert len(session.calls) == 2


def test_a_name_neither_registry_knows_is_unresolvable_not_failed() -> None:
    """Metabolon-style names miss the exact-name endpoints often. That is a property of the input,
    and it must not read as a network failure."""
    lookup, _ = _lookup({"refmet/name": _FakeResponse({}, status=404), "pubchem": _FakeResponse({}, status=404)})
    result = lookup.lookup("X-12345")
    assert result.outcome is TierBOutcome.UNRESOLVABLE
    assert result.inchikey_block is None


def test_a_rate_limited_service_is_lookup_failed_not_unresolvable() -> None:
    """Otherwise a throttled PubChem turns the published curve into a network artifact that nothing
    in the provenance records."""
    lookup, _ = _lookup(
        {"refmet/name": _FakeResponse({}, status=404), "pubchem": _FakeResponse({}, status=503)},
        max_attempts=2,
    )
    result = lookup.lookup("glucose")
    assert result.outcome is TierBOutcome.LOOKUP_FAILED


def test_transport_errors_are_swallowed_into_lookup_failed() -> None:
    """The swallow-everything try/except lives only in ``StructureResolver.inchikey_block``; calling
    the fetchers directly bypasses it and a raise propagates into the mapping loop."""
    lookup, _ = _lookup({"refmet/name": RuntimeError("connection reset")})
    result = lookup.lookup("glucose")
    assert result.outcome is TierBOutcome.LOOKUP_FAILED


def test_retries_back_off_before_giving_up() -> None:
    slept: list[float] = []
    session = _FakeSession({"refmet/name": _FakeResponse({}, status=503)})
    lookup = IndependentStructureLookup(
        # Throttle disabled so the recorded sleeps are the backoff and nothing else.
        session=session,
        sleep=slept.append,
        clock=lambda: 0.0,
        min_interval_s=0,
        max_attempts=3,
        backoff_base_s=0.5,
    )
    lookup.lookup("glucose")
    assert len(slept) >= 2
    assert slept[1] > slept[0], "backoff must grow"


def test_throttle_paces_outbound_calls() -> None:
    """PUG-REST is rate-limited and Tier B moves these calls from a small conflict subset to every
    unique query name across every arm."""
    now = [0.0]
    slept: list[float] = []
    session = _FakeSession({"refmet/name": _FakeResponse({"inchi_key": MW_KEY})})
    lookup = IndependentStructureLookup(session=session, sleep=slept.append, clock=lambda: now[0], min_interval_s=0.2)
    lookup.lookup("a")
    lookup.lookup("b")
    assert slept and slept[0] > 0


def test_repeated_names_are_memoized_without_a_second_call() -> None:
    lookup, session = _lookup({"refmet/name": _FakeResponse({"inchi_key": MW_KEY})})
    first = lookup.lookup("glucose")
    second = lookup.lookup("glucose")
    assert len(session.calls) == 1
    assert first.inchikey_block == second.inchikey_block
    assert second.cache_state == "process_memo"


def test_cache_state_is_recorded_from_the_response() -> None:
    lookup, _ = _lookup({"refmet/name": _FakeResponse({"inchi_key": MW_KEY}, from_cache=True)})
    assert lookup.lookup("glucose").cache_state == "hit"


def test_resolution_rate_is_reported_beside_every_operating_point() -> None:
    """``MW_INCHIKEY_URL`` is the EXACT-name endpoint while the annotator uses fuzzy
    ``refmet/match``, so corroboration would otherwise be computed on a biased easy subset with no
    way for a reader to see it."""
    lookup, _ = _lookup(
        {
            "refmet/name": _FakeResponse({}, status=404),
            "pubchem": _FakeResponse({"PropertyTable": {"Properties": [{"InChIKey": PUBCHEM_KEY}]}}),
        }
    )
    lookup.lookup("glucose")
    lookup.lookup("glucose")  # memoized; must not double-count
    stats = lookup.stats()
    assert stats["n_unique_query_names"] == 1
    assert stats["n_tier_b_resolved"] == 1
    assert stats["resolution_rate"] == 1.0


def test_an_empty_name_is_unresolvable_without_any_call() -> None:
    lookup, session = _lookup({})
    assert lookup.lookup("").outcome is TierBOutcome.UNRESOLVABLE
    assert session.calls == []


@pytest.mark.parametrize("name", ["glucose", "X-12345"])
def test_lookup_never_touches_kestrel(name: str) -> None:
    """Tier B resolves the QUERY NAME against external registries only. A Kestrel call here would
    make the 'independent' evidence a second read of the same graph."""
    lookup, session = _lookup({"refmet/name": _FakeResponse({"inchi_key": MW_KEY})})
    lookup.lookup(name)
    assert all("kestrel" not in url and "krakenkg" not in url for url in session.calls)


def test_a_slash_in_the_query_name_is_percent_encoded() -> None:
    """A slash must not survive into the URL as a path separator.

    ``quote`` defaults to ``safe="/"``. With that default a lipid-shorthand name splits the URL into
    extra path segments, the registry 404s, and the row is recorded ``unresolvable`` -- a statement
    about the NAME -- when nothing about the name was ever actually asked. That biases the
    ``resolution_rate`` the figure's admissibility gate reads, in the direction that makes the gate
    look satisfied.

    The same default was live in shipped code at ``StructureResolver._fetch_mw_inchikey`` and
    ``_fetch_pubchem_inchikey``; ``test_structure_resolver_name_fallback_encodes_slashes`` pins those.
    """
    name = "GlcCer 16:0;O3/10:0"
    lookup, session = _lookup({"refmet/name": _FakeResponse({"inchi_key": MW_KEY})})
    lookup.lookup(name)

    assert session.calls, "no request was made"
    url = session.calls[0]
    assert "%2F" in url, f"the slash was not encoded: {url}"
    # Exactly the service's own path separators remain: nothing the name contributed.
    assert url.count("/") == MW_INCHIKEY_URL.count("/") + 2, f"the name injected a path segment: {url}"


def test_structure_resolver_name_fallback_encodes_slashes() -> None:
    """The same bug pre-existed on the shipped name-fallback rung, which fires whenever the KG
    carries no InChIKey -- so a slash-bearing name 404s and is recorded structurally unresolvable."""
    from biomapper2.core.structure_resolver import StructureResolver

    resolver = StructureResolver.__new__(StructureResolver)
    session = _FakeSession({"refmet/name": _FakeResponse({"inchi_key": MW_KEY})})
    resolver._session = session  # type: ignore[assignment]  # fake session, records the outbound URL only

    resolver._fetch_mw_inchikey("GlcCer 16:0;O3/10:0")
    assert "%2F" in session.calls[0], f"the slash was not encoded: {session.calls[0]}"

    session.responses = {"pug/compound/name": _FakeResponse({"PropertyTable": {"Properties": [{"InChIKey": MW_KEY}]}})}
    resolver._fetch_pubchem_inchikey("PC 16:0/18:1")
    assert "%2F" in session.calls[1], f"the slash was not encoded: {session.calls[1]}"
