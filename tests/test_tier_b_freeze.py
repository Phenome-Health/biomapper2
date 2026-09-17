"""Tier B freeze-first behaviour and the on-by-default contract.

Everything here runs against injectable fakes: a recording session, a forced-open breaker. Nothing
touches Metabolomics Workbench, PubChem or Kestrel. These tests pin the properties that make Tier B
SAFE to ship on by default: the freeze is consulted first (a hit costs no network), a freeze miss is
loud and falls back to the live path behind a circuit breaker, and an open breaker degrades to
``lookup_failed`` with no call.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pytest

from biomapper2.core import tier_b_snapshot
from biomapper2.core.certificate import TierBOutcome
from biomapper2.core.tier_b import IndependentStructureLookup

FIXTURE = Path(__file__).parent / "fixtures" / "tier_b_freeze_fixture.tsv"
MW_KEY = "BSYNRYMUTXBXSQ-UHFFFAOYSA-N"


class _FakeResponse:
    def __init__(self, payload: Any, status: int = 200, from_cache: bool = False) -> None:
        self._payload = payload
        self.status_code = status
        self.from_cache = from_cache

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self) -> Any:
        return self._payload


class _RecordingSession:
    """Records every URL requested so a test can assert the freeze path made no call."""

    def __init__(self, responses: dict[str, Any]) -> None:
        self.responses = responses
        self.calls: list[str] = []

    def get(self, url: str, timeout: float | None = None) -> _FakeResponse:  # noqa: ARG002
        self.calls.append(url)
        for fragment, response in self.responses.items():
            if fragment in url:
                return response
        return _FakeResponse({}, status=404)


class _OpenBreaker:
    """A breaker that is always open: allow() is False and it never trips on its own."""

    def allow(self) -> bool:
        return False

    def record_success(self) -> None:
        pass

    def record_failure(self) -> None:
        pass


def _lookup(responses: dict[str, Any], **kwargs) -> tuple[IndependentStructureLookup, _RecordingSession]:
    session = _RecordingSession(responses)
    return (
        IndependentStructureLookup(session=session, sleep=lambda _s: None, clock=lambda: 0.0, **kwargs),
        session,
    )


@pytest.fixture(autouse=True)
def _reset_snapshot_cache():
    tier_b_snapshot.reset()
    yield
    tier_b_snapshot.reset()


@pytest.fixture
def _freeze(monkeypatch):
    monkeypatch.setenv("BIOMAPPER2_TIER_B_SNAPSHOT_PATH", str(FIXTURE))
    tier_b_snapshot.reset()


def test_freeze_hit_resolves_with_no_network_call(_freeze) -> None:
    lookup, session = _lookup({"refmet/name": _FakeResponse({"inchi_key": MW_KEY})})
    result = lookup.lookup("glucose")
    assert result.outcome is TierBOutcome.RESOLVED
    assert result.inchikey_block == "WQZGKKKJIJFFOK-GASJEMHNSA-N"
    assert result.source == "pubchem"
    assert result.cache_state == "frozen"
    assert session.calls == [], "a freeze hit must never touch the network"


def test_freeze_miss_is_loud_and_falls_back_to_live(_freeze, caplog) -> None:
    lookup, session = _lookup({"refmet/name": _FakeResponse({"inchi_key": MW_KEY})})
    with caplog.at_level(logging.WARNING):
        result = lookup.lookup("a name not in the freeze")
    assert result.outcome is TierBOutcome.RESOLVED
    assert result.source == "metabolomics-workbench"
    assert session.calls, "a freeze miss must fall back to the live path"
    assert any("freeze miss" in rec.message.lower() for rec in caplog.records), "a freeze miss must log loudly"


def test_breaker_open_degrades_to_lookup_failed_with_no_call() -> None:
    # No freeze configured, so every name is a live-path candidate; the open breaker must degrade it
    # to lookup_failed WITHOUT any network call rather than crashing the mapping loop.
    lookup, session = _lookup({"refmet/name": _FakeResponse({"inchi_key": MW_KEY})}, breaker=_OpenBreaker())
    result = lookup.lookup("glucose")
    assert result.outcome is TierBOutcome.LOOKUP_FAILED
    assert session.calls == [], "an open breaker must make no network call"


def test_three_state_resolution_matrix(monkeypatch) -> None:
    """The default posture is COUPLED to freeze presence, so a fresh deploy never silently hits live
    services: unset + freeze -> enabled_freeze, unset + no freeze -> inert, explicit truthy + no
    freeze -> enabled_live (the supervised-sweep path), falsy -> disabled regardless of freeze."""
    from biomapper2 import config

    monkeypatch.delenv("BIOMAPPER2_TIER_B_ENABLED", raising=False)
    assert config.resolve_tier_b_state(snapshot_present=True) == config.TIER_B_STATE_ENABLED_FREEZE
    assert config.resolve_tier_b_state(snapshot_present=False) == config.TIER_B_STATE_INERT

    monkeypatch.setenv("BIOMAPPER2_TIER_B_ENABLED", "true")
    assert config.resolve_tier_b_state(snapshot_present=False) == config.TIER_B_STATE_ENABLED_LIVE
    assert config.resolve_tier_b_state(snapshot_present=True) == config.TIER_B_STATE_ENABLED_FREEZE

    monkeypatch.setenv("BIOMAPPER2_TIER_B_ENABLED", "false")
    assert config.resolve_tier_b_state(snapshot_present=True) == config.TIER_B_STATE_DISABLED
    assert config.resolve_tier_b_state(snapshot_present=False) == config.TIER_B_STATE_DISABLED


def test_freeze_version_is_a_first_class_certificate_field(_freeze) -> None:
    """Frozen independent evidence must be auditable on the same footing as RefMet's: the freeze
    version is a FIRST-CLASS certificate field and a flat top-level column (not only provenance),
    populated on a freeze hit and None on a live result."""
    from biomapper2.core.certificate import TierBOutcome, TierBResult, issue

    def _cert(tier_b):
        return issue(
            chosen_kg_id="CHEBI:4167",
            is_small_molecule=True,
            kg_equivalent_ids={"INCHIKEY": ["WQZGKKKJIJFFOK-GASJEMHNSA-N"]},
            equivalent_ids_lookup_ok=True,
            tier_b=tier_b,
            tier_b_enabled=True,
        )

    lookup, _ = _lookup({})  # no responses needed; glucose is a freeze hit
    frozen = lookup.lookup("glucose")
    assert frozen.version == "tier-b-fixture-v1"

    frozen_cert = _cert(frozen)
    assert frozen_cert.tier_b_snapshot_version == "tier-b-fixture-v1"
    assert frozen_cert.to_flat_columns()["certificate_tier_b_snapshot_version"] == "tier-b-fixture-v1"
    assert frozen_cert.to_api_dict()["tier_b_snapshot_version"] == "tier-b-fixture-v1"
    # RefMet parity: the version is a first-class field, NOT carried in provenance.
    assert "tier_b_snapshot_version" not in frozen_cert.provenance

    # A live (non-frozen) result carries no freeze version.
    live = TierBResult(source="pubchem", inchikey_block="WQZGKKKJIJFFOK", outcome=TierBOutcome.RESOLVED)
    live_cert = _cert(live)
    assert live_cert.tier_b_snapshot_version is None
    assert live_cert.to_flat_columns()["certificate_tier_b_snapshot_version"] is None


def test_non_small_molecule_stays_out_of_scope_under_an_enabled_run() -> None:
    """Scope is the first safety property: an enabled run never looks Tier B up for a
    non-small-molecule row, and the certificate records out_of_scope rather than a live verdict."""
    from biomapper2.core.certificate import CertificateState, issue

    certificate = issue(
        chosen_kg_id="HGNC:1",
        is_small_molecule=False,
        kg_equivalent_ids=None,
        equivalent_ids_lookup_ok=True,
        tier_b=None,
        tier_b_enabled=True,
    )
    assert certificate.tier_b_outcome is TierBOutcome.OUT_OF_SCOPE
    assert certificate.state is CertificateState.NOT_APPLICABLE
