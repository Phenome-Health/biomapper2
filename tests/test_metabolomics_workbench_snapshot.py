"""Snapshot rollout for the RefMet annotator (core/annotators/metabolomics_workbench.py).

Fixture + monkeypatch only; NEVER a live call. Proves the safe rollout:
- snapshot PRESENT: freeze served first (voted/no_match -> local_snapshot), a miss is a deterministic
  not_in_snapshot NO_MATCH with NO network, and an OPEN breaker does not touch the default path;
- snapshot ABSENT: the live path is used, exactly as before (source live_api);
- live_api_fallback=True: a freeze miss falls through to the live path.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from circuitbreaker import CircuitBreakerError

from biomapper2.core.annotators import refmet_snapshot
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "refmet_freeze_fixture.tsv"


def _no_network(*_args, **_kwargs):
    raise AssertionError("network was called on the deterministic snapshot path")


@pytest.fixture(autouse=True)
def _reset_snapshot_cache():
    refmet_snapshot.reset()
    yield
    refmet_snapshot.reset()


@pytest.fixture
def snapshot_present(monkeypatch):
    monkeypatch.setenv("REFMET_SNAPSHOT_PATH", str(FIXTURE))
    refmet_snapshot.reset()


@pytest.fixture
def snapshot_absent(monkeypatch):
    monkeypatch.delenv("REFMET_SNAPSHOT_PATH", raising=False)
    refmet_snapshot.reset()


def test_voted_hit_from_snapshot(snapshot_present):
    ann = MetabolomicsWorkbenchAnnotator()
    ann._do_refmet_request = _no_network  # any network use would be a bug on this path
    result = ann._fetch_refmet_data("Cholic acid")
    assert result.status == "voted"
    assert result.source == "local_snapshot"
    assert result.version == "fixture-v1"
    assert result.data == {"refmet_id": "RM0041813"}
    # And the emitted vote uses the frozen refmet_id.
    annotations = ann.get_annotations({"name": "Cholic acid"}, "name", "biolink:SmallMolecule")
    assert annotations == {ann.slug: {"refmet_id": {"RM0041813": {}}}}


def test_no_match_from_snapshot(snapshot_present):
    ann = MetabolomicsWorkbenchAnnotator()
    ann._do_refmet_request = _no_network
    result = ann._fetch_refmet_data("definitely not a real metabolite")
    assert result.status == "no_match"
    assert result.source == "local_snapshot"
    assert ann.get_annotations({"name": "definitely not a real metabolite"}, "name", "biolink:SmallMolecule") == {
        ann.slug: {}
    }


def test_miss_is_not_in_snapshot_with_no_network(snapshot_present):
    ann = MetabolomicsWorkbenchAnnotator()  # live_api_fallback defaults OFF
    ann._request_once = _no_network
    ann._do_refmet_request = _no_network
    result = ann._fetch_refmet_data("a name absent from the freeze")
    assert result.status == "no_match"
    assert result.source == "not_in_snapshot"
    assert result.version == "fixture-v1"
    assert ann.get_source({"name": "a name absent from the freeze"}, "name") == {ann.slug: "not_in_snapshot"}


def test_open_breaker_does_not_affect_the_snapshot_path(snapshot_present):
    ann = MetabolomicsWorkbenchAnnotator()

    def _breaker_open(*_a, **_k):
        raise CircuitBreakerError("simulated open breaker")

    ann._do_refmet_request = _breaker_open
    # A voted hit still resolves from the freeze; a miss is still deterministic — the breaker is out.
    assert ann._fetch_refmet_data("Cholic acid").source == "local_snapshot"
    assert ann._fetch_refmet_data("a name absent from the freeze").source == "not_in_snapshot"


def test_snapshot_absent_uses_live_path(snapshot_absent):
    ann = MetabolomicsWorkbenchAnnotator()
    ann._request_once = lambda metabolite_name: {"refmet_id": "RM9999"}
    result = ann._fetch_refmet_data("anything")
    assert result.status == "voted"
    assert result.source == "live_api"
    assert result.version is None


def test_live_api_fallback_on_miss_uses_live(snapshot_present):
    ann = MetabolomicsWorkbenchAnnotator(live_api_fallback=True)
    ann._request_once = lambda metabolite_name: {"refmet_id": "RM9999"}
    result = ann._fetch_refmet_data("a name absent from the freeze")
    assert result.status == "voted"
    assert result.source == "live_api"
    assert result.data == {"refmet_id": "RM9999"}
