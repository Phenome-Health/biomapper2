"""Offline/caching guarantees for BiolinkClient — the fix for the flaky offline gate.

bmt.Toolkit downloads the model schema (when given a URL) and the predicate map on every
construction. The offline test gate builds a fresh ``Normalizer()``/``Mapper()`` per case, so those
downloads ran hundreds of times and flaked on transient GitHub-raw resets ("Connection reset by
peer"). BiolinkClient now (1) disk-caches the schema and (2) builds the Toolkit once per process.
These tests pin both, plus the download retry that absorbs a single transient failure.

Network is only touched to warm the process-wide cache once (the single build the whole suite
shares); every assertion below runs against the cached, offline path.
"""

from __future__ import annotations

import pytest
import requests

import biomapper2.biolink_client as bc
from biomapper2.biolink_client import BiolinkClient, _get_with_retry


@pytest.fixture(autouse=True)
def _warm_toolkit():
    # Build once (may fetch) so the offline assertions exercise the cached path, not a first-ever
    # download. In a full run other tests have already warmed it, so this reuses the cache.
    BiolinkClient()


def test_toolkit_is_shared_across_clients():
    # The read-only Toolkit is cached process-wide, so separate clients reuse one instance instead
    # of each re-downloading the model.
    assert BiolinkClient().bmt is BiolinkClient().bmt


def test_construction_is_offline_once_warmed(monkeypatch):
    # With the schema on disk and the Toolkit cached, constructing a client must not touch the
    # network. A hit here is the exact regression this fix exists to prevent.
    def _boom(*args, **kwargs):
        raise AssertionError("network was hit during offline BiolinkClient construction")

    monkeypatch.setattr(requests, "get", _boom)
    client = BiolinkClient()
    assert client.standardize_entity_type("metabolite") == "biolink:SmallMolecule"


def test_get_with_retry_recovers_after_transient_errors(monkeypatch):
    monkeypatch.setattr(bc.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    class _Resp:
        text = "ok"

        def raise_for_status(self):
            return None

    def _flaky(url, timeout=60):
        calls["n"] += 1
        if calls["n"] < 3:
            raise requests.exceptions.ConnectionError("simulated reset")
        return _Resp()

    monkeypatch.setattr(requests, "get", _flaky)
    assert _get_with_retry("http://example/x").text == "ok"
    assert calls["n"] == 3


def test_get_with_retry_raises_when_every_attempt_fails(monkeypatch):
    # Positive control that the guard can fail: if every attempt errors, the last error propagates
    # rather than being swallowed.
    monkeypatch.setattr(bc.time, "sleep", lambda *_: None)

    def _always_fail(url, timeout=60):
        raise requests.exceptions.ConnectionError("simulated reset")

    monkeypatch.setattr(requests, "get", _always_fail)
    with pytest.raises(requests.exceptions.ConnectionError):
        _get_with_retry("http://example/x")


def test_atomic_write_publishes_full_content(tmp_path):
    target = tmp_path / "schema.yaml"
    bc._atomic_write_text(str(target), "complete")
    assert target.read_text() == "complete"
    assert not list(tmp_path.glob("*.tmp"))  # no temp file left behind


def test_atomic_write_leaves_no_partial_on_failure(tmp_path, monkeypatch):
    # If the rename is interrupted, the target must never appear as a partial file (which the cache
    # would then trust forever); only the temp is touched, and it is cleaned up.
    target = tmp_path / "schema.yaml"

    def _fail_replace(src, dst):
        raise OSError("simulated interruption before rename")

    monkeypatch.setattr(bc.os, "replace", _fail_replace)
    with pytest.raises(OSError):
        bc._atomic_write_text(str(target), "partial content")
    assert not target.exists()
    assert not list(tmp_path.glob("*.tmp"))
