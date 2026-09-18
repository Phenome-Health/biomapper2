"""Mode ``live_backup`` control flow (core/annotators/metabolomics_workbench.py).

Fixture + monkeypatch only; NEVER a live call. Proves the write-through design:
- live VOTED -> source=live_api, written back to the store when the stored value differs (D7);
- live VOTED identical to the store -> NO write-back (no write amplification);
- live NO_MATCH -> written back as a negative cache (D6);
- live DOWN (breaker open) + store hit -> source=freeze_backup (the flip-killing backup, D3);
- live DOWN + store miss -> unavailable;
- modes ``off`` / ``frozen`` still dispatch to the untouched live / snapshot paths.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from circuitbreaker import CircuitBreakerError

from biomapper2.core.annotators import refmet_snapshot, refmet_store
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "refmet_freeze_fixture.tsv"


def _no_network(*_args, **_kwargs):
    raise AssertionError("network was called where it must not be")


def _breaker_open(*_args, **_kwargs):
    raise CircuitBreakerError("simulated open breaker")


@pytest.fixture(autouse=True)
def _reset_snapshot_cache():
    refmet_snapshot.reset()
    yield
    refmet_snapshot.reset()


@pytest.fixture
def store_db(monkeypatch, tmp_path: Path) -> Path:
    db = tmp_path / "store.sqlite"
    monkeypatch.setenv("REFMET_STORE_PATH", str(db))
    # No snapshot configured: __init__ seeding is a no-op, so each test controls store contents.
    monkeypatch.delenv("REFMET_SNAPSHOT_PATH", raising=False)
    return db


def test_live_voted_writes_back_when_different(store_db: Path):
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    ann._request_once = lambda metabolite_name: {"refmet_id": "RM9999"}
    result = ann._fetch_refmet_data("novel metabolite")
    assert result.status == "voted"
    assert result.source == "live_api"
    # Written back so a later outage can serve it.
    hit = refmet_store.get("novel metabolite")
    assert hit is not None and hit.status == "voted" and hit.refmet_id == "RM9999"


def test_live_voted_identical_does_not_write_back(store_db: Path, monkeypatch):
    refmet_store.upsert("caffeine", "voted", refmet_id="RM9999")  # pre-seed identical value
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    ann._request_once = lambda metabolite_name: {"refmet_id": "RM9999"}

    calls: list[str] = []
    monkeypatch.setattr(refmet_store, "upsert", lambda *a, **k: calls.append("upsert"))
    result = ann._fetch_refmet_data("caffeine")
    assert result.status == "voted" and result.source == "live_api"
    assert calls == []  # identical (status, refmet_id) -> no write


def test_live_no_match_is_written_back_as_negative_cache(store_db: Path):
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    ann._request_once = lambda metabolite_name: None  # /match "-" -> genuine no-match
    result = ann._fetch_refmet_data("not a metabolite")
    assert result.status == "no_match" and result.source == "live_api"
    hit = refmet_store.get("not a metabolite")
    assert hit is not None and hit.status == "no_match" and hit.refmet_id is None


def test_breaker_open_serves_store_backup(store_db: Path):
    # Positive control: retinol seeded, live down -> backup serves the correct id (flip killed).
    refmet_store.upsert("retinol", "voted", refmet_id="CHEBI:12777")
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    ann._do_refmet_request = _breaker_open
    result = ann._fetch_refmet_data("retinol")
    assert result.status == "voted"
    assert result.source == "freeze_backup"
    assert result.data == {"refmet_id": "CHEBI:12777"}


def test_breaker_open_with_store_miss_is_unavailable(store_db: Path):
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    ann._do_refmet_request = _breaker_open
    result = ann._fetch_refmet_data("never seen")
    assert result.status == "unavailable"
    assert result.source == "unavailable"
    assert result.data is None


def test_mode_off_uses_live_and_never_touches_store(store_db: Path, monkeypatch):
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="off")
    ann._request_once = lambda metabolite_name: {"refmet_id": "RM1"}
    calls: list[str] = []
    monkeypatch.setattr(refmet_store, "upsert", lambda *a, **k: calls.append("upsert"))
    result = ann._fetch_refmet_data("anything")
    assert result.status == "voted" and result.source == "live_api"
    assert calls == []  # off mode has no write-through
    assert refmet_store.get("anything") is None


def test_mode_frozen_uses_snapshot(monkeypatch, store_db: Path):
    monkeypatch.setenv("REFMET_SNAPSHOT_PATH", str(FIXTURE))
    refmet_snapshot.reset()
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="frozen")
    ann._do_refmet_request = _no_network  # frozen keeps the breaker out of the path
    result = ann._fetch_refmet_data("Cholic acid")
    assert result.status == "voted"
    assert result.source == "local_snapshot"
    assert result.data == {"refmet_id": "RM0041813"}


def test_store_failure_is_best_effort_never_aborts(store_db: Path, monkeypatch):
    # The backup store is best-effort: an unwritable/unreadable store must NEVER abort a valid live
    # resolution (write path) nor raise on the outage path (read path) — it degrades, not fails.
    def _boom(*_a, **_k):
        raise sqlite3.OperationalError("simulated unwritable/locked store")

    monkeypatch.setattr(refmet_store, "get", _boom)
    monkeypatch.setattr(refmet_store, "upsert", _boom)
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")

    ann._request_once = lambda metabolite_name: {"refmet_id": "RM1"}
    live_ok = ann._fetch_refmet_data("x")  # live succeeds; store write blows up -> still VOTED
    assert live_ok.status == "voted" and live_ok.source == "live_api"

    ann._request_once = _breaker_open  # live down; store read blows up -> unavailable, no exception
    down = ann._fetch_refmet_data("y")
    assert down.status == "unavailable"
