"""Mode ``live_backup`` control flow (core/annotators/metabolomics_workbench.py).

Fixture + monkeypatch only; NEVER a live call. Proves the write-through design:
- live VOTED -> source=live_api, written back to the store when the stored value differs (D7);
- live VOTED identical to the store -> NO write-back (no write amplification);
- live NO_MATCH with no stored vote -> written back as a negative cache (D6);
- live NO_MATCH with a stored vote -> the vote is kept, source=kept_live_no_match, logged once;
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


# --------------------------------------------------------------------------------------------
# Keep the last-known-good vote against a live no-match
# --------------------------------------------------------------------------------------------

DHA = "docosahexaenoate (dha; 22:6n3)"


def _annotator(live: dict | None) -> MetabolomicsWorkbenchAnnotator:
    """A live_backup annotator whose live /match answers ``live`` (None = a genuine no-match)."""
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    ann._request_once = lambda metabolite_name: live
    return ann


def _kinds() -> list[str]:
    return [c.kind for c in refmet_store.list_changes()]


def test_live_no_match_keeps_the_stored_vote(store_db: Path, caplog):
    refmet_store.upsert(DHA, "voted", refmet_id="RM0153631", snapshot_version="mw_match_fixture")
    ann = _annotator(None)

    with caplog.at_level("WARNING"):
        result = ann._fetch_refmet_data(DHA)

    assert result.status == "voted"
    assert result.source == "kept_live_no_match"
    assert result.data == {"refmet_id": "RM0153631"}
    assert result.version == "mw_match_fixture"
    hit = refmet_store.get(DHA)
    assert hit is not None and (hit.status, hit.refmet_id) == ("voted", "RM0153631")
    assert _kinds() == ["vote_kept"]
    assert sum("kept" in r.getMessage() and DHA in r.getMessage() for r in caplog.records) == 1

    # The same request again: same result, no second record, no second warning.
    caplog.clear()
    with caplog.at_level("WARNING"):
        again = ann._fetch_refmet_data(DHA)
    assert (again.status, again.source, again.data) == ("voted", "kept_live_no_match", {"refmet_id": "RM0153631"})
    assert _kinds() == ["vote_kept"]
    assert not any(DHA in r.getMessage() for r in caplog.records)


def test_repeat_kept_vote_makes_no_store_write(store_db: Path, monkeypatch):
    refmet_store.upsert(DHA, "voted", refmet_id="RM0153631")
    ann = _annotator(None)
    ann._fetch_refmet_data(DHA)

    calls: list[str] = []
    monkeypatch.setattr(refmet_store, "apply_live_answer", lambda name, status, **k: calls.append(name))
    monkeypatch.setattr(refmet_store, "upsert", lambda *a, **k: calls.append("upsert"))
    assert ann._fetch_refmet_data(DHA).source == "kept_live_no_match"
    assert calls == []


def test_live_match_after_a_kept_vote_is_recorded_as_recovered(store_db: Path):
    refmet_store.upsert(DHA, "voted", refmet_id="RM0153631")
    _annotator(None)._fetch_refmet_data(DHA)

    result = _annotator({"refmet_id": "RM0153631"})._fetch_refmet_data(DHA)

    assert (result.status, result.source, result.data) == ("voted", "live_api", {"refmet_id": "RM0153631"})
    assert _kinds() == ["live_recovered", "vote_kept"]


def test_live_match_with_a_different_id_wins(store_db: Path):
    refmet_store.upsert(DHA, "voted", refmet_id="RM0153631")
    result = _annotator({"refmet_id": "RM9"})._fetch_refmet_data(DHA)
    assert (result.status, result.source, result.data) == ("voted", "live_api", {"refmet_id": "RM9"})
    hit = refmet_store.get(DHA)
    assert hit is not None and (hit.status, hit.refmet_id) == ("voted", "RM9")
    assert _kinds() == ["id_changed"]


def test_live_no_match_without_a_stored_vote_logs_nothing(store_db: Path):
    result = _annotator(None)._fetch_refmet_data("not a metabolite")
    assert (result.status, result.source) == ("no_match", "live_api")
    hit = refmet_store.get("not a metabolite")
    assert hit is not None and hit.status == "no_match"
    assert _kinds() == []


def test_live_match_for_a_stored_no_match_is_recorded(store_db: Path):
    refmet_store.upsert("x", "no_match")
    result = _annotator({"refmet_id": "RM2"})._fetch_refmet_data("x")
    assert (result.status, result.source) == ("voted", "live_api")
    hit = refmet_store.get("x")
    assert hit is not None and (hit.status, hit.refmet_id) == ("voted", "RM2")
    assert _kinds() == ["match_appeared"]


def test_live_match_without_an_id_leaves_the_store_alone(store_db: Path):
    refmet_store.upsert(DHA, "voted", refmet_id="RM0153631")
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    ann._request_once = lambda metabolite_name: {"refmet_name": "DHA"}  # a match that carries no id
    result = ann._fetch_refmet_data(DHA)
    assert (result.status, result.source) == ("voted", "live_api")
    hit = refmet_store.get(DHA)
    assert hit is not None and hit.refmet_id == "RM0153631" and hit.live_status is None
    assert _kinds() == []


def test_live_unreachable_with_a_stored_vote_is_still_freeze_backup(store_db: Path):
    refmet_store.upsert(DHA, "voted", refmet_id="RM0153631")
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    ann._do_refmet_request = _breaker_open
    result = ann._fetch_refmet_data(DHA)
    assert (result.status, result.source) == ("voted", "freeze_backup")
    assert _kinds() == []


def test_live_no_match_with_an_unreadable_store_returns_the_no_match_and_writes_nothing(store_db: Path, monkeypatch):
    refmet_store.upsert(DHA, "voted", refmet_id="RM0153631")

    def _boom(name, *, path=None):
        raise sqlite3.OperationalError("simulated unreadable store")

    writes: list[str] = []
    monkeypatch.setattr(refmet_store, "get", _boom)
    monkeypatch.setattr(refmet_store, "apply_live_answer", lambda name, status, **k: writes.append(name))
    monkeypatch.setattr(refmet_store, "upsert", lambda *a, **k: writes.append("upsert"))
    result = _annotator(None)._fetch_refmet_data(DHA)
    assert (result.status, result.source) == ("no_match", "live_api")
    assert writes == []  # without the row it cannot know whether a vote exists; never overwrite blind


def test_kept_vote_is_returned_when_recording_the_observation_fails(store_db: Path, monkeypatch):
    refmet_store.upsert(DHA, "voted", refmet_id="RM0153631")

    def _boom(name, status, *, refmet_id=None, path=None):
        raise sqlite3.OperationalError("simulated locked store")

    monkeypatch.setattr(refmet_store, "apply_live_answer", _boom)
    result = _annotator(None)._fetch_refmet_data(DHA)
    assert (result.status, result.source, result.data) == ("voted", "kept_live_no_match", {"refmet_id": "RM0153631"})


def test_a_case_twin_row_is_not_used_to_keep_a_vote(store_db: Path):
    refmet_store.upsert(DHA.upper(), "voted", refmet_id="RM0153631")
    result = _annotator(None)._fetch_refmet_data(DHA)
    assert (result.status, result.source) == ("no_match", "live_api")
    assert _kinds() == []


def test_deleting_the_store_row_clears_a_kept_vote(store_db: Path):
    refmet_store.upsert(DHA, "voted", refmet_id="RM0153631")
    ann = _annotator(None)
    assert ann._fetch_refmet_data(DHA).source == "kept_live_no_match"

    conn = sqlite3.connect(str(store_db))
    conn.execute("DELETE FROM refmet_by_query WHERE query = ?", (DHA,))
    conn.commit()
    conn.close()

    result = ann._fetch_refmet_data(DHA)
    assert (result.status, result.source) == ("no_match", "live_api")
    hit = refmet_store.get(DHA)
    assert hit is not None and hit.status == "no_match"
    assert _kinds() == ["vote_kept"]  # only the record from before the delete


@pytest.mark.parametrize("stored_status", ["unavailable", "voted"])
def test_a_row_without_a_last_known_good_is_treated_as_no_row(store_db: Path, stored_status: str):
    refmet_store.upsert("x", stored_status, refmet_id=None)
    result = _annotator(None)._fetch_refmet_data("x")
    assert (result.status, result.source) == ("no_match", "live_api")
    hit = refmet_store.get("x")
    assert hit is not None and hit.status == "no_match"
    assert _kinds() == []


def test_store_error_during_construction_does_not_fail_startup(store_db: Path, monkeypatch, caplog):
    monkeypatch.setenv("REFMET_SNAPSHOT_PATH", str(FIXTURE))

    def _boom(tsv_path, *, path=None):
        raise sqlite3.OperationalError("simulated migration failure")

    monkeypatch.setattr(refmet_store, "seed_from_tsv", _boom)
    with caplog.at_level("WARNING"):
        ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    assert ann.freeze_mode == "live_backup"
    assert any("seed" in r.getMessage().lower() for r in caplog.records)


def test_second_identical_live_match_makes_no_store_write(store_db: Path, monkeypatch):
    ann = _annotator({"refmet_id": "RM9999"})
    ann._fetch_refmet_data("novel metabolite")  # first match is written back

    calls: list[str] = []
    monkeypatch.setattr(refmet_store, "apply_live_answer", lambda name, status, **k: calls.append(name))
    monkeypatch.setattr(refmet_store, "upsert", lambda *a, **k: calls.append("upsert"))
    result = ann._fetch_refmet_data("novel metabolite")
    assert (result.status, result.source) == ("voted", "live_api")
    assert calls == []


def test_kept_vote_on_the_canonical_spelling_skips_the_raw_spelling(store_db: Path):
    # Scope boundary: canonical-first order is unchanged. A kept vote is `voted`, so the raw spelling
    # is not queried, exactly as before the drift.
    refmet_store.upsert("acetylcarnitine (C2)", "voted", refmet_id="RM0154009")
    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    asked: list[str] = []

    def _live(metabolite_name: str):
        asked.append(metabolite_name)
        return None

    ann._request_once = _live
    result = ann._fetch_refmet_data("acetylcarnitine (c2)")
    assert (result.status, result.source) == ("voted", "kept_live_no_match")
    assert asked == ["acetylcarnitine (C2)"]
