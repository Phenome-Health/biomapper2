"""Write-through RefMet store (core/annotators/refmet_store.py).

SQLite temp DB only; NO network. Proves the store primitives the ``live_backup`` mode relies on:
upsert/get, seed-from-freeze (idempotent), update-on-conflict (one row, newest wins), exact-query
keying (case variants never cross-serve; the legacy folded-key table is never read),
thread-concurrent upserts (no corruption, last-writer-wins), and config path discovery.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import pytest

from biomapper2.core.annotators import refmet_store

pytestmark = pytest.mark.unit

FIXTURE = Path(__file__).parent / "fixtures" / "refmet_freeze_fixture.tsv"


@pytest.fixture
def db(tmp_path: Path) -> Path:
    return tmp_path / "refmet_store.sqlite"


def _row_count(db: Path, query: str) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("SELECT COUNT(*) FROM refmet_by_query WHERE query = ?", (query,)).fetchone()[0]
    finally:
        conn.close()


def test_upsert_then_get(db: Path):
    refmet_store.upsert("Cholic acid", "voted", refmet_id="RM0041813", path=db)
    hit = refmet_store.get("Cholic acid", path=db)
    assert hit is not None
    assert hit.status == "voted"
    assert hit.refmet_id == "RM0041813"
    assert hit.updated_at is not None
    # A genuine miss is None (never a fabricated row).
    assert refmet_store.get("some other name", path=db) is None


def test_seed_from_tsv_is_bulk_and_idempotent(db: Path):
    inserted = refmet_store.seed_from_tsv(FIXTURE, path=db)
    assert inserted == 2  # the fixture carries one voted + one no_match row
    assert refmet_store.get("Cholic acid", path=db).refmet_id == "RM0041813"  # type: ignore[union-attr]
    assert refmet_store.get("definitely not a real metabolite", path=db).status == "no_match"  # type: ignore[union-attr]
    # Second seed is a no-op because the store is no longer empty.
    assert refmet_store.seed_from_tsv(FIXTURE, path=db) == 0


def test_update_on_conflict_keeps_one_row_newest_wins(db: Path):
    refmet_store.upsert("retinol", "voted", refmet_id="CHEBI:12777", path=db)
    refmet_store.upsert("retinol", "voted", refmet_id="CHEBI:999", path=db)
    hit = refmet_store.get("retinol", path=db)
    assert hit is not None and hit.refmet_id == "CHEBI:999"
    assert _row_count(db, "retinol") == 1


def test_rows_are_keyed_by_the_exact_query(db: Path):
    # RefMet /match is case-sensitive, so a stored answer belongs to the exact string that was sent.
    refmet_store.upsert("Cholic  Acid", "voted", refmet_id="RM0041813", path=db)
    assert refmet_store.get("Cholic  Acid", path=db).refmet_id == "RM0041813"  # type: ignore[union-attr]
    assert refmet_store.get("cholic acid", path=db) is None
    assert refmet_store.get("Cholic Acid", path=db) is None
    assert _row_count(db, "Cholic  Acid") == 1


def test_case_twins_never_cross_serve(db: Path):
    # Recorded live 2026-11-04: the Metabolon spelling votes CAR 2:0, the lower-cased NECS spelling is a
    # RefMet no-match. Under the old case-folded key these shared one last-writer-wins row, so in an
    # outage one spelling was served the other's answer. Each spelling now keeps its own row.
    refmet_store.upsert("acetylcarnitine (C2)", "voted", refmet_id="RM0154009", path=db)
    refmet_store.upsert("acetylcarnitine (c2)", "no_match", path=db)
    upper = refmet_store.get("acetylcarnitine (C2)", path=db)
    lower = refmet_store.get("acetylcarnitine (c2)", path=db)
    assert upper is not None and (upper.status, upper.refmet_id) == ("voted", "RM0154009")
    assert lower is not None and (lower.status, lower.refmet_id) == ("no_match", None)


def test_legacy_case_folded_rows_are_never_served_and_never_dropped(db: Path):
    # A store written by the previous release: table `refmet`, keyed on the folded name. Its key cannot
    # say which spelling wrote the row, so it must not be served for either spelling; it is kept so a
    # rollback still finds it.
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE refmet (name_norm TEXT PRIMARY KEY, refmet_id TEXT, refmet_name TEXT, inchi_key TEXT, "
        "formula TEXT, exactmass TEXT, status TEXT NOT NULL, source TEXT, updated_at TEXT, snapshot_version TEXT)"
    )
    conn.execute(
        "INSERT INTO refmet (name_norm, refmet_id, status) VALUES (?, ?, ?)",
        ("acetylcarnitine (c2)", "RM0154009", "voted"),
    )
    conn.commit()
    conn.close()

    assert refmet_store.get("acetylcarnitine (c2)", path=db) is None
    assert refmet_store.get("acetylcarnitine (C2)", path=db) is None
    # The new table starts empty, so the freeze seed still runs on an upgraded store.
    assert refmet_store.seed_from_tsv(FIXTURE, path=db) == 2
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute("SELECT COUNT(*) FROM refmet").fetchone()[0] == 1
    finally:
        conn.close()


def test_seed_keys_rows_by_the_exact_freeze_query_name(db: Path):
    refmet_store.seed_from_tsv(FIXTURE, path=db)
    assert refmet_store.get("Cholic acid", path=db) is not None
    assert refmet_store.get("cholic acid", path=db) is None


def test_concurrent_upserts_no_corruption(db: Path):
    refmet_store.upsert("caffeine", "voted", refmet_id="RM0", path=db)  # create the WAL db first

    def worker(i: int) -> None:
        refmet_store.upsert("caffeine", "voted", refmet_id=f"RM{i}", path=db)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    hit = refmet_store.get("caffeine", path=db)
    assert hit is not None and hit.status == "voted"
    assert hit.refmet_id in {f"RM{i}" for i in range(16)}  # last writer won, value intact
    assert _row_count(db, "caffeine") == 1  # no duplicate / corruption


def test_config_path_discovery(monkeypatch, db: Path):
    monkeypatch.delenv("REFMET_STORE_PATH", raising=False)
    assert refmet_store.is_configured() is False
    assert refmet_store.get("caffeine") is None  # no store -> miss, no write

    monkeypatch.setenv("REFMET_STORE_PATH", str(db))
    assert refmet_store.is_configured() is True
    # With no explicit path the primitives resolve REFMET_STORE_PATH.
    refmet_store.upsert("caffeine", "voted", refmet_id="RM7")
    hit = refmet_store.get("caffeine")
    assert hit is not None and hit.refmet_id == "RM7"


# --------------------------------------------------------------------------------------------
# Last observed live answer + change log (the kept-vote change)
# --------------------------------------------------------------------------------------------

_LEGACY_CREATE = (
    "CREATE TABLE refmet_by_query (query TEXT PRIMARY KEY, refmet_id TEXT, refmet_name TEXT, inchi_key TEXT, "
    "formula TEXT, exactmass TEXT, status TEXT NOT NULL, source TEXT, updated_at TEXT, snapshot_version TEXT)"
)
# The exact statement the previous release's ``get`` runs: it must keep working on a migrated file.
_PREVIOUS_RELEASE_SELECT = (
    "SELECT status, refmet_id, refmet_name, inchi_key, formula, exactmass, source, "
    "updated_at, snapshot_version FROM refmet_by_query WHERE query = ?"
)


def _legacy_store(db: Path) -> None:
    """A store file as the previous release created it: no live columns, no change log."""
    conn = sqlite3.connect(str(db))
    conn.execute("PRAGMA journal_mode=WAL")  # the previous release always opened its store in WAL mode
    conn.execute(_LEGACY_CREATE)
    conn.execute(
        "INSERT INTO refmet_by_query (query, refmet_id, status, source, updated_at) VALUES (?, ?, ?, ?, ?)",
        ("dha", "RM0153631", "voted", "live_api", "2026-11-06T00:00:00+00:00"),
    )
    conn.commit()
    conn.close()


def _columns(db: Path) -> list[str]:
    conn = sqlite3.connect(str(db))
    try:
        return [row[1] for row in conn.execute("PRAGMA table_info(refmet_by_query)")]
    finally:
        conn.close()


def _tables(db: Path) -> set[str]:
    conn = sqlite3.connect(str(db))
    try:
        return {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    finally:
        conn.close()


def test_fresh_store_has_live_columns_and_change_log(db: Path):
    refmet_store.upsert("caffeine", "voted", refmet_id="RM7", path=db)
    columns = _columns(db)
    for column in ("live_status", "live_refmet_id", "live_changed_at"):
        assert columns.count(column) == 1
    assert "refmet_change_log" in _tables(db)
    hit = refmet_store.get("caffeine", path=db)
    assert hit is not None and (hit.status, hit.refmet_id) == ("voted", "RM7")
    assert (hit.live_status, hit.live_refmet_id, hit.live_changed_at) == (None, None, None)
    assert refmet_store.list_changes(path=db) == []


def test_previous_release_store_is_migrated_with_rows_intact(db: Path):
    _legacy_store(db)
    hit = refmet_store.get("dha", path=db)
    assert hit is not None and (hit.status, hit.refmet_id) == ("voted", "RM0153631")
    assert hit.updated_at == "2026-11-06T00:00:00+00:00"
    assert "live_status" in _columns(db) and "refmet_change_log" in _tables(db)
    # Rollback safety: the previous release names its columns, so its read still works.
    conn = sqlite3.connect(str(db))
    try:
        assert conn.execute(_PREVIOUS_RELEASE_SELECT, ("dha",)).fetchone()[:2] == ("voted", "RM0153631")
    finally:
        conn.close()


def test_two_threads_migrate_the_same_legacy_store(db: Path):
    _legacy_store(db)
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker() -> None:
        try:
            barrier.wait()
            refmet_store.get("dha", path=db)
        except BaseException as exc:  # noqa: BLE001 - the test reports any failure
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert errors == []
    columns = _columns(db)
    for column in ("live_status", "live_refmet_id", "live_changed_at"):
        assert columns.count(column) == 1


def test_no_match_against_a_vote_is_logged_once_and_keeps_the_vote(db: Path):
    refmet_store.upsert("dha", "voted", refmet_id="RM0153631", path=db)
    before = refmet_store.get("dha", path=db)
    assert before is not None

    assert refmet_store.apply_live_answer("dha", "no_match", path=db) == "vote_kept"

    after = refmet_store.get("dha", path=db)
    assert after is not None
    assert (after.status, after.refmet_id, after.updated_at) == ("voted", "RM0153631", before.updated_at)
    assert (after.live_status, after.live_refmet_id) == ("no_match", None)
    assert after.live_changed_at is not None
    changes = refmet_store.list_changes(path=db)
    assert len(changes) == 1
    record = changes[0]
    assert record.query == "dha"
    assert (record.old_status, record.old_refmet_id) == ("voted", "RM0153631")
    assert (record.new_status, record.new_refmet_id) == ("no_match", None)
    assert record.kind == "vote_kept"
    assert record.old_basis == "assumed"  # live columns were empty: the old value is the row itself
    assert record.changed_at.endswith("+00:00")

    # The same observation again: no change, no second record.
    assert refmet_store.apply_live_answer("dha", "no_match", path=db) is None
    assert len(refmet_store.list_changes(path=db)) == 1


def test_match_again_after_a_kept_vote_is_logged_as_recovered(db: Path):
    refmet_store.upsert("dha", "voted", refmet_id="RM0153631", path=db)
    refmet_store.apply_live_answer("dha", "no_match", path=db)

    assert refmet_store.apply_live_answer("dha", "voted", refmet_id="RM0153631", path=db) == "live_recovered"

    hit = refmet_store.get("dha", path=db)
    assert hit is not None and (hit.live_status, hit.live_refmet_id) == ("voted", "RM0153631")
    newest = refmet_store.list_changes(path=db)[0]
    assert newest.kind == "live_recovered"
    assert (newest.old_status, newest.new_status, newest.new_refmet_id) == ("no_match", "voted", "RM0153631")
    assert newest.old_basis == "observed"  # the old value came from the live columns


def test_confirming_a_vote_with_the_same_id_logs_nothing(db: Path):
    refmet_store.upsert("dha", "voted", refmet_id="RM0153631", path=db)
    assert refmet_store.apply_live_answer("dha", "voted", refmet_id="RM0153631", path=db) is None
    assert refmet_store.list_changes(path=db) == []


def test_observation_for_an_unknown_query_logs_nothing_and_creates_no_vote(db: Path):
    assert refmet_store.apply_live_answer("never seen", "no_match", path=db) is None
    hit = refmet_store.get("never seen", path=db)
    assert hit is not None and (hit.status, hit.refmet_id) == ("no_match", None)
    assert refmet_store.list_changes(path=db) == []
    # A first match for an unknown query is written back (as before) and is not a "change".
    assert refmet_store.apply_live_answer("novel", "voted", refmet_id="RM1", path=db) is None
    novel = refmet_store.get("novel", path=db)
    assert novel is not None and (novel.status, novel.refmet_id, novel.source) == ("voted", "RM1", "live_api")
    assert refmet_store.list_changes(path=db) == []


def test_two_connections_recording_the_same_change_log_it_once(db: Path):
    refmet_store.upsert("dha", "voted", refmet_id="RM0153631", path=db)
    results: list[str | None] = []

    def worker() -> None:
        results.append(refmet_store.apply_live_answer("dha", "no_match", path=db))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(r for r in results if r is not None) == ["vote_kept"]
    assert len(refmet_store.list_changes(path=db)) == 1


def test_no_match_never_replaces_a_vote_even_without_a_prior_read(db: Path):
    # R4 lives in the store function itself, not only in the annotator's read-then-decide.
    refmet_store.upsert("dha", "voted", refmet_id="RM0153631", path=db)
    refmet_store.apply_live_answer("dha", "no_match", path=db)
    refmet_store.apply_live_answer("dha", "no_match", path=db)
    hit = refmet_store.get("dha", path=db)
    assert hit is not None and (hit.status, hit.refmet_id) == ("voted", "RM0153631")


def test_different_id_replaces_the_vote_and_logs_id_changed(db: Path):
    refmet_store.upsert("dha", "voted", refmet_id="RM0153631", path=db)
    assert refmet_store.apply_live_answer("dha", "voted", refmet_id="RM9", path=db) == "id_changed"
    hit = refmet_store.get("dha", path=db)
    assert hit is not None and (hit.status, hit.refmet_id, hit.source) == ("voted", "RM9", "live_api")
    changes = refmet_store.list_changes(path=db)
    assert [c.kind for c in changes] == ["id_changed"]
    assert (changes[0].old_refmet_id, changes[0].new_refmet_id) == ("RM0153631", "RM9")


def test_match_for_a_stored_no_match_logs_match_appeared(db: Path):
    refmet_store.upsert("x", "no_match", path=db)
    assert refmet_store.apply_live_answer("x", "voted", refmet_id="RM2", path=db) == "match_appeared"
    hit = refmet_store.get("x", path=db)
    assert hit is not None and (hit.status, hit.refmet_id) == ("voted", "RM2")
    assert [c.kind for c in refmet_store.list_changes(path=db)] == ["match_appeared"]
    # A stored no-match that stays a no-match writes nothing.
    refmet_store.upsert("y", "no_match", path=db)
    assert refmet_store.apply_live_answer("y", "no_match", path=db) is None
    assert len(refmet_store.list_changes(path=db)) == 1


def test_unusable_rows_are_overwritten_without_a_log_row(db: Path):
    # An `unavailable` row and a `voted` row with no id carry no last-known-good: treated as no row.
    refmet_store.upsert("a", "unavailable", path=db)
    refmet_store.upsert("b", "voted", refmet_id=None, path=db)
    assert refmet_store.apply_live_answer("a", "no_match", path=db) is None
    assert refmet_store.apply_live_answer("b", "no_match", path=db) is None
    for query in ("a", "b"):
        hit = refmet_store.get(query, path=db)
        assert hit is not None and hit.status == "no_match"
    assert refmet_store.list_changes(path=db) == []


def test_a_match_without_an_id_is_never_applied(db: Path):
    refmet_store.upsert("dha", "voted", refmet_id="RM0153631", path=db)
    assert refmet_store.apply_live_answer("dha", "voted", refmet_id=None, path=db) is None
    hit = refmet_store.get("dha", path=db)
    assert hit is not None and hit.refmet_id == "RM0153631" and hit.live_status is None


def test_recording_and_listing_are_no_ops_without_a_store(monkeypatch):
    monkeypatch.delenv("REFMET_STORE_PATH", raising=False)
    assert refmet_store.apply_live_answer("dha", "no_match") is None
    assert refmet_store.list_changes() == []


def test_listing_is_newest_first_and_honors_the_limit(db: Path):
    for query in ("a", "b", "c"):
        refmet_store.upsert(query, "voted", refmet_id="RM1", path=db)
        refmet_store.apply_live_answer(query, "no_match", path=db)
    assert [c.query for c in refmet_store.list_changes(path=db)] == ["c", "b", "a"]
    assert [c.query for c in refmet_store.list_changes(limit=2, path=db)] == ["c", "b"]
