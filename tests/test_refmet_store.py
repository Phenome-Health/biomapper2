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
