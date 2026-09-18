"""Write-through RefMet store (core/annotators/refmet_store.py).

SQLite temp DB only; NO network. Proves the store primitives the ``live_backup`` mode relies on:
upsert/get, seed-from-freeze (idempotent), update-on-conflict (one row, newest wins), normalized-name
lookup, thread-concurrent upserts (no corruption, last-writer-wins), and config path discovery.
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


def _row_count(db: Path, name_norm: str) -> int:
    conn = sqlite3.connect(str(db))
    try:
        return conn.execute("SELECT COUNT(*) FROM refmet WHERE name_norm = ?", (name_norm,)).fetchone()[0]
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


def test_normalized_name_lookup(db: Path):
    # Stored with odd casing + doubled whitespace; retrieved by the normalized key.
    refmet_store.upsert("Cholic  Acid", "voted", refmet_id="RM0041813", path=db)
    hit = refmet_store.get("cholic acid", path=db)
    assert hit is not None and hit.refmet_id == "RM0041813"
    assert _row_count(db, "cholic acid") == 1


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
