"""Write-through RefMet store — the mutable, self-healing RefMet source for mode ``live_backup``.

Why this exists
---------------
Mode ``frozen`` (see ``refmet_snapshot.py``) is an immutable TSV: deterministic and great for
benchmark reproducibility, but corpus-bound and stale vs RefMet releases. Mode ``live_backup`` (this
module) is the PROD source: live ``/match`` is tried first, its authoritative answer is written back
here, and when live is DOWN this store serves the last-known-good value as a backup. That keeps full
live coverage + freshness while still killing the breaker-driven flip (a frozen name resolves
correctly from backup even when live is unreachable).

Storage / concurrency choice
-----------------------------
A single SQLite database (WAL journal) with one table keyed on the normalized query name. Prod runs
multiple worker PROCESSES; to stay consistent across them we do NOT hold a long-lived in-memory copy
(D2). Every ``get``/``upsert``/``seed_from_tsv`` opens a SHORT-LIVED connection, does its one
statement, and closes it. WAL lets readers and a single writer proceed concurrently; ``busy_timeout``
(~2s) makes a writer wait out a competing writer instead of raising ``database is locked``. Upserts
are idempotent last-writer-wins, so two workers writing the same name converge on one row with the
same value. A per-process pooled connection would be a latency optimization; per-call is the simplest
correct approach for the multi-process deployment and is what this module implements.

Table ``refmet``::

    name_norm TEXT PRIMARY KEY   -- normalized query name (lower/strip/collapse-ws, shared with the freeze)
    refmet_id, refmet_name, inchi_key, formula, exactmass
    status                       -- voted | no_match | unavailable (mirrors the freeze statuses)
    source                       -- provenance of the WRITE (live_api for write-backs, local_snapshot for seeds)
    updated_at                   -- ISO-format UTC timestamp of the last write (D3 age surfacing)
    snapshot_version             -- freeze version when a seed populated the row, else None
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ... import config
from .base import AVAILABILITY_VOTED, REFMET_SOURCE_LIVE, REFMET_SOURCE_LOCAL
from .refmet_snapshot import _VALID_STATUSES, _normalize

logger = logging.getLogger(__name__)

_BUSY_TIMEOUT_MS = 2000

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS refmet (
    name_norm TEXT PRIMARY KEY,
    refmet_id TEXT,
    refmet_name TEXT,
    inchi_key TEXT,
    formula TEXT,
    exactmass TEXT,
    status TEXT NOT NULL,
    source TEXT,
    updated_at TEXT,
    snapshot_version TEXT
)
"""

# Freeze TSV columns copied verbatim into the store when seeding (see refmet_snapshot for the format).
_SEED_FIELDS = ("refmet_id", "refmet_name", "inchi_key", "formula", "exactmass")


@dataclass(frozen=True)
class StoreHit:
    """One row of the write-through store: the last-known RefMet outcome for a normalized name."""

    status: str
    refmet_id: str | None
    refmet_name: str | None = None
    inchi_key: str | None = None
    formula: str | None = None
    exactmass: str | None = None
    source: str | None = None
    updated_at: str | None = None
    snapshot_version: str | None = None


def _now() -> str:
    """ISO-format UTC timestamp for ``updated_at`` (D3: backup age is derived from this)."""
    return datetime.now(timezone.utc).isoformat()


def _resolve_path(path: Path | None) -> Path | None:
    """Explicit path wins (tests); otherwise the configured ``REFMET_STORE_PATH`` (read per call)."""
    return path if path is not None else config.get_refmet_store_path()


def is_configured() -> bool:
    """True iff a store path is configured (``REFMET_STORE_PATH`` set). The mode's rollout switch."""
    return config.get_refmet_store_path() is not None


def _connect(path: Path) -> sqlite3.Connection:
    """Open a short-lived connection, ensure the schema, and apply the WAL/durability pragmas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=_BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    conn.execute(_CREATE_TABLE)
    return conn


def get(name: str, *, path: Path | None = None) -> StoreHit | None:
    """Last-known store row for ``name`` (normalized key), or None on a miss / no store configured."""
    if not name:
        return None
    resolved = _resolve_path(path)
    if resolved is None:
        return None
    key = _normalize(name)
    conn = _connect(resolved)
    try:
        row = conn.execute(
            "SELECT status, refmet_id, refmet_name, inchi_key, formula, exactmass, source, "
            "updated_at, snapshot_version FROM refmet WHERE name_norm = ?",
            (key,),
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    return StoreHit(
        status=row["status"],
        refmet_id=row["refmet_id"],
        refmet_name=row["refmet_name"],
        inchi_key=row["inchi_key"],
        formula=row["formula"],
        exactmass=row["exactmass"],
        source=row["source"],
        updated_at=row["updated_at"],
        snapshot_version=row["snapshot_version"],
    )


def upsert(
    name: str,
    status: str,
    *,
    refmet_id: str | None = None,
    refmet_name: str | None = None,
    inchi_key: str | None = None,
    formula: str | None = None,
    exactmass: str | None = None,
    source: str = REFMET_SOURCE_LIVE,
    snapshot_version: str | None = None,
    path: Path | None = None,
) -> None:
    """Insert or update the row for ``name`` (normalized key), stamping ``updated_at`` on write.

    Idempotent last-writer-wins: ``INSERT ... ON CONFLICT(name_norm) DO UPDATE``. No-op when no store
    is configured or the name is empty (the caller guards "different" before writing; this is a plain
    write primitive).
    """
    if not name:
        return
    resolved = _resolve_path(path)
    if resolved is None:
        return
    key = _normalize(name)
    conn = _connect(resolved)
    try:
        conn.execute(
            """
            INSERT INTO refmet (
                name_norm, refmet_id, refmet_name, inchi_key, formula, exactmass,
                status, source, updated_at, snapshot_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(name_norm) DO UPDATE SET
                refmet_id = excluded.refmet_id,
                refmet_name = excluded.refmet_name,
                inchi_key = excluded.inchi_key,
                formula = excluded.formula,
                exactmass = excluded.exactmass,
                status = excluded.status,
                source = excluded.source,
                updated_at = excluded.updated_at,
                snapshot_version = excluded.snapshot_version
            """,
            (key, refmet_id, refmet_name, inchi_key, formula, exactmass, status, source, _now(), snapshot_version),
        )
        conn.commit()
    finally:
        conn.close()


def _is_empty(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM refmet LIMIT 1").fetchone() is None


def seed_from_tsv(tsv_path: Path, *, path: Path | None = None) -> int:
    """Bulk-import the freeze TSV rows into the store ONCE, only if the store is empty.

    Returns the number of rows inserted (0 when the store already had rows, no store is configured, or
    the TSV is unreadable). Seeded rows carry ``source=local_snapshot`` and the freeze's version so a
    day-one deployment is pre-covered by the pinned corpus; the store then grows from live traffic.
    """
    resolved = _resolve_path(path)
    if resolved is None:
        return 0
    snapshot_version = config.derive_refmet_snapshot_version(tsv_path)
    conn = _connect(resolved)
    try:
        if not _is_empty(conn):
            return 0
        try:
            handle = tsv_path.open("r", encoding="utf-8", newline="")
        except OSError as exc:
            logger.warning("RefMet store seed skipped: could not read freeze %s (%s)", tsv_path, exc)
            return 0
        inserted = 0
        skipped = 0
        now = _now()
        with handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if reader.fieldnames is None or "query_name" not in reader.fieldnames or "status" not in reader.fieldnames:
                logger.warning("RefMet store seed skipped: freeze %s missing query_name/status header", tsv_path)
                return 0
            rows: list[tuple[object, ...]] = []
            seen: set[str] = set()
            for row in reader:
                query_name = (row.get("query_name") or "").strip()
                status = (row.get("status") or "").strip()
                if not query_name or not status:
                    continue
                key = _normalize(query_name)
                if key in seen:
                    continue  # first occurrence wins, mirroring the freeze loader
                values = {f: ((row.get(f) or "").strip() or None) for f in _SEED_FIELDS}
                # Mirror the freeze loader's validation (refmet_snapshot._parse): reject an unknown status
                # or a `voted` row with no refmet_id. A malformed row must not be seeded — it would later
                # replay to a misleading `unavailable` and silently remove outage coverage. Skip (don't
                # mark `seen`), so a valid later occurrence of the same name can still win.
                if status not in _VALID_STATUSES or (status == AVAILABILITY_VOTED and not values["refmet_id"]):
                    skipped += 1
                    continue
                seen.add(key)
                rows.append(
                    (
                        key,
                        values["refmet_id"],
                        values["refmet_name"],
                        values["inchi_key"],
                        values["formula"],
                        values["exactmass"],
                        status,
                        REFMET_SOURCE_LOCAL,
                        now,
                        snapshot_version,
                    )
                )
            if rows:
                conn.executemany(
                    "INSERT OR IGNORE INTO refmet ("
                    "name_norm, refmet_id, refmet_name, inchi_key, formula, exactmass, "
                    "status, source, updated_at, snapshot_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    rows,
                )
                conn.commit()
                inserted = len(rows)
        if skipped:
            logger.warning("RefMet store seed skipped %d malformed row(s) from %s", skipped, tsv_path)
        if inserted:
            logger.info("Seeded RefMet store at %s with %d rows from freeze %s", resolved, inserted, tsv_path)
        return inserted
    finally:
        conn.close()
