"""Write-through RefMet store — the mutable, self-healing RefMet source for mode ``live_backup``.

Why this exists
---------------
Mode ``frozen`` (see ``refmet_snapshot.py``) is an immutable TSV: deterministic and great for
benchmark reproducibility, but corpus-bound and stale vs RefMet releases. Mode ``live_backup`` (this
module) is the PROD source: live ``/match`` is tried first, its answer is written back here, and when
live is DOWN this store serves the last-known-good value as a backup. That keeps full live coverage +
freshness while still killing the breaker-driven flip (a frozen name resolves correctly from backup
even when live is unreachable).

One live answer is NOT written back: a no-match for a query whose row holds a vote with an id. RefMet
``/match`` stopped matching names it used to match around 2026-11-12, so such a no-match is far more
likely upstream name-matching drift than a retired entry. The vote stays, the annotator serves it as
``kept_live_no_match``, and the change is recorded (see "Last live answer and change log"). Issue:
https://github.com/Phenome-Health/biomapper2/issues/159

Storage / concurrency choice
-----------------------------
A single SQLite database (WAL journal) with one table keyed on the EXACT query string. Prod runs
multiple worker PROCESSES; to stay consistent across them we do NOT hold a long-lived in-memory copy
(D2). Every ``get``/``upsert``/``seed_from_tsv`` opens a SHORT-LIVED connection, does its one
statement, and closes it. WAL lets readers and a single writer proceed concurrently; ``busy_timeout``
(~2s) makes a writer wait out a competing writer instead of raising ``database is locked``. Upserts
are idempotent last-writer-wins, so two workers writing the same name converge on one row with the
same value. A per-process pooled connection would be a latency optimization; per-call is the simplest
correct approach for the multi-process deployment and is what this module implements.

Table ``refmet_by_query``::

    query TEXT PRIMARY KEY       -- the exact string sent to /match (no case or whitespace folding)
    refmet_id, refmet_name, inchi_key, formula, exactmass
    status                       -- voted | no_match | unavailable (mirrors the freeze statuses)
    source                       -- provenance of the WRITE (live_api for write-backs, local_snapshot for seeds)
    updated_at                   -- ISO-format UTC timestamp of the last write (D3 age surfacing)
    snapshot_version             -- freeze version when a seed populated the row, else None
    live_status, live_refmet_id  -- what live /match LAST said for this query (see below)
    live_changed_at              -- UTC time that live answer last CHANGED (not when it was last checked)

Last live answer and change log
-------------------------------
``(status, refmet_id)`` is the last-known-good vote; ``(live_status, live_refmet_id)`` is what live last
said. They differ exactly while a vote is being kept against a live no-match. A row whose live columns
are empty is read as "live last said what the row says": before these columns existed every live
answer overwrote the row, so the row WAS the last live answer. For a row seeded from the freeze or
restored by a repair that is an assumption, so each log row says whether its old value was ``observed``
(read from the live columns) or ``assumed`` (read from the row).

Table ``refmet_change_log`` is append-only, one row per change of a query's live answer::

    id INTEGER PRIMARY KEY AUTOINCREMENT
    query
    old_status, old_refmet_id    -- the live answer before the change
    new_status, new_refmet_id    -- the live answer after it
    kind                         -- vote_kept | live_recovered | id_changed | match_appeared | match_lost | repair
    old_basis                    -- observed | assumed | stored_row (repair: old/new describe the stored vote)
    changed_at                   -- ISO-format UTC timestamp

``apply_live_answer`` does the live-column update, the log insert and the row write-back in ONE write
transaction, and the update carries its own "is it different?" condition. So two workers that see the
same new answer produce one log row, and a write-back can never hide the change it belongs to.

Schema migration is additive and runs on connect: the three columns are added when missing (a
``duplicate column name`` error means the other worker added it first and is ignored) and the log table
is created if absent. The previous release names its columns when it reads and writes, so it still
works against a migrated file; if rolled back it would resume overwriting votes and the log would go
stale, which is harmless.

Why the key is the exact query (and what happened to table ``refmet``)
----------------------------------------------------------------------
RefMet ``/match`` is case-sensitive: ``acetylcarnitine (C2)`` votes CAR 2:0 and ``acetylcarnitine (c2)``
is a no-match. The original table ``refmet`` was keyed on a case-folded ``name_norm`` with last-writer-
wins upserts, so the two spellings shared one row, and during an outage a name could be served its
case-twin's answer. Rows are now keyed on the exact query string, so case variants never cross-serve.

The legacy ``refmet`` table is left in place and is NEVER read: its folded key cannot say which spelling
wrote a row, so no legacy row can be served safely. It is not dropped either, so rolling back to the
previous release finds its data intact. On first start the new table is empty and is seeded from the
pinned freeze (exact ``query_name`` keys), then grows again from live write-backs; until a name is
re-learned live its outage fallback is UNAVAILABLE, never a wrong answer.
"""

from __future__ import annotations

import csv
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ... import config
from .base import AVAILABILITY_NO_MATCH, AVAILABILITY_VOTED, REFMET_SOURCE_LIVE, REFMET_SOURCE_LOCAL
from .refmet_snapshot import _VALID_STATUSES

logger = logging.getLogger(__name__)

_BUSY_TIMEOUT_MS = 2000

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS refmet_by_query (
    query TEXT PRIMARY KEY,
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

# Added with the kept-vote change; ``_connect`` adds each one to a store file that predates it.
_LIVE_COLUMNS = ("live_status", "live_refmet_id", "live_changed_at")

_CREATE_CHANGE_LOG = """
CREATE TABLE IF NOT EXISTS refmet_change_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    old_status TEXT,
    old_refmet_id TEXT,
    new_status TEXT,
    new_refmet_id TEXT,
    kind TEXT NOT NULL,
    old_basis TEXT NOT NULL,
    changed_at TEXT NOT NULL
)
"""

# Change kinds written to ``refmet_change_log.kind``.
CHANGE_VOTE_KEPT = "vote_kept"  # live went to no-match while a vote is stored; the vote is kept
CHANGE_LIVE_RECOVERED = "live_recovered"  # a kept vote's query matches again with the same id
CHANGE_ID_CHANGED = "id_changed"  # live matches with a different id; live wins
CHANGE_MATCH_APPEARED = "match_appeared"  # a stored no-match now matches
CHANGE_MATCH_LOST = "match_lost"  # live went to no-match and no vote is stored (only after a manual row edit)
CHANGE_REPAIR = "repair"  # an operator restored the row from a freeze file

# Where a log row's old value came from.
BASIS_OBSERVED = "observed"  # the row's live columns
BASIS_ASSUMED = "assumed"  # the row itself (live columns empty)
BASIS_STORED_ROW = "stored_row"  # repair: old/new are the stored vote, not a live answer

# Freeze TSV columns copied verbatim into the store when seeding (see refmet_snapshot for the format).
_SEED_FIELDS = ("refmet_id", "refmet_name", "inchi_key", "formula", "exactmass")


@dataclass(frozen=True)
class StoreHit:
    """One row of the write-through store: the last-known RefMet outcome for one exact query."""

    status: str
    refmet_id: str | None
    refmet_name: str | None = None
    inchi_key: str | None = None
    formula: str | None = None
    exactmass: str | None = None
    source: str | None = None
    updated_at: str | None = None
    snapshot_version: str | None = None
    # What live /match last said for this query; all None until a live answer is recorded for the row.
    live_status: str | None = None
    live_refmet_id: str | None = None
    live_changed_at: str | None = None


@dataclass(frozen=True)
class ChangeRecord:
    """One row of ``refmet_change_log``: a change in what live ``/match`` says for one exact query."""

    query: str
    old_status: str | None
    old_refmet_id: str | None
    new_status: str | None
    new_refmet_id: str | None
    kind: str
    old_basis: str
    changed_at: str


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
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add the live-answer columns and the change log to a store file that predates them."""
    present = {row["name"] for row in conn.execute("PRAGMA table_info(refmet_by_query)")}
    for column in _LIVE_COLUMNS:
        if column in present:
            continue
        try:
            conn.execute(f"ALTER TABLE refmet_by_query ADD COLUMN {column} TEXT")
        except sqlite3.OperationalError as exc:
            # SQLite has no ADD COLUMN IF NOT EXISTS. Another worker migrating the same file at the
            # same moment may have added the column between the check above and this statement.
            if "duplicate column name" not in str(exc).lower():
                raise
    conn.execute(_CREATE_CHANGE_LOG)


def _connect_readonly(path: Path) -> sqlite3.Connection:
    """Open an existing store read-only: never creates the file, never migrates it."""
    conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True, timeout=_BUSY_TIMEOUT_MS / 1000)
    conn.row_factory = sqlite3.Row
    conn.execute(f"PRAGMA busy_timeout={_BUSY_TIMEOUT_MS}")
    return conn


def _has_table(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)).fetchone()
    return row is not None


def get(name: str, *, path: Path | None = None) -> StoreHit | None:
    """Last-known store row for the exact query ``name``, or None on a miss / no store configured."""
    if not name:
        return None
    resolved = _resolve_path(path)
    if resolved is None:
        return None
    conn = _connect(resolved)
    try:
        row = conn.execute(
            "SELECT status, refmet_id, refmet_name, inchi_key, formula, exactmass, source, "
            "updated_at, snapshot_version, live_status, live_refmet_id, live_changed_at "
            "FROM refmet_by_query WHERE query = ?",
            (name,),
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
        live_status=row["live_status"],
        live_refmet_id=row["live_refmet_id"],
        live_changed_at=row["live_changed_at"],
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
    """Insert or update the row for the exact query ``name``, stamping ``updated_at`` on write.

    Idempotent last-writer-wins: ``INSERT ... ON CONFLICT(query) DO UPDATE``. No-op when no store
    is configured or the name is empty (the caller guards "different" before writing; this is a plain
    write primitive).
    """
    if not name:
        return
    resolved = _resolve_path(path)
    if resolved is None:
        return
    conn = _connect(resolved)
    try:
        conn.execute(
            """
            INSERT INTO refmet_by_query (
                query, refmet_id, refmet_name, inchi_key, formula, exactmass,
                status, source, updated_at, snapshot_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(query) DO UPDATE SET
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
            (name, refmet_id, refmet_name, inchi_key, formula, exactmass, status, source, _now(), snapshot_version),
        )
        conn.commit()
    finally:
        conn.close()


def apply_live_answer(
    name: str,
    status: str,
    *,
    refmet_id: str | None = None,
    path: Path | None = None,
) -> str | None:
    """Record what live ``/match`` just said for the exact query ``name``; write it back where due.

    One write transaction does three things:

    1. updates the row's live-answer columns, only if the answer differs from the last one (the
       comparison is the UPDATE's own condition);
    2. appends a ``refmet_change_log`` row, only if step 1 changed the row;
    3. writes the answer to the row's vote columns, EXCEPT a no-match is never written over a vote
       that has an id (the vote is kept).

    Returns the change kind that was logged, or None when nothing was logged. A query with no row, or
    with a row that carries no last-known-good (``unavailable``, or ``voted`` without an id), has the
    answer written as a plain write-back and logs nothing. A match without an id is never applied.
    No-op when no store is configured. Raises on a store error: the caller is best-effort and logs it.
    """
    if not name or status not in (AVAILABILITY_VOTED, AVAILABILITY_NO_MATCH):
        return None
    if status == AVAILABILITY_VOTED and not refmet_id:
        return None
    if status == AVAILABILITY_NO_MATCH:
        refmet_id = None
    resolved = _resolve_path(path)
    if resolved is None:
        return None
    conn = _connect(resolved)
    try:
        conn.isolation_level = None  # explicit transaction control below
        conn.execute("BEGIN IMMEDIATE")  # take the write lock before reading the row
        try:
            kind = _apply_live_answer(conn, name, status, refmet_id)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:  # SQLite already rolled back on some errors (full disk, I/O)
                conn.execute("ROLLBACK")
            raise
        return kind
    finally:
        conn.close()


def _apply_live_answer(conn: sqlite3.Connection, name: str, status: str, refmet_id: str | None) -> str | None:
    """Body of :func:`apply_live_answer`; runs inside the caller's write transaction."""
    now = _now()
    row = conn.execute(
        "SELECT status, refmet_id, live_status, live_refmet_id FROM refmet_by_query WHERE query = ?", (name,)
    ).fetchone()
    has_vote = row is not None and row["status"] == AVAILABILITY_VOTED and bool(row["refmet_id"])
    is_no_match = row is not None and row["status"] == AVAILABILITY_NO_MATCH
    if row is None or not (has_vote or is_no_match):
        # No last-known-good to compare with: a plain write-back, nothing to log.
        conn.execute(
            """
            INSERT INTO refmet_by_query (
                query, refmet_id, status, source, updated_at, live_status, live_refmet_id, live_changed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(query) DO UPDATE SET
                refmet_id = excluded.refmet_id,
                refmet_name = NULL,
                inchi_key = NULL,
                formula = NULL,
                exactmass = NULL,
                status = excluded.status,
                source = excluded.source,
                updated_at = excluded.updated_at,
                snapshot_version = NULL,
                live_status = excluded.live_status,
                live_refmet_id = excluded.live_refmet_id,
                live_changed_at = excluded.live_changed_at
            """,
            (name, refmet_id, status, REFMET_SOURCE_LIVE, now, status, refmet_id, now),
        )
        return None

    stored_id = row["refmet_id"] if has_vote else None
    observed = row["live_status"] is not None
    old_status = row["live_status"] if observed else row["status"]
    old_id = (row["live_refmet_id"] if observed else stored_id) if old_status == AVAILABILITY_VOTED else None

    # 1. The live columns change only if the answer differs from the last one. With empty live columns
    #    the last answer is the row itself. `IS` compares NULLs as equal.
    changed = (
        conn.execute(
            """
            UPDATE refmet_by_query
               SET live_status = ?, live_refmet_id = ?, live_changed_at = ?
             WHERE query = ?
               AND NOT (
                   COALESCE(live_status, status) IS ?
                   AND (CASE WHEN COALESCE(live_status, status) != 'voted' THEN NULL
                             WHEN live_status IS NULL THEN refmet_id
                             ELSE live_refmet_id END) IS ?
               )
            """,
            (status, refmet_id, now, name, status, refmet_id),
        ).rowcount
        == 1
    )

    # 3. Write-back, in the same transaction so it cannot hide the change from step 1. A no-match
    #    never replaces a vote that has an id (R4).
    keeps_vote = status == AVAILABILITY_NO_MATCH and has_vote
    if not keeps_vote and (row["status"], stored_id) != (status, refmet_id):
        conn.execute(
            """
            UPDATE refmet_by_query
               SET status = ?, refmet_id = ?, refmet_name = NULL, inchi_key = NULL, formula = NULL,
                   exactmass = NULL, source = ?, updated_at = ?, snapshot_version = NULL
             WHERE query = ?
            """,
            (status, refmet_id, REFMET_SOURCE_LIVE, now, name),
        )

    if not changed:
        return None

    # 2. Exactly one log row per change.
    if status == AVAILABILITY_NO_MATCH:
        kind = CHANGE_VOTE_KEPT if has_vote else CHANGE_MATCH_LOST
    elif is_no_match:
        kind = CHANGE_MATCH_APPEARED
    elif stored_id == refmet_id and old_status == AVAILABILITY_NO_MATCH:
        kind = CHANGE_LIVE_RECOVERED
    else:
        kind = CHANGE_ID_CHANGED
    _log_change(
        conn,
        query=name,
        old=(old_status, old_id),
        new=(status, refmet_id),
        kind=kind,
        old_basis=BASIS_OBSERVED if observed else BASIS_ASSUMED,
        changed_at=now,
    )
    return kind


def _log_change(
    conn: sqlite3.Connection,
    *,
    query: str,
    old: tuple[str | None, str | None],
    new: tuple[str | None, str | None],
    kind: str,
    old_basis: str,
    changed_at: str,
) -> None:
    conn.execute(
        "INSERT INTO refmet_change_log (query, old_status, old_refmet_id, new_status, new_refmet_id, kind, "
        "old_basis, changed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (query, old[0], old[1], new[0], new[1], kind, old_basis, changed_at),
    )


def list_changes(*, limit: int | None = None, path: Path | None = None) -> list[ChangeRecord]:
    """Recorded live-answer changes, newest first, optionally limited to the newest ``limit``.

    Read-only: it never creates or migrates the store. Returns an empty list when no store is
    configured, the file does not exist, or the file predates the change log.
    """
    resolved = _resolve_path(path)
    if resolved is None or not resolved.is_file():
        return []
    conn = _connect_readonly(resolved)
    try:
        if not _has_table(conn, "refmet_change_log"):
            return []
        sql = (
            "SELECT query, old_status, old_refmet_id, new_status, new_refmet_id, kind, old_basis, changed_at "
            "FROM refmet_change_log ORDER BY id DESC"
        )
        rows = (
            conn.execute(sql + " LIMIT ?", (limit,)).fetchall() if limit is not None else conn.execute(sql).fetchall()
        )
    finally:
        conn.close()
    return [ChangeRecord(**dict(row)) for row in rows]


def _is_empty(conn: sqlite3.Connection) -> bool:
    return conn.execute("SELECT 1 FROM refmet_by_query LIMIT 1").fetchone() is None


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
                key = query_name  # exact query, like the freeze's exact table: case variants stay apart
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
                    "INSERT OR IGNORE INTO refmet_by_query ("
                    "query, refmet_id, refmet_name, inchi_key, formula, exactmass, "
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


# --------------------------------------------------------------------------------------------
# Operator functions (scripts/refmet_store_admin.py): inspect, back up, and repair a store.
#
# Unlike the mapping-path functions above these RAISE ``StoreAdminError`` instead of no-op'ing: an
# operator pointed at the wrong path must get an error, never a fresh empty store and a falsely
# clean result. The read-only ones open the file read-only and never create or migrate it.
# --------------------------------------------------------------------------------------------


class StoreAdminError(Exception):
    """An operator command cannot run: no store configured, store file missing, or unusable freeze."""


@dataclass(frozen=True)
class Contradiction:
    """A store row that is a no-match while the freeze has a vote with an id for the same exact query."""

    query: str
    freeze_refmet_id: str
    updated_at: str | None


def _require_store(path: Path | None) -> Path:
    """The store file an operator command works on. It must be configured and must already exist."""
    resolved = _resolve_path(path)
    if resolved is None:
        raise StoreAdminError("no RefMet store configured: REFMET_STORE_PATH is not set")
    if not resolved.is_file():
        raise StoreAdminError(f"RefMet store file does not exist: {resolved}")
    return resolved


def store_info(*, path: Path | None = None) -> tuple[Path, int]:
    """The resolved store path and its row count. Read-only."""
    resolved = _require_store(path)
    conn = _connect_readonly(resolved)
    try:
        if not _has_table(conn, "refmet_by_query"):
            raise StoreAdminError(f"not a RefMet store (no refmet_by_query table): {resolved}")
        return resolved, conn.execute("SELECT COUNT(*) FROM refmet_by_query").fetchone()[0]
    finally:
        conn.close()


def _read_freeze_votes(tsv_path: Path) -> dict[str, dict[str, str | None]]:
    """Freeze rows that are a vote with an id, keyed by the exact ``query_name``.

    Reads the TSV directly, the way ``seed_from_tsv`` does (first valid occurrence of a query wins),
    and never through ``refmet_snapshot.lookup``, which falls back to a case-folded key. Surrounding
    spaces in ``query_name`` are kept: a padded query is a different query from the unpadded one.
    """
    try:
        handle = tsv_path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise StoreAdminError(f"could not read freeze {tsv_path}: {exc}") from exc
    first: dict[str, dict[str, str | None]] = {}
    with handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if reader.fieldnames is None or "query_name" not in reader.fieldnames or "status" not in reader.fieldnames:
            raise StoreAdminError(f"freeze {tsv_path} has no query_name/status header")
        for row in reader:
            # The key is the query exactly as written: no strip, so " dha" never stands in for "dha".
            # (seed_from_tsv does strip its keys; a padded freeze row then simply has no repair match.)
            query_name = row.get("query_name") or ""
            status = (row.get("status") or "").strip()
            if not query_name.strip() or query_name in first:
                continue
            values: dict[str, str | None] = {f: ((row.get(f) or "").strip() or None) for f in _SEED_FIELDS}
            if status not in _VALID_STATUSES or (status == AVAILABILITY_VOTED and not values["refmet_id"]):
                continue  # malformed, as in seed_from_tsv: a valid later occurrence can still win
            values["status"] = status
            first[query_name] = values
    return {query: values for query, values in first.items() if values["status"] == AVAILABILITY_VOTED}


def _parse_utc(value: str) -> datetime:
    """An ISO date or timestamp as an aware UTC datetime (a value without an offset is taken as UTC)."""
    parsed = datetime.fromisoformat(value)
    return parsed.replace(tzinfo=timezone.utc) if parsed.tzinfo is None else parsed.astimezone(timezone.utc)


def find_contradictions(tsv_path: Path, *, since: str | None = None, path: Path | None = None) -> list[Contradiction]:
    """Store rows that are a no-match where the freeze has a vote with an id for the same exact query.

    Read-only. No case or whitespace folding: a freeze row that differs from the store row only by
    case is a different query. ``since`` (an ISO date or timestamp, UTC) keeps only rows last updated
    on or after it; a row with no usable ``updated_at`` is then left out. Sorted by query.
    """
    resolved = _require_store(path)
    votes = _read_freeze_votes(tsv_path)
    try:
        cutoff = _parse_utc(since) if since is not None else None
    except ValueError as exc:
        raise StoreAdminError(f"--since is not an ISO date or timestamp: {since!r}") from exc
    conn = _connect_readonly(resolved)
    try:
        if not _has_table(conn, "refmet_by_query"):
            raise StoreAdminError(f"not a RefMet store (no refmet_by_query table): {resolved}")
        rows = conn.execute(
            "SELECT query, updated_at FROM refmet_by_query WHERE status = ? ORDER BY query", (AVAILABILITY_NO_MATCH,)
        ).fetchall()
    finally:
        conn.close()
    found: list[Contradiction] = []
    for row in rows:
        vote = votes.get(row["query"])
        if vote is None:
            continue
        if cutoff is not None:
            try:
                if row["updated_at"] is None or _parse_utc(row["updated_at"]) < cutoff:
                    continue
            except ValueError:
                continue
        found.append(Contradiction(row["query"], str(vote["refmet_id"]), row["updated_at"]))
    return found


def restore_from_freeze(tsv_path: Path, queries: list[str], *, path: Path | None = None) -> list[str]:
    """Restore the given exact queries to the freeze's vote; return the ones actually restored.

    One write transaction. A query is restored only if the freeze has a vote with an id for it AND
    its store row is still a no-match at write time, so a row that changed since the operator's dry
    run is left alone. A restored row looks like a seeded one (``source=local_snapshot``, the freeze's
    version) and its live answer is set to no-match: that is what live last said, so the next request
    keeps the vote without logging a second change. Each restored row gets one ``repair`` log row.
    """
    resolved = _require_store(path)
    votes = _read_freeze_votes(tsv_path)
    snapshot_version = config.derive_refmet_snapshot_version(tsv_path)
    restored: list[str] = []
    conn = _connect(resolved)
    try:
        conn.isolation_level = None
        conn.execute("BEGIN IMMEDIATE")
        try:
            now = _now()
            for query in queries:
                vote = votes.get(query)
                if vote is None:
                    continue
                updated = conn.execute(
                    """
                    UPDATE refmet_by_query
                       SET refmet_id = ?, refmet_name = ?, inchi_key = ?, formula = ?, exactmass = ?,
                           status = ?, source = ?, snapshot_version = ?,
                           live_changed_at = CASE WHEN live_status = 'no_match' THEN live_changed_at
                                                  WHEN live_status IS NULL THEN updated_at
                                                  ELSE ? END,
                           live_status = 'no_match', live_refmet_id = NULL,
                           updated_at = ?
                     WHERE query = ? AND status = 'no_match'
                    """,
                    (
                        *(vote[f] for f in _SEED_FIELDS),
                        AVAILABILITY_VOTED,
                        REFMET_SOURCE_LOCAL,
                        snapshot_version,
                        now,
                        now,
                        query,
                    ),
                ).rowcount
                if updated != 1:
                    continue
                _log_change(
                    conn,
                    query=query,
                    old=(AVAILABILITY_NO_MATCH, None),
                    new=(AVAILABILITY_VOTED, vote["refmet_id"]),
                    kind=CHANGE_REPAIR,
                    old_basis=BASIS_STORED_ROW,
                    changed_at=now,
                )
                restored.append(query)
            conn.execute("COMMIT")
        except BaseException:
            if conn.in_transaction:  # SQLite already rolled back on some errors (full disk, I/O)
                conn.execute("ROLLBACK")
            raise
    finally:
        conn.close()
    return restored


def backup(dest: Path, *, path: Path | None = None) -> int:
    """Write a consistent copy of the store to ``dest``; return the copy's row count.

    Uses SQLite's online backup, which is safe while the API workers are writing. A plain file copy of
    a live WAL database is not: it can miss pages still in the ``-wal`` file. Never overwrites ``dest``.
    """
    resolved = _require_store(path)
    if dest.exists():
        raise StoreAdminError(f"backup destination already exists: {dest}")
    source = _connect_readonly(resolved)
    try:
        target = sqlite3.connect(str(dest))
        try:
            source.backup(target)
            return target.execute("SELECT COUNT(*) FROM refmet_by_query").fetchone()[0]
        finally:
            target.close()
    finally:
        source.close()
