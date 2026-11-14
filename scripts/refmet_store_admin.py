#!/usr/bin/env python3
"""Inspect, back up and repair the write-through RefMet store used by ``REFMET_FREEZE_MODE=live_backup``.

The store is the SQLite file at ``REFMET_STORE_PATH`` (read from the environment or ``.env``). Run this
from the repository root as the user the API service runs as, so the store keeps its ownership::

    uv run python scripts/refmet_store_admin.py history [--limit N]
    uv run python scripts/refmet_store_admin.py repair --freeze PATH [--since DATE]
    uv run python scripts/refmet_store_admin.py repair --freeze PATH [--since DATE] --apply --expect-count N
    uv run python scripts/refmet_store_admin.py backup DEST

``history`` lists every recorded change in what live RefMet ``/match`` says for a query, newest first
(a vote kept against a live no-match, a kept vote matching again, an id change, a no-match starting to
match, a repair).

``repair`` lists store rows that are a no-match where the freeze file has a vote with an id for the
same exact query (no case or whitespace folding). Without ``--apply`` it changes nothing. With
``--apply`` it recomputes the list, refuses to go on unless the list has exactly ``--expect-count``
rows (the number you reviewed in the dry run), restores those rows to the freeze's vote, and records
one ``repair`` change per row. ``--since`` keeps only rows last updated on or after a UTC date.

``backup`` writes a consistent copy of the store with SQLite's online backup. Use it, not ``cp``, on a
store the API is writing to.

``history`` and a ``repair`` dry run open the store read-only. Every command exits non-zero when no
store is configured or the store file does not exist; none of them creates a store. Output is plain
tab-separated lines so it can be saved next to a run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from biomapper2 import config
from biomapper2.core.annotators import refmet_store


def _print_store() -> None:
    path, rows = refmet_store.store_info()
    print(f"store\t{path}")
    print(f"store_rows\t{rows}")


def _history(args: argparse.Namespace) -> int:
    _print_store()
    changes = refmet_store.list_changes(limit=args.limit)
    print("changed_at\tquery\tkind\told_status\told_refmet_id\tnew_status\tnew_refmet_id\told_basis")
    for c in changes:
        fields = (c.changed_at, c.query, c.kind, c.old_status, c.old_refmet_id, c.new_status, c.new_refmet_id)
        print("\t".join("" if f is None else f for f in (*fields, c.old_basis)))
    print(f"changes\t{len(changes)}")
    return 0


def _repair(args: argparse.Namespace) -> int:
    if args.apply and args.expect_count is None:
        raise refmet_store.StoreAdminError("--apply needs --expect-count N (the row count from the dry run)")
    freeze = Path(args.freeze)
    _print_store()
    print(f"freeze\t{freeze}\t{config.derive_refmet_snapshot_version(freeze)}")
    if args.since:
        print(f"since\t{args.since}")
    found = refmet_store.find_contradictions(freeze, since=args.since)
    print("query\tfreeze_refmet_id\tstore_updated_at")
    for c in found:
        print(f"{c.query}\t{c.freeze_refmet_id}\t{c.updated_at or ''}")
    print(f"contradictions\t{len(found)}")
    if not args.apply:
        print("dry_run\tnothing written; re-run with --apply --expect-count N to restore these rows")
        return 0
    if len(found) != args.expect_count:
        raise refmet_store.StoreAdminError(
            f"--expect-count is {args.expect_count} but the list has {len(found)} row(s); nothing written. "
            "Review the list above and re-run."
        )
    restored = refmet_store.restore_from_freeze(freeze, [c.query for c in found])
    for query in restored:
        print(f"restored_query\t{query}")
    print(f"restored\t{len(restored)}")
    skipped = len(found) - len(restored)
    if skipped:
        print(f"skipped\t{skipped}\tno longer a no-match when the write ran")
    return 0


def _backup(args: argparse.Namespace) -> int:
    _print_store()
    dest = Path(args.dest)
    rows = refmet_store.backup(dest)
    print(f"backup\t{dest}")
    print(f"backup_rows\t{rows}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)

    history = commands.add_parser("history", help="list recorded live-answer changes, newest first")
    history.add_argument("--limit", type=int, default=None, help="show only the newest N changes")
    history.set_defaults(run=_history)

    repair = commands.add_parser("repair", help="list (and with --apply restore) rows that contradict a freeze")
    repair.add_argument("--freeze", required=True, help="freeze TSV to compare against, by exact query_name")
    repair.add_argument("--since", default=None, help="only rows last updated on or after this UTC date")
    repair.add_argument("--apply", action="store_true", help="restore the listed rows (default: dry run)")
    repair.add_argument("--expect-count", type=int, default=None, help="row count reviewed in the dry run")
    repair.set_defaults(run=_repair)

    backup = commands.add_parser("backup", help="write a consistent copy of the store")
    backup.add_argument("dest", help="destination file; must not exist")
    backup.set_defaults(run=_backup)

    args = parser.parse_args(argv)
    try:
        return args.run(args)
    except refmet_store.StoreAdminError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
