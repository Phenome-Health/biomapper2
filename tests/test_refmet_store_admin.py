"""RefMet store admin: change history, the one-time repair, and the consistent backup.

Temporary SQLite and TSV files only; NO network. The logic under test lives in
``core/annotators/refmet_store.py``; ``scripts/refmet_store_admin.py`` is the thin command line over it.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path

import pytest

from biomapper2.core.annotators import refmet_snapshot, refmet_store
from biomapper2.core.annotators.metabolomics_workbench import MetabolomicsWorkbenchAnnotator
from scripts import refmet_store_admin

pytestmark = pytest.mark.unit

DHA = "docosahexaenoate (dha; 22:6n3)"
DPA = "docosapentaenoate (n3 dpa; 22:5n3)"
HEADER = "query_name\tstatus\trefmet_id\trefmet_name\tinchi_key\tformula\texactmass\n"


@pytest.fixture(autouse=True)
def _reset_snapshot_cache():
    refmet_snapshot.reset()
    yield
    refmet_snapshot.reset()


@pytest.fixture
def db(monkeypatch, tmp_path: Path) -> Path:
    path = tmp_path / "store.sqlite"
    monkeypatch.setenv("REFMET_STORE_PATH", str(path))
    monkeypatch.delenv("REFMET_SNAPSHOT_PATH", raising=False)
    return path


@pytest.fixture
def freeze(tmp_path: Path) -> Path:
    path = tmp_path / "refmet_freeze_mw_match_20261022T164109Z.tsv"
    path.write_text(
        HEADER
        + f"{DHA}\tvoted\tRM0153631\tDHA\tMBMBGCFOFBJSGT-KUBAVDMBSA-N\tC22H32O2\t328.2402\n"
        + f"{DPA}\tvoted\tRM0153164\tDPA\t\t\t\n"
        + "already votes\tvoted\tRM1\tX\t\t\t\n"
        + "freeze says no\tno_match\t\t\t\t\t\n"
        + "vote without id\tvoted\t\t\t\t\t\n"
        + "Case Twin\tvoted\tRM2\tY\t\t\t\n",
        encoding="utf-8",
    )
    return path


def _overwritten_store(db: Path) -> None:
    """The store as prod left it: live no-matches written over two votes the freeze still has."""
    refmet_store.upsert(DHA, "no_match", path=db)
    refmet_store.upsert(DPA, "no_match", path=db)
    refmet_store.upsert("already votes", "voted", refmet_id="RM1", path=db)
    refmet_store.upsert("freeze says no", "no_match", path=db)
    refmet_store.upsert("vote without id", "no_match", path=db)
    refmet_store.upsert("case twin", "no_match", path=db)  # differs from the freeze only by case
    refmet_store.upsert("not in the freeze", "no_match", path=db)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _set_updated_at(db: Path, query: str, value: str) -> None:
    conn = sqlite3.connect(str(db))
    conn.execute("UPDATE refmet_by_query SET updated_at = ? WHERE query = ?", (value, query))
    conn.commit()
    conn.close()


# ---- store functions -----------------------------------------------------------------------


def test_contradictions_are_listed_by_exact_query_with_update_time(db: Path, freeze: Path):
    _overwritten_store(db)
    before = _digest(db)

    found = refmet_store.find_contradictions(freeze, path=db)

    assert [c.query for c in found] == sorted([DHA, DPA])
    by_query = {c.query: c for c in found}
    assert by_query[DHA].freeze_refmet_id == "RM0153631"
    hit = refmet_store.get(DHA, path=db)
    assert hit is not None and by_query[DHA].updated_at == hit.updated_at
    assert hit.status == "no_match"  # listing changed nothing
    assert _digest(db) == before


def test_since_excludes_rows_last_updated_before_the_date(db: Path, freeze: Path):
    _overwritten_store(db)
    _set_updated_at(db, DHA, "2026-11-12T04:44:00+00:00")
    _set_updated_at(db, DPA, "2026-11-01T00:00:00+00:00")
    found = refmet_store.find_contradictions(freeze, since="2026-11-12", path=db)
    assert [c.query for c in found] == [DHA]


def test_restore_puts_back_the_freeze_vote_and_logs_one_repair(db: Path, freeze: Path):
    _overwritten_store(db)

    restored = refmet_store.restore_from_freeze(freeze, [DHA], path=db)

    assert restored == [DHA]
    hit = refmet_store.get(DHA, path=db)
    assert hit is not None
    assert (hit.status, hit.refmet_id, hit.refmet_name) == ("voted", "RM0153631", "DHA")
    assert (hit.source, hit.snapshot_version) == ("local_snapshot", freeze.stem)
    assert (hit.live_status, hit.live_refmet_id) == ("no_match", None)  # live still says no-match
    untouched = refmet_store.get(DPA, path=db)
    assert untouched is not None and untouched.status == "no_match"
    changes = refmet_store.list_changes(path=db)
    assert len(changes) == 1
    assert (changes[0].query, changes[0].kind, changes[0].old_basis) == (DHA, "repair", "stored_row")
    assert (changes[0].old_status, changes[0].new_status, changes[0].new_refmet_id) == (
        "no_match",
        "voted",
        "RM0153631",
    )


def test_restore_skips_a_row_that_is_no_longer_a_no_match(db: Path, freeze: Path):
    _overwritten_store(db)
    found = [c.query for c in refmet_store.find_contradictions(freeze, path=db)]
    # Between the dry run and the apply, live matched DPA again with another id.
    refmet_store.apply_live_answer(DPA, "voted", refmet_id="RM_NEW", path=db)

    restored = refmet_store.restore_from_freeze(freeze, found, path=db)

    assert restored == [DHA]
    hit = refmet_store.get(DPA, path=db)
    assert hit is not None and hit.refmet_id == "RM_NEW"


def test_restore_ignores_queries_the_freeze_does_not_vote_for(db: Path, freeze: Path):
    _overwritten_store(db)
    restored = refmet_store.restore_from_freeze(
        freeze, ["freeze says no", "vote without id", "case twin", "nope"], path=db
    )
    assert restored == []
    assert refmet_store.list_changes(path=db) == []


def test_live_no_match_after_a_repair_keeps_the_vote_without_a_second_record(db: Path, freeze: Path):
    _overwritten_store(db)
    refmet_store.restore_from_freeze(freeze, [DHA], path=db)

    ann = MetabolomicsWorkbenchAnnotator(freeze_mode="live_backup")
    ann._request_once = lambda metabolite_name: None  # live still says no-match
    result = ann._fetch_refmet_data(DHA)

    assert (result.status, result.source, result.data) == ("voted", "kept_live_no_match", {"refmet_id": "RM0153631"})
    assert [c.kind for c in refmet_store.list_changes(path=db)] == ["repair"]


def test_unreadable_or_malformed_freeze_is_an_error(db: Path, tmp_path: Path):
    _overwritten_store(db)
    with pytest.raises(refmet_store.StoreAdminError):
        refmet_store.find_contradictions(tmp_path / "missing.tsv", path=db)
    bad = tmp_path / "bad.tsv"
    bad.write_text("name\tid\nx\ty\n", encoding="utf-8")
    with pytest.raises(refmet_store.StoreAdminError):
        refmet_store.find_contradictions(bad, path=db)
    with pytest.raises(refmet_store.StoreAdminError):
        refmet_store.restore_from_freeze(bad, [DHA], path=db)


def test_admin_functions_refuse_a_missing_store_and_create_no_file(db: Path, freeze: Path, tmp_path: Path):
    for call in (
        lambda: refmet_store.find_contradictions(freeze, path=db),
        lambda: refmet_store.restore_from_freeze(freeze, [DHA], path=db),
        lambda: refmet_store.backup(tmp_path / "copy.sqlite", path=db),
        lambda: refmet_store.store_info(path=db),
    ):
        with pytest.raises(refmet_store.StoreAdminError):
            call()
    assert not db.exists()
    assert not (tmp_path / "copy.sqlite").exists()


def test_backup_is_a_store_with_the_same_rows(db: Path, tmp_path: Path):
    _overwritten_store(db)
    dest = tmp_path / "backups" / "copy.sqlite"
    dest.parent.mkdir()

    assert refmet_store.backup(dest, path=db) == 7

    hit = refmet_store.get("already votes", path=dest)
    assert hit is not None and (hit.status, hit.refmet_id) == ("voted", "RM1")
    assert refmet_store.store_info(path=dest)[1] == 7
    with pytest.raises(refmet_store.StoreAdminError):
        refmet_store.backup(dest, path=db)  # never overwrite an existing backup


# ---- command line --------------------------------------------------------------------------


def test_cli_dry_run_lists_contradictions_and_writes_nothing(db: Path, freeze: Path, capsys):
    _overwritten_store(db)
    before = _digest(db)

    assert refmet_store_admin.main(["repair", "--freeze", str(freeze)]) == 0

    out = capsys.readouterr().out
    assert f"store\t{db}" in out and "store_rows\t7" in out and str(freeze) in out
    assert f"{DHA}\tRM0153631\t" in out and f"{DPA}\tRM0153164\t" in out
    assert "contradictions\t2" in out
    assert "Case Twin" not in out and "case twin" not in out
    assert _digest(db) == before
    hit = refmet_store.get(DHA, path=db)
    assert hit is not None and hit.status == "no_match"


def test_cli_read_only_commands_leave_a_pre_migration_store_untouched(db: Path, freeze: Path, capsys):
    conn = sqlite3.connect(str(db))
    conn.execute(
        "CREATE TABLE refmet_by_query (query TEXT PRIMARY KEY, refmet_id TEXT, refmet_name TEXT, inchi_key TEXT, "
        "formula TEXT, exactmass TEXT, status TEXT NOT NULL, source TEXT, updated_at TEXT, snapshot_version TEXT)"
    )
    conn.execute(
        "INSERT INTO refmet_by_query (query, status, updated_at) VALUES (?, 'no_match', ?)",
        (DHA, "2026-11-12T04:44:00+00:00"),
    )
    conn.commit()
    conn.close()
    before = _digest(db)

    assert refmet_store_admin.main(["repair", "--freeze", str(freeze)]) == 0
    assert "contradictions\t1" in capsys.readouterr().out
    assert refmet_store_admin.main(["history"]) == 0
    assert "changes\t0" in capsys.readouterr().out
    assert _digest(db) == before


def test_cli_apply_restores_exactly_the_reviewed_rows(db: Path, freeze: Path, capsys):
    _overwritten_store(db)

    assert refmet_store_admin.main(["repair", "--freeze", str(freeze), "--apply", "--expect-count", "2"]) == 0

    assert "restored\t2" in capsys.readouterr().out
    for query, refmet_id in ((DHA, "RM0153631"), (DPA, "RM0153164")):
        hit = refmet_store.get(query, path=db)
        assert hit is not None and (hit.status, hit.refmet_id) == ("voted", refmet_id)
    assert [c.kind for c in refmet_store.list_changes(path=db)] == ["repair", "repair"]
    # Nothing left to repair: a second apply writes nothing and reports zero.
    assert refmet_store_admin.main(["repair", "--freeze", str(freeze), "--apply", "--expect-count", "0"]) == 0
    assert "restored\t0" in capsys.readouterr().out
    assert len(refmet_store.list_changes(path=db)) == 2


def test_cli_apply_refuses_a_count_that_differs_from_the_reviewed_one(db: Path, freeze: Path, capsys):
    _overwritten_store(db)
    assert refmet_store_admin.main(["repair", "--freeze", str(freeze), "--apply", "--expect-count", "5"]) != 0
    assert "expect-count" in capsys.readouterr().err
    hit = refmet_store.get(DHA, path=db)
    assert hit is not None and hit.status == "no_match"
    assert refmet_store.list_changes(path=db) == []
    # --apply without --expect-count is refused too.
    assert refmet_store_admin.main(["repair", "--freeze", str(freeze), "--apply"]) != 0
    assert refmet_store.list_changes(path=db) == []


def test_cli_since_limits_the_list(db: Path, freeze: Path, capsys):
    _overwritten_store(db)
    _set_updated_at(db, DPA, "2026-11-01T00:00:00+00:00")
    _set_updated_at(db, DHA, "2026-11-12T04:44:00+00:00")
    assert refmet_store_admin.main(["repair", "--freeze", str(freeze), "--since", "2026-11-12"]) == 0
    out = capsys.readouterr().out
    assert "contradictions\t1" in out and DPA not in out


def test_cli_errors_exit_non_zero_and_write_nothing(db: Path, freeze: Path, tmp_path: Path, monkeypatch, capsys):
    # A configured store file that does not exist.
    for argv in (["history"], ["repair", "--freeze", str(freeze)], ["backup", str(tmp_path / "b.sqlite")]):
        assert refmet_store_admin.main(argv) != 0
        assert "does not exist" in capsys.readouterr().err
    assert not db.exists() and not (tmp_path / "b.sqlite").exists()

    # A bad freeze.
    _overwritten_store(db)
    before = _digest(db)
    assert refmet_store_admin.main(["repair", "--freeze", str(tmp_path / "missing.tsv")]) != 0
    assert capsys.readouterr().err
    assert (
        refmet_store_admin.main(["repair", "--freeze", str(tmp_path / "missing.tsv"), "--apply", "--expect-count", "0"])
        != 0
    )
    assert _digest(db) == before

    # No store configured.
    monkeypatch.delenv("REFMET_STORE_PATH")
    assert refmet_store_admin.main(["history"]) != 0
    assert "REFMET_STORE_PATH" in capsys.readouterr().err


def test_cli_history_prints_changes_newest_first(db: Path, capsys):
    for query in ("a", "b", "c"):
        refmet_store.upsert(query, "voted", refmet_id="RM1", path=db)
        refmet_store.apply_live_answer(query, "no_match", path=db)

    assert refmet_store_admin.main(["history", "--limit", "2"]) == 0

    lines = [line for line in capsys.readouterr().out.splitlines() if "\tvote_kept\t" in line]
    assert [line.split("\t")[1] for line in lines] == ["c", "b"]


def test_cli_backup_writes_a_consistent_copy(db: Path, tmp_path: Path, capsys):
    _overwritten_store(db)
    dest = tmp_path / "copy.sqlite"
    assert refmet_store_admin.main(["backup", str(dest)]) == 0
    assert "backup_rows\t7" in capsys.readouterr().out
    assert refmet_store.store_info(path=dest)[1] == 7
