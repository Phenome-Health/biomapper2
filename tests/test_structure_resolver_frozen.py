"""Axis 4: StructureResolver serves a node/name InChIKey from the pinned RefMet freeze when present.

A RefMet ``RM:`` node has no KG-asserted InChIKey, so its structure otherwise resolves via a LIVE
MW/PubChem name lookup whose fail-soft ``None`` on a transient error flips ``connectivity_match`` and
thus the committed node run-to-run. Serving the InChIKey from the immutable freeze makes it
deterministic. These tests pin that: frozen value wins with NO network, and a blank/absent freeze
entry still falls through to the live hop. No network.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from biomapper2.core.annotators import refmet_snapshot
from biomapper2.core.structure_resolver import StructureResolver

pytestmark = pytest.mark.unit

FIXTURE = (
    Path(__file__).parent / "fixtures" / "refmet_freeze_fixture.tsv"
)  # 'Cholic acid' -> BHQCQFFYRZLCQQ-OELDTZBJSA-N


@pytest.fixture(autouse=True)
def _reset_snapshot():
    refmet_snapshot.reset()
    yield
    refmet_snapshot.reset()


@pytest.fixture
def frozen(monkeypatch):
    monkeypatch.setenv("REFMET_SNAPSHOT_PATH", str(FIXTURE))
    refmet_snapshot.reset()


def _boom(*_a, **_k):
    raise AssertionError("live structure lookup called where the freeze should have answered")


def test_frozen_inchikey_returns_upper_when_present(frozen):
    assert StructureResolver._frozen_inchikey("Cholic acid") == "BHQCQFFYRZLCQQ-OELDTZBJSA-N"


def test_frozen_inchikey_none_when_absent_or_no_snapshot(monkeypatch):
    monkeypatch.delenv("REFMET_SNAPSHOT_PATH", raising=False)
    refmet_snapshot.reset()
    assert StructureResolver._frozen_inchikey("Cholic acid") is None  # no snapshot configured
    monkeypatch.setenv("REFMET_SNAPSHOT_PATH", str(FIXTURE))
    refmet_snapshot.reset()
    assert StructureResolver._frozen_inchikey("not a metabolite in the freeze") is None  # name absent


def test_resolve_name_key_prefers_freeze_and_skips_live(frozen, monkeypatch):
    sr = StructureResolver(linker=MagicMock())
    monkeypatch.setattr(sr, "_fetch_mw_inchikey", _boom)
    monkeypatch.setattr(sr, "_fetch_pubchem_inchikey", _boom)
    assert sr._resolve_name_key("Cholic acid") == "BHQCQFFYRZLCQQ-OELDTZBJSA-N"  # no network, deterministic


def test_resolve_name_key_falls_to_live_when_not_frozen(frozen, monkeypatch):
    # A name not in the freeze still uses the live hop (unchanged behavior).
    sr = StructureResolver(linker=MagicMock())
    monkeypatch.setattr(sr, "_fetch_mw_inchikey", lambda name: "AAAAAAAAAAAAAA-BBBBBBBBFC-N")
    monkeypatch.setattr(sr, "_fetch_pubchem_inchikey", lambda name: None)
    assert sr._resolve_name_key("novel name not frozen") == "AAAAAAAAAAAAAA-BBBBBBBBFC-N"


def test_connectivity_is_deterministic_via_freeze_when_live_would_fail(frozen, monkeypatch):
    # Committed RM-style node has no KG InChIKey; its structure comes from the freeze, so a transient
    # live failure cannot flip connectivity_match. Node A (frozen name) vs node B (KG InChIKey).
    records = {
        "RM:0041813": {"name": "Cholic acid", "equivalent_ids": {}},  # no KG INCHIKEY -> name hop -> freeze
        "CHEBI:16359": {"name": "cholic acid", "equivalent_ids": {"INCHIKEY": ["BHQCQFFYRZLCQQ-OELDTZBJSA-N"]}},
    }
    lk = MagicMock()
    lk.get_node_records.return_value = records
    sr = StructureResolver(linker=lk)
    monkeypatch.setattr(sr, "_fetch_mw_inchikey", _boom)  # live must NOT be reached for the RM node
    monkeypatch.setattr(sr, "_fetch_pubchem_inchikey", _boom)
    assert sr.connectivity_match("RM:0041813", "CHEBI:16359") is True  # shared block, deterministic


def test_frozen_sentinel_and_blank_are_not_structure(monkeypatch, tmp_path):
    # A regenerated freeze carrying the upstream "-" missing sentinel (or a blank) must NOT be accepted
    # as an InChIKey — it is "no structure", same as the live MW lookup's own "-" filter.
    cols = [
        "query_name",
        "status",
        "refmet_id",
        "refmet_name",
        "inchi_key",
        "formula",
        "exactmass",
        "pubchem_cid",
        "super_class",
        "main_class",
        "sub_class",
        "panels",
    ]
    rows = ["dash sentinel\tvoted\tRM1\t\t-\t\t\t\t\t\t\t", "blank key\tvoted\tRM2\t\t\t\t\t\t\t\t\t"]
    freeze = tmp_path / "freeze.tsv"
    freeze.write_text("\t".join(cols) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    monkeypatch.setenv("REFMET_SNAPSHOT_PATH", str(freeze))
    refmet_snapshot.reset()
    assert StructureResolver._frozen_inchikey("dash sentinel") is None
    assert StructureResolver._frozen_inchikey("blank key") is None


def test_frozen_lookup_is_fail_soft(frozen, monkeypatch):
    # A broken snapshot must degrade to the live hop, never abort resolution.
    monkeypatch.setattr(refmet_snapshot, "lookup", lambda *_a, **_k: (_ for _ in ()).throw(RuntimeError("corrupt")))
    assert StructureResolver._frozen_inchikey("Cholic acid") is None
