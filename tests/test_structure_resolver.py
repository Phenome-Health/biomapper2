"""Unit tests for the layered InChIKey StructureResolver (no live APIs)."""

from unittest.mock import MagicMock

import pytest

from biomapper2.core.structure_resolver import StructureResolver


def _linker(records):
    lk = MagicMock()
    lk.get_node_records.return_value = records
    return lk


def _fail(*_args, **_kwargs):
    pytest.fail("external structure lookup should not run on a KG hit")


def test_kg_hit_matches_skips_external(monkeypatch):
    sr = StructureResolver(
        _linker(
            {
                "CHEBI:1": {"name": "x", "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-BBBBBBBBBB-N"]}},
                "CHEBI:2": {"name": "y", "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-CCCCCCCCCC-M"]}},
            }
        )
    )
    monkeypatch.setattr(sr, "_fetch_mw_inchikey", _fail)
    monkeypatch.setattr(sr, "_fetch_pubchem_inchikey", _fail)
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is True  # same 1st block, no external call


def test_kg_hit_differs_returns_false():
    sr = StructureResolver(
        _linker(
            {
                "CHEBI:1": {"name": "x", "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-BB-N"]}},
                "CHEBI:2": {"name": "y", "equivalent_ids": {"INCHIKEY": ["ZZZZZZZZZZZZZZ-BB-N"]}},
            }
        )
    )
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is False


def test_kg_miss_falls_through_to_mw(monkeypatch):
    sr = StructureResolver(
        _linker(
            {
                "CHEBI:1": {"name": "acid", "equivalent_ids": {}},  # no KG inchikey
                "CHEBI:2": {"name": "anion", "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-BB-N"]}},
            }
        )
    )
    monkeypatch.setattr(sr, "_fetch_mw_inchikey", lambda name: "AAAAAAAAAAAAAA-XX-M" if name == "acid" else None)
    monkeypatch.setattr(
        sr, "_fetch_pubchem_inchikey", lambda name: pytest.fail("PubChem must not run when MW resolves")
    )
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is True


def test_mw_miss_falls_through_to_pubchem(monkeypatch):
    sr = StructureResolver(
        _linker(
            {
                "CHEBI:1": {"name": "obscure", "equivalent_ids": {}},
                "CHEBI:2": {"name": "anion", "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-BB-N"]}},
            }
        )
    )
    monkeypatch.setattr(sr, "_fetch_mw_inchikey", lambda name: None)
    monkeypatch.setattr(
        sr, "_fetch_pubchem_inchikey", lambda name: "AAAAAAAAAAAAAA-YY-M" if name == "obscure" else None
    )
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is True


def test_all_layers_miss_returns_none(monkeypatch):
    sr = StructureResolver(
        _linker(
            {
                "CHEBI:1": {"name": "obscure", "equivalent_ids": {}},
                "CHEBI:2": {"name": "anion", "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-BB-N"]}},
            }
        )
    )
    monkeypatch.setattr(sr, "_fetch_mw_inchikey", lambda name: None)
    monkeypatch.setattr(sr, "_fetch_pubchem_inchikey", lambda name: None)
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is None  # one node unresolvable


def test_missing_node_name_blocks_fallback():
    sr = StructureResolver(
        _linker(
            {
                "CHEBI:1": {"name": None, "equivalent_ids": {}},  # no name -> can't query MW/PubChem
                "CHEBI:2": {"name": "y", "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-BB-N"]}},
            }
        )
    )
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is None


def test_external_error_returns_none(monkeypatch):
    sr = StructureResolver(
        _linker(
            {
                "CHEBI:1": {"name": "acid", "equivalent_ids": {}},
                "CHEBI:2": {"name": "y", "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-BB-N"]}},
            }
        )
    )

    def boom(name):
        raise RuntimeError("network down")

    monkeypatch.setattr(sr, "_fetch_mw_inchikey", boom)
    monkeypatch.setattr(sr, "_fetch_pubchem_inchikey", lambda name: None)
    # _inchikey_block must swallow the error and return None, not raise.
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is None


# --------------------------- Multi-valued INCHIKEY (D2) ---------------------------
# A KG node's INCHIKEY list is multi-valued (neutral parent, conjugate anion, salt, stereoisomers) and
# keys[0] ordering is arbitrary, so `connectivity_match` compared one arbitrary representation against
# another. That is the same artifact PR #36 fixed in the Hajjar scorer (gold sat at position 2-4 in
# 12/19 misses); `connectivity_match` now tests set intersection over ALL entries instead.


def test_match_on_a_non_first_inchikey_entry():
    """The keys[0] artifact: the shared block is entry 2 on one side and entry 1 on the other."""
    sr = StructureResolver(
        _linker(
            {
                "CHEBI:1": {
                    "name": "acid",
                    "equivalent_ids": {"INCHIKEY": ["ZZZZZZZZZZZZZZ-BB-N", "AAAAAAAAAAAAAA-BB-N"]},
                },
                "CHEBI:2": {"name": "anion", "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-CC-M"]}},
            }
        )
    )
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is True


def test_disjoint_multivalued_lists_still_return_false():
    """Loosening to intersection must not turn genuinely different molecules into a match."""
    sr = StructureResolver(
        _linker(
            {
                "CHEBI:1": {
                    "name": "x",
                    "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-BB-N", "BBBBBBBBBBBBBB-BB-N"]},
                },
                "CHEBI:2": {
                    "name": "y",
                    "equivalent_ids": {"INCHIKEY": ["YYYYYYYYYYYYYY-BB-N", "ZZZZZZZZZZZZZZ-BB-N"]},
                },
            }
        )
    )
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is False


def test_multivalued_side_still_falls_back_by_name_when_kg_is_silent(monkeypatch):
    """The name fallback survives D2, so the change cannot inflate `conflict_no_structure`."""
    sr = StructureResolver(
        _linker(
            {
                "CHEBI:1": {"name": "acid", "equivalent_ids": {}},
                "CHEBI:2": {
                    "name": "y",
                    "equivalent_ids": {"INCHIKEY": ["QQQQQQQQQQQQQQ-BB-N", "AAAAAAAAAAAAAA-BB-N"]},
                },
            }
        )
    )
    monkeypatch.setattr(sr, "_fetch_mw_inchikey", lambda name: "AAAAAAAAAAAAAA-XX-M" if name == "acid" else None)
    monkeypatch.setattr(sr, "_fetch_pubchem_inchikey", lambda name: None)
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is True


def test_conflated_list_can_produce_a_false_agreement():
    """The documented cost of D2, pinned so it cannot regress silently into a surprise.

    Loosening is one-directional: a False can become True, never the reverse. If a KG node's INCHIKEY
    list is itself conflated (entries for two genuinely different molecules), a single shared entry now
    reads as "same molecule" and suppresses the `divergent_refmet` review flag. Accepted trade — a
    suppressed prompt beats the old form's actively wrong conflict adjudication.
    """
    sr = StructureResolver(
        _linker(
            {
                # Conflated node: carries a block belonging to a different molecule alongside its own.
                "CHEBI:1": {
                    "name": "x",
                    "equivalent_ids": {"INCHIKEY": ["AAAAAAAAAAAAAA-BB-N", "ZZZZZZZZZZZZZZ-BB-N"]},
                },
                "CHEBI:2": {"name": "y", "equivalent_ids": {"INCHIKEY": ["ZZZZZZZZZZZZZZ-CC-M"]}},
            }
        )
    )
    assert sr.connectivity_match("CHEBI:1", "CHEBI:2") is True
