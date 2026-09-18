"""Unit 6 wiring: the Mapper widens the lipid population so a committed lipid node with no graph
InChIKey becomes in scope for the structure-free composition check.

The check parses the committed node's NAME (one /get-nodes read, no MW/PubChem lookup) and compares it
to the query's Goslin parse. Every collaborator is a small fake, so no pygoslin grammar is built and no
service is touched.
"""

from __future__ import annotations

from typing import Any

from biomapper2.core.annotators.goslin_grammar import LipidParse
from biomapper2.core.certificate import LipidResolutionLevel

NODE = "RM:0010728"
NO_KEY = {"HMDB": ["HMDB0010728"]}
WITH_KEY = {"INCHIKEY": ["BSYNRYMUTXBXSQ-UHFFFAOYSA-N"]}


def _lipid_row(canonical: str = "PC 34:1") -> dict:
    """An entity carrying goslin-lipid votes, the signal ``_goslin_base_metadata`` reads."""
    meta = {"goslin_canonical": canonical, "goslin_level_names": {"CLASS": "PC", "SPECIES": canonical}}
    return {"assigned_ids": {"goslin-lipid": {"REFMET": {"RM0010728": meta}}}}


class _FakeLinker:
    def __init__(self, node_name: str | None) -> None:
        self._node_name = node_name
        self.calls = 0

    def get_node_records(self, node_ids):
        self.calls += 1
        return {node_ids[0]: {"name": self._node_name, "equivalent_ids": {}}}


class _FakeLipidResolver:
    """Parses a name into a LipidParse only when it looks like the fixture's lipid; else None."""

    def parse(self, name):
        if not name or not str(name).startswith("PC"):
            return None
        species = str(name)
        return LipidParse(
            input_name=species,
            canonical_name=species,
            sum_formula=None,
            monoisotopic_mass=None,
            level="LipidLevel.SPECIES",
            dialect="Goslin",
            level_names={"CLASS": "PC", "SPECIES": species},
        )


class _Mapper:
    """The narrowest object exposing ``_lipid_structure_evidence``."""

    def __init__(self, node_name: str | None, lipid_resolver: Any = None) -> None:
        from biomapper2.mapper import Mapper

        self.linker = _FakeLinker(node_name)
        self.lipid_resolver: Any = lipid_resolver if lipid_resolver is not None else _FakeLipidResolver()
        self._lipid_structure_evidence = Mapper._lipid_structure_evidence.__get__(self, _Mapper)


def test_a_no_inchikey_lipid_node_gets_a_structure_free_verdict() -> None:
    mapper = _Mapper(node_name="PC 34:1")
    evidence = mapper._lipid_structure_evidence(node_id=NODE, kg_equivalent_ids=NO_KEY, lipid_row=_lipid_row())
    assert evidence is not None
    assert evidence.level is LipidResolutionLevel.LIPID_SPECIES
    assert evidence.mapping_relation == "exact"


def test_a_node_that_disagrees_on_composition_contradicts() -> None:
    mapper = _Mapper(node_name="PC 36:2")
    evidence = mapper._lipid_structure_evidence(node_id=NODE, kg_equivalent_ids=NO_KEY, lipid_row=_lipid_row("PC 34:1"))
    assert evidence is not None
    assert evidence.level is LipidResolutionLevel.CONTRADICTED


def test_an_inchikey_bearing_node_is_left_to_the_block_path() -> None:
    mapper = _Mapper(node_name="PC 34:1")
    evidence = mapper._lipid_structure_evidence(node_id=NODE, kg_equivalent_ids=WITH_KEY, lipid_row=_lipid_row())
    assert evidence is None


def test_a_non_lipid_query_is_out_of_scope_for_the_structure_free_check() -> None:
    mapper = _Mapper(node_name="PC 34:1")
    evidence = mapper._lipid_structure_evidence(node_id=NODE, kg_equivalent_ids=NO_KEY, lipid_row={"assigned_ids": {}})
    assert evidence is None


def test_an_enrichment_outage_skips_the_structure_free_fetch() -> None:
    # During a /get-nodes enrichment outage the row is unavailable regardless, so the check must not
    # buy a second redundant /get-nodes round trip whose verdict the certificate would discard.
    mapper = _Mapper(node_name="PC 34:1")
    evidence = mapper._lipid_structure_evidence(
        node_id=NODE, kg_equivalent_ids={}, lipid_row=_lipid_row(), equivalent_ids_lookup_ok=False
    )
    assert evidence is None
    assert mapper.linker.calls == 0


def test_the_check_is_inert_when_tier_b_is_disabled() -> None:
    # lipid_resolver is None exactly when Tier B is off; the check must not fetch or parse anything.
    mapper = _Mapper(node_name="PC 34:1")
    mapper.lipid_resolver = None
    evidence = mapper._lipid_structure_evidence(node_id=NODE, kg_equivalent_ids=NO_KEY, lipid_row=_lipid_row())
    assert evidence is None
    assert mapper.linker.calls == 0  # no /get-nodes read when the check is off
